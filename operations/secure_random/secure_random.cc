#include <tuple>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "sodium.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

#define CHACHABLOCKSIZE 64
#define NUMBER_OF_SEEDS randombytes_SEEDBYTES / sizeof(int32)

template <typename uInt, typename uWide>
std::tuple<uInt, uInt> wmul(uInt x, uInt y) {
    uWide x_w = static_cast<uWide>(x);
    uWide y_w = static_cast<uWide>(y);

    uWide tmp = x_w * y_w;

    // shift top half by size of not wide int to get mod
    // lower half is the result of x * y

    auto hi = static_cast<uInt>(tmp >> (sizeof(uInt) * 8));
    auto lo = static_cast<uInt>(tmp);

    auto tup = std::make_tuple(hi, lo);

    return tup;
}

static Status SecureRandomShape(shape_inference::InferenceContext* context) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(seed, 0), NUMBER_OF_SEEDS, &unused));

  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(context->MakeShapeFromShapeTensor(0, &out));
  context->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("SecureRandom")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32} = DT_INT32")
    .SetShapeFn(SecureRandomShape);

// this function allows you to skip ahead in the chacha stream so you don't
// have to allocate more memory than you need to, used in rejection sampling and
// will be handy for parallelizing this operation
void randombytes_buf_deterministic_ic(void * const buf, const size_t size, uint32 count,
                              const unsigned char seed[randombytes_SEEDBYTES])
{
    static const unsigned char nonce[crypto_stream_chacha20_ietf_NONCEBYTES] = {
        'L', 'i', 'b', 's', 'o', 'd', 'i', 'u', 'm', 'D', 'R', 'G'
    };

    unsigned char * u_buf = (unsigned char *)buf;

    memset(u_buf, 0, size);

    crypto_stream_chacha20_ietf_xor_ic(u_buf, u_buf, (unsigned long long) size,
                                       nonce, count, seed);
}


template <typename T, typename Wide>
class Generator {
public:
  Tensor *output = NULL;
  const unsigned char * seeds = NULL;

  Generator(Tensor* output, const unsigned char * seeds) : output(output), seeds(seeds) {
    auto flat = output->flat<T>();

    count_ = flat.size();
    bytes_count_ = count_ * sizeof(T);
    buf_ = static_cast<T *>(flat.data());

    elements_per_block_ = CHACHABLOCKSIZE / sizeof(T);
    extra_block_ = static_cast<T *>(malloc(CHACHABLOCKSIZE));

    block_counter_ = bytes_count_ / CHACHABLOCKSIZE + 1;

    // prepare the extra block if any values get rejected in the rejection sampling
    randombytes_buf_deterministic_ic(extra_block_, CHACHABLOCKSIZE, block_counter_, seeds);
  }

  ~ Generator() {
    free(extra_block_);
  }

  void GenerateData(T minval, T maxval) {
    auto flat = output->flat<T>();

    randombytes_buf_deterministic(buf_, bytes_count_, seeds);

    Uniform(minval, maxval - 1);

    std::copy(buf_, buf_ + flat.size(), flat.data());
  }

private:
  T *buf_ = NULL;
  uint32 count_ = 0;
  uint32 bytes_count_ = 0;

  T * extra_block_ = NULL;
  uint32 block_counter_ = 0;
  uint32 elements_per_block_ = 0;
  uint32 inner_block_index_ = 0;

  // The following random uniform distribution is based on a an implementation from
  // https://github.com/rust-random/rand/blob/3eadab75c8a5871d1be729091795a6c4e1dc19bb/src/distributions/uniform.rs#L310
  // There is quite a bit of documentation at that link which is explains the implementation.
  // See below for other inline docs.

  // inclusive uniform!
  void Uniform(T low, T high) {
    typedef typename std::make_unsigned<T>::type uT;

    // add one for inclusive range, subtract 1 from high input to get exclusive range
    auto range = static_cast<uT>(high) - static_cast<uT>(low) + 1;

    auto unsigned_max = std::numeric_limits<uT>::max();

    // find the number of integers to reject
    auto ints_to_reject = (unsigned_max - range + 1) % range;

    // find the allowed zone, multiple of the range
    auto zone = unsigned_max - ints_to_reject;

    // loop through all of the values to check for numbers to reject
    for (uint32_t i = 0; i < count_; ++i) {
      // we need the unsigned version here
      auto unsign = static_cast<uT>(buf_[i]);

      uT hi, lo;
      // returns a tuple of result of the widening multiplication
      // hi word contains the result, i.e. the left over after the cast back to normal width
      // lo word contains the product minus the first 32 bit after cast back to normal width
      std::tie(hi, lo) = wmul<uT, Wide>(unsign, range);

      // if lo is out of the zone reject and get another number
      while(lo > zone) {
        // rejection sampling, get the next valid number in the stream
        buf_[i] = GetNextValidData();
        unsign = static_cast<uT>(buf_[i]);

        std::tie(hi, lo) = wmul<uT, Wide>(unsign, range);
      }

      // shift hi by the lower bound to get the value in between lower/upper bound
      buf_[i] = random::SignedAdd(low, hi);
    }
  }

  T GetNextValidData() {
    // if the extra block has been used up get the next available block
    if(inner_block_index_ + 1 == elements_per_block_) {
      inner_block_index_ = 0;
      block_counter_++;

      randombytes_buf_deterministic_ic(extra_block_, CHACHABLOCKSIZE, block_counter_, seeds);
    }

    T ret = extra_block_[inner_block_index_];
    inner_block_index_++;

    return ret;
  }
};

template <typename T, typename Wide>
class SecureRandomOp : public OpKernel {
public:
  explicit SecureRandomOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_tensor = context->input(0);
    const Tensor& seed_tensor = context->input(1);
    const Tensor& minval = context->input(2);
    const Tensor& maxval = context->input(3);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_tensor, &shape));
    OP_REQUIRES(context, seed_tensor.dims() == 1 && seed_tensor.dim_size(0) == NUMBER_OF_SEEDS,
                errors::InvalidArgument("seed must have shape [", NUMBER_OF_SEEDS, "], not ",
                                        seed_tensor.shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    T hi = maxval.scalar<T>()();
    T lo = minval.scalar<T>()();
    OP_REQUIRES(
      context, lo < hi,
      errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    OP_REQUIRES(context, shape.num_elements() > 0, errors::InvalidArgument("Shape contains zero elements"));
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize, try again"));



    int32 *seeds = static_cast<int *>(malloc(sizeof(int32) * NUMBER_OF_SEEDS));
    auto seed_vals = seed_tensor.flat<int32>().data();

    for(auto i = 0; i < NUMBER_OF_SEEDS; i++) {
      seeds[i] = seed_vals[i];
    }

    const unsigned char * seed_bytes = reinterpret_cast<const unsigned char*>(seeds);

    Generator<T, Wide> gen(output, seed_bytes);

    gen.GenerateData(lo, hi);

    free(seeds);
  }
};


REGISTER_KERNEL_BUILDER(
  Name("SecureRandom")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  SecureRandomOp<int32, uint64>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandom")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  SecureRandomOp<int64, __uint128_t>);
