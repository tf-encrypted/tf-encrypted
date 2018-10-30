#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "sodium.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

static Status SecureRandomShape(shape_inference::InferenceContext* context) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(seed, 0), 8, &unused));

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



template <typename T>
void Uniform(T *buf, int len, T lo, T hi) {
    auto range = static_cast<typename std::make_unsigned<T>::type>(hi) -
                 static_cast<typename std::make_unsigned<T>::type>(lo);
    for (int i = 0; i < len; ++i) {
      buf[i] = random::SignedAdd(lo, buf[i] % range);
    }
}

template <typename T>
class SecureRandomOp : public OpKernel {
public:
  explicit SecureRandomOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    const Tensor& minval = context->input(2);
    const Tensor& maxval = context->input(3);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_t, &shape));
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 8,
                errors::InvalidArgument("seed must have shape [8], not ",
                                        seed_t.shape().DebugString()));

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
    if (shape.num_elements() == 0) return;

    if (sodium_init() < 0) {
      return;
    }

    GenerateData(seed_t, output, lo, hi);
  }

private:
  void GenerateData(const Tensor& seed_t, Tensor* output, int minval, int maxval) {
    int number_of_seeds = randombytes_SEEDBYTES / sizeof(int32);

    int32 *seeds = static_cast<int *>(malloc(sizeof(int32) * number_of_seeds));
    auto seed_vals = seed_t.flat<int32>().data();

    for(auto i = 0; i < number_of_seeds; i++) {
      seeds[i] = seed_vals[i];
    }

    auto flat = output->flat<T>();

    auto bytes = reinterpret_cast<const unsigned char*>(seeds);

    int len = output->flat<T>().size();
    int bytes_len = len * sizeof(T);
    T *buf = static_cast<T *>(malloc(bytes_len));

    randombytes_buf_deterministic(buf, bytes_len, bytes);

    Uniform<T>(buf, len, minval, maxval);

    std::copy(buf, buf + flat.size(), flat.data());

    free(seeds);
    free(buf);
  }
};


REGISTER_KERNEL_BUILDER(
  Name("SecureRandom")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  SecureRandomOp<int32>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandom")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  SecureRandomOp<int64>);
