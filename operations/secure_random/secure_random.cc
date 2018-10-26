#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
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
    .Output("output: dtype")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32} = DT_INT32")
    .SetShapeFn(SecureRandomShape);

template <typename T>
class SecureRandomOp : public OpKernel {
public:
  explicit SecureRandomOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_t, &shape));
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 8,
                errors::InvalidArgument("seed must have shape [8], not ",
                                        seed_t.shape().DebugString()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    if (sodium_init() < 0) {
      return;
    }
  
    int number_of_seeds = randombytes_SEEDBYTES / sizeof(int32);

    int32 *seeds = static_cast<int *>(malloc(sizeof(int32) * number_of_seeds));
    auto seed_vals = seed_t.flat<int32>().data();

    for(auto i = 0; i < number_of_seeds; i++) {
      seeds[i] = seed_vals[i];
    }

    auto flat = output->flat<T>();
    int data_size = flat.size() * sizeof(T);

    auto bytes = reinterpret_cast<const unsigned char*>(seeds);
    T *buf = static_cast<T *>(malloc(data_size));

    randombytes_buf_deterministic(buf, data_size, bytes);

    std::copy(buf, buf + flat.size(), flat.data());
    
    free(buf);
    free(seeds);
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
