#include "generators.h"

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

static Status RandomUniformShapeCommon(InferenceContext* context) {
  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(context->MakeShapeFromShapeTensor(0, &out));
  context->set_output(0, out);
  return Status::OK();
}

static Status SeededRandomUniformShape(InferenceContext* context) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(seed, 0), NUMBER_OF_SEEDS, &unused));

  return RandomUniformShapeCommon(context);
}

REGISTER_OP("SecureSeededRandomUniform")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32} = DT_INT32")
    .SetShapeFn(SeededRandomUniformShape);

REGISTER_OP("SecureRandomUniform")
    .Input("shape: T")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .SetShapeFn(RandomUniformShapeCommon);

REGISTER_OP("SecureSeed")
    .Output("output: int32");

template <typename T, typename Gen>
class SeededRandomUniformOp : public OpKernel {
public:
  explicit SeededRandomUniformOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::cout << "HI" << std::endl;
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
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));

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

    auto seed_vals = seed_tensor.flat<int32>().data();
    const unsigned char * seed_bytes = reinterpret_cast<const unsigned char*>(seed_vals);

    Gen gen(output, seed_bytes);


    gen.GenerateData(lo, hi);
  }
};

template <typename T, typename Gen>
class RandomUniformOp : public OpKernel {
public:
  explicit RandomUniformOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_tensor = context->input(0);
    const Tensor& minval = context->input(1);
    const Tensor& maxval = context->input(2);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_tensor, &shape));

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));

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

    Gen gen(output);

    gen.GenerateData(lo, hi);
  }
};

class SeedOp : public OpKernel {
public:
  explicit SeedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorShape shape({NUMBER_OF_SEEDS});

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize, try again"));

    randombytes_buf(output->flat<int32>().data(), randombytes_SEEDBYTES);
  }
};

REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  SeededRandomUniformOp<int32, SeededGenerator<int32, int64>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  SeededRandomUniformOp<int64, SeededGenerator<int64, __uint128_t>>);

REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  RandomUniformOp<int32, Generator<int32, int64>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  RandomUniformOp<int64, Generator<int64, __uint128_t>>);

REGISTER_KERNEL_BUILDER(
  Name("SecureSeed")
  .Device(DEVICE_CPU),
  SeedOp);
