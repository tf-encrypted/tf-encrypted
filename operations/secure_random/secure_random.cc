#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/tensor_util.h"

#include "generators.h"

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using tensorflow::tensor::MakeShape;

using i128_t = __int128_t;
using u128_t = __uint128_t;

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

static std::vector<shape_inference::DimensionHandle> _takeDims(shape_inference::InferenceContext* c,
                                                               shape_inference::ShapeHandle const& shape) {
  int32 rank = c->Rank(shape);
  std::vector<shape_inference::DimensionHandle> dims;
  for (int32 i = 0; i < rank; ++i) dims.emplace_back(c->Dim(shape, i));
  return dims;
}

static Status I128RandomUniformShapeCommon(InferenceContext* context) {
  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(context->MakeShapeFromShapeTensor(0, &out));
  auto dims = _takeDims(context, out);
  dims.emplace_back(context->MakeDim(2));
  context->set_output(0, context->MakeShape(dims));
  return Status::OK();
}

static Status I128SeededRandomUniformShape(InferenceContext* context) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(seed, 0), NUMBER_OF_SEEDS, &unused));

  return I128RandomUniformShapeCommon(context);
}

REGISTER_OP("SecureSeededRandomUniform")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int8, int16, int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32} = DT_INT32")
    .SetShapeFn(SeededRandomUniformShape);

REGISTER_OP("SecureRandomUniform")
    .Input("shape: T")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int8, int16, int32, int64} = DT_INT32")
    .Attr("T: {int32, int64} = DT_INT32")
    .SetShapeFn(RandomUniformShapeCommon);
  
REGISTER_OP("I128SecureSeededRandomUniform")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int64} = DT_INT64")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32} = DT_INT32")
    .SetIsStateful()
    .SetShapeFn(I128SeededRandomUniformShape);

REGISTER_OP("I128SecureRandomUniform")
    .Input("shape: T")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int64} = DT_INT64")
    .Attr("T: {int32, int64} = DT_INT32")
    .SetIsStateful()
    .SetShapeFn(I128RandomUniformShapeCommon);

REGISTER_OP("SecureSeed")
    .Output("output: int32")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
        if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
        c->set_output(0, c->MakeShape({c->MakeDim(NUMBER_OF_SEEDS)}));
        return Status::OK();
    });

template <typename T, typename Gen>
class SeededRandomUniformOp : public OpKernel {
public:
  explicit SeededRandomUniformOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    auto data = output->flat<T>().data();
    Gen gen(data, shape.num_elements(), seed_bytes);

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

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    int num_threads = worker_threads.num_threads;

    // this calculated number doesn't seem to make much difference in the
    // time it takes but its meant to just be an estimation of how many compute cycles
    // each numbers takes to generate so that the shard function can efficiently schedule
    // the shards to take advantage of as much parallelizism as possible
    int estimated_chacha_cpb = 3;
    int estimated_chacha_per_number = 3 * sizeof(T);
    int uniform_dist_per_number = 11;
    int total_cycles = (estimated_chacha_per_number + uniform_dist_per_number) * shape.num_elements();
    Shard(num_threads, worker_threads.workers, shape.num_elements(), total_cycles,
            [output, lo, hi](int64 start_group, int64 limit_group) {
              auto data = output->flat<T>().data();
              int64 size = limit_group - start_group;

              Gen gen(data + start_group, size);

              gen.GenerateData(lo, hi);
            });
  }
};

class I128SeededRandomUniformOp : public OpKernel {
public:
  explicit I128SeededRandomUniformOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    i128_t lo = 0, hi = 0;
    int64_t* loI64View = ((int64_t*) &lo);
    int64_t* hiI64View = ((int64_t*) &hi);
    if (TensorShapeUtils::IsScalar(minval.shape())) {
        loI64View[0] = minval.scalar<int64>()();
        if (loI64View[0] < 0)
            loI64View[1] = -1;
    }
    else {
        OP_REQUIRES(context, minval.dims() == 1 && minval.dim_size(0) == 2,
                errors::InvalidArgument("minval must be int64[2]"));
        lo = *((i128_t*) minval.flat<int64>().data());
    }

    if (TensorShapeUtils::IsScalar(maxval.shape())) {
        hiI64View[0] = maxval.scalar<int64>()();
        if (hiI64View[0] < 0)
            hiI64View[1] = -1;
    }
    else {
        OP_REQUIRES(context, maxval.dims() == 1 && maxval.dim_size(0) == 2,
                errors::InvalidArgument("maxval must be int64[2]"));
        hi = *((i128_t*) maxval.flat<int64>().data());
    }

    OP_REQUIRES(
        context, lo < hi,
        errors::InvalidArgument("Error: minval >= maxval"));

    // Allocate output
    shape.AddDim(2); // Need two int64 to hold one int128
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

    auto seed_vals = seed_tensor.flat<int32>().data();
    const unsigned char * seed_bytes = reinterpret_cast<const unsigned char*>(seed_vals);

    i128_t* data = (i128_t*) (output->flat<int64>().data());
    SeededGenerator<i128_t> gen(data, shape.num_elements() / 2, seed_bytes);
    gen.GenerateData(lo, hi);
  }
};

class I128RandomUniformOp : public OpKernel {
public:
  explicit I128RandomUniformOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_tensor = context->input(0);
    const Tensor& minval = context->input(1);
    const Tensor& maxval = context->input(2);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_tensor, &shape));

    i128_t lo = 0, hi = 0;
    int64_t* loI64View = ((int64_t*) &lo);
    int64_t* hiI64View = ((int64_t*) &hi);
    if (TensorShapeUtils::IsScalar(minval.shape())) {
        loI64View[0] = minval.scalar<int64>()();
        if (loI64View[0] < 0)
            loI64View[1] = -1;
    }
    else {
        OP_REQUIRES(context, minval.dims() == 1 && minval.dim_size(0) == 2,
                errors::InvalidArgument("minval must be int64[2]"));
        lo = *((i128_t*) minval.flat<int64>().data());
    }

    if (TensorShapeUtils::IsScalar(maxval.shape())) {
        hiI64View[0] = maxval.scalar<int64>()();
        if (hiI64View[0] < 0)
            hiI64View[1] = -1;
    }
    else {
        OP_REQUIRES(context, maxval.dims() == 1 && maxval.dim_size(0) == 2,
                errors::InvalidArgument("maxval must be int64[2]"));
        hi = *((i128_t*) maxval.flat<int64>().data());
    }

    OP_REQUIRES(
        context, lo < hi,
        errors::InvalidArgument("Error: minval >= maxval"));

    // Allocate output
    shape.AddDim(2); // Need two int64 to hold one int128
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    int num_threads = worker_threads.num_threads;

    // this calculated number doesn't seem to make much difference in the
    // time it takes but its meant to just be an estimation of how many compute cycles
    // each numbers takes to generate so that the shard function can efficiently schedule
    // the shards to take advantage of as much parallelizism as possible
    int estimated_chacha_cpb = 3;
    int estimated_chacha_per_number = 3 * sizeof(i128_t);
    int uniform_dist_per_number = 11;
    int total_cycles = (estimated_chacha_per_number + uniform_dist_per_number) * shape.num_elements();
    Shard(num_threads, worker_threads.workers, shape.num_elements() / 2, total_cycles,
            [output, lo, hi](int64 start_group, int64 limit_group) {
              i128_t* data = (i128_t*) (output->flat<int64>().data());
              int64 size = limit_group - start_group;

              Generator<i128_t> gen(data + start_group, size);

              gen.GenerateData(lo, hi);
            });
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

    generate_seed(output->flat<int32>().data());
  }
};

REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int8>("dtype"),
  SeededRandomUniformOp<int8, SeededGenerator<int8>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int16>("dtype"),
  SeededRandomUniformOp<int16, SeededGenerator<int16>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  SeededRandomUniformOp<int32, SeededGenerator<int32>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  SeededRandomUniformOp<int64, SeededGenerator<int64>>);
REGISTER_KERNEL_BUILDER(
  Name("I128SecureSeededRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  I128SeededRandomUniformOp);

REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int8>("dtype"),
  RandomUniformOp<int8, Generator<int8>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int16>("dtype"),
  RandomUniformOp<int16, Generator<int16>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("dtype"),
  RandomUniformOp<int32, Generator<int32>>);
REGISTER_KERNEL_BUILDER(
  Name("SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  RandomUniformOp<int64, Generator<int64>>);
REGISTER_KERNEL_BUILDER(
  Name("I128SecureRandomUniform")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("dtype"),
  I128RandomUniformOp);

REGISTER_KERNEL_BUILDER(
  Name("SecureSeed")
  .Device(DEVICE_CPU),
  SeedOp);
