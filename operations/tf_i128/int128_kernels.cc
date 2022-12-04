#include "./int128_tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
static bool IsValidateI128Tensor(TensorShape const& shape) {
    return shape.dims() > 0 && shape.dim_size(shape.dims() - 1) == tf_i128::N_LIMBS;
}

static TensorShape shapeAfterReduction(TensorShape in, long axis, bool keepdim) {
    if (axis < 0) {
        if (keepdim) {
            for (long d = 0; d < in.dims() - 1; ++d) in.set_dim(d, 1);
            in.set_dim(in.dims() - 1, tf_i128::N_LIMBS);
            return in;
        } else {
            return TensorShape({tf_i128::N_LIMBS});
        }
    }

    if (keepdim)
        in.set_dim(axis, 1);
    else
        in.RemoveDim(axis);
    return in;
}

static Status binaryOpShape(TensorShape& out, const TensorShape& shape0, const TensorShape& shape1) {
    //CHECK(IsValidateI128Tensor(shape0));
    //CHECK(IsValidateI128Tensor(shape1));

    std::vector<int32> dims;
    for (long i = shape0.dims() - 1, j = shape1.dims() - 1; i >= 0 || j >= 0; --i, --j) {
        int32 d0 = i >= 0 ? shape0.dim_size(i) : 1;
        int32 d1 = j >= 0 ? shape1.dim_size(j) : 1;
        dims.push_back(std::max(d0, d1));
    }
    std::reverse(dims.begin(), dims.end());

    return TensorShapeUtils::MakeShape(dims.data(), dims.size(), &out);
}

static std::vector<shape_inference::DimensionHandle> _takeDims(shape_inference::InferenceContext* c,
                                                               shape_inference::ShapeHandle const& shape) {
    int32 rank = c->Rank(shape);
    std::vector<shape_inference::DimensionHandle> dims;
    for (int32 i = 0; i < rank; ++i) dims.emplace_back(c->Dim(shape, i));
    return dims;
}

static std::vector<shape_inference::DimensionHandle> do_broadcast(shape_inference::InferenceContext* c) {
    auto shape0 = c->input(0);
    auto shape1 = c->input(1);
    int rank0   = c->Rank(shape0);
    int rank1   = c->Rank(shape1);

    std::vector<shape_inference::DimensionHandle> dims;
    if (rank0 == 0) {
        dims = _takeDims(c, shape1);
    } else if (rank1 == 0) {
        dims = _takeDims(c, shape0);
    } else {
        int rank = std::max(rank0, rank1);
        shape_inference::DimensionHandle bc_dim;
        for (long i = 0; i < rank; ++i) {
            int32 dim0 = i < rank0 ? c->Value(c->Dim(shape0, i)) : 1;
            int32 dim1 = i < rank1 ? c->Value(c->Dim(shape1, i)) : 1;
            dims.emplace_back(c->MakeDim(std::max(dim0, dim1)));
        }
    }
    if (dims.empty())  // when only operators are with rank = 0
        dims.push_back(c->MakeDim(tf_i128::N_LIMBS));
    return dims;
}

static Status _broadcastAndDropLastDim(shape_inference::InferenceContext* c) {
    if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
    auto dims = do_broadcast(c);
    CHECK(!dims.empty());
    dims.pop_back();
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
}

static Status _broadcast(shape_inference::InferenceContext* c) {
    if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
    auto dims = do_broadcast(c);
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
}


#define DEFINE_I128_UnaryOP(OpName, Method)                                     \
    class OpName : public OpKernel {                                            \
    public:                                                                     \
        explicit OpName(OpKernelConstruction* context) : OpKernel(context) {}   \
        void Compute(OpKernelContext* ctx) override {                           \
            const Tensor& op0 = ctx->input(0);                                  \
            CHECK(IsValidateI128Tensor(op0.shape()));                           \
            Tensor* output;                                                     \
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, op0.shape(), &output)); \
            Method(*output, op0);                                             \
        }                                                                       \
    }


#define DEFINE_I128_BinaryOP(OpName, Method)                                  \
    class OpName : public OpKernel {                                          \
    public:                                                                   \
        explicit OpName(OpKernelConstruction* context) : OpKernel(context) {} \
        void Compute(OpKernelContext* ctx) override {                         \
            const Tensor& op0         = ctx->input(0);                        \
            const Tensor& op1         = ctx->input(1);                        \
            const TensorShape& shape0 = op0.shape();                          \
            const TensorShape& shape1 = op1.shape();                          \
            Tensor* output;                                                   \
            TensorShape out_shape;                                            \
            CHECK(binaryOpShape(out_shape, shape0, shape1).ok());             \
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output)); \
            Method(*output, op0, op1);                                    \
        }                                                                     \
    }


#define DEFINE_I128_ShiftOP(OpName, Method)                                 \
    class OpName : public OpKernel {                                            \
    public:                                                                     \
        explicit OpName(OpKernelConstruction* context) : OpKernel(context) {}   \
        void Compute(OpKernelContext* ctx) override {                           \
            const Tensor& op0         = ctx->input(0);                          \
            const Tensor& op1         = ctx->input(1);                          \
            const TensorShape& shape0 = op0.shape();                            \
            const TensorShape& shape1 = op1.shape();                            \
            TensorShape op0I128Shape(shape0);                                   \
            op0I128Shape.RemoveLastDims(1);                                     \
            Tensor* output;                                                     \
            TensorShape out_shape;                                              \
            CHECK(binaryOpShape(out_shape, op0I128Shape, shape1).ok());         \
            out_shape.AddDim(2);                                             \
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));   \
            Method(*output, op0, op1);                                          \
        }                                                                       \
    }


#define DEFINE_I128_ReduceOP(OpName, Method)                                   \
    class OpName : public OpKernel {                                           \
    public:                                                                    \
        explicit OpName(OpKernelConstruction* context) : OpKernel(context) {}  \
        void Compute(OpKernelContext* ctx) override {                          \
            const Tensor& op0        = ctx->input(0);                          \
            const TensorShape& shape = op0.shape();                            \
            CHECK(IsValidateI128Tensor(shape));                                \
            long axis    = ctx->input(1).scalar<int64>()();                    \
            bool keepdim = ctx->input(2).scalar<bool>()();                     \
            TensorShape out_shape {shapeAfterReduction(shape, axis, keepdim)}; \
            Tensor* output;                                                    \
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));  \
            Method(*output, op0, axis, keepdim);                             \
        }                                                                      \
    }

//! Juhous: Add, Sub, Mul will be casted to AddScalar, SubScalar and MulScalar
//! when the right-hand-side operator is a scalar (i.e., a single i128 value)
DEFINE_I128_BinaryOP(I128AddOp, tf_i128::i128TensorAdd);
DEFINE_I128_BinaryOP(I128SubOp, tf_i128::i128TensorSub);
DEFINE_I128_BinaryOP(I128MulOp, tf_i128::i128TensorMul);

DEFINE_I128_UnaryOP(I128NegateOp, tf_i128::i128TensorNegate);
DEFINE_I128_UnaryOP(I128AbsOp, tf_i128::i128TensorAbs);
DEFINE_I128_UnaryOP(I128BitReverseOp, tf_i128::i128TensorBitReverse);
DEFINE_I128_UnaryOP(I128XorIndicesOp, tf_i128::i128TensorXorIndices);

DEFINE_I128_ShiftOP(I128LeftShiftOp, tf_i128::i128TensorLeftShift);
DEFINE_I128_ShiftOP(I128RightShiftOp, tf_i128::i128TensorRightShift);
DEFINE_I128_ShiftOP(I128LogicRightShiftOp, tf_i128::i128TensorLogicalRightShift);
DEFINE_I128_ReduceOP(I128ReduceSumOp, tf_i128::i128TensorReduceSum);
#undef DEFINE_I128_BinaryOP
#undef DEFINE_I128_UnaryOP
#undef DEFINE_I128_ShiftOP
#undef DEFINE_I128_ReduceOP

class I128MatMulOp : public OpKernel {
public:
    explicit I128MatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& op0 = ctx->input(0);
        const Tensor& op1 = ctx->input(1);
        CHECK(IsValidateI128Tensor(op0.shape()));
        CHECK(IsValidateI128Tensor(op1.shape()));
        Tensor* output;
        long d0 = op0.shape().dim_size(0);
        long d1 = op1.shape().dim_size(1);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({d0, d1, tf_i128::N_LIMBS}), &output));
        tf_i128::I128TensorView view0(op0);
        tf_i128::I128TensorView view1(op1);
        tf_i128::i128TensorMatmul(*output, view0, view1);
    }
};

class ToI128Op : public OpKernel {
public:
    explicit ToI128Op(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& op0    = ctx->input(0);
        const uint64 scale = ctx->input(1).scalar<uint64>()();
        Tensor* output;
        TensorShape out_shape {op0.shape()};
        out_shape.AddDim(tf_i128::N_LIMBS);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
        tf_i128::i128TensorConvert(*output, op0, scale);
    }
};


class FromI128Op : public OpKernel {
public:
    explicit FromI128Op(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& op0    = ctx->input(0);
        const uint64 scale = ctx->input(1).scalar<uint64>()();
        CHECK(IsValidateI128Tensor(op0.shape()));
        Tensor* output;
        TensorShape out_shape {op0.shape()};
        out_shape.RemoveLastDims(1);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
        tf_i128::i128TensorConvertBack(*output, op0, scale);
    }
};


class I128EqualOp : public OpKernel {
public:
    explicit I128EqualOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& op0 = ctx->input(0);
        const Tensor& op1 = ctx->input(1);
        CHECK(IsValidateI128Tensor(op0.shape()));
        CHECK(IsValidateI128Tensor(op1.shape()));
        tf_i128::I128TensorView view0(op0);
        tf_i128::I128TensorView view1(op1);
        Tensor* output;
        //! Shape of view has no the N_LIMBS dimension.
        TensorShape out_shape {op0.dims() > op1.dims() ? view0.shape() : view1.shape()};
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
        tf_i128::i128TensorEqual(*output, view0, view1);
    }
};

class I128BitGatherOp : public OpKernel {
    int start_;
    int stride_;
public:
    explicit I128BitGatherOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("start", &start_));
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
    }
    void Compute(OpKernelContext *ctx) override {
        const Tensor& op0 = ctx->input(0);
        CHECK(IsValidateI128Tensor(op0.shape()));
        Tensor* output;
        TensorShape out_shape {op0.shape()};
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
        tf_i128::I128TensorView view0(op0);
        tf_i128::i128TensorGatherBits(*output, view0, start_, stride_);
    }
};

class I128BitSplitAndGatherOp : public OpKernel {
    int stride_;
public:
    explicit I128BitSplitAndGatherOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
        OP_REQUIRES(context, 128%stride_ == 0, tensorflow::errors::InvalidArgument("Bit length of dtype is not a multiple of stride"));
    }
    void Compute(OpKernelContext *ctx) override {
        const Tensor& op0 = ctx->input(0);
        CHECK(IsValidateI128Tensor(op0.shape()));
        int input_num_elements = op0.NumElements() / 2;

        Tensor *output;
        TensorShape out_shape {op0.shape()};
        out_shape.InsertDim(0, stride_);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
        int output_num_elements = output->NumElements() / 2;

        tf_i128::I128TensorView view0(op0);
        const u128_t* src = (const u128_t*)view0.data();
        const u128_t* end = src + view0.numElements();
        u128_t* dst       = (u128_t*)output->flat<tensorflow::int64>().data();

        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        auto worker_func = [=, &op0](tensorflow::int64 lo, tensorflow::int64 hi) {
            OP_REQUIRES(ctx, lo % input_num_elements == 0, tensorflow::errors::InvalidArgument("Task separation is invalid"));
            int start = lo / input_num_elements;
            for (int k = 0; k < input_num_elements; k++) {
                const u128_t one {1};
                u128_t ans {0};
                for (long d = start, i = 0; d < 128; d += stride_, i++) {
                    if ((src[k] >> d) & 1) 
                        ans |= (one << i);
                }
                dst[lo + k] = ans;
            }
        };
        
        if (worker_threads.workers->NumThreads() > 1) {
            worker_threads.workers->TransformRangeConcurrently(
                    input_num_elements,
                    output_num_elements, 
                    worker_func);
        }
        else {
            for (int i = 0; i < stride_; i++)
                worker_func(input_num_elements * i, input_num_elements * (i+1));
        }
    }
};

REGISTER_OP("I128Add")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("I128Sub")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("I128Mul")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("I128Equal")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
        shape_inference::ShapeHandle out;
        TF_RETURN_IF_ERROR(shape_inference::BroadcastBinaryOpOutputShapeFnHelper(c, c->input(0), c->input(1), true, &out));
        auto dims = _takeDims(c, out);
        CHECK(!dims.empty());
        dims.pop_back();
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });

REGISTER_OP("I128Negate").Input("op0: int64").Output("output: int64").SetShapeFn(shape_inference::UnchangedShape);
REGISTER_OP("I128Abs").Input("op0: int64").Output("output: int64").SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128MatMul")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
        std::vector<shape_inference::DimensionHandle> dims;
        dims.push_back(c->Dim(c->input(0), 0));
        dims.push_back(c->Dim(c->input(1), 1));
        dims.push_back(c->MakeDim(tf_i128::N_LIMBS));
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });

REGISTER_OP("I128LeftShift")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128RightShift")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128LogicRightShift")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128ReduceSum").Input("op0: int64").Input("axis: int64").Input("keepdims: bool").Output("output: int64");

REGISTER_OP("I128MulScalar")
    .Input("op0: int64")
    .Input("op1: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ToI128")
    .Input("op0: float64")
    .Input("scale: int64")
    .Output("output: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
        auto dims = _takeDims(c, c->input(0));
        dims.push_back(c->MakeDim(tf_i128::N_LIMBS));
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });

REGISTER_OP("FromI128")
    .Input("op0: int64")
    .Input("scale: int64")
    .Output("output: float64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        if (!c) return errors::Internal("empty shape_inference::InferenceContext pointer");
        auto dims = _takeDims(c, c->input(0));
        if (!dims.empty()) {
            dims.pop_back();
        }
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });

REGISTER_OP("I128BitReverse")
    .Input("op0: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128BitGather")
    .Input("op0: int64")
    .Output("output: int64")
    .Attr("start: int")
    .Attr("stride: int")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I128BitSplitAndGather")
    .Input("op0: int64")
    .Output("output: int64")
    .Attr("stride: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        if (!c) return tensorflow::errors::Internal("empty shape_inference::InferenceContext pointer");
        tensorflow::int32 rank = c->Rank(c->input(0));
        std::vector<shape_inference::DimensionHandle> dims;
        int stride;
        c->GetAttr("stride", &stride);
        dims.emplace_back(c->MakeDim(stride));
        for (tensorflow::int32 i = 0; i < rank; ++i) dims.emplace_back(c->Dim(c->input(0), i));
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });

REGISTER_OP("I128XorIndices")
    .Input("op0: int64")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("I128Add").Device(DEVICE_CPU), I128AddOp);
REGISTER_KERNEL_BUILDER(Name("I128Sub").Device(DEVICE_CPU), I128SubOp);
REGISTER_KERNEL_BUILDER(Name("I128Mul").Device(DEVICE_CPU), I128MulOp);
REGISTER_KERNEL_BUILDER(Name("I128Equal").Device(DEVICE_CPU), I128EqualOp);
REGISTER_KERNEL_BUILDER(Name("I128Negate").Device(DEVICE_CPU), I128NegateOp);
REGISTER_KERNEL_BUILDER(Name("I128Abs").Device(DEVICE_CPU), I128AbsOp);
REGISTER_KERNEL_BUILDER(Name("I128MatMul").Device(DEVICE_CPU), I128MatMulOp);
REGISTER_KERNEL_BUILDER(Name("I128LeftShift").Device(DEVICE_CPU), I128LeftShiftOp);
REGISTER_KERNEL_BUILDER(Name("I128RightShift").Device(DEVICE_CPU), I128RightShiftOp);
REGISTER_KERNEL_BUILDER(Name("I128LogicRightShift").Device(DEVICE_CPU), I128LogicRightShiftOp);
REGISTER_KERNEL_BUILDER(Name("I128ReduceSum").Device(DEVICE_CPU), I128ReduceSumOp);
REGISTER_KERNEL_BUILDER(Name("ToI128").Device(DEVICE_CPU), ToI128Op);
REGISTER_KERNEL_BUILDER(Name("FromI128").Device(DEVICE_CPU), FromI128Op);
REGISTER_KERNEL_BUILDER(Name("I128BitReverse").Device(DEVICE_CPU), I128BitReverseOp);
REGISTER_KERNEL_BUILDER(Name("I128BitGather").Device(DEVICE_CPU), I128BitGatherOp);
REGISTER_KERNEL_BUILDER(Name("I128BitSplitAndGather").Device(DEVICE_CPU), I128BitSplitAndGatherOp);
REGISTER_KERNEL_BUILDER(Name("I128XorIndices").Device(DEVICE_CPU), I128XorIndicesOp);
