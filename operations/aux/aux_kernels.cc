#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include <type_traits>

namespace tf = tensorflow;
using tf::OpKernel;
using tf::OpKernelContext;
using tf::OpKernelConstruction;
using tf::Tensor;
using tf::shape_inference::UnchangedShape;
using tf::shape_inference::DimensionHandle;
using tf::shape_inference::InferenceContext;
using tf::TensorShape;
using tf::DEVICE_CPU;
using tf::Status;

template <typename T>
class BitGatherOp : public OpKernel {
    int start_;
    int stride_;
public:
    explicit BitGatherOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("start", &start_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("stride", &stride_));
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor &op = ctx->input(0);
        Tensor *output;
        TensorShape out_shape {op.shape()};
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

        using uT = typename std::make_unsigned<T>::type;
        const uT *src = (const uT *)op.flat<T>().data();
        const uT *end = src + op.NumElements();
        T *dst = output->flat<T>().data();
        std::transform(src, end, dst, [&](uT v) -> T {
            const uT one {1};
            uT ans {0};
            for (long d = start_, i = 0; d < sizeof(T) * 8; d += stride_, i++) {
                if ((v >> d) & 1)  
                    ans |= (one << i);
            }
            return ans;
        });
    }
};

template <typename T>
class BitSplitAndGatherOp : public OpKernel {
    int stride_;
public:
    explicit BitSplitAndGatherOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("stride", &stride_));
        OP_REQUIRES(ctx, sizeof(T)*8%stride_ == 0, tf::errors::InvalidArgument("Bit length of dtype is not a multiple of stride"));
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor &op = ctx->input(0);
        Tensor *output;
        TensorShape out_shape {op.shape()};
        out_shape.InsertDim(0, stride_);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

        using uT = typename std::make_unsigned<T>::type;
        const uT *src = (const uT *)op.flat<T>().data();
        const uT *end = src + op.NumElements();
        T *dst = output->flat<T>().data();

        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        auto worker_func = [=, &op](tf::int64 lo, tf::int64 hi) {
            OP_REQUIRES(ctx, lo % op.NumElements() == 0, tf::errors::InvalidArgument("Task separation is invalid"));
            int start = lo / op.NumElements();
            for (int k = 0; k < op.NumElements(); k++) {
                const uT one {1};
                uT ans {0};
                for (long d = start, i = 0; d < sizeof(T) * 8; d += stride_, i++) {
                    if ((src[k] >> d) & 1) 
                        ans |= (one << i);
                }
                dst[lo + k] = ans;
            }
        };
        
        if (worker_threads.workers->NumThreads() > 1) {
            worker_threads.workers->TransformRangeConcurrently(
                    op.NumElements(),
                    output->NumElements(), 
                    worker_func);
        }
        else {
            for (int i = 0; i < stride_; i++)
                worker_func(op.NumElements() * i, op.NumElements() * (i+1));
        }
    }
};


REGISTER_OP("BitGather")
    .Input("op: dtype")
    .Output("output: dtype")
    .Attr("start: int")
    .Attr("stride: int")
    .Attr("dtype: {int8, int16, int32, int64}")
    .SetShapeFn(UnchangedShape);

REGISTER_OP("BitSplitAndGather")
    .Input("op: dtype")
    .Output("output: dtype")
    .Attr("stride: int")
    .Attr("dtype: {int8, int16, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
        if (!c) return tf::errors::Internal("empty shape_inference::InferenceContext pointer");
        tf::int32 rank = c->Rank(c->input(0));
        std::vector<DimensionHandle> dims;
        int stride;
        c->GetAttr("stride", &stride);
        dims.emplace_back(c->MakeDim(stride));
        for (tf::int32 i = 0; i < rank; ++i) dims.emplace_back(c->Dim(c->input(0), i));
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });


REGISTER_KERNEL_BUILDER(
        Name("BitGather").Device(DEVICE_CPU).template TypeConstraint<tf::int8>("dtype"), 
        BitGatherOp<tf::int8>);
REGISTER_KERNEL_BUILDER(
        Name("BitGather").Device(DEVICE_CPU).template TypeConstraint<tf::int16>("dtype"), 
        BitGatherOp<tf::int16>);
REGISTER_KERNEL_BUILDER(
        Name("BitGather").Device(DEVICE_CPU).template TypeConstraint<tf::int32>("dtype"), 
        BitGatherOp<tf::int32>);
REGISTER_KERNEL_BUILDER(
        Name("BitGather").Device(DEVICE_CPU).template TypeConstraint<tf::int64>("dtype"), 
        BitGatherOp<tf::int64>);

REGISTER_KERNEL_BUILDER(
        Name("BitSplitAndGather").Device(DEVICE_CPU).template TypeConstraint<tf::int8>("dtype"), 
        BitSplitAndGatherOp<tf::int8>);
REGISTER_KERNEL_BUILDER(
        Name("BitSplitAndGather").Device(DEVICE_CPU).template TypeConstraint<tf::int16>("dtype"), 
        BitSplitAndGatherOp<tf::int16>);
REGISTER_KERNEL_BUILDER(
        Name("BitSplitAndGather").Device(DEVICE_CPU).template TypeConstraint<tf::int32>("dtype"), 
        BitSplitAndGatherOp<tf::int32>);
REGISTER_KERNEL_BUILDER(
        Name("BitSplitAndGather").Device(DEVICE_CPU).template TypeConstraint<tf::int64>("dtype"), 
        BitSplitAndGatherOp<tf::int64>);

