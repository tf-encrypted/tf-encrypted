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

using u32 = tensorflow::uint32;
using u64 = tensorflow::uint64;
using i64 = tensorflow::int64;

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

class BitReverseOp : public OpKernel {
public:
    explicit BitReverseOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext *ctx) override {
        const Tensor &op0 = ctx->input(0);
        Tensor *output;
        TensorShape out_shape {op0.shape()};
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

        auto u32_reverse = [](u32 v) -> u32 {
            v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
            // swap consecutive pairs
            v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
            // swap nibbles ...
            v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
            // swap bytes
            v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
            // swap 2-byte long pairs
            v = (v >> 16) | (v << 16);
            return v;
        };

        const u64 *src = (const u64 *)op0.flat<i64>().data();
        const u64 *end = src + op0.NumElements();
        i64 *dst       = output->flat<i64>().data();
        std::transform(src, end, dst, [u32_reverse](u64 v) -> i64 {
            u32 *raw = (u32 *)&v;
            raw[0]   = u32_reverse(raw[0]);
            raw[1]   = u32_reverse(raw[1]);
            std::swap(raw[0], raw[1]);
            return (i64)v;
        });
    }
};


class XorIndicesOp : public OpKernel {
public:
    explicit XorIndicesOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext *ctx) override {
        const Tensor &op0 = ctx->input(0);
        Tensor *output;
        TensorShape out_shape {op0.shape()};
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

        const u64 *src = (const u64 *)op0.flat<i64>().data();
        const u64 *end = src + op0.NumElements();
        u64 *dst       = (u64 *)output->flat<i64>().data();
        std::transform(src, end, dst, [](u64 v) -> u64 {
            long ans {0};
            for (long d = 0; v > 0 && d < 64; ++d) {
                if (v & 1) ans ^= d;
                v >>= 1;
            }
            return (u64)ans;
        });
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

REGISTER_OP("BitReverse").Input("op0: int64").Output("output: int64").SetShapeFn(UnchangedShape);

REGISTER_OP("XorIndices").Input("op0: int64").Output("output: int64").SetShapeFn(UnchangedShape);


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

REGISTER_KERNEL_BUILDER(Name("BitReverse").Device(DEVICE_CPU), BitReverseOp);
REGISTER_KERNEL_BUILDER(Name("XorIndices").Device(DEVICE_CPU), XorIndicesOp);

