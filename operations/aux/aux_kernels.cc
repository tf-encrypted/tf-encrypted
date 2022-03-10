#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <type_traits>

namespace tf = tensorflow;
using tf::OpKernel;
using tf::OpKernelContext;
using tf::OpKernelConstruction;
using tf::Tensor;
using tf::shape_inference::UnchangedShape;
using tf::TensorShape;
using tf::DEVICE_CPU;

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


REGISTER_OP("BitGather")
    .Input("op: dtype")
    .Output("output: dtype")
    .Attr("start: int")
    .Attr("stride: int")
    .Attr("dtype: {int8, int16, int32, int64}")
    .SetShapeFn(UnchangedShape);


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
