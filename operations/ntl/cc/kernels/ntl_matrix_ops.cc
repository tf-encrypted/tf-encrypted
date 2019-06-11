#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ.h>

#include "ntl_matrix.h"

using namespace tensorflow;

class InitModulusOp : public OpKernel {
public:
  explicit InitModulusOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(0);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(value.shape()),
        errors::InvalidArgument(
            "value expected to be a 2d matrix ",
            "but got shape: ", value.shape().DebugString()));

  }
};

class CreateNTLMatrixOp : public OpKernel {
public:
  explicit CreateNTLMatrixOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(0);
    const Tensor& mod = ctx->input(1);

    // TODO support vector?? scalar??
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(value.shape()),
        errors::InvalidArgument(
            "value expected to be a 2d matrix ",
            "but got shape: ", value.shape().DebugString()));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(mod.shape()),
        errors::InvalidArgument(
            "value expected to be a scalar ",
            "but got shape: ", mod.shape().DebugString()));

    NTL::ZZ_p::init(NTL::ZZ(mod.scalar<int64>()()));

    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));

    auto rows = value.dim_size(0), cols = value.dim_size(1);
    NTLMatrix m(rows, cols);

    auto mat = value.matrix<string>();

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            m.m[i][j] = NTL::conv<NTL::ZZ_p>(mat(i, j).c_str());
        }
    }

    result->scalar<Variant>()() = std::move(m);
  }
};

Status GetNTLMatrix(OpKernelContext* ctx, int index, const NTLMatrix** wrapper) {
    const Tensor& input = ctx->input(index);

    // TODO: check scalar type
    const NTLMatrix * w = input.scalar<Variant>()().get<NTLMatrix>();
    if(w == nullptr) {
        return errors::InvalidArgument(
        "Input handle is not a ntl wrapper. Saw: '",
        input.scalar<Variant>()().DebugString(), "'");
    }

    *wrapper = w;
    return Status::OK();
}

class MatMulNTLOp : public OpKernel {
public:
  explicit MatMulNTLOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const NTLMatrix * m1 = nullptr;
    OP_REQUIRES_OK(ctx, GetNTLMatrix(ctx, 0, &m1));

    const NTLMatrix * m2 = nullptr;
    OP_REQUIRES_OK(ctx, GetNTLMatrix(ctx, 1, &m2));

    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));

    NTLMatrix res;

    res.m = m1->m * m2->m;

    // TODO: free old memory (m1, m2)???
    result->scalar<Variant>()() = std::move(res);
  }
};

class NTLToStringOp : public OpKernel {
public:
    explicit NTLToStringOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      const NTLMatrix * w = nullptr;
      OP_REQUIRES_OK(ctx, GetNTLMatrix(ctx, 0, &w));

      auto rows = w->m.NumRows(), cols = w->m.NumCols();

      Tensor* result;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{rows, cols}, &result));

      for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
          std::stringstream buffer;
          buffer << w->m[i][j];
          result->matrix<string>()(i, j) = buffer.str();
        }
      }
    }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(NTLMatrix, NTLMatrix::kTypeName);

// Input just a string for now, we make this more robust in the future
REGISTER_OP("CreateNTLMatrix")
    .Input("value: string")
    .Input("modulus: int64")
    .Output("ntl: variant")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("CreateNTLMatrix")
  .Device(DEVICE_CPU),
  CreateNTLMatrixOp);

REGISTER_OP("MatMulNTL")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("MatMulNTL")
  .Device(DEVICE_CPU),
  MatMulNTLOp);

REGISTER_OP("NTLToString")
    .Input("ntl: variant")
    .Output("str: string")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("NTLToString")
  .Device(DEVICE_CPU),
  NTLToStringOp);