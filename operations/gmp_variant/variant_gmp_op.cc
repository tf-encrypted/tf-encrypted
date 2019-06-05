#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

#include <gmp.h>

using namespace tensorflow;

struct MpzWrapper {
 public:
  MpzWrapper();
  //MpzWrapper(string value);
  MpzWrapper(const MpzWrapper& other);

  static const char kTypeName[];
  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  string DebugString() const { return "MpzWrapper"; }

  mpz_t value;
};

MpzWrapper::MpzWrapper() {
    mpz_init(this->value);
}

MpzWrapper::MpzWrapper(const MpzWrapper& other) {
    mpz_init_set(this->value, other.value);
}

void MpzWrapper::Encode(VariantTensorData * data) const {
    size_t count_p;

    gmp_printf("HELELLELELELLE %Zd\n", value);

    char * p = (char *)mpz_export(NULL, &count_p, 1, sizeof(unsigned long), 0, 0, value);

    int total_size = count_p * sizeof(unsigned long);

    std::string s(p, total_size);

    data->set_type_name(TypeName());
    data->set_metadata(s);
}

bool MpzWrapper::Decode(const VariantTensorData& data) {
    string metadata("");
    data.get_metadata(&metadata);
    mpz_import(value, 1, 1, sizeof(unsigned long), 0, 0, metadata.c_str());

    gmp_printf("%Zd", value);

    std::cout << "HI" << std::endl;

    return true;
}

const char MpzWrapper::kTypeName[] = "MpzWrapper";

class CreateMpzVariantOp : public OpKernel {
public:
  explicit CreateMpzVariantOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(value.shape()),
        errors::InvalidArgument(
            "value expected to be a scalar ",
            "but got shape: ", value.shape().DebugString()));

    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));

    MpzWrapper i;
    mpz_init_set_str(i.value, value.scalar<string>()().c_str(), 10);

    result->scalar<Variant>()() = std::move(i);
  }
};

Status GetMpzWrapper(OpKernelContext* ctx, int index, const MpzWrapper** wrapper) {
    const Tensor& input = ctx->input(index);

    // TODO: check scalar type
    const MpzWrapper * w = input.scalar<Variant>()().get<MpzWrapper>();
    if(w == nullptr) {
        return errors::InvalidArgument(
        "Input handle is not a mpz wrapper. Saw: '",
        input.scalar<Variant>()().DebugString(), "'");
    }

    *wrapper = w;
    return Status::OK();
}

class AddMpzOp : public OpKernel {
public:
  explicit AddMpzOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const MpzWrapper * m1 = nullptr;
    OP_REQUIRES_OK(ctx, GetMpzWrapper(ctx, 0, &m1));

    const MpzWrapper * m2 = nullptr;
    OP_REQUIRES_OK(ctx, GetMpzWrapper(ctx, 1, &m2));

    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));

    MpzWrapper res;
    mpz_init(res.value);

    mpz_add(res.value, m1->value, m2->value);

    // TODO: free old memory???
    result->scalar<Variant>()() = std::move(res);
  }
};

class MpzToStringOp : public OpKernel {
public:
    explicit MpzToStringOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
        const MpzWrapper * w = nullptr;
        OP_REQUIRES_OK(ctx, GetMpzWrapper(ctx, 0, &w));

        Tensor* result;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));

        char buf[50];
        gmp_sprintf(buf, "%Zd", w->value);

        string s(buf);

        result->scalar<string>()() = s;
    }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(MpzWrapper, MpzWrapper::kTypeName);

// Input just a string for now, we make this more robust in the future
REGISTER_OP("CreateMpzVariant")
    .Input("value: string")
    .Output("mpz: variant")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("CreateMpzVariant")
  .Device(DEVICE_CPU),
  CreateMpzVariantOp);

REGISTER_OP("AddMpz")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("AddMpz")
  .Device(DEVICE_CPU),
  AddMpzOp);

REGISTER_OP("MpzToString")
    .Input("mpz: variant")
    .Output("str: string")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(
  Name("MpzToString")
  .Device(DEVICE_CPU),
  MpzToStringOp);