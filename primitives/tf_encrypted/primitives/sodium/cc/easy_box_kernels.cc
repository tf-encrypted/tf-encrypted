// See libsodium documentation at https://download.libsodium.org/doc/public-key_cryptography/authenticated_encryption

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "sodium.h"

using namespace tensorflow;

class SodiumEasyBoxGenKeypair : public OpKernel {
public:
  explicit SodiumEasyBoxGenKeypair(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize"));

    //
    // Allocate outputs
    //

    Tensor* pk_t;
    TensorShape pk_shape({crypto_box_PUBLICKEYBYTES});
    OP_REQUIRES_OK(context, context->allocate_output(0, pk_shape, &pk_t));
    auto pk_data = pk_t->flat<tensorflow::uint8>().data();
    unsigned char* pk = reinterpret_cast<unsigned char*>(pk_data);
    
    Tensor* sk_t;
    TensorShape sk_shape({crypto_box_SECRETKEYBYTES});
    OP_REQUIRES_OK(context, context->allocate_output(1, sk_shape, &sk_t));
    auto sk_data = sk_t->flat<tensorflow::uint8>().data();
    unsigned char* sk = reinterpret_cast<unsigned char*>(sk_data);

    //
    // Computation
    //

    auto res = crypto_box_keypair(pk, sk);
    OP_REQUIRES(context, res == 0, errors::Internal("libsodium keypair operation failed"));
  }
};

class SodiumEasyBoxGenNonce : public OpKernel {
public:
  explicit SodiumEasyBoxGenNonce(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize"));

    //
    // Allocate outputs
    //

    Tensor* nonce_t;
    TensorShape nonce_shape({crypto_box_NONCEBYTES});
    OP_REQUIRES_OK(context, context->allocate_output(0, nonce_shape, &nonce_t));
    auto nonce_data = nonce_t->flat<tensorflow::uint8>().data();
    unsigned char* nonce = reinterpret_cast<unsigned char*>(nonce_data);

    //
    // Computation
    //

    randombytes_buf(nonce, crypto_box_NONCEBYTES);
  }
};

template <typename T>
class SodiumEasyBoxSealDetached : public OpKernel {
public:
  explicit SodiumEasyBoxSealDetached(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize"));

    //
    // Retrieve inputs and verify shapes
    //

    const Tensor& plaintext_t = context->input(0);
    const unsigned char* plaintext_data = reinterpret_cast<const unsigned char*>(plaintext_t.flat<T>().data());

    const Tensor& nonce_t = context->input(1);
    const auto nonce_flat = nonce_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, nonce_flat.size() == crypto_box_NONCEBYTES,
        errors::Internal("Nonce is not of required size ", crypto_box_NONCEBYTES));
    const unsigned char* nonce_data = reinterpret_cast<const unsigned char*>(nonce_flat.data());

    const Tensor& pk_receiver_t = context->input(2);
    const auto pk_receiver_flat = pk_receiver_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, pk_receiver_flat.size() == crypto_box_PUBLICKEYBYTES,
        errors::Internal("Public key is not of required size ", crypto_box_PUBLICKEYBYTES));
    const unsigned char* pk_receiver_data = reinterpret_cast<const unsigned char*>(pk_receiver_flat.data());

    const Tensor& sk_sender_t = context->input(3);
    const auto sk_receiver_flat = sk_sender_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, sk_receiver_flat.size() == crypto_box_SECRETKEYBYTES,
        errors::Internal("Secret key is not of required size ", crypto_box_SECRETKEYBYTES));
    const unsigned char* sk_sender_data = reinterpret_cast<const unsigned char*>(sk_receiver_flat.data());

    //
    // Allocate outputs
    //

    Tensor* ciphertext_t;
    TensorShape ciphertext_shape = plaintext_t.shape();
    ciphertext_shape.AddDim(sizeof(T));
    OP_REQUIRES_OK(context, context->allocate_output(0, ciphertext_shape, &ciphertext_t));
    auto ciphertext_flat = ciphertext_t->flat<tensorflow::uint8>();
    unsigned char* ciphertext_data = reinterpret_cast<unsigned char*>(ciphertext_flat.data());

    Tensor* mac_t;
    TensorShape mac_shape({crypto_box_MACBYTES});
    OP_REQUIRES_OK(context, context->allocate_output(1, mac_shape, &mac_t));
    auto mac_flat = mac_t->flat<tensorflow::uint8>();
    unsigned char* mac_data = reinterpret_cast<unsigned char*>(mac_flat.data());

    //
    // Computation
    //

    auto plaintext_size = ciphertext_flat.size();
    auto res = crypto_box_detached(ciphertext_data, mac_data, plaintext_data, plaintext_size,
        nonce_data, pk_receiver_data, sk_sender_data);
    OP_REQUIRES(context, res == 0, errors::Internal("libsodium seal operation failed"));
  }
};

template <typename T>
class SodiumEasyBoxOpenDetached : public OpKernel {
public:
  explicit SodiumEasyBoxOpenDetached(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, sodium_init() >= 0, errors::Internal("libsodium failed to initialize"));

    //
    // Retrieve inputs and verify shapes
    //

    const Tensor& ciphertext_t = context->input(0);
    TensorShape ciphertext_shape = ciphertext_t.shape();
    OP_REQUIRES(context, ciphertext_shape.dim_size(ciphertext_shape.dims() - 1) == sizeof(T),
        errors::Internal("Last dim of ciphertext should equal ", sizeof(T)));
    const auto ciphertext_flat = ciphertext_t.flat<tensorflow::uint8>();
    const unsigned char* ciphertext_data = reinterpret_cast<const unsigned char*>(ciphertext_flat.data());

    const Tensor& mac_t = context->input(1);
    auto mac_flat = mac_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, mac_flat.size() == crypto_box_MACBYTES,
        errors::Internal("Mac is not of required size ", crypto_box_MACBYTES));
    const unsigned char* mac_data = reinterpret_cast<const unsigned char*>(mac_flat.data());

    const Tensor& nonce_t = context->input(2);
    const auto nonce_flat = nonce_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, nonce_flat.size() == crypto_box_NONCEBYTES,
        errors::Internal("Nonce is not of required size ", crypto_box_NONCEBYTES));
    const unsigned char* nonce_data = reinterpret_cast<const unsigned char*>(nonce_flat.data());

    const Tensor& pk_sender_t = context->input(3);
    const auto pk_sender_flat = pk_sender_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, pk_sender_flat.size() == crypto_box_PUBLICKEYBYTES,
        errors::Internal("Public key is not of required size ", crypto_box_PUBLICKEYBYTES));
    const unsigned char* pk_sender_data = reinterpret_cast<const unsigned char*>(pk_sender_flat.data());

    const Tensor& sk_receiver_t = context->input(4);
    const auto sk_receiver_flat = sk_receiver_t.flat<tensorflow::uint8>();
    OP_REQUIRES(context, sk_receiver_flat.size() == crypto_box_SECRETKEYBYTES,
        errors::Internal("Secret key is not of required size ", crypto_box_SECRETKEYBYTES));
    const unsigned char* sk_receiver_data = reinterpret_cast<const unsigned char*>(sk_receiver_flat.data());

    //
    // Allocate outputs
    //

    Tensor* plaintext_t;
    TensorShape plaintext_shape = ciphertext_shape;
    plaintext_shape.RemoveLastDims(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, plaintext_shape, &plaintext_t));
    auto plaintext_flat = plaintext_t->flat<T>();
    unsigned char* plaintext_data = reinterpret_cast<unsigned char*>(plaintext_flat.data());

    //
    // Computation
    //

    const auto ciphertext_size = ciphertext_flat.size();
    auto res = crypto_box_open_detached(plaintext_data, ciphertext_data, mac_data, ciphertext_size, nonce_data, pk_sender_data, sk_receiver_data);
    OP_REQUIRES(context, res == 0, errors::Internal("libsodium open operation failed"));
  }
};


REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxGenKeypair").Device(DEVICE_CPU), SodiumEasyBoxGenKeypair);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxGenNonce").Device(DEVICE_CPU), SodiumEasyBoxGenNonce);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<bfloat16>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<bfloat16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<float>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<float>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<double>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<double>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<int8>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<int8>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<int16>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<int16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<int32>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<int32>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<int64>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<int64>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<uint8>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<uint8>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<uint16>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<uint16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<uint32>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<uint32>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<uint64>("plaintext_dtype"), 
    SodiumEasyBoxSealDetached<uint64>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<bfloat16>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<bfloat16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<float>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<float>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<double>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<double>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<int8>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<int8>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<int16>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<int16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<int32>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<int32>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<int64>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<int64>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<uint8>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<uint8>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<uint16>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<uint16>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<uint32>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<uint32>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<uint64>("plaintext_dtype"), 
    SodiumEasyBoxOpenDetached<uint64>);
