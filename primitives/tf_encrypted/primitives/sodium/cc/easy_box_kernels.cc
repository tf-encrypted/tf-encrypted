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
    // Retrieve inputs
    //

    const Tensor& plaintext_t = context->input(0);
    auto plaintext_data = plaintext_t.flat<T>().data();
    const unsigned char* plaintext = reinterpret_cast<const unsigned char*>(plaintext_data);

    const Tensor& nonce_t = context-> input(1);
    auto nonce_data = nonce_t.flat<tensorflow::uint8>().data();
    const unsigned char* nonce = reinterpret_cast<const unsigned char*>(nonce_data);

    const Tensor& pk_receiver_t = context->input(2);
    auto pk_receiver_data = pk_receiver_t.flat<tensorflow::uint8>().data();
    const unsigned char* pk_receiver = reinterpret_cast<const unsigned char*>(pk_receiver_data);

    const Tensor& sk_sender_t = context->input(3);
    auto sk_sender_data = sk_sender_t.flat<tensorflow::uint8>().data();
    const unsigned char* sk_sender = reinterpret_cast<const unsigned char*>(sk_sender_data);
    
    //
    // Allocate outputs
    //

    Tensor* ciphertext_t;
    TensorShape ciphertext_shape = plaintext_t.shape();
    ciphertext_shape.AddDim(sizeof(T));
    OP_REQUIRES_OK(context, context->allocate_output(0, ciphertext_shape, &ciphertext_t));
    auto ciphertext_data = ciphertext_t->flat<tensorflow::uint8>().data();
    unsigned char* ciphertext = reinterpret_cast<unsigned char*>(ciphertext_data);

    Tensor* mac_t;
    TensorShape mac_shape({crypto_box_MACBYTES});
    OP_REQUIRES_OK(context, context->allocate_output(1, mac_shape, &mac_t));
    auto mac_data = mac_t->flat<tensorflow::uint8>().data();
    unsigned char* mac = reinterpret_cast<unsigned char*>(mac_data);

    //
    // Computation
    //

    auto res = crypto_box_detached(ciphertext, mac, plaintext, ciphertext_shape.num_elements(), nonce, pk_receiver, sk_sender);
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
    // Retrieve inputs
    //

    const Tensor& ciphertext_t = context->input(0);
    auto ciphertext_data = ciphertext_t.flat<tensorflow::uint8>().data();
    const unsigned char* ciphertext = reinterpret_cast<const unsigned char*>(ciphertext_data);

    const Tensor& mac_t = context-> input(1);
    auto mac_data = mac_t.flat<tensorflow::uint8>().data();
    const unsigned char* mac = reinterpret_cast<const unsigned char*>(mac_data);

    const Tensor& nonce_t = context-> input(2);
    auto nonce_data = nonce_t.flat<tensorflow::uint8>().data();
    const unsigned char* nonce = reinterpret_cast<const unsigned char*>(nonce_data);

    const Tensor& pk_sender_t = context->input(3);
    auto pk_sender_data = pk_sender_t.flat<tensorflow::uint8>().data();
    const unsigned char* pk_sender = reinterpret_cast<const unsigned char*>(pk_sender_data);

    const Tensor& sk_receiver_t = context->input(4);
    auto sk_receiver_data = sk_receiver_t.flat<tensorflow::uint8>().data();
    const unsigned char* sk_receiver = reinterpret_cast<const unsigned char*>(sk_receiver_data);
    
    //
    // Allocate outputs
    //

    Tensor* plaintext_t;
    auto ciphertext_shape = ciphertext_t.shape();
    auto last_dim_index = ciphertext_shape.dims() - 1;
    auto last_dim = ciphertext_shape.dim_size(last_dim_index);
    OP_REQUIRES(context, last_dim == sizeof(T),
        errors::Internal("Last dim of ciphertext_shape should equal ", sizeof(T)));
    TensorShape plaintext_shape = ciphertext_shape;
    plaintext_shape.RemoveLastDims(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, plaintext_shape, &plaintext_t));
    auto plaintext_data = plaintext_t->flat<T>().data();
    unsigned char* plaintext = reinterpret_cast<unsigned char*>(plaintext_data);

    //
    // Computation
    //

    auto res = crypto_box_open_detached(plaintext, ciphertext, mac, ciphertext_t.shape().num_elements(), nonce, pk_sender, sk_receiver);
    OP_REQUIRES(context, res == 0, errors::Internal("libsodium open operation failed"));
  }
};


REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxGenKeypair").Device(DEVICE_CPU), SodiumEasyBoxGenKeypair);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxGenNonce").Device(DEVICE_CPU), SodiumEasyBoxGenNonce);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<float>("plaintext_type"), 
    SodiumEasyBoxSealDetached<float>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxSealDetached").Device(DEVICE_CPU).TypeConstraint<uint8>("plaintext_type"), 
    SodiumEasyBoxSealDetached<uint8>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<float>("plaintext_type"), 
    SodiumEasyBoxOpenDetached<float>);
REGISTER_KERNEL_BUILDER(Name("SodiumEasyBoxOpenDetached").Device(DEVICE_CPU).TypeConstraint<uint8>("plaintext_type"), 
    SodiumEasyBoxOpenDetached<uint8>);
