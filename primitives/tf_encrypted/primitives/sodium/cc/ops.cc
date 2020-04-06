#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "sodium.h"

using namespace tensorflow;

//
// Easy Box ops
//

REGISTER_OP("SodiumEasyBoxGenKeypair")
    .Output("pk: uint8")
    .Output("sk: uint8")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context){
        shape_inference::ShapeHandle pk_shape = context->MakeShape({crypto_box_PUBLICKEYBYTES});
        context->set_output(0, pk_shape);

        shape_inference::ShapeHandle sk_shape = context->MakeShape({crypto_box_SECRETKEYBYTES});
        context->set_output(1, sk_shape);

        return Status::OK();
    });;

REGISTER_OP("SodiumEasyBoxGenNonce")
    .Output("nonce: uint8")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context){
        shape_inference::ShapeHandle nonce_shape = context->MakeShape({crypto_box_NONCEBYTES});
        context->set_output(0, nonce_shape);
        return Status::OK();
    });

REGISTER_OP("SodiumEasyBoxSealDetached")
    .Attr("plaintext_dtype: {uint8, float32}")
    .Input("plaintext: plaintext_dtype")
    .Input("nonce: uint8")
    .Input("pk_receiver: uint8")
    .Input("sk_sender: uint8")
    .Output("ciphertext: uint8")
    .Output("mac: uint8")
    .SetIsStateful();

REGISTER_OP("SodiumEasyBoxOpenDetached")
    .Attr("plaintext_dtype: {uint8, float32}")
    .Input("ciphertext: uint8")
    .Input("mac: uint8")
    .Input("nonce: uint8")
    .Input("pk_sender: uint8")
    .Input("sk_receiver: uint8")
    .Output("plaintext: plaintext_dtype")
    .SetIsStateful();
