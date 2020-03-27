#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

//
// Easy Box ops
//

REGISTER_OP("SodiumEasyBoxGenKeypair")
    .Output("pk: uint8")
    .Output("sk: uint8")
    .SetIsStateful();

REGISTER_OP("SodiumEasyBoxGenNonce")
    .Output("nonce: uint8")
    .SetIsStateful();

REGISTER_OP("SodiumEasyBoxSealDetached")
    .Input("plaintext: uint8")
    .Input("nonce: uint8")
    .Input("pk_receiver: uint8")
    .Input("sk_sender: uint8")
    .Output("ciphertext: uint8")
    .Output("mac: uint8")
    .SetIsStateful();

REGISTER_OP("SodiumEasyBoxOpenDetached")
    .Input("ciphertext: uint8")
    .Input("mac: uint8")
    .Input("nonce: uint8")
    .Input("pk_sender: uint8")
    .Input("sk_receiver: uint8")
    .Output("plaintext: uint8")
    .SetIsStateful();
