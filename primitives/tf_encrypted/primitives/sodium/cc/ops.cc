#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

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
    .Attr("plaintext_dtype: {uint8, uint16, uint32, uint64, int8, int16, int32, int64, bfloat16, float32, float64}")
    .Input("plaintext: plaintext_dtype")
    .Input("nonce: uint8")
    .Input("pk_receiver: uint8")
    .Input("sk_sender: uint8")
    .Output("ciphertext: uint8")
    .Output("mac: uint8")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c){

        //
        // Check input shapes; also done in kernel
        //
        
        if (!c->MergeInput(1, c->MakeShape({crypto_box_NONCEBYTES}))){
            return errors::Internal("Nonce not of the required shape");
        }
        if (!c->MergeInput(2, c->MakeShape({crypto_box_PUBLICKEYBYTES}))){
            return errors::Internal("Public key not of the required shape");
        }
        if (!c->MergeInput(3, c->MakeShape({crypto_box_SECRETKEYBYTES}))){
            return errors::Internal("Secret key not of the required shape");
        }

        //
        // Compute output shapes
        //

        tensorflow::DataType plaintext_dtype;
        TF_RETURN_IF_ERROR(c->GetAttr("plaintext_dtype", &plaintext_dtype));
        int ciphertext_expansion = tensorflow::DataTypeSize(plaintext_dtype);

        shape_inference::ShapeHandle plaintext_shape = c->input(0);
        shape_inference::ShapeHandle ciphertext_shape;
        TF_RETURN_IF_ERROR(c->Concatenate(plaintext_shape, c->MakeShape({ciphertext_expansion}), &ciphertext_shape));
        c->set_output(0, ciphertext_shape);

        shape_inference::ShapeHandle mac_shape = c->MakeShape({crypto_box_MACBYTES});
        c->set_output(1, mac_shape);

        return Status::OK();
    });

REGISTER_OP("SodiumEasyBoxOpenDetached")
    .Attr("plaintext_dtype: {uint8, uint16, uint32, uint64, int8, int16, int32, int64, bfloat16, float32, float64}")
    .Input("ciphertext: uint8")
    .Input("mac: uint8")
    .Input("nonce: uint8")
    .Input("pk_sender: uint8")
    .Input("sk_receiver: uint8")
    .Output("plaintext: plaintext_dtype")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c){

        //
        // Check input shapes; also done in kernel
        //

        tensorflow::DataType plaintext_dtype;
        TF_RETURN_IF_ERROR(c->GetAttr("plaintext_dtype", &plaintext_dtype));
        int ciphertext_expansion = tensorflow::DataTypeSize(plaintext_dtype);

        shape_inference::ShapeHandle ciphertext_shape = c->input(0);
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(ciphertext_shape, 1, &ciphertext_shape));
        int64 last_dim = c->Value(c->Dim(ciphertext_shape, -1));
        if (last_dim != ciphertext_expansion) {
            return errors::Internal("Last dimension of ciphertext does not match plaintext_dtype ", last_dim);
        }

        if (!c->MergeInput(1, c->MakeShape({crypto_box_MACBYTES}))){
            return errors::Internal("Mac not of the required shape");
        }
        if (!c->MergeInput(2, c->MakeShape({crypto_box_NONCEBYTES}))){
            return errors::Internal("Nonce not of the required shape");
        }
        if (!c->MergeInput(3, c->MakeShape({crypto_box_PUBLICKEYBYTES}))){
            return errors::Internal("Public key not of the required shape");
        }
        if (!c->MergeInput(4, c->MakeShape({crypto_box_SECRETKEYBYTES}))){
            return errors::Internal("Secret key not of the required shape");
        }

        //
        // Compute output shape
        //

        shape_inference::ShapeHandle plaintext_shape;
        TF_RETURN_IF_ERROR(c->Subshape(ciphertext_shape, 0, -1, &plaintext_shape));
        c->set_output(0, plaintext_shape);

        return Status::OK();
    });
