#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

REGISTER_OP("CreateNTLMatrix")
    .Attr("T: {int32, int64, string}")
    .Input("value: T")
    .Input("modulus: int64")
    .Output("ntl: variant")
    .SetIsStateful();

REGISTER_OP("MatMulNTL")
    .Input("val1: variant")
    .Input("val2: variant")
    .Input("modulus: int64")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_OP("NTLToNative")
    .Attr("T: {int32, int64, string}")
    .Input("ntl: variant")
    .Output("native: T")
    .SetIsStateful();
