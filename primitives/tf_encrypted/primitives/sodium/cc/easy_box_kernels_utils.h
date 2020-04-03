
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


template <typename T>
TensorShape makeShapeSeal(const TensorShape& plaintext_shape);

template <>
inline TensorShape makeShapeSeal<float>(const TensorShape& plaintext_shape){
  auto ciphertext_shape = plaintext_shape;
  ciphertext_shape.AddDim(sizeof(float));
  return ciphertext_shape;
}

template <>
inline TensorShape makeShapeSeal<uint8>(const TensorShape& plaintext_shape){
  auto ciphertext_shape = plaintext_shape;
  ciphertext_shape.AddDim(sizeof(uint8));
  return ciphertext_shape;
}

template <typename T>
TensorShape makeShapeOpen(const TensorShape& ciphertext);

template <>
inline TensorShape makeShapeOpen<float>(const TensorShape& ciphertext_shape){
  TensorShape plaintext_shape = ciphertext_shape;
  plaintext_shape.RemoveLastDims(1);
  return plaintext_shape;
}

template <>
inline TensorShape makeShapeOpen<uint8>(const TensorShape& ciphertext_shape){
  TensorShape plaintext_shape = ciphertext_shape;
  plaintext_shape.RemoveLastDims(1);
  return plaintext_shape;
}
