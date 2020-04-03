
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
TensorShape makeShapeSeal(const Tensor& ciphertext){
  auto ciphertext_shape = ciphertext.shape();
  return ciphertext_shape;
}

template <>
inline TensorShape makeShapeSeal<float>(const Tensor& ciphertext){
  auto ciphertext_shape = ciphertext.shape();
  ciphertext_shape.AddDim(sizeof(float));
  return ciphertext_shape;
}

template <>
TensorShape makeShapeSeal<uint8>(const Tensor& ciphertext){
  auto ciphertext_shape = ciphertext.shape();
  return ciphertext_shape;
}

template <typename T>
TensorShape makeShapeOpen(const Tensor& plaintext){
  TensorShape plaintext_shape = plaintext.shape();
  return plaintext_shape;
}

template <>
TensorShape makeShapeOpen<float>(const Tensor& plaintext){
	TensorShape plaintext_shape = plaintext.shape();
  plaintext_shape.RemoveLastDims(1);
  return plaintext_shape;
}

template <>
TensorShape makeShapeOpen<uint8>(const Tensor& plaintext){
  TensorShape plaintext_shape = plaintext.shape();
  return plaintext_shape;
}
