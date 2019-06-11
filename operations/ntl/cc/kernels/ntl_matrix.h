#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

#include <NTL/ZZ.h>
#include <NTL/vector.h>
#include <NTL/matrix.h>
#include <NTL/tools.h>
#include <NTL/ZZ_p.h>

using namespace tensorflow;

struct NTLMatrix {
 public:
  NTLMatrix();
  NTLMatrix(const NTLMatrix& other);

  NTLMatrix(int n, int m);

  static const char kTypeName[];
  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  string DebugString() const { return "NTLMatrix"; }

  NTL::Mat<NTL::ZZ_p> m;
};