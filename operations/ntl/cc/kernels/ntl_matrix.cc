#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <NTL/ZZ.h>
#include <NTL/vector.h>
#include <NTL/matrix.h>
#include <NTL/tools.h>
#include <NTL/ZZ_p.h>
#include <NTL/tools.h>

#include "ntl_matrix.h"

using namespace tensorflow;

NTLMatrix::NTLMatrix() : m() {}

NTLMatrix::NTLMatrix(const NTLMatrix& other) : m(other.m) {}

NTLMatrix::NTLMatrix(int n, int m) : m(NTL::INIT_SIZE, n, m) {}

void NTLMatrix::Encode(VariantTensorData * data) const {
    TensorShape shape = TensorShape{m.NumRows(), m.NumCols()};
    Tensor t(DT_STRING, shape);

    auto mat = t.matrix<string>();

    for(int i = 0; i < m.NumRows(); i++) {
        for(int j = 0; j < m.NumCols(); j++) {
          auto zz_p = m[i][j];
          auto b = NTL::NumBytes(zz_p.LoopHole());

          char buf[b];
          NTL::BytesFromZZ((unsigned char *)buf, zz_p.LoopHole(), b);

          mat(i, j).assign(buf, b);
        }
    }

    *data->add_tensors() = t;

    string metadata;

    long mod = NTL::conv<long>(m[0][0].modulus());
    core::PutVarint64(&metadata, static_cast<uint64>(mod));

    TensorShapeProto shape_proto;
    shape.AsProto(&shape_proto);
    shape_proto.AppendToString(&metadata);
    data->set_metadata(metadata);
}

bool NTLMatrix::Decode(const VariantTensorData& data) {
    auto mat = data.tensors()[0].matrix<string>();

    TensorShape shape;
    string metadata;
    data.get_metadata(&metadata);

    uint64 scratch;
    StringPiece iter(metadata);
    core::GetVarint64(&iter, &scratch);

    NTL::ZZ_p::init(NTL::ZZ((long)scratch));

    TensorShapeProto shape_proto;
    shape_proto.ParseFromString(string(iter.data(), iter.size()));
    shape = TensorShape(shape_proto);

    m.SetDims(shape.dim_size(0), shape.dim_size(1));

    for(int i = 0; i < m.NumRows(); i++) {
        for(int j = 0; j < m.NumCols(); j++) {
            unsigned char * c_str = (unsigned char *)mat(i, j).c_str();
            m[i][j] = NTL::conv<NTL::ZZ_p>(NTL::ZZFromBytes(c_str, mat(i, j).length()));
        }
    }

    return true;
}

const char NTLMatrix::kTypeName[] = "NTLMatrix";