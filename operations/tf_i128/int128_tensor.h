#ifndef TF_I128_TENSOR_H
#define TF_I128_TENSOR_H
#include <memory>  // unique_ptr
#include <array>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "Eigen/Core"
#include "Eigen/Dense"

#ifdef __APPLE__
using i128_t = __int128_t;
using u128_t = __uint128_t;
#else
using i128_t = int __attribute__((mode(TI)));
using u128_t = unsigned int __attribute__((mode(TI)));
#endif

namespace tf_i128 {
namespace tf = tensorflow;
constexpr int N_LIMBS = 2; // number of underlying limbs
// \breif A view (i.e., no holding data) for int128 2D tensor.
struct I128TensorView {
    using Scalar     = i128_t;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatView    = Eigen::Map<MatrixType>;

    template <int N>
    using TensorType = Eigen::Tensor<Scalar, N, Eigen::RowMajor>;

    template <int N>
    using TensorView = Eigen::TensorMap< TensorType<N> >; 


    explicit I128TensorView(const tf::Tensor &source);

    // NOTICE the shape is in the view of int128, not the shape of source.
    const tf::TensorShape &shape() const { return shape_; }

    const MatView &view() const { return *mat_view_; }

    int dims () const { return shape_.dims();}

    template<int N>
    std::array<int64_t, N> dim_array () const;

    /**
    Return the tensor's original view of int128.
    */
    template<int N>
    TensorView<N> view () const; 

    template<int N>
    TensorView<N> view (const std::array<int64_t, N>& reshape) const;

    /** 
    Return the tensor view in the given shape.

    @param reshape The target shape of the return view.
    */
    template<int N>
    TensorView<N> view (const tf::TensorShape& reshape) const;

    const Scalar *data() const { return data_; }
    Scalar *data() { return data_; }
    long numElements() const { return shape().num_elements(); }
    bool isScalar() const { return numElements() == 1; }


    Scalar* data_;
    tf::TensorShape shape_;
    std::shared_ptr<MatView> mat_view_;
};

bool i128TensorMatmul(tf::Tensor &out, I128TensorView const &lhs, I128TensorView const &rhs);

bool i128TensorGatherBits(tf::Tensor &out, I128TensorView const &in, int start, int stride);

bool i128TensorEqual(tf::Tensor &out, I128TensorView const &lhs, I128TensorView const &rhs);

//! broadcast operations
bool i128TensorEqualScalar(tf::Tensor &out, I128TensorView const &lhs, I128TensorView::Scalar const &rhs);

bool i128TensorConvert(tf::Tensor &out, tf::Tensor const &in, tf::uint64 scale);
bool i128TensorConvertBack(tf::Tensor &out, tf::Tensor const &in, tf::uint64 scale);


bool i128TensorMul(tf::Tensor &out, tf::Tensor const &lhs, tf::Tensor const &rhs);
bool i128TensorAdd(tf::Tensor &out, tf::Tensor const &lhs, tf::Tensor const &rhs);
bool i128TensorSub(tf::Tensor &out, tf::Tensor const &lhs, tf::Tensor const &rhs);
bool i128TensorLeftShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt);
bool i128TensorRightShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt);
bool i128TensorLogicalRightShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt);

bool i128TensorReduceSum(tf::Tensor& out, tf::Tensor const& in, const long* axis_vec, long axis_num, bool keepdims);

bool i128TensorNegate(tf::Tensor& out, tf::Tensor const& in); 
bool i128TensorAbs(tf::Tensor &out, tf::Tensor const &in);
bool i128TensorBitReverse(tf::Tensor &out, tf::Tensor const &in);
// For the bits in an int128 integer, XOR the position that with bit set, i.e., (XOR_{i} i) where the i-th bit is 1.
bool i128TensorXorIndices(tf::Tensor &out, tf::Tensor const &in);
}  // namespace tf_i128
#endif  // TF_INT128_INT128_TENSOR_H
