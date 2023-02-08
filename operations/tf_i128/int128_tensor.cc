#include "./int128_tensor.h"
#include <mutex>
#include <exception>
#include "tensorflow/core/util/bcast.h"

namespace tf_i128 {
namespace tf     = tensorflow;
using MatView    = I128TensorView::MatView;
using Scalar     = I128TensorView::Scalar;
template <int N>
using TensorType = I128TensorView::TensorType<N>;
template <int N>
using TensorView = I128TensorView::TensorView<N>;



static void setZeroTensor(tf::Tensor& in) {
    tf::int64* p = in.flat<tf::int64>().data();
    std::memset(p, 0, sizeof(tf::int64) * in.shape().num_elements());
}

static bool sameShape(tf::Tensor const& i64, I128TensorView const& i128) {
    //! Juhou: i64 should be an output tensor, so the last dimension is N_LIMBS
    const tf::TensorShape i64shape {i64.shape()};
    const tf::TensorShape i128shape {i128.shape()};
    long n_dims = i128shape.dims();
    if (n_dims + 1 != i64shape.dims()) return false;
    for (long i = 0; i < n_dims; ++i) {
        if (i64shape.dim_size(i) != i128shape.dim_size(i)) return false;
    }
    return true;
}

//! Juhou: receives 1D, 2D, and 3D tensor (the last dimension should be 2)
//! 1D tensor -> view of a single scalar
//! 2D tensor -> view of 1D tensor
//! 3D tensor -> view of 2D tensor
template <class ViewType = MatView>
static ViewType makeView(const tf::Tensor& in) {
    const auto& shape = in.shape();
    long n_dims       = shape.dims();
    //CHECK_LT(n_dims, 4);  // at most 3 dimensions
    CHECK_GT(n_dims, 0);
    CHECK_EQ(shape.dim_size(n_dims - 1), N_LIMBS);  // the last dimension should be 2.
    auto cnst_data_ptr = in.flat<tf::int64>().data();
    //! shape: (2, ) -> (1, 1, 2)
    //!        (N, 2) -> (N, 1, 2)
    //!        (N, D, 2) -> (N, D, 2)
    return ViewType((Scalar*)cnst_data_ptr, n_dims > 1 ? shape.dim_size(0) : 1, n_dims > 2 ? shape.dim_size(1) : 1);
}

//! constructor
I128TensorView::I128TensorView(const tf::Tensor& source) : shape_(source.shape()) {
    shape_.RemoveLastDims(1);  // drop the last size-2 dimension
    mat_view_    = std::make_shared<MatView>(makeView<MatView>(source));

    data_ = (Scalar*) (source.flat<tf::int64>().data());
}

template<int N>
std::array<int64_t, N> I128TensorView::dim_array () const {
    CHECK_EQ(N, shape_.dims());
    std::array<int64_t, N> d;
    for (int i = 0; i < N; i++)
        d[i] = shape_.dim_size(i);
    return d;
}

template<int N>
TensorView<N> I128TensorView::view () const {
    CHECK_EQ(N, shape_.dims());
    return TensorView<N>(data_, dim_array<N>());
}

template<int N>
TensorView<N> I128TensorView::view (const std::array<int64_t, N>& reshape) const {
    return TensorView<N>(data_, reshape);
}

template<int N>
TensorView<N> I128TensorView::view (const tf::TensorShape& reshape) const {
    CHECK_EQ(N, reshape.dims());
    std::array<int64_t, N> shapeArr;
    for (int i = 0; i < N; i++)
        shapeArr[i] = reshape.dim_size(i);
    return view<N>(shapeArr);
}


struct I128BCast {
    //! Compute the broadcast array for the binary operation.
    //! Shapes with different number of dimensions should be reshaped via adding prefix dimension(s) of '1'.
    //! Dimensions with different size should be broadcasted via using the larger dimension size.
    explicit I128BCast(tf::TensorShape const& lshape, tf::TensorShape const& rshape)
        : lbcast_({1, 1}), lreshape_({1, 1}), rbcast_({1, 1}), rreshape_({1, 1}) {
        long n_dim_l = lshape.dims();
        long n_dim_r = rshape.dims();
        CHECK_LT(n_dim_l, 3);
        CHECK_LT(n_dim_r, 3);

        //! Process from the suffix
        auto lbc_itr    = lbcast_.rbegin();
        auto lshape_itr = lreshape_.rbegin();
        auto rbc_itr    = rbcast_.rbegin();
        auto rshape_itr = rreshape_.rbegin();
        for (long i = n_dim_l - 1, j = n_dim_r - 1; i >= 0 || j >= 0; --i, --j) {
            int ldim = i >= 0 ? lshape.dim_size(i) : 1;
            int rdim = j >= 0 ? rshape.dim_size(j) : 1;
            CHECK((ldim == rdim || (ldim == 1 || rdim == 1)) && "Invalid shapes for broadcasting");

            int bigger_dim = std::max(ldim, rdim);
            *lshape_itr++  = ldim;
            *lbc_itr++     = ldim == 1 ? bigger_dim : 1;

            *rshape_itr++ = rdim;
            *rbc_itr++    = rdim == 1 ? bigger_dim : 1;
        }

        if (n_dim_l == 1 && n_dim_r == 1) {
            // output_shape (1, N) -> (N, 1)
            std::swap(lreshape_[0], lreshape_[1]);
            std::swap(rreshape_[0], rreshape_[1]);
            std::swap(lbcast_[0], lbcast_[1]);
            std::swap(rbcast_[0], rbcast_[1]);
        }
    }

    const std::array<int, 2>& lhs_bcast() const { return lbcast_; }
    const std::array<int, 2>& lhs_reshape() const { return lreshape_; }
    const std::array<int, 2>& rhs_bcast() const { return rbcast_; }
    const std::array<int, 2>& rhs_reshape() const { return rreshape_; }

private:
    std::array<int, 2> lbcast_, rbcast_;
    std::array<int, 2> lreshape_, rreshape_;
};


/**
Return a new shape that prepends the given shape with new dims of 1.
If the given shape's number of dims is not less than the target dims,
the function simply returns a copy of the given shape.

@param src The shape to be expanded.
@param target_dims The target number of dimensions to be expanded to.
*/
tf::TensorShape expandToDims (tf::TensorShape const& src, int target_dims) {
    if (src.dims() >= target_dims)
        return src;
    tf::TensorShape result(src);
    for (int idx = target_dims - src.dims() - 1; idx >= 0; idx--)
        result.InsertDim(0, 1);
    return result;
}

/**
Return the broadcasting array to use for broadcasting op1 to be operable with op2. 

Reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
*/
template<int N>
std::array<int64_t, N> bcastArray (const tf::TensorShape &op1, const tf::TensorShape &op2) {
    CHECK_EQ(N, op1.dims());
    CHECK_EQ(N, op2.dims());
    std::array<int64_t, N> result;
    for (int i = 0; i < N; i++) {
        CHECK((op1.dim_size(i) == 1 || op2.dim_size(i) == 1 || op1.dim_size(i) == op2.dim_size(i)) && "Non-broadcastable tensors.");
        result[i] = op1.dim_size(i) == 1 ? op2.dim_size(i) : 1;
    }
    return result;
}

/**
Return the broadcasting array to use for broadcasting op1 to be operable with op2. 

Reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
*/
template<int N>
std::array<int64_t, N> bcastArray (const TensorView<N>& op1, const TensorView<N>& op2) {
    std::array<int64_t, N> result;
    for (int i = 0; i < N; i++) {
        CHECK((op1.dimension(i) == 1 || op2.dimension(i) == 1 || op1.dimension(i) == op2.dimension(i)) && "Non-broadcastable tensors.");
        result[i] = op1.dimension(i) == 1 ? op2.dimension(i) : 1;
    }
    return result;
}


template <int N>
struct BroadcastMul {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
            * rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view));
    }
};

template <int N>
struct BroadcastAdd {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
            + rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view));
    }
};

template <int N>
struct BroadcastSub {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
            - rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view));
    }
};

template <int N>
struct BroadcastRightShift {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        auto rshift_op = [](const Scalar& x, const Scalar& step) -> Scalar {
            if (step <= 0) return x;
            if (step >= 128) return x < 0 ? -1 : 0;
            return x >> step;

            // Should change to the following?
            //y = y & ~(y>>63);
            //y = y & 0x7F;
            //return x >> y;
        };

        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
                .binaryExpr(rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view)), rshift_op);
    }
};

template <int N>
struct BroadcastLogicalRightShift {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        auto logical_rshift_op = [](const Scalar& x, const Scalar& step) -> Scalar {
            if (step <= 0) return x;
            if (step >= 128) return 0;
            return ((u128_t) x) >> step;
        };

        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
                .binaryExpr(rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view)), logical_rshift_op);
    }
};

template <int N>
struct BroadcastLeftShift {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &lhs_eigen_view, const TensorView<N> &rhs_eigen_view) {
        auto lshift_op = [](const Scalar& x, const Scalar& step) -> Scalar {
            if (step <= 0) return x;
            if (step >= 128) return 0;
            return x << step;
        };

        out_eigen_view = lhs_eigen_view.broadcast(bcastArray<N>(lhs_eigen_view, out_eigen_view))
                .binaryExpr(rhs_eigen_view.broadcast(bcastArray<N>(rhs_eigen_view, out_eigen_view)), lshift_op);
    }
};

template <int N>
struct CwiseNegate {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &in_eigen_view) {
        out_eigen_view = -in_eigen_view; 
    }
};

template <int N>
struct CwiseAbs {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &in_eigen_view) {
        auto abs_op = [](const Scalar& x) -> Scalar {
            return x < 0 ? -x : x;
        };
        out_eigen_view = in_eigen_view.unaryExpr(abs_op);
        ////! Juhou: Eigen calls std::abs() in the cwiseAbs() method, but in the standard library,
        ////! the overload for i128_t is absent.
    }
};

template <int N>
struct CwiseBitReverse {
    void operator() (TensorView<N> &out_eigen_view, const TensorView<N> &in_eigen_view) {
        using u32 = tf::uint32;

        auto u32_reverse = [](u32 v) -> u32 {
            v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
            // swap consecutive pairs
            v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
            // swap nibbles ...
            v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
            // swap bytes
            v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
            // swap 2-byte long pairs
            v = (v >> 16) | (v << 16);
            return v;
        };
        
        auto u128_reverse = [&](Scalar v) -> Scalar {
            u32* raw = (u32*)&v;
            for (int i = 0; i < 4; ++i) {
                raw[i] = u32_reverse(raw[i]);
            }
            std::swap(raw[0], raw[3]);
            std::swap(raw[1], raw[2]);
            return (Scalar)v;
        };

        out_eigen_view = in_eigen_view.unaryExpr(u128_reverse);
    }
};

template <int N, template<int> class OP>
void i128TensorCwiseCompute (I128TensorView &out_view, I128TensorView const& lhs_view, I128TensorView const& rhs_view) {
    // NOTE: Prepend a dim of 1 for handling the case of N == 0, because it doesn't support broadcasting in Eigen when N = 0.
    // If we don't use Eigen's broadcasting, then the prepend can be removed.

    tf::TensorShape out_reshape {out_view.shape()};
    out_reshape.InsertDim(0, 1);
    tf::TensorShape lhs_reshape = expandToDims(lhs_view.shape(), out_reshape.dims());
    tf::TensorShape rhs_reshape = expandToDims(rhs_view.shape(), out_reshape.dims()); 
    TensorView<N+1> out_eigen_view = out_view.view<N+1>(out_reshape);
    TensorView<N+1> lhs_eigen_view = lhs_view.view<N+1>(lhs_reshape);
    TensorView<N+1> rhs_eigen_view = rhs_view.view<N+1>(rhs_reshape);

    OP<N+1>()(out_eigen_view, lhs_eigen_view, rhs_eigen_view); 
}

template <int N, template<int> class OP>
void i128TensorCwiseCompute (I128TensorView &out_view, I128TensorView const& in_view) {
    CHECK(out_view.shape() == in_view.shape());

    TensorView<N> out_eigen_view = out_view.view<N>();
    TensorView<N> in_eigen_view = in_view.view<N>();

    OP<N>()(out_eigen_view, in_eigen_view); 
}

bool i128TensorMul (tf::Tensor& out, tf::Tensor const& lhs, tf::Tensor const& rhs) {
    I128TensorView out_view(out), lhs_view(lhs), rhs_view(rhs);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastMul>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}

bool i128TensorAdd(tf::Tensor& out, tf::Tensor const& lhs, tf::Tensor const& rhs) {
    I128TensorView out_view(out), lhs_view(lhs), rhs_view(rhs);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastAdd>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;

    return true;
}


bool i128TensorSub(tf::Tensor& out, tf::Tensor const& lhs, tf::Tensor const& rhs) {
    I128TensorView out_view(out), lhs_view(lhs), rhs_view(rhs);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastSub>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}

/**
Simply cast a tf::int64 Tensor to a int128 Tensor.
*/
tf::Tensor i128TensorConvert(const tf::Tensor &in) {
    CHECK_EQ(in.dtype(), tf::DT_INT64);

    tf::TensorShape out_shape(in.shape());
    out_shape.AddDim(N_LIMBS);
    tf::Tensor out(tf::DT_INT64, out_shape);

    Scalar* dst   = (Scalar*) (out.flat<tf::int64>().data());
    const tf::int64* src = in.flat<tf::int64>().data();
    const tf::int64* end = src + in.NumElements();

    std::transform(src, end, dst, [](tf::int64 x) -> Scalar {
            return (Scalar) x;
    });

    return std::move(out);
}


bool i128TensorRightShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt) {
    tf::Tensor i128Shmt = i128TensorConvert(shmt);
    
    I128TensorView out_view(out), lhs_view(in), rhs_view(i128Shmt);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastRightShift>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}

bool i128TensorLogicalRightShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt) {
    tf::Tensor i128Shmt = i128TensorConvert(shmt);
    
    I128TensorView out_view(out), lhs_view(in), rhs_view(i128Shmt);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastLogicalRightShift>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}


bool i128TensorLeftShift(tf::Tensor& out, tf::Tensor const& in, tf::Tensor const& shmt) {
    tf::Tensor i128Shmt = i128TensorConvert(shmt);
    
    I128TensorView out_view(out), lhs_view(in), rhs_view(i128Shmt);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, BroadcastLeftShift>(out_view, lhs_view, rhs_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}


bool i128TensorEqual(tf::Tensor& out, I128TensorView const& lhs, I128TensorView const& rhs) {
    CHECK_EQ(out.dtype(), tf::DT_BOOL);
    bool lhs_scalar = lhs.isScalar();
    bool rhs_scalar = rhs.isScalar();
    bool* dst       = out.flat<bool>().data();
    if (lhs_scalar && rhs_scalar) {
        *dst = (*lhs.data()) == (*rhs.data());
        return true;
    }

    if (lhs_scalar) return i128TensorEqualScalar(out, rhs, *lhs.data());
    if (rhs_scalar) return i128TensorEqualScalar(out, lhs, *rhs.data());
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(out.shape(), lhs.shape());  // bool tensor do not contain the last size-2 dimension.

    const Scalar* op0 = lhs.data();
    const Scalar* end = op0 + lhs.numElements();
    const Scalar* op1 = rhs.data();
    while (op0 != end) {
        *dst++ = *op0++ == *op1++;
    }
    return true;
}

bool i128TensorMatmul(tf::Tensor& out, I128TensorView const& lhs, I128TensorView const& rhs) {
    const auto& out_shape = out.shape();
    const auto& lhs_shape = lhs.shape();
    const auto& rhs_shape = rhs.shape();
    CHECK_EQ(out_shape.dims(), 3);
    CHECK_EQ(out_shape.dim_size(0), lhs_shape.dim_size(0));
    CHECK_EQ(out_shape.dim_size(1), rhs_shape.dim_size(1));
    CHECK_EQ(lhs_shape.dim_size(1), rhs_shape.dim_size(0));

    auto out_view {makeView<MatView>(out)};
    out_view = (lhs.view() * rhs.view());
    return true;
}


bool i128TensorNegate(tf::Tensor& out, tf::Tensor const& in) {
    I128TensorView out_view(out), in_view(in);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, CwiseNegate>(out_view, in_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, CwiseNegate>(out_view, in_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, CwiseNegate>(out_view, in_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, CwiseNegate>(out_view, in_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, CwiseNegate>(out_view, in_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, CwiseNegate>(out_view, in_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}

bool i128TensorAbs(tf::Tensor& out, tf::Tensor const& in) {
    I128TensorView out_view(out), in_view(in);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, CwiseAbs>(out_view, in_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, CwiseAbs>(out_view, in_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, CwiseAbs>(out_view, in_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, CwiseAbs>(out_view, in_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, CwiseAbs>(out_view, in_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, CwiseAbs>(out_view, in_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }

    return true;
}


template <int N, long D>
void i128TensorReduceSumCompute(I128TensorView &out_view, const I128TensorView &in_view, const long* axis_vec) {
    CHECK(N > 0);
    TensorView<N> in_eigen_view = in_view.view<N>();
    TensorView<1> out_eigen_view = out_view.view<1>(std::array<int64_t, 1>{out_view.numElements()});

    std::array<int64_t, D> reduce_dims;
    for(long i = 0; i < D; i++){
        reduce_dims[i] = *(axis_vec+i);
    }
    TensorType<N-D> result = in_eigen_view.sum(reduce_dims);
    out_eigen_view = TensorView<1>(result.data(), std::array<int64_t, 1>{out_view.numElements()});
}

bool i128TensorReduceSum(tf::Tensor& out, tf::Tensor const& in, const long* axis_vec, long axis_num, bool keepdims) {
    I128TensorView in_view(in), out_view(out);
    
    // TODO(zjn) This is pretty ugly, need a better implementation.
    switch (in_view.dims()) {
        case 0:
            *out_view.data() = *in_view.data();
            break;
        case 1:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<1, 1>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            i128TensorReduceSumCompute<1, 1>(out_view, in_view, axis_vec);
            break;
        case 2:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<2, 1>(out_view, in_view, axis_vec);
                    break;
                case 2:
                    i128TensorReduceSumCompute<2, 2>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            break;
        case 3:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<3, 1>(out_view, in_view, axis_vec);
                    break;
                case 2:
                    i128TensorReduceSumCompute<3, 2>(out_view, in_view, axis_vec);
                    break;
                case 3:
                    i128TensorReduceSumCompute<3, 3>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            break;
        case 4:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<4, 1>(out_view, in_view, axis_vec);
                    break;
                case 2:
                    i128TensorReduceSumCompute<4, 2>(out_view, in_view, axis_vec);
                    break;
                case 3:
                    i128TensorReduceSumCompute<4, 3>(out_view, in_view, axis_vec);
                    break;
                case 4:
                    i128TensorReduceSumCompute<4, 4>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            break;
        case 5:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<5, 1>(out_view, in_view, axis_vec);
                    break;
                case 2:
                    i128TensorReduceSumCompute<5, 2>(out_view, in_view, axis_vec);
                    break;
                case 3:
                    i128TensorReduceSumCompute<5, 3>(out_view, in_view, axis_vec);
                    break;
                case 4:
                    i128TensorReduceSumCompute<5, 4>(out_view, in_view, axis_vec);
                    break;
                case 5:
                    i128TensorReduceSumCompute<5, 5>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            break;
        case 6:
            switch (axis_num){
                case 1:
                    i128TensorReduceSumCompute<6, 1>(out_view, in_view, axis_vec);
                    break;
                case 2:
                    i128TensorReduceSumCompute<6, 2>(out_view, in_view, axis_vec);
                    break;
                case 3:
                    i128TensorReduceSumCompute<6, 3>(out_view, in_view, axis_vec);
                    break;
                case 4:
                    i128TensorReduceSumCompute<6, 4>(out_view, in_view, axis_vec);
                    break;
                case 5:
                    i128TensorReduceSumCompute<6, 5>(out_view, in_view, axis_vec);
                    break;
                case 6:
                    i128TensorReduceSumCompute<6, 6>(out_view, in_view, axis_vec);
                    break;
                default:
                    std::cout << "input axis num not support " << axis_num << std::endl;
                    CHECK((axis_num <= in_view.dims()) && "Unsupported axis nums");
            }
            break;
        default:
            std::cout << "input tesor dims not support " << in_view.dims() << std::endl;
            CHECK((in_view.dims() >=0 && in_view.dims() <= 6) && "Unsupported tensor dims");
    }
    
    return true;
}

bool i128TensorConvert(tf::Tensor& out, tf::Tensor const& in, tf::uint64 scale) {
    const auto& out_shape = out.shape();
    const auto& in_shape  = in.shape();
    const long n_dims     = in_shape.dims();
    CHECK(scale > 0.);
    CHECK_EQ(out.dtype(), tf::DT_INT64);
    CHECK_EQ(in.dtype(), tf::DT_DOUBLE);
    CHECK_EQ(out_shape.dims(), n_dims + 1);
    CHECK_EQ(out_shape.dim_size(n_dims), N_LIMBS);
    for (long i = 0; i < n_dims; ++i) CHECK_EQ(out_shape.dim_size(i), in_shape.dim_size(i));

    tf::uint64* dst   = (tf::uint64*)out.flat<tf::int64>().data();
    const double* src = in.flat<double>().data();
    const double* end = src + in.NumElements();

    while (src != end) {
        Scalar v = (Scalar)((*src++) * scale);
        *dst++   = (tf::uint64)v;
        *dst++   = (tf::uint64)(v >> 64);
    }
    return true;
}

bool i128TensorConvertBack(tf::Tensor& out, tf::Tensor const& in, tf::uint64 scale) {
    const auto& out_shape = out.shape();
    const auto& in_shape  = in.shape();
    const long n_dims     = out_shape.dims();
    CHECK(scale > 0.);
    CHECK_EQ(out.dtype(), tf::DT_DOUBLE);
    CHECK_EQ(in.dtype(), tf::DT_INT64);
    CHECK_EQ(in_shape.dims(), n_dims + 1);
    CHECK_EQ(in_shape.dim_size(n_dims), N_LIMBS);
    for (long i = 0; i < n_dims; ++i) CHECK_EQ(out_shape.dim_size(i), in_shape.dim_size(i));

    double* dst       = out.flat<double>().data();
    const Scalar* src = (const Scalar*)in.flat<tf::int64>().data();
    const Scalar* end = src + in.NumElements() / N_LIMBS;
    double _scale     = (double)scale;
    std::transform(src, end, dst, [_scale](const Scalar& v) -> double { return v / _scale; });
    return true;
}

bool i128TensorEqualScalar(tf::Tensor& out, I128TensorView const& lhs, I128TensorView::Scalar const& rhs) {
    CHECK_EQ(out.dtype(), tf::DT_BOOL);
    CHECK_EQ(out.shape(), lhs.shape());  // bool tensor do not contain the last size-2 dimension.
    const Scalar* op0 = lhs.data();
    const Scalar* end = op0 + lhs.numElements();
    bool* dst         = out.flat<bool>().data();
    while (op0 != end) {
        *dst++ = *op0++ == rhs;
    }
    return true;
}

bool i128TensorBitReverse(tf::Tensor& out, tf::Tensor const& in) {
    I128TensorView out_view(out), in_view(in);

    switch (out_view.dims()) {
        case 0:
            i128TensorCwiseCompute<0, CwiseBitReverse>(out_view, in_view);
            break;
        case 1:
            i128TensorCwiseCompute<1, CwiseBitReverse>(out_view, in_view);
            break;
        case 2:
            i128TensorCwiseCompute<2, CwiseBitReverse>(out_view, in_view);
            break;
        case 3:
            i128TensorCwiseCompute<3, CwiseBitReverse>(out_view, in_view);
            break;
        case 4:
            i128TensorCwiseCompute<4, CwiseBitReverse>(out_view, in_view);
            break;
        case 5:
            i128TensorCwiseCompute<5, CwiseBitReverse>(out_view, in_view);
            break;
        default:
            CHECK((out_view.dims() >=0 && out_view.dims() <= 5) && "Unsupported tensor dims");
    }
    return true;
}

bool i128TensorGatherBits(tf::Tensor& out, I128TensorView const& in, int start, int stride) {
    CHECK(sameShape(out, in));
    const u128_t* src = (const u128_t*)in.data();
    const u128_t* end = src + in.numElements();
    u128_t* dst       = (u128_t*)out.flat<tf::int64>().data();
    std::transform(src, end, dst, [=](u128_t v) -> u128_t {
        const u128_t one {1};
        u128_t ans {0};
        v >>= start;
        for (long d=start, i = 0; v > 0 && d < 128; d+=stride, i++) {
            if (v & 1) ans |= (one << i);
            v >>= stride;
        }
        return ans;
    });
    return true;
}

bool i128TensorXorIndices(tf::Tensor& out, tf::Tensor const& in) {
    CHECK(out.shape() == in.shape());
    const u128_t* src = (const u128_t*)in.flat<tf::int64>().data();
    const u128_t* end = src + (in.NumElements() / 2);
    u128_t* dst       = (u128_t*)out.flat<tf::int64>().data();
    std::transform(src, end, dst, [](u128_t v) -> u128_t {
        long ans {0};
        for (long d = 0; v > 0 && d < 128; ++d) {
            if (v & 1) ans ^= d;
            v >>= 1;
        }
        return (u128_t)ans;
    });
    return true;
}

}  // namespace tf_i128
