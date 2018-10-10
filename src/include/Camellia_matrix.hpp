#ifndef CAMELLIA_MATRIX_HPP
#define CAMELLIA_MATRIX_HPP

/*
 Adapted from Omega_h Matrix class.
 */

#include <Camellia_array.hpp>
#include <Camellia_vector.hpp>

namespace Camellia {

/* column-first storage and indexing !!! */
template <typename Scalar, int m, int n>
class Matrix : public Array<Vector<Scalar,m>, n> {
 public:
  KOKKOS_INLINE_FUNCTION Matrix() {}
  /* these constructors accept the matrix in
   * row-first order for convenience.
   */
  inline Matrix(std::initializer_list<Vector<Scalar,m>> l) : Array<Vector<Scalar,m>, n>(l) {}
  inline Matrix(std::initializer_list<Scalar> l);
  KOKKOS_INLINE_FUNCTION void operator=(Matrix<Scalar, m, n> const& rhs) volatile {
    Array<Vector<Scalar,m>, n>::operator=(rhs);
  }
  KOKKOS_INLINE_FUNCTION Matrix(Matrix<Scalar, m, n> const& rhs) : Array<Vector<Scalar, m>, n>(rhs) {}
  KOKKOS_INLINE_FUNCTION Matrix(const volatile Matrix<Scalar, m, n>& rhs)
      : Array<Vector<Scalar, m>, n>(rhs) {}
  KOKKOS_INLINE_FUNCTION Scalar& operator()(int i, int j) {
    return Array<Vector<Scalar, m>, n>::operator[](j)[i];
  }
  KOKKOS_INLINE_FUNCTION Scalar const& operator()(int i, int j) const {
    return Array<Vector<Scalar, m>, n>::operator[](j)[i];
  }
  KOKKOS_INLINE_FUNCTION Scalar volatile& operator()(int i, int j) volatile {
    return Array<Vector<Scalar, m>, n>::operator[](j)[i];
  }
  KOKKOS_INLINE_FUNCTION Scalar const volatile& operator()(int i, int j) const volatile {
    return Array<Vector<Scalar, m>, n>::operator[](j)[i];
  }
};

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Scalar* scalar_ptr(Matrix<Scalar, m, n>& a) {
  return &a[0][0];
}
template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Scalar const* scalar_ptr(Matrix<Scalar, m, n> const& a) {
  return &a[0][0];
}

template <typename Scalar, int m, int n>
inline Matrix<Scalar, m, n>::Matrix(std::initializer_list<Scalar> l) {
  int k = 0;
  for (Scalar v : l) {
    (*this)[k % n][k / n] = v;
    ++k;
  }
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar, m> operator*(Matrix<Scalar, m, n> a, Vector<Scalar, n> b) {
  Vector<Scalar, m> c = a[0] * b[0];
  for (int j = 1; j < n; ++j) c = c + a[j] * b[j];
  return c;
}

template <typename Scalar, int m, int n, int p>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator*(Matrix<Scalar, m, p> a, Matrix<Scalar, p, n> b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = a * b[j];
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, n, m> transpose(Matrix<Scalar, m, n> a) {
  Matrix<Scalar, n, m> b;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) b[i][j] = a[j][i];
  return b;
}

template <typename Scalar, int max_m, int max_n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, max_m, max_n> identity_matrix(int m, int n) {
  Matrix<Scalar, max_m, max_n> a;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) a[j][i] = (i == j);
  return a;
}

template <typename Scalar, int max_m, int max_n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, max_m, max_n> identity_matrix() {
  return identity_matrix<Scalar, max_m, max_n>(max_m, max_n);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 1, 1> matrix_1x1(Scalar a) {
  Matrix<Scalar, 1, 1> o;
  o[0][0] = a;
  return o;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 2, 2> matrix_2x2(Scalar a, Scalar b, Scalar c, Scalar d) {
  Matrix<Scalar, 2, 2> o;
  o[0] = vector_2(a, c);
  o[1] = vector_2(b, d);
  return o;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> matrix_3x3(
    Scalar a, Scalar b, Scalar c, Scalar d, Scalar e, Scalar f, Scalar g, Scalar h, Scalar i) {
  Matrix<Scalar, 3, 3> o;
  o[0] = vector_3(a, d, g);
  o[1] = vector_3(b, e, h);
  o[2] = vector_3(c, f, i);
  return o;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator*(Matrix<Scalar, m, n> a, Scalar b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = a[j] * b;
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n>& operator*=(Matrix<Scalar, m, n>& a, Scalar b) {
  a = a * b;
  return a;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator*(Scalar a, Matrix<Scalar, m, n> b) {
  return b * a;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator/(Matrix<Scalar, m, n> a, Scalar b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = a[j] / b;
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n>& operator/=(Matrix<Scalar, m, n>& a, Scalar b) {
  a = a / b;
  return a;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator+(Matrix<Scalar, m, n> a, Matrix<Scalar, m, n> b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = a[j] + b[j];
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n>& operator+=(Matrix<Scalar, m, n>& a, Matrix<Scalar, m, n> b) {
  a = a + b;
  return a;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator-(Matrix<Scalar, m, n> a, Matrix<Scalar, m, n> b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = a[j] - b[j];
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n>& operator-=(Matrix<Scalar, m, n>& a, Matrix<Scalar, m, n> b) {
  a = a - b;
  return a;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> operator-(Matrix<Scalar, m, n> a) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j) c[j] = -a[j];
  return c;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Scalar max_norm(Matrix<Scalar, m, n> a) {
  Scalar x = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) x = max2(x, fabs(a[j][i]));
  return x;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION bool are_close(
    Matrix<Scalar, m, n> a, Matrix<Scalar, m, n> b, Scalar tol, Scalar floor) {
  for (int j = 0; j < n; ++j)
    if (!are_close(a[j], b[j], tol, floor)) return false;
  return true;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> outer_product(Vector<Scalar, m> a, Vector<Scalar, n> b) {
  Matrix<Scalar, m, n> c;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) c[j][i] = a[i] * b[j];
  return c;
}

template <typename Scalar, int m>
KOKKOS_INLINE_FUNCTION Scalar trace(Matrix<Scalar, m, m> a) __attribute__((pure));
template <typename Scalar, int m>
KOKKOS_INLINE_FUNCTION Scalar trace(Matrix<Scalar, m, m> a) {
  Scalar t = a[0][0];
  for (int i = 1; i < m; ++i) t += a[i][i];
  return t;
}

template <typename Scalar, int m>
KOKKOS_INLINE_FUNCTION Vector<Scalar, m> diagonal(Matrix<Scalar, m, m> a) {
  Vector<Scalar, m> v;
  for (int i = 0; i < m; ++i) v[i] = a[i][i];
  return v;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, n> zero_matrix() {
  Matrix<Scalar, m, n> a;
  for (int j = 0; j < n; ++j) a[j] = zero_vector<Scalar,m>();
  return a;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar determinant(Matrix<Scalar, 1, 1> m) { return m[0][0]; }

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar determinant(Matrix<Scalar, 2, 2> m) {
  Scalar a = m[0][0];
  Scalar b = m[1][0];
  Scalar c = m[0][1];
  Scalar d = m[1][1];
  return a * d - b * c;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar determinant(Matrix<Scalar, 3, 3> m) {
  Scalar a = m[0][0];
  Scalar b = m[1][0];
  Scalar c = m[2][0];
  Scalar d = m[0][1];
  Scalar e = m[1][1];
  Scalar f = m[2][1];
  Scalar g = m[0][2];
  Scalar h = m[1][2];
  Scalar i = m[2][2];
  return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) -
         (a * f * h);
}

template <typename Scalar, int max_m, int max_n>
KOKKOS_INLINE_FUNCTION Scalar inner_product(
    int m, int n, Matrix<Scalar, max_m, max_n> a, Matrix<Scalar, max_m, max_n> b) {
  Scalar out = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      out += a[j][i] * b[j][i];
    }
  }
  return out;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Scalar inner_product(Matrix<Scalar, m, n> a, Matrix<Scalar, m, n> b) {
  return inner_product(m, n, a, b);
}

template <typename Scalar, int max_m, int max_n>
KOKKOS_INLINE_FUNCTION Scalar norm(int m, int n, Matrix<Scalar, max_m, max_n> a) {
  return sqrt(inner_product(m, n, a, a));
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION Scalar norm(Matrix<Scalar, m, n> a) {
  return norm(m, n, a);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> cross(Vector<Scalar, 3> a) {
  return matrix_3x3(0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar, 3> uncross(Matrix<Scalar, 3, 3> c) {
  return vector_3(c[1][2] - c[2][1], c[2][0] - c[0][2], c[0][1] - c[1][0]) / 2.;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 1, 1> invert(Matrix<Scalar, 1, 1> m) {
  return matrix_1x1(1.0 / m[0][0]);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 2, 2> invert(Matrix<Scalar, 2, 2> m) {
  Scalar a = m[0][0];
  Scalar b = m[1][0];
  Scalar c = m[0][1];
  Scalar d = m[1][1];
  return matrix_2x2(d, -b, -c, a) / determinant(m);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> invert(Matrix<Scalar, 3, 3> a) {
  Matrix<Scalar, 3, 3> b;
  b[0] = cross(a[1], a[2]);
  b[1] = cross(a[2], a[0]);
  b[2] = cross(a[0], a[1]);
  return transpose(b) / determinant(a);
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION typename std::enable_if<(n < m), Matrix<Scalar, n, m>>::type
pseudo_invert(Matrix<Scalar, m, n> a) {
  auto at = transpose(a);
  return invert(at * a) * at;
}

template <typename Scalar, int m, int n>
KOKKOS_INLINE_FUNCTION typename std::enable_if<(n > m), Matrix<Scalar, n, m>>::type
pseudo_invert(Matrix<Scalar, m, n> a) {
  auto at = transpose(a);
  return at * invert(a * at);
}

template <typename Scalar, int m>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, m, m> diagonal(Vector<Scalar, m> v) {
  Matrix<Scalar, m, m> a;
  for (int i = 0; i < m; ++i)
    for (int j = i + 1; j < m; ++j) a[i][j] = a[j][i] = 0.0;
  for (int i = 0; i < m; ++i) a[i][i] = v[i];
  return a;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr int symm_ncomps(int dim) {
  return ((dim + 1) * dim) / 2;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar, 1> symm2vector(Matrix<Scalar, 1, 1> symm) {
  return vector_1(symm[0][0]);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 1, 1> vector2symm(Vector<Scalar, 1> v) {
  return matrix_1x1(v[0]);
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar, 3> symm2vector(Matrix<Scalar, 2, 2> symm) {
  Vector<Scalar, 3> v;
  v[0] = symm[0][0];
  v[1] = symm[1][1];
  v[2] = symm[1][0];
  return v;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 2, 2> vector2symm(Vector<Scalar, 3> v) {
  Matrix<Scalar, 2, 2> symm;
  symm[0][0] = v[0];
  symm[1][1] = v[1];
  symm[1][0] = v[2];
  symm[0][1] = symm[1][0];
  return symm;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar, 6> symm2vector(Matrix<Scalar, 3, 3> symm) {
  Vector<Scalar, 6> v;
  v[0] = symm[0][0];
  v[1] = symm[1][1];
  v[2] = symm[2][2];
  v[3] = symm[1][0];
  v[4] = symm[2][1];
  v[5] = symm[2][0];
  return v;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> vector2symm(Vector<Scalar, 6> v) {
  Matrix<Scalar, 3, 3> symm;
  symm[0][0] = v[0];
  symm[1][1] = v[1];
  symm[2][2] = v[2];
  symm[1][0] = v[3];
  symm[2][1] = v[4];
  symm[2][0] = v[5];
  symm[0][1] = symm[1][0];
  symm[1][2] = symm[2][1];
  symm[0][2] = symm[2][0];
  return symm;
}

KOKKOS_INLINE_FUNCTION constexpr int matrix_ncomps(int dim) { return dim * dim; }

template <typename Scalar, int dim>
KOKKOS_INLINE_FUNCTION Vector<Scalar, matrix_ncomps(dim)> matrix2vector(Matrix<Scalar, dim, dim> m) {
  Vector<Scalar, matrix_ncomps(dim)> v;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      v[j * dim + i] = m[i][j];
    }
  }
  return v;
}

template <typename Scalar, int dim>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, dim, dim> vector2matrix(Vector<Scalar, matrix_ncomps(dim)> v) {
  Matrix<Scalar, dim, dim> m;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      m[i][j] = v[j * dim + i];
    }
  }
  return m;
}

/* Rodrigues' Rotation Formula */
template<typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> rotate(Scalar angle, Vector<Scalar, 3> axis) {
  return cos(angle) * identity_matrix<Scalar, 3, 3>() + sin(angle) * cross(axis) +
         (1 - cos(angle)) * outer_product(axis, axis);
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 2, 2> rotate(Scalar angle) {
  return matrix_2x2(cos(angle), -sin(angle), sin(angle), cos(angle));
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar rotation_angle(Matrix<Scalar, 2, 2> r) { return acos(r[0][0]); }

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar rotation_angle(Matrix<Scalar, 3, 3> r) __attribute__((pure));
template<typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar rotation_angle(Matrix<Scalar, 3, 3> r) {
  return acos((trace(r) - 1.0) / 2.0);
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 1, 1> form_ortho_basis(Vector<Scalar, 1> v) {
  return matrix_1x1(v[0]);
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 2, 2> form_ortho_basis(Vector<Scalar, 2> v) {
  Matrix<Scalar, 2, 2> A;
  A[0] = v;
  A[1] = perp(A[0]);
  return A;
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, 3, 3> form_ortho_basis(Vector<Scalar, 3> v) {
  Matrix<Scalar, 3, 3> A;
  A[0] = v;
  /* tiny custom code to sort components by absolute value */
  struct {
    int i;
    Scalar m;
  } s[3] = {{0, fabs(v[0])}, {1, fabs(v[1])}, {2, fabs(v[2])}};
  if (s[2].m > s[1].m) swap2(s[1], s[2]);
  if (s[1].m > s[0].m) swap2(s[0], s[1]);
  if (s[2].m > s[1].m) swap2(s[1], s[2]);
  /* done, components sorted by increasing magnitude */
  int lc = s[0].i;
  int mc = s[1].i;
  int sc = s[2].i;
  /* use the 2D rotation on the largest components
     (rotate v around the smallest axis) */
  A[1][lc] = -v[mc];
  A[1][mc] = v[lc];
  /* and make the last component zero so that A[0] * A[1] == 0 */
  A[1][sc] = 0;
  A[1] = normalize(A[1]);
  /* now we have 2 orthogonal unit vectors, cross product gives the third */
  A[2] = cross(A[0], A[1]);
  return A;
}

template <typename Scalar, int dim>
KOKKOS_INLINE_FUNCTION Matrix<Scalar, dim, dim> deviator(Matrix<Scalar, dim, dim> a) {
  return (a - ((1.0 / dim) * trace(a) * identity_matrix<dim, dim>()));
}


}  // end namespace Camellia

#endif
