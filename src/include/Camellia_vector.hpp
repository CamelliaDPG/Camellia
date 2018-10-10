#ifndef CAMELLIA_VECTOR_HPP
#define CAMELLIA_VECTOR_HPP

/*
 Adapted from Omega_h Vector class.
 */

#include <Camellia_array.hpp>

namespace Camellia {

template <class Scalar, int n>
class Vector : public Array<Scalar, n> {
 public:
  KOKKOS_INLINE_FUNCTION Vector() {}
  inline Vector(std::initializer_list<Scalar> l) : Array<Scalar, n>(l) {}
  KOKKOS_INLINE_FUNCTION void operator=(Vector<Scalar,n> const& rhs) volatile {
    Array<Scalar, n>::operator=(rhs);
  }
  KOKKOS_INLINE_FUNCTION Vector(Vector<Scalar,n> const& rhs) : Array<Scalar, n>(rhs) {}
  KOKKOS_INLINE_FUNCTION Vector(const volatile Vector<Scalar,n>& rhs) : Array<Scalar, n>(rhs) {}
#define CAMELLIA_VECTOR_AT return Array<Scalar, n>::operator[](i)
  KOKKOS_INLINE_FUNCTION Scalar& operator()(int i) { CAMELLIA_VECTOR_AT; }
  KOKKOS_INLINE_FUNCTION Scalar const& operator()(int i) const { CAMELLIA_VECTOR_AT; }
  KOKKOS_INLINE_FUNCTION Scalar volatile& operator()(int i) volatile {
    CAMELLIA_VECTOR_AT;
  }
  KOKKOS_INLINE_FUNCTION Scalar const volatile& operator()(int i) const volatile {
    CAMELLIA_VECTOR_AT;
  }
#undef CAMELLIA_VECTOR_AT
};

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Scalar* scalar_ptr(Vector<Scalar,n>& v) {
  return &v[0];
}
template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Scalar const* scalar_ptr(Vector<Scalar,n> const& v) {
  return &v[0];
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator+(Vector<Scalar,n> a, Vector<Scalar,n> b) {
  Vector<Scalar,n> c;
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n>& operator+=(Vector<Scalar,n>& a, Vector<Scalar,n> b) {
  a = a + b;
  return a;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator-(Vector<Scalar,n> a, Vector<Scalar,n> b) {
  Vector<Scalar,n> c;
  for (int i = 0; i < n; ++i) c[i] = a[i] - b[i];
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n>& operator-=(Vector<Scalar,n>& a, Vector<Scalar,n> b) {
  a = a - b;
  return a;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator-(Vector<Scalar,n> a) {
  Vector<Scalar,n> c;
  for (int i = 0; i < n; ++i) c[i] = -a[i];
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator*(Vector<Scalar,n> a, Scalar b) {
  Vector<Scalar,n> c;
  for (int i = 0; i < n; ++i) c[i] = a[i] * b;
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n>& operator*=(Vector<Scalar,n>& a, Scalar b) {
  a = a * b;
  return a;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator*(Scalar a, Vector<Scalar,n> b) {
  return b * a;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> operator/(Vector<Scalar,n> a, Scalar b) {
  Vector<Scalar,n> c;
  for (int i = 0; i < n; ++i) c[i] = a[i] / b;
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n>& operator/=(Vector<Scalar,n>& a, Scalar b) {
  a = a / b;
  return a;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Scalar operator*(Vector<Scalar,n> a, Vector<Scalar,n> b) {
  Scalar c = a[0] * b[0];
  for (int i = 1; i < n; ++i) c += a[i] * b[i];
  return c;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Scalar norm_squared(Vector<Scalar,n> v) {
  return v * v;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Scalar norm(Vector<Scalar,n> v) {
  return sqrt(norm_squared(v));
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> normalize(Vector<Scalar,n> v) {
  return v / norm(v);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar,1> vector_1(Scalar x) {
  Vector<Scalar,1> v;
  v[0] = x;
  return v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar,2> vector_2(Scalar x, Scalar y) {
  Vector<Scalar,2> v;
  v[0] = x;
  v[1] = y;
  return v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar,3> vector_3(Scalar x, Scalar y, Scalar z) {
  Vector<Scalar,3> v;
  v[0] = x;
  v[1] = y;
  v[2] = z;
  return v;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION bool are_close(
    Vector<Scalar,n> a, Vector<Scalar,n> b, double tol, Scalar floor) {
  for (int i = 0; i < n; ++i)
    if (!are_close(a[i], b[i], tol, floor)) return false;
  return true;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> fill_vector(Scalar value) {
  Vector<Scalar,n> v;
  for (int i = 0; i < n; ++i) v[i] = value;
  return v;
}

template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> zero_vector() {
  return fill_vector<Scalar,n>(0.0);
}

/* Moore-Penrose pseudo-inverse of a vector */
template <class Scalar, int n>
KOKKOS_INLINE_FUNCTION Vector<Scalar,n> pseudo_invert(Vector<Scalar,n> a) {
  auto nsq = a * a;
  if (nsq < 1e-15) return zero_vector<n>();
  return a / nsq;
}

/* a function to disambiguate a unit vector
   from its negative. we treat the signs of
   the components as bits of an integer,
   and negate the components if the resulting
   bit pattern makes a larger integer */
template <int n>
KOKKOS_INLINE_FUNCTION Vector<double,n> positivize(Vector<double,n> v) {
  std::uint32_t bits = 0;
  for (int i = 0; i < n; ++i) bits |= (std::uint32_t(v[i] >= 0.0) << i);
  std::uint32_t neg_bits = (~bits) & ((std::uint32_t(1) << n) - 1);
  if (neg_bits > bits) return v * -1.0;
  return v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar cross(Vector<Scalar,2> a, Vector<Scalar,2> b) {
  return (a[0] * b[1] - a[1] * b[0]);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar,3> cross(Camellia::Vector<Scalar,3> a, Camellia::Vector<Scalar,3> b) {
  return vector_3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Vector<Scalar,2> perp(Vector<Scalar,2> v) { return vector_2(-v[1], v[0]); }

template <class Scalar, int n, class Arr>
KOKKOS_INLINE_FUNCTION void set_vector(Arr const& a, int i, Vector<Scalar,n> v) {
  for (int j = 0; j < n; ++j) a[i * n + j] = v[j];
}

template <class Scalar, int n, class Arr>
KOKKOS_INLINE_FUNCTION Vector<Scalar, n> get_vector(Arr const& a, int i) {
  Vector<Scalar, n> v;
  for (int j = 0; j < n; ++j) v[j] = a[i * n + j];
  return v;
}
  
}  // namespace Camellia

#endif
