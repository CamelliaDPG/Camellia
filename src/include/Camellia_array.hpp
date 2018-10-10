#ifndef CAMELLIA_ARRAY_HPP
#define CAMELLIA_ARRAY_HPP

#include <initializer_list>
#include <new>
#include <type_traits>

#include <Camellia_c.h> // defines CAMELLIA_CHECK
#include <Kokkos_Core.hpp>

/*
 Adapted from Omega_h's Few class.
 */

namespace Camellia {

template <typename T, int n>
class Array {
  /* Can't use this because volatile forces us to make the class look non-POD
   * static_assert(std::is_pod<T>::value, "T must be POD for Array<T>");
   */
  T array_[n];

 public:
  enum { size = n };
  KOKKOS_INLINE_FUNCTION T* data() { return array_; }
  KOKKOS_INLINE_FUNCTION T const* data() const { return array_; }
  KOKKOS_INLINE_FUNCTION T volatile* data() volatile { return array_; }
  KOKKOS_INLINE_FUNCTION T const volatile* data() const volatile { return array_; }
#ifdef CAMELLIA_CHECK_BOUNDS
#define CAMELLIA_ARRAY_AT                                                         \
  CAMELLIA_CHECK(0 <= i);                                                       \
  CAMELLIA_CHECK(i < size);                                                     \
  return array_[i]
#else
#define CAMELLIA_ARRAY_AT return array_[i]
#endif
  KOKKOS_INLINE_FUNCTION T& operator[](int i) { CAMELLIA_ARRAY_AT; }
  KOKKOS_INLINE_FUNCTION T const& operator[](int i) const { CAMELLIA_ARRAY_AT; }
  KOKKOS_INLINE_FUNCTION T volatile& operator[](int i) volatile { CAMELLIA_ARRAY_AT; }
  KOKKOS_INLINE_FUNCTION T const volatile& operator[](int i) const volatile {
    CAMELLIA_ARRAY_AT;
  }
#undef CAMELLIA_ARRAY_AT
  Array(std::initializer_list<T> l) {
    int i = 0;
    for (auto it = l.begin(); it != l.end(); ++it) {
      new (array_ + (i++)) T(*it);
    }
  }
  KOKKOS_INLINE_FUNCTION Array() {}
  KOKKOS_INLINE_FUNCTION ~Array() {}
  KOKKOS_INLINE_FUNCTION void operator=(Array<T, n> const& rhs) volatile {
    for (int i = 0; i < n; ++i) array_[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION void operator=(Array<T, n> const& rhs) {
    for (int i = 0; i < n; ++i) array_[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION void operator=(Array<T, n> const volatile& rhs) {
    for (int i = 0; i < n; ++i) array_[i] = rhs[i];
  }
  KOKKOS_INLINE_FUNCTION Array(Array<T, n> const& rhs) {
    for (int i = 0; i < n; ++i) new (array_ + i) T(rhs[i]);
  }
  KOKKOS_INLINE_FUNCTION Array(Array<T, n> const volatile& rhs) {
    for (int i = 0; i < n; ++i) new (array_ + i) T(rhs[i]);
  }
};

template <int capacity, typename T>
KOKKOS_INLINE_FUNCTION void add_unique(Array<T, capacity>& stack, int& n, T e) {
  for (int i = 0; i < n; ++i)
    if (stack[i] == e) return;
  stack[n++] = e;
}

template <int n, typename T>
KOKKOS_INLINE_FUNCTION T average(Array<T, n> x) {
  auto avg = x[0];
  for (int i = 1; i < n; ++i) avg = avg + x[i];
  return avg / n;
}

template <int n, typename T>
KOKKOS_INLINE_FUNCTION T minimum(Array<T, n> x) {
  auto out = x[0];
  for (int i = 1; i < n; ++i) out = min2(out, x[i]);
  return out;
}

template <int n, typename T>
KOKKOS_INLINE_FUNCTION T maximum(Array<T, n> x) {
  auto out = x[0];
  for (int i = 1; i < n; ++i) out = max2(out, x[i]);
  return out;
}

template <int n_max, typename T>
KOKKOS_INLINE_FUNCTION T maximum_magnitude(Array<T, n_max> x, int n) {
  auto out = x[0];
  auto max_mag = std::abs(x[0]);
  for (int i = 1; i < n; ++i) {
    auto mag = std::abs(x[i]);
    if (mag > max_mag) {
      max_mag = mag;
      out = x[i];
    }
  }
  return out;
}

template <int n, typename T>
KOKKOS_INLINE_FUNCTION T sum(Array<T, n> x) {
  auto out = x[0];
  for (int i = 1; i < n; ++i) out = out + x[i];
  return out;
}

template <int n, typename T>
KOKKOS_INLINE_FUNCTION T product(Array<T, n> x) {
  auto out = x[0];
  for (int i = 1; i < n; ++i) out = out * x[i];
  return out;
}

}  // namespace Camellia

#endif
