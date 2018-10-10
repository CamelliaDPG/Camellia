#ifndef CAMELLIA_C_H
#define CAMELLIA_C_H

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  void Camellia_fail(char const* format, ...)
  __attribute__((noreturn, format(printf, 1, 2)));
  
#ifdef __cplusplus
}  // end of extern "C" block
#endif

#ifdef __CUDA_ARCH__
#define CAMELLIA_CHECK(cond) assert(cond)
#else
#define CAMELLIA_CHECK(cond)                                                    \
((cond) ? ((void)0)                                                          \
: Camellia_fail(                                                      \
"assertion %s failed at %s +%d\n", #cond, __FILE__, __LINE__))
#endif

#ifdef __clang__
#define CAMELLIA_NORETURN(x) CAMELLIA_CHECK(false)
#else
#define CAMELLIA_NORETURN(x)                                                    \
do {                                                                         \
CAMELLIA_CHECK(false);                                                             \
return x;                                                                  \
} while (false)
#endif

#endif
