#ifndef PROJECTOR
#define PROJECTOR

#include "TypeDefs.h"

#include "IP.h"

#include "Basis.h"

namespace Camellia {
  template <typename Scalar>
  class Projector {
  public:
    // newest version:
    static void projectFunctionOntoBasis(Intrepid::FieldContainer<Scalar> &basisCoefficients,
        TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr basisCache,
        TIPPtr<Scalar> ip, VarPtr v,
        std::set<int>fieldIndicesToSkip = std::set<int>());

    static void projectFunctionOntoBasis(Intrepid::FieldContainer<Scalar> &basisCoefficients,
        TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr basisCache);

    static void projectFunctionOntoBasisInterpolating(Intrepid::FieldContainer<Scalar> &basisCoefficients,
        TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr domainBasisCache);
  };

  extern template class Projector<double>;
}
#endif
