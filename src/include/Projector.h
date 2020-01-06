// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef PROJECTOR
#define PROJECTOR

#include "TypeDefs.h"

#include "IP.h"

#include "Basis.h"

namespace Camellia
{
template <typename Scalar>
class Projector
{
public:
  static void projectFunctionOntoBasis(Camellia::FieldContainer<Scalar> &basisCoefficients,
                                       TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr basisCache,
                                       TIPPtr<Scalar> ip, VarPtr v,
                                       std::set<int>fieldIndicesToSkip = std::set<int>());

  // this version invokes the one above
  static void projectFunctionOntoBasis(Camellia::FieldContainer<Scalar> &basisCoefficients,
                                       TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr basisCache);
  
  static void projectFunctionOntoBasisInterpolating(Camellia::FieldContainer<Scalar> &basisCoefficients,
      TFunctionPtr<Scalar> fxn, BasisPtr basis, BasisCachePtr domainBasisCache);
  
  // versions with subdomain caches:
  static void projectChildFunctionOntoParentBasis(Camellia::FieldContainer<Scalar> &basisCoefficients,
                                                  TFunctionPtr<Scalar> childFxn, BasisPtr parentBasis,
                                                  RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches,
                                                  TIPPtr<Scalar> ip, VarPtr v, set<int> fieldIndicesToSkip);
  
  static void projectChildFunctionOntoParentBasisInterpolating(Camellia::FieldContainer<Scalar> &basisCoefficients,
                                                               TFunctionPtr<Scalar> childFxn, BasisPtr parentBasis,
                                                               RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches);
};

extern template class Projector<double>;
}
#endif
