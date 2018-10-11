//
//  BasisCacheFactory.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 10/10/18.
//

#ifndef BasisCacheFactory_hpp
#define BasisCacheFactory_hpp

#include "TypeDefs.h"

namespace Camellia
{
  template<typename Scalar>
  class BasisCacheFactory
  {
  public:
    // ! Create a BasisCache that is the union of the child BasisCaches specified
    // ! The points in the union will be in the order of the subdomainCaches -- all the points for the first subdomain, then all for the next, etc.
    // ! The subdomainCaches are assumed to be in the same order as the child ordinals in refPattern.
    static BasisCachePtr unionBasisCacheForParentCell(RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches);
    
    // ! Create a Function that can be evaluated on a BasisCache returned by unionBasisCacheForParentCell()
    // ! such that the specified (wrapped) Function does not see the union cache, but the subdomain caches in sequence.
    // ! Note that this Function is "fragile" in that it will only work with the union BasisCache, or one that imitates it identically in terms of point layout.
    static TFunctionPtr<Scalar> unionBasisCacheSubdomainFunctionWrapper(TFunctionPtr<Scalar> subdomainFunction, const std::vector<BasisCachePtr> &subdomainCaches);
  };
}


#endif /* BasisCacheFactory_hpp */
