//
//  IPFilter.h
//  Camellia
//
//  Created by Nate Roberts on 10/10/14.
//
//

#ifndef Camellia_IPFilter_h
#define Camellia_IPFilter_h

#include "IP.h"
#include "LocalStiffnessMatrixFilter.h"

class IPFilter : public LocalStiffnessMatrixFilter {
  IPPtr _ip;
public:
  IPFilter(IPPtr ip); // idea is to use an IP on the trial space to add L^2 penalty terms.  In the case of Stokes I want integral of p = 0, so I might use an ip defined as (p,p).
  void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector,
              BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc);
};

#endif
