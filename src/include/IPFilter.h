//
//  IPFilter.h
//  Camellia
//
//  Created by Nate Roberts on 10/10/14.
//
//

#ifndef Camellia_IPFilter_h
#define Camellia_IPFilter_h

#include "TypeDefs.h"

#include "IP.h"
#include "LocalStiffnessMatrixFilter.h"

namespace Camellia {
	/*
	 Prototype IPFilter.
	 
	 Allows the definition of a LocalStiffnessMatrixFilter via a trial-space IP object.  Notion is that you might want to minimize (e_i, e_j)_E + (e_i, e_j)_ip, where E is the energy norm, and _ip is the inner product defined by the IP object.  Could be used for resolution of the extra constant mode in Stokes operator, e.g., if _ip were defined as (p,p)_L^2, say.
	 */

	class IPFilter : public LocalStiffnessMatrixFilter {
	  IPPtr _ip;
	public:
	  IPFilter(IPPtr ip); // idea is to use an IP on the trial space to add L^2 penalty terms.  In the case of Stokes I want integral of p = 0, so I might use an ip defined as (p,p).
	  void filter(Intrepid::FieldContainer<double> &localStiffnessMatrix, Intrepid::FieldContainer<double> &localRHSVector,
	              BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc);
	};
}

#endif
