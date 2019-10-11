// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  EnergyErrorIndicator.h
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#ifndef __Camellia__EnergyErrorIndicator__
#define __Camellia__EnergyErrorIndicator__

#include "ErrorIndicator.h"
#include "LinearTerm.h"
#include "IP.h"

namespace Camellia
{
  template <typename Scalar>
  class EnergyErrorIndicator : public ErrorIndicator
  {
    TSolutionPtr<Scalar> _solution;
    TRieszRepPtr<Scalar> _rieszRep;
    
    int _rieszRepCubatureEnrichment = 0;
  public:
    //! Solution-based energy error computation
    EnergyErrorIndicator(TSolutionPtr<Scalar> soln);
    
    //! Energy error computation using the RieszRep class.
    EnergyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, int cubatureEnrichmentDegree = 0);
    
    //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
    virtual void measureError();
  };
}

#endif /* defined(__Camellia__EnergyErrorIndicator__) */
