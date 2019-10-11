//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  EnergyErrorIndicator.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#include "EnergyErrorIndicator.h"
#include "RieszRep.h"

using namespace Camellia;

//! Solution-based energy error computation
template <typename Scalar>
EnergyErrorIndicator<Scalar>::EnergyErrorIndicator(TSolutionPtr<Scalar> soln) : ErrorIndicator(soln->mesh())
{
  _solution = soln;
}

//! Energy error computation using the RieszRep class.
template <typename Scalar>
EnergyErrorIndicator<Scalar>::EnergyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                   int cubatureEnrichmentDegree) : ErrorIndicator(mesh)
{
  _rieszRep = Teuchos::rcp( new TRieszRep<Scalar>(mesh, ip, residual) );
  _rieszRepCubatureEnrichment = cubatureEnrichmentDegree;
}

//! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
template <typename Scalar>
void EnergyErrorIndicator<Scalar>::measureError()
{
  const map<GlobalIndexType, double>* rankLocalEnergyError;
  bool energyErrorIsSquared;
  if (_rieszRep.get() != NULL)
  {
    _rieszRep->computeRieszRep(_rieszRepCubatureEnrichment);
    rankLocalEnergyError = &_rieszRep->getNormsSquared();
    // will need to take square roots:
    energyErrorIsSquared = true;
  }
  else
  {
    rankLocalEnergyError = &_solution->rankLocalEnergyError();
    // square roots have already been taken
    energyErrorIsSquared = false;
  }
  _localErrorMeasures.clear();
  for (auto entry : *rankLocalEnergyError)
  {
    GlobalIndexType cellID = entry.first;
    double error = energyErrorIsSquared ? sqrt(entry.second) : entry.second;
    _localErrorMeasures[cellID] = error;
  }
}

// ETI for double 
template class Camellia::EnergyErrorIndicator<double>;
