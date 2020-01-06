//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ExactSolutionFunction.h"

#include "BasisCache.h"
#include "ExactSolution.h"

using namespace Camellia;

template <typename Scalar>
ExactSolutionFunction<Scalar>::ExactSolutionFunction(Teuchos::RCP<ExactSolution<Scalar>> exactSolution, int trialID)
  : TFunction<Scalar>(exactSolution->exactFunctions().find(trialID)->second->rank())
{
  _exactSolution = exactSolution;
  _trialID = trialID;
}
template <typename Scalar>
void ExactSolutionFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  _exactSolution->solutionValues(values,_trialID,basisCache);
}

namespace Camellia
{
template class ExactSolutionFunction<double>;
}
