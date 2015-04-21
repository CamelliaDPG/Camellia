#include "ExactSolutionFunction.h"

#include "BasisCache.h"
#include "ExactSolution.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
ExactSolutionFunction<Scalar>::ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID)
: TFunction<Scalar>(exactSolution->exactFunctions().find(trialID)->second->rank()) {
  _exactSolution = exactSolution;
  _trialID = trialID;
}
template <typename Scalar>
void ExactSolutionFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
  _exactSolution->solutionValues(values,_trialID,basisCache);
}

namespace Camellia {
  template class ExactSolutionFunction<double>;
}