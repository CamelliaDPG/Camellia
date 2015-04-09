#include "ExactSolutionFunction.h"

#include "BasisCache.h"
#include "ExactSolution.h"

using namespace Camellia;
using namespace Intrepid;

ExactSolutionFunction::ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID)
: Function(exactSolution->exactFunctions().find(trialID)->second->rank()) {
  _exactSolution = exactSolution;
  _trialID = trialID;
}
void ExactSolutionFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  _exactSolution->solutionValues(values,_trialID,basisCache);
}
