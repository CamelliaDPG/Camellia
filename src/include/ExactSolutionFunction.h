//
//  ExactSolutionFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ExactSolutionFunction_h
#define Camellia_ExactSolutionFunction_h

#include "ExactSolution.h"
#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class ExactSolutionFunction : public Function<Scalar> { // for scalars, for now
    Teuchos::RCP<ExactSolution> _exactSolution;
    int _trialID;
  public:
    ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };
}
#endif
