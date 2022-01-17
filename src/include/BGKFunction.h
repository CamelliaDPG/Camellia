//
//  BGKFunction.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 12/22/21.
//

#ifndef BGKFunction_h
#define BGKFunction_h

#include "Function.h"
#include "SolutionMoments.h"

namespace Camellia
{
  class BGKFunction : public TFunction<double>
  {
    double _tau;
    FunctionPtr _f_soln;
    Teuchos::RCP<SolutionMoments> _f_solnMoments;
    
    FunctionPtr _zeroMoment;
    std::vector<FunctionPtr> _firstMoments; // up to three of these
    FunctionPtr _secondMoment;
    
  public:
    BGKFunction(const double tau, FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments);
    
    using TFunction<double>::values;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };
}

#endif /* BGKFunction_h */
