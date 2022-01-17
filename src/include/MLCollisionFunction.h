//
//  MLCollisionFunction.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/14/22.
//

#ifndef MLCollisionFunction_h
#define MLCollisionFunction_h

#include "Function.h"
#include "SolutionMoments.h"

#ifdef HAVE_MLC
#include "MLC_CollisionModel.hpp"
#include "MLC_MomentConversion.hpp"
#endif

namespace Camellia
{
  class MLCollisionFunction : public TFunction<double>
  {
    FunctionPtr _f_soln;
    Teuchos::RCP<SolutionMoments> _f_solnMoments;
    
    FunctionPtr _bgkFunction;

    // MLC arguments:
    std::string _modelDirectory;
    
    // CollisionModel takes numPoints as a construction argument.  Until we make mlc more flexible on this front, we need to reload from disk every time we want a different number of points.
    int _numPointsForLoadedModel = 0;
    Teuchos::RCP<mlc::CollisionModel> _loadedModel;
  public:
    MLCollisionFunction(FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments, std::string &modelDirPath);
    
    static int maxMomentNumber(std::string &modelDirPath);
    
    using TFunction<double>::values;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };
}

#endif /* MLCCollisionFunction_h */
