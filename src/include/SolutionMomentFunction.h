//
//  SolutionMomentFunction.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/24/21.
//

#ifndef SolutionMomentFunction_h
#define SolutionMomentFunction_h

#include "Camellia_FieldContainerConversion.hpp"
#include "Camellia_TensorIndices.h"
#include "CubatureFactory.h"
#include "Function.h"
#include "MPIWrapper.h"
#include "SolutionMoments.h"
#include "SubExtrudedMeshTopology.h"
#include "TensorBasis.h"

namespace Camellia
{
  class SolutionMomentFunction : public TFunction<double>
  {
    SolutionMoments & _solutionMoments; // common storage, etc. for all moments
    const int _momentEntryOrdinal;
  public:
    SolutionMomentFunction(SolutionMoments &solutionMoments, const int &momentEntryOrdinal);
    
    using TFunction<double>::values;
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };
}

#endif /* SolutionMomentFunction_h */
