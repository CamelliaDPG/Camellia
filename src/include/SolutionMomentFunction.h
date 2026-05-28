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
    
    int momentEntryOrdinal() const { return _momentEntryOrdinal; }
    
    using TFunction<double>::values;
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };

  class RelativisticFirstMomentFunction : public TFunction<double>
  {
    // Non-owning references are intentional here: these functions may be retained inside RHS terms for
    // configuration-space solves, and strong ownership would unnecessarily retain the phase-space solution
    // / mesh graph and its associated topology/observer objects.
    SolutionPtr _phaseSpaceSolution;
    VarPtr _fieldVar;
    MeshPtr _configurationSpaceMesh;
    double _c;
    int _velocityComponent;
  public:
    RelativisticFirstMomentFunction(SolutionPtr phaseSpaceSolution, VarPtr fieldVar,
                                    MeshPtr configurationSpaceMesh, double c,
                                    int velocityComponent);

    using TFunction<double>::values;
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };
}

#endif /* SolutionMomentFunction_h */
