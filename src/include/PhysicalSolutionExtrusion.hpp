//
//  PhysicalSolutionExtrusion.hpp
//  BoltzmannTests
//
//  Created by Roberts, Nathan V on 12/6/21.
//

#ifndef PhysicalSolutionExtrusion_hpp
#define PhysicalSolutionExtrusion_hpp

#include "Camellia.h"
#include "Function.h"

namespace Camellia
{

  class PhysicalSolutionExtrusion : public TFunction<double>
  {
    SolutionPtr _physicalSoln;    // the solution on the physical mesh
    VarPtr      _fieldVar;        // the physical field var whose solution we extrude.
    FunctionPtr _physicalSolnFxn; // the solution function for the field var, defined on the physical mesh.
    
    MeshPtr     _phaseMesh;       // the  phase space mesh
    static inline int variationDimensions(SolutionPtr physicalSoln)
    {
      auto physicalMeshDims = physicalSoln->mesh()->getDimension();
      int variationDimensions = VARIES_IN_NO_DIMENSION;
      for (int d=0; d<physicalMeshDims; d++)
      {
        int varDim = VARIES_IN_DIMENSION_0 << d; // will give VARIES_IN_DIMENSION_1 for d == 1, etc.
        variationDimensions |= varDim;
      }
      return variationDimensions;
    }
  public:
    PhysicalSolutionExtrusion( SolutionPtr physicalSoln, VarPtr fieldVar, MeshPtr phaseMesh);
    
    std::string displayString() override;
    
    void updateValues();
    
    using TFunction<double>::values;
    void values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache) override;
    
    bool boundaryValueOnly() override;
  };

}

#endif /* PhysicalSolutionExtrusion_hpp */
