//
//  SpatialIntegralOfSpaceTimeSolution.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 11/23/22.
//

#ifndef SpatialIntegralOfSpaceTimeSolution_hpp
#define SpatialIntegralOfSpaceTimeSolution_hpp

#include "TypeDefs.h"

namespace Camellia
{
  //! Defined on the temporal mesh.
  class SpatialIntegralOfSpaceTimeSolution
  {
    using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
    
    BasisPtr _extrusionBasis;
    BasisPtr _spatialBasis; // _spatialBasis x extrusionBasis = _spaceTimeBasis.
    BasisPtr _spaceTimeBasis;
    
    MeshPtr     _temporalMesh;
    SolutionPtr _temporalSolution; // defined on the temporal mesh
    FunctionPtr _temporalFunction; // defined with reference to _temporalSolution
  public:
    //! spatialWeight: a function to integrate against (e.g., could be sin (x/2) for a Fourier mode computation).  quadratureEnrichment: how much higher-order to go for quadrature to account for integrating against spatialWeight.
    SpatialIntegralOfSpaceTimeSolution(SolutionPtr spaceTimeSoln, VarPtr fieldVar, FunctionPtr spatialWeight, int quadratureEnrichment);
    
    MeshPtr temporalMesh();
    FunctionPtr temporalFunction(); // defined on the temporal mesh
    SolutionPtr temporalSolution();
    
    void exportViz(std::string parentDir, std::string name);
  };
}

#endif /* SpatialIntegralOfSpaceTimeSolution_hpp */
