#include <stdio.h>

//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BGKFunctionTests.cpp
//  runTests
//
//  Created by Roberts, Nathan V on 3/24/21.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia_TensorIndices.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "ExtrudedMeshTopology.h"
#include "SolutionMoments.h"
#include "BGKFunction.h"
#include "MaxwellianFunction.h"
#include "SubExtrudedMeshTopology.h"
#include "MeshFactory.h"
#include "TensorBasis.h"

using namespace Camellia;
using namespace std;

namespace
{
  static const double EXTRUSION_EXTENT =  2.0; // extent of each extrusion dimension
  static const double EXTRUSION_ORIGIN = -1.0; // starting coordinate in each extrusion dimension
  static const double PHYSICAL_EXTENT  =  1.0; // extent in each physical dimension
  static const double PHYSICAL_ORIGIN  =  0.0; // starting coordinate in each physical dimension
  
  static const double TEMPORAL_ORIGIN  =  0.0;  // used for space-time meshes
  static const double TEMPORAL_EXTENT  =  1.0;  // used for space-time meshes
  
  static const int PHYSICAL_MESH_WIDTH  = 2;
  static const int EXTRUSION_MESH_WIDTH = 2;
  static const int TEMPORAL_MESH_WIDTH  = 2;  // used for space-time meshes

  static const double TEST_TAU = 0.25; // arbitrary value; mostly want to avoid 1.0 value

  Teuchos::RCP<ExtrudedMeshTopology> makeTestMeshTopology( int numSpatialDims, int numVelocityDims, bool useSpaceTime )
  {
    using std::vector;
    vector<double> spatialDims(numSpatialDims,PHYSICAL_EXTENT);
    vector<int> spatialElementCounts(numSpatialDims,PHYSICAL_MESH_WIDTH);
    vector<double> spatialOrigin(numSpatialDims,PHYSICAL_ORIGIN);
    auto configurationMeshTopo = MeshFactory::rectilinearMeshTopology(spatialDims, spatialElementCounts, spatialOrigin);
    
    if (useSpaceTime)
    {
      const int numTemporalExtrusionDims = 1;
      vector<double> temporalStartCoords(numTemporalExtrusionDims, TEMPORAL_ORIGIN);
      vector<double> temporalExtents(numTemporalExtrusionDims, TEMPORAL_EXTENT);
      vector<int> temporalElementCounts(numTemporalExtrusionDims,TEMPORAL_MESH_WIDTH);
      
      configurationMeshTopo = Teuchos::rcp(new ExtrudedMeshTopology(configurationMeshTopo,temporalStartCoords,temporalExtents,temporalElementCounts));
    }
    
    vector<double> extrusionStartCoords(numVelocityDims, EXTRUSION_ORIGIN);
    vector<double> extrusionExtents(numVelocityDims, EXTRUSION_EXTENT);
    vector<int> extrusionElementCounts(numVelocityDims,EXTRUSION_MESH_WIDTH);
    
    auto phaseMeshTopology = Teuchos::rcp(new ExtrudedMeshTopology(configurationMeshTopo,extrusionStartCoords,extrusionExtents,extrusionElementCounts));
    return phaseMeshTopology;
  }

  void testBGKOfMaxwellianIsZero(const int numSpatialDims, const int numVelocityDims, const bool useSpaceTime,
                                 const int meshPolyOrder,
                                 const std::vector<double> u, const double sigmaSquared, const double n,
                                 bool refine,
                                 const double tol,
                                 Teuchos::FancyOStream &out, bool &success)
  {
    const int numConfigurationDims = useSpaceTime ? numSpatialDims + 1 : numSpatialDims;
    
    auto maxwellian = MaxwellianFunction::maxwellian(numConfigurationDims, numVelocityDims, u, sigmaSquared, n);
    
    const int delta_k = 0;
    
    const int numPhaseDims = numConfigurationDims + numVelocityDims;
    
    std::vector<int> H1Order(numPhaseDims, meshPolyOrder + 1);
    
    auto meshTopo = makeTestMeshTopology(numSpatialDims, numVelocityDims, useSpaceTime);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", HGRAD);
    
    BFPtr bf = BF::bf(vf);
    
    Epetra_CommPtr comm = MPIWrapper::CommWorld();
    const bool allowPruning = false;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
    
    SolutionPtr soln = Solution::solution(mesh);
    std::map<int, FunctionPtr> functionMap;
    functionMap[f->ID()] = maxwellian;
    
    soln->projectOntoMesh(functionMap);
    
    const int maxMomentNumber = 2; // BGK requires 2nd moment
    
    MeshTopologyViewPtr configurationMeshTopology = meshTopo->physicalMeshTopology();
    
    
    std::vector<int> configurationH1Order{numConfigurationDims, meshPolyOrder + 1};

    MeshPtr physicalMesh = MeshFactory::minRuleMesh(configurationMeshTopology, bf, configurationH1Order, delta_k, comm, allowPruning);
    
    const int survivingExtrusionDim = -1; // SolutionMoments should integrate in all extrusion dimensions
    SolutionMoments solnMoments(maxMomentNumber, soln, f, physicalMesh, survivingExtrusionDim);
    solnMoments.update();
    
    // TODO: instantiate BGK Operator, and test that its values are near zero…
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_1V )
  {
    // TODO: write this test
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_2V )
  {
    // TODO: write this test
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_3V )
  {
    // TODO: write this test
  }
} // end anonymous namespace
