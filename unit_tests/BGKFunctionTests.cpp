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

#include "BasisCache.h"
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

  using Data = Intrepid2::Data<double, Camellia::DeviceType>;

  template <class ViewType>
  std::vector<double> flattenedViewValues(const ViewType &view)
  {
    std::vector<double> values(view.size());
    const double* viewData = view.data();
    for (size_t i=0; i<values.size(); i++)
    {
      values[i] = viewData[i];
    }
    return values;
  }

  std::vector<double> flattenedDataValues(const Data &data)
  {
    switch (data.getUnderlyingViewRank())
    {
      case 1:
        return flattenedViewValues(data.getUnderlyingView<1>());
      case 2:
        return flattenedViewValues(data.getUnderlyingView<2>());
      case 3:
        return flattenedViewValues(data.getUnderlyingView<3>());
      case 4:
        return flattenedViewValues(data.getUnderlyingView<4>());
      case 5:
        return flattenedViewValues(data.getUnderlyingView<5>());
      case 6:
        return flattenedViewValues(data.getUnderlyingView<6>());
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported BGK test Data rank");
    }
  }

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
                                 Teuchos::FancyOStream &out, bool &success,
                                 const bool useHierarchicalFieldBasis = false)
  {
    const int numConfigurationDims = useSpaceTime ? numSpatialDims + 1 : numSpatialDims;
    
    auto maxwellian = MaxwellianFunction::maxwellian(numConfigurationDims, numVelocityDims, u, sigmaSquared, n);
    
    const int delta_k = 0;
    
    const int numPhaseDims = numConfigurationDims + numVelocityDims;
    const int numConfigurationTensorComponents = useSpaceTime ? 2 : 1;
    const int numTensorComponents = numConfigurationTensorComponents + numVelocityDims;
    
    std::vector<int> H1Order(numTensorComponents, meshPolyOrder + 1);
    
    auto meshTopo = makeTestMeshTopology(numSpatialDims, numVelocityDims, useSpaceTime);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", HGRAD);
    
    BFPtr bf = BF::bf(vf);
    
    if (useHierarchicalFieldBasis)
    {
      f->setBasisFamily(BasisFamily::hierarchicalBasisFamily());
    }
    
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const bool allowPruning = false;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
    
    SolutionPtr soln = Solution::solution(mesh);
    std::map<int, FunctionPtr> functionMap;
    functionMap[f->ID()] = maxwellian;
    
    soln->projectOntoMesh(functionMap);
    
    const int maxMomentNumber = 2; // BGK requires 2nd moment
    
    MeshTopologyViewPtr configurationMeshTopology = meshTopo->physicalMeshTopology();
    
    std::vector<int> configurationH1Order(numConfigurationDims, meshPolyOrder + 1);

    MeshPtr physicalMesh = MeshFactory::minRuleMesh(configurationMeshTopology, bf, configurationH1Order, delta_k, comm, allowPruning);
    
    const int survivingExtrusionDim = -1; // SolutionMoments should integrate in all extrusion dimensions
    Teuchos::RCP<SolutionMoments> solnMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, soln, f, physicalMesh, survivingExtrusionDim));
    solnMoments->update();
    
    auto f_soln = Function::solution(f, soln);
    
    auto bgkFunction = Teuchos::rcp( new BGKFunction(TEST_TAU, f_soln, solnMoments) );
    
    using Data = Intrepid2::Data<double, Camellia::DeviceType>;
    
    auto myCellIDs = mesh->cellIDsInPartition();
    std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(), myCellIDs.end());
    bgkFunction->importCellData(myCellIDsVector);
    
    for (const auto & cellID : myCellIDs)
    {
      auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      Data bgkData = bgkFunction->checkoutValuesData(basisCache);
      
      bgkFunction->values(bgkData, basisCache);
      
      std::vector<double> actualValues = flattenedDataValues(bgkData);
      std::vector<double> expectedZeroValues(actualValues.size(), 0.0);
      
      out << "Checking BGK values are near zero on cell " << cellID << " with " << actualValues.size() << " entries" << std::endl;
      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(actualValues, expectedZeroValues, 0.0, tol);
      
      bgkFunction->checkinValuesData(bgkData);
    }
  }

  void testBGKOfConstantOneMatchesAnalyticValue(const int meshPolyOrder,
                                                const double relTol, const double absTol,
                                                Teuchos::FancyOStream &out, bool &success)
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 1;
    const bool useSpaceTime = false;
    const int numConfigurationDims = numSpatialDims;
    const int numPhaseDims = numConfigurationDims + numVelocityDims;
    const int delta_k = 0;
    
    std::vector<int> H1Order(numPhaseDims, meshPolyOrder + 1);
    
    auto meshTopo = makeTestMeshTopology(numSpatialDims, numVelocityDims, useSpaceTime);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", HGRAD);
    
    BFPtr bf = BF::bf(vf);
    
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const bool allowPruning = false;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
    
    SolutionPtr soln = Solution::solution(mesh);
    std::map<int, FunctionPtr> functionMap;
    functionMap[f->ID()] = Function::constant(1.0);
    
    soln->projectOntoMesh(functionMap);
    
    const int maxMomentNumber = 2;
    MeshTopologyViewPtr configurationMeshTopology = meshTopo->physicalMeshTopology();
    std::vector<int> configurationH1Order(numConfigurationDims, meshPolyOrder + 1);
    MeshPtr physicalMesh = MeshFactory::minRuleMesh(configurationMeshTopology, bf, configurationH1Order, delta_k, comm, allowPruning);
    
    const int survivingExtrusionDim = -1;
    Teuchos::RCP<SolutionMoments> solnMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, soln, f, physicalMesh, survivingExtrusionDim));
    solnMoments->update();
    
    auto f_soln = Function::solution(f, soln);
    auto bgkFunction = Teuchos::rcp( new BGKFunction(TEST_TAU, f_soln, solnMoments) );
    
    auto myCellIDs = mesh->cellIDsInPartition();
    std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(), myCellIDs.end());
    bgkFunction->importCellData(myCellIDsVector);
    
    for (const auto & cellID : myCellIDs)
    {
      auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      const auto &points = basisCache->getPhysicalCubaturePointsFC();
      const int numCells = points.dimension(0);
      const int numPoints = points.dimension(1);
      
      FieldContainer<double> bgkValues(numCells, numPoints);
      bgkFunction->values(bgkValues, basisCache);
      
      // For f(v) = 1 on [-1,1], the finite-domain moments are n = 2, u = 0,
      // and sigma^2 = (1/n) int_{-1}^1 v^2 dv = 1/3.  BGK should therefore be
      // -1/tau times the difference between f and the Maxwellian reconstructed
      // from those finite-domain moments.
      const double n = EXTRUSION_EXTENT;
      const double sigmaSquared = 1.0 / 3.0;
      const double pi = std::acos(-1.0);
      const double normalization = n / std::sqrt(2.0 * pi * sigmaSquared);
      
      std::vector<double> actualValues;
      std::vector<double> expectedValues;
      actualValues.reserve(numCells * numPoints);
      expectedValues.reserve(numCells * numPoints);
      
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double velocity = points(cellOrdinal, pointOrdinal, numConfigurationDims);
          const double fM = normalization * std::exp(-0.5 * velocity * velocity / sigmaSquared);
          actualValues.push_back(bgkValues(cellOrdinal, pointOrdinal));
          expectedValues.push_back((-1.0 / TEST_TAU) * (1.0 - fM));
        }
      }
      
      TEUCHOS_TEST_FOR_EXCEPTION(actualValues.size() != expectedValues.size(), std::invalid_argument, "BGK test expected value count does not match actual value count");
      
      out << "Checking BGK values match the analytic non-equilibrium constant-distribution result on cell " << cellID << std::endl;
      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(actualValues, expectedValues, relTol, absTol);
    }
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_1V )
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 1;
    const bool useSpaceTime = true; // exercises rank-3 configuration moment data
    const int meshPolyOrder = 8;
    
    // Keep the distribution centered and well inside the finite velocity interval [-1,1].
    // Otherwise, the moments computed over the truncated velocity domain do not reproduce
    // the full-space Maxwellian parameters, and BGK(f_M) is not expected to be zero.
    const std::vector<double> u { 0.0 };
    const double sigmaSquared = 0.04;
    const double n = 1.0;
    const bool refine = false;
    
    const double tol = 2e-3;
    
    testBGKOfMaxwellianIsZero(numSpatialDims, numVelocityDims, useSpaceTime,
                              meshPolyOrder,
                              u, sigmaSquared, n,
                              refine,
                              tol,
                              out, success);
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistribution_NonzeroDrift_1V )
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 1;
    const bool useSpaceTime = true;
    const int meshPolyOrder = 8;
    
    // This exercises the first-moment path and velocity-coordinate offset with a
    // drift that is still far enough from the finite velocity-domain boundary to
    // keep truncation error modest.
    const std::vector<double> u { 0.2 };
    const double sigmaSquared = 0.04;
    const double n = 1.0;
    const bool refine = false;
    
    const double tol = 1e-2;
    
    testBGKOfMaxwellianIsZero(numSpatialDims, numVelocityDims, useSpaceTime,
                              meshPolyOrder,
                              u, sigmaSquared, n,
                              refine,
                              tol,
                              out, success);
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistribution_Rank4MomentData )
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 2;
    const bool useSpaceTime = true; // 1 spatial + 1 temporal configuration dims exercises rank-4 moment data
    const int meshPolyOrder = 5;
    
    const std::vector<double> u { 0.0, 0.0 };
    const double sigmaSquared = 0.04;
    const double n = 1.0;
    const bool refine = false;
    
    const double tol = 5e-2;
    
    const bool useHierarchicalFieldBasis = true;
    testBGKOfMaxwellianIsZero(numSpatialDims, numVelocityDims, useSpaceTime,
                              meshPolyOrder,
                              u, sigmaSquared, n,
                              refine,
                              tol,
                              out, success,
                              useHierarchicalFieldBasis);
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_2V )
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 2;
    const bool useSpaceTime = false;
    const int meshPolyOrder = 6;
    
    const std::vector<double> u { 0.0, 0.0 };
    const double sigmaSquared = 0.04;
    const double n = 1.0;
    const bool refine = false;
    
    const double tol = 1e-2;
    
    testBGKOfMaxwellianIsZero(numSpatialDims, numVelocityDims, useSpaceTime,
                              meshPolyOrder,
                              u, sigmaSquared, n,
                              refine,
                              tol,
                              out, success);
  }

  TEUCHOS_UNIT_TEST( BGKFunction, MaxwellianDistributions_3V )
  {
    const int numSpatialDims = 1;
    const int numVelocityDims = 3;
    const bool useSpaceTime = false;
    const int meshPolyOrder = 5;
    
    const std::vector<double> u { 0.0, 0.0, 0.0 };
    const double sigmaSquared = 0.04;
    const double n = 1.0;
    const bool refine = false;
    
    const double tol = 5e-2;
    
    testBGKOfMaxwellianIsZero(numSpatialDims, numVelocityDims, useSpaceTime,
                              meshPolyOrder,
                              u, sigmaSquared, n,
                              refine,
                              tol,
                              out, success);
  }

  TEUCHOS_UNIT_TEST( BGKFunction, ConstantDistribution_1V_AnalyticValue )
  {
    const int meshPolyOrder = 2;
    const double relTol = 1e-8;
    const double absTol = 1e-8;
    
    testBGKOfConstantOneMatchesAnalyticValue(meshPolyOrder, relTol, absTol, out, success);
  }
} // end anonymous namespace
