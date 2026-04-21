#include <stdio.h>

//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SolutionMomentFunctionTests.cpp
//  runTests
//
//  Created by Roberts, Nathan V on 3/24/21.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia_TensorIndices.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "ExtrudedMeshTopology.h"
#include "MaxwellianFunction.h"
#include "MonomialFunctions.h"
#include "SolutionMoments.h"
#include "SolutionMomentFunction.h"
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

  Teuchos::RCP<ExtrudedMeshTopology> makeTestMeshTopology( int numPhysicalDims, int numExtrusionDims, bool useSpaceTime,
                                                          double extrusionOrigin = EXTRUSION_ORIGIN, double extrusionExtent = EXTRUSION_EXTENT,
                                                          int extrusionMeshWidth = EXTRUSION_MESH_WIDTH)
  {
    using std::vector;
    vector<double> physicalDims(numPhysicalDims,PHYSICAL_EXTENT);
    vector<int> physicalElementCounts(numPhysicalDims,PHYSICAL_MESH_WIDTH);
    vector<double> physicalOrigin(numPhysicalDims,PHYSICAL_ORIGIN);
    auto physicalMeshTopo = MeshFactory::rectilinearMeshTopology(physicalDims, physicalElementCounts, physicalOrigin);
    
    if (useSpaceTime)
    {
      const int numTemporalExtrusionDims = 1;
      vector<double> temporalStartCoords(numTemporalExtrusionDims, TEMPORAL_ORIGIN);
      vector<double> temporalExtents(numTemporalExtrusionDims, TEMPORAL_EXTENT);
      vector<int> temporalElementCounts(numTemporalExtrusionDims,TEMPORAL_MESH_WIDTH);
      
      physicalMeshTopo = Teuchos::rcp(new ExtrudedMeshTopology(physicalMeshTopo,temporalStartCoords,temporalExtents,temporalElementCounts));
    }
    
    vector<double> extrusionStartCoords(numExtrusionDims, extrusionOrigin);
    vector<double> extrusionExtents(numExtrusionDims, extrusionExtent);
    vector<int> extrusionElementCounts(numExtrusionDims,extrusionMeshWidth);
    
    auto extrudedMeshTopology = Teuchos::rcp(new ExtrudedMeshTopology(physicalMeshTopo,extrusionStartCoords,extrusionExtents,extrusionElementCounts));
    return extrudedMeshTopology;
  }

  void testIntegrate(const int numPhysicalDims, const int numExtrusionDims, const bool useSpaceTime,
                     int survivingExtrusionDim, // use -1 for no surviving dims
                     const int fxnPolyOrder,
                     const bool useSerendipity,
                     FunctionPtr phaseSpaceFxn, FunctionPtr expectedIntegral,
                     const int &iMomentNumber, const int &jMomentNumber, const int &kMomentNumber,
                     bool refine, bool pCoarsen,
                     const double tol,
                     Teuchos::FancyOStream &out, bool &success)
  {
    // TODO: something to test anisotropy in p
    
    const int delta_k = 0;
    
    std::vector<int> H1Order(1 + numExtrusionDims, fxnPolyOrder + 1);
    
    auto meshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims, useSpaceTime);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", HGRAD);
    
    if (useSerendipity)
    {
      f->setBasisFamily(BasisFamily::serendipityBasisFamily());
    }
    else if (pCoarsen)
    {
      // variable-p requires hierarchical basis
      f->setBasisFamily(BasisFamily::hierarchicalBasisFamily());
    }
    
    BFPtr bf = BF::bf(vf);
    
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const bool allowPruning = false;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
    
    std::vector<int> physicalH1Order{fxnPolyOrder + 1};
    
    MeshTopologyViewPtr extrudedPhysicalMeshTopology;
    if (survivingExtrusionDim == -1)
    {
      // all are integrated
      extrudedPhysicalMeshTopology = meshTopo->physicalMeshTopology();
    }
    else
    {
      extrudedPhysicalMeshTopology = Teuchos::rcp( new SubExtrudedMeshTopology(meshTopo, survivingExtrusionDim) );
    }
    
    MeshPtr physicalMesh = MeshFactory::minRuleMesh(extrudedPhysicalMeshTopology, bf, physicalH1Order, delta_k, comm, allowPruning);
    
    if (survivingExtrusionDim == -1)
    {
      // while SubExtrudedMeshTopology knows about phase space mesh topology, plain physical mesh does not
      // for that reason, we need to register for updates
      PhysicalMeshUpdater::registerForUpdates(mesh, physicalMesh);
    }
    
    // confirm cell counts match expected before any refinement
    int startingPhysicalCellCount = 1;
    for (int d=0; d<numPhysicalDims; d++)
    {
      startingPhysicalCellCount *= PHYSICAL_MESH_WIDTH;
    }
    if (useSpaceTime)
    {
      startingPhysicalCellCount *= TEMPORAL_MESH_WIDTH;
    }
    int expectedActivePhaseSpaceCellCount = startingPhysicalCellCount;
    for (int d=0; d<numExtrusionDims; d++)
    {
      expectedActivePhaseSpaceCellCount *= EXTRUSION_MESH_WIDTH;
    }
    int expectedActivePhysicalCellCount = startingPhysicalCellCount;
    if (survivingExtrusionDim != -1)
    {
      expectedActivePhysicalCellCount *= EXTRUSION_MESH_WIDTH;
    }
    TEST_EQUALITY(expectedActivePhysicalCellCount, physicalMesh->numActiveElements());
    TEST_EQUALITY(expectedActivePhaseSpaceCellCount, mesh->numActiveElements());
    
    if (refine)
    {
      const IndexType cellToRefine = 0;
      
      int numTemporalDims = useSpaceTime ? 1 : 0;
      
      int newCellsPerRefinement = 1 << (numExtrusionDims + numPhysicalDims + numTemporalDims); // isotropic refinement
      
      set<GlobalIndexType> cellsToRefine {cellToRefine};
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(mesh->getElementType(cellToRefine)->cellTopoPtr);
      mesh->hRefine(cellsToRefine, refPattern);
      
      TEST_EQUALITY(newCellsPerRefinement, refPattern->numChildren());
      
      expectedActivePhaseSpaceCellCount += newCellsPerRefinement - 1; // -1: the parent cell is no longer active
      
      int newPhysicalCellsPerRefinement = 1 << (numPhysicalDims + numTemporalDims);
      const int newExtrusionCellsPerRefinement = 2;
      if (survivingExtrusionDim != -1)
      {
        expectedActivePhysicalCellCount = (startingPhysicalCellCount + newPhysicalCellsPerRefinement - 1) * (EXTRUSION_MESH_WIDTH + newExtrusionCellsPerRefinement - 1);
      }
      else
      {
        expectedActivePhysicalCellCount = (startingPhysicalCellCount + newPhysicalCellsPerRefinement - 1);
      }
      
      TEST_EQUALITY(expectedActivePhysicalCellCount, physicalMesh->numActiveElements());
      TEST_EQUALITY(expectedActivePhaseSpaceCellCount, mesh->numActiveElements());
    }
    
    // now that mesh is set up, project phaseSpaceFxn onto solution.
    SolutionPtr soln = Solution::solution(mesh);
    std::map<int, FunctionPtr> functionMap;
    functionMap[f->ID()] = phaseSpaceFxn;
    
    mesh->registerSolution(soln);
    
    soln->projectOntoMesh(functionMap);
    
    const int maxMomentNumber = iMomentNumber + jMomentNumber + kMomentNumber;
    SolutionMoments solnMoments(maxMomentNumber, soln, f, physicalMesh, survivingExtrusionDim);
    
    FunctionPtr solnMomentFunction;
    const int numSurvivingDims  = (survivingExtrusionDim == -1) ? 0 : 1;
    const int numIntegratedDims = numExtrusionDims - numSurvivingDims;
    if (numIntegratedDims == 1)
    {
      solnMomentFunction = solnMoments.solutionMomentFunction(iMomentNumber);
    }
    else if (numIntegratedDims == 2)
    {
      solnMomentFunction = solnMoments.solutionMomentFunction(iMomentNumber,jMomentNumber);
    }
    else if (numIntegratedDims == 3)
    {
      solnMomentFunction = solnMoments.solutionMomentFunction(iMomentNumber,jMomentNumber,kMomentNumber);
    }
    
    // because we don't allow pruning, cell 0 is known everywhere.  We use it to determine spatial quadrature points on the reference element.
    BasisCachePtr sampleBasisCache = BasisCache::basisCacheForCell(physicalMesh, 0);
    solnMoments.update();
    
    if (pCoarsen)
    {
      // we reduce p in every odd-numbered active cell by 1
      auto activeCellIDs = mesh->getActiveCellIDsGlobal();
      std::set<GlobalIndexType> oddCellIDs;
      for (auto cellID : activeCellIDs)
      {
        if ((cellID % 2) == 1)
        {
          oddCellIDs.insert(cellID);
        }
      }
      mesh->pRefine(oddCellIDs, -1);
    }
    solnMoments.update();
    
    FunctionPtr errorFxn = expectedIntegral - solnMomentFunction;
    
    if (numExtrusionDims == 1)
    {
      out << "Testing whether solnMomentFunction Data path matches View path" << std::endl;
      const auto & myCells = physicalMesh->cellIDsInPartition();
      for (const auto & physicalCellID : myCells)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
        testFunctionDataPathMatchesViewPath1V(solnMomentFunction, basisCache, tol, out, success);
      }
    }
    else if (numExtrusionDims == 3)
    {
      out << "Testing whether solnMomentFunction Data path matches View path" << std::endl;
      const auto & myCells = physicalMesh->cellIDsInPartition();
      for (const auto & physicalCellID : myCells)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
        testFunctionDataPathMatchesViewPath3V(solnMomentFunction, basisCache, tol, out, success);
      }
    }
    
    bool errorMeetsTol = true;
    
    const double expectedL2 = expectedIntegral->l2norm(physicalMesh);
    const double error = errorFxn->l2norm(physicalMesh);
    
    const double actualL2 = solnMomentFunction->l2norm(physicalMesh);
    
    out << "L^2 norm of expected integral: " << expectedL2 << std::endl;
    out << "L^2 norm of actual integral:   " << actualL2 << std::endl;
    
    if (expectedL2 < tol)
    {
      // use absolute comparison if the expected value is very small
      errorMeetsTol = (error <= tol);
      TEST_COMPARE(error, <=, tol);
    }
    else
    {
      const double relativeError = error / expectedL2;
      errorMeetsTol = (relativeError <= tol);
      TEST_COMPARE(relativeError, <=, tol);
    }
    
    if (!errorMeetsTol)
    {
      const int numSurvivingExtrusionDims = (survivingExtrusionDim == -1) ? 0 : 1;
      auto refCellPoints = sampleBasisCache->getRefCellPoints();
      const int numPoints = refCellPoints.extent_int(0);
      FieldContainer<double> physicalValuesExpected(1,numPoints);
      FieldContainer<double> physicalValuesActual(1,numPoints);
      const auto & myCellIDs = physicalMesh->cellIDsInPartition();
      for (IndexType cellID : myCellIDs)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, cellID);
        expectedIntegral->values(physicalValuesExpected, basisCache);
        solnMomentFunction->values(physicalValuesActual, basisCache);
        
        const auto & physicalCubaturePointsFC = basisCache->getPhysicalCubaturePointsFC();
        for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
        {
          const double &actualValue   = physicalValuesActual(0,pointOrdinal);
          const double &expectedValue = physicalValuesExpected(0,pointOrdinal);
          
          bool differs = !errMeetsAbsAndRelTol(expectedValue, actualValue, tol, tol);
          
          if (differs)
          {
            const double relPointError = fabs(actualValue - expectedValue) / max(actualValue,expectedValue);
            out << "At point (";
            for (int d=0; d<numPhysicalDims+numSurvivingExtrusionDims; d++)
            {
              out << physicalCubaturePointsFC(0,pointOrdinal,d);
            }
            out << "), expected value (" << expectedValue << ") differs from actual (" << actualValue << "); relative error is " << relPointError << std::endl;
          }
        }
      }
    }
  }

  void testIntegrate(const int numPhysicalDims, const int numExtrusionDims, const bool useSpaceTime,
                     int survivingExtrusionDim, // use -1 for no surviving dims
                     const int fxnPolyOrder,
                     const bool useSerendipity,
                     FunctionPtr phaseSpaceFxn, FunctionPtr expectedIntegral,
                     const int &iMomentNumber, const int &jMomentNumber, const int &kMomentNumber,
                     bool refine,
                     const double tol,
                     Teuchos::FancyOStream &out, bool &success)
  {
    bool pCoarsen = false;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, useSpaceTime,
                  survivingExtrusionDim,
                  fxnPolyOrder, useSerendipity,
                  phaseSpaceFxn, expectedIntegral,
                  iMomentNumber, jMomentNumber, kMomentNumber,
                  refine, pCoarsen, tol, out, success);
  }

  void testIntegrate(const int numPhysicalDims, const int numExtrusionDims, const bool useSpaceTime,
                     int survivingExtrusionDim, // use -1 for no surviving dims
                     const int fxnPolyOrder,
                     const bool useSerendipity,
                     FunctionPtr phaseSpaceFxn, FunctionPtr expectedIntegral,
                     bool refine,
                     const double tol,
                     Teuchos::FancyOStream &out, bool &success)
  {
    testIntegrate(numPhysicalDims, numExtrusionDims, useSpaceTime,
                  survivingExtrusionDim,
                  fxnPolyOrder, useSerendipity,
                  phaseSpaceFxn, expectedIntegral,
                  0,0,0,
                  refine, tol, out, success);
  }

void testIntegrate(const int numPhysicalDims, const int numExtrusionDims, const bool useSpaceTime,
                   int survivingExtrusionDim, // use -1 for no surviving dims
                   const int fxnPolyOrder,
                   const bool useSerendipity,
                   FunctionPtr phaseSpaceFxn, FunctionPtr expectedIntegral,
                   bool refine, bool pCoarsen,
                   const double tol,
                   Teuchos::FancyOStream &out, bool &success)
{
  testIntegrate(numPhysicalDims, numExtrusionDims, useSpaceTime,
                survivingExtrusionDim,
                fxnPolyOrder, useSerendipity,
                phaseSpaceFxn, expectedIntegral,
                0,0,0,
                refine, pCoarsen, tol, out, success);
}


//! expectedMoments: has momentNumber+1 entries in each of the three dimensions.  When i + j + k > momentNumber, entries ignored (may be null)
void testMoments(const int numPhysicalDims, const int numExtrusionDims, const bool useSpaceTime,
                 int survivingExtrusionDim, // use -1 for no surviving dims
                 const int fxnPolyOrder,
                 FunctionPtr phaseSpaceFxn, std::vector<std::vector<std::vector<FunctionPtr> > > expectedMoments,
                 const int &momentNumber,
                 bool refine,
                 const double tol,
                 Teuchos::FancyOStream &out, bool &success,
                 double extrusionOrigin = EXTRUSION_ORIGIN, double extrusionExtent = EXTRUSION_EXTENT,
                 int extrusionMeshWidth = EXTRUSION_MESH_WIDTH)
{
  // TODO: something to test anisotropy in p
  
  const int delta_k = 0;
  
  std::vector<int> H1Order(1 + numExtrusionDims, fxnPolyOrder + 1);
  
  auto meshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims, useSpaceTime, extrusionOrigin, extrusionExtent, extrusionMeshWidth);
  
  VarFactoryPtr vf = VarFactory::varFactory();
  auto f = vf->fieldVar("f");
  auto v = vf->testVar("v", HGRAD);
  
  BFPtr bf = BF::bf(vf);
  
  Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
  const bool allowPruning = false;
  MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
  
  std::vector<int> physicalH1Order{fxnPolyOrder + 1};
  
  MeshTopologyViewPtr extrudedPhysicalMeshTopology;
  if (survivingExtrusionDim == -1)
  {
    // all are integrated
    extrudedPhysicalMeshTopology = meshTopo->physicalMeshTopology();
  }
  else
  {
    extrudedPhysicalMeshTopology = Teuchos::rcp( new SubExtrudedMeshTopology(meshTopo, survivingExtrusionDim) );
  }
  
  MeshPtr physicalMesh = MeshFactory::minRuleMesh(extrudedPhysicalMeshTopology, bf, physicalH1Order, delta_k, comm, allowPruning);
  
  if (survivingExtrusionDim == -1)
  {
    // while SubExtrudedMeshTopology knows about phase space mesh topology, plain physical mesh does not
    // for that reason, we need to register for updates
    PhysicalMeshUpdater::registerForUpdates(mesh, physicalMesh);
  }
  
  // confirm cell counts match expected before any refinement
  int startingPhysicalCellCount = 1;
  for (int d=0; d<numPhysicalDims; d++)
  {
    startingPhysicalCellCount *= PHYSICAL_MESH_WIDTH;
  }
  if (useSpaceTime)
  {
    startingPhysicalCellCount *= TEMPORAL_MESH_WIDTH;
  }
  int expectedActivePhaseSpaceCellCount = startingPhysicalCellCount;
  for (int d=0; d<numExtrusionDims; d++)
  {
    expectedActivePhaseSpaceCellCount *= extrusionMeshWidth;
  }
  int expectedActivePhysicalCellCount = startingPhysicalCellCount;
  if (survivingExtrusionDim != -1)
  {
    expectedActivePhysicalCellCount *= extrusionMeshWidth;
  }
  TEST_EQUALITY(expectedActivePhysicalCellCount, physicalMesh->numActiveElements());
  TEST_EQUALITY(expectedActivePhaseSpaceCellCount, mesh->numActiveElements());
  
  if (refine)
  {
    const IndexType cellToRefine = 0;
    
    int numTemporalDims = useSpaceTime ? 1 : 0;
    
    int newCellsPerRefinement = 1 << (numExtrusionDims + numPhysicalDims + numTemporalDims); // isotropic refinement
    
    set<GlobalIndexType> cellsToRefine {cellToRefine};
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(mesh->getElementType(cellToRefine)->cellTopoPtr);
    mesh->hRefine(cellsToRefine, refPattern);
    
    TEST_EQUALITY(newCellsPerRefinement, refPattern->numChildren());
    
    expectedActivePhaseSpaceCellCount += newCellsPerRefinement - 1; // -1: the parent cell is no longer active
    
    int newPhysicalCellsPerRefinement = 1 << (numPhysicalDims + numTemporalDims);
    const int newExtrusionCellsPerRefinement = 2;
    if (survivingExtrusionDim != -1)
    {
      expectedActivePhysicalCellCount = (startingPhysicalCellCount + newPhysicalCellsPerRefinement - 1) * (EXTRUSION_MESH_WIDTH + newExtrusionCellsPerRefinement - 1);
    }
    else
    {
      expectedActivePhysicalCellCount = (startingPhysicalCellCount + newPhysicalCellsPerRefinement - 1);
    }
    
    TEST_EQUALITY(expectedActivePhysicalCellCount, physicalMesh->numActiveElements());
    TEST_EQUALITY(expectedActivePhaseSpaceCellCount, mesh->numActiveElements());
  }
  
  // now that mesh is set up, project phaseSpaceFxn onto solution.
  SolutionPtr soln = Solution::solution(mesh);
  std::map<int, FunctionPtr> functionMap;
  functionMap[f->ID()] = phaseSpaceFxn;
  
  soln->projectOntoMesh(functionMap);
  
  SolutionMoments solnMoments(momentNumber, soln, f, physicalMesh, survivingExtrusionDim);
  solnMoments.update();
  
  const int numSurvivingDims  = (survivingExtrusionDim == -1) ? 0 : 1;
  const int numIntegratedDims = numExtrusionDims - numSurvivingDims;
  
  // the commented-out phaseViz export only works for total dim <= 3.  The configViz should work for any spatial dimension, though.
//  {
//    // DEBUGGING: export viz
//
//    FunctionPtr f_soln = Function::solution(f, soln);
//    FunctionPtr f_err  = phaseSpaceFxn - f_soln;
//    std::vector<FunctionPtr> fxnsToExport {phaseSpaceFxn, f_err};
//    std::vector<std::string> fxnNames {"f_exact", "f_err"};
//
//    auto phaseViz = Teuchos::rcp(new HDF5Exporter(soln->mesh(), "TestMoments", "/tmp"));
//    phaseViz->exportSolution(soln, fxnsToExport, fxnNames);
//
//    FunctionPtr zeroMoment;
//    if (numIntegratedDims == 1)
//    {
//      zeroMoment = solnMoments.solutionMomentFunction(0);
//    }
//    else if (numIntegratedDims == 2)
//    {
//      zeroMoment = solnMoments.solutionMomentFunction(0,0);
//    }
//    else if (numIntegratedDims == 3)
//    {
//      zeroMoment = solnMoments.solutionMomentFunction(0,0,0);
//    }
//
//    auto configViz = Teuchos::rcp(new HDF5Exporter(solnMoments.mesh(), "TestMomentsConfig", "/tmp"));
//    configViz->exportFunction({zeroMoment}, "M_000");
//  }
  
  for (int i=0; i<=momentNumber; i++)
  {
    const int j_max = (numIntegratedDims > 1) ? momentNumber - i : 0;
    for (int j=0; j<=j_max; j++)
    {
      const int k_max = (numIntegratedDims > 2) ? momentNumber - i - j : 0;
      for (int k=0; k<=k_max; k++)
      {
        out << "*** Moment (" << i << "," << j << "," << k << ") ***\n";
        FunctionPtr solnMomentFunction;
        if (numIntegratedDims == 1)
        {
          solnMomentFunction = solnMoments.solutionMomentFunction(i);
        }
        else if (numIntegratedDims == 2)
        {
          solnMomentFunction = solnMoments.solutionMomentFunction(i,j);
        }
        else if (numIntegratedDims == 3)
        {
          solnMomentFunction = solnMoments.solutionMomentFunction(i,j,k);
        }
        
        // because we don't allow pruning, cell 0 is known everywhere.  We use it to determine spatial quadrature points on the reference element.
        BasisCachePtr sampleBasisCache = BasisCache::basisCacheForCell(physicalMesh, 0);
        
        FunctionPtr expectedIntegral = expectedMoments[i][j][k];
        
        FunctionPtr errorFxn = expectedIntegral - solnMomentFunction;
        
        if (numExtrusionDims == 1)
        {
          out << "Testing whether solnMomentFunction Data path matches View path" << std::endl;
          const auto & myCells = physicalMesh->cellIDsInPartition();
          for (const auto & physicalCellID : myCells)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
            testFunctionDataPathMatchesViewPath1V(solnMomentFunction, basisCache, tol, out, success);
          }
        }
        else if (numExtrusionDims == 3)
        {
          out << "Testing whether solnMomentFunction Data path matches View path" << std::endl;
          const auto & myCells = physicalMesh->cellIDsInPartition();
          for (const auto & physicalCellID : myCells)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
            testFunctionDataPathMatchesViewPath3V(solnMomentFunction, basisCache, tol, out, success);
          }
        }
        
        bool errorMeetsTol = true;
        
        const double expectedL2 = expectedIntegral->l2norm(physicalMesh);
        const double error = errorFxn->l2norm(physicalMesh);
        
        const double actualL2 = solnMomentFunction->l2norm(physicalMesh);
        
        out << "L^2 norm of expected integral: " << expectedL2 << std::endl;
        out << "L^2 norm of actual integral:   " << actualL2 << std::endl;
        
        if (expectedL2 < tol)
        {
          // use absolute comparison if the expected value is very small
          errorMeetsTol = (error <= tol);
          TEST_COMPARE(error, <=, tol);
        }
        else
        {
          const double relativeError = error / expectedL2;
          TEST_COMPARE(relativeError, <=, tol);
        }
        
        if (!errorMeetsTol)
        {
          const int numSurvivingExtrusionDims = (survivingExtrusionDim == -1) ? 0 : 1;
          auto refCellPoints = sampleBasisCache->getRefCellPoints();
          const int numPoints = refCellPoints.extent_int(0);
          FieldContainer<double> physicalValuesExpected(1,numPoints);
          FieldContainer<double> physicalValuesActual(1,numPoints);
          const auto & myCellIDs = physicalMesh->cellIDsInPartition();
          for (IndexType cellID : myCellIDs)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, cellID);
            expectedIntegral->values(physicalValuesExpected, basisCache);
            solnMomentFunction->values(physicalValuesActual, basisCache);
            
            const auto & physicalCubaturePointsFC = basisCache->getPhysicalCubaturePointsFC();
            for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
            {
              const double &actualValue   = physicalValuesActual(0,pointOrdinal);
              const double &expectedValue = physicalValuesExpected(0,pointOrdinal);
              
              bool differs = !errMeetsAbsAndRelTol(expectedValue, actualValue, tol, tol);
              
              if (differs)
              {
                const double relPointError = fabs(actualValue - expectedValue) / max(actualValue,expectedValue);
                out << "At point (";
                for (int d=0; d<numPhysicalDims+numSurvivingExtrusionDims; d++)
                {
                  out << physicalCubaturePointsFC(0,pointOrdinal,d);
                }
                out << "), expected value (" << expectedValue << ") differs from actual (" << actualValue << "); relative error is " << relPointError << std::endl;
              }
            }
          }
        }
      }
    }
  }
}

void testConstantMoments(const int numPhysicalDims, const int numExtrusionDims,
                         bool refine,
                         const double tol,
                         Teuchos::FancyOStream &out, bool &success)
{
  const int survivingExtrusionDim = -1;
  const bool spaceTime       = false;
  const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
  
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr f = one;
  
  int maxMoment = 5;

  using std::vector;
  vector<FunctionPtr> nullFunctions(maxMoment + 1);
  vector<vector<vector<FunctionPtr>>> expectedMoments(maxMoment+1, vector<vector<FunctionPtr>>(maxMoment+1, nullFunctions));
  
  // nth-moment integral in vx is (vx)^(n+1) / (n+1), evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
  vector<FunctionPtr> integrals(maxMoment+1);
  for (int n=0; n<=maxMoment; n++)
  {
    const double v_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double v_min = EXTRUSION_ORIGIN;
    const double v_integral = (pow(v_max, n+1) - pow(v_min, n+1)) / (n+1);
    FunctionPtr integralFxn = Function::constant(v_integral);
    integrals[n] = integralFxn;
  }
  
  for (int i=0; i<=maxMoment; i++)
  {
    if (numExtrusionDims == 1)
    {
      FunctionPtr integral = integrals[i];
      expectedMoments[i][0][0] = integral;
    }
    else
    {
      for (int j=0; j<=maxMoment-i; j++)
      {
        if (numExtrusionDims == 2)
        {
          FunctionPtr integral = integrals[i] * integrals[j];
          expectedMoments[i][j][0] = integral;
        }
        else
        {
          for (int k=0; k<=maxMoment-i-j; k++)
          {
            FunctionPtr integral = integrals[i] * integrals[j] * integrals[k];
            expectedMoments[i][j][k] = integral;
          }
        }
      }
    }
  }
  
  testMoments(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder,
              f, expectedMoments, maxMoment, refine, tol, out, success);
}

void testMaxwellianMoments(const int numPhysicalDims, const int numExtrusionDims,
                           const double n,
                           const double ux, const double uy, const double uz,
                           const double sigmaSquared,
                           bool refine,
                           const double extrusionOrigin, const double extrusionExtent,
                           const int extrusionMeshWidth,
                           const double tol,
                           Teuchos::FancyOStream &out, bool &success)
{
  const int survivingExtrusionDim = -1;
  const bool spaceTime       = false;
  const int polyOrder = 6; // for L^2 fields, one less than H^1 order; here we use 6 to get a decent approximation of the Maxwellian on only two elements…
  
  std::vector<double> u;
  if      (numExtrusionDims == 1) u = std::vector<double>{ux};
  else if (numExtrusionDims == 2) u = std::vector<double>{ux,uy};
  else                            u = std::vector<double>{ux,uy,uz};
  FunctionPtr f = MaxwellianFunction::maxwellian(numPhysicalDims, numExtrusionDims, u, sigmaSquared, n);
  
  int maxMoment = 5;

  using std::vector;
  vector<FunctionPtr> nullFunctions(maxMoment + 1);
  vector<vector<vector<FunctionPtr>>> expectedMoments(maxMoment+1, vector<vector<FunctionPtr>>(maxMoment+1, nullFunctions));
  
  std::vector<vector<double> > integrals_1D(numExtrusionDims, vector<double>(maxMoment+1));
  for (int i=0; i<=maxMoment; i++)
  {
    for (int d=0; d<numExtrusionDims; d++)
    {
      const double moment = MaxwellianFunction::momentNormalized1D(u[d], sigmaSquared, i);
      integrals_1D[d][i] = moment;
    }
  }
  
  for (int i=0; i<=maxMoment; i++)
  {
    if (numExtrusionDims == 1)
    {
      const double moment = n * integrals_1D[0][i];
      expectedMoments[i][0][0] = Function::constant(moment);
    }
    else
    {
      for (int j=0; j<=maxMoment-i; j++)
      {
        if (numExtrusionDims == 2)
        {
          const double moment = n * integrals_1D[0][i] * integrals_1D[1][j];
          expectedMoments[i][j][0] = Function::constant(moment);
        }
        else
        {
          for (int k=0; k<=maxMoment-i-j; k++)
          {
            const double moment = n * integrals_1D[0][i] * integrals_1D[1][j] * integrals_1D[2][k];
            expectedMoments[i][j][k] = Function::constant(moment);
          }
        }
      }
    }
  }
  
  testMoments(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder,
              f, expectedMoments, maxMoment, refine, tol, out, success, extrusionOrigin, extrusionExtent, extrusionMeshWidth);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunction, DataPathMatchesViewPath_OnSubExtrudedMesh )
{
  const int numPhysicalDims  = 1;
  const int numExtrusionDims = 2;
  const int survivingExtrusionDim = 0; // x,vx survive
  const bool useSpaceTime    = true;
  const bool useSerendipity  = false;
  const int polyOrder = 2;
  const int delta_k   = 0;

  std::vector<int> H1Order(1 + numExtrusionDims + 1, polyOrder + 1); // +1 for time

  auto meshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims, useSpaceTime);

  VarFactoryPtr vf = VarFactory::varFactory();
  auto f = vf->fieldVar("f");
  auto v = vf->testVar("v", HGRAD);

  if (useSerendipity)
  {
    f->setBasisFamily(BasisFamily::serendipityBasisFamily());
  }

  BFPtr bf = BF::bf(vf);

  Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
  const bool allowPruning = false;
  MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);

  std::vector<int> physicalH1Order{polyOrder + 1};

  MeshTopologyViewPtr extrudedPhysicalMeshTopology =
    Teuchos::rcp(new SubExtrudedMeshTopology(meshTopo, survivingExtrusionDim));

  MeshPtr physicalMesh =
    MeshFactory::minRuleMesh(extrudedPhysicalMeshTopology, bf, physicalH1Order, delta_k, comm, allowPruning);

  // Project a phase-space function that yields a nontrivial moment function.
  SolutionPtr soln = Solution::solution(mesh);
  FunctionPtr phaseSpaceFxn = Function::xn(1) * MonomialFunction::xi_n(3,2); // x * vy^2

  std::map<int, FunctionPtr> functionMap;
  functionMap[f->ID()] = phaseSpaceFxn;
  soln->projectOntoMesh(functionMap);

  const int maxMomentNumber = 0;
  SolutionMoments solnMoments(maxMomentNumber, soln, f, physicalMesh, survivingExtrusionDim);
  solnMoments.update();

  FunctionPtr momentFxn = solnMoments.solutionMomentFunction(0); // integral over vy

  const IndexType cellID = 0;
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, cellID);

  testScalarFunctionDataPathMatchesViewPath(momentFxn, basisCache, 1e-14, out, success);
}

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const bool refine          = false;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const bool refine          = true;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D2V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const bool refine          = false;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D2V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const bool refine          = true;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D3V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const bool refine          = false;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_1D3V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const bool refine          = true;
    const double tol = 1e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_2D1V_Refine )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 1;
    const bool refine          = true;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_2D2V )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 2;
    const bool refine          = false;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_2D2V_Refine )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 2;
    const bool refine          = true;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_2D3V )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 3;
    const bool refine          = false;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_2D3V_Refine )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 3;
    const bool refine          = true;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D1V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 1;
    const bool refine          = false;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D1V_Refine )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 1;
    const bool refine          = true;
    const double tol = 1e-15;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D2V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 2;
    const bool refine          = false;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D2V_Refine )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 2;
    const bool refine          = true;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D3V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 3;
    const bool refine          = false;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, AllMoments_Constant_3D3V_Refine )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 3;
    const bool refine          = true;
    const double tol = 5e-14;
    
    testConstantMoments(numPhysicalDims, numExtrusionDims, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, MaxwellianMoments_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const bool refine          = false;
    const double tol = 1e-13; // first 5 moments can be resolved to this tolerance with 64-wide velocity-space mesh
    
    const double n  = 1.0;
    const double ux = 0.0;
    const double uy = 0.0;
    const double uz = 0.0;
    const double sigmaSquared = 0.10; // need to keep this not too wide (< O(1)) to keep the tails from hitting the edge of the domain (at ±1)
    
    const double extrusionOrigin = -10.0;
    const double extrusionExtent =  20.0;
    const int extrusionMeshWidth =    64;
    
    testMaxwellianMoments(numPhysicalDims, numExtrusionDims, n, ux, uy, uz, sigmaSquared, refine,
                          extrusionOrigin, extrusionExtent, extrusionMeshWidth,
                          tol, out, success);
  }

// 2-24-22: the test commented out below passes, but is quite expensive.  We might want to revisit once we have Serendipity working.
//  TEUCHOS_UNIT_TEST( SolutionMomentFunction, MaxwellianMoments_1D3V_SuperSlow )
//  {
//    const int numPhysicalDims  = 1;
//    const int numExtrusionDims = 3;
//    const bool refine          = false;
//    const double tol = 0.25; // 128 4D 6th-order elements only resolves this to tol 0.21 or so, but doing even this much as part of a standard unit test is prohibitive (takes about 10 seconds on 24 MPI ranks)
//
//
//    const double n  = 1.0;
//    const double ux = 0.0;
//    const double uy = 0.0;
//    const double uz = 0.0;
//    const double sigmaSquared = 0.10; // need to keep this not too wide (< O(1)) to keep the tails from hitting the edge of the domain (at ±1)
//
//    const double extrusionOrigin = -10.0;
//    const double extrusionExtent =  20.0;
//    const int extrusionMeshWidth =     4;
//
//    testMaxwellianMoments(numPhysicalDims, numExtrusionDims, n, ux, uy, uz, sigmaSquared, refine,
//                          extrusionOrigin, extrusionExtent, extrusionMeshWidth,
//                          tol, out, success);
//  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1V )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1V_Serendipity )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1V_Refine )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  /*
   The "1T" here is a temporal dimension.  For space-time Vlasov, we have a space-time physical mesh,
   which is then further extruded in velocity dimensions.  So the lower-dimensional mesh is also an
   extruded mesh, so that e.g. "1D1T1V" is a slightly different use case than e.g. 2D1V.
   */
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1T1V )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  /*
   The "1T" here is a temporal dimension.  For space-time Vlasov, we have a space-time physical mesh,
   which is then further extruded in velocity dimensions.  So the lower-dimensional mesh is also an
   extruded mesh, so that e.g. "1D1T1V" is a slightly different use case than e.g. 2D1V.
   */
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1T1V_Serendipity )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1T1V_Refine )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D1T1V_Refine_Serendipity )
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = true;
    const bool useSerendipity  = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D2V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D2V_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D2V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D2V_Refine_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D3V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_1D3V_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_2D1V )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_2D2V )
  {
    const int numPhysicalDims  = 2;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_2D3V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_2D3V_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_3D1V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_3D2V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 0; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_Constant_3D3V )
  {
    const int numPhysicalDims  = 3;
    const int numExtrusionDims = 3;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 5e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1V_pCoarsen )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const bool pCoarsen        = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, pCoarsen, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1V_Refine_pCoarsen )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const bool pCoarsen        = true;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, pCoarsen, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1T1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1T1V_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_x_1D1T1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::xn(1);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D1V_pCoarsen )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const bool pCoarsen        = true;
    const int polyOrder = 3; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, pCoarsen, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D1V_Refine_pCoarsen )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const bool pCoarsen        = true;
    const int polyOrder = 3; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, pCoarsen, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xz2_1D1T1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z2 = Function::zn(2);
    FunctionPtr f = x * z2;
    // integral in z is z^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double z_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double z_min = EXTRUSION_ORIGIN;
    const double z_integral = (pow(z_max, 3) - pow(z_min, 3)) / 3.0;
    FunctionPtr f_integral = z_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xz2_1D1T1V_Serendipity )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = false;
    const bool useSerendipity  = true;
    const int polyOrder = 3; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z2 = Function::zn(2);
    FunctionPtr f = x * z2;
    // integral in z is z^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double z_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double z_min = EXTRUSION_ORIGIN;
    const double z_integral = (pow(z_max, 3) - pow(z_min, 3)) / 3.0;
    FunctionPtr f_integral = z_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xz2_1D1T1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = true;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z2 = Function::zn(2);
    FunctionPtr f = x * z2;
    // integral in z is z^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double z_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double z_min = EXTRUSION_ORIGIN;
    const double z_integral = (pow(z_max, 3) - pow(z_min, 3)) / 3.0;
    FunctionPtr f_integral = z_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D2V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    const double z_integral = EXTRUSION_EXTENT; // does not vary in z
    FunctionPtr f_integral = y_integral * x * z_integral;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_xy2_1D2V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    const double z_integral = EXTRUSION_EXTENT; // does not vary in z
    FunctionPtr f_integral = y_integral * x * z_integral;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_y2_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, Integrate_y2_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateFirstMoment_Constant_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr f = one;
    // first-moment integral in y is y^2 / 2, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 2) - pow(y_min, 2)) / 2.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 1;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateFirstMoment_Constant_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr f = one;
    // first-moment integral in y is y^2 / 2, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 2) - pow(y_min, 2)) / 2.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 1;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateFirstMoment_y_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y = Function::yn(1);
    FunctionPtr f = y;
    // first-moment integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 1;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateFirstMoment_y_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y = Function::yn(1);
    FunctionPtr f = y;
    // first-moment integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 1;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateSecondMoment_Constant_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr f = one;
    // second-moment integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 2;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateSecondMoment_Constant_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order; here we use 1 because 0 is a bit too easy to pass, since there is then only one dof per cell
    const double tol = 1e-15;
    
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr f = one;
    // second-moment integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 2;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateSecondMoment_y_1D1V )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y = Function::yn(1);
    FunctionPtr f = y;
    // second-moment integral in y is y^4 / 4, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 4) - pow(y_min, 4)) / 4.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 2;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, IntegrateSecondMoment_y_1D1V_Refine )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 1;
    const int survivingExtrusionDim = -1;
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr y = Function::yn(1);
    FunctionPtr f = y;
    // first-moment integral in y is y^4 / 4, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 4) - pow(y_min, 4)) / 4.0;
    FunctionPtr f_integral = Function::constant(y_integral);
    
    const int iMoment = 2;
    const int jMoment = 0;
    const int kMoment = 0;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity,
                  f, f_integral,
                  iMoment, jMoment, kMoment,
                  refine,
                  tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, PartiallyIntegrate_Constant_1D2V_to_xz ) // to xz: x,z dimensions survive integration
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = 1; // z dimension (final extrusion dim) does not get integrated
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims-1) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, PartiallyIntegrate_Constant_1D2V_to_xz_Refine ) // to xz: x,z dimensions survive integration
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = 1; // z dimension (final extrusion dim) does not get integrated
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 1; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr f = Function::constant(1.0);
    // integral in extrusion dimensions results in multiplication by extent in those dimensions:
    FunctionPtr f_integral = pow(EXTRUSION_EXTENT,numExtrusionDims-1) * f;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, PartiallyIntegrate_xy2_1D2V_to_xz ) // to xz: x,z dimensions survive integration
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = 1; // z dimension (final extrusion dim) does not get integrated
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, PartiallyIntegrate_xy2_1D2V_to_xz_Refine ) // to xz: x,z dimensions survive integration
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = 1; // z dimension (final extrusion dim) does not get integrated
    const bool spaceTime       = false;
    const bool refine          = true;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr f = x * y2;
    // integral in y is y^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double y_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double y_min = EXTRUSION_ORIGIN;
    const double y_integral = (pow(y_max, 3) - pow(y_min, 3)) / 3.0;
    FunctionPtr f_integral = y_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, PartiallyIntegrate_xz2_1D3V_to_xy ) // to xy: (x,v_x) dimensions survive integration
  {
    MPIWrapper::CommWorld()->Barrier(); // for breakpoint
    
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    const int survivingExtrusionDim = 0; // y dimension (first extrusion dim) does not get integrated
    const bool spaceTime       = false;
    const bool refine          = false;
    const bool useSerendipity  = false;
    const int polyOrder = 2; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z2 = Function::zn(2);
    FunctionPtr f = x * z2;
    // integral in z is z^3 / 3, evaluated at EXTRUSION_ORIGIN + EXTRUSION_EXTENT and EXTRUSION_ORIGIN
    const double z_max = EXTRUSION_ORIGIN + EXTRUSION_EXTENT;
    const double z_min = EXTRUSION_ORIGIN;
    const double z_integral = (pow(z_max, 3) - pow(z_min, 3)) / 3.0;
    FunctionPtr f_integral = z_integral * x;
    
    testIntegrate(numPhysicalDims, numExtrusionDims, spaceTime, survivingExtrusionDim, polyOrder, useSerendipity, f, f_integral, refine, tol, out, success);
  }

  // TODO: test higher-order moments
  // TODO: test more refinements
  // TODO: test more partial integrals

  TEUCHOS_UNIT_TEST( SolutionMomentFunction, TensorHelperMethods )
  {
    // TODO: revise this to avoid allocating the whole tensor-product containers (use Intrepid2 tensor containers instead)
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 3;
    const bool spaceTime       = false;
    const int polyOrder = 3; // for L^2 fields, one less than H^1 order
    const double tol = 1e-15;
    const int delta_k = 0;
    
    std::vector<int> H1Order(1 + numExtrusionDims, polyOrder + 1);
    
    auto meshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims, spaceTime);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", HGRAD);
    
    BFPtr bf = BF::bf(vf);
    
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const bool allowPruning = false;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, bf, H1Order, delta_k, comm, allowPruning);
    
    const IndexType sampleCellID = 0;
    BasisPtr phaseSpaceBasis = mesh->getElementType(sampleCellID)->trialOrderPtr->getBasis(f->ID());
    
    std::vector<int> physicalH1Order{polyOrder + 1};
    MeshPtr physicalMesh = MeshFactory::minRuleMesh(meshTopo->physicalMeshTopology(), bf, physicalH1Order, delta_k, comm, allowPruning);
    
    BasisPtr physicalBasis = physicalMesh->getElementType(sampleCellID)->trialOrderPtr->getBasis(f->ID());

    BasisCachePtr phaseBasisCache    = BasisCache::basisCacheForCell(mesh, sampleCellID);
    BasisCachePtr physicalBasisCache = BasisCache::basisCacheForCell(physicalMesh, sampleCellID);
    
    const auto & physicalRefPoints     = physicalBasisCache->getRefCellPointsFC();
    const int physicalBasisCardinality = physicalBasis->getCardinality();
    const int numPhysicalPoints        = physicalRefPoints.dimension(0);
    FieldContainer<double> physicalBasisValues(physicalBasisCardinality, numPhysicalPoints);
    physicalBasis->getValues(physicalBasisValues, physicalRefPoints, OP_VALUE);
    
    const auto & phaseRefPoints     = phaseBasisCache->getRefCellPointsFC();
    const int phaseBasisCardinality = phaseSpaceBasis->getCardinality();
    const int numPhasePoints        = phaseRefPoints.dimension(0);
    FieldContainer<double> phaseBasisValues(phaseBasisCardinality, numPhasePoints);
    phaseSpaceBasis->getValues(phaseBasisValues, phaseRefPoints, OP_VALUE);
    
    auto basisComponents = phaseSpaceBasis->intrepid2TensorBasisComponents();
    std::vector<BasisPtr> extrusionBases(numExtrusionDims);
    int compOffset = basisComponents.size() - numExtrusionDims;
    for (int d=0; d<numExtrusionDims; d++)
    {
      extrusionBases[d] = basisComponents[d+compOffset];
    }
    std::vector<int> numExtrusionPoints(numExtrusionDims);
    std::vector<FieldContainer<double>> extrusionPoints(numExtrusionDims);
    std::vector<FieldContainer<double>> extrusionValues(numExtrusionDims);
    int totalExtrusionCardinality = 1;
    for (int d=0; d<numExtrusionDims; d++)
    {
      BasisPtr extrusionBasis = extrusionBases[d];
      totalExtrusionCardinality *= extrusionBasis->getCardinality();
      auto extrusionPointsView = phaseBasisCache->getRefCellPoints().getTensorComponent(1+d);
      numExtrusionPoints[d] = extrusionPointsView.extent_int(0);
      extrusionPoints[d] = FieldContainerConversion<double>::fromDynRankView(extrusionPointsView);
      extrusionValues[d] = FieldContainer<double>(extrusionBasis->getCardinality(),numExtrusionPoints[d]);
      extrusionBasis->getValues(extrusionValues[d], extrusionPoints[d], OP_VALUE);
    }
    
    int totalExtrusionPointCount = numPhasePoints / numPhysicalPoints;
    
    using namespace ::Camellia::TensorIndices;
    // confirm that component points/point ordinals line up
    for (int spatialPointOrdinal=0; spatialPointOrdinal<numPhysicalPoints; spatialPointOrdinal++)
    {
      for (int extrusionPointOrdinal=0; extrusionPointOrdinal<totalExtrusionPointCount; extrusionPointOrdinal++)
      {
        std::vector<int> extrusionPointOrdinals(numExtrusionDims);
        int remainingExtrusionPointOrdinal = extrusionPointOrdinal;
        for (int d=0; d<numExtrusionDims; d++)
        {
          extractNextComponentOrdinal(extrusionPointOrdinals[d], remainingExtrusionPointOrdinal, numExtrusionPoints[d]);
        }
        
        int phasePointOrdinal;
        setTensorOrdinal(phasePointOrdinal, spatialPointOrdinal, numPhysicalPoints, extrusionPointOrdinal);
        
        for (int d=0; d<numPhysicalDims+numExtrusionDims; d++)
        {
          double expectedCoord_d;
          if (d<numPhysicalDims)
          {
            expectedCoord_d = physicalRefPoints(spatialPointOrdinal,d);
          }
          else
          {
            const int extrusionOrdinal = d - numPhysicalDims;
            expectedCoord_d = extrusionPoints[extrusionOrdinal](extrusionPointOrdinals[extrusionOrdinal],0);
          }
          TEST_EQUALITY(expectedCoord_d, phaseRefPoints(phasePointOrdinal,d));
        }
      }
    }
    
    for (int spatialBasisOrdinal=0; spatialBasisOrdinal<physicalBasisCardinality; spatialBasisOrdinal++)
    {
      for (int extrusionFieldOrdinal=0; extrusionFieldOrdinal<totalExtrusionCardinality; extrusionFieldOrdinal++)
      {
        int phaseFieldOrdinal;
        setTensorOrdinal(phaseFieldOrdinal, spatialBasisOrdinal, physicalBasisCardinality, extrusionFieldOrdinal);
        std::vector<int> extrusionComponentOrdinals(numExtrusionDims);
        int remainingExtrusionOrdinal = extrusionFieldOrdinal;
        for (int d=0; d<numExtrusionDims; d++)
        {
          extractNextComponentOrdinal(extrusionComponentOrdinals[d], remainingExtrusionOrdinal, extrusionBases[d]->getCardinality());
        }
        for (int spatialPointOrdinal=0; spatialPointOrdinal<numPhysicalPoints; spatialPointOrdinal++)
        {
          for (int extrusionPointOrdinal=0; extrusionPointOrdinal<totalExtrusionPointCount; extrusionPointOrdinal++)
          {
            std::vector<int> extrusionPointOrdinals(numExtrusionDims);
            int remainingExtrusionPointOrdinal = extrusionPointOrdinal;
            for (int d=0; d<numExtrusionDims; d++)
            {
              extractNextComponentOrdinal(extrusionPointOrdinals[d], remainingExtrusionPointOrdinal, numExtrusionPoints[d]);
            }
            
            int phasePointOrdinal;
            setTensorOrdinal(phasePointOrdinal, spatialPointOrdinal, numPhysicalPoints, extrusionPointOrdinal);
            
            double expectedValue = physicalBasisValues(spatialBasisOrdinal,spatialPointOrdinal);
            for (int d=0; d<numExtrusionDims; d++)
            {
              double extrusionValue = extrusionValues[d](extrusionComponentOrdinals[d],extrusionPointOrdinals[d]);
              expectedValue *= extrusionValue;
            }
            double actualValue = phaseBasisValues(phaseFieldOrdinal,phasePointOrdinal);
            TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
          }
        }
      }
    }
  }

  // TODO: move this to test group SubExtrudedMeshTopology, in SubExtrudedMeshTopologyTests.cpp (which we should create)
  TEUCHOS_UNIT_TEST( SolutionMomentFunction, SubExtrudedMeshTopology_CellHalo )
  {
    const int numPhysicalDims  = 1;
    const int numExtrusionDims = 2;
    
    const int physicalMeshWidth  = 2; // # of elements in each dimension
    const int extrusionMeshWidth = 2; // # of elements in each dimension
    
    using std::vector;
    vector<double> physicalDims(numPhysicalDims,PHYSICAL_EXTENT);
    vector<int> physicalElementCounts(numPhysicalDims,physicalMeshWidth);
    vector<double> physicalOrigin(numPhysicalDims,PHYSICAL_ORIGIN);
    auto physicalMeshTopo = MeshFactory::rectilinearMeshTopology(physicalDims, physicalElementCounts, physicalOrigin);
    
    vector<double> extrusionStartCoords(numExtrusionDims, EXTRUSION_ORIGIN);
    vector<double> extrusionExtents(numExtrusionDims, EXTRUSION_EXTENT);
    vector<int> extrusionElementCounts(numExtrusionDims,extrusionMeshWidth);
        
    auto extrudedMeshTopology = Teuchos::rcp(new ExtrudedMeshTopology(physicalMeshTopo,extrusionStartCoords,extrusionExtents,extrusionElementCounts));
    auto subExtrudedMeshTopology = Teuchos::rcp(new SubExtrudedMeshTopology(extrudedMeshTopology, 0));
    
    unsigned dimForNeighborRelation = (numPhysicalDims + 1) - 1;
    
    const GlobalIndexType sampleCellID = 0;
    CellPtr sampleCell = subExtrudedMeshTopology->getCell(sampleCellID);
    
    std::set<GlobalIndexType> expectedHalo {sampleCellID};
    auto sampleCellNeighbors = sampleCell->getNeighbors(subExtrudedMeshTopology);
    // expect there to one neighbor per dimension in MeshTopo (since this is a boundary cell in each dimension of interest)
    TEST_EQUALITY(numPhysicalDims + 1, sampleCellNeighbors.size());
    for (auto neighbor : sampleCellNeighbors)
    {
      expectedHalo.insert(neighbor->cellIndex());
    }
    
    std::set<GlobalIndexType> actualHalo;
    std::set<GlobalIndexType> cellIndices {0};
    subExtrudedMeshTopology->cellHalo(actualHalo, cellIndices, dimForNeighborRelation);
    
    TEST_EQUALITY(expectedHalo.size(), actualHalo.size());
    
    if (expectedHalo.size() == actualHalo.size())
    {
      auto expectedIt = expectedHalo.begin();
      auto actualIt   = actualHalo.begin();
      
      while (expectedIt != expectedHalo.end())
      {
        TEST_EQUALITY(*expectedIt, *actualIt);
        expectedIt++;
        actualIt++;
      }
    }
  }
} // end anonymous namespace
