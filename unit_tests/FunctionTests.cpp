//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include <CamelliaCellTools.h>
#include "CellTopology.h"
#include "Function.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RieszRep.h"
#include "Solution.h"

using namespace Camellia;

namespace
{

void testSpaceTimeNormalTimeComponent(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 1;
  bool createSideCache = true;
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FunctionPtr spaceTimeNormalComponent = Function::normalSpaceTime()->t();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    Camellia::FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPoints().dimension(0));
    spaceTimeNormalComponent->values(spaceTimeNormals,spaceTimeSideCache);

    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal);
        TEST_COMPARE(abs(spaceTimeTemporalNormalComponent), <, 1e-15);
      }
    }
    else
    {
      // otherwise, we expect 0 in every component, except the last, where we expect ±1
      int temporalSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      double expectedValue = (temporalSideOrdinal == 0) ? -1.0 : 1.0;
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal);
        TEST_FLOATING_EQUALITY(spaceTimeTemporalNormalComponent, expectedValue, 1e-15);
      }
    }
  }
}

Camellia::FieldContainer<double> getScaledTranslatedRefNodes(CellTopoPtr topo, double nodeScaling, double nodeTranslation)
{
  Camellia::FieldContainer<double> nodes(topo->getNodeCount(),topo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(nodes, topo);
  for (int nodeOrdinal=0; nodeOrdinal<topo->getNodeCount(); nodeOrdinal++)
  {
    for (int d=0; d<topo->getDimension(); d++)
    {
      nodes(nodeOrdinal,d) *= nodeScaling;
      nodes(nodeOrdinal,d) += nodeTranslation;
    }
  }
  return nodes;
}

void setTemporalNodes(CellTopoPtr spaceTimeTopo, Camellia::FieldContainer<double> &spaceTimeNodes, double t0, double t1)
{
  int d_time = spaceTimeTopo->getDimension() - 1;
  CellTopoPtr spaceTopo = spaceTimeTopo->getTensorialComponent();
  vector<unsigned> tensorComponentNodes = {0,0};
  for (unsigned spaceNode=0; spaceNode<spaceTopo->getNodeCount(); spaceNode++)
  {
    unsigned timeZeroNode = spaceTimeTopo->getNodeFromTensorialComponentNodes({spaceNode, 0});
    unsigned timeOneNode = spaceTimeTopo->getNodeFromTensorialComponentNodes({spaceNode, 1});
    spaceTimeNodes(timeZeroNode,d_time) = t0;
    spaceTimeNodes(timeOneNode,d_time)  = t1;
  }
}

void testSpaceTimeIntegrateByPartsInTime(CellTopoPtr spaceTopo, FunctionPtr f, Teuchos::FancyOStream &out, bool &success)
{
  /* Use the fact that
     (df/dt, 1)_K = < f, n_t >_dK - (f, 0) = < f, n_t >_dK
   */
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 2;
  bool createSideCache = true;
  double spaceNodeScaling = 0.5;
  double spaceNodeTranslation = 0.5; // scale, then translate
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  Camellia::FieldContainer<double> physicalNodesSpaceTime = getScaledTranslatedRefNodes(spaceTimeTopo, spaceNodeScaling, spaceNodeTranslation);
  double t0 = 0.0, t1 = 1.0;
  setTemporalNodes(spaceTimeTopo,physicalNodesSpaceTime,t0,t1);
  physicalNodesSpaceTime.resize(1,physicalNodesSpaceTime.dimension(0),physicalNodesSpaceTime.dimension(1));
  spaceTimeBasisCache->setPhysicalCellNodes(physicalNodesSpaceTime, vector<GlobalIndexType>(), createSideCache);

  FunctionPtr n_spacetime = Function::normalSpaceTime();

  double lhs_integral = f->dt()->integrate(spaceTimeBasisCache);
  double rhs_integral = (f * n_spacetime->t())->integrate(spaceTimeBasisCache);

  double diff = abs(lhs_integral-rhs_integral);
  double tol = 1e-14;

  TEST_COMPARE(diff, <, tol);
  if (diff >= tol)
  {
    out << "lhs_integral: " << lhs_integral << endl;
    out << "rhs_integral: " << rhs_integral << endl;
  }
}

void testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopoPtr spaceTopo, FunctionPtr f_spatial, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 2;
  bool createSideCache = true;
  double spaceNodeScaling = .5;
  double spaceNodeTranslation = .5; // scale, then translate
  BasisCachePtr spaceBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, cubatureDegree, createSideCache);
  Camellia::FieldContainer<double> physicalNodesSpace = getScaledTranslatedRefNodes(spaceTopo, spaceNodeScaling, spaceNodeTranslation);
  physicalNodesSpace.resize(1,physicalNodesSpace.dimension(0),physicalNodesSpace.dimension(1));
  spaceBasisCache->setPhysicalCellNodes(physicalNodesSpace, vector<GlobalIndexType>(), createSideCache);
  double spatialIntegral = f_spatial->integrate(spaceBasisCache);
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  Camellia::FieldContainer<double> physicalNodesSpaceTime = getScaledTranslatedRefNodes(spaceTimeTopo, spaceNodeScaling, spaceNodeTranslation);
  double t0 = 0.0, t1 = .5;
  double temporalExtent = t1 - t0;
  setTemporalNodes(spaceTimeTopo,physicalNodesSpaceTime,t0,t1);
  physicalNodesSpaceTime.resize(1,physicalNodesSpaceTime.dimension(0),physicalNodesSpaceTime.dimension(1));
  spaceTimeBasisCache->setPhysicalCellNodes(physicalNodesSpaceTime, vector<GlobalIndexType>(), createSideCache);
  double temporalIntegralActual = f_spatial->integrate(spaceTimeBasisCache);
  double temporalIntegralExpected = spatialIntegral * temporalExtent;
  double diff = abs(temporalIntegralExpected - temporalIntegralActual);

  double tol = 1e-14;
  TEST_COMPARE(diff, <, tol);

  if (diff > tol)
  {
    out << "temporalIntegralActual: " << temporalIntegralActual << endl;
    out << "temporalIntegralExpected: " << temporalIntegralExpected << endl;
  }
}

void testSpaceTimeNormal(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 1;
  bool createSideCache = true;
  BasisCachePtr spaceBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, cubatureDegree, createSideCache);
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FunctionPtr spaceTimeNormal = Function::normalSpaceTime();
  FunctionPtr spaceNormal = Function::normal();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    Camellia::FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPoints().dimension(0),spaceTimeTopo->getDimension());
    spaceTimeNormal->values(spaceTimeNormals,spaceTimeSideCache);

    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      // expect spaceTimeNormals to match spatial normals in the first d dimensions, and to be 0 in the final dimension
      int spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
      BasisCachePtr spaceSideCache = spaceBasisCache->getSideBasisCache(spatialSideOrdinal);

      Camellia::FieldContainer<double> spaceNormals(1,spaceSideCache->getRefCellPoints().dimension(0),spaceTopo->getDimension());
      spaceNormal->values(spaceNormals, spaceSideCache);

      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        // assume all normals are the same on the (spatial) side, and that there exists at least one point:
        for (int d=0; d<spaceTopo->getDimension(); d++)
        {
          double spaceNormalComponent = spaceNormals(0,0,d);
          double spaceTimeNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,d);
          TEST_FLOATING_EQUALITY(spaceNormalComponent, spaceTimeNormalComponent, 1e-15);
        }
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,spaceTopo->getDimension());
        TEST_COMPARE(abs(spaceTimeTemporalNormalComponent), <, 1e-15);
      }
    }
    else
    {
      // otherwise, we expect 0 in every component, except the last, where we expect ±1
      int temporalSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      double expectedValue = (temporalSideOrdinal == 0) ? -1.0 : 1.0;
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        // assume all normals are the same on the (spatial) side, and that there exists at least one point:
        for (int d=0; d<spaceTopo->getDimension(); d++)
        {
          double spaceTimeNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,d);
          TEST_COMPARE(abs(spaceTimeNormalComponent), <, 1e-15);
        }
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,spaceTopo->getDimension());

        TEST_FLOATING_EQUALITY(spaceTimeTemporalNormalComponent, expectedValue, 1e-15);
      }
    }
  }
}

  void testHFunction(CellTopoPtr cellTopo, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     For a simple, generic test of hFunction, we take the h value measured for the reference
     cell, and then scale each vertex in the reference cell by a factor.  We then expect the
     h value for the new cell to scale by that same factor.
     */
    
    int nodeCount = cellTopo->getNodeCount();
    int dim = cellTopo->getDimension();
    Camellia::FieldContainer<double> refCellNodes(nodeCount, dim);
    Camellia::FieldContainer<double> scaledRefCellNodes = refCellNodes;
    CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
    
    double scalingFactor = 2.0;
    for (int i=0; i<refCellNodes.size(); i++)
    {
      scaledRefCellNodes[i] = scalingFactor * refCellNodes[i];
    }
    // add "cell" dimension
    int cellCount = 1;
    refCellNodes.resize(cellCount, nodeCount, dim);
    scaledRefCellNodes.resize(cellCount, nodeCount, dim);
    int cubDegree = 1;
    
    BasisCachePtr refBasisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubDegree, refCellNodes);
    BasisCachePtr scaledBasisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubDegree, scaledRefCellNodes);
    
    FunctionPtr h = Function::h();
    int numPoints = refBasisCache->getPhysicalCubaturePoints().dimension(1); // C, P, D
    Camellia::FieldContainer<double> refValues(cellCount,numPoints);
    Camellia::FieldContainer<double> scaledRefValues(cellCount,numPoints);
    
    h->values(refValues, refBasisCache);
    h->values(scaledRefValues, scaledBasisCache);
    
    double tol = 1e-15;
    double maxDiff = 0;
    for (int i=0; i<refValues.size(); i++)
    {
      double expected = refValues[i] * scalingFactor;
      double actual = scaledRefValues[i];
      double diff = abs(expected-actual);
      if (diff > tol)
      {
        success = false;
      }
      maxDiff = max(maxDiff,diff);
    }
    if (!success)
    {
      out << "Expected and actual values differ; refValues were:\n" << refValues;
      out << "Expected these to be scaled by " << scalingFactor << " in scaledRefValues:\n" << scaledRefValues;
      out << "Maximum difference is " << maxDiff << endl;
    }
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Line )
  {
    CellTopoPtr cellTopo = CellTopology::line();
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Quad )
  {
    CellTopoPtr cellTopo = CellTopology::quad();
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Triangle )
  {
    CellTopoPtr cellTopo = CellTopology::triangle();
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Tetrahedron )
  {
    CellTopoPtr cellTopo = CellTopology::tetrahedron();
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Hexahedron )
  {
    CellTopoPtr cellTopo = CellTopology::hexahedron();
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, hFunction_Hexahedron_x_Line )
  {
    CellTopoPtr hex = CellTopology::hexahedron();
    CellTopoPtr cellTopo = CellTopology::lineTensorTopology(hex);
    testHFunction(cellTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Function, L2NormOfJumps_RepFunction )
  {
    // In this test, we set up a 2x2 mesh with unit RepFunction values on the lower-left and upper-right cells,
    // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
    // With this setup, the solution jumps should be 1.0 everywhere on the interior of the mesh.  The interior
    // mesh skeleton has total length of 4.0, so that the squared L^2 norm of the jumps is 4.0...
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {2,2};
    set<int> unitCellIDs = {0,3};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {2.0,2.0}, elemCounts, H1Order);
    
    LinearTermPtr residual; // leave as a null pointer for now; we shouldn't actually use this...
    IPPtr ip = form.bf()->graphNorm(); // we actually shouldn't use this either
    RieszRepPtr rieszRep = Teuchos::rcp(new RieszRep(mesh, ip, residual));
    
    auto myCellIDs = mesh->cellIDsInPartition();
    
    for (int cellID : myCellIDs)
    {
      auto testOrdering = mesh->getElementType(cellID)->testOrderPtr;
      Camellia::FieldContainer<double> rieszCoefficients(testOrdering->totalDofs());
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      if (hasUnitSolution) rieszCoefficients.initialize(1.0);
      else                 rieszCoefficients.initialize(0.0); // this will give all variables a constant 1.0 value
      
      rieszRep->setCoefficientsForCell(cellID, rieszCoefficients);
    }
    
    // v has scalar nodal H^1 basis, for which the unit coefficients definitely give us a unit function
    // (I'm not sure that that's true for the H(div) basis we use for tau.)
    FunctionPtr repFunction = RieszRep::repFunction(form.v(), rieszRep);
    
    // we expect the jumps to be 1 everywhere on the interior; each interior side has unit length,
    // and each cell has two interior sides, so that we have a total cell contribution of 2.0 on
    // the cells that are not set to unity, and a contribution of 4.0 on those that are
    map<GlobalIndexType,double> squaredL2NormOfJumpOnCell;
    for (auto cellID : myCellIDs)
    {
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      
      double squaredL2ValueExpected;
      
      if (hasUnitSolution) squaredL2ValueExpected = 4.0;
      else                 squaredL2ValueExpected = 2.0;
      
      squaredL2NormOfJumpOnCell[cellID] = squaredL2ValueExpected;
    }
    bool weightBySideMeasure = false;
    int cubatureDegreeEnrichment = 0;
    map<GlobalIndexType, double> cellL2Norms = repFunction->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
    
    for (auto entry : cellL2Norms)
    {
      auto cellID = entry.first;
      double l2OfJumpActual = entry.second;
      TEUCHOS_TEST_FLOATING_EQUALITY(l2OfJumpActual, squaredL2NormOfJumpOnCell[cellID], 1e-14, out, success);
    }
  }
  
  TEUCHOS_UNIT_TEST( Function, L2NormOfJumps_SolutionFunction )
  {
    // In this test, we set up a 2x2 mesh with unit solution values on the lower-left and upper-right cells,
    // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
    // With this setup, the solution jumps should be 1.0 everywhere on the interior of the mesh.  The interior
    // mesh skeleton has total length of 4.0, so that the squared L^2 norm of the jumps is 4.0...
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {2,2};
    set<int> unitCellIDs = {0,3};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {2.0,2.0}, elemCounts, H1Order);
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    LinearTermPtr dummyGoal = 1.0 * form.u(); // just something to let there be two solutions...
    solution->setGoalOrientedRHS(dummyGoal);
    
    solution->initializeLHSVector();
    
    auto myCellIDs = mesh->cellIDsInPartition();
    
    int solutionOrdinal = 1; // use the second solution for our actual test
    for (GlobalIndexType cellID : myCellIDs)
    {
      auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
      Camellia::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      if (hasUnitSolution) solnCoefficients.initialize(1.0);
      else                 solnCoefficients.initialize(0.0); // this will give all variables a constant 1.0 value
      
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, solutionOrdinal);
      
      // to emulate a solve context, there should be solution values for both solutions;
      // we just put 0 values for the other solution
      int otherSolutionOrdinal = 1 - solutionOrdinal;
      solnCoefficients.initialize(0.0);
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, otherSolutionOrdinal);
    }
    
    bool weightByParity = false;
    FunctionPtr u_goal = Function::solution(form.u(), solution, weightByParity, solutionOrdinal);
    
    // we expect the jumps to be 1 everywhere on the interior; each interior side has unit length,
    // and each cell has two interior sides, so that we have a total cell contribution of 2.0 on
    // the cells that are not set to unity, and a contribution of 4.0 on those that are
    map<GlobalIndexType,double> squaredL2NormOfJumpOnCell;
    for (auto cellID : myCellIDs)
    {
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      
      double squaredL2ValueExpected;
      
      if (hasUnitSolution) squaredL2ValueExpected = 4.0;
      else                 squaredL2ValueExpected = 2.0;
      
      squaredL2NormOfJumpOnCell[cellID] = squaredL2ValueExpected;
    }
    bool weightBySideMeasure = false;
    int cubatureDegreeEnrichment = 0;
    map<GlobalIndexType, double> cellL2Norms = u_goal->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
    
    for (auto entry : cellL2Norms)
    {
      auto cellID = entry.first;
      double l2OfJumpActual = entry.second;
      TEUCHOS_TEST_FLOATING_EQUALITY(l2OfJumpActual, squaredL2NormOfJumpOnCell[cellID], 1e-14, out, success);
    }
  }
  
  TEUCHOS_UNIT_TEST( Function, L2NormOfJumps_SolutionFunction_EdgeWeighted )
  {
    // In this test, we set up a 2x2 mesh with unit solution values on the lower-left and upper-right cells,
    // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
    // With this setup, the solution jumps should be 1.0 everywhere on the interior of the mesh.
    // Each element has two interior edges of length 0.5, so the edge-weighted L^2 norm of the jump is
    //  2 * 0.5 * 0.5.
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {2,2};
    set<int> unitCellIDs = {0,3};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, elemCounts, H1Order);
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    LinearTermPtr dummyGoal = 1.0 * form.u(); // just something to let there be two solutions...
    solution->setGoalOrientedRHS(dummyGoal);
    
    solution->initializeLHSVector();
    
    auto myCellIDs = mesh->cellIDsInPartition();
    
    int solutionOrdinal = 1; // use the second solution for our actual test
    for (int cellID : myCellIDs)
    {
      auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
      Camellia::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      if (hasUnitSolution) solnCoefficients.initialize(1.0);
      else                 solnCoefficients.initialize(0.0); // this will give all variables a constant 1.0 value
      
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, solutionOrdinal);
      
      // to emulate a solve context, there should be solution values for both solutions;
      // we just put 0 values for the other solution
      int otherSolutionOrdinal = 1 - solutionOrdinal;
      solnCoefficients.initialize(0.0);
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, otherSolutionOrdinal);
    }
    
    bool weightByParity = false;
    FunctionPtr u_goal = Function::solution(form.u(), solution, weightByParity, solutionOrdinal);
    
    // we expect the jumps to be 1 everywhere on the interior; each interior side has length 0.5,
    // and each cell has two interior sides, so that we have a total cell contribution of (2 * 0.5 * 0.5) = 0.5
    // on the cells that are not set to unity, and a contribution of (4 * 0.5 * 0.5) = 1.0 on those that are
    map<GlobalIndexType,double> squaredL2NormOfJumpOnCell;
    for (auto cellID : myCellIDs)
    {
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      
      double squaredL2ValueExpected;
      
      if (hasUnitSolution) squaredL2ValueExpected = 1.0;
      else                 squaredL2ValueExpected = 0.5;
      
      squaredL2NormOfJumpOnCell[cellID] = squaredL2ValueExpected;
    }
    bool weightBySideMeasure = true;
    int cubatureDegreeEnrichment = 0;
    map<GlobalIndexType, double> cellL2Norms = u_goal->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
    
    for (auto entry : cellL2Norms)
    {
      auto cellID = entry.first;
      double l2OfJumpActual = entry.second;
      TEUCHOS_TEST_FLOATING_EQUALITY(l2OfJumpActual, squaredL2NormOfJumpOnCell[cellID], 1e-14, out, success);
    }
    
  }
  
TEUCHOS_UNIT_TEST( Function, MinAndMaxFunctions )
{
  FunctionPtr one = Function::constant(1);
  FunctionPtr two = Function::constant(2);
  FunctionPtr minFcn = Function::min(one,two);
  FunctionPtr maxFcn = Function::max(one,two);
  double x0 = 0, y0 = 0;
  double expectedValue = 1.0;
  double actualValue = Function::evaluate(minFcn, x0, y0);
  double tol = 1e-14;
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
  expectedValue = 2.0;
  actualValue = Function::evaluate(maxFcn, x0, y0);
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralLine )
{
  FunctionPtr x = Function::xn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::line(), x, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralQuad )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::quad(), x * y, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralTriangle )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::triangle(), x * y, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralHexahedron )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::hexahedron(), x * y * z, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralTetrahedron )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::tetrahedron(), x * y * z, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeLine )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::line(), x*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeQuad )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::quad(), x*y*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeTriangle )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::triangle(), x*y*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalLine )
{
  testSpaceTimeNormal(CellTopology::line(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalQuad )
{
  testSpaceTimeNormal(CellTopology::quad(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTriangle )
{
  testSpaceTimeNormal(CellTopology::triangle(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalHexahedron )
{
  testSpaceTimeNormal(CellTopology::hexahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTetrahedron )
{
  testSpaceTimeNormal(CellTopology::tetrahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentLine )
{
  testSpaceTimeNormalTimeComponent(CellTopology::line(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentQuad )
{
  testSpaceTimeNormalTimeComponent(CellTopology::quad(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentTriangle )
{
  testSpaceTimeNormalTimeComponent(CellTopology::triangle(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentHexahedron )
{
  testSpaceTimeNormalTimeComponent(CellTopology::hexahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentTetrahedron )
{
  testSpaceTimeNormalTimeComponent(CellTopology::tetrahedron(), out, success);
}

// locally defined function class for the ValuesDevicePath test below
class FunctionNoDevicePath : public TFunction<double>
{
private:
  double _value = 1.0;
public:
  FunctionNoDevicePath(double value) : TFunction<>(0) // rank 0 (scalar) function
  {
    _value = value;
  }
  
  std::string displayString() override
  {
    return "FunctionNoDevicePath (for test)";
  }
  
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    for (int i=0; i<values.size(); i++)
    {
      values[i] = _value;
    }
  }
};
  
TEUCHOS_UNIT_TEST( Function, ValuesDevicePath )
{
  const int dim0 = 5, dim1 = 2;
  const double existingValue = 3.141592; // the value already in the view prior to calling values()
  const double constantValue = 1.5;      // function value
  double expectedValue;
  
  FunctionNoDevicePath f(constantValue); // class does not implement the device-path values(); relies on base class implementation
  
  std::vector<ValueModificationType> modTypes = {ASSIGN, ADD, SUBTRACT, MULTIPLY, DIVIDE};
  for (auto modType : modTypes)
  {
    switch (modType)
    {
      case Camellia::ASSIGN:
        expectedValue = constantValue;
        break;
      case Camellia::ADD:
        expectedValue = existingValue + constantValue;
        break;
      case Camellia::SUBTRACT:
        expectedValue = existingValue - constantValue;
        break;
      case Camellia::MULTIPLY:
        expectedValue = existingValue * constantValue;
        break;
      case Camellia::DIVIDE:
        expectedValue = existingValue / constantValue;
    }
    auto fc = FieldContainer<double>(dim0,dim1);
    auto fcView = fc.checkoutDeviceView(false); // false: don't copy from host
    Kokkos::deep_copy( fcView, existingValue);
    
    BasisCachePtr nullBasisCache;
    f.TFunction<double>::values(fcView, modType, nullBasisCache); // device path
    fc.checkinDeviceView(true); // true: copy back to host
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }
}
  
TEUCHOS_UNIT_TEST( Function, VectorMultiply )
{
  FunctionPtr x2 = Function::xn(2);
  FunctionPtr y4 = Function::yn(4);
  vector<double> weight(2);
  weight[0] = 3;
  weight[1] = 2;
  FunctionPtr g = Function::vectorize(x2,y4);
  double x0 = 2, y0 = 3;
  double expectedValue = weight[0] * x0 * x0 + weight[1] * y0 * y0 * y0 * y0;
  double actualValue = Function::evaluate(g * weight, x0, y0);
  double tol = 1e-14;
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
}

TEUCHOS_UNIT_TEST( Function, AddToValues )
{
  const int dim0 = 5, dim1 = 2;
  auto fc = FieldContainer<double>(dim0,dim1);
  auto f1 = TFunction<double>::constant(1.0);
  auto f2 = TFunction<double>::constant(2.0);
  BasisCachePtr voidBasisCache;
  
  double expectedValue = 3.0;
  double tol = 1.0e-14;
  
  f1->values(fc, voidBasisCache);
  f2->addToValues(fc, voidBasisCache);
  
  for (int i = 0; i < dim0*dim1; i++) {
    TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
  }
}

TEUCHOS_UNIT_TEST( Function, ScalarModifyFunctionValues )
{
  const int dim0 = 5, dim1 = 2;
  auto fc_mul = FieldContainer<double>(dim0,dim1);
  auto fc_div = FieldContainer<double>(dim0,dim1);
  auto f2 = TFunction<double>::constant(2.0);
  auto f4 = TFunction<double>::constant(4.0);
  BasisCachePtr voidBasisCache;
  
  double expectedValueMultiply = 8.0;
  double expectedValueDivide   = 0.5;
  double tol = 1.0e-14;
  
  f2->values(fc_mul, voidBasisCache);
  f2->values(fc_div, voidBasisCache);
  f4->scalarMultiplyFunctionValues(fc_mul, voidBasisCache);
  f4->scalarDivideFunctionValues  (fc_div, voidBasisCache);
  
  for (int i = 0; i < dim0*dim1; i++) {
    TEST_FLOATING_EQUALITY(expectedValueMultiply,fc_mul[i],tol);
    TEST_FLOATING_EQUALITY(expectedValueDivide  ,fc_div[i],tol);
  }
}

TEUCHOS_UNIT_TEST( Function, ScalarModifyBasisValues )
{
  const int dim0 = 5, dim1 = 2, dim2 = 2;
  auto fc_mul = FieldContainer<double>(dim0,dim1,dim2);
  auto fc_div = FieldContainer<double>(dim0,dim1,dim2);
  std::vector<double> v2 = {2.0,2.0};
  auto f2 = TFunction<double>::constant(v2);
  auto f4 = TFunction<double>::constant(4.0);
  f4 = TFunction<double>::min(f4,4.0);
  BasisCachePtr voidBasisCache;
  
  double expectedValueMultiply = 8.0;
  double expectedValueDivide   = 0.5;
  double tol = 1.0e-14;
  
  f2->values(fc_mul, voidBasisCache);
  f2->values(fc_div, voidBasisCache);
  f4->scalarMultiplyBasisValues(fc_mul, voidBasisCache);
  f4->scalarDivideBasisValues  (fc_div, voidBasisCache);
  
  for (int i = 0; i < dim0*dim1*dim2; i++) {
    TEST_FLOATING_EQUALITY(expectedValueMultiply,fc_mul[i],tol);
    TEST_FLOATING_EQUALITY(expectedValueDivide  ,fc_div[i],tol);
  }
}

TEUCHOS_UNIT_TEST( Function, ValuesDottedWithTensor )
{
  std::vector<double> v1 = {1.0,2.0};
  std::vector<double> v2 = {1.0,2.0};
  auto f1 = TFunction<double>::constant(v1);
  auto f2 = TFunction<double>::constant(v2);
  
  bool conformingTraces = true; int spaceDim = 2;
  PoissonFormulation form(spaceDim,conformingTraces);
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, 1);
  
  // |1.0| dotProd |1.0| = 1.0*1.0 + 2.0*2.0 = 5.0
  // |2.0|         |2.0|
  double expectedValue = 5.0;
  double tol = 1.0e-14;
  
  for (int cellID : mesh->cellIDsInPartition()) {
    auto fc = FieldContainer<double>(1,1);
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    f1->valuesDottedWithTensor(fc, f2, basisCache);
    TEST_FLOATING_EQUALITY(expectedValue,fc(0,0),tol);
  }
}
} // namespace
