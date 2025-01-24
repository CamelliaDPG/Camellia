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
  const bool tensorTopologyMeansSpaceTime = true;
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache, tensorTopologyMeansSpaceTime);
  FunctionPtr spaceTimeNormalComponent = Function::normalSpaceTime()->t();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    Camellia::FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPointsFC().dimension(0));
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
  const bool tensorTopologyMeansSpaceTime = true;
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache, tensorTopologyMeansSpaceTime);
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
  const bool tensorTopologyMeansSpaceTime = true;
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache, tensorTopologyMeansSpaceTime);
  FunctionPtr spaceTimeNormal = Function::normalSpaceTime();
  FunctionPtr spaceNormal = Function::normal();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    Camellia::FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPointsFC().dimension(0),spaceTimeTopo->getDimension());
    spaceTimeNormal->values(spaceTimeNormals,spaceTimeSideCache);

    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      // expect spaceTimeNormals to match spatial normals in the first d dimensions, and to be 0 in the final dimension
      int spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
      BasisCachePtr spaceSideCache = spaceBasisCache->getSideBasisCache(spatialSideOrdinal);

      Camellia::FieldContainer<double> spaceNormals(1,spaceSideCache->getRefCellPointsFC().dimension(0),spaceTopo->getDimension());
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
    int numPoints = refBasisCache->getPhysicalCubaturePointsFC().dimension(1); // C, P, D
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
    
    // Test device path values implementation with a different modType
    auto dv_scaledRefValues = scaledRefValues.checkoutDeviceView();
    h->values(dv_scaledRefValues, ValueModificationType::ADD, scaledBasisCache);
    scaledRefValues.checkinDeviceView(true); // true: copy back to host
    for (int i=0; i<refValues.size(); i++)
    {
      double expected = 2.0 * (refValues[i] * scalingFactor);
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
    
    const double domain_x = 2.0;
    const double domain_y = 2.0;
    
    const double h_x         = domain_x / elemCounts[0];
    const double h_y         = domain_y / elemCounts[1];
    const double ref_space_h = 2.0; // [-1,1] ref space line
    const double h_scaling_x = h_x / ref_space_h;
    const double h_scaling_y = h_y / ref_space_h;
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {domain_x,domain_y}, elemCounts, H1Order);
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    LinearTermPtr dummyGoal = 1.0 * form.u(); // just something to let there be two solutions...
    solution->setGoalOrientedRHS(dummyGoal);
    
    solution->initializeLHSVector();
    
    auto myCellIDs = mesh->cellIDsInPartition();
    
    const double coefficient_scaling = (form.u()->space() == L2) ? h_scaling_x * h_scaling_y : 1.0;
    
    SolutionComponent solutionComponent = INFLUENCE_SOLUTION; // use the second solution for our actual test
    int solutionStage = 0;
    int solutionOrdinal = solution->solutionOrdinal(solutionComponent, solutionStage);
    for (GlobalIndexType cellID : myCellIDs)
    {
      auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
      Camellia::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      if (hasUnitSolution) solnCoefficients.initialize(1.0 * coefficient_scaling);
      else                 solnCoefficients.initialize(0.0 * coefficient_scaling); // this will give all variables a constant 1.0 value
      
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, solutionOrdinal);
      
      // to emulate a solve context, there should be solution values for both solutions;
      // we just put 0 values for the other solution
      int otherSolutionOrdinal = 1 - solutionOrdinal;
      solnCoefficients.initialize(0.0);
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, otherSolutionOrdinal);
    }
    
    bool weightByParity = false;
    FunctionPtr u_goal = Function::solution(form.u(), solution, weightByParity, solutionComponent, solutionStage);
    
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
    
    const double domain_x = 1.0;
    const double domain_y = 1.0;
    
    const double h_x         = domain_x / elemCounts[0];
    const double h_y         = domain_y / elemCounts[1];
    const double ref_space_h = 2.0; // [-1,1] ref space line
    const double h_scaling_x = h_x / ref_space_h;
    const double h_scaling_y = h_y / ref_space_h;
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {domain_x,domain_y}, elemCounts, H1Order);
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    LinearTermPtr dummyGoal = 1.0 * form.u(); // just something to let there be two solutions...
    solution->setGoalOrientedRHS(dummyGoal);
    
    solution->initializeLHSVector();
    
    auto myCellIDs = mesh->cellIDsInPartition();
    
    const double coefficient_scaling = (form.u()->space() == L2) ? h_scaling_x * h_scaling_y : 1.0;
    
    SolutionComponent solutionComponent = INFLUENCE_SOLUTION; // use the second solution for our actual test
    int solutionStage = 0;
    int solutionOrdinal = solution->solutionOrdinal(solutionComponent, solutionStage);
    for (int cellID : myCellIDs)
    {
      auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
      Camellia::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
      bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
      if (hasUnitSolution) solnCoefficients.initialize(1.0 * coefficient_scaling);
      else                 solnCoefficients.initialize(0.0 * coefficient_scaling); // this will give all variables a constant 1.0 value
      
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, solutionOrdinal);
      
      // to emulate a solve context, there should be solution values for both solutions;
      // we just put 0 values for the other solution
      int otherSolutionOrdinal = 1 - solutionOrdinal;
      solnCoefficients.initialize(0.0);
      solution->setSolnCoeffsForCellID(solnCoefficients, cellID, otherSolutionOrdinal);
    }
    
    bool weightByParity = false;
    FunctionPtr u_goal = Function::solution(form.u(), solution, weightByParity, solutionComponent, solutionStage);
    
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

TEUCHOS_UNIT_TEST( Function, NeighborFunction )
{
  // In this test, we set up a 2x2 mesh with unit solution values on the lower-left and upper-right cells,
  // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
  // We test the values pointwise.
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  int H1Order = 1;
  vector<int> elemCounts = {2,2};
  set<int> unitCellIDs = {0,3};
  
  const double domain_x = 1.0;
  const double domain_y = 1.0;
  
  const double h_x         = domain_x / elemCounts[0];
  const double h_y         = domain_y / elemCounts[1];
  const double ref_space_h = 2.0; // [-1,1] ref space line
  const double h_scaling_x = h_x / ref_space_h;
  const double h_scaling_y = h_y / ref_space_h;
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {domain_x,domain_y}, elemCounts, H1Order);
  SolutionPtr solution = Solution::solution(form.bf(), mesh);
  
  const double coefficient_scaling = (form.u()->space() == L2) ? h_scaling_x * h_scaling_y : 1.0;
  
  solution->initializeLHSVector();
  
  auto myCellIDs = mesh->cellIDsInPartition();
  
  for (int cellID : myCellIDs)
  {
    auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
    Intrepid::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
    bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
    if (hasUnitSolution) solnCoefficients.initialize(1.0 * coefficient_scaling);
    else                 solnCoefficients.initialize(0.0 * coefficient_scaling); // this will give all variables a constant 1.0 value
    
    solution->setSolnCoeffsForCellID(solnCoefficients, cellID);
  }
  
  VarPtr u = form.u();
  FunctionPtr u_soln = Function::solution(u, solution);
  auto u_neighbor = Function::neighborFunction(u_soln, mesh);
  
  std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(),myCellIDs.end());
  u_neighbor->importCellData(myCellIDsVector);
  
  for (int cellID : myCellIDs)
  {
    bool cellHasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
    double myValue = cellHasUnitSolution ? 1.0 : 0.0;
    
    auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      // for boundary sides, the "neighbor" value is the same as the value in the current cell
      // for non-boundary sides, the neighbor value is 1.0 - myValue
      bool isBoundary = mesh->getTopology()->getCell(cellID)->isBoundary(sideOrdinal);
      double expectedValue = isBoundary ? myValue : 1.0 - myValue;
      auto sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      int numCells = 1;
      int numPoints = sideBasisCache->getRefCellPointsFC().dimension(0); // (P,D) container
      Camellia::FieldContainer<double> neighborValues(numCells,numPoints);
      u_neighbor->values(neighborValues, sideBasisCache);
      
      double tol = 1e-15;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        auto actualValue = neighborValues(0,pointOrdinal);
        if (expectedValue == 0.0)
        {
          TEST_COMPARE(abs(actualValue), <, tol);
        }
        else
        {
          TEST_FLOATING_EQUALITY(actualValue, expectedValue, tol);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Function, NeighborFunctionWithNormal )
{
  /*
    *-------*-------*
    |       |       |
    |   1   |   3   |
    |       |       |
    *-------*-------*
    |       |       |
    |   0   |   2   |
    |       |       |
    *-------*-------*
  */
  
  // In this test, we set up a 2x2 mesh with a linear solution everywhere ( u(x,y) = x ).
  // We test the neighbor values on the mesh skeleton pointwise for the normal flux ( grad(u)*n ).
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  int H1Order = 1;
  vector<int> elemCounts = {2,2};
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, elemCounts, H1Order);
  SolutionPtr solution = Solution::solution(form.bf(), mesh);
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr u_exact = x;
  BCPtr bc = BC::bc();
  bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_exact);
  
  FunctionPtr n = Function::normal();
  auto u_neighbor = Function::neighborFunction(u_exact->grad(2)*n, mesh);
  
  auto myCellIDs = mesh->cellIDsInPartition();
  std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(),myCellIDs.end());
  u_neighbor->importCellData(myCellIDsVector);
  
  for (int cellID : myCellIDs)
  {
    auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      auto sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      
      // for sides normal to x, the "neighbor" value is +1 or -1, depending on the neighbor's normal direction
      // for sides normal to y, the "neighbor" value is 0
      double expectedValue;
      bool isBoundary = mesh->getTopology()->getCell(cellID)->isBoundary(sideOrdinal);
      // given a single cell and straight edges, the side normal values will be the same at every (C,P) index
      const int sampleCellOrdinal = 0;
      const int sampleCellPoint   = 0;
      const int dim_x             = 0; // since the gradient of x vanishes at all sides for which the y component is nonzero, we only need to consider the x component of the normal
      auto normal_x = sideBasisCache->getSideNormals()(sampleCellOrdinal,sampleCellPoint,dim_x);
      if (isBoundary) { expectedValue =  normal_x; } // on the boundary, neighbor function duplicates the interior of the cell
      else            { expectedValue = -normal_x; }
      
      int numCells = 1;
      int numPoints = sideBasisCache->getRefCellPointsFC().dimension(0); // (P,D) container
      Camellia::FieldContainer<double> neighborValues(numCells,numPoints);
      u_neighbor->values(neighborValues, sideBasisCache);
      
      double tol = 1e-13;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        auto actualValue = neighborValues(0,pointOrdinal);
        if (expectedValue == 0.0)
        {
          TEST_COMPARE(abs(actualValue), <, tol);
        }
        else
        {
          TEST_FLOATING_EQUALITY(actualValue, expectedValue, tol);
        }
      }
    }
  }
}
  
TEUCHOS_UNIT_TEST( Function, NeighborFunctionWithHangingNode )
{
  // In this test, we set up a 2x2 mesh with unit solution values on the lower-left through upper-right cells,
  // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
  // We refine the lower-left cell, resulting in something of the form
  /*
   _________________
   |       |       |
   |   1   |   3   |
   |       |       |
   |_______|_______|
   | 7 | 6 |       |
   |___|___|   2   |
   | 4 | 5 |       |
   |___|___|_______|
   
   Here, 4,6,3 have unit values.
   */
  // We test the values pointwise.
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  enum ExpectedValueCategory
  {
    ZERO,
    ONE,
    ZERO_FOR_X_LESS_THAN_POINT_25, // (1.0 above 0.25)
    ZERO_FOR_Y_LESS_THAN_POINT_25  // (1.0 above 0.25)
  };
  
  std::map<int, std::vector<ExpectedValueCategory> > expectedValueMap; // index: [cellID][sideOrdinal]
  expectedValueMap[1] = {ZERO_FOR_X_LESS_THAN_POINT_25,ONE,ZERO,ZERO}; // side numbering: SENW
  expectedValueMap[2] = {ZERO,ZERO,ONE,ZERO_FOR_Y_LESS_THAN_POINT_25};
  expectedValueMap[3] = {ZERO,ONE,ONE,ZERO};
  expectedValueMap[4] = {ONE,ZERO,ZERO,ONE};
  expectedValueMap[5] = {ZERO,ZERO,ONE,ONE};
  expectedValueMap[6] = {ZERO,ZERO,ZERO,ZERO};
  expectedValueMap[7] = {ONE,ONE,ZERO,ZERO};
  
  int H1Order = 1;
  vector<int> elemCounts = {2,2};
  set<int> unitCellIDs = {3,4,6};
  
  const double domain_x = 1.0;
  const double domain_y = 1.0;
  
  const double h_x         = domain_x / elemCounts[0];
  const double h_y         = domain_y / elemCounts[1];
  const double ref_space_h = 2.0; // [-1,1] ref space line
  const double h_scaling_x = h_x / ref_space_h;
  const double h_scaling_y = h_y / ref_space_h;
  
  const double coefficient_scaling = (form.u()->space() == L2) ? h_scaling_x * h_scaling_y : 1.0;
  const double refined_coefficient_scaling = coefficient_scaling / 4.0;
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {domain_x,domain_y}, elemCounts, H1Order);
  const vector<GlobalIndexType> cellsToHRefine = {0};
  mesh->hRefine(cellsToHRefine);
  
  SolutionPtr solution = Solution::solution(form.bf(), mesh);
  
  solution->initializeLHSVector();
  
  auto myCellIDs = mesh->cellIDsInPartition();
  
  for (int cellID : myCellIDs)
  {
    auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
    Intrepid::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
    bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
    if (hasUnitSolution)
    {
      // this will give all variables a constant 1.0 value:
      if (cellID == 3) solnCoefficients.initialize(1.0 * coefficient_scaling);
      else             solnCoefficients.initialize(1.0 * refined_coefficient_scaling); // cells 4 and 6 are refined
    }
    else
    {
      solnCoefficients.initialize(0.0 * coefficient_scaling);
    }
    
    solution->setSolnCoeffsForCellID(solnCoefficients, cellID);
  }
  
  VarPtr u = form.u();
  FunctionPtr u_soln = Function::solution(u, solution);
  auto u_neighbor = Function::neighborFunction(u_soln, mesh);
  
  std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(),myCellIDs.end());
  u_neighbor->importCellData(myCellIDsVector);
  
  for (int cellID : myCellIDs)
  {
    out << "Checking cell " << cellID << std::endl;
    auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      out << "Checking side " << sideOrdinal << std::endl;
      auto expectedValueCategory = expectedValueMap[cellID][sideOrdinal];
      auto sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      int numCells = 1;
      int numPoints = sideBasisCache->getRefCellPointsFC().dimension(0); // (P,D) container
      Camellia::FieldContainer<double> neighborValues(numCells,numPoints);
      u_neighbor->values(neighborValues, sideBasisCache);
      
      double tol = 1e-15;
      auto &physicalPoints = sideBasisCache->getPhysicalCubaturePointsFC();
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double expectedValue;
        double x = physicalPoints(0,pointOrdinal,0);
        double y = physicalPoints(0,pointOrdinal,1);
        out << "Checking point (" << x << "," << y << ")" << std::endl;
        switch (expectedValueCategory) {
          case ZERO:
            expectedValue = 0.0;
            break;
          case ONE:
            expectedValue = 1.0;
            break;
          case ZERO_FOR_X_LESS_THAN_POINT_25:
            if (x == 0.25)
            {
              expectedValue = 0.5; // two cells match the point at 0.25, and we take the average
            }
            else if (x < 0.25)
            {
              expectedValue = 0.0;
            }
            else
            {
              expectedValue = 1.0;
            }
            break;
          case ZERO_FOR_Y_LESS_THAN_POINT_25:
            if (y == 0.25)
            {
              expectedValue = 0.5; // two cells match the point at 0.25, and we take the average
            }
            else if (y < 0.25)
            {
              expectedValue = 0.0;
            }
            else
            {
              expectedValue = 1.0;
            }
            break;
        }
        
        auto actualValue = neighborValues(0,pointOrdinal);
        if (expectedValue == 0.0)
        {
          TEST_COMPARE(abs(actualValue), <, tol);
        }
        else
        {
          TEST_FLOATING_EQUALITY(actualValue, expectedValue, tol);
        }
      }
    }
  }
}
  
TEUCHOS_UNIT_TEST( Function, SolutionFunctionSkeletonRestriction )
{
  // In this test, we set up a 2x2 mesh with unit solution values on the lower-left and upper-right cells,
  // and zero solutions in the others.  The lower-left cell has ID 0; the upper-right, 3.
  // The idea here is that we have a discontinuous solution, and want to contribute to some global integral
  // by integrating local contributions.  (It is necessary that the thing you integrate against is in some
  // sense globally continuous: i.e., the "left" and "right" cell values should be tested with the same test function.)
  // We test that the integral on each cell matches the expected value.
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  int H1Order = 1;
  vector<int> elemCounts = {2,2};
  set<int> unitCellIDs = {0,3};
  
  const double domain_x = 1.0;
  const double domain_y = 1.0;
  
  const double h_x         = domain_x / elemCounts[0];
  const double h_y         = domain_y / elemCounts[1];
  const double ref_space_h = 2.0; // [-1,1] ref space line
  const double h_scaling_x = h_x / ref_space_h;
  const double h_scaling_y = h_y / ref_space_h;
  const double coefficient_scaling = (form.u()->space() == L2) ? h_scaling_x * h_scaling_y : 1.0;
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {domain_x,domain_y}, elemCounts, H1Order);
  SolutionPtr solution = Solution::solution(form.bf(), mesh);
  
  solution->initializeLHSVector();
  
  auto myCellIDs = mesh->cellIDsInPartition();
  
  for (int cellID : myCellIDs)
  {
    auto trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
    Intrepid::FieldContainer<double> solnCoefficients(trialOrdering->totalDofs());
    bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
    if (hasUnitSolution) solnCoefficients.initialize(1.0 * coefficient_scaling);
    else                 solnCoefficients.initialize(0.0 * coefficient_scaling); // this will give all variables a constant 1.0 value
    
    solution->setSolnCoeffsForCellID(solnCoefficients, cellID);
  }
  
  FunctionPtr gamma   = Function::meshBoundaryCharacteristic();
  FunctionPtr gamma_h = Function::meshSkeletonCharacteristic();
  
  VarPtr u = form.u();
  FunctionPtr u_soln = Function::solution(u, solution);
  // we want the value of u on the mesh boundary, and the average value of u ("left" and "right") on the interior skeleton:
  //    u_soln * gamma + (gamma_h - gamma) * 0.5 * u_soln;
  // this simplifies to:
  //    0.5 * (gamma_h + gamma) * u_soln
  FunctionPtr u_restriction = 0.5 * (gamma_h + gamma) * u_soln;
  
  // set up expected integrals on each cell:
  // - on cell with zero value, we expect 0
  // - on a cell with unit value, we expect:
  //   interior edges: edge_length * 0.5 * 1.0 = 0.5 * 0.5 * 1.0 = 0.25
  //   boundary edges: edge_length * 1.0 * 1.0 = 0.5
  //   each cell has 2 interior edges and 2 boundary edges, so the expected integral is 1.5
  for (int cellID : myCellIDs)
  {
    double actualValue = u_restriction->integrate(cellID, mesh);
    bool hasUnitSolution = unitCellIDs.find(cellID) != unitCellIDs.end();
    double expectedValue = hasUnitSolution ? 1.5 : 0.0;
    double tol = 1e-15;
    if (expectedValue == 0.0)
    {
      TEST_COMPARE(abs(actualValue), <, tol);
    }
    else
    {
      TEST_FLOATING_EQUALITY(actualValue, expectedValue, tol);
    }
  }
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
  
  using TFunction<double>::values;
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
        break;
      case Camellia::MINIMUM:
        expectedValue = (existingValue < constantValue) ? existingValue : constantValue;
        break;
      case Camellia::MAXIMUM:
        expectedValue = (existingValue > constantValue) ? existingValue : constantValue;
    }
    auto fc = FieldContainer<double>(dim0,dim1);
    auto fcView = fc.checkoutDeviceView(false); // false: don't copy from host
    Kokkos::deep_copy( fcView, existingValue);
    
    BasisCachePtr simpleBasisCache = BasisCache::basisCache1D(0, 1, 3);
    FieldContainer<double> refCellPoints(dim1,1);
    FieldContainer<double> cellNodes(dim0,2,1); // 2 points define line
    simpleBasisCache->setRefCellPointsFC(refCellPoints);
    simpleBasisCache->setPhysicalCellNodes(cellNodes, {0,1}, false);
    f.TFunction<double>::values(fcView, modType, simpleBasisCache); // device path
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
  BasisCachePtr nullBasisCache;
  
  double expectedValue = 3.0;
  double tol = 1.0e-14;
  
  f1->values(fc, nullBasisCache);
  f2->addToValues(fc, nullBasisCache);
  
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
  BasisCachePtr nullBasisCache;
  
  double expectedValueMultiply = 8.0;
  double expectedValueDivide   = 0.5;
  double tol = 1.0e-14;
  
  f2->values(fc_mul, nullBasisCache);
  f2->values(fc_div, nullBasisCache);
  f4->scalarMultiplyFunctionValues(fc_mul, nullBasisCache);
  f4->scalarDivideFunctionValues  (fc_div, nullBasisCache);
  
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
  BasisCachePtr nullBasisCache;
  
  double expectedValueMultiply = 8.0;
  double expectedValueDivide   = 0.5;
  double tol = 1.0e-14;
  
  f2->values(fc_mul, nullBasisCache);
  f2->values(fc_div, nullBasisCache);
  f4->scalarMultiplyBasisValues(fc_mul, nullBasisCache);
  f4->scalarDivideBasisValues  (fc_div, nullBasisCache);
  
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

TEUCHOS_UNIT_TEST( Function, Equals )
{
  auto f1 = TFunction<double>::constant(1.00);
  auto f2 = TFunction<double>::constant(1.05);
  auto f3 = TFunction<double>::constant(0.05);
  
  int spaceDim = 2; bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, 1);
  
  double tol = 1.0e-14;
  for (int cellID : mesh->cellIDsInPartition()) {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    TEST_EQUALITY(f1->equals(f2        , basisCache, tol), false);
    TEST_EQUALITY(f2->equals(f1+f3     , basisCache, tol), true );
    TEST_EQUALITY(f3->equals((f2-f1)*f1, basisCache, tol), true );
  }
}

TEUCHOS_UNIT_TEST( Function, MeshBoundaryCharacteristic )
{
  FunctionPtr gamma = Function::meshBoundaryCharacteristic();
  
  bool conformingTraces = true; int spaceDim = 2;
  PoissonFormulation form(spaceDim,conformingTraces);
  MeshPtr mesh1 = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, 1);
  
  double tol = 1.0e-14;
  double expectedValue0 = 0.0;
  double expectedValue1 = 1.0;
  double expectedValue2 = 2.0;
  
  for (int cellID : mesh1->cellIDsInPartition())
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh1, cellID);
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      auto fc = FieldContainer<double>(1,1);
      gamma->values(fc, sideBasisCache);
      TEST_FLOATING_EQUALITY(expectedValue1, fc(0,0), tol);
      auto dv = fc.checkoutDeviceView();
      gamma->values(dv, ValueModificationType::ADD, sideBasisCache);
      fc.checkinDeviceView(true); // true: deep copy to host
      TEST_FLOATING_EQUALITY(expectedValue2, fc(0,0), tol);
    }
  }
  
  /*
    *-----*-----*
    |     |     |
    |  0  |  1  |
    |     |     |
    *-----*-----*
  */
  MeshPtr mesh2 = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {2,1}, 1);
  double expectedValue[2][4] = { {1, 0, 1, 1}, {1, 1, 1, 0} };
  for (int cellID : mesh2->cellIDsInPartition())
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh2, cellID);
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      auto fc = FieldContainer<double>(1,1);
      gamma->values(fc, sideBasisCache);
      TEST_FLOATING_EQUALITY(expectedValue[cellID][sideOrdinal], fc(0,0), tol);
      auto dv = fc.checkoutDeviceView();
      gamma->values(dv, ValueModificationType::SUBTRACT, sideBasisCache);
      fc.checkinDeviceView(true); // true: copy to host
      TEST_FLOATING_EQUALITY(expectedValue0, fc(0,0), tol);
    }
  }
}

} // namespace
