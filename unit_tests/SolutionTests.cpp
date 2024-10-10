//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CellTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Camellia_FieldContainer.hpp"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CamelliaTestingHelpers.h"
#include "Cell.h"
#include "ERKTableau.h"
#include "GlobalDofAssignment.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MeshUtilities.h"
#include "MonomialFunctions.h"
#include "PoissonFormulation.h"
#include "Projector.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "StokesVGPFormulation.h"
#include "Var.h"

using namespace Camellia;

namespace
{
  
  MeshPtr poissonUniformMesh(vector<int> elementWidths, int H1Order, bool useConformingTraces)
  {
    int spaceDim = elementWidths.size();
    int testSpaceEnrichment = spaceDim; //
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(spaceDim,span);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
    return mesh;
  }
  
  MeshPtr poissonUniformMesh(int spaceDim, int elementWidth, int H1Order, bool useConformingTraces)
  {
    vector<int> elementCounts(spaceDim,elementWidth);
    return poissonUniformMesh(elementCounts, H1Order, useConformingTraces);
  }
  
  MeshPtr poissonIrregularMesh(int spaceDim, int irregularity, int H1Order)
  {
    bool useConformingTraces = true;
    
    int elementWidth = 2;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    int meshIrregularity = 0;
    vector<GlobalIndexType> cellsToRefine = {1};
    int mySideOrdinal = 0;
    int globalSideOrdinal;
    if (mesh->cellIDsInPartition().find(cellsToRefine[0]) != mesh->cellIDsInPartition().end())
    {
      CellPtr cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
      const int sideCount = cellToRefine->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        if (cellToRefine->getNeighbor(sideOrdinal, mesh->getTopology()) != Teuchos::null)
        {
          mySideOrdinal = sideOrdinal;
          break;
        }
      }
    }
    mesh->Comm()->SumAll(&mySideOrdinal, &globalSideOrdinal, 1);
    
    while (meshIrregularity < irregularity)
    {
      //      print("refining cells", cellsToRefine);
      mesh->hRefine(cellsToRefine);
      meshIrregularity++;
      
      mySideOrdinal = 0;
      if (mesh->getTopology()->isValidCellIndex(cellsToRefine[0]))
      {
        CellPtr cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
        const int sideCount = cellToRefine->getSideCount();
        for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
        {
          if (cellToRefine->getNeighbor(sideOrdinal, mesh->getTopology()) != Teuchos::null)
          {
            mySideOrdinal = sideOrdinal;
            break;
          }
        }
      }
      mesh->Comm()->MaxAll(&mySideOrdinal, &globalSideOrdinal, 1);
      
      // setup for the next refinement, if any:
      mySideOrdinal = 0;
      int myChildWithNeighborCellID = 0;
      if (mesh->getTopology()->isValidCellIndex(cellsToRefine[0]))
      {
        CellPtr cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
        auto childEntry = cellToRefine->childrenForSide(globalSideOrdinal)[0];
        myChildWithNeighborCellID = childEntry.first;
      }
      int globalChildWithNeighborCellID;
      mesh->Comm()->MaxAll(&myChildWithNeighborCellID, &globalChildWithNeighborCellID, 1);
      cellsToRefine = {(GlobalIndexType)globalChildWithNeighborCellID};
    }
    return mesh;
  }
  
  void testCondensedSolveZeroMeanConstraint(bool minRule, Teuchos::FancyOStream &out, bool &success)
  {
    double tol = 1e-11;
    
    int spaceDim = 2;
    bool conformingTraces = false; // false mostly because I want to do cavity flow with non-H^1 BCs
    double mu = 1.0;
    StokesVGPFormulation stokesForm = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    
    VarPtr u1 = stokesForm.u(1);
    VarPtr u2 = stokesForm.u(2);
    VarPtr p = stokesForm.p();
    
    VarPtr u1hat = stokesForm.u_hat(1);
    VarPtr u2hat = stokesForm.u_hat(2);
    
    BFPtr bf = stokesForm.bf();
    
    // robust test norm
    IPPtr ip = bf->graphNorm();
    
    ////////////////////   SPECIFY RHS   ///////////////////////
    
    RHSPtr rhs = RHS::rhs(); // zero RHS
    
    ////////////////////   CREATE BCs   ///////////////////////
    // cavity flow
    BCPtr bc = BC::bc();
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1.0);
    SpatialFilterPtr wallBoundary = SpatialFilter::negatedFilter(topBoundary);
    
    FunctionPtr n = Function::normal();
    
    bc->addDirichlet(u1hat, topBoundary, Function::constant(1.0));
    bc->addDirichlet(u1hat, wallBoundary, Function::zero());
    bc->addDirichlet(u2hat, wallBoundary, Function::zero());
    bc->addZeroMeanConstraint(p);
    
    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 2;
    int pToAdd = 2;
    
    // first, single-element mesh
    MeshPtr mesh;
    if (!minRule)
      mesh = MeshUtilities::buildUnitQuadMesh(1, bf, H1Order, H1Order+pToAdd);
    else
      mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAdd, 1.0, 1.0, 1, 1);
    
    ////////////////////   REFINE & SOLVE   ///////////////////////
    SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    SolutionPtr condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution->setUseCondensedSolve(true);
    
    solution->solve();
    condensedSolution->solve(false);
    condensedSolution->setUseCondensedSolve(false); // not sure if this makes a difference, or why it should (just trying something)
    FunctionPtr u1_soln = Function::solution(u1,solution);
    FunctionPtr u1_condensed_soln = Function::solution(u1,condensedSolution);
    double diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve with zero-mean constraint on single-element mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    FunctionPtr p_soln = Function::solution(p,solution);
    FunctionPtr p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      cout << "Failing test: Condensed solve pressure solution with zero-mean constraint on single-element mesh does not match regular solve" << endl;
      cout << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    double p_mean_soln = p_soln->integrate(mesh);
    TEST_COMPARE(abs(p_mean_soln), <, 1e-12);
    double p_mean_condensed_soln = p_condensed_soln->integrate(mesh);
    TEST_COMPARE(abs(p_mean_condensed_soln), <, 1e-12);
    
    int numCells = 2;
    if (!minRule)
      mesh = MeshUtilities::buildUnitQuadMesh(numCells, bf, H1Order, H1Order+pToAdd);
    else
      mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAdd, 1.0, 1.0, numCells, numCells);
    mesh->hRefine(set<GlobalIndexType>{0});
    
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    solution->solve();
    condensedSolution->condensedSolve();
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    
    if (diff>tol)
    {
      cout << "Failing test: Condensed solve with zero-mean constraint on refined (hanging-node) mesh does not match regular solve" << endl;
      cout << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      
#ifdef HAVE_EPETRAEXT_HDF5
      ostringstream dir_name;
      if (minRule)
        dir_name << "refinedMaxRuleMeshStandardVsCondensedSolve";
      else
        dir_name << "refinedMinRuleMeshStandardVsCondensedSolve";
      HDF5Exporter exporter(mesh,dir_name.str());
      VarFactoryPtr vf = bf->varFactory();
      exporter.exportSolution(solution,0);
      exporter.exportSolution(condensedSolution,1);
#endif
      success=false;
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      cout << "Failing test: Condensed solve pressure solution with zero-mean constraint on refined (hanging-node) mesh does not match regular solve" << endl;
      cout << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    p_mean_soln = p_soln->integrate(mesh);
    TEST_COMPARE(abs(p_mean_soln), <, 1e-12);
    p_mean_condensed_soln = p_condensed_soln->integrate(mesh);
    TEST_COMPARE(abs(p_mean_condensed_soln), <, 1e-12);
  }
  
  void testImportOffRankCellSolution(int spaceDim, int meshWidth, Teuchos::FancyOStream &out, bool &success)
  {
    // just want any bilinear form; we'll use Poisson
    bool useConformingTraces = true;
    int H1Order = 1;
    MeshPtr mesh = poissonUniformMesh(spaceDim, meshWidth, H1Order, useConformingTraces);
    SolutionPtr soln = Solution::solution(mesh);
    
    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
    
    int rank = Teuchos::GlobalMPISession::getRank();
    
    // set up some dummy data
    for (GlobalIndexType cellID : myCells)
    {
      Camellia::FieldContainer<double> cellDofs(mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      cellDofs.initialize((double)rank);
      soln->setLocalCoefficientsForCell(cellID, cellDofs);
    }
    
    set<GlobalIndexType> myCellsAndNeighbors;
    for (GlobalIndexType cellID : myCells)
    {
      myCellsAndNeighbors.insert(cellID);
      CellPtr cell = mesh->getTopology()->getCell(cellID);
      set<GlobalIndexType> neighbors = cell->getActiveNeighborIndices(mesh->getTopology());
      myCellsAndNeighbors.insert(neighbors.begin(), neighbors.end());
    }
    
    soln->importSolutionForOffRankCells(myCellsAndNeighbors);
    
    for (GlobalIndexType cellID : myCellsAndNeighbors)
    {
      Camellia::FieldContainer<double> cellDofs = soln->allCoefficientsForCellID(cellID, false); // false: don't warn about off-rank requests
      
      TEST_ASSERT(cellDofs.size() == mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      
      PartitionIndexType rankForCell = soln->mesh()->partitionForCellID(cellID);
      TEST_EQUALITY(cellDofs[0], rankForCell);
      for (int i=1; i<cellDofs.size(); i++)
      {
        TEST_EQUALITY(cellDofs[i-1], cellDofs[i]);
      }
    }
  }
  
  void testAddSolution(int spaceDim, int H1Order, int elementWidth, bool includeHangingNodes, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    MPIWrapper::CommWorld()->Barrier();
    bool useConformingTraces = true;
    
    MeshPtr poissonMesh;
    if (includeHangingNodes)
    {
      const int irregularity = 1;
      poissonMesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
    }
    else
    {
      poissonMesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    }
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    
    VarPtr u_hat = poissonForm.u_hat();
    FunctionPtr value = Function::constant(1.0);
    SolutionPtr solution = Solution::solution(poissonMesh);
    solution->projectOntoMesh({{u_hat->ID(), value}});
    
    SolutionPtr solutionAdded = Solution::solution(poissonMesh);
    solutionAdded->addSolution(solution, 1.0);
    
    FunctionPtr uHatSoln = Function::solution(u_hat, solution, false);
    FunctionPtr uHatSolnAdded = Function::solution(u_hat, solutionAdded, false);
    
    // sanity check: both L^2 norms are > 0
    double l2norm = uHatSoln->l2norm(poissonMesh);
    TEST_COMPARE(l2norm, >, 0);
    
    double l2normAdded = uHatSolnAdded->l2norm(poissonMesh);
    TEST_COMPARE(l2normAdded, >, 0);
    
    TEST_FLOATING_EQUALITY(l2norm, l2normAdded, tol);
    
    double err = (uHatSoln - uHatSolnAdded)->l2norm(poissonMesh);
    TEST_COMPARE(err, <, tol);
    
    solutionAdded->addSolution(solution, 1.0);
    
    err = (2 * uHatSoln - uHatSolnAdded)->l2norm(poissonMesh);
    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolution_1D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 1;
    const bool hangingNodes = false;
    double tol = 1e-14;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolution_2D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 2;
    const bool hangingNodes = false;
    double tol = 1e-14;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolution_3D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 3;
    const bool hangingNodes = false;
    double tol = 1e-14;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolutionWithHangingNodes_1D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 1;
    const bool hangingNodes = true;
    double tol = 1e-14;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolutionWithHangingNodes_2D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 2;
    const bool hangingNodes = true;
    double tol = 1e-13;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, AddSolutionWithHangingNodes_3D )
  {
    const int H1Order = 2;
    const int elementWidth = 2;
    const int spaceDim = 3;
    const bool hangingNodes = true;
    double tol = 1e-11;
    
    testAddSolution(spaceDim, H1Order, elementWidth, hangingNodes, tol, out, success);
  }
  
  // TODO: finish moving AddSolution and AddSolutionCondensed, commented out below, from DPGTests (then delete there)
  //  TEUCHOS_UNIT_TEST( Solution, AddSolution )
  //  {
  //    double weight = 3.141592;
  //    double tol = 1e-12;
  //
  //    Camellia::FieldContainer<double> expectedValuesU(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> expectedValuesSIGMA1(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> expectedValuesSIGMA2(_testPoints.dimension(0));
  //    _confusionSolution2_2x2->solutionValues(expectedValuesU, ConfusionBilinearForm::U_ID, _testPoints);
  //    _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA1, ConfusionBilinearForm::SIGMA_1_ID, _testPoints);
  //    _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA2, ConfusionBilinearForm::SIGMA_2_ID, _testPoints);
  //
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesU, weight+1.0);
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesSIGMA1, weight+1.0);
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesSIGMA2, weight+1.0);
  //
  //    _confusionSolution1_2x2->addSolution(_confusionSolution2_2x2, weight);
  //    Camellia::FieldContainer<double> valuesU(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> valuesSIGMA1(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> valuesSIGMA2(_testPoints.dimension(0));
  //
  //    _confusionSolution1_2x2->solutionValues(valuesU, ConfusionBilinearForm::U_ID, _testPoints);
  //    _confusionSolution1_2x2->solutionValues(valuesSIGMA1, ConfusionBilinearForm::SIGMA_1_ID, _testPoints);
  //    _confusionSolution1_2x2->solutionValues(valuesSIGMA2, ConfusionBilinearForm::SIGMA_2_ID, _testPoints);
  //
  //    for (int pointIndex=0; pointIndex < valuesU.size(); pointIndex++)
  //    {
  //      double diff = abs(valuesU[pointIndex] - expectedValuesU[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of U: " << expectedValuesU[pointIndex] << "; actual: " << valuesU[pointIndex] << endl;
  //      }
  //
  //      diff = abs(valuesSIGMA1[pointIndex] - expectedValuesSIGMA1[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of SIGMA1: " << expectedValuesSIGMA1[pointIndex] << "; actual: " << valuesSIGMA1[pointIndex] << endl;
  //      }
  //
  //      diff = abs(valuesSIGMA2[pointIndex] - expectedValuesSIGMA2[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of SIGMA2: " << expectedValuesSIGMA2[pointIndex] << "; actual: " << valuesSIGMA2[pointIndex] << endl;
  //      }
  //    }
  //  }
  //
  //  TEUCHOS_UNIT_TEST( Solution, AddSolutionCondensed ) // copied pretty much wholesale from DPGTests
  //  {
  //    Camellia::FieldContainer<double> quadPoints(4,2);
  //
  //    quadPoints(0,0) = 0.0; // x1
  //    quadPoints(0,1) = 0.0; // y1
  //    quadPoints(1,0) = 1.0;
  //    quadPoints(1,1) = 0.0;
  //    quadPoints(2,0) = 1.0;
  //    quadPoints(2,1) = 1.0;
  //    quadPoints(3,0) = 0.0;
  //    quadPoints(3,1) = 1.0;
  //
  //    double esigmalon = 1e-2;
  //    double beta_x = 1.0, beta_y = 1.0;
  //    ConvectionDiffusionFormulation form;
  //
  //    _confusionExactSolution = Teuchos::rcp( new ConfusionManufacturedSolution(esigmalon,beta_x,beta_y) );
  //
  //    bool useConformingTraces = true;
  //    int polyOrder = 2; // 2 is minimum for projecting QuadraticFunction exactly
  //    _poissonExactSolution =
  //    Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::POLYNOMIAL,
  //                                           polyOrder, useConformingTraces) );
  //    _poissonExactSolution->setUseSinglePointBCForPHI(false, -1); // impose zero-mean constraint
  //
  //    int H1Order = polyOrder+1;
  //    int horizontalCells = 2;
  //    int verticalCells = 2;
  //
  //    // before we hRefine, compute a solution for comparison after refinement
  //    IPPtr ipConfusion = Teuchos::rcp(new MathInnerProduct(_confusionExactSolution->bilinearForm()));
  //    IPPtr ipPoisson = Teuchos::rcp(new MathInnerProduct(_poissonExactSolution->bilinearForm()));
  //    Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _confusionExactSolution->bilinearForm(), H1Order, H1Order+1);
  //
  //    Teuchos::RCP<Mesh> poissonMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _poissonExactSolution->bilinearForm(), H1Order, H1Order+2);
  //    Teuchos::RCP<Mesh> poissonMesh1x1 = MeshFactory::buildQuadMesh(quadPoints, 1, 1, _poissonExactSolution->bilinearForm(), H1Order, H1Order+2);
  //    IPPtr poissonIp = Teuchos::rcp(new MathInnerProduct(_poissonExactSolution->bilinearForm()));
  //
  //    _confusionSolution1_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->ExactSolution::bc(), _confusionExactSolution->ExactSolution::rhs(), ipConfusion) );
  //    _confusionSolution2_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->ExactSolution::bc(), _confusionExactSolution->ExactSolution::rhs(), ipConfusion) );
  //    _poissonSolution = Teuchos::rcp( new Solution(poissonMesh, _poissonExactSolution->ExactSolution::bc(),_poissonExactSolution->ExactSolution::rhs(), ipPoisson));
  //    _poissonSolution_1x1 = Teuchos::rcp( new Solution(poissonMesh1x1, _poissonExactSolution->ExactSolution::bc(),_poissonExactSolution->ExactSolution::rhs(), ipPoisson));
  //    _poissonSolution_1x1_unsolved = Teuchos::rcp( new Solution(poissonMesh1x1, _poissonExactSolution->ExactSolution::bc(),_poissonExactSolution->ExactSolution::rhs(), ipPoisson));
  //
  //    _confusionUnsolved = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->ExactSolution::bc(), _confusionExactSolution->ExactSolution::rhs(), ipConfusion) );
  //
  //    _poissonSolution_1x1->solve();
  //    _confusionSolution1_2x2->solve();
  //    _confusionSolution2_2x2->solve();
  //    _poissonSolution->solve();
  //
  //
  //    double weight = 3.141592;
  //    double tol = 1e-12;
  //
  //    double soln2_coefficientWeight = 2.0;
  //
  //    FunctionPtr c = Function::vectorize(Function::constant(0.5), Function::constant(0.5));
  //    ConvectionFormulation convectionForm(2, c);
  //
  //    BFPtr bf = convectionForm.bf();
  //
  //    Teuchos::ParameterList pl;
  //
  //    int H1Order = 1;
  //    int pToAddTest = 2;
  //    int horizontalElements = 1;
  //    int verticalElements = 1;
  //    double width = 1.0;
  //    double height = 1.0;
  //    double x0 = 0;
  //    double y0 = 0;
  //    bool divideIntoTriangles = false;
  //
  //    pl.set("useMinRule", true);
  //    pl.set("bf",bf);
  //    pl.set("H1Order", H1Order);
  //    pl.set("delta_k", pToAddTest);
  //    pl.set("horizontalElements", horizontalElements);
  //    pl.set("verticalElements", verticalElements);
  //    pl.set("width", width);
  //    pl.set("height", height);
  //    pl.set("divideIntoTriangles", divideIntoTriangles);
  //    pl.set("x0",x0);
  //    pl.set("y0",y0);
  //
  //    MeshPtr mesh = MeshFactory::quadMesh(pl);
  //
  //    //  MeshPtr mesh = MeshFactory::quadMesh(bf, 2); // min-rule mesh, single element
  //
  //    // inflow BCs; set to x+1 and 2*y+1 for soln1.
  //    SpatialFilterPtr x_equals_0 = SpatialFilter::matchingX(0.0);
  //    SpatialFilterPtr y_equals_0 = SpatialFilter::matchingY(0.0);
  //
  //    // so that the fields scale linearly with the trace data (which are weighted by soln2_coefficientWeight),
  //    // we scale the BC and RHS data for soln2 with soln2_coefficientWeight.
  //
  //    BCPtr bc = BC::bc();
  //    BCPtr bc2 = BC::bc();
  //
  //    FunctionPtr x = Function::xn(1);
  //    FunctionPtr y = Function::yn(1);
  //
  //    FunctionPtr in_x = 2*y + 1;
  //    FunctionPtr in_y = x + 1;
  //
  //    bc->addDirichlet(convectionForm.q_n_hat(), x_equals_0, in_x);
  //    bc->addDirichlet(convectionForm.q_n_hat(), y_equals_0, in_y);
  //
  //    bc2->addDirichlet(convectionForm.q_n_hat(), x_equals_0, in_x * soln2_coefficientWeight);
  //    bc2->addDirichlet(convectionForm.q_n_hat(), y_equals_0, in_y * soln2_coefficientWeight);
  //
  //    RHSPtr rhs = RHS::rhs();
  //    RHSPtr rhs2 = RHS::rhs();
  //
  //    rhs->addTerm(convectionForm.v());
  //    rhs2->addTerm(soln2_coefficientWeight * convectionForm.v());
  //
  //    IPPtr ip = bf->graphNorm();
  //
  //    SolutionPtr soln1 = Solution::solution(mesh, bc, rhs, ip);
  //    soln1->setUseCondensedSolve(true);
  //    soln1->solve(); // to force computation of local stiffness matrices, etc.
  //    SolutionPtr soln2 = Solution::solution(mesh, bc2, rhs2, ip);
  //    soln2->setUseCondensedSolve(true);
  //    soln2->solve();
  //
  //    Teuchos::RCP< Epetra_FEVector > lhsVector1 = soln1->getLHSVector();
  //    Teuchos::RCP< Epetra_FEVector > lhsVector2 = soln2->getLHSVector();
  //
  //    // load lhsVector1 and 2 with some arbitrary data
  //
  //    if (lhsVector1->Map().NumMyElements() > 0)
  //    {
  //      for (int i=lhsVector1->Map().MinLID(); i<=lhsVector1->Map().MaxLID(); i++)
  //      {
  //        GlobalIndexType gid = lhsVector1->Map().GID(i);
  //        (*lhsVector1)[0][i] = (double) gid;
  //      }
  //    }
  //
  //    if (lhsVector2->Map().NumMyElements() > 0)
  //    {
  //      for (int i=lhsVector2->Map().MinLID(); i<=lhsVector2->Map().MaxLID(); i++)
  //      {
  //        GlobalIndexType gid = lhsVector2->Map().GID(i);
  //        (*lhsVector2)[0][i] = (double) soln2_coefficientWeight * gid;
  //      }
  //    }
  //
  //    // determine cell-local coefficients:
  //    soln1->importSolution();
  //    soln2->importSolution();
  //
  //    GlobalIndexType cellID = 0;
  //
  //    Camellia::FieldContainer<double> soln1_cell0 = soln1->allCoefficientsForCellID(cellID);
  //    Camellia::FieldContainer<double> soln2_cell0 = soln2->allCoefficientsForCellID(cellID);
  //
  //    //  cout << "soln1_cell0:\n" << soln1_cell0;
  //
  //    {
  //      // DEBUGGING: check for linear dependence of cell0 coefficients on the lhsVector coefficients
  //      Camellia::FieldContainer<double> soln1_doubled = soln1_cell0;
  //      SerialDenseWrapper::multiplyFCByWeight(soln1_doubled, soln2_coefficientWeight);
  //      double tol = 1e-14;
  //      double maxDiff = 0;
  //      if ( !TestSuite::fcsAgree(soln1_doubled, soln2_cell0, tol, maxDiff) )
  //      {
  //        cout << "Error: before calling addSolution, coefficients for soln2 aren't as expected...\n";
  //        success = false;
  //      }
  //    }
  //
  //    soln1->addSolution(soln2, weight);
  //
  //    Camellia::FieldContainer<double> actualValues = soln1->allCoefficientsForCellID(cellID);
  //    Camellia::FieldContainer<double> expectedValues = soln1_cell0;
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValues, soln2_coefficientWeight * weight + 1);
  //    double maxDiff = 0;
  //    if ( !TestSuite::fcsAgree(expectedValues, actualValues, tol, maxDiff) )
  //    {
  //      cout << "Error: after calling addSolution, actual coefficients for sum differ from expected by as much as " << maxDiff << "...\n";
  //      success = false;
  //    }
  //
  //    // now repeat, but with a different version of addSolution
  //    // first, reset soln1:
  //    {
  //      if (lhsVector1->Map().NumMyElements() > 0)
  //      {
  //        for (int i=lhsVector1->Map().MinLID(); i<=lhsVector1->Map().MaxLID(); i++)
  //        {
  //          GlobalIndexType gid = lhsVector1->Map().GID(i);
  //          (*lhsVector1)[0][i] = (double) gid;
  //        }
  //      }
  //      soln1->importSolution();
  //    }
  //    set<int> varsToAdd = mesh->getElementType(cellID)->trialOrderPtr->getVarIDs(); //simple test: apply to all varIDs
  //    soln1->addSolution(soln2, weight, varsToAdd);
  //
  //    actualValues = soln1->allCoefficientsForCellID(cellID);
  //    maxDiff = 0;
  //    if ( !TestSuite::fcsAgree(expectedValues, actualValues, tol, maxDiff) )
  //    {
  //      cout << "Error: after calling addSolution (varID filtered version), actual coefficients for sum differ from expected by as much as " << maxDiff << "...\n";
  //      success = false;
  //    }
  //
  //    weight = 1.0; // since we don't weight the rhs or the BCs below, this is require to ensure the correctness of the test...
  //
  //    _confusionSolution2_2x2->setUseCondensedSolve(true);
  //    _confusionSolution1_2x2->setUseCondensedSolve(true);
  //
  //    _confusionSolution1_2x2->solve();
  //    _confusionSolution2_2x2->solve();
  //
  //    Camellia::FieldContainer<double> expectedValuesU(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> expectedValuesSIGMA1(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> expectedValuesSIGMA2(_testPoints.dimension(0));
  //    _confusionSolution2_2x2->solutionValues(expectedValuesU, ConfusionBilinearForm::U_ID, _testPoints);
  //    _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA1, ConfusionBilinearForm::SIGMA_1_ID, _testPoints);
  //    _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA2, ConfusionBilinearForm::SIGMA_2_ID, _testPoints);
  //
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesU, weight+1.0);
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesSIGMA1, weight+1.0);
  //    SerialDenseWrapper::multiplyFCByWeight(expectedValuesSIGMA2, weight+1.0);
  //
  //    Teuchos::RCP< Epetra_FEVector > vector1_copy = Teuchos::rcp( new Epetra_FEVector(*_confusionSolution1_2x2->getLHSVector().get()) );
  //    Teuchos::RCP< Epetra_FEVector > vector2 = _confusionSolution2_2x2->getLHSVector();
  //
  //    map<GlobalIndexType, Camellia::FieldContainer<double> > cellCoefficientsForRank;
  //    set<GlobalIndexType> rankLocalCells = _confusionSolution1_2x2->mesh()->cellIDsInPartition();
  //
  //    for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  //    {
  //      GlobalIndexType cellID = *cellIDIt;
  //      Camellia::FieldContainer<double> coefficients = _confusionSolution1_2x2->allCoefficientsForCellID(cellID);
  //      cellCoefficientsForRank[cellID] = coefficients;
  //    }
  //
  //    //  cout << "local coefficients, cell 0:\n";
  //    //  GlobalIndexType cellID = 0;
  //    //  cout << _confusionSolution1_2x2->allCoefficientsForCellID(cellID);
  //
  //    //  cout << "vector 1:\n";
  //    //  for (int i = vector1_copy->Map().MinLID(); i <= vector1_copy->Map().MaxLID(); i++) {
  //    //    cout << vector1_copy->Map().GID(i) << ": " << vector1_copy->Values()[i] << endl;
  //    //  }
  //
  //    //  cout << "vector 2:\n";
  //    //  for (int i = vector2->Map().MinLID(); i <= vector2->Map().MaxLID(); i++) {
  //    //    cout << vector2->Map().GID(i) << ": " << vector2->Values()[i] << endl;
  //    //  }
  //
  //    _confusionSolution1_2x2->addSolution(_confusionSolution2_2x2, weight);
  //
  //    // check that the cell-local coefficients are as expected (multiplied by weight + 1)
  //
  //    for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++)
  //    {
  //      GlobalIndexType cellID = *cellIDIt;
  //      Camellia::FieldContainer<double> actualCoefficients = _confusionSolution1_2x2->allCoefficientsForCellID(cellID);
  //      Camellia::FieldContainer<double> expectedCoefficients = cellCoefficientsForRank[cellID];
  //      SerialDenseWrapper::multiplyFCByWeight(expectedCoefficients, weight + 1.0);
  //      double maxDiff = 0;
  //      if (! TestSuite::fcsAgree(expectedCoefficients, actualCoefficients, tol, maxDiff) )
  //      {
  //        cout << "Error: expected coefficients for cell ID " << cellID << " differ from actual by " << maxDiff << endl;
  //
  //        cout << "expectedCoefficients:\n" << expectedCoefficients;
  //        cout << "actualCoefficients:\n" << actualCoefficients;
  //        success = false;
  //      }
  //    }
  //
  //    //  cout << "local coefficients, cell 0, after summing:\n";
  //    //  cout << _confusionSolution1_2x2->allCoefficientsForCellID(cellID);
  //
  //    Teuchos::RCP< Epetra_FEVector > lhsVector = _confusionSolution1_2x2->getLHSVector();
  //
  //    //  cout << "weighted sum:\n";
  //    //  for (int i = lhsVector->Map().MinLID(); i <= lhsVector->Map().MaxLID(); i++) {
  //    //    cout << lhsVector->Map().GID(i) << ": " << lhsVector->Values()[i] << endl;
  //    //  }
  //
  //    Camellia::FieldContainer<double> valuesU(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> valuesSIGMA1(_testPoints.dimension(0));
  //    Camellia::FieldContainer<double> valuesSIGMA2(_testPoints.dimension(0));
  //
  //    _confusionSolution1_2x2->solutionValues(valuesU, ConfusionBilinearForm::U_ID, _testPoints);
  //    _confusionSolution1_2x2->solutionValues(valuesSIGMA1, ConfusionBilinearForm::SIGMA_1_ID, _testPoints);
  //    _confusionSolution1_2x2->solutionValues(valuesSIGMA2, ConfusionBilinearForm::SIGMA_2_ID, _testPoints);
  //
  //    for (int pointIndex=0; pointIndex < valuesU.size(); pointIndex++)
  //    {
  //      double diff = abs(valuesU[pointIndex] - expectedValuesU[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of U: " << expectedValuesU[pointIndex] << "; actual: " << valuesU[pointIndex] << endl;
  //      }
  //
  //      diff = abs(valuesSIGMA1[pointIndex] - expectedValuesSIGMA1[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of SIGMA1: " << expectedValuesSIGMA1[pointIndex] << "; actual: " << valuesSIGMA1[pointIndex] << endl;
  //      }
  //
  //      diff = abs(valuesSIGMA2[pointIndex] - expectedValuesSIGMA2[pointIndex]);
  //      if (diff > tol)
  //      {
  //        success = false;
  //        cout << "expected value of SIGMA2: " << expectedValuesSIGMA2[pointIndex] << "; actual: " << valuesSIGMA2[pointIndex] << endl;
  //      }
  //    }
  //  }
  
  TEUCHOS_UNIT_TEST( Solution, CondensedSolve )
  {
    // very simple problem: Poisson with unit load and zero BCs
    int spaceDim = 1;
    bool conformingTraces = false;
    PoissonFormulation form(spaceDim, conformingTraces);
    double xLeft = 0, xRight = 1;
    
    int numCells = 2;
    int H1Order = 3, delta_k = 1;
    MeshPtr mesh = MeshFactory::intervalMesh(form.bf(), xLeft, xRight, numCells, H1Order, delta_k);
    
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(1.0 * form.tau());
    
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), Function::zero());
    
    SolutionPtr soln = Solution::solution(mesh,bc,rhs,form.bf()->graphNorm());
    SolutionPtr solnCondensed = Solution::solution(mesh,bc,rhs,form.bf()->graphNorm());
    solnCondensed->setUseCondensedSolve(true);
    
    soln->solve();
    solnCondensed->solve();
    
    FunctionPtr u = Function::solution(form.u(), soln);
    FunctionPtr uCondensed = Function::solution(form.u(), solnCondensed);
    
    double diff_l2 = (u - uCondensed)->l2norm(mesh);
    double tol = 1e-14;
    TEST_COMPARE(diff_l2, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( Solution, CondensedSolveWithPointConstraint_Slow )
  {
    /*
     This test copied over, more or less wholesale, from the legacy DPGTests::SolutionTests.
     It's not particularly granular; a refactoring to split into several tests might be useful.
     */
    double tol = 1e-10;
    vector<double> pointForImposition = {0.0,0.0};
    
    int rank = Teuchos::GlobalMPISession::getRank();
    
    int spaceDim = 2;
    bool conformingTraces = false; // false mostly because I want to do cavity flow with non-H^1 BCs
    double mu = 1.0;
    StokesVGPFormulation stokesForm = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    
    VarPtr u1 = stokesForm.u(1);
    VarPtr u2 = stokesForm.u(2);
    VarPtr p = stokesForm.p();
    
    VarPtr u1hat = stokesForm.u_hat(1);
    VarPtr u2hat = stokesForm.u_hat(2);
    
    BFPtr bf = stokesForm.bf();
    
    // robust test norm
    IPPtr ip = bf->graphNorm();
    
    ////////////////////   SPECIFY RHS   ///////////////////////
    
    RHSPtr rhs = RHS::rhs(); // zero RHS
    
    ////////////////////   BUILD MESH   ///////////////////////
    
    int H1Order = 2;
    int pToAdd = 2;
    
    // first, single-element mesh
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order);
    
    ////////////////////   CREATE BCs   ///////////////////////
    // cavity flow
    BCPtr bc = BC::bc();
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1.0);
    SpatialFilterPtr wallBoundary = SpatialFilter::negatedFilter(topBoundary);
    
    FunctionPtr n = Function::normal();
    
    bc->addDirichlet(u1hat, topBoundary, Function::constant(1.0));
    bc->addDirichlet(u1hat, wallBoundary, Function::zero());
    bc->addDirichlet(u2hat, wallBoundary, Function::zero());
    bc->addSpatialPointBC(p->ID(), 0.0, pointForImposition);
    
    ////////////////////   REFINE & SOLVE   ///////////////////////
    SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    SolutionPtr condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    
    //  condensedSolution->setWriteMatrixToFile(true, "/tmp/condensed_legacy_single_element_max_rule.dat");
    //  condensedSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/rhs_legacy.dat");
    
    solution->solve(false);
    condensedSolution->condensedSolve();
    
    //  out << "legacy interface, coefficients for cell 0:\n" << condensedSolution->allCoefficientsForCellID(0);
    //  out << "legacy interface, coefficients for cell 0 (uncondensed solve):\n" << solution->allCoefficientsForCellID(0);
    
    FunctionPtr u1_soln = Function::solution(u1,solution);
    FunctionPtr u1_condensed_soln = Function::solution(u1,condensedSolution);
    double diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on single-element max rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    FunctionPtr p_soln = Function::solution(p,solution);
    FunctionPtr p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on single-element max rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    
    // repeat, but now with the newer interface for condensed solve:
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution->setUseCondensedSolve(true);
    
    //  condensedSolution->setWriteMatrixToFile(true, "/tmp/condensed_new_interface_single_element_max_rule.dat");
    //  condensedSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/rhs_new.dat");
    
    solution->solve(false);
    condensedSolution->solve(false);
    
    //  out << "new interface, coefficients for cell 0:\n" << condensedSolution->allCoefficientsForCellID(0);
    //  out << "new interface, coefficients for cell 0 (uncondensed solve):\n" << solution->allCoefficientsForCellID(0);
    
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on single-element max rule mesh does not match regular solve";
      out << " when using newer setUseCondensedSolve() method." << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on single-element max rule mesh does not match regular solve";
      out << " when using newer setUseCondensedSolve() method." << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    
    // now, same thing, but with a single-element minimum-rule mesh:
    mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAdd, 1.0, 1.0, 1, 1);
    
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    //  condensedSolution->setUseCondensedSolve(true);
    
    solution->solve(false);
    condensedSolution->condensedSolve();
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff>tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on single-element min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on single-element min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    
    int numCells = 2;
    // MAX RULE, multi-element refined mesh
    mesh = MeshFactory::quadMesh(bf, H1Order);
    
    set<GlobalIndexType> cell0;
    cell0.insert(0);
    mesh->hRefine(cell0, RefinementPattern::regularRefinementPatternQuad());
    
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    solution->solve(false);
    condensedSolution->condensedSolve();
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    
    if (diff>tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on refined max rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on refined max rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    
    // MIN RULE, multi-element compatible mesh
    mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAdd, 1.0, 1.0, numCells, numCells);
    
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    solution->solve(false);
    condensedSolution->condensedSolve();
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff>tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on multi-element (compatible) min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
#ifdef HAVE_EPETRAEXT_HDF5
      ostringstream dir_name;
      dir_name << "multiElementMinRuleMeshStandardVsCondensedSolve";
      HDF5Exporter exporter(mesh,dir_name.str());
      VarFactoryPtr vf = bf->varFactory();
      exporter.exportSolution(solution,vf,0);
      exporter.exportSolution(condensedSolution,vf,1);
#endif
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on multi-element (compatible) min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
    
    // MIN RULE, multi-element refined mesh
    mesh->hRefine(cell0, RefinementPattern::regularRefinementPatternQuad());
    
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    condensedSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    solution->solve(false);
    condensedSolution->condensedSolve();
    u1_soln = Function::solution(u1,solution);
    u1_condensed_soln = Function::solution(u1,condensedSolution);
    diff = (u1_soln-u1_condensed_soln)->l2norm(mesh,H1Order);
    if (diff>tol)
    {
      out << "Failing test: Condensed solve with single-point constraint on refined min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
      int cellID = 6;
      Camellia::FieldContainer<double> cell6coeffs_standard = solution->allCoefficientsForCellID(cellID, false); // false: don't warn if off-rank
      Camellia::FieldContainer<double> cell6coeffs_condensed = condensedSolution->allCoefficientsForCellID(cellID, false); // false: don't warn if off-rank
      if (rank==0)
      {
        out << "cell " << cellID << ", standard solution coefficients:\n" << cell6coeffs_standard;
        out << "cell " << cellID << ", condensed solution coefficients:\n" << cell6coeffs_condensed;
      }
    }
    p_soln = Function::solution(p,solution);
    p_condensed_soln = Function::solution(p,condensedSolution);
    
    int vertexDim = 0;
    IndexType pointForImpositionEntityIndex;
    bool vertexFound = mesh->getTopology()->getVertexIndex(pointForImposition, pointForImpositionEntityIndex);
    
    // make sure at least someone found the vertex
    int myFoundVertex = vertexFound ? 1 : 0;
    int globalFoundVertex;
    mesh->Comm()->SumAll(&myFoundVertex, &globalFoundVertex, 1);
    TEST_ASSERT(globalFoundVertex > 0);
    
    vector<pair<IndexType,unsigned>> cellsForVertex;
    if (vertexFound)
    {
      cellsForVertex = mesh->getTopology()->getActiveCellIndices(vertexDim, pointForImpositionEntityIndex);
    }
    
    const set<GlobalIndexType>* myCellIDs = &mesh->cellIDsInPartition();
    for (pair<IndexType,unsigned> cellEntry : cellsForVertex)
    {
      GlobalIndexType cellID = cellEntry.first;
      int vertexOrdinal = cellEntry.second;
      if (myCellIDs->find(cellID) != myCellIDs->end())
      {
        CellPtr cell = mesh->getTopology()->getCell(cellID);
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
        Camellia::FieldContainer<double> refCellPoints(cell->topology()->getVertexCount(),cell->topology()->getDimension());
        CamelliaCellTools::refCellNodesForTopology(refCellPoints, cell->topology());
        Camellia::FieldContainer<double> refPoint(1,cell->topology()->getDimension());
        refPoint(0,0) = refCellPoints(vertexOrdinal,0);
        refPoint(0,1) = refCellPoints(vertexOrdinal,1);
        basisCache->setRefCellPointsFC(refPoint);
        Camellia::FieldContainer<double> solnValue(1,1);
        Camellia::FieldContainer<double> condensedSolnValue(1,1);
        p_soln->values(solnValue, basisCache);
        p_condensed_soln->values(condensedSolnValue, basisCache);
        // both values should be exactly zero; we don't require quite that, just an extremely tight tolerance
        TEST_COMPARE(abs(solnValue[0]), <, 1e-15);
        TEST_COMPARE(abs(condensedSolnValue[0]), <, 1e-15);
      }
    }
    
    diff = (p_soln-p_condensed_soln)->l2norm(mesh,H1Order);
    if (diff > tol)
    {
      out << "Failing test: Condensed solve pressure solution with single-point constraint on refined min rule mesh does not match regular solve" << endl;
      out << "L2 norm of difference is " << diff << "; tol is " << tol << endl;
      success=false;
    }
  }
  
  TEUCHOS_UNIT_TEST( Solution, CondensedSolveWithZeroMeanConstraintMaxRule_Slow)
  {
    bool minRule = false;
    testCondensedSolveZeroMeanConstraint(minRule, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, CondensedSolveWithZeroMeanConstraintMinRule_Slow)
  {
    bool minRule = true;
    testCondensedSolveZeroMeanConstraint(minRule, out, success);
  }
  
  void testERKTableauConvergenceRate(ERKTableauPtr tableau, const double dt_initial, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     In this test, we solve an exponential decay problem
     u' = - lambda u
     which has solution
     u(t) = u_0 e^(-lambda t)
     using an ERKTableau, and we check that this has the right convergence rate in the time step.
     
     Because the ERKTableau solve is embedded within Solution, we do need to set up a variational
     formulation.  Because we are only concerned with temporal convergence here, we can manage
     with a very simple formulation: one constant variable, one 1D element.
     */
    const double rateTolerance = 1e-2; // look for rates in range expectedRate ± rateTolerance
    const double errTolerance = 1e-18; // for linear term check; will loosen later
    const double t_final    = 10 * dt_initial;
    SolutionStagePtr solnStage = SolutionStage::solutionStage(tableau, dt_initial);
    
    const double lambda  = 1.0;
    const double u0      = 1.0;
    const double u_final = u0 * exp(-lambda * t_final);
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto u = vf->fieldVar("u");
    auto v = vf->testVar("v", L2);
    
    FunctionPtr dt = solnStage->dt();
    
    BFPtr bf = BF::bf(vf);
    bf->addTerm(u / dt, v);
    
    const double xLeft = 0.0;
    const double xRight = 1.0;
    const int H1Order = 1; // --> constant L^2 variables
    const int numElements = 1;
    const int delta_k = 0; // Bubnov-Galerkin
    
    auto mesh = MeshFactory::intervalMesh(bf, xLeft, xRight, numElements, H1Order, delta_k);
    auto soln = Solution::solution(bf, mesh);
    auto solnWeakPtr = Teuchos::rcp(soln.get(), false); // avoid circular strong reference
    
    const bool weightFluxesBySideParity = false; // not applicable here
    const int solutionStage = -1; // the "current" previous solution
    auto u_prev = Function::solution(u, solnWeakPtr, weightFluxesBySideParity, STANDARD_SOLUTION, solutionStage);
    
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm((1/dt) * u_prev * v - lambda * u_prev * v); // soln will end up adding a -(1/dt) * u_prev * v term
    
    soln->setRHS(rhs);
    soln->setSolutionStage(solnStage); // soln will modify the RHS, adding -(1/dt) * u_prev * v term
    
    soln->setBC(BC::bc());
    
    std::map<int, FunctionPtr> initialSoln;
    initialSoln[u->ID()] = Function::constant(u0);
    soln->projectOntoMesh(initialSoln);
    
    // let's confirm that the RHS is as we expect it to be; should be equal to
    //  - lambda * u_prev * v
    LinearTermPtr expectedRHSTerm = - lambda * u_prev * v;
    LinearTermPtr actualRHSTerm = soln->rhs()->linearTerm();
    LinearTermPtr errTerm = actualRHSTerm - expectedRHSTerm;
    IPPtr testIP = IP::ip();
    testIP->addTerm(v);
    double err = errTerm->computeNorm(testIP, mesh);
    
    TEST_COMPARE(err, <, errTolerance);
    
    while (solnStage->getTime() < t_final)
    {
      soln->solve();
    }
    
    double err_initial = (u_final - u_prev)->l2norm(mesh);
    out << "err_initial = " << err_initial << std::endl;
    
    // compute again, this time with time step cut in half:
    solnStage->setTime(0);
    soln->projectOntoMesh(initialSoln);
    solnStage->setTimeStep(dt_initial/2.0);
    while (solnStage->getTime() < t_final)
    {
      soln->solve();
    }
    
    double err_final = (u_final - u_prev)->l2norm(mesh);
    out << "err_final = " << err_final << std::endl;
    
    double err_rate = std::log(err_initial / err_final) / std::log(2.0);
    out << "err_rate = " << err_rate << std::endl;
    
    double expectedRate = tableau->expectedRate();
    
    bool rateMatches = (err_rate < expectedRate + rateTolerance) && (err_rate > expectedRate - rateTolerance);
    if (!rateMatches)
    {
      success = false;
      out << "FAILURE: actual rate of " << err_rate << " is not within " << rateTolerance << " of expected rate of ";
      out << expectedRate << std::endl;
    }
  }
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauConvergenceRateForwardEuler )
  {
    ERKTableauPtr tableau = ERKTableau::forwardEuler();
    const double dt_initial = .001;
    testERKTableauConvergenceRate(tableau, dt_initial, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauConvergenceRateHeun )
  {
    ERKTableauPtr tableau = ERKTableau::heun();
    const double dt_initial = .001;
    testERKTableauConvergenceRate(tableau, dt_initial, out, success);
  }
  
  /*
   NOTE: we do not include a test for the rate of RK4 because the rate test
         above for dt_initial > .001 appears to be in the pre-asymptotic regime
         (rates are higher than expected), while dt_initial <= .001 gives
         err of 0 to machine precision, so that rates are nan or -inf.
         There is likely a better way to test the rate of RK4.  For now, we
         make do with the single-step test below.
   */
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauConvergenceRateSSPRK3 )
  {
    ERKTableauPtr tableau = ERKTableau::ssprk3();
    const double dt_initial = .001;
    testERKTableauConvergenceRate(tableau, dt_initial, out, success);
  }
  
  void testERKTableauSingleStep(ERKTableauPtr tableau, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     In this test, we solve an exponential decay problem
     u' = - lambda u
     which has solution
     u(t) = u_0 e^(-lambda t)
     using an ERKTableau, and we check that solution after the first step matches the expected value
     (computed manually).
     
     Because the ERKTableau solve is embedded within Solution, we do need to set up a variational
     formulation.  Because we are only concerned with temporal convergence here, we can manage
     with a very simple formulation: one constant variable, one 1D element.
     */
    const double relTolerance = 1e-16;
    const double dtValue = 0.01;
    SolutionStagePtr solnStage = SolutionStage::solutionStage(tableau, dtValue);
    
    const double lambda  = 1.0;
    const double u0      = 1.0;
    
    using std::vector;
    vector<double> k(tableau->numStages());
    for (int stageOrdinal=1; stageOrdinal<=tableau->numStages(); stageOrdinal++)
    {
      const int i = stageOrdinal-1;
      double u_for_stage = u0;
      for (int j=0; j<i; j++)
      {
        const double a_ij = tableau->a(i, j);
        u_for_stage += dtValue * a_ij * k[j];
      }
      out << "u for stage " << stageOrdinal << ": " << u_for_stage << std::endl;
      k[i] = - lambda * u_for_stage;
    }
    
    double u_expected = u0;
    for (int stageOrdinal=1; stageOrdinal<=tableau->numStages(); stageOrdinal++)
    {
      const int i=stageOrdinal-1;
      const double b_i = tableau->b(i);
      u_expected += dtValue * b_i * k[i];
    }
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto u = vf->fieldVar("u");
    auto v = vf->testVar("v", L2);
    
    FunctionPtr dt = solnStage->dt();
    
    BFPtr bf = BF::bf(vf);
    bf->addTerm(u / dt, v);
    
    const double xLeft = 0.0;
    const double xRight = 1.0;
    const int H1Order = 1; // --> constant L^2 variables
    const int numElements = 1;
    const int delta_k = 0; // Bubnov-Galerkin
    
    auto mesh = MeshFactory::intervalMesh(bf, xLeft, xRight, numElements, H1Order, delta_k);
    auto soln = Solution::solution(bf, mesh);
    auto solnWeakPtr = Teuchos::rcp(soln.get(), false); // avoid circular strong reference
    
    const bool weightFluxesBySideParity = false; // not applicable here
    const int solutionStage = -1; // the "current" previous solution
    auto u_prev = Function::solution(u, solnWeakPtr, weightFluxesBySideParity, STANDARD_SOLUTION, solutionStage);
    
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm((1./dt) * u_prev * v - lambda * u_prev * v); // soln will end up adding a -(1/dt) * u_prev * v term
    
    soln->setRHS(rhs);
    soln->setSolutionStage(solnStage); // soln will modify the RHS, adding -(1/dt) * u_prev * v term
    
    soln->setBC(BC::bc());
    
    std::map<int, FunctionPtr> initialSoln;
    initialSoln[u->ID()] = Function::constant(u0);
    soln->projectOntoMesh(initialSoln);
    
    // single step:
    soln->solve();
    
    // single, constant element of width 1: the value is the same as the integral
    double u_value = u_prev->integrate(mesh);
    out << "actual u   = " << u_value << std::endl;
    out << "expected u = " << u_expected << std::endl;
    
    double err = (u_expected - u_prev)->l2norm(mesh);
    out << "err = " << err << std::endl;
    
    double relErr = u_expected * err;
    out << "relErr = " << relErr << std::endl;
    TEST_COMPARE(relErr, <, relTolerance);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauSingleStepForwardEuler )
  {
    ERKTableauPtr tableau = ERKTableau::forwardEuler();
    testERKTableauSingleStep(tableau, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauSingleStepHeun )
  {
    ERKTableauPtr tableau = ERKTableau::heun();
    testERKTableauSingleStep(tableau, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ERKTableauSingleStepRK4 )
  {
    ERKTableauPtr tableau = ERKTableau::rk4();
    testERKTableauSingleStep(tableau, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ERKTableauSingleStepSSPRK3 )
  {
    ERKTableauPtr tableau = ERKTableau::ssprk3();
    testERKTableauSingleStep(tableau, out, success);
  }

  void testERKTableauSolutionValues(ERKTableauPtr tableau, Teuchos::FancyOStream &out, bool &success)
  {
    // no solve here; we just load up the LHS vector with arbitrary data and check that solutionValues
    // returns the appropriately weighted result
    const double relTolerance = 1e-16;
    
    const double dtValue = 0.01;
    SolutionStagePtr solnStage = SolutionStage::solutionStage(tableau, dtValue);
    
    const double lambda  = 1.0;
    const double u0      = 1.0;
    
    using std::vector;
    vector<double> k(tableau->numStages());
    vector<double> solnForStageOrdinal(1+tableau->numStages()); // what actually is stored in solution vector
    solnForStageOrdinal[0] = u0;
    vector<double> uForStageOrdinal(1+tableau->numStages()); // the "stage solution" for the stage ordinal: u0 + …
    uForStageOrdinal[0] = u0;
    for (int stageOrdinal=1; stageOrdinal<=tableau->numStages(); stageOrdinal++)
    {
      const int i = stageOrdinal-1;
      double u_for_stage = u0;
      for (int j=0; j<i; j++)
      {
        const double a_ij = tableau->a(i, j);
        u_for_stage += a_ij * solnForStageOrdinal[j+1];
      }
      uForStageOrdinal[stageOrdinal] = u_for_stage;
      out << "u for stage " << stageOrdinal << ": " << u_for_stage << std::endl;
      k[i] = - lambda * u_for_stage;
      solnForStageOrdinal[stageOrdinal] = dtValue * k[i];
    }
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto u = vf->fieldVar("u");
    auto v = vf->testVar("v", L2);
    
    FunctionPtr dt = solnStage->dt();
    
    BFPtr bf = BF::bf(vf);
    bf->addTerm(u / dt, v);
    
    const double xLeft = 0.0;
    const double xRight = 1.0;
    const int H1Order = 1; // --> constant L^2 variables
    const int numElements = 1;
    const int delta_k = 0; // Bubnov-Galerkin
    
    auto mesh = MeshFactory::intervalMesh(bf, xLeft, xRight, numElements, H1Order, delta_k);
    auto soln = Solution::solution(bf, mesh);
    auto solnWeakPtr = Teuchos::rcp(soln.get(), false); // avoid circular strong reference
    
    const bool weightFluxesBySideParity = false; // not applicable here
    const int solutionStage = -1; // the "current" previous solution
    auto u_prev = Function::solution(u, solnWeakPtr, weightFluxesBySideParity, STANDARD_SOLUTION, solutionStage);
    
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm((1./dt) * u_prev * v - lambda * u_prev * v); // soln will end up adding a -(1/dt) * u_prev * v term
    
    soln->setRHS(rhs);
    soln->setSolutionStage(solnStage); // soln will modify the RHS, adding -(1/dt) * u_prev * v term
    
    std::map<int, FunctionPtr> solnValues;
    const int numSolnForStageOrdinalEntries = solnForStageOrdinal.size();
    for (int i=0; i<numSolnForStageOrdinalEntries; i++)
    {
      solnValues[u->ID()] = Function::constant(solnForStageOrdinal[i]);
      soln->projectOntoMesh(solnValues,i);
    }

    for (int stageOrdinal=0; stageOrdinal<numSolnForStageOrdinalEntries; stageOrdinal++)
    {
      out << "Testing stage " << stageOrdinal << std::endl;
      auto u_prev = Function::solution(u, solnWeakPtr, weightFluxesBySideParity, STANDARD_SOLUTION, stageOrdinal);
      // single, constant element of width 1: the value is the same as the integral
      const double u_value    = u_prev->integrate(mesh);
      const double u_expected = uForStageOrdinal[stageOrdinal];
      out << "actual u   = " << u_value << std::endl;
      out << "expected u = " << u_expected << std::endl;
      
      double err = (u_expected - u_prev)->l2norm(mesh);
      out << "err = " << err << std::endl;
      
      double relErr = u_expected * err;
      out << "relErr = " << relErr << std::endl;
      TEST_COMPARE(relErr, <, relTolerance);
    }
  }

  TEUCHOS_UNIT_TEST( Solution, ERKTableauSolutionValues_Heun )
  {
    ERKTableauPtr tableau = ERKTableau::heun();
    testERKTableauSolutionValues(tableau, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ERKTableauSolutionValues_SSPRK3 )
  {
    ERKTableauPtr tableau = ERKTableau::ssprk3();
    testERKTableauSolutionValues(tableau, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ImportOffRankCellData_1D )
  {
    int spaceDim = 1;
    int meshWidth = 4;
    
    testImportOffRankCellSolution(spaceDim, meshWidth, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ImportOffRankCellData_2D )
  {
    int spaceDim = 2;
    int meshWidth = 2;
    
    testImportOffRankCellSolution(spaceDim, meshWidth, out, success);
  }
  
  
  TEUCHOS_UNIT_TEST( Solution, ImportOffRankCellData_3D )
  {
    int spaceDim = 3;
    int meshWidth = 2;
    
    testImportOffRankCellSolution(spaceDim, meshWidth, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ImposeBCs )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useTriangles = true;
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::ULTRAWEAK);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {2,1};
    
    int H1Order = 1, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    double prescribedValue = 1.0;
    BCPtr bc = BC::bc();
    VarPtr u_hat = form.u_hat();
    bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::constant(prescribedValue));
    SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, RHS::rhs(), form.bf()->graphNorm());
    soln->initializeLHSVector();
    soln->initializeStiffnessAndLoad();
    
    soln->populateStiffnessAndLoad();
    
    // we prescribe 1 at at all the bc indices, and leave the solution coefficients as 0 everywhere else.
    // then we check the values on the elements.
    
    soln->imposeBCs();
    soln->importSolution();
    // for each of my cells, find vertices on the mesh boundary.
    set<GlobalIndexType> myCellIDs = mesh->cellIDsInPartition();
    int numPoints = 4;
    Camellia::FieldContainer<double> refLinePoints(numPoints,1);
    refLinePoints(0,0) = -1.0;
    refLinePoints(1,0) = -0.5;
    refLinePoints(2,0) =  0.5;
    refLinePoints(3,0) =  1.0;
    
    double tol = 1e-14;
    FunctionPtr u_soln = Function::solution(u_hat, soln);
    Camellia::FieldContainer<double> uValues(1,numPoints);
    for (GlobalIndexType cellID : myCellIDs)
    {
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
      out << "cell " << cellID << ", nonzero coefficients:\n";
      printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, soln->allCoefficientsForCellID(cellID));
      
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      int sideCount = basisCache->cellTopology()->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
        sideCache->setRefCellPointsFC(refLinePoints);
        u_soln->values(uValues, sideCache);
        
        // are there some physical points that lie on the boundary?
        // we expect to match prescribedValue at these
        const Camellia::FieldContainer<double>* physicalPoints = &sideCache->getPhysicalCubaturePointsFC();
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          double x = (*physicalPoints)(0,pointOrdinal,0);
          double y = (*physicalPoints)(0,pointOrdinal,1);
          if ((abs(x-0) < tol) || (abs(x-1) < tol) || (abs(y-0) < tol) || (abs(y-1) < tol))
          {
            double actualValue = uValues(0,pointOrdinal);
            out << "\n\ntesting value for (" << x << "," << y << ") on cell " << cellID << ", side " << sideOrdinal << endl;
            TEST_FLOATING_EQUALITY(actualValue, prescribedValue, tol);
          }
        }
      }
    }
  }
  
  void testProjectTraceOnTensorMesh(CellTopoPtr spaceTopo, int H1Order, FunctionPtr f, VarType traceOrFlux,
                                    Teuchos::FancyOStream &out, bool &success)
  {
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), spaceTopo->getTensorialDegree() + 1);
    
    // very simply, take a one-element, reference space mesh, project a polynomial onto a trace variable,
    // and check whether we correctly project a function onto it...
    
    // define a VarFactory with just a trace variable, and an HGRAD test
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr v = vf->testVarSpaceTime("v", HGRAD, HGRAD);
    VarPtr uhat;
    if (traceOrFlux == TRACE)
      uhat = vf->traceVarSpaceTime("uhat");
    else if (traceOrFlux == FLUX)
      uhat = vf->fluxVarSpaceTime("u_n");
    
    BFPtr bf = BF::bf(vf);
    
    vector< vector<double> > refCellNodes;
    CamelliaCellTools::refCellNodesForTopology(refCellNodes,spaceTimeTopo);
    
    int spaceDim = spaceTimeTopo->getDimension();
    int pToAdd = 1; // for this test, doesn't really affect much
    
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
    meshTopo->addCell(spaceTimeTopo, refCellNodes);
    
    MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopo, bf, H1Order, pToAdd) );
    mesh->setTensorTopologyMeansSpaceTime(true);
    
    SolutionPtr soln = Solution::solution(mesh);
    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = f;
    
    soln->projectOntoMesh(functionMap);
    
    // Now, manually project onto the basis for the trace to compute some expected coefficients
    Camellia::FieldContainer<double> basisCoefficientsExpected;
    
    double tol = 1e-15;
    
    const set<GlobalIndexType> & cellIDs = mesh->cellIDsInPartition();
    
    for (const auto &cellID : cellIDs)
    {
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
      
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

      const int spaceTimeSideCount = spaceTimeTopo->getSideCount();
      for (int sideOrdinal = 0; sideOrdinal < spaceTimeSideCount; sideOrdinal++)
      {
        CellTopoPtr sideTopo = spaceTimeTopo->getSide(sideOrdinal);
        BasisPtr sideBasis = trialOrder->getBasis(uhat->ID(), sideOrdinal);
        BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
        
        int numCells = 1;
        basisCoefficientsExpected.resize(numCells,sideBasis->getCardinality());
        
        Projector<double>::projectFunctionOntoBasis(basisCoefficientsExpected, f, sideBasis, sideBasisCache);
        
        Camellia::FieldContainer<double> basisCoefficientsActual(sideBasis->getCardinality());
        
        soln->solnCoeffsForCellID(basisCoefficientsActual,cellID,uhat->ID(),sideOrdinal);
        
        for (int basisOrdinal=0; basisOrdinal < sideBasis->getCardinality(); basisOrdinal++)
        {
          double diff = basisCoefficientsActual[basisOrdinal] - basisCoefficientsExpected[basisOrdinal];
          TEST_COMPARE(abs(diff),<,tol);
        }
      }
      //      { // DEBUGGING:
      //        cout << "CellID " << cellID << " info:\n";
      //        Camellia::FieldContainer<double> localCoefficients = soln->allCoefficientsForCellID(cellID);
      //        Camellia::printLabeledDofCoefficients(vf, trialOrder, localCoefficients);
      //      }
    }
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectTrace_1D)
  {
    const double tol = 1e-15;
    const int spaceDim = 1;
    const int elementWidth = 2; // use two elements so that there is one shared trace dof (hardens test slightly)
    const int H1Order = 1;
    bool useConformingTraces = true; // does not matter in 1D
    MeshPtr poissonMesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    
    VarPtr u_hat = poissonForm.u_hat();
    const double constantValue = 1.0;
    FunctionPtr value = Function::constant(constantValue);
    SolutionPtr solution = Solution::solution(poissonMesh);
    const int solutionOrdinal = 0;
    solution->projectOntoMesh({{u_hat->ID(), value}}, solutionOrdinal);
    
    double expectedCoefficientValue = constantValue; // constant basis at points; always has value 1.0
    const auto & myCellIDs = poissonMesh->cellIDsInPartition();
    for (auto cellID : myCellIDs)
    {
      auto trialDofOrdering  = poissonMesh->getElementType(cellID)->trialOrderPtr;
      auto leftSideDofIndex  = trialDofOrdering->getDofIndex(u_hat->ID(), 0, 0); // dofOrdinal on side, sideOrdinal
      auto rightSideDofIndex = trialDofOrdering->getDofIndex(u_hat->ID(), 0, 1); // dofOrdinal on side, sideOrdinal
      auto cellCoefficients  = solution->allCoefficientsForCellID(cellID);
      auto leftCoefficient   = cellCoefficients(leftSideDofIndex);
      auto rightCoefficient  = cellCoefficients(rightSideDofIndex);
      TEST_FLOATING_EQUALITY(expectedCoefficientValue, leftCoefficient, tol);
      TEST_FLOATING_EQUALITY(expectedCoefficientValue, rightCoefficient, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectTraceOnOneElementTensorMesh1D )
  {
    int H1Order = 2;
    FunctionPtr f = Function::xn(1);
    testProjectTraceOnTensorMesh(CellTopology::line(), H1Order, f, TRACE, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnOneElementTensorMesh1D )
  {
    int H1Order = 3;
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity;
    testProjectTraceOnTensorMesh(CellTopology::line(), H1Order, f, FLUX, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnTwoElements2D )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {2,1};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {2.0,1.0}, elemCounts, H1Order);
    
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    map<int, FunctionPtr> solutionMap;
    FunctionPtr n = Function::normal();
    FunctionPtr n_parity = Function::normal() * Function::sideParity();
    FunctionPtr exactFxn = n_parity->x();
    VarPtr sigma_n_hat = form.sigma_n_hat();
    solutionMap[sigma_n_hat->ID()] = exactFxn;
    
    solution->projectOntoMesh(solutionMap);
    
    double tol = 1e-14;
    FunctionPtr solnFxn = Function::solution(sigma_n_hat, solution, false);
    
    double err = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(err, <, tol, out, success);
    
    if (!success)
    {
      set<GlobalIndexType> cellIDs = mesh->getTopology()->getMyActiveCellIndices();
      for (GlobalIndexType cellID : cellIDs)
      {
        Camellia::FieldContainer<double> coefficients;
        int sideOrdinal = 3;
        solution->solnCoeffsForCellID(coefficients, cellID, sigma_n_hat->ID(), sideOrdinal);
        out << "coefficients for side ordinal 3: \n" << coefficients;
      }
    }
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnRefinedMesh2D )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {1,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,2.0}, elemCounts, H1Order);
    
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    map<int, FunctionPtr> solutionMap;
    VarPtr sigma = form.sigma();
    solutionMap[sigma->ID()] = Function::constant({1.0,1.0});
    
    VarPtr sigma_n_hat = form.sigma_n_hat();
    FunctionPtr exactFxn = sigma_n_hat->termTraced()->evaluate(solutionMap);
    
    solutionMap[sigma_n_hat->ID()] = exactFxn;
    
    solution->projectOntoMesh(solutionMap);
    FunctionPtr solnFxn = Function::solution(sigma_n_hat, solution, false);
    
    // as sanity check, confirm that (solnFxn == exactFxn) before refinement
    double tol = 1e-14;
    double errBeforeRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errBeforeRefinement, <, tol, out, success);
    
    // register solution so that mesh will project parents onto children
    mesh->registerSolution(solution);
    
    // after projection, refine.  Should preserve the condition (exactFxn == solnFxn) even under refinement
    mesh->hRefine(vector<GlobalIndexType>{1});
    double errAfterOneRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errAfterOneRefinement, <, tol, out, success);
    
    // refine
    mesh->hRefine(vector<GlobalIndexType>{0,2});
    double errAfterTwoRefinements = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errAfterTwoRefinements, <, tol, out, success);
    
    if (!success)
    {
#ifdef HAVE_EPETRAEXT_HDF5
      HDF5Exporter exporter(mesh, "projectFluxOnRefinedMesh2DFailure");
      exporter.exportFunction({solnFxn-exactFxn,solnFxn, exactFxn},{"error", "solution", "exact"},0);
      
      vector<GlobalIndexType> cellsToReportOn = {9,10,11};
      
      for (auto cellID : cellsToReportOn)
      {
        if (mesh->myCellsInclude(cellID))
        {
          BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID);
          FunctionPtr errFxn = (solnFxn - exactFxn);
          double errOnCell = sqrt((errFxn * errFxn)->integrate(basisCache));
          out << "error on cell " << cellID << ": " << errOnCell << endl;
        }
      }
#endif
    }
  }
  
  /*
   {
   vector<GlobalIndexType> cellIDs = {1};
   mesh->hRefine(cellIDs, repartitionAndRebuild);
   mesh->enforceOneIrregularity();
   mesh->repartitionAndRebuild();
   //      outputMesh(meshOrdinal++);
   cellIDs = {4};
   mesh->hRefine(cellIDs, repartitionAndRebuild);
   mesh->enforceOneIrregularity();
   mesh->repartitionAndRebuild();
   }
   */

  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnRefinedMesh2D_2x2_Starting_Slow )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, elemCounts, H1Order);
    
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    map<int, FunctionPtr> solutionMap;
    VarPtr sigma = form.sigma();
    solutionMap[sigma->ID()] = Function::constant({1.0,1.0});
    
    VarPtr sigma_n_hat = form.sigma_n_hat();
    FunctionPtr exactFxn = sigma_n_hat->termTraced()->evaluate(solutionMap);
    
    solutionMap[sigma_n_hat->ID()] = exactFxn;
    
    solution->projectOntoMesh(solutionMap);
    FunctionPtr solnFxn = Function::solution(sigma_n_hat, solution, false);
    
    // as sanity check, confirm that (solnFxn == exactFxn) before refinement
    double tol = 1e-14;
    double errBeforeRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errBeforeRefinement, <, tol, out, success);
    
    // register solution so that mesh will project parents onto children
    mesh->registerSolution(solution);
    
    // after projection, refine.  Should preserve the condition (exactFxn == solnFxn) even under refinement
    bool repartitionAndRebuild = false;
    vector<GlobalIndexType> cellIDs = {1};
    mesh->hRefine(cellIDs, repartitionAndRebuild);
    mesh->enforceOneIrregularity();
    mesh->repartitionAndRebuild();

    double errAfterOneRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errAfterOneRefinement, <, tol, out, success);

    auto outputMesh = [mesh] (int ordinal) -> void
    {
      MeshTopologyPtr meshCopy = mesh->getTopology()->getGatheredCopy();
      int rank = mesh->Comm()->MyPID();
      if (rank == 0)
      {
        bool labelCells = true;
        int numPointsPerEdge = 2;
        ostringstream name;
        name << "meshSequence-" << ordinal;
        GnuPlotUtil::writeExactMeshSkeleton(name.str(), meshCopy.get(), numPointsPerEdge, labelCells);
      }
    };

//    outputMesh(0);

    if (!success)
    {
#ifdef HAVE_EPETRAEXT_HDF5
      HDF5Exporter exporter(mesh, "ProjectFluxOnRefinedMesh2D_2x2_Starting, after one refinement");
      exporter.exportFunction({solnFxn-exactFxn,solnFxn, exactFxn},{"error", "solution", "exact"},0);
      
      Camellia::print("myCells",mesh->cellIDsInPartition());
      
//      set<GlobalIndexType> cellsToReportOn = mesh->getActiveCellIDsGlobal();
//      print("allCells", cellsToReportOn);
      vector<GlobalIndexType> cellsToReportOn = {0,2,3,4,5,6,7};
      
      for (auto cellID : cellsToReportOn)
      {
        if (mesh->myCellsInclude(cellID))
        {
          BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID);
          FunctionPtr errFxn = (solnFxn - exactFxn);
          double errOnCell = sqrt((errFxn * errFxn)->integrate(basisCache));
          cout << "error on cell " << cellID << ": " << errOnCell << endl;
        }
      }
#endif
    }
    
    cellIDs = {4};
    mesh->hRefine(cellIDs, repartitionAndRebuild);
    mesh->enforceOneIrregularity();
    mesh->repartitionAndRebuild();
    
    if (success) // if we've already failed, focus on that failure...
    {
      double errAfterTwoRefinements = (solnFxn - exactFxn)->l2norm(mesh);
      TEUCHOS_TEST_COMPARE(errAfterTwoRefinements, <, tol, out, success);
      
      if (!success)
      {
        outputMesh(0);
#ifdef HAVE_EPETRAEXT_HDF5
        HDF5Exporter exporter(mesh, "ProjectFluxOnRefinedMesh2D_2x2_Starting");
        exporter.exportFunction({solnFxn-exactFxn,solnFxn, exactFxn},{"error", "solution", "exact"},0);
        
        Camellia::print("myCells",mesh->cellIDsInPartition());
        
        vector<GlobalIndexType> cellsToReportOn = {8,9,15};
        
        for (auto cellID : cellsToReportOn)
        {
          if (mesh->myCellsInclude(cellID))
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID);
            FunctionPtr errFxn = (solnFxn - exactFxn);
            double errOnCell = sqrt((errFxn * errFxn)->integrate(basisCache));
            cout << "error on cell " << cellID << ": " << errOnCell << endl;
          }
        }
#endif
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnRefinedMesh2DSimpler )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int H1Order = 1;
    vector<int> elemCounts = {1,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,2.0}, elemCounts, H1Order);
    
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    map<int, FunctionPtr> solutionMap;
    VarPtr sigma = form.sigma();
    solutionMap[sigma->ID()] = Function::constant({1.0,1.0});

    VarPtr sigma_n_hat = form.sigma_n_hat();
    FunctionPtr exactFxn = sigma_n_hat->termTraced()->evaluate(solutionMap);
    
    solutionMap[sigma_n_hat->ID()] = exactFxn;
    
    solution->projectOntoMesh(solutionMap);
    FunctionPtr solnFxn = Function::solution(sigma_n_hat, solution, false);

    // as sanity check, confirm that (solnFxn == exactFxn) before refinement
    double tol = 1e-14;
    double errBeforeRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errBeforeRefinement, <, tol, out, success);

    // register solution so that mesh will project parents onto children
    mesh->registerSolution(solution);
    
    // after projection, refine.  Should preserve the condition (exactFxn == solnFxn) even under refinement
    mesh->hRefine(vector<GlobalIndexType>{0});
    double errAfterOneRefinement = (solnFxn - exactFxn)->l2norm(mesh);
    TEUCHOS_TEST_COMPARE(errAfterOneRefinement, <, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh1D )
  {
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(CellTopology::line(), tensorialDegree);
    
    vector<double> v00 = {-1,-1};
    vector<double> v10 = {1,-1};
    vector<double> v20 = {2,-1};
    vector<double> v01 = {-1,1};
    vector<double> v11 = {1,1};
    vector<double> v21 = {2,1};
    
    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00); // 0
    spaceTimeVertices.push_back(v10); // 1
    spaceTimeVertices.push_back(v20); // 2
    spaceTimeVertices.push_back(v01); // 3
    spaceTimeVertices.push_back(v11); // 4
    spaceTimeVertices.push_back(v21); // 5
    
    vector<IndexType> spaceTimeLine1VertexList;
    vector<IndexType> spaceTimeLine2VertexList;
    spaceTimeLine1VertexList.push_back(0);
    spaceTimeLine1VertexList.push_back(1);
    spaceTimeLine1VertexList.push_back(3);
    spaceTimeLine1VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(1);
    spaceTimeLine2VertexList.push_back(2);
    spaceTimeLine2VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(5);
    
    vector< vector<IndexType> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    spaceTimeElementVertices.push_back(spaceTimeLine2VertexList);
    
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    spaceTimeCellTopos.push_back(line_x_time);
    
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr v = varFactory->testVarSpaceTime("v", HGRAD);
    
    // define trial variables
    VarPtr uhat = varFactory->fluxVarSpaceTime("uhat");
    
    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    
    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 3, pToAdd = 1;
    MeshPtr spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
    spaceTimeMesh->setTensorTopologyMeansSpaceTime(true);
    
    SolutionPtr spaceTimeSolution = Solution::solution(spaceTimeMesh);
    
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity;
    
    map<int, FunctionPtr > functionMap;
    
    functionMap[uhat->ID()] = f;
    spaceTimeSolution->projectOntoMesh(functionMap);
    
    //    for (GlobalIndexType cellID=0; cellID <= 1; cellID++) {
    //      cout << "CellID " << cellID << " info:\n";
    //      Camellia::FieldContainer<double> localCoefficients = spaceTimeSolution->allCoefficientsForCellID(cellID);
    //
    //      DofOrderingPtr trialOrder = spaceTimeMesh->getElementType(cellID)->trialOrderPtr;
    //
    //      Camellia::printLabeledDofCoefficients(varFactory, trialOrder, localCoefficients);
    //    }
    
    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++)
    {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution, false);
      
      int cubDegreeEnrichment = 0;
      bool spatialSidesOnly = false;
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh, cubDegreeEnrichment, spatialSidesOnly);
      TEST_COMPARE(err_L2, <, tol);
      
      // pointwise comparison
      set<GlobalIndexType> cellIDs = spaceTimeMesh->cellIDsInPartition();
      for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
      {
        GlobalIndexType cellID = *cellIDIt;
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh, cellID);
        if ((trialVar->varType() == FLUX) || (trialVar->varType() == TRACE))
        {
          int sideCount = spaceTimeMesh->getElementType(cellID)->cellTopoPtr->getSideCount();
          for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
          {
            BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
            Camellia::FieldContainer<double> physicalPoints = sideCache->getPhysicalCubaturePointsFC();
            int numPoints = physicalPoints.dimension(1); // (C,P,D)
            out << "physicalPoints for side " << sideOrdinal << ":\n" << physicalPoints;
            Camellia::FieldContainer<double> actualValues(1,numPoints); // assumes scalar-valued
            Camellia::FieldContainer<double> expectedValues(1,numPoints); // assumes scalar-valued
            f_actual->values(actualValues, sideCache);
            f_expected->values(expectedValues, sideCache);
            TEST_COMPARE_FLOATING_ARRAYS(expectedValues, actualValues, tol);
          }
        }
        else
        {
          Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePointsFC();
          int numPoints = physicalPoints.dimension(1); // (C,P,D)
          out << "physicalPoints:\n" << physicalPoints;
          Camellia::FieldContainer<double> actualValues(1,numPoints); // assumes scalar-valued
          Camellia::FieldContainer<double> expectedValues(1,numPoints); // assumes scalar-valued
          f_actual->values(actualValues, basisCache);
          f_expected->values(expectedValues, basisCache);
          TEST_COMPARE_FLOATING_ARRAYS(expectedValues, actualValues, tol);
        }
      }
    }
    
    //    map<GlobalIndexType,GlobalIndexType> cellMap_t0, cellMap_t1;
    //    MeshPtr meshSlice_t0 = MeshTools::timeSliceMesh(spaceTimeMesh, 0, cellMap_t0, H1Order);
    //    FunctionPtr sliceFunction_t0 = MeshTools::timeSliceFunction(spaceTimeMesh, cellMap_t0, Function::xn(1), 0);
    //    HDF5Exporter exporter0(meshSlice_t0, "Function1D_t0");
    //    exporter0.exportFunction(sliceFunction_t0, "x");
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh2D_Slow )
  {
    int tensorialDegree = 1;
    CellTopoPtr quad_x_time = CellTopology::cellTopology(CellTopology::quad(), tensorialDegree);
    CellTopoPtr tri_x_time = CellTopology::cellTopology(CellTopology::triangle(), tensorialDegree);
    
    // let's draw a little house
    vector<double> v00 = {-1,0,0};
    vector<double> v10 = {1,0,0};
    vector<double> v20 = {1,2,0};
    vector<double> v30 = {-1,2,0};
    vector<double> v40 = {0.0,3,0};
    vector<double> v01 = {-1,0,1};
    vector<double> v11 = {1,0,1};
    vector<double> v21 = {1,2,1};
    vector<double> v31 = {-1,2,1};
    vector<double> v41 = {0.0,3,1};
    
    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v30);
    spaceTimeVertices.push_back(v40);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);
    spaceTimeVertices.push_back(v31);
    spaceTimeVertices.push_back(v41);
    
    vector<IndexType> spaceTimeQuadVertexList;
    spaceTimeQuadVertexList.push_back(0);
    spaceTimeQuadVertexList.push_back(1);
    spaceTimeQuadVertexList.push_back(2);
    spaceTimeQuadVertexList.push_back(3);
    spaceTimeQuadVertexList.push_back(5);
    spaceTimeQuadVertexList.push_back(6);
    spaceTimeQuadVertexList.push_back(7);
    spaceTimeQuadVertexList.push_back(8);
    vector<IndexType> spaceTimeTriVertexList;
    spaceTimeTriVertexList.push_back(3);
    spaceTimeTriVertexList.push_back(2);
    spaceTimeTriVertexList.push_back(4);
    spaceTimeTriVertexList.push_back(8);
    spaceTimeTriVertexList.push_back(7);
    spaceTimeTriVertexList.push_back(9);
    
    vector< vector<IndexType> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeQuadVertexList);
    spaceTimeElementVertices.push_back(spaceTimeTriVertexList);
    
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(quad_x_time);
    spaceTimeCellTopos.push_back(tri_x_time);
    
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr tau = varFactory->testVarSpaceTime("tau", HDIV);
    VarPtr v = varFactory->testVarSpaceTime("v", HGRAD);
    
    // define trial variables
    VarPtr uhat = varFactory->traceVarSpaceTime("uhat");
    VarPtr fhat = varFactory->fluxVarSpaceTime("fhat");
    VarPtr u = varFactory->fieldVarSpaceTime("u");
    VarPtr sigma1 = varFactory->fieldVarSpaceTime("sigma1", L2);
    VarPtr sigma2 = varFactory->fieldVarSpaceTime("sigma2", L2);
    
    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma1, tau->x());
    bf->addTerm(sigma2, tau->y());
    bf->addTerm(u, tau->div());
    bf->addTerm(-uhat, tau->dot_normal());
    
    // v terms:
    bf->addTerm( sigma1, v->dx() );
    bf->addTerm( sigma2, v->dy() );
    bf->addTerm( fhat, v);
    
    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 3, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
    spaceTimeMesh->setTensorTopologyMeansSpaceTime(true);
    
    SolutionPtr spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );
    
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f_flux = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity + Function::zn(1) * n->z() * parity;
    
    FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = Function::xn(1) * traceFxn;
    functionMap[fhat->ID()] = f_flux;
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma1->ID()] = Function::xn(1);
    functionMap[sigma2->ID()] = Function::yn(1);
    spaceTimeSolution->projectOntoMesh(functionMap);
    
    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++)
    {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution, false);
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
      TEST_COMPARE(err_L2, <, tol);
    }
  }

  void testProjectExtrudedMesh(int baseDim, int extrusionDim, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    // project an exactly-recoverable function onto a single-element mesh.
    // We use H^1 order of 2, which means L^2 variables get linear bases.
    std::vector<int> H1Orders(1+extrusionDim, 2); // we set one polynomial order in the base dimension, and one in each extrusion dimension
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    SolutionPtr soln = Solution::solution(mesh);
    
    FunctionPtr f = Function::constant(1.0);
    for (int d=0; d<baseDim+extrusionDim; d++)
    {
      FunctionPtr xi = MonomialFunction::xi_n(d,1);
      f = xi * f;
    }
    
    std::map<int,FunctionPtr> projectionMap;
    VarPtr uField = mesh->bilinearForm()->varFactory()->fieldVar("u");
    projectionMap[uField->ID()] = f;
    
    soln->projectOntoMesh(projectionMap);
    
    FunctionPtr u_soln = Function::solution(uField, soln);
    
    double err_l2 = (f - u_soln)->l2norm(mesh);
    
    TEST_COMPARE(err_l2, <, tol);
  }

  void testProject1D1VMeshAnisotropicallyRefined(double tol, Teuchos::FancyOStream &out, bool &success)
  {
    // project an exactly-recoverable function onto a single-element mesh.
    // We use H^1 order of 2, which means L^2 variables get linear bases.
    std::vector<int> H1Orders(2, 2); // we set one polynomial order in the base dimension, and one in each extrusion dimension
    auto mesh = makeSingleElementExtrudedTestMesh(1, 1, H1Orders);
    mesh->hRefine(vector<GlobalIndexType>{0}); // make 2x2 mesh
    
    auto xAnisotropicRef = RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
    // refine one element anisotropically:
    mesh->hRefine(vector<GlobalIndexType>{1}, xAnisotropicRef);
    // the above makes a hanging node; refine one of its children -- these are cells 4 and 5
    mesh->hRefine(vector<GlobalIndexType>{5}, xAnisotropicRef);
    // now we have a 2-irregular mesh; fix that:
    mesh->enforceOneIrregularity();
    
    SolutionPtr soln = Solution::solution(mesh);
    
    FunctionPtr f = Function::constant(1.0);
    for (int d=0; d<2; d++)
    {
      FunctionPtr xi = MonomialFunction::xi_n(d,1);
      f = xi * f;
    }
    
    std::map<int,FunctionPtr> projectionMap;
    VarPtr uField = mesh->bilinearForm()->varFactory()->fieldVar("u");
    projectionMap[uField->ID()] = f;
    VarPtr uFlux  = mesh->bilinearForm()->varFactory()->fluxVar("uhat");
    projectionMap[uFlux->ID()] = f;
    
    soln->projectOntoMesh(projectionMap);
    
    FunctionPtr u_soln = Function::solution(uField, soln);
    
    double err_l2 = (f - u_soln)->l2norm(mesh);
    
    TEST_COMPARE(err_l2, <, tol);
    
    FunctionPtr sideParity = Function::sideParity();
    FunctionPtr uFlux_soln = Function::solution(uFlux, soln, true);
    double flux_err_l2 = (f * sideParity - uFlux_soln)->l2norm(mesh);
    
    TEST_COMPARE(flux_err_l2, <, tol);
  }

  void testProjectExtrudedMeshConstantExtrusions(int baseDim, int extrusionDim, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    // project a spatial (base-mesh) function onto an extruded mesh
    // test that the derivative in extruded directions is zero
    std::vector<int> H1Orders(1+extrusionDim, 2); // we set one polynomial order in the base dimension, and one in each extrusion dimension
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    SolutionPtr soln = Solution::solution(mesh);
    
    FunctionPtr f = Function::xn(1);
    
    std::map<int,FunctionPtr> projectionMap;
    VarPtr uField = mesh->bilinearForm()->varFactory()->fieldVar("u");
    projectionMap[uField->ID()] = f;
    
    soln->projectOntoMesh(projectionMap);
    
    FunctionPtr u_soln = Function::solution(uField, soln);
    
    // di() is 1-based; di(1) means dx(), etc.
    for (int i=1; i<=baseDim+extrusionDim; i++)
    {
      out << "checking di, i=" << i << std::endl;
      double err_l2 = (f->di(i) - u_soln->di(i))->l2norm(mesh);
      TEST_COMPARE(err_l2, <, tol);
    }
  }

  TEUCHOS_UNIT_TEST( Solution, Project1D1VMeshAnisotropicallyRefined )
  {
    const double tol = 1e-14;
    testProject1D1VMeshAnisotropicallyRefined(tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMesh1D1V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 1;
    testProjectExtrudedMesh(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMesh1D2V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 2;
    testProjectExtrudedMesh(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMesh1D3V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 3;
    testProjectExtrudedMesh(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMesh2D3V )
  {
    const double tol = 1e-14;
    const int baseDim = 2;
    const int extrusionDim = 3;
    testProjectExtrudedMesh(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMesh3D3V )
  {
    const double tol = 1e-12;
    const int baseDim = 3;
    const int extrusionDim = 3;
    testProjectExtrudedMesh(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMeshConstantExtrusions1D1V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 1;
    testProjectExtrudedMeshConstantExtrusions(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMeshConstantExtrusions1D2V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 2;
    testProjectExtrudedMeshConstantExtrusions(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMeshConstantExtrusions1D3V )
  {
    const double tol = 1e-14;
    const int baseDim = 1;
    const int extrusionDim = 3;
    testProjectExtrudedMeshConstantExtrusions(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMeshConstantExtrusions2D3V )
  {
    const double tol = 1e-13;
    const int baseDim = 2;
    const int extrusionDim = 3;
    testProjectExtrudedMeshConstantExtrusions(baseDim, extrusionDim, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnExtrudedTensorMeshConstantExtrusions3D3V )
  {
    const double tol = 1e-12;
    const int baseDim = 3;
    const int extrusionDim = 3;
    testProjectExtrudedMeshConstantExtrusions(baseDim, extrusionDim, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh3D_Slow )
  {
    int tensorialDegree = 1;
    CellTopoPtr hex_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >(), tensorialDegree);
    
    // let's draw a little box
    vector<double> v00 = {0,0,0,0};
    vector<double> v10 = {1,0,0,0};
    vector<double> v20 = {1,1,0,0};
    vector<double> v30 = {0,1,0,0};
    vector<double> v40 = {0,0,1,0};
    vector<double> v50 = {1,0,1,0};
    vector<double> v60 = {1,1,1,0};
    vector<double> v70 = {0,1,1,0};
    vector<double> v01 = {0,0,0,1};
    vector<double> v11 = {1,0,0,1};
    vector<double> v21 = {1,1,0,1};
    vector<double> v31 = {0,1,0,1};
    vector<double> v41 = {0,0,1,1};
    vector<double> v51 = {1,0,1,1};
    vector<double> v61 = {1,1,1,1};
    vector<double> v71 = {0,1,1,1};
    
    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v30);
    spaceTimeVertices.push_back(v40);
    spaceTimeVertices.push_back(v50);
    spaceTimeVertices.push_back(v60);
    spaceTimeVertices.push_back(v70);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);
    spaceTimeVertices.push_back(v31);
    spaceTimeVertices.push_back(v41);
    spaceTimeVertices.push_back(v51);
    spaceTimeVertices.push_back(v61);
    spaceTimeVertices.push_back(v71);
    
    vector<IndexType> spaceTimeHexVertexList;
    spaceTimeHexVertexList.push_back(0);
    spaceTimeHexVertexList.push_back(1);
    spaceTimeHexVertexList.push_back(2);
    spaceTimeHexVertexList.push_back(3);
    spaceTimeHexVertexList.push_back(4);
    spaceTimeHexVertexList.push_back(5);
    spaceTimeHexVertexList.push_back(6);
    spaceTimeHexVertexList.push_back(7);
    spaceTimeHexVertexList.push_back(8);
    spaceTimeHexVertexList.push_back(9);
    spaceTimeHexVertexList.push_back(10);
    spaceTimeHexVertexList.push_back(11);
    spaceTimeHexVertexList.push_back(12);
    spaceTimeHexVertexList.push_back(13);
    spaceTimeHexVertexList.push_back(14);
    spaceTimeHexVertexList.push_back(15);
    
    vector< vector<IndexType> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeHexVertexList);
    
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(hex_x_time);
    
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr tau = varFactory->testVarSpaceTime("tau", HDIV);
    VarPtr v = varFactory->testVarSpaceTime("v", HGRAD);
    
    // define trial variables
    VarPtr uhat = varFactory->traceVarSpaceTime("uhat");
    VarPtr fhat = varFactory->fluxVarSpaceTime("fhat");
    VarPtr u = varFactory->fieldVarSpaceTime("u");
    VarPtr sigma1 = varFactory->fieldVarSpaceTime("sigma1", L2);
    VarPtr sigma2 = varFactory->fieldVarSpaceTime("sigma2", L2);
    VarPtr sigma3 = varFactory->fieldVarSpaceTime("sigma3", L2);
    
    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma1, tau->x());
    bf->addTerm(sigma2, tau->y());
    bf->addTerm(sigma3, tau->z());
    bf->addTerm(u, tau->div());
    bf->addTerm(-uhat, tau->dot_normal());
    
    // v terms:
    bf->addTerm( sigma1, v->dx() );
    bf->addTerm( sigma2, v->dy() );
    bf->addTerm( sigma3, v->dz() );
    bf->addTerm( fhat, v);
    
    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 4, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
    spaceTimeMesh->setTensorTopologyMeansSpaceTime(true);
    
    SolutionPtr spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );
    
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f_flux = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity + Function::zn(1) * n->z() * parity;
    
    FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = Function::xn(1) * traceFxn;
    functionMap[fhat->ID()] = f_flux;
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma1->ID()] = Function::xn(1);
    functionMap[sigma2->ID()] = Function::yn(1);
    functionMap[sigma3->ID()] = Function::zn(1);
    spaceTimeSolution->projectOntoMesh(functionMap);
    
    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++)
    {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      out << "testing " << trialVar->name() << std::endl;
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution, false);
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
      TEST_COMPARE(err_L2, <, tol);
    }
  }
  
  void testSaveAndLoad2D(BFPtr bf, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2;
    vector<double> dimensions = {1.0, 2.0}; // 1 x 2 domain
    vector<int> elementCounts = {3, 2}; // 3 x 2 mesh
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    BCPtr bc = BC::bc();
    RHSPtr rhs = RHS::rhs();
    IPPtr ip = bf->graphNorm();
    SolutionPtr soln = Solution::solution(mesh,bc,rhs,ip);
    
#ifdef HAVE_EPETRAEXT_HDF5
    string filePrefix = "SavedSolution";
    soln->save(filePrefix);
    
    soln->load(bf, filePrefix);
    MeshPtr loadedMesh = soln->mesh();
    TEST_EQUALITY(loadedMesh->globalDofCount(), mesh->globalDofCount());
    
    // delete the files we created
    remove((filePrefix+".soln").c_str());
    remove((filePrefix+".mesh").c_str());
    
    // just to confirm that we can manipulate the loaded mesh:
    set<GlobalIndexType> cellsToRefine;
    cellsToRefine.insert(0);
    loadedMesh->pRefine(cellsToRefine);
#else
    out << "SKIPPING testSaveAndLoad2D() test because Trilinos built without Epetra HDF5 support.\n";
#endif
  }
  
  TEUCHOS_UNIT_TEST( Solution, SaveAndLoadPoissonConforming )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    testSaveAndLoad2D(form.bf(), out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, SaveAndLoadStokesConforming )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    double mu = 1.0;
    StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim,mu,conformingTraces);
    testSaveAndLoad2D(form.bf(), out, success);
  }
  
  void testUnrefine(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2; // linear field order
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int meshWidth = 2;
    vector<double> domainDim(spaceDim, 1.0);
    vector<int> meshDim(spaceDim, meshWidth);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), domainDim, meshDim, H1Order);
    
    GlobalIndexType parentCellID = 1;
    
    SolutionPtr soln = Solution::solution(mesh);
    mesh->registerSolution(soln);
    
    // add a concrete solution, so we can check that projections are done appropriately
    // we will just use one that can be exactly represented (a patch test)
    
    FunctionPtr u_exact = Function::xn(1);
    std::map<int, FunctionPtr> solnMap;
    solnMap[form.u()->ID()] = u_exact;
    
    soln->projectOntoMesh(solnMap);
    
    if (mesh->myCellsInclude(parentCellID))
    {
      auto elemType = mesh->getElementType(parentCellID);
      auto basis = elemType->trialOrderPtr->getBasis(form.u()->ID());
      
      Camellia::FieldContainer<double> basisCoeffs(basis->getCardinality());
      soln->solnCoeffsForCellID(basisCoeffs, parentCellID, form.u()->ID());
      
      out << "parentCoefficients before refinement:\n" << basisCoeffs;
    }
    
    FunctionPtr u_soln = Function::solution(form.u(), soln);
    
    // sanity check: verify that u_soln and u_exact match in the original mesh
    double tol = 1e-14;
    double err = (u_soln - u_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    
    vector<GlobalIndexType> cellsToRefine = {parentCellID};
    mesh->hRefine(cellsToRefine);
    
    // check that after the forward refinement, u_soln still matches u_exact
    err = (u_soln - u_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
    
    // for this to be an effective test, we discard parent cell's old coefficients -- these should be recomputed during hUnrefine()
    soln->discardInactiveCellCoefficients();
    
    bool parentHasCoefficients = soln->cellHasCoefficientsAssigned(parentCellID);
    TEST_ASSERT(!parentHasCoefficients);
    
    mesh->hUnrefine({parentCellID});
    
    if (mesh->myCellsInclude(parentCellID))
    {
      parentHasCoefficients = soln->cellHasCoefficientsAssigned(parentCellID);
      TEST_ASSERT(parentHasCoefficients);
      
      auto elemType = mesh->getElementType(parentCellID);
      auto basis = elemType->trialOrderPtr->getBasis(form.u()->ID());
      
      Camellia::FieldContainer<double> basisCoeffs(basis->getCardinality());

      soln->solnCoeffsForCellID(basisCoeffs, parentCellID, form.u()->ID());
    
      out << "parentCoefficients after unrefinement:\n" << basisCoeffs;
    }
    
    // check that after the unrefinement, u_soln still matches u_exact
    err = (u_soln - u_exact)->l2norm(mesh);
    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( Solution, Unrefine_1D )
  {
    int spaceDim = 1;
    testUnrefine(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, Unrefine_2D )
  {
    int spaceDim = 2;
    testUnrefine(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Solution, Unrefine_3D )
  {
    int spaceDim = 3;
    testUnrefine(spaceDim, out, success);
  }
} // namespace
