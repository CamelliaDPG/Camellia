//
//  GMGOperatorTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/4/15.
//
//

#include "EpetraExt_RowMatrixOut.h"
#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaDebugUtility.h"
#include "CamelliaTestingHelpers.h"
#include "GDAMinimumRule.h"
#include "GMGOperator.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "StokesVGPFormulation.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
  enum FormulationChoice {
    Poisson, Stokes
  };
  
  void testEpetraMatrixIsIdentity(Teuchos::RCP<Epetra_CrsMatrix> P, Teuchos::FancyOStream &out, bool &success)
  {
    int myRowCount = P->Map().NumMyElements();
    
    GlobalIndexTypeToCast globalEntryCount = P->Map().NumGlobalElements();
    Intrepid::FieldContainer<GlobalIndexTypeToCast> colIndices(globalEntryCount);
    Intrepid::FieldContainer<double> colValues(globalEntryCount);
    
//    printMapSummary(P->Map(), "P->Map()");
    
    for (int localRowIndex=0; localRowIndex<myRowCount; localRowIndex++)
    {
      GlobalIndexTypeToCast globalRowIndex = P->Map().GID(localRowIndex);
      int numEntries;
      P->ExtractMyRowCopy(localRowIndex, globalEntryCount, numEntries, &colValues(0), &colIndices(0));
      bool diagEntryFound = false;
      
      double tol=1e-15;
      for (int colEntryOrdinal=0; colEntryOrdinal<numEntries; colEntryOrdinal++)
      {
        GlobalIndexTypeToCast localColIndex = colIndices(colEntryOrdinal);
        GlobalIndexTypeToCast globalColIndex = P->DomainMap().GID(localColIndex);
        double expectedValue;
        double actualValue = colValues(colEntryOrdinal);
        if (globalColIndex != globalRowIndex)
        {
          // expect 0 for off-diagonals
          TEST_COMPARE(abs(actualValue), <, tol);
        }
        else
        {
          // expect 1 on the diagonal
          expectedValue = 1.0;
          TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
          diagEntryFound = true;
        }
      }
      if (!diagEntryFound)
      {
        int rank = Teuchos::GlobalMPISession::getRank();
        cout << "on rank " << rank << ", no diagonal entry found for global row " << globalRowIndex;
        cout << " (num col entries: " << numEntries << ")\n";
      }
      TEST_ASSERT(diagEntryFound);
    }
    
//    { // DEBUGGING: export to disk
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";
//    }
  }

  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorHangingNode_2D)
  {
    // take a mesh with a hanging node, using the same mesh for coarse and fine in a GMGOperator.
    // test that the prolongation operator is the identity
    int spaceDim = 2;
    bool useConformingTraces = true;
    int H1Order = useConformingTraces ? 1 : 2; // make trace variables linear
    bool useStaticCondensation = false;
    vector<int> cellCounts = {1,2}; // two elements
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);
    
    // refine element 0:
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    BCPtr bc = BC::bc();
    IPPtr ip = bf->graphNorm();
    RHSPtr rhs = RHS::rhs();
    
    SolutionPtr fineSoln = Solution::solution(mesh);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setBC(bc);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    
    GMGOperator gmgOperator(bc,mesh,ip,mesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.constructProlongationOperator();
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.getProlongationOperator();
    testEpetraMatrixIsIdentity(P, out, success);
  }
  
  void testIdentityProlongationOperatorUniform(FormulationChoice formChoice, int spaceDim,
                                               bool useConformingTraces, bool useStaticCondensation,
                                               Teuchos::FancyOStream &out, bool &success)
  {
    // take a uniform mesh, using the same mesh for coarse and fine in a GMGOperator.
    // test that the prolongation operator is the identity
//    int H1Order = useConformingTraces ? 1 : 2; // make trace variables linear
    int H1Order = 1;

    vector<int> cellCounts(spaceDim,2);
    BFPtr bf;
    BCPtr bc = BC::bc();
    if (formChoice == Poisson)
    {
      PoissonFormulation form(spaceDim, useConformingTraces);
      bf = form.bf();
    }
    else
    {
      StokesVGPFormulation form(spaceDim, useConformingTraces);
      bf = form.bf();
      bc->addSinglePointBC(form.p()->ID(), 0.0);
    }
    
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);
    
    IPPtr ip = bf->graphNorm();
    RHSPtr rhs = RHS::rhs();
    
    SolutionPtr fineSoln = Solution::solution(mesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    
    GMGOperator gmgOperator(bc,mesh,ip,mesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.constructProlongationOperator();
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.getProlongationOperator();
    testEpetraMatrixIsIdentity(P, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorUniform_2D )
  {
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useStaticCondensation = false;

    testIdentityProlongationOperatorUniform(Poisson, spaceDim, useConformingTraces, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorUniform_StokesCondensed2D )
  {
    int spaceDim = 2;
    bool useConformingTraces = false;
    bool useStaticCondensation = true;
    
    testIdentityProlongationOperatorUniform(Stokes, spaceDim, useConformingTraces, useStaticCondensation, out, success);
  }
  
TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorLine )
{
  /*

   Take a 1D, 1-element mesh with an exact solution.
   Refine.
   Compute the GMGOperator's prolongation of the coarse exact solution onto the fine mesh.
   Check that the solution on the fine mesh is exact.

   */

  int spaceDim = 1;
  bool useConformingTraces = false;
  PoissonFormulation form(spaceDim, useConformingTraces);
  BFPtr bf = form.bf();
  VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
  FunctionPtr phi_exact = Function::xn(2); // x^2 exact solution
  FunctionPtr psi_exact = phi_exact->dx();

  double xLeft = 0.0, xRight = 1.0;
  int coarseElementCount = 1;
  int H1Order = 3, delta_k = spaceDim;
  MeshPtr mesh = MeshFactory::intervalMesh(bf, xLeft, xRight, coarseElementCount, H1Order, delta_k);
  // for debugging, do a refinement first:
  mesh->hRefine(mesh->getActiveCellIDs());
  
  SolutionPtr coarseSoln = Solution::solution(mesh);

  VarPtr q = form.q();
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(phi_exact->dx()->dx() * q);
  coarseSoln->setRHS(rhs);

  IPPtr ip = bf->graphNorm();
  coarseSoln->setIP(ip);

  FunctionPtr n = Function::normal_1D();
  FunctionPtr parity = Function::sideParity();

  map<int, FunctionPtr> exactSolnMap;
  exactSolnMap[phi->ID()] = phi_exact;
  exactSolnMap[psi->ID()] = psi_exact;
  exactSolnMap[phi_hat->ID()] = phi_exact * parity * n;
  exactSolnMap[psi_n_hat->ID()] = psi_exact * parity * n;

  coarseSoln->projectOntoMesh(exactSolnMap);

  double energyError = coarseSoln->energyErrorTotal();

  // sanity check: our exact solution should give us 0 energy error
  double tol = 1e-14;
  TEST_COMPARE(energyError, <, tol);
  
  MeshPtr fineMesh = mesh->deepCopy();

  fineMesh->hRefine(fineMesh->getActiveCellIDs());
//  fineMesh->hRefine(fineMesh->getActiveCellIDs());

  SolutionPtr fineSoln = Solution::solution(fineMesh);
  fineSoln->setIP(ip);
  fineSoln->setRHS(rhs);

  // again, a sanity check, now on the fine solution:
  fineSoln->projectOntoMesh(exactSolnMap);
  bool warnAboutOffRank = false;
  set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
  for (GlobalIndexType cellID : myCellIDs) {
    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
  }
  
  energyError = fineSoln->energyErrorTotal();
  TEST_COMPARE(energyError, <, tol);

  BCPtr bc = BC::bc();
  SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
  bool useStaticCondensation = false;
  GMGOperator gmgOperator(bc,mesh,ip,fineMesh,fineSoln->getDofInterpreter(),
                          fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);

  Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
  Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();

  fineSoln->initializeLHSVector();
  fineSoln->getLHSVector()->PutScalar(0); // set a 0 global solution

  fineSoln->importSolution();      // imports the 0 solution onto each cell
  fineSoln->clearComputedResiduals();

  P->Multiply(false, *coarseSolutionVector, *fineSoln->getLHSVector());

  fineSoln->importSolution();

  energyError = fineSoln->energyErrorTotal();
  TEST_COMPARE(energyError, <, tol);

//  for (GlobalIndexType cellID : myCellIDs) {
//    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
//    out << "\n\n******************** Dofs for cell " << cellID << " (fineSoln) ********************\n\n";
//    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
//    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
//  }
//  
//  { // DEBUGGING: export to disk
//    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
//    int rank = Teuchos::GlobalMPISession::getRank();
//    if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";
//    
//    GDAMinimumRule* fineGDA = dynamic_cast<GDAMinimumRule*>(fineMesh->globalDofAssignment().get());
//    fineGDA->printGlobalDofInfo();
//    
//    GDAMinimumRule* coarseGDA = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//    coarseGDA->printGlobalDofInfo();
//  }
}

TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_Simple )
{
  /*

   Take a 2D, 1-element mesh with exactly representable solution data
   Refine.
   Compute the GMGOperator's prolongation of the coarse solution onto the fine mesh.
   Check that the prolongated solution on the fine mesh matches the refined coarse solution.

   */

  int spaceDim = 2;
  bool useConformingTraces = false;
  PoissonFormulation form(spaceDim, useConformingTraces);
  BFPtr bf = form.bf();
  VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
  //    FunctionPtr phi_exact = Function::xn(2) * Function::yn(1); // x^2 y exact solution
  //    FunctionPtr psi_exact = phi_exact->grad();

  FunctionPtr phi_exact = Function::constant(3.14159);
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr two = Function::constant(2.0);
  FunctionPtr psi_exact = Function::vectorize(one, two);

  int coarseElementCount = 1;
  int H1Order = 2, delta_k = spaceDim;
  vector<double> dimensions(2,1.0);
  vector<int> elementCounts(2,coarseElementCount);
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k);

  SolutionPtr coarseSoln = Solution::solution(mesh);

  map<int, FunctionPtr> exactSolnMap;
  exactSolnMap[phi->ID()] = phi_exact;
  exactSolnMap[psi->ID()] = psi_exact;

  FunctionPtr phi_hat_exact   =   phi_hat->termTraced()->evaluate(exactSolnMap);
  FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactSolnMap);

  exactSolnMap[phi_hat->ID()]   = phi_hat_exact;
  exactSolnMap[psi_n_hat->ID()] = psi_n_hat_exact;

  coarseSoln->projectOntoMesh(exactSolnMap);

  MeshPtr fineMesh = mesh->deepCopy();
  fineMesh->hRefine(fineMesh->getActiveCellIDs());

  SolutionPtr exactSoln = Solution::solution(fineMesh);
  exactSoln->projectOntoMesh(exactSolnMap);

  SolutionPtr fineSoln = Solution::solution(fineMesh);

  BCPtr bc = BC::bc();
  SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
  bool useStaticCondensation = false;
  GMGOperator gmgOperator(bc,mesh,bf->graphNorm(),fineMesh,fineSoln->getDofInterpreter(),
                          fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);

//    LocalDofMapperPtr dofMapper_1 = gmgOperator.getLocalCoefficientMap(1);
//    dofMapper_1->printMappingReport();

  Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
  Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();

  fineSoln->initializeLHSVector();
  fineSoln->getLHSVector()->PutScalar(0); // set a 0 global solution
  fineSoln->importSolution();      // imports the 0 solution onto each cell
  fineSoln->clearComputedResiduals();

  P->Multiply(false, *coarseSolutionVector, *fineSoln->getLHSVector());

  fineSoln->importSolution();

  set<GlobalIndexType> cellIDs = fineSoln->mesh()->getActiveCellIDs();

  // import global solution data onto each rank:
//  fineSoln->importSolutionForOffRankCells(cellIDs);
//  exactSoln->importSolutionForOffRankCells(cellIDs);

//    bool warnAboutOffRank = false;
//    VarFactoryPtr vf = bf->varFactory();
//    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
//      cout << "\n\n******************** Dofs for cell " << *cellIDIt << " (fineSoln before subtracting exact) ********************\n";
//      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(*cellIDIt, warnAboutOffRank);
//      DofOrderingPtr trialOrder = fineMesh->getElementType(*cellIDIt)->trialOrderPtr;
//      printLabeledDofCoefficients(vf, trialOrder, coefficients);
//    }

  set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
  
  bool warnAboutOffRank = false;
  for (GlobalIndexType cellID : myCellIDs) {
    FieldContainer<double> coefficients = exactSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
  }
  
  for (GlobalIndexType cellID : myCellIDs) {
    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    out << "\n\n******************** Dofs for cell " << cellID << " (fineSoln) ********************\n\n";
    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
  }
  
  fineSoln->addSolution(exactSoln, -1.0); // should recover zero solution this way

  fineSoln->importSolutionForOffRankCells(cellIDs);
  
  double tol = 1e-14;
  for (GlobalIndexType cellID : cellIDs) {
    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, false);
    FieldContainer<double> expectedCoefficients(coefficients.size()); // zero coefficients
    
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(coefficients, expectedCoefficients, tol, tol);
//    cout << "\n\n******************** Dofs for cell " << cellID << " (fineSoln after subtracting exact) ********************\n\n";
//    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
//    printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, coefficients);
  }

}

TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_Slow )
{
  /*

   Take a 2D, 1-element mesh with an exact solution.
   Refine.
   Compute the GMGOperator's prolongation of the coarse exact solution onto the fine mesh.
   Check that the solution on the fine mesh is exact.

   */

  int spaceDim = 2;
  bool useConformingTraces = false;
  PoissonFormulation form(spaceDim, useConformingTraces);
  BFPtr bf = form.bf();
  VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
//    FunctionPtr phi_exact = Function::xn(2) * Function::yn(1); // x^2 y exact solution
//    FunctionPtr psi_exact = phi_exact->grad();

  FunctionPtr phi_exact = Function::xn(1) + Function::yn(1);
  FunctionPtr psi_exact = phi_exact->grad();

  int coarseElementCount = 1;
  int H1Order = 2, delta_k = spaceDim;
  vector<double> dimensions(2,1.0);
  vector<int> elementCounts(2,coarseElementCount);
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k);

  SolutionPtr coarseSoln = Solution::solution(mesh);

  VarPtr q = form.q();
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  rhs->addTerm(f * q);
  coarseSoln->setRHS(rhs);

  IPPtr ip = bf->graphNorm();
  coarseSoln->setIP(ip);

  map<int, FunctionPtr> exactSolnMap;
  exactSolnMap[phi->ID()] = phi_exact;
  exactSolnMap[psi->ID()] = psi_exact;

  FunctionPtr phi_hat_exact   =   phi_hat->termTraced()->evaluate(exactSolnMap);
  FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactSolnMap);

  exactSolnMap[phi_hat->ID()]   = phi_hat_exact;
  exactSolnMap[psi_n_hat->ID()] = psi_n_hat_exact;

  coarseSoln->projectOntoMesh(exactSolnMap);

  double energyError = coarseSoln->energyErrorTotal();

  // sanity check: our exact solution should give us 0 energy error
  double tol = 1e-14;
  TEST_COMPARE(energyError, <, tol);

  MeshPtr fineMesh = mesh->deepCopy();

  fineMesh->hRefine(fineMesh->getActiveCellIDs());

  SolutionPtr exactSoln = Solution::solution(fineMesh);
  exactSoln->setIP(ip);
  exactSoln->setRHS(rhs);

  // again, a sanity check, now on the exact fine solution:
  exactSoln->projectOntoMesh(exactSolnMap);
  energyError = exactSoln->energyErrorTotal();
  TEST_COMPARE(energyError, <, tol);

  SolutionPtr fineSoln = Solution::solution(fineMesh);
  fineSoln->setIP(ip);
  fineSoln->setRHS(rhs);

  BCPtr bc = BC::bc();
  SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
  bool useStaticCondensation = false;
  GMGOperator gmgOperator(bc,mesh,ip,fineMesh,fineSoln->getDofInterpreter(),
                          fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);

  Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
  Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();

  fineSoln->initializeLHSVector();
  fineSoln->getLHSVector()->PutScalar(0); // set a 0 global solution
  fineSoln->importSolution();      // imports the 0 solution onto each cell
  fineSoln->clearComputedResiduals();

  P->Multiply(false, *coarseSolutionVector, *fineSoln->getLHSVector());

  fineSoln->importSolution();

  set<GlobalIndexType> cellIDs = fineSoln->mesh()->getActiveCellIDs();

  energyError = fineSoln->energyErrorTotal();
  TEST_COMPARE(energyError, <, tol);

  bool warnAboutOffRank = false;
  VarFactoryPtr vf = bf->varFactory();
//    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
//      cout << "\n\n******************** Dofs for cell " << *cellIDIt << " (fineSoln before subtracting exact) ********************\n";
//      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(*cellIDIt, warnAboutOffRank);
//      DofOrderingPtr trialOrder = fineMesh->getElementType(*cellIDIt)->trialOrderPtr;
//      printLabeledDofCoefficients(vf, trialOrder, coefficients);
//    }

  set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
  
  for (GlobalIndexType cellID : myCellIDs) {
    FieldContainer<double> coefficients = exactSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
  }
  
  for (GlobalIndexType cellID : myCellIDs) {
    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    out << "\n\n******************** Dofs for cell " << cellID << " (fineSoln) ********************\n\n";
    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
  }
  
  fineSoln->addSolution(exactSoln, -1.0);

  // import global solution data onto each rank:
  fineSoln->importSolutionForOffRankCells(cellIDs);
  
  for (GlobalIndexType cellID : cellIDs) {
    FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
    FieldContainer<double> expectedCoefficients(coefficients.size()); // zero coefficients
    
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(coefficients, expectedCoefficients, tol, tol);
    //    cout << "\n\n******************** Dofs for cell " << cellID << " (fineSoln after subtracting exact) ********************\n\n";
    //    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
    //    printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, coefficients);
  }

}
} // namespace
