//
//  GMGSolverTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/16/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "GMGSolver.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;

namespace
{
  FunctionPtr getPhiExact(int spaceDim)
  {
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    if (spaceDim==1)
    {
      return x * x + 1;
    }
    else if (spaceDim==2)
    {
      return x * y + x * x;
    }
    else if (spaceDim==3)
    {
      return x * y * z + z * z * x;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
  }
  
  SolutionPtr poissonExactSolution(vector<int> cellCounts, int H1Order, FunctionPtr phiExact, bool useH1Traces)
  {
    int spaceDim = cellCounts.size();
    bool useConformingTraces = false;
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
    //    FunctionPtr phi_exact = Function::xn(2) * Function::yn(1); // x^2 y exact solution
    FunctionPtr psiExact = (spaceDim > 1) ? phiExact->grad() : phiExact->dx();


    int coarseElementCount = 1;
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,coarseElementCount);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k);
    
    SolutionPtr coarseSoln = Solution::solution(mesh);
    
    map<int, FunctionPtr> exactSolnMap;
    exactSolnMap[phi->ID()] = phiExact;
    exactSolnMap[psi->ID()] = psiExact;
    
    FunctionPtr phi_hat_exact   =   phi_hat->termTraced()->evaluate(exactSolnMap);
    FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactSolnMap);
    
    exactSolnMap[phi_hat->ID()]   = phi_hat_exact;
    exactSolnMap[psi_n_hat->ID()] = psi_n_hat_exact;
    
    coarseSoln->projectOntoMesh(exactSolnMap);
    
    MeshPtr fineMesh = mesh->deepCopy();
    fineMesh->hRefine(fineMesh->getActiveCellIDs());
    
    // rhs = f * q, where f = \Delta phi
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f;
    switch (spaceDim)
    {
      case 1:
        f = phiExact->dx()->dx();
        break;
      case 2:
        f = phiExact->dx()->dx() + phiExact->dy()->dy();
        break;
      case 3:
        f = phiExact->dx()->dx() + phiExact->dy()->dy() + phiExact->dz()->dz();
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
        break;
    }
//    cout << "rhs: " << f->displayString() << " * q\n";
    
    rhs->addTerm(f * form.q());
    
    IPPtr graphNorm = bf->graphNorm();
    
    BCPtr bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    SolutionPtr solution;
    
    bc->addDirichlet(phi_hat, boundary, phiExact);
    solution = Solution::solution(mesh, bc, rhs, graphNorm);

    solution->projectOntoMesh(exactSolnMap);
    
    return solution;
  }
  
  // ! This test adapted from one that used to reside in GMGTests (testGMGSolverIdentityUniformMeshes)
  void testIdentityUniformMeshes(int spaceDim, int meshWidth, bool useConformingTraces, bool useStaticCondensation,
                                 bool applySmoothing, Teuchos::FancyOStream &out, bool &success)
  {
    PoissonFormulation form(spaceDim, useConformingTraces);
    vector<int> cellCount;
    for (int d=0; d<spaceDim; d++)
    {
      cellCount.push_back(meshWidth);
    }
    
    int H1Order = 1;
    bool useH1Traces = false;
    FunctionPtr phiExact = getPhiExact(spaceDim);
    SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
    SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
    
    exactPoissonSolution->setUseCondensedSolve(useStaticCondensation);
    actualPoissonSolution->setUseCondensedSolve(useStaticCondensation);
    
    BCPtr poissonBC = exactPoissonSolution->bc();
    BCPtr zeroBCs = poissonBC->copyImposingZero();
    MeshPtr mesh = exactPoissonSolution->mesh();
    BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
    IPPtr graphNorm = bf->graphNorm();
    
    if (useStaticCondensation)
    {
      // need to populate local stiffness matrices for condensed dof interpreter
      exactPoissonSolution->initializeLHSVector();
      exactPoissonSolution->initializeStiffnessAndLoad();
      exactPoissonSolution->populateStiffnessAndLoad();
    }
    
    Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
    int maxIters = 100;
    double iter_tol = 1e-14;
    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                                                    exactPoissonSolution->getDofInterpreter(),
                                                                    exactPoissonSolution->getPartitionMap(),
                                                                    maxIters, iter_tol, coarseSolver, useStaticCondensation) );
    gmgSolver->setComputeConditionNumberEstimate(false);
    
    // before we test the solve proper, let's check that with smoothing off, ApplyInverse acts just like the standard solve
    //        exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A_direct.dat");
    //        exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b_direct.dat");
    exactPoissonSolution->initializeLHSVector();
    exactPoissonSolution->initializeStiffnessAndLoad();
    exactPoissonSolution->populateStiffnessAndLoad();
    Teuchos::RCP<Epetra_MultiVector> rhsVector = exactPoissonSolution->getRHSVector();
    // since I'm not totally sure that the KluSolver won't clobber the rhsVector, make a copy:
    Epetra_MultiVector rhsVectorCopy(*rhsVector);
    exactPoissonSolution->solve();
    Teuchos::RCP<Epetra_MultiVector> lhsVector = exactPoissonSolution->getLHSVector();
    //        EpetraExt::MultiVectorToMatlabFile("/tmp/x_direct.dat",*lhsVector);
    
    Epetra_MultiVector gmg_lhsVector(rhsVectorCopy.Map(), 1); // lhs has same distribution structure as rhs
    gmgSolver->setApplySmoothingOperator(applySmoothing);
    
    // since we may change the RHS vector below, let's make a copy and use that
    Epetra_MultiVector rhsVectorCopy2(rhsVectorCopy);
    
    Epetra_CrsMatrix *A = exactPoissonSolution->getStiffnessMatrix().get();
    
    const Epetra_Map* map = &A->RowMatrixRowMap();
    
    Teuchos::RCP<Epetra_Vector> diagA = Teuchos::rcp( new Epetra_Vector(*map) );
    A->ExtractDiagonalCopy(*diagA);
    
    gmgSolver->gmgOperator()->setStiffnessDiagonal(diagA);
    
    Teuchos::RCP<Epetra_Vector> diagA_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
    Teuchos::RCP<Epetra_Vector> diagA_sqrt_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
    diagA_inv->Reciprocal(*diagA);
    if (map->NumMyElements() > 0)
    {
      for (int lid = map->MinLID(); lid <= map->MaxLID(); lid++)
      {
        (*diagA_sqrt_inv)[lid] = 1.0 / sqrt((*diagA)[lid]);
      }
    }
    Teuchos::RCP<Epetra_Vector> diagA_sqrt = Teuchos::rcp( new Epetra_Vector(*map, 1) );
    diagA_sqrt->Reciprocal(*diagA_sqrt_inv);
    
    //              EpetraExt::RowMatrixToMatlabFile("/tmp/A.dat",*A);
    //              EpetraExt::MultiVectorToMatlabFile("/tmp/rhs.dat",rhsVectorCopy2);
    
    // determine the expected value
    Epetra_MultiVector directValue(*lhsVector); // x
    if (applySmoothing)
    {
      // x + D^-1 b
      directValue.Multiply(1.0, rhsVectorCopy2, *diagA_inv, 1.0);
    }
    
    // if applySmoothing = false, then we expect exact agreement between direct solution and iterative.
    // If applySmoothing = true,  then we expect iterative = exact + D^-1 b
    
    if (applySmoothing)
    {
      gmgSolver->gmgOperator()->setSmootherType(GMGOperator::POINT_JACOBI);
    }
    else
    {
      gmgSolver->gmgOperator()->setSmootherType(GMGOperator::NONE);
    }
    
    gmgSolver->gmgOperator()->setFineStiffnessMatrix(A);
    gmgSolver->gmgOperator()->ApplyInverse(rhsVectorCopy2, gmg_lhsVector);
    
    double tol = 1e-10;
    int minLID = gmg_lhsVector.Map().MinLID();
    int numLIDs = gmg_lhsVector.Map().NumMyElements();
    for (int lid=minLID; lid < minLID + numLIDs; lid++ )
    {
      double direct_val = directValue[0][lid];
      double gmg_val = gmg_lhsVector[0][lid];
      double diff = abs(direct_val - gmg_val);
      if (diff > tol)
      {
        GlobalIndexType gid = gmg_lhsVector.Map().GID(lid);
        out << "FAILURE: For meshWidth = " << meshWidth << " in " << spaceDim << "D, ";
        out << "GMG ApplyInverse and direct solve differ for gid " << gid << " with difference = " << diff << ".\n";
        success = false;
      }
    }
    
    // do "multi" grid between mesh and itself.  Solution should match phiExact.
    maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration
    
    if (useStaticCondensation)
    {
      // need to populate local stiffness matrices in the
      actualPoissonSolution->initializeLHSVector();
      actualPoissonSolution->initializeStiffnessAndLoad();
      actualPoissonSolution->populateStiffnessAndLoad();
    }
    
    gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                            actualPoissonSolution->getDofInterpreter(),
                                            actualPoissonSolution->getPartitionMap(),
                                            maxIters, iter_tol, coarseSolver, useStaticCondensation) );
    
    gmgSolver->setComputeConditionNumberEstimate(false);
    gmgSolver->setApplySmoothingOperator(applySmoothing);
    
    Teuchos::RCP<Solver> fineSolver = gmgSolver;
    
    actualPoissonSolution->solve(fineSolver);
    exactPoissonSolution->solve(coarseSolver);
    
    VarPtr phi = form.phi();
    
    FunctionPtr exactPhiSoln = Function::solution(phi, exactPoissonSolution);
    FunctionPtr actualPhiSoln = Function::solution(phi, actualPoissonSolution);
    
    double l2_diff = (exactPhiSoln-actualPhiSoln)->l2norm(mesh);
    
    tol = iter_tol * 10;
    if (l2_diff > tol)
    {
      success = false;
      out << "FAILURE: For mesh width = " << meshWidth << " in " << spaceDim << "D, ";
      
      out << "GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
    }
  }
  
  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_1D_Slow)
  {
    int spaceDim = 1;
    bool useConformingTraces = false;
    
    vector<bool> staticCondensationChoices = {false, true};
    vector<bool> applySmoothingChoices = {false, true};
    vector<int> meshWidths = {1,2};
    
    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (bool applySmoothing : applySmoothingChoices)
      {
        for (int meshWidth : meshWidths)
        {
          testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation, applySmoothing, out, success);
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_2D_Slow)
  {
    int spaceDim = 2;
    bool useConformingTraces = false;
    
    vector<bool> staticCondensationChoices = {false, true};
    vector<bool> applySmoothingChoices = {false, true};
    vector<int> meshWidths = {1,2};
    
    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (bool applySmoothing : applySmoothingChoices)
      {
        for (int meshWidth : meshWidths)
        {
          testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation, applySmoothing, out, success);
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_3D_Slow)
  {
    int spaceDim = 3;
    bool useConformingTraces = false;
    
    vector<bool> staticCondensationChoices = {false, true};
    vector<bool> applySmoothingChoices = {false, true};
    vector<int> meshWidths = {1,2};
    
    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (bool applySmoothing : applySmoothingChoices)
      {
        for (int meshWidth : meshWidths)
        {
          testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation, applySmoothing, out, success);
        }
      }
    }
  }
  

  // for the moment, disabling the tests below.  They do pass in the sense of achieving the required tolerance, but I'd like to
  // get clearer on what exactly I am testing with them.
  
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGrid_1D )
//  {
//    int spaceDim = 1;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 3, delta_k = spaceDim;
//    int H1Order_coarse = H1Order_fine - 2;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//    
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr phi_exact = x * x;
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm(phi_exact->dx()->dx() * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp( new GMGSolver(fineSolution,{coarseMesh,fineMesh},maxIters,tol) );
//    solver->setComputeConditionNumberEstimate(false);
//    solver->setAztecOutput(1);
//    
//    fineSolution->solve(solver);
//    
//    // TODO: check the solution.
//  }
//  
//  
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGrid_1D )
//  {
//    int spaceDim = 1;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 3, delta_k = spaceDim;
//    int H1Order_coarse = H1Order_fine - 2;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarsestMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//    
//    coarseMesh->hRefine(vector<GlobalIndexType>({0}));
//    fineMesh->hRefine(vector<GlobalIndexType>({0}));
//    
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr phi_exact = x * x;
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm(phi_exact->dx()->dx() * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp( new GMGSolver(fineSolution,{coarsestMesh, coarseMesh, fineMesh},maxIters,tol) );
//    solver->setComputeConditionNumberEstimate(false);
//    solver->setAztecOutput(1);
//    
//    fineSolution->solve(solver);
//  }
//  
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGrid_2D )
//  {
//    int spaceDim = 2;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 8, delta_k = spaceDim;
//    int H1Order_coarse = 4;
//    int H1Order_coarsest = 2;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarsestMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarsest, delta_k);
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//    
//    coarsestMesh->hRefine(vector<GlobalIndexType>({0}));
//    coarseMesh->hRefine(vector<GlobalIndexType>({0}));
//    fineMesh->hRefine(vector<GlobalIndexType>({0}));
//    
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    FunctionPtr phi_exact = x * x + x * y;
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm(phi_exact->dx()->dx() * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp( new GMGSolver(fineSolution,{coarsestMesh, coarseMesh, fineMesh},maxIters,tol) );
//    solver->setComputeConditionNumberEstimate(false);
//    solver->setAztecOutput(1);
//    
//    fineSolution->solve(solver);
//  }
//  
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGrid_2D )
//  {
//    int spaceDim = 2;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 8, delta_k = spaceDim;
//    int H1Order_coarse = 4;
//    int H1Order_coarsest = 2;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarsestMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarsest, delta_k);
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//    
//    coarsestMesh->hRefine(vector<GlobalIndexType>({0}));
//    coarseMesh->hRefine(vector<GlobalIndexType>({0}));
//    fineMesh->hRefine(vector<GlobalIndexType>({0}));
//    
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    FunctionPtr phi_exact = x * x + x * y;
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm(phi_exact->dx()->dx() * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp( new GMGSolver(fineSolution,{coarsestMesh, fineMesh},maxIters,tol) );
//    solver->setComputeConditionNumberEstimate(false);
//    solver->setAztecOutput(1);
//    
//    fineSolution->solve(solver);
//  }
} // namespace