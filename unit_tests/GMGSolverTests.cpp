//
//  GMGSolverTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/16/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "EpetraExt_RowMatrixOut.h"

#include "GDAMinimumRule.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"
#include "SuperLUDistSolver.h"

using namespace Camellia;

namespace
{
  void writeOperatorsToDisk(Teuchos::RCP<GMGOperator> op)
  {
    int rank = Teuchos::GlobalMPISession::getRank();

    Teuchos::RCP<Epetra_RowMatrix> opMatrix = op->getMatrixRepresentation();
    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/op.dat",*opMatrix,NULL,NULL,false);
    if (rank==0) cout << "wrote op matrix to /tmp/op.dat\n";

    Teuchos::RCP<Epetra_RowMatrix> P = op->getProlongationOperator();
    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
    if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";

    Teuchos::RCP<Epetra_RowMatrix> coarseMatrix = op->getCoarseStiffnessMatrix();
    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/coarse.dat",*coarseMatrix,NULL,NULL,false);
    if (rank==0) cout << "wrote coarse stiffness matrix to /tmp/coarse.dat\n";

    if (op->getSmootherType() != GMGOperator::NONE)
    {
      Teuchos::RCP<Epetra_RowMatrix> smoother = op->getSmootherAsMatrix();
      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/smoother.dat",*smoother,NULL,NULL,false);
      if (rank==0) cout << "wrote smoother matrix to /tmp/smoother.dat\n";
    }

    string fineStiffnessLocation = "/tmp/A.dat";
    EpetraExt::RowMatrixToMatrixMarketFile(fineStiffnessLocation.c_str(),*op->getFineStiffnessMatrix(),NULL,NULL,false);
    if (rank==0) cout << "wrote fine stiffness matrix to " << fineStiffnessLocation << endl;
  }

  void turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver)
  {
    Teuchos::RCP<GMGOperator> gmgOperator = gmgSolver->gmgOperator();
    while (gmgOperator->getCoarseOperator() != Teuchos::null)
    {
      gmgOperator = gmgOperator->getCoarseOperator();
    }
    SolverPtr coarseSolver = gmgOperator->getCoarseSolver();
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
    SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
    if (superLUSolver)
    {
      superLUSolver->setRunSilent(true);
    }
#endif
  }

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

    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);

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

  SolutionPtr poissonExactSolutionRefined_2D(int H1Order, FunctionPtr phi_exact, bool useH1Traces, int refinementSetOrdinal)
  {
    vector<int> numCells;
    numCells.push_back(2);
    numCells.push_back(2);
    SolutionPtr soln = poissonExactSolution(numCells, H1Order, phi_exact, useH1Traces);

    MeshPtr mesh = soln->mesh();

    set<GlobalIndexType> cellIDs;
    switch (refinementSetOrdinal)
    {
      case 0: // no refinements
        break;
      case 1: // one refinement
        cellIDs = {3};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
        break;
      case 2:
        cellIDs = {3};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {6,7};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {1};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
        cellIDs.clear();
        break;

      case 3:
        cellIDs = {1,3};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {6,7,8,10,11};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {2};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {4,9,12,14,19,26,31};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);

        cellIDs = {0,5};
        mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported refinement number");
    }

    return soln;
  }

  // ! In this test, the prolongation operator is the identity: we have the same mesh for coarse and fine.
  void testIdentity(int spaceDim, bool useRefinedMeshes, int refinementNumber, int meshWidth,
                    bool useConformingTraces, bool useStaticCondensation,
                    GMGOperator::SmootherApplicationType smootherApplicationType,
                    Teuchos::FancyOStream &out, bool &success)
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
    SolutionPtr exactPoissonSolution, actualPoissonSolution;
    if (!useRefinedMeshes)
    {
      exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
    }
    else if (spaceDim == 2)
    {
      exactPoissonSolution = poissonExactSolutionRefined_2D(H1Order, phiExact, useH1Traces, refinementNumber);
      actualPoissonSolution = poissonExactSolutionRefined_2D(H1Order, phiExact, useH1Traces, refinementNumber);
    }

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
    turnOffSuperLUDistOutput(gmgSolver);
    gmgSolver->setComputeConditionNumberEstimate(false);

//    gmgSolver->gmgOperator()->setDebugMode(true);

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

    // since we may change the RHS vector below, let's make a copy and use that
    Epetra_MultiVector rhsVectorCopy2(rhsVectorCopy);

    Epetra_CrsMatrix *A = exactPoissonSolution->getStiffnessMatrix().get();

    const Epetra_Map* map = &A->RowMatrixRowMap();

    Teuchos::RCP<Epetra_Vector> diagA = Teuchos::rcp( new Epetra_Vector(*map) );
    A->ExtractDiagonalCopy(*diagA);

//    gmgSolver->gmgOperator()->setStiffnessDiagonal(diagA);

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
    // if smoother applied before coarse solve, then we expect exact agreement between direct solution and iterative.
    // If smoother is not applied before coarse solve and not after (i.e. a strict two-level operator),  then we expect iterative = exact + D^-1 b
    // If smoother is not applied before coarse solve but is applied after,  then we expect iterative = exact + D^-1 b - D^-1 A D^-1 b
    Epetra_MultiVector directValue(*lhsVector); // x
    if (smootherApplicationType == GMGOperator::ADDITIVE)
    {
      // x + D^-1 b
      directValue.Multiply(1.0, rhsVectorCopy2, *diagA_inv, 1.0);
    }
//    else
//    {
//      // DEBUGGING multiplicative option:
//      gmgSolver->gmgOperator()->setDebugMode(true);
//
//      cout << "x:\n" << directValue;
//    }

    gmgSolver->gmgOperator()->setSmootherType(GMGOperator::POINT_JACOBI);

    gmgSolver->gmgOperator()->setFineStiffnessMatrix(A);
    gmgSolver->gmgOperator()->setSmootherApplicationType(smootherApplicationType);
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
//    if (!success)
//    {
//      writeOperatorsToDisk(gmgSolver->gmgOperator());
//    }

    // do "multi" grid between mesh and itself.  Solution should match phiExact.
    maxIters = (smootherApplicationType == GMGOperator::MULTIPLICATIVE) ? 1 : 100; // if smoother applied multiplicatively, then GMG should recover exactly the direct solution, in 1 iteration

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

    turnOffSuperLUDistOutput(gmgSolver);
    gmgSolver->setComputeConditionNumberEstimate(false);
    gmgSolver->gmgOperator()->setSmootherApplicationType(smootherApplicationType);

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
      out << "FAILURE: For mesh width = " << meshWidth << " in " << spaceDim << "D, with smootherApplicationType = ";
      if (smootherApplicationType == GMGOperator::ADDITIVE)
        out << "additive";
      else
        out << "multiplicative";

      out << ", GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";

      Teuchos::RCP<GMGOperator> op = gmgSolver->gmgOperator();
      writeOperatorsToDisk(op);
    }
  }

  // ! This test adapted from one that used to reside in GMGTests (testGMGSolverIdentityUniformMeshes)
  // ! In this test, the prolongation operator is the identity: we have the same mesh for coarse and fine.
  void testIdentityUniformMeshes(int spaceDim, int meshWidth, bool useConformingTraces, bool useStaticCondensation,
                                 GMGOperator::SmootherApplicationType smootherApplicationType,
                                 Teuchos::FancyOStream &out, bool &success)
  {
    bool useRefinedMeshes = false;
    int refinementNumber = -1;
    testIdentity(spaceDim, useRefinedMeshes, refinementNumber, meshWidth, useConformingTraces,
                 useStaticCondensation, smootherApplicationType, out, success);
  }

  // ! This test adapted from one that used to reside in GMGTests (testGMGSolverIdentityUniformMeshes)
  // ! In this test, the prolongation operator is the identity: we have the same mesh for coarse and fine.
  void testIdentityRefined2DMeshes(int refinementSequence, bool useConformingTraces, bool useStaticCondensation,
                                   GMGOperator::SmootherApplicationType smootherApplicationType,
                                   Teuchos::FancyOStream &out, bool &success)
  {
    bool useRefinedMeshes = true;
    int spaceDim = 2;
    int meshWidth = 2;
    testIdentity(spaceDim, useRefinedMeshes, refinementSequence, meshWidth, useConformingTraces,
                 useStaticCondensation, smootherApplicationType, out, success);
  }

  void testOperatorIsSPD(Teuchos::RCP<GMGOperator> op, Teuchos::FancyOStream &out, bool &success)
  {
    Intrepid::FieldContainer<double> denseMatrix;
    Teuchos::RCP<Epetra_RowMatrix> opMatrix = op->getMatrixRepresentation();
    SerialDenseWrapper::extractFCFromEpetra_RowMatrix(*opMatrix, denseMatrix);

    double relTol = 1e-6; // we don't need very tight tolerance
    double absTol = relTol; // what counts as a zero, basically
    vector<pair<int,int>> asymmetricEntries;
    if (! SerialDenseWrapper::matrixIsSymmetric(denseMatrix,asymmetricEntries,relTol,absTol))
    {
      success = false;
      out << "matrix is not symmetric.  Details:\n";
      for (pair<int,int> entry : asymmetricEntries)
      {
        int i = entry.first, j = entry.second;
        out << "A[" << i << "," << j << "] = " << denseMatrix(i,j);
        out << " ≠ " << denseMatrix(j,i) << " = A[" << j << "," << i << "]" << endl;
      }
      // out << "full matrix:\n" << denseMatrix;

      writeOperatorsToDisk(op);
    }
//    else
//    {
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "even though we pass the test, for debugging, writing things out to disk\n";
//      writeOperatorsToDisk(op);
//    }

    // TODO: implement SerialDenseWrapper::matrixIsPositiveDefinite() using eigenvalue solve
//    if (! SerialDenseWrapper::matrixIsPositiveDefinite(denseMatrix))
//    {
//      success = false;
//      out << "matrix is not positive definite:\n" << denseMatrix;
//    }
  }

  TEUCHOS_UNIT_TEST( GMGSolver, MeshesForMultigrid)
  {
    int spaceDim = 1;
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> cellCounts(spaceDim,1);
    
    int k_coarse = 0;
    int H1Order_coarse = k_coarse + 1;
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dimensions, cellCounts, H1Order_coarse, delta_k);
    
    vector<GlobalIndexType> expectedDofCounts;
    expectedDofCounts.push_back(mesh->numGlobalDofs());
    
    int numHRefs = 3;
    for (int i=0; i<numHRefs; i++)
    {
      mesh->hRefine(mesh->getActiveCellIDs());
      expectedDofCounts.push_back(mesh->numGlobalDofs());
    }
    
    expectedDofCounts.push_back(mesh->numGlobalDofs()); // expect to have the fine h mesh twice
    
    int H1Order_fine = 5;
    int numPRefs = H1Order_fine - H1Order_coarse;
    for (int i=0; i< numPRefs; i++)
    {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
    
    expectedDofCounts.push_back(mesh->numGlobalDofs());
    
    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, k_coarse, delta_k);
    
    vector<GlobalIndexType> actualDofCounts;
    for (MeshPtr mesh : meshes)
    {
      actualDofCounts.push_back(mesh->numGlobalDofs());
    }
    
    TEST_COMPARE_ARRAYS(actualDofCounts, expectedDofCounts);
    
    if (!success)
    {
      Camellia::print("actualDofCounts", actualDofCounts);
      Camellia::print("expectedDofCounts", expectedDofCounts);
    }
  }
  
  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_1D_Slow)
  {
    int spaceDim = 1;
    bool useConformingTraces = false;

    vector<bool> staticCondensationChoices = {false, true};
    vector<GMGOperator::SmootherApplicationType> smootherApplicationTypes = {GMGOperator::ADDITIVE, GMGOperator::MULTIPLICATIVE};
    vector<int> meshWidths = {1,2};
//    vector<bool> staticCondensationChoices = {false};
//    vector<bool> applySmootherBeforeCoarseSolveChoices = {false};
//    vector<bool> applySmootherAfterCoarseSolveChoices = {true};
//    vector<int> meshWidths = {1};

    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (GMGOperator::SmootherApplicationType smootherApplicationType : smootherApplicationTypes)
      {
          for (int meshWidth : meshWidths)
          {
            testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation,
                                      smootherApplicationType, out, success);
          }
      }
    }
  }

  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_2D_Slow)
  {
    int spaceDim = 2;
    bool useConformingTraces = false;

    vector<bool> staticCondensationChoices = {false, true};
    // to make the test faster, just use the most common choice for applying smoother (corresponds to two-level)
    vector<GMGOperator::SmootherApplicationType> smootherApplicationTypes = {GMGOperator::ADDITIVE};

    vector<int> meshWidths = {2};

    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (GMGOperator::SmootherApplicationType smootherApplicationType : smootherApplicationTypes)
      {
        for (int meshWidth : meshWidths)
        {
          testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation,
                                    smootherApplicationType, out, success);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( GMGSolver, RefinedIdentity_2D_Slow)
  {
    bool useConformingTraces = false;

    vector<bool> staticCondensationChoices = {false, true};
    vector<GMGOperator::SmootherApplicationType> smootherApplicationTypes = {GMGOperator::ADDITIVE};
    vector<int> refinementSequences = {0,1,2,3};

    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (GMGOperator::SmootherApplicationType smootherApplicationType : smootherApplicationTypes)
      {
        for (int refinementSequence : refinementSequences)
        {
          testIdentityRefined2DMeshes(refinementSequence, useConformingTraces, useStaticCondensation,
                                      smootherApplicationType, out, success);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( GMGSolver, UniformIdentity_3D_Slow)
  {
    int spaceDim = 3;
    bool useConformingTraces = false;

    vector<bool> staticCondensationChoices = {false}; // to keep test cost down, we'll consider testing static condensation in 1D and 2D good enough.
      // to make the test faster, just use the most common choices for applying smoother (corresponds to two-level)
    vector<GMGOperator::SmootherApplicationType> smootherApplicationTypes = {GMGOperator::ADDITIVE};

    vector<int> meshWidths = {2};

    for (bool useStaticCondensation : staticCondensationChoices)
    {
      for (GMGOperator::SmootherApplicationType smootherApplicationType : smootherApplicationTypes)
      {
        for (int meshWidth : meshWidths)
        {
          testIdentityUniformMeshes(spaceDim, meshWidth, useConformingTraces, useStaticCondensation,
                                    smootherApplicationType, out, success);
        }

      }
    }
  }

  void setupPoissonGMGSolver_TwoGrid_h(Teuchos::RCP<GMGSolver> &gmgSolver, SolutionPtr &fineSolution,
                                       int spaceDim, FunctionPtr phi_exact)
  {
    bool useConformingTraces = false;
    PoissonFormulation form(spaceDim, useConformingTraces);
    int coarseElementCount = 1;
    int H1Order_fine = 1, delta_k = 1;
    int H1Order_coarse = 1;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,coarseElementCount);
    BFPtr bf = form.bf();

    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);

    fineMesh->hRefine(set<GlobalIndexType>({0}));

    RHSPtr rhs = RHS::rhs();
    if (spaceDim==1)
      rhs->addTerm(phi_exact->dx()->dx() * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy()) * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz())  * form.q());
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
    fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
    int maxIters = 1000;
    double tol = 1e-6;
    gmgSolver = Teuchos::rcp( new GMGSolver(fineSolution,{coarseMesh, fineMesh},maxIters,tol) );
    turnOffSuperLUDistOutput(gmgSolver);
  }

  void setupPoissonGMGSolver_TwoGrid_p(Teuchos::RCP<GMGSolver> &gmgSolver, SolutionPtr &fineSolution,
                                       int spaceDim, FunctionPtr phi_exact)
  {
    bool useConformingTraces = false;
    PoissonFormulation form(spaceDim, useConformingTraces);
    int coarseElementCount = 1;
    int H1Order_fine = 3, delta_k = 1;
    int H1Order_coarse = 1;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,coarseElementCount);
    BFPtr bf = form.bf();

    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);

    RHSPtr rhs = RHS::rhs();
    if (spaceDim==1)
      rhs->addTerm(phi_exact->dx()->dx() * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy()) * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz())  * form.q());
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
    fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
    int maxIters = 1000;
    double tol = 1e-6;
    gmgSolver = Teuchos::rcp( new GMGSolver(fineSolution,{coarseMesh, fineMesh},maxIters,tol) );
    turnOffSuperLUDistOutput(gmgSolver);
  }

  void setupPoissonGMGSolver_ThreeGrid(Teuchos::RCP<GMGSolver> &gmgSolver, SolutionPtr &fineSolution,
                                       int spaceDim, FunctionPtr phi_exact, int coarseElementCount)
  {
    bool useConformingTraces = false;
    PoissonFormulation form(spaceDim, useConformingTraces);
    int H1Order_fine = 2, delta_k = 1;
    int H1Order_coarse = 1;
    int H1Order_coarsest = 1;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,coarseElementCount);
    BFPtr bf = form.bf();
    MeshPtr coarsestMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarsest, delta_k);
    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);

    coarseMesh->hRefine(vector<GlobalIndexType>({0}));
    fineMesh->hRefine(vector<GlobalIndexType>({0}));

    RHSPtr rhs = RHS::rhs();
    if (spaceDim==1)
      rhs->addTerm(phi_exact->dx()->dx() * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy()) * form.q());
    else if (spaceDim==3)
      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz())  * form.q());
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
    fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
    int maxIters = 1000;
    double tol = 1e-6;
    gmgSolver = Teuchos::rcp( new GMGSolver(fineSolution,{coarsestMesh, coarseMesh, fineMesh},maxIters,tol) );
    turnOffSuperLUDistOutput(gmgSolver);
  }

  enum GridType {
    TwoGrid_h,
    TwoGrid_p,
    ThreeGrid,
    ThreeGrid_Width_2
  };

void testOperatorIsSPD(int spaceDim, GridType gridType, GMGOperator::SmootherApplicationType smootherApplicationType, Teuchos::FancyOStream &out, bool &success)
  {
    FunctionPtr phiExact;
    if (spaceDim == 1)
    {
      FunctionPtr x = Function::xn(1);
      phiExact = x * x;
    }
    else if (spaceDim == 2)
    {
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      phiExact = x * x + x * y;
    }
    else if (spaceDim == 3)
    {
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      FunctionPtr z = Function::zn(1);
      phiExact = x * x + x * y;
    }
    SolutionPtr fineSolution;
    Teuchos::RCP<GMGSolver> solver;

    switch (gridType) {
      case TwoGrid_h:
        setupPoissonGMGSolver_TwoGrid_h(solver, fineSolution, spaceDim, phiExact);
        break;
      case TwoGrid_p:
        setupPoissonGMGSolver_TwoGrid_p(solver, fineSolution, spaceDim, phiExact);
        break;
      case ThreeGrid:
        setupPoissonGMGSolver_ThreeGrid(solver, fineSolution, spaceDim, phiExact, 1);
        break;
      case ThreeGrid_Width_2:
        setupPoissonGMGSolver_ThreeGrid(solver, fineSolution, spaceDim, phiExact, 2);
        break;
    }

    fineSolution->initializeLHSVector();
    fineSolution->initializeStiffnessAndLoad();
    fineSolution->populateStiffnessAndLoad();
    solver->gmgOperator()->setFineStiffnessMatrix(fineSolution->getStiffnessMatrix().get());

    vector<GMGOperator::SmootherChoice> smootherChoices = {GMGOperator::NONE, GMGOperator::IFPACK_ADDITIVE_SCHWARZ, GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ};

    for (GMGOperator::SmootherChoice smoother : smootherChoices)
    {
      string smootherString = GMGOperator::smootherString(smoother);
      out << "***************** Testing smoother choice " << smootherString << " *****************" << endl;
      solver->gmgOperator()->setSmootherType(smoother);
      solver->gmgOperator()->setFineStiffnessMatrix(fineSolution->getStiffnessMatrix().get());
      if (smoother == GMGOperator::NONE)
      {
        testOperatorIsSPD(solver->gmgOperator(), out, success);
      }
      else
      {
        solver->gmgOperator()->setSmootherApplicationType(smootherApplicationType);
        if (smootherApplicationType == GMGOperator::ADDITIVE)
        {
          out << "          *** Testing smootherApplicationType = ADDITIVE ***" << endl;
        }
        else
        {
          out << "          *** Testing smootherApplicationType = MULTIPLICATIVE ***" << endl;
        }
        testOperatorIsSPD(solver->gmgOperator(), out, success);
      }
    }
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_1D_h )
  {
    int spaceDim = 1;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_h;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_1D_h_multiplicative )
  {
    int spaceDim = 1;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = TwoGrid_h;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_1D_p )
  {
    int spaceDim = 1;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_p;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_1D_p_multiplicative )
  {
    int spaceDim = 1;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = TwoGrid_p;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGridOperatorIsSPD_1D )
  {
    int spaceDim = 1;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = ThreeGrid;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_2D_h )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_h;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_2D_p )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_p;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_2D_p_multiplicative )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = TwoGrid_p;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_2D_h_multiplicative )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = TwoGrid_h;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGridOperatorIsSPD_2D_Slow )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = ThreeGrid;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }


  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGridOperatorIsSPD_2D_Multiplicative_Slow )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = ThreeGrid;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGridWidth2OperatorIsSPD_2D_Multiplicative_Slow )
  {
    int spaceDim = 2;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
    GridType gridType = ThreeGrid_Width_2;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_3D_h_Slow )
  {
    int spaceDim = 3;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_h;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGridOperatorIsSPD_3D_p_Slow )
  {
    int spaceDim = 3;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = TwoGrid_p;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGridOperatorIsSPD_3D_Slow )
  {
    int spaceDim = 3;
    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::ADDITIVE;
    GridType gridType = ThreeGrid;
    testOperatorIsSPD(spaceDim, gridType, smootherApplicationType, out, success);
  }

//  TEUCHOS_UNIT_TEST( GMGSolver, DebuggingOperatorApplyInverse )
//  {
//    int rank = Teuchos::GlobalMPISession::getRank();
//    int spaceDim = 1;
//    bool useConformingTraces = false;
//    GMGOperator::SmootherChoice smoother = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
//    GMGOperator::SmootherApplicationType smootherApplicationType = GMGOperator::MULTIPLICATIVE;
//
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 1, delta_k = 1;
//    int H1Order_coarse = 1;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//
//    fineMesh->hRefine(set<GlobalIndexType>({0}));
//
//    FunctionPtr phi_exact = getPhiExact(spaceDim);
//
//    RHSPtr rhs = RHS::rhs();
//    if (spaceDim==1)
//      rhs->addTerm(phi_exact->dx()->dx() * form.q());
//    else if (spaceDim==3)
//      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy()) * form.q());
//    else if (spaceDim==3)
//      rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz())  * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(fineSolution,{coarseMesh, fineMesh},maxIters,tol) );
//    turnOffSuperLUDistOutput(gmgSolver);
//
//    Teuchos::RCP<GMGOperator> op = gmgSolver->gmgOperator();
//    op->setSmootherType(smoother);
//    op->setSmootherApplicationType(smootherApplicationType);
//    op->setDebugMode(false);
//
//    fineSolution->initializeLHSVector();
//    fineSolution->initializeStiffnessAndLoad();
//    fineSolution->populateStiffnessAndLoad();
//    op->setFineStiffnessMatrix(fineSolution->getStiffnessMatrix().get());
//
//    Intrepid::FieldContainer<double> denseMatrix;
//    Teuchos::RCP<Epetra_RowMatrix> opMatrix = op->getMatrixRepresentation();
//    SerialDenseWrapper::extractFCFromEpetra_RowMatrix(*opMatrix, denseMatrix);
//
//    if (rank==0) cout << "For debugging, writing things out to disk\n";
//    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/op.dat",*opMatrix,NULL,NULL,false);
//    if (rank==0) cout << "wrote op matrix to /tmp/op.dat\n";
//
//    Teuchos::RCP<Epetra_RowMatrix> P = op->getProlongationOperator();
//    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
//    if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";
//
//    Teuchos::RCP<Epetra_RowMatrix> coarseMatrix = op->getCoarseStiffnessMatrix();
//    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/coarse.dat",*coarseMatrix,NULL,NULL,false);
//    if (rank==0) cout << "wrote coarse stiffness matrix to /tmp/coarse.dat\n";
//
//    if (op->getSmootherType() != GMGOperator::NONE)
//    {
//      Teuchos::RCP<Epetra_RowMatrix> smoother = op->getSmootherAsMatrix();
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/smoother.dat",*smoother,NULL,NULL,false);
//      if (rank==0) cout << "wrote smoother matrix to /tmp/smoother.dat\n";
//    }
//
//    string fineStiffnessLocation = "/tmp/A.dat";
//    EpetraExt::RowMatrixToMatrixMarketFile(fineStiffnessLocation.c_str(),*op->getFineStiffnessMatrix(),NULL,NULL,false);
//    if (rank==0) cout << "wrote fine stiffness matrix to " << fineStiffnessLocation << endl;
//
//    int GID = 0;
//    cout << "setting up fine vector for testing, with global ordinal " << GID << " --> 1";
//    op->setDebugMode(true);
//
//    Epetra_MultiVector v(P->RowMatrixRowMap(), 1);
//    int lid = v.Map().LID(GID);
//    if (lid != -1)
//    {
//      v[0][lid] = 1.0;
//    }
//    Epetra_MultiVector y(P->RowMatrixRowMap(), 1);
//    op->ApplyInverse(v, y);
//    cout << "y:\n";
//    y.Comm().Barrier();
//    cout << y;
//    y.Comm().Barrier();
//
//    opMatrix->Apply(v, y);
//    cout << "y from opMatrix:\n";
//    y.Comm().Barrier();
//    cout << y;
//    y.Comm().Barrier();
//  }

  // for the moment, disabling the tests below.  They do pass (at least when doing two-level) in the sense of achieving the required tolerance, but I'd like to
  // get clearer on what exactly I am testing with them.

//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGrid_1D_h_multigrid )
//  {
//    int spaceDim = 1;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 2, delta_k = spaceDim;
//    int H1Order_coarse = H1Order_fine;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//
//    fineMesh->hRefine(set<GlobalIndexType>({0}));
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
//    solver->setComputeConditionNumberEstimate(true);
//    solver->setAztecOutput(100);
//
//    int err = fineSolution->solve(solver);
//
//    TEST_EQUALITY(err, 0);
//  }
//
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonTwoGrid_1D_p_multigrid )
//  {
//    int spaceDim = 1;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 2;
//    int H1Order_fine = 3, delta_k = spaceDim;
//    int H1Order_coarse = 1;
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
//    int err = fineSolution->solve(solver);
//    TEST_EQUALITY(err, 0);
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
//    solver->setAztecOutput(10);
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
//
//    solver->setUseConjugateGradient(false); // suspicion is we violate symmetry with default options in GMGOperator
//
////    // use two-level:
////    solver->gmgOperator()->setSmoothBeforeCoarseSolve(false);
////    solver->gmgOperator()->setSmoothAfterCoarseSolve(false);
//    solver->setComputeConditionNumberEstimate(true);
//    solver->setAztecOutput(10);
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
//    solver->setUseConjugateGradient(false); // suspicion is we violate symmetry with default options in GMGOperator
//    solver->setComputeConditionNumberEstimate(true);
//    solver->setAztecOutput(10);
//
//    fineSolution->solve(solver);
//  }
//
//  TEUCHOS_UNIT_TEST( GMGSolver, PoissonThreeGrid_3D )
//  {
//    int spaceDim = 3;
//    bool useConformingTraces = false;
//    PoissonFormulation form(spaceDim, useConformingTraces);
//    int coarseElementCount = 1;
//    int H1Order_fine = 4, delta_k = 1;
//    int H1Order_coarse = 2;
//    int H1Order_coarsest = 2;
//    vector<double> dimensions(spaceDim,1.0);
//    vector<int> elementCounts(spaceDim,coarseElementCount);
//    BFPtr bf = form.bf();
//    MeshPtr coarsestMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarsest, delta_k);
//    MeshPtr coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k);
//    MeshPtr fineMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_fine, delta_k);
//
//    coarseMesh->hRefine(vector<GlobalIndexType>({0}));
//    fineMesh->hRefine(vector<GlobalIndexType>({0}));
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr y = Function::yn(1);
//    FunctionPtr phi_exact = x * x + x * y;
//    RHSPtr rhs = RHS::rhs();
//    rhs->addTerm((phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz()) * form.q());
//    BCPtr bc = BC::bc();
//    bc->addDirichlet(form.phi_hat(), SpatialFilter::allSpace(), phi_exact);
//    SolutionPtr fineSolution = Solution::solution(fineMesh, bc, rhs, bf->graphNorm());
//    int maxIters = 1000;
//    double tol = 1e-6;
//    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp( new GMGSolver(fineSolution,{coarsestMesh, coarseMesh, fineMesh},maxIters,tol) );
//
//    solver->setUseConjugateGradient(true); // suspicion is we violate symmetry with default options in GMGOperator
//
//    // use two-level:
//    solver->gmgOperator()->setSmoothBeforeCoarseSolve(false);
//    solver->gmgOperator()->setSmoothAfterCoarseSolve(false);
//    solver->setComputeConditionNumberEstimate(true);
//
//    // TODO: add a test that the GMGOperator here is symmetric.
//    out << "test unfinished\n";
//    success = false;
//
//    solver->setAztecOutput(10);
//
//    fineSolution->solve(solver);
//  }
} // namespace
