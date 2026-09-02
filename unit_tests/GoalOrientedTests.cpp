//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TestTemplate
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "GlobalDofAssignment.h"

using namespace Camellia;

namespace
{
  MeshPtr poissonUniformMesh(const vector<int> & elementWidths, int H1Order, bool useConformingTraces)
  {
    int spaceDim = elementWidths.size();
    int testSpaceEnrichment = spaceDim; //
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(spaceDim,span);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
    return mesh;
  }
  
  SolutionPtr simplePoissonSolution(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs)
  {
    // unit forcing
    auto mesh = poissonUniformMesh(elementWidths, H1Order, useConformingTraces);
    const int spaceDim = elementWidths.size();
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    auto rhs = poissonForm.rhs(Function::constant(1.0));
    auto u_hat = poissonForm.u_hat();
    auto bc = BC::bc();
    if (homogeneousBCs)
    {
      bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
    }
    else
    {
      bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::constant(1.0));
    }
    auto graphNorm = poissonForm.bf()->graphNorm();
    auto soln = Solution::solution(poissonForm.bf(),mesh,bc,rhs,graphNorm);
    
    mesh->registerSolution(soln);
    
    return soln;
  }
  
  SolutionPtr simplePoissonSolutionWithSecondaryRHS(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs)
  {
    auto soln = simplePoissonSolution(elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    const int spaceDim = elementWidths.size();
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    
    LinearTermPtr g = 1*u;
    soln->setGoalOrientedRHS(g);
    
    return soln;
  }
  
  SolutionPtr simplePoissonSolutionWithZeroSecondaryRHS(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs)
  {
    auto soln = simplePoissonSolution(elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    const int spaceDim = elementWidths.size();
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    
    LinearTermPtr g = 0.0*u;
    soln->setGoalOrientedRHS(g);
    
    return soln;
  }
  
  void testPoissonSolveMatches(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs, double tol, bool &success, Teuchos::FancyOStream &out)
  {
    // simple test that Poisson solve with secondary ("goal-oriented") RHS
    // still (a) has the same primary solution
    // and   (b) gets the same adaptive refinements
    
    const int spaceDim = elementWidths.size();
    
    auto soln_OneRHS = simplePoissonSolution                (elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    auto soln_TwoRHS = simplePoissonSolutionWithSecondaryRHS(elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    
    //    {
    //      // DEBUGGING: write out the matrices and RHSes to file
    //      soln_OneRHS->setWriteMatrixToMatrixMarketFile(true, "/tmp/A_one.dat");
    //      soln_TwoRHS->setWriteMatrixToMatrixMarketFile(true, "/tmp/A_two.dat");
    //      soln_OneRHS->setWriteRHSToMatrixMarketFile(true, "/tmp/b_one.dat");
    //      soln_TwoRHS->setWriteRHSToMatrixMarketFile(true, "/tmp/b_two.dat");
    //    }
    
    soln_OneRHS->solve();
    soln_TwoRHS->solve();
    
    // we want to check that the (primary) solutions match each other
    // we'll do some higher-level tests below, but to start with, we'll check that the solution coefficients match
    auto lhs_OneRHS = soln_OneRHS->getLHSVector();
    auto lhs_TwoRHS = soln_TwoRHS->getLHSVector(); // the first column of this should match the first column of lhs_OneRHS
    
    int localLength = lhs_OneRHS->getLocalLength();
    TEST_EQUALITY(localLength, lhs_TwoRHS->getLocalLength()); // the value distribution across MPI should also match
    auto lhsOneView = lhs_OneRHS->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto lhsTwoView = lhs_TwoRHS->getLocalViewHost(Tpetra::Access::ReadOnly);
    
    // we take advantage of the fact that the value distribution should match here
    for (int i=0; i<localLength; i++)
    {
      auto oneRHSValue = lhsOneView(i,0);
      auto twoRHSValue = lhsTwoView(i,0);
      // also use tol as a ceiling for rounding to zero:
      if ((abs(oneRHSValue) < tol) && ((abs(twoRHSValue) < tol)))
      {
        continue;
      }
      TEST_FLOATING_EQUALITY(oneRHSValue, twoRHSValue, tol);
    }
    
    // Solution *also* stores a cell-local representation of solution coefficients, and this should also match
    // for the first (0) solution ordinals.
    auto & myCellIDs = soln_OneRHS->mesh()->cellIDsInPartition();
    for (auto cellID : myCellIDs)
    {
      bool warnAboutOffRank = true; // should all be on-rank
      const int solutionOrdinal = 0;
      auto & coeffsOne = soln_OneRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,solutionOrdinal);
      auto & coeffsTwo = soln_TwoRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,solutionOrdinal);
      if (coeffsOne.size() != coeffsTwo.size())
      {
        out << "FAILURE: Sizes differ: coeffsOne is of length " << coeffsOne.size();
        out << ", while coeffsTwo is of length " << coeffsTwo.size() << endl;
        success = false;
      }
      else
      {
        int dofCount = coeffsOne.size();
        bool savedSuccess = success; // allows us to detect local failure
        success = true;
        for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
        {
          auto dofOne = coeffsOne[dofOrdinal];
          auto dofTwo = coeffsTwo[dofOrdinal];
          if ((abs(dofOne) < tol) && (abs(dofTwo) < tol))
          {
            // both zero, essentially
            continue;
          }
          else
          {
            TEST_FLOATING_EQUALITY(dofOne, dofTwo, tol);
          }
        }
        if (!success) // local failure
        {
          out << "Dofs do not match on cell ID " << cellID << endl;
          out << "coeffsOne:\n" << coeffsOne;
          out << "coeffsTwo:\n" << coeffsTwo;
        }
        // copy back saved value of success
        success = savedSuccess;
      }
    }
    
    // higher-level check below -- check that the solutions globally match
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    
    auto u_OneRHS = Function::solution(u, soln_OneRHS);
    auto u_TwoRHS = Function::solution(u, soln_TwoRHS);
    auto u_diff   = u_OneRHS - u_TwoRHS;
    
    auto u_err_L2 = u_diff->l2norm(soln_OneRHS->mesh());
    TEST_COMPARE(u_err_L2, <, tol);
  }
  
  void testZeroSecondaryRHSHasZeroSolution(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs, double tol, bool &success, Teuchos::FancyOStream &out)
  {
    // simple test that Poisson solve with a zero secondary ("goal-oriented") RHS
    // has a zero secondary (influence) solution
    
    const int spaceDim = elementWidths.size();
    
    auto soln_TwoRHS = simplePoissonSolutionWithZeroSecondaryRHS(elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    
    soln_TwoRHS->solve();
    
    // high-level check below -- check that the solution is zero
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    VarPtr u = poissonForm.u();
    VarPtr u_hat = poissonForm.u_hat();
    std::vector<VarPtr> sigma(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      sigma[d] = poissonForm.sigma(d);
    }
    VarPtr sigma_n_hat = poissonForm.sigma_n_hat();
    
    const bool weightFluxesByParity = false; // should not matter
    const SolutionComponent solutionComponent = INFLUENCE_SOLUTION; // secondary solution
    const int solutionStage = 0;
    auto u_influence           = Function::solution(u,           soln_TwoRHS, weightFluxesByParity, solutionComponent, solutionStage);
    auto u_hat_influence       = Function::solution(u_hat,       soln_TwoRHS, weightFluxesByParity, solutionComponent, solutionStage);
    std::vector<FunctionPtr> sigma_influence(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      sigma_influence[d] = Function::solution(sigma[d], soln_TwoRHS, weightFluxesByParity, solutionComponent, solutionStage);
    }
    
    auto sigma_n_hat_influence = Function::solution(sigma_n_hat, soln_TwoRHS, weightFluxesByParity, solutionComponent, solutionStage);
    // u_influence should be zero; test this:
    auto u_err_L2 = u_influence->l2norm(soln_TwoRHS->mesh());
    TEST_COMPARE(u_err_L2, <, tol);
    auto u_hat_err_L2 = u_hat_influence->l2norm(soln_TwoRHS->mesh());
    TEST_COMPARE(u_hat_err_L2, <, tol);
    double sigma_err_L2_squared = 0.0;
    for (int d=0; d<spaceDim; d++)
    {
      double sigma_err_L2_d = sigma_influence[d]->l2norm(soln_TwoRHS->mesh());
      sigma_err_L2_squared += sigma_err_L2_d * sigma_err_L2_d;
    }
    double sigma_err_L2 = sqrt( sigma_err_L2_squared );
    TEST_COMPARE(sigma_err_L2, <, tol);
    auto sigma_n_hat_err_L2 = sigma_n_hat_influence->l2norm(soln_TwoRHS->mesh());
    TEST_COMPARE(sigma_n_hat_err_L2, <, tol);
  }
  
  void testRefinedSolutionMatches(const vector<int> &elementWidths, int H1Order, bool useConformingTraces, bool homogeneousBCs, double tol,
                                  bool &success, Teuchos::FancyOStream &out)
  {
    // simple test that synthetic Poisson solution with secondary ("goal-oriented") RHS
    // that starts with the same coefficients in primary and secondary solutions still has
    // the same coefficients after a uniform mesh refinement
    
    const int spaceDim = elementWidths.size();
    
    auto soln_TwoRHS = simplePoissonSolutionWithSecondaryRHS(elementWidths, H1Order, useConformingTraces, homogeneousBCs);
    
    auto Comm = soln_TwoRHS->mesh()->Comm();
    int rank = Comm->getRank();
    
    // barrier for debugger attachment:
    Comm->barrier();
    
    // we want to check that the (primary) solutions match each other
    // we'll do some higher-level tests below, but to start with, we'll check that the solution coefficients match
    soln_TwoRHS->initializeLHSVector();
    auto lhs_TwoRHS = soln_TwoRHS->getLHSVector(); // the first column of this should match the first column of lhs_OneRHS
    lhs_TwoRHS->putScalar(1.0); // puts 1.0 in each entry
    
    const int numRefinements = 2;
    for (int refinementNumber=0; refinementNumber<=numRefinements; refinementNumber++)
    {
//      cout << "Rank " << rank << ", starting refinement " << refinementNumber << endl;
      int localLength = lhs_TwoRHS->getLocalLength();
      auto lhsView = lhs_TwoRHS->getLocalViewHost(Tpetra::Access::ReadOnly);
      for (int i=0; i<localLength; i++)
      {
        auto firstValue  = lhsView(i,0);
        auto secondValue = lhsView(i,1);
        // also use tol as a ceiling for rounding to zero:
        if ((abs(firstValue) < tol) && ((abs(secondValue) < tol)))
        {
          continue;
        }
        TEST_FLOATING_EQUALITY(firstValue, secondValue, tol);
      }
      
      // construct cell-local representations of the solution coefficients
      soln_TwoRHS->importSolution();
      
      // Solution *also* stores a cell-local representation of solution coefficients, and this should also match
      // for the first (0) solution ordinals.
      auto & myCellIDs = soln_TwoRHS->mesh()->cellIDsInPartition();
//      {
//        // DEBUGGING
//        cout << "Rank " << rank << ", myCellIDs: ";
//        for (auto cellID : myCellIDs)
//        {
//          cout << cellID << " ";
//        }
//        cout << endl;
//      }
      
      for (auto cellID : myCellIDs)
      {
        bool warnAboutOffRank = true; // should all be on-rank
        auto & coeffsOne = soln_TwoRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,0);
        auto & coeffsTwo = soln_TwoRHS->allCoefficientsForCellID(cellID,warnAboutOffRank,1);
        if (coeffsOne.size() != coeffsTwo.size())
        {
          out << "FAILURE: Sizes differ: coeffsOne is of length " << coeffsOne.size();
          out << ", while coeffsTwo is of length " << coeffsTwo.size() << endl;
          success = false;
        }
        else
        {
          int dofCount = coeffsOne.size();
          bool savedSuccess = success; // allows us to detect local failure
          success = true;
          for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
          {
            auto dofOne = coeffsOne[dofOrdinal];
            auto dofTwo = coeffsTwo[dofOrdinal];
            if ((abs(dofOne) < tol) && (abs(dofTwo) < tol))
            {
              // both zero, essentially
              continue;
            }
            else
            {
              TEST_FLOATING_EQUALITY(dofOne, dofTwo, tol);
            }
          }
          if (!success) // local failure
          {
            out << "Dofs do not match on cell ID " << cellID << endl;
            out << "coeffsOne:\n" << coeffsOne;
            out << "coeffsTwo:\n" << coeffsTwo;
          }
          // copy back saved value of success
          success = savedSuccess;
        }
      }
      
      // higher-level check below -- check that the solutions globally match
      PoissonFormulation poissonForm(spaceDim, useConformingTraces);
      VarPtr u = poissonForm.u();
      
      bool weightFluxesByParity = false;
      
      int solutionStage = 0;
      auto u_0 = Function::solution(u, soln_TwoRHS, weightFluxesByParity, STANDARD_SOLUTION, solutionStage);
      auto u_1 = Function::solution(u, soln_TwoRHS, weightFluxesByParity, INFLUENCE_SOLUTION, solutionStage);
      auto u_diff   = u_0 - u_1;
      
      auto u_err_L2 = u_diff->l2norm(soln_TwoRHS->mesh());
      TEST_COMPARE(u_err_L2, <, tol);
      
      if (refinementNumber != numRefinements)
      {
        auto mesh = soln_TwoRHS->mesh();
        
        // confirm that soln_TwoRHS is registered with mesh/GDA
        auto registeredSolutions = mesh->globalDofAssignment()->getRegisteredSolutions();
        TEST_EQUALITY(1, registeredSolutions.size());
        if (registeredSolutions.size() == 1)
        {
          TSolution<double>* registeredSolutionPtr = registeredSolutions[0].get();
          TEST_EQUALITY(registeredSolutionPtr, soln_TwoRHS.get());
        }
        else
        {
          cout << "registeredSolutions.size() == " << registeredSolutions.size() << endl;
        }
        
        auto activeCellIDs = mesh->getActiveCellIDsGlobal();
//        {
//          // DEBUGGING
//          int rank = mesh->Comm()->MyPID();
//          cout << "On rank " << rank << ", active cellIDs = ";
//          for (auto cellID : activeCellIDs)
//          {
//            cout << cellID << " ";
//          }
//          cout << endl;
//        }
        
        mesh->hRefine(activeCellIDs);
        
      } // if (refinementNumber != numRefinements)
    } // for (int refinementNumber=0; refinementNumber<=numRefinements; refinementNumber++)
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_1D )
  {
    const int spaceDim = 1;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 4;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-14;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }

  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_1D_NonhomogeneousBCs )
  {
    const int spaceDim = 1;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 4;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = false;
    double tol = 1e-14;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }

  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_2D )
  {
    const int spaceDim = 2;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-14;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_2D_NonhomogeneousBCs )
  {
    const int spaceDim = 2;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = false;
    double tol = 1e-14;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_3D )
  {
    const int spaceDim = 3;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-13;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }

  TEUCHOS_UNIT_TEST( GoalOriented, PoissonSolveMatches_3D_NonhomogeneousBCs )
  {
    const int spaceDim = 3;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = false;
    double tol = 1e-13;
    
    testPoissonSolveMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  // The following 1D test fails for reasons that *look* like they do not have to do with goal-oriented stuff.
  // TODO: dig into this, and try adding a refinement test elsewhere (SolutionTests, maybe), and fix whatever's wrong.
  //   (It looks like some of the parent-to-child projection code doesn't quite do the right thing in 1D, which is frankly a
  //    bit surprising -- so maybe I'm missing something, and this really does reflect some issue with the goal-oriented
  //    stuff...)
//  TEUCHOS_UNIT_TEST( GoalOriented, RefinedSolutionMatches_1D )
//  {
//    const int spaceDim = 1;
//    const int meshWidth = 2;
//    auto elementWidths = vector<int>(spaceDim,meshWidth);
//    int H1Order = 4;
//    bool useConformingTraces = true;
//    double tol = 1e-16;
//
//    testRefinedSolutionMatches(elementWidths, H1Order, useConformingTraces, tol, success, out);
//  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, RefinedSolutionMatches_2D )
  {
    const int spaceDim = 2;
    const int meshWidth = 1;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-16;
    
    testRefinedSolutionMatches(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, ZeroGoalHasZeroSolution_1D )
  {
    const int spaceDim = 1;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 4;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-16;
    
    testZeroSecondaryRHSHasZeroSolution(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  
  TEUCHOS_UNIT_TEST( GoalOriented, ZeroGoalHasZeroSolution_2D )
  {
    const int spaceDim = 2;
    const int meshWidth = 2;
    auto elementWidths = vector<int>(spaceDim,meshWidth);
    int H1Order = 1;
    bool useConformingTraces = true;
    bool useHomogeneousBCs = true;
    double tol = 1e-16;
    
    testZeroSecondaryRHSHasZeroSolution(elementWidths, H1Order, useConformingTraces, useHomogeneousBCs, tol, success, out);
  }
  
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace
