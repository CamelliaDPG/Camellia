//
//  BFTests
//  Camellia
//
//  Created by Nate Roberts on 6/17/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BF.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

using namespace Camellia;
using namespace Intrepid;

namespace
{
  TEUCHOS_UNIT_TEST( BF, FactoredCholeskySolve_Identities )
  {
    int testCount = 5, trialCount = 4;
    
    // inputs: square and rectangular identities, and a vector of 1's
    FieldContainer<double> ip(testCount,testCount);
    FieldContainer<double> stiffnessEnriched(trialCount,testCount);
    FieldContainer<double> rhsEnriched(testCount,1);
    
    for (int i=0; i<testCount; i++)
    {
      ip(i,i) = 1.0;
      rhsEnriched(i,0) = 1.0;
    }
    for (int i=0; i<trialCount; i++)
    {
      stiffnessEnriched(i,i) = 1.0;
    }
    
    FieldContainer<double> stiffness(trialCount,trialCount);
    FieldContainer<double> rhs(trialCount,1);
    
    TBF<double>::factoredCholeskySolve(ip, stiffnessEnriched, rhsEnriched, stiffness, rhs);
    
    // expect stiffness to be identity, and rhs to be vector of 1's
    double tol = 1e-14;
    for (int i=0; i<trialCount; i++)
    {
      for (int j=0; j<trialCount; j++)
      {
        if (i==j)
        {
          TEST_FLOATING_EQUALITY(1.0, stiffness(i,j), tol);
        }
        else
        {
          TEST_COMPARE(abs(stiffness(i,j)), <, tol);
        }
      }
    }
    for (int i=0; i<trialCount; i++)
    {
      TEST_FLOATING_EQUALITY(rhs(i,0), 1.0, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( BF, FactoredCholeskySolve_LowestOrderPoissonAgrees_2D )
  {
    // check that lowest-order Poisson formulation gives same results for both standard Cholesky and factored Cholesky
    int spaceDim = 2;
    bool useConformingTraces = true;
    
    PoissonFormulation form(spaceDim, useConformingTraces, PoissonFormulation::ULTRAWEAK);
    BFPtr bf = form.bf();
    
    int H1Order = 1;
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, H1Order);
    RHSPtr rhsPtr = RHS::rhs();
    rhsPtr->addTerm(1.0 * form.q());
    GlobalIndexType cellZero = 0;
    if (mesh->myCellsInclude(cellZero))
    {
      ElementTypePtr elemType = mesh->getElementType(cellZero);
      int trialCount = elemType->trialOrderPtr->totalDofs();
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellZero);
      BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(mesh, cellZero, true);
      int numCells = 1;
      FieldContainer<double> stiffnessExpected(numCells,trialCount,trialCount), rhsExpected(numCells,trialCount);
      bf->setOptimalTestSolver(TBF<>::CHOLESKY);
      bf->localStiffnessMatrixAndRHS(stiffnessExpected, rhsExpected, bf->graphNorm(), ipBasisCache, rhsPtr, basisCache);
      
      FieldContainer<double> stiffness(numCells,trialCount,trialCount), rhs(numCells,trialCount);
      bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);
      bf->localStiffnessMatrixAndRHS(stiffness, rhs, bf->graphNorm(), ipBasisCache, rhsPtr, basisCache);
      // resize to strip cell dim:
      stiffness.resize(trialCount,trialCount);
      stiffnessExpected.resize(trialCount,trialCount);
      rhs.resize(trialCount);
      rhsExpected.resize(trialCount);
      double tol = 1e-12;
      for (int i=0; i<trialCount; i++)
      {
        for (int j=0; j<trialCount; j++)
        {
          if (abs(stiffnessExpected(i,j)) > tol)
          {
            TEST_FLOATING_EQUALITY(stiffnessExpected(i,j), stiffness(i,j), tol);
          }
          else
          {
            TEST_COMPARE(abs(stiffness(i,j)), <, tol);
          }
        }
      }
      for (int i=0; i<trialCount; i++)
      {
        if (abs(rhsExpected(i)) > tol)
        {
          TEST_FLOATING_EQUALITY(rhsExpected(i), rhs(i), tol);
        }
        else
        {
          TEST_COMPARE(abs(rhs(i)), <, tol);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( BF, FactoredCholeskySolve_SimpleRectangularMatrices )
  {
    int testCount = 3, trialCount = 2;
    
    /* 
     
     inputs: hardcoded matrices as follows:
     
     G = [ 1.0  0.5  0.0]  B = [ 3 2 ]  l = [ 1 ]
         [ 0.5  1.0  2.0]      [ 1 0 ]      [ 2 ]
         [ 0.0  2.0  8.0]      [ 2 2 ]      [ 3 ]
    
     expected outputs:
    
     stiffness: B^T G^-1 B
     rhs:       B^T G^-1 l
     
     Numerically for our example:
     stiffness: [ 13.5   12.5 ]
                [ 12.5   13.5 ]
     
     rhs: [  0.75 ]
          [ -1.75 ]
     */
    
    FieldContainer<double> ip(testCount,testCount);
    FieldContainer<double> stiffnessEnriched(trialCount,testCount);
    FieldContainer<double> rhsEnriched(testCount,1);
    
    // column-major order (Fortran's ordering, and LAPACK's):
    ip[0] = 1.0; ip[3] = 0.5; ip[6] = 0.0;
    ip[1] = 0.5; ip[4] = 1.0; ip[7] = 2.0;
    ip[2] = 0.0; ip[5] = 2.0; ip[8] = 8.0;

    stiffnessEnriched[0] = 3;     stiffnessEnriched[3] = 2;
    stiffnessEnriched[1] = 1;     stiffnessEnriched[4] = 0;
    stiffnessEnriched[2] = 2;     stiffnessEnriched[5] = 2;
    
    rhsEnriched[0] = 1;
    rhsEnriched[1] = 2;
    rhsEnriched[2] = 3;
    
    FieldContainer<double> stiffnessExpected(trialCount,trialCount);
    FieldContainer<double> rhsExpected(trialCount,1);
    
    stiffnessExpected[0] = 13.5;     stiffnessExpected[2] = 12.5;
    stiffnessExpected[1] = 12.5;     stiffnessExpected[3] = 13.5;
    
    rhsExpected[0] =  0.75;
    rhsExpected[1] = -1.75;
    
    FieldContainer<double> stiffness(trialCount,trialCount);
    FieldContainer<double> rhs(trialCount,1);
    
    TBF<double>::factoredCholeskySolve(ip, stiffnessEnriched, rhsEnriched, stiffness, rhs);
    
    // expect stiffness to be identity, and rhs to be vector of 1's
    double tol = 1e-14;
    for (int i=0; i<trialCount; i++)
    {
      for (int j=0; j<trialCount; j++)
      {
        if (abs(stiffnessExpected(i,j)) > tol)
        {
          TEST_FLOATING_EQUALITY(stiffnessExpected(i,j), stiffness(i,j), tol);
        }
        else
        {
          TEST_COMPARE(abs(stiffness(i,j)), <, tol);
        }
      }
    }
    for (int i=0; i<trialCount; i++)
    {
      if (abs(rhsExpected(i,0)) > tol)
      {
        TEST_FLOATING_EQUALITY(rhsExpected(i,0), rhs(i,0), tol);
      }
      else
      {
        TEST_COMPARE(abs(rhs(i,0)), <, tol);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BF, FactoredCholeskySolve_SimpleSquareMatrices )
  {
    int testCount = 2, trialCount = 2;
    
    /*
     
     inputs: hardcoded matrices as follows:
     
     G = [ 1.0   0.5 ]  B = [ 3 2 ]  l = [ 1 ]
         [ 0.5   1.0 ]      [ 1 0 ]      [ 2 ]
     
     expected outputs:
     
     stiffness: B^T G^-1 B
     rhs:       B^T G^-1 l
     
     Numerically for our example:
     stiffness: [ 28/3 20/3 ]
     [ 20/3 16/3 ]
     
     rhs: [ 2 ]
     [ 0 ]
     */
    
    FieldContainer<double> ip(testCount,testCount);
    FieldContainer<double> stiffnessEnriched(trialCount,testCount);
    FieldContainer<double> rhsEnriched(testCount,1);
    
    // column-major order (Fortran's ordering, and LAPACK's):
    ip[0] = 1.0; ip[2] = 0.5;
    ip[1] = 0.5; ip[3] = 1.0;
    
    stiffnessEnriched[0] = 3;     stiffnessEnriched[2] = 2;
    stiffnessEnriched[1] = 1;     stiffnessEnriched[3] = 0;
    
    rhsEnriched[0] = 1;
    rhsEnriched[1] = 2;
    
    FieldContainer<double> stiffnessExpected(trialCount,trialCount);
    FieldContainer<double> rhsExpected(trialCount,1);
    
    stiffnessExpected[0] = 28.0/3.0;     stiffnessExpected[2] = 20.0/3.0;
    stiffnessExpected[1] = 20.0/3.0;     stiffnessExpected[3] = 16.0/3.0;
    
    rhsExpected[0] = 2.0;
    rhsExpected[1] = 0.0;
    
    FieldContainer<double> stiffness(trialCount,trialCount);
    FieldContainer<double> rhs(trialCount,1);
    
    TBF<double>::factoredCholeskySolve(ip, stiffnessEnriched, rhsEnriched, stiffness, rhs);
    
    // expect stiffness to be identity, and rhs to be vector of 1's
    double tol = 1e-14;
    for (int i=0; i<trialCount; i++)
    {
      for (int j=0; j<trialCount; j++)
      {
        if (abs(stiffnessExpected(i,j)) > tol)
        {
          TEST_FLOATING_EQUALITY(stiffnessExpected(i,j), stiffness(i,j), tol);
        }
        else
        {
          TEST_COMPARE(abs(stiffness(i,j)), <, tol);
        }
      }
    }
    for (int i=0; i<trialCount; i++)
    {
      if (abs(rhsExpected(i,0)) > tol)
      {
        TEST_FLOATING_EQUALITY(rhsExpected(i,0), rhs(i,0), tol);
      }
      else
      {
        TEST_COMPARE(abs(rhs(i,0)), <, tol);
      }
    }
  }
} // namespace
