//
//  SerialDenseWrapperTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/6/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "SerialDenseWrapper.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FieldContainer.hpp"

#include <Teuchos_BLAS.hpp>

using namespace Camellia;
using namespace Intrepid;
using namespace std;

namespace
{
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, DeterminantAndInverse_Simple )
  {
    // a couple hard-coded examples
    int numCells = 1;
    int numPoints = 1;
    int spaceDim = 2;
    FieldContainer<double> A(numCells,numPoints,spaceDim,spaceDim);
    A(0,0,0,0) = 1.0;
    A(0,0,1,1) = 1.0;
    FieldContainer<double> expectedInverse = A;
    FieldContainer<double> expectedDet(numCells,numPoints);
    expectedDet[0] = 1.0;
    
    FieldContainer<double> detValues(numCells,numPoints);
    FieldContainer<double> inverse(numCells,numPoints,spaceDim,spaceDim);
    
    SerialDenseWrapper::determinantAndInverse(detValues, inverse, A);
    
    TEST_COMPARE_FLOATING_ARRAYS(detValues, expectedDet, 1e-15);
    TEST_COMPARE_FLOATING_ARRAYS(inverse, expectedInverse, 1e-15);
    
    A(0,0,0,0) = 1.0;
    A(0,0,0,1) = 2.0;
    A(0,0,1,0) = 3.0;
    A(0,0,1,1) = 4.0;
    expectedInverse(0,0,0,0) = -2.0;
    expectedInverse(0,0,0,1) =  1.0;
    expectedInverse(0,0,1,0) =  1.5;
    expectedInverse(0,0,1,1) = -0.5;
    expectedDet[0] = -2;
    
    SerialDenseWrapper::determinantAndInverse(detValues, inverse, A);
    
    TEST_COMPARE_FLOATING_ARRAYS(detValues, expectedDet, 1e-15);
    TEST_COMPARE_FLOATING_ARRAYS(inverse, expectedInverse, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, DeterminantAndInverse_IntrepidComparison )
  {
    // tests the method SerialDenseWrapper::determinantAndInverse() in its initial use case: computing determinants of Jacobian matrices
    
    // we try with arbitrary "Jacobian" data in 1, 2, and 3D, since these are the cases that the existing Intrepid functions support.
    
    int numCells = 2;
    int numPoints = 3;
    for (int spaceDim=1; spaceDim <= 3; spaceDim++)
    {
      FieldContainer<double> jacobianData(numCells,numPoints,spaceDim,spaceDim);
      
      // set up some arbitrary data -- hopefully we don't stumble onto singular matrices this way
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          for (int d1=0; d1<spaceDim; d1++)
          {
            for (int d2=0; d2<spaceDim; d2++)
            {
              double value = (d1 + d2 + 1) * (d1 + d2 + 1) + d1 + d2 + (cellOrdinal + 2) * (ptOrdinal + 1);
              jacobianData(cellOrdinal,ptOrdinal,d1,d2) = value;
            }
          }
        }
      }
      
      FieldContainer<double> detValues(numCells,numPoints);
      FieldContainer<double> outInverses(numCells,numPoints,spaceDim,spaceDim);
      
      int err = SerialDenseWrapper::determinantAndInverse(detValues, outInverses, jacobianData);
      
      TEST_EQUALITY(0, err);
      
      if (err != 0)
      {
        // possibly a singular matrix; output array so we can check
        std::cout << "Error encountered while computing determinant and inverse.  Matrices:\n";
        std::cout << jacobianData;
      }
      
      FieldContainer<double> expectedDetValues(numCells,numPoints);
      FieldContainer<double> expectedOutInverses(numCells,numPoints,spaceDim,spaceDim);
      
      CellTools<double>::setJacobianDet(expectedDetValues, jacobianData);
      CellTools<double>::setJacobianInv(expectedOutInverses, jacobianData );
      
      SerialDenseWrapper::roundZeros(expectedOutInverses, 1e-14);
      SerialDenseWrapper::roundZeros(outInverses, 1e-14);
      
      TEST_COMPARE_FLOATING_ARRAYS(detValues, expectedDetValues, 1e-13);
      TEST_COMPARE_FLOATING_ARRAYS(outInverses, expectedOutInverses, 1e-13);
    }
  }
  
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, Multiply )
  {
    // X = A * B
    FieldContainer<double> A(2,2);
    FieldContainer<double> B(2,2);
    FieldContainer<double> X(2,2), X_expected(2,2);
    
    A(0,0) = 1.0; A(0,1) = 4.0;
    A(1,0) = 2.0; A(1,1) = 5.0;
    
    B(0,0) = 2.0; B(0,1) = 8.0;
    B(1,0) = 3.0; B(1,1) = 0.0;
    
    X_expected(0,0) = 14.0;  X_expected(0,1) =  8.0;
    X_expected(1,0) = 19.0;  X_expected(1,1) = 16.0;
    
    SerialDenseWrapper::multiply(X, A, B, 'N', 'N');
    
    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
  }
  
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, Transpose )
  {
    // X = A * B
    int m = 20;
    int n = 30;
    FieldContainer<double> A(m,n);
    FieldContainer<double> A_T(n,m);
    
    for (int i=0; i<A.size(); i++)
    {
      A[i] = i;
    }
    
    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        A_T(j,i) = A(i,j);
    
    SerialDenseWrapper::transposeMatrix(A);
    
    TEST_COMPARE_ARRAYS(A, A_T);
  }
  
//  TEUCHOS_UNIT_TEST( SerialDenseWrapper, Multiply_HUGE_CommentMeOut_Slow )
//  {
//    // just timing -- multiplies a bunch of zeroes, doesn't check output
//
//    
//    // X = A * B
//    FieldContainer<double> A(2000,2000);
//    FieldContainer<double> B(2000,2000);
//    FieldContainer<double> X(2000,2000), X_expected(2000,2000);
//    
//    A(0,0) = 1.0; A(0,1) = 4.0;
//    A(1,0) = 2.0; A(1,1) = 5.0;
//    
//    B(0,0) = 2.0; B(0,1) = 8.0;
//    B(1,0) = 3.0; B(1,1) = 0.0;
//    
//    X_expected(0,0) = 14.0;  X_expected(0,1) =  8.0;
//    X_expected(1,0) = 19.0;  X_expected(1,1) = 16.0;
//    
//    SerialDenseWrapper::multiply(X, A, B, 'N', 'N');
//    // the above takes ~0.37 seconds with the built-in BLAS on one core on Ivy Bridge 2.7 GHz i7 (Early 2013 MB Pro)
//    // (appears to use threads there; Activity Monitor shows 350% CPU usage for the process)
//    // (this is about 25 GFlops, not actually that bad.)
//    
////    Teuchos::BLAS<int,double> blas;
////    blas.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 2000, 2000, 2000, 1.0, &A[0], 2000, &B[0], 2000, 0.0, &X[0], 2000);
//    
////    double tol = 1e-15;
////    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
//  }
} // namespace
