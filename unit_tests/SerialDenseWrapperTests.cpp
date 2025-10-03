//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
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
#include "Camellia_FieldContainer.hpp"

#include <Teuchos_BLAS.hpp>

using namespace Camellia;
using namespace std;

namespace
{
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, ConvertFCtoSDMandBack)
{
    /*
     static Teuchos::SerialDenseMatrix<int,double> convertFCToSDM(A, transpose, Teuchos::DataAccess);
     static void convertSDMToFC(A_fc, A, transpose);
     */
    
    out << "\nTesting A[3x2]\n";
    const int M = 3;
    const int N = 2;
    Camellia::FieldContainer<double> A(M,N);
    Camellia::FieldContainer<double> A_fc(M,N);
    
    A(0,0) = 1.0; A(0,1) = 4.0;
    A(1,0) = 2.0; A(1,1) = 5.0;
    A(2,0) = 3.0; A(2,1) = 6.0;
    
    std::vector<bool> transposeCases {true, false};
    std::vector<Teuchos::DataAccess> dataAccessCases {Teuchos::Copy, Teuchos::View};
    
    auto runTestCases = [&]()
    {
      bool hasUnitDimension = (A.dimension(0) == 1) || (A.dimension(1) == 1);
      for (const auto transposeIn : transposeCases)
      {
        for (const auto & da : dataAccessCases)
        {
          if (!transposeIn && (da == Teuchos::View) && !hasUnitDimension) continue; // can't use View if transposeIn is false (data needs to move)
          for (const auto transposeOut : transposeCases)
          {
            if (transposeIn) out << "transpose in/";
            else             out << "no transpose in/";
            
            if (da == Teuchos::Copy) out << "copy/";
            else                     out << "view/";
            
            if (transposeOut) out << "transpose out\n";
            else              out << "no transpose out\n";
            
            auto A_SDM = SerialDenseWrapper::convertFCToSDM(A, transposeIn, da);
            
            if (transposeOut)
            {
              A_fc.resize(A_SDM.numCols(),A_SDM.numRows());
            }
            else
            {
              A_fc.resize(A_SDM.numRows(), A_SDM.numCols());
            }
            
            SerialDenseWrapper::convertSDMToFC(A_fc, A_SDM, transposeOut);
            
            bool needTranspose = (transposeIn && !transposeOut) || (!transposeIn && transposeOut);
            if (needTranspose)
            {
              SerialDenseWrapper::transposeMatrix(A_fc);
            }
            
            TEST_COMPARE_FLOATING_ARRAYS(A, A_fc, 0.0); // expect exact equality: no opportunity for roundoff
          }
        }
      }
    };
    
    runTestCases();
    
    // now try with column vector, for which we can always use View
    out << "\nTesting A[3x1]\n";
    A.resize(3,1);
    A(0,0) = 1.0;
    A(1,0) = 2.0;
    A(2,0) = 3.0;
    
    A_fc.resize(3,1);
    
    runTestCases();
    
    // now try with row vector, for which we can always use View
    out << "\nTesting A[1x3]\n";
    A.resize(1,3);
    A(0,0) = 1.0;
    A(0,1) = 2.0;
    A(0,2) = 3.0;
    
    A_fc.resize(1,3);
    
    runTestCases();
  }

  TEUCHOS_UNIT_TEST( SerialDenseWrapper, DeterminantAndInverse_Simple )
  {
    // a couple hard-coded examples
    int numCells = 1;
    int numPoints = 1;
    int spaceDim = 2;
    Camellia::FieldContainer<double> A(numCells,numPoints,spaceDim,spaceDim);
    A(0,0,0,0) = 1.0;
    A(0,0,1,1) = 1.0;
    Camellia::FieldContainer<double> expectedInverse = A;
    Camellia::FieldContainer<double> expectedDet(numCells,numPoints);
    expectedDet[0] = 1.0;
    
    Camellia::FieldContainer<double> detValues(numCells,numPoints);
    Camellia::FieldContainer<double> inverse(numCells,numPoints,spaceDim,spaceDim);
    
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
      Camellia::FieldContainer<double> jacobianData(numCells,numPoints,spaceDim,spaceDim);
      
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
      
      Camellia::FieldContainer<double> detValues(numCells,numPoints);
      Camellia::FieldContainer<double> outInverses(numCells,numPoints,spaceDim,spaceDim);
      
      int err = SerialDenseWrapper::determinantAndInverse(detValues, outInverses, jacobianData);
      
      TEST_EQUALITY(0, err);
      
      if (err != 0)
      {
        // possibly a singular matrix; output array so we can check
        std::cout << "Error encountered while computing determinant and inverse.  Matrices:\n";
        std::cout << jacobianData;
      }
      
      Camellia::FieldContainer<double> expectedDetValues(numCells,numPoints);
      Camellia::FieldContainer<double> expectedOutInverses(numCells,numPoints,spaceDim,spaceDim);
      
      Intrepid::CellTools<double>::setJacobianDet(expectedDetValues, jacobianData);
      Intrepid::CellTools<double>::setJacobianInv(expectedOutInverses, jacobianData );
      
      SerialDenseWrapper::roundZeros(expectedOutInverses, 1e-14);
      SerialDenseWrapper::roundZeros(outInverses, 1e-14);
      
      TEST_COMPARE_FLOATING_ARRAYS(detValues, expectedDetValues, 1e-13);
      TEST_COMPARE_FLOATING_ARRAYS(outInverses, expectedOutInverses, 1e-13);
    }
  }
  
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, Multiply )
  {
    // X = A * B
    out << "\nTesting A[2x2] x B[2x2] = X[2x2]\n";
    Camellia::FieldContainer<double> A(2,2);
    Camellia::FieldContainer<double> B(2,2);
    Camellia::FieldContainer<double> X(2,2), X_expected(2,2);
    
    A(0,0) = 1.0; A(0,1) = 4.0;
    A(1,0) = 2.0; A(1,1) = 5.0;
    
    B(0,0) = 2.0; B(0,1) = 8.0;
    B(1,0) = 3.0; B(1,1) = 0.0;
    
    X_expected(0,0) = 14.0;  X_expected(0,1) =  8.0;
    X_expected(1,0) = 19.0;  X_expected(1,1) = 16.0;
    
    out << "A: no transpose, B: no transpose\n";
    char transposeA = 'N';
    char transposeB = 'N';
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
    
    out << "A: no transpose, B: transpose\n";
    transposeA = 'N';
    transposeB = 'T';
    
    X_expected(0,0) = 34.0;  X_expected(0,1) = 3.0;
    X_expected(1,0) = 44.0;  X_expected(1,1) = 6.0;
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
    
    out << "A: transpose, B: no transpose\n";
    transposeA = 'T';
    transposeB = 'N';
    
    X_expected(0,0) =  8.0;  X_expected(0,1) =  8.0;
    X_expected(1,0) = 23.0;  X_expected(1,1) = 32.0;
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
    
    out << "A: transpose, B: transpose\n";
    transposeA = 'T';
    transposeB = 'T';
    
    X_expected(0,0) = 18.0;  X_expected(0,1) =  3.0;
    X_expected(1,0) = 48.0;  X_expected(1,1) = 12.0;
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
    
    out << "\nTesting A[3x2] x B[2x2] = X[3x2]\n";
    
    A.resize(3,2);
    B.resize(2,2);
    X.resize(3,2);
    X_expected.resize(3,2);
    
    A(0,0) = 1.0; A(0,1) = 4.0;
    A(1,0) = 2.0; A(1,1) = 5.0;
    A(2,0) = 3.0; A(2,1) = 6.0;
    
    B(0,0) = 2.0; B(0,1) = 8.0;
    B(1,0) = 3.0; B(1,1) = 0.0;
    
    out << "A: no transpose, B: no transpose\n";
    transposeA = 'N';
    transposeB = 'N';
    
    X_expected(0,0) = 14.0;  X_expected(0,1) =  8.0;
    X_expected(1,0) = 19.0;  X_expected(1,1) = 16.0;
    X_expected(2,0) = 24.0;  X_expected(2,1) = 24.0;
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
    
    out << "A: no transpose, B: transpose\n";
    transposeA = 'N';
    transposeB = 'T';
    
    X_expected(0,0) = 34.0;  X_expected(0,1) = 3.0;
    X_expected(1,0) = 44.0;  X_expected(1,1) = 6.0;
    X_expected(2,0) = 54.0;  X_expected(2,1) = 9.0;
    
    SerialDenseWrapper::multiply(X, A, B, transposeA, transposeB);
    
    TEST_COMPARE_FLOATING_ARRAYS(X, X_expected, tol);
  }

  TEUCHOS_UNIT_TEST( SerialDenseWrapper, SolveIdentitySPD)
  {
//    int SerialDenseWrapper::solveSPDSystemMultipleRHS(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A_SPD,
//                                                      Camellia::FieldContainer<double> &b, bool allowOverwriteOfA)
    Camellia::FieldContainer<double> A(3,3);
    A(0,0) = 1.0;
    A(1,1) = 1.0;
    A(2,2) = 1.0;
    
    Camellia::FieldContainer<double> b(3,2);
    b(0,0) = 1.0;  b(0,1) = 4.0;
    b(1,0) = 2.0;  b(1,1) = 5.0;
    b(2,0) = 3.0;  b(2,1) = 6.0;
    
    Camellia::FieldContainer<double> x(3,2);
    
    Camellia::FieldContainer<double> xExpected = b;
    
    bool allowOverwriteOfA = false;
    
    SerialDenseWrapper::solveSPDSystemMultipleRHS(x, A, b, allowOverwriteOfA);
    
    const double tol = 1e-15;
    
    TEST_COMPARE_FLOATING_ARRAYS(x, xExpected, tol);
  }

  TEUCHOS_UNIT_TEST( SerialDenseWrapper, SolveDiagonalSPD)
  {
    Camellia::FieldContainer<double> A(3,3);
    A(0,0) = 1.0;
    A(1,1) = 2.0;
    A(2,2) = 3.0;
    
    Camellia::FieldContainer<double> b(3,2);
    b(0,0) = 1.0;  b(0,1) = 4.0;
    b(1,0) = 2.0;  b(1,1) = 5.0;
    b(2,0) = 3.0;  b(2,1) = 6.0;
    
    Camellia::FieldContainer<double> x(3,2);
    
    Camellia::FieldContainer<double> xExpected(3,2);
    
    for (int i=0; i<xExpected.dimension(0); i++)
    {
      for (int j=0; j<xExpected.dimension(1); j++)
      {
        xExpected(i,j) = b(i,j) / A(i,i);
      }
    }
    
    bool allowOverwriteOfA = true;
    
    SerialDenseWrapper::solveSPDSystemMultipleRHS(x, A, b, allowOverwriteOfA);
    
    const double tol = 1e-15;
    
    TEST_COMPARE_FLOATING_ARRAYS(x, xExpected, tol);
  }

  TEUCHOS_UNIT_TEST( SerialDenseWrapper, SolveSPD)
  {
    Camellia::FieldContainer<double> A(3,3);
    A(0,0) =  3.0;  A(0,1) = -1.0;  A(0,2) =  0.0;
    A(1,0) = -1.0;  A(1,1) =  3.0;  A(1,2) = -1.0;
    A(2,0) =  0.0;  A(2,1) = -1.0;  A(2,2) =  3.0;
    
    Camellia::FieldContainer<double> xExpected(3,2);
    xExpected(0,0) = 1.0;  xExpected(0,1) = 4;
    xExpected(1,0) = 2.0;  xExpected(1,1) = 5;
    xExpected(2,0) = 3.0;  xExpected(2,1) = 6;
    
    Camellia::FieldContainer<double> b(3,2);
    for (int i=0; i<A.dimension(0); i++)
    {
      for (int rhs=0; rhs<xExpected.dimension(1); rhs++)
      {
        double value = 0;
        for (int j=0; j<A.dimension(1); j++)
        {
          const auto A_ij = A(i,j);
          const auto x_j  = xExpected(j,rhs);
          value += A_ij * x_j;
        }
        b(i,rhs) = value;
      }
    }
    
    Camellia::FieldContainer<double> x(3,2);
    
    bool allowOverwriteOfA = true;
    
    SerialDenseWrapper::solveSPDSystemMultipleRHS(x, A, b, allowOverwriteOfA);
    
    const double tol = 1e-15;
    
    TEST_COMPARE_FLOATING_ARRAYS(x, xExpected, tol);
  }
  
  TEUCHOS_UNIT_TEST( SerialDenseWrapper, Transpose )
  {
    // X = A * B
    int m = 20;
    int n = 30;
    Camellia::FieldContainer<double> A(m,n);
    Camellia::FieldContainer<double> A_T(n,m);
    
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
//    Camellia::FieldContainer<double> A(2000,2000);
//    Camellia::FieldContainer<double> B(2000,2000);
//    Camellia::FieldContainer<double> X(2000,2000), X_expected(2000,2000);
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
