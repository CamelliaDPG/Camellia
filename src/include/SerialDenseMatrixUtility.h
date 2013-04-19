//
//  SerialDenseMatrixUtility.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#ifndef Camellia_debug_SerialDenseMatrixUtility_h
#define Camellia_debug_SerialDenseMatrixUtility_h

#include "Intrepid_FieldContainer.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialDenseSVD.h"
#include "Epetra_DataAccess.h"

class SerialDenseMatrixUtility {
  static void transposeSquareMatrix(FieldContainer<double> &A) {
    int rows = A.dimension(0), cols = A.dimension(1);
    TEUCHOS_TEST_FOR_EXCEPTION(rows != cols, std::invalid_argument, "matrix not square");
    for (int i=0; i<rows; i++) {
      for (int j=0; j<i; j++) {
        double temp = A(i,j);
        A(i,j) = A(j,i);
        A(j,i) = temp;
      }
    }
  }
  
public:
  static void solveSystem(FieldContainer<double> &x, FieldContainer<double> &A, FieldContainer<double> &b, bool useATranspose = false) {
    // solves Ax = b, where
    // A = (N,N)
    // b = N
    // x = N
    Epetra_SerialDenseSolver solver;
    
    int N = A.dimension(0);
    
    if (! useATranspose) {
      transposeSquareMatrix(A); // FCs are in row-major order, so we swap for compatibility with SDM
    }
    
    Epetra_SerialDenseMatrix AMatrix(Copy,
                                     &A(0,0),
                                     A.dimension(1),
                                     A.dimension(1),
                                     A.dimension(0)); // stride -- fc stores in row-major order (a.o.t. SDM)
        
    Epetra_SerialDenseVector bVector(Copy,
                                     &b(0),
                                     b.dimension(0));
    
    Epetra_SerialDenseVector xVector(x.dimension(0));
   
    solver.SetMatrix(AMatrix);
    int info = solver.SetVectors(xVector,bVector);
    if (info!=0){
      cout << "solveSystem: failed to SetVectors with error " << info << endl;
    }
    
    bool equilibrated = false;
    if (solver.ShouldEquilibrate()){
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }
    
    info = solver.Solve();
    if (info!=0){
      cout << "solveSystem: failed to solve with error " << info << endl;
    }
    
    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "solveSystem: unequilibration FAILED with error: " << successLocal << endl;
      }
    }
    
    for (int i=0;i<N;i++){
      x(i) = xVector(i);
    }
    
    // now that we're done, if we transposed, reverse the operation
    if (! useATranspose) {
      transposeSquareMatrix(A); // FCs are in row-major order, so we swap for compatibility with SDM
    }
  }

  static void solveSystemMultipleRHS(FieldContainer<double> &x, FieldContainer<double> &A, FieldContainer<double> &b, bool useATranspose = false){
    // solves Ax = b, where
    // A = (N,N)
    // b = N
    // x = N
    Epetra_SerialDenseSolver solver;
    
    int N = A.dimension(0);
    int nRHS = b.dimension(1);

    if (! useATranspose) {
      transposeSquareMatrix(A); // FCs are in row-major order, so we swap for compatibility with SDM
    }
    
    Epetra_SerialDenseMatrix AMatrix(Copy,
                                     &A(0,0),
                                     A.dimension(0),
                                     A.dimension(1),
                                     A.dimension(0)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    /*
    Epetra_SerialDenseMatrix bVectors(Copy,
				      &b(0,0),
				      N,
				      b.dimension(0),b.dimension(1));
    */
    // TODO: figure out why the above COPY doesn't work
    Epetra_SerialDenseMatrix bVectors(N,nRHS);
    for (int i = 0;i<N;i++){
      for (int j = 0;j<nRHS;j++){
        bVectors(i,j) = b(i,j);
      }
    }

    Epetra_SerialDenseMatrix xVectors(N,nRHS);
   
    solver.SetMatrix(AMatrix);
    int info = solver.SetVectors(xVectors,bVectors);
    if (info!=0){
      cout << "solveSystem: failed to SetVectors with error " << info << endl;
    }
    
    bool equilibrated = false;
    if (solver.ShouldEquilibrate()){
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }
    
    info = solver.Solve();
    if (info!=0){
      cout << "solveSystem: failed to solve with error " << info << endl;
    }
    
    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "solveSystem: unequilibration FAILED with error: " << successLocal << endl;
      }
    }

    for (int i=0;i<N;i++){
      for (int j=0;j<nRHS;j++){
        x(i,j) = xVectors(i,j);
      }
    }
  }
  
  static void jacobiScaleMatrix(FieldContainer<double> &A) {
    int N = A.dimension(0);
    if ((N != A.dimension(1)) || (A.rank() != 2)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "badly shaped matrix");
    }
    for (int i=0; i<N; i++) {
      double diag = A(i,i);
      double *val = &A(i,0);
      for (int j=0; j<N; j++) {
        *val /= diag;
        val++;
      }
    }
  }

  static double estimate2NormConditionNumber(FieldContainer<double> &A, bool ignoreZeroEigenvalues = true) {
    Epetra_SerialDenseSVD svd;
    
    int N = A.dimension(0);
    if ((N != A.dimension(1)) || (A.rank() != 2)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "badly shaped matrix");
    }
    
    transposeSquareMatrix(A); // FCs are in row-major order, so we swap for compatibility with SDM
    
    Epetra_SerialDenseMatrix AMatrix(Copy,
                                     &A(0,0),
                                     A.dimension(0),
                                     A.dimension(1),
                                     A.dimension(0)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    svd.SetMatrix(AMatrix);
    int result = svd.Factor();
    
    if (result == 0) {
      // then the singular values are stored in svd.S_
      double maxSingularValue = svd.S_[0];
      double minSingularValue = svd.S_[N-1];
      double tol = 1e-14;
      if (ignoreZeroEigenvalues) {
        int index = N-1;
        while ((abs(minSingularValue) < tol) && (index > 0)) {
          index--;
          minSingularValue = svd.S_[index];
        }
      }
      if (maxSingularValue < tol) {
        cout << "maxSingularValue is zero for matrix:\n" << A;
      }
      return maxSingularValue / minSingularValue;
    } else {
      cout << "SVD failed for matrix:\n" << A;
      return -1.0;
    }
  }
  
  
  static double estimate1NormConditionNumber(FieldContainer<double> &A, bool useATranspose = false) {
    Epetra_SerialDenseSolver solver;
    
    int N = A.dimension(0);
    if ((N != A.dimension(1)) || (A.rank() != 2)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "badly shaped matrix");
    }
    
    if (! useATranspose) {
      transposeSquareMatrix(A); // FCs are in row-major order, so we swap for compatibility with SDM
    }
    
    Epetra_SerialDenseMatrix AMatrix(Copy,
                                     &A(0,0),
                                     A.dimension(0),
                                     A.dimension(1),
                                     A.dimension(0)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    solver.SetMatrix(AMatrix);
      
    double rcond=0;
    double result = solver.ReciprocalConditionEstimate(rcond);
    
//    // experimental code: try equilibriating first.  Just output that result to console for now.
//    solver.EquilibrateMatrix();
//    double rcond2;
//    double result2 = solver.ReciprocalConditionEstimate(rcond2);
//    cout << "1/rcond2 = " << 1.0 / rcond2 << endl;
    
    if (result == 0) // success
      return 1.0 / rcond;
    else
      return -1.0;
  }
};

#endif
