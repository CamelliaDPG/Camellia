//
//  SerialDenseWrapper.h
//  Camellia-debug
//
//
//

#ifndef SerialDenseWrapper_h
#define SerialDenseWrapper_h

#include "Intrepid_FieldContainer.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

class SerialDenseWrapper {
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
  static Epetra_SerialDenseMatrix convertFCToSDM(FieldContainer<double> &A){
    int n = A.dimension(0);
    int m = A.dimension(1);
    Epetra_SerialDenseMatrix Amatrix(n,m);
    for (int i = 0;i<n;i++){
      for (int j = 0;j<m;j++){
        Amatrix(i,j) = A(i,j);
      }
    }
    return Amatrix;
  }
  static void convertSDMToFC(FieldContainer<double>& A_fc, Epetra_SerialDenseMatrix &A){
    int n = A.M();    
    int m = A.N();
    A_fc.resize(n,m);
    for (int i = 0;i<n;i++){
      for (int j = 0;j<m;j++){
        A_fc(i,j) = A(i,j);
      }
    }
  }
  
public:
  // gives X = scalarA*A+scalarB*B (overwrites A)
  static void add(FieldContainer<double> &X, FieldContainer<double> &A, FieldContainer<double> &B, double scalarA = 1.0, double scalarB = 1.0){
    int N = A.dimension(0);
    int M = A.dimension(1);
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B);
    AMatrix.Scale(scalarA);
    BMatrix.Scale(scalarB);
    AMatrix += BMatrix;
    convertSDMToFC(X,AMatrix);
  }

  // gives X = A*B.  Must pass in 2D arrays, even for vectors! 
  static void multiply(FieldContainer<double> &X, FieldContainer<double> &A, FieldContainer<double> &B, char TransposeA = 'N', char TransposeB = 'N'){  
    multiplyAndAdd(X,A,B,TransposeA,TransposeB,1.0,0.0);
  }

  // wrapper for SDM multiply + add routine.  Must pass in 2D arrays, even for vectors! 
  // X = ScalarThis*X + ScalarAB*A*B
  static void multiplyAndAdd(FieldContainer<double> &X, FieldContainer<double> &A, FieldContainer<double> &B, char TransposeA, char TransposeB, double ScalarAB, double ScalarThis){
    int N = X.dimension(0);
    int M = X.dimension(1);
    
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B);
    Epetra_SerialDenseMatrix XMatrix = convertFCToSDM(X);
    
    int success = XMatrix.Multiply(TransposeA,TransposeB,ScalarAB,AMatrix,BMatrix,ScalarThis);
    if (success != 0){
      cout << "Error in SerialDenseWrapper::multiplyAndAdd with error code " << success << endl;
    }

    convertSDMToFC(X,XMatrix);
  }

  static void solveSystem(FieldContainer<double> &x, FieldContainer<double> &A, FieldContainer<double> &b, bool useATranspose = false) {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N)
    if (b.rank()==1){
      b.resize(b.dimension(0),1);
      x.resize(x.dimension(0),1);
      solveSystemMultipleRHS(x, A, b, useATranspose);
      x.resize(x.dimension(0));
    }
  }

  static void solveSystemMultipleRHS(FieldContainer<double> &x, FieldContainer<double> &A, FieldContainer<double> &b, bool useATranspose = false){
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N)
    Epetra_SerialDenseSolver solver;
    
    int N = A.dimension(0);
    int nRHS = b.dimension(1);
 
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix bVectors = convertFCToSDM(b);
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
    
    convertSDMToFC(x,xVectors);
  }

  static double getMatrixConditionNumber(FieldContainer<double> &A){
    int N = A.dimension(0);
    int M = A.dimension(1);
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseSolver solver;
    solver.SetMatrix(AMatrix); 
    double invCond;
    solver.ReciprocalConditionEstimate(invCond);
    return 1.0/invCond;
  }

};

#endif
