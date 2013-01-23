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
  
public:
  // gives X = A*B
  static void multiply(FieldContainer<double> &X, FieldContainer<double> &A, FieldContainer<double> &B, char TransposeA = 'N', char TransposeB = 'N'){
    int N = B.dimension(0);
    int nRHS = B.dimension(1);
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B);
    Epetra_SerialDenseMatrix XMatrix(N,nRHS);    

    XMatrix.Multiply(TransposeA,TransposeB,1.0,AMatrix,BMatrix,0.0);
    
    for (int i=0;i<N;i++){
      for (int j=0;j<nRHS;j++){
	X(i,j) = XMatrix(i,j);
      }
    }
  }

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

    for (int i=0;i<N;i++){
      for (int j=0;j<nRHS;j++){
	x(i,j) = xVectors(i,j);
      }
    }
  }


};

#endif
