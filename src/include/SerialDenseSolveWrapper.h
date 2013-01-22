//
//  SerialDenseSolveWrapper.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#ifndef Camellia_debug_SerialDenseSolveWrapper_h
#define Camellia_debug_SerialDenseSolveWrapper_h

#include "Intrepid_FieldContainer.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

class SerialDenseSolveWrapper {
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


};

#endif
