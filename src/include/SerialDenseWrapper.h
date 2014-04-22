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

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

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
  static Epetra_SerialDenseMatrix convertFCToSDM(const FieldContainer<double> &A){
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
  static void convertSDMToFC(FieldContainer<double>& A_fc, const Epetra_SerialDenseMatrix &A){
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
  static void add(FieldContainer<double> &X, const FieldContainer<double> &A, const FieldContainer<double> &B, double scalarA = 1.0, double scalarB = 1.0){
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
  static void multiply(FieldContainer<double> &X, const FieldContainer<double> &A, const FieldContainer<double> &B, char TransposeA = 'N', char TransposeB = 'N'){
    multiplyAndAdd(X,A,B,TransposeA,TransposeB,1.0,0.0);
  }

  // wrapper for SDM multiply + add routine.  Must pass in 2D arrays, even for vectors! 
  // X = ScalarThis*X + ScalarAB*A*B
  static void multiplyAndAdd(FieldContainer<double> &X, const FieldContainer<double> &A, const FieldContainer<double> &B, char TransposeA, char TransposeB, double ScalarAB, double ScalarThis){
    int N = X.dimension(0);
    int M = X.dimension(1);
    if ((N==0) || (M==0)) {
      cout << "ERROR: empty result matrix passed in to SerialDenseWrapper::multiplyAndAdd.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "empty result matrix passed in to SerialDenseWrapper::multiplyAndAdd");
    }
    
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B);
    Epetra_SerialDenseMatrix XMatrix = convertFCToSDM(X);
    
    int success = XMatrix.Multiply(TransposeA,TransposeB,ScalarAB,AMatrix,BMatrix,ScalarThis);

    convertSDMToFC(X,XMatrix);
    
    
    if (success != 0){
      cout << "Error in SerialDenseWrapper::multiplyAndAdd with error code " << success << endl;
      
      cout << "A:\n" << A;
      cout << "B:\n" << B;
      cout << "X:\n" << X;
    }
  }
  
  static FieldContainer<double> getSubMatrix(FieldContainer<double> &A, set<unsigned> &rowIndices, set<unsigned> &colIndices,
                                             bool warnOfNonzeroOffBlockEntries = false) {
    FieldContainer<double> subMatrix(rowIndices.size(),colIndices.size());
    
    unsigned subrowIndex = 0;
    for (set<unsigned>::iterator rowIndexIt = rowIndices.begin(); rowIndexIt != rowIndices.end(); rowIndexIt++, subrowIndex++) {
      unsigned subcolIndex = 0;
      for (set<unsigned>::iterator colIndexIt = colIndices.begin(); colIndexIt != colIndices.end(); colIndexIt++, subcolIndex++) {
        subMatrix(subrowIndex,subcolIndex) = A(*rowIndexIt,*colIndexIt);
      }
    }
    
    if (warnOfNonzeroOffBlockEntries) {
      int numRows = A.dimension(0);
      int numCols = A.dimension(1);
      double tol = 1e-14;
      for (set<unsigned>::iterator rowIndexIt = rowIndices.begin(); rowIndexIt != rowIndices.end(); rowIndexIt++) {
        for (int j=0; j<numCols; j++) {
          if (colIndices.find(j) == colIndices.end()) {
            double val = A(*rowIndexIt,j);
            if (abs(val) > tol) {
              cout << "WARNING: off-block entry (" << *rowIndexIt << "," << j << ") = " << val << " is non-zero.\n";
            }
          }
        }
      }
      for (set<unsigned>::iterator colIndexIt = colIndices.begin(); colIndexIt != colIndices.end(); colIndexIt++) {
        for (int i=0; i<numRows; i++) {
          if (rowIndices.find(i) == rowIndices.end()) {
            double val = A(i,*colIndexIt);
            if (abs(val) > tol) {
              cout << "WARNING: off-block entry (" << i <<  "," << *colIndexIt << ") = " << val << " is non-zero.\n";
            }
          }
        }
      }
    }
    return subMatrix;
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

  static int solveSystemMultipleRHS(FieldContainer<double> &x, FieldContainer<double> &A, FieldContainer<double> &b, bool useATranspose = false){
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
      return info;
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
      return info;
    }
    
    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "solveSystem: unequilibration FAILED with error: " << successLocal << endl;
        return successLocal;
      }
    }
    
    convertSDMToFC(x,xVectors);
    return 0;
  }
  
  static int solveSPDSystemMultipleRHS(FieldContainer<double> &x, FieldContainer<double> &A_SPD, FieldContainer<double> &b){
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N)
    Epetra_SerialSpdDenseSolver solver;
    
    int N = A_SPD.dimension(0);
    int nRHS = b.dimension(1);
    
    int result = 0;
    
    Epetra_SerialSymDenseMatrix AMatrix(Copy, &A_SPD(0,0),
                                          N, // stride -- fc stores in row-major order (a.o.t. SDM)
                                          N);
    
    Epetra_SerialDenseMatrix bVectors = convertFCToSDM(b);
    Epetra_SerialDenseMatrix xVectors(N,nRHS);
    
    solver.SetMatrix(AMatrix);
    int info = solver.SetVectors(xVectors,bVectors);
    if (info!=0){
      result = info;
      cout << "solveSPDSystemMultipleRHS: failed to SetVectors with error " << info << endl;
      return result;
    }
    
    if ( solver.ShouldEquilibrate() ) {
      solver.FactorWithEquilibration(true);
      solver.SolveToRefinedSolution(false); // false: don't use iterative refinements...
    }
    info = solver.Factor();
    if (info != 0) {
      result = info;
      cout << "solveSPDSystemMultipleRHS: Factor failed with code " << result << endl;
      return result;
    }
    
    info = solver.Solve();
    
    if (info != 0) {
      cout << "BilinearForm::optimalTestWeights: Solve FAILED with error: " << info << endl;
      result = info;
    }
    
    convertSDMToFC(x,xVectors);
    
    return result;
  }

  static double getMatrixConditionNumber(FieldContainer<double> &A) {
    int N = A.dimension(0);
    int M = A.dimension(1);
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseSolver solver;
    solver.SetMatrix(AMatrix); 
    double invCond;
    int result = solver.ReciprocalConditionEstimate(invCond);
    if (result == 0) // success
      return 1.0/invCond;
    else // failure
      return -1;
  }

  static void writeMatrixToMatlabFile(const string& filePath, FieldContainer<double> &A){
    int N = A.dimension(0);
    int M = A.dimension(1);
    ofstream matrixFile;
    matrixFile.open(filePath.c_str());
    
    for (int i = 0;i<N;i++){
      for (int j = 0;j<M;j++){
        matrixFile << A(i,j) << " ";
      }
      matrixFile << endl;
    }
    matrixFile.close();
  }
};

#endif
