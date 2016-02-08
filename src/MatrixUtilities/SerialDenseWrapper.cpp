//
//  SerialDenseWrapper.cpp
//  Camellia
//
//
//

#include "SerialDenseWrapper.h"

#include "Intrepid_FieldContainer.hpp"

#include "Epetra_RowMatrix.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialDenseSVD.h"
#include "Epetra_DataAccess.h"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialQRDenseSolver.hpp"

#include "MPIWrapper.h"

namespace Camellia {
  
  void SerialDenseWrapper::transposeSquareMatrix(Intrepid::FieldContainer<double> &A)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(A.rank() != 2, std::invalid_argument, "A must be rank 2!");
    int rows = A.dimension(0), cols = A.dimension(1);
    TEUCHOS_TEST_FOR_EXCEPTION(rows != cols, std::invalid_argument, "matrix not square");
    for (int i=0; i<rows; i++)
    {
      for (int j=0; j<i; j++)
      {
        double temp = A(i,j);
        A(i,j) = A(j,i);
        A(j,i) = temp;
      }
    }
  }
  
  Epetra_SerialDenseMatrix SerialDenseWrapper::convertFCToSDM(const Intrepid::FieldContainer<double> &A, Epetra_DataAccess CV)
  {
    //  FC is row major, SDM expects column major data, so the roles of rows and columns get swapped
    // distance between rows in FC is the column length (dimension 1)
    int n = A.dimension(0);
    int m = A.dimension(1);
    double *firstEntry = (double *) &A[0]; // a bit dangerous: cast away the const.  Not dangerous if we're doing Copy, of course.
    Epetra_SerialDenseMatrix Amatrix(CV,firstEntry,m,m,n);
    return Amatrix;
  }
  
  void SerialDenseWrapper::convertSDMToFC(Intrepid::FieldContainer<double>& A_fc, const Epetra_SerialDenseMatrix &A)
  {
    int n = A.M();
    int m = A.N();
    Teuchos::Array<int> dim(2);
    dim[0] = m;
    dim[1] = n;
    double * firstEntry = (double *) &A(0,0); // again, casting away the const.  OK since we copy below.
    A_fc = Intrepid::FieldContainer<double>(dim,firstEntry,true); // true: copy
  }\
  
  // gives X = scalarA*A+scalarB*B (overwrites A)
  void SerialDenseWrapper::add(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A, const Intrepid::FieldContainer<double> &B,
                               double scalarA, double scalarB)
  {
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B);
    AMatrix.Scale(scalarA);
    BMatrix.Scale(scalarB);
    AMatrix += BMatrix;
    convertSDMToFC(X,AMatrix);
  }
  
  double SerialDenseWrapper::dot(const Intrepid::FieldContainer<double> &a, const Intrepid::FieldContainer<double> &b)
  {
    if (((a.rank() != 1) && (a.rank() != 2)) || ((b.rank() != 1) && (b.rank() != 2)))
    {
      std::cout << "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)");
    }
    if (a.rank()==2)
    {
      if ((a.dimension(0) != 1) && (a.dimension(1) != 1))
      {
        std::cout << "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)");
      }
    }
    if (b.rank()==2)
    {
      if ((b.dimension(0) != 1) && (b.dimension(1) != 1))
      {
        std::cout << "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "a and b must have rank 1 or 2; if rank 2, one of the two ranks' dimensions must be 1.  (I.e. a and b must both be vectors.)");
      }
    }
    if (b.size() != a.size())
    {
      std::cout << "a and b vectors must have the same length.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "a and b vectors must have the same length.");
    }
    double sum = 0.0;
    for (int i=0; i<a.size(); i++)
    {
      sum += a[i] * b[i];
    }
    return sum;
  }
  
  int SerialDenseWrapper::determinantAndInverse(Intrepid::FieldContainer<double> &detValues, Intrepid::FieldContainer<double> &outInverses,
                                                const Intrepid::FieldContainer<double> &inMatrices)
  {
    // uses LAPACK LU factorization to compute the determinant and inverse of a matrix
    // inMatrices and outMatrices should have shape (C,P,D,D)--this initial implementation is meant for computing Jacobian determinants and inverses
    // detValues should have shape (C,P)
    
    if ((inMatrices.rank() != 4) || (outInverses.rank() != 4))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inMatrices and outInverses must have rank 4");
    }
    
    int numCells = inMatrices.dimension(0);
    int numPoints = inMatrices.dimension(1);
    int spaceDim = inMatrices.dimension(2);
    
    if (inMatrices.dimension(3) != spaceDim)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inMatrices 3rd and 4th dimension must match!");
    }
    
    if ((detValues.rank() != 2) || (detValues.dimension(0) != numCells) || (detValues.dimension(1) != numPoints))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "detValues must have shape (C,P)");
    }
    if ((outInverses.rank() != 4) || (outInverses.dimension(0) != numCells) || (outInverses.dimension(1) != numPoints)
        || (outInverses.dimension(2) != spaceDim) || (outInverses.dimension(3) != spaceDim) )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "outInverses must have shape (C,P,D,D)");
    }
    
    Intrepid::FieldContainer<double> workspace = inMatrices; // copy, since the below will overwrite the matrix data
    
    Intrepid::FieldContainer<int> pivots(spaceDim);
    
    int errOut = 0;
    
    int err;
    
    Teuchos::LAPACK<int, double> lapack;
    
    outInverses.initialize(0); // initial 0's make seeding the identity simpler below
    
    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal < numPoints; ptOrdinal++)
      {
        double *matData = &workspace(cellOrdinal,ptOrdinal,0,0);
        
        lapack.GETRF(spaceDim, spaceDim, matData, spaceDim, &pivots(0), &err);
        
        if (err != 0)
        {
          errOut = err;
        }
        
        // determinant:
        // determinant is the product of the diagonal entries, times 1 or -1 depending on pivots
        int pivotParity = 1;
        double det = 1.0;
        for (int d=0; d<spaceDim; d++)
        {
          if (pivots[d] != d+1) pivotParity *= -1;
          det *= matData[d*spaceDim + d];
        }
        det *= pivotParity;
        detValues(cellOrdinal,ptOrdinal) = det;
        
        // inverse:
        double *outData = &outInverses(cellOrdinal,ptOrdinal,0,0);
        // populate outData matrix with identity so we'll get the inverse out...
        for (int d=0; d<spaceDim; d++)
        {
          outData[d*spaceDim + d] = 1.0;
        }
        
        lapack.GETRS('N', spaceDim, spaceDim, matData, spaceDim, &pivots(0), outData, spaceDim, &err);
        if (err != 0)
        {
          errOut = err;
        }
      }
    }
    return errOut;
  }
  
  int SerialDenseWrapper::extractFCFromEpetra_RowMatrix(const Epetra_RowMatrix &A, Intrepid::FieldContainer<double> &A_fc)
  {
    // this is not the most efficient way to do this; using an Epetra_Importer would probably be faster
    // (I write it this way because it's easier for me to think about this way, and this method is meant primarily for
    //  testing and debugging.)
    A_fc.resize(A.NumGlobalRows(), A.NumGlobalCols());
    // fill in the locally-known entries:
    for (int myRow=0; myRow < A.NumMyRows(); myRow++)
    {
      int i = A.RowMatrixRowMap().GID(myRow); // global row ID
      int numCols = -1;
      int err = A.NumMyRowEntries(myRow, numCols);
      TEUCHOS_TEST_FOR_EXCEPTION(err !=0, std::invalid_argument, "Error encountered during row extraction");
      Intrepid::FieldContainer<int> localColIDs(numCols);
      Intrepid::FieldContainer<double> values(numCols);
      int numColsExtracted = 0;
      if (numCols > 0)
      {
        A.ExtractMyRowCopy(myRow,numCols,numColsExtracted,&values[0],&localColIDs[0]);
        TEUCHOS_TEST_FOR_EXCEPTION(numCols != numColsExtracted, std::invalid_argument, "");
        for (int colOrdinal=0; colOrdinal < numColsExtracted; colOrdinal++)
        {
          int localColID = localColIDs(colOrdinal);
          int j = A.RowMatrixColMap().GID(localColID);
          A_fc(i,j) = values(colOrdinal);
        }
      }
    }
    MPIWrapper::entryWiseSum<double>(A.Comm(),A_fc);
    return 0;
  }
  
  void SerialDenseWrapper::filterMatrix(Intrepid::FieldContainer<double> &filteredMatrix, const Intrepid::FieldContainer<double> &matrix,
                                        const std::set<int> &rowOrdinals, const std::set<int> &colOrdinals)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(matrix.rank() != 2, std::invalid_argument, "matrix must have rank 2");
    
    filteredMatrix.resize(rowOrdinals.size(), colOrdinals.size());
    
    int i=0;
    for (int rowOrdinal : rowOrdinals)
    {
      int j=0;
      for (int colOrdinal : colOrdinals)
      {
        filteredMatrix(i,j) = matrix(rowOrdinal,colOrdinal);
        j++;
      }
      i++;
    }
  }
  
  bool SerialDenseWrapper::matrixIsSymmetric(Intrepid::FieldContainer<double> &A, double relTol, double absTol)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(A.rank() != 2, std::invalid_argument, "A must have rank 2");
    int numRows = A.dimension(0);
    int numCols = A.dimension(1);
    if (numRows != numCols) return false;
    for (int i=0; i<numRows; i++)
    {
      for (int j=0; j<i; j++)
      {
        double absDiff = abs(A(i,j)-A(j,i));
        if (absDiff > absTol)
        {
          double relDiff = absDiff / std::min(abs(A(i,j)),abs(A(j,i)));
          if (relDiff > relTol)
          {
            return false;
          }
        }
      }
    }
    return true;
  }
  
  bool SerialDenseWrapper::matrixIsSymmetric(Intrepid::FieldContainer<double> &A, std::vector<std::pair<int,int>> &asymmetricEntries,
                                             double relTol, double absTol)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(A.rank() != 2, std::invalid_argument, "A must have rank 2");
    asymmetricEntries.clear();
    int numRows = A.dimension(0);
    int numCols = A.dimension(1);
    if (numRows != numCols) return false;
    for (int i=0; i<numRows; i++)
    {
      for (int j=0; j<i; j++)
      {
        double absDiff = abs(A(i,j)-A(j,i));
        if (absDiff > absTol)
        {
          double relDiff = absDiff / std::min(abs(A(i,j)),abs(A(j,i)));
          if (relDiff > relTol)
          {
            asymmetricEntries.push_back({i,j});
          }
        }
      }
    }
    return asymmetricEntries.size() == 0;
  }
  
  // gives X = A*B.  Must pass in 2D arrays, even for vectors!
  void SerialDenseWrapper::multiply(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A,
                                    const Intrepid::FieldContainer<double> &B, char TransposeA, char TransposeB)
  {
    multiplyAndAdd(X,A,B,TransposeA,TransposeB,1.0,0.0);
  }
  
  void SerialDenseWrapper::multiplyFCByWeight(Intrepid::FieldContainer<double> & fc, double weight)
  {
    int size = fc.size();
    double *valuePtr = &fc[0]; // to make this as fast as possible, do some pointer arithmetic...
    for (int i=0; i<size; i++)
    {
      *valuePtr *= weight;
      valuePtr++;
    }
  }
  
  // wrapper for SDM multiply + add routine.  Must pass in 2D arrays, even for vectors!
  // X = ScalarThis*X + ScalarAB*A*B
  void SerialDenseWrapper::multiplyAndAdd(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A,
                                          const Intrepid::FieldContainer<double> &B,
                                          char TransposeA, char TransposeB, double ScalarAB, double ScalarThis)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(X.rank() != 2, std::invalid_argument, "multiplyAndAdd() only supports rank-2 containers.");
    TEUCHOS_TEST_FOR_EXCEPTION(A.rank() != 2, std::invalid_argument, "multiplyAndAdd() only supports rank-2 containers.");
    TEUCHOS_TEST_FOR_EXCEPTION(B.rank() != 2, std::invalid_argument, "multiplyAndAdd() only supports rank-2 containers.");
    int N = X.dimension(0);
    int M = X.dimension(1);
    if ((N==0) || (M==0))
    {
      std::cout << "ERROR: empty result matrix passed in to SerialDenseWrapper::multiplyAndAdd.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "empty result matrix passed in to SerialDenseWrapper::multiplyAndAdd");
    }
    transposeMatrix(X); // SDMs are transposed relative to FCs
    
    int A_cols = (TransposeA=='T') ? A.dimension(0) : A.dimension(1);
    int A_rows = (TransposeA=='T') ? A.dimension(1) : A.dimension(0);
    
    int B_cols = (TransposeB=='T') ? B.dimension(0) : B.dimension(1);
    int B_rows = (TransposeB=='T') ? B.dimension(1) : B.dimension(0);
    
    if (A_cols != B_rows)
    {
      std::cout << "error: A_cols != B_rows\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error: A_cols != B_rows");
    }
    
    if (A_rows != N)
    {
      std::cout << "error: A_rows != X_rows\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error: A_rows != X_rows");
    }
    
    if (B_cols != M)
    {
      std::cout << "error: B_cols != X_cols\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error: B_cols != X_cols");
    }
    
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A,::View);
    Epetra_SerialDenseMatrix BMatrix = convertFCToSDM(B,::View);
    Epetra_SerialDenseMatrix XMatrix = convertFCToSDM(X,::View);
    
    // since SDMs are transposed relative to FCs, reverse the transposal
    char transposeAMatrix = (TransposeA=='T') ? 'N' : 'T';
    char transposeBMatrix = (TransposeB=='T') ? 'N' : 'T';
    
    //    std::cout << "transposeA = " << TransposeA << std::endl;
    //    std::cout << "transposeB = " << TransposeB << std::endl;
    //    std::cout << "A.M() = " << AMatrix.M() << std::endl;
    //    std::cout << "A.N() = " << AMatrix.N() << std::endl;
    //    std::cout << "B.M() = " << BMatrix.M() << std::endl;
    //    std::cout << "B.N() = " << BMatrix.N() << std::endl;
    //    std::cout << "X.M() = " << XMatrix.M() << std::endl;
    //    std::cout << "X.N() = " << XMatrix.N() << std::endl;
    //    std::cout << "AMatrix.LDA() = " << AMatrix.LDA() << std::endl;
    //    std::cout << "BMatrix.LDA() = " << BMatrix.LDA() << std::endl;
    //    std::cout << "XMatrix.LDA() = " << XMatrix.LDA() << std::endl;
    
    int success = XMatrix.Multiply(transposeAMatrix,transposeBMatrix,ScalarAB,AMatrix,BMatrix,ScalarThis);
    
    transposeMatrix(X); // SDMs are transposed relative to FCs
    //    convertSDMToFC(X,XMatrix); // not needed when using View.
    
    if (success != 0)
    {
      std::cout << "Error in SerialDenseWrapper::multiplyAndAdd with error code " << success << std::endl;
      
      std::cout << "A:\n" << A;
      std::cout << "B:\n" << B;
      std::cout << "X:\n" << X;
    }
  }
  
  Intrepid::FieldContainer<double> SerialDenseWrapper::getSubMatrix(Intrepid::FieldContainer<double> &A, std::set<unsigned> &rowIndices,
                                                                    std::set<unsigned> &colIndices, bool warnOfNonzeroOffBlockEntries)
  {
    Intrepid::FieldContainer<double> subMatrix(rowIndices.size(),colIndices.size());
    
    unsigned subrowIndex = 0;
    for (std::set<unsigned>::iterator rowIndexIt = rowIndices.begin(); rowIndexIt != rowIndices.end(); rowIndexIt++, subrowIndex++)
    {
      unsigned subcolIndex = 0;
      for (std::set<unsigned>::iterator colIndexIt = colIndices.begin(); colIndexIt != colIndices.end(); colIndexIt++, subcolIndex++)
      {
        subMatrix(subrowIndex,subcolIndex) = A(*rowIndexIt,*colIndexIt);
      }
    }
    
    if (warnOfNonzeroOffBlockEntries)
    {
      int numRows = A.dimension(0);
      int numCols = A.dimension(1);
      double tol = 1e-14;
      for (std::set<unsigned>::iterator rowIndexIt = rowIndices.begin(); rowIndexIt != rowIndices.end(); rowIndexIt++)
      {
        for (int j=0; j<numCols; j++)
        {
          if (colIndices.find(j) == colIndices.end())
          {
            double val = A(*rowIndexIt,j);
            if (abs(val) > tol)
            {
              std::cout << "WARNING: off-block entry (" << *rowIndexIt << "," << j << ") = " << val << " is non-zero.\n";
            }
          }
        }
      }
      for (std::set<unsigned>::iterator colIndexIt = colIndices.begin(); colIndexIt != colIndices.end(); colIndexIt++)
      {
        for (int i=0; i<numRows; i++)
        {
          if (rowIndices.find(i) == rowIndices.end())
          {
            double val = A(i,*colIndexIt);
            if (abs(val) > tol)
            {
              std::cout << "WARNING: off-block entry (" << i <<  "," << *colIndexIt << ") = " << val << " is non-zero.\n";
            }
          }
        }
      }
    }
    return subMatrix;
  }
  
  void SerialDenseWrapper::roundZeros(Intrepid::FieldContainer<double> &A, double tol)
  {
    for (int i=0; i<A.size(); i++)
    {
      if (abs(A[i]) < tol) A[i] = 0;
    }
  }
  
  void SerialDenseWrapper::scaleBySymmetricDiagonal(Intrepid::FieldContainer<double> &A)
  {
    // requires that A's diagonal be non-negative in every entry
    // the below is a first pass implementation--would be more efficient not to construct the diagonal matrix explicitly
    
    if ((A.rank() != 2) || (A.dimension(0) != A.dimension(1)))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "A must be N x N matrix");
    }
    
    //    std::cout << "A before scaling:\n" << A;
    
    int N = A.dimension(0);
    Intrepid::FieldContainer<double> diag_inv_sqrt(N,N);
    for (int i=0; i<N; i++)
    {
      double diag_ii = A(i,i);
      if (diag_ii < 0)
      {
        std::cout << "A(" << i << "," << i << ") = " << diag_ii << " < 0!\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "diag(A) must be non-negative!\n");
      }
      diag_inv_sqrt(i,i) = 1.0 / sqrt(diag_ii);
    }
    Intrepid::FieldContainer<double> A_temp(N,N); // not sure if we can safely use A while multiplying by A
    multiply(A_temp, diag_inv_sqrt, A);
    multiply(A, A_temp, diag_inv_sqrt);
    
    //    std::cout << "A after scaling:\n" << A;
  }
  
  int SerialDenseWrapper::solveSystem(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                                      Intrepid::FieldContainer<double> &b, bool useATranspose)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N)
    if (b.rank()==1)
    {
      b.resize(b.dimension(0),1);
      x.resize(x.dimension(0),1);
      int result = solveSystemMultipleRHS(x, A, b, useATranspose);
      x.resize(x.dimension(0));
      b.resize(b.dimension(0));
      return result;
    }
    return solveSystemMultipleRHS(x, A, b, useATranspose);
  }
  
  int SerialDenseWrapper::solveSystemLeastSquares(Intrepid::FieldContainer<double> &x,
                                                  const Intrepid::FieldContainer<double> &A,
                                                  const Intrepid::FieldContainer<double> &b)
  {
    // solves Ax = b, where
    // A = (N,M), N >= M
    // b = (N)
    // x = (M)
    //   OR
    // b = (N,L)
    // x = (M,L)
    Intrepid::FieldContainer<double> bCopy = b;
    
    if (bCopy.rank()==1)
    {
      bCopy.resize(b.dimension(0),1);
      x.resize(x.dimension(0),1);
    }
    
    int N = A.dimension(0);
    int M = A.dimension(1);
    int L = bCopy.dimension(1);
    
    if (N < M)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "N < M");
    }
    if (N != b.dimension(0))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "N != b.dimension(0)");
    }
    if (M != x.dimension(0))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x.dimension(0) != M");
    }
    
    Intrepid::FieldContainer<double> A_T = A;
    transposeMatrix(A_T);
    
    Intrepid::FieldContainer<double> ATA(M,M);
    multiply(ATA, A_T, A);
    
    Intrepid::FieldContainer<double> ATb(M,L);
    multiply(ATb, A_T, b);
    
    return solveSystemMultipleRHS(x, ATA, ATb, false);
  }
  
  int SerialDenseWrapper::solveSystemMultipleRHS(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                                                 Intrepid::FieldContainer<double> &b, bool useATranspose)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N, M)
    Epetra_SerialDenseSolver solver;
    
    int N = A.dimension(0);
    int nRHS = b.dimension(1);
    
    // take care of the fact that SDMs are transposed relative to FCs:
    transposeMatrix(b);
    
    if (!useATranspose)
    {
      transposeMatrix(A);
    }
    
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    Epetra_SerialDenseMatrix bVectors = convertFCToSDM(b);
    Epetra_SerialDenseMatrix xVectors(N,nRHS);
    
    solver.SetMatrix(AMatrix);
    int info = solver.SetVectors(xVectors,bVectors);
    if (info!=0)
    {
      std::cout << "solveSystem: failed to SetVectors with error " << info << std::endl;
      return info;
    }
    
    bool equilibrated = false;
    if (solver.ShouldEquilibrate())
    {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }
    
    info = solver.Solve();
    if (info!=0)
    {
      std::cout << "solveSystem: failed to solve with error " << info << std::endl;
      return info;
    }
    
    if (equilibrated)
    {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0)
      {
        std::cout << "solveSystem: unequilibration FAILED with error: " << successLocal << std::endl;
        return successLocal;
      }
    }
    
    // if we transposed A above, transpose it back here--leaving things as we found them
    if (!useATranspose)
    {
      transposeMatrix(A);
    }
    
    transposeMatrix(b);
    
    convertSDMToFC(x,xVectors);
    
    transposeMatrix(x);
    
    return 0;
  }
  
  int SerialDenseWrapper::solveSystemUsingQR(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                                             Intrepid::FieldContainer<double> &b, bool useATranspose, // would be better default to true because that involves no data movement
                                             bool allowOverwriteOfA)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N,M)
    if ((x.rank() != 2) || (A.rank() != 2) || (b.rank() != 2))
    {
      std::cout << "x, A, and b must each be a rank 2 FieldContainer!!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x, A, and b must each be a rank 2 FieldContainer!!");
    }
    if (A.dimension(0) != A.dimension(1))
    {
      std::cout << "A must be square!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "A must be square!");
    }
    if ((A.dimension(0) != b.dimension(0)) || (A.dimension(0) != x.dimension(0)))
    {
      std::cout << "x and b's first dimension must match the dimension of A!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x and b's first dimension must match the dimension of A!");
    }
    
    { // DEBUGGING:
      bool printMatrixInfo = false;
      if (printMatrixInfo)
      {
        int N = A.dimension(0);
        int nnz = 0;
        double tol = 1e-15;
        for (int i=0; i<N; i++)
        {
          for (int j=0; j<N; j++)
          {
            if (abs(A(i,j)) > tol)
            {
              nnz++;
            }
          }
        }
        std::cout << "A has dimensions " << N << " x " << N << " and has " << nnz << " non-zero entries.\n";
      }
    }
    
    int N = A.dimension(0);
    int nRHS = b.dimension(1); // M
    
    int info = 0; // default return value
    
    // transpose data in b container (we'll transpose it back before we return)
    transposeMatrix(b);
    
    // by wrapping the below in braces, we allow the things we allocate there to go out of scope
    // before the transpose of the x FieldContainer, which as presently implemented copies x.  This can
    // take a sizable chunk of memory for higher-order computations, so being able to reclaim the memory
    // allocated for bSDM--which is the same size as x--actually guarantees that we won't run out of memory
    // during the transposition (we would have run out when allocating bSDM if we were that constrained).
    {
      int numCols = N;
      int numRows = N;
      int rowStride = numCols;
      
      Teuchos::DataAccess dataAccessFieldContainer;
      if (!allowOverwriteOfA)
      {
        // we want one copy to be made, into the SDM we actually use in QR solve
        if (!useATranspose)
        {
          dataAccessFieldContainer = Teuchos::View; // will copy during transpose (creation of ASDM)
        }
        else
        {
          dataAccessFieldContainer = Teuchos::Copy; // won't compute transpose: copy now
        }
      }
      else
      {
        dataAccessFieldContainer = Teuchos::View;
      }
      
      Teuchos::SerialDenseMatrix<int, double> ATranspose(dataAccessFieldContainer, &A(0,0), rowStride,
                                                         numCols, numRows); // this is the transpose of A because SDM is column-major
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double>> ASDM;
      if (!useATranspose) // avoid transpose computation when useATranspose = true
        ASDM = Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, double>( ATranspose, Teuchos::TRANS));
      
      Teuchos::SerialDenseMatrix<int, double> bSDM( Teuchos::View, &b(0,0), N, N, nRHS);
      
      // transpose x since SDM is transposed relative to FC:
      x.resize(nRHS,N);
      Teuchos::SerialDenseMatrix<int, double> xSDM( Teuchos::View, &x(0,0), N, N, nRHS);
      
      Teuchos::SerialQRDenseSolver<int,double> qrSolver;
      
      if (useATranspose)
      {
        qrSolver.setMatrix( Teuchos::rcp( &ATranspose, false ) );
      }
      else
      {
        qrSolver.setMatrix( ASDM );
      }
      qrSolver.setVectors( Teuchos::rcp( &xSDM, false ), Teuchos::rcp( &bSDM, false ) );
      info = qrSolver.factor();
      if (info != 0)
      {
        std::cout << "Teuchos::SerialQRDenseSolver::factor() returned : " << info << std::endl;
        // we transposed b above; transpose back
        transposeMatrix(b);

        return info;
      }
      
      info = qrSolver.solve();
      if (info != 0)
      {
        // we transposed b above; transpose back before returning
        transposeMatrix(b);
        
        std::cout << "Teuchos::SerialQRDenseSolver::solve() returned : " << info << std::endl;
        writeMatrixToMatlabFile("/tmp/A.dat", A);
        writeMatrixToMatlabFile("/tmp/b.dat", b);
        std::cout << "wrote matrices to /tmp/A.dat, /tmp/b.dat.\n";
        //      std::cout << "A:\n" << A;
        //      std::cout << "b:\n" << b;

        return info;
      }
    }
  
    // the SDM computation for x has put things in there transposed relative to caller's expectations; transpose
    transposeMatrix(x);
    
    // we transposed b above; transpose back
    transposeMatrix(b);
    
    return 0;
  }

  int SerialDenseWrapper::solveSPDSystemMultipleRHS(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A_SPD,
                                                    Intrepid::FieldContainer<double> &b, bool allowOverwriteOfA)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N)
    Epetra_SerialSpdDenseSolver solver;
    
    int N = A_SPD.dimension(0);
    int nRHS = b.dimension(1);
    
    int result = 0;
    
    Epetra_DataAccess A_dataAccess = allowOverwriteOfA ? ::View : ::Copy;

    Epetra_SerialSymDenseMatrix AMatrix(A_dataAccess, &A_SPD(0,0),
                                        N, // stride -- fc stores in row-major order (a.o.t. SDM)
                                        N);

    
    transposeMatrix(b); // a bit lame that we need to do this...
    Epetra_SerialDenseMatrix bVectors = convertFCToSDM(b);
    transposeMatrix(b); // restore data to be nice to caller (again, a bit lame)
    Epetra_SerialDenseMatrix xVectors(N,nRHS);
    
    solver.SetMatrix(AMatrix);
    int info = solver.SetVectors(xVectors,bVectors);
    if (info!=0)
    {
      result = info;
      std::cout << "solveSPDSystemMultipleRHS: failed to SetVectors with error " << info << std::endl;
      return result;
    }
    
    if ( solver.ShouldEquilibrate() )
    {
      solver.FactorWithEquilibration(true);
      solver.SolveToRefinedSolution(false); // false: don't use iterative refinements...
    }
    info = solver.Factor();
    if (info != 0)
    {
      result = info;
      std::cout << "solveSPDSystemMultipleRHS: Factor failed with code " << result << std::endl;
      return result;
    }
    
    info = solver.Solve();
    
    if (info != 0)
    {
      std::cout << "BilinearForm::optimalTestWeights: Solve FAILED with error: " << info << std::endl;
      result = info;
    }
    
    convertSDMToFC(x,xVectors);
    
    transposeMatrix(x); // perhaps we should change the expected inputs so we don't have to do this?
    
    return result;
  }
  
//  int SerialDenseWrapper::solveSPDSystemMultipleRHS(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A_SPD,
//                                                    Intrepid::FieldContainer<double> &b, bool allowOverwriteOfA)
//  {
//    // solves Ax = b, where
//    // A = (N,N)
//    // x, b = (N)
//    Epetra_SerialSpdDenseSolver solver;
//    
//    x.resize(x.dimension(1),x.dimension(0)); // transpose without moving data (we'll overwrite) -- we'll transpose back (with data movement) below
//    
//    int N = A_SPD.dimension(0);
//    int nRHS = b.dimension(1);
//    
//    int result = 0;
//    
//    Epetra_DataAccess A_dataAccess = allowOverwriteOfA ? ::View : ::Copy;
//    
//    Epetra_SerialSymDenseMatrix AMatrix(A_dataAccess, &A_SPD(0,0),
//                                        N, // stride -- fc stores in row-major order (a.o.t. SDM)
//                                        N);
//
//    transposeMatrix(b); // a bit lame that we need to do this...
//    Epetra_SerialDenseMatrix bVectors = convertFCToSDM(b);
//    transposeMatrix(b); // restore data to be nice to caller (again, a bit lame)
//    
//    Epetra_SerialDenseMatrix xTranspose(::View, &x(0,0), N, N, nRHS);
//    
//    solver.SetMatrix(AMatrix);
//    int info = solver.SetVectors(xTranspose,bVectors);
//    if (info!=0)
//    {
//      result = info;
//      std::cout << "solveSPDSystemMultipleRHS: failed to SetVectors with error " << info << std::endl;
//      return result;
//    }
//    
//    if ( solver.ShouldEquilibrate() )
//    {
//      solver.FactorWithEquilibration(true);
//      solver.SolveToRefinedSolution(false); // false: don't use iterative refinements...
//    }
//    info = solver.Factor();
//    if (info != 0)
//    {
//      result = info;
//      std::cout << "solveSPDSystemMultipleRHS: Factor failed with code " << result << std::endl;
//      return result;
//    }
//    
//    info = solver.Solve();
//    
//    if (info != 0)
//    {
//      std::cout << "BilinearForm::optimalTestWeights: Solve FAILED with error: " << info << std::endl;
//      result = info;
//    }
//    
//    transposeMatrix(x); // perhaps we should change the expected inputs so we don't have to do this?
//    
//    return result;
//  }
  
  void SerialDenseWrapper::transposeMatrix(Intrepid::FieldContainer<double> &A)
  {
    int n = A.dimension(0);
    int m = A.dimension(1);
    if (n==m)
    {
      transposeSquareMatrix(A);
    }
    else
    {
      Intrepid::FieldContainer<double> A_copy = A;
      A.resize(m,n);
      for (int i=0; i<n; i++)
      {
        for (int j=0; j<m; j++)
        {
          A(j,i) = A_copy(i,j);
        }
      }
    }
  }
  
  //! Returns the reciprocal of the 1-norm condition number of the matrix in A
  /*!
   \param A In
   A rank-2 FieldContainer with equal first and second dimensions.
   
   \return the 1-norm condition number if successful; -1 otherwise.
   */
  double SerialDenseWrapper::getMatrixConditionNumber(Intrepid::FieldContainer<double> &A)
  {
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
  
  //! Returns the reciprocal of the 2-norm condition number of the matrix in A
  /*!
   \param A In
   A rank-2 FieldContainer with equal first and second dimensions.
   
   \return the 2-norm condition number if successful; -1 otherwise.
   */
  double SerialDenseWrapper::getMatrixConditionNumber2Norm(Intrepid::FieldContainer<double> &A, bool ignoreZeroEigenvalues)
  {
    Epetra_SerialDenseSVD svd;
    
    int N = A.dimension(0);
    if ((N != A.dimension(1)) || (A.rank() != 2))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "badly shaped matrix");
    }
    
    Epetra_SerialDenseMatrix AMatrix = convertFCToSDM(A);
    
    svd.SetMatrix(AMatrix);
    int result = svd.Factor();
    
    if (result == 0)
    {
      // then the singular values are stored in svd.S_
      double maxSingularValue = svd.S_[0];
      double minSingularValue = svd.S_[N-1];
      double tol = 1e-14;
      if (ignoreZeroEigenvalues)
      {
        int index = N-1;
        while ((abs(minSingularValue) < tol) && (index > 0))
        {
          index--;
          minSingularValue = svd.S_[index];
        }
      }
      if (maxSingularValue < tol)
      {
        //        std::cout << "maxSingularValue is zero for matrix:\n" << A;
      }
      //      std::cout << "minSingularValue: " << minSingularValue << std::endl;
      //      std::cout << "maxSingularValue: " << maxSingularValue << std::endl;
      return maxSingularValue / minSingularValue;
    }
    else
    {
      //      std::cout << "SVD failed for matrix:\n" << A;
      return -1.0;
    }
  }
  
  // These methods for testing positive (semi)definiteness do not work.  As noted in comments below, need to do an eigenvalue solve.
  //  //! Returns true if all the eigenvalues are greater than zero
  //  /*!
  //   \param A In
  //   A rank-2 FieldContainer with equal first and second dimensions.
  //
  //   \return true if all the eigenvalues are positive; false otherwise.
  //   */
  //  static bool matrixIsPositiveDefinite(Intrepid::FieldContainer<double> &A)
  //  {
  //    cout << "WARNING: matrixIsPositiveDefinite() does not work.  Need to set up an eigenvalue solve, not an SVD!\n";
  //    /*
  //     See:
  //     http://www.physics.orst.edu/~rubin/nacphy/lapack/eigen.html
  //     dsytrd - Reduces a symmetric/Hermitian matrix to real symmetric tridiagonal form by an orthogonal/unitary similarity transformation
  //     dsteqr - Computes all eigenvalues and eigenvectors of a real	symmetric tridiagonal matrix, using the implicit QL or QR algorithm
  //
  //     See also:
  //     http://www.netlib.org/lapack/lug/node70.html
  //
  //     */
  //    return getMatrixConditionNumber2Norm(A, false) > 0.0;
  //  }
  //
  //  //! Returns true if all the eigenvalues are greater than or equal to zero
  //  /*!
  //   \param A In
  //   A rank-2 FieldContainer with equal first and second dimensions.
  //
  //   \return true if all the eigenvalues are non-negative; false otherwise.
  //   */
  //  static bool matrixIsPositiveSemiDefinite(Intrepid::FieldContainer<double> &A)
  //  {
  //    cout << "WARNING: matrixIsPositiveSemiDefinite() does not work.  Need to set up an eigenvalue solve, not an SVD!\n";
  //    return getMatrixConditionNumber2Norm(A, true) > 0.0;
  //  }
  
  void SerialDenseWrapper::writeMatrixToMatlabFile(const std::string& filePath, Intrepid::FieldContainer<double> &A)
  {
    int N = A.dimension(0);
    int M = A.dimension(1);
    std::ofstream matrixFile;
    matrixFile.open(filePath.c_str());
    
    for (int i = 0; i<N; i++)
    {
      for (int j = 0; j<M; j++)
      {
        matrixFile << A(i,j) << " ";
      }
      matrixFile << std::endl;
    }
    matrixFile.close();
  }
  
  
  void SerialDenseWrapper::addFCs(Intrepid::FieldContainer<double> &A, const Intrepid::FieldContainer<double> &B,
                                  double B_weight, double A_weight)
  {
    if (A.size() != B.size() )
    {
      std::cout << "addFCs: A and B must have the same size!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "addFCs: Array sizes do not match.\n");
    }
    double* ptrA = &A[0];
    const double* ptrB = &B[0];
    for (int i=0; i<A.size(); i++)
    {
      *ptrA = *ptrA * A_weight + *ptrB * B_weight;
      ptrA++;
      ptrB++;
    }
  }
}