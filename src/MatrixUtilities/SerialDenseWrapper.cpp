//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  SerialDenseWrapper.cpp
//  Camellia
//
//
//

#include "SerialDenseWrapper.h"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_matrix.hpp"

#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialQRDenseSolver.hpp"

#include "MPIWrapper.h"

namespace Camellia {
  
  void SerialDenseWrapper::transposeSquareMatrix(Camellia::FieldContainer<double> &A)
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
  
  double SerialDenseWrapper::condest(const Camellia::FieldContainer<double> &A, bool assumeSPD)
  {
    // copy, since LAPACK will want to overwrite
    Camellia::FieldContainer<double> A_copy = A;
    
    Teuchos::LAPACK<int, double> lapack;
    Teuchos::BLAS<int, double> blas;

    int N = A.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != A.dimension(1), std::invalid_argument, "A must be square!");
    
    int result = 0;
    int INFO;

    double A_norm = 0; // 1-norm of A (max column sum)
    for (int i=0; i<N; i++)
    {
      double col_sum = 0;
      for (int j=0; j<N; j++)
      {
        col_sum += A(i,j);
      }
      A_norm = std::max(A_norm, abs(col_sum));
    }
    
    double condest;
    
    if (assumeSPD)
    {
      char UPLO = 'L'; // lower-triangular
      
      // factor
      lapack.POTRF(UPLO, N, &A_copy[0], N, &INFO);
      
      if (INFO != 0) result = INFO;
      
      Camellia::FieldContainer<double> WORK(3*N);
      Camellia::FieldContainer<int> IWORK(N);
      
      double rcond;
      lapack.POCON(UPLO, N, &A_copy[0], N, A_norm, &rcond, &WORK[0], &IWORK[0], &INFO);
      
      if (INFO != 0) result = INFO;
      
      if (result != 0) condest = -1.0; // indication there was an error
      else condest = 1.0 / rcond;
    }
    else
    {
      Camellia::FieldContainer<int> IPIV(N);
      
      // factor
      lapack.GETRF(N, N, &A_copy[0], N, &IPIV[0], &INFO);
      
      if (INFO != 0) result = INFO;
      
      Camellia::FieldContainer<double> WORK(4*N);
      Camellia::FieldContainer<int> IWORK(N);

      char NORM = '1'; // 1-norm
      double rcond;
      
      lapack.GECON(NORM, N, &A_copy[0], N, A_norm, &rcond, &WORK[0], &IWORK[0], &INFO);
      
      if (INFO != 0) result = INFO;
      
      if (result != 0) condest = -1.0; // indication there was an error
      else condest = 1.0 / rcond;
    }
    return condest;
  }
  
  //! Mimics what Epetra SerialDenseMatrix did.
  bool SerialDenseWrapper::computeEquilibrationScaling(Teuchos::SerialDenseMatrix<int, double> &A,
                                                       std::vector<double> &rowScalings, std::vector<double> &colScalings)
  {
    bool shouldEquilibrate = false;
    
    const int M = A.numRows();
    const int N = A.numCols();
    rowScalings.resize(M);
    colScalings.resize(N);
    
    double ROWCND, COLCND, AMAX;
    int INFO;
    
    Teuchos::LAPACK<int,double> lapack;
    
    lapack.GEEQU(M, N, A.values(), A.stride(),
                 rowScalings.data(), colScalings.data(),
                 &ROWCND, &COLCND, &AMAX, &INFO);
    if (INFO != 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GEEQU returned error code " << INFO);
    }
    
    const double Epetra_Overflow = 1.79E308; // Used to test if equilibration should be done.
    const double Epetra_Underflow = 2.23E-308;

    if (COLCND < 0.1 || ROWCND < 0.1 || AMAX < Epetra_Underflow || AMAX > Epetra_Overflow)
    {
      shouldEquilibrate = true;
    }

    return shouldEquilibrate;
  }

  bool SerialDenseWrapper::computeEquilibrationScalingSPD(Teuchos::SerialDenseMatrix<int, double> &A_SPD, std::vector<double> &scalings)
  {
    const int N = A_SPD.numRows();
    scalings.resize(N);
        
    double SCOND, AMAX;
    int INFO;

    Teuchos::LAPACK<int,double> lapack;
    
    lapack.POEQU(N, A_SPD.values(), A_SPD.stride(), scalings.data(), &SCOND, &AMAX, &INFO);
    if (INFO != 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "POEQU returned error code " << INFO);
    }
    
    bool shouldEquilibrate = false;

    const double Epetra_Overflow = 1.79E308; // Used to test if equilibration should be done.
    const double Epetra_Underflow = 2.23E-308;
    
    if (SCOND < 0.1 || AMAX < Epetra_Underflow || AMAX > Epetra_Overflow)
    {
      shouldEquilibrate = true;
    }

    return shouldEquilibrate;
  }

  Teuchos::SerialDenseMatrix<int, double> SerialDenseWrapper::convertFCToSDM(const Camellia::FieldContainer<double> &A, bool transpose, Teuchos::DataAccess CV)
  {
    bool hasAUnitDimension = (A.dimension(0) == 1) || (A.dimension(1) == 1);
    if (!transpose)
    {
      // then we must make a copy, UNLESS one of the dimensions is 1
      if (CV == Teuchos::View)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(!hasAUnitDimension, std::invalid_argument,
                                   "Can only convert this FC with CV=Teuchos::Copy");
      }
    }
    
    if (A.rank() != 2)
    {
      std::cout << "A must be a rank 2 Camellia::FieldContainer\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x, A, and b must each be a rank 2 Camellia::FieldContainer!!");
    }
    const int A_rows = A.dimension(0);
    const int A_cols = A.dimension(1);
    
    int dataStride = A_cols; // A is row-major
    
    int info = 0; // default return value
    
    Teuchos::RCP<const Camellia::FieldContainer<double>> matrixPtr; // points to either A or its transpose, depending on what we need for
    if (!transpose && !hasAUnitDimension)
    {
      auto newMatrix = new FieldContainer<double>(A);
      transposeMatrix(*newMatrix);
      dataStride = A_rows;
      matrixPtr = Teuchos::rcp(newMatrix);
    }
    else
    {
      matrixPtr = Teuchos::rcp(&A, false);
      if (!transpose && hasAUnitDimension)
      {
        dataStride = A_rows;
      }
    }
    
    const int SDM_cols = transpose ? A_rows : A_cols;
    const int SDM_rows = transpose ? A_cols : A_rows;
    
    double* data = (A_rows * A_cols > 0) ? const_cast<double*>(&(*matrixPtr)(0,0)) : NULL;
    
    Teuchos::SerialDenseMatrix<int, double> A_SDM(CV, data, dataStride, SDM_rows, SDM_cols);
    return A_SDM;
  }

  void SerialDenseWrapper::convertSDMToFC(Camellia::FieldContainer<double> &A_fc, const Teuchos::SerialDenseMatrix<int, double> &A, bool transpose)
  {
    int m = A.numRows();
    int n = A.numCols();
    Teuchos::Array<int> dim(2);
    dim[0] = m;
    dim[1] = n;
    const double * A_Entry = &A(0,0);
    if (transpose)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(A_fc.dimension(0) != n, std::invalid_argument, "A_fc dimension 0 must match A.N()");
      TEUCHOS_TEST_FOR_EXCEPTION(A_fc.dimension(1) != m, std::invalid_argument, "A_fc dimension 1 must match A.M()");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(A_fc.dimension(0) != m, std::invalid_argument, "A_fc dimension 0 must match A.M()");
      TEUCHOS_TEST_FOR_EXCEPTION(A_fc.dimension(1) != n, std::invalid_argument, "A_fc dimension 1 must match A.N()");
    }
    
    double * A_fc_Entry = (double*) &A_fc(0,0);
    // deep copy, respecting whatever ownership that A_fc was initialized with
    memcpy(A_fc_Entry, A_Entry, n*m*sizeof(double));
    
    if (!transpose)
    {
      A_fc.resize(A_fc.dimension(1),A_fc.dimension(0));
      transposeMatrix(A_fc);
    }
  }

  // gives X = scalarA*A+scalarB*B (overwrites A)
  void SerialDenseWrapper::add(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A, const Camellia::FieldContainer<double> &B,
                               double scalarA, double scalarB)
  {
    auto AMatrix = convertFCToSDM(A);
    auto BMatrix = convertFCToSDM(B);
    AMatrix.scale(scalarA);
    BMatrix.scale(scalarB);
    AMatrix += BMatrix;
    convertSDMToFC(X,AMatrix);
  }
  
  double SerialDenseWrapper::dot(const Camellia::FieldContainer<double> &a, const Camellia::FieldContainer<double> &b)
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
  
  int SerialDenseWrapper::determinantAndInverse(Camellia::FieldContainer<double> &detValues, Camellia::FieldContainer<double> &outInverses,
                                                const Camellia::FieldContainer<double> &inMatrices)
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
    
    if ((spaceDim >= 1) && (spaceDim <= 3))
    {
      for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal < numPoints; ptOrdinal++)
        {
          if (spaceDim == 1)
          {
            const double & value                   = inMatrices(cellOrdinal,ptOrdinal,0,0);
            detValues(cellOrdinal,ptOrdinal)       = value;
            outInverses(cellOrdinal,ptOrdinal,0,0) = 1.0 / value;
          }
          else if (spaceDim == 2)
          {
            Matrix<double, 2, 2> mat2, mat2_inv;
            for (int d1=0; d1<spaceDim; d1++)
            {
              for (int d2=0; d2<spaceDim; d2++)
              {
                mat2(d1,d2) = inMatrices(cellOrdinal,ptOrdinal,d1,d2);
              }
            }
            detValues(cellOrdinal,ptOrdinal) = determinant(mat2);
            mat2_inv = invert(mat2);
            for (int d1=0; d1<spaceDim; d1++)
            {
              for (int d2=0; d2<spaceDim; d2++)
              {
                outInverses(cellOrdinal,ptOrdinal,d1,d2) = mat2_inv(d1,d2);
              }
            }
          }
          else if (spaceDim == 3)
          {
            Matrix<double, 3, 3> mat3, mat3_inv;
            for (int d1=0; d1<spaceDim; d1++)
            {
              for (int d2=0; d2<spaceDim; d2++)
              {
                mat3(d1,d2) = inMatrices(cellOrdinal,ptOrdinal,d1,d2);
              }
            }
            detValues(cellOrdinal,ptOrdinal) = determinant(mat3);
            mat3_inv = invert(mat3);
            for (int d1=0; d1<spaceDim; d1++)
            {
              for (int d2=0; d2<spaceDim; d2++)
              {
                outInverses(cellOrdinal,ptOrdinal,d1,d2) = mat3_inv(d1,d2);
              }
            }
          }
        }
      }
      return 0; // success
    }
    else
    {
      // more general, LAPACK-based version
      Camellia::FieldContainer<double> workspace = inMatrices; // copy, since the below will overwrite the matrix data
      
      Camellia::FieldContainer<int> pivots(spaceDim);
      
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
  }

  int SerialDenseWrapper::eigenvalues(const Camellia::FieldContainer<double> &A,
                                      Camellia::FieldContainer<double> &lambda_real, Camellia::FieldContainer<double> &lambda_imag)
  {
    Camellia::FieldContainer<double> eigvectors;
    bool computeRight = true; // ignored for empty eigvectors container
    return eigenvalues(A, lambda_real, lambda_imag, eigvectors, computeRight);
  }
  
  int SerialDenseWrapper::eigenvalues(const Camellia::FieldContainer<double> &A,
                                      Camellia::FieldContainer<double> &lambda_real,
                                      Camellia::FieldContainer<double> &lambda_imag,
                                      Camellia::FieldContainer<double> &eigvectors, // if empty, skips eigvector determination
                                      bool computeRight)
  {
//    GEEV (const char JOBVL, const char JOBVR, const OrdinalType n, ScalarType *A, const OrdinalType lda, MagnitudeType *WR, MagnitudeType *WI, ScalarType *VL, const OrdinalType ldvl, ScalarType *VR, const OrdinalType ldvr, ScalarType *WORK, const OrdinalType lwork, MagnitudeType *RWORK, OrdinalType *info) const
    
    int N = A.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != A.dimension(1), std::invalid_argument, "A must be square");
    TEUCHOS_TEST_FOR_EXCEPTION(N != lambda_real.dimension(0), std::invalid_argument, "lambda_real must be of length N");
    TEUCHOS_TEST_FOR_EXCEPTION(N != lambda_imag.dimension(0), std::invalid_argument, "lambda_imag must be of length N");
    
    Teuchos::LAPACK<int, double> lapack;
    char JOBVL = 'N'; // don't compute left eigenvectors
    char JOBVR = 'N'; // don't compute right eigenvectors
    
    double* VL = NULL;
    double* VR = NULL;
    
    if (eigvectors.size() != 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(eigvectors.dimension(0) != N, std::invalid_argument, "eigvectors must either be empty or have dimension N x N");
      TEUCHOS_TEST_FOR_EXCEPTION(eigvectors.dimension(1) != N, std::invalid_argument, "eigvectors must either be empty or have dimension N x N");
      if (computeRight)
      {
        JOBVR = 'V';
        VR = &eigvectors[0];
      }
      else
      {
        JOBVL = 'V';
        VL = &eigvectors[0];
      }
    }
    
    Camellia::FieldContainer<double> A_copy = A; // A_copy will be overwritten; therefore we can't use the const input A
    int LDA = N;
    int LDVL = N;
    int LDVR = N;
    int LWORK = 4*N;
    Camellia::FieldContainer<double> WORK(LWORK);
    int INFO;
    lapack.GEEV(JOBVL, JOBVR, N, &A_copy[0], LDA, &lambda_real[0], &lambda_imag[0], VL, LDVL, VR, LDVR, &WORK[0], LWORK, &INFO);
    return INFO;
  }
  
  void SerialDenseWrapper::equilibrateMatrix(Teuchos::SerialDenseMatrix<int, double> &A,
                                             const std::vector<double> &rowScalings, const std::vector<double> &colScalings)
  {
    auto M   = A.numRows();
    auto N   = A.numCols();
    auto LDA = A.stride();
    double * ptr;
    for (int j=0; j<N; j++) {
      ptr = A.values() + j*LDA;
      const double &s1 = colScalings[j];
      for (int i=0; i<M; i++) {
        *ptr = *ptr*s1*rowScalings[i];
        ptr++;
      }
    }
  }

  void SerialDenseWrapper::equilibrateRHS(Teuchos::SerialDenseMatrix<int, double> &b, const std::vector<double> &rowScalings)
  {
    auto M     = b.numRows();
    auto NRHS  = b.numCols();
    auto LDB = b.stride();
    
    double * ptr;
    for (int j=0; j<NRHS; j++) {
      ptr = b.values() + j*LDB;
      for (int i=0; i<M; i++) {
        *ptr = *ptr*rowScalings[i];
        ptr++;
      }
    }
  }

  int SerialDenseWrapper::extractFCFromTMatrix(TMatrixPtr<double> A, Camellia::FieldContainer<double> &A_fc)
  {
    int myRank = A->getComm()->getRank();
    auto numRowsGlobal = A->getGlobalNumRows();
    auto numMyRows = (myRank == 0) ? numRowsGlobal : 0;
    
    TMapPtr importMap = Teuchos::rcp( new TMap(numRowsGlobal, numMyRows, 0, A->getComm()) );
    Tpetra::Import<>  rowImporter(A->getRowMap(), importMap); // source, destination
    
    TMatrix<double> A0(importMap,0);
    A0.doImport(*A, rowImporter, Tpetra::INSERT);
    A0.fillComplete();
    
    A_fc.resize(static_cast<int>(numRowsGlobal), static_cast<int>(A->getGlobalNumCols()));
    
    // fill in on rank 0
    if (myRank == 0)
    {
      int maxCols = static_cast<int>(A0.getLocalMaxNumRowEntries());
      Camellia::FieldContainer<int> localColIDs(maxCols); // upper bound on columns for any row
      Camellia::FieldContainer<double> values(maxCols);
      
      auto rowGidsView = A0.getRowMap()->getMyGlobalIndices();
      auto colGidsView = A0.getColMap()->getMyGlobalIndices();
      
      using GlobalIndicesHostView = TMatrix<double>::nonconst_global_inds_host_view_type;
      GlobalIndicesHostView tColGIDs("Tpetra matrix columns host view", maxCols);
      using ValuesViewHost = TMatrix<double>::nonconst_values_host_view_type;
      ValuesViewHost tValues("Tpetra matrix row values", maxCols);
      
      for (int myRow=0; myRow < numRowsGlobal; myRow++)
      {
        auto gRow = rowGidsView[myRow]; // global row ID
        int numCols = -1;
        size_t numRowEntries = 0;
        
        A0.getGlobalRowCopy(gRow, tColGIDs, tValues, numRowEntries);
        
        for (int colOrdinal=0; colOrdinal < numRowEntries; colOrdinal++)
        {
          auto gCol = tColGIDs(colOrdinal);
          A_fc(static_cast<int>(gRow),static_cast<int>(gCol)) = values(colOrdinal);
        }
      }
    }
    
    // now, broadcast A_fc from rank 0:
    char* fcData = A_fc.size() > 0 ? reinterpret_cast<char*>(&A_fc[0]) : NULL;
    A->getComm()->broadcast(0, A_fc.size(), fcData);
    
    return 0;
  }
  
  void SerialDenseWrapper::filterMatrix(Camellia::FieldContainer<double> &filteredMatrix, const Camellia::FieldContainer<double> &matrix,
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
  
  int SerialDenseWrapper::getEnumeration(const std::vector<int> &index, const Intrepid::FieldContainer<double> &fc)
  {
    switch (fc.rank())
    {
      case 0:
        return 0;
      case 1:
        return index[0];
      case 2:
        return index[0] * fc.dimension(1) + index[1];
      case 3:
        return (index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2];
      case 4:
        return ((index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2]) * fc.dimension(3) + index[3];
      case 5:
        return (((index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2]) * fc.dimension(3) + index[3]) * fc.dimension(4) + index[4];
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled Camellia::FieldContainer rank.");
    }
  }
  
  int SerialDenseWrapper::getEnumeration(const std::vector<int> &index, const Camellia::FieldContainer<double> &fc)
  {
    switch (fc.rank())
    {
      case 0:
        return 0;
      case 1:
        return index[0];
      case 2:
        return index[0] * fc.dimension(1) + index[1];
      case 3:
        return (index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2];
      case 4:
        return ((index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2]) * fc.dimension(3) + index[3];
      case 5:
        return (((index[0] * fc.dimension(1) + index[1]) * fc.dimension(2) + index[2]) * fc.dimension(3) + index[3]) * fc.dimension(4) + index[4];
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled Camellia::FieldContainer rank.");
    }
  }
  
  std::function<int()> SerialDenseWrapper::getEnumerator(const std::vector<int> &index, const Camellia::FieldContainer<double> &fc)
  {
    std::function<int()> enumerator;
    int dim1, dim2, dim3, dim4;
    const int *coord0, *coord1, *coord2, *coord3, *coord4;
    switch(fc.rank())
    {
      case 0:
        enumerator = []() {
          return 0;
        };
        break;
      case 1:
        coord0 = &index[0];
        enumerator = [coord0]() {
          return *coord0;
        };
        break;
      case 2:
        dim1 = fc.dimension(1);
        coord0 = &index[0];
        coord1 = &index[1];
        enumerator = [dim1, coord0, coord1]() {
          return *coord0 * dim1 + *coord1;
        };
        break;
      case 3:
        dim1 = fc.dimension(1);
        dim2 = fc.dimension(2);
        coord0 = &index[0];
        coord1 = &index[1];
        coord2 = &index[2];
        enumerator = [dim1, dim2, coord0, coord1, coord2]() {
          return (*coord0 * dim1 + *coord1) * dim2 + *coord2;
        };
        break;
      case 4:
        dim1 = fc.dimension(1);
        dim2 = fc.dimension(2);
        dim3 = fc.dimension(3);
        coord0 = &index[0];
        coord1 = &index[1];
        coord2 = &index[2];
        coord3 = &index[3];
        enumerator = [dim1, dim2, dim3, coord0, coord1, coord2, coord3]() {
          return ((*coord0 * dim1 + *coord1) * dim2 + *coord2) * dim3 + *coord3;
        };
        break;
      case 5:
        dim1 = fc.dimension(1);
        dim2 = fc.dimension(2);
        dim3 = fc.dimension(3);
        dim4 = fc.dimension(4);
        coord0 = &index[0];
        coord1 = &index[1];
        coord2 = &index[2];
        coord3 = &index[3];
        coord4 = &index[4];
        enumerator = [dim1, dim2, dim3, dim4, coord0, coord1, coord2, coord3, coord4]() {
          return (((*coord0 * dim1 + *coord1) * dim2 + *coord2) * dim3 + *coord3) * dim4 + *coord4;
        };
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "Unhandled outputValues rank");
    }
    return enumerator;
  }
  
  void SerialDenseWrapper::jacobiScaleMatrix(Camellia::FieldContainer<double> &A)
  {
    int N = A.dimension(0);
    if ((N != A.dimension(1)) || (A.rank() != 2))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "badly shaped matrix");
    }
    for (int i=0; i<N; i++)
    {
      double diag = A(i,i);
      double *val = &A(i,0);
      if (diag == 0) diag = 1.0; // simply omit rescaling of 0 diagonal entries
      for (int j=0; j<N; j++)
      {
        *val /= diag;
        val++;
      }
    }
  }

  double SerialDenseWrapper::estimate2NormConditionNumber(Camellia::FieldContainer<double> &A, bool ignoreZeroEigenvalues)
  {
    std::vector<double> singularValues;
    int result = solveSVD(singularValues, A);
    
    const int N = A.dimension(0);
  
    if (result == 0)
    {
      double maxSingularValue = singularValues[0];
      double minSingularValue = singularValues[N-1];
      double tol = 1e-14;
      if (ignoreZeroEigenvalues)
      {
        int index = N-1;
        while ((abs(minSingularValue) < tol) && (index > 0))
        {
          index--;
          minSingularValue = singularValues[index];
        }
      }
      if (maxSingularValue < tol)
      {
        std::cout << "maxSingularValue is zero for matrix:\n" << A;
      }
      return maxSingularValue / minSingularValue;
    }
    else
    {
      std::cout << "SVD failed for matrix:\n" << A;
      return -1.0;
    }
  }

  bool SerialDenseWrapper::matrixIsSymmetric(Camellia::FieldContainer<double> &A, double relTol, double absTol)
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
  
  bool SerialDenseWrapper::matrixIsSymmetric(Camellia::FieldContainer<double> &A, std::vector<std::pair<int,int>> &asymmetricEntries,
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
  void SerialDenseWrapper::multiply(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A,
                                    const Camellia::FieldContainer<double> &B, char TransposeA, char TransposeB)
  {
    multiplyAndAdd(X,A,B,TransposeA,TransposeB,1.0,0.0);
  }
  
  void SerialDenseWrapper::multiplyFCByWeight(Camellia::FieldContainer<double> & fc, double weight)
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
  void SerialDenseWrapper::multiplyAndAdd(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A,
                                          const Camellia::FieldContainer<double> &B,
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
    
    auto ATMatrix = convertFCToSDM(A,true,Teuchos::View); // true: transpose
    auto BTMatrix = convertFCToSDM(B,true,Teuchos::View);
    auto XTMatrix = convertFCToSDM(X,true,Teuchos::View);
    
    // since SDMs are transposed relative to FCs, reverse the transposal
    Teuchos::ETransp transposeAMatrix = (TransposeA=='T') ? Teuchos::NO_TRANS : Teuchos::TRANS;
    Teuchos::ETransp transposeBMatrix = (TransposeB=='T') ? Teuchos::NO_TRANS : Teuchos::TRANS;
    
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
    
    int success = XTMatrix.multiply(transposeAMatrix,transposeBMatrix,ScalarAB,ATMatrix,BTMatrix,ScalarThis);
    
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
  
  Camellia::FieldContainer<double> SerialDenseWrapper::getSubMatrix(Camellia::FieldContainer<double> &A, std::set<unsigned> &rowIndices,
                                                                    std::set<unsigned> &colIndices, bool warnOfNonzeroOffBlockEntries)
  {
    Camellia::FieldContainer<double> subMatrix(rowIndices.size(),colIndices.size());
    
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
  
  void SerialDenseWrapper::roundZeros(Camellia::FieldContainer<double> &A, double tol)
  {
    for (int i=0; i<A.size(); i++)
    {
      if (abs(A[i]) < tol) A[i] = 0;
    }
  }
  
  void SerialDenseWrapper::scaleBySymmetricDiagonal(Camellia::FieldContainer<double> &A)
  {
    // requires that A's diagonal be non-negative in every entry
    // the below is a first pass implementation--would be more efficient not to construct the diagonal matrix explicitly
    
    if ((A.rank() != 2) || (A.dimension(0) != A.dimension(1)))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "A must be N x N matrix");
    }
    
    //    std::cout << "A before scaling:\n" << A;
    
    int N = A.dimension(0);
    Camellia::FieldContainer<double> diag_inv_sqrt(N,N);
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
    Camellia::FieldContainer<double> A_temp(N,N); // not sure if we can safely use A while multiplying by A
    multiply(A_temp, diag_inv_sqrt, A);
    multiply(A, A_temp, diag_inv_sqrt);
    
    //    std::cout << "A after scaling:\n" << A;
  }
  
  int SerialDenseWrapper::solveSystem(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                                      Camellia::FieldContainer<double> &b, bool useATranspose)
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
  
  int SerialDenseWrapper::solveSystemLeastSquares(Camellia::FieldContainer<double> &x,
                                                  const Camellia::FieldContainer<double> &A,
                                                  const Camellia::FieldContainer<double> &b)
  {
    // solves Ax = b, where
    // A = (N,M), N >= M
    // b = (N)
    // x = (M)
    //   OR
    // b = (N,L)
    // x = (M,L)
    Camellia::FieldContainer<double> bCopy = b;
    
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
    
    Camellia::FieldContainer<double> A_T = A;
    transposeMatrix(A_T);
    
    Camellia::FieldContainer<double> ATA(M,M);
    multiply(ATA, A_T, A);
    
    Camellia::FieldContainer<double> ATb(M,L);
    multiply(ATb, A_T, b);
    
    const bool useATATranspose = true; // use ATA transpose (avoids some data movement; ATA is symmetric)
    const bool allowOverwriteOfATA = true; // local matrix, discarded after this
    return solveSystemUsingQR(x, ATA, ATb, useATATranspose, allowOverwriteOfATA);
  }
  
  int SerialDenseWrapper::solveSystemMultipleRHS(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                                                 Camellia::FieldContainer<double> &b, bool useATranspose)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N, M)
    int N = A.dimension(0);
    int nRHS = b.dimension(1);
    
    // take care of the fact that SDMs are transposed relative to FCs:
    transposeMatrix(b);
    
    if (!useATranspose)
    {
      transposeMatrix(A);
    }
    
    auto AMatrix = convertFCToSDM(A, useATranspose);
    auto bxVectors = convertFCToSDM(b);
    Teuchos::SerialDenseMatrix<int,double> xVectors(N,nRHS);
    
    std::vector<double> rowScalings, colScalings;
    bool shouldEquilibrate = computeEquilibrationScaling(AMatrix, rowScalings, colScalings);
    
    if (shouldEquilibrate)
    {
      equilibrateMatrix(AMatrix,rowScalings,colScalings);
      equilibrateRHS(bxVectors,rowScalings);
    }
    
    std::vector<int> piv(N);
    Teuchos::LAPACK<int,double> lapack;
    int info = 0;
    lapack.GETRF(
      N,          // # rows
      N,          // # cols
      AMatrix.values(), // pointer to A(0,0)
      AMatrix.stride(), // leading dimension (stride)
      piv.data(),// pivot indices (1-based)
      &info);
    
    TEUCHOS_TEST_FOR_EXCEPTION(info < 0, std::runtime_error, "GETRF: illegal argument at position " << -info);
    TEUCHOS_TEST_FOR_EXCEPTION(info > 0, std::runtime_error, "GETRF: U(" << info << "," << info << ") is zero; matrix is singular.");
    
    const char trans = 'N'; // no transpose
    lapack.GETRS(
      trans,
      N,                 // # rows
      nRHS,              // # cols
      AMatrix.values(),  // LU factors from GETRF
      AMatrix.stride(),  // leading dimension
      piv.data(), // pivot indices
      bxVectors.values(),  // RHS on entry, solution on exit
      bxVectors.stride(),  // leading dimension
      &info);
    
    if (shouldEquilibrate)
    {
      unequilibrateLHS(bxVectors, colScalings);
    }
    
    // if we transposed A above, transpose it back here--leaving things as we found them
    if (!useATranspose)
    {
      transposeMatrix(A);
    }
    
    transposeMatrix(b);
    
    x.resize(x.dimension(1),x.dimension(0)); //transposeMatrix(x);
    convertSDMToFC(x,bxVectors);
    transposeMatrix(x);
    
    return 0;
  }
  
  int SerialDenseWrapper::solveSystemUsingQR(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                                             Camellia::FieldContainer<double> &b, bool useATranspose, // would be better default to true because that involves no data movement
                                             bool allowOverwriteOfA)
  {
    // solves Ax = b, where
    // A = (N,N)
    // x, b = (N,M)
    if ((x.rank() != 2) || (A.rank() != 2) || (b.rank() != 2))
    {
      std::cout << "x, A, and b must each be a rank 2 Camellia::FieldContainer!!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x, A, and b must each be a rank 2 Camellia::FieldContainer!!");
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
    // before the transpose of the x Camellia::FieldContainer, which as presently implemented copies x.  This can
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

  int SerialDenseWrapper::solveSPDSystemLAPACKCholesky(Camellia::FieldContainer<double> &bx, Camellia::FieldContainer<double> &A_SPD)
  {
    int N = A_SPD.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != A_SPD.dimension(1), std::invalid_argument, "A must be square!");
    
    bool shouldEquilibrate;
    std::vector<double> scalings;
    
    bool transpose = true;
    Teuchos::DataAccess dataAccess = Teuchos::View;
    auto AMatrix = convertFCToSDM(A_SPD, transpose, dataAccess);
    shouldEquilibrate = computeEquilibrationScalingSPD(AMatrix, scalings);
    
    int M = (bx.rank() == 1) ? 1 : bx.dimension(1);
    
    if (M != 1)
    {
      transposeMatrix(bx); // moves data in memory
    }
    
    auto bxVectors = convertFCToSDM(bx, M != 1, dataAccess); // transpose if M != 1
    
    if (shouldEquilibrate)
    {
      equilibrateMatrix(AMatrix,scalings,scalings);
      equilibrateRHS(bxVectors,scalings);
    }
    
    char UPLO = 'L'; // lower-triangular
    
    int result = 0;
    int INFO;
    
    Teuchos::LAPACK<int, double> lapack;
    Teuchos::BLAS<int, double> blas;
    
    // factor
    lapack.POTRF(UPLO, N, &A_SPD[0], N, &INFO);
    
    if (INFO != 0) result = INFO;
    
    // back-substitute
    lapack.POTRS(UPLO, N, M, AMatrix.values(), N, bxVectors.values(), N, &INFO);

    if (shouldEquilibrate)
    {
      unequilibrateLHS(bxVectors, scalings);
    }
    
    if (INFO != 0) result = INFO;
    
    if (M != 1)
    {
      transposeMatrix(bx);
    }
    
    return result;
  }
  
  int SerialDenseWrapper::solveSPDSystemMultipleRHS(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A_SPD,
                                                    Camellia::FieldContainer<double> &b, bool allowOverwriteOfA)
  {
    x = b;
    if (!allowOverwriteOfA)
    {
      Camellia::FieldContainer<double> A_copy = A_SPD;
      return solveSPDSystemLAPACKCholesky(x, A_copy);
    }
    else
    {
      return solveSPDSystemLAPACKCholesky(x, A_SPD);
    }
  }
  
  int SerialDenseWrapper::solveSVD(std::vector<double> &singularValuesOut, Camellia::FieldContainer<double> &A)
  {
    const int N = A.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != A.dimension(1), std::invalid_argument, "A must be square");
    TEUCHOS_TEST_FOR_EXCEPTION(A.rank() != 2, std::invalid_argument, "A must be rank 2");
    
    singularValuesOut.resize(N);
    
    Camellia::FieldContainer<double> A_copy = A;
    
    // SDM wants column-major; input is row-major
    transposeMatrix(A_copy);
    
    const int &rowStride = N; // numCols
    Teuchos::SerialDenseMatrix<int, double> M(Teuchos::View, &A_copy(0,0), rowStride, N, N);
    
    Teuchos::LAPACK<int,double> lapack;
    int info = 0;
    
    // we only want singular values, not U or VT
    const char jobu  = 'N';
    const char jobvt = 'N';
    
    // workspace query
    int lwork = -1;
    double wkopt = 0.0;
    // note: we pass nullptr (or 0) for U and VT since JOBU=JOBVT='N'
    lapack.GESVD(jobu, jobvt,
                 N, N,
                 M.values(), M.stride(),
                 singularValuesOut.data(),
                 nullptr, 1,
                 nullptr, 1,
                 &wkopt, lwork,
                 nullptr,   // iwork not used for GESVD of real
                 &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info<0, std::runtime_error, "LAPACK GESVD workspace query failed with info="<<info);
    
    lwork = (int)wkopt;
    std::vector<double> work(lwork);
    
    // actual SVD computation
    lapack.GESVD(jobu, jobvt,
                 N, N,
                 M.values(), M.stride(),
                 singularValuesOut.data(),
                 nullptr, 1,
                 nullptr, 1,
                 work.data(), lwork,
                 nullptr,
                 &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info!=0, std::runtime_error, "LAPACK GESVD failed with info="<<info);
    
    return info;
  }
  
  void SerialDenseWrapper::transposeMatrix(Camellia::FieldContainer<double> &A)
  {
    int n = A.dimension(0);
    int m = A.dimension(1);
    if (n==m)
    {
      transposeSquareMatrix(A);
    }
    else
    {
      Camellia::FieldContainer<double> A_copy = A;
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

  void SerialDenseWrapper::unequilibrateLHS(Teuchos::SerialDenseMatrix<int, double> &x, const std::vector<double> &colScalings)
  {
    auto N    = x.numRows();
    auto NRHS = x.numCols();
    auto LDX  = x.stride();
    
    double * ptr;
    for (int j=0; j<NRHS; j++) {
      ptr = x.values() + j*LDX;
      for (int i=0; i<N; i++) {
        *ptr = *ptr*colScalings[i];
        ptr++;
      }
    }
  }
  
  //! Returns the reciprocal of the 2-norm condition number of the matrix in A
  /*!
   \param A In
   A rank-2 Camellia::FieldContainer with equal first and second dimensions.
   
   \return the 2-norm condition number if successful; -1 otherwise.
   */
  double SerialDenseWrapper::getMatrixConditionNumber2Norm(Camellia::FieldContainer<double> &A, bool ignoreZeroEigenvalues)
  {
    return estimate2NormConditionNumber(A, ignoreZeroEigenvalues);
  }
  
  void SerialDenseWrapper::writeMatrixToMatlabFile(const std::string& filePath, Camellia::FieldContainer<double> &A)
  {
    int N = A.dimension(0);
    int M = A.dimension(1);
    std::ofstream matrixFile;
    matrixFile.open(filePath.c_str());
    matrixFile << std::setprecision(15);
    
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
  
  
  void SerialDenseWrapper::addFCs(Camellia::FieldContainer<double> &A, const Camellia::FieldContainer<double> &B,
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
