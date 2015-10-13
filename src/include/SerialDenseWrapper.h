//
//  SerialDenseWrapper.h
//  Camellia-debug
//
//
//

#ifndef SerialDenseWrapper_h
#define SerialDenseWrapper_h

#include "Intrepid_FieldContainer.hpp"

#include "Epetra_RowMatrix.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_DataAccess.h"

namespace Camellia
{
  class SerialDenseWrapper
  {
    static void transposeSquareMatrix(Intrepid::FieldContainer<double> &A);
    static Epetra_SerialDenseMatrix convertFCToSDM(const Intrepid::FieldContainer<double> &A, Epetra_DataAccess CV = ::Copy);
    static void convertSDMToFC(Intrepid::FieldContainer<double>& A_fc, const Epetra_SerialDenseMatrix &A);
  public:
    // gives X = scalarA*A+scalarB*B (overwrites A)
    static void add(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A, const Intrepid::FieldContainer<double> &B, double scalarA = 1.0, double scalarB = 1.0);
    static double dot(const Intrepid::FieldContainer<double> &a, const Intrepid::FieldContainer<double> &b);
    
    static int determinantAndInverse(Intrepid::FieldContainer<double> &detValues, Intrepid::FieldContainer<double> &outInverses,
                                     const Intrepid::FieldContainer<double> &inMatrices);
    
    static int extractFCFromEpetra_RowMatrix(const Epetra_RowMatrix &A, Intrepid::FieldContainer<double> &A_fc);
    
    static void filterMatrix(Intrepid::FieldContainer<double> &filteredMatrix, const Intrepid::FieldContainer<double> &matrix,
                             const std::set<int> &rowOrdinals, const std::set<int> &colOrdinals);
    
    static bool matrixIsSymmetric(Intrepid::FieldContainer<double> &A, double relTol = 1e-12, double absTol = 1e-14);
    
    static bool matrixIsSymmetric(Intrepid::FieldContainer<double> &A, std::vector<std::pair<int,int>> &asymmetricEntries,
                                  double relTol = 1e-12, double absTol = 1e-14);
    
    // gives X = A*B.  Must pass in 2D arrays, even for vectors!
    static void multiply(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A,
                         const Intrepid::FieldContainer<double> &B, char TransposeA = 'N', char TransposeB = 'N');
    
    static void multiplyFCByWeight(Intrepid::FieldContainer<double> & fc, double weight);
    
    // wrapper for SDM multiply + add routine.  Must pass in 2D arrays, even for vectors!
    // X = ScalarThis*X + ScalarAB*A*B
    static void multiplyAndAdd(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A,
                               const Intrepid::FieldContainer<double> &B,
                               char TransposeA, char TransposeB, double ScalarAB, double ScalarThis);
    
    static Intrepid::FieldContainer<double> getSubMatrix(Intrepid::FieldContainer<double> &A, std::set<unsigned> &rowIndices,
                                                         std::set<unsigned> &colIndices, bool warnOfNonzeroOffBlockEntries = false);
    
    static void roundZeros(Intrepid::FieldContainer<double> &A, double tol);
    
    static void scaleBySymmetricDiagonal(Intrepid::FieldContainer<double> &A);
    
    static int solveSystem(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                           Intrepid::FieldContainer<double> &b, bool useATranspose = false);
    
    static int solveSystemLeastSquares(Intrepid::FieldContainer<double> &x,
                                       const Intrepid::FieldContainer<double> &A,
                                       const Intrepid::FieldContainer<double> &b);
    
    static int solveSystemMultipleRHS(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                                      Intrepid::FieldContainer<double> &b, bool useATranspose = false);
    
    static int solveSystemUsingQR(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A,
                                  Intrepid::FieldContainer<double> &b, bool useATranspose = false, // would be better default to true because that involves no data movement
                                  bool allowOverwriteOfA = false);
    
    static int solveSPDSystemMultipleRHS(Intrepid::FieldContainer<double> &x, Intrepid::FieldContainer<double> &A_SPD,
                                         Intrepid::FieldContainer<double> &b, bool allowOverwriteOfA = false);
    
    static void transposeMatrix(Intrepid::FieldContainer<double> &A);
    
    //! Returns the reciprocal of the 1-norm condition number of the matrix in A
    /*!
     \param A In
     A rank-2 FieldContainer with equal first and second dimensions.
     
     \return the 1-norm condition number if successful; -1 otherwise.
     */
    static double getMatrixConditionNumber(Intrepid::FieldContainer<double> &A);
    
    //! Returns the reciprocal of the 2-norm condition number of the matrix in A
    /*!
     \param A In
     A rank-2 FieldContainer with equal first and second dimensions.
     
     \return the 2-norm condition number if successful; -1 otherwise.
     */
    static double getMatrixConditionNumber2Norm(Intrepid::FieldContainer<double> &A, bool ignoreZeroEigenvalues = true);
    
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
    
    static void writeMatrixToMatlabFile(const std::string& filePath, Intrepid::FieldContainer<double> &A);
    
    
    static void addFCs(Intrepid::FieldContainer<double> &A, const Intrepid::FieldContainer<double> &B,
                       double B_weight = 1.0, double A_weight = 1.0);
  };
}

#endif
