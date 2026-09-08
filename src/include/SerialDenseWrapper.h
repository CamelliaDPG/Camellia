// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  SerialDenseWrapper.h
//  Camellia
//

#ifndef SerialDenseWrapper_h
#define SerialDenseWrapper_h

#include "Camellia_FieldContainer.hpp"

#include "Epetra_RowMatrix.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_DataAccess.h"

#include <set>

namespace Camellia
{
  class SerialDenseWrapper
  {
  public:
    // the following made public so we can test them.  They are not intended for general use, however.
    static Epetra_SerialDenseMatrix convertFCToSDM(const Camellia::FieldContainer<double> &A, Epetra_DataAccess CV = ::Copy);
    static void convertSDMToFC(Camellia::FieldContainer<double>& A_fc, const Epetra_SerialDenseMatrix &A);
  public:
    // gives X = scalarA*A+scalarB*B (overwrites A)
    static void add(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A, const Camellia::FieldContainer<double> &B, double scalarA = 1.0, double scalarB = 1.0);
    
    static double condest(const Camellia::FieldContainer<double> &A, bool assumeSPD = false);
    
    static double dot(const Camellia::FieldContainer<double> &a, const Camellia::FieldContainer<double> &b);
    
    static int determinantAndInverse(Camellia::FieldContainer<double> &detValues, Camellia::FieldContainer<double> &outInverses,
                                     const Camellia::FieldContainer<double> &inMatrices);
    
    static int extractFCFromEpetra_RowMatrix(const Epetra_RowMatrix &A, Camellia::FieldContainer<double> &A_fc);
    
    static void filterMatrix(Camellia::FieldContainer<double> &filteredMatrix, const Camellia::FieldContainer<double> &matrix,
                             const std::set<int> &rowOrdinals, const std::set<int> &colOrdinals);
    
    static int eigenvalues(const Camellia::FieldContainer<double> &A,
                           Camellia::FieldContainer<double> &lambda_real,
                           Camellia::FieldContainer<double> &lambda_imag);
    
    static int eigenvalues(const Camellia::FieldContainer<double> &A,
                           Camellia::FieldContainer<double> &lambda_real,
                           Camellia::FieldContainer<double> &lambda_imag,
                           Camellia::FieldContainer<double> &eigvectors, // if empty, skips eigvector determination
                           bool computeRight); // if computeRight = false, will compute left eigenvectors.
    
    // ! provides an implementation of getEnumeration that uses std::vector<int> instead.
    static int getEnumeration(const std::vector<int> &index, const Camellia::FieldContainer<double> &fc);

    // ! returns a lambda function that depends dynamically on the values in index, and at generation time on the dimensions of fc.  Intended to provide a faster version than either Camellia::FieldContainer or the above getEnumeration() (which should perhaps be deleted).
    static std::function<int()> getEnumerator(const std::vector<int> &index, const Camellia::FieldContainer<double> &fc);
    
    static bool matrixIsSymmetric(Camellia::FieldContainer<double> &A, double relTol = 1e-12, double absTol = 1e-14);
    
    static bool matrixIsSymmetric(Camellia::FieldContainer<double> &A, std::vector<std::pair<int,int>> &asymmetricEntries,
                                  double relTol = 1e-12, double absTol = 1e-14);
    
    // gives X = A*B.  Must pass in 2D arrays, even for vectors!
    static void multiply(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A,
                         const Camellia::FieldContainer<double> &B, char TransposeA = 'N', char TransposeB = 'N');
    
    static void multiplyFCByWeight(Camellia::FieldContainer<double> & fc, double weight);
    
    // wrapper for SDM multiply + add routine.  Must pass in 2D arrays, even for vectors!
    // X = ScalarThis*X + ScalarAB*A*B
    static void multiplyAndAdd(Camellia::FieldContainer<double> &X, const Camellia::FieldContainer<double> &A,
                               const Camellia::FieldContainer<double> &B,
                               char TransposeA, char TransposeB, double ScalarAB, double ScalarThis);
    
    static Camellia::FieldContainer<double> getSubMatrix(Camellia::FieldContainer<double> &A, std::set<unsigned> &rowIndices,
                                                         std::set<unsigned> &colIndices, bool warnOfNonzeroOffBlockEntries = false);
    
    static void roundZeros(Camellia::FieldContainer<double> &A, double tol);
    
    static void scaleBySymmetricDiagonal(Camellia::FieldContainer<double> &A);
    
    static int solveSystem(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                           Camellia::FieldContainer<double> &b, bool useATranspose = false);
    
    static int solveSystemLeastSquares(Camellia::FieldContainer<double> &x,
                                       const Camellia::FieldContainer<double> &A,
                                       const Camellia::FieldContainer<double> &b);
    
    static int solveSystemMultipleRHS(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                                      Camellia::FieldContainer<double> &b, bool useATranspose = false);
    
    static int solveSystemUsingQR(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A,
                                  Camellia::FieldContainer<double> &b, bool useATranspose = false, // would be better default to true because that involves no data movement
                                  bool allowOverwriteOfA = false);
    
    // ! A_SPD is left as in lower-triangular factored format; bx on input is the RHS b; on output, it's the solution x = A \ b.
    static int solveSPDSystemLAPACKCholesky(Camellia::FieldContainer<double> &bx, Camellia::FieldContainer<double> &A_SPD);
    
    static int solveSPDSystemMultipleRHS(Camellia::FieldContainer<double> &x, Camellia::FieldContainer<double> &A_SPD,
                                         Camellia::FieldContainer<double> &b, bool allowOverwriteOfA = false);
    
    static void transposeMatrix(Camellia::FieldContainer<double> &A);
    
    static void transposeSquareMatrix(Camellia::FieldContainer<double> &A);
    
    //! Returns the reciprocal of the 1-norm condition number of the matrix in A
    /*!
     \param A In
     A rank-2 Camellia::FieldContainer with equal first and second dimensions.
     
     \return the 1-norm condition number if successful; -1 otherwise.
     */
    static double getMatrixConditionNumber(Camellia::FieldContainer<double> &A);
    
    //! Returns the reciprocal of the 2-norm condition number of the matrix in A
    /*!
     \param A In
     A rank-2 Camellia::FieldContainer with equal first and second dimensions.
     
     \return the 2-norm condition number if successful; -1 otherwise.
     */
    static double getMatrixConditionNumber2Norm(Camellia::FieldContainer<double> &A, bool ignoreZeroEigenvalues = true);
    
    // These methods for testing positive (semi)definiteness do not work.  As noted in comments below, need to do an eigenvalue solve.
    //  //! Returns true if all the eigenvalues are greater than zero
    //  /*!
    //   \param A In
    //   A rank-2 Camellia::FieldContainer with equal first and second dimensions.
    //
    //   \return true if all the eigenvalues are positive; false otherwise.
    //   */
    //  static bool matrixIsPositiveDefinite(Camellia::FieldContainer<double> &A)
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
    //   A rank-2 Camellia::FieldContainer with equal first and second dimensions.
    //
    //   \return true if all the eigenvalues are non-negative; false otherwise.
    //   */
    //  static bool matrixIsPositiveSemiDefinite(Camellia::FieldContainer<double> &A)
    //  {
    //    cout << "WARNING: matrixIsPositiveSemiDefinite() does not work.  Need to set up an eigenvalue solve, not an SVD!\n";
    //    return getMatrixConditionNumber2Norm(A, true) > 0.0;
    //  }
    
    static void writeMatrixToMatlabFile(const std::string& filePath, Camellia::FieldContainer<double> &A);
    
    
    static void addFCs(Camellia::FieldContainer<double> &A, const Camellia::FieldContainer<double> &B,
                       double B_weight = 1.0, double A_weight = 1.0);
  };
}

#endif
