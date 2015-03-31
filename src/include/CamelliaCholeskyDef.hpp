#include "cholesky.hpp"

template<class Scalar>
int Cholesky<Scalar>::solve(Intrepid::FieldContainer<double> &X_fc,
                             const Intrepid::FieldContainer<double> &A_fc,
                             const Intrepid::FieldContainer<double> &B_fc,
                             bool transposeBandX) {
  // shapes:
  // A: (C,N,N)
  // B: (C,N,M)
  // C: (C,N,M)
  // unless transposeBandX = true, in which case
  // B: (C,M,N)
  // C: (C,M,N)
  int solvedAll = 0;
  
  typedef ublas::row_major  ORI;
  int numCells = A_fc.dimension(0);
  int n = A_fc.dimension(1);
  int m = transposeBandX ? B_fc.dimension(1) : B_fc.dimension(2);
  
  TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(0) != numCells, std::invalid_argument, "A and B must agree in numCells.");
  if (transposeBandX) {
    TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(2) != n, std::invalid_argument, "A and B must agree in the dimension N.");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(1) != n, std::invalid_argument, "A and B must agree in the dimension N.");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(0) != C_fc.dimension(0), std::invalid_argument, "B and C must agree in shape.");
  TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(1) != C_fc.dimension(1), std::invalid_argument, "B and C must agree in shape.");
  TEUCHOS_TEST_FOR_EXCEPTION(B_fc.dimension(2) != C_fc.dimension(2), std::invalid_argument, "B and C must agree in shape.");
  
  ublas::matrix<Scalar, ORI> A (n, n);
  ublas::matrix<Scalar, ORI> L (n, n);
  ublas::vector<Scalar> x (n);

  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    // copy the inner product for this cell into matrix A
    // (could optimize this using pointer arithmetic)
    for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
        A(i,j) = A_fc(cellIndex,i,j);
      }
    }
    size_t res = cholesky_decompose(A, L);
    if (res != 0) { // failure: communicate by setting solvedAll
      solvedAll = res;
    }
    // now solve for each rhs corresponding to the stiffness matrix columns for this cell
    for (int j=0; j<m; j++) {
      // copy from transposed stiffness matrix:
      for (int i=0; i<n; i++) {
        if (transposeBandX)
          x(i) = B_fc(cellIndex,j,i);
        else
          x(i) = B_fc(cellIndex,i,j);
      }
      cholesky_solve(L, x, ublas::lower());
      for (int i=0; i<n; i++) {
        if (transposeBandX)
          X_fc(cellIndex,j,i) = x(i);
        else
          X_fc(cellIndex,i,j) = x(i);
      }
    }
  }
  return solvedAll;
}
