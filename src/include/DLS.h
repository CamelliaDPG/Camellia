//
//  DLS.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/7/17.
//
//

#ifndef DLS_hpp
#define DLS_hpp

#include "Solution.h"
#include "TypeDefs.h"

namespace Camellia
{
  template <typename Scalar = double>
  class DLS {
    TMatrixPtr<Scalar> _dlsMatrix; // rectangular least-squares matrix (scaled on the right by square root of the normal matrix diagonal), with columns corresponding to BCs eliminated
    TVectorPtr<Scalar> _lhsVector; // the solution (with BCs eliminated)
    TVectorPtr<Scalar> _rhsVector; // adjusted according to BC values
    TSolutionPtr<Scalar> _soln;
    
    Teuchos::RCP<Tpetra::Vector<Scalar>> _diag_sqrt_inverse; // inverse of the diagonal of the normal matrix (A^T * A)
    
    typedef Teuchos::RCP<Tpetra::Map<>>  SolutionMapPtr;
    // Tpetra Map that goes from our DLS solution to the _soln LHS vector:
    SolutionMapPtr _dlsToSolnMap;
    // (the reason being that we'll eliminate BCs from our DLS LHS, but these exist explicitly in _soln)
  public:
    DLS(TSolutionPtr<Scalar> solution);
    
    int assemble(); // will take BCs into account
    int solveProblemLSQR(int maxIters = -1, double tol = 1e-6);
    
    TMatrixPtr<Scalar> matrix(); // test by trial; has been multiplied on the right by D^-1/2
    TVectorPtr<Scalar> rhs(); // the "enriched" RHS vector (has length = # test dofs)
    TVectorPtr<Scalar> lhs(); // the solution vector (with all Dirichlet dofs eliminated); has been equilibriated by multiplying by D^-1/2 to get the true solution (i.e. due to scaling of matrix, it is not the solution to (matrix^T * matrix) \ rhs, but instead is D^-1/2 * ((matrix^T * matrix) \ rhs)

    TVectorPtr<Scalar> normalDiagInverseSqrt(); // the diagonal of the inverse of the square root of the normal matrix

    TSolutionPtr<Scalar> solution();

    // The below method commented out because it seems like a relic, before we decided to maintain
    // numbering between the two -- the only difference is that solution()->lhsVector() has some extra
    // dofs, which are set to whatever the BCs were...  The map is an identity map, albeit a non-bijective one...
//    // Map to allow translation between the Solution returned by DLS::solution()->lhsVector()
//    // and that returned by DLS::lhs()
//    SolutionMapPtr solutionMapPtr();
  };
}

#endif /* DLS_hpp */
