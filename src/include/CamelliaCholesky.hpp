//
//  CamelliaCholesky.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/28/13.
//
//

#ifndef Camellia_debug_CamelliaCholesky_hpp
#define Camellia_debug_CamelliaCholesky_hpp

namespace Camellia {
  template class<Scalar> Cholesky {
    static int solve(Intrepid::FieldContainer<double> &X, const Intrepid::FieldContainer<double> &A, const Intrepid::FieldContainer<double> &B, bool transposeBandX = false);
  };
}

#include "CamelliaCholeskyDef.hpp"

#endif