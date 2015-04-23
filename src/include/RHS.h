#ifndef DPG_RHS
#define DPG_RHS

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  RHS.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

#include "BasisCache.h"

#include "LinearTerm.h"

#include <vector>

using namespace std;

namespace Camellia {
  template <typename Scalar>
  class TRHS {
    bool _legacySubclass;

    TLinearTermPtr<Scalar> _lt;
    set<int> _varIDs;
  public:
    TRHS(bool legacySubclass) : _legacySubclass(legacySubclass) {}
    virtual bool nonZeroRHS(int testVarID);
    virtual vector<Camellia::EOperator> operatorsForTestID(int testID);
    // TODO: change the API here so that values is the first argument (fitting a convention in the rest of the code)
    virtual void rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, Intrepid::FieldContainer<Scalar> &values);
    virtual void rhs(int testVarID, int operatorIndex, const Intrepid::FieldContainer<double> &physicalPoints, Intrepid::FieldContainer<Scalar> &values);
    virtual void rhs(int testVarID, const Intrepid::FieldContainer<double> &physicalPoints, Intrepid::FieldContainer<Scalar> &values);
    // physPoints (numCells,numPoints,spaceDim)
    // values: either (numCells,numPoints) or (numCells,numPoints,spaceDim)

    virtual void integrateAgainstStandardBasis(Intrepid::FieldContainer<Scalar> &rhsVector, Teuchos::RCP<DofOrdering> testOrdering,
                                               BasisCachePtr basisCache);
    virtual void integrateAgainstOptimalTests(Intrepid::FieldContainer<Scalar> &rhsVector, const Intrepid::FieldContainer<Scalar> &optimalTestWeights,
                                              Teuchos::RCP<DofOrdering> testOrdering, BasisCachePtr basisCache);

    void addTerm( TLinearTermPtr<Scalar> rhsTerm );
    void addTerm( VarPtr v );

    TLinearTermPtr<Scalar> linearTerm(); // MUTABLE reference (change this, RHS will change!)
    TLinearTermPtr<Scalar> linearTermCopy(); // copy of RHS as a LinearTerm

    virtual ~TRHS() {}

    static TRHSPtr<Scalar> rhs() { return Teuchos::rcp(new TRHS<Scalar>(false) ); }
  };

  extern template class TRHS<double>;
}

#endif
