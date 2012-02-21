
#ifndef DPG_OPTIMAL_INNER_PRODUCT
#define DPG_OPTIMAL_INNER_PRODUCT

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
 *  OptimalInnerProduct.h
 *
 *  Created by Nathan Roberts on 7/22/11.
 *
 */

#include "BilinearForm.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"

/*
 Implements quasi-optimal inner product for H1, H(div) test functions
 */

class OptimalInnerProduct : public DPGInnerProduct {
  typedef pair<IntrepidExtendedTypes::EOperatorExtended, int > OpOpIndexPair;
  map< pair<int, int>, 
       vector< pair < pair< OpOpIndexPair, OpOpIndexPair >, int> > > _testCombos;
  /*
   _testCombos is a map like:
  map< pair<testID1, testID2>, 
       vector< pair< pair<pair<op1, opIndexForTrialTest1 >,
                          pair<op2, opIndexForTrialTest2 > >,
                     trialID > >
   */
  double _beta; // TODO: make this a vector, one value per flux/trace, or even somehow problem/mesh-dependent...
public:
  OptimalInnerProduct(Teuchos::RCP< BilinearForm > bilinearForm);
  
  void operators(int testID1, int testID2, 
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2);
  
  void applyInnerProductData(FieldContainer<double> &testValues1,
                             FieldContainer<double> &testValues2,
                             int testID1, int testID2, int operatorIndex,
                             const FieldContainer<double>& physicalPoints);
};

#endif
