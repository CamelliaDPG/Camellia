#ifndef INNER_PRODUCT
#define INNER_PRODUCT

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

#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "BilinearForm.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DofOrdering.h"

using namespace Intrepid;
using namespace std;

class BasisCache; // BasisCache.h and DofOrdering.h #include each other...
typedef Teuchos::RCP<BasisCache> BasisCachePtr;

class DPGInnerProduct {
protected:
  Teuchos::RCP< BilinearForm > _bilinearForm;
public:
  DPGInnerProduct(Teuchos::RCP< BilinearForm > bfs);
  virtual ~DPGInnerProduct() {}
  
  virtual void computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         shards::CellTopology &cellTopo,
                                         FieldContainer<double>& physicalCellNodes);
  
  virtual void computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         BasisCachePtr basisCache);
  
  virtual void printInteractions();
  
  virtual void operators(int testID1, int testID2, 
                         vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                         vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2) = 0;
  
  virtual void applyInnerProductData(FieldContainer<double> &testValues1,
                                     FieldContainer<double> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     const FieldContainer<double>& physicalPoints) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override some version of applyInnerProductData!");
  }
  
  virtual void applyInnerProductData(FieldContainer<double> &testValues1,
                                     FieldContainer<double> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     BasisCachePtr basisCache);
  
  virtual bool hasBoundaryTerms(); // used for deciding whether to create side caches or not.
};

typedef Teuchos::RCP<DPGInnerProduct> InnerProductPtr;

#endif
