#ifndef DPG_L2_INNER_PRODUCT
#define DPG_L2_INNER_PRODUCT

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

#include "BilinearForm.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "AbstractFunction.h"

#include "IP.h"

/*
  Implements "standard" inner product for H1, H(div) test functions
*/

class L2InnerProduct : public IP {
 public:

 L2InnerProduct(Teuchos::RCP< BilinearForm > bfs) : IP(bfs) {}
  
  void operators(int testID1, int testID2, 
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2);
  
  void applyInnerProductData(FieldContainer<double> &testValues1,
                             FieldContainer<double> &testValues2,
			     int testID1, int testID2, int operatorIndex,
			     const FieldContainer<double>& physicalPoints);

  // overwrite the original computeInnerProductMatrix routine - instead compute trial L2 matrix
  void computeInnerProductMatrix(FieldContainer<double> &innerProduct,
				   Teuchos::RCP<DofOrdering> dofOrdering,
				   shards::CellTopology &cellTopo,
				   FieldContainer<double>& physicalCellNodes);

  void computeInnerProductVector(FieldContainer<double> &innerProduct,
				   Teuchos::RCP<DofOrdering> dofOrdering,
				   shards::CellTopology &cellTopo,
				 FieldContainer<double>& physicalCellNodes,
				 Teuchos::RCP<AbstractFunction> fxn);
  
};

#endif
