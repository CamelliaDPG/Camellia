// @HEADER
//
// Copyright © 2011 Nathan V. Roberts. All Rights Reserved.
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
// THIS SOFTWARE IS PROVIDED BY NATHAN V. ROBERTS "AS IS" AND ANY 
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

#ifndef DPG_PATCH_BASIS
#define DPG_PATCH_BASIS

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace std;
using namespace Intrepid;

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

class PatchBasis : public DoubleBasis {
  shards::CellTopology _patchCellTopo;
  shards::CellTopology _parentTopo;
  FieldContainer<double> _patchNodesInParentRefCell;
  BasisPtr _parentBasis;
  FieldContainer<double> _parentRefNodes;
  
  void computeCellJacobians(FieldContainer<double> &cellJacobian, FieldContainer<double> &cellJacobInv,
                            FieldContainer<double> &cellJacobDet, const FieldContainer<double> &inputPointsParentRefCell) const;
  
  void initializeTags();
public:
  PatchBasis(BasisPtr parentBasis, FieldContainer<double> &patchNodesInParentRefCell, shards::CellTopology &patchCellTopo);
  
  void getValues(FieldContainer<double> &outputValues, const FieldContainer<double> &  inputPoints,
                 const EOperator operatorType) const;
  void getValues(FieldContainer<double> & outputValues,
                 const FieldContainer<double> &   inputPoints,
                 const FieldContainer<double> &    cellVertices,
                 const EOperator        operatorType) const;

  BasisPtr parentBasis();
};

#endif