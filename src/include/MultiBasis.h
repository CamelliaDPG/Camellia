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

#ifndef DPG_MULTI_BASIS
#define DPG_MULTI_BASIS

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

class MultiBasis : public DoubleBasis {
  shards::CellTopology _cellTopo;
  FieldContainer<double> _subRefNodes;
  vector< BasisPtr > _bases;
  int _numLeaves;
  
  void computeCellJacobians(FieldContainer<double> &cellJacobian, FieldContainer<double> &cellJacobInv,
                            FieldContainer<double> &cellJacobDet, const FieldContainer<double> &inputPointsSubRefCell,
                            int subRefCellIndex) const;
  
  void initializeTags();
public:
  // below, subRefNodes means the coordinates of the nodes of the children in the parent/reference cell
  // if there are N nodes in D-dimensional cellTopo and C bases in bases, then subRefNodes should have dimensions (C,N,D)
  MultiBasis(vector< BasisPtr > bases, FieldContainer<double> &subRefNodes, shards::CellTopology &cellTopo);

  void getValues(FieldContainer<double> &outputValues, const FieldContainer<double> &  inputPoints,
                 const EOperator operatorType) const;
  void getValues(FieldContainer<double> & outputValues,
                 const FieldContainer<double> &   inputPoints,
                 const FieldContainer<double> &    cellVertices,
                 const EOperator        operatorType) const;
  
  BasisPtr getSubBasis(int basisIndex);
  BasisPtr getLeafBasis(int leafOrdinal);
  
  vector< pair<int,int> > adjacentVertexOrdinals(); // NOTE: prototype, untested code!
  
  int numLeafNodes();
  int numSubBases();
  
  int relativeToAbsoluteDofOrdinal(int basisDofOrdinal, int leafOrdinal);
  
  void getCubature(FieldContainer<double> &cubaturePoints, FieldContainer<double> &cubatureWeights, int maxTestDegree);
  
  void printInfo();
};

#endif
