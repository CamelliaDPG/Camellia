// @HEADER
//
// Copyright © 2013 Nathan V. Roberts. All Rights Reserved.
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

#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=::Intrepid::FieldContainer<double> > class PatchBasis;
  template<class Scalar, class ArrayScalar> class PatchBasis : public Camellia::Basis<Scalar,ArrayScalar> {
    CellTopoPtr _patchCellTopo;
    CellTopoPtr _parentTopo;
    ArrayScalar _patchNodesInParentRefCell;
    BasisPtr _parentBasis;
    ArrayScalar _parentRefNodes;
    
    void computeCellJacobians(ArrayScalar &cellJacobian, ArrayScalar &cellJacobInv,
                              ArrayScalar &cellJacobDet, const ArrayScalar &inputPointsParentRefCell) const;
    
    void initializeTags() const;
  public:

    PatchBasis(BasisPtr parentBasis, ArrayScalar &patchNodesInParentRefCell, shards::CellTopology &patchCellTopo);
    
    void getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                   const Intrepid::EOperator operatorType) const;
    
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > getSubBasis(int basisIndex) const;
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > getLeafBasis(int leafOrdinal) const;
    
    vector< pair<int,int> > adjacentVertexOrdinals() const; // NOTE: prototype, untested code!
    
    // domain info on which the basis is defined:
    CellTopoPtr domainTopology() const;
    
    // dof ordinal subsets:
  //  std::set<int> dofOrdinalsForEdges(bool includeVertices = true);
  //  std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true);
  //  std::set<int> dofOrdinalsForInterior();
  //  std::set<int> dofOrdinalsForVertices();
    
  //  int getDofOrdinal(const int subcDim, const int subcOrd, const int subcDofOrd) const;
    
    // range info for basis values:
    int rangeDimension() const;
    int rangeRank() const;
    
    int relativeToAbsoluteDofOrdinal(int basisDofOrdinal, int leafOrdinal) const;
    
    void getCubature(ArrayScalar &cubaturePoints, ArrayScalar &cubatureWeights, int maxTestDegree) const;
    
    BasisPtr nonPatchAncestorBasis() const; // the ancestor of whom all descendants are PatchBases
    BasisPtr parentBasis() const; // the immediate parent
  };

  typedef Teuchos::RCP< PatchBasis<> > PatchBasisPtr;
}

#include "PatchBasisDef.h"

#endif