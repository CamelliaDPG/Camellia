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

#ifndef DPG_MULTI_BASIS
#define DPG_MULTI_BASIS

#include "Basis.h"
#include "Intrepid_FieldContainer.hpp"

#include "CellTopology.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class MultiBasis;
template<class Scalar, class ArrayScalar> class MultiBasis : public Basis<Scalar,ArrayScalar>
{
  CellTopoPtr _cellTopo;
  Intrepid::FieldContainer<double> _subRefNodes;
  std::vector< Teuchos::RCP< Basis<Scalar,ArrayScalar> > > _bases;
  int _numLeaves;

  void computeCellJacobians(ArrayScalar &cellJacobian, ArrayScalar &cellJacobInv,
                            ArrayScalar &cellJacobDet, const ArrayScalar &inputPointsSubRefCell,
                            int subRefCellIndex) const;

  void initializeTags() const;
public:
  // below, subRefNodes means the coordinates of the nodes of the children in the parent/reference cell
  // if there are N nodes in D-dimensional cellTopo and C bases in bases, then subRefNodes should have dimensions (C,N,D)
  MultiBasis(std::vector< Teuchos::RCP< Basis<Scalar,ArrayScalar> > > bases, ArrayScalar &subRefNodes, shards::CellTopology &cellTopo);

  void getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                 const Intrepid::EOperator operatorType) const;

  Teuchos::RCP< Basis<Scalar,ArrayScalar> > getSubBasis(int basisIndex) const;
  Teuchos::RCP< Basis<Scalar,ArrayScalar> > getLeafBasis(int leafOrdinal) const;

  std::vector< std::pair<int,int> > adjacentVertexOrdinals() const; // NOTE: prototype, untested code!

  // domain info on which the basis is defined:
  CellTopoPtr domainTopology() const;

  // dof ordinal subsets:
  //  std::set<int> dofOrdinalsForEdges(bool includeVertices = true);
  //  std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true);
  //  std::set<int> dofOrdinalsForInterior();
  //  std::set<int> dofOrdinalsForVertices();

  int getDofOrdinal(const int subcDim, const int subcOrd, const int subcDofOrd) const;

  // range info for basis values:
  int rangeDimension() const;
  int rangeRank() const;

  int numLeafNodes() const;
  int numSubBases() const;

  int relativeToAbsoluteDofOrdinal(int basisDofOrdinal, int leafOrdinal) const;

  void getCubature(ArrayScalar &cubaturePoints, ArrayScalar &cubatureWeights, int maxTestDegree) const;

  void printInfo() const;
};

typedef Teuchos::RCP< MultiBasis<> > MultiBasisPtr;

} // namespace Camellia
#include "MultiBasisDef.h"

#endif