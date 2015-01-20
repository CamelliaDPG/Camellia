//
//  PatchBasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
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

#include "Intrepid_CellTools.hpp"

template<class Scalar, class ArrayScalar>
PatchBasis<Scalar, ArrayScalar>::PatchBasis(BasisPtr parentBasis, ArrayScalar &patchNodesInParentRefCell, shards::CellTopology &patchCellTopo) {
  this -> _parentBasis = parentBasis;
  this -> _patchNodesInParentRefCell = patchNodesInParentRefCell;
  this -> _patchCellTopo = Camellia::CellTopology::cellTopology(patchCellTopo);
  this -> _parentTopo = parentBasis->domainTopology();
  this -> _functionSpace = parentBasis->functionSpace();
  
  if (patchCellTopo.getKey() != shards::Line<2>::key ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PatchBasis only supports lines right now.");
  }
  
  if (_patchNodesInParentRefCell.rank() != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "patchNodes should be rank 2.");
  }
  
  if (_parentTopo->getDimension() != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PatchBasis only supports lines right now.");
  }
  
  // resize _patchNodesInParentRefCell for convenience in mapToPhysicalFrame:
  _patchNodesInParentRefCell.resize(1,_patchNodesInParentRefCell.dimension(0),_patchNodesInParentRefCell.dimension(1));
  
  // I don't think we actually need _parentRefNodes...
  // TODO: delete this
  // otherwise, set parent's ref nodes to be that of Line<2>
  _parentRefNodes = ::Intrepid::FieldContainer<double>(1,2,1); // for convenience in mapToPhysicalFrame, make this a rank-3 container
  _parentRefNodes(0,0,0) = 0.0;
  _parentRefNodes(0,1,0) = 1.0;
  
  //  _childRefNodes = Intrepid::FieldContainer<double>(1,2,1);
  //  _childRefNodes(0,0,0) = -1.0;
  //  _childRefNodes(0,1,0) = 1.0;
  
  // in 1D, each subRefCell ought to have 2 nodes
  if (_patchNodesInParentRefCell.dimension(1) != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "patchNodes requires two nodes per line segment.");
  }
  if (_patchNodesInParentRefCell.dimension(2) != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PatchBasis requires patchNodes to have dimensions (numNodesPerCell,spaceDim).  Right now, spaceDim must==1.");
  }
  
  this -> _basisCardinality  = _parentBasis->getCardinality();
  this -> _basisDegree       = _parentBasis->getDegree();
//  this -> basisCellTopology_ = _patchCellTopo;
//  this -> basisTagsAreSet_   = false;
  // TODO: figure out what to do about tag initialization...
}

template<class Scalar, class ArrayScalar>
void PatchBasis<Scalar, ArrayScalar>::getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                                                const Intrepid::EOperator operatorType) const {
  // compute cellJacobian, etc. for inputPoints:
  // inputPoints dimensions (P, D)
  // outputValues dimensions (F,P), (F,P,D), or (F,P,D,D)
  
  int numPoints = inputPoints.dimension(0);
  int spaceDim = inputPoints.dimension(1);
  if (spaceDim != _patchCellTopo->getDimension() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "spaceDim != _patchCellTopo.getDimension()");
  }
  if (spaceDim != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "spaceDim != 1");
  }
  if (operatorType != ::Intrepid::OPERATOR_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "PatchBasis only supports OPERATOR_VALUE right now.");
  }
  
  typedef ::Intrepid::CellTools<double>  CellTools;
  ArrayScalar parentInputPoints(numPoints,spaceDim);
  
  // first, transform the inputPoints into parent's reference frame:
  int parentCellIndex = 0;
  CellTools::mapToPhysicalFrame (parentInputPoints, inputPoints, _patchNodesInParentRefCell, _parentTopo->getShardsTopology(), parentCellIndex);
  //  cout << "_patchNodesInParentRefCell:\n" << _patchNodesInParentRefCell;
  //  cout << "inputPoints:\n" << inputPoints;
  //  cout << "parentInputPoints:\n" << parentInputPoints;
  
  _parentBasis->getValues(outputValues, parentInputPoints, operatorType);
  
  // TODO: for 3D meshes, figure out whether we need to do a value transform
  // (we get away without figuring this out in 2D because all our interfaces are 1D fluxes and traces, computed
  //  with an H1 basis--so that the value transform is just the identity…)
  
  // finally, transform values using the cellJacobian (for H1 values, this will be an identity transform)
  // transform the values back to this reference cell
  //  ArrayScalar cellJacobian,cellJacobInv,cellJacobDet;
  //  computeCellJacobians(cellJacobian,cellJacobInv,cellJacobDet, parentInputPoints);
  ////
  //  Teuchos::RCP< ArrayScalar > transformedValues = BasisEvaluation::getTransformedValues(_parentBasis,
  //                                                                  (IntrepidExtendedTypes::EOperator)operatorType,
  //                                                                  parentInputPoints,
  //                                                                  cellJacobian, cellJacobInv, cellJacobDet);
  // copy back to outputValues
  //  outputValues = *transformedValues;
}

//template<class Scalar, class ArrayScalar>
//void PatchBasis<Scalar, ArrayScalar>::initializeTags() {
//  // TODO: finish implementing this -- not quite clear on what the internal_dof/edge_dof should be, but it might need to know about whether this particular patch shares the parent's edges, which isn't something we presently have determined.
//  // TODO: generalize to 2D
//  
//  //cout << "PatchBasis<Scalar, ArrayScalar>::initializeTags() called.\n";
//  
//  // The following adapted from Basis_HGRAD_LINE_Cn_FEM
//  // unlike there, we do assume that the edge's endpoints are included in the first and last bases...
//  
//  // Basis-dependent initializations
//  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
//  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
//  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
//  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
//  
//  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
//  
//  int N = this->getCardinality();
//  
//  int *tags = new int[ tagSize * N ];
//  
//  int internal_dof = N - 2; // all but the endpoints
//  int edge_dof;
//  
//  tags[0] = 0;
//  tags[1] = 0;
//  tags[2] = 0;
//  tags[3] = 1;
//  edge_dof = 1;
//  
//  int n = N-1;
//  
//  for (int i=1;i < n;i++) {
//    tags[4*i] = 1;
//    tags[4*i+1] = 0;
//    tags[4*i+2] = -edge_dof + i;
//    tags[4*i+3] = internal_dof;
//  }
//  tags[4*n] = 0;
//  tags[4*n+1] = 1;
//  tags[4*n+2] = 0;
//  tags[4*n+3] = 1;
//  
//  Intrepid::setOrdinalTagData(this -> tagToOrdinal_,
//                              this -> ordinalToTag_,
//                              tags,
//                              this -> basisCardinality_,
//                              tagSize,
//                              posScDim,
//                              posScOrd,
//                              posDfOrd);
//  
//  delete []tags;
//}

// private method:
template<class Scalar, class ArrayScalar>
void PatchBasis<Scalar, ArrayScalar>::computeCellJacobians(ArrayScalar &cellJacobian, ArrayScalar &cellJacobInv,
                                                           ArrayScalar &cellJacobDet, const ArrayScalar &parentInputPoints) const {
  // TODO: implement this.  (the below code is copied from MultiBasis.)
  // inputPointsSubRefCell: the points in *reference* coordinates, as seen by the reference sub-cell.
  // (i.e. for cubature points, we'd expect these to span (-1,1), not to be confined to, e.g., (-1,0).)
  //  int numPoints = inputPointsSubRefCell.dimension(0);
  //  int spaceDim = inputPointsSubRefCell.dimension(1);
  //  int numNodesPerCell = _subRefNodes.dimension(1);
  //
  //  cellJacobian.resize(1, numPoints, spaceDim, spaceDim);
  //  cellJacobInv.resize(1, numPoints, spaceDim, spaceDim);
  //  cellJacobDet.resize(1, numPoints);
  //
  //  typedef Intrepid::CellTools<double>  CellTools;
  //  CellTools::setJacobian(cellJacobian, parentInputPoints, _childRefNodes, _cellTopo);
  //  CellTools::setJacobianInv(cellJacobInv, cellJacobian );
  //  CellTools::setJacobianDet(cellJacobDet, cellJacobian );
}


template<class Scalar, class ArrayScalar>
CellTopoPtr PatchBasis<Scalar, ArrayScalar>::domainTopology() const {
  return _patchCellTopo;
}

template<class Scalar, class ArrayScalar>
void PatchBasis<Scalar, ArrayScalar>::initializeTags() const {
  // TODO: finish implementing this -- not quite clear on what the internal_dof/edge_dof should be, but it might need to know about whether this particular patch shares the parent's edges, which isn't something we presently have determined.
  // TODO: generalize to 2D
  
  //cout << "PatchBasis::initializeTags() called.\n";
  
  // The following adapted from Basis_HGRAD_LINE_Cn_FEM
  // unlike there, we do assume that the edge's endpoints are included in the first and last bases...
  
  // Basis-dependent initializations
  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
  
  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
  
  int N = this->getCardinality();
  
  int *tags = new int[ tagSize * N ];
  
  int internal_dof = N - 2; // all but the endpoints
  int edge_dof;
  
  tags[0] = 0;
  tags[1] = 0;
  tags[2] = 0;
  tags[3] = 1;
  edge_dof = 1;
  
  int n = N-1;
  
  for (int i=1;i < n;i++) {
    tags[4*i] = 1;
    tags[4*i+1] = 0;
    tags[4*i+2] = -edge_dof + i;
    tags[4*i+3] = internal_dof;
  }
  tags[4*n] = 0;
  tags[4*n+1] = 1;
  tags[4*n+2] = 0;
  tags[4*n+3] = 1;
  
  ::Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
                                this -> _ordinalToTag,
                                tags,
                                this -> _basisCardinality,
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
  
  delete []tags;
}

template<class Scalar, class ArrayScalar>
BasisPtr PatchBasis<Scalar, ArrayScalar>::nonPatchAncestorBasis() const {
  BasisPtr basis = _parentBasis;
  PatchBasis<Scalar, ArrayScalar>* patchBasis = dynamic_cast< PatchBasis<Scalar, ArrayScalar>*>(basis.get());
  
  while (patchBasis != NULL) {
    basis = patchBasis->parentBasis();
    patchBasis = dynamic_cast< PatchBasis<Scalar, ArrayScalar>*>(basis.get());
  }
  return basis;
}

template<class Scalar, class ArrayScalar>
BasisPtr PatchBasis<Scalar, ArrayScalar>::parentBasis() const {
  return _parentBasis;
}

template<class Scalar, class ArrayScalar>
int PatchBasis<Scalar, ArrayScalar>::rangeDimension() const {
  return _parentBasis->rangeDimension();
}

template<class Scalar, class ArrayScalar>
int PatchBasis<Scalar, ArrayScalar>::rangeRank() const {
  return _parentBasis->rangeRank();
}