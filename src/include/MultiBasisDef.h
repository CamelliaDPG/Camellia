//
//  MultiBasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
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

#include "BasisEvaluation.h"

#include "Mesh.h"

#include "Intrepid_CellTools.hpp"

template<class Scalar, class ArrayScalar>
MultiBasis<Scalar, ArrayScalar>::MultiBasis(vector< Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > > bases, ArrayScalar &subRefNodes, shards::CellTopology &cellTopo) {
  this -> _bases = bases;
  this -> _subRefNodes = subRefNodes;
  this -> _cellTopo = cellTopo;
  this -> _functionSpace = bases[0]->functionSpace();
  
  if (_cellTopo.getKey() != shards::Line<2>::key ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis only supports lines right now.");
  }
  
  if (_subRefNodes.rank() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subRefNodes should be rank 3.");
  }
  
  // in 1D, each subRefCell ought to have 2 nodes
  if (_subRefNodes.dimension(1) != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis requires two nodes per line segment.");
  }
  if (_subRefNodes.dimension(2) != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis requires subRefNodes to have dimensions (numSubRefCells,numNodesPerCell,spaceDim).  Right now, spaceDim must==1.");
  }
  
  int numSharedNodes = 0; // only valid for lines
  
  int basisCardinality = 0, basisDegree = 0;
  _numLeaves = 0;
  
  typename std::vector< Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > >::iterator basisIt;
  for (basisIt = _bases.begin(); basisIt != _bases.end(); basisIt++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > basis = *basisIt;
    basisCardinality += basis->getCardinality();
    basisDegree = max(basisDegree, basis->getDegree());
    
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(basis.get());
    
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      _numLeaves += multiBasis->numLeafNodes();
    } else {
      _numLeaves += 1;
    }
  }
  
  basisCardinality -= numSharedNodes;
  
  this -> _basisCardinality  = basisCardinality;
  this -> _basisDegree       = basisDegree;
  
  if (bases[0]->domainTopology().getKey() != _cellTopo.getKey() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis bases[0] must have baseCellTopo == cellTopo");
  }
  if (bases[1]->domainTopology().getKey() != _cellTopo.getKey() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis bases[1] must have baseCellTopo == cellTopo");
  }
//  this -> basisCellTopology_ = _cellTopo;
//  this -> basisType_         = bases[0]->getBasisType();
//  this -> basisCoordinates_  = bases[0]->getCoordinateSystem();
//  this -> basisTagsAreSet_   = false;
  // TODO: figure out what to do about tag initialization...
}

// whenever side cubature is determined (e.g. in BasisCache; elsewhere??), make sure this gets called when basis is MultiBasis...
template<class Scalar, class ArrayScalar>
void MultiBasis<Scalar, ArrayScalar>::getCubature(ArrayScalar &cubaturePoints, ArrayScalar &cubatureWeights, int maxTestDegree) const {
  typedef CellTools<Scalar>  CellTools;
  // maxTestDegree: the maximum degree of functions being integrated against us.
  int numBases = _bases.size();
  int spaceDim = _cellTopo.getDimension();
  vector< ArrayScalar > cubPointsVector, cubWeightsVector;
  int totalCubPoints=0;
  DefaultCubatureFactory<Scalar>  cubFactory;
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    ArrayScalar cubPointsForBasis, cubWeightsForBasis;
    int numCubPoints;
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(_bases[basisIndex].get());
    
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      multiBasis->getCubature(cubPointsForBasis, cubWeightsForBasis, maxTestDegree);
      numCubPoints = cubPointsForBasis.dimension(0);
    } else {
      int cubDegree = maxTestDegree + _bases[basisIndex]->getDegree();
      Teuchos::RCP<Cubature<Scalar> > cellCub = cubFactory.create(_cellTopo, cubDegree);
      numCubPoints = cellCub->getNumPoints();
      cubPointsForBasis.resize(numCubPoints, spaceDim);
      cubWeightsForBasis.resize(numCubPoints);
      cellCub->getCubature(cubPointsForBasis, cubWeightsForBasis);
    }
    Teuchos::Array<int> dimensions;
    cubPointsForBasis.dimensions(dimensions);
    
    // map to quasi-physical frame (from the sub-ref cell to the ref cell)
    ArrayScalar cubPointsInRefCell(dimensions);
    CellTools::mapToPhysicalFrame (cubPointsInRefCell, cubPointsForBasis, _subRefNodes, _cellTopo, basisIndex);
    
    // weight according to (sub-)cell measure
    ArrayScalar cellJacobian,cellJacobInv,cellJacobDet;
    computeCellJacobians(cellJacobian,cellJacobInv,cellJacobDet, cubPointsForBasis,basisIndex);
    ArrayScalar weightedMeasure(1,numCubPoints);
    FunctionSpaceTools::computeCellMeasure<Scalar>(weightedMeasure,cellJacobDet,cubWeightsForBasis);
    weightedMeasure.resize(numCubPoints);
    
    cubPointsVector.push_back(cubPointsInRefCell);
    cubWeightsVector.push_back(weightedMeasure);
    totalCubPoints += numCubPoints;
  }
  cubaturePoints.resize(totalCubPoints,spaceDim);
  cubatureWeights.resize(totalCubPoints);
  int pointsEnumIndex = 0, weightsEnumIndex = 0; // for copying...
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    int basisCubPointsSize = cubPointsVector[basisIndex].size();
    for (int i=0; i<basisCubPointsSize; i++) {
      cubaturePoints[pointsEnumIndex++] = cubPointsVector[basisIndex][i];
    }
    int basisCubWeightsSize = cubWeightsVector[basisIndex].size();
    for (int i=0; i<basisCubWeightsSize; i++) {
      cubatureWeights[weightsEnumIndex++] = cubWeightsVector[basisIndex][i];
    }
  }
  //  cout << "MultiBasis: cubaturePoints:\n" << cubaturePoints;
  //  cout << "MultiBasis: cubatureWeights:\n" << cubatureWeights;
}

template<class Scalar, class ArrayScalar>
void MultiBasis<Scalar, ArrayScalar>::getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                                                const EOperator operatorType) const {
  // compute cellJacobian, etc. for inputPoints:
  // inputPoints dimensions (P, D)
  // outputValues dimensions (F,P), (F,P,D), or (F,P,D,D)
  // each basis is nonzero for just some of the points
  // -- the ones inside the (convex hull of the) appropriate entry in subRefNodes
  int numPoints = inputPoints.dimension(0);
  int numSubRefCells = _subRefNodes.dimension(0);
  //  int numNodesPerCell = _subRefNodes.dimension(1);
  int spaceDim = inputPoints.dimension(1);
  if (spaceDim != _cellTopo.getDimension() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "spaceDim != _cellTopo.getDimension()");
  }
  if (spaceDim != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "spaceDim != 1");
  }
  //map<int,int> subRefCellForPoint; // key: pointIndex; value: subRefCellIndex
  map<int,vector<int> > pointsForSubRefCell; // key: subRefCellIndex; values: vector<pointIndex>
  
  typedef CellTools<Scalar>  CellTools;
  for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
    // TODO: extend this to a more general determination of whether the point is in the convex hull
    // (which would work for 2D or 3D--what's here is 1D specific)
    double x = inputPoints(pointIndex,0);
    for (int cellIndex=0; cellIndex<numSubRefCells; cellIndex++) {
      // in 1D, each subRefCell ought to have 2 nodes
      double xMin = min(_subRefNodes(cellIndex,0,0), _subRefNodes(cellIndex,1,0));
      double xMax = max(_subRefNodes(cellIndex,0,0), _subRefNodes(cellIndex,1,0));
      if ( (x >= xMin) && (x <= xMax) ) {
        //subRefCellForPoint[pointIndex] = cellIndex;
        pointsForSubRefCell[cellIndex].push_back(pointIndex);
        // break; // we've found our match, so break out of cellIndex for loop
        // (commented out the above, because vertex points may lie on *two* sub-elements)
      }
    }
  }
  
  int numBases = _bases.size();
  
  Teuchos::Array<int> dimensions; // for basisValues
  outputValues.dimensions(dimensions);
  Teuchos::Array<int> outputValueLocation = dimensions; // just to copy the size
  int numValuesPerPoint = 1;
  for (int i=0; i<outputValueLocation.size(); i++) {
    outputValueLocation[i] = 0;
    if (i > 1) {
      numValuesPerPoint *= dimensions[i];
    }
  }
  
  outputValues.initialize(0.0); // set 0s: the sub-bases only have support on their subRefCells
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > basis = _bases[basisIndex];
    int refCellIndex = basisIndex; // the one for this basis
    // collect input points and map them to the ref cell for basis
    int numPointsForSubRefCell = pointsForSubRefCell[refCellIndex].size();
    if (numPointsForSubRefCell == 0) {
      outputValueLocation[0] += basis->getCardinality();
      continue; // next basisIndex
    }
    ArrayScalar inputPointsQuasiPhysical(numPointsForSubRefCell,spaceDim);
    for (int pointIndexIndex=0; pointIndexIndex<numPointsForSubRefCell; pointIndexIndex++) {
      int pointIndex = pointsForSubRefCell[refCellIndex][pointIndexIndex];
      for (int dim=0; dim<spaceDim; dim++) {
        inputPointsQuasiPhysical(pointIndexIndex,dim) = inputPoints(pointIndex,dim);
      }
    }
    ArrayScalar inputPointsRefCell(numPointsForSubRefCell,spaceDim);
    
    CellTools::mapToReferenceFrame (inputPointsRefCell, inputPointsQuasiPhysical, _subRefNodes, _cellTopo, refCellIndex);
    
    // now get all the values for basis (including the ones we'll skip)
    int basisCardinality = basis->getCardinality();
    dimensions[0] = basisCardinality;
    dimensions[1] = numPointsForSubRefCell;
    //ArrayScalar basisOutputValues(dimensions);
    //basis->getValues(basisOutputValues,inputPointsRefCell,operatorType);
    
    // transform the values back to this reference cell
    ArrayScalar cellJacobian,cellJacobInv,cellJacobDet;
    computeCellJacobians(cellJacobian,cellJacobInv,cellJacobDet, inputPointsRefCell,refCellIndex);
    
    Teuchos::RCP< ArrayScalar > transformedValues = BasisEvaluation::getTransformedValues(basis,
                                                                                          (IntrepidExtendedTypes::EOperatorExtended)operatorType,
                                                                                          inputPointsRefCell,
                                                                                          cellJacobian, cellJacobInv, cellJacobDet);
    
    //    cout << "transformedValues for basis " << basisIndex << ":\n" << *transformedValues;
    Teuchos::Array<int> basisValueLocation = outputValueLocation;
    basisValueLocation.insert(basisValueLocation.begin(),0); // cell dimension
    // copy the values to the right spot in outputValues
    for (int fieldIndex=0; fieldIndex<basisCardinality; fieldIndex++) {
      // if the basis is not a multiBasis, then we should permute the field ordering
      // so that it matches the permutation expected relative to a neighbor's field order
      // copy the values from basisOutputValues into appropriate outputValues location
      basisValueLocation[1] = fieldIndex;
      for (int pointIndexIndex=0; pointIndexIndex<numPointsForSubRefCell; pointIndexIndex++) {
        int pointIndex = pointsForSubRefCell[refCellIndex][pointIndexIndex];
        outputValueLocation[1] = pointIndex;
        basisValueLocation[2] = pointIndexIndex;
        int outputEnumerationIndex = outputValues.getEnumeration(outputValueLocation);
        int basisEnumerationIndex = transformedValues->getEnumeration(basisValueLocation);
        for (int i=0; i<numValuesPerPoint; i++) {
          outputValues[i+outputEnumerationIndex] = (*transformedValues)[i+basisEnumerationIndex];
        }
      }
      outputValueLocation[0]++; // go to next fieldIndex in the multi-basis
    }
  }
  //  cout << "MultiBasis inputPoints:\n" << inputPoints;
  //  cout << "MultiBasis outputValues:\n" << outputValues;
}

/*
template<class Scalar, class ArrayScalar>
void MultiBasis<Scalar, ArrayScalar>::initializeTags() {
  // TODO: finish implementing this
  // TODO: generalize to 2D multiBasis
  
  //cout << "MultiBasis<Scalar, ArrayScalar>::initializeTags() called.\n";
  
  // we need to at least set this up for the first and last vertices
  int firstVertexDofOrdinal, secondVertexDofOrdinal;
  firstVertexDofOrdinal = _bases[0]->getDofOrdinal(0,0,0);
  Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > lastBasis = _bases[_bases.size() - 1];
  int lastBasisOrdinalOffset = this->_basisCardinality - lastBasis->getCardinality();
  secondVertexDofOrdinal = lastBasis->getDofOrdinal(0,1,0) + lastBasisOrdinalOffset;
  
  // The following adapted from Basis_HGRAD_LINE_Cn_FEM
  // unlike there, we do assume that the edge's endpoints are included in the first and last bases...
  
  // Basis-dependent initializations
  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
  
  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
  
  int N = this->getCardinality();
  
  // double-check that our assumptions about the sub-bases have not been violated:
  if (firstVertexDofOrdinal != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has first vertex dofOrdinal in unexpected spot." );
  }
  if (secondVertexDofOrdinal != N-1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has second vertex dofOrdinal in unexpected spot." );
  }
  
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
  
  Intrepid::setOrdinalTagData(this -> tagToOrdinal_,
                              this -> ordinalToTag_,
                              tags,
                              this -> basisCardinality_,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
  
  delete []tags;
}*/

// private method:
template<class Scalar, class ArrayScalar>
void MultiBasis<Scalar, ArrayScalar>::computeCellJacobians(ArrayScalar &cellJacobian, ArrayScalar &cellJacobInv,
                                                           ArrayScalar &cellJacobDet, const ArrayScalar &inputPointsSubRefCell,
                                                           int subRefCellIndex) const {
  // inputPointsSubRefCell: the points in *reference* coordinates, as seen by the reference sub-cell.
  // (i.e. for cubature points, we'd expect these to span (-1,1), not to be confined to, e.g., (-1,0).)
  int numPoints = inputPointsSubRefCell.dimension(0);
  int spaceDim = inputPointsSubRefCell.dimension(1);
  int numNodesPerCell = _subRefNodes.dimension(1);
  
  cellJacobian.resize(1, numPoints, spaceDim, spaceDim);
  cellJacobInv.resize(1, numPoints, spaceDim, spaceDim);
  cellJacobDet.resize(1, numPoints);
  
  ArrayScalar thisSubRefNode(1,numNodesPerCell,spaceDim);
  
  for (int nodeIndex=0; nodeIndex<numNodesPerCell; nodeIndex++) {
    // in 3D, this will have to become the application of the neighbor's side symmetry to the points in subRefNodes
    // (i.e. a transformation of the points)
    int permutedNodeIndex = Mesh::neighborDofPermutation(nodeIndex,numNodesPerCell);
    for (int dim=0; dim<spaceDim; dim++) {
      thisSubRefNode(0,permutedNodeIndex,dim) = _subRefNodes(subRefCellIndex,nodeIndex,dim);
    }
  }
  
  typedef CellTools<Scalar>  CellTools;
  CellTools::setJacobian(cellJacobian, inputPointsSubRefCell, thisSubRefNode, _cellTopo);
  CellTools::setJacobianInv(cellJacobInv, cellJacobian );
  CellTools::setJacobianDet(cellJacobDet, cellJacobian );
}

template <class Scalar, class ArrayScalar>
void MultiBasis<Scalar,ArrayScalar>::initializeTags() const {
  // TODO: finish implementing this
  // TODO: generalize to 2D multiBasis
  
  //cout << "MultiBasis::initializeTags() called.\n";
  
  // we need to at least set this up for the first and last vertices
  int firstVertexDofOrdinal, secondVertexDofOrdinal;
  firstVertexDofOrdinal = _bases[0]->getDofOrdinal(0,0,0);
  BasisPtr lastBasis = _bases[_bases.size() - 1];
  int lastBasisOrdinalOffset = this->_basisCardinality - lastBasis->getCardinality();
  secondVertexDofOrdinal = lastBasis->getDofOrdinal(0,1,0) + lastBasisOrdinalOffset;
  
  // The following adapted from Basis_HGRAD_LINE_Cn_FEM
  // unlike there, we do assume that the edge's endpoints are included in the first and last bases...
  
  // Basis-dependent initializations
  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
  
  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
  
  int N = this->getCardinality();
  
  // double-check that our assumptions about the sub-bases have not been violated:
  if (firstVertexDofOrdinal != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has first vertex dofOrdinal in unexpected spot." );
  }
  if (secondVertexDofOrdinal != N-1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has second vertex dofOrdinal in unexpected spot." );
  }
  
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
  
  Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
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
int MultiBasis<Scalar, ArrayScalar>::numLeafNodes() const {
  return _numLeaves;
}

template<class Scalar, class ArrayScalar>
shards::CellTopology MultiBasis<Scalar, ArrayScalar>::domainTopology() const {
  return _bases[0]->domainTopology();
}


template<class Scalar, class ArrayScalar>
int MultiBasis<Scalar, ArrayScalar>::numSubBases() const {
  return _bases.size();
}

template<class Scalar, class ArrayScalar>
Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > MultiBasis<Scalar, ArrayScalar>::getSubBasis(int basisIndex) const {
  return _bases[basisIndex];
}

template<class Scalar, class ArrayScalar>
Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > MultiBasis<Scalar, ArrayScalar>::getLeafBasis(int leafOrdinal) const {
  int leafOrdinalOffset = 0;
  for (int subBasisIndex=0; subBasisIndex < _bases.size(); subBasisIndex++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > subBasis = _bases[subBasisIndex];
    int numLeaves = 1; // 1 if not MultiBasis
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(subBasis.get());
    
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      numLeaves = multiBasis->numLeafNodes();
    }
    if (leafOrdinal < leafOrdinalOffset + numLeaves) {
      // reachable by (or identical to) this subBasis
      if (multiBasis != NULL) {
        int relativeLeafOrdinal = leafOrdinal - leafOrdinalOffset;
        return multiBasis->getLeafBasis(relativeLeafOrdinal);
      } else {
        return subBasis;
      }
    }
    leafOrdinalOffset += numLeaves;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "leafOrdinal basis unreachable");
}

template<class Scalar, class ArrayScalar>
vector< pair<int,int> > MultiBasis<Scalar, ArrayScalar>::adjacentVertexOrdinals() const { // NOTE: prototype, untested code!
  // assumes that each basis has one vertex at each end, and the last one of one basis is adjacent to the
  // first of the next... (i.e. very much depends on being a 1D basis)
  // (we can also precompute this on construction, or lazily store construct and store it once asked...)
  int dofOffset = 0;
  vector< pair<int, int> > adjacencies;
  int numBases = _bases.size();
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > basis = _bases[basisIndex];
    
    if (basisIndex > 0) { // as in, this is not our first time through
      // then pair the last one in the previous basis with the first in this basis
      adjacencies.push_back( make_pair( dofOffset - 1, dofOffset ) );
    }
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(basis.get());
    
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      vector< pair<int,int> > subAdjacencies = multiBasis->adjacentVertexOrdinals();
      for (vector< pair<int,int> >::iterator adjIt = subAdjacencies.begin();
           adjIt != subAdjacencies.end(); adjIt++) {
        adjacencies.push_back( make_pair( adjIt->first + dofOffset, adjIt->second + dofOffset ) );
      }
    }
    dofOffset += basis->getCardinality();
  }
  return adjacencies;
}

template<class Scalar, class ArrayScalar>
int MultiBasis<Scalar, ArrayScalar>::getDofOrdinal(const int subcDim, const int subcOrd, const int subcDofOrd) const {
  if (subcDim!=0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis::getDofOrdinal() only supports vertices right now");
  }
  if (subcOrd>1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MultiBasis::getDofOrdinal() only supports 1D elements right now (lines, 2 vertices)");
  }
  if (subcDofOrd>0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only 1 dof defined for each vertex.");
  }
  int firstVertexDofOrdinal, secondVertexDofOrdinal;
  firstVertexDofOrdinal = _bases[0]->getDofOrdinal(0,0,0);
  Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > lastBasis = _bases[_bases.size() - 1];
  int lastBasisOrdinalOffset = this->_basisCardinality - lastBasis->getCardinality();
  secondVertexDofOrdinal = lastBasis->getDofOrdinal(0,1,0) + lastBasisOrdinalOffset;
  if (subcOrd==0) {
    return firstVertexDofOrdinal;
  } else {
    return secondVertexDofOrdinal;
  }
}

template<class Scalar, class ArrayScalar>
int MultiBasis<Scalar, ArrayScalar>::rangeDimension() const {
  return _bases[0]->rangeDimension();
}

template<class Scalar, class ArrayScalar>
int MultiBasis<Scalar, ArrayScalar>::rangeRank() const {
  return _bases[0]->rangeRank();
}

template<class Scalar, class ArrayScalar>
int MultiBasis<Scalar, ArrayScalar>::relativeToAbsoluteDofOrdinal(int basisDofOrdinal, int leafOrdinal) const {
  int numBases = _bases.size();
  int maxReachableLeaf = 0; // for a given basis
  int previousMaxReachable = 0;
  Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > basis; // the basis that contains the leaf...
  int dofOffset = 0;
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    basis = _bases[basisIndex];
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(basis.get());
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      maxReachableLeaf += multiBasis->numLeafNodes();
    } else {
      maxReachableLeaf += 1;
    }
    if (leafOrdinal < maxReachableLeaf) {
      MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(basis.get());
      
      if ( multiBasis != NULL ) { // basis is a MultiBasis, then
        return dofOffset + multiBasis->relativeToAbsoluteDofOrdinal(basisDofOrdinal, leafOrdinal - previousMaxReachable);
      } else {
        return dofOffset + basisDofOrdinal;
        // (we'll need to be more careful in 3D!)
      }
    }
    dofOffset += basis->getCardinality();
    previousMaxReachable = maxReachableLeaf;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested leafOrdinal out of bounds");
  return -1;
}

template<class Scalar, class ArrayScalar>
void MultiBasis<Scalar, ArrayScalar>::printInfo() const {
  cout << "MultiBasis with " << _numLeaves << " leaves:\n";
  for (int leafOrdinal=0; leafOrdinal<_numLeaves; leafOrdinal++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > leafBasis = getLeafBasis(leafOrdinal);
    cout << "Leaf " << leafOrdinal << ": cardinality " << leafBasis->getCardinality() << endl;
  }
  int numBases = _bases.size();
  for (int basisIndex=0; basisIndex<numBases; basisIndex++) {
    Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > basis = _bases[basisIndex];
    cout << "*** sub-basis " << basisIndex << " ***\n";
    MultiBasis<Scalar, ArrayScalar>* multiBasis = dynamic_cast< MultiBasis<Scalar, ArrayScalar>*>(basis.get());
    
    if ( multiBasis != NULL ) { // basis is a MultiBasis, then
      multiBasis->printInfo();
    } else {
      cout << "Leaf basis: cardinality " << basis->getCardinality() << endl;
    }
  }
}