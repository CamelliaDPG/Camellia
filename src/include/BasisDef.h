//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Intrepid_Basis.hpp"

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  Basis<Scalar,ArrayScalar>::Basis() {
    _basisTagsAreSet = false;
  }

  template<class Scalar, class ArrayScalar>
  void Basis<Scalar,ArrayScalar>::CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    // for VALUE, GRAD, and DIV, we can say what happens to the rank:
    // (for CURL, it's different between 2D and 3D--and in 2D it depends on whether you're taking the curl of a scalar or a vector quantity)
    int UNKNOWN_RANK_CHANGE = -2;
    int rankChange=UNKNOWN_RANK_CHANGE;
    if (operatorType == Intrepid::OPERATOR_VALUE) {
      rankChange = 0;
    } else if (operatorType == Intrepid::OPERATOR_DIV) {
      rankChange = -1;
    } else if (operatorType == Intrepid::OPERATOR_GRAD) {
      rankChange = 1;
    }
    
    if (rankChange != UNKNOWN_RANK_CHANGE) {
      // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
      TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2 + rangeRank() + rankChange, std::invalid_argument, "values should have shape (F,P).");
    }
    // refPoints should have shape: (P,D)
    TEUCHOS_TEST_FOR_EXCEPTION(refPoints.rank() != 2, std::invalid_argument, "refPoints should have shape (P,D).");
    TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != domainTopology().getDimension(), std::invalid_argument, "refPoints should have shape (P,D).");
  }

  template<class Scalar, class ArrayScalar>
  shards::CellTopology Basis<Scalar,ArrayScalar>::domainTopology() const {
    return _domainTopology;
  }
  
  template<class Scalar, class ArrayScalar>
  int Basis<Scalar,ArrayScalar>::getCardinality() const {
    return this->_basisCardinality;
  }

  template<class Scalar, class ArrayScalar>
  int Basis<Scalar,ArrayScalar>::getDegree() const {
    return this->_basisDegree;
  }
  
  template<class Scalar, class ArrayScalar>
  int Basis<Scalar, ArrayScalar>::getDofOrdinal(const int subcDim,
                                                const int subcOrd,
                                                const int subcDofOrd) const {
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    // Use .at() for bounds checking
    int dofOrdinal = _tagToOrdinal.at(subcDim).at(subcOrd).at(subcDofOrd);
    TEUCHOS_TEST_FOR_EXCEPTION( (dofOrdinal == -1), std::invalid_argument,
                               ">>> ERROR (Basis): Invalid DoF tag");
    return dofOrdinal;
  }
  
  template<class Scalar,class ArrayScalar>
  const std::vector<std::vector<std::vector<int> > > & Basis<Scalar, ArrayScalar>::getDofOrdinalData( ) const
  {
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    return _tagToOrdinal;
  }
  
  
  template<class Scalar, class ArrayScalar>
  const std::vector<int>&  Basis<Scalar, ArrayScalar>::getDofTag(int dofOrd) const {
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    // Use .at() for bounds checking
    return _ordinalToTag.at(dofOrd);
  }

  template<class Scalar, class ArrayScalar>
  const std::vector<std::vector<int> > & Basis<Scalar, ArrayScalar>::getAllDofTags() const {
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    return _ordinalToTag;
  }

  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdge(int edgeIndex) const {
    int edgeDim = 1;
    std::set<int> dofOrdinals;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unfinished method");
    // TODO: figure out how best to set numDofsForEdge
    int numDofsForEdge;
    for (int edgeDofIndex=0; edgeDofIndex<numDofsForEdge; edgeDofIndex++) {
      int dofOrdinal = this->getDofOrdinal(edgeDim,edgeIndex,edgeDofIndex);
      dofOrdinals.insert(dofOrdinal);
    }
    return dofOrdinals;
  }

  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const {
    std::set<int> dofOrdinals = includeVertices ? this->dofOrdinalsForVertices() : std::set<int>();
    int numEdges = this->domainTopology().getEdgeCount();
    int edgeDim = 1;
    for (int edgeIndex=0; edgeIndex<numEdges; edgeIndex++) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unfinished method");
      // TODO: figure out how best to set numDofsForEdge
      int numDofsForEdge;
      for (int edgeDofIndex=0; edgeDofIndex<numDofsForEdge; edgeDofIndex++) {
        int dofOrdinal = this->getDofOrdinal(edgeDim,edgeIndex,edgeDofIndex);
        dofOrdinals.insert(dofOrdinal);
      }
    }
    return dofOrdinals;
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const {
    std::set<int> dofOrdinals = includeVerticesAndEdges ? this->dofOrdinalsForEdges(true) : std::set<int>();
    int numFaces = this->domainTopology().getFaceCount();
    int faceDim = 2;
    for (int faceIndex=0; faceIndex<numFaces; faceIndex++) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unfinished method");
      // TODO: figure out how best to set numDofsForFace
      int numDofsForFace;
      for (int faceDofIndex=0; faceDofIndex<numDofsForFace; faceDofIndex++) {
        int dofOrdinal = this->getDofOrdinal(faceDim,faceIndex,faceDofIndex);
        dofOrdinals.insert(dofOrdinal);
      }
    }
    return dofOrdinals;
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForInterior() const {
    std::set<int> dofOrdinals;
    int interiorDim = this->domainTopology().getDimension();
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method");
    // TODO: figure out the number of dofOrdinals for the interior, and use getDofOrdinal() to look them up.
    // (or use tags, like intrepid does)
  }

  template<class Scalar, class ArrayScalar>
  int Basis<Scalar,ArrayScalar>::dofOrdinalForVertex(int vertexIndex) const {
    int dofOrdinal = this->getDofOrdinal(0,vertexIndex,0);
    return dofOrdinal;
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForVertices() const {
    std::set<int> dofOrdinals;
    int numVertices = this->domainTopology().getVertexCount();
    for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
      int dofOrdinal = this->getDofOrdinal(0,vertexIndex,0);
      dofOrdinals.insert(dofOrdinal);
    }
    return dofOrdinals;
  }
  
  template<class Scalar, class ArrayScalar>
  bool Basis<Scalar,ArrayScalar>::isConforming() const {
    return false;
  }
  
  template<class Scalar, class ArrayScalar>
  bool Basis<Scalar,ArrayScalar>::isNodal() const {
    return false;
  }

  template<class Scalar, class ArrayScalar>
  int Basis<Scalar,ArrayScalar>::rangeDimension() const {
    return _rangeDimension;
  }
  
  template<class Scalar, class ArrayScalar>
  int Basis<Scalar,ArrayScalar>::rangeRank() const {
    return _rangeRank;
  }
  
  template<class Scalar, class ArrayScalar>
  bool IntrepidBasisWrapper<Scalar,ArrayScalar>::isConforming() const {
    return true;
  }
  
  template<class Scalar, class ArrayScalar>
  bool IntrepidBasisWrapper<Scalar,ArrayScalar>::isNodal() const {
    return true;
  }
  
  template<class Scalar, class ArrayScalar>
  IntrepidBasisWrapper<Scalar,ArrayScalar>::IntrepidBasisWrapper(Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis,
                                                                 int rangeDimension, int rangeRank) {
    _intrepidBasis = intrepidBasis;
    this->_rangeDimension = rangeDimension;
    this->_rangeRank = rangeRank;
  }

  template<class Scalar, class ArrayScalar>
  int IntrepidBasisWrapper<Scalar,ArrayScalar>::getCardinality() const {
    return _intrepidBasis->getCardinality();
  }

  template<class Scalar, class ArrayScalar>
  int IntrepidBasisWrapper<Scalar,ArrayScalar>::getDegree() const {
    return _intrepidBasis->getDegree();
  }

  // domain info on which the basis is defined:

  template<class Scalar, class ArrayScalar>
  shards::CellTopology IntrepidBasisWrapper<Scalar,ArrayScalar>::domainTopology() const {
    return _intrepidBasis->getBaseCellTopology();
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::getSubcellDofs(int subcellDimStart, int subcellDimEnd) const {
    shards::CellTopology cellTopo = _intrepidBasis->getBaseCellTopology();
    std::set<int> indices;
    for (int subcellDim = subcellDimStart; subcellDim <= subcellDimEnd; subcellDim++) {
      int numSubcells = cellTopo.getSubcellCount(subcellDim);
      for (int subcellIndex=0; subcellIndex<numSubcells; subcellIndex++) {
        // check that there is at least one dof for the subcell before asking for the first one:
        if (   (_intrepidBasis->getDofOrdinalData().size() > subcellDim)
            && (_intrepidBasis->getDofOrdinalData()[subcellDim].size() > subcellIndex)
            && (_intrepidBasis->getDofOrdinalData()[subcellDim][subcellIndex].size() > 0) ) {
          int firstDofOrdinal = _intrepidBasis->getDofOrdinal(subcellDim, subcellIndex, 0);
          int numDofs = _intrepidBasis->getDofTag(firstDofOrdinal)[3];
          for (int dof=0; dof<numDofs; dof++) {
            indices.insert(_intrepidBasis->getDofOrdinal(subcellDim, subcellIndex, dof));
          }
        }
      }
    }
    return indices;
  }

  // dof ordinal subsets:
  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const {
    int edgeDim = 1;
    int subcellDimStart = includeVertices ? 0 : edgeDim;
    return getSubcellDofs(subcellDimStart, edgeDim);
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const {
    int faceDim = 2;
    int subcellDimStart = includeVerticesAndEdges ? 0 : faceDim;
    return getSubcellDofs(subcellDimStart, faceDim);
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForInterior() const {
    shards::CellTopology cellTopo = domainTopology();
    int dim = cellTopo.getDimension();
    return getSubcellDofs(dim, dim);
  }
  template<class Scalar, class ArrayScalar>
  std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForVertices() const {
    int vertexDim = 0;
    return getSubcellDofs(vertexDim, vertexDim);
  }
  
  template<class Scalar, class ArrayScalar>
  int IntrepidBasisWrapper<Scalar, ArrayScalar>::getDofOrdinal(const int subcDim,
                                                const int subcOrd,
                                                const int subcDofOrd) const {
    return _intrepidBasis->getDofOrdinal(subcDim,subcOrd,subcDofOrd);
  }
  
  template<class Scalar,class ArrayScalar>
  const std::vector<std::vector<std::vector<int> > > & IntrepidBasisWrapper<Scalar, ArrayScalar>::getDofOrdinalData( ) const
  {
    return _intrepidBasis->getDofOrdinalData();
  }
  
  
  template<class Scalar, class ArrayScalar>
  const std::vector<int>&  IntrepidBasisWrapper<Scalar, ArrayScalar>::getDofTag(int dofOrd) const {
    return _intrepidBasis->getDofTag(dofOrd);
  }
  
  template<class Scalar, class ArrayScalar>
  const std::vector<std::vector<int> > & IntrepidBasisWrapper<Scalar, ArrayScalar>::getAllDofTags() const {
    return _intrepidBasis->getAllDofTags();
  }

  template<class Scalar, class ArrayScalar>
  void IntrepidBasisWrapper<Scalar,ArrayScalar>::initializeTags() const {
    // we leave tag initialization to the _intrepidBasis object.
  }
  
  template<class Scalar, class ArrayScalar>
  void IntrepidBasisWrapper<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
    return _intrepidBasis->getValues(values,refPoints,operatorType);
  }
} // namespace Camellia