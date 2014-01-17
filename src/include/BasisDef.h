//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Intrepid_Basis.hpp"
//#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  Basis<Scalar,ArrayScalar>::Basis() {
    _basisTagsAreSet = false;
    _functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_UNKNOWN;
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
    if (refPoints.rank() != 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(refPoints.rank() != 2, std::invalid_argument, "refPoints should have shape (P,D).");
    }
    if ( refPoints.dimension(1) != domainTopology().getDimension() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != domainTopology().getDimension(), std::invalid_argument, "refPoints should have shape (P,D).");
    }
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
//    std::cout << "int Basis<Scalar, ArrayScalar>::getDofOrdinal" << std::endl;
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
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex) const {
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    std::set<int> dofOrdinals;
    try {
      // Use .at() for bounds checking
      int firstDofOrdinal = this->_tagToOrdinal.at(subcellDim).at(subcellIndex).at(0);
      if (firstDofOrdinal == -1) { // no matching dof ordinals
        return dofOrdinals;
      }
      int numDofs = _tagToOrdinal[subcellDim][subcellIndex].size();
      
      for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
        int dofOrdinal = _tagToOrdinal.at(subcellDim).at(subcellIndex).at(dofIndex); // -1 indicates invalid entry...
        if (dofOrdinal >= 0) {
          dofOrdinals.insert(dofOrdinal);
        }
      }
    } catch (std::out_of_range e) {
      // we can be out of range if there aren't any dofs defined, but not otherwise...
      if ( dofOrdinals.size() > 0) { // that's unexpected, then.
        throw e;
      }
    }
    return dofOrdinals;
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex, int minimumSubSubcellDimension) const {
    if (minimumSubSubcellDimension > subcellDim) {
      return std::set<int>();
    }
    // to start, get the ones defined on the subcell itself:
    std::set<int> dofOrdinals = this->dofOrdinalsForSubcell(subcellDim, subcellIndex);
    
    const CellTopologyData *cellTopoData = this->domainTopology().getCellTopologyData();
    const CellTopologyData *sideTopoData = this->domainTopology().getCellTopologyData(subcellDim, subcellIndex);
    
    // Now, we want to look up all the subcells of the "side" of dimension > minimumSubSubcellDimension
    // For each of these, we need to:
    //  - find the appropriate subcell index in the larger topology
    //  - add all dof ordinals associated with that subcell index
    
    // ssc: sub-subcell
    for (int d_ssc = minimumSubSubcellDimension; d_ssc < subcellDim; d_ssc++) {
      int numSubSubcells_d = sideTopoData->subcell_count[d_ssc]; // numSubSubcells of dim d
      for (int ssc=0; ssc<numSubSubcells_d; ssc++) { //
        int numNodes_ssc = sideTopoData->subcell[d_ssc][ssc].topology->node_count;
        std::set<int> cellNodeIndices;
        for (int i=0; i < numNodes_ssc; i++) {
          unsigned subcellNodeIndex = sideTopoData->subcell[d_ssc][ssc].node[i];
          unsigned cellNodeIndex = cellTopoData->subcell[subcellDim][subcellIndex].node[subcellNodeIndex];
          cellNodeIndices.insert(cellNodeIndex);
        }
        // this is a bit involved, because CellTopology doesn't give a way to go from a set of nodes to a subcell defined on those nodes
        
        // now, examine each of the d_ssc-dimensional subcells of cell to look for one that matches all cellNodeIndices
        unsigned numSubcells = cellTopoData->subcell_count[d_ssc];
        unsigned matchingSubcellIndex = numSubcells;
        
        if (d_ssc==0) { // vertex
          // map the nodeIndex ssc in the subcell to the cellNodeIndex in the cell
          matchingSubcellIndex = cellTopoData->subcell[subcellDim][subcellIndex].node[ssc];
        } else {
          for (unsigned sc=0; sc<numSubcells; sc++) {
            bool matches = true;
            if (numNodes_ssc != cellTopoData->subcell[d_ssc][sc].topology->node_count) {
              matches = false;
            } else {
              for (int i=0; i < numNodes_ssc; i++) {
                unsigned cellNodeIndex = cellTopoData->subcell[d_ssc][sc].node[i];
                if (cellNodeIndices.find(cellNodeIndex) == cellNodeIndices.end()) {
                  matches = false;
                  break;
                }
              }
            }
            if (matches == true) {
              matchingSubcellIndex = sc;
              break;
            }
          }
          TEUCHOS_TEST_FOR_EXCEPTION(matchingSubcellIndex >= numSubcells, std::invalid_argument, "matching subcell not found");
        }
        std::set<int> dofOrdinals_ssc = this->dofOrdinalsForSubcell(d_ssc,matchingSubcellIndex);
        dofOrdinals.insert(dofOrdinals_ssc.begin(), dofOrdinals_ssc.end());
      }
    }
    return dofOrdinals;
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdge(int edgeIndex) const {
    int edgeDim = 1;
    return dofOrdinalsForSubcell(edgeDim, edgeIndex);
  }

  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const {
    std::set<int> dofOrdinals;
    if (!_basisTagsAreSet) {
      initializeTags();
      _basisTagsAreSet = true;
    }
    if ((subcellDim > 0) && includeLesserDimensions) {
      dofOrdinals = dofOrdinalsForSubcells(subcellDim-1,true);
    }
    if (this->_tagToOrdinal.size() < subcellDim+1) { // none of dimension subcellDim
      return dofOrdinals;
    }
    
    int numSubcells = this->_tagToOrdinal[subcellDim].size();
    for (int subcellIndex=0; subcellIndex<numSubcells; subcellIndex++) {
      std::set<int> dofOrdinalsForSubcell = this->dofOrdinalsForSubcell(subcellDim,subcellIndex);
      dofOrdinals.insert(dofOrdinalsForSubcell.begin(), dofOrdinalsForSubcell.end());
    }
    return dofOrdinals;
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const {
    int edgeDim = 1;
    return dofOrdinalsForSubcells(edgeDim,includeVertices);
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const {
    int faceDim = 2;
    return dofOrdinalsForSubcells(faceDim,includeVerticesAndEdges);
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForInterior() const {
    int interiorDim = this->domainTopology().getDimension();
    return dofOrdinalsForSubcells(interiorDim, false);
  }

  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForVertex(int vertexIndex) const {
    int vertexDim = 0;
    return this->dofOrdinalsForSubcell(vertexDim, vertexIndex);
  }
  
  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForVertices() const {
    int vertexDim = 0;
    return this->dofOrdinalsForSubcells(vertexDim, false);
  }
  
  template<class Scalar, class ArrayScalar>
  IntrepidExtendedTypes::EFunctionSpaceExtended Basis<Scalar,ArrayScalar>::functionSpace() const {
    return this->_functionSpace;
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
                                                                 int rangeDimension, int rangeRank,
                                                                 IntrepidExtendedTypes::EFunctionSpaceExtended fs) {
    _intrepidBasis = intrepidBasis;
    this->_rangeDimension = rangeDimension;
    this->_rangeRank = rangeRank;
    this->_functionSpace = fs;
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
    // we leave tag initialization to the _intrepidBasis object, but we'll keep a copy ourselves:
    this->_tagToOrdinal = _intrepidBasis->getDofOrdinalData();
    this->_ordinalToTag = _intrepidBasis->getAllDofTags();
  }
  
  template<class Scalar, class ArrayScalar>
  Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > IntrepidBasisWrapper<Scalar,ArrayScalar>::intrepidBasis() {
    return _intrepidBasis;
  }
  
  template<class Scalar, class ArrayScalar>
  void IntrepidBasisWrapper<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
    return _intrepidBasis->getValues(values,refPoints,operatorType);
  }
  
} // namespace Camellia