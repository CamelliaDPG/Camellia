// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include <Intrepid2_TensorBasis.hpp>

#include "Camellia_KokkosHelpers.h"

namespace Camellia
{
template<class Scalar, class ArrayScalar>
Basis<Scalar,ArrayScalar>::Basis()
{
  _basisTagsAreSet = false;
  _functionSpace = Camellia::FUNCTION_SPACE_UNKNOWN;
}

template<class Scalar, class ArrayScalar>
Intrepid2::BasisValues<Scalar,Camellia::DeviceType> Basis<Scalar,ArrayScalar>::allocateBasisValues( const Intrepid2::TensorPoints<Scalar,Camellia::DeviceType> &points,
                                                                                                             const Intrepid2::EOperator op) const
{
  // base implementation throws exception; Intrepid2BasisWrapper overrides
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This is only valid for Intrepid2 bases");
}

template<class Scalar, class ArrayScalar>
void Basis<Scalar,ArrayScalar>::getValues( Intrepid2::BasisValues<Scalar,Camellia::DeviceType> &outputValues,
                                          const Intrepid2::TensorPoints<Scalar,Camellia::DeviceType>  &inputPoints,
                                          const Intrepid2::EOperator op ) const
{
  // base implementation throws exception; Intrepid2BasisWrapper overrides
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This is only valid for Intrepid2 bases");
}

template<class Scalar, class ArrayScalar>
void Basis<Scalar,ArrayScalar>::CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const
{
  // for VALUE, GRAD, and DIV, we can say what happens to the rank:
  // (for CURL, it's different between 2D and 3D--and in 2D it depends on whether you're taking the curl of a scalar or a vector quantity)
  int UNKNOWN_RANK_CHANGE = -2;
  int rankChange=UNKNOWN_RANK_CHANGE;

  if (operatorType == OP_VALUE)
  {
    rankChange = 0;
  }
  else if (operatorType == OP_DIV)
  {
    rankChange = -1;
  }
  else if (operatorType == OP_GRAD)
  {
    rankChange = 1;
  }
  if (operatorType == OP_CURL)
  {
    if (this->rangeDimension() == 3)
    {
      rankChange = 0;
    }
    else if (this->rangeDimension() == 2)
    {
      if (this->rangeRank() == 0)
      {
        rankChange = 1;
      }
      else if (this->rangeRank() == 1)
      {
        rankChange = -1;
      }
    }
  }

  if (rankChange != UNKNOWN_RANK_CHANGE)
  {
    // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
    if (values.rank() != 2 + rangeRank() + rankChange)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2 + rangeRank() + rankChange, std::invalid_argument, "values should have shape (F,P,[D,D,...]).");
    }
    for (int d=0; d<rangeRank() + rankChange; d++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2+d) != rangeDimension(), std::invalid_argument, "values should have shape (F,P,[D,D,...]).");
    }
  }
  // refPoints should have shape: (P,D)
  if (refPoints.rank() != 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(refPoints.rank() != 2, std::invalid_argument, "refPoints should have shape (P,D).");
  }
  const int domainDim = domainTopology()->getDimension();
  if (refPoints.dimension(1) != domainDim)
  {
    // we allow the two dimensions to differ in exactly one case: bases defined on the node topology, where we allow refPoints dimension to be 1.
    const bool nodeTopo = (refPoints.dimension(1) == 1) && (domainDim == 0);
    if (!nodeTopo)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != domainDim, std::invalid_argument, "refPoints should have shape (P,D).");
    }
  }
}

template<class Scalar, class ArrayScalar>
Teuchos::RCP< ::Camellia::Basis<Scalar, ArrayScalar> >  Basis<Scalar,ArrayScalar>::continuousBasis()
{
  if (_continuousBasis != Teuchos::null)
  {
    return _continuousBasis;
  }
  else
  {
    return Teuchos::rcp(this, false);
  }
}

template<class Scalar, class ArrayScalar>
CellTopoPtr Basis<Scalar,ArrayScalar>::domainTopology() const
{
  return _domainTopology;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::getCardinality() const
{
  return this->_basisCardinality;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::getDegree() const
{
  return this->_basisDegree;
}
  
  template<class Scalar, class ArrayScalar>
  void Basis<Scalar, ArrayScalar>::getDofCoords(ArrayScalar & DofCoords) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getDofCoords not implemented by basis!");
  }
  
template<class Scalar, class ArrayScalar>
int Basis<Scalar, ArrayScalar>::getDofOrdinal(const int subcDim,
    const int subcOrd,
    const int subcDofOrd) const
{
//    std::cout << "int Basis<Scalar, ArrayScalar>::getDofOrdinal" << std::endl;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
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
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  return _tagToOrdinal;
}


template<class Scalar, class ArrayScalar>
const std::vector<int>&  Basis<Scalar, ArrayScalar>::getDofTag(int dofOrd) const
{
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  // Use .at() for bounds checking
  return _ordinalToTag.at(dofOrd);
}

template<class Scalar, class ArrayScalar>
const std::vector<std::vector<int> > & Basis<Scalar, ArrayScalar>::getAllDofTags() const
{
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  return _ordinalToTag;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::getNumTensorialComponents() const
{
  using BasisBase   = Intrepid2::Basis<Camellia::DeviceType,double,double>;
  using TensorBasis = Intrepid2::Basis_TensorBasis<BasisBase>;
  
  auto tensorBasis = dynamic_cast<TensorBasis*>(this->intrepid2Basis().get());
  if (tensorBasis != NULL)
  {
    return tensorBasis->getTensorBasisComponents().size();
  }
  else
  {
    return 1; // trivial tensor structure
  }
}

  template<class Scalar, class ArrayScalar>
  std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSide(int sideOrdinal) const
  {
    int sideDim = this->_domainTopology->getDimension() - 1;
    std::vector<int> dofOrdinals = this->dofOrdinalsForSubcell(sideDim, sideOrdinal, 0);
    std::set<int> dofOrdinalsSet(dofOrdinals.begin(),dofOrdinals.end());
    return dofOrdinalsSet;
  }
  
template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex) const
{
//  static std::vector<int> emptyVector;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
//  std::vector<int> dofOrdinals;
//  int firstDofOrdinal = -1;
//  if (_tagToOrdinal.size() > subcellDim)
//  {
//    if (_tagToOrdinal[subcellDim].size() > subcellIndex)
//    {
//      if (_tagToOrdinal[subcellDim][subcellIndex].size() > 0)
//      {
//        firstDofOrdinal = _tagToOrdinal[subcellDim][subcellIndex][0];
//      }
//    }
//  }
//  if (firstDofOrdinal == -1)   // no matching dof ordinals
//  {
//    return emptyVector;
//  }
//  int numDofs = _ordinalToTag.at(firstDofOrdinal)[3];
////  int numDofs = _tagToOrdinal[subcellDim][subcellIndex].size();
//
//  for (int dofIndex=0; dofIndex<numDofs; dofIndex++)
//  {
//    int dofOrdinal = _tagToOrdinal.at(subcellDim).at(subcellIndex).at(dofIndex); // -1 indicates invalid entry...
//    TEUCHOS_TEST_FOR_EXCEPTION(dofOrdinal < 0, std::invalid_argument, "invalid entry encountered in _tagToOrdinal");
//  }
  
  if (this->_functionSpace != Camellia::FUNCTION_SPACE_HDIV) // there appears to be a bug in Intrepid's HDIV basis for hexahedra, such that _ordinalToTag is not consistent with _tagToOrdinal.  _tagToOrdinal is OK, though, I think.
  { // DEBUGGING -- sanity checks:
    for (int dofOrdinal : _tagToOrdinal[subcellDim][subcellIndex])
    {
      TEUCHOS_TEST_FOR_EXCEPTION(dofOrdinal < 0, std::invalid_argument, "dofOrdinal can't be < 0");
    }
    int firstDofOrdinal = -1;
    if (int(_tagToOrdinal.size()) > subcellDim)
    {
      if (int(_tagToOrdinal[subcellDim].size()) > subcellIndex)
      {
        if (_tagToOrdinal[subcellDim][subcellIndex].size() > 0)
        {
          firstDofOrdinal = _tagToOrdinal[subcellDim][subcellIndex][0];
        }
      }
    }
    int numDofs;
    
    if (firstDofOrdinal == -1)   // no matching dof ordinals
    {
      numDofs = 0;
    }
    else
    {
      numDofs = _ordinalToTag.at(firstDofOrdinal)[3];
      if (numDofs == 0)
      {
        // inconsistent, but we can survive that
        // (bug in HDIV_HEX_In, at least when n=1; hopefully not otherwise)
        numDofs = 1;
      }
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(numDofs != int(_tagToOrdinal[subcellDim][subcellIndex].size()), std::invalid_argument, "wrong number of dofs in trimmed container");
  }
  
  return _tagToOrdinal[subcellDim][subcellIndex];
}
  
template<class Scalar, class ArrayScalar>
std::vector<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcell(int subcellDim, int subcellIndex, int minimumSubSubcellDimension) const
{
  if (minimumSubSubcellDimension > subcellDim)
  {
    return std::vector<int>();
  }
  // to start, get the ones defined on the subcell itself:
  std::vector<int> dofOrdinals = this->dofOrdinalsForSubcell(subcellDim, subcellIndex);

  CellTopoPtr cellTopo = this->domainTopology();
  CellTopoPtr sideTopo = cellTopo->getSubcell(subcellDim, subcellIndex);

//    const CellTopologyData *cellTopoData = this->domainTopology()->getCellTopologyData();
//    const CellTopologyData *sideTopoData = this->domainTopology()->getCellTopologyData(subcellDim, subcellIndex);

  // Now, we want to look up all the subcells of the "side" of dimension > minimumSubSubcellDimension
  // For each of these, we need to:
  //  - find the appropriate subcell index in the larger topology
  //  - add all dof ordinals associated with that subcell index

  // ssc: sub-subcell
  for (int d_ssc = minimumSubSubcellDimension; d_ssc < subcellDim; d_ssc++)
  {
    int numSubSubcells_d = sideTopo->getSubcellCount(d_ssc); // numSubSubcells of dim d
    for (int ssc=0; ssc<numSubSubcells_d; ssc++)   //
    {
      CellTopoPtr subSubCellTopo = sideTopo->getSubcell(d_ssc,ssc);
      int numNodes_ssc = subSubCellTopo->getNodeCount();
      std::set<int> cellNodeIndices;
      for (int i=0; i < numNodes_ssc; i++)
      {
        unsigned subcellNodeIndex = sideTopo->getNodeMap(d_ssc, ssc, i); //  ->subcell[d_ssc][ssc].node[i];
        unsigned cellNodeIndex = cellTopo->getNodeMap(subcellDim, subcellIndex, subcellNodeIndex); //cellTopoData->subcell[subcellDim][subcellIndex].node[subcellNodeIndex];
        cellNodeIndices.insert(cellNodeIndex);
      }
      // this is a bit involved, because CellTopology doesn't give a way to go from a set of nodes to a subcell defined on those nodes

      // now, examine each of the d_ssc-dimensional subcells of cell to look for one that matches all cellNodeIndices
      unsigned numSubcells = cellTopo->getSubcellCount(d_ssc);
      unsigned matchingSubcellIndex = numSubcells;

      if (d_ssc==0)   // vertex
      {
        // map the nodeIndex ssc in the subcell to the cellNodeIndex in the cell
        matchingSubcellIndex = cellTopo->getNodeMap(subcellDim, subcellIndex, ssc); // cellTopoData->subcell[subcellDim][subcellIndex].node[ssc];
      }
      else
      {
        for (unsigned sc=0; sc<numSubcells; sc++)
        {
          bool matches = true;
          if (numNodes_ssc != int(subSubCellTopo->getNodeCount()))
          {
            matches = false;
          }
          else
          {
            for (int i=0; i < numNodes_ssc; i++)
            {
              unsigned cellNodeIndex = cellTopo->getNodeMap(d_ssc,sc,i); // ->subcell[d_ssc][sc].node[i];
              if (cellNodeIndices.find(cellNodeIndex) == cellNodeIndices.end())
              {
                matches = false;
                break;
              }
            }
          }
          if (matches == true)
          {
            matchingSubcellIndex = sc;
            break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(matchingSubcellIndex >= numSubcells, std::invalid_argument, "matching subcell not found");
      }
      const std::vector<int> dofOrdinals_ssc = this->dofOrdinalsForSubcell(d_ssc,matchingSubcellIndex);
      dofOrdinals.insert(dofOrdinals.end(), dofOrdinals_ssc.begin(), dofOrdinals_ssc.end());
    }
  }
  std::sort(dofOrdinals.begin(), dofOrdinals.end());
  return dofOrdinals;
}

template<class Scalar, class ArrayScalar>
const std::vector<int>& Basis<Scalar,ArrayScalar>::dofOrdinalsForEdge(int edgeIndex) const
{
  static const int edgeDim = 1;
  return dofOrdinalsForSubcell(edgeDim, edgeIndex);
}


template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const
{
  std::set<int> dofOrdinals;
  if (!_basisTagsAreSet)
  {
    initializeTagsAndTrim();
    _basisTagsAreSet = true;
  }
  if ((subcellDim > 0) && includeLesserDimensions)
  {
    dofOrdinals = dofOrdinalsForSubcells(subcellDim-1,true);
  }
  if (int(this->_tagToOrdinal.size()) < subcellDim+1)   // none of dimension subcellDim
  {
    return dofOrdinals;
  }

  int numSubcells = this->domainTopology()->getSubcellCount(subcellDim);
  for (int subcellIndex=0; subcellIndex<numSubcells; subcellIndex++)
  {
    std::vector<int> dofOrdinalsForSubcell = this->dofOrdinalsForSubcell(subcellDim,subcellIndex);
    dofOrdinals.insert(dofOrdinalsForSubcell.begin(), dofOrdinalsForSubcell.end());
  }
  return dofOrdinals;
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const
{
  int edgeDim = 1;
  return dofOrdinalsForSubcells(edgeDim,includeVertices);
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const
{
  int faceDim = 2;
  return dofOrdinalsForSubcells(faceDim,includeVerticesAndEdges);
}

template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForInterior() const
{
  int interiorDim = this->domainTopology()->getDimension();
  return dofOrdinalsForSubcell(interiorDim, false);
}

template<class Scalar, class ArrayScalar>
const std::vector<int> &Basis<Scalar,ArrayScalar>::dofOrdinalsForVertex(int vertexIndex) const
{
  const static int vertexDim = 0;
  return this->dofOrdinalsForSubcell(vertexDim, vertexIndex);
}

template<class Scalar, class ArrayScalar>
std::set<int> Basis<Scalar,ArrayScalar>::dofOrdinalsForVertices() const
{
  const static int vertexDim = 0;
  return this->dofOrdinalsForSubcells(vertexDim, false);
}

template<class Scalar, class ArrayScalar>
Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace() const
{
  return this->_functionSpace;
}

template<class Scalar, class ArrayScalar>
Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace(int tensorialComponent) const
{
  using BasisBase   = Intrepid2::Basis<Camellia::DeviceType,double,double>;
  using TensorBasis = Intrepid2::Basis_TensorBasis<BasisBase>;
  
  auto tensorBasis = dynamic_cast<TensorBasis*>(this->intrepid2Basis().get());
  if (tensorBasis != NULL)
  {
    auto basisComponents = tensorBasis->getTensorBasisComponents();
    const int numBasisComponents = basisComponents.size();
    if (numBasisComponents > tensorialComponent)
    {
      auto intrepid2FS = basisComponents[tensorialComponent]->getFunctionSpace();
      return fromIntrepid2FunctionSpace(intrepid2FS);
    }
  }
  if (tensorialComponent==0)
  {
    return this->_functionSpace;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"tensorialComponent exceeds the tensorial degree of the basis");
  }
}
  
  template<class Scalar, class ArrayScalar>
  Camellia::EFunctionSpace Basis<Scalar,ArrayScalar>::functionSpace(int subcDim, int subcOrdinal) const
  {
    // TODO: determine how this is used, and whether we should revise it now that we've eliminated the space-time enums used in the code commented out below…
    return this->_functionSpace;
//    switch (this->_functionSpace)
//    {
//      case FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
//      case FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
//      {
//        TEUCHOS_TEST_FOR_EXCEPTION(this->_domainTopology()->getTensorialDegree() == 0, std::invalid_argument, "mixed function spaces only supported for tensorial domains");
//        CellTopoPtr subcell= this->_domainTopology()->getSubcell(subcDim,subcOrdinal);
//        if (subcell->getTensorialDegree() < this->_domainTopology()->getTensorialDegree())
//        {
//          // then this is the spatial part
//          return this->functionSpace(0); // 0: tensorial rank of spatial part
//        }
//        else
//        {
//          // any tensorial subcell other than Node x Line_2 has a mixed function space; Node x Line_2 has the temporal function space
//          if (subcell->getNodeCount() == 2)
//          {
//            return this->functionSpace(1); // 1: tensorial rank of temporal part
//          }
//          else
//          {
//            return this->_functionSpace;
//          }
//        }
//      }
//        break;
//      default:
//        return this->_functionSpace;
//    }
  }

  template<class Scalar, class ArrayScalar>
  void Basis<Scalar,ArrayScalar>::initializeTagsAndTrim() const
  {
    initializeTags();
    
    // trim so that _tagToOrdinal vectors have the right size (no -1 filling)
    CellTopoPtr domainTopo = this->domainTopology();
    int domainDim = domainTopo->getDimension();
    
    auto tagToOrdinalTrimmed = std::vector<std::vector<std::vector<int> > >(domainDim + 1);
    tagToOrdinalTrimmed.resize(domainDim + 1);
    
    for (int d=0; d<=domainDim; d++)
    {
      int subcellCount = domainTopo->getSubcellCount(d);
      tagToOrdinalTrimmed[d].resize(subcellCount);
      for (int subcOrd=0; subcOrd < subcellCount; subcOrd++)
      {
        vector<int> dofOrdinals;
        // some subclasses don't fill the container for subcells when subcells of a given dimension don't have any dofOrdinals defined...  We'll set it to be an empty vector<int> in that case...
        if ((d < int(this->_tagToOrdinal.size())) && (subcOrd < int(this->_tagToOrdinal[d].size())))
        {
          int numDofs = this->_tagToOrdinal[d][subcOrd].size();
          for (int dofIndex = 0; dofIndex < numDofs; dofIndex++)
          {
            int dofOrdinal = this->_tagToOrdinal[d][subcOrd][dofIndex];
            if (dofOrdinal >= 0)
            {
              dofOrdinals.push_back(dofOrdinal);
            }
          }
          std::sort(dofOrdinals.begin(),dofOrdinals.end());
        }
        tagToOrdinalTrimmed[d][subcOrd] = dofOrdinals;
      }
    }
    _tagToOrdinal = tagToOrdinalTrimmed;
  }

template<class Scalar, class ArrayScalar>
Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> >
Basis<Scalar,ArrayScalar>::intrepid2Basis() const
{
  // Intrepid2 basis wrapper overrides
  return Teuchos::null;
}

template<class Scalar, class ArrayScalar>
std::vector< Teuchos::RCP< ::Camellia::Basis<Scalar, ArrayScalar> > >
Basis<Scalar,ArrayScalar>::intrepid2TensorBasisComponents()
{
  if (_intrepid2TensorBasisComponents.size() > 0) return _intrepid2TensorBasisComponents;
  using BasisBase        = Intrepid2::Basis<Camellia::DeviceType,double,double>;
  using TensorBasis      = Intrepid2::Basis_TensorBasis<BasisBase>;
  
  // compute tensorial components:
  auto tensorBasis = dynamic_cast<TensorBasis*>(this->intrepid2Basis().get());

  if (tensorBasis == NULL)
  {
    _intrepid2TensorBasisComponents.push_back(Teuchos::rcp(this,false));
    return _intrepid2TensorBasisComponents;
  }

  auto basisComponents = tensorBasis->getTensorBasisComponents();
  
  for (const auto & basisComponent : basisComponents)
  {
    Camellia::EFunctionSpace fs = fromIntrepid2FunctionSpace(basisComponent->getFunctionSpace());
    const bool isVectorFS = (fs == FUNCTION_SPACE_HCURL) || (fs == FUNCTION_SPACE_HDIV) || functionSpaceIsVectorized(fs);
    const int rank = isVectorFS ? 1 : 0;
    const int spaceDim = basisComponent->getBaseCellTopology().getDimension();
    auto camelliaBasis = Teuchos::rcp(new Intrepid2BasisWrapper<Scalar,ArrayScalar>( basisComponent, spaceDim, rank, fs) );
    _intrepid2TensorBasisComponents.push_back(camelliaBasis);
  }
  return _intrepid2TensorBasisComponents;
}

template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isConforming() const
{
  return false;
}

template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isModal() const
{
  return false;
}
  
template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::isNodal() const
{
  return false;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::rangeDimension() const
{
  return _rangeDimension;
}

template<class Scalar, class ArrayScalar>
int Basis<Scalar,ArrayScalar>::rangeRank() const
{
  return _rangeRank;
}

template<class Scalar, class ArrayScalar>
bool Basis<Scalar,ArrayScalar>::supportsOperator(::Camellia::EOperator &op) const
{
  // default implementation returns true if the op is known to Intrepid/Intrepid2.
  const bool standardIntrepidOperator = (op >= Camellia::OP_VALUE) && (op <= Camellia::OP_D10);
  
  // subclasses should override insofar as they support other operators.
  return standardIntrepidOperator;
}

// IntrepidBasisWrapper method definitions removed; use Intrepid2BasisWrapper.

template<class Scalar, class ArrayScalar>
bool Intrepid2BasisWrapper<Scalar,ArrayScalar>::isConforming() const
{
  bool isDiscontinuous =  Camellia::functionSpaceIsDiscontinuous(this->_functionSpace); // returns true for L^2 spaces as well as HGRAD_DISC, etc.
  
  bool functionSpaceIsL2 = (this->_functionSpace == FUNCTION_SPACE_HVOL) || (this->_functionSpace == FUNCTION_SPACE_VECTOR_HVOL) || (this->_functionSpace == FUNCTION_SPACE_TENSOR_HVOL);
  
  if (isDiscontinuous && !functionSpaceIsL2)
    return false; // discontinuous and not L^2: definitely not conforming
  else if (!isDiscontinuous && functionSpaceIsL2)
    return false; // not discontinuous but L^2: definitely not conforming (and a bit weird!)
  else // not discontinuous
    return true;  // in the usual case, if not discontinuous, we are conforming.  There could be exceptions to this, but not among the Intrepid2 bases we expect to support.
}

template<class Scalar, class ArrayScalar>
bool Intrepid2BasisWrapper<Scalar,ArrayScalar>::isNodal() const
{
  if (_intrepid2Basis->getBasisType() == Intrepid2::BASIS_FEM_LAGRANGIAN)
    return true; // for Lagrangian, we know it's nodal
  else
    return false; // we're not sure about non-Lagrangian bases, but in general they are not nodal
}

template<class Scalar, class ArrayScalar>
Intrepid2BasisWrapper<Scalar,ArrayScalar>::Intrepid2BasisWrapper(Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> > intrepid2Basis,
    int rangeDimension, int rangeRank,
    Camellia::EFunctionSpace fs)
{
  _intrepid2Basis = intrepid2Basis;
  this->_rangeDimension = rangeDimension;
  this->_rangeRank = rangeRank;
  this->_functionSpace = fs;
  this->_basisCardinality = _intrepid2Basis->getCardinality();
  this->_basisDegree = _intrepid2Basis->getDegree();
  this->_domainTopology = CellTopology::cellTopology( _intrepid2Basis->getBaseCellTopology(), _intrepid2Basis->getNumTensorialExtrusions() );

  bool isDiscontinuous =  Camellia::functionSpaceIsDiscontinuous(this->_functionSpace);
  
  if (isDiscontinuous)
  {
    bool upgradeHVOL = true;
    Camellia::EFunctionSpace continuousFS = Camellia::continuousSpaceForDiscontinuous(fs,upgradeHVOL);
    this->_continuousBasis = Teuchos::rcp( new Intrepid2BasisWrapper(intrepid2Basis, rangeDimension,
                                                                     rangeRank, continuousFS));
  }
}

template<class Scalar, class ArrayScalar>
void Intrepid2BasisWrapper<Scalar,ArrayScalar>::initializeTags() const
{
  using OrdinalToTagType = typename Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar>::OrdinalTypeArray2DHost;
  using TagToOrdinalType = typename Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar>::OrdinalTypeArray3DHost;
    
  TagToOrdinalType tagToOrdinalIntrepid2 = _intrepid2Basis->getAllDofOrdinal();
  OrdinalToTagType ordinalToTagIntrepid2 = _intrepid2Basis->getAllDofTags();
  
  /*
     mutable std::vector<std::vector<int> > _ordinalToTag;
     mutable std::vector<std::vector<std::vector<int> > > _tagToOrdinal;
   */
  
  using std::vector;
  vector<int> ttoInnermost(tagToOrdinalIntrepid2.extent_int(2),-1);
  vector<vector<int>> ttoSecond(tagToOrdinalIntrepid2.extent_int(1),ttoInnermost);
  this->_tagToOrdinal = vector<vector<vector<int>>>(tagToOrdinalIntrepid2.extent_int(0),ttoSecond);
  
  vector<int> ottInner(ordinalToTagIntrepid2.extent_int(1));
  this->_ordinalToTag = vector<vector<int>>(ordinalToTagIntrepid2.extent_int(0),ottInner);
//  this->_tagToOrdinal = ArrayScalar(tagToOrdinalIntrepid2.extent_int(0),tagToOrdinalIntrepid2.extent_int(1),tagToOrdinalIntrepid2.extent_int(2));
//  this->_ordinalToTag = ArrayScalar(ordinalToTagIntrepid2.extent_int(0),ordinalToTagIntrepid2.extent_int(2));

  for (int i=0; i<tagToOrdinalIntrepid2.extent_int(0); i++)
  {
    for (int j=0; j<tagToOrdinalIntrepid2.extent_int(1); j++)
    {
      for (int k=0; k<tagToOrdinalIntrepid2.extent_int(2); k++)
      {
        this->_tagToOrdinal[i][j][k] = tagToOrdinalIntrepid2(i,j,k);
      }
    }
  }
  
  for (int i=0; i<ordinalToTagIntrepid2.extent_int(0); i++)
  {
    for (int j=0; j<ordinalToTagIntrepid2.extent_int(1); j++)
    {
      this->_ordinalToTag[i][j] = ordinalToTagIntrepid2(i,j);
    }
  }
  
  bool isDiscontinuous =  Camellia::functionSpaceIsDiscontinuous(this->_functionSpace);

  // if this is an L^2 basis (potentially wrapping a non-L^2 Intrepid basis--to date, Intrepid doesn't have any L^2 bases, so we usually use H^1 of one lower degree for L^2),
  // then we should rework the data structures a bit...
  if (isDiscontinuous)
  {
    std::vector<int> tag(4);
    int domainDimension = this->domainTopology()->getDimension();
    this->_ordinalToTag = std::vector< std::vector<int> >(this->_basisCardinality);
    this->_tagToOrdinal = std::vector<std::vector<std::vector<int> > >(domainDimension + 1);
    for (int d=0; d<=domainDimension; d++)
    {
      int subcellCount = this->domainTopology()->getSubcellCount(d);
      this->_tagToOrdinal[d].resize(subcellCount);
    }
    this->_tagToOrdinal[domainDimension] = std::vector<std::vector<int> >(1);
    this->_tagToOrdinal[domainDimension][0] = std::vector<int>(this->_basisCardinality);

//    for (int d=0; d<this->_tagToOrdinal.size(); d++)
//    {
//      for (int subcOrd=0; subcOrd < this->_tagToOrdinal[d].size(); subcOrd++)
//      {
//        for (int subcDofOrd=0; subcDofOrd< this->_tagToOrdinal[d][subcOrd].size(); subcDofOrd++)
//        {
//          this->_tagToOrdinal[d][subcOrd][subcDofOrd] = -1;
//        }
//      }
//    }

    for (int dofOrdinal = 0; dofOrdinal < this->_basisCardinality; dofOrdinal++)
    {
      tag[0] = domainDimension; // dimension of the subcell
      tag[1] = 0; // subcell ordinal
      tag[2] = dofOrdinal; // ordinal of the dof relative to subcell
      tag[3] = this->_basisCardinality; // total # of dofs associated with subcell
      this->_ordinalToTag[dofOrdinal] = tag;
      this->_tagToOrdinal[domainDimension][0][dofOrdinal] = dofOrdinal;
    }
  }
}

template<class Scalar, class ArrayScalar>
Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> > Intrepid2BasisWrapper<Scalar,ArrayScalar>::intrepid2Basis() const
{
  return _intrepid2Basis;
}
  
  template<class Scalar, class ArrayScalar>
  void Intrepid2BasisWrapper<Scalar, ArrayScalar>::getDofCoords(ArrayScalar & dofCoords) const
  {
    Camellia::FieldContainer<Scalar> dofCoordsCamellia = dofCoords;
    
    auto dofCoordsView = dofCoordsCamellia.checkoutDeviceView(false); // false: don't deep copy from host
    
    // on CUDA, layouts may differ; we therefore allocate a second view and copy.  This is obviously inefficient; we should use Intrepid2-compatible views in all performance-critical code.
    using Intrepid2Values = Intrepid2::Basis<Camellia::DeviceType,double,double>::OutputViewType;
    Intrepid2Values intrepid2DofCoords = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(dofCoordsView);
    
    _intrepid2Basis->getDofCoords( intrepid2DofCoords );
    
    deepCopy(dofCoordsView,intrepid2DofCoords);
    
    // client code will depend on host availability of dof coords; let's copy from device if need be
    dofCoordsCamellia.checkinDeviceView(true); // true: copy from device to host
    
    dofCoords = dofCoordsCamellia;
  }

  template<class Scalar, class ArrayScalar>
  std::set<int> Intrepid2BasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForSide(int sideOrdinal) const
  {
    if (this->_continuousBasis == Teuchos::null)
    {
      return this->Basis<Scalar,ArrayScalar>::dofOrdinalsForSide(sideOrdinal);
    }
    else
    {
      return this->_continuousBasis->dofOrdinalsForSide(sideOrdinal);
    }
  }
  
template<class Scalar, class ArrayScalar>
void Intrepid2BasisWrapper<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const
{
  this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
  
  const bool convertible = ((operatorType >= OP_VALUE) && (operatorType <= OP_D10));
  TEUCHOS_TEST_FOR_EXCEPTION(!convertible, std::invalid_argument, "operator is not in the valid range for Intrepid2");
  
  const Intrepid2::EOperator opIntrepid2 = static_cast<Intrepid2::EOperator>(operatorType);
  
  // (This version does a bunch of extra copies to achieve compatibility with the original Intrepid basis interface.)
  
  Camellia::FieldContainer<Scalar> refPointsCamellia = refPoints;
  Camellia::FieldContainer<Scalar> valuesCamellia = values;
  
  auto refPointsView = refPointsCamellia.checkoutDeviceView(true); // true: do deep copy from host
  auto valuesView    = valuesCamellia.checkoutDeviceView(false);   // false: don't deep copy from host
    
  // on CUDA, layouts may differ; we therefore allocate a second view and copy.  This is obviously inefficient; we should use Intrepid2-compatible views in all performance-critical code.
  using Intrepid2Values = Intrepid2::Basis<Camellia::DeviceType,double,double>::OutputViewType;
  Intrepid2Values intrepid2Values = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(valuesView);
  Intrepid2Values intrepid2Points = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(refPointsView);
  deepCopy(intrepid2Points, refPointsView);

  _intrepid2Basis->getValues(intrepid2Values,intrepid2Points,opIntrepid2);

  deepCopy(valuesView,intrepid2Values);
  
  refPointsCamellia.checkinDeviceView(false); // false: don't deep copy from device to host
  valuesCamellia.checkinDeviceView(true);     // true: do deep copy from device to host
  
  // finally, copy back to the FieldContainer that was passed in:
  values = valuesCamellia;
}

template<class Scalar, class ArrayScalar>
Intrepid2::BasisValues<Scalar,Camellia::DeviceType>
Intrepid2BasisWrapper<Scalar,ArrayScalar>::allocateBasisValues( const Intrepid2::TensorPoints<Scalar,Camellia::DeviceType> &points,
                                                                const Intrepid2::EOperator op) const
{
  return _intrepid2Basis->allocateBasisValues(points,op);
}

template<class Scalar, class ArrayScalar>
void Intrepid2BasisWrapper<Scalar,ArrayScalar>::getValues(Intrepid2::BasisValues<Scalar,Camellia::DeviceType> &outputValues,
                                                          const Intrepid2::TensorPoints<Scalar,Camellia::DeviceType> &inputPoints,
                                                          const Intrepid2::EOperator op) const
{
  _intrepid2Basis->getValues(outputValues,inputPoints,op);
}

} // namespace Camellia
