// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef DPG_PATCH_BASIS
#define DPG_PATCH_BASIS

#include "Camellia_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=::Intrepid::FieldContainer<double> > class PatchBasis;
template<class Scalar, class ArrayScalar> class PatchBasis : public Camellia::Basis<Scalar,ArrayScalar>
{
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
                 Camellia::EOperator operatorType) const;

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
