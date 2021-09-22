// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef DPG_MULTI_BASIS
#define DPG_MULTI_BASIS

#include "Basis.h"
#include "Camellia_FieldContainer.hpp"

#include "CellTopology.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class MultiBasis;
template<class Scalar, class ArrayScalar> class MultiBasis : public Basis<Scalar,ArrayScalar>
{
  using CamelliaArrayScalar = Camellia::FieldContainer<double>;
  
  CellTopoPtr _cellTopo;
  Camellia::FieldContainer<double> _subRefNodes;
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

  using Basis<Scalar,ArrayScalar>::getValues;
  void getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                 Camellia::EOperator operatorType) const;

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

  void getCubature(CamelliaArrayScalar &cubaturePoints, CamelliaArrayScalar &cubatureWeights, int maxTestDegree) const;

  void printInfo() const;
};

typedef Teuchos::RCP< MultiBasis<> > MultiBasisPtr;

} // namespace Camellia
#include "MultiBasisDef.h"

#endif
