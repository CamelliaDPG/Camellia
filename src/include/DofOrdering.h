// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef DOF_ORDERING
#define DOF_ORDERING

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Camellia_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

#include "CellTopology.h"

#include "VarFactory.h"

namespace Camellia
{
  const unsigned VOLUME_INTERIOR_SIDE_ORDINAL = static_cast<unsigned>(-1);
  const int VOLUME_INTERIOR_SIDE_INDEX = -1; // "index": signed integer type; "ordinal": unsigned
class DofOrdering
{
  int _indexNeedsToBeRebuilt;
  int _nextIndex;
  
  int _volumeIndex; // where the volume indices are stored in _indices
  
  //  vector<int> varIDs;
  std::set<int> varIDs;
  //  std::vector<int> varIDsVector;
  std::map< std::pair<int, std::pair<int, int> >, std::pair<int, int> > dofIdentifications; // keys: <varID, <sideIndex, dofOrdinal> >
  // values: <sideIndex, dofOrdinal>
  std::map< int, std::vector<int> > _sidesForVarID; // to replace numSidesForVarID
  std::map<int,int> numSidesForVarID;
//  std::map< std::pair<int,int>, std::vector<int> > indices; // keys for indices are <varID, sideIndex >, where sideIndex = VOLUME_INTERIOR_SIDE_INDEX for field (volume) variables
  std::vector<map<int,vector<int>>> _indices; // outer index is sideOrdinal (or _volumeIndex); next is varID.
  // values for indices: list of the indices used in the DofOrdering by this <varID, sideIndex> pair's basis, ordered according to that basis's ordering
  std::map< std::pair<int,int>, BasisPtr > bases; // keys are <varID, sideIndex>
  std::map< int, int > basisRanks; // keys are varIDs; values are 0,1,2,... (scalar, vector, tensor)

  std::map< int, CellTopoPtr > _cellTopologyForSide; // -1 is field variable
  
  // optional registration of VarFactory and whether this is a trial ordering or a test ordering:
  // (this is only used for debugging output at the moment)
  VarFactoryPtr _varFactory;
  bool _isTrialOrdering = false;
public:
  DofOrdering(CellTopoPtr cellTopo = Teuchos::null); // constructor

  void addEntry(int varID, BasisPtr basis, int basisRank, int sideIndex = VOLUME_INTERIOR_SIDE_INDEX);

  bool hasBasisEntry(int varID, int sideIndex) const;
  bool hasSideVarIDs();

  void copyLikeCoefficients( Camellia::FieldContainer<double> &newValues, Teuchos::RCP<DofOrdering> oldDofOrdering,
                             const Camellia::FieldContainer<double> &oldValues );

  // get the varIndex variable's dof with basis ordinal dofId in the Dof ordering:
  int getDofIndex(int varID, int basisDofOrdinal, int sideIndex=VOLUME_INTERIOR_SIDE_INDEX, int subSideIndex = -1);

  const std::vector<int> & getDofIndices(int varID, int sideIndex=VOLUME_INTERIOR_SIDE_INDEX) const;

  const std::set<int> & getVarIDs() const;

  std::set<int> getTraceDofIndices(); // returns dof indices corresponding to variables with numSides > 1.

  int getBasisCardinality(int varID, int sideIndex);

  BasisPtr getBasis(int varID, int sideIndex = VOLUME_INTERIOR_SIDE_INDEX) const;

  int getBasisRank(int varID)
  {
    return basisRanks[varID];
  }

  // ! returns the dof coordinates (nodes in case of a nodal basis) for the variable on the side indicated.
  // ! The order of the coordinates corresponds to the basis ordering.  The coordinates are in volume reference space.
  // ! dofCoords shape: (F,D), where D is the dimension of the volume topology.
  void getDofCoords(Camellia::FieldContainer<double> &dofCoords, int varID, int sideOrdinal = VOLUME_INTERIOR_SIDE_INDEX);

  // ! returns the dof coordinates (nodes in case of a nodal basis) for the variable on the side indicated, given a physical cell
  // ! with nodes as indicated.
  // ! The order of the coordinates corresponds to the basis ordering.  The coordinates are in physical space.
  // ! dofCoords shape: (C,F,D), where D is the dimension of the volume topology.
  void getDofCoords(const Camellia::FieldContainer<double> &physicalCellNodes, Camellia::FieldContainer<double> &dofCoords,
                    int varID, int sideOrdinal = VOLUME_INTERIOR_SIDE_INDEX);
  
  bool hasEntryForVarID( int varID ); // returns true if we have any basis on any side for varID

  int getNumSidesForVarID(int varID); // will be deprecated soon.  Use getSidesForVarID instead

  const vector<int> & getSidesForVarID(int varID) const;

  int getTotalBasisCardinality(); // sum of all the *distinct* bases' cardinalities

  void addIdentification(int varID, int side1, int basisDofOrdinal1,
                         int side2, int basisDofOrdinal2);

  CellTopoPtr cellTopology(int sideIndex = -1) const;

  int minimumSubcellDimensionForContinuity() const; // across all bases
  int maxBasisDegree();
  int maxBasisDegreeForVolume();

  void print(std::ostream& os) const;
  
  // ! prints the coordinates of varID's basis/bases, in reference space
  void printDofCoords(int varID, std::ostream& os) const;
  
  int totalDofs() const
  {
    return _nextIndex;
  }

  void rebuildIndex();
  
  // let the DofOrdering know the associated VarFactory
  void setVarFactory(VarFactoryPtr varFactory, bool isTrialOrdering);
  
  VarFactoryPtr getVarFactory() const;
  
  bool isTrialOrdering() const;
  
  // ! Returns vector containing (varID,vector<sideOrdinal>) entries corresponding to variables with nonzero coefficients in the provided container
  vector<pair<int,vector<int>>> variablesWithNonZeroEntries(const Camellia::FieldContainer<double> &localCoefficients, double tol = 0.0) const;
};
}

inline bool operator==(const Camellia::DofOrdering& lhs, const Camellia::DofOrdering& rhs)
{
  if (lhs.cellTopology()->getKey() != rhs.cellTopology()->getKey()) return false;
  if (lhs.getVarIDs() != rhs.getVarIDs()) return false;
  for (int varID : lhs.getVarIDs())
  {
    if (lhs.getSidesForVarID(varID) != rhs.getSidesForVarID(varID)) return false;
    for (int side : lhs.getSidesForVarID(varID))
    {
      if (lhs.getDofIndices(varID,side) != rhs.getDofIndices(varID,side)) return false;
    }
  }
  return true;
}

inline bool operator!=(const Camellia::DofOrdering& lhs, const Camellia::DofOrdering& rhs){ return !(lhs == rhs); }

std::ostream& operator << (std::ostream& os, const Camellia::DofOrdering& dofOrdering);

#endif
