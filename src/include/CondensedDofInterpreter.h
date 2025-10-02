// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  CondensedDofInterpreter.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//

#ifndef Camellia_debug_CondensedDofInterpreter_h
#define Camellia_debug_CondensedDofInterpreter_h

#include "BC.h"
#include "DofInterpreter.h"
#include "Camellia_FieldContainer.hpp"
#include "Epetra_Vector.h"
#include "Mesh.h"
#include "LagrangeConstraints.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialDenseVector.h"
#include "RHS.h"

namespace Camellia
{
/**
 Note on naming of variables:
 Essentially, the notion is that the condensed dof interpreter sits on a layer above the GDA object (owned by the Mesh),
 which sits atop individual elements.  When we use "interpreted" below we mean interpreted by the GDA.  When we say
 "global" we mean the level above the condensed dof interpreter, the dofs that actually enter the global solve.

 See Nate's dissertation, pp. 44-45 for a brief explanation of static condensation.
 **/

template <typename Scalar>
class CondensedDofInterpreter : public DofInterpreter
{
protected:
  map<GlobalIndexType, Camellia::FieldContainer<Scalar> > _localLoadVectors;       // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
private:
  bool _storeLocalStiffnessMatrices;
  bool _skipLocalFields; // when true, need not include local field degrees of freedom in interpretGlobalCoefficients()
  MeshPtr _mesh; // for element type lookup, and for determination of which dofs are trace dofs
  TIPPtr<Scalar> _ip;
  TRHSPtr<Scalar> _rhs;
  TBCPtr<Scalar> _bc;
  LagrangeConstraints* _lagrangeConstraints;
  std::set<int> _uncondensibleVarIDs;
  // ! Container stores any local indices for the cell corresponding to variables that would otherwise be condensed, but have some local exception.  (This is used for singleton BCs right now.)
  std::map<GlobalIndexType,vector<int>> _cellLocalUncondensibleDofIndices;
  
  GlobalIndexType _meshLastKnownGlobalDofCount;
  std::set<GlobalIndexType> _offRankCellsToInclude;
  
  map<pair<DofOrdering*,int>, vector<int>> _fieldRowIndices; // key is (trialOrdering, varID)
  
  map<GlobalIndexType, Camellia::FieldContainer<Scalar> > _localStiffnessMatrices; // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
  map<GlobalIndexType, Camellia::FieldContainer<GlobalIndexType> > _localInterpretedDofIndices;       // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
  map<GlobalIndexType, Teuchos::RCP<Epetra_SerialDenseMatrix> > _fluxToFieldMapForIterativeSolves;
  
  GlobalIndexType _myGlobalDofIndexOffset;
  GlobalIndexType _myGlobalDofIndexCount;
  
  vector<GlobalIndexType> _globalDofIndexOffsets;

  set<GlobalIndexType> _interpretedFluxDofIndices; // the "global" dof indices prior to condensation

//  map< GlobalIndexType, map< pair<int, int>, Camellia::FieldContainer<GlobalIndexType> > > _interpretedDofIndicesForBasis; // outer map: cellID is index.  Inner: (varID, sideOrdinal)

  map<GlobalIndexType, GlobalIndexType> _interpretedToGlobalDofIndexMap; // maps from the interpreted dof indices to the new ("outer") global dof indices (we only store the ones that are seen by the local MPI rank)

  void getSubmatrices(set<int> fieldIndices, set<int> fluxIndices,
                      const Camellia::FieldContainer<Scalar> &K, Epetra_SerialDenseMatrix &K_field,
                      Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux);

  void getSubvectors(set<int> fieldIndices, set<int> fluxIndices, const Camellia::FieldContainer<Scalar> &b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux);

  void initializeGlobalDofIndices();

  // ! For fluxes owned by the local partition.
  map<GlobalIndexType, GlobalIndexType> interpretedFluxMapLocal(const set<GlobalIndexType> &cellsForFluxInterpretation);

  void getLocalData(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &stiffness, Camellia::FieldContainer<Scalar> &load,
                    Camellia::FieldContainer<GlobalIndexType> &interpretedDofIndices);
  
  // new version:
  void getLocalData(GlobalIndexType cellID, Teuchos::RCP<Epetra_SerialDenseSolver> &fieldSolver,
                    Epetra_SerialSymDenseMatrix &FieldField, Epetra_SerialDenseMatrix &FluxField, Epetra_SerialDenseVector &b_field,
                    Camellia::FieldContainer<GlobalIndexType> &interpretedDofIndices, set<int> &fieldIndices, set<int> &fluxIndices);
  
  bool varDofsAreUsuallyCondensible(int varID, int sideOrdinal, DofOrderingPtr dofOrdering) const;
public:
  CondensedDofInterpreter(MeshPtr mesh, TIPPtr<Scalar> ip, TRHSPtr<Scalar> rhs, TBCPtr<Scalar> bc,
                          LagrangeConstraints* lagrangeConstraints, const set<int> &fieldIDsToExclude,
                          bool storeLocalStiffnessMatrices, std::set<GlobalIndexType> offRankCellsToInclude);

  void addSolution(CondensedDofInterpreter* otherSolnDofInterpreter, Scalar weight);
  
  // ! Storage cost in bytes.  (This neglects the STL map overhead.)
  long long approximateStiffnessAndLoadMemoryCost();
  
  void clearStiffnessAndLoad();
  
  void computeAndStoreLocalStiffnessAndLoad(GlobalIndexType cellID);

  // ! Returns a matrix with shape field x flux for specified cellID, which allows determination of field
  // ! coefficients from flux coefficients, under the assumption that the field portion of the RHS is 0
  // ! (a reasonable supposition when doing an iterative solve).
  Teuchos::RCP<Epetra_SerialDenseMatrix> fluxToFieldMapForIterativeSolves(GlobalIndexType cellID);
  
  // ! returns row indices as found in e.g. fluxToFieldMap for the specified field ID
  std::vector<int> fieldRowIndices(GlobalIndexType cellID, int condensibleVarID);
  
  // ! returns a map whose keys are flux indices -- e.g. the column indices of fluxToFieldMap -- and whose values are the corresponding dof indices in the (uncondensed) local cell
  std::vector<int> fluxIndexLookupLocalCell(GlobalIndexType cellID);
  
  GlobalIndexType globalDofCount();
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType rank);

  void importInterpretedMapForNeighborGlobalIndices(const map<PartitionIndexType,set<GlobalIndexType>> &partitionToMeshGlobalIndices);
  
  void importInterpretedMapForOffRankCells(const set<GlobalIndexType> &offRankCells);
  
  void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localData,
                          Camellia::FieldContainer<Scalar> &globalData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices);

  // note that interpretLocalData stores the stiffness and load.  If you call this with data different from the actual stiffness and load
  // (as Solution does when imposing zero-mean constraints), it can be important to restore the actual stiffness and load afterward.
  // (see storeLoadForCell, storeStiffnessForCell)
  void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localStiffnessData, const Camellia::FieldContainer<Scalar> &localLoadData,
                          Camellia::FieldContainer<Scalar> &globalStiffnessData, Camellia::FieldContainer<Scalar> &globalLoadData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices);

  using DofInterpreter::interpretLocalCoefficients;
  virtual void interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localCoefficients, Epetra_MultiVector &globalCoefficients, int columnOrdinal);

  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Camellia::FieldContainer<Scalar> &basisCoefficients,
                                       Camellia::FieldContainer<Scalar> &globalCoefficients, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices);

  void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &localDofs, const Epetra_MultiVector &globalDofs);

  bool isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const;
  
  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID);
  set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime);

  GlobalIndexType condensedGlobalIndex(GlobalIndexType meshGlobalIndex); // meshGlobalIndex aka interpretedGlobalIndex

  set<int> condensibleVariableIDs();
  
  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex );

  void reinitialize(); // clear stiffness matrices, etc., and rebuild global dof index map
  bool canSkipLocalFieldInInterpretGlobalCoefficients() const;
  void setCanSkipLocalFieldInInterpretGlobalCoefficients(bool value);
  
  void storeLoadForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &load);
  void storeStiffnessForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &stiffness);

  const Camellia::FieldContainer<Scalar> & storedLocalLoadForCell(GlobalIndexType cellID);
  const Camellia::FieldContainer<Scalar> & storedLocalStiffnessForCell(GlobalIndexType cellID);
  
  bool varDofsAreCondensible(GlobalIndexType cellID, int varID, int sideOrdinal, DofOrderingPtr dofOrdering);
};

extern template class CondensedDofInterpreter<double>;
}


#endif
