//
//  GDAMinimumRule.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMinimumRule.h"

#include "MPIWrapper.h"

#include "CamelliaCellTools.h"

#include "Teuchos_GlobalMPISession.hpp"

GDAMinimumRule::GDAMinimumRule(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
: GlobalDofAssignment(meshTopology,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement)
{
  
}


void GDAMinimumRule::didChangePartitionPolicy() {
  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHRefine(parentCellIDs);
  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
    GlobalIndexType parentCellID = *cellIDIt;
    CellPtr parentCell = _meshTopology->getCell(parentCellID);
    vector<IndexType> childIDs = parentCell->getChildIndices();
    int parentH1Order = _cellH1Orders[parentCellID];
    for (vector<IndexType>::iterator childIDIt = childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
      _cellH1Orders[*childIDIt] = parentH1Order;
      assignInitialElementType(*childIDIt);
    }
  }
  rebuildLookups();
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) {
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    assignInitialElementType(*cellIDIt);
  }
  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHUnrefine(parentCellIDs);
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::didHUnrefine() unimplemented.\n";
  rebuildLookups();
}

ElementTypePtr GDAMinimumRule::elementType(GlobalIndexType cellID) {
  return Teuchos::rcp( (ElementType*) NULL);
//  return _elementTypeForCell[cellID];
}

GlobalIndexType GDAMinimumRule::globalDofCount() {
  // TODO: implement this
  cout << "WARNING: globalDofCount() unimplemented.\n";
  return 0;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  // TODO: implement this
  set<GlobalIndexType> globalDofIndices;
  cout << "WARNING: GDAMinimumRule::globalDofIndicesForPartition() unimplemented.\n";
  return globalDofIndices;
}

int GDAMinimumRule::H1Order(GlobalIndexType cellID, unsigned sideOrdinal) {
  // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
  return _cellH1Orders[cellID];
}

void GDAMinimumRule::interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs) {
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::interpretGlobalData() unimplemented.\n";
}

void GDAMinimumRule::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                                        FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) {
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::interpretLocalData() unimplemented.\n";
}

IndexType GDAMinimumRule::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

CellConstraints GDAMinimumRule::getCellConstraints(GlobalIndexType cellID) {
  CellConstraints cellConstraints;
  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = cell->topology();
  unsigned sideCount = topo->getSideCount();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;
  unsigned subsideDim = sideDim - 1;
  vector<IndexType> sideEntityIndices = cell->getEntityIndices(sideDim);
  
  cellConstraints.sideConstraints = vector< ConstrainingCellInfo >(sideCount);
  cellConstraints.subsideConstraints = vector< vector<ConstrainingSubsideInfo> >(sideCount);
  ConstrainingCellInfo sideConstraint;
  ConstrainingSubsideInfo subsideConstraint;
  
  typedef pair< IndexType, unsigned > CellPair;
  
  // determine subcell ownership from the perspective of the cell (for now, we redundantly store it from the perspective of side as well, below)
  cellConstraints.owningCellIDForSubcell = vector< vector<GlobalIndexType> >(sideDim+1);
  for (int d=0; d<spaceDim; d++) {
    vector<IndexType> scIndices = cell->getEntityIndices(d);
    unsigned scCount = scIndices.size();
    cellConstraints.owningCellIDForSubcell[d] = vector<GlobalIndexType>(scCount);
    for (int scOrdinal = 0; scOrdinal < scCount; scOrdinal++) {
      IndexType entityIndex = scIndices[scOrdinal];
      IndexType constrainingEntityIndex = _meshTopology->getConstrainingEntityIndex(d, entityIndex);
      GlobalIndexType owningCellID = _meshTopology->leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(d, constrainingEntityIndex);
      cellConstraints.owningCellIDForSubcell[d][scOrdinal] = owningCellID;
    }
  }
  
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    IndexType sideEntityIndex = sideEntityIndices[sideOrdinal];
    
    // determine subcell ownership from the perspective of the side (for now, we redundantly store it from the perspective of cell as well, above)
    sideConstraint.owningCellIDForSideSubcell = vector< vector<GlobalIndexType> >(subsideDim + 1);
    for (int d=0; d<sideDim; d++) {
      unsigned scCount = _meshTopology->getSubEntityCount(sideDim, sideEntityIndex, d);
      cellConstraints.owningCellIDForSubcell[d] = vector<GlobalIndexType>(scCount);
      for (int scOrdinal = 0; scOrdinal < scCount; scOrdinal++) {
        IndexType entityIndex = _meshTopology->getSubEntityIndex(sideDim, sideEntityIndex, d, scOrdinal);
        IndexType constrainingEntityIndex = _meshTopology->getConstrainingEntityIndex(d, entityIndex);
        GlobalIndexType owningCellID = _meshTopology->leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(d, constrainingEntityIndex);
        cellConstraints.owningCellIDForSubcell[d][scOrdinal] = owningCellID;
      }
    }
    
    // first, establish side constraint
    IndexType constrainingSideEntityIndex = _meshTopology->getConstrainingEntityIndex(sideDim, sideEntityIndex);
    if (sideEntityIndex == constrainingSideEntityIndex) {
      if (_meshTopology->getCellCountForSide(sideEntityIndex) == 1) { // then we know that this cell is the constraining one on the side
        sideConstraint.cellID = cellID;
        sideConstraint.sideOrdinal = sideOrdinal;
      } else { // there are two cells; we need to figure out which has the lower H1 order
        CellPair cell1 = _meshTopology->getFirstCellForSide(sideEntityIndex);
        CellPair cell2 = _meshTopology->getSecondCellForSide(sideEntityIndex);
        unsigned cell1Order = _cellH1Orders[cell1.first];
        unsigned cell2Order = _cellH1Orders[cell1.first];
        
        int whichCell;
        if (cell1Order < cell2Order) {
          whichCell = 1;
        } else if (cell2Order < cell1Order) {
          whichCell = 2;
        } else { // tie -- broken by cellID (lower one is the constraint)
          if (cell1.first < cell2.first) {
            whichCell = 1;
          } else {
            whichCell = 2;
          }
        }
        if (whichCell==1) {
          sideConstraint.cellID = cell1.first;
          sideConstraint.sideOrdinal = cell1.second;
        } else if (whichCell==2) {
          sideConstraint.cellID = cell2.first;
          sideConstraint.sideOrdinal = cell2.second;
        }
      }
    } else {
      // there should be exactly one cell with the specified sideEntityIndex listed in _meshTopology::_cellsForSideEntities.
      if (_meshTopology->getCellCountForSide(constrainingSideEntityIndex) != 1) {
        cout << "ERROR while applying GDAMinimumRule: extra cell(s) defined for constrainingSideEntityIndex.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: extra cell(s) defined for constrainingSideEntityIndex.");
      }
      CellPair constrainingCellPair = _meshTopology->getFirstCellForSide(constrainingSideEntityIndex);
      sideConstraint.cellID = constrainingCellPair.first;
      sideConstraint.sideOrdinal = constrainingCellPair.second;
    }
    cellConstraints.sideConstraints[sideOrdinal] = sideConstraint;
    
    // next, establish subside constraints
    shards::CellTopology sideTopo = topo->getCellTopologyData(sideDim, sideOrdinal);
    unsigned subsideCount = sideTopo.getSideCount();
    cellConstraints.subsideConstraints[sideOrdinal] = vector<ConstrainingSubsideInfo>(subsideCount);
    for (unsigned subsideOrdinal = 0; subsideOrdinal < subsideCount; subsideOrdinal++) {
      // determine the subside's entity index
      unsigned subsideEntityIndex = _meshTopology->getSubEntityIndex(sideDim, sideEntityIndex, subsideDim, subsideOrdinal);
      // determine the constraining entity index
      unsigned constrainingEntityIndex = _meshTopology->getConstrainingEntityIndex(subsideDim, subsideEntityIndex);
      // determine all sides that contain the constraining entity -- find the cell containing such a side with the lowest H^1 order (using cellID to break ties)
      set< CellPair > cellsForSubside = _meshTopology->getCellsContainingEntity(subsideDim, constrainingEntityIndex);
      unsigned leastH1Order = (unsigned)-1;
      set< CellPair > cellsWithLeastH1Order;
      for (set< CellPair >::iterator subsideCellIt = cellsForSubside.begin(); subsideCellIt != cellsForSubside.end(); subsideCellIt++) {
        IndexType subsideCellID = subsideCellIt->first;
        if (_cellH1Orders[subsideCellID] == leastH1Order) {
          cellsWithLeastH1Order.insert(*subsideCellIt);
        } else if (_cellH1Orders[subsideCellID] < leastH1Order) {
          cellsWithLeastH1Order.clear();
          leastH1Order = _cellH1Orders[subsideCellID];
          cellsWithLeastH1Order.insert(*subsideCellIt);
        }
      }
      if (cellsWithLeastH1Order.size() == 0) {
        cout << "ERROR: No cells found for constraining subside entity.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No cells found for constraining subside entity.");
      }
      CellPair constrainingCellPair = *cellsWithLeastH1Order.begin(); // first one will have the earliest cell ID, given the sorting of set/pair.
      subsideConstraint.cellID = constrainingCellPair.first;
      subsideConstraint.sideOrdinal = constrainingCellPair.second;
      
      // now, we need to find the subcellOrdinal within cell for the subside:
      CellPtr constrainingCell = _meshTopology->getCell(subsideConstraint.cellID);
      
      shards::CellTopology constrainingSide = constrainingCell->topology()->getCellTopologyData(subsideConstraint.sideOrdinal, sideDim);
      int subsideCountInConstrainingSide = constrainingSide.getSubcellCount(subsideDim);
      constrainingSideEntityIndex = constrainingCell->entityIndex(sideDim, subsideConstraint.sideOrdinal);
      subsideConstraint.subsideOrdinalInSide = -1;
      for (int subsideOrdinalInConstrainingSide = 0; subsideOrdinalInConstrainingSide < subsideCountInConstrainingSide; subsideOrdinalInConstrainingSide++) {
        IndexType entityIndex =  _meshTopology->getSubEntityIndex(sideDim, constrainingSideEntityIndex, subsideDim, subsideOrdinalInConstrainingSide);
        if (entityIndex == constrainingEntityIndex) {
          subsideConstraint.subsideOrdinalInSide = subsideOrdinalInConstrainingSide;
        }
      }
      if (subsideConstraint.subsideOrdinalInSide == -1) {
        cout << "ERROR: Could not determine subside ordinal in side for subside constraint.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not determine subside ordinal in side for subside constraint.");
      }
      cellConstraints.subsideConstraints[sideOrdinal][subsideOrdinal] = subsideConstraint;
    }
  }
  
  return cellConstraints;
}

void GDAMinimumRule::rebuildLookups() {
  determineActiveElements(); // call to super: constructs cell partitionings
  
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> myCellIDs = _partitions[rank];
  
  map<int, VarPtr> trialVars = _varFactory.trialVars();
  
  GlobalIndexType partitionDofCount = 0; // how many dofs we own locally
  _cellDofOffsets.clear(); // within the partition, offsets for the owned dofs in cell
  
  int spaceDim = _meshTopology->getSpaceDim();
  int sideDim = spaceDim - 1;
  
  // pieces of this remain fairly ugly--the brute force searches are limited to entities on a cell (i.e. < O(12) items to search),
  // and I've done a reasonable job only doing them when we need the result, but they still are brute force searches.  By tweaking
  // the design of MeshTopology and Cell to take better advantage of regularities (or just to store better lookups), we should be able to do better.
  // But in the interest of avoiding wasting development time on premature optimization, I'm
  for (vector<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = cell->topology();
    int sideCount = topo->getSideCount();
    CellConstraints constraints = getCellConstraints(cellID);
    DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
    for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++) {
      VarPtr var = varIt->second;
      bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
      if ( varHasSupportOnVolume ) { // i.e. a variable with support on the volume
        BasisPtr basis = trialOrdering->getBasis(var->ID());
        if (var->space() == L2) { // unconstrained / local
          partitionDofCount += basis->getCardinality();
        } else {
          partitionDofCount += basis->dofOrdinalsForInterior().size();
        }
      }

      if ( ! (varHasSupportOnVolume && (var->space() == L2)) ) { // if space is L^2 on the volume, we'll have claimed all the dofs above, and we can skip further processing
        vector< set<unsigned> > processedSubcells(sideDim);
        for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
          shards::CellTopology sideTopo = topo->getCellTopologyData(sideDim, sideOrdinal);
          GlobalIndexType owningCellIDForSide = constraints.owningCellIDForSubcell[sideDim][sideOrdinal];
          if (owningCellIDForSide == cellID) {
            ConstrainingCellInfo sideConstraint = constraints.sideConstraints[sideOrdinal];
            DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[sideConstraint.cellID]->trialOrderPtr;
            if (varHasSupportOnVolume) {
              BasisPtr basis = constrainingTrialOrdering->getBasis(var->ID());
              set<int> sideBasisOrdinals = basis->dofOrdinalsForSubcell(sideDim, sideConstraint.sideOrdinal, sideDim);
              partitionDofCount += sideBasisOrdinals.size();
            } else {
              BasisPtr basis = constrainingTrialOrdering->getBasis(var->ID(),sideConstraint.sideOrdinal);
              set<int> sideBasisOrdinals = basis->dofOrdinalsForInterior();
              partitionDofCount += sideBasisOrdinals.size();
            }
          }
          
          int subsideCount = sideTopo.getSideCount();
          vector< map< unsigned, unsigned > > subsideMap(subsideCount); // outer vector indexed by dimension.  map goes from scOrdinalInSide to a subside containing that subcell.  (This is not uniquely defined, but that should be OK.)
          for (int d=0; d<sideDim-1; d++) {
            for (int ssOrdinal=0; ssOrdinal<subsideCount; ssOrdinal++) {
            shards::CellTopology subside = sideTopo.getCellTopologyData(sideDim-1, ssOrdinal);
              unsigned scCount = subside.getSubcellCount(d);
              for (int scOrdinalInSubside=0; scOrdinalInSubside<scCount; scOrdinalInSubside++) {
                unsigned scOrdinalInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, sideDim-1, ssOrdinal, d, scOrdinalInSubside);
                subsideMap[d][scOrdinalInSide] = ssOrdinal;
              }
            }
          }
          
          for (int d=0; d<sideDim; d++) {
            unsigned scCount = sideTopo.getSubcellCount(d);
            for (unsigned scOrdinalInSide = 0; scOrdinalInSide < scCount; scOrdinalInSide++) {
              unsigned scOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(*topo, sideDim, sideOrdinal, d, scOrdinalInSide);
              if (processedSubcells[d].find(scOrdinalInCell) == processedSubcells[d].end()) { // haven't processed this one yet
                GlobalIndexType owningCellID = constraints.owningCellIDForSubcell[d][scOrdinalInCell];
                
                // determine the subcell index in _meshTopology (will be used below to determine the subcell ordinal of the constraining subcell in the constraining cell)
                IndexType scIndex = cell->entityIndex(d, scOrdinalInCell);
                IndexType constrainingScIndex = _meshTopology->getConstrainingEntityIndex(d, scIndex);
                
                if (owningCellID == cellID) {
                  if (varHasSupportOnVolume) {
                    ConstrainingCellInfo sideConstraint = constraints.sideConstraints[sideOrdinal];
                    DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[sideConstraint.cellID]->trialOrderPtr;
                    BasisPtr basis = constrainingTrialOrdering->getBasis(var->ID());
                    CellPtr constrainingCell = _meshTopology->getCell(sideConstraint.cellID);
                    
                    unsigned scOrdinalInConstrainingCell = constrainingCell->findSubcellOrdinal(d,constrainingScIndex);
                    set<int> scBasisOrdinals = basis->dofOrdinalsForSubcell(d, scOrdinalInConstrainingCell, d);
                    partitionDofCount += scBasisOrdinals.size();
                  } else {
                    // here, we are dealing with a subside or one of its constituents.  We want to use the constraint info for that subside.
                    unsigned subsideOrdinal = subsideMap[d][scOrdinalInSide];
                    ConstrainingSubsideInfo subsideConstraint = constraints.subsideConstraints[sideOrdinal][subsideOrdinal];
                    DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[subsideConstraint.cellID]->trialOrderPtr;
                    BasisPtr basis = constrainingTrialOrdering->getBasis(var->ID(),subsideConstraint.sideOrdinal);
                    
                    CellPtr constrainingCell = _meshTopology->getCell(subsideConstraint.cellID);
                    unsigned scOrdinalInConstrainingCell = constrainingCell->findSubcellOrdinal(d,constrainingScIndex);
                    unsigned scOrdinalInConstrainingSide = CamelliaCellTools::subcellReverseOrdinalMap(*constrainingCell->topology(),
                                                                                                       sideDim, subsideConstraint.sideOrdinal,
                                                                                                       d, scOrdinalInConstrainingCell);

                    set<int> scBasisOrdinals = basis->dofOrdinalsForSubcell(d, scOrdinalInConstrainingSide, d);
                    partitionDofCount += scBasisOrdinals.size();
                  }
                }
                
                processedSubcells[d].insert(scOrdinalInCell);
              }
            }
          }
        }
      }
    }
    _cellDofOffsets[cellID] = partitionDofCount;
  }
  int numRanks = Teuchos::GlobalMPISession::getNProc();
  FieldContainer<int> partitionDofCounts(numRanks);
  partitionDofCounts[rank] = partitionDofCount;
  MPIWrapper::entryWiseSum(partitionDofCounts);
  _partitionDofOffset = 0; // add this to a local partition dof index to get the global dof index
  for (int i=0; i<rank; i++) {
    _partitionDofOffset += partitionDofCounts[i];
  }
  _globalDofCount = _partitionDofOffset;
  for (int i=rank; i<numRanks; i++) {
    _globalDofCount += partitionDofCounts[i];
  }
}