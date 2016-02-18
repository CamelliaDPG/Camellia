//
//  GDAMinimumRule.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMinimumRule.h"

#include "BasisFactory.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "MPIWrapper.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace std;
using namespace Camellia;

GDAMinimumRule::GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
  : GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, vector<int>(1,initialH1OrderTrial), testOrderEnhancement, false)
{

}

GDAMinimumRule::GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               vector<int> initialH1OrderTrial, unsigned testOrderEnhancement)
  : GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement, false)
{

}

vector<unsigned> GDAMinimumRule::allBasisDofOrdinalsVector(int basisCardinality)
{
  vector<unsigned> ordinals(basisCardinality);
  for (int i=0; i<basisCardinality; i++)
  {
    ordinals[i] = i;
  }
  return ordinals;
}

GlobalDofAssignmentPtr GDAMinimumRule::deepCopy()
{
  return Teuchos::rcp(new GDAMinimumRule(*this) );
}

void GDAMinimumRule::didChangePartitionPolicy()
{
  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs)
{
  set<GlobalIndexType> neighborsOfNewElements;
  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType parentCellID = *cellIDIt;
//    cout << "GDAMinimumRule: h-refining " << parentCellID << endl;
    CellPtr parentCell = _meshTopology->getCell(parentCellID);
    vector<IndexType> childIDs = parentCell->getChildIndices(_meshTopology);
    vector<int> parentH1Order = _cellH1Orders[parentCellID];
    for (vector<IndexType>::iterator childIDIt = childIDs.begin(); childIDIt != childIDs.end(); childIDIt++)
    {
      GlobalIndexType childCellID = *childIDIt;
      _cellH1Orders[childCellID] = parentH1Order;
      assignInitialElementType(childCellID);
      assignParities(childCellID);
      // determine neighbors, so their parities can be updated below:
      CellPtr childCell = _meshTopology->getCell(childCellID);

      unsigned childSideCount = childCell->getSideCount();
      for (int childSideOrdinal=0; childSideOrdinal<childSideCount; childSideOrdinal++)
      {
        GlobalIndexType neighborCellID = childCell->getNeighborInfo(childSideOrdinal, _meshTopology).first;
        if (neighborCellID != -1)
        {
          neighborsOfNewElements.insert(neighborCellID);
        }
      }
    }
  }
  // this set is not as lean as it might be -- we could restrict to peer neighbors, I think -- but it's a pretty cheap operation.
  for (set<GlobalIndexType>::iterator cellIDIt = neighborsOfNewElements.begin(); cellIDIt != neighborsOfNewElements.end(); cellIDIt++)
  {
    assignParities(*cellIDIt);
  }
//  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
//    GlobalIndexType parentCellID = *cellIDIt;
//    ElementTypePtr elemType = elementType(parentCellID);
//    for (vector< TSolution<double>* >::iterator solutionIt = _registeredSolutions.begin();
//         solutionIt != _registeredSolutions.end(); solutionIt++) {
//      // do projection
//      vector<IndexType> childIDsLocalIndexType = _meshTopology->getCell(parentCellID)->getChildIndices();
//      vector<GlobalIndexType> childIDs(childIDsLocalIndexType.begin(),childIDsLocalIndexType.end());
//      (*solutionIt)->projectOldCellOntoNewCells(parentCellID,elemType,childIDs);
//    }
//  }

  this->GlobalDofAssignment::didHRefine(parentCellIDs);
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)
{
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);

  // the above assigns _cellH1Orders for active elements; now we take minimums for parents (inactive elements)
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    CellPtr cell = _meshTopology->getCell(*cellIDIt);
    CellPtr parent = cell->getParent();
    while (parent.get() != NULL)
    {
      vector<IndexType> childIndices = parent->getChildIndices(_meshTopology);
      vector<int> minH1Order = _cellH1Orders[*cellIDIt];
      for (int childOrdinal=0; childOrdinal<childIndices.size(); childOrdinal++)
      {
        if (_cellH1Orders.find(childIndices[childOrdinal])==_cellH1Orders.end())
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sibling H1 order not found");
        }
        else
        {
          // take minimum component-wise
          for (int pComponent=0; pComponent < minH1Order.size(); pComponent++)
          {
            minH1Order[pComponent] = min(minH1Order[pComponent],_cellH1Orders[childIndices[childOrdinal]][pComponent]);
          }
        }
      }
      _cellH1Orders[parent->cellIndex()] = minH1Order;
      parent = parent->getParent();
    }
  }

  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    ElementTypePtr oldType = elementType(*cellIDIt);
    assignInitialElementType(*cellIDIt);
    for (typename vector< TSolutionPtr<double> >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++)
    {
      // do projection
      vector<IndexType> childIDs(1,*cellIDIt); // "child" ID is just the cellID
      (*solutionIt)->projectOldCellOntoNewCells(*cellIDIt,oldType,childIDs);
    }
  }
//  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<GlobalIndexType> &parentCellIDs)
{
  this->GlobalDofAssignment::didHUnrefine(parentCellIDs);
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::didHUnrefine() unimplemented.\n";
  // will need to treat cell side parities here--probably suffices to redo those in parentCellIDs plus all their neighbors.
//  rebuildLookups();
}

ElementTypePtr GDAMinimumRule::elementType(GlobalIndexType cellID)
{
  return _elementTypeForCell[cellID];
}

GlobalIndexType GDAMinimumRule::globalDofCount()
{
  // assumes the lookups have been rebuilt since the last change that would affect the count

  // TODO: Consider working out a way to guard against a stale value here.  E.g. could have a "dirty" flag that gets set anytime there's a change to the refinements, and cleared when lookups are rebuilt.  If we're dirty when we get here, we rebuild before returning the global dof count.
  return _globalDofCount;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForCell(GlobalIndexType cellID)
{
  set<GlobalIndexType> globalDofIndices;

  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();

  globalDofIndices.insert(globalIndexVector.begin(),globalIndexVector.end());
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned int dim, unsigned int subcellOrdinal)
{
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  set<GlobalIndexType> globalDofIndices = dofMapper->globalIndicesForSubcell(varID, dim, subcellOrdinal);
  
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForPartition(PartitionIndexType partitionNumber)
{
  int rank = _partitionPolicy->Comm()->MyPID();

  if (partitionNumber==-1) partitionNumber = rank;

  // Added this exception because I suspected that in our usage, partitionNumber is always -1 and therefore we can/should
  // dispense with the argument...  (The recently added case which violates this is in CondensedDofInterpreter)
//  TEUCHOS_TEST_FOR_EXCEPTION(partitionNumber != rank, std::invalid_argument, "partitionNumber must be -1 or the current MPI rank");
  set<GlobalIndexType> globalDofIndices;
  if (partitionNumber == rank)
  {
    // by construction, our globalDofIndices are contiguously numbered, starting with _partitionDofOffset
    for (GlobalIndexType i=0; i<_partitionDofCount; i++)
    {
      globalDofIndices.insert(_partitionDofOffset + i);
    }
  }
  else
  {
    GlobalIndexType partitionDofOffset = 0;
    for (int i=0; i<partitionNumber; i++)
    {
      partitionDofOffset += _partitionDofCounts[i];
    }
    IndexType partitionDofCount = _partitionDofCounts[partitionNumber];
    for (IndexType i=0; i<partitionDofCount; i++)
    {
      globalDofIndices.insert(partitionDofOffset + i);
    }
  }

  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::ownedGlobalDofIndicesForCell(GlobalIndexType cellID)
{
  set<GlobalIndexType> globalDofIndices;

  CellConstraints constraints = getCellConstraints(cellID);
  SubCellDofIndexInfo owningCellDofIndexInfo = getOwnedGlobalDofIndices(cellID, constraints);

  CellTopoPtr cellTopo = _meshTopology->getCell(cellID)->topology();

  int dim = cellTopo->getDimension();
  for (int d=0; d<=dim; d++)
  {
    int scCount = cellTopo->getSubcellCount(d);
    for (int scord = 0; scord < scCount; scord++)
    {
      map<int, vector<GlobalIndexType> > globalDofOrdinalsMapForSubcell = owningCellDofIndexInfo[d][scord]; // keys are varIDs
      for (map<int, vector<GlobalIndexType> >::iterator entryIt = globalDofOrdinalsMapForSubcell.begin();
           entryIt != globalDofOrdinalsMapForSubcell.end(); entryIt++)
      {
        globalDofIndices.insert(entryIt->second.begin(), entryIt->second.end());
      }
    }
  }

  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalFieldIndices()
{
  // compute complement of fluxes and traces
  set<GlobalIndexType> fieldDofIndices;
  // by construction, our globalDofIndices are contiguously numbered, starting with _partitionDofOffset
  for (GlobalIndexType i=0; i<_partitionDofCount; i++)
  {
    if ((_partitionTraceIndexOffsets.find(i) == _partitionTraceIndexOffsets.end()) &&
        (_partitionFluxIndexOffsets.find(i) == _partitionFluxIndexOffsets.end()) )
    {
      fieldDofIndices.insert(_partitionDofOffset + i);
    }
  }
  return fieldDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedIndicesForVariables(set<int> varIDs)
{
  set<GlobalIndexType> varIndices;
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
  {
    for (set<IndexType>::iterator varOffsetIt=_partitionIndexOffsetsForVarID[*varIt].begin();
         varOffsetIt != _partitionIndexOffsetsForVarID[*varIt].end(); varOffsetIt++)
    {
      varIndices.insert(*varOffsetIt + _partitionDofOffset);
    }
  }
  return varIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalFluxIndices()
{
  set<GlobalIndexType> fluxIndices;
  for (set<IndexType>::iterator fluxOffsetIt=_partitionFluxIndexOffsets.begin(); fluxOffsetIt != _partitionFluxIndexOffsets.end(); fluxOffsetIt++)
  {
    fluxIndices.insert(*fluxOffsetIt + _partitionDofOffset);
  }
  return fluxIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalTraceIndices()
{
  set<GlobalIndexType> traceIndices;
  for (set<IndexType>::iterator traceOffsetIt=_partitionTraceIndexOffsets.begin(); traceOffsetIt != _partitionTraceIndexOffsets.end(); traceOffsetIt++)
  {
    traceIndices.insert(*traceOffsetIt + _partitionDofOffset);
  }
  return traceIndices;
}

vector<int> GDAMinimumRule::H1Order(GlobalIndexType cellID, unsigned sideOrdinal)
{
  // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
  return _cellH1Orders[cellID];
}

void GDAMinimumRule::interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localCoefficients,
                                                 const Epetra_MultiVector &globalCoefficients)
{
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();

  // DEBUGGING
//  if (cellID==0) {
//    cout << "interpretGlobalData, mapping report for cell " << cellID << ":\n";
//    dofMapper->printMappingReport();
//  }

  if (globalCoefficients.NumVectors() == 1)
  {
    bool useFieldContainer = false;
    
    if (useFieldContainer)
    {
      Intrepid::FieldContainer<double> globalCoefficientsFC(globalIndexVector.size());
      Epetra_BlockMap partMap = globalCoefficients.Map();
      for (int i=0; i<globalIndexVector.size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          globalCoefficientsFC[i] = globalCoefficients[0][localIndex];
        }
        else
        {
          // non-local coefficient: ignore
          globalCoefficientsFC[i] = 0;
        }
      }
      localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
    }
    else
    {
      Epetra_BlockMap partMap = globalCoefficients.Map();
      map<GlobalIndexType,double> globalCoefficientsToMap;
      for (int i=0; i<globalIndexVector.size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          if (globalCoefficients[0][localIndex] != 0.0)
            globalCoefficientsToMap[globalIndex] = globalCoefficients[0][localIndex];
        }
      }
//      if (globalCoefficientsToMap.size() == 0) cout << "***********************  Empty globalCoefficientsToMap **************************\n";
      localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsToMap);
    }
  }
  else
  {
    Intrepid::FieldContainer<double> globalCoefficientsFC(globalCoefficients.NumVectors(), globalIndexVector.size());
    Epetra_BlockMap partMap = globalCoefficients.Map();
    for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.NumVectors(); vectorOrdinal++)
    {
      for (int i=0; i<globalIndexVector.size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          globalCoefficientsFC(vectorOrdinal,i) = globalCoefficients[vectorOrdinal][localIndex];
        }
        else
        {
          // non-local coefficient: ignore
          globalCoefficientsFC(vectorOrdinal,i) = 0;
        }
      }
    }
    localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
  }
//  cout << "For cellID " << cellID << ", mapping globalData:\n " << globalDataFC;
//  cout << " to localData:\n " << localDofs;
}

template <typename Scalar>
void GDAMinimumRule::interpretGlobalCoefficients2(GlobalIndexType cellID, Intrepid::FieldContainer<Scalar> &localCoefficients, const TVectorPtr<Scalar> globalCoefficients)
{
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();

  // DEBUGGING
//  if (cellID==4) {
//    cout << "interpretGlobalData, mapping report for cell " << cellID << ":\n";
//    dofMapper->printMappingReport();
//  }

  if (globalCoefficients->getNumVectors() == 1)
  {
    // TODO: Test whether this is actually correct
    Teuchos::ArrayRCP<Scalar> globalCoefficientsArray = globalCoefficients->getDataNonConst(globalIndexVector.size());
    Intrepid::FieldContainer<Scalar> globalCoefficientsFC(globalIndexVector.size());
    Teuchos::ArrayView<Scalar> arrayView = globalCoefficientsArray.view(0,globalIndexVector.size());
    globalCoefficientsFC.setValues(arrayView);
    // ConstMapPtr partMap = globalCoefficients->getMap();
    // for (int i=0; i<globalIndexVector.size(); i++) {
    //   GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
    //   int localIndex = partMap->getLocalElement(globalIndex);
    //   if (localIndex != Teuchos::OrdinalTraits<IndexType>::invalid()) {
    //     globalCoefficientsFC[i] = globalCoefficients[0][localIndex];
    //   } else {
    //     // non-local coefficient: ignore
    //     globalCoefficientsFC[i] = 0;
    //   }
    // }
    localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "InterpretGlobalCoefficients not implemented for more than one vector");
    // Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar>> globalCoefficients2DArray = globalCoefficients->get2dViewNonConst(globalIndexVector.size());
    // Intrepid::FieldContainer<Scalar> globalCoefficientsFC(globalCoefficients.numVectors(), globalIndexVector.size());
    // for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.numVectors(); vectorOrdinal++) {
    //   Teuchos::ArrayRCP<Scalar> globalCoefficientsArray = globalCoefficients2DArray[vectorOrdinal];
    //   Teuchos::ArrayView<Scalar> arrayView = globalCoefficientsArray.view(0,globalIndexVector.size());
    //   globalCoefficientsFC.setValues(arrayView);
    // }
    //   Intrepid::FieldContainer<Scalar> globalCoefficientsFC(globalCoefficients.NumVectors(), globalIndexVector.size());
    //   Epetra_BlockMap partMap = globalCoefficients.Map();
    //   for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.NumVectors(); vectorOrdinal++) {
    //     for (int i=0; i<globalIndexVector.size(); i++) {
    //       GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
    //       int localIndex = partMap.LID(globalIndex);
    //       if (localIndex != -1) {
    //         globalCoefficientsFC(vectorOrdinal,i) = globalCoefficients[vectorOrdinal][localIndex];
    //       } else {
    //         // non-local coefficient: ignore
    //         globalCoefficientsFC(vectorOrdinal,i) = 0;
    //       }
    //     }
    // }
    // localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
  }
//  cout << "For cellID " << cellID << ", mapping globalData:\n " << globalDataFC;
//  cout << " to localData:\n " << localDofs;
}

template void GDAMinimumRule::interpretGlobalCoefficients2(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localCoefficients, const TVectorPtr<double> globalCoefficients);

void GDAMinimumRule::interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localData,
                                        Intrepid::FieldContainer<double> &globalData, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);

  // DEBUGGING
//  if (Teuchos::GlobalMPISession::getRank()==0) {
//    if (cellID==1) {
//      cout << "interpretLocalData, mapping report for cell " << cellID << ":\n";
//      dofMapper->printMappingReport();
//    }
//  }

  globalData = dofMapper->mapLocalData(localData, false);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();
  globalDofIndices.resize(globalIndexVector.size());
  for (int i=0; i<globalIndexVector.size(); i++)
  {
    globalDofIndices(i) = globalIndexVector[i];
  }

//  // mostly for debugging purposes, let's sort according to global dof index:
//  if (globalData.rank() == 1) {
//    map<GlobalIndexType, double> globalDataMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalDataMap[globalIndex] = globalData(i);
//    }
//    int i=0;
//    for (map<GlobalIndexType, double>::iterator mapIt = globalDataMap.begin(); mapIt != globalDataMap.end(); mapIt++) {
//      globalDofIndices(i) = mapIt->first;
//      globalData(i) = mapIt->second;
//      i++;
//    }
//  } else if (globalData.rank() == 2) {
//    cout << "globalData.rank() == 2.\n";
//    Intrepid::FieldContainer<double> globalDataCopy = globalData;
//    map<GlobalIndexType,int> globalIndexToOrdinalMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalIndexToOrdinalMap[globalIndex] = i;
//    }
//    int i=0;
//    for (map<GlobalIndexType,int>::iterator i_mapIt = globalIndexToOrdinalMap.begin();
//         i_mapIt != globalIndexToOrdinalMap.end(); i_mapIt++) {
//      int j=0;
//      globalDofIndices(i) = i_mapIt->first;
//      for (map<GlobalIndexType,int>::iterator j_mapIt = globalIndexToOrdinalMap.begin();
//           j_mapIt != globalIndexToOrdinalMap.end(); j_mapIt++) {
//        globalData(i,j) = globalDataCopy(i_mapIt->second,j_mapIt->second);
//
//        j++;
//      }
//      i++;
//    }
//  }

//  cout << "localData:\n" << localData;
//  cout << "globalData:\n" << globalData;
//  if (cellID==1)
//    cout << "globalIndices:\n" << globalDofIndices;
}

void GDAMinimumRule::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Intrepid::FieldContainer<double> &basisCoefficients,
    Intrepid::FieldContainer<double> &globalCoefficients, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints, varID, sideOrdinal);

  if (dofMapper->isPermutation())
  {
    map<int, GlobalIndexType> permutationMap = dofMapper->getPermutationMap();
    int entryOrdinal = 0;
    DofOrderingPtr dofOrdering = elementType(cellID)->trialOrderPtr;
    if (dofOrdering->hasBasisEntry(varID, sideOrdinal))
    {
      // we are mapping the whole basis in this case
      const vector<int>* localDofIndices = &dofOrdering->getDofIndices(varID, sideOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localDofIndices->size() != basisCoefficients.size(), std::invalid_argument, "basisCoefficients should be sized to match either the basis cardinality (when whole bases are matched) or the cardinality of the restriction");
      int numEntries = basisCoefficients.size();
      globalCoefficients.resize(numEntries);
      globalDofIndices.resize(numEntries);
      for (int localDofIndex : *localDofIndices)
      {
        globalCoefficients[entryOrdinal] = basisCoefficients[entryOrdinal];
        globalDofIndices[entryOrdinal] = permutationMap[localDofIndex];
        entryOrdinal++;
      }
    }
    else
    {
      // we are restricting a volume basis to the side indicated
      TEUCHOS_TEST_FOR_EXCEPTION(!dofOrdering->hasBasisEntry(varID, VOLUME_INTERIOR_SIDE_ORDINAL), std::invalid_argument, "Basis not found");
      BasisPtr volumeBasis = dofOrdering->getBasis(varID);
      set<int> basisDofOrdinals = volumeBasis->dofOrdinalsForSide(sideOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(basisDofOrdinals.size() != basisCoefficients.size(), std::invalid_argument, "basisCoefficients should be sized to match either the basis cardinality (when whole bases are matched) or the cardinality of the restriction");
      const vector<int>* localDofIndices = &dofOrdering->getDofIndices(varID);
      int numEntries = basisCoefficients.size();
      globalCoefficients.resize(numEntries);
      globalDofIndices.resize(numEntries);
      for (int basisDofOrdinal : basisDofOrdinals)
      {
        int localDofIndex = (*localDofIndices)[basisDofOrdinal];
        globalCoefficients[entryOrdinal] = basisCoefficients[entryOrdinal];
        globalDofIndices[entryOrdinal] = permutationMap[localDofIndex];
        entryOrdinal++;
      }
    }
  }
  else
  {
    globalCoefficients = dofMapper->fitLocalCoefficients(basisCoefficients);
    const vector<GlobalIndexType>* globalIndexVector = &dofMapper->fittableGlobalIndices();
    globalDofIndices.resize(globalIndexVector->size());

    for (int i=0; i<globalIndexVector->size(); i++)
    {
      globalDofIndices(i) = (*globalIndexVector)[i];
    }
  }
}

IndexType GDAMinimumRule::localDofCount()
{
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

typedef vector< SubBasisDofMapperPtr > BasisMap;
// volume variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofIndexInfo, VarPtr var)
{
  BasisMap varVolumeMap;
  
  vector<SubBasisMapInfo> subBasisMaps;
  SubBasisMapInfo subBasisMap;

  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;

  // assumption is that the basis is defined on the whole cell
  BasisPtr basis = trialOrdering->getBasis(var->ID());

  const static int DEBUG_VAR_ID = 0;
  const static GlobalIndexType DEBUG_CELL_ID = 3;
  const static GlobalIndexType DEBUG_GLOBAL_DOF = 0;
  const static int DEBUG_LOCAL_DOF = 5;
  
  // to begin, let's map the volume-interior dofs:
  vector<GlobalIndexType> globalDofOrdinals = dofIndexInfo[spaceDim][0][var->ID()];
  set<int> basisDofOrdinals = BasisReconciliation::interiorDofOrdinalsForBasis(basis); // basis->dofOrdinalsForInterior(); // TODO: change interiorDofOrdinalsForBasis to return set<unsigned>...

  if (basisDofOrdinals.size() > 0)
  {
    varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinals, globalDofOrdinals));
  }

  if (basisDofOrdinals.size() == basis->getCardinality())
  {
    // then we're done...
    return varVolumeMap;
  }
  
  static const int VOLUME_BASIS_SUBCORD = 0; // volume always has subcell ordinal 0 in volume
  
  int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
  
  for (int d = sideDim; d >= minimumConstraintDimension; d--)
  {
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int subcord=0; subcord < subcellCount; subcord++)
    {
      AnnotatedEntity subcellConstraint = getCellConstraints(cellID).subcellConstraints[d][subcord];
      DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[subcellConstraint.cellID]->trialOrderPtr;
      
      CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
      BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID());
      
      unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim,
                                                                                       subcellConstraint.sideOrdinal,
                                                                                       subcellConstraint.dimension,
                                                                                       subcellConstraint.subcellOrdinal);
      CellTopoPtr constrainingTopo = constrainingCell->topology()->getSubcell(subcellConstraint.dimension,
                                                                              subcellOrdinalInConstrainingCell);
      
      SubBasisReconciliationWeights weightsForSubcell;
      
      CellPtr ancestralCell = cell->ancestralCellForSubcell(d, subcord, _meshTopology);
      
      RefinementBranch volumeRefinements = cell->refinementBranchForSubcell(d, subcord, _meshTopology);
      if (volumeRefinements.size()==0)
      {
        // could be, we'd do better to revise Cell::refinementBranchForSubcell() to ensure that we always have a refinement, but for now
        // we just create a RefinementBranch with a trivial refinement here:
        RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(cell->topology());
        volumeRefinements = {{noRefinementPattern.get(),0}};
      }
      
      pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(d, subcord, _meshTopology);
      
      unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
      unsigned ancestralSubcellDimension = ancestralSubcell.second;
      
      if (ancestralSubcellOrdinal == -1)
      {
        cout << "Internal error: ancestral subcell ordinal was not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: ancestral subcell ordinal was not found.");
      }
    
      /***
       It is possible use the common computeConstrainedWeights() for both the case where the subcell dimension is the same as
       constraining and the one where it's different.  It appears, however, that this version of computeConstrainedWeights is a *LOT* 
       slower; overall runtime of runTests about doubled when I tried using this for both.
       ***/
      IndexType constrainingEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
      unsigned ancestralSubcellOrdinalInCell = ancestralCell->findSubcellOrdinal(subcellConstraint.dimension, constrainingEntityIndex);
      
      if (subcellConstraint.dimension != d)
      {
        // ancestralPermutation goes from canonical to cell's side's ancestor's ordering:
        unsigned ancestralCellPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);
        // constrainingPermutation goes from canonical to the constraining side's ordering
        unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(subcellConstraint.dimension, subcellOrdinalInConstrainingCell); // subcell permutation as seen from the perspective of the constraining cell's side
        
        // ancestralPermutationInverse goes from ancestral view to canonical
        unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralCellPermutation);
        unsigned ancestralToConstrainedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingCellPermutation);

        weightsForSubcell = BasisReconciliation::computeConstrainedWeights(d, basis, subcord,
                                                                           volumeRefinements,
                                                                           VOLUME_BASIS_SUBCORD,
                                                                           ancestralCell->topology(),
                                                                           subcellConstraint.dimension,
                                                                           constrainingBasis, ancestralSubcellOrdinalInCell,
                                                                           VOLUME_BASIS_SUBCORD, ancestralToConstrainedPermutation);
      }
      else
      {
        // from canonical to ancestral view:
        unsigned ancestralPermutation = ancestralCell->subcellPermutation(subcellConstraint.dimension, ancestralSubcellOrdinalInCell); // subcell permutation as seen from the perspective of the fine cell's ancestor
        // from canonical to constraining view:
        unsigned constrainingPermutation = constrainingCell->subcellPermutation(subcellConstraint.dimension,
                                                                                subcellOrdinalInConstrainingCell);
        
        // from ancestral to canonical:
        unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralPermutation);
        unsigned ancestralToConstrainingPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingPermutation);
        
        weightsForSubcell = _br.constrainedWeights(d, basis, subcord, volumeRefinements, constrainingBasis,
                                                   ancestralSubcellOrdinalInCell, ancestralToConstrainingPermutation);
      }
      
      // add sub-basis map for dofs interior to the constraining subcell
      // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
      SubBasisReconciliationWeights weightsForWholeSubcell = weightsForSubcell; // copy to save before we filter
      weightsForSubcell = BasisReconciliation::weightsForCoarseSubcell(weightsForSubcell, constrainingBasis,
                                                                       subcellConstraint.dimension, ancestralSubcellOrdinalInCell,
                                                                       false);
      
      if ((weightsForSubcell.coarseOrdinals.size() > 0) && (weightsForSubcell.fineOrdinals.size() > 0))
      {
        CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
        OwnershipInfo ownershipInfo = constrainingCellConstraints.owningCellIDForSubcell[subcellConstraint.dimension][subcellOrdinalInConstrainingCell];
        CellConstraints owningCellConstraints = getCellConstraints(ownershipInfo.cellID);
        SubCellDofIndexInfo owningCellDofIndexInfo = getOwnedGlobalDofIndices(ownershipInfo.cellID, owningCellConstraints);
        unsigned owningSubcellOrdinal = _meshTopology->getCell(ownershipInfo.cellID)->findSubcellOrdinal(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
        vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo[ownershipInfo.dimension][owningSubcellOrdinal][var->ID()];
        
        // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
        vector<int> basisOrdinalsVector = constrainingBasis->dofOrdinalsForSubcell(subcellConstraint.dimension, ancestralSubcellOrdinalInCell);
//        vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
        vector<GlobalIndexType> globalDofOrdinals;
        for (int i=0; i<basisOrdinalsVector.size(); i++)
        {
          if (weightsForSubcell.coarseOrdinals.find(basisOrdinalsVector[i]) != weightsForSubcell.coarseOrdinals.end())
          {
            globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
          }
        }
        
        if (weightsForSubcell.coarseOrdinals.size() != globalDofOrdinals.size())
        {
          cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
          Camellia::print("coarseOrdinals", weightsForSubcell.coarseOrdinals);
          Camellia::print("globalDofOrdinals", globalDofOrdinals);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
        }
        
        subBasisMap.weights = weightsForSubcell.weights;
        subBasisMap.globalDofOrdinals = globalDofOrdinals;
        subBasisMap.basisDofOrdinals = weightsForSubcell.fineOrdinals;
        
//        { // DEBUGGING
//          if (cellID == DEBUG_CELL_ID)
//          {
//            if (var->ID() == DEBUG_VAR_ID)
//            {
//              if (std::find(subBasisMap.globalDofOrdinals.begin(), subBasisMap.globalDofOrdinals.end(), DEBUG_GLOBAL_DOF) != subBasisMap.globalDofOrdinals.end())
//              {
//                if (subBasisMap.basisDofOrdinals.find(DEBUG_LOCAL_DOF) != subBasisMap.basisDofOrdinals.end())
//                {
//                  cout << CamelliaCellTools::entityTypeString(d) << " " << subcord << " on cell " << cellID << " constrained by ";
//                  cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
//                  cout << subcellConstraint.cellID << ".\n";
//                  
//                  Camellia::print("weights coarse ordinals", weightsForSubcell.coarseOrdinals);
//                  Camellia::print("subBasisMap.basisDofOrdinals", subBasisMap.basisDofOrdinals);
//                  Camellia::print("subBasisMap.globalDofOrdinals", subBasisMap.globalDofOrdinals);
//                  cout << "subBasisMap.weights:\n" << weightsForSubcell.weights;
//                }
//              }
//            }
//          }
//        }
        
        subBasisMaps.push_back(subBasisMap);
      }
      
      // process subcells of the coarse subcell (new code, new idea as of 2-8-16; passes tests thus far, but coverage isn't terribly thorough)
      for (int subsubcdim=minimumConstraintDimension; subsubcdim<subcellConstraint.dimension; subsubcdim++)
      {
        int subsubcellCount = constrainingTopo->getSubcellCount(subsubcdim);
        for (int subsubcellOrdinal = 0; subsubcellOrdinal < subsubcellCount; subsubcellOrdinal++)
        {
          // first question: is this subcell of the original constraining subcell further constrained?
          // (In a 1-irregular mesh, I believe this is only possible if the original constraint did not involve a hanging node--could be a permutation,
          //  or a trivial constraint.)
          // If it is further constrained, then I *think* this will naturally be handled at some other point.
          int subsubcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), subcellConstraint.dimension,
                                                                                         subcellOrdinalInConstrainingCell, subsubcdim, subsubcellOrdinal);
          
          CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
          AnnotatedEntity subsubcellConstraints = constrainingCellConstraints.subcellConstraints[subsubcdim][subsubcellOrdinalInConstrainingCell];
          bool furtherConstrained;
          if (subsubcellConstraints.dimension != subsubcdim)
            furtherConstrained = true;
          else
          {
            CellPtr subsubcellConstrainingCell = _meshTopology->getCell(subsubcellConstraints.cellID);
            int subsubcellConstrainingSubcellOrdinalInItsConstrainingCell = CamelliaCellTools::subcellOrdinalMap(subsubcellConstrainingCell->topology(), sideDim,
                                                                                                                 subsubcellConstraints.sideOrdinal, subsubcdim,
                                                                                                                 subsubcellConstraints.subcellOrdinal);
            IndexType constrainingEntityIndex = subsubcellConstrainingCell->entityIndex(subsubcdim, subsubcellConstrainingSubcellOrdinalInItsConstrainingCell);
            IndexType subsubcellEntityIndex = constrainingCell->entityIndex(subsubcdim, subsubcellOrdinalInConstrainingCell);
            furtherConstrained = (constrainingEntityIndex != subsubcellEntityIndex);
          }
          if (furtherConstrained) continue;
          
          SubBasisReconciliationWeights weightsForSubSubcell = BasisReconciliation::weightsForCoarseSubcell(weightsForWholeSubcell, constrainingBasis,
                                                                                                            subsubcdim, subsubcellOrdinalInConstrainingCell,
                                                                                                            false);
          
          // copied and pasted from above.  Could refactor:
          if ((weightsForSubSubcell.coarseOrdinals.size() > 0) && (weightsForSubSubcell.fineOrdinals.size() > 0))
          {
            OwnershipInfo ownershipInfo = constrainingCellConstraints.owningCellIDForSubcell[subsubcdim][subsubcellOrdinalInConstrainingCell];
            CellConstraints owningCellConstraints = getCellConstraints(ownershipInfo.cellID);
            SubCellDofIndexInfo owningCellDofIndexInfo = getOwnedGlobalDofIndices(ownershipInfo.cellID, owningCellConstraints);
            unsigned owningSubcellOrdinal = _meshTopology->getCell(ownershipInfo.cellID)->findSubcellOrdinal(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
            vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo[ownershipInfo.dimension][owningSubcellOrdinal][var->ID()];
            
            // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
            
            const vector<int>* constrainingBasisOrdinalsForSubcell = &constrainingBasis->dofOrdinalsForSubcell(subsubcdim, subsubcellOrdinalInConstrainingCell);
//            set<int> constrainingBasisOrdinalsForSubcell = constrainingBasis->dofOrdinalsForSubcell(subsubcdim, subsubcellOrdinalInConstrainingCell);
//            vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
            vector<GlobalIndexType> globalDofOrdinals;
            for (int i=0; i<constrainingBasisOrdinalsForSubcell->size(); i++)
            {
              int constrainingBasisOrdinal = (*constrainingBasisOrdinalsForSubcell)[i];
              if (weightsForSubSubcell.coarseOrdinals.find(constrainingBasisOrdinal) != weightsForSubSubcell.coarseOrdinals.end())
              {
                globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
              }
            }
            
            if (weightsForSubSubcell.coarseOrdinals.size() != globalDofOrdinals.size())
            {
              cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
              Camellia::print("coarseOrdinals", weightsForSubSubcell.coarseOrdinals);
              Camellia::print("globalDofOrdinals", globalDofOrdinals);
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
            }
            
            subBasisMap.weights = weightsForSubSubcell.weights;
            subBasisMap.globalDofOrdinals = globalDofOrdinals;
            subBasisMap.basisDofOrdinals = weightsForSubSubcell.fineOrdinals;
            
//            { // DEBUGGING
//              if (cellID == DEBUG_CELL_ID)
//              {
//                if (var->ID() == DEBUG_VAR_ID)
//                {
//                  if (std::find(subBasisMap.globalDofOrdinals.begin(), subBasisMap.globalDofOrdinals.end(), DEBUG_GLOBAL_DOF) != subBasisMap.globalDofOrdinals.end())
//                  {
//                    if (subBasisMap.basisDofOrdinals.find(DEBUG_LOCAL_DOF) != subBasisMap.basisDofOrdinals.end())
//                    {
//                      cout << CamelliaCellTools::entityTypeString(d) << " " << subcord << " on cell " << cellID << " constrained by ";
//                      cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
//                      cout << subcellConstraint.cellID << "; " << CamelliaCellTools::entityTypeString(subsubcdim) << " " << subsubcellOrdinalInConstrainingCell;
//                      cout << " on cell " << subcellConstraint.cellID << " constrained by ";
//                      cout << CamelliaCellTools::entityTypeString(subsubcellConstraints.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
//                      cout << subsubcellConstraints.cellID << ".\n";
//                      
//                      Camellia::print("weights coarse ordinals", weightsForSubSubcell.coarseOrdinals);
//                      Camellia::print("subBasisMap.basisDofOrdinals", subBasisMap.basisDofOrdinals);
//                      Camellia::print("subBasisMap.globalDofOrdinals", subBasisMap.globalDofOrdinals);
//                      cout << "subBasisMap.weights:\n" << weightsForSubSubcell.weights;
//                    }
//                  }
//                }
//              }
//            }
            
            subBasisMaps.push_back(subBasisMap);
            
            //              { // DEBUGGING
            //                cout << "weights for whole subcell:\n";
            //                Camellia::print("coarseOrdinals", weightsForWholeSubcell.coarseOrdinals);
            //                Camellia::print("fineOrdinals", weightsForWholeSubcell.fineOrdinals);
            //                cout << weightsForWholeSubcell.weights;
            //
            //                cout << "weights for sub-subcell:\n";
            //                Camellia::print("coarseOrdinals", weightsForSubSubcell.coarseOrdinals);
            //                Camellia::print("fineOrdinals", weightsForSubSubcell.fineOrdinals);
            //                cout << weightsForSubSubcell.weights;
            //              }
          }
        }
      }
    }
  }
  
  // now, we collect the local basis coefficients corresponding to each global ordinal
  // likely there is a more efficient way to do this, but for now this is our approach
  map< GlobalIndexType, map<int, double> > weightsForGlobalOrdinal;
  map< int, set<GlobalIndexType> > globalOrdinalsForFineOrdinal;
  
  for (vector<SubBasisMapInfo>::iterator subBasisIt = subBasisMaps.begin(); subBasisIt != subBasisMaps.end(); subBasisIt++)
  {
    subBasisMap = *subBasisIt;
    vector<GlobalIndexType> globalDofOrdinals = subBasisMap.globalDofOrdinals;
    set<int>* basisDofOrdinals = &subBasisMap.basisDofOrdinals;
    // weights are fine x coarse
    for (int j=0; j<subBasisMap.weights.dimension(1); j++)
    {
      GlobalIndexType globalDofOrdinal = globalDofOrdinals[j];
      map<int, double> fineOrdinalCoefficientsThusFar = weightsForGlobalOrdinal[globalDofOrdinal];
      auto fineOrdinalPtr = basisDofOrdinals->begin();
      for (int i=0; i<subBasisMap.weights.dimension(0); i++)
      {
        int fineOrdinal = *fineOrdinalPtr++;
        double coefficient = subBasisMap.weights(i,j);
        if (coefficient != 0)
        {
          if (fineOrdinalCoefficientsThusFar.find(fineOrdinal) != fineOrdinalCoefficientsThusFar.end())
          {
            double tol = 1e-14;
            double previousCoefficient = fineOrdinalCoefficientsThusFar[fineOrdinal];
            if (abs(previousCoefficient-coefficient) > tol)
            {
              cout  << "ERROR: incompatible entries for fine ordinal " << fineOrdinal << " in representation of global ordinal " << globalDofOrdinal << endl;
              cout << "previousCoefficient = " << previousCoefficient << endl;
              cout << "coefficient = " << coefficient << endl;
              cout << "diff = " << abs(previousCoefficient - coefficient) << endl;
              cout << "Encountered the incompatible entry while processing variable " << var->name() << " on cell " << cellID << ", on volume" << endl;
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal Error: incompatible entries for fine ordinal " );
            }
          }
          fineOrdinalCoefficientsThusFar[fineOrdinal] = coefficient;
          globalOrdinalsForFineOrdinal[fineOrdinal].insert(globalDofOrdinal);
//          {
//            // DEBUGGING
//            if (cellID == DEBUG_CELL_ID)
//            {
//              if (var->ID() == DEBUG_VAR_ID)
//              {
//                if ((fineOrdinal == DEBUG_LOCAL_DOF) && (globalDofOrdinal == DEBUG_GLOBAL_DOF))
//                {
//                  cout << "subBasisMap.weights:\n" << subBasisMap.weights;
//                  cout << "Added weight " << coefficient << " for global dof " << globalDofOrdinal << " representation by local basis ordinal " << fineOrdinal << endl;
//                }
//              }
//            }
//          }
        }
      }
      weightsForGlobalOrdinal[globalDofOrdinal] = fineOrdinalCoefficientsThusFar;
    }
  }
  
  // partition global ordinals according to which fine ordinals they interact with -- this is definitely not super-efficient
  set<GlobalIndexType> partitionedGlobalDofOrdinals;
  vector< set<GlobalIndexType> > globalDofOrdinalPartitions;
  vector< set<int> > fineOrdinalsForPartition;
  
  for (map< GlobalIndexType, map<int, double> >::iterator globalWeightsIt = weightsForGlobalOrdinal.begin();
       globalWeightsIt != weightsForGlobalOrdinal.end(); globalWeightsIt++)
  {
    GlobalIndexType globalOrdinal = globalWeightsIt->first;
    if (partitionedGlobalDofOrdinals.find(globalOrdinal) != partitionedGlobalDofOrdinals.end()) continue;
    
    set<GlobalIndexType> partition;
    partition.insert(globalOrdinal);
    
    set<int> fineOrdinals;
    
    set<GlobalIndexType> globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed;
    
    map<int, double> fineCoefficients = globalWeightsIt->second;
    for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
    {
      int fineOrdinal = coefficientIt->first;
      fineOrdinals.insert(fineOrdinal);
      set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
      partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
    }
    
    globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
    
    while (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.size() != partition.size())
    {
      for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
      {
        GlobalIndexType globalOrdinal = *globalOrdIt;
        if (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.find(globalOrdinal) !=
            globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.end()) continue;
        map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
        for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
        {
          int fineOrdinal = coefficientIt->first;
          fineOrdinals.insert(fineOrdinal);
          set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
          partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
        }
        
        globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
      }
    }
    
    for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
    {
      GlobalIndexType globalOrdinal = *globalOrdIt;
      partitionedGlobalDofOrdinals.insert(globalOrdinal);
    }
    globalDofOrdinalPartitions.push_back(partition);
    fineOrdinalsForPartition.push_back(fineOrdinals);
  }
  
  for (int i=0; i<globalDofOrdinalPartitions.size(); i++)
  {
    set<GlobalIndexType> partition = globalDofOrdinalPartitions[i];
    set<int> fineOrdinals = fineOrdinalsForPartition[i];
    vector<int> fineOrdinalsVector(fineOrdinals.begin(), fineOrdinals.end());
    map<int,int> fineOrdinalRowLookup;
    for (int i=0; i<fineOrdinalsVector.size(); i++)
    {
      fineOrdinalRowLookup[fineOrdinalsVector[i]] = i;
    }
    Intrepid::FieldContainer<double> weights(fineOrdinals.size(),partition.size());
    int col=0;
    for (set<GlobalIndexType>::iterator globalDofIt=partition.begin(); globalDofIt != partition.end(); globalDofIt++)
    {
      GlobalIndexType globalOrdinal = *globalDofIt;
      map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
      for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
      {
        int fineOrdinal = coefficientIt->first;
        double coefficient = coefficientIt->second;
        int row = fineOrdinalRowLookup[fineOrdinal];
        weights(row,col) = coefficient;
      }
      col++;
    }
    vector<GlobalIndexType> globalOrdinals(partition.begin(),partition.end());
    SubBasisDofMapperPtr subBasisMap = SubBasisDofMapper::subBasisDofMapper(fineOrdinals, globalOrdinals, weights);
    varVolumeMap.push_back(subBasisMap);
  }
  
  return varVolumeMap;
}

BasisMap GDAMinimumRule::getBasisMapDiscontinuousVolumeRestrictedToSide(GlobalIndexType cellID, SubCellDofIndexInfo& dofOwnershipInfo, VarPtr var, int sideOrdinal)
{
   BasisMap volumeMap = getBasisMap(cellID, dofOwnershipInfo, var);

  // We're interested in the restriction of the map to the side
  
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  BasisPtr basis = trialOrdering->getBasis(var->ID());
  set<int> basisDofOrdinalsForSide = basis->dofOrdinalsForSide(sideOrdinal);
  
  // for now, the only case we will support is the one where each volume dof is identified with a single global dof
  // (this is the case when using discontinuous volume dofs, which is the case for ultraweak DPG as well as DG)
  // we assert that condition here:
  {
    for (SubBasisDofMapperPtr subBasisDofMapper : volumeMap)
    {
      // does this mapper involve any of the dof ordinals we're interested in?
      // if so, we require that it is a permutation -- i.e. no constraints imposed
      bool containsSideDofOrdinal = false;
      for (int basisDofOrdinal : basisDofOrdinalsForSide)
      {
        if (subBasisDofMapper->basisDofOrdinalFilter().find(basisDofOrdinal) != subBasisDofMapper->basisDofOrdinalFilter().end())
        {
          containsSideDofOrdinal = true;
          break;
        }
      }
      if (! containsSideDofOrdinal) break;
      
      TEUCHOS_TEST_FOR_EXCEPTION(!subBasisDofMapper->isPermutation(), std::invalid_argument, "getDofMapper() only supports side restrictions of volume variables in cases where the volume variables do not have any constraints imposed");
    }
  }

  vector<GlobalIndexType> globalDofOrdinals;
  for (int basisDofOrdinalForSide : basisDofOrdinalsForSide)
  {
    for (SubBasisDofMapperPtr subBasisDofMapper : volumeMap)
    {
      if (subBasisDofMapper->basisDofOrdinalFilter().find(basisDofOrdinalForSide) != subBasisDofMapper->basisDofOrdinalFilter().end())
      {
        // this dof mapper contains the guy we're looking for
        set<int> basisDofOrdinalSet = {basisDofOrdinalForSide};
        set<GlobalIndexType> mappedGlobalDofOrdinals = subBasisDofMapper->mappedGlobalDofOrdinalsForBasisOrdinals(basisDofOrdinalSet);
        if (mappedGlobalDofOrdinals.size() != 1)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getDofMapper() only supports side restrictions of volume variables if volume mapping is a permutation");
        }
        GlobalIndexType mappedGlobalDofOrdinal = *mappedGlobalDofOrdinals.begin();
        
        globalDofOrdinals.push_back(mappedGlobalDofOrdinal);
        break; // break out of subBasisDofMapper iteration loop -- we found our guy
      }
    }
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(globalDofOrdinals.size() != basisDofOrdinalsForSide.size(), std::invalid_argument, "Error: did not find globalDofOrdinals for all basisDofOrdinals on the side");
  
  BasisMap varVolumeMap;
  
  if (basisDofOrdinalsForSide.size() > 0)
  {
    varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinalsForSide, globalDofOrdinals));
  }

  return varVolumeMap;
}

// trace variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofIndexInfo, VarPtr var, int sideOrdinal)
{
  vector<SubBasisMapInfo> subBasisMaps;

  SubBasisMapInfo subBasisMap;

//  const static int DEBUG_VAR_ID = 2;
//  const static GlobalIndexType DEBUG_CELL_ID = 0;
//  const static unsigned DEBUG_SIDE_ORDINAL = 2;
//  const static GlobalIndexType DEBUG_GLOBAL_DOF = -3;

  const static int DEBUG_VAR_ID = -1;
  const static GlobalIndexType DEBUG_CELL_ID = -1;
  const static unsigned DEBUG_SIDE_ORDINAL = -1;
  const static GlobalIndexType DEBUG_GLOBAL_DOF = -3;

//  if ((cellID==2) && (sideOrdinal==1) && (var->ID() == 0)) {
//    cout << "DEBUGGING: (cellID==2) && (sideOrdinal==1) && (var->ID() == 0).\n";
//  }

  // TODO: move the permutation computation outside of this method -- might include in CellConstraints, e.g. -- this obviously will not change from one var to the next, but we compute it redundantly each time...

  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;
  CellTopoPtr sideTopo = topo->getSubcell(sideDim, sideOrdinal);

  // assumption is that the basis is defined on the side
  BasisPtr basis = trialOrdering->getBasis(var->ID(), sideOrdinal);

  typedef pair<AnnotatedEntity, SubBasisReconciliationWeights > AppliedWeightPair;
  typedef vector< AppliedWeightPair > AppliedWeightVector;

  vector< map< GlobalIndexType, AppliedWeightVector > > appliedWeights(sideDim+1); // map keys are the entity indices; these are used to ensure that we don't apply constraints for a given entity multiple times.

  AnnotatedEntity defaultConstraint;
  defaultConstraint.cellID = cellID;
  defaultConstraint.sideOrdinal = sideOrdinal;
  defaultConstraint.subcellOrdinal = 0;
  defaultConstraint.dimension = sideDim;

  SubBasisReconciliationWeights unitWeights;
  unitWeights.weights.resize(basis->getCardinality(), basis->getCardinality());
  set<int> allOrdinals;
  for (int i=0; i<basis->getCardinality(); i++)
  {
    allOrdinals.insert(i);
    unitWeights.weights(i,i) = 1.0;
  }
  unitWeights.fineOrdinals = allOrdinals;
  unitWeights.coarseOrdinals = allOrdinals;

  GlobalIndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
  appliedWeights[sideDim][sideEntityIndex].push_back(make_pair(defaultConstraint, unitWeights));

  int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);

  int appliedWeightsGreatestEntryDimension = sideDim; // the greatest dimension for which appliedWeights is non-empty
  while (appliedWeightsGreatestEntryDimension >= minimumConstraintDimension)
  {
    int d = appliedWeightsGreatestEntryDimension; // the dimension of the subcell being constrained.

    map< GlobalIndexType, vector< pair<AnnotatedEntity, SubBasisReconciliationWeights > > > appliedWeightsForDimension = appliedWeights[d];

    // clear these out from the main container:
    appliedWeights[d].clear();

    map< GlobalIndexType, AppliedWeightVector >::iterator appliedWeightsIt;
    for (appliedWeightsIt = appliedWeightsForDimension.begin(); appliedWeightsIt != appliedWeightsForDimension.end(); appliedWeightsIt++)
    {
      // appliedWeightVectorForSubcell seems misnamed to me; just appliedWeightVector would seem more appropriate
      // (we don't yet have a particular subcell in mind; the entries in "appliedWeightVectorForSubcell" specify which
      //  subcell they apply to, right?)
      AppliedWeightVector appliedWeightVectorForSubcell = appliedWeightsIt->second;

      for (AppliedWeightVector::iterator appliedWeightPairIt = appliedWeightVectorForSubcell.begin();
           appliedWeightPairIt != appliedWeightVectorForSubcell.end(); appliedWeightPairIt++)
      {
        AppliedWeightPair appliedWeightsForSubcell = *appliedWeightPairIt;

        AnnotatedEntity subcellInfo = appliedWeightsForSubcell.first;
        SubBasisReconciliationWeights prevWeights = appliedWeightsForSubcell.second;

        if (subcellInfo.dimension != d)
        {
          cout << "INTERNAL ERROR: subcellInfo.dimension should be d!\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "INTERNAL ERROR: subcellInfo.dimension should be d!");
        }

        CellPtr appliedConstraintCell = _meshTopology->getCell(subcellInfo.cellID);

        unsigned subcordInAppliedConstraintCell = CamelliaCellTools::subcellOrdinalMap(appliedConstraintCell->topology(), sideDim,
            subcellInfo.sideOrdinal, d, subcellInfo.subcellOrdinal);

        DofOrderingPtr appliedConstraintTrialOrdering = _elementTypeForCell[subcellInfo.cellID]->trialOrderPtr;
        BasisPtr appliedConstraintBasis = appliedConstraintTrialOrdering->getBasis(var->ID(), subcellInfo.sideOrdinal);

//        CellConstraints cellConstraints = getCellConstraints(subcellInfo.cellID);
        AnnotatedEntity subcellConstraint = getCellConstraints(subcellInfo.cellID).subcellConstraints[d][subcordInAppliedConstraintCell];

        DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[subcellConstraint.cellID]->trialOrderPtr;
        
//        if (! constrainingTrialOrdering->hasBasisEntry(var->ID(), subcellConstraint.sideOrdinal))
//        {
//          // assumption is, "conforming" space-time trace that's not supported on the temporal side.  We should treat this guy as
//          // self-constrained:
//          subcellConstraint = subcellInfo;
//          // TODO: check if this is reasonably treated below...
//          cout << "WARNING: constrainingTrialOrdering->hasBasisEntry(var->ID(), subcellConstraint.sideOrdinal) returned false.\n";
//          constrainingTrialOrdering = _elementTypeForCell[subcellConstraint.sideOrdinal]->trialOrderPtr;
//        }
        
        CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
        BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID(), subcellConstraint.sideOrdinal);


        unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim, subcellConstraint.sideOrdinal,
            subcellConstraint.dimension, subcellConstraint.subcellOrdinal);

        // DEBUGGING
        if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
        {
          IndexType appliedConstraintSubcellEntityIndex = appliedConstraintCell->entityIndex(subcellInfo.dimension, subcordInAppliedConstraintCell);
          IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);

          cout << "while getting basis map for cell " << DEBUG_CELL_ID << ", side " << DEBUG_SIDE_ORDINAL << ": cell " << subcellInfo.cellID << ", side " << subcellInfo.sideOrdinal;
          cout << ", subcell " << subcellInfo.subcellOrdinal << " of dimension " << subcellInfo.dimension << " is ";
          if ((subcellInfo.cellID == subcellConstraint.cellID) &&
              (subcellInfo.sideOrdinal == subcellConstraint.sideOrdinal) &&
              (subcellInfo.dimension == subcellConstraint.dimension) &&
              (subcellInfo.subcellOrdinal == subcellConstraint.subcellOrdinal))
          {
            cout << "unconstrained.\n";
            cout << CamelliaCellTools::entityTypeString(subcellInfo.dimension) << " " << appliedConstraintSubcellEntityIndex;
            cout << " is unconstrained.\n";
          }
          else
          {
            cout << "constrained by cell " << subcellConstraint.cellID << ", side " << subcellConstraint.sideOrdinal;
            cout << ", subcell " << subcellConstraint.subcellOrdinal << " of dimension " << subcellConstraint.dimension << endl;
            cout << CamelliaCellTools::entityTypeString(subcellInfo.dimension) << " " << appliedConstraintSubcellEntityIndex;
            cout << " constrained by " << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << constrainingSubcellEntityIndex;
            cout << endl;
          }
        }

        CellTopoPtr constrainingTopo = constrainingCell->topology()->getSubcell(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);

        SubBasisReconciliationWeights composedWeights;

        CellPtr ancestralCell = appliedConstraintCell->ancestralCellForSubcell(d, subcordInAppliedConstraintCell, _meshTopology);

        RefinementBranch volumeRefinements = appliedConstraintCell->refinementBranchForSubcell(d, subcordInAppliedConstraintCell, _meshTopology);
        if (volumeRefinements.size()==0)
        {
          // could be, we'd do better to revise Cell::refinementBranchForSubcell() to ensure that we always have a refinement, but for now
          // we just create a RefinementBranch with a trivial refinement here:
          RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(appliedConstraintCell->topology());
          volumeRefinements = {{noRefinementPattern.get(),0}};
        }

        pair<unsigned, unsigned> ancestralSubcell = appliedConstraintCell->ancestralSubcellOrdinalAndDimension(d, subcordInAppliedConstraintCell, _meshTopology);

        unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
        unsigned ancestralSubcellDimension = ancestralSubcell.second;

        if (ancestralSubcellOrdinal == -1)
        {
          cout << "Internal error: ancestral subcell ordinal was not found.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: ancestral subcell ordinal was not found.");
        }

        unsigned ancestralSideOrdinal;
        /*
         How we know that there is always an ancestralSideOrdinal to speak of:
         In the extreme case, consider what happens when you have a triangular refinement and an interior triangle selected,
         and you're concerned with one of its vertices: even then there is a side ordinal even then, and a unique one.  Whatever constraints
         there are eventually come through some constraining side (or the subcell of some constraining side).
         */
        if (ancestralSubcellDimension == sideDim)
        {
          ancestralSideOrdinal = ancestralSubcellOrdinal;
        }
        else
        {

          IndexType ancestralSubcellEntityIndex = ancestralCell->entityIndex(ancestralSubcellDimension, ancestralSubcellOrdinal);

          // for subcells constrained by subcells of unlike dimension, we can handle any side that contains the ancestral subcell,
          // but for like-dimensional constraints, we do need the ancestralSideOrdinal to be the ancestor of the side in subcellInfo...

          if (subcellConstraint.dimension == d)
          {
            IndexType descendantSideEntityIndex = appliedConstraintCell->entityIndex(sideDim, subcellInfo.sideOrdinal);

            ancestralSideOrdinal = -1;
            int sideCount = ancestralCell->getSideCount();
            for (int side=0; side<sideCount; side++)
            {
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
              if (ancestralSideEntityIndex == descendantSideEntityIndex)
              {
                ancestralSideOrdinal = side;
                break;
              }

              if (_meshTopology->entityIsAncestor(sideDim, ancestralSideEntityIndex, descendantSideEntityIndex))
              {
                ancestralSideOrdinal = side;
                break;
              }
            }

            if (ancestralSideOrdinal == -1)
            {
              cout << "Error: no ancestor of side found.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: no ancestor of side contains the ancestral subcell.");
            }

            {
              // a sanity check:
              vector<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, ancestralSideOrdinal);
              if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) == sidesForSubcell.end())
              {
                cout << "Error: the ancestral side does not contain the ancestral subcell.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "the ancestral side does not contain the ancestral subcell.");
              }
            }

          }
          else
          {
            // find some side in the ancestral cell that contains the ancestral subcell, then (there should be at least two; which one shouldn't matter)
            vector<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);

            ancestralSideOrdinal = -1;
            int sideCount = ancestralCell->getSideCount();
            for (int side=0; side<sideCount; side++)
            {
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
              if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) != sidesForSubcell.end())
              {
                ancestralSideOrdinal = side;
                break;
              }
            }
          }
        }

        if (ancestralSideOrdinal == -1)
        {
          cout << "Error: ancestralSideOrdinal not found.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: ancestralSideOrdinal not found.");
        }

        // 5-21-15: It is possible use the common computeConstrainedWeights() for both the case where the subcell dimension is the same as constraining and the
        //          one where it's different.  There are bugs, perhaps with the treatment of permutations, that are revealed by doing so--two runTests tests,
        //          Solution_ProjectOnTensorMesh2D_Slow and Solution_ProjectOnTensorMesh3D_Slow, fail that don't with the separate treatment.  However, even once
        //          said bugs are fixed, it is appears this version of computeConstrainedWeights is a *LOT* slower; overall runtime of runTests about doubled when
        //          I tried using this for both.
        if (subcellConstraint.dimension != d)
        {
          // 5-25-15: changing permutations here to be relative to *volumes*
          // ancestralPermutation goes from canonical to cell's side's ancestor's ordering:
          unsigned ancestralCellPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);
          // constrainingPermutation goes from canonical to the constraining side's ordering
          unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(subcellConstraint.dimension, subcellOrdinalInConstrainingCell); // subcell permutation as seen from the perspective of the constraining cell's side

          // ancestralPermutationInverse goes from ancestral view to canonical
          unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralCellPermutation);
          unsigned ancestralToConstrainedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingCellPermutation);


          // 5-21-15: second-to-last argument of the following call was ancestralSideOrdinal.  AFAIK it won't make a difference on the present implementation--since
          //          I believe the ancestor is generally the constraining cell, but I'm pretty sure the conceptually correct thing here is not ancestralSideOrdinal
          //          but subcellConstraint.sideOrdinal.  So I have replaced this just now.
          SubBasisReconciliationWeights newWeightsToApply = BasisReconciliation::computeConstrainedWeights(d, appliedConstraintBasis, subcellInfo.subcellOrdinal,
              volumeRefinements, subcellInfo.sideOrdinal,
              ancestralCell->topology(), subcellConstraint.dimension,
              constrainingBasis, subcellConstraint.subcellOrdinal,
              subcellConstraint.sideOrdinal, ancestralToConstrainedPermutation);



          composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(prevWeights, newWeightsToApply);

// DEBUGGING
          if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
          {
            cout << "subcellInfo:\n" << subcellInfo << endl;
            cout << "subcellConstraint:\n" << subcellConstraint << endl;

            cout << "ancestralToConstrainedPermutation: " << ancestralToConstrainedPermutation << endl;

            cout << "prevWeights:\n";
            Camellia::print("prevWeights fine ordinals", prevWeights.fineOrdinals);
            Camellia::print("prevWeights coarse ordinals", prevWeights.coarseOrdinals);
            cout << "prevWeights weights:\n" << prevWeights.weights;

            cout << "newWeightsToApply:\n";
            Camellia::print("newWeightsToApply fine ordinals", newWeightsToApply.fineOrdinals);
            Camellia::print("newWeightsToApply coarse ordinals", newWeightsToApply.coarseOrdinals);
            cout << "newWeightsToApply weights:\n" << newWeightsToApply.weights;

            cout << "composedWeights:\n";
            Camellia::print("composedWeights fine ordinals", composedWeights.fineOrdinals);
            Camellia::print("composedWeights coarse ordinals", composedWeights.coarseOrdinals);
            cout << "composedWeights weights:\n" << composedWeights.weights;
          }

        }
        else
        {
          RefinementBranch sideRefinements = RefinementPattern::subcellRefinementBranch(volumeRefinements, sideDim, ancestralSideOrdinal);

          IndexType constrainingEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);

          unsigned ancestralSubcellOrdinalInCell = ancestralCell->findSubcellOrdinal(subcellConstraint.dimension, constrainingEntityIndex);

          unsigned ancestralSubcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(ancestralCell->topology(), sideDim, ancestralSideOrdinal, subcellConstraint.dimension, ancestralSubcellOrdinalInCell);

          // from canonical to ancestral view:
          unsigned ancestralPermutation = ancestralCell->sideSubcellPermutation(ancestralSideOrdinal, d, ancestralSubcellOrdinalInSide); // subcell permutation as seen from the perspective of the fine cell's side's ancestor
          // from canonical to constraining view:
          unsigned constrainingPermutation = constrainingCell->sideSubcellPermutation(subcellConstraint.sideOrdinal, subcellConstraint.dimension, subcellConstraint.subcellOrdinal); // subcell permutation as seen from the perspective of the constraining cell's side

          // 5-14-15: I think this permutation goes the wrong way
//          unsigned constrainingPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, constrainingPermutation);
//          unsigned composedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, constrainingPermutationInverse, ancestralPermutation);

          // 5-14-15: trying this instead:
          // from ancestral to canonical:
          unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralPermutation);
          unsigned ancestralToConstrainingPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingPermutation);

          SubBasisReconciliationWeights newWeightsToApply = _br.constrainedWeights(d, appliedConstraintBasis,
                                                                                   subcellInfo.subcellOrdinal,
                                                                                   sideRefinements, constrainingBasis,
                                                                                   subcellConstraint.subcellOrdinal,
                                                                                   ancestralToConstrainingPermutation);

          // DEBUGGING
          if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
          {
            cout << "newWeightsToApply:\n";
            Camellia::print("newWeightsToApply fine ordinals", newWeightsToApply.fineOrdinals);
            Camellia::print("newWeightsToApply coarse ordinals", newWeightsToApply.coarseOrdinals);
            cout << "newWeightsToApply weights:\n" << newWeightsToApply.weights;
          }
          
          // compose the new weights with existing weights for this subcell
          composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(prevWeights, newWeightsToApply);

          // DEBUGGING
          if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
          {
            cout << "subcellInfo:\n" << subcellInfo << endl;
            cout << "subcellConstraint:\n" << subcellConstraint << endl;

            cout << "ancestralToConstrainingPermutation: " << ancestralToConstrainingPermutation << endl;

            cout << "prevWeights:\n";
            Camellia::print("prevWeights fine ordinals", prevWeights.fineOrdinals);
            Camellia::print("prevWeights coarse ordinals", prevWeights.coarseOrdinals);
            cout << "prevWeights weights:\n" << prevWeights.weights;

            cout << "newWeightsToApply:\n";
            Camellia::print("newWeightsToApply fine ordinals", newWeightsToApply.fineOrdinals);
            Camellia::print("newWeightsToApply coarse ordinals", newWeightsToApply.coarseOrdinals);
            cout << "newWeightsToApply weights:\n" << newWeightsToApply.weights;

            cout << "composedWeights:\n";
            Camellia::print("composedWeights fine ordinals", composedWeights.fineOrdinals);
            Camellia::print("composedWeights coarse ordinals", composedWeights.coarseOrdinals);
            cout << "composedWeights weights:\n" << composedWeights.weights;
          }
        }

        // populate the containers for the (d-1)-dimensional constituents of the constraining subcell
        if (subcellConstraint.dimension >= minimumConstraintDimension + 1)
        {
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          CellTopoPtr constrainingSideTopo = constrainingCellTopo->getSide(subcellConstraint.sideOrdinal);
          
          int d1 = subcellConstraint.dimension-1;
          unsigned sscCount = constrainingTopo->getSubcellCount(d1);
          for (unsigned ssubcord=0; ssubcord<sscCount; ssubcord++)
          {
            // 5/28/15: the commented-out code below belongs to an effort to guarantee that fine and coarse are reconciled only using the
            //          full intersection of their domains.  This, however, does not quite suffice, at least in the present approach.  We can
            //          have a fine and a coarse domain that intersect in an edge, and then one of the edge's vertices is constrained by a face
            //          which intersects the original fine domain in a full edge.  So we need more than a "local" guarantee.
//            unsigned ssubcordInSide = CamelliaCellTools::subcellOrdinalMap(constrainingSideTopo, subcellConstraint.dimension, subcellConstraint.subcellOrdinal, d1, ssubcord);;
//            if (cellConstraints.sideSubcellConstraintEnforcedBySuper[subcellInfo.sideOrdinal][d1][ssubcordInSide])
//            {
//              // then the containing subcells have already taken care of the sub-subcell
//              continue;
//            }
            unsigned ssubcordInCell = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, subcellConstraint.dimension, subcellOrdinalInConstrainingCell, d1, ssubcord);
            IndexType ssEntityIndex = constrainingCell->entityIndex(d1, ssubcordInCell);
            // DEBUGGING:
            //            if ((d1==0) && (ssEntityIndex==12)) {
            //              cout << "Adding vertex 12 to the list.\n";
            //            }
            
            AnnotatedEntity subsubcellConstraint;
            subsubcellConstraint.cellID = subcellConstraint.cellID;
            subsubcellConstraint.sideOrdinal = subcellConstraint.sideOrdinal;
            subsubcellConstraint.dimension = d1;
            
            CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
            subsubcellConstraint.subcellOrdinal = CamelliaCellTools::subcellReverseOrdinalMap(constrainingCellTopo, sideDim, subsubcellConstraint.sideOrdinal,
                                                                                              d1, ssubcordInCell);
            
            //          if ((cellID==21) && (sideOrdinal==2) && (d1==0) && (subcellConstraint.cellID==6) && ((ssubcordInCell==3) || (ssubcordInCell==0))) {
            //            cout << "About to compute composed weights for sub subcell.\n";
            //          }
            
            SubBasisReconciliationWeights composedWeightsForSubSubcell = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, d1,
                                                                                                                      subsubcellConstraint.subcellOrdinal, true);
            // DEBUGGING
            if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
            {
              cout << "subsubcellConstraint: " << subsubcellConstraint << endl;
              Camellia::print("composedWeightsForSubSubcell.coarseOrdinals", composedWeightsForSubSubcell.coarseOrdinals);
              Camellia::print("composedWeightsForSubSubcell.fineOrdinals", composedWeightsForSubSubcell.fineOrdinals);
              cout << "composed weights for subSubSubcell:\n" << composedWeightsForSubSubcell.weights;
            }
            
            if (composedWeightsForSubSubcell.weights.size() > 0)
            {
              appliedWeights[d1][ssEntityIndex].push_back(make_pair(subsubcellConstraint, composedWeightsForSubSubcell));
            }
          }
        }

        // add sub-basis map for dofs interior to the constraining subcell
        // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
        SubBasisReconciliationWeights subcellInteriorWeights = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, subcellConstraint.dimension,
            subcellConstraint.subcellOrdinal, false);

        if ((subcellInteriorWeights.coarseOrdinals.size() > 0) && (subcellInteriorWeights.fineOrdinals.size() > 0))
        {
          CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
          OwnershipInfo ownershipInfo = constrainingCellConstraints.owningCellIDForSubcell[subcellConstraint.dimension][subcellOrdinalInConstrainingCell];
          CellConstraints owningCellConstraints = getCellConstraints(ownershipInfo.cellID);
          SubCellDofIndexInfo owningCellDofIndexInfo = getOwnedGlobalDofIndices(ownershipInfo.cellID, owningCellConstraints);
          unsigned owningSubcellOrdinal = _meshTopology->getCell(ownershipInfo.cellID)->findSubcellOrdinal(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo[ownershipInfo.dimension][owningSubcellOrdinal][var->ID()];

          // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
          const vector<int>* constrainingBasisOrdinalsForSubcell = &constrainingBasis->dofOrdinalsForSubcell(subcellConstraint.dimension, subcellConstraint.subcellOrdinal);
//          vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
//          set<int> constrainingBasisOrdinalsForSubcell = constrainingBasis->dofOrdinalsForSubcell(subcellConstraint.dimension, subcellConstraint.subcellOrdinal);
//          vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
          vector<GlobalIndexType> globalDofOrdinals;
          for (int i=0; i<constrainingBasisOrdinalsForSubcell->size(); i++)
          {
            int constrainingBasisOrdinal = (*constrainingBasisOrdinalsForSubcell)[i];
            if (subcellInteriorWeights.coarseOrdinals.find(constrainingBasisOrdinal) != subcellInteriorWeights.coarseOrdinals.end())
            {
              globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
              // DEBUGGING:
              if ((globalDofOrdinalsForSubcell[i]==DEBUG_GLOBAL_DOF) && (cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL))
              {
                cout << "globalDofOrdinalsForSubcell includes global dof ordinal " << DEBUG_GLOBAL_DOF << ".\n";
                auto fineOrdinalIt = subcellInteriorWeights.fineOrdinals.begin();
                for (int fineWeightOrdinal=0; fineWeightOrdinal<subcellInteriorWeights.fineOrdinals.size(); fineWeightOrdinal++,
                     fineOrdinalIt++)
                {
                  cout << "fine ordinal " << *fineOrdinalIt << " weight for global dof " << DEBUG_GLOBAL_DOF << ": " << subcellInteriorWeights.weights(fineWeightOrdinal,i) << endl;
                }
              }
            }
          }

          if (subcellInteriorWeights.coarseOrdinals.size() != globalDofOrdinals.size())
          {
            cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
            Camellia::print("coarseOrdinals", subcellInteriorWeights.coarseOrdinals);
            Camellia::print("globalDofOrdinals", globalDofOrdinals);
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
          }

          // DEBUGGING:
          if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
          {
            set<unsigned> globalDofOrdinalsSet(globalDofOrdinals.begin(),globalDofOrdinals.end());
            IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);

            cout << "Determined constraints imposed by ";
            cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << constrainingSubcellEntityIndex;
            cout << " (owned by " << CamelliaCellTools::entityTypeString(ownershipInfo.dimension) << " " << ownershipInfo.owningSubcellEntityIndex << ")\n";

            ostringstream basisOrdinalsString, globalOrdinalsString;
            basisOrdinalsString << "basisDofOrdinals mapped on cell " << DEBUG_CELL_ID;
            basisOrdinalsString << ", sideOrdinal " << DEBUG_SIDE_ORDINAL;

            globalOrdinalsString << "globalDofOrdinals mapped on cell " << DEBUG_CELL_ID;
            globalOrdinalsString << ", sideOrdinal " << DEBUG_SIDE_ORDINAL;

            Camellia::print(basisOrdinalsString.str(),subcellInteriorWeights.fineOrdinals);
            Camellia::print(globalOrdinalsString.str(), globalDofOrdinalsSet);
            cout << "weights:\n" << subcellInteriorWeights.weights;
          }

          subBasisMap.weights = subcellInteriorWeights.weights;
          subBasisMap.globalDofOrdinals = globalDofOrdinals;
          subBasisMap.basisDofOrdinals = subcellInteriorWeights.fineOrdinals;

          subBasisMaps.push_back(subBasisMap);
        }

      }
    }

    appliedWeightsGreatestEntryDimension = -1;
    for (int d=sideDim; d >= minimumConstraintDimension; d--)
    {
      if (appliedWeights[d].size() > 0)
      {
        appliedWeightsGreatestEntryDimension = d;
        break;
      }
    }
  } // (appliedWeightsGreatestEntryDimension >= 0)

  // now, we collect the local basis coefficients corresponding to each global ordinal
  // likely there is a more efficient way to do this, but for now this is our approach
  map< GlobalIndexType, map<int, double> > weightsForGlobalOrdinal;

  map< int, set<GlobalIndexType> > globalOrdinalsForFineOrdinal;

  for (vector<SubBasisMapInfo>::iterator subBasisIt = subBasisMaps.begin(); subBasisIt != subBasisMaps.end(); subBasisIt++)
  {
    subBasisMap = *subBasisIt;
    vector<GlobalIndexType> globalDofOrdinals = subBasisMap.globalDofOrdinals;
    set<int> basisDofOrdinals = subBasisMap.basisDofOrdinals;
    vector<int> basisDofOrdinalsVector(basisDofOrdinals.begin(),basisDofOrdinals.end());
    // weights are fine x coarse
    for (int j=0; j<subBasisMap.weights.dimension(1); j++)
    {
      GlobalIndexType globalDofOrdinal = globalDofOrdinals[j];
      map<int, double> fineOrdinalCoefficientsThusFar = weightsForGlobalOrdinal[globalDofOrdinal];
      for (int i=0; i<subBasisMap.weights.dimension(0); i++)
      {
        int fineOrdinal = basisDofOrdinalsVector[i];
        double coefficient = subBasisMap.weights(i,j);
        if (coefficient != 0)
        {
          if (fineOrdinalCoefficientsThusFar.find(fineOrdinal) != fineOrdinalCoefficientsThusFar.end())
          {
            double tol = 1e-14;
            double previousCoefficient = fineOrdinalCoefficientsThusFar[fineOrdinal];
            if (abs(previousCoefficient-coefficient) > tol)
            {
              cout  << "ERROR: incompatible entries for fine ordinal " << fineOrdinal << " in representation of global ordinal " << globalDofOrdinal << endl;
              cout << "previousCoefficient = " << previousCoefficient << endl;
              cout << "coefficient = " << coefficient << endl;
              cout << "diff = " << abs(previousCoefficient - coefficient) << endl;
              cout << "Encountered the incompatible entry while processing variable " << var->name() << " on cell " << cellID << ", side " << sideOrdinal << endl;
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal Error: incompatible entries for fine ordinal " );
            }
          }
          fineOrdinalCoefficientsThusFar[fineOrdinal] = coefficient;
          globalOrdinalsForFineOrdinal[fineOrdinal].insert(globalDofOrdinal);
        }
      }
      weightsForGlobalOrdinal[globalDofOrdinal] = fineOrdinalCoefficientsThusFar;
    }
  }

  // partition global ordinals according to which fine ordinals they interact with -- this is definitely not super-efficient
  set<GlobalIndexType> partitionedGlobalDofOrdinals;
  vector< set<GlobalIndexType> > globalDofOrdinalPartitions;
  vector< set<int> > fineOrdinalsForPartition;

  for (map< GlobalIndexType, map<int, double> >::iterator globalWeightsIt = weightsForGlobalOrdinal.begin();
       globalWeightsIt != weightsForGlobalOrdinal.end(); globalWeightsIt++)
  {
    GlobalIndexType globalOrdinal = globalWeightsIt->first;
    if (partitionedGlobalDofOrdinals.find(globalOrdinal) != partitionedGlobalDofOrdinals.end()) continue;

    set<GlobalIndexType> partition;
    partition.insert(globalOrdinal);

    set<int> fineOrdinals;

    set<GlobalIndexType> globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed;

    map<int, double> fineCoefficients = globalWeightsIt->second;
    for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
    {
      int fineOrdinal = coefficientIt->first;
      fineOrdinals.insert(fineOrdinal);
      set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
      partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
    }

    globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);

    while (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.size() != partition.size())
    {
      for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
      {
        GlobalIndexType globalOrdinal = *globalOrdIt;
        if (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.find(globalOrdinal) !=
            globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.end()) continue;
        map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
        for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
        {
          int fineOrdinal = coefficientIt->first;
          fineOrdinals.insert(fineOrdinal);
          set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
          partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
        }

        globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
      }
    }

    for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
    {
      GlobalIndexType globalOrdinal = *globalOrdIt;
      partitionedGlobalDofOrdinals.insert(globalOrdinal);
    }
    globalDofOrdinalPartitions.push_back(partition);
    fineOrdinalsForPartition.push_back(fineOrdinals);
  }

  BasisMap varSideMap;
  for (int i=0; i<globalDofOrdinalPartitions.size(); i++)
  {
    set<GlobalIndexType> partition = globalDofOrdinalPartitions[i];
    set<int> fineOrdinals = fineOrdinalsForPartition[i];
    vector<int> fineOrdinalsVector(fineOrdinals.begin(), fineOrdinals.end());
    map<int,int> fineOrdinalRowLookup;
    for (int i=0; i<fineOrdinalsVector.size(); i++)
    {
      fineOrdinalRowLookup[fineOrdinalsVector[i]] = i;
    }
    Intrepid::FieldContainer<double> weights(fineOrdinals.size(),partition.size());
    int col=0;
    for (set<GlobalIndexType>::iterator globalDofIt=partition.begin(); globalDofIt != partition.end(); globalDofIt++)
    {
      GlobalIndexType globalOrdinal = *globalDofIt;
      map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
      for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
      {
        int fineOrdinal = coefficientIt->first;
        double coefficient = coefficientIt->second;
        int row = fineOrdinalRowLookup[fineOrdinal];
        weights(row,col) = coefficient;
      }
      col++;
    }
    vector<GlobalIndexType> globalOrdinals(partition.begin(),partition.end());
    SubBasisDofMapperPtr subBasisMap = SubBasisDofMapper::subBasisDofMapper(fineOrdinals, globalOrdinals, weights);
    varSideMap.push_back(subBasisMap);

    // DEBUGGING
//    if ((cellID==4) && (sideOrdinal==3) && (var->ID() ==0)) {
//      cout << "Adding subBasisDofMapper.  Details:\n";
//      Camellia::print("fineOrdinals", fineOrdinals);
//      Camellia::print("globalOrdinals", globalOrdinals);
//      cout << "weights:\n" << weights;
//    }
  }

  return varSideMap;
}

CellConstraints GDAMinimumRule::getCellConstraints(GlobalIndexType cellID)
{
  if (_constraintsCache.find(cellID) == _constraintsCache.end())
  {

//    cout << "Getting cell constraints for cellID " << cellID << endl;

    typedef pair< IndexType, unsigned > CellPair;

    CellPtr cell = _meshTopology->getCell(cellID);
    DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
    CellTopoPtr topo = cell->topology();
    unsigned spaceDim = topo->getDimension();
    unsigned sideDim = spaceDim - 1;

    vector< vector< bool > > processedSubcells(spaceDim+1); // we process dimensions from high to low -- since we deal here with the volume basis case, we initialize this to spaceDim + 1
    vector< vector< AnnotatedEntity > > constrainingSubcellInfo(spaceDim + 1);

    AnnotatedEntity emptyConstraintInfo;
    emptyConstraintInfo.cellID = -1;

    for (int d=0; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      processedSubcells[d] = vector<bool>(scCount, false);
      constrainingSubcellInfo[d] = vector< AnnotatedEntity >(scCount, emptyConstraintInfo);
    }

    for (int d=sideDim; d >= 0; d--)
    {
      int scCount = topo->getSubcellCount(d);
      for (int subcord=0; subcord < scCount; subcord++)   // subcell ordinals in cell
      {
        if (! processedSubcells[d][subcord])   // i.e. we don't yet know the constraining entity for this subcell
        {

          IndexType entityIndex = cell->entityIndex(d, subcord);

//          cout << "entity of dimension " << d << " with entity index " << entityIndex << ": " << endl;
//          _meshTopology->printEntityVertices(d, entityIndex);
          pair<IndexType,unsigned> constrainingEntity = _meshTopology->getConstrainingEntity(d, entityIndex); // the constraining entity of the same dimension as this entity

          unsigned constrainingEntityDimension = constrainingEntity.second;
          IndexType constrainingEntityIndex = constrainingEntity.first;

//          cout << "is constrained by entity of dimension " << constrainingEntity.second << "  with entity index " << constrainingEntity.first << ": " << endl;
//          _meshTopology->printEntityVertices(constrainingEntity.second, constrainingEntity.first);

          set< CellPair > cellsForSubcell = _meshTopology->getCellsContainingEntity(constrainingEntityDimension, constrainingEntityIndex);

          TEUCHOS_TEST_FOR_EXCEPTION(cellsForSubcell.size() == 0, std::invalid_argument, "no cells found that match constraining entity");
          
          // for now, we just use the first component of H1Order; this should be OK so long as refinements are isotropic,
          // but if / when we support anisotropic refinements, we'll want to revisit this (probably we need a notion of the
          // appropriate order along the *interface* in question)
          int leastH1Order = INT_MAX;
          set< CellPair > cellsWithLeastH1Order;
          for (set< CellPair >::iterator cellForSubcellIt = cellsForSubcell.begin(); cellForSubcellIt != cellsForSubcell.end(); cellForSubcellIt++)
          {
            IndexType subcellCellID = cellForSubcellIt->first;
            if (_cellH1Orders[subcellCellID][0] == leastH1Order)
            {
              cellsWithLeastH1Order.insert(*cellForSubcellIt);
            }
            else if (_cellH1Orders[subcellCellID][0] < leastH1Order)
            {
              cellsWithLeastH1Order.clear();
              leastH1Order = _cellH1Orders[subcellCellID][0];
              cellsWithLeastH1Order.insert(*cellForSubcellIt);
            }
            if (cellsWithLeastH1Order.size() == 0)
            {
              cout << "ERROR: No cells found for constraining subside entity.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No cells found for constraining subside entity.");
            }
          }
          
          CellPair constrainingCellPair = *cellsWithLeastH1Order.begin(); // first one will have the earliest cell ID, given the sorting of set/pair.
          
          // if one of the cellsWithLeastH1Order is active, we prefer that...
//          for (CellPair cellPair : cellsWithLeastH1Order)
//          {
//            if (_meshTopology->getActiveCellIndices().find(cellPair.first) != _meshTopology->getActiveCellIndices().end())
//            {
//              // active cell: we prefer this one
//              constrainingCellPair = cellPair;
//              break;
//            }
//          }
          
          constrainingSubcellInfo[d][subcord].cellID = constrainingCellPair.first;

          unsigned constrainingCellID = constrainingCellPair.first;
          unsigned constrainingSideOrdinal = constrainingCellPair.second;

          CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          if (constrainingCellTopo->getTensorialDegree() > 0)
          {
            // presumption is space-time.  In this case, we prefer spatial sides when we have a choice
            // (because some trace/flux variables are only defined on spatial sides)
            if (constrainingEntityDimension < sideDim) // if constrainingEntityDimension == sideDim, then the constraining entity is a side, so there isn't another side on constrainingCell that contains the constraining entity
            {
              if (! constrainingCellTopo->sideIsSpatial(constrainingSideOrdinal))
              {
                for (unsigned newSideOrdinal=0; newSideOrdinal<constrainingCellTopo->getSideCount(); newSideOrdinal++)
                {
                  if (constrainingCellTopo->sideIsSpatial(newSideOrdinal))
                  {
                    // spatial side.  Does it contain the constraining entity?
                    unsigned subcellOrdinalInSide = constrainingCell->findSubcellOrdinalInSide(constrainingEntityDimension, constrainingEntityIndex, newSideOrdinal);
                    if (subcellOrdinalInSide != -1)
                    {
                      constrainingSideOrdinal = newSideOrdinal;
                      break;
                    }
                  }
                }
              }
            }
          }

          constrainingSubcellInfo[d][subcord].sideOrdinal = constrainingSideOrdinal;
          unsigned subcellOrdinalInConstrainingSide = constrainingCell->findSubcellOrdinalInSide(constrainingEntityDimension, constrainingEntityIndex, constrainingSideOrdinal);

          constrainingSubcellInfo[d][subcord].subcellOrdinal = subcellOrdinalInConstrainingSide;
          constrainingSubcellInfo[d][subcord].dimension = constrainingEntityDimension;

          if (constrainingEntityDimension < d)
          {
            cout << "Internal error: constrainingEntityDimension < entity dimension.  This should not happen!\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constrainingEntityDimension < entity dimension.  This should not happen!");
          }

          if ((d==0) && (constrainingEntityDimension==0))
          {
            if (constrainingEntityIndex != entityIndex)
            {
              cout << "Internal error: one vertex constrained by another.  This should not happen!\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "one vertex constrained by another.  Shouldn't happen!");
            }
          }

//          cout << "constraining subcell of dimension " << d << ", ordinal " << subcord << " with:\n";
//          cout << "  cellID " << constrainingSubcellInfo[d][subcord].cellID;
//          cout << ", sideOrdinal " << constrainingSubcellInfo[d][subcord].sideOrdinal;
//          cout << ", subcellOrdinalInSide " << constrainingSubcellInfo[d][subcord].subcellOrdinalInSide;
//          cout << ", dimension " << constrainingSubcellInfo[d][subcord].dimension << endl;
        }
      }
    }

    // fill in constraining subcell info for the volume (namely, it constrains itself):
    constrainingSubcellInfo[spaceDim][0].cellID = cellID;
    constrainingSubcellInfo[spaceDim][0].subcellOrdinal = -1;
    constrainingSubcellInfo[spaceDim][0].sideOrdinal = -1;
    constrainingSubcellInfo[spaceDim][0].dimension = spaceDim;

    CellConstraints cellConstraints;
    cellConstraints.subcellConstraints = constrainingSubcellInfo;

    // determine subcell ownership from the perspective of the cell
    cellConstraints.owningCellIDForSubcell = vector< vector< OwnershipInfo > >(spaceDim+1);
    for (int d=0; d<spaceDim; d++)
    {
      vector<IndexType> scIndices = cell->getEntityIndices(d);
      unsigned scCount = scIndices.size();
      cellConstraints.owningCellIDForSubcell[d] = vector< OwnershipInfo >(scCount);
      for (int scOrdinal = 0; scOrdinal < scCount; scOrdinal++)
      {
        IndexType entityIndex = scIndices[scOrdinal];
        unsigned constrainingDimension = constrainingSubcellInfo[d][scOrdinal].dimension;
        IndexType constrainingEntityIndex;
        if (d==constrainingDimension)
        {
          constrainingEntityIndex = _meshTopology->getConstrainingEntity(d, entityIndex).first;
        }
        else
        {
          CellPtr constrainingCell = _meshTopology->getCell(constrainingSubcellInfo[d][scOrdinal].cellID);
          unsigned constrainingSubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim, constrainingSubcellInfo[d][scOrdinal].sideOrdinal,
              constrainingDimension, constrainingSubcellInfo[d][scOrdinal].subcellOrdinal);
          constrainingEntityIndex = constrainingCell->entityIndex(constrainingDimension, constrainingSubcellOrdinalInCell);
        }
        pair<GlobalIndexType,GlobalIndexType> owningCellInfo = _meshTopology->owningCellIndexForConstrainingEntity(constrainingDimension, constrainingEntityIndex);
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].cellID = owningCellInfo.first;
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].owningSubcellEntityIndex = owningCellInfo.second; // the constrained entity in owning cell (which is a same-dimensional descendant of the constraining entity)
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].dimension = constrainingDimension;
      }
    }
    // cell owns (and constrains) its interior:
    cellConstraints.owningCellIDForSubcell[spaceDim] = vector< OwnershipInfo >(1);
    cellConstraints.owningCellIDForSubcell[spaceDim][0].cellID = cellID;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].owningSubcellEntityIndex = cellID;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].dimension = spaceDim;

    /* 5-28-15:
     
     Something like the idea (or at least the goal) of sideSubcellConstraintEnforcedBySuper is good, but
     I need to get mathematically and geometrically precise on what the domain of enforcement should be,
     and the commented-out code below does not suffice to get us to uniqueness of that domain.
     
     It *might* be something like: take the AnnotatedEntity that constrains a given subcell; local degrees of freedom
     belonging to that subcell should be eliminated from any "enforcement path" that does not begin with that AnnotatedEntity.
     
     */
    /* Finally, build sideSubcellConstraintEnforcedBySuper.
     The point of this is to guarantee that during getBasisMap, as we split off lower-dimensioned subcells,
     we never enforce constraints on anything less than the intersection of the fine and coarse domains.  The
     rule is, if the side that constrains the subcell is the same as the sub-subcell, then the sub-subcell
     constraint is taken care of, and we can mark "true" for that sub-subcell.
     
     */
//    int sideCount = topo->getSideCount();
//    cellConstraints.sideSubcellConstraintEnforcedBySuper = vector< vector< vector<bool> > >(sideCount);
//    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
//    {
//      cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal] = vector< vector<bool> >(sideDim+1);
//      CellTopoPtr sideTopo = topo->getSide(sideOrdinal);
//      
//      // initialize all entries to false:
//      for (int scdim=0; scdim<=sideDim; scdim++)
//      {
//        int subcellCount = sideTopo->getSubcellCount(scdim);
//        cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal][scdim] = vector<bool>(subcellCount,false);
//      }
//      
//      for (int scdim=sideDim; scdim>0; scdim--)
//      {
//        int subcellCount = sideTopo->getSubcellCount(scdim);
//        for (int scord=0; scord<subcellCount; scord++) // ordinal in side
//        {
////          { // DEBUGGING:
////            if ((sideOrdinal==5) && (cellID==18) && (scdim==1) && (scord==0))
////            {
////              cout << "(sideOrdinal==5) && (cellID==18) && (scdim==1) && (scord==0).\n"; //breakpoint here
////            }
////          }
//          
//          int scordInCell = CamelliaCellTools::subcellOrdinalMap(topo, sideDim, sideOrdinal, scdim, scord);
//          GlobalIndexType constrainingCellID = cellConstraints.subcellConstraints[scdim][scordInCell].cellID;
//          unsigned constrainingSideOrdinal = cellConstraints.subcellConstraints[scdim][scordInCell].sideOrdinal;
//          CellTopoPtr subcell = sideTopo->getSubcell(scdim, scord);
//          for (int sscdim=scdim-1; sscdim >= 0; sscdim--)
//          {
//            int subsubcellCount = subcell->getSubcellCount(sscdim);
//            for (int sscord=0; sscord<subsubcellCount; sscord++) // ordinal in subcell
//            {
//              int sscordInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, scdim, scord, sscdim, sscord);
//              int sscordInCell = CamelliaCellTools::subcellOrdinalMap(topo, sideDim, sideOrdinal, sscdim, sscordInSide);
//              GlobalIndexType sscConstrainingCellID = cellConstraints.subcellConstraints[sscdim][sscordInCell].cellID;
//              GlobalIndexType sscConstrainingSideOrdinal = cellConstraints.subcellConstraints[sscdim][sscordInCell].sideOrdinal;
//              if ((sscConstrainingCellID == constrainingCellID) && (constrainingSideOrdinal == sscConstrainingSideOrdinal))
//              {
//                cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal][sscdim][sscordInSide] = true;
//              }
//            }
//          }
//        }
//      }
//    }
    
    _constraintsCache[cellID] = cellConstraints;

//    if (cellID==4) { // DEBUGGING
//      printConstraintInfo(cellID);
//    }
  }

  return _constraintsCache[cellID];
}

// ! returns the permutation that goes from the ancestor of the indicated cell's view of the subcell to the constraining cell's view.
unsigned GDAMinimumRule::getConstraintPermutation(GlobalIndexType cellID, unsigned subcdim, unsigned subcord)
{
  CellPtr cell = _meshTopology->getCell(cellID);
  CellPtr ancestralCell = cell->ancestralCellForSubcell(subcdim, subcord, _meshTopology);

  pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(subcdim, subcord, _meshTopology);

  unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
  unsigned ancestralSubcellDimension = ancestralSubcell.second;
  unsigned ancestralPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);

  CellConstraints cellConstraints = getCellConstraints(cellID);
  GlobalIndexType constrainingCellID = cellConstraints.subcellConstraints[subcdim][subcord].cellID;
  GlobalIndexType constrainingSubcellOrdinal = cellConstraints.subcellConstraints[subcdim][subcord].subcellOrdinal;
  GlobalIndexType constrainingSubcellDimension = cellConstraints.subcellConstraints[subcdim][subcord].dimension;

  CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
  unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(constrainingSubcellOrdinal, constrainingSubcellDimension);

  // ancestral and constrainingCell permutations are from canonical.  We want to go from ancestor to constraining:
  CellTopoPtr ancestralSubcellTopo = ancestralCell->topology()->getSubcell(ancestralSubcellDimension, ancestralSubcellOrdinal);
  unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(ancestralSubcellTopo, ancestralPermutation);
  unsigned ancestralToConstraint = CamelliaCellTools::permutationComposition(ancestralSubcellTopo, ancestralPermutationInverse, constrainingCellPermutation);

  return ancestralToConstraint;
}

// ! returns the permutation that goes from the ancestor of the indicated side's view of the subcell to the constraining cell's view.
unsigned GDAMinimumRule::getConstraintPermutation(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcdim, unsigned subcord)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Method not implemented");
}

typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension

set<GlobalIndexType> GDAMinimumRule::getFittableGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints, int sideOrdinal,
                                                                 int varID)
{
  pair<GlobalIndexType,pair<int,unsigned>> key = {cellID,{varID,sideOrdinal}};
  if (_fittableGlobalIndicesCache.find(key) != _fittableGlobalIndicesCache.end())
  {
    return _fittableGlobalIndicesCache[key];
  }
  
  // returns the global dof indices for basis functions which have support on the given side.  This is determined by taking the union of the global dof indices defined on all the constraining sides for the given side (the constraining sides are by definition unconstrained).
  SubCellDofIndexInfo dofIndexInfo = getGlobalDofIndices(cellID, constraints);
  CellPtr cell = _meshTopology->getCell(cellID);
  int sideDim = _meshTopology->getDimension() - 1;

  GlobalIndexType constrainingCellID = constraints.subcellConstraints[sideDim][sideOrdinal].cellID;
  unsigned constrainingCellSideOrdinal = constraints.subcellConstraints[sideDim][sideOrdinal].sideOrdinal;

  CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);

  CellConstraints constrainingCellConstraints = getCellConstraints(constrainingCellID);

  SubCellDofIndexInfo constrainingCellDofIndexInfo = getGlobalDofIndices(constrainingCellID, constrainingCellConstraints);

  set<GlobalIndexType> fittableDofIndices;

  /*
   Exactly *which* of the constraining side's dofs to include is a little tricky.  We want the ones interior to the side for sure (since these are
   not further constrained), and definitely want to exclude e.g. edge dofs that are further constrained by other edges.  But then it turns out we
   also want to exclude the vertices that are included in those constrained edges.  The general rule seems to be: if a subcell is excluded, then
   all its constituent subcells should also be excluded.
   */

  // iterate through all the subcells of the constraining side, collecting dof indices
  CellTopoPtr constrainingCellTopo = constrainingCell->topology();
  CellTopoPtr constrainingSideTopo = constrainingCellTopo->getSubcell(sideDim, constrainingCellSideOrdinal);
  set<unsigned> constrainingCellConstrainedNodes; // vertices in the constraining cell that belong to subcell entities that are constrained by other cells
  for (int d=sideDim; d>=0; d--)
  {
    int scCount = constrainingSideTopo->getSubcellCount(d);
    for (int scOrdinal=0; scOrdinal<scCount; scOrdinal++)
    {
      int scordInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, sideDim, constrainingCellSideOrdinal, d, scOrdinal);
      set<unsigned> scNodesInConstrainingCell;
      bool nodeNotKnownToBeConstrainedFound = false;
      if (d==0)
      {
        scNodesInConstrainingCell.insert(scordInConstrainingCell);
        if (constrainingCellConstrainedNodes.find(scordInConstrainingCell) == constrainingCellConstrainedNodes.end())
        {
          nodeNotKnownToBeConstrainedFound = true;
        }
      }
      else
      {
        int nodeCount = constrainingCellTopo->getNodeCount(d, scordInConstrainingCell);
        for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++)
        {
          unsigned nodeInConstrainingCell = constrainingCellTopo->getNodeMap(d, scordInConstrainingCell, nodeOrdinal);
          scNodesInConstrainingCell.insert(nodeInConstrainingCell);
          if (constrainingCellConstrainedNodes.find(nodeInConstrainingCell) == constrainingCellConstrainedNodes.end())
          {
            nodeNotKnownToBeConstrainedFound = true;
          }
        }
      }

      if (!nodeNotKnownToBeConstrainedFound) continue; // all nodes constrained, so we can skip further processing

      IndexType subcellIndex = constrainingCell->entityIndex(d, scordInConstrainingCell);
      pair<IndexType,unsigned> constrainingSubcell = _meshTopology->getConstrainingEntity(d, subcellIndex);
      if ((constrainingSubcell.second == d) && (constrainingSubcell.first == subcellIndex))   // i.e. the subcell is not further constrained.
      {
        map<int, vector<GlobalIndexType> > dofIndicesInConstrainingSide = constrainingCellDofIndexInfo[d][scordInConstrainingCell];
        for (auto dofIndexEntry : dofIndicesInConstrainingSide)
        {
          int entryVarID = dofIndexEntry.first;
          if ((varID == -1) || (varID == entryVarID))
            fittableDofIndices.insert(dofIndexEntry.second.begin(), dofIndexEntry.second.end());
        }
      }
      else
      {
        constrainingCellConstrainedNodes.insert(scNodesInConstrainingCell.begin(),scNodesInConstrainingCell.end());
      }
    }
  }
  _fittableGlobalIndicesCache[key] = fittableDofIndices;
  return fittableDofIndices;
}

SubCellDofIndexInfo GDAMinimumRule::getOwnedGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints)
{
  // there's a lot of redundancy between this method and the dof-counting bit of rebuild lookups.  May be worth factoring that out.

  if (_ownedGlobalDofIndicesCache.find(cellID) != _ownedGlobalDofIndicesCache.end())
  {
    return _ownedGlobalDofIndicesCache[cellID];
  }

  int spaceDim = _meshTopology->getDimension();
  int sideDim = spaceDim - 1;

  SubCellDofIndexInfo scInfo(spaceDim+1);

  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;

  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory->trialVars();

//  cout << "Owned global dof indices for cell " << cellID << endl;

  GlobalIndexType globalDofIndex = _globalCellDofOffsets[cellID]; // this cell's first globalDofIndex

  for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++)
  {
    map< pair<unsigned,IndexType>, pair<unsigned, unsigned> > entitiesClaimedForVariable; // maps from the constraining entity claimed to the (d, scord) entry that claimed it.
    VarPtr var = varIt->second;
    unsigned scordForBasis;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
//    cout << " var " << var->name() << ":\n";

    for (int d=0; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        OwnershipInfo ownershipInfo = constraints.owningCellIDForSubcell[d][scord];
        if (ownershipInfo.cellID == cellID)   // owned by this cell: count all the constraining dofs as entries for this cell
        {
          GlobalIndexType constrainingCellID = constraints.subcellConstraints[d][scord].cellID;
          unsigned constrainingDimension = constraints.subcellConstraints[d][scord].dimension;
          DofOrderingPtr trialOrdering = _elementTypeForCell[constrainingCellID]->trialOrderPtr;
          BasisPtr basis; // the constraining basis for the subcell
          if (varHasSupportOnVolume)
          {
            if (d==spaceDim)
            {
              // then there is only one subcell ordinal (and there will be -1's in sideOrdinal and subcellOrdinalInSide....
              scordForBasis = 0;
            }
            else
            {
              scordForBasis = CamelliaCellTools::subcellOrdinalMap(_meshTopology->getCell(constrainingCellID)->topology(), sideDim,
                              constraints.subcellConstraints[d][scord].sideOrdinal,
                              constrainingDimension, constraints.subcellConstraints[d][scord].subcellOrdinal);
            }
            basis = trialOrdering->getBasis(var->ID());
          }
          else
          {
            if (d==spaceDim) continue; // side bases don't have any support on the interior of the cell...
            if (! trialOrdering->hasBasisEntry(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal) ) continue;
            scordForBasis = constraints.subcellConstraints[d][scord].subcellOrdinal; // the basis sees the side, so that's the view to use for subcell ordinal
            basis = trialOrdering->getBasis(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal);
          }
          int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
          if (minimumConstraintDimension > d) continue; // then we don't enforce (or own) anything for this subcell/basis combination

          pair<unsigned, IndexType> owningSubcellEntity = make_pair(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          if (entitiesClaimedForVariable.find(owningSubcellEntity) != entitiesClaimedForVariable.end())
          {
            // already processed this guy on this cell: just copy
            pair<unsigned,unsigned> previousConstrainedSubcell = entitiesClaimedForVariable[owningSubcellEntity];
            scInfo[d][scord][var->ID()] = scInfo[previousConstrainedSubcell.first][previousConstrainedSubcell.second][var->ID()];
            continue;
          }
          else
          {
            entitiesClaimedForVariable[owningSubcellEntity] = make_pair(d, scord);
          }

          int dofOrdinalCount = basis->dofOrdinalsForSubcell(constrainingDimension, scordForBasis).size();
          vector<GlobalIndexType> globalDofIndices;

//          cout << "   dim " << d << ", scord " << scord << ":";

          for (int i=0; i<dofOrdinalCount; i++)
          {
//            cout << " " << globalDofIndex;
            globalDofIndices.push_back(globalDofIndex++);
          }
//          cout << endl;
          if (scInfo.size() < d+1)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: scInfo vector not big enough");
          }
//          if (scInfo[d].find(scord) == scInfo[d].end()) {
//            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: scord not found");
//          }
//          if (scInfo[d][scord].find(var->ID())==scInfo[d][scord].end()) {
//            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: var ID not found");
//          }
          scInfo[d][scord][var->ID()] = globalDofIndices;
        }
      }
    }
  }
  _ownedGlobalDofIndicesCache[cellID] = scInfo;
  return scInfo;
}

void printDofIndexInfo(GlobalIndexType cellID, SubCellDofIndexInfo &dofIndexInfo)
{
  //typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension
  cout << "Dof Index info for cell ID " << cellID << ":\n";
  ostringstream varIDstream;
  for (int d=0; d<dofIndexInfo.size(); d++)
  {
    //    typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
    //    typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
    cout << "****** dimension " << d << " *******\n";
    SubCellOrdinalToMap scordMap = dofIndexInfo[d];
    for (SubCellOrdinalToMap::iterator scordMapIt = scordMap.begin(); scordMapIt != scordMap.end(); scordMapIt++)
    {
      cout << "  scord " << scordMapIt->first << ":\n";
      VarIDToDofIndices varMap = scordMapIt->second;
      for (VarIDToDofIndices::iterator varIt = varMap.begin(); varIt != varMap.end(); varIt++)
      {
        varIDstream.str("");
        varIDstream << "     var " << varIt->first << ", global dofs";
        if (varIt->second.size() > 0) Camellia::print(varIDstream.str(), varIt->second);
      }
    }
  }
}

SubCellDofIndexInfo& GDAMinimumRule::getGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints)
{
  if (_globalDofIndicesForCellCache.find(cellID) == _globalDofIndicesForCellCache.end())
  {
    
    /**************** ESTABLISH OWNERSHIP ****************/
    SubCellDofIndexInfo dofIndexInfo = getOwnedGlobalDofIndices(cellID, constraints);
    
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
    
    int spaceDim = topo->getDimension();
    
    // fill in the other global dof indices (the ones not owned by this cell):
    map< GlobalIndexType, SubCellDofIndexInfo > otherDofIndexInfoCache; // local lookup, to avoid a bunch of redundant calls to getOwnedGlobalDofIndices
    for (int d=0; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        if (dofIndexInfo[d].find(scord) == dofIndexInfo[d].end())   // this one not yet filled in
        {
          OwnershipInfo owningCellInfo = constraints.owningCellIDForSubcell[d][scord];
          GlobalIndexType owningCellID = owningCellInfo.cellID;
          CellConstraints owningConstraints = getCellConstraints(owningCellID);
          GlobalIndexType scEntityIndex = owningCellInfo.owningSubcellEntityIndex;
          CellPtr owningCell = _meshTopology->getCell(owningCellID);
          unsigned owningCellScord = owningCell->findSubcellOrdinal(owningCellInfo.dimension, scEntityIndex);
          if (otherDofIndexInfoCache.find(owningCellID) == otherDofIndexInfoCache.end())
          {
            otherDofIndexInfoCache[owningCellID] = getOwnedGlobalDofIndices(owningCellID, owningConstraints);
          }
          SubCellDofIndexInfo owningDofIndexInfo = otherDofIndexInfoCache[owningCellID];
          dofIndexInfo[d][scord] = owningDofIndexInfo[owningCellInfo.dimension][owningCellScord];
        }
      }
    }
    _globalDofIndicesForCellCache[cellID] = dofIndexInfo;
  }
  
  // DEBUGGING
//  printDofIndexInfo(cellID, dofIndexInfo);

  return _globalDofIndicesForCellCache[cellID];
}

set<GlobalIndexType> GDAMinimumRule::getGlobalDofIndicesForIntegralContribution(GlobalIndexType cellID, int sideOrdinal)   // assuming an integral is being done over the whole mesh skeleton, returns either an empty set or the global dof indices associated with the given side, depending on whether the cell "owns" the side for the purpose of such contributions.
{
  set<GlobalIndexType> indices;

  CellPtr cell = _meshTopology->getCell(cellID);
  bool ownsSide = cell->ownsSide(sideOrdinal, _meshTopology);

  if (ownsSide)
  {
    CellConstraints cellConstraints = getCellConstraints(cellID);
    SubCellDofIndexInfo dofIndexInfo = getGlobalDofIndices(cellID, cellConstraints);
    int spaceDim =  _meshTopology->getDimension();

    CellTopoPtr cellTopo = cell->topology();
    CellTopoPtr sideTopo = cellTopo->getSubcell(spaceDim-1, sideOrdinal);

    for (int d=0; d<spaceDim; d++)
    {
      int scCount = sideTopo->getSubcellCount(d);
      for (int scordSide=0; scordSide<scCount; scordSide++)
      {
        int scordCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, spaceDim-1, sideOrdinal, d, scordSide);
        map<int, vector<GlobalIndexType> > dofIndices = dofIndexInfo[d][scordCell];


        for (map<int, vector<GlobalIndexType> >::iterator dofIndicesIt = dofIndices.begin(); dofIndicesIt != dofIndices.end(); dofIndicesIt++)
        {
          indices.insert(dofIndicesIt->second.begin(), dofIndicesIt->second.end());
        }
      }
    }
  }

  return indices;
}

set<GlobalIndexType> GDAMinimumRule::getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal)
{
  CellConstraints constraints = getCellConstraints(cellID);
  
  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
  int spaceDim = topo->getDimension();
  
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory->trialVars();
  
  VarPtr var = trialVars[varID];
  
  set<GlobalIndexType> fittableIndexSet;
  if ((var->varType() == FLUX) || (var->varType() == TRACE))
  {
    return getFittableGlobalDofIndices(cellID, constraints, sideOrdinal, varID);
  }
  else if (sideOrdinal == -1)
  {
    SubCellDofIndexInfo* dofIndexInfo = &getGlobalDofIndices(cellID, constraints);
    fittableIndexSet.insert((*dofIndexInfo)[spaceDim][0][varID].begin(),(*dofIndexInfo)[spaceDim][0][varID].end());
    BasisPtr basis = trialOrdering->getBasis(varID);
    if (! functionSpaceIsDiscontinuous(basis->functionSpace()))
    {
      // then we also need to add fittable ordinals from the sides
      int sideCount = basis->domainTopology()->getSideCount();
      for (int domainSideOrdinal=0; domainSideOrdinal<sideCount; domainSideOrdinal++)
      {
        set<GlobalIndexType> fittableIndicesOnSide = getFittableGlobalDofIndices(cellID, constraints, domainSideOrdinal, varID);
        fittableIndexSet.insert(fittableIndicesOnSide.begin(),fittableIndicesOnSide.end());
      }
    }
  }
  else
  {
    // sideOrdinal != -1, and var is not a flux or trace
    return getFittableGlobalDofIndices(cellID, constraints, sideOrdinal, varID);
  }
  
  return fittableIndexSet;
}

vector<GlobalIndexType> GDAMinimumRule::globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID)
{
  map<int, VarPtr> trialVars = _varFactory->trialVars();
  VarPtr trialVar = trialVars[varID];
  
  EFunctionSpace fs = efsForSpace(trialVar->space());
  TEUCHOS_TEST_FOR_EXCEPTION(!functionSpaceIsDiscontinuous(fs), std::invalid_argument, "globalDofIndicesForFieldVariable() only supports discontinuous field variables right now");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVar->varType() != FIELD, std::invalid_argument, "globalDofIndicesForFieldVariable() requires a discontinuous field variable");
  
  CellConstraints constraints = getCellConstraints(cellID);
  SubCellDofIndexInfo dofIndexInfo = getOwnedGlobalDofIndices(cellID, constraints);
  
  int spaceDim = _mesh->getTopology()->getDimension();
  vector<GlobalIndexType> globalIndices(dofIndexInfo[spaceDim][0][trialVar->ID()].begin(),
                                        dofIndexInfo[spaceDim][0][trialVar->ID()].end());
  
//  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints, varID, VOLUME_INTERIOR_SIDE_ORDINAL);
//  
//  TEUCHOS_TEST_FOR_EXCEPTION(!dofMapper->isPermutation(), std::invalid_argument, "GDAMinimumRule only supports globalDofIndicesForFieldVariable() for discontinuous variables");
//  
//  map<int, GlobalIndexType> permutationMap = dofMapper->getPermutationMap();
//  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
//  BasisPtr volumeBasis = trialOrdering->getBasis(varID);
//  
//  const vector<int>* localDofIndices = &trialOrdering->getDofIndices(varID);
//  vector<GlobalIndexType> globalIndices;
//  for (int localDofIndex : *localDofIndices)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(permutationMap.find(localDofIndex) == permutationMap.end(), std::invalid_argument,
//                               "Error: permutation map does not contain localDofIndex");
//    globalIndices.push_back(permutationMap[localDofIndex]);
//  }
  return globalIndices;
}

LocalDofMapperPtr GDAMinimumRule::getDofMapper(GlobalIndexType cellID, CellConstraints &constraints, int varIDToMap, int sideOrdinalToMap)
{
//  { // DEBUGGING
//    if ((cellID==0) && (sideOrdinalToMap==3))
//    {
//      cout << "set breakpoint here.\n";
//    }
//  }
  
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1))
  {
    // a mapper for the whole dof ordering: we cache these separately...
    if (_dofMapperCache.find(cellID) != _dofMapperCache.end())
    {
      return _dofMapperCache[cellID];
    }
  }
  else
  {
    map< GlobalIndexType, map<int, map<int, LocalDofMapperPtr> > >::iterator cellMapEntry = _dofMapperForVariableOnSideCache.find(cellID);
    if (cellMapEntry != _dofMapperForVariableOnSideCache.end())
    {
      map<int, map<int, LocalDofMapperPtr> >::iterator sideMapEntry = cellMapEntry->second.find(sideOrdinalToMap);
      if (sideMapEntry != cellMapEntry->second.end())
      {
        map<int, LocalDofMapperPtr>::iterator varMapEntry = sideMapEntry->second.find(varIDToMap);
        if (varMapEntry != sideMapEntry->second.end())
        {
          return varMapEntry->second;
        }
      }
    }
  }

  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
  int sideCount = topo->getSideCount();
  int spaceDim = topo->getDimension();

  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory->trialVars();

  typedef vector< SubBasisDofMapperPtr > BasisMap;
  map< int, BasisMap > volumeMap; // keys are variable IDs
  vector< map< int, BasisMap > > sideMaps(sideCount);

  bool determineFittableGlobalDofOrdinalsOnSides = false;
  
  if (varIDToMap != -1)
  {
    // replace trialVars appropriately
    map<int, VarPtr> oneEntryTrialVars;
    oneEntryTrialVars[varIDToMap] = trialVars[varIDToMap];
    trialVars = oneEntryTrialVars;
  }
  
  for (auto varEntry : trialVars)
  {
    VarPtr var = varEntry.second;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
    if (varHasSupportOnVolume)
    {
      // we will only care about this variable on the sides if it's not discontinuous
      Camellia::EFunctionSpace fs = efsForSpace(var->space());
      if (!functionSpaceIsDiscontinuous(fs))
      {
        determineFittableGlobalDofOrdinalsOnSides = true;
        break;
      }
    }
    else
    {
      // there is a trace variable of interest; we should determine fittable global dof ordinals on sides
      determineFittableGlobalDofOrdinalsOnSides = true;
      break;
    }
  }
  
  vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(sideCount);
  
  if (determineFittableGlobalDofOrdinalsOnSides)
  {
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
      fittableGlobalDofOrdinalsOnSides[sideOrdinal] = getFittableGlobalDofIndices(cellID, constraints, sideOrdinal);
  //    cout << "sideOrdinal " << sideOrdinal;
  //    Camellia::print(", fittableGlobalDofIndices", fittableGlobalDofOrdinalsOnSides[sideOrdinal]);
    }
  }

  set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;

  /**************** ESTABLISH OWNERSHIP ****************/
  SubCellDofIndexInfo dofIndexInfo = getGlobalDofIndices(cellID, constraints);

  /**************** CREATE SUB-BASIS MAPPERS ****************/

  for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++)
  {
    VarPtr var = varIt->second;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);

    if (varHasSupportOnVolume)
    {
      if (sideOrdinalToMap == -1)
      {
        volumeMap[var->ID()] = getBasisMap(cellID, dofIndexInfo, var);
        // first, get interior dofs
        fittableGlobalDofOrdinalsInVolume.insert(dofIndexInfo[spaceDim][0][var->ID()].begin(),dofIndexInfo[spaceDim][0][var->ID()].end());
        
        // now, sides
        for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
        {
          set<GlobalIndexType> fittableGlobalDofOrdinalsOnSide = getFittableGlobalDofIndices(cellID, constraints, sideOrdinal, var->ID());
          fittableGlobalDofOrdinalsInVolume.insert(fittableGlobalDofOrdinalsOnSide.begin(),fittableGlobalDofOrdinalsOnSide.end());
        }
      }
      else
      {
        if (functionSpaceIsDiscontinuous(efsForSpace(var->space())))
        {
          // then there is no chance of any minimum-rule constraints to impose
          // and we need to specially "extract" the side degrees of freedom
          // (this comes up when imposing BCs in a DG context)
          volumeMap[var->ID()] = getBasisMapDiscontinuousVolumeRestrictedToSide(cellID, dofIndexInfo, var, sideOrdinalToMap);
          for (auto subMap : volumeMap[var->ID()])
          {
            fittableGlobalDofOrdinalsInVolume.insert(subMap->mappedGlobalDofOrdinals().begin(),subMap->mappedGlobalDofOrdinals().end());
          }
        }
        else
        {
          // if the function space is not discontinuous, then we restrict the usual basis map to the global dof ordinals on the side
          DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
          BasisPtr basis = trialOrdering->getBasis(var->ID());
          set<int> basisDofOrdinalsForSide = basis->dofOrdinalsForSide(sideOrdinalToMap);
          
          BasisMap unrestrictedMap = getBasisMap(cellID, dofIndexInfo, var);
          volumeMap[var->ID()] = getRestrictedBasisMap(unrestrictedMap, basisDofOrdinalsForSide);
          set<GlobalIndexType> fittableGlobalDofOrdinalsOnSide = getFittableGlobalDofIndices(cellID, constraints, sideOrdinalToMap, var->ID());
          fittableGlobalDofOrdinalsInVolume.insert(fittableGlobalDofOrdinalsOnSide.begin(),fittableGlobalDofOrdinalsOnSide.end());
        }
      }
    }
    else
    {
      for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
      {
        if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
        if (! trialOrdering->hasBasisEntry(var->ID(), sideOrdinal)) continue; // skip this side/var combo...
        sideMaps[sideOrdinal][var->ID()] = getBasisMap(cellID, dofIndexInfo, var, sideOrdinal);
      }
    }
  }

  set<GlobalIndexType> emptyGlobalIDSet; // the "extra" guys to map
  LocalDofMapperPtr dofMapper = Teuchos::rcp( new LocalDofMapper(trialOrdering,volumeMap,fittableGlobalDofOrdinalsInVolume,
                                                                 sideMaps,fittableGlobalDofOrdinalsOnSides,emptyGlobalIDSet,
                                                                 varIDToMap,sideOrdinalToMap) );
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1))
  {
    // a mapper for the whole dof ordering: we cache these...
    _dofMapperCache[cellID] = dofMapper;
    return _dofMapperCache[cellID];
  }
  else
  {
    _dofMapperForVariableOnSideCache[cellID][sideOrdinalToMap][varIDToMap] = dofMapper;
    return dofMapper;
  }
}

BasisMap GDAMinimumRule::getRestrictedBasisMap(BasisMap &basisMap, const set<int> &basisDofOrdinalRestriction) // restricts to part of the basis
{
  BasisMap newBasisMap;
  for (SubBasisDofMapperPtr subBasisMap : basisMap)
  {
    subBasisMap = subBasisMap->restrictDofOrdinalFilter(basisDofOrdinalRestriction);
    if (subBasisMap->basisDofOrdinalFilter().size() > 0)
    {
      newBasisMap.push_back(subBasisMap);
    }
  }
  return newBasisMap;
}

PartitionIndexType GDAMinimumRule::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex )
{
  PartitionIndexType numRanks = _partitionDofCounts.size();
  GlobalIndexType totalDofCount = 0;
  for (PartitionIndexType i=0; i<numRanks; i++)
  {
    totalDofCount += _partitionDofCounts(i);
    if (totalDofCount > globalDofIndex)
    {
      return i;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid globalDofIndex");
}

void GDAMinimumRule::printConstraintInfo(GlobalIndexType cellID)
{
  CellConstraints cellConstraints = getCellConstraints(cellID);
  cout << "***** Constraints for cell " << cellID << " ****** \n";
  CellPtr cell = _meshTopology->getCell(cellID);
  int spaceDim = cell->topology()->getDimension();
  for (int d=0; d<spaceDim; d++)
  {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++)
    {
      AnnotatedEntity constraintInfo = cellConstraints.subcellConstraints[d][scord];
      cout << "    ordinal " << scord  << " constrained by cell " << constraintInfo.cellID;
      cout << ", side " << constraintInfo.sideOrdinal << "'s dimension " << constraintInfo.dimension << " subcell ordinal " << constraintInfo.subcellOrdinal << endl;
    }
  }

  cout << "Ownership info:\n";
  for (int d=0; d<spaceDim; d++)
  {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++)
    {
      cout << "    ordinal " << scord  << " owned by cell " << cellConstraints.owningCellIDForSubcell[d][scord].cellID << endl;
    }
  }
}

void GDAMinimumRule::printGlobalDofInfo()
{
  set<GlobalIndexType> cellIDs = _meshTopology->getActiveCellIndices();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    CellConstraints cellConstraints = getCellConstraints(cellID);
    SubCellDofIndexInfo dofIndexInfo = getOwnedGlobalDofIndices(cellID, cellConstraints);
    printDofIndexInfo(cellID, dofIndexInfo);
  }
}

void GDAMinimumRule::rebuildLookups()
{
  _constraintsCache.clear(); // to free up memory, could clear this again after the lookups are rebuilt.  Having the cache is most important during the construction below.
  _dofMapperCache.clear();
  _dofMapperForVariableOnSideCache.clear();
  _ownedGlobalDofIndicesCache.clear();
  _globalDofIndicesForCellCache.clear();
  _fittableGlobalIndicesCache.clear();

  _partitionFluxIndexOffsets.clear();
  _partitionTraceIndexOffsets.clear();
  _partitionIndexOffsetsForVarID.clear();

  int rank = _partitionPolicy->Comm()->MyPID();
//  cout << "GDAMinimumRule: Rebuilding lookups on rank " << rank << endl;
  set<GlobalIndexType>* myCellIDs = &_partitions[rank];

  map<int, VarPtr> trialVars = _varFactory->trialVars();

  _cellDofOffsets.clear(); // within the partition, offsets for the owned dofs in cell

  int spaceDim = _meshTopology->getDimension();
  int sideDim = spaceDim - 1;

  // pieces of this remain fairly ugly--the brute force searches are limited to entities on a cell (i.e. < O(12) items to search in a hexahedron),
  // and I've done a reasonable job only doing them when we need the result, but they still are brute force searches.  By tweaking
  // the design of MeshTopology and Cell to take better advantage of regularities (or just to store better lookups), we should be able to do better.
  // But in the interest of avoiding wasting development time on premature optimization, I'm leaving it as is for now...

  _partitionDofCount = 0; // how many dofs we own locally
  for (GlobalIndexType cellID : *myCellIDs)
  {
    _cellDofOffsets[cellID] = _partitionDofCount;
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = cell->topology();
    CellConstraints constraints = getCellConstraints(cellID);

    set< pair<unsigned,IndexType> > entitiesClaimedForCell;

    for (int d=0; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        OwnershipInfo ownershipInfo = constraints.owningCellIDForSubcell[d][scord];
        if (ownershipInfo.cellID == cellID)   // owned by this cell: count all the constraining dofs as entries for this cell
        {
          pair<unsigned, IndexType> owningSubcellEntity = make_pair(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          if (entitiesClaimedForCell.find(owningSubcellEntity) != entitiesClaimedForCell.end())
          {
            continue; // already processed this guy on this cell
          }
          else
          {
            entitiesClaimedForCell.insert(owningSubcellEntity);
          }
          GlobalIndexType constrainingCellID = constraints.subcellConstraints[d][scord].cellID;
          unsigned constrainingSubcellDimension = constraints.subcellConstraints[d][scord].dimension;
          DofOrderingPtr trialOrdering = _elementTypeForCell[constrainingCellID]->trialOrderPtr;
          for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++)
          {
            VarPtr var = varIt->second;
            unsigned scordForBasis;
            bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
            BasisPtr basis; // the constraining basis for the subcell
            if (varHasSupportOnVolume)
            {
              // volume basis => the basis sees the cell as a whole: in constraining cell, map from side scord to the volume
              if (constrainingSubcellDimension==spaceDim)
              {
                // then there is only one subcell ordinal (and there will be -1's in sideOrdinal and subcellOrdinalInSide....
                scordForBasis = 0;
              }
              else
              {
                scordForBasis = CamelliaCellTools::subcellOrdinalMap(_meshTopology->getCell(constrainingCellID)->topology(), sideDim,
                                constraints.subcellConstraints[d][scord].sideOrdinal,
                                constrainingSubcellDimension, constraints.subcellConstraints[d][scord].subcellOrdinal);
              }
              basis = trialOrdering->getBasis(var->ID());
              // field
              int ordinalCount = basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
              for (int ordinal=0; ordinal<ordinalCount; ordinal++)
              {
                _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
              }
            }
            else
            {
              if (constrainingSubcellDimension==spaceDim) continue; // side bases don't have any support on the interior of the cell...
              if (!trialOrdering->hasBasisEntry(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal)) continue;

              scordForBasis = constraints.subcellConstraints[d][scord].subcellOrdinal; // the basis sees the side, so that's the view to use for subcell ordinal
              basis = trialOrdering->getBasis(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal);

              int ordinalCount = basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
              if (var->varType()==FLUX)
              {
                for (int ordinal=0; ordinal<ordinalCount; ordinal++)
                {
                  _partitionFluxIndexOffsets.insert(ordinal+_partitionDofCount);
                  _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
                }
              }
              else if (var->varType() == TRACE)
              {
                for (int ordinal=0; ordinal<ordinalCount; ordinal++)
                {
                  _partitionTraceIndexOffsets.insert(ordinal+_partitionDofCount);
                  _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
                }
              }
            }
            _partitionDofCount += basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
          }
        }
      }
    }
  }
  int numRanks = _partitionPolicy->Comm()->NumProc();
  _partitionDofCounts.resize(numRanks);
  _partitionDofCounts.initialize(0.0);
  _partitionDofCounts[rank] = _partitionDofCount;
  MPIWrapper::entryWiseSumAfterCasting<GlobalIndexType,long long>(*_partitionPolicy->Comm(),_partitionDofCounts);
//  if (rank==0) cout << "partitionDofCounts:\n" << _partitionDofCounts;
  _partitionDofOffset = 0; // add this to a local partition dof index to get the global dof index
  for (int i=0; i<rank; i++)
  {
    _partitionDofOffset += _partitionDofCounts[i];
  }
  _globalDofCount = _partitionDofOffset;
  for (int i=rank; i<numRanks; i++)
  {
    _globalDofCount += _partitionDofCounts[i];
  }
//  if (rank==0) cout << "globalDofCount: " << _globalDofCount << endl;
  // collect and communicate global cell dof offsets:
  int activeCellCount = _meshTopology->getActiveCellIndices().size();
  Intrepid::FieldContainer<int> globalCellIDDofOffsets(activeCellCount);
  int partitionCellOffset = 0;
  for (int i=0; i<rank; i++)
  {
    partitionCellOffset += _partitions[i].size();
  }
  // fill in our _cellDofOffsets:

  int i=0;
  for (GlobalIndexType cellID : *myCellIDs)
  {
    globalCellIDDofOffsets[partitionCellOffset+i] = _cellDofOffsets[cellID] + _partitionDofOffset;
    i++;
  }
  // global copy:
  MPIWrapper::entryWiseSum(*_partitionPolicy->Comm(),globalCellIDDofOffsets);
  // fill in the lookup table:
  _globalCellDofOffsets.clear();
  int globalCellIndex = 0;
  for (int i=0; i<numRanks; i++)
  {
    set<GlobalIndexType> rankCellIDs = _partitions[i];
    for (set<GlobalIndexType>::iterator cellIDIt = rankCellIDs.begin(); cellIDIt != rankCellIDs.end(); cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      _globalCellDofOffsets[cellID] = globalCellIDDofOffsets[globalCellIndex];
//      if (rank==numRanks-1) cout << "global dof offset for cell " << cellID << ": " << _globalCellDofOffsets[cellID] << endl;
      globalCellIndex++;
    }
  }

  _cellIDsForElementType = vector< map< ElementType*, vector<GlobalIndexType> > >(numRanks);
  for (int i=0; i<numRanks; i++)
  {
    set<GlobalIndexType> cellIDs = _partitions[i];
    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      ElementTypePtr elemType = _elementTypeForCell[cellID];
      _cellIDsForElementType[i][elemType.get()].push_back(cellID);
    }
  }
}

namespace Camellia
{
  std::ostream& operator << (std::ostream& os, AnnotatedEntity& annotatedEntity)
  {
    os << "cell " << annotatedEntity.cellID;
    if (annotatedEntity.sideOrdinal != -1)
    {
      os << "'s side " << annotatedEntity.sideOrdinal;
    }
    os << "'s " << CamelliaCellTools::entityTypeString(annotatedEntity.dimension);
    os << " " << annotatedEntity.subcellOrdinal;
    
    return os;
  }
}