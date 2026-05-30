//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CellDataMigration.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/15/14.
//
//

#include <Teuchos_GlobalMPISession.hpp>

#include "CellDataMigration.h"

#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "Solution.h"

using namespace Camellia;

typedef pair<RefinementBranch,vector<GlobalIndexType>> LabeledRefinementBranch; // first cellID indicates the root cell ID; after that, each cellID indicates first child of each refinement
typedef pair<LabeledRefinementBranch, vector<vector<double>> > RootedLabeledRefinementBranch; // second contains vertex coordinates for root cell

// TODO: move READ_ADVANCE and WRITE_ADVANCE into Camellia_Serialization.h or something like that
#define READ_ADVANCE(dataLocation,variable) readAndAdvance(&variable,dataLocation,sizeof(variable))
#define READ_ADVANCE_ARRAY(dataLocation,array) if (array.size() > 0) readAndAdvance(array.data(),dataLocation,sizeof(array[0])*array.size())
#define WRITE_ADVANCE(dataLocation,variable) writeAndAdvance(dataLocation,&variable,sizeof(variable))
#define WRITE_ADVANCE_ARRAY(dataLocation,array) if (array.size() > 0) writeAndAdvance(dataLocation,array.data(),sizeof(array[0])*array.size())

std::vector<std::pair<std::pair<GlobalIndexType,bool>,Camellia::FieldContainer<double>>> CellDataMigration::solutionCoefficients;

void CellDataMigration::addMigratedGeometry(MeshTopology* meshTopo,
                                            const vector<RootedLabeledRefinementBranch> &rootedLabeledBranches)
{
  for (int i=0; i<rootedLabeledBranches.size(); i++)
  {
    const RootedLabeledRefinementBranch* rootedLabeledBranch = &rootedLabeledBranches[i];
    const vector<GlobalIndexType>* labels = &rootedLabeledBranch->first.second;
    const RefinementBranch* refBranch = &rootedLabeledBranch->first.first;
    const vector<vector<double>>* rootVertices = &rootedLabeledBranch->second;
    // first, check whether we know the root cell ID
    GlobalIndexType rootCellID = (*labels)[0];
    if (! meshTopo->isValidCellIndex(rootCellID))
    {
      CellTopoPtr rootCellTopo = (*refBranch)[0].first->parentTopology();
      meshTopo->addMigratedCell(rootCellID, rootCellTopo, *rootVertices);
    }
    for (int j=1; j<labels->size(); j++)
    {
      GlobalIndexType grandparentFirstChildCellID = (*labels)[j-1];
      int parentOffset = 0;
      if (j >= 2)
      {
        parentOffset = (*refBranch)[j-2].second; // parent's childOrdinal in grandparent
      }
      GlobalIndexType parentCellID = grandparentFirstChildCellID + parentOffset;
      GlobalIndexType firstChildCellID = (*labels)[j];
      RefinementPatternPtr refPattern = RefinementPattern::refinementPattern((*refBranch)[j-1].first->getKey());
      bool allChildrenKnown = true;
      for (int childOrdinal = 0; childOrdinal < refPattern->numChildren(); childOrdinal++)
      {
        if (! meshTopo->isValidCellIndex(childOrdinal + firstChildCellID))
        {
          allChildrenKnown = false;
          break;
        }
      }
      
      if (! allChildrenKnown)
      {
        // get the RefinementPatternPtr stored for the RefinementPattern:
        meshTopo->refineCell(parentCellID, refPattern, firstChildCellID);
      }
    }
  }
}

int CellDataMigration::dataSize(Mesh *mesh, GlobalIndexType cellID, bool willPackParentDofs)
{
  int solutionSize = solutionDataSize(mesh, cellID, willPackParentDofs);
  int geometrySize = geometryDataSize(mesh, cellID);
  int size = solutionSize + geometrySize;
//  cout << "geometrySize for cellID " << cellID << ": " << geometrySize << endl;
//  cout << "solutionSize for cellID " << cellID << ": " << solutionSize << endl;
//  cout << "dataSize for cellID " << cellID << ": " << size << endl;
  return size;
}

int CellDataMigration::geometryDataSize(Mesh* mesh, GlobalIndexType cellID)
{
  int size = 0;
  bool geometryIsDistributed = mesh->getTopology()->isDistributed(); // otherwise, redundantly stored, and we don't need to send
  
  vector<RootedLabeledRefinementBranch> cellHaloGeometry;
  if (geometryIsDistributed) getCellHaloGeometry(mesh, cellID, cellHaloGeometry);
  int numLabeledBranches = cellHaloGeometry.size();
  size += sizeof(numLabeledBranches);
  
  for (int i=0; i<numLabeledBranches; i++)
  {
    // top level: labeledRefinementBranch, vertices
    
    // LabeledRefinementBranch has two entries: RefinementBranch, labels
    int numLabels = cellHaloGeometry[i].first.second.size();
    size += sizeof(numLabels);
    size += numLabels * sizeof(GlobalIndexType);
    
    // at each level, RefinementBranch will yield two entries: RefinementPatternKey and childOrdinal
    int numLevels = cellHaloGeometry[i].first.first.size();
    size += numLevels * (sizeof(RefinementPatternKey) + sizeof(unsigned));
    
    // we are able to extract the number of vertices and the spaceDim from RefinementPatternKey and Mesh
    int spaceDim = mesh->getDimension();
    int numVertexEntries = spaceDim * cellHaloGeometry[i].second.size();
    size += numVertexEntries * sizeof(double);
  }

  // entity sets
  int spaceDim = mesh->getDimension();
  const MeshTopology* meshTopo = mesh->getTopology()->baseMeshTopology();
  vector<EntityHandle> entityHandles;
  if (geometryIsDistributed) entityHandles = meshTopo->getEntityHandlesForCell(cellID);
  int numHandles = entityHandles.size();
  size += sizeof(numHandles);
  for (int handleOrdinal=0; handleOrdinal<numHandles; handleOrdinal++)
  {
    EntityHandle handle = entityHandles[handleOrdinal];
    size += sizeof(handle);
    EntitySetPtr entitySet = meshTopo->getEntitySet(handle);
    for (int d=0; d<spaceDim; d++)
    {
      vector<unsigned> subcellOrdinals = entitySet->subcellOrdinals(mesh->getTopology(), cellID, d);
      int numSubcells = subcellOrdinals.size();
      size += sizeof(numSubcells);
      size += numSubcells * sizeof(unsigned);
    }
  }
  
  return size;
}

void CellDataMigration::getCellGeometry(const MeshTopology* meshTopo, GlobalIndexType cellID, set<GlobalIndexType> &knownCells,
                                        RootedLabeledRefinementBranch &cellGeometry)
{
  // clear cellGeometry
  cellGeometry = RootedLabeledRefinementBranch();
  vector<vector<double>>* rootCellVertices = &cellGeometry.second;
  RefinementBranch* cellRefBranch = &cellGeometry.first.first;
  vector<GlobalIndexType>* cellLabels = &cellGeometry.first.second;
  
  CellPtr cell = meshTopo->getCell(cellID);
  *cellRefBranch = cell->refinementBranch();
  
  GlobalIndexType rootCellIndex = cell->rootCellIndex();
  cellLabels->push_back(rootCellIndex);
  CellPtr ancestralCell = meshTopo->getCell(cell->rootCellIndex());
  int vertexCount = ancestralCell->vertices().size();
  for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    rootCellVertices->push_back(meshTopo->getVertex(ancestralCell->vertices()[vertexOrdinal]));
  }
  
  knownCells.insert(rootCellIndex); // keep track of cells already included in the data structure, to avoid redundancy
  for (auto refEntry : *cellRefBranch)
  {
    RefinementPattern* refPattern = refEntry.first;
    int childCount = refPattern->numChildren();
    
    if (childCount == 1)
    {
      // null refinement pattern -- child ID is the same as cell ID
      cellLabels->push_back(ancestralCell->cellIndex());
      knownCells.insert(ancestralCell->cellIndex());
    }
    else
    {
      unsigned firstChildOrdinal = 0;
      GlobalIndexType firstChildCellIndex = ancestralCell->children()[firstChildOrdinal]->cellIndex();
      cellLabels->push_back(firstChildCellIndex);
      
      for (int childOrdinal=0; childOrdinal<childCount; childOrdinal++)
      {
        knownCells.insert(firstChildCellIndex + childOrdinal);
      }
      ancestralCell = ancestralCell->children()[refEntry.second];
    }
  }
}

void CellDataMigration::getCellHaloGeometry(Mesh *mesh, GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloGeometry)
{
  ElementTypePtr elemType = mesh->getElementType(cellID);
  int dimForContinuity = elemType->trialOrderPtr->minimumSubcellDimensionForContinuity();
  
  getCellHaloGeometry(mesh->getTopology()->baseMeshTopology(), dimForContinuity, cellID, cellHaloGeometry);
}

void CellDataMigration::getCellHaloGeometry(const MeshTopology *meshTopo, unsigned dimForContinuity,
                                            GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloGeometry)
{
  cellHaloGeometry.resize(1);
  
  set<GlobalIndexType> knownCells;
  getCellGeometry(meshTopo, cellID, knownCells, cellHaloGeometry[0]);
  
  CellPtr cell = meshTopo->getCell(cellID);
  ConstMeshTopologyPtr meshTopoPtr = Teuchos::rcp(meshTopo,false);
//  set<GlobalIndexType> cellHaloIndices = cell->getActiveNeighborIndices(dimForContinuity, meshTopoPtr);
  set<GlobalIndexType> cellHaloIndices;
  meshTopo->cellHalo(cellHaloIndices, {cellID}, dimForContinuity);
  
  // since children have cell indices greater than their parents, we get a more compact structure if we
  // iterate in reverse order
  for (auto neighborIDIt = cellHaloIndices.rbegin(); neighborIDIt != cellHaloIndices.rend(); neighborIDIt++)
  {
    GlobalIndexType neighborID = *neighborIDIt;
    if (knownCells.find(neighborID) == knownCells.end())
    {
      RootedLabeledRefinementBranch neighborCellGeometry;
      getCellGeometry(meshTopo, neighborID, knownCells, neighborCellGeometry);
      cellHaloGeometry.push_back(neighborCellGeometry);
    }
  }
}

void CellDataMigration::packData(Mesh *mesh, GlobalIndexType cellID, bool packParentDofs, char *dataBuffer, int size)
{
  // ideally, we'd pack the global coefficients for this cell and simply remap them when unpacking
  // however, producing the map is an implementation challenge, particularly in the presence of refined elements
  // so what we do instead is map local data, and then use the local to global mapper that we build anyway to map
  // to global values when unpacking.
  //  int myRank                    = mesh->Comm()->MyPID();
  //  cout << "CellDataMigration::packData() called for cell " << cellID << " on rank " << myRank << endl;
  char* dataLocation = dataBuffer;
  if (size<dataSize(mesh, cellID, packParentDofs))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "undersized dataBuffer");
  }
  
  // pack geometry info first
  packGeometryData(mesh,cellID,dataLocation,size);
  
  packSolutionData(mesh, cellID, packParentDofs, dataLocation, size);
}

void CellDataMigration::packGeometryData(Mesh *mesh, GlobalIndexType cellID, char* &dataLocation, int size)
{
  bool geometryIsDistributed = mesh->getTopology()->isDistributed(); // otherwise, redundantly stored, and we don't need to send
  
  vector<RootedLabeledRefinementBranch> cellHaloGeometry;
  if (geometryIsDistributed) getCellHaloGeometry(mesh, cellID, cellHaloGeometry);
  int numLabeledBranches = cellHaloGeometry.size();
  
  memcpy(dataLocation, &numLabeledBranches, sizeof(numLabeledBranches));
  dataLocation += sizeof(numLabeledBranches);
  
  for (int i=0; i<numLabeledBranches; i++)
  {
    // top level: LabeledRefinementBranch, vertices
    // LabeledRefinementBranch has two entries: RefinementBranch, labels
    int numLabels = cellHaloGeometry[i].first.second.size();
    memcpy(dataLocation, &numLabels, sizeof(numLabels));
    dataLocation += sizeof(numLabels);
    
    memcpy(dataLocation, &cellHaloGeometry[i].first.second[0], numLabels * sizeof(GlobalIndexType));
    dataLocation += numLabels * sizeof(GlobalIndexType);
    
    // numLevels = numLabels - 1;
    // at each level, RefinementBranch will yield two entries: RefinementPatternKey and childOrdinal
    int numLevels = cellHaloGeometry[i].first.first.size();
    TEUCHOS_TEST_FOR_EXCEPTION(numLevels != numLabels-1, std::invalid_argument, "numLevels != numLabels - 1");
    for (int level=0; level<numLevels; level++)
    {
      RefinementPatternKey key = cellHaloGeometry[i].first.first[level].first->getKey();
      memcpy(dataLocation, &key, sizeof(key));
      dataLocation += sizeof(key);
      
      unsigned childOrdinal = cellHaloGeometry[i].first.first[level].second;
      memcpy(dataLocation, &childOrdinal, sizeof(childOrdinal));
      dataLocation += sizeof(childOrdinal);
    }
    
    // we are able to extract the number of vertices and the spaceDim from RefinementPatternKey and Mesh
    int spaceDim = mesh->getDimension();
    int vertexCount = cellHaloGeometry[i].second.size();
    for (int vertexOrdinal=0; vertexOrdinal < vertexCount; vertexOrdinal++)
    {
      memcpy(dataLocation, &cellHaloGeometry[i].second[vertexOrdinal][0], spaceDim * sizeof(double));
      dataLocation += spaceDim * sizeof(double);
    }
  }
  
  int spaceDim = mesh->getDimension();
  // entity sets
  const MeshTopology* meshTopo = mesh->getTopology()->baseMeshTopology();
  vector<EntityHandle> entityHandles;
  if (geometryIsDistributed) entityHandles = meshTopo->getEntityHandlesForCell(cellID);
  int numHandles = entityHandles.size();
  memcpy(dataLocation, &numHandles, sizeof(numHandles));
  dataLocation += sizeof(numHandles);
  for (int handleOrdinal=0; handleOrdinal<numHandles; handleOrdinal++)
  {
    EntityHandle handle = entityHandles[handleOrdinal];
    memcpy(dataLocation, &handle, sizeof(handle));
    dataLocation += sizeof(handle);
    EntitySetPtr entitySet = meshTopo->getEntitySet(handle);
    for (int d=0; d<spaceDim; d++)
    {
      vector<unsigned> subcellOrdinals = entitySet->subcellOrdinals(mesh->getTopology(), cellID, d);
      int numSubcells = subcellOrdinals.size();
      memcpy(dataLocation, &numSubcells, sizeof(numSubcells));
      dataLocation += sizeof(numSubcells);
      memcpy(dataLocation, subcellOrdinals.data(), sizeof(unsigned)*numSubcells);
      dataLocation += sizeof(unsigned)*numSubcells;
    }
  }
}

void CellDataMigration::packSolutionData(Mesh *mesh, GlobalIndexType cellID, bool packParentDofs, char* &dataLocation, int size)
{
  int packedDofsBelongToParent = packParentDofs ? 1 : 0;
  memcpy(dataLocation, &packedDofsBelongToParent, sizeof(packedDofsBelongToParent));
  dataLocation += sizeof(packedDofsBelongToParent);
  
  //  cout << "packed data for cell " << cellID << ": ";
  //  ElementTypePtr elemType = mesh->getElementType(cellID);
  vector<TSolutionPtr<double>> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  int numSolutions = solutions.size();
  memcpy(dataLocation, &numSolutions, sizeof(numSolutions));
  dataLocation += sizeof(numSolutions);
  
  GlobalIndexType cellIDForCoefficients;
  if (packParentDofs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    cellIDForCoefficients = cell->getParent()->cellIndex();
  }
  else
  {
    cellIDForCoefficients = cellID;
  }
  
  //  cout << numSolutions << " ";
  for (int i=0; i<numSolutions; i++)
  {
    int numLHSes = solutions[i]->numSolutions();
    for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
    {
      bool thisLHSHasCoefficients = solutions[i]->cellHasCoefficientsAssigned(cellIDForCoefficients, lhsOrdinal);
      if ( !thisLHSHasCoefficients )
      {
        int localDofs = 0;
        memcpy(dataLocation, &localDofs, sizeof(localDofs));
        dataLocation += sizeof(localDofs);
        continue; // no dofs to assign; proceed to next solution
      }
      // # dofs per solution
      const Camellia::FieldContainer<double>* solnCoeffs = &solutions[i]->allCoefficientsForCellID(cellIDForCoefficients, false, lhsOrdinal); // false: don't warn
//      cout << "cellIDForCoefficients = " << cellIDForCoefficients;
//      cout << "; lhOrdinal = " << lhsOrdinal;
      int localDofs = solnCoeffs->size();
//      cout << "; localDofs = " << localDofs << endl;
      //    int localDofs = elemType->trialOrderPtr->totalDofs();
      memcpy(dataLocation, &localDofs, sizeof(localDofs));
      //    cout << localDofs << " ";
      dataLocation += sizeof(localDofs);
      
      if (localDofs > 0)
      {
        memcpy(dataLocation, &(*solnCoeffs)[0], localDofs * sizeof(double));
      }
      // the dofs themselves
      dataLocation += localDofs * sizeof(double);
      //    for (int j=0; j<solnCoeffs->size(); j++) {
      //      cout << (*solnCoeffs)[j] << " ";
      //    }
      //    cout << ";";
    }
  }
  //  cout << endl;
}

void CellDataMigration::processSolutionCoefficients(Mesh* mesh)
{
  vector<TSolutionPtr<double>> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  int numSolutions = solutions.size();
  int solnOrdinal = 0;
  
  for (auto entry : solutionCoefficients)
  {
    GlobalIndexType cellID = entry.first.first;
    bool coefficientsBelongToParent = entry.first.second;
    const Camellia::FieldContainer<double>& solnCoeffs = entry.second;
    
    int numLHSes = solutions[solnOrdinal]->numSolutions();
    for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
    {
      if (!coefficientsBelongToParent)
      {
        solutions[solnOrdinal]->setSolnCoeffsForCellID(solnCoeffs, cellID, lhsOrdinal);
      }
      else
      {
        CellPtr cell = mesh->getTopology()->getCell(cellID);
        CellPtr parent = cell->getParent();
        int childOrdinal = -1;
        vector<IndexType> childIndices = parent->getChildIndices(mesh->getTopology());
        for (int i=0; i<childIndices.size(); i++)
        {
          if (childIndices[i]==cellID) childOrdinal = i;
          else childIndices[i] = -1; // indication that Solution should not compute the projection for this child
        }
        if (childOrdinal == -1)
        {
          cout << "ERROR: child not found.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child not found!");
        }
        //      cout << "determining cellID " << parent->cellIndex() << "'s child " << childOrdinal << "'s coefficients.\n";
        solutions[solnOrdinal]->projectOldCellOntoNewCells(parent->cellIndex(), mesh->getElementType(parent->cellIndex()),
                                                           solnCoeffs, childIndices, lhsOrdinal);
      }
    }
    solnOrdinal = (solnOrdinal + 1) % numSolutions;
  }
  solutionCoefficients.clear();
  for (SolutionPtr soln : solutions)
  {
    // reset the global LHS vector from the local coefficients
    soln->initializeLHSVector();
    // import any coefficients that belong to neighbors -- and reset local coefficients accordingly
    soln->importSolution();
  }
}

int CellDataMigration::solutionDataSize(Mesh *mesh, GlobalIndexType cellID, bool willPackParentDofs)
{
  int size = 0;
  
  GlobalIndexType cellIDForCoefficients;
  if (willPackParentDofs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    cellIDForCoefficients = cell->getParent()->cellIndex();
  }
  else
  {
    cellIDForCoefficients = cellID;
  }
  ElementTypePtr elemType = mesh->getElementType(cellIDForCoefficients);
  
//  int packedDofsBelongToParent = 0; // 0 for false, anything else for true
  size += sizeof(int); // sizeof(packedDofsBelongToParent)
  
  vector<TSolutionPtr<double>> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  // store # of solution objects
  int numSolutions = solutions.size();
  size += sizeof(numSolutions);
//  for (int i=0; i<numSolutions; i++)
//  {
//    // # dofs per solution
//    int localDofs = elemType->trialOrderPtr->totalDofs();
//    size += sizeof(localDofs);
//    // the dofs themselves
//    size += localDofs * sizeof(double);
//  }
  for (int i=0; i<numSolutions; i++)
  {
    int numLHSes = solutions[i]->numSolutions();
    for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
    {
      bool thisLHSHasCoefficients = solutions[i]->cellHasCoefficientsAssigned(cellIDForCoefficients, lhsOrdinal);
      if ( !thisLHSHasCoefficients )
      {
//        int localDofs = 0;
        size += sizeof(int); // localDof count
        continue; // no dofs to assign; proceed to next solution
      }
      int localDofs = elemType->trialOrderPtr->totalDofs();
      size += sizeof(localDofs);
      // the dofs themselves
      size += localDofs * sizeof(double);
    }
  }
  
  return size;
}

void CellDataMigration::unpackData(Mesh *mesh, GlobalIndexType cellID, const char *dataBuffer, int size)
{
  //  int myRank                    = mesh->Comm()->MyPID();
  //  cout << "CellDataMigration::unpackData() called for cell " << cellID << " on rank " << myRank << endl;
  const char* dataLocation = dataBuffer;
  
  // we can't know what the right size is anymore, since that will depend on the geometry
  // unpack geometry info first
  vector<RootedLabeledRefinementBranch> rootedLabeledBranches;
  unpackGeometryData(mesh, cellID, dataLocation, size, rootedLabeledBranches);
  unpackSolutionData(mesh, cellID, dataLocation, size);
}

void CellDataMigration::unpackGeometryData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size,
                                           vector<RootedLabeledRefinementBranch> &cellHaloGeometry)
{
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
  if (meshTopo == NULL)
  {
    // then the MeshTopology had better *not* be distributed
    // there should be two ints, numLabeledBranches, numHandles, both 0, and nothing else
  
    int numLabeledBranches;
    
    memcpy(&numLabeledBranches, dataLocation, sizeof(numLabeledBranches));
    dataLocation += sizeof(numLabeledBranches);
    
    TEUCHOS_TEST_FOR_EXCEPTION(numLabeledBranches != 0, std::invalid_argument, "unpackGeometryData() called with nontrivial migrated geometry on a Mesh that does not have a full MeshTopology as its MeshTopologyView");
    
    int numHandles;
    memcpy(&numHandles, dataLocation, sizeof(numHandles));
    dataLocation += sizeof(numHandles);
    
    TEUCHOS_TEST_FOR_EXCEPTION(numHandles != 0, std::invalid_argument, "unpackGeometryData() called with nontrivial migrated geometry on a Mesh that does not have a full MeshTopology as its MeshTopologyView");
    
    return;
  }
    
  cellHaloGeometry.clear();
  int numLabeledBranches;
  
  memcpy(&numLabeledBranches, dataLocation, sizeof(numLabeledBranches));
  dataLocation += sizeof(numLabeledBranches);
  
  for (int i=0; i<numLabeledBranches; i++)
  {
    // top level: LabeledRefinementBranch, vertices
    // LabeledRefinementBranch has two entries: RefinementBranch, labels
    int numLabels;
    memcpy(&numLabels, dataLocation, sizeof(numLabels));
    dataLocation += sizeof(numLabels);
    
    vector<GlobalIndexType> labels;
    for (int labelOrdinal=0; labelOrdinal<numLabels; labelOrdinal++)
    {
      GlobalIndexType cellID;
      memcpy(&cellID, dataLocation, sizeof(cellID));
      dataLocation += sizeof(cellID);
      labels.push_back(cellID);
    }
    
    // numLevels = numLabels - 1;
    // at each level, RefinementBranch will yield two entries: RefinementPatternKey and childOrdinal
    RefinementBranch refBranch;
    int numLevels = numLabels - 1;
    for (int level=0; level<numLevels; level++)
    {
      RefinementPatternKey key;
      memcpy(&key, dataLocation, sizeof(key));
      dataLocation += sizeof(key);
      
      unsigned childOrdinal;
      memcpy(&childOrdinal, dataLocation, sizeof(childOrdinal));
      dataLocation += sizeof(childOrdinal);
      
      RefinementPatternPtr refPattern = RefinementPattern::refinementPattern(key);
      if (refPattern == Teuchos::null)
      {
        // repeat the call for simple debugging
        refPattern = RefinementPattern::refinementPattern(key);
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementPattern deserialization failed!");
      }
      refBranch.push_back({refPattern.get(), childOrdinal});
    }
    
    // we are able to extract the number of vertices and the spaceDim from RefinementPatternKey and Mesh
    int spaceDim = mesh->getDimension();
    int vertexCount = refBranch[0].first->parentTopology()->getVertexCount();
    vector<vector<double>> rootVertices(vertexCount,vector<double>(spaceDim));
    for (int vertexOrdinal=0; vertexOrdinal < vertexCount; vertexOrdinal++)
    {
      memcpy(&rootVertices[vertexOrdinal][0], dataLocation, spaceDim * sizeof(double));
      dataLocation += spaceDim * sizeof(double);
    }
    RootedLabeledRefinementBranch rootedBranch = {{refBranch,labels},rootVertices};
    cellHaloGeometry.push_back(rootedBranch);
  }
  
  addMigratedGeometry(meshTopo, cellHaloGeometry);
  
  // entity sets
  CellPtr cell = meshTopo->getCell(cellID);
  int spaceDim = mesh->getDimension();
  int numHandles;
  memcpy(&numHandles, dataLocation, sizeof(numHandles));
  dataLocation += sizeof(numHandles);

  for (int handleOrdinal=0; handleOrdinal<numHandles; handleOrdinal++)
  {
    EntityHandle handle;
    memcpy(&handle, dataLocation, sizeof(handle));
    dataLocation += sizeof(handle);
    EntitySetPtr entitySet = meshTopo->getEntitySet(handle);
    for (int d=0; d<spaceDim; d++)
    {
      int numSubcells;
      memcpy(&numSubcells, dataLocation, sizeof(numSubcells));
      dataLocation += sizeof(numSubcells);
      for (int i=0; i<numSubcells; i++)
      {
        unsigned subcord;
        memcpy(&subcord, dataLocation, sizeof(subcord));
        dataLocation += sizeof(subcord);
        IndexType entityIndex = cell->entityIndex(d, subcord);
        entitySet->addEntity(d, entityIndex);
      }
    }
  }
  
  // finally, force parities for the cell and its neighbors to be recomputed:
  mesh->globalDofAssignment()->assignParities(cellID);
  set<GlobalIndexType> neighborIDs = meshTopo->getCell(cellID)->getActiveNeighborIndices(mesh->getTopology());
  for (GlobalIndexType neighborID : neighborIDs)
  {
    mesh->globalDofAssignment()->assignParities(neighborID);
  }
}

void CellDataMigration::unpackSolutionData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size)
{
  int parentDataPackedFlag;
  memcpy(&parentDataPackedFlag, dataLocation, sizeof(parentDataPackedFlag));
  dataLocation += sizeof(parentDataPackedFlag);
  bool coefficientsBelongToParent = parentDataPackedFlag != 0;
  
  const set<GlobalIndexType>* rankLocalCellIDs = &mesh->cellIDsInPartition();
  if (rankLocalCellIDs->find(cellID) == rankLocalCellIDs->end())
  {
    // it may be that when we do ghost cells, this shouldn't be an exception--or maybe the ghost cells will be packed in with the active cell
    cout << "unpackData called for a non-rank-local cellID " << cellID << "\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unpackData called for a non-rank-local cellID");
  }
  ElementTypePtr elemType;
  if (!coefficientsBelongToParent)
  {
    elemType = mesh->getElementType(cellID);
  }
  else
  {
    GlobalIndexType parentCellID = mesh->getTopology()->getCell(cellID)->getParent()->cellIndex();
    elemType = mesh->getElementType(parentCellID);
  }
  vector<TSolutionPtr<double>> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  int numSolutions = solutions.size();
  int numSolutionsPacked;
  memcpy(&numSolutionsPacked, dataLocation, sizeof(numSolutionsPacked));
  if (numSolutions != numSolutionsPacked)
  {
    cout << "numSolutions != numSolutionsPacked.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numSolutions != numSolutionsPacked");
  }
  dataLocation += sizeof(numSolutions);
  for (int solnOrdinal=0; solnOrdinal<numSolutions; solnOrdinal++)
  {
    int numLHSes = solutions[solnOrdinal]->numSolutions();
    for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
    {
      // # dofs per solution
      int localDofs;
      memcpy(&localDofs, dataLocation, sizeof(localDofs));
      dataLocation += sizeof(localDofs);
      if (localDofs==0)
      {
        // no dofs assigned -- proceed to next solution
        continue;
      }
      
      if (localDofs != elemType->trialOrderPtr->totalDofs())
      {
        cout << "localDofs != elemType->trialOrderPtr->totalDofs().\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localDofs != elemType->trialOrderPtr->totalDofs()");
      }
      
      Camellia::FieldContainer<double> solnCoeffs(localDofs);
      if (localDofs > 0)
      {
        memcpy(&solnCoeffs[0], dataLocation, localDofs * sizeof(double));
      }
      // the dofs themselves
      dataLocation += localDofs * sizeof(double);
      
      solutionCoefficients.push_back({{cellID,coefficientsBelongToParent},solnCoeffs});
      //    cout << "setting solution coefficients for cellID " << cellID << endl << solnCoeffs;
    }
  }
}

void CellDataMigration::getGeometry(const MeshTopologyView* meshTopo, MeshGeometryInfo &geometryInfo)
{
  const set<IndexType>* myRootCellIndices = &meshTopo->getRootCellIndicesLocal();
  geometryInfo.rootVertices.resize(myRootCellIndices->size());
  geometryInfo.rootCellIDs.resize(myRootCellIndices->size());
  geometryInfo.rootCellTopos.resize(myRootCellIndices->size());
  geometryInfo.globalCellCount = meshTopo->cellCount();
  geometryInfo.globalActiveCellCount = meshTopo->activeCellCount();
  int rootCellOrdinal = 0;
  ConstMeshTopologyViewPtr meshTopoRCP = Teuchos::rcp(meshTopo,false);
  vector<GlobalIndexType> thisLevelParents;
  for (IndexType rootCellIndex : *myRootCellIndices)
  {
    geometryInfo.rootCellIDs[rootCellOrdinal] = rootCellIndex;
    CellPtr rootCell = meshTopo->getCell(rootCellIndex);
    geometryInfo.rootCellTopos[rootCellOrdinal] = rootCell->topology()->getKey();
    const vector<IndexType>* vertexIndices = &rootCell->vertices();
    int vertexCount = vertexIndices->size();
    geometryInfo.rootVertices[rootCellOrdinal].resize(vertexCount);
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
    {
      IndexType vertexIndex = (*vertexIndices)[vertexOrdinal];
      geometryInfo.rootVertices[rootCellOrdinal][vertexOrdinal] = meshTopo->getVertex(vertexIndex);
    }
    if (rootCell->isParent(meshTopoRCP))
    {
      thisLevelParents.push_back(rootCellIndex);
    }
    rootCellOrdinal++;
  }
  
  geometryInfo.refinementLevels.clear();
  
  while (thisLevelParents.size() > 0)
  {
    vector<IndexType> nextLevelParents;
    RefinementLevel refinementLevel;
    for (GlobalIndexType parentCellID : thisLevelParents)
    {
      CellPtr parentCell = meshTopo->getCell(parentCellID);
      vector<IndexType> childCellIndices = parentCell->getChildIndices(meshTopoRCP);
      GlobalIndexType firstChildID = childCellIndices[0];
      RefinementPatternKey refPatternKey = parentCell->refinementPattern()->getKey();
      refinementLevel[refPatternKey].push_back({parentCellID,firstChildID});
      
      // look to see if any children should be included in the next mesh refinement level:
      for (IndexType childCellID : childCellIndices)
      {
        if (meshTopo->isValidCellIndex(childCellID))
        {
          CellPtr childCell = meshTopo->getCell(childCellID);
          if (childCell->isParent(meshTopoRCP))
          {
            nextLevelParents.push_back(childCellID);
          }
        }
      }
    }
    thisLevelParents = nextLevelParents;
    geometryInfo.refinementLevels.push_back(refinementLevel);
  }
  
  geometryInfo.myCellIDs.clear();
  const set<IndexType>* myCellIDs = &meshTopo->getMyActiveCellIndices();
  geometryInfo.myCellIDs.insert(geometryInfo.myCellIDs.begin(), myCellIDs->begin(), myCellIDs->end());
  
  // TODO: treat periodic BCs
}

int CellDataMigration::getGeometryDataSize(const MeshGeometryInfo &geometryInfo)
{
  int dataSize = 0;
  dataSize += sizeof(geometryInfo.globalCellCount);
  dataSize += sizeof(geometryInfo.globalActiveCellCount);
  int rootCellCount = geometryInfo.rootVertices.size();
  dataSize += sizeof(rootCellCount);
  for (int rootCellOrdinal=0; rootCellOrdinal<rootCellCount; rootCellOrdinal++)
  {
    dataSize += sizeof(geometryInfo.rootCellIDs[rootCellOrdinal]);
    dataSize += sizeof(geometryInfo.rootCellTopos[rootCellOrdinal]);
    int vertexCount = geometryInfo.rootVertices[rootCellOrdinal].size();
    dataSize += sizeof(vertexCount);
    int spaceDim = CellTopology::cellTopology(geometryInfo.rootCellTopos[rootCellOrdinal])->getDimension();
    dataSize += vertexCount * sizeof(double) * spaceDim;
  }
  
  int numLevels = geometryInfo.refinementLevels.size();
  dataSize += sizeof(numLevels);
  for (int level=0; level<numLevels; level++)
  {
    int numRefTypes = geometryInfo.refinementLevels[level].size();
    dataSize += sizeof(numRefTypes);
    for (auto entry : geometryInfo.refinementLevels[level])
    {
      dataSize += sizeof(entry.first); // sizeof(RefinementPatternKey)
      int numRefinements = entry.second.size();
      dataSize += sizeof(numRefinements);
      dataSize += numRefinements * sizeof(pair<GlobalIndexType,GlobalIndexType>);
    }
  }
  
  int numMyCellIDs = geometryInfo.myCellIDs.size();
  dataSize += sizeof(numMyCellIDs);
  dataSize += sizeof(GlobalIndexType) * numMyCellIDs;
  
  // TODO: treat periodic BCs
  return dataSize;
}

void CellDataMigration::writeGeometryData(const MeshGeometryInfo &geometryInfo, char* &dataLocation, int bufferSize)
{
  int minSize = getGeometryDataSize(geometryInfo);
  TEUCHOS_TEST_FOR_EXCEPTION(bufferSize < minSize, std::invalid_argument, "buffer is too small!");
  
  WRITE_ADVANCE(dataLocation, geometryInfo.globalCellCount);
  WRITE_ADVANCE(dataLocation, geometryInfo.globalActiveCellCount);
  
  int rootCellCount = geometryInfo.rootVertices.size();
  WRITE_ADVANCE(dataLocation, rootCellCount);
  
  for (int rootCellOrdinal=0; rootCellOrdinal<rootCellCount; rootCellOrdinal++)
  {
    WRITE_ADVANCE(dataLocation, geometryInfo.rootCellIDs[rootCellOrdinal]);
    WRITE_ADVANCE(dataLocation, geometryInfo.rootCellTopos[rootCellOrdinal]);
    int vertexCount = geometryInfo.rootVertices[rootCellOrdinal].size();
    WRITE_ADVANCE(dataLocation, vertexCount);
    int spaceDim = CellTopology::cellTopology(geometryInfo.rootCellTopos[rootCellOrdinal])->getDimension();
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        WRITE_ADVANCE(dataLocation, geometryInfo.rootVertices[rootCellOrdinal][vertexOrdinal][d]);
      }
    }
  }
  
  int numLevels = geometryInfo.refinementLevels.size();
  WRITE_ADVANCE(dataLocation, numLevels);
  
  for (int level=0; level<numLevels; level++)
  {
    int numRefTypes = geometryInfo.refinementLevels[level].size();
    WRITE_ADVANCE(dataLocation, numRefTypes);
    for (auto entry : geometryInfo.refinementLevels[level])
    {
      RefinementPatternKey refPatternKey = entry.first;
      WRITE_ADVANCE(dataLocation, refPatternKey);
      int numRefinements = entry.second.size();
      WRITE_ADVANCE(dataLocation, numRefinements);
      for (int refinementOrdinal=0; refinementOrdinal<numRefinements; refinementOrdinal++)
      {
        WRITE_ADVANCE(dataLocation, entry.second[refinementOrdinal]);
      }
    }
  }
  
  int numMyCellIDs = geometryInfo.myCellIDs.size();
  WRITE_ADVANCE(dataLocation, numMyCellIDs);
  WRITE_ADVANCE_ARRAY(dataLocation, geometryInfo.myCellIDs);
  
    // TODO: treat periodic BCs
}

void CellDataMigration::readGeometryData(const char* &dataLocation, int bufferSize, MeshGeometryInfo &geometryInfo)
{
  const char* startingDataLocation = dataLocation;
  
  READ_ADVANCE(dataLocation, geometryInfo.globalCellCount);
  READ_ADVANCE(dataLocation, geometryInfo.globalActiveCellCount);
  
  int rootCellCount;
  READ_ADVANCE(dataLocation, rootCellCount);
  geometryInfo.rootCellIDs.resize(rootCellCount);
  geometryInfo.rootCellTopos.resize(rootCellCount);
  geometryInfo.rootVertices.resize(rootCellCount);
  
  for (int rootCellOrdinal=0; rootCellOrdinal<rootCellCount; rootCellOrdinal++)
  {
    READ_ADVANCE(dataLocation, geometryInfo.rootCellIDs[rootCellOrdinal]);
    READ_ADVANCE(dataLocation, geometryInfo.rootCellTopos[rootCellOrdinal]);
    int vertexCount;
    READ_ADVANCE(dataLocation, vertexCount);
    geometryInfo.rootVertices[rootCellOrdinal].resize(vertexCount);
    int spaceDim = CellTopology::cellTopology(geometryInfo.rootCellTopos[rootCellOrdinal])->getDimension();
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
    {
      geometryInfo.rootVertices[rootCellOrdinal][vertexOrdinal].resize(spaceDim);
      READ_ADVANCE_ARRAY(dataLocation, geometryInfo.rootVertices[rootCellOrdinal][vertexOrdinal]);
    }
  }
  
  int numLevels;
  READ_ADVANCE(dataLocation, numLevels);
  geometryInfo.refinementLevels.resize(numLevels);
  
  for (int level=0; level<numLevels; level++)
  {
    int numRefTypes;
    READ_ADVANCE(dataLocation, numRefTypes);
    for (int refTypeOrdinal=0; refTypeOrdinal<numRefTypes; refTypeOrdinal++)
    {
      RefinementPatternKey refPatternKey;
      READ_ADVANCE(dataLocation, refPatternKey);
      int numRefinements;
      READ_ADVANCE(dataLocation, numRefinements);
      geometryInfo.refinementLevels[level][refPatternKey].resize(numRefinements);
      for (int refinementOrdinal=0; refinementOrdinal<numRefinements; refinementOrdinal++)
      {
        READ_ADVANCE(dataLocation,geometryInfo.refinementLevels[level][refPatternKey][refinementOrdinal]);
      }
    }
  }
  
  int numMyCellIDs;
  READ_ADVANCE(dataLocation, numMyCellIDs);
  geometryInfo.myCellIDs.resize(numMyCellIDs);
  READ_ADVANCE_ARRAY(dataLocation, geometryInfo.myCellIDs);

  // TODO: treat periodic BCs
  
  // have we read the whole buffer?
  int remainingBufferSize = bufferSize - (dataLocation - startingDataLocation);
  TEUCHOS_TEST_FOR_EXCEPTION(remainingBufferSize < 0, std::invalid_argument, "Read past the edge of the buffer!!");
  bool bufferExhausted = (remainingBufferSize == 0);
  
  if (!bufferExhausted)
  {
    // if not, then there must be another entry to read.  First read, then merge:
    MeshGeometryInfo otherGeometryInfo;
    readGeometryData(dataLocation, remainingBufferSize, otherGeometryInfo);
    
    // *******  MERGE  *******
    // sanity check: the two geometry info objects should agree on global counts
    TEUCHOS_TEST_FOR_EXCEPTION(geometryInfo.globalCellCount != otherGeometryInfo.globalCellCount, std::invalid_argument, "Two geometry chunks differ on globalCellCount");
    TEUCHOS_TEST_FOR_EXCEPTION(geometryInfo.globalActiveCellCount != otherGeometryInfo.globalActiveCellCount, std::invalid_argument, "Two geometry chunks differ on globalActiveCellCount");
    
    // merge root cells:
    int otherRootCellOrdinal = 0;
    for (GlobalIndexType otherRootCellID : otherGeometryInfo.rootCellIDs)
    {
      if (std::find(geometryInfo.rootCellIDs.begin(), geometryInfo.rootCellIDs.end(), otherRootCellID) == geometryInfo.rootCellIDs.end())
      {
        auto lowerIt = std::lower_bound(geometryInfo.rootCellIDs.begin(), geometryInfo.rootCellIDs.end(), otherRootCellID);
        int insertLocation = lowerIt - geometryInfo.rootCellIDs.begin();
        geometryInfo.rootCellIDs.insert(geometryInfo.rootCellIDs.begin() + insertLocation, otherRootCellID);
        geometryInfo.rootCellTopos.insert(geometryInfo.rootCellTopos.begin() + insertLocation, otherGeometryInfo.rootCellTopos[otherRootCellOrdinal]);
        geometryInfo.rootVertices.insert(geometryInfo.rootVertices.begin() + insertLocation, otherGeometryInfo.rootVertices[otherRootCellOrdinal]);
      }
      
      otherRootCellOrdinal++;
    }
    
    // merge refinement levels:
    int totalLevels = max(geometryInfo.refinementLevels.size(),otherGeometryInfo.refinementLevels.size());
    for (int level=0; level<totalLevels; level++)
    {
      if (level >= geometryInfo.refinementLevels.size())
      {
        // then copy otherGeometryInfo for this level to geometryInfo
        geometryInfo.refinementLevels.push_back(otherGeometryInfo.refinementLevels[level]);
      }
      else if (level >= otherGeometryInfo.refinementLevels.size())
      {
        // then geometryInfo has the right information
      }
      else
      {
        // collect all refinement pattern keys in the two geometry objects
        set<RefinementPatternKey> refKeys;
        for (auto entry : geometryInfo.refinementLevels[level])
        {
          refKeys.insert(entry.first);
        }
        for (auto entry : otherGeometryInfo.refinementLevels[level])
        {
          refKeys.insert(entry.first);
        }
        for (RefinementPatternKey refKey : refKeys)
        {
          if (otherGeometryInfo.refinementLevels[level].find(refKey) == otherGeometryInfo.refinementLevels[level].end())
          {
            // then geometryInfo is already correct for this refKey
          }
          else if (geometryInfo.refinementLevels[level].find(refKey) == geometryInfo.refinementLevels[level].end())
          {
            // otherGeometryInfo is correct:
            geometryInfo.refinementLevels[level][refKey] = otherGeometryInfo.refinementLevels[level][refKey];
          }
          else
          {
            // we actually need to merge
            // easy way to do the merge is using set
            set<pair<GlobalIndexType,GlobalIndexType>> refinements;
            for (pair<GlobalIndexType,GlobalIndexType> refinement : geometryInfo.refinementLevels[level][refKey])
            {
              refinements.insert(refinement);
            }
            for (pair<GlobalIndexType,GlobalIndexType> refinement : otherGeometryInfo.refinementLevels[level][refKey])
            {
              refinements.insert(refinement);
            }
            vector<pair<GlobalIndexType,GlobalIndexType>> refinementsVector(refinements.begin(),refinements.end());
            geometryInfo.refinementLevels[level][refKey] = refinementsVector;
          }
        }
      }
    }
    
    // merge active cells:
    // concatenate:
    geometryInfo.myCellIDs.insert(geometryInfo.myCellIDs.begin(),otherGeometryInfo.myCellIDs.begin(),otherGeometryInfo.myCellIDs.end());
    // sort: (may not be important, but it seems a little cleaner)
    std::sort(geometryInfo.myCellIDs.begin(), geometryInfo.myCellIDs.end());
    // *******  END MERGE  *******
  }
}
