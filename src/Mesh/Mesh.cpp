// @HEADER
//
// Original Version Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

/*
 *  Mesh.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "Mesh.h"
#include "BilinearFormUtility.h"
#include "ElementType.h"
#include "DofOrderingFactory.h"
#include "BasisFactory.h"
#include "BasisCache.h"

#include "Solution.h"

#include "MeshTransformationFunction.h"

#include "CamelliaCellTools.h"

#include "GlobalDofAssignment.h"

#include "GDAMinimumRule.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include <Teuchos_GlobalMPISession.hpp>

#include "MeshFactory.h"

using namespace Intrepid;

map<int,int> Mesh::_emptyIntIntMap;

Mesh::Mesh(MeshTopologyPtr meshTopology, BilinearFormPtr bilinearForm, int H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements) {
  
  _meshTopology = meshTopology;
  
  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  
  _gda = Teuchos::rcp( new GDAMinimumRule(_meshTopology, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  
  setBilinearForm(bilinearForm);
  _boundary.setMesh(this);
}

Mesh::Mesh(const vector<vector<double> > &vertices, vector< vector<unsigned> > &elementVertices,
           Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAddTest, bool useConformingTraces,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements, vector<PeriodicBCPtr> periodicBCs) {

  cout << "in legacy mesh constructor, periodicBCs size is " << periodicBCs.size() << endl;
  
  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices) );
  _meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry, periodicBCs) );
  
  DofOrderingFactoryPtr dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  
  _gda = Teuchos::rcp( new GDAMaximumRule2D(_meshTopology, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                            partitionPolicy, H1Order, pToAddTest, _enforceMBFluxContinuity) );
  
  setBilinearForm(bilinearForm);
  
  _useConformingTraces = useConformingTraces;
  _usePatchBasis = false;

  // DEBUGGING: check how we did:
  int numVertices = vertices.size();
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++ ) {
    vector<double> vertex = _meshTopology->getVertex(vertexIndex);

    unsigned assignedVertexIndex;
    bool vertexFound = _meshTopology->getVertexIndex(vertex, assignedVertexIndex);
    
    if (!vertexFound) {
      cout << "INTERNAL ERROR: vertex not found by vertex lookup.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error");
    }
    
    if (assignedVertexIndex != vertexIndex) {
      cout << "INTERNAL ERROR: assigned vertex index is incorrect.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error");
    }
  }
  
  _boundary.setMesh(this);
  
  _pToAddToTest = pToAddTest;
}

GlobalIndexType Mesh::numInitialElements(){
  return _meshTopology->getRootCellIndices().size();
}

GlobalIndexType Mesh::activeCellOffset() {
  return _gda->activeCellOffset();
}

vector< ElementPtr > Mesh::activeElements() {
  set< IndexType > activeCellIndices = _meshTopology->getActiveCellIndices();
  
  vector< ElementPtr > activeElements;
  
  for (set< IndexType >::iterator cellIt = activeCellIndices.begin(); cellIt != activeCellIndices.end(); cellIt++) {
    activeElements.push_back(getElement(*cellIt));
  }
  
  return activeElements;
}

ElementPtr Mesh::ancestralNeighborForSide(ElementPtr elem, int sideIndex, int &elemSideIndexInNeighbor) {
  CellPtr cell = _meshTopology->getCell(elem->cellID());
  pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighbor(sideIndex);
  elemSideIndexInNeighbor = neighborInfo.second;
  
  if (neighborInfo.first == -1) return Teuchos::rcp( (Element*) NULL );
  
  return getElement(neighborInfo.first);
}

BilinearFormPtr Mesh::bilinearForm() {
  return _bilinearForm;
}

void Mesh::setBilinearForm( BilinearFormPtr bf) {
  // must match the original in terms of variable IDs, etc...
  _bilinearForm = bf;
}

Boundary & Mesh::boundary() {
  return _boundary; 
}

GlobalIndexType Mesh::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber) {
  return _gda->cellID(elemTypePtr, cellIndex, partitionNumber);
}

vector< GlobalIndexType > Mesh::cellIDsOfType(ElementTypePtr elemType) {
  int rank = Teuchos::GlobalMPISession::getRank();
  return cellIDsOfType(rank,elemType);
}

vector< GlobalIndexType > Mesh::cellIDsOfType(int partitionNumber, ElementTypePtr elemTypePtr) {
  // returns the cell IDs for a given partition and element type
  return _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
}

vector< GlobalIndexType > Mesh::cellIDsOfTypeGlobal(ElementTypePtr elemTypePtr) {
  vector< GlobalIndexType > cellIDs;
  int partititionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partititionCount; partitionNumber++) {
    vector< GlobalIndexType > cellIDsForType = cellIDsOfType(partitionNumber, elemTypePtr);
    cellIDs.insert(cellIDs.end(), cellIDsForType.begin(), cellIDsForType.end());
  }
  return cellIDs;
}

int Mesh::cellPolyOrder(GlobalIndexType cellID) { // aka H1Order
  return _gda->getH1Order(cellID);
}

bool Mesh::cellContainsPoint(GlobalIndexType cellID, vector<double> &point) {
  // note that this design, with a single point being passed in, will be quite inefficient
  // if there are many points.  TODO: revise to allow multiple points (returning vector<bool>, maybe)
  int numCells = 1, numPoints = 1, spaceDim = _meshTopology->getSpaceDim();
  FieldContainer<double> physicalPoints(numCells,numPoints,spaceDim);
  for (int d=0; d<spaceDim; d++) {
    physicalPoints(0,0,d) = point[d];
  }
  //  cout << "cell " << elem->cellID() << ": (" << x << "," << y << ") --> ";
  FieldContainer<double> refPoints(numCells,numPoints,spaceDim);
  MeshPtr thisPtr = Teuchos::rcp(this,false);
  CamelliaCellTools::mapToReferenceFrame(refPoints, physicalPoints, thisPtr, cellID);
  
  CellTopoPtr cellTopo = _meshTopology->getCell(cellID)->topology();
  
  int result = CellTools<double>::checkPointInclusion(&refPoints[0], spaceDim, *cellTopo);
  return result == 1;
}

vector<ElementPtr> Mesh::elementsForPoints(const FieldContainer<double> &physicalPoints) {
  // returns a vector of an active element per point, or null if there is no element including that point
  vector<ElementPtr> elemsForPoints;
//  cout << "entered elementsForPoints: \n" << physicalPoints;
  int numPoints = physicalPoints.dimension(0);
  // TODO: work out what to do here for 3D
  // figure out the last element of the original mesh:

  set<GlobalIndexType> rootCellIndices = _meshTopology->getRootCellIndices();
  
  // NOTE: the above does depend on the domain of the mesh remaining fixed after refinements begin.
  
  for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
    double x = physicalPoints(pointIndex,0);
    double y = physicalPoints(pointIndex,1);
    vector<double> point(2);
    point[0] = x;
    point[1] = y;
    
    // find the element from the original mesh that contains this point
    ElementPtr elem;
    for (set<GlobalIndexType>::iterator cellIt = rootCellIndices.begin(); cellIt != rootCellIndices.end(); cellIt++) {
      GlobalIndexType cellID = *cellIt;
      if (cellContainsPoint(cellID,point)) {
        elem = getElement(cellID);
        break;
      }
    }
    if (elem.get() != NULL) {
      while ( elem->isParent() ) {
        int numChildren = elem->numChildren();
        bool foundMatchingChild = false;
        for (int childIndex = 0; childIndex < numChildren; childIndex++) {
          ElementPtr child = elem->getChild(childIndex);
          if ( cellContainsPoint(child->cellID(),point) ) {
            elem = child;
            foundMatchingChild = true;
            break;
          }
        }
        if (!foundMatchingChild) {
          cout << "parent matches, but none of its children do... will return nearest cell centroid\n";
          int numVertices = elem->numSides();
          int spaceDim = 2;
          FieldContainer<double> vertices(numVertices,spaceDim);
          verticesForCell(vertices, elem->cellID());
          cout << "parent vertices:\n" << vertices;
          double minDistance = numeric_limits<double>::max();
          int childSelected = -1;
          for (int childIndex = 0; childIndex < numChildren; childIndex++) {
            ElementPtr child = elem->getChild(childIndex);
            verticesForCell(vertices, child->cellID());
            cout << "child " << childIndex << ", vertices:\n" << vertices;
            vector<double> cellCentroid = getCellCentroid(child->cellID());
            double d = sqrt((cellCentroid[0] - x) * (cellCentroid[0] - x) + (cellCentroid[1] - y) * (cellCentroid[1] - y));
            if (d < minDistance) {
              minDistance = d;
              childSelected = childIndex;
            }
          }
          elem = elem->getChild(childSelected);
        }
      }
    }
    elemsForPoints.push_back(elem);
  }
//  cout << "Returning from elementsForPoints\n";
  return elemsForPoints;
}

void Mesh::enforceOneIrregularity() {
  bool meshIsNotRegular = true; // assume it's not regular and check elements
  while (meshIsNotRegular) {
    vector <GlobalIndexType> irregularTriangleCells;
    vector <GlobalIndexType> irregularQuadCells;
    vector< Teuchos::RCP< Element > > newActiveElements = activeElements();
    vector< Teuchos::RCP< Element > >::iterator newElemIt;
    
    for (newElemIt = newActiveElements.begin(); newElemIt != newActiveElements.end(); newElemIt++) {
      Teuchos::RCP< Element > current_element = *(newElemIt);
      bool isIrregular = false;
      for (int sideIndex=0; sideIndex < current_element->numSides(); sideIndex++) {
        int mySideIndexInNeighbor;
        ElementPtr neighbor = current_element->getNeighbor(mySideIndexInNeighbor, sideIndex);
        if (neighbor.get() != NULL) {
          int numNeighborsOnSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor).size();
          if (numNeighborsOnSide > 2) isIrregular=true;
        }
      }
      
      if (isIrregular){
        if ( 3 == current_element->numSides() ) {
          irregularTriangleCells.push_back(current_element->cellID());
        }
        else if (4 == current_element->numSides() ) {
          irregularQuadCells.push_back(current_element->cellID());
        }
      }
    }
    if ((irregularQuadCells.size()>0) || (irregularTriangleCells.size()>0)) {
      hRefine(irregularTriangleCells,RefinementPattern::regularRefinementPatternTriangle());
      hRefine(irregularQuadCells,RefinementPattern::regularRefinementPatternQuad());
      irregularTriangleCells.clear();
      irregularQuadCells.clear();
    } else {
      meshIsNotRegular=false;
    }
  }
}

FieldContainer<double> Mesh::cellSideParities( ElementTypePtr elemTypePtr ) {
  // old version (using lookup table)
  // return dynamic_cast<GDAMaximumRule2D*>(_gda.get())->cellSideParities(elemTypePtr);
  
  // new implementation below:
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);
  
  int numCells = cellIDs.size();
  int numSides = elemTypePtr->cellTopoPtr->getSideCount();
  
  FieldContainer<double> sideParities(numCells, numSides);
  for (int i=0; i<numCells; i++) {
    FieldContainer<double> iParities = cellSideParitiesForCell(cellIDs[i]);
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
      sideParities(i,sideOrdinal) = iParities(0,sideOrdinal);
    }
  }
  
  return sideParities;
}

FieldContainer<double> Mesh::cellSideParitiesForCell( GlobalIndexType cellID ) {
  return _gda->cellSideParitiesForCell(cellID);
}

vector<double> Mesh::getCellCentroid(GlobalIndexType cellID){
  return _meshTopology->getCellCentroid(cellID);
}

vector< ElementPtr > Mesh::elementsInPartition(PartitionIndexType partitionNumber){
  vector< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(partitionNumber);
  vector< ElementPtr > elements;
  for (vector< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementPtr element = getElement(cellID);
    elements.push_back(element);
  }
  return elements;
}

vector< ElementPtr > Mesh::elementsOfType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr) {
  // returns the elements for a given partition and element type
  vector< ElementPtr > elementsOfType;
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
  for (vector<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    elementsOfType.push_back(getElement(*cellIt));
  }
  return elementsOfType;
}

vector< ElementPtr > Mesh::elementsOfTypeGlobal(ElementTypePtr elemTypePtr) {
  vector< ElementPtr > elementsOfTypeVector;
  int partitionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partitionCount; partitionNumber++) {
    vector< ElementPtr > elementsOfTypeForPartition = elementsOfType(partitionNumber,elemTypePtr);
    elementsOfTypeVector.insert(elementsOfTypeVector.end(),elementsOfTypeForPartition.begin(),elementsOfTypeForPartition.end());
  }
  return elementsOfTypeVector;
}

vector< ElementTypePtr > Mesh::elementTypes(PartitionIndexType partitionNumber) {
  return _gda->elementTypes(partitionNumber);
}

set<GlobalIndexType> Mesh::getActiveCellIDs() {
  set<IndexType> activeCellIndices = _meshTopology->getActiveCellIndices();
  set<GlobalIndexType> activeCellIDs(activeCellIndices.begin(), activeCellIndices.end());
  return activeCellIDs;
}

int Mesh::getDimension() {
  return _meshTopology->getSpaceDim();
}

DofOrderingFactory & Mesh::getDofOrderingFactory() {
  return *_gda->getDofOrderingFactory().get();
}

ElementPtr Mesh::getElement(GlobalIndexType cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  
  ElementTypePtr elemType = _gda->elementType(cellID);
  
  IndexType cellIndex = _gda->partitionLocalCellIndex(cellID);
  
  GlobalIndexType globalCellIndex = _gda->globalCellIndex(cellID);
  
  ElementPtr element = Teuchos::rcp( new Element(this, cellID, elemType, cellIndex, globalCellIndex) );
  
  return element;
}

ElementTypePtr Mesh::getElementType(GlobalIndexType cellID) {
  return _gda->elementType(cellID);
}

ElementTypeFactory & Mesh::getElementTypeFactory() {
  return _gda->getElementTypeFactory();
}

GlobalIndexType Mesh::getVertexIndex(double x, double y, double tol) {
  vector<double> vertex;
  vertex.push_back(x);
  vertex.push_back(y);
  
  IndexType vertexIndex; // distributed mesh will need to use some sort of offset...
  if (! _meshTopology->getVertexIndex(vertex, vertexIndex) ) {
    return -1;
  } else {
    return vertexIndex;
  }
}

const map< pair<GlobalIndexType,IndexType>, GlobalIndexType>& Mesh::getLocalToGlobalMap() {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "getLocalToGlobalMap only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getLocalToGlobalMap only supported for max rule.");
  }
  return maxRule->getLocalToGlobalMap();
}

map<IndexType, GlobalIndexType> Mesh::getGlobalVertexIDs(const FieldContainer<double> &vertices) {
  double tol = 1e-12; // tolerance for vertex equality
  
  map<IndexType, GlobalIndexType> localToGlobalVertexIndex;
  int numVertices = vertices.dimension(0);
  for (int i=0; i<numVertices; i++) {
    localToGlobalVertexIndex[i] = getVertexIndex(vertices(i,0), vertices(i,1),tol);
  }
  return localToGlobalVertexIndex;
}

FunctionPtr Mesh::getTransformationFunction() {
  // will be NULL for meshes without edge curves defined
  
  // for now, we recompute the transformation function each time the edge curves get updated
  // we might later want to do something lazier, updating/creating it here if it's out of date
  
  return _meshTopology->transformationFunction();
}

GlobalDofAssignmentPtr Mesh::globalDofAssignment() {
  return _gda;
}

GlobalIndexType Mesh::globalDofCount() {
  return numGlobalDofs(); // TODO: eliminate numGlobalDofs in favor of globalDofCount
}

GlobalIndexType Mesh::globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "globalDofIndex lookup only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndex lookup only supported for max rule.");
  }
  return maxRule->globalDofIndex(cellID, localDofIndex);
}

set<GlobalIndexType> Mesh::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  return _gda->globalDofIndicesForPartition(partitionNumber);
}

//void Mesh::hRefine(vector<GlobalIndexType> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
//  hRefine(cellIDs,refPattern,vector< Teuchos::RCP<Solution> >()); 
//}

void Mesh::hRefine(const vector<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  set<GlobalIndexType> cellSet(cellIDs.begin(),cellIDs.end());
  hRefine(cellSet,refPattern);
}

void Mesh::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  // refine any registered meshes
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    (*meshIt)->hRefine(cellIDs,refPattern);
  }
  
  set<GlobalIndexType>::const_iterator cellIt;
  
  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    
    _meshTopology->refineCell(cellID, refPattern);

    set<GlobalIndexType> cellIDset;
    cellIDset.insert(cellID);
  
    _gda->didHRefine(cellIDset);
    
    // let transformation function know about the refinement that just took place
    if (_meshTopology->transformationFunction().get()) {
      _meshTopology->transformationFunction()->didHRefine(cellIDset);
    }
  }
  _gda->rebuildLookups();
  _boundary.buildLookupTables();
}

void Mesh::hUnrefine(const set<GlobalIndexType> &cellIDs) {
  // refine any registered meshes
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    (*meshIt)->hUnrefine(cellIDs);
  }
  
//  set<GlobalIndexType>::const_iterator cellIt;
//  set< pair<GlobalIndexType, int> > affectedNeighborSides; // (cellID, sideIndex)
//  set< GlobalIndexType > deletedCellIDs;
//  
//  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
//    GlobalIndexType cellID = *cellIt;
//    ElementPtr elem = getElement(cellID);
//    elem->deleteChildrenFromMesh(affectedNeighborSides, deletedCellIDs);
//  }
//  
//  set<int> affectedNeighbors;
//  // for each nullified neighbor relationship, need to figure out the correct element type...
//  for ( set< pair<int, int> >::iterator neighborIt = affectedNeighborSides.begin();
//       neighborIt != affectedNeighborSides.end(); neighborIt++) {
//    ElementPtr elem = _elements[ neighborIt->first ];
//    if (elem->isActive()) {
//      matchNeighbor( elem, neighborIt->second );
//    }
//  }
//  
//  // delete any boundary entries for deleted elements
//  for (set<int>::iterator cellIt = deletedCellIDs.begin(); cellIt != deletedCellIDs.end(); cellIt++) {
//    int cellID = *cellIt;
//    ElementPtr elem = _elements[cellID];
//    for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
//      // boundary allows us to delete even combinations that weren't there to begin with...
//      _boundary.deleteElement(cellID, sideIndex);
//    }
//  }
  
//  // add in any new boundary elements:
//  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
//    int cellID = *cellIt;
//    ElementPtr elem = _elements[cellID];
//    if (elem->isActive()) {
//      int elemSideIndexInNeighbor;
//      for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
//        if (ancestralNeighborForSide(elem, sideIndex, elemSideIndexInNeighbor)->cellID() == -1) {
//          // boundary
//          _boundary.addElement(cellID, sideIndex);
//        }
//      }
//    }
//  }

//  // added by Jesse to try to fix bug
//  for (set<int>::iterator cellIt = deletedCellIDs.begin(); cellIt != deletedCellIDs.end(); cellIt++) {
//    // erase from _elements list
//    for (int i = 0; i<_elements.size();i++){
//      if (_elements[i]->cellID()==(*cellIt)){
//        _elements.erase(_elements.begin()+i);
//        break;
//      }
//    }
//    // erase any pairs from _edgeToCellIDs having to do with deleted cellIDs
//    for (map<pair<int,int>, vector<pair<int,int> > >::iterator mapIt = _edgeToCellIDs.begin(); mapIt!=_edgeToCellIDs.end();mapIt++){
//      vector<pair<int,int> > cellIDSideIndices = mapIt->second;
//      bool eraseEntry = false;
//      for (int i = 0;i<cellIDSideIndices.size();i++){
//        int cellID = cellIDSideIndices[i].first;
//        if (cellID==(*cellIt)){
//          eraseEntry = true;
//        }
//        if (eraseEntry)
//          break;
//      }
//      if (eraseEntry){
//        _edgeToCellIDs.erase(mapIt);
////        cout << "deleting edge to cell entry " << mapIt->first.first << " --> " << mapIt->first.second << endl;
//      }
//    }
//  }
  
  _gda->didHUnrefine(cellIDs);
  _gda->rebuildLookups();
  _boundary.buildLookupTables();
}

void Mesh::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients, const Epetra_Vector &globalCoefficients) {
  _gda->interpretGlobalCoefficients(cellID, localCoefficients, globalCoefficients);
}

void Mesh::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                   FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) {
  _gda->interpretLocalBasisCoefficients(cellID, varID, sideOrdinal, basisCoefficients, globalCoefficients, globalDofIndices);
}

void Mesh::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                              FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) {
  _gda->interpretLocalData(cellID, localDofs, globalDofs, globalDofIndices);
}

GlobalIndexType Mesh::numActiveElements() {
  return _meshTopology->activeCellCount();
}

GlobalIndexType Mesh::numElements() {
  return _meshTopology->cellCount();
}

GlobalIndexType Mesh::numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr ) {
  // returns the global total (across all MPI nodes)
  int numElements = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber<partitionCount; partitionNumber++) {
    numElements += _gda->cellIDsOfElementType(partitionNumber, elemTypePtr).size();
  }
  return numElements;
}

GlobalIndexType Mesh::numFluxDofs(){
  return numGlobalDofs()-numFieldDofs();
}

GlobalIndexType Mesh::numFieldDofs(){
  GlobalIndexType numFieldDofs = 0;
  set<GlobalIndexType> activeCellIDs = getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementTypePtr elemTypePtr = _gda->elementType(cellID);
    vector< int > fieldIDs = _bilinearForm->trialVolumeIDs();
    vector< int >::iterator fieldIDit;
    for (fieldIDit = fieldIDs.begin(); fieldIDit != fieldIDs.end() ; fieldIDit++){    
      int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(*fieldIDit,0);
      numFieldDofs += numDofs;
    }
  }
  return numFieldDofs;
}

GlobalIndexType Mesh::numGlobalDofs() {
  return _gda->globalDofCount();
}

int Mesh::parityForSide(GlobalIndexType cellID, int sideOrdinal) {
  int parity = _gda->cellSideParitiesForCell(cellID)[sideOrdinal];
  return parity;
}

PartitionIndexType Mesh::partitionForCellID( GlobalIndexType cellID ) {
  return _gda->partitionForCellID(cellID);
}

PartitionIndexType Mesh::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  return _gda->partitionForGlobalDofIndex(globalDofIndex);
}

GlobalIndexType Mesh::partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "partitionLocalIndexForGlobalDofIndex only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "partitionLocalIndexForGlobalDofIndex only supported for max rule.");
  }
  return maxRule->partitionLocalIndexForGlobalDofIndex(globalDofIndex);
}

FieldContainer<double> Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr) {
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);
  
  return physicalCellNodes(elemTypePtr, cellIDs);
}

FieldContainer<double> Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr, vector<GlobalIndexType> &cellIDs ) {
  int numCells = cellIDs.size();
  int numVertices = elemTypePtr->cellTopoPtr->getVertexCount();
  int spaceDim = _meshTopology->getSpaceDim();
  
  FieldContainer<double> physicalNodes(numCells, numVertices, spaceDim);
  for (int i=0; i<numCells; i++) {
    FieldContainer<double> iPhysicalNodes = physicalCellNodesForCell(cellIDs[i]);
    for (int vertexOrdinal=0; vertexOrdinal<numVertices; vertexOrdinal++) {
      for (int d=0; d<spaceDim; d++) {
        physicalNodes(i,vertexOrdinal,d) = iPhysicalNodes(0,vertexOrdinal,d);
      }
    }
  }
  return physicalNodes;
}

FieldContainer<double> Mesh::physicalCellNodesForCell( GlobalIndexType cellID ) {
  CellPtr cell = _meshTopology->getCell(cellID);
  int vertexCount = cell->topology()->getVertexCount();
  int spaceDim = _meshTopology->getSpaceDim();
  int numCells = 1;
  FieldContainer<double> physicalCellNodes(numCells,vertexCount,spaceDim);
  
  FieldContainer<double> cellVertices(vertexCount,spaceDim);
  vector<unsigned> vertexIndices = _meshTopology->getCell(cellID)->vertices();
  for (int vertex=0; vertex<vertexCount; vertex++) {
    unsigned vertexIndex = vertexIndices[vertex];
    for (int i=0; i<spaceDim; i++) {
      physicalCellNodes(0,vertex,i) = _meshTopology->getVertex(vertexIndex)[i];
    }
  }
  return physicalCellNodes;
}

FieldContainer<double> Mesh::physicalCellNodesGlobal( Teuchos::RCP< ElementType > elemTypePtr ) {
  int numRanks = Teuchos::GlobalMPISession::getNProc();
  
  vector<GlobalIndexType> globalCellIDs;
  for (int rank=0; rank<numRanks; rank++) {
    vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);
    globalCellIDs.insert(globalCellIDs.end(), cellIDs.begin(), cellIDs.end());
  }
  
  return physicalCellNodes(elemTypePtr, globalCellIDs);
}

void Mesh::printLocalToGlobalMap() {
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> localToGlobalMap = this->getLocalToGlobalMap();
  
  for (map< pair<GlobalIndexType,IndexType>, GlobalIndexType>::iterator entryIt = localToGlobalMap.begin();
       entryIt != localToGlobalMap.end(); entryIt++) {
    int cellID = entryIt->first.first;
    int localDofIndex = entryIt->first.second;
    int globalDofIndex = entryIt->second;
    cout << "(" << cellID << "," << localDofIndex << ") --> " << globalDofIndex << endl;
  }
}

void Mesh::printVertices() {
  cout << "Vertices:\n";
  unsigned vertexDim = 0;
  unsigned vertexCount = _meshTopology->getEntityCount(vertexDim);
  for (unsigned vertexIndex=0; vertexIndex<vertexCount; vertexIndex++) {
    vector<double> vertex = _meshTopology->getVertex(vertexIndex);
    cout << vertexIndex << ": (" << vertex[0] << ", " << vertex[1] << ")\n";
  }
}

void Mesh::registerObserver(Teuchos::RCP<RefinementObserver> observer) {
  _registeredObservers.push_back(observer);
}

void Mesh::registerSolution(Teuchos::RCP<Solution> solution) {
  _gda->registerSolution(solution.get());
}

void Mesh::unregisterObserver(RefinementObserver* observer) {
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    if ( (*meshIt).get() == observer ) {
      _registeredObservers.erase(meshIt);
      return;
    }
  }
  cout << "WARNING: Mesh::unregisterObserver: Observer not found.\n";
}

void Mesh::unregisterObserver(Teuchos::RCP<RefinementObserver> mesh) {
  this->unregisterObserver(mesh.get());
}

void Mesh::unregisterSolution(Teuchos::RCP<Solution> solution) {
  _gda->unregisterSolution(solution.get());
}

void Mesh::pRefine(const vector<GlobalIndexType> &cellIDsForPRefinements) {
  set<GlobalIndexType> cellSet;
  for (vector<GlobalIndexType>::const_iterator cellIt=cellIDsForPRefinements.begin();
       cellIt != cellIDsForPRefinements.end(); cellIt++) {
    cellSet.insert(*cellIt);
  }
  pRefine(cellSet);
}

void Mesh::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements){
  pRefine(cellIDsForPRefinements,1);
}

void Mesh::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements, int pToAdd) {
  // refine any registered meshes
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    (*meshIt)->pRefine(cellIDsForPRefinements);
  }
  
  _gda->didPRefine(cellIDsForPRefinements, pToAdd);
  
  // let transformation function know about the refinement that just took place
  if (_meshTopology->transformationFunction().get()) {
    _meshTopology->transformationFunction()->didPRefine(cellIDsForPRefinements);
  }
  
  _gda->rebuildLookups();
  _boundary.buildLookupTables();
}

int Mesh::condensedRowSizeUpperBound() {
  // includes multiplicity
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber < partitionCount; partitionNumber++) {
    vector<ElementTypePtr> elementTypes = _gda->elementTypes(partitionNumber);
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
      ElementTypePtr elemTypePtr = *elemTypeIt;
      int numSides = elemTypePtr->cellTopoPtr->getSideCount();
      vector< int > fluxIDs = _bilinearForm->trialBoundaryIDs();
      vector< int >::iterator fluxIDIt;
      int numFluxDofs = 0;
      for (fluxIDIt = fluxIDs.begin(); fluxIDIt != fluxIDs.end(); fluxIDIt++) {
        int fluxID = *fluxIDIt;
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideIndex);
          numFluxDofs += numDofs;
        }
      }
      int maxPossible = numFluxDofs * 2 + numSides*fluxIDs.size();  // a side can be shared by 2 elements, and vertices can be shared
      maxRowSize = max(maxPossible, maxRowSize);
    }
  }
  return maxRowSize;
}

void Mesh::rebuildLookups() {
  _gda->rebuildLookups();
  _boundary.buildLookupTables();
}

int Mesh::rowSizeUpperBound() {
  // includes multiplicity
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber < partitionCount; partitionNumber++) {
    vector<ElementTypePtr> elementTypes = _gda->elementTypes(partitionNumber);
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
      ElementTypePtr elemTypePtr = *elemTypeIt;
      int numSides = elemTypePtr->cellTopoPtr->getSideCount();
      vector< int > fluxIDs = _bilinearForm->trialBoundaryIDs();
      vector< int >::iterator fluxIDIt;
      int numFluxDofs = 0;
      for (fluxIDIt = fluxIDs.begin(); fluxIDIt != fluxIDs.end(); fluxIDIt++) {
        int fluxID = *fluxIDIt;
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideIndex);
          numFluxDofs += numDofs;
        }
      }
      int numFieldDofs = elemTypePtr->trialOrderPtr->totalDofs() - numFluxDofs;
      int maxPossible = numFluxDofs * 2 + numSides*fluxIDs.size() + numFieldDofs;  // a side can be shared by 2 elements, and vertices can be shared
      maxRowSize = max(maxPossible, maxRowSize);
    }
  }
  return maxRowSize;
}

vector< ParametricCurvePtr > Mesh::parametricEdgesForCell(GlobalIndexType cellID, bool neglectCurves) {
  return _meshTopology->parametricEdgesForCell(cellID, neglectCurves);
}

// TODO: consider adding/moving this logic into MeshTopology
void Mesh::setEdgeToCurveMap(const map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > &edgeToCurveMap) {
  MeshPtr thisPtr = Teuchos::rcp(this, false);
  map< pair<IndexType, IndexType>, ParametricCurvePtr > localMap(edgeToCurveMap.begin(),edgeToCurveMap.end());
  _meshTopology->setEdgeToCurveMap(localMap, thisPtr);
}

void Mesh::setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "setElementType only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setElementType only supported for max rule.");
  }
  maxRule->setElementType(cellID, newType, sideUpgradeOnly);
}

void Mesh::setEnforceMultiBasisFluxContinuity( bool value ) {
  _enforceMBFluxContinuity = value;
}

void Mesh::setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy ) {
  _gda->setPartitionPolicy(partitionPolicy);
}

void Mesh::setUsePatchBasis( bool value ) {
  // TODO: throw an exception if we've already been refined??
  _usePatchBasis = value;
}

bool Mesh::usePatchBasis() {
  return _usePatchBasis;
}

MeshTopologyPtr Mesh::getTopology() {
  return _meshTopology;
}

vector<unsigned> Mesh::vertexIndicesForCell(GlobalIndexType cellID) {
  return _meshTopology->getCell(cellID)->vertices();
}

FieldContainer<double> Mesh::vertexCoordinates(GlobalIndexType vertexIndex) {
  int spaceDim = _meshTopology->getSpaceDim();
  FieldContainer<double> vertex(spaceDim);
  for (int d=0; d<spaceDim; d++) {
    vertex(d) = _meshTopology->getVertex(vertexIndex)[d];
  }
  return vertex;
}

void Mesh::verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<unsigned> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();
  int spaceDim = _meshTopology->getSpaceDim();

  //vertices.resize(numVertices,dimension);
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    for (int d=0; d<spaceDim; d++) {
      vertices(vertexIndex,d) = _meshTopology->getVertex(vertexIndices[vertexIndex])[d];
    }
  }
}

// global across all MPI nodes:
void Mesh::verticesForElementType(FieldContainer<double>& vertices, ElementTypePtr elemTypePtr) {
  int spaceDim = _meshTopology->getSpaceDim();
  int numVertices = elemTypePtr->cellTopoPtr->getNodeCount();
  int numCells = numElementsOfType(elemTypePtr);
  vertices.resize(numCells,numVertices,spaceDim);

  Teuchos::Array<int> dim; // for an individual cell
  dim.push_back(numVertices);
  dim.push_back(spaceDim);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = this->cellID(elemTypePtr,cellIndex);
    FieldContainer<double> cellVertices(dim,&vertices(cellIndex,0,0));
    this->verticesForCell(cellVertices, cellID);
  }
}

void Mesh::verticesForCells(FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs) {
  // all cells represented in cellIDs must have the same topology
  int spaceDim = _meshTopology->getSpaceDim();
  int numCells = cellIDs.size();
  
  if (numCells == 0) {
    vertices.resize(0,0,0);
    return;
  }
  unsigned firstCellID = cellIDs[0];
  int numVertices = _meshTopology->getCell(firstCellID)->vertices().size();
 
  vertices.resize(numCells,numVertices,spaceDim);

  Teuchos::Array<int> dim; // for an individual cell
  dim.push_back(numVertices);
  dim.push_back(spaceDim);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = cellIDs[cellIndex];
    FieldContainer<double> cellVertices(dim,&vertices(cellIndex,0,0));
    this->verticesForCell(cellVertices, cellID);
  }
}

void Mesh::verticesForSide(FieldContainer<double>& vertices, GlobalIndexType cellID, int sideIndex) {
  CellPtr cell = _meshTopology->getCell(cellID);
  int spaceDim = _meshTopology->getSpaceDim();
  int sideDim = spaceDim - 1;
  unsigned sideEntityIndex = cell->entityIndex(sideDim, sideIndex);
  vector<unsigned> vertexIndices = _meshTopology->getEntityVertexIndices(sideDim, sideEntityIndex);

  int numVertices = vertexIndices.size();
  vertices.resize(numVertices,spaceDim);
  
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    for (int d=0; d<spaceDim; d++) {
      vertices(vertexIndex,d) = _meshTopology->getVertex(vertexIndex)[d];
    }
  }
}

void Mesh::writeMeshPartitionsToFile(const string & fileName){
  ofstream myFile;
  myFile.open(fileName.c_str());
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  myFile << "numPartitions="<< partitionCount <<";"<<endl;

  int maxNumVertices=0;
  int maxNumElems=0;
  int spaceDim = 2;

  //initialize verts
  for (int i=0;i<partitionCount;i++){
    vector< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0;l<spaceDim;l++){
      myFile << "verts{"<< i+1 <<","<< l+1 << "} = zeros(" << maxNumVertices << ","<< maxNumElems << ");"<< endl;
      for (int j=0;j<cellsInPartition.size();j++){
        CellPtr cell = _meshTopology->getCell(cellsInPartition[j]);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> verts(numVertices,spaceDim); // gets resized inside verticesForCell
        verticesForCell(verts, cellsInPartition[j]);  //verts(numVertsForCell,dim)
        maxNumVertices = max(maxNumVertices,verts.dimension(0));
        maxNumElems = max(maxNumElems,(int)cellsInPartition.size());
      }
    }
  }  
  cout << "max number of elems = " << maxNumElems << endl;

  for (int i=0;i<partitionCount;i++){
    vector< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0;l<spaceDim;l++){
      for (int j=0;j<cellsInPartition.size();j++){
        CellPtr cell = _meshTopology->getCell(cellsInPartition[j]);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> vertices(numVertices,spaceDim);
        verticesForCell(vertices, cellsInPartition[j]);  //vertices(numVertsForCell,dim)
        
        // write vertex coordinates to file
        for (int k=0;k<numVertices;k++){
          myFile << "verts{"<< i+1 <<","<< l+1 <<"}("<< k+1 <<","<< j+1 <<") = "<< vertices(k,l) << ";"<<endl; // verts{numPartitions,spaceDim}
        }
      }
      
    }
  }
  myFile.close();
}

double Mesh::getCellMeasure(GlobalIndexType cellID)
{
  FieldContainer<double> physicalCellNodes = physicalCellNodesForCell(cellID);
  ElementPtr elem = getElement(cellID);
  Teuchos::RCP< ElementType > elemType = elem->elementType();
  Teuchos::RCP< shards::CellTopology > cellTopo = elemType->cellTopoPtr;  
  BasisCache basisCache(physicalCellNodes, *cellTopo, 1);
  return basisCache.getCellMeasures()(0);
}

double Mesh::getCellXSize(GlobalIndexType cellID){
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Anisotropic cell measures only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim); 
  verticesForCell(vertices, cellID);
  double xDist = vertices(1,0)-vertices(0,0);
  double yDist = vertices(1,1)-vertices(0,1);
  return sqrt(xDist*xDist + yDist*yDist);
}

double Mesh::getCellYSize(GlobalIndexType cellID){
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Anisotropic cell measures only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim); 
  verticesForCell(vertices, cellID);
  double xDist = vertices(3,0)-vertices(0,0);
  double yDist = vertices(3,1)-vertices(0,1);
  return sqrt(xDist*xDist + yDist*yDist);
}

vector<double> Mesh::getCellOrientation(GlobalIndexType cellID){
  ElementPtr elem = getElement(cellID);
  int spaceDim = 2; // assuming 2D
  int numSides = elem->numSides();
  TEUCHOS_TEST_FOR_EXCEPTION(numSides!=4, std::invalid_argument, "Cell orientation only defined for quads right now.");
  FieldContainer<double> vertices(numSides,spaceDim); 
  verticesForCell(vertices, cellID);
  double xDist = vertices(3,0)-vertices(0,0);
  double yDist = vertices(3,1)-vertices(0,1);
  vector<double> orientation;
  orientation.push_back(xDist);
  orientation.push_back(yDist);
  return orientation;
}


Teuchos::RCP<Mesh> Mesh::readMsh(string filePath, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::readMsh() deprecated.  Use MeshFactory::readMesh() instead.\n";
  
  return MeshFactory::readMesh(filePath, bilinearForm, H1Order, pToAdd);
}

Teuchos::RCP<Mesh> Mesh::readTriangle(string filePath, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::readTriangle() deprecated.  Use MeshFactory::readTriangle() instead.\n";
  
  return MeshFactory::readTriangle(filePath, bilinearForm, H1Order, pToAdd);
}

Teuchos::RCP<Mesh> Mesh::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                                       int horizontalElements, int verticalElements,
                                       Teuchos::RCP< BilinearForm > bilinearForm,
                                       int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                                       map<int,int> trialOrderEnhancements,
                                       map<int,int> testOrderEnhancements) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::buildQuadMesh() deprecated.  Use MeshFactory::buildQuadMesh() instead.\n";
  
  return MeshFactory::buildQuadMesh(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, H1Order, pTest, triangulate, useConformingTraces, trialOrderEnhancements, testOrderEnhancements);
}

Teuchos::RCP<Mesh> Mesh::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
                                             int horizontalElements, int verticalElements,
                                             Teuchos::RCP< BilinearForm > bilinearForm,
                                             int H1Order, int pTest, bool useConformingTraces) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::buildQuadMeshHybrid() deprecated.  Use MeshFactory::buildQuadMeshHybrid() instead.\n";
  
  return MeshFactory::buildQuadMeshHybrid(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, \
                                          H1Order, pTest, useConformingTraces);
}

void Mesh::quadMeshCellIDs(FieldContainer<int> &cellIDs,
                           int horizontalElements, int verticalElements,
                           bool useTriangles) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: Mesh::quadMeshCellIDs() deprecated.  Use MeshFactory::quadMeshCellIDs() instead.\n";
  
  MeshFactory::quadMeshCellIDs(cellIDs, horizontalElements, verticalElements, useTriangles);
}

