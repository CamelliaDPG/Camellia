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

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

#include "MeshFactory.h"

#include "MPIWrapper.h"

#include "ZoltanMeshPartitionPolicy.h"

#include <algorithm>

using namespace Intrepid;
using namespace Camellia;
using namespace std;

template <typename Scalar>
map<int,int> TMesh<Scalar>::_emptyIntIntMap;

template <typename Scalar>
TMesh<Scalar>::TMesh(MeshTopologyPtr meshTopology, TBFPtr<Scalar> bilinearForm, vector<int> H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy) : DofInterpreter(Teuchos::rcp(this,false)) {

  _meshTopology = meshTopology;

  DofOrderingFactoryPtr<Scalar> dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory<Scalar>(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  //  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  if ( partitionPolicy.get() == NULL )
    partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy() );

  TMeshPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  setBilinearForm(bilinearForm);
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

template <typename Scalar>
TMesh<Scalar>::TMesh(MeshTopologyPtr meshTopology, TBFPtr<Scalar> bilinearForm, int H1Order, int pToAddTest,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
           MeshPartitionPolicyPtr partitionPolicy) : DofInterpreter(Teuchos::rcp(this,false)) {

  _meshTopology = meshTopology;

  DofOrderingFactoryPtr<Scalar> dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory<Scalar>(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
//  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  if ( partitionPolicy.get() == NULL )
    partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy() );

  TMeshPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMinimumRule(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                          partitionPolicy, H1Order, pToAddTest));
  _gda->repartitionAndMigrate();

  setBilinearForm(bilinearForm);
  _boundary.setMesh(Teuchos::rcp(this,false));

  _meshTopology->setGlobalDofAssignment(_gda.get());

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

template <typename Scalar>
TMesh<Scalar>::TMesh(const vector<vector<double> > &vertices, vector< vector<unsigned> > &elementVertices,
           TBFPtr<Scalar> bilinearForm, int H1Order, int pToAddTest, bool useConformingTraces,
           map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements, vector<PeriodicBCPtr> periodicBCs) : DofInterpreter(Teuchos::rcp(this,false)) {

//  cout << "in legacy mesh constructor, periodicBCs size is " << periodicBCs.size() << endl;

  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices) );
  _meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry, periodicBCs) );

  DofOrderingFactoryPtr<Scalar> dofOrderingFactoryPtr = Teuchos::rcp( new DofOrderingFactory<Scalar>(bilinearForm, trialOrderEnhancements,testOrderEnhancements) );
  _enforceMBFluxContinuity = false;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy() );

  TMeshPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
  _gda = Teuchos::rcp( new GDAMaximumRule2D(thisPtr, bilinearForm->varFactory(), dofOrderingFactoryPtr,
                                            partitionPolicy, H1Order, pToAddTest, _enforceMBFluxContinuity) );
  _gda->repartitionAndMigrate();

  _meshTopology->setGlobalDofAssignment(_gda.get());

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

  _boundary.setMesh(Teuchos::rcp(this,false));

  _pToAddToTest = pToAddTest;

  // Teuchos::RCP< RefinementHistory > refHist = Teuchos::rcp( &_refinementHistory, false );
  // cout << "Has ownership " << refHist.has_ownership() << endl;
  // this->registerObserver(refHist);
  this->registerObserver(Teuchos::rcp( &_refinementHistory, false ));
}

// private constructor for use by deepCopy()
template <typename Scalar>
TMesh<Scalar>::TMesh(MeshTopologyPtr meshTopology, Teuchos::RCP<GlobalDofAssignment> gda, TBFPtr<Scalar> bf,
           int pToAddToTest, bool useConformingTraces, bool usePatchBasis, bool enforceMBFluxContinuity) : DofInterpreter(Teuchos::rcp(this,false)) {
  _meshTopology = meshTopology;
  _gda = gda;
  _bilinearForm = bf;
  _pToAddToTest = pToAddToTest;
  _useConformingTraces = useConformingTraces;
  _usePatchBasis = usePatchBasis;
  _enforceMBFluxContinuity = enforceMBFluxContinuity;

  _boundary.setMesh(Teuchos::rcp(this,false));

}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numInitialElements(){
  return _meshTopology->getRootCellIndices().size();
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::activeCellOffset() {
  return _gda->activeCellOffset();
}

template <typename Scalar>
vector< ElementPtr > TMesh<Scalar>::activeElements() {
  set< IndexType > activeCellIndices = _meshTopology->getActiveCellIndices();

  vector< ElementPtr > activeElements;

  for (set< IndexType >::iterator cellIt = activeCellIndices.begin(); cellIt != activeCellIndices.end(); cellIt++) {
    activeElements.push_back(getElement(*cellIt));
  }

  return activeElements;
}

template <typename Scalar>
ElementPtr TMesh<Scalar>::ancestralNeighborForSide(ElementPtr elem, int sideIndex, int &elemSideIndexInNeighbor) {
  CellPtr cell = _meshTopology->getCell(elem->cellID());
  pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideIndex);
  elemSideIndexInNeighbor = neighborInfo.second;

  if (neighborInfo.first == -1) return Teuchos::rcp( (Element*) NULL );

  return getElement(neighborInfo.first);
}

template <typename Scalar>
TBFPtr<Scalar> TMesh<Scalar>::bilinearForm() {
  return _bilinearForm;
}

template <typename Scalar>
void TMesh<Scalar>::setBilinearForm( TBFPtr<Scalar> bf) {
  // must match the original in terms of variable IDs, etc...
  _bilinearForm = bf;
}

template <typename Scalar>
Boundary<Scalar> & TMesh<Scalar>::boundary() {
  return _boundary;
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber) {
  return _gda->cellID(elemTypePtr, cellIndex, partitionNumber);
}

template <typename Scalar>
vector< GlobalIndexType > TMesh<Scalar>::cellIDsOfType(ElementTypePtr elemType) {
  int rank = Teuchos::GlobalMPISession::getRank();
  return cellIDsOfType(rank,elemType);
}

template <typename Scalar>
vector< GlobalIndexType > TMesh<Scalar>::cellIDsOfType(int partitionNumber, ElementTypePtr elemTypePtr) {
  // returns the cell IDs for a given partition and element type
  return _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
}

template <typename Scalar>
vector< GlobalIndexType > TMesh<Scalar>::cellIDsOfTypeGlobal(ElementTypePtr elemTypePtr) {
  vector< GlobalIndexType > cellIDs;
  int partititionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partititionCount; partitionNumber++) {
    vector< GlobalIndexType > cellIDsForType = cellIDsOfType(partitionNumber, elemTypePtr);
    cellIDs.insert(cellIDs.end(), cellIDsForType.begin(), cellIDsForType.end());
  }
  return cellIDs;
}

template <typename Scalar>
set<GlobalIndexType> TMesh<Scalar>::cellIDsInPartition() {
  return _gda->cellsInPartition(-1);
}

template <typename Scalar>
int TMesh<Scalar>::cellPolyOrder(GlobalIndexType cellID) { // aka H1Order
  return _gda->getH1Order(cellID)[0];
}

template <typename Scalar>
vector<int> TMesh<Scalar>::cellTensorPolyOrder(GlobalIndexType cellID) { // aka H1Order
  return _gda->getH1Order(cellID);
}

template <typename Scalar>
vector<GlobalIndexType> TMesh<Scalar>::cellIDsForPoints(const FieldContainer<double> &physicalPoints, bool minusOnesIfOffRank) {
  vector<GlobalIndexType> cellIDs = _meshTopology->cellIDsForPoints(physicalPoints);

  if (minusOnesIfOffRank) {
    set<GlobalIndexType> rankLocalCellIDs = cellIDsInPartition();
    for (int i=0; i<cellIDs.size(); i++) {
      if (rankLocalCellIDs.find(cellIDs[i]) == rankLocalCellIDs.end()) {
        cellIDs[i] = -1;
      }
    }
  }
  return cellIDs;
}

template <typename Scalar>
TMeshPtr<Scalar> TMesh<Scalar>::deepCopy() {
  MeshTopologyPtr meshTopoCopy = _meshTopology->deepCopy();
  GlobalDofAssignmentPtr gdaCopy = _gda->deepCopy();

  TMeshPtr<Scalar> meshCopy = Teuchos::rcp( new TMesh<Scalar>(meshTopoCopy, gdaCopy, _bilinearForm, _pToAddToTest, _useConformingTraces, _usePatchBasis, _enforceMBFluxContinuity ));
  gdaCopy->setMeshAndMeshTopology(meshCopy);
  return meshCopy;
}

template <typename Scalar>
vector<ElementPtr> TMesh<Scalar>::elementsForPoints(const FieldContainer<double> &physicalPoints, bool nullElementsIfOffRank) {

  vector<GlobalIndexType> cellIDs = cellIDsForPoints(physicalPoints, nullElementsIfOffRank);
  vector<ElementPtr> elemsForPoints(cellIDs.size());

  for (int i=0; i<cellIDs.size(); i++) {
    GlobalIndexType cellID = cellIDs[i];
    ElementPtr elem;
    if (cellID==-1) {
      elem = Teuchos::rcp( (Element*) NULL);
    } else {
      elem = getElement(cellID);
    }
    elemsForPoints[i] = elem;
  }
  return elemsForPoints;
}

template <typename Scalar>
void TMesh<Scalar>::enforceOneIrregularity() {
  int rank = Teuchos::GlobalMPISession::getRank();
  bool meshIsNotRegular = true; // assume it's not regular and check elements
  while (meshIsNotRegular) {
    int spaceDim = _meshTopology->getSpaceDim();
    if (spaceDim == 1) return;

    map< Camellia::CellTopologyKey, set<GlobalIndexType> > irregularCellIDs; // key is CellTopology key
    set< GlobalIndexType > activeCellIDs = _meshTopology->getActiveCellIndices();
    set< GlobalIndexType >::iterator cellIDIt;

    for (cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;

      CellPtr cell = _meshTopology->getCell(cellID);
      bool isIrregular = false;
      int sideCount = cell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++) {
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal);
        unsigned mySideIndexInNeighbor = neighborInfo.second;

        if (neighborInfo.first != -1) {
          CellPtr neighbor = _meshTopology->getCell(neighborInfo.first);
          int numNeighborsOnSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor).size();
          if (spaceDim==2) {
            if (numNeighborsOnSide > 2) isIrregular=true;
          } else if (spaceDim==3) {
            // TODO: fix this criterion so that it will be correct for anisotropic refinements (consider two vertical cuts: could have as few as 3 neighbors and still be 2-irregular).  The right criterion is the depth of the side's refinement tree: this should be at most 1.  (This holds in general dimensions, not just 3.)
            if (numNeighborsOnSide > 4) isIrregular=true;
          }
        }
      }

      if (isIrregular) {
//        if (rank==0) cout << "cellID " << cellID << " is irregular.\n";
        irregularCellIDs[cell->topology()->getKey()].insert(cellID);
      }
    }
    if (irregularCellIDs.size() > 0) {
      for (map< Camellia::CellTopologyKey, set<GlobalIndexType> >::iterator mapIt = irregularCellIDs.begin();
           mapIt != irregularCellIDs.end(); mapIt++) {
        Camellia::CellTopologyKey cellKey = mapIt->first;
        hRefine(mapIt->second, RefinementPattern::regularRefinementPattern(cellKey));
      }
      irregularCellIDs.clear();
    } else {
      meshIsNotRegular=false;
    }
  }
}

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::cellSideParities( ElementTypePtr elemTypePtr ) {
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

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::cellSideParitiesForCell( GlobalIndexType cellID ) {
  return _gda->cellSideParitiesForCell(cellID);
}

template <typename Scalar>
vector<double> TMesh<Scalar>::getCellCentroid(GlobalIndexType cellID){
  return _meshTopology->getCellCentroid(cellID);
}

template <typename Scalar>
vector< ElementPtr > TMesh<Scalar>::elementsInPartition(PartitionIndexType partitionNumber){
  set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(partitionNumber);
  vector< ElementPtr > elements;
  for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementPtr element = getElement(cellID);
    elements.push_back(element);
  }
  return elements;
}

template <typename Scalar>
vector< ElementPtr > TMesh<Scalar>::elementsOfType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr) {
  // returns the elements for a given partition and element type
  vector< ElementPtr > elementsOfType;
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(partitionNumber, elemTypePtr);
  for (vector<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    elementsOfType.push_back(getElement(*cellIt));
  }
  return elementsOfType;
}

template <typename Scalar>
vector< ElementPtr > TMesh<Scalar>::elementsOfTypeGlobal(ElementTypePtr elemTypePtr) {
  vector< ElementPtr > elementsOfTypeVector;
  int partitionCount = _gda->getPartitionCount();
  for (int partitionNumber=0; partitionNumber<partitionCount; partitionNumber++) {
    vector< ElementPtr > elementsOfTypeForPartition = elementsOfType(partitionNumber,elemTypePtr);
    elementsOfTypeVector.insert(elementsOfTypeVector.end(),elementsOfTypeForPartition.begin(),elementsOfTypeForPartition.end());
  }
  return elementsOfTypeVector;
}

template <typename Scalar>
vector< ElementTypePtr > TMesh<Scalar>::elementTypes(PartitionIndexType partitionNumber) {
  return _gda->elementTypes(partitionNumber);
}

template <typename Scalar>
set<GlobalIndexType> TMesh<Scalar>::getActiveCellIDs() {
  set<IndexType> activeCellIndices = _meshTopology->getActiveCellIndices();
  set<GlobalIndexType> activeCellIDs(activeCellIndices.begin(), activeCellIndices.end());
  return activeCellIDs;
}

template <typename Scalar>
int TMesh<Scalar>::getDimension() {
  return _meshTopology->getSpaceDim();
}

template <typename Scalar>
DofOrderingFactory<Scalar> & TMesh<Scalar>::getDofOrderingFactory() {
  return *_gda->getDofOrderingFactory().get();
}

template <typename Scalar>
ElementPtr TMesh<Scalar>::getElement(GlobalIndexType cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);

  ElementTypePtr elemType = _gda->elementType(cellID);

  IndexType cellIndex = _gda->partitionLocalCellIndex(cellID);

  GlobalIndexType globalCellIndex = _gda->globalCellIndex(cellID);

  ElementPtr element = Teuchos::rcp( new Element(this, cellID, elemType, cellIndex, globalCellIndex) );

  return element;
}

template <typename Scalar>
ElementTypePtr TMesh<Scalar>::getElementType(GlobalIndexType cellID) {
  return _gda->elementType(cellID);
}

template <typename Scalar>
ElementTypeFactory & TMesh<Scalar>::getElementTypeFactory() {
  return _gda->getElementTypeFactory();
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::getVertexIndex(double x, double y, double tol) {
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

template <typename Scalar>
const map< pair<GlobalIndexType,IndexType>, GlobalIndexType>& TMesh<Scalar>::getLocalToGlobalMap() {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "getLocalToGlobalMap only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getLocalToGlobalMap only supported for max rule.");
  }
  return maxRule->getLocalToGlobalMap();
}

template <typename Scalar>
bool TMesh<Scalar>::meshUsesMaximumRule() {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  return (maxRule != NULL);
}

template <typename Scalar>
bool TMesh<Scalar>::meshUsesMinimumRule() {
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule *>(_gda.get());
  return (minRule != NULL);
}

template <typename Scalar>
map<IndexType, GlobalIndexType> TMesh<Scalar>::getGlobalVertexIDs(const FieldContainer<double> &vertices) {
  double tol = 1e-12; // tolerance for vertex equality

  map<IndexType, GlobalIndexType> localToGlobalVertexIndex;
  int numVertices = vertices.dimension(0);
  for (int i=0; i<numVertices; i++) {
    localToGlobalVertexIndex[i] = getVertexIndex(vertices(i,0), vertices(i,1),tol);
  }
  return localToGlobalVertexIndex;
}

template <typename Scalar>
TFunctionPtr<double> TMesh<Scalar>::getTransformationFunction() {
  // will be NULL for meshes without edge curves defined

  // for now, we recompute the transformation function each time the edge curves get updated
  // we might later want to do something lazier, updating/creating it here if it's out of date

  return _meshTopology->transformationFunction();
}

template <typename Scalar>
GlobalDofAssignmentPtr TMesh<Scalar>::globalDofAssignment() {
  return _gda;
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::globalDofCount() {
  return numGlobalDofs(); // TODO: eliminate numGlobalDofs in favor of globalDofCount
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "globalDofIndex lookup only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndex lookup only supported for max rule.");
  }
  return maxRule->globalDofIndex(cellID, localDofIndex);
}

template <typename Scalar>
set<GlobalIndexType> TMesh<Scalar>::globalDofIndicesForCell(GlobalIndexType cellID) {
  return _gda->globalDofIndicesForCell(cellID);
}

template <typename Scalar>
set<GlobalIndexType> TMesh<Scalar>::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  return _gda->globalDofIndicesForPartition(partitionNumber);
}

//void TMesh<Scalar>::hRefine(vector<GlobalIndexType> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
//  hRefine(cellIDs,refPattern,vector< TSolutionPtr<double> >());
//}

template <typename Scalar>
void TMesh<Scalar>::hRefine(const vector<GlobalIndexType> &cellIDs) {
  set<GlobalIndexType> cellSet(cellIDs.begin(),cellIDs.end());
  hRefine(cellSet);
}

template <typename Scalar>
void TMesh<Scalar>::hRefine(const set<GlobalIndexType> &cellIDs) {
  map< CellTopologyKey, set<GlobalIndexType> > cellIDsForTopo;
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end();
       cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellTopoPtr cellTopo = getElementType(cellID)->cellTopoPtr;
    cellIDsForTopo[cellTopo->getKey()].insert(cellID);
  }

  for (map< CellTopologyKey, set<GlobalIndexType> >::iterator entryIt = cellIDsForTopo.begin();
       entryIt != cellIDsForTopo.end(); entryIt++) {
    CellTopologyKey cellTopoKey = entryIt->first;
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopoKey);
    hRefine(entryIt->second, refPattern);
  }
}

template <typename Scalar>
void TMesh<Scalar>::hRefine(const vector<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  set<GlobalIndexType> cellSet(cellIDs.begin(),cellIDs.end());
  hRefine(cellSet,refPattern);
}

template <typename Scalar>
void TMesh<Scalar>::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  hRefine(cellIDs, refPattern, true);
}

template <typename Scalar>
void TMesh<Scalar>::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild) {
  if (cellIDs.size() == 0) return;

  // send h-refinement message any registered observers (may be meshes)
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    (*meshIt)->hRefine(_meshTopology,cellIDs,refPattern);
  }

  set<GlobalIndexType>::const_iterator cellIt;

  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;

    if (_meshTopology->getActiveCellIndices().find(cellID) == _meshTopology->getActiveCellIndices().end()) {
      cout << "cellID " << cellID << " is not active, but Mesh received request for h-refinement.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inactive cell");
    }

    _meshTopology->refineCell(cellID, refPattern);

    // TODO: figure out what it is that breaks in GDAMaximumRule when we use didHRefine to notify about all cells together outside this loop
    //       (and/or try moving outside the loop if and only if we are using the minimum rule)
    set<GlobalIndexType> cellIDset;
    cellIDset.insert(cellID);

    // TODO: consider making GDA a refinementObserver, using that interface to send it the notification
    _gda->didHRefine(cellIDset);
  }

  // NVR 12/10/14 the code below moved from inside the loop above, where it was doing the below one cell at a time...
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator observerIt = _registeredObservers.begin();
       observerIt != _registeredObservers.end(); observerIt++) {
    (*observerIt)->didHRefine(_meshTopology,cellIDs,refPattern);
  }

  // TODO: consider making transformation function a refinementObserver, using that interface to send it the notification
  // let transformation function know about the refinement that just took place
  if (_meshTopology->transformationFunction().get()) {
    _meshTopology->transformationFunction()->didHRefine(cellIDs);
  }

  if (repartitionAndRebuild) {
    _gda->repartitionAndMigrate();
    _boundary.buildLookupTables();

    for (vector< Teuchos::RCP<RefinementObserver> >::iterator observerIt = _registeredObservers.begin();
         observerIt != _registeredObservers.end(); observerIt++) {
      (*observerIt)->didRepartition(_meshTopology);
    }
  }
}

template <typename Scalar>
void TMesh<Scalar>::hUnrefine(const set<GlobalIndexType> &cellIDs) {
  if (cellIDs.size() == 0) return;

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

  // TODO: consider making GDA a RefinementObserver, and using that interface to send the notification of unrefinement.
  _gda->didHUnrefine(cellIDs);

  // notify observers that of the unrefinement that just happened
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    (*meshIt)->didHUnrefine(_meshTopology,cellIDs);
  }

  _gda->repartitionAndMigrate();
  _boundary.buildLookupTables();
}

template <typename Scalar>
void TMesh<Scalar>::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients) {
  _gda->interpretGlobalCoefficients(cellID, localCoefficients, globalCoefficients);
}

template <typename Scalar>
void TMesh<Scalar>::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                   FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) {
  _gda->interpretLocalBasisCoefficients(cellID, varID, sideOrdinal, basisCoefficients, globalCoefficients, globalDofIndices);
}

template <typename Scalar>
void TMesh<Scalar>::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                              FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) {
  _gda->interpretLocalData(cellID, localDofs, globalDofs, globalDofIndices);
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numActiveElements() {
  return _meshTopology->activeCellCount();
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numElements() {
  return _meshTopology->cellCount();
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr ) {
  // returns the global total (across all MPI nodes)
  int numElements = 0;
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  for (PartitionIndexType partitionNumber=0; partitionNumber<partitionCount; partitionNumber++) {
    numElements += _gda->cellIDsOfElementType(partitionNumber, elemTypePtr).size();
  }
  return numElements;
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numFluxDofs(){
  GlobalIndexType fluxDofsForPartition = _gda->partitionOwnedGlobalFluxIndices().size();
  GlobalIndexType traceDofsForPartition = _gda->partitionOwnedGlobalTraceIndices().size();

  return MPIWrapper::sum(fluxDofsForPartition + traceDofsForPartition);
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numFieldDofs(){
  GlobalIndexType fieldDofsForPartition = _gda->partitionOwnedGlobalFieldIndices().size();
  return MPIWrapper::sum(fieldDofsForPartition);
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::numGlobalDofs() {
  return _gda->globalDofCount();
}

template <typename Scalar>
int TMesh<Scalar>::parityForSide(GlobalIndexType cellID, int sideOrdinal) {
  int parity = _gda->cellSideParitiesForCell(cellID)[sideOrdinal];
  return parity;
}

template <typename Scalar>
PartitionIndexType TMesh<Scalar>::partitionForCellID( GlobalIndexType cellID ) {
  return _gda->partitionForCellID(cellID);
}

template <typename Scalar>
PartitionIndexType TMesh<Scalar>::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  return _gda->partitionForGlobalDofIndex(globalDofIndex);
}

template <typename Scalar>
GlobalIndexType TMesh<Scalar>::partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "partitionLocalIndexForGlobalDofIndex only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "partitionLocalIndexForGlobalDofIndex only supported for max rule.");
  }
  return maxRule->partitionLocalIndexForGlobalDofIndex(globalDofIndex);
}

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr) {
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);

  return physicalCellNodes(elemTypePtr, cellIDs);
}

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr, vector<GlobalIndexType> &cellIDs ) {
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

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::physicalCellNodesForCell( GlobalIndexType cellID ) {
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

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::physicalCellNodesGlobal( Teuchos::RCP< ElementType > elemTypePtr ) {
//  int numRanks = Teuchos::GlobalMPISession::getNProc();

  // user should call cellIDsOfTypeGlobal() to get the corresponding cell IDs (the cell nodes are *NOT* sorted by cell ID)

  vector<GlobalIndexType> globalCellIDs = cellIDsOfTypeGlobal(elemTypePtr);
//  for (int rank=0; rank<numRanks; rank++) {
//    vector<GlobalIndexType> cellIDs = _gda->cellIDsOfElementType(rank, elemTypePtr);
//    globalCellIDs.insert(globalCellIDs.end(), cellIDs.begin(), cellIDs.end());
//  }

  return physicalCellNodes(elemTypePtr, globalCellIDs);
}

template <typename Scalar>
void TMesh<Scalar>::printLocalToGlobalMap() {
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> localToGlobalMap = this->getLocalToGlobalMap();

  for (map< pair<GlobalIndexType,IndexType>, GlobalIndexType>::iterator entryIt = localToGlobalMap.begin();
       entryIt != localToGlobalMap.end(); entryIt++) {
    int cellID = entryIt->first.first;
    int localDofIndex = entryIt->first.second;
    int globalDofIndex = entryIt->second;
    cout << "(" << cellID << "," << localDofIndex << ") --> " << globalDofIndex << endl;
  }
}

template <typename Scalar>
void TMesh<Scalar>::printVertices() {
  cout << "Vertices:\n";
  unsigned vertexDim = 0;
  unsigned vertexCount = _meshTopology->getEntityCount(vertexDim);
  for (unsigned vertexIndex=0; vertexIndex<vertexCount; vertexIndex++) {
    vector<double> vertex = _meshTopology->getVertex(vertexIndex);
    cout << vertexIndex << ": (" << vertex[0] << ", " << vertex[1] << ")\n";
  }
}

template <typename Scalar>
void TMesh<Scalar>::registerObserver(Teuchos::RCP<RefinementObserver> observer) {
  _registeredObservers.push_back(observer);
}

template <typename Scalar>
void TMesh<Scalar>::registerSolution(TSolutionPtr<Scalar> solution) {
  _gda->registerSolution(solution);
}

template <typename Scalar>
void TMesh<Scalar>::unregisterObserver(RefinementObserver* observer) {
  for (vector< Teuchos::RCP<RefinementObserver> >::iterator meshIt = _registeredObservers.begin();
       meshIt != _registeredObservers.end(); meshIt++) {
    if ( (*meshIt).get() == observer ) {
      _registeredObservers.erase(meshIt);
      return;
    }
  }
  cout << "WARNING: TMesh<Scalar>::unregisterObserver: Observer not found.\n";
}

template <typename Scalar>
void TMesh<Scalar>::unregisterObserver(Teuchos::RCP<RefinementObserver> mesh) {
  this->unregisterObserver(mesh.get());
}

template <typename Scalar>
void TMesh<Scalar>::unregisterSolution(TSolutionPtr<Scalar> solution) {
  _gda->unregisterSolution(solution);
}

template <typename Scalar>
void TMesh<Scalar>::pRefine(const vector<GlobalIndexType> &cellIDsForPRefinements) {
  set<GlobalIndexType> cellSet;
  for (vector<GlobalIndexType>::const_iterator cellIt=cellIDsForPRefinements.begin();
       cellIt != cellIDsForPRefinements.end(); cellIt++) {
    cellSet.insert(*cellIt);
  }
  pRefine(cellSet);
}

template <typename Scalar>
void TMesh<Scalar>::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements){
  pRefine(cellIDsForPRefinements,1);
}

template <typename Scalar>
void TMesh<Scalar>::pRefine(const set<GlobalIndexType> &cellIDsForPRefinements, int pToAdd) {
  if (cellIDsForPRefinements.size() == 0) return;

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

  _gda->repartitionAndMigrate();
  _boundary.buildLookupTables();
}

template <typename Scalar>
int TMesh<Scalar>::condensedRowSizeUpperBound() {
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
      maxRowSize = std::max(maxPossible, maxRowSize);
    }
  }
  return maxRowSize;
}

template <typename Scalar>
void TMesh<Scalar>::rebuildLookups() {
  _gda->repartitionAndMigrate();
  _boundary.buildLookupTables();
}

template <typename Scalar>
int TMesh<Scalar>::rowSizeUpperBound() {
  // includes multiplicity
  static const int MAX_SIZE_TO_PRESCRIBE = 100; // the below is a significant over-estimate.  Eventually, we want something more precise, that will analyze the BF to determine which variables actually talk to each other, and perhaps even provide a precise per-row count to the Epetra_CrsMatrix.  For now, we just cap the estimate.  (On construction, Epetra_CrsMatrix appears to be allocating the row size provided for every row, which is also wasteful.)
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
      maxRowSize = std::max(maxPossible, maxRowSize);
    }
  }
  GlobalIndexType numGlobalDofs = this->numGlobalDofs();
  maxRowSize = std::min(maxRowSize, (int) numGlobalDofs);
  return std::min(maxRowSize, MAX_SIZE_TO_PRESCRIBE);
}

template <typename Scalar>
vector< ParametricCurvePtr > TMesh<Scalar>::parametricEdgesForCell(GlobalIndexType cellID, bool neglectCurves) {
  return _meshTopology->parametricEdgesForCell(cellID, neglectCurves);
}

// TODO: consider adding/moving this logic into MeshTopology
template <typename Scalar>
void TMesh<Scalar>::setEdgeToCurveMap(const map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > &edgeToCurveMap) {
  TMeshPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
  map< pair<IndexType, IndexType>, ParametricCurvePtr > localMap(edgeToCurveMap.begin(),edgeToCurveMap.end());
  _meshTopology->setEdgeToCurveMap(localMap, thisPtr);
}

template <typename Scalar>
void TMesh<Scalar>::setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly) {
  GDAMaximumRule2D* maxRule = dynamic_cast<GDAMaximumRule2D *>(_gda.get());
  if (maxRule == NULL) {
    cout << "setElementType only supported for max rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setElementType only supported for max rule.");
  }
  maxRule->setElementType(cellID, newType, sideUpgradeOnly);
}

template <typename Scalar>
void TMesh<Scalar>::setEnforceMultiBasisFluxContinuity( bool value ) {
  _enforceMBFluxContinuity = value;
}

template <typename Scalar>
void TMesh<Scalar>::setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy ) {
  _gda->setPartitionPolicy(partitionPolicy);
}

template <typename Scalar>
void TMesh<Scalar>::setUsePatchBasis( bool value ) {
  // TODO: throw an exception if we've already been refined??
  _usePatchBasis = value;
}

template <typename Scalar>
bool TMesh<Scalar>::usePatchBasis() {
  return _usePatchBasis;
}

template <typename Scalar>
MeshTopologyPtr TMesh<Scalar>::getTopology() {
  return _meshTopology;
}

template <typename Scalar>
vector<unsigned> TMesh<Scalar>::vertexIndicesForCell(GlobalIndexType cellID) {
  return _meshTopology->getCell(cellID)->vertices();
}

template <typename Scalar>
FieldContainer<double> TMesh<Scalar>::vertexCoordinates(GlobalIndexType vertexIndex) {
  int spaceDim = _meshTopology->getSpaceDim();
  FieldContainer<double> vertex(spaceDim);
  for (int d=0; d<spaceDim; d++) {
    vertex(d) = _meshTopology->getVertex(vertexIndex)[d];
  }
  return vertex;
}

template <typename Scalar>
vector< vector<double> > TMesh<Scalar>::verticesForCell(GlobalIndexType cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<unsigned> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();

  vector< vector<double> > vertices(numVertices);
  //vertices.resize(numVertices,dimension);
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    vertices[vertexIndex] = _meshTopology->getVertex(vertexIndices[vertexIndex]);
  }
  return vertices;
}

template <typename Scalar>
void TMesh<Scalar>::verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID) {
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
template <typename Scalar>
void TMesh<Scalar>::verticesForElementType(FieldContainer<double>& vertices, ElementTypePtr elemTypePtr) {
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

template <typename Scalar>
void TMesh<Scalar>::verticesForCells(FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs) {
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

template <typename Scalar>
void TMesh<Scalar>::verticesForSide(FieldContainer<double>& vertices, GlobalIndexType cellID, int sideIndex) {
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

template <typename Scalar>
void TMesh<Scalar>::writeMeshPartitionsToFile(const string & fileName){
  // TODO: rewrite this code to only talk to rank-local cells.

  ofstream myFile;
  myFile.open(fileName.c_str());
  PartitionIndexType partitionCount = _gda->getPartitionCount();
  myFile << "numPartitions="<< partitionCount <<";"<<endl;

  int maxNumVertices=0;
  int maxNumElems=0;
  int spaceDim = 2;

  //initialize verts
  for (int i=0;i<partitionCount;i++){
    set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0;l<spaceDim;l++){
      myFile << "verts{"<< i+1 <<","<< l+1 << "} = zeros(" << maxNumVertices << ","<< maxNumElems << ");"<< endl;
      for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++) {
        CellPtr cell = _meshTopology->getCell(*cellIt);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> verts(numVertices,spaceDim); // gets resized inside verticesForCell
        verticesForCell(verts, *cellIt);  //verts(numVertsForCell,dim)
        maxNumVertices = std::max(maxNumVertices,verts.dimension(0));
        maxNumElems = std::max(maxNumElems,(int)cellsInPartition.size());
      }
    }
  }
  cout << "max number of elems = " << maxNumElems << endl;

  for (int i=0;i<partitionCount;i++){
    set< GlobalIndexType > cellsInPartition = _gda->cellsInPartition(i);
    for (int l=0;l<spaceDim;l++){
      int j=0;
      for (set< GlobalIndexType >::iterator cellIt = cellsInPartition.begin(); cellIt != cellsInPartition.end(); cellIt++) {
        CellPtr cell = _meshTopology->getCell(*cellIt);
        int numVertices = cell->topology()->getVertexCount();
        FieldContainer<double> vertices(numVertices,spaceDim);
        verticesForCell(vertices, *cellIt);  //vertices(numVertsForCell,dim)

        // write vertex coordinates to file
        for (int k=0;k<numVertices;k++){
          myFile << "verts{"<< i+1 <<","<< l+1 <<"}("<< k+1 <<","<< j+1 <<") = "<< vertices(k,l) << ";"<<endl; // verts{numPartitions,spaceDim}
        }
        j++;
      }

    }
  }
  myFile.close();
}

template <typename Scalar>
double TMesh<Scalar>::getCellMeasure(GlobalIndexType cellID)
{
  FieldContainer<double> physicalCellNodes = physicalCellNodesForCell(cellID);
  ElementPtr elem = getElement(cellID);
  Teuchos::RCP< ElementType > elemType = elem->elementType();
  CellTopoPtr cellTopo = elemType->cellTopoPtr;
  BasisCache basisCache(physicalCellNodes, cellTopo, 1);
  return basisCache.getCellMeasures()(0);
}

template <typename Scalar>
double TMesh<Scalar>::getCellXSize(GlobalIndexType cellID){
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

template <typename Scalar>
double TMesh<Scalar>::getCellYSize(GlobalIndexType cellID){
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

template <typename Scalar>
vector<double> TMesh<Scalar>::getCellOrientation(GlobalIndexType cellID){
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


#ifdef HAVE_EPETRAEXT_HDF5
template <typename Scalar>
void TMesh<Scalar>::saveToHDF5(string filename)
{
  int commRank = Teuchos::GlobalMPISession::getRank();

  if (commRank == 0)
  {
    set<IndexType> rootCellIndicesSet = getTopology()->getRootCellIndices();
    vector<IndexType> rootCellIndices(rootCellIndicesSet.begin(), rootCellIndicesSet.end());
    vector<IndexType> rootVertexIndices;
    vector<Camellia::CellTopologyKey> rootKeys;
    vector<double> rootVertices;
    for (int i=0; i < rootCellIndices.size(); i++)
    {
      CellPtr cell = getTopology()->getCell(rootCellIndices[i]);
      vector< unsigned > vertexIndices = cell->vertices();
      rootKeys.push_back(cell->topology()->getKey());
      rootVertexIndices.insert(rootVertexIndices.end(), vertexIndices.begin(), vertexIndices.end());
    }
    IndexType maxVertexIndex = *max_element(rootVertexIndices.begin(), rootVertexIndices.end());
    for (int i=0; i <= maxVertexIndex; i++)
    {
      vector< double > vertex = getTopology()->getVertex(i);
      rootVertices.insert(rootVertices.end(), vertex.begin(), vertex.end());
//      cout << "vertex " << i << ":\n";
//      Camellia::print("vertex coords", vertex);
    }
    map<int, int> trialOrderEnhancements = getDofOrderingFactory().getTrialOrderEnhancements();
    map<int, int> testOrderEnhancements = getDofOrderingFactory().getTestOrderEnhancements();
    vector<int> trialOrderEnhancementsVec;
    vector<int> testOrderEnhancementsVec;
    for (map<int,int>::iterator it=trialOrderEnhancements.begin(); it!=trialOrderEnhancements.end(); ++it)
    {
      trialOrderEnhancementsVec.push_back(it->first);
      trialOrderEnhancementsVec.push_back(it->second);
    }
    for (map<int,int>::iterator it=testOrderEnhancements.begin(); it!=testOrderEnhancements.end(); ++it)
    {
      testOrderEnhancementsVec.push_back(it->first);
      testOrderEnhancementsVec.push_back(it->second);
    }
    int vertexIndicesSize = rootVertexIndices.size();
    int topoKeysIntSize = rootKeys.size() * sizeof(Camellia::CellTopologyKey) / sizeof(int);
    int topoKeysSize = rootKeys.size();
    int verticesSize = rootVertices.size();
    int trialOrderEnhancementsSize = trialOrderEnhancementsVec.size();
    int testOrderEnhancementsSize = testOrderEnhancementsVec.size();


    FieldContainer<GlobalIndexType> partitions;
    bool partitionsSet = globalDofAssignment()->getPartitions(partitions);
    int numPartitions, maxPartitionSize;
    if (partitionsSet) {
      numPartitions = partitions.dimension(0);
      maxPartitionSize = partitions.dimension(1);
    } else {
      numPartitions = 0;
      maxPartitionSize = 0;
    }
    FieldContainer<int> partitionsCastToInt;
    if (partitions.size() > 0) {
      partitionsCastToInt.resize(numPartitions, maxPartitionSize);
      for (int i=0; i<numPartitions; i++) {
        for (int j=0; j<maxPartitionSize; j++) {
          partitionsCastToInt(i,j) = (int) partitions(i,j);
        }
      }
    }

    vector<int> initialH1Order = globalDofAssignment()->getInitialH1Order();

    Epetra_SerialComm Comm;
    EpetraExt::HDF5 hdf5(Comm);
    hdf5.Create(filename);
    hdf5.Write("Mesh", "vertexIndicesSize", vertexIndicesSize);
    hdf5.Write("Mesh", "topoKeysSize", topoKeysSize);
    hdf5.Write("Mesh", "verticesSize", verticesSize);
    hdf5.Write("Mesh", "trialOrderEnhancementsSize", trialOrderEnhancementsSize);
    hdf5.Write("Mesh", "testOrderEnhancementsSize", testOrderEnhancementsSize);
    hdf5.Write("Mesh", "dimension", getDimension());
    hdf5.Write("Mesh", "vertexIndices", H5T_NATIVE_INT, rootVertexIndices.size(), &rootVertexIndices[0]);
    hdf5.Write("Mesh", "topoKeys", H5T_NATIVE_INT, topoKeysIntSize, &rootKeys[0]);
    hdf5.Write("Mesh", "vertices", H5T_NATIVE_DOUBLE, rootVertices.size(), &rootVertices[0]);
    hdf5.Write("Mesh", "H1OrderSize", (int)initialH1Order.size());
    hdf5.Write("Mesh", "deltaP", globalDofAssignment()->getTestOrderEnrichment());
    hdf5.Write("Mesh", "numPartitions", numPartitions);
    hdf5.Write("Mesh", "maxPartitionSize", maxPartitionSize);
    if (numPartitions > 0) {
      hdf5.Write("Mesh", "partitions", H5T_NATIVE_INT, partitionsCastToInt.size(), &partitionsCastToInt[0]);
    } else {
      hdf5.Write("Mesh", "partitions", H5T_NATIVE_INT, partitionsCastToInt.size(), NULL);
    }
    if (meshUsesMaximumRule())
      hdf5.Write("Mesh", "GDARule", "max");
    else if(meshUsesMinimumRule())
      hdf5.Write("Mesh", "GDARule", "min");
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid GDA");
    hdf5.Write("Mesh", "trialOrderEnhancements", H5T_NATIVE_INT, trialOrderEnhancementsVec.size(), &trialOrderEnhancementsVec[0]);
    hdf5.Write("Mesh", "testOrderEnhancements", H5T_NATIVE_INT, testOrderEnhancementsVec.size(), &testOrderEnhancementsVec[0]);
    hdf5.Write("Mesh", "H1Order", H5T_NATIVE_INT, initialH1Order.size(), &initialH1Order[0]);
    _refinementHistory.saveToHDF5(hdf5);
    hdf5.Close();
  }
}
// end HAVE_EPETRAEXT_HDF5 include guard
#endif


template <typename Scalar>
TMeshPtr<Scalar> TMesh<Scalar>::readMsh(string filePath, TBFPtr<Scalar> bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: TMesh<Scalar>::readMsh() deprecated.  Use MeshFactory::readMesh() instead.\n";

  return MeshFactory::readMesh(filePath, bilinearForm, H1Order, pToAdd);
}

template <typename Scalar>
TMeshPtr<Scalar> TMesh<Scalar>::readTriangle(string filePath, TBFPtr<Scalar> bilinearForm, int H1Order, int pToAdd)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: TMesh<Scalar>::readTriangle() deprecated.  Use MeshFactory::readTriangle() instead.\n";

  return MeshFactory::readTriangle(filePath, bilinearForm, H1Order, pToAdd);
}

template <typename Scalar>
TMeshPtr<Scalar> TMesh<Scalar>::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                                       int horizontalElements, int verticalElements,
                                       TBFPtr<Scalar> bilinearForm,
                                       int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                                       map<int,int> trialOrderEnhancements,
                                       map<int,int> testOrderEnhancements) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: TMesh<Scalar>::buildQuadMesh() deprecated.  Use MeshFactory::buildQuadMesh() instead.\n";

  return MeshFactory::buildQuadMesh(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, H1Order, pTest, triangulate, useConformingTraces, trialOrderEnhancements, testOrderEnhancements);
}

template <typename Scalar>
TMeshPtr<Scalar> TMesh<Scalar>::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
                                             int horizontalElements, int verticalElements,
                                             TBFPtr<Scalar> bilinearForm,
                                             int H1Order, int pTest, bool useConformingTraces) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: TMesh<Scalar>::buildQuadMeshHybrid() deprecated.  Use MeshFactory::buildQuadMeshHybrid() instead.\n";

  return MeshFactory::buildQuadMeshHybrid(quadBoundaryPoints, horizontalElements, verticalElements, bilinearForm, \
                                          H1Order, pTest, useConformingTraces);
}

template <typename Scalar>
void TMesh<Scalar>::quadMeshCellIDs(FieldContainer<int> &cellIDs,
                           int horizontalElements, int verticalElements,
                           bool useTriangles) {
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "Warning: TMesh<Scalar>::quadMeshCellIDs() deprecated.  Use MeshFactory::quadMeshCellIDs() instead.\n";

  MeshFactory::quadMeshCellIDs(cellIDs, horizontalElements, verticalElements, useTriangles);
}

namespace Camellia {
  template class TMesh<double>;
}
