/*
 *  Boundary.cpp
 *
 */

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
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

#include "Boundary.h"
#include "Intrepid_PointTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Mesh.h"
#include "Function.h"
#include "Projector.h"
#include "VarFactory.h"

#include "BC.h"
#include "BCFunction.h"

Boundary::Boundary() {
  
}

void Boundary::setMesh(Mesh* mesh) {
  _mesh = mesh;
}

void Boundary::addElement( int cellID, int sideIndex ) {
  //cout << "Boundary: addElement( cellID = " << cellID << ", sideIndex = " << sideIndex << " )\n";
  _boundaryElements.insert( make_pair(cellID,sideIndex) );
}

bool Boundary::boundaryElement(int cellID) {
  int numSides = _mesh->getElement(cellID)->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    if (boundaryElement(cellID,sideIndex)) {
      return true;
    }
  }
  return false;
}

bool Boundary::boundaryElement( int cellID, int sideIndex ) {
  pair<int,int> key = make_pair(cellID,sideIndex);  
  return _boundaryElements.find(key) != _boundaryElements.end();
}

void Boundary::deleteElement( int cellID, int sideIndex ) {
  //cout << "Boundary: deleteElement( cellID = " << cellID << ", sideIndex = " << sideIndex << " )\n";
  pair<int,int> key = make_pair(cellID,sideIndex);
  if ( _boundaryElements.find(key) != _boundaryElements.end() ) {
    _boundaryElements.erase( key );
  }
}

void Boundary::buildLookupTables() {
  _boundaryElementsByType.clear();
  _boundaryCellIDs.clear();
  set< pair< int, int > >::iterator entryIt;
  //cout << "# Boundary entries: " << _boundaryElements.size() << ":\n";
  for (entryIt=_boundaryElements.begin(); entryIt!=_boundaryElements.end(); entryIt++) {
    int cellID = (*entryIt).first;
    TEUCHOS_TEST_FOR_EXCEPTION(cellID < 0, std::invalid_argument, "cellID should be >= 0.");
    Teuchos::RCP< Element > elemPtr = _mesh->elements()[cellID]; 
    int sideIndex = (*entryIt).second;
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    _boundaryElementsByType[elemTypePtr.get()].push_back( make_pair( elemPtr->globalCellIndex(), sideIndex ) );
    _boundaryCellIDs[elemTypePtr.get()].push_back( elemPtr->cellID() );
    //cout << "cellID:\t" << elemPtr->cellID() << "\t sideIndex:\t" << sideIndex << endl;
  }
}

vector< pair< int, int > > Boundary::boundaryElements(Teuchos::RCP< ElementType > elemTypePtr) {
  return _boundaryElementsByType[elemTypePtr.get()];
}

void Boundary::bcsToImpose(FieldContainer<int> &globalIndices, FieldContainer<double> &globalValues, BC &bc, 
                           set<int> &globalIndexFilter) {
  // there's a more efficient way to do this--but it's probably not worth it just yet
  FieldContainer<int> allGlobalIndices;
  FieldContainer<double> allGlobalValues;
  this->bcsToImpose(allGlobalIndices,allGlobalValues,bc);
  set<int> matchingFCIndices;
  int i;
  for (i=0; i<allGlobalIndices.size(); i++) {
    int globalIndex = allGlobalIndices(i);
    if (globalIndexFilter.find(globalIndex) != globalIndexFilter.end() ) {
      matchingFCIndices.insert(i);
    }
  }
  int numIndices = matchingFCIndices.size();
  globalIndices.resize(numIndices);
  globalValues.resize(numIndices);
  
  i=-1;
  for (set<int>::iterator setIt = matchingFCIndices.begin();
       setIt != matchingFCIndices.end(); setIt++) {
    int matchingFCIndex = *setIt;
    i++;
    globalIndices(i) = allGlobalIndices(matchingFCIndex);
    globalValues(i)  =  allGlobalValues(matchingFCIndex);
    //cout << "BC: " << globalIndices(i) << " = " << globalValues(i) << endl;
  }
}

void Boundary::bcsToImpose(FieldContainer<int> &globalIndices,
                           FieldContainer<double> &globalValues, BC &bc) {
  
  // first, let's check for any singletons (one-point BCs)
  map<int,bool> isSingleton;
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    isSingleton[trialID] = bc.singlePointBC(trialID);
  }
  
  map< ElementType*, map< int, double> > bcsForElementType;
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int numIndices = 0;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    map< int, double> bcGlobalIndicesAndValues;
    bcsToImpose( bcGlobalIndicesAndValues, bc, elemTypePtr, isSingleton);
    bcsForElementType[elemTypePtr.get()] = bcGlobalIndicesAndValues;
    numIndices += bcGlobalIndicesAndValues.size();
  }
  
  globalIndices.resize(numIndices);
  globalValues.resize(numIndices);
  globalIndices.initialize(0);
  globalValues.initialize(0.0);
  int currentIndex = 0;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    map< int, double> bcGlobalIndicesAndValues = bcsForElementType[elemTypePtr.get()];
    map< int, double>::iterator bcIt;
    for (bcIt = bcGlobalIndicesAndValues.begin(); bcIt != bcGlobalIndicesAndValues.end(); bcIt++) {
      int index = (*bcIt).first;
      double value = (*bcIt).second;
      globalIndices(currentIndex) = index;
      globalValues(currentIndex) = value;
      //cout << "BC: " << globalIndices(currentIndex) << " = " << globalValues(currentIndex) << endl;
      currentIndex++;
      if (index < 0) {
        TEUCHOS_TEST_FOR_EXCEPTION( true,
                           std::invalid_argument,
                           "bcsToImpose: error: index < 0.");
      }
    }
  }
  
  // check to make sure all our singleton BCs got imposed:
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (isSingleton[trialID]) {
      // that means that it was NOT imposed: warn the user
      cout << "WARNING: singleton BC requested for trial variable " << _mesh->bilinearForm()->trialName(trialID);
      cout << ", but no BC was imposed for this variable (imposeHere never returned true for any point)." << endl;
    }
  }
  
  //cout << "bcsToImpose: globalIndices:" << endl << globalIndices;
}

void Boundary::bcsToImpose( map<  int, double > &globalDofIndicesAndValues, BC &bc, 
                           Teuchos::RCP< ElementType > elemTypePtr, map<int,bool> &isSingleton) {
  // define a couple of important inner products:
  IPPtr ipL2 = Teuchos::rcp( new IP );
  IPPtr ipH1 = Teuchos::rcp( new IP );
  VarFactory varFactory;
  VarPtr trace = varFactory.traceVar("trace");
  VarPtr flux = varFactory.traceVar("flux");
  ipL2->addTerm(flux);
  ipH1->addTerm(trace);
  ipH1->addTerm(trace->grad());
  globalDofIndicesAndValues.clear();
  DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  int spaceDim = elemTypePtr->cellTopoPtr->getDimension();
  int sideDim = spaceDim - 1;
  Teuchos::RCP<Mesh> meshPtr = Teuchos::rcp(_mesh,false); // create an RCP that doesn't own the memory....
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, meshPtr) );
  vector< pair< int, int > > boundaryIndicesForType = _boundaryElementsByType[elemTypePtr.get()];
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    bool isTrace = _mesh->bilinearForm()->functionSpaceForTrial(trialID) == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    // we assume if it's not a trace, then it's a flux (i.e. L2 projection is appropriate)
    if ( bc.bcsImposed(trialID) ) {
      // 1. Collect the physicalCellNodes according to sideIndex
      // 2. Determine global dof indices and values, in one pass per side
      
      // 1. Collect the physicalCellNodes according to sideIndex
      int numSides = elemTypePtr->cellTopoPtr->getSideCount();
      vector< vector<int> > physicalCellIndicesPerSide;
      vector< vector<int> > cellIDsPerSide;
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        vector<int> thisSideIndices;
        physicalCellIndicesPerSide.push_back(thisSideIndices);
        vector<int> cellIDs;
        cellIDsPerSide.push_back(cellIDs);
      }
      vector< pair< int, int > >::iterator indexIterator;
      vector< int >::iterator cellIDIterator = _boundaryCellIDs[elemTypePtr.get()].begin();
      for (indexIterator=boundaryIndicesForType.begin(); indexIterator != boundaryIndicesForType.end(); 
           indexIterator++) {
        int cellIndex = indexIterator->first;
        int sideIndex = indexIterator->second;
        physicalCellIndicesPerSide[sideIndex].push_back(cellIndex);
        cellIDsPerSide[sideIndex].push_back(*cellIDIterator);
        cellIDIterator++;
      }
      Teuchos::Array<int> dimensions;
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
      physicalCellNodes.dimensions(dimensions);
      vector< FieldContainer<double> > physicalCellNodesPerSide;
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        dimensions[0] = physicalCellIndicesPerSide[sideIndex].size();
        FieldContainer<double> physicalCellNodesForSide(dimensions);
        int numCells = dimensions[0];
        int numPoints = dimensions[1];
        int spaceDim = dimensions[2];
        for (int localCellIndex = 0; localCellIndex<numCells; localCellIndex++) {
          int cellIndex = physicalCellIndicesPerSide[sideIndex][localCellIndex];
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            for (int dim=0; dim<spaceDim; dim++) {
              physicalCellNodesForSide(localCellIndex,ptIndex,dim) = physicalCellNodes(cellIndex,ptIndex,dim);
            }
          }
        }
        physicalCellNodesPerSide.push_back(physicalCellNodesForSide);
      }
      
      bool impositionReported = false;
      // 2. Determine global dof indices and values, in one pass per side
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        BasisPtr basis = trialOrderingPtr->getBasis(trialID,sideIndex);
        int numDofs = basis->getCardinality();
        
        int numCells = physicalCellNodesPerSide[sideIndex].dimension(0);
        vector<int> cellIDs = cellIDsPerSide[sideIndex];
        
        if (numCells > 0) {
          FieldContainer<double> dirichletValues(numCells,numDofs);

          // project bc function onto side basis:
          basisCache->setPhysicalCellNodes(physicalCellNodesPerSide[sideIndex],cellIDs,true);
          BCPtr bcPtr = Teuchos::rcp(&bc, false);
          Teuchos::RCP<BCFunction> bcFunction = Teuchos::rcp(new BCFunction(bcPtr, trialID, isTrace));
          // TODO: test the below.  (New as of 9/15/12, basically.)
          // (NOT YET WORKING.  NEED TO ADD SUPPORT FOR grad() to BCFunction, but it's unclear how best to do so...)
//          if (isTrace && (bcFunction->grad().get() != NULL)) { // TODO: in case grad() is NULL, better to interpolate than do the L^2 projection, probably
//            Projector::projectFunctionOntoBasis(dirichletValues, bcFunction, basis, 
//                                                basisCache->getSideBasisCache(sideIndex), ipH1, trace);
//          } else {
//            Projector::projectFunctionOntoBasis(dirichletValues, bcFunction, basis, 
//                                                basisCache->getSideBasisCache(sideIndex), ipL2, flux);
//          }
          // was:
          // Projector::projectFunctionOntoBasis(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideIndex));
          
          // new idea: (trouble with this is that the basisCache already assumes a particular cubature setup, specifically violated when doing straight interpolation.  But this can probably be fixed by the interpolating BC subclass...)
          bcPtr->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideIndex));
//          cout << "imposing values for " << meshPtr->bilinearForm()->trialName(trialID);
//          cout << " at points: \n" << basisCache->getSideBasisCache(sideIndex)->getPhysicalCubaturePoints();
//          cout << "dirichletValues:" << endl << dirichletValues;
          
          for (int localCellIndex=0; localCellIndex<numCells; localCellIndex++) {
            if (bcFunction->imposeOnCell(localCellIndex)) {
              for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
                int cellID = cellIDsPerSide[sideIndex][localCellIndex];
                double value = dirichletValues(localCellIndex,dofOrdinal);
                int localDofIndex = trialOrderingPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                int globalDofIndex = _mesh->globalDofIndex(cellID,localDofIndex);
//                cout << "BC: " << globalDofIndex << " = " << value << " @ (" << dofPointsSidePhysical(localCellIndex,localDofIndex,0) << ", " << dofPointsSidePhysical(localCellIndex,localDofIndex,1) << ")\n";
                if (globalDofIndex < 0) {
                  TEUCHOS_TEST_FOR_EXCEPTION( true,
                                     std::invalid_argument,
                                     "bcsToImpose: error: globalDofIndex < 0.");
                }
                globalDofIndicesAndValues[globalDofIndex] = value;
                if ( ! impositionReported ) {
                  //cout << "imposed BC values for variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
                  impositionReported = true;
                }
              }
            }
          }
        }
      }
    }
  }
  
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    // now, deal with the singletons:
    int trialID = *trialIt;
    TEUCHOS_TEST_FOR_EXCEPTION((isSingleton[trialID]) && ( _mesh->bilinearForm()->isFluxOrTrace(trialID) ),
                       std::invalid_argument,
                       "Singleton BCs on traces and fluxes unsupported...");
    
    if ((isSingleton[trialID]) && ( ! _mesh->bilinearForm()->isFluxOrTrace(trialID) ) ) {
      // (we only support singletons on the interior)
      shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
      // find the physical points for the vertices.
      // nodal H1 basis: we know that one basis function will be 1 at a given vertex,
      // and the others will be 0.
      // (a bit of a hack, but then so is imposing a BC at a single point)
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
      if (physicalCellNodes.dimension(0) > 0) {
        // then we have a cell on which to impose the BC
        int numCells = physicalCellNodes.dimension(0);
        int numPoints = physicalCellNodes.dimension(1);
        int spaceDim = physicalCellNodes.dimension(2);
        
        DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
        BasisPtr basis = trialOrderingPtr->getBasis(trialID,0);
        if (! basis->isNodal()) {
          // could we relax this to just requiring a conforming basis?  I think any conforming basis will be "nodal"
          // with respect to the vertices.  (I.e. each vertex has exactly one basis function on each element that is non-zero there.)
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "single-point BC imposition requires a nodal basis");
        }
        int basisCardinality = basis->getCardinality();
        FieldContainer<double> refPoints(numCells,numPoints,spaceDim);
        CellTools<double>::mapToReferenceFrame(refPoints,physicalCellNodes,physicalCellNodes,cellTopo);
        FieldContainer<double> basisValues(basisCardinality,numPoints);
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          if ( isSingleton[trialID] ) {
            Teuchos::Array<int> dimensions;
            refPoints.dimensions(dimensions);
            //strip off the cellIndex dimension:
            dimensions.erase(dimensions.begin());
            FieldContainer<double> cellRefPoints(dimensions, &refPoints(cellIndex,0,0));
            basis->getValues(basisValues, cellRefPoints, Intrepid::OPERATOR_VALUE);
            
            FieldContainer<int> basisOrdinalForPointFC(numPoints);
            for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
              int basisOrdinalForPoint = -1;
              double tol = 1e-12;
              for (int i=0; i< basisCardinality; i++) {
                if (basisOrdinalForPoint == -1) {
                  if (abs(basisValues(i,ptIndex)-1.0) < tol ) {
                    basisOrdinalForPoint = i;
                  } else if (abs(basisValues(i,ptIndex) > tol) ) {
                    TEUCHOS_TEST_FOR_EXCEPTION(true,
                                       std::invalid_argument,
                                       "basis value at node neither 1.0 nor 0.0"); 
                  }
                } else {
                  if (abs(basisValues(i,ptIndex)-1.0) < tol ) {
                    TEUCHOS_TEST_FOR_EXCEPTION(true,
                                       std::invalid_argument,
                                       "multiple basis values at node == 1.0"); 
                  } else if (abs(basisValues(i,ptIndex)) > tol) {
                    cout << "error: basisValue at node neither 1 nor 0: " << basisValues(i,ptIndex) << endl;
                    TEUCHOS_TEST_FOR_EXCEPTION(true,
                                       std::invalid_argument,
                                       "basis value at node neither 1.0 nor 0.0");               
                  }            
                }
              }
              if (basisOrdinalForPoint == -1) {
                TEUCHOS_TEST_FOR_EXCEPTION(true,
                                   std::invalid_argument,
                                   "no nonzero basis function found at node");
              }
              // otherwise, we have our basis ordinal...
              basisOrdinalForPointFC(ptIndex) = basisOrdinalForPoint;
            }
            FieldContainer<double> sideNormals;
            FieldContainer<double> dirichletValues(numCells,numPoints);
            FieldContainer<bool> imposeHere(numCells,numPoints);
            bc.imposeBC(trialID, physicalCellNodes, sideNormals, dirichletValues, imposeHere);
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
              if ( imposeHere(cellIndex,ptIndex) && isSingleton[trialID] ) {
                int cellID = _mesh->cellID(elemTypePtr, cellIndex);
                int localDofIndex = trialOrderingPtr->getDofIndex(trialID,basisOrdinalForPointFC(ptIndex));
                int globalDofIndex = _mesh->globalDofIndex(cellID, localDofIndex);
                globalDofIndicesAndValues[globalDofIndex] = dirichletValues(cellIndex,ptIndex);
                isSingleton[trialID] = false; // we've imposed it...
//                cout << "imposed singleton BC value " << dirichletValues(cellIndex,ptIndex);
//                cout << " for variable " << _mesh->bilinearForm()->trialName(trialID) << " at point: (";
//                cout << physicalCellNodes(cellIndex,ptIndex,0) << "," << physicalCellNodes(cellIndex,ptIndex,1);
//                cout << ")" << endl;
              }
            }
          }
        }
      }
    }
  }
}
