/*
 *  Boundary.cpp
 *
 */

// @HEADER
//
// Copyright © 2014 Nathan V. Roberts. All Rights Reserved.
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

#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaCellTools.h"

#include "GlobalDofAssignment.h"
#include "CondensedDofInterpreter.h"

#include "CamelliaDebugUtility.h"

Boundary::Boundary() {
  _dofInterpreter = NULL;
  _mesh = NULL;
}

void Boundary::setDofInterpreter(DofInterpreter* dofInterpreter, Teuchos::RCP<Epetra_Map> globalDofMap) { // must be called after setMesh(); setMesh will also set dofInterpreter to be the mesh
  _dofInterpreter = dofInterpreter;
  _globalDofMap = globalDofMap;
}

void Boundary::setMesh(Mesh* mesh) {
  _mesh = mesh;
  _dofInterpreter = _mesh;
  buildLookupTables();
}

void Boundary::buildLookupTables() {
  _boundaryElements.clear();
  
  int rank = Teuchos::GlobalMPISession::getRank();
  
  set< GlobalIndexType > rankLocalCells = _mesh->cellIDsInPartition();
  for (set< GlobalIndexType >::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellPtr cell = _mesh->getTopology()->getCell(cellID);
    vector<unsigned> boundarySides = cell->boundarySides();
    for (int i=0; i<boundarySides.size(); i++) {
      _boundaryElements.insert(make_pair(cellID, boundarySides[i]));
    }
  }
  
  _imposeSingletonBCsOnThisRank = (_mesh->globalDofAssignment()->cellsInPartition(rank).size() > 0);  // want this to be true for the first rank that has some active cells
  for (int i=0; i<rank; i++) {
    int activeCellCount = _mesh->globalDofAssignment()->cellsInPartition(i).size();
    if (activeCellCount > 0) {
      _imposeSingletonBCsOnThisRank = false;
      break;
    }
  }
}

void Boundary::bcsToImpose(FieldContainer<GlobalIndexType> &globalIndices, FieldContainer<double> &globalValues, BC &bc,
                           set<GlobalIndexType> &globalIndexFilter) {
//  int rank = Teuchos::GlobalMPISession::getRank();
//  ostringstream rankLabel;
//  rankLabel << "on rank " << rank << ", globalIndexFilter";
//  Camellia::print(rankLabel.str(), globalIndexFilter);
  
  FieldContainer<GlobalIndexType> allGlobalIndices; // "all" belonging to cells that belong to us...
  FieldContainer<double> allGlobalValues;
  this->bcsToImpose(allGlobalIndices,allGlobalValues,bc);
//  cout << "rank " << rank << " allGlobalIndices:\n" << allGlobalIndices;
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
//    cout << "BC: " << globalIndices(i) << " = " << globalValues(i) << endl;
  }
}

void Boundary::bcsToImpose(FieldContainer<GlobalIndexType> &globalIndices,
                           FieldContainer<double> &globalValues, BC &bc) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  // first, let's check for any singletons (one-point BCs)
  map<int,bool> isSingleton;
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    isSingleton[trialID] = bc.singlePointBC(trialID);
  }
  
  set< GlobalIndexType > rankLocalCells = _mesh->cellIDsInPartition();
  map< GlobalIndexType, double> bcGlobalIndicesAndValues;
  for (set< GlobalIndexType >::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++) {
    bcsToImpose(bcGlobalIndicesAndValues, bc, *cellIDIt, isSingleton);
  }
  
  globalIndices.resize(bcGlobalIndicesAndValues.size());
  globalValues.resize(bcGlobalIndicesAndValues.size());
  globalIndices.initialize(0);
  globalValues.initialize(0.0);
  int entryOrdinal = 0;
  for (map< GlobalIndexType, double>::iterator bcEntry = bcGlobalIndicesAndValues.begin(); bcEntry != bcGlobalIndicesAndValues.end(); bcEntry++, entryOrdinal++) {
    globalIndices[entryOrdinal] = bcEntry->first;
    globalValues[entryOrdinal] = bcEntry->second;
  }
  
  // check to make sure all our singleton BCs got imposed:
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if ((isSingleton[trialID]) && _imposeSingletonBCsOnThisRank) {
      // that means that it was NOT imposed: warn the user
      cout << "WARNING: singleton BC requested for trial variable " << _mesh->bilinearForm()->trialName(trialID);
      cout << ", but no BC was imposed for this variable (possibly because imposeHere never returned true for any point)." << endl;
    }
  }
  
  //cout << "bcsToImpose: globalIndices:" << endl << globalIndices;
}

void Boundary::bcsToImpose( map<  GlobalIndexType, double > &globalDofIndicesAndValues, BC &bc, GlobalIndexType cellID, map<int,bool> &isSingleton) {
  int rank = Teuchos::GlobalMPISession::getRank();

  CellPtr cell = _mesh->getTopology()->getCell(cellID);
  
  // define a couple of important inner products:
  IPPtr ipL2 = Teuchos::rcp( new IP );
  IPPtr ipH1 = Teuchos::rcp( new IP );
  VarFactory varFactory;
  VarPtr trace = varFactory.traceVar("trace");
  VarPtr flux = varFactory.traceVar("flux");
  ipL2->addTerm(flux);
  ipH1->addTerm(trace);
  ipH1->addTerm(trace->grad());
  ElementTypePtr elemType = _mesh->getElementType(cellID);
  DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  int spaceDim = elemType->cellTopoPtr->getDimension();
  int sideDim = spaceDim - 1;
  Teuchos::RCP<Mesh> meshPtr = Teuchos::rcp(_mesh,false); // create an RCP that doesn't own the memory....
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(meshPtr, cellID);
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    bool isTrace = _mesh->bilinearForm()->functionSpaceForTrial(trialID) == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    // we assume if it's not a trace, then it's a flux (i.e. L2 projection is appropriate)
    if ( bc.bcsImposed(trialID) ) {
      vector<unsigned> boundarySides = cell->boundarySides();
      
      // 2. Determine global dof indices and values, in one pass per side
      for (int i=0; i<boundarySides.size(); i++) {
        unsigned sideOrdinal = boundarySides[i];
        BasisPtr basis = trialOrderingPtr->getBasis(trialID,sideOrdinal);
        int numDofs = basis->getCardinality();
        GlobalIndexType numCells = 1;
        if (numCells > 0) {
          FieldContainer<double> dirichletValues(numCells,numDofs);
          // project bc function onto side basis:
          BCPtr bcPtr = Teuchos::rcp(&bc, false);
          Teuchos::RCP<BCFunction> bcFunction = BCFunction::bcFunction(bcPtr, trialID, isTrace);
          bcPtr->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideOrdinal));
          dirichletValues.resize(numDofs);
          if (bcFunction->imposeOnCell(0)) {
            FieldContainer<double> globalData;
            FieldContainer<GlobalIndexType> globalDofIndices;
            _dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, dirichletValues, globalData, globalDofIndices);
              for (int globalDofOrdinal=0; globalDofOrdinal<globalDofIndices.size(); globalDofOrdinal++) {
                GlobalIndexType globalDofIndex = globalDofIndices(globalDofOrdinal);
                globalDofIndicesAndValues[globalDofIndex] = globalData(globalDofOrdinal);
              }
            }
          }
        }
      }
  }
  
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    // now, deal with the singletons:
    int trialID = *trialIt;
    
    shards::CellTopology cellTopo = *(elemType->cellTopoPtr.get());
    
    if (! _mesh->meshUsesMaximumRule()) {
      // a bit less nice than the maximum rule singleton BC imposition; we don't impose a non-zero value (because
      // we don't figure out physical points, etc.), and we also neglect any spatial filtering.
      // We just find some GlobalDofIndex that belongs to the variable in question, and impose zero on it.
      
      // OTOH, here we do support traces and fluxes for single-point BCs
      if (_imposeSingletonBCsOnThisRank) {
        if (isSingleton[trialID]) {
        
          set<GlobalIndexType> globalIndicesForVariable;
          DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
          
          int numSides = _mesh->bilinearForm()->isFluxOrTrace(trialID) ? cellTopo.getSideCount() : 1;
          
          for (int sideOrdinal=0; sideOrdinal < numSides; sideOrdinal++) {
            BasisPtr basis = trialOrderingPtr->getBasis(trialID,sideOrdinal);
            int basisCardinality = basis->getCardinality();
            FieldContainer<double> basisCoefficients(basisCardinality);
            FieldContainer<double> globalCoefficients; // we'll ignore this
            FieldContainer<GlobalIndexType> globalDofIndices;
            _dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients,
                                                             globalCoefficients, globalDofIndices);
            if (_globalDofMap.get() != NULL) {
              for (int i=0; i<globalDofIndices.size(); i++) {
                if (_globalDofMap->LID((GlobalIndexTypeToCast)globalDofIndices[i]) != -1) {
                  globalDofIndicesAndValues[globalDofIndices[i]] = 0.0;
                  cout << "Imposed single-point BC on variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
                  isSingleton[trialID] = false; // we've imposed it...
                  break;
                }
              }
              
            } else {
              // _globalDofMap not set, so presumably mesh->GDA sees an accurate picture of the global dofs.
              set<GlobalIndexType> rankLocalDofIndices = _mesh->globalDofAssignment()->globalDofIndicesForPartition(-1); // current rank
              for (int i=0; i<globalDofIndices.size(); i++) {
                if (rankLocalDofIndices.find(globalDofIndices[i]) != rankLocalDofIndices.end()) {
                  globalDofIndicesAndValues[globalDofIndices[i]] = 0.0;
                  cout << "Imposed single-point BC on variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
                  isSingleton[trialID] = false; // we've imposed it...
                  break;
                }
              }
            }
            if (!isSingleton[trialID]) { // imposed it, so skip other sides
              break;
            }
          }
        }
      }
    } else { // maximum rule
      TEUCHOS_TEST_FOR_EXCEPTION((isSingleton[trialID]) && ( _mesh->bilinearForm()->isFluxOrTrace(trialID) ),
                               std::invalid_argument,
                               "Singleton BCs on traces and fluxes unsupported...");

    if ((isSingleton[trialID]) && ( ! _mesh->bilinearForm()->isFluxOrTrace(trialID) ) ) {
      // (we only support singletons on the interior)
      // find the physical points for the vertices.
      // nodal H1 basis: we know that one basis function will be 1 at a given vertex,
      // and the others will be 0.
      // (a bit of a hack, but then so is imposing a BC at a single point)
        // then we have a cell on which to impose the BC
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
      int numCells = physicalCellNodes.dimension(0);
      int numNodes = physicalCellNodes.dimension(1);
      int spaceDim = physicalCellNodes.dimension(2);
        
      DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
      BasisPtr basis = trialOrderingPtr->getBasis(trialID,0);
      if (! basis->isNodal()) {
        // could we relax this to just requiring a conforming basis?  I think any conforming basis will be "nodal"
        // with respect to the vertices.  (I.e. each vertex has exactly one basis function on each element that is non-zero there.)
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "single-point BC imposition requires a nodal basis");
      }
      int basisCardinality = basis->getCardinality();
      FieldContainer<double> refNodes(numNodes,spaceDim);
      CamelliaCellTools::refCellNodesForTopology(refNodes, cellTopo);
      
      FieldContainer<double> basisValues(basisCardinality,numNodes);
      if ( isSingleton[trialID] ) {
        if (_mesh->meshUsesMaximumRule()) {
          basis->getValues(basisValues, refNodes, Intrepid::OPERATOR_VALUE);
          
          FieldContainer<int> basisOrdinalForPointFC(numNodes);
          for (int ptIndex=0; ptIndex < numNodes; ptIndex++) {
            int basisOrdinalForPoint = -1;
            double tol = 1e-12;
            for (int i=0; i< basisCardinality; i++) {
              if (basisOrdinalForPoint == -1) {
                if (abs(basisValues(i,ptIndex)-1.0) < tol ) {
                  basisOrdinalForPoint = i;
                } else if (abs(basisValues(i,ptIndex)) > tol ) {
                  cout << "basis value at node " << ptIndex << " is neither 1 nor 0.  Values:" << endl;
                  cout << basisValues;
                  cout << "Reference points:\n" << refNodes;
                  cout << "Physical cell nodes:\n" << physicalCellNodes;
                  
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
          FieldContainer<double> dirichletValues(numCells,numNodes);
          FieldContainer<bool> imposeHere(numCells,numNodes);
          bc.imposeBC(trialID, physicalCellNodes, sideNormals, dirichletValues, imposeHere);

          for (int ptIndex=0; ptIndex<numNodes; ptIndex++) {
            if ( imposeHere(0,ptIndex) && isSingleton[trialID] && _imposeSingletonBCsOnThisRank ) { // only impose singleton BCs on lowest rank with active cells
              int localDofIndex = trialOrderingPtr->getDofIndex(trialID,basisOrdinalForPointFC(ptIndex));
              GlobalIndexType globalDofIndex = _mesh->globalDofIndex(cellID, localDofIndex);;
              if (_dofInterpreter != NULL) {
                // then map from mesh's view to the condensed view of the global dof index
                // (only supported for CondensedDofInterpreter right now)
                CondensedDofInterpreter* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(_dofInterpreter);
                
                if (condensedDofInterpreter != NULL) {
                  globalDofIndex = condensedDofInterpreter->condensedGlobalIndex(globalDofIndex);
                } else {
                  Mesh* meshAsInterpreter = dynamic_cast<Mesh*>(_dofInterpreter);
                  if (meshAsInterpreter == NULL) {
                    cout << "Unsupported dof interpreter for singleton BCs.\n";
                    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported dof interpreter for singleton BCs");
                  }
                }

              }
              globalDofIndicesAndValues[globalDofIndex] = dirichletValues(0,ptIndex);
              isSingleton[trialID] = false; // we've imposed it...
              break;

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
