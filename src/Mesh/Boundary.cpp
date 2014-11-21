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

#include "BasisFactory.h"

#include "BC.h"
#include "BCFunction.h"

#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaCellTools.h"

#include "GlobalDofAssignment.h"
#include "CondensedDofInterpreter.h"

#include "CamelliaDebugUtility.h"

Boundary::Boundary() {
  _mesh = NULL;
}

void Boundary::setMesh(Mesh* mesh) {
  _mesh = mesh;
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
                           set<GlobalIndexType> &globalIndexFilter, DofInterpreter* dofInterpreter, const Epetra_Map *globalDofMap) {
//  int rank = Teuchos::GlobalMPISession::getRank();
//  ostringstream rankLabel;
//  rankLabel << "on rank " << rank << ", globalIndexFilter";
//  Camellia::print(rankLabel.str(), globalIndexFilter);
  
  FieldContainer<GlobalIndexType> allGlobalIndices; // "all" belonging to cells that belong to us...
  FieldContainer<double> allGlobalValues;
  this->bcsToImpose(allGlobalIndices,allGlobalValues,bc, dofInterpreter, globalDofMap);
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
                           FieldContainer<double> &globalValues, BC &bc,
                           DofInterpreter* dofInterpreter, const Epetra_Map *globalDofMap) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  set< GlobalIndexType > rankLocalCells = _mesh->cellIDsInPartition();
  
  // first, let's check for any singletons (one-point BCs)
  map<IndexType, set < pair<int, unsigned> > > singletonsForCell;
  
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (bc.singlePointBC(trialID)) {
      GlobalIndexType vertexNumberForImposition = bc.vertexForSinglePointBC(trialID);
      if (vertexNumberForImposition == -1) vertexNumberForImposition = 0; // just pick the first point in the mesh.  This should never be a hanging node.
      unsigned vertexDim = 0;
      IndexType leastActiveCellIndex = -1;
      set< pair<IndexType, unsigned> > cellsForVertex = _mesh->getTopology()->getCellsContainingEntity(vertexDim, vertexNumberForImposition);
      for (set< pair<IndexType, unsigned> >::iterator cellEntryIt = cellsForVertex.begin(); cellEntryIt != cellsForVertex.end(); cellEntryIt++) {
        IndexType cellIndex = cellEntryIt->first;
        if (_mesh->getTopology()->getActiveCellIndices().find(cellIndex) != _mesh->getTopology()->getActiveCellIndices().end()) {
          leastActiveCellIndex = cellIndex;
          break;
        }
      }
      if (rankLocalCells.find(leastActiveCellIndex) != rankLocalCells.end()) { // we own this cell, so we're responsible for imposing the singleton BC
        CellPtr cell = _mesh->getTopology()->getCell(leastActiveCellIndex);
        int vertexCount = cell->vertices().size();
        for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++) {
          if (vertexNumberForImposition == cell->vertices()[vertexOrdinal]) {
            singletonsForCell[leastActiveCellIndex].insert(make_pair(trialID, vertexOrdinal));
          }
        }
      }
    }
  }
  
  map< GlobalIndexType, double> bcGlobalIndicesAndValues;
  set < pair<int, unsigned> > noSingletons;
  
  for (set< GlobalIndexType >::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++) {
    if (singletonsForCell.find(*cellIDIt) != singletonsForCell.end()) {
      bcsToImpose(bcGlobalIndicesAndValues, bc, *cellIDIt, singletonsForCell[*cellIDIt], dofInterpreter, globalDofMap);
    } else {
      bcsToImpose(bcGlobalIndicesAndValues, bc, *cellIDIt, noSingletons, dofInterpreter, globalDofMap);
    }
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
  
//  // check to make sure all our singleton BCs got imposed:
//  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
//    int trialID = *trialIt;
//    if ((isSingleton[trialID]) && _imposeSingletonBCsOnThisRank) {
//      // that means that it was NOT imposed: warn the user
//      cout << "WARNING: singleton BC requested for trial variable " << _mesh->bilinearForm()->trialName(trialID);
//      cout << ", but no BC was imposed for this variable (possibly because imposeHere never returned true for any point)." << endl;
//    }
//  }
  
  //cout << "bcsToImpose: globalIndices:" << endl << globalIndices;
}

void Boundary::bcsToImpose( map<  GlobalIndexType, double > &globalDofIndicesAndValues, BC &bc,
                           GlobalIndexType cellID, set < pair<int, unsigned> > &singletons,
                           DofInterpreter* dofInterpreter, const Epetra_Map *globalDofMap) {
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
  Teuchos::RCP<Mesh> meshPtr = Teuchos::rcp(_mesh,false); // create an RCP that doesn't own the memory....
  
  vector<unsigned> boundarySides = cell->boundarySides();
  if (boundarySides.size() > 0) {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(meshPtr, cellID);
    for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      bool isTrace = _mesh->bilinearForm()->functionSpaceForTrial(trialID) == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
      // we assume if it's not a trace, then it's a flux (i.e. L2 projection is appropriate)
      if ( bc.bcsImposed(trialID) ) {
        // Determine global dof indices and values, in one pass per side
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
              dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, dirichletValues, globalData, globalDofIndices);
                for (int globalDofOrdinal=0; globalDofOrdinal<globalDofIndices.size(); globalDofOrdinal++) {
                  GlobalIndexType globalDofIndex = globalDofIndices(globalDofOrdinal);
                  globalDofIndicesAndValues[globalDofIndex] = globalData(globalDofOrdinal);
                }
              }
            }
          }
        }
    }
  }
  
  for (set<pair<int, unsigned> >::iterator singletonIt = singletons.begin(); singletonIt != singletons.end(); singletonIt++) {
    int trialID = singletonIt->first;
    unsigned vertexOrdinalInCell = singletonIt->second;
    
    shards::CellTopology cellTopo = *(elemType->cellTopoPtr.get());

    // in some ways less nice than the previous version of singleton BC imposition; we don't impose a non-zero value (because
    // we don't figure out physical points, etc.), and we also neglect any spatial filtering.
    
    // OTOH, here we do support traces and fluxes for single-point BCs, and this is significantly simpler
    
    set<GlobalIndexType> globalIndicesForVariable;
    DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
    
    IndexType vertexIndex = cell->vertices()[vertexOrdinalInCell];
    int numSides = cell->getSideCount();
    
    int vertexOrdinal;
    
    int sideForVertex = -1;
    int sideDim = cellTopo.getDimension() - 1;
    if (!_mesh->bilinearForm()->isFluxOrTrace(trialID)) {
      vertexOrdinal = vertexOrdinalInCell;
      sideForVertex = 0; // for volume trialIDs, the "side" in DofOrdering is 0
    } else {
      int vertexOrdinalInSide = -1;
      for (int sideOrdinal=0; sideOrdinal < numSides; sideOrdinal++) {
        vector<IndexType> vertexIndicesForSide = cell->getEntityVertexIndices(sideDim, sideOrdinal);
        for (int vertexOrdinal=0; vertexOrdinal < vertexIndicesForSide.size(); vertexOrdinal++) {
          if (vertexIndicesForSide[vertexOrdinal] == vertexIndex) {
            sideForVertex = sideOrdinal;
            vertexOrdinalInSide = vertexOrdinal;
            break;
          }
        }
        if (sideForVertex != -1) break;
      }
      if (sideForVertex == -1) {
        cout << "ERROR: sideForVertex not found during singleton BC imposition.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideForVertex not found during singleton BC imposition");
      }
      vertexOrdinal = vertexOrdinalInSide;
    }
    BasisPtr basis = trialOrderingPtr->getBasis(trialID,sideForVertex);
    
    // upgrade basis to continuous one of the same cardinality, if it is discontinuous.
    if ((basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) || (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL_DISC)) {
      basis = BasisFactory::basisFactory()->getBasis(basis->getDegree(), basis->domainTopology(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    } else if (IntrepidExtendedTypes::functionSpaceIsDiscontinuous(basis->functionSpace())) {
      IntrepidExtendedTypes::EFunctionSpaceExtended fsContinuous = IntrepidExtendedTypes::continuousSpaceForDiscontinuous((basis->functionSpace()));
      basis = BasisFactory::basisFactory()->getBasis(basis->getDegree(), basis->domainTopology(), fsContinuous);
    }
    
    std::set<int> dofOrdinals = basis->dofOrdinalsForVertex(vertexOrdinal);
    if (dofOrdinals.size() != 1) {
      cout << "ERROR: dofOrdinals.size() != 1 during singleton BC imposition.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "dofOrdinals.size() != 1 during singleton BC imposition");
    }
    int dofOrdinal = *dofOrdinals.begin();
    int basisCardinality = basis->getCardinality();
    FieldContainer<double> basisCoefficients(basisCardinality);
    basisCoefficients[dofOrdinal] = 1.0;
    FieldContainer<double> globalCoefficients; // we'll ignore this
    FieldContainer<GlobalIndexType> globalDofIndices;
    dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideForVertex, basisCoefficients,
                                                    globalCoefficients, globalDofIndices);
    double tol = 1e-14;
    int nonzeroEntryOrdinal = -1;
    for (int fieldOrdinal=0; fieldOrdinal < globalCoefficients.size(); fieldOrdinal++) {
      if (abs(globalCoefficients[fieldOrdinal]) > tol) {
        if (nonzeroEntryOrdinal != -1) {
          // previous nonzero entry found; this is a problem--it means we have multiple global coefficients that depend on this vertex
          // (could happen if user specified a hanging node)
          cout << "Error: vertex for single-point imposition has multiple global degrees of freedom.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: vertex for single-point imposition has multiple global degrees of freedom.");
        }
        // nonzero entry: store the fact, and impose the constraint
        nonzeroEntryOrdinal = fieldOrdinal;
        if (globalDofMap != NULL) {
          if (globalDofMap->LID((GlobalIndexTypeToCast)globalDofIndices[fieldOrdinal]) != -1) {
            globalDofIndicesAndValues[globalDofIndices[fieldOrdinal]] = bc.valueForSinglePointBC(trialID) * globalCoefficients[fieldOrdinal];
          } else {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: global dof index for single-point BC is not locally owned");
          }
        } else {
          // _globalDofMap not set, so presumably mesh->GDA sees an accurate picture of the global dofs.
          set<GlobalIndexType> rankLocalDofIndices = _mesh->globalDofAssignment()->globalDofIndicesForPartition(-1); // current rank
          if (rankLocalDofIndices.find(globalDofIndices[fieldOrdinal]) != rankLocalDofIndices.end()) {
            globalDofIndicesAndValues[globalDofIndices[fieldOrdinal]] = 0.0;
          } else {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: global dof index for single-point BC is not locally owned");
          }
        }
      }
    }
  }
}
