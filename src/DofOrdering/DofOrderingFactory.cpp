/*
 *  DofOrderingFactory.cpp
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

#include "BasisFactory.h"

#include "DofOrderingFactory.h"

#include "BilinearForm.h"

typedef Teuchos::RCP<Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;

DofOrderingFactory::DofOrderingFactory(Teuchos::RCP<BilinearForm> bilinearForm) {
  _bilinearForm = bilinearForm;
}

DofOrderingPtr DofOrderingFactory::testOrdering(int polyOrder, 
                                                           const shards::CellTopology &cellTopo) {
  vector<int> testIDs = _bilinearForm->testIDs();
  vector<int>::iterator testIterator;
  
  DofOrderingPtr testOrder = Teuchos::rcp(new DofOrdering());
  
  for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
    int testID = *testIterator;
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = _bilinearForm->functionSpaceForTest(testID);
    Teuchos::RCP< Intrepid::Basis<double,FieldContainer<double> > > basis;
    int basisRank;
    basis = BasisFactory::getBasis( basisRank, polyOrder, cellTopo.getKey(), fs);
    testOrder->addEntry(testID,basis,basisRank);
  }
  
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  return *(_testOrderings.insert(testOrder).first); 
}

DofOrderingPtr DofOrderingFactory::trialOrdering(int polyOrder, 
                                                 const shards::CellTopology &cellTopo,
                                                 bool conformingVertices) {
  // right now, only works for 2D topologies
  vector<int> trialIDs = _bilinearForm->trialIDs();
  vector<int>::iterator trialIterator;
  
  DofOrderingPtr trialOrder = Teuchos::rcp(new DofOrdering());
  
  for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++) {
    int trialID = *trialIterator;
    
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = _bilinearForm->functionSpaceForTrial(trialID);
    
    BasisPtr basis;
    
    int basisRank;
    
    if (_bilinearForm->isFluxOrTrace(trialID)) { //lines, in 2D case (TODO: extend to arbitrary dimension)
      int numSides = cellTopo.getSideCount();
      basis = BasisFactory::getBasis( basisRank, polyOrder, shards::Line<2>::key, fs);
      for (int j=0; j<numSides; j++) {
        trialOrder->addEntry(trialID,basis,basisRank,j);
      }
      if ( conformingVertices
          && fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) {
        // then we want to identify basis dofs at the vertices...
        
        addConformingVertexPairings(trialID, trialOrder, cellTopo);
      }
    } else {
      basis = BasisFactory::getBasis( basisRank, polyOrder, cellTopo.getKey(), fs);
      trialOrder->addEntry(trialID,basis,basisRank,0);
    }
  }
  trialOrder->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  trialOrder = *(_trialOrderings.insert(trialOrder).first);
  _isConforming[trialOrder.get()] = conformingVertices;
  return trialOrder;
}

DofOrderingPtr DofOrderingFactory::getTrialOrdering(DofOrdering &ordering) {
  DofOrderingPtr orderingPtr = Teuchos::rcp(&ordering,false);
  set<DofOrderingPtr, Comparator >::iterator orderingIt = _trialOrderings.find(orderingPtr);
  if ( orderingIt != _trialOrderings.end() ) {
    return *orderingIt;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ordering not found");
}

DofOrderingPtr DofOrderingFactory::getTestOrdering(DofOrdering &ordering) {
  DofOrderingPtr orderingPtr = Teuchos::rcp(&ordering,false);
  set<DofOrderingPtr, Comparator >::iterator orderingIt = _testOrderings.find(orderingPtr);
  if ( orderingIt != _testOrderings.end() ) {
    return *orderingIt;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ordering not found");
}

void DofOrderingFactory::addConformingVertexPairings(int varID, DofOrderingPtr dofOrdering,
                                                     const shards::CellTopology &cellTopo) {
  // then we want to identify basis dofs at the vertices...
  map< int, pair<int,int> > cellVertexOrdinalToSideVertexOrdinal; // vertexOrdinal --> pair<sideNumber, vertexNumber>
  int numSides = cellTopo.getSideCount();
  for (int j=0; j<numSides; j++) {
    int numVerticesPerSide = cellTopo.getVertexCount(1,j); // should be 2
    for (int i=0; i < numVerticesPerSide; i++) {
      unsigned vertexOrdinal = cellTopo.getNodeMap(1,j,i);
      if ( cellVertexOrdinalToSideVertexOrdinal.find(vertexOrdinal) 
          == cellVertexOrdinalToSideVertexOrdinal.end() ) {
        // haven't seen this one yet
        cellVertexOrdinalToSideVertexOrdinal[vertexOrdinal] = make_pair(j,i);
      } else {
        pair<int,int> pairedSideVertex = cellVertexOrdinalToSideVertexOrdinal[vertexOrdinal];
        int firstSide = pairedSideVertex.first;
        int firstVertex = pairedSideVertex.second;
        int secondSide = j;
        int secondVertex = i;
        BasisPtr firstBasis = dofOrdering->getBasis(varID,firstSide);
        BasisPtr secondBasis = dofOrdering->getBasis(varID,secondSide);
        int firstDofOrdinal = firstBasis->getDofOrdinal(0,firstVertex,0);
        int secondDofOrdinal = secondBasis->getDofOrdinal(0,secondVertex,0);
        dofOrdering->addIdentification(varID,firstSide,firstDofOrdinal,
                                       secondSide,secondDofOrdinal);
      }
    }
  }
}

int DofOrderingFactory::polyOrder(DofOrderingPtr dofOrdering) {
  set<int> varIDs = dofOrdering->getVarIDs();
  set<int>::iterator idIt;
  int interiorVariable;
  bool interiorVariableFound = false;
  int minSidePolyOrder = INT_MAX;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = dofOrdering->getNumSidesForVarID(varID);
    if (numSides == 1) {
      interiorVariable = varID;
      interiorVariableFound = true;
      break;
    } else {
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        int polyOrder = BasisFactory::basisPolyOrder( dofOrdering->getBasis(varID,sideIndex) );
        minSidePolyOrder = min(minSidePolyOrder,polyOrder);
      }
    }
  }
  if ( ! interiorVariableFound) {
    // all side variables, which is a bit weird
    // if we have some idea of what the minimum poly order is for a side, then we return that.
    // otherwise, throw an exception
    TEUCHOS_TEST_FOR_EXCEPTION( minSidePolyOrder == INT_MAX,
                       std::invalid_argument,
                       "DofOrdering appears not to have any interior (volume) varIDs--DofOrderingFactory cannot pRefine.");
    return minSidePolyOrder;
  }
  BasisPtr interiorBasis = dofOrdering->getBasis(interiorVariable);
  return BasisFactory::basisPolyOrder(interiorBasis);
}

map<int, BasisPtr> DofOrderingFactory::getMultiBasisUpgradeMap(vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide) {
  vector< BasisPtr > bases;
  set<int> varIDs = (childTrialOrdersForSide[0].first)->getVarIDs();
  map<int, BasisPtr> varIDsToUpgrade;
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = childTrialOrdersForSide[0].first->getNumSidesForVarID(varID);
    if (numSides > 1) { // a variable that lives on the sides: we need to match basis
      int numBases = childTrialOrdersForSide.size();
      for (int i=numBases-1; i>=0; i-- ) {
        // reverse order of bases ( valid only for 2D; in 3D we'll need a more general permutation )
        int childSideIndex = childTrialOrdersForSide[i].second;
        BasisPtr basis  = childTrialOrdersForSide[i].first->getBasis(varID,childSideIndex);
        bases.push_back(basis);
      }
      if (bases.size() != 1) {
        BasisPtr multiBasis = BasisFactory::getMultiBasis(bases);
        varIDsToUpgrade[varID] = multiBasis;
      } else {
        varIDsToUpgrade[varID] = bases[0];
      }
      bases.clear();
    }
  }
  return varIDsToUpgrade;
}

map<int, BasisPtr> DofOrderingFactory::getPatchBasisUpgradeMap(const DofOrderingPtr childTrialOrdering, int childSideIndex,
                                                               const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
                                                               int childIndexInParentSide) {
  set<int> varIDs = childTrialOrdering->getVarIDs();
  map<int, BasisPtr> varIDsToUpgrade;
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = childTrialOrdering->getNumSidesForVarID(varID);
    if (numSides > 1) { // a variable that lives on the sides: we need to match basis
      BasisPtr basis = parentTrialOrdering->getBasis(varID,parentSideIndex);
      FieldContainer<double> nodes(2,1); // just 1D patches supported right now
      if (childIndexInParentSide==0) {
        nodes(0,0) = -1.0;
        nodes(1,0) = 0.0;
      } else {
        nodes(0,0) = 0.0;
        nodes(1,0) = 1.0;
      }
      BasisPtr patchBasis = BasisFactory::getPatchBasis(basis, nodes);
      varIDsToUpgrade[varID] = patchBasis;
    }
  }
  return varIDsToUpgrade;
}

void DofOrderingFactory::assignMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex, 
                                          const shards::CellTopology &cellTopo,
                                          vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide ) {
  map<int, BasisPtr> varIDsToUpgrade = getMultiBasisUpgradeMap(childTrialOrdersForSide);
  trialOrdering = upgradeSide(trialOrdering,cellTopo,varIDsToUpgrade,sideIndex);
}

void DofOrderingFactory::assignPatchBasis(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                                          const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
                                          int childIndexInParentSide, const shards::CellTopology &childCellTopo) {
  TEUCHOS_TEST_FOR_EXCEPTION(childIndexInParentSide >= 2, std::invalid_argument, "assignPatchBasis only supports 2 children on a side right now.");
  map<int, BasisPtr> varIDsToUpgrade = getPatchBasisUpgradeMap(childTrialOrdering, childSideIndex, parentTrialOrdering,
                                                               parentSideIndex, childIndexInParentSide);
  childTrialOrdering = upgradeSide(childTrialOrdering,childCellTopo,varIDsToUpgrade,childSideIndex);
}

bool DofOrderingFactory::sideHasMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex) {
  set<int> varIDs = trialOrdering->getVarIDs();
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = trialOrdering->getNumSidesForVarID(varID);
    if (numSides > 1) { // a variable that lives on the sides
      BasisPtr basis = trialOrdering->getBasis(varID,sideIndex);
      // as one side basis goes, so go they all:
      return BasisFactory::isMultiBasis(basis);
    }
  }
  // if we get here, we didn't really have a side...
  return false;
}

void DofOrderingFactory::childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                                          const shards::CellTopology &childTopo,
                                          int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
                                          DofOrderingPtr &parentTrialOrdering, int sideIndex,
                                          const shards::CellTopology &parentTopo) {
  // basic strategy: if parent has MultiBasis on that side, then child should get a piece of that
  //                 otherwise, we can simply use matchSides as it is...
  if ( sideHasMultiBasis(parentTrialOrdering,sideIndex) ) {
    set<int> varIDs = parentTrialOrdering->getVarIDs();
    map<int, BasisPtr> varIDsToUpgrade;
    set<int>::iterator idIt;
    for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
      int varID = *idIt;
      int numSides = parentTrialOrdering->getNumSidesForVarID(varID);
      if (numSides > 1) { // a variable that lives on the sides: we need to match basis
        BasisPtr basis  = parentTrialOrdering->getBasis(varID,sideIndex);
        if (! BasisFactory::isMultiBasis(basis) ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "if one basis is multibasis, they all should be");
        }
        MultiBasis* multiBasis = (MultiBasis*) basis.get();
        varIDsToUpgrade[varID] = multiBasis->getSubBasis(childIndexInParentSide);
      }
    }
    childTrialOrdering = upgradeSide(childTrialOrdering,childTopo,varIDsToUpgrade,childSideIndex);
  } else {
    int upgradedSide = matchSides(childTrialOrdering,childSideIndex,childTopo,
                                  parentTrialOrdering,sideIndex,parentTopo);
    if (upgradedSide == 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "parent should never be upgraded!");  
    }
  }
}

int DofOrderingFactory::matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex, 
                                    const shards::CellTopology &firstCellTopo,
                                    DofOrderingPtr &secondOrdering, int secondSideIndex,
                                    const shards::CellTopology &secondCellTopo) {
  // upgrades the lesser-order basis 
  map<int, BasisPtr> varIDsToUpgrade;
  int orderingToUpgrade = 0; // 0 means neither, 1 first, 2 second, -1 means PatchBasis (i.e. can't matchSides w/o more Mesh info)
  set<int> varIDs = firstOrdering->getVarIDs();
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = firstOrdering->getNumSidesForVarID(varID);
    if (numSides > 1) { // a variable that lives on the sides: we need to match basis
      BasisPtr firstBasis = firstOrdering->getBasis(varID,firstSideIndex);
      BasisPtr secondBasis = secondOrdering->getBasis(varID,secondSideIndex);
      if (BasisFactory::isPatchBasis(firstBasis) || BasisFactory::isPatchBasis(secondBasis)) {
        return -1; // then we need to deal with ancestors, etc.--and we can't do that here
      }
      
      // use cardinality instead of degree to compare so that multiBasis > singleBasis
      if ( firstBasis->getCardinality() > secondBasis->getCardinality() ) {
        if (orderingToUpgrade == 1) {
          TEUCHOS_TEST_FOR_EXCEPTION( true,
                             std::invalid_argument,
                             "DofOrderings vary in terms of which has higher degree.  Unhandled case in DofOrderingFactory.");
        }
        // otherwise
        orderingToUpgrade = 2;
        varIDsToUpgrade[varID] = firstBasis;
      } else if (secondBasis->getCardinality() > firstBasis->getCardinality() ) {
        if (orderingToUpgrade == 2) {
          TEUCHOS_TEST_FOR_EXCEPTION( true,
                             std::invalid_argument,
                             "DofOrderings vary in terms of which has higher degree.  Unhandled case in DofOrderingFactory.");
        }
        // otherwise
        orderingToUpgrade = 1;
        varIDsToUpgrade[varID] = secondBasis;
      }
    }
  }
  // now that we know which ones to upgrade, rebuild the DofOrdering, overriding with those when needed...
  // TODO: ? (Don't forget to worry about conforming bases...)
  
  if (orderingToUpgrade==1) {
    firstOrdering = upgradeSide(firstOrdering,firstCellTopo,varIDsToUpgrade,firstSideIndex);
  } else if (orderingToUpgrade==2) {
    secondOrdering = upgradeSide(secondOrdering,secondCellTopo,varIDsToUpgrade,secondSideIndex);
  }
  
  return orderingToUpgrade;
}

DofOrderingPtr DofOrderingFactory::upgradeSide(DofOrderingPtr dofOrdering,
                                               const shards::CellTopology &cellTopo, 
                                               map<int,BasisPtr> varIDsToUpgrade,
                                               int sideToUpgrade) {
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering());
  
  set<int> varIDs = dofOrdering->getVarIDs();
  set<int>::iterator idIt;

  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = dofOrdering->getNumSidesForVarID(varID);
    if ((varIDsToUpgrade.find(varID) != varIDsToUpgrade.end()) && numSides == 1) {
      TEUCHOS_TEST_FOR_EXCEPTION( true,
                         std::invalid_argument,
                         "upgradeSide requested for varID on interior.");
    }
    IntrepidExtendedTypes::EFunctionSpaceExtended fs;
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      BasisPtr basis = dofOrdering->getBasis(varID,sideIndex);
      fs = BasisFactory::getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::getBasisRank(basis);
      if ((varIDsToUpgrade.find(varID) == varIDsToUpgrade.end()) || (sideIndex != sideToUpgrade)) {
        // use existing basis
        newOrdering->addEntry(varID,basis,basisRank,sideIndex);
      } else {
        // upgrade basis
        basis = varIDsToUpgrade[varID];
        newOrdering->addEntry(varID,basis,basisRank,sideToUpgrade);
      }
    }

    if ((numSides > 1) && (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) && (conforming)) {
      addConformingVertexPairings(varID, newOrdering, cellTopo);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderings.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}

DofOrderingPtr DofOrderingFactory::pRefine(DofOrderingPtr dofOrdering,
                                           const shards::CellTopology &cellTopo, int pToAdd) {
  // could consider adding a cache that lets you go from (DofOrdering*,pToAdd) --> enrichedDofOrdering...
  // (since likely we'll be upgrading the same DofOrdering a bunch of times)
  set<int> varIDs = dofOrdering->getVarIDs();
  int interiorPolyOrder = polyOrder(dofOrdering); // rule is, any bases with polyOrder < interiorPolyOrder+pToAdd get upgraded 
  int newPolyOrder = interiorPolyOrder + pToAdd;
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering());
  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = dofOrdering->getNumSidesForVarID(varID);
    IntrepidExtendedTypes::EFunctionSpaceExtended fs;
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      BasisPtr basis = dofOrdering->getBasis(varID,sideIndex);
      fs = BasisFactory::getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::getBasisRank(basis);
      if (BasisFactory::basisPolyOrder(basis) >= newPolyOrder) {
        newOrdering->addEntry(varID,basis,basisRank,sideIndex);
      } else {
        // upgrade basis
        basis = BasisFactory::setPolyOrder(basis, newPolyOrder);
        newOrdering->addEntry(varID,basis,basisRank,sideIndex);
      }
    }
    if ((numSides > 1) && (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) && (conforming)) {
      addConformingVertexPairings(varID, newOrdering, cellTopo);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderings.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}

DofOrderingPtr DofOrderingFactory::setSidePolyOrder(DofOrderingPtr dofOrdering, int sideIndexToSet,
                                                    int newPolyOrder, bool replacePatchBasis) {
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering());
  set<int> varIDs = dofOrdering->getVarIDs();
  Teuchos::RCP< shards::CellTopology > cellTopoPtr = dofOrdering->cellTopology();
  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++) {
    int varID = *idIt;
    int numSides = dofOrdering->getNumSidesForVarID(varID);
    IntrepidExtendedTypes::EFunctionSpaceExtended fs;
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      BasisPtr basis = dofOrdering->getBasis(varID,sideIndex);
      if (replacePatchBasis) {
        if (BasisFactory::isPatchBasis(basis)) {
          // if we have a PatchBasis, then we want to get the underlying basis...
          basis = ((PatchBasis*)basis.get())->nonPatchAncestorBasis();
        }
      }
      fs = BasisFactory::getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::getBasisRank(basis);
      int basisPolyOrder = BasisFactory::basisPolyOrder(basis);
      if ( (numSides > 1) && (sideIndex==sideIndexToSet) && (basisPolyOrder < newPolyOrder) ) {
        // upgrade basis
        basis = BasisFactory::setPolyOrder(basis, newPolyOrder);
      }
      newOrdering->addEntry(varID,basis,basisRank,sideIndex);
    }
    if ((numSides > 1) && (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) && (conforming)) {
      addConformingVertexPairings(varID, newOrdering, *cellTopoPtr);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderings.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}