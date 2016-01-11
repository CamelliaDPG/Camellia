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

#include "Basis.h"

#include "CamelliaCellTools.h"

using namespace Intrepid;
using namespace Camellia;

DofOrderingFactory::DofOrderingFactory(VarFactoryPtr varFactory)
{
  _varFactory = varFactory;
}

DofOrderingFactory::DofOrderingFactory(VarFactoryPtr varFactory,
                                       map<int,int> trialOrderEnhancements,
                                       map<int,int> testOrderEnhancements)
{
  _varFactory = varFactory;
  _trialOrderEnhancements = trialOrderEnhancements;
  _testOrderEnhancements = testOrderEnhancements;
}

DofOrderingFactory::DofOrderingFactory(TBFPtr<double> bilinearForm)
{
  // _bilinearForm = bilinearForm;
  _varFactory = bilinearForm->varFactory();
}

DofOrderingFactory::DofOrderingFactory(TBFPtr<double> bilinearForm,
                                       map<int,int> trialOrderEnhancements,
                                       map<int,int> testOrderEnhancements)
{
  // _bilinearForm = bilinearForm;
  _varFactory = bilinearForm->varFactory();
  _trialOrderEnhancements = trialOrderEnhancements;
  _testOrderEnhancements = testOrderEnhancements;
}

DofOrderingPtr DofOrderingFactory::testOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo)
{
  auto key = make_pair(polyOrder,cellTopo->getKey());
  if (_testOrderings.find(key) != _testOrderings.end()) return _testOrderings[key];
  
  // vector<int> testIDs = _bilinearForm->testIDs();
  vector<int> testIDs = _varFactory->testIDs();
  vector<int>::iterator testIterator;

  DofOrderingPtr testOrder = Teuchos::rcp(new DofOrdering(cellTopo));

  vector<int> testIDPolyOrder(polyOrder.size());

  for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++)
  {
    int testID = *testIterator;
    // Camellia::EFunctionSpace fs = _bilinearForm->functionSpaceForTest(testID);
    Camellia::EFunctionSpace fs = efsForSpace(_varFactory->test(testID)->space());
    BasisPtr basis;
    for (int pComponent = 0; pComponent < polyOrder.size(); pComponent++)
    {
      testIDPolyOrder[pComponent] = polyOrder[pComponent] + _testOrderEnhancements[testID]; // uses the fact that map defaults to 0 for entries that aren't found
    }
    Camellia::EFunctionSpace fsTemporal = FUNCTION_SPACE_HGRAD; // tests should use HGRAD, so we can take time derivatives...
    basis = BasisFactory::basisFactory()->getBasis( testIDPolyOrder, cellTopo, fs, fsTemporal);
    int basisRank = basis->rangeRank();
    testOrder->addEntry(testID,basis,basisRank);
  }
  
  testOrder = *(_testOrderingsSet.insert(testOrder).first);
  _testOrderings[key] = testOrder;
  return testOrder;
}

DofOrderingPtr DofOrderingFactory::testOrdering(vector<int> &polyOrder,
    const shards::CellTopology &shardsTopo)
{
  CellTopoPtr cellTopo = CellTopology::cellTopology(shardsTopo);
  return testOrdering(polyOrder, cellTopo);
}

DofOrderingPtr DofOrderingFactory::trialOrdering(vector<int> &polyOrder,
    CellTopoPtr cellTopo,
    bool conformingVertices)
{
  auto key = make_pair(make_pair(polyOrder,cellTopo->getKey()),conformingVertices);
  if (_trialOrderings.find(key) != _trialOrderings.end()) return _trialOrderings[key];
  
  // conformingVertices = true only works for 2D topologies
  if ((cellTopo->getDimension() != 2) && conformingVertices)
  {
    cout << "ERROR: DofOrderingFactory only supports conformingVertices = true for 2D topologies.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "DofOrderingFactory only supports conformingVertices = true for 2D topologies");
  }

  // vector<int> trialIDs = _bilinearForm->trialIDs();
  vector<int> trialIDs = _varFactory->trialIDs();
  vector<int>::iterator trialIterator;

  DofOrderingPtr trialOrder = Teuchos::rcp(new DofOrdering(cellTopo));
  DofOrderingPtr traceOrder = Teuchos::rcp(new DofOrdering(cellTopo));
  DofOrderingPtr fieldOrder = Teuchos::rcp(new DofOrdering(cellTopo));

  // VarFactory vf = _bilinearForm->varFactory();

  vector<int> trialIDPolyOrder(polyOrder.size());

  for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++)
  {
    int trialID = *trialIterator;
    VarPtr trialVar = _varFactory->trialVars().find(trialID)->second;
    for (int pComponent = 0; pComponent < polyOrder.size(); pComponent++)
    {
      trialIDPolyOrder[pComponent] = polyOrder[pComponent] + _trialOrderEnhancements[trialID]; // uses the fact that map defaults to 0 for entries that aren't found
    }

    Camellia::EFunctionSpace fs = efsForSpace(_varFactory->trial(trialID)->space());

    BasisPtr basis;

    int basisRank;

    Camellia::EFunctionSpace temporalFS = FUNCTION_SPACE_HVOL;
    Camellia::EFunctionSpace spatialFS = fs; // default to using fs for the spatial component
    if (fs == FUNCTION_SPACE_HGRAD)
    {
      temporalFS = FUNCTION_SPACE_HGRAD;
    }
    else if (fs == FUNCTION_SPACE_HVOL)
    {
      temporalFS = FUNCTION_SPACE_HVOL;
    }
    else if (fs == FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME)
    {
      spatialFS = FUNCTION_SPACE_HVOL;
      temporalFS = FUNCTION_SPACE_HGRAD;
    }
    else if (fs == FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME)
    {
      spatialFS = FUNCTION_SPACE_HGRAD;
      temporalFS = FUNCTION_SPACE_HVOL;
    }
    
    // if (_bilinearForm->isFluxOrTrace(trialID)) {
    VarType varType = _varFactory->trial(trialID)->varType();
    if ((varType == FLUX) || (varType == TRACE))
    {
      int sideDim = cellTopo->getDimension() - 1;
      int numSides = cellTopo->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        CellTopoPtr sideTopo = cellTopo->getSubcell(sideDim, sideOrdinal);
        basis = BasisFactory::basisFactory()->getConformingBasis( trialIDPolyOrder, sideTopo, spatialFS, temporalFS );
        basisRank = basis->rangeRank();

        bool temporalSide = ! cellTopo->sideIsSpatial(sideOrdinal);

        if ( temporalSide && !trialVar->isDefinedOnTemporalInterface() )
        {
          // skip adding on this side
          continue;
        }

        trialOrder->addEntry(trialID,basis,basisRank,sideOrdinal);
        traceOrder->addEntry(trialID,basis,basisRank,sideOrdinal);
      }
      if ( conformingVertices
           && fs == Camellia::FUNCTION_SPACE_HGRAD)
      {
        // then we want to identify basis dofs at the vertices...

        addConformingVertexPairings(trialID, trialOrder, cellTopo);
        addConformingVertexPairings(trialID, traceOrder, cellTopo);
      }
    }
    else
    {
      basis = BasisFactory::basisFactory()->getConformingBasis(trialIDPolyOrder, cellTopo, spatialFS, temporalFS);
      basisRank = basis->rangeRank();
      trialOrder->addEntry(trialID,basis,basisRank,VOLUME_INTERIOR_SIDE_ORDINAL);
      fieldOrder->addEntry(trialID,basis,basisRank,VOLUME_INTERIOR_SIDE_ORDINAL);
    }
  }
  trialOrder->rebuildIndex();
  traceOrder->rebuildIndex();
  fieldOrder->rebuildIndex();
  trialOrder = *(_trialOrderingsSet.insert(trialOrder).first);
  traceOrder = *(_trialOrderingsSet.insert(traceOrder).first);
  fieldOrder = *(_trialOrderingsSet.insert(fieldOrder).first);
  _isConforming[trialOrder.get()] = conformingVertices;
  _isConforming[traceOrder.get()] = conformingVertices;
  _isConforming[fieldOrder.get()] = conformingVertices;
  _traceOrderingForTrial[trialOrder.get()] = traceOrder;
  _fieldOrderingForTrial[trialOrder.get()] = fieldOrder;
  _trialOrderings[key] = trialOrder;

  return trialOrder;
}

DofOrderingPtr DofOrderingFactory::trialOrdering(vector<int> &polyOrder,
    const shards::CellTopology &shardsTopo,
    bool conformingVertices)
{
  CellTopoPtr cellTopoPtr = CellTopology::cellTopology(shardsTopo);
  return trialOrdering(polyOrder, cellTopoPtr, conformingVertices);
}

DofOrderingPtr DofOrderingFactory::getFieldOrdering(DofOrderingPtr trialOrdering)
{
  // returns the sub-ordering that contains only the traces
  if (_fieldOrderingForTrial.find(trialOrdering.get()) == _fieldOrderingForTrial.end())
  {
    cout << "trialOrdering not found in DofOrderingFactory!  Did you use DofOrderingFactory to create it??\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialOrdering not found in DofOrderingFactory!  Did you use DofOrderingFactory to create it??");
  }
  return _fieldOrderingForTrial[trialOrdering.get()];
}

DofOrderingPtr DofOrderingFactory::getTraceOrdering(DofOrderingPtr trialOrdering)
{
  // returns the sub-ordering that contains only the traces
  if (_traceOrderingForTrial.find(trialOrdering.get()) == _traceOrderingForTrial.end())
  {
    cout << "trialOrdering not found in DofOrderingFactory!  Did you use DofOrderingFactory to create it??\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialOrdering not found in DofOrderingFactory!  Did you use DofOrderingFactory to create it??");
  }
  return _traceOrderingForTrial[trialOrdering.get()];
}

map<int, int> DofOrderingFactory::getTestOrderEnhancements()
{
  return _testOrderEnhancements;
}

map<int, int> DofOrderingFactory::getTrialOrderEnhancements()
{
  return _trialOrderEnhancements;
}

void DofOrderingFactory::addConformingVertexPairings(int varID, DofOrderingPtr dofOrdering, CellTopoPtr cellTopo)
{
  // then we want to identify basis dofs at the vertices...
  map< int, pair<int,int> > cellVertexOrdinalToSideVertexOrdinal; // vertexOrdinal --> pair<sideNumber, vertexNumber>
  int numSides = cellTopo->getSideCount();
  for (int j=0; j<numSides; j++)
  {
    int numVerticesPerSide = cellTopo->getVertexCount(1,j); // should be 2
    for (int i=0; i < numVerticesPerSide; i++)
    {
      unsigned vertexOrdinal = cellTopo->getNodeMap(1,j,i);
      if ( cellVertexOrdinalToSideVertexOrdinal.find(vertexOrdinal)
           == cellVertexOrdinalToSideVertexOrdinal.end() )
      {
        // haven't seen this one yet
        cellVertexOrdinalToSideVertexOrdinal[vertexOrdinal] = make_pair(j,i);
      }
      else
      {
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

int DofOrderingFactory::polyOrder(DofOrderingPtr dofOrdering, bool isTestOrdering)
{
  set<int> varIDs = dofOrdering->getVarIDs();
  set<int>::iterator idIt;
  int interiorVariable;
  bool interiorVariableFound = false;
  int minSidePolyOrder = INT_MAX;

  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    int numSides = sidesForVar->size();
    int varIDEnhancement = isTestOrdering ? _testOrderEnhancements[varID] : _trialOrderEnhancements[varID];
    if (numSides == 1)
    {
      interiorVariable = varID;
      interiorVariableFound = true;
      break;
    }
    else
    {
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;
        int polyOrder = BasisFactory::basisFactory()->basisPolyOrder( dofOrdering->getBasis(varID,sideOrdinal) ) - varIDEnhancement;
        minSidePolyOrder = min(minSidePolyOrder,polyOrder);
      }
    }
  }
  if ( ! interiorVariableFound)
  {
    // all side variables, which is a bit weird
    // if we have some idea of what the minimum poly order is for a side, then we return that.
    // otherwise, throw an exception
    TEUCHOS_TEST_FOR_EXCEPTION( minSidePolyOrder == INT_MAX,
                                std::invalid_argument,
                                "DofOrdering appears not to have any interior (volume) varIDs--DofOrderingFactory cannot pRefine.");
    return minSidePolyOrder;
  }
  int varIDEnhancement = isTestOrdering ? _testOrderEnhancements[interiorVariable] : _trialOrderEnhancements[interiorVariable];
  BasisPtr interiorBasis = dofOrdering->getBasis(interiorVariable);
  return BasisFactory::basisFactory()->basisPolyOrder(interiorBasis) - varIDEnhancement;
}

map<int, BasisPtr> DofOrderingFactory::getMultiBasisUpgradeMap(vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide)
{
  vector< BasisPtr > bases;
  set<int> varIDs = (childTrialOrdersForSide[0].first)->getVarIDs();
  map<int, BasisPtr> varIDsToUpgrade;
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    int numSides = childTrialOrdersForSide[0].first->getSidesForVarID(varID).size();
    if (numSides > 1)   // a variable that lives on the sides: we need to match basis
    {
      int numBases = childTrialOrdersForSide.size();
      for (int i=numBases-1; i>=0; i-- )
      {
        // reverse order of bases ( valid only for 2D; in 3D we'll need a more general permutation )
        int childSideIndex = childTrialOrdersForSide[i].second;
        BasisPtr basis  = childTrialOrdersForSide[i].first->getBasis(varID,childSideIndex);
        bases.push_back(basis);
      }
      if (bases.size() != 1)
      {
        BasisPtr multiBasis = BasisFactory::basisFactory()->getMultiBasis(bases);
        varIDsToUpgrade[varID] = multiBasis;
      }
      else
      {
        varIDsToUpgrade[varID] = bases[0];
      }
      bases.clear();
    }
  }
  return varIDsToUpgrade;
}

map<int, BasisPtr> DofOrderingFactory::getPatchBasisUpgradeMap(const DofOrderingPtr childTrialOrdering, int childSideIndex,
    const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
    int childIndexInParentSide)
{
  set<int> varIDs = childTrialOrdering->getVarIDs();
  map<int, BasisPtr> varIDsToUpgrade;
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    int numSides = childTrialOrdering->getSidesForVarID(varID).size();
    if (numSides > 1)   // a variable that lives on the sides: we need to match basis
    {
      BasisPtr basis = parentTrialOrdering->getBasis(varID,parentSideIndex);
      FieldContainer<double> nodes(2,1); // just 1D patches supported right now
      if (childIndexInParentSide==0)
      {
        nodes(0,0) = -1.0;
        nodes(1,0) = 0.0;
      }
      else
      {
        nodes(0,0) = 0.0;
        nodes(1,0) = 1.0;
      }
      BasisPtr patchBasis = BasisFactory::basisFactory()->getPatchBasis(basis, nodes);
      varIDsToUpgrade[varID] = patchBasis;
    }
  }
  return varIDsToUpgrade;
}

void DofOrderingFactory::assignMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex,
    CellTopoPtr cellTopo,
    vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide )
{
  map<int, BasisPtr> varIDsToUpgrade = getMultiBasisUpgradeMap(childTrialOrdersForSide);
  trialOrdering = upgradeSide(trialOrdering,cellTopo,varIDsToUpgrade,sideIndex);
}

void DofOrderingFactory::assignPatchBasis(DofOrderingPtr &childTrialOrdering, int childSideIndex,
    const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
    int childIndexInParentSide, CellTopoPtr childCellTopo)
{
  TEUCHOS_TEST_FOR_EXCEPTION(childIndexInParentSide >= 2, std::invalid_argument, "assignPatchBasis only supports 2 children on a side right now.");
  map<int, BasisPtr> varIDsToUpgrade = getPatchBasisUpgradeMap(childTrialOrdering, childSideIndex, parentTrialOrdering,
                                       parentSideIndex, childIndexInParentSide);
  childTrialOrdering = upgradeSide(childTrialOrdering,childCellTopo,varIDsToUpgrade,childSideIndex);
}

bool DofOrderingFactory::sideHasMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex)
{
  set<int> varIDs = trialOrdering->getVarIDs();
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    int numSides = trialOrdering->getSidesForVarID(varID).size();
    if (numSides > 1)   // a variable that lives on the sides
    {
      BasisPtr basis = trialOrdering->getBasis(varID,sideIndex);
      // as one side basis goes, so go they all:
      return BasisFactory::basisFactory()->isMultiBasis(basis);
    }
  }
  // if we get here, we didn't really have a side...
  return false;
}

void DofOrderingFactory::childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
    CellTopoPtr childTopo,
    int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
    DofOrderingPtr &parentTrialOrdering, int sideIndex,
    CellTopoPtr parentTopo)
{
  // basic strategy: if parent has MultiBasis on that side, then child should get a piece of that
  //                 otherwise, we can simply use matchSides as it is...
  if ( sideHasMultiBasis(parentTrialOrdering,sideIndex) )
  {
    set<int> varIDs = parentTrialOrdering->getVarIDs();
    map<int, BasisPtr> varIDsToUpgrade;
    set<int>::iterator idIt;
    for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
    {
      int varID = *idIt;
      int numSides = parentTrialOrdering->getSidesForVarID(varID).size();
      if (numSides > 1)   // a variable that lives on the sides: we need to match basis
      {
        BasisPtr basis  = parentTrialOrdering->getBasis(varID,sideIndex);
        if (! BasisFactory::basisFactory()->isMultiBasis(basis) )
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "if one basis is multibasis, they all should be");
        }
        MultiBasis<>* multiBasis = (MultiBasis<>*) basis.get();
        varIDsToUpgrade[varID] = multiBasis->getSubBasis(childIndexInParentSide);
      }
    }
    childTrialOrdering = upgradeSide(childTrialOrdering,childTopo,varIDsToUpgrade,childSideIndex);
  }
  else
  {
    int upgradedSide = matchSides(childTrialOrdering,childSideIndex,childTopo,
                                  parentTrialOrdering,sideIndex,parentTopo);
    if (upgradedSide == 2)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "parent should never be upgraded!");
    }
  }
}

int DofOrderingFactory::matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex,
                                   CellTopoPtr firstCellTopo,
                                   DofOrderingPtr &secondOrdering, int secondSideIndex,
                                   CellTopoPtr secondCellTopo)
{
  // upgrades the lesser-order basis
  map<int, BasisPtr > varIDsToUpgrade;
  int orderingToUpgrade = 0; // 0 means neither, 1 first, 2 second, -1 means PatchBasis (i.e. can't matchSides w/o more Mesh info)
  set<int> varIDs = firstOrdering->getVarIDs();
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    int numSides = firstOrdering->getSidesForVarID(varID).size();
    if (numSides > 1)   // a variable that lives on the sides: we need to match basis
    {
      BasisPtr firstBasis = firstOrdering->getBasis(varID,firstSideIndex);
      BasisPtr secondBasis = secondOrdering->getBasis(varID,secondSideIndex);
      if (BasisFactory::basisFactory()->isPatchBasis(firstBasis) || BasisFactory::basisFactory()->isPatchBasis(secondBasis))
      {
        return -1; // then we need to deal with ancestors, etc.--and we can't do that here
      }

      // use cardinality instead of degree to compare so that multiBasis > singleBasis
      if ( firstBasis->getCardinality() > secondBasis->getCardinality() )
      {
        if (orderingToUpgrade == 1)
        {
          TEUCHOS_TEST_FOR_EXCEPTION( true,
                                      std::invalid_argument,
                                      "DofOrderings vary in terms of which has higher degree.  Unhandled case in DofOrderingFactory.");
        }
        // otherwise
        orderingToUpgrade = 2;
        varIDsToUpgrade[varID] = firstBasis;
      }
      else if (secondBasis->getCardinality() > firstBasis->getCardinality() )
      {
        if (orderingToUpgrade == 2)
        {
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

  if (orderingToUpgrade==1)
  {
    firstOrdering = upgradeSide(firstOrdering,firstCellTopo,varIDsToUpgrade,firstSideIndex);
  }
  else if (orderingToUpgrade==2)
  {
    secondOrdering = upgradeSide(secondOrdering,secondCellTopo,varIDsToUpgrade,secondSideIndex);
  }

  return orderingToUpgrade;
}

DofOrderingPtr DofOrderingFactory::upgradeSide(DofOrderingPtr dofOrdering,
    CellTopoPtr cellTopo,
    map<int,BasisPtr> varIDsToUpgrade,
    int sideToUpgrade)
{
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));

  set<int> varIDs = dofOrdering->getVarIDs();
  set<int>::iterator idIt;

  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    int numSides = sidesForVar->size();
    if ((varIDsToUpgrade.find(varID) != varIDsToUpgrade.end()) && numSides == 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION( true,
                                  std::invalid_argument,
                                  "upgradeSide requested for varID on interior.");
    }
    Camellia::EFunctionSpace fs;
    for (int sideOrdinal : *sidesForVar)
    {
      BasisPtr basis = dofOrdering->getBasis(varID,sideOrdinal);
      fs = BasisFactory::basisFactory()->getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      if ((varIDsToUpgrade.find(varID) == varIDsToUpgrade.end()) || (sideOrdinal != sideToUpgrade))
      {
        // use existing basis
        newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
      }
      else
      {
        // upgrade basis
        basis = varIDsToUpgrade[varID];
        newOrdering->addEntry(varID,basis,basisRank,sideToUpgrade);
      }
    }

    if ((numSides > 1) && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
    {
      addConformingVertexPairings(varID, newOrdering, cellTopo);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderingsSet.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}

DofOrderingPtr DofOrderingFactory::pRefine(DofOrderingPtr dofOrdering, CellTopoPtr cellTopo, int pToAdd, bool isTestOrdering)
{
  // could consider adding a cache that lets you go from (DofOrdering*,pToAdd) --> enrichedDofOrdering...
  // (since likely we'll be upgrading the same DofOrdering a bunch of times)
  set<int> varIDs = dofOrdering->getVarIDs();
  int interiorPolyOrder = polyOrder(dofOrdering, isTestOrdering); // rule is, any bases with polyOrder < interiorPolyOrder+pToAdd get upgraded
  int newPolyOrder = interiorPolyOrder + pToAdd;
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));
  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    int numSides = sidesForVar->size();
    Camellia::EFunctionSpace fs;
    int newPolyOrderForVarID;
    if (isTestOrdering)
    {
      newPolyOrderForVarID = newPolyOrder + _testOrderEnhancements[varID];
    }
    else
    {
      newPolyOrderForVarID = newPolyOrder + _trialOrderEnhancements[varID];
    }
    for (int sideOrdinal : *sidesForVar)
    {
      BasisPtr basis = dofOrdering->getBasis(varID,sideOrdinal);
      fs = BasisFactory::basisFactory()->getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      if (BasisFactory::basisFactory()->basisPolyOrder(basis) >= newPolyOrderForVarID)
      {
        newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
      }
      else
      {
        // upgrade basis
        basis = BasisFactory::basisFactory()->setPolyOrder(basis, newPolyOrderForVarID);
        newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
      }
    }
    if ((numSides > 1) && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
    {
      addConformingVertexPairings(varID, newOrdering, cellTopo);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderingsSet.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}


DofOrderingPtr DofOrderingFactory::pRefineTest(DofOrderingPtr testOrdering,
    const shards::CellTopology &cellTopo, int pToAdd)
{
  CellTopoPtr cellTopoPtr = CellTopology::cellTopology(cellTopo);
  return pRefine(testOrdering, cellTopoPtr, pToAdd, true);
}

DofOrderingPtr DofOrderingFactory::pRefineTrial(DofOrderingPtr trialOrdering,
    const shards::CellTopology &cellTopo, int pToAdd)
{
  CellTopoPtr cellTopoPtr = CellTopology::cellTopology(cellTopo);
  return pRefine(trialOrdering, cellTopoPtr, pToAdd, false);
}

DofOrderingPtr DofOrderingFactory::pRefineTest(DofOrderingPtr testOrdering,
    CellTopoPtr cellTopo, int pToAdd)
{
  return pRefine(testOrdering, cellTopo, pToAdd, true);
}

DofOrderingPtr DofOrderingFactory::pRefineTrial(DofOrderingPtr trialOrdering,
    CellTopoPtr cellTopo, int pToAdd)
{
  return pRefine(trialOrdering, cellTopo, pToAdd, false);
}

int DofOrderingFactory::testPolyOrder(DofOrderingPtr testOrdering)
{
  return polyOrder(testOrdering,true);
}

int DofOrderingFactory::trialPolyOrder(DofOrderingPtr trialOrdering)
{
  return polyOrder(trialOrdering,false);
}

DofOrderingPtr DofOrderingFactory::setBasisDegree(DofOrderingPtr dofOrdering, int basisDegreeToSet, bool replaceDiscontinuousFSWithContinuous)
{
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));

  DofOrderingPtr newTraceOrder = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));
  DofOrderingPtr newFieldOrder = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));

  set<int> varIDs = dofOrdering->getVarIDs();
  CellTopoPtr cellTopoPtr = dofOrdering->cellTopology();
  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    Camellia::EFunctionSpace fs;
    for (int sideOrdinal : *sidesForVar)
    {
      BasisPtr basis = dofOrdering->getBasis(varID,sideOrdinal);

      fs = BasisFactory::basisFactory()->getBasisFunctionSpace(basis);
      if (replaceDiscontinuousFSWithContinuous)
      {
        if (Camellia::functionSpaceIsDiscontinuous(fs))
        {
          fs = Camellia::continuousSpaceForDiscontinuous(fs);
        }
      }
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      int currentBasisDegree = basis->getDegree();
      int delta_k = basisDegreeToSet - currentBasisDegree;
      // upgrade basis
      int currentPolyOrder = BasisFactory::basisFactory()->basisPolyOrder(basis);
//      basis = BasisFactory::basisFactory()->setPolyOrder(basis, currentPolyOrder + delta_k );
      basis = BasisFactory::basisFactory()->getBasis( currentPolyOrder + delta_k, basis->domainTopology(), fs );
      newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
      if (sidesForVar->size() == 1)
      {
        newFieldOrder->addEntry(varID, basis, basisRank, sideOrdinal);
      }
      else
      {
        newTraceOrder->addEntry(varID, basis, basisRank, sideOrdinal);
      }
    }
    if ((sidesForVar->size() > 1) && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
    {
      addConformingVertexPairings(varID, newOrdering, cellTopoPtr);
      addConformingVertexPairings(varID, newTraceOrder, cellTopoPtr);
    }
  }
  newOrdering->rebuildIndex();
  newTraceOrder->rebuildIndex();
  newFieldOrder->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderingsSet.insert(newOrdering).first);
  newTraceOrder = *(_trialOrderingsSet.insert(newTraceOrder).first);
  newFieldOrder = *(_trialOrderingsSet.insert(newFieldOrder).first);
  _isConforming[newOrdering.get()] = conforming;
  _isConforming[newTraceOrder.get()] = conforming;
  _isConforming[newFieldOrder.get()] = conforming;
  _traceOrderingForTrial[newOrdering.get()] = newTraceOrder;
  _fieldOrderingForTrial[newOrdering.get()] = newFieldOrder;
  return newOrdering;
}

DofOrderingPtr DofOrderingFactory::setSidePolyOrder(DofOrderingPtr dofOrdering, int sideIndexToSet,
    int newPolyOrder, bool replacePatchBasis)
{
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));
  set<int> varIDs = dofOrdering->getVarIDs();
  CellTopoPtr cellTopoPtr = dofOrdering->cellTopology();

  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    Camellia::EFunctionSpace fs;
    for (int sideOrdinal : *sidesForVar)
    {
      BasisPtr basis = dofOrdering->getBasis(varID,sideOrdinal);
      if (replacePatchBasis)
      {
        if (BasisFactory::basisFactory()->isPatchBasis(basis))
        {
          // if we have a PatchBasis, then we want to get the underlying basis...
          basis = ((PatchBasis<>*)basis.get())->nonPatchAncestorBasis();
        }
      }
      fs = BasisFactory::basisFactory()->getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      int basisPolyOrder = BasisFactory::basisFactory()->basisPolyOrder(basis);
      if ( (sidesForVar->size() > 1) && (sideOrdinal==sideIndexToSet) && (basisPolyOrder < newPolyOrder) )
      {
        // upgrade basis
        basis = BasisFactory::basisFactory()->setPolyOrder(basis, newPolyOrder);
      }
      newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
    }
    if ((sidesForVar->size() > 1) && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
    {
      addConformingVertexPairings(varID, newOrdering, cellTopoPtr);
    }
  }
  newOrdering->rebuildIndex();
  // return Teuchos::RCP to the old element if there was one, or the newly inserted element
  newOrdering = *(_trialOrderingsSet.insert(newOrdering).first);
  _isConforming[newOrdering.get()] = conforming;
  return newOrdering;
}

