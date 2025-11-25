//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "BasisFactory.h"

#include "DofOrderingFactory.h"

#include "Basis.h"

#include "CamelliaCellTools.h"
#include "PolynomialOrder.hpp"

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
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument, "bilinearForm argument is null!");
  _varFactory = bilinearForm->varFactory();
  _trialOrderEnhancements = trialOrderEnhancements;
  _testOrderEnhancements = testOrderEnhancements;
}

using tag = std::pair<std::string,int>;

DofOrderingPtr DofOrderingFactory::testOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo)
{
  std::set<Tag> tagsForVolume;
  const std::map<int, std::set<Tag>> tagsForSide;
  
  return this->testOrdering(polyOrder,cellTopo,tagsForVolume,tagsForSide);
}

DofOrderingPtr DofOrderingFactory::testOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo, const std::set<Tag> &tagsForVolume, 
                                                const std::map<int, std::set<Tag>> &tagsForSide)
{
  PolynomialOrder::checkPolyOrderValidity(polyOrder, cellTopo);
  const int polyOrderTime = PolynomialOrder::temporalPolyOrder(polyOrder);
  
  using namespace std;
  set<Tag> keyTagsForVolume;
  map<int, set<Tag>> keyTagsForSide;
  const auto & testVars = _varFactory->testVars();
  for (const auto & varEntry : testVars)
  {
    const auto & var = varEntry.second;
    if (var->hasTag())
    {
      auto tag = var->getTag();
      if (var->isDefinedOnVolume())
      {
        if (tagsForVolume.find(tag) != tagsForVolume.end())
        {
          keyTagsForVolume.insert(tag);
        }
      }
      else
      {
        for (auto sideEntry : tagsForSide)
        {
          const int &sideOrdinal = sideEntry.first;
          const set<Tag> &tags   = sideEntry.second;
          
          if (tags.find(tag) != tags.end())
          {
            keyTagsForSide[sideOrdinal].insert(tag);
          }
        }
      }
    }
  }
  
  // trial_key arguments: (polyOrder, CellTopo, conforming (legacy, for max. rule), volume tags, side tags.  Volume tags, side tags only included if there are tag-filtered variables matching them.
  // test_key is the same, but skip the "conforming" bool argument.
  const auto key = make_tuple(polyOrder,cellTopo->getKey(),keyTagsForVolume,keyTagsForSide);
  // std::tuple<vector<int>,CellTopologyKey,     set<Tag>,map<int,Tag> >;
  if (_testOrderings.find(key) != _testOrderings.end()) return _testOrderings[key];
  
  // vector<int> testIDs = _bilinearForm->testIDs();
  vector<int> testIDs = _varFactory->testIDs();
  
  DofOrderingPtr testOrder = Teuchos::rcp(new DofOrdering(cellTopo));

  vector<int> testIDPolyOrder(polyOrder.size());
  vector<Camellia::EFunctionSpace> testIDFunctionSpace(polyOrder.size());

  for (int testID : testIDs)
  {
    VarPtr testVar = _varFactory->test(testID);
    VarType varType = _varFactory->test(testID)->varType();
    bool sideVar = (varType == FLUX) || (varType == TRACE);
    if (testVar->hasTag())
    {
      auto tag = testVar->getTag();
      // then we need to check if this variable exists on the element
      if (!sideVar) // tagsForVolume governs
      {
        if (tagsForVolume.find(tag) == tagsForVolume.end())
        {
          // no match; proceed to next trialID
          continue;
        }
      }
      else // tagsForSide governs
      {
        bool matchFound = false;
        for (const auto & mapEntry : tagsForSide)
        {
          const auto & sideTags = mapEntry.second;
          if (sideTags.find(tag) != sideTags.end())
          {
            matchFound = true;
            break;
          }
        }
        if (!matchFound) continue; // next testID
      }
    }
    
    Camellia::EFunctionSpace fs = efsForSpace(testVar->space());
    BasisPtr basis;
    const int testPolyOrderTime = polyOrderTime + getTestOrderEnhancement(testID);
    for (int pComponent = 0; pComponent < polyOrder.size(); pComponent++)
    {
      testIDPolyOrder[pComponent] = polyOrder[pComponent] + getTestOrderEnhancement(testID);
      testIDFunctionSpace[pComponent] = fs;
    }
    Camellia::EFunctionSpace fsTemporal = efsForSpace(testVar->temporalFunctionSpace());
    
    if (sideVar)
    {
      int sideDim = cellTopo->getDimension() - 1;
      int numSides = cellTopo->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        CellTopoPtr sideTopo = cellTopo->getSubcell(sideDim, sideOrdinal);
        
        // temporalSide: somewhat confusingly, this means the *normal* in the time dimension has unit length.  (i.e., this is a side that is not extruded in time.)
        // (sideIsSpatial() means the side is a side of the spatial cell topology, extruded in time.)
        bool temporalSide = testVar->isSpaceTime() && !cellTopo->sideIsSpatial(sideOrdinal);
        
        if ( temporalSide && !testVar->isDefinedOnTemporalInterface() )
        {
          // skip adding on this side
          continue;
        }
        if (testVar->hasTag())
        {
          if (tagsForSide.find(sideOrdinal) == tagsForSide.end())
          {
            // no tags at all on this side; skip variable on this side
            continue;
          }
          else
          {
            auto tagSetForSide = tagsForSide.find(sideOrdinal)->second;
            auto tag = testVar->getTag();
            if (tagSetForSide.find(tag) == tagSetForSide.end())
            {
              // no tag matching this variable on this side; skip variable on this side
              continue;
            }
          }
        }
        
        if (testVar->getBasisFamily() != Teuchos::null)
        {
          std::vector<int> pVector = testIDPolyOrder;
          pVector.push_back(testPolyOrderTime);
          basis = testVar->getBasis(cellTopo, pVector, sideOrdinal);
        }
        else if (testVar->isSpaceTime() && cellTopo->sideIsSpatial(sideOrdinal))
        {
          basis = BasisFactory::basisFactory()->getSpaceTimeConformingBasis(testIDPolyOrder, testPolyOrderTime, sideTopo, testIDFunctionSpace, fsTemporal);
        }
        else
        {
          std::vector<int> pVector = testIDPolyOrder;
          if (pVector.size() > sideTopo->getTensorialDegree() + 1)
          {
            // we should do something here in terms of selecting the right dimensions from pVector,
            // but for now, I'm just going to check isotropic in p, and then reduce the number of entries
            // by one.
            
            int p = pVector[0];
            for (int d=1; d<pVector.size(); d++)
            {
              TEUCHOS_TEST_FOR_EXCEPTION(p != pVector[d], std::invalid_argument, "this code path assumes isotropic in p right now");
            }
            
            pVector.pop_back();
          }
          auto fsVector = testIDFunctionSpace;
          if (fsVector.size() > sideTopo->getTensorialDegree() + 1)
          {
            // similar story for function spaces
            auto fs = fsVector[0];
            for (int d=1; d<fsVector.size(); d++)
            {
              TEUCHOS_TEST_FOR_EXCEPTION(fs != fsVector[d], std::invalid_argument, "this code path assumes isotropic in function space right now");
            }
            fsVector.pop_back();
          }
          basis = BasisFactory::basisFactory()->getConformingBasis( pVector, sideTopo, fsVector );
        }
        int basisRank = basis->rangeRank();
        
        testOrder->addEntry(testID,basis,basisRank,sideOrdinal);
      }
    }
    else
    {
      if (testVar->getBasisFamily() != Teuchos::null)
      {
        basis = testVar->getBasis(cellTopo, testIDPolyOrder);
      }
      else if (testVar->isSpaceTime())
      {
        vector<int> testIDPolyOrderSpace;
        vector<Camellia::EFunctionSpace> testIDFunctionSpaceSpace;
        if (testIDPolyOrder.size() == 1)
        {
          // isotropic in space-time
          testIDPolyOrderSpace = testIDPolyOrder;
          testIDFunctionSpaceSpace = testIDFunctionSpace;
        }
        else
        {
          testIDPolyOrderSpace = vector<int>(testIDPolyOrder.begin(),testIDPolyOrder.end()-1);
          testIDFunctionSpaceSpace = vector<Camellia::EFunctionSpace>(testIDFunctionSpace.begin(),testIDFunctionSpace.end()-1);
        }
        basis = BasisFactory::basisFactory()->getSpaceTimeConformingBasis(testIDPolyOrderSpace, testPolyOrderTime, cellTopo, testIDFunctionSpaceSpace, fsTemporal);
      }
      else
      {
        basis = BasisFactory::basisFactory()->getConformingBasis( testIDPolyOrder, cellTopo, testIDFunctionSpace );
      }
      int basisRank = basis->rangeRank();
      testOrder->addEntry(testID,basis,basisRank);
    }
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

using tag = std::pair<std::string,int>;

DofOrderingPtr DofOrderingFactory::trialOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo, bool conformingVertices)
{
  std::set<Tag> tagsForVolume;
  const std::map<int, std::set<Tag>> tagsForSide;
  
  return this->trialOrdering(polyOrder,cellTopo,conformingVertices,tagsForVolume,tagsForSide);
}

DofOrderingPtr DofOrderingFactory::trialOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo, bool conformingVertices,
                                                 const std::set<Tag> &tagsForVolume, const std::map<int, std::set<Tag>> &tagsForSide)
{
  PolynomialOrder::checkPolyOrderValidity(polyOrder, cellTopo);
  const int polyOrderTime = PolynomialOrder::temporalPolyOrder(polyOrder);
  
  using namespace std;
  set<Tag> keyTagsForVolume;
  map<int, set<Tag>> keyTagsForSide;
  const auto & trialVars = _varFactory->trialVars();
  for (const auto & varEntry : trialVars)
  {
    const auto & var = varEntry.second;
    if (var->hasTag())
    {
      auto tag = var->getTag();
      if (var->isDefinedOnVolume())
      {
        if (tagsForVolume.find(tag) != tagsForVolume.end())
        {
          keyTagsForVolume.insert(tag);
        }
      }
      else
      {
        for (auto sideEntry : tagsForSide)
        {
          const int &sideOrdinal = sideEntry.first;
          const set<Tag> &tags   = sideEntry.second;
          
          if (tags.find(tag) != tags.end())
          {
            keyTagsForSide[sideOrdinal].insert(tag);
          }
        }
      }
    }
  }
  
  // trial_key arguments: (polyOrder, CellTopo, conforming (legacy, for max. rule), volume tags, side tags.  Volume tags, side tags only included if there are tag-filtered variables matching them.
  auto key = make_tuple(polyOrder,cellTopo->getKey(),conformingVertices,keyTagsForVolume,keyTagsForSide);
  if (_trialOrderings.find(key) != _trialOrderings.end()) return _trialOrderings[key];
  
  // conformingVertices = true only works for 2D topologies
  if ((cellTopo->getDimension() != 2) && conformingVertices)
  {
    cout << "ERROR: DofOrderingFactory only supports conformingVertices = true for 2D topologies.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "DofOrderingFactory only supports conformingVertices = true for 2D topologies");
  }

  // vector<int> trialIDs = _bilinearForm->trialIDs();
  vector<int> trialIDs = _varFactory->trialIDs();

  DofOrderingPtr trialOrder = Teuchos::rcp(new DofOrdering(cellTopo));
  DofOrderingPtr traceOrder = Teuchos::rcp(new DofOrdering(cellTopo));
  DofOrderingPtr fieldOrder = Teuchos::rcp(new DofOrdering(cellTopo));

  vector<int> trialIDPolyOrder(polyOrder.size());
  
  for (const int &trialID : trialIDs)
  {
    VarPtr trialVar = _varFactory->trialVars().find(trialID)->second;
    VarType varType = _varFactory->trial(trialID)->varType();
    bool sideVar = (varType == FLUX) || (varType == TRACE);
    if (trialVar->hasTag())
    {
      auto tag = trialVar->getTag();
      // then we need to check if this variable exists on the element
      if (!sideVar) // tagsForVolume governs
      {
        if (tagsForVolume.find(tag) == tagsForVolume.end())
        {
          // no match; proceed to next trialID
          continue;
        }
      }
      else // tagsForSide governs
      {
        bool matchFound = false;
        for (const auto & mapEntry : tagsForSide)
        {
          const auto & sideTags = mapEntry.second;
          if (sideTags.find(tag) != sideTags.end())
          {
            matchFound = true;
            break;
          }
        }
        if (!matchFound) continue; // next trialID
      }
    }
    
    Camellia::EFunctionSpace fs         = efsForSpace(trialVar->space());
    Camellia::EFunctionSpace temporalFS = efsForSpace(trialVar->temporalFunctionSpace());
    
    int trialOrderEnhancement = getTrialOrderEnhancement(trialID);
    for (int pComponent = 0; pComponent < polyOrder.size(); pComponent++)
    {
      trialIDPolyOrder[pComponent]     = polyOrder[pComponent] + trialOrderEnhancement; // uses the fact that map defaults to 0 for entries that aren't found
    }
    const int trialIDPolyOrderTime = polyOrderTime + trialOrderEnhancement;

    BasisPtr basis;

    int basisRank;

    if (sideVar)
    {
      int sideDim = cellTopo->getDimension() - 1;
      int numSides = cellTopo->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        // temporalSide: somewhat confusingly, this means the *normal* in the time dimension has unit length.  (i.e., this is a side that is not extruded in time.)
        // (sideIsSpatial() means the side is a side of the spatial cell topology, extruded in time.)
        bool temporalSide = trialVar->isSpaceTime() && !cellTopo->sideIsSpatial(sideOrdinal);
        if ( temporalSide && !trialVar->isDefinedOnTemporalInterface() )
        {
          // skip adding on this side
          continue;
        }
        if (trialVar->hasTag())
        {
          if (tagsForSide.find(sideOrdinal) == tagsForSide.end())
          {
            // no tags at all on this side; skip variable on this side
            continue;
          }
          else
          {
            auto tagSetForSide = tagsForSide.find(sideOrdinal)->second;
            auto tag = trialVar->getTag();
            if (tagSetForSide.find(tag) == tagSetForSide.end())
            {
              // no tag matching this variable on this side; skip variable on this side
              continue;
            }
          }
        }
        
        vector<int> trialIDPolyOrderSide = PolynomialOrder::sidePolynomialOrder(trialIDPolyOrder, cellTopo, sideOrdinal);
        
        CellTopoPtr sideTopo = cellTopo->getSubcell(sideDim, sideOrdinal);
        if (trialVar->getBasisFamily() != Teuchos::null)
        {
          basis = trialVar->getBasis(cellTopo, trialIDPolyOrder, sideOrdinal);
        }
        else if (trialVar->isSpaceTime() && cellTopo->sideIsSpatial(sideOrdinal))
        {
          // side is extruded in time
          vector<int> trialIDPolyOrderSideSpatialPart;
          if (trialIDPolyOrderSide.size() == 1)
          {
            // isotropic (on side, anyway)
            trialIDPolyOrderSideSpatialPart = trialIDPolyOrderSide;
          }
          else
          {
            // spatial part is all but the last entry, which belongs to time
            trialIDPolyOrderSideSpatialPart = vector<int>(trialIDPolyOrderSide.begin(),trialIDPolyOrderSide.end()-1);
            // sanity check: the last entry in trialIDPolyOrderSide should match trialIDPolyOrderTime
            TEUCHOS_TEST_FOR_EXCEPTION(trialIDPolyOrderTime != trialIDPolyOrderSide[trialIDPolyOrderSide.size()-1], std::logic_error, "poly order in time does not match the final poly order in a side that is extruded in time.");
          }
          basis = BasisFactory::basisFactory()->getSpaceTimeConformingBasis(trialIDPolyOrderSideSpatialPart, trialIDPolyOrderTime, sideTopo, fs, temporalFS);
        }
        else
        {
          basis = BasisFactory::basisFactory()->getConformingBasis( trialIDPolyOrderSide, sideTopo, fs );
        }
        int basisRank = basis->rangeRank();
        
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
      if (trialVar->getBasisFamily() != Teuchos::null)
      {
        basis = trialVar->getBasis(cellTopo, trialIDPolyOrder);
      }
      else if (trialVar->isSpaceTime())
      {
        vector<int> trialIDPolyOrderSpace;
        if (trialIDPolyOrder.size() == 1)
        {
          // isotropic in space-time
          trialIDPolyOrderSpace = trialIDPolyOrder;
        }
        else
        {
          trialIDPolyOrderSpace = vector<int>(trialIDPolyOrder.begin(),trialIDPolyOrder.end()-1);
        }
        
        basis = BasisFactory::basisFactory()->getSpaceTimeConformingBasis(trialIDPolyOrderSpace, trialIDPolyOrderTime, cellTopo, fs, temporalFS);
      }
      else
      {
        basis = BasisFactory::basisFactory()->getConformingBasis(trialIDPolyOrder, cellTopo, fs);
      }
      basisRank = basis->rangeRank();
      trialOrder->addEntry(trialID,basis,basisRank,VOLUME_INTERIOR_SIDE_INDEX);
      fieldOrder->addEntry(trialID,basis,basisRank,VOLUME_INTERIOR_SIDE_INDEX);
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

  for (const auto & varID : varIDs)
  {
    const vector<int> & sidesForVar = dofOrdering->getSidesForVarID(varID);
    int varIDEnhancement = isTestOrdering ? getTestOrderEnhancement(varID) : getTrialOrderEnhancement(varID);
    if (dofOrdering->isVolumeVar(varID))
    {
      interiorVariable = varID;
      interiorVariableFound = true;
      break;
    }
    else
    {
      for (const auto & sideOrdinal : sidesForVar)
      {
        auto basis = dofOrdering->getBasis(varID,sideOrdinal);
        int p_H1 = getH1Order( basis->getDegree(), basis->functionSpace() ) - varIDEnhancement;
        minSidePolyOrder = min(minSidePolyOrder,p_H1);
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
  int varIDEnhancement = isTestOrdering ? getTestOrderEnhancement(interiorVariable) : getTrialOrderEnhancement(interiorVariable);
  BasisPtr interiorBasis = dofOrdering->getBasis(interiorVariable);
  return BasisFactory::basisFactory()->basisPolyOrder(interiorBasis) - varIDEnhancement;
}

int DofOrderingFactory::getTestOrderEnhancement(int varID) const
{
  auto foundEntry = _testOrderEnhancements.find(varID);
  if (foundEntry != _testOrderEnhancements.end())
    return foundEntry->second;
  else
    return 0;
}

int DofOrderingFactory::getTrialOrderEnhancement(int varID) const
{
  auto foundEntry = _trialOrderEnhancements.find(varID);
  if (foundEntry != _trialOrderEnhancements.end())
    return foundEntry->second;
  else
    return 0;
}

void DofOrderingFactory::childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
    CellTopoPtr childTopo,
    int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
    DofOrderingPtr &parentTrialOrdering, int sideIndex,
    CellTopoPtr parentTopo)
{
  int upgradedSide = matchSides(childTrialOrdering,childSideIndex,childTopo,
                                parentTrialOrdering,sideIndex,parentTopo);
  if (upgradedSide == 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "parent should never be upgraded!");
  }
}

int DofOrderingFactory::matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex,
                                   CellTopoPtr firstCellTopo,
                                   DofOrderingPtr &secondOrdering, int secondSideIndex,
                                   CellTopoPtr secondCellTopo)
{
  // upgrades the lesser-order basis
  map<int, BasisPtr > varIDsToUpgrade;
  int orderingToUpgrade = 0; // 0 means neither, 1 first, 2 second
  set<int> varIDs = firstOrdering->getVarIDs();
  set<int>::iterator idIt;
  for (idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    if (!firstOrdering->isVolumeVar(varID))   // a variable that lives on the sides: we need to match basis
    {
      BasisPtr firstBasis = firstOrdering->getBasis(varID,firstSideIndex);
      BasisPtr secondBasis = secondOrdering->getBasis(varID,secondSideIndex);

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
    const vector<int> & sidesForVar = dofOrdering->getSidesForVarID(varID);
    const bool isVolumeVar = dofOrdering->isVolumeVar(varID);
    if ((varIDsToUpgrade.find(varID) != varIDsToUpgrade.end()) && isVolumeVar)
    {
      TEUCHOS_TEST_FOR_EXCEPTION( true,
                                  std::invalid_argument,
                                  "upgradeSide requested for varID on interior.");
    }
    Camellia::EFunctionSpace fs;
    for (const int & sideOrdinal : sidesForVar)
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

    if (!isVolumeVar && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
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
  // TODO: eliminate the dependence on BasisFactory here
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
    bool isVolumeVar = dofOrdering->isVolumeVar(varID);
    Camellia::EFunctionSpace fs;
    int enhancement = isTestOrdering ? getTestOrderEnhancement(varID) : getTrialOrderEnhancement(varID);
    int newPolyOrderForVarID = newPolyOrder + enhancement;
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
    if (!isVolumeVar && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
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
  // TODO: eliminate the dependence on BasisFactory here
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
    const bool isVolumeVar = dofOrdering->isVolumeVar(varID);
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
    if (!isVolumeVar && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
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
    int newPolyOrder)
{
  // TODO: eliminate the dependence on BasisFactory here
  bool conforming = _isConforming[dofOrdering.get()];
  DofOrderingPtr newOrdering = Teuchos::rcp(new DofOrdering(dofOrdering->cellTopology()));
  set<int> varIDs = dofOrdering->getVarIDs();
  CellTopoPtr cellTopoPtr = dofOrdering->cellTopology();

  for (set<int>::iterator idIt = varIDs.begin(); idIt != varIDs.end(); idIt++)
  {
    int varID = *idIt;
    const vector<int>* sidesForVar = &dofOrdering->getSidesForVarID(varID);
    const bool isVolumeVar = dofOrdering->isVolumeVar(varID);
    Camellia::EFunctionSpace fs;
    for (int sideOrdinal : *sidesForVar)
    {
      BasisPtr basis = dofOrdering->getBasis(varID,sideOrdinal);
      fs = BasisFactory::basisFactory()->getBasisFunctionSpace(basis);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      int basisPolyOrder = BasisFactory::basisFactory()->basisPolyOrder(basis);
      if ( !isVolumeVar && (sideOrdinal==sideIndexToSet) && (basisPolyOrder < newPolyOrder) )
      {
        // upgrade basis
        basis = BasisFactory::basisFactory()->setPolyOrder(basis, newPolyOrder);
      }
      newOrdering->addEntry(varID,basis,basisRank,sideOrdinal);
    }
    if (!isVolumeVar && (fs == Camellia::FUNCTION_SPACE_HGRAD) && (conforming))
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

