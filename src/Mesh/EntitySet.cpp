//
//  EntitySet.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/11/15.
//
//

#include "CamelliaCellTools.h"
#include "EntitySet.h"
#include "MeshTopology.h"

using namespace Camellia;
using namespace std;

EntitySet::EntitySet(EntityHandle thisHandle)
{
  _handle = thisHandle;
}

void EntitySet::addEntity(int d, IndexType entityIndex)
{
  if (d >= _entities.size())
  {
    _entities.resize(d+1);
  }
  _entities[d].insert(entityIndex);
}

bool EntitySet::containsEntity(int d, IndexType entityIndex) const
{
  if (d >= _entities.size()) return false;
  return _entities[d].find(entityIndex) != _entities[d].end();
}

EntityHandle EntitySet::getHandle() const
{
  return _handle;
}

void EntitySet::removeEntity(int d, IndexType entityIndex)
{
  if (d >= _entities.size()) return;
  _entities[d].erase(entityIndex);
}

// ! Returns the cellIDs from the provided set that contain at least one entity in the set, according to the provided MeshTopologyView.
set<IndexType> EntitySet::cellIDsThatMatch(MeshTopologyViewPtr meshTopo, const set<IndexType> &cellIDs) const
{
  // When there are very few cellIDs in the provided set relative to the number of entities in our set,
  // then it would be better to iterate over their entities and take the intersection with our entities.
  
  // For now, we instead iterate over our entities and take the intersection with cellIDs.
  // Later, we could check the number of cellIDs and how many entities we have, and choose accordingly.
  
  set<IndexType> matchingCellIDs;
  for (int d=0; d<_entities.size(); d++)
  {
    for (IndexType entityIndex : _entities[d])
    {
      set< pair<IndexType, unsigned> > cellEntries = meshTopo->getCellsContainingEntity(d, entityIndex);
      for (auto cellEntry : cellEntries)
      {
        if (cellIDs.find(cellEntry.first) != cellIDs.end())
        {
          matchingCellIDs.insert(cellEntry.first);
        }
      }
    }
  }
  return matchingCellIDs;
}

vector<unsigned> EntitySet::subcellOrdinals(MeshTopologyViewPtr meshTopo, IndexType cellID, unsigned d) const
{
  if ((d >= _entities.size()) || (_entities[d].size() == 0)) return vector<unsigned>();
  
  vector<unsigned> subcellOrdinals;
  CellPtr cell = meshTopo->getCell(cellID);
  
  int subcellCount = cell->topology()->getSubcellCount(d);
  for (int subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++)
  {
    IndexType entityIndex = cell->entityIndex(d, subcellOrdinal);
    if (containsEntity(d, entityIndex)) subcellOrdinals.push_back(subcellOrdinal);
  }
  return subcellOrdinals;
}

vector<unsigned> EntitySet::subcellOrdinalsOnSide(MeshTopologyViewPtr meshTopo, IndexType cellID, unsigned sideOrdinal, unsigned d) const
{
  if ((d >= _entities.size()) || (_entities[d].size() == 0)) return vector<unsigned>();
  
  vector<unsigned> subcellOrdinals;
  CellPtr cell = meshTopo->getCell(cellID);
  
  int sideDim = cell->topology()->getDimension() - 1;
  int subcellCount = cell->topology()->getSide(sideOrdinal)->getSubcellCount(d);

  for (int sideSubcellOrdinal=0; sideSubcellOrdinal<subcellCount; sideSubcellOrdinal++)
  {
    unsigned cellSubcellOrdinal = CamelliaCellTools::subcellOrdinalMap(cell->topology(), sideDim, sideOrdinal, d, sideSubcellOrdinal);
    IndexType entityIndex = cell->entityIndex(d, cellSubcellOrdinal);
    if (containsEntity(d, entityIndex)) subcellOrdinals.push_back(sideSubcellOrdinal);
  }
  return subcellOrdinals;
}
