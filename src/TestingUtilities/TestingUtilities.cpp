//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "TestingUtilities.h"
#include "ElementType.h"
#include "Element.h"
#include "RHS.h"

using namespace Camellia;

// checks if dof has a BC applied to it
bool TestingUtilities::isBCDof(GlobalIndexType globalDofIndex, TSolutionPtr<double> solution)
{
  Camellia::FieldContainer<GlobalIndexType> globalIndices;
  Camellia::FieldContainer<double> globalValues;
  solution->mesh()->boundary().bcsToImpose(globalIndices, globalValues, *(solution->bc()), NULL);
  for (int i = 0; i < globalIndices.size(); i++)
  {
    if (globalIndices[i]==globalDofIndex)
    {
      return true;
    }
  }
  return false;
}

bool TestingUtilities::isFluxOrTraceDof(MeshPtr mesh, GlobalIndexType globalDofIndex)
{
  map<GlobalIndexType,set<GlobalIndexType> > fluxInds, fieldInds;
  getGlobalFieldFluxDofInds(mesh, fluxInds,fieldInds);
  bool value = false;
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDsGlobal();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++)
  {
    GlobalIndexType cellID = *cellIt;
    ElementPtr elem = mesh->getElement(cellID);
    if (fluxInds[cellID].find(globalDofIndex)!=fluxInds[cellID].end())
    {
      value = true;
    }
  }
  return value;
}
