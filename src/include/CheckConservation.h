#ifndef CHECKCONSERVATION_H
#define CHECKCONSERVATION_H

#include "InnerProductScratchPad.h"
#include "VarFactory.h"
#include "Mesh.h"

#include <Teuchos_Tuple.hpp>

Teuchos::Tuple<double, 3> checkConservation(FunctionPtr flux, FunctionPtr source, VarFactory& varFactory, Teuchos::RCP<Mesh> mesh, int cubatureEnrichment = 0)
{
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  vector<ElementPtr> elems = mesh->activeElements();
  for (vector<ElementPtr>::iterator it = elems.begin(); it != elems.end(); ++it)
  {
    ElementPtr elem = *it;
    int cellID = elem->cellID();
    ElementTypePtr elemType = elem->elementType();
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, cubatureEnrichment);
    FieldContainer<double> volumeIntegral(1);
    source->integrate(volumeIntegral, basisCache, true);
    int numSides = basisCache->cellTopology().getSideCount();
    double surfaceIntegral = 0;
    for (int sideIndex = 0; sideIndex < numSides; sideIndex++)
    {
      FieldContainer<double> sideIntegral(1);
      flux->integrate(sideIntegral, basisCache->getSideBasisCache(sideIndex), true);
      surfaceIntegral += sideIntegral(0);
    }
    double massFlux = surfaceIntegral - volumeIntegral(0);

    maxMassFluxIntegral = max(abs(massFlux), maxMassFluxIntegral);
    totalMassFlux += massFlux;
    totalAbsMassFlux += abs( massFlux );
  }

  Teuchos::Tuple<double, 3> fluxImbalances = Teuchos::tuple(maxMassFluxIntegral, totalMassFlux, totalAbsMassFlux);

  return fluxImbalances;
}

#endif /* end of include guard: CHECKCONSERVATION_H */
