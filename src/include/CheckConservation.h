#ifndef CHECKCONSERVATION_H
#define CHECKCONSERVATION_H

#include "InnerProductScratchPad.h"
#include "VarFactory.h"
#include "Mesh.h"

#include "CamelliaCellTools.h"

#include <Teuchos_Tuple.hpp>

namespace Camellia {
  Teuchos::Tuple<double, 3> checkConservation(FunctionPtr<double> flux, FunctionPtr<double> source, Teuchos::RCP<Mesh> mesh, int cubatureEnrichment = 0)
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
      Intrepid::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, cubatureEnrichment);
      Intrepid::FieldContainer<double> volumeIntegral(1);
      source->integrate(volumeIntegral, basisCache, true);
      int numSides = CamelliaCellTools::getSideCount(basisCache->cellTopology());
      double surfaceIntegral = 0;
      for (int sideIndex = 0; sideIndex < numSides; sideIndex++)
      {
        Intrepid::FieldContainer<double> sideIntegral(1);
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

  double computeFluxOverElementSides(FunctionPtr<double> flux, Teuchos::RCP<Mesh> mesh, vector< pair<ElementPtr, int> > originalElemFaces, int cubatureEnrichment=0)
  {
     double totalMassFlux = 0.0;
     for (vector< pair<ElementPtr, int> >::iterator origIt = originalElemFaces.begin(); origIt != originalElemFaces.end(); ++origIt)
     {
        int originalSideIndex = origIt->second;
        vector< pair<int, int> > cellFaces = origIt->first->getDescendantsForSide(originalSideIndex);
        for (vector< pair<int, int> >::iterator it = cellFaces.begin(); it != cellFaces.end(); ++it)
        {
           int cellID = it->first;
           int sideIndex = it->second;
           BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, cubatureEnrichment);
           BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
           // Intrepid::FieldContainer<double> physicalCubaturePoints = sideBasisCache->getPhysicalCubaturePoints();
           // double xCell0 = physicalCubaturePoints(0,0,0);
           // cout << physicalCubaturePoints << endl;
           Intrepid::FieldContainer<double> sideIntegral(1);
           flux->integrate(sideIntegral, sideBasisCache, true);
           totalMassFlux += sideIntegral(0);
        }
     }
     return totalMassFlux;
  }
}

#endif /* end of include guard: CHECKCONSERVATION_H */
