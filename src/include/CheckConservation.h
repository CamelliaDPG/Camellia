#ifndef CHECKCONSERVATION_H
#define CHECKCONSERVATION_H

#include "InnerProductScratchPad.h"
#include "VarFactory.h"
#include "Mesh.h"

#include <Teuchos_Tuple.hpp>

Teuchos::Tuple<double, 3> checkConservation(FunctionPtr flux, FunctionPtr source, VarFactory& varFactory, Teuchos::RCP<Mesh> mesh, int fakeTestOrder = 3)
{
  // Check conservation by testing against one
  VarPtr testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  // Create a fake bilinear form for the testing
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  // Define our mass flux
  LinearTermPtr fluxTerm = flux * testOne;
  LinearTermPtr sourceTerm = source * testOne;

  Teuchos::RCP<shards::CellTopology> quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(fakeBF);
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);

  int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
  vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
  map<int, double> massFluxIntegral; // cellID -> integral
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
    vector<int> cellIDs;
    for (int i=0; i<elems.size(); i++) {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh) );
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    // FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
    FieldContainer<double> surfaceIntegrals(elems.size(),testOrdering->totalDofs());
    FieldContainer<double> volumeIntegrals(elems.size(),testOrdering->totalDofs());
    // massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    fluxTerm->integrate(surfaceIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    sourceTerm->integrate(volumeIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = surfaceIntegrals(i,testOneIndex) + volumeIntegrals(i,testOneIndex);
    }
    // find the largest:
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
    }
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }

  Teuchos::Tuple<double, 3> fluxImbalances = Teuchos::tuple(maxMassFluxIntegral, totalMassFlux, totalAbsMassFlux);

  return fluxImbalances;
}

#endif /* end of include guard: CHECKCONSERVATION_H */
