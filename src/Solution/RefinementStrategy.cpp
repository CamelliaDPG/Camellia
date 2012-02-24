//
//  RefinementStrategy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RefinementStrategy.h"

RefinementStrategy::RefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold) {
  _solution = solution;
  _relativeEnergyThreshold = relativeEnergyThreshold;
}

void RefinementStrategy::refine(bool printToConsole) {
  // greedy refinement algorithm - mark cells for refinement
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  map<int, double> energyError;
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  
  _solution->energyError(energyError);
  
  vector<int> triangleCellsToRefine;
  vector<int> quadCellsToRefine;
  double maxError = 0.0;
  double totalEnergyError = 0.0;
  
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    maxError = max(energyError[cellID],maxError);
    totalEnergyError += energyError[current_element->cellID()]*energyError[current_element->cellID()]; 
  }
  
  // record results prior to refinement
  RefinementResults results;
  setResults(results, mesh->numElements(), mesh->numGlobalDofs(), sqrt(totalEnergyError));
  _results.push_back(results);
  
  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    if ( energyError[cellID] >= maxError * _relativeEnergyThreshold ) {
      if (current_element->numSides()==3) {
        triangleCellsToRefine.push_back(cellID);
      } else if (current_element->numSides()==4) {
        quadCellsToRefine.push_back(cellID);
      }
    }
  }
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  mesh->enforceOneIrregularity();

  if (printToConsole) {
    cout << "Prior to refinement, energy error: " << sqrt(totalEnergyError) << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

void RefinementStrategy::setResults(RefinementResults &solnResults, int numElements, int numDofs,
                                    double totalEnergyError) {
  solnResults.numElements = numElements;
  solnResults.numDofs = numDofs;
  solnResults.totalEnergyError = totalEnergyError;
}