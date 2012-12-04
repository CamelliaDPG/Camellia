//
//  RefinementStrategy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RefinementStrategy.h"
#include "Mesh.h"
#include "Solution.h"

RefinementStrategy::RefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold) {
  _solution = solution;
  _relativeEnergyThreshold = relativeEnergyThreshold;
  _enforceOneIrregularity = true;
  _reportPerCellErrors = false;
}

void RefinementStrategy::setEnforceOneIrregularity(bool value) {
  _enforceOneIrregularity = value;
}

void RefinementStrategy::setReportPerCellErrors(bool value) {
  _reportPerCellErrors = value;
}

void RefinementStrategy::refine(bool printToConsole) {
  // greedy refinement algorithm - mark cells for refinement
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  const map<int, double>* energyError = &(_solution->energyError());
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  
  double maxError = 0.0;
  double totalEnergyError = 0.0;
  
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    maxError = max(cellEnergyError,maxError);
    totalEnergyError += cellEnergyError * cellEnergyError; 
  }
  totalEnergyError = sqrt(totalEnergyError);
  if ( printToConsole && _reportPerCellErrors ) {
    cout << "per-cell Energy Error Squared for cells with > 0.1% of squared energy error\n";
    for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
         activeElemIt != activeElements.end(); activeElemIt++) {
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      double cellEnergyError = energyError->find(cellID)->second;
      double percent = (cellEnergyError*cellEnergyError) / (totalEnergyError*totalEnergyError) * 100;
      if (percent > 0.1) {
        cout << cellID << ": " << cellEnergyError*cellEnergyError << " ( " << percent << " %)\n";
      }
    }
  }
  
  // record results prior to refinement
  RefinementResults results;
  setResults(results, mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);
  
  vector<int> cellsToRefine;
  
  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    if ( cellEnergyError >= maxError * _relativeEnergyThreshold ) {
      //      cout << "refining cellID " << cellID << endl;
      cellsToRefine.push_back(cellID);
    }
  }
  
  refineCells(cellsToRefine);
  
  if (_enforceOneIrregularity)
    mesh->enforceOneIrregularity();
  
  if (printToConsole) {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

void RefinementStrategy::getCellsAboveErrorThreshhold(vector<int> &cellsToRefine){
  // greedy refinement algorithm - mark cells for refinement
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  const map<int, double>* energyError = &(_solution->energyError());
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  
  double maxError = 0.0;
  double totalEnergyError = 0.0;
  
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    maxError = max(cellEnergyError,maxError);
    totalEnergyError += cellEnergyError * cellEnergyError; 
  }
  totalEnergyError = sqrt(totalEnergyError);

  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    if ( cellEnergyError >= maxError * _relativeEnergyThreshold ) {
      cellsToRefine.push_back(cellID);
    }
  }
}

// defaults to h-refinement
void RefinementStrategy::refineCells(vector<int> &cellIDs) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  hRefineCells(mesh, cellIDs);
}

void RefinementStrategy::pRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs) {
  mesh->pRefine(cellIDs);  
}


void RefinementStrategy::hRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs) {
  vector<int> triangleCellsToRefine;
  vector<int> quadCellsToRefine;
  
  for (vector< int >::const_iterator cellIDIt = cellIDs.begin();
       cellIDIt != cellIDs.end(); cellIDIt++){
    int cellID = *cellIDIt;
    
    if (mesh->getElement(cellID)->numSides()==3) {
      triangleCellsToRefine.push_back(cellID);
    } else if (mesh->getElement(cellID)->numSides()==4) {
      quadCellsToRefine.push_back(cellID);
    }
  }
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

void RefinementStrategy::hRefineUniformly(Teuchos::RCP<Mesh> mesh) {
  vector<int> cellsToRefine;
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    cellsToRefine.push_back(current_element->cellID());
  }
  hRefineCells(mesh, cellsToRefine);
}

void RefinementStrategy::setResults(RefinementResults &solnResults, int numElements, int numDofs,
                                    double totalEnergyError) {
  solnResults.numElements = numElements;
  solnResults.numDofs = numDofs;
  solnResults.totalEnergyError = totalEnergyError;
}
