//
//  ErrorPercentageRefinementStrategy.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/20/14.
//
//

#include "ErrorPercentageRefinementStrategy.h"

#include "CamelliaDebugUtility.h"

#include "Solution.h"

using namespace Camellia;

ErrorPercentageRefinementStrategy::ErrorPercentageRefinementStrategy(SolutionPtr<double> soln, double percentageThreshold, double min_h, int max_p, bool preferPRefinements)
                                                : RefinementStrategy(soln, 1.0, min_h, max_p, preferPRefinements) {
  // percentageThreshold should be a number in [0,1]
  _percentageThreshold = percentageThreshold;
}

void ErrorPercentageRefinementStrategy::refine(bool printToConsole) {
  MeshPtr mesh = this->mesh();

  map<GlobalIndexType, double> energyError;
  if (_rieszRep.get() != NULL) {
    _rieszRep->computeRieszRep();
    energyError = _rieszRep->getNormsSquaredGlobal();
    // take square roots:
    for (map<GlobalIndexType, double>::iterator energyEntryIt = energyError.begin();
         energyEntryIt != energyError.end(); energyEntryIt++) {
      energyEntryIt->second = sqrt( energyEntryIt->second );
    }
  } else {
    energyError = _solution->globalEnergyError();
  }
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();

  double maxError = 0.0;
  double totalEnergyErrorSquared = 0.0;

  map<GlobalIndexType, double> cellMeasures;
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt=cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    cellMeasures[cellID] = mesh->getCellMeasure(cellID);
  }

  map<double, vector<GlobalIndexType> > errorReverseLookup; // an easy way to sort by error amount

  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError.find(cellID)->second;

    errorReverseLookup[cellEnergyError].push_back(cellID);

    double h = sqrt(cellMeasures[cellID]);
    if (h > _min_h) {
      maxError = max(cellEnergyError,maxError);
    }
    totalEnergyErrorSquared += cellEnergyError * cellEnergyError;
  }
  double totalEnergyError = sqrt(totalEnergyErrorSquared);
  if ( printToConsole && _reportPerCellErrors ) {
    cout << "per-cell Energy Error Squared for cells with > 0.1% of squared energy error\n";
    for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
         activeElemIt != activeElements.end(); activeElemIt++) {
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      double cellEnergyError = energyError.find(cellID)->second;
      double percent = (cellEnergyError*cellEnergyError) / totalEnergyErrorSquared * 100;
      if (percent > 0.1) {
        cout << cellID << ": " << cellEnergyError*cellEnergyError << " ( " << percent << " %)\n";
      }
    }
  }

  // record results prior to refinement
  RefinementResults results = setResults(mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);

  vector<GlobalIndexType> cellsToRefine;
  vector<GlobalIndexType> cellsToPRefine;

  double errorSquaredThatWeCanIgnore = (1- _percentageThreshold) * totalEnergyErrorSquared;

  double errorSquaredEncounteredThusFar = 0;
  for (map<double, vector<GlobalIndexType> >::iterator errorEntryIt=errorReverseLookup.begin(); errorEntryIt != errorReverseLookup.end(); errorEntryIt++) {
    double error = errorEntryIt->first;
    vector<GlobalIndexType> cells = errorEntryIt->second;
    for (vector<GlobalIndexType>::iterator cellIt = cells.begin(); cellIt != cells.end(); cellIt++) {
      errorSquaredEncounteredThusFar += error * error;
      if (errorSquaredEncounteredThusFar > errorSquaredThatWeCanIgnore) {
        GlobalIndexType cellID = *cellIt;
        double h = sqrt(cellMeasures[cellID]);
        int p = mesh->cellPolyOrder(cellID);

        //      cout << "refining cellID " << cellID << endl;
        if (!_preferPRefinements) {
          if (h > _min_h) {
            cellsToRefine.push_back(cellID);
          } else {
            cellsToPRefine.push_back(cellID);
          }
        } else {
          if (p < _max_p) {
            cellsToPRefine.push_back(cellID);
          } else {
            cellsToRefine.push_back(cellID);
          }
        }
      }
    }
  }

  if (printToConsole) {
    if (cellsToRefine.size() > 0) Camellia::print("cells for h-refinement", cellsToRefine);
    if (cellsToPRefine.size() > 0) Camellia::print("cells for p-refinement", cellsToPRefine);
  }
  refineCells(cellsToRefine);
  pRefineCells(mesh, cellsToPRefine);

  if (_enforceOneIrregularity)
    mesh->enforceOneIrregularity();

  if (printToConsole) {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}
