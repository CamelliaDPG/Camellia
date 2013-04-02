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

RefinementStrategy::RefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold, double min_h) {
  _solution = solution;
  _relativeEnergyThreshold = relativeEnergyThreshold;
  _enforceOneIrregularity = true;
  _reportPerCellErrors = false;
  _anisotropicThreshhold = 10.0;
  _maxAspectRatio = 2^5; // five anisotropic refinements of an element
  _min_h = min_h;
}

void RefinementStrategy::setAnisotropicThreshhold(double value){
  _anisotropicThreshhold = value;
}

void RefinementStrategy::setMaxAspectRatio(double value){
  _maxAspectRatio = value;
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
  
  map<int, double> cellMeasures;
  set<int> cellIDs = mesh->getActiveCellIDs();
  for (set<int>::iterator cellIt=cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    cellMeasures[cellID] = mesh->getCellMeasure(cellID);
  }
  
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++) {
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double cellEnergyError = energyError->find(cellID)->second;
    
    double h = sqrt(cellMeasures[cellID]);
    if (h > _min_h) {
      maxError = max(cellEnergyError,maxError);
    }
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
  vector<int> cellsToPRefine;
  
  // do refinements on cells with error above threshold
  for (vector< Teuchos::RCP< Element > >::iterator activeElemIt = activeElements.begin();
       activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    double h = sqrt(cellMeasures[cellID]);
      double cellEnergyError = energyError->find(cellID)->second;
      if ( cellEnergyError >= maxError * _relativeEnergyThreshold ) {
        //      cout << "refining cellID " << cellID << endl;
        if (h > _min_h) {
          cellsToRefine.push_back(cellID);
        } else {
          cellsToPRefine.push_back(cellID);
        }
      }
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

// without variable anisotropic threshholding
void RefinementStrategy::refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr) {
  // greedy refinement algorithm - mark cells for refinement
  Teuchos::RCP< Mesh > mesh = _solution->mesh();

  vector<int> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells);
 
  // record results prior to refinement
  RefinementResults results;
  double totalEnergyError = _solution->energyErrorTotal();
  setResults(results, mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);
  
  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());    
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());    
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());        
    
  if (_enforceOneIrregularity)
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);
    
  if (printToConsole) {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

void RefinementStrategy::refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr, map<int,double> &threshMap) {
  map<int,double> hRefMap;
  vector<ElementPtr> elems = _solution->mesh()->activeElements();
  for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){    
    hRefMap[(*elemIt)->cellID()] = true; // default to h-refinement
  }
}

// with variable anisotropic threshholding and p-refinement specification
void RefinementStrategy::refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr, map<int,double> &threshMap, map<int, bool> useHRefMap) {

  // greedy refinement algorithm - mark cells for refinement
  Teuchos::RCP< Mesh > mesh = _solution->mesh();

  vector<int> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells, threshMap);
 
  // record results prior to refinement
  RefinementResults results;
  double totalEnergyError = _solution->energyErrorTotal();
  setResults(results, mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  _results.push_back(results);

  // check if any cells should be marked for p-refinement
  vector<int> pCells;
  for (vector<int>::iterator cellIt = xCells.begin();cellIt!=xCells.end();cellIt++){
    int cellID = *cellIt;
    if (!useHRefMap[cellID]){
      pCells.push_back(cellID);
      xCells.erase(cellIt);
    }
  }
  for (vector<int>::iterator cellIt = yCells.begin();cellIt!=yCells.end();cellIt++){
    int cellID = *cellIt;
    if (!useHRefMap[cellID]){
      pCells.push_back(cellID);
      yCells.erase(cellIt);
    }
  }
  for (vector<int>::iterator cellIt = regCells.begin();cellIt!=regCells.end();cellIt++){
    int cellID = *cellIt;
    if (!useHRefMap[cellID]){
      pCells.push_back(cellID);
      regCells.erase(cellIt);
    }
  }
  
  mesh->pRefine(pCells); // p-refine FIRST
  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());    
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());    
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());        
    
  if (_enforceOneIrregularity){
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);
  }
    
  if (printToConsole) {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

void RefinementStrategy::getAnisotropicCellsToRefine(map<int,double> &xErr, map<int,double> &yErr, vector<int> &xCells, vector<int> &yCells, vector<int> &regCells){  
  map<int,double> threshMap;
  vector<ElementPtr> elems = _solution->mesh()->activeElements();
  for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){    
    threshMap[(*elemIt)->cellID()] = _anisotropicThreshhold;
  }
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells,threshMap);
}

// anisotropy with variable threshholding
void RefinementStrategy::getAnisotropicCellsToRefine(map<int,double> &xErr, map<int,double> &yErr, vector<int> &xCells, vector<int> &yCells, vector<int> &regCells, map<int,double> &threshMap){  
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  vector<int> cellsToRefine;
  getCellsAboveErrorThreshhold(cellsToRefine);
  for (vector<int>::iterator cellIt = cellsToRefine.begin();cellIt!=cellsToRefine.end();cellIt++){
    int cellID = *cellIt;    
    double h1 = mesh->getCellXSize(cellID);
    double h2 = mesh->getCellYSize(cellID);
    double min_h = min(h1,h2);
    
    double thresh = threshMap[cellID];
    double ratio = xErr[cellID]/yErr[cellID];
    bool doXAnisotropy = ratio > thresh;
    bool doYAnisotropy = ratio < 1.0/thresh;
    double aspectRatio = max(h1/h2,h2/h1); // WARNING: this assumes a *non-stretched* element (just skewed)
    double maxAspect = _maxAspectRatio; // the conservative aspect ratio from LD's DPG III: Adaptivity paper is 100. 
    if (doXAnisotropy && aspectRatio < maxAspect){ // if ratio is small = y err bigger than xErr
      xCells.push_back(cellID); // cut along y-axis
    }else if (doYAnisotropy && aspectRatio < maxAspect){ // if ratio is small = y err bigger than xErr
      yCells.push_back(cellID); // cut along x-axis
    }else{
      regCells.push_back(cellID);
    }        
  }
}

// enforcing one-irregularity with anisotropy - ONLY FOR QUADS RIGHT NOW.  ALSO NOT PARALLELIZED

bool RefinementStrategy::enforceAnisotropicOneIrregularity(vector<int> &xCells, vector<int> &yCells){
  bool success = true;
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  int maxIters = mesh->numActiveElements(); // should not refine more than the number of elements...

  // build children list - for use in "upgrading" refinements to prevent deadlocking
  vector<int> xChildren,yChildren;
  for (vector<int>::iterator cellIt = xCells.begin();cellIt!=xCells.end();cellIt++){
    ElementPtr elem = mesh->getElement(*cellIt);
    for (int i = 0;i<elem->numChildren();i++){
      xChildren.push_back(elem->getChild(i)->cellID());
    }
  }
  // build children list
  for (vector<int>::iterator cellIt = yCells.begin();cellIt!=yCells.end();cellIt++){
    ElementPtr elem = mesh->getElement(*cellIt);
    for (int i = 0;i<elem->numChildren();i++){
      yChildren.push_back(elem->getChild(i)->cellID());
    }   
  }

  bool meshIsNotRegular = true; // assume it's not regular and check elements
  int i = 0;
  while (meshIsNotRegular && i<maxIters) {
    vector <int> irregularQuadCells,xUpgrades,yUpgrades;
    vector< Teuchos::RCP< Element > > newActiveElements = mesh->activeElements();
    vector< Teuchos::RCP< Element > >::iterator newElemIt;
    
    for (newElemIt = newActiveElements.begin(); newElemIt != newActiveElements.end(); newElemIt++) {
      Teuchos::RCP< Element > current_element = *(newElemIt);
      bool isIrregular = false;
      for (int sideIndex=0; sideIndex < current_element->numSides(); sideIndex++) {
        int mySideIndexInNeighbor;
        Element* neighbor; // may be a parent
        current_element->getNeighbor(neighbor, mySideIndexInNeighbor, sideIndex);
        int numNeighborsOnSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor).size();
        if (numNeighborsOnSide > 2) isIrregular=true;
      }
      if (isIrregular){
	int cellID = current_element->cellID();
	bool isXRefined = std::find(xChildren.begin(),xChildren.end(),cellID)!=xChildren.end();
	bool isYRefined = std::find(yChildren.begin(),yChildren.end(),cellID)!=yChildren.end();
	bool isPreviouslyRefined = (isXRefined || isYRefined);
	if (!isPreviouslyRefined){ // if the cell to refine has already been refined anisotropically, don't refine it again, 
	  irregularQuadCells.push_back(cellID);
	}else if (isXRefined){ 
	  yUpgrades.push_back(cellID);
	}else if (isYRefined){ 
	  xUpgrades.push_back(cellID);
	}
      }
    }
    if (irregularQuadCells.size()>0) {
      mesh->hRefine(irregularQuadCells,RefinementPattern::regularRefinementPatternQuad());
      mesh->hRefine(xUpgrades,RefinementPattern::xAnisotropicRefinementPatternQuad());
      mesh->hRefine(yUpgrades,RefinementPattern::yAnisotropicRefinementPatternQuad());
      irregularQuadCells.clear(); xUpgrades.clear(); yUpgrades.clear();
    } else {
      meshIsNotRegular=false;
    }
    ++i;
  }
  if (i>=maxIters){
    success = false;
  }
  return success;
}
