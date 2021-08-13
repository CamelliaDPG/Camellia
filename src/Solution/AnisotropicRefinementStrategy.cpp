// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  AnisotropicRefinementStrategy.h
//  Camellia
//
//  Created by Nate Roberts on 8/9/16, based on code from Jesse Chan.
//
//

#include "AnisotropicRefinementStrategy.h"

using namespace Camellia;

template <typename Scalar>
AnisotropicRefinementStrategy<Scalar>::AnisotropicRefinementStrategy(TSolutionPtr<Scalar> solution, double relativeEnergyThreshold,
                                                                     double min_h, int max_p, bool preferPRefinements)
{
  _solution = solution;
  this->_relativeErrorThreshold = relativeEnergyThreshold;
  this->_enforceOneIrregularity = true;
  _anisotropicThreshhold = 10.0;
  _maxAspectRatio = 1 << 5; // five anisotropic refinements of an element
  this->_min_h = min_h;
  this->_preferPRefinements = preferPRefinements;
  this->_max_p = max_p;
}

template <typename Scalar>
AnisotropicRefinementStrategy<Scalar>::AnisotropicRefinementStrategy(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                                     double relativeEnergyThreshold, double min_h, int max_p,
                                                                     bool preferPRefinements)
{
  _rieszRep = Teuchos::rcp( new TRieszRep<Scalar>(mesh, ip, residual) );
  this->_relativeErrorThreshold = relativeEnergyThreshold;
  this->_enforceOneIrregularity = true;
  _anisotropicThreshhold = 10.0;
  _maxAspectRatio = 1 << 5; // five anisotropic refinements of an element
  this->_min_h = min_h;
  this->_preferPRefinements = preferPRefinements;
  this->_max_p = max_p;
}

// without variable anisotropic threshholding
template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();
  
  vector<GlobalIndexType> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells);
  
  // record results prior to refinement
  double totalEnergyError = _solution->energyErrorTotal();
  RefinementResults results = this->setResults(mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  this->_results.push_back(results);
  
  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());
  
  if (this->_enforceOneIrregularity)
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);
  
  if (printToConsole)
  {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap)
{
  map<GlobalIndexType,bool> hRefMap;
  set<GlobalIndexType> cellIDs = _solution->mesh()->getActiveCellIDsGlobal();
  for (GlobalIndexType cellID : cellIDs)
  {
    hRefMap[cellID] = true; // default to h-refinement
  }
  refine(printToConsole,xErr,yErr,threshMap,hRefMap);
}

// with variable anisotropic threshholding and p-refinement specification
template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::refine(bool printToConsole, map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, map<GlobalIndexType,double> &threshMap, map<GlobalIndexType, bool> useHRefMap)
{
  
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();
  
  vector<GlobalIndexType> xCells, yCells, regCells;
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells, threshMap);
  
  // record results prior to refinement
  double totalEnergyError = _solution->energyErrorTotal();
  RefinementResults results = this->setResults(mesh->numElements(), mesh->numGlobalDofs(), totalEnergyError);
  this->_results.push_back(results);
  
  // check if any cells should be marked for p-refinement
  vector<GlobalIndexType> pCells;
  for (vector<GlobalIndexType>::iterator cellIt = xCells.begin(); cellIt!=xCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      xCells.erase(cellIt);
    }
  }
  for (vector<GlobalIndexType>::iterator cellIt = yCells.begin(); cellIt!=yCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      yCells.erase(cellIt);
    }
  }
  for (vector<GlobalIndexType>::iterator cellIt = regCells.begin(); cellIt!=regCells.end(); cellIt++)
  {
    int cellID = *cellIt;
    if (!useHRefMap[cellID])
    {
      pCells.push_back(cellID);
      regCells.erase(cellIt);
    }
  }
  
  mesh->pRefine(pCells); // p-refine FIRST
  mesh->hRefine(xCells, RefinementPattern::xAnisotropicRefinementPatternQuad());
  mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());
  mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());
  
  if (this->_enforceOneIrregularity)
  {
    //    mesh->enforceOneIrregularity();
    enforceAnisotropicOneIrregularity(xCells,yCells);
  }
  
  if (printToConsole)
  {
    cout << "Prior to refinement, energy error: " << totalEnergyError << endl;
    cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
  }
}

template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells)
{
  map<GlobalIndexType,double> threshMap;
  set<GlobalIndexType> cellIDs = _solution->mesh()->getActiveCellIDsGlobal();
  for (GlobalIndexType cellID : cellIDs)
  {
    threshMap[cellID] = _anisotropicThreshhold;
  }
  getAnisotropicCellsToRefine(xErr,yErr,xCells,yCells,regCells,threshMap);
}

// anisotropy with variable threshholding
template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::getAnisotropicCellsToRefine(map<GlobalIndexType,double> &xErr, map<GlobalIndexType,double> &yErr, vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells, vector<GlobalIndexType> &regCells, map<GlobalIndexType,double> &threshMap)
{
  MeshPtr mesh = this->mesh();
  vector<GlobalIndexType> cellsToRefine;
  this->getCellsAboveErrorThreshhold(cellsToRefine);
  for (vector<GlobalIndexType>::iterator cellIt = cellsToRefine.begin(); cellIt!=cellsToRefine.end(); cellIt++)
  {
    int cellID = *cellIt;
    double h1 = mesh->getCellXSize(cellID);
    double h2 = mesh->getCellYSize(cellID);
    double min_h = min(h1,h2);
    
    double thresh = threshMap[cellID];
    double ratio = xErr[cellID]/yErr[cellID];
    
    /*
     double anisoErr = xErr[cellID] + yErr[cellID];
     double energyErr = energyError[cellID];
     double anisoPercentage = anisoErr/energyErr;
     cout << "aniso percentage = " << anisoPercentage << endl;
     */
    bool doXAnisotropy = ratio > thresh;
    bool doYAnisotropy = ratio < 1.0/thresh;
    double aspectRatio = max(h1/h2,h2/h1); // WARNING: this assumes a *non-squashed/stretched* element (just skewed)
    double maxAspect = _maxAspectRatio; // the conservative aspect ratio from LD's DPG III: Adaptivity paper is 100.
    // don't refine if h is already too small
    bool doAnisotropy = (aspectRatio < maxAspect);
    if (min_h > this->_min_h)
    {
      if (doXAnisotropy && doAnisotropy)   // if ratio is small = y err bigger than xErr
      {
        xCells.push_back(cellID); // cut along y-axis
      }
      else if (doYAnisotropy && doAnisotropy)     // if ratio is small = y err bigger than xErr
      {
        yCells.push_back(cellID); // cut along x-axis
      }
      else
      {
        regCells.push_back(cellID);
      }
    }
  }
}

// enforcing one-irregularity with anisotropy - ONLY FOR QUADS RIGHT NOW.  ALSO NOT PARALLELIZED
template <typename Scalar>
bool AnisotropicRefinementStrategy<Scalar>::enforceAnisotropicOneIrregularity(vector<GlobalIndexType> &xCells, vector<GlobalIndexType> &yCells)
{
  bool success = true;
  MeshPtr mesh = this->mesh();
  int maxIters = mesh->numActiveElements(); // should not refine more than the number of elements...
  
  // build children list - for use in "upgrading" refinements to prevent deadlocking
  vector<GlobalIndexType> xChildren,yChildren;
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  for (GlobalIndexType xCellID : xCells)
  {
    CellPtr cell = meshTopo->getCell(xCellID);
    vector<GlobalIndexType> cellChildren = cell->getChildIndices(meshTopo);
    xChildren.insert(xChildren.end(),cellChildren.begin(),cellChildren.end());
  }
  // build children list
  for (GlobalIndexType yCellID : yCells)
  {
    CellPtr cell = meshTopo->getCell(yCellID);
    vector<GlobalIndexType> cellChildren = cell->getChildIndices(meshTopo);
    yChildren.insert(yChildren.end(),cellChildren.begin(),cellChildren.end());
  }
  
  bool meshIsNotRegular = true; // assume it's not regular and check elements
  int i = 0;
  while (meshIsNotRegular && i<maxIters)
  {
    vector<GlobalIndexType> irregularQuadCells,xUpgrades,yUpgrades;
    set<GlobalIndexType> newActiveCellIDs = mesh->getActiveCellIDsGlobal();
    
    for (GlobalIndexType activeCellID : newActiveCellIDs)
    {
      CellPtr cell = meshTopo->getCell(activeCellID);
      bool isIrregular = false;
      for (int sideIndex=0; sideIndex < cell->getSideCount(); sideIndex++)
      {
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideIndex, meshTopo); // ->getNeighbor(mySideIndexInNeighbor, sideIndex);
        GlobalIndexType neighborCellID = neighborInfo.first;
        int mySideIndexInNeighbor = neighborInfo.second;
        
        if (neighborCellID != -1)
        {
          CellPtr neighborCell = meshTopo->getCell(neighborCellID);
          int numNeighborsOnSide = neighborCell->getDescendantsForSide(mySideIndexInNeighbor, meshTopo).size();
          if (numNeighborsOnSide > 2) isIrregular=true;
        }
      }
      if (isIrregular)
      {
        std::sort(xChildren.begin(), xChildren.end());
        std::sort(yChildren.begin(), yChildren.end());
        bool isXRefined = std::find(xChildren.begin(),xChildren.end(),activeCellID)!=xChildren.end();
        bool isYRefined = std::find(yChildren.begin(),yChildren.end(),activeCellID)!=yChildren.end();
        bool isPreviouslyRefined = (isXRefined || isYRefined);
        if (!isPreviouslyRefined)   // if the cell to refine has already been refined anisotropically, don't refine it again,
        {
          irregularQuadCells.push_back(activeCellID);
        }
        else if (isXRefined)
        {
          yUpgrades.push_back(activeCellID);
        }
        else if (isYRefined)
        {
          xUpgrades.push_back(activeCellID);
        }
      }
    }
    if (irregularQuadCells.size()>0)
    {
      mesh->hRefine(irregularQuadCells,RefinementPattern::regularRefinementPatternQuad());
      mesh->hRefine(xUpgrades,RefinementPattern::xAnisotropicRefinementPatternQuad());
      mesh->hRefine(yUpgrades,RefinementPattern::yAnisotropicRefinementPatternQuad());
      irregularQuadCells.clear();
      xUpgrades.clear();
      yUpgrades.clear();
    }
    else
    {
      meshIsNotRegular=false;
    }
    ++i;
  }
  if (i>=maxIters)
  {
    success = false;
  }
  return success;
}


template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::setAnisotropicThreshhold(double value)
{
  _anisotropicThreshhold = value;
}

template <typename Scalar>
void AnisotropicRefinementStrategy<Scalar>::setMaxAspectRatio(double value)
{
  _maxAspectRatio = value;
}
