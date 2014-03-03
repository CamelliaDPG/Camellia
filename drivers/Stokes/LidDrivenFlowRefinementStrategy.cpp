//
//  LidDrivenFlowRefinementStrategy.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "LidDrivenFlowRefinementStrategy.h"

#include "IndexType.h"

set<GlobalIndexType> LidDrivenFlowRefinementStrategy::symmetricCellIDs(set<GlobalIndexType> &cellIDs) {
  // find the
  set<GlobalIndexType> symmetricCellIDs;
  int spaceDim = 2;
  FieldContainer<double> cellPoints(cellIDs.size(), spaceDim);
  int i=0;
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++, i++) {
    GlobalIndexType cellID = *cellIt;
    vector<double> centroid = _solution->mesh()->getCellCentroid(cellID);
    double x = centroid[0];
    double y = centroid[1];
    // we're in [0,1]^2, and we want symmetry across the horizontal midline:
    cellPoints(i,0) = x;
    cellPoints(i,1) = 1-y;
  }
  vector< ElementPtr > elements = _solution->mesh()->elementsForPoints(cellPoints);
  for (vector< ElementPtr >::iterator elementIt = elements.begin();
       elementIt != elements.end(); elementIt++) {
    GlobalIndexType cellID = (*elementIt)->cellID();
    symmetricCellIDs.insert(cellID);
  }
  return symmetricCellIDs;
}

void LidDrivenFlowRefinementStrategy::setSymmetricRefinements(bool value) {
  _symmetricRefinements = value;
}

void LidDrivenFlowRefinementStrategy::refineCells(vector<int> &cellIDs) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  set<GlobalIndexType> triangleCellsToRefine;
  set<GlobalIndexType> quadCellsToRefine;
  set<GlobalIndexType> pCellsToRefine;
  
  int spaceDim = 2;
  FieldContainer<double> triangleVertices(3,spaceDim);
  FieldContainer<double> quadVertices(4,spaceDim);
  
  vector<GlobalIndexType> cellIDVector(1);
  for (vector< int >::iterator cellIDIt = cellIDs.begin();
       cellIDIt != cellIDs.end(); cellIDIt++){
    GlobalIndexType cellID = *cellIDIt;
    cellIDVector[0] = cellID;
    ElementTypePtr elemType = mesh->getElement(cellID)->elementType();
    BasisCachePtr basisCacheOneCell = Teuchos::rcp( new BasisCache(elemType, mesh) );
    basisCacheOneCell->setPhysicalCellNodes(mesh->physicalCellNodesForCell(cellID),cellIDVector,false); // false: don't createSideCacheToo
    double h = sqrt( basisCacheOneCell->getCellMeasures()(0) );
    //cout << "cellID " << cellID << " h: " << h << endl;
    
    // check if it's a corner element:
    quadVertices.initialize(0.0); // quad vertices will work for both triangles and quads
    mesh->verticesForCell(quadVertices,cellID);
    bool cornerCell = false;
    double tol = 1e-14;
    for (int i=0; i<mesh->getElement(cellID)->numSides(); i++) {
      double x = quadVertices(i,0);
      double y = quadVertices(i,1);
      if ((abs(1-x) < tol) && (abs(1-y) < tol)) {
        // top right
        cornerCell = true;
      } else if ((abs(x) < tol) && (abs(y) < tol)) {
        // bottom left
        cornerCell = true;
      } else if ((abs(1-x) < tol) && (abs(y) < tol)) {
        // bottom right
        cornerCell = true;
      } else if ((abs(x) < tol) && (abs(1-y) < tol)) {
        // top left
        cornerCell = true;
      }
    }
    
    int polyOrder = mesh->cellPolyOrder(cellID);
    if ((!cornerCell || (h - tol <= _hmin)) && (polyOrder < _maxPolyOrder) ) {
      pCellsToRefine.insert(cellID);
      if (_printToConsole)
        cout << "p-refining " << cellID << " (polyOrder prior to refinement: " << polyOrder << ")" << endl;
    } else if (h - tol > _hmin) {
      if (_printToConsole)
        cout << "h-refining " << cellID << " (h: " << h << ")" << endl;
      //cout << "cornerCell: " << cornerCell << endl;
      //cout << "polyOrder: " << polyOrder << endl;
      if (mesh->getElement(cellID)->numSides()==3) {
        triangleCellsToRefine.insert(cellID);
      } else if (mesh->getElement(cellID)->numSides()==4) {
        quadCellsToRefine.insert(cellID);
      }
    } else {
      if (_printToConsole)
        cout << "Skipping refinement of cellID " << cellID << " because min h " << h << " and max p " << polyOrder << " have been attained.\n";
    }
  }
  
  if (_symmetricRefinements) {
    set<GlobalIndexType> symmetricTriangleCells = symmetricCellIDs(triangleCellsToRefine);
    triangleCellsToRefine.insert(symmetricTriangleCells.begin(),symmetricTriangleCells.end());
    
    set<GlobalIndexType> symmetricQuadCells = symmetricCellIDs(quadCellsToRefine);
    quadCellsToRefine.insert(symmetricQuadCells.begin(),symmetricQuadCells.end());
    
    set<GlobalIndexType> symmetricPCells = symmetricCellIDs(pCellsToRefine);
    pCellsToRefine.insert(symmetricPCells.begin(),symmetricPCells.end());
  }
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  mesh->pRefine(pCellsToRefine);
}
