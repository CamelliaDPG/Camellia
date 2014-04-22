//
//  BackwardFacingStepRefinementStrategy.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "BackwardFacingStepRefinementStrategy.h"

void BackwardFacingStepRefinementStrategy::addCorner( double x, double y ) {
  _corners.push_back( make_pair(x,y) );
}

void  BackwardFacingStepRefinementStrategy::clearCorners() {
  _corners.clear();
}

void BackwardFacingStepRefinementStrategy::refineCells(vector<int> &cellIDs) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  set<unsigned> triangleCellsToRefine;
  set<unsigned> quadCellsToRefine;
  set<unsigned> pCellsToRefine;
  
  int spaceDim = 2;
  FieldContainer<double> triangleVertices(3,spaceDim);
  FieldContainer<double> quadVertices(4,spaceDim);
  
  vector<unsigned> cellIDVector(1);
  for (vector< int >::iterator cellIDIt = cellIDs.begin();
       cellIDIt != cellIDs.end(); cellIDIt++){
    int cellID = *cellIDIt;
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
      for ( vector< pair<double, double> >::iterator cornerIt = _corners.begin(); cornerIt != _corners.end(); cornerIt++) {
        double corner_x = cornerIt->first;
        double corner_y = cornerIt->second;
        if ((abs(x-corner_x) < tol) && (abs(y-corner_y) < tol)) {
          cornerCell = true;
        }
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
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  mesh->pRefine(pCellsToRefine);
}
