//
//  LidDrivenFlowRefinementStrategy.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "LidDrivenFlowRefinementStrategy.h"

void LidDrivenFlowRefinementStrategy::refineCells(vector<int> &cellIDs) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  vector<int> triangleCellsToRefine;
  vector<int> quadCellsToRefine;
  vector<int> pCellsToRefine;
  
  int maxPolyOrder = 11; // corresponds to L^2 (field) order of 10
  int spaceDim = 2;
  FieldContainer<double> triangleVertices(3,spaceDim);
  FieldContainer<double> quadVertices(4,spaceDim);
  
  for (vector< int >::iterator cellIDIt = cellIDs.begin();
       cellIDIt != cellIDs.end(); cellIDIt++){
    int cellID = *cellIDIt;
    
    // check if it's a corner element:
    quadVertices.initialize(0.0); // quad vertices will work for both triangles and quads
    mesh->verticesForCell(quadVertices,cellID);
    bool cornerCell = false;
    for (int i=0; i<mesh->getElement(cellID)->numSides(); i++) {
      double tol = 1e-14;
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
    
    if (!cornerCell && (mesh->cellPolyOrder(cellID) < maxPolyOrder)) {
      pCellsToRefine.push_back(cellID);
    } else {
      if (mesh->getElement(cellID)->numSides()==3) {
        triangleCellsToRefine.push_back(cellID);
      } else if (mesh->getElement(cellID)->numSides()==4) {
        quadCellsToRefine.push_back(cellID);
      }
    }
  }
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  mesh->pRefine(pCellsToRefine);
}