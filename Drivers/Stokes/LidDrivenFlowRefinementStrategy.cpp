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
  
  int maxPolyOrder = 6; // corresponds to L^2 (field) order of 5
  int spaceDim = 2;
  FieldContainer<double> triangleVertices(3,spaceDim);
  FieldContainer<double> quadVertices(4,spaceDim);
  
  vector<int> cellIDVector(1);
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
    if ((!cornerCell || (h <= 1.0 / 64.0)) && (polyOrder < maxPolyOrder) ) {
      pCellsToRefine.push_back(cellID);
//      cout << "p-refining " << cellID << endl;
    } else if (h - tol > 1.0 / 64.0) {
      //cout << "h-refining " << cellID << " (h: " << h << ")" << endl;
      //cout << "cornerCell: " << cornerCell << endl;
      //cout << "polyOrder: " << polyOrder << endl;
      if (mesh->getElement(cellID)->numSides()==3) {
        triangleCellsToRefine.push_back(cellID);
      } else if (mesh->getElement(cellID)->numSides()==4) {
        quadCellsToRefine.push_back(cellID);
      }
    } else {
      cout << "Skipping refinement of cellID " << cellID << " because min h and max p have been attained.\n";
    }
  }
  
  mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  mesh->pRefine(pCellsToRefine);
}
