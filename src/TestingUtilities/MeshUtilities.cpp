#include "MeshUtilities.h"

// builds a [0,1]x[0,1] square mesh with evenly spaced horizontal/vertical cells
MeshPtr MeshUtilities::buildUnitQuadMesh(int horizontalCells, int verticalCells, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest){
  FieldContainer<double> quadPoints(4,2);
  double squareSize = 1.0;
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = squareSize;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = squareSize;
  quadPoints(2,1) = squareSize;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = squareSize;
  
  // create a pointer to a new mesh:
  return Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bilinearForm, H1Order, pTest);
}

MeshPtr MeshUtilities::buildUnitQuadMesh(int nCells, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest){
  return MeshUtilities::buildUnitQuadMesh(nCells,nCells, bilinearForm, H1Order, pTest);
}

