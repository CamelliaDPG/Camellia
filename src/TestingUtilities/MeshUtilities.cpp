#include "MeshUtilities.h"
#include "SerialDenseSolveWrapper.h"

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

void MeshUtilities::writeMatrixToSparseDataFile(FieldContainer<double> &matrix, string filename) {
  // matlab-friendly format (use spconvert)
  int rows = matrix.dimension(0);
  int cols = matrix.dimension(1);
  ofstream fout(filename.c_str());
  // specify dimensions:
  fout << rows << "\t" << cols << "\t"  << 0 << endl;
  double tol = 1e-15;
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      if (abs(matrix(i,j)) > tol) { // nonzero
        fout << i+1 << "\t" << j+1 << "\t" << matrix(i,j) << endl;
      }
    }
  }
  fout.close();
}

double MeshUtilities::computeMaxLocalConditionNumber(IPPtr ip, MeshPtr mesh, string sparseFileToWriteTo) {
  set<int> cellIDs = mesh->getActiveCellIDs();
  FieldContainer<double> maxConditionNumberIPMatrix;
  int maxCellID = -1;
  double maxConditionNumber = -1;
  for (set<int>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    bool testVsTest = true;
    BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);
    DofOrderingPtr testSpace = mesh->getElement(cellID)->elementType()->testOrderPtr;
    int testDofs = testSpace->totalDofs();
    int numCells = 1;
    FieldContainer<double> innerProductMatrix(numCells,testDofs,testDofs);
    ip->computeInnerProductMatrix(innerProductMatrix, testSpace, cellBasisCache);
    // reshape:
    innerProductMatrix.resize(testDofs,testDofs);
    double conditionNumber = SerialDenseSolveWrapper::estimateConditionNumber(innerProductMatrix);
    if (conditionNumber > maxConditionNumber) {
      maxConditionNumber = conditionNumber;
      maxConditionNumberIPMatrix = innerProductMatrix;
      maxCellID = cellID;
    } else if (maxConditionNumberIPMatrix.size()==0) {
      // could be that the estimation failed--we still want a copy of the matrix written to file.
      maxConditionNumberIPMatrix = innerProductMatrix;
    }
  }
  if (sparseFileToWriteTo.length() > 0) {
    if (maxConditionNumberIPMatrix.size() > 0) {
      writeMatrixToSparseDataFile(maxConditionNumberIPMatrix, sparseFileToWriteTo);
    }
  }
//  cout << "max condition number occurs in cellID " << maxCellID << endl;
  return maxConditionNumber;
}
