#include "MeshUtilities.h"
#include "SerialDenseSolveWrapper.h"

//static const double RAMP_HEIGHT = 0.0;

class RampWallBoundary : public SpatialFilter {
  double _RAMP_HEIGHT;
public:
  RampWallBoundary(double rampHeight){
    _RAMP_HEIGHT = rampHeight;
  }
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool onRamp = (abs(_RAMP_HEIGHT*(x-1)-y)<tol) && (x>1.0);
    return onRamp;
  }
};

SpatialFilterPtr MeshUtilities::rampBoundary(double RAMP_HEIGHT){
  return Teuchos::rcp(new RampWallBoundary(RAMP_HEIGHT));
}

// builds a [0,2]x[0,1] L shaped domain with 2 main blocks
MeshPtr MeshUtilities::buildRampMesh(double rampHeight, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest){

  MeshPtr mesh;
  // L-shaped domain for double ramp problem
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2);
  A(0) = 0.0; A(1) = 0.0;
  B(0) = 1.0; B(1) = 0.0;
  C(0) = 2.0; C(1) = 0.0 + rampHeight;
  D(0) = 0.0; D(1) = 1.0;
  E(0) = 1.0; E(1) = 1.0;
  F(0) = 2.0; F(1) = 1.0; //
  vector<FieldContainer<double> > vertices;
  vertices.push_back(A); int A_index = 0;
  vertices.push_back(B); int B_index = 1;
  vertices.push_back(C); int C_index = 2;
  vertices.push_back(D); int D_index = 3;
  vertices.push_back(E); int E_index = 4;
  vertices.push_back(F); int F_index = 5;
  vector< vector<int> > elementVertices;
  vector<int> el1, el2;
  // left patch:
  el1.push_back(A_index); el1.push_back(B_index); el1.push_back(E_index); el1.push_back(D_index);
  // right:
  el2.push_back(B_index); el2.push_back(C_index); el2.push_back(F_index); el2.push_back(E_index);

  elementVertices.push_back(el1);
  elementVertices.push_back(el2);
  int pToAdd = pTest-H1Order;
  mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bilinearForm, H1Order, pToAdd) );  
  return mesh;
}



// builds a [0,2]x[0,2] L shaped domain with 3 main blocks
MeshPtr MeshUtilities::buildFrontFacingStep(Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest){

  MeshPtr mesh;
  // L-shaped domain for double ramp problem
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A(0) = 0.0; A(1) = 0.0;
  B(0) = 1.0; B(1) = 0.0;
  C(0) = 0.0; C(1) = 1.0;
  D(0) = 1.0; D(1) = 1.0;
  E(0) = 2.0; E(1) = 1.0;
  F(0) = 0.0; F(1) = 2.0;
  G(0) = 1.0; G(1) = 2.0;
  H(0) = 2.0; H(1) = 2.0;
  vector<FieldContainer<double> > vertices;
  vertices.push_back(A); int A_index = 0;
  vertices.push_back(B); int B_index = 1;
  vertices.push_back(C); int C_index = 2;
  vertices.push_back(D); int D_index = 3;
  vertices.push_back(E); int E_index = 4;
  vertices.push_back(F); int F_index = 5;
  vertices.push_back(G); int G_index = 6;
  vertices.push_back(H); int H_index = 7;
  vector< vector<int> > elementVertices;
  vector<int> el1, el2, el3;
  // left patch:
  el1.push_back(A_index); el1.push_back(B_index); el1.push_back(D_index); el1.push_back(C_index);
  // top right:
  el2.push_back(C_index); el2.push_back(D_index); el2.push_back(G_index); el2.push_back(F_index);
  // bottom right:
  el3.push_back(D_index); el3.push_back(E_index); el3.push_back(H_index); el3.push_back(G_index);

  elementVertices.push_back(el1);
  elementVertices.push_back(el2);
  elementVertices.push_back(el3);
  int pToAdd = pTest-H1Order;
  mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bilinearForm, H1Order, pToAdd) );  
  return mesh;
}


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
