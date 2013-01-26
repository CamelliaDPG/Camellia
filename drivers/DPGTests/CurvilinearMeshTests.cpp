//
//  CurvilinearMeshTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#include "CurvilinearMeshTests.h"

#include "MeshFactory.h"
#include "Mesh.h"
#include "Function.h"

#include "GnuPlotUtil.h"

#include "StokesFormulation.h"

const static double PI  = 3.141592653589793238462;

void CurvilinearMeshTests::setup() {
  
}

void CurvilinearMeshTests::teardown() {
  
}

void CurvilinearMeshTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testEdgeLength()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testStraightEdgeMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testCylinderMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool CurvilinearMeshTests::testCylinderMesh() {
  bool success = true;
  
  FunctionPtr one = Function::constant(1.0);
  
  double width = 3.0;
  double height = 3.0;
  double r = 1.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  double trueArea = width * height - PI * r * r;
  
  int H1Order = 1;
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  MeshPtr mesh = MeshFactory::hemkerMesh(width, height, r, bf, H1Order, pToAdd);
  
//  GnuPlotUtil::writeExactMeshSkeleton("/tmp/cylinderFlowExactMesh.dat", mesh, 10);
  
//  double approximateArea = one->integrate(mesh);
  
//  cout << setprecision(15);
//  cout << "Exact area:" << trueArea << endl;
//  cout << "Approximate area on straight-line mesh: " << approximateArea << endl;
//
  
  int numCells = mesh->numElements();
  set<int> allCells;
  for (int i=0; i<numCells; i++) {
    allCells.insert(i);
  }
  
  double tol = 1e-10;
  cout << setprecision(15);
  // test p-convergence of mesh area
  double previousError = 1000;
  int numPRefinements = 5;
  for (int i=1; i<=numPRefinements; i++) {
    double approximateArea = one->integrate(mesh);
//    double impliedPi = (width * height - approximateArea) / (r*r);
//    cout << "For k=" << i << ", implied value of pi: " << impliedPi;
//    cout << " (error " << abs(PI-impliedPi) << ")\n";
//    cout << "Area with H1Order " << H1Order << ": " << approximateArea << endl;
    double error = abs(trueArea - approximateArea);
    if ((error > previousError) && (error > tol)) { // non-convergence
      success = false;
      cout << "Error with H1Order = " << i << " is greater than with H1Order = " << i - 1 << endl;
      cout << "Current error = " << error << "; previous = " << previousError << endl;
    }
//    ostringstream filePath;
//    filePath << "/tmp/cylinderFlowMesh" << H1Order << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    previousError = error;
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(allCells);
    }
  }
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::hemkerMesh(width, height, r, bf, H1Order, pToAdd);
  previousError = 1000;
  int numHRefinements = 5;
  for (int i=0; i<=numHRefinements; i++) {
    double approximateArea = one->integrate(mesh);
//    double impliedPi = (width * height - approximateArea) / (r*r);
//    cout << "For h-refinement " << i << ", implied value of pi: " << impliedPi;
//    cout << " (error " << abs(PI-impliedPi) << ")\n";
    //    cout << "Area with H1Order " << H1Order << ": " << approximateArea << endl;
    double error = abs(trueArea - approximateArea);
    if ((error > previousError) && (error > tol)) { // non-convergence
      success = false;
      cout << "Error for h-refinement " << i << " is greater than for h-refinement " << i - 1 << endl;
      cout << "Current error = " << error << "; previous = " << previousError << endl;
    }
//    ostringstream filePath;
//    filePath << "/tmp/cylinderFlowMesh_h" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
//    filePath.str("");
//    filePath << "/tmp/cylinderFlowMesh_h" << i << "_straight_lines.dat";
//    GnuPlotUtil::writeExactMeshSkeleton(filePath.str(), mesh, 2);
    previousError = error;
    
    // h-refine
    if (i<numHRefinements) {
      mesh->hRefine(mesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
    }
  }
  
  return success;
}

bool CurvilinearMeshTests::testEdgeLength() {
  bool success = true;
  
  // to begin, a very simple test: do we compute the correct area for a square?
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr oneOnBoundary = Function::restrictToCellBoundary(one);
  
  double radius = 1.0;
  
  double meshWidth = radius * sqrt(2.0);
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  // make a single-element mesh:
  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  double perimeter = oneOnBoundary->integrate(mesh);
  
  double tol = 1e-14;
  double expectedPerimeter = 4*(meshWidth);
  double err = abs(perimeter - expectedPerimeter);
  if (err > tol) {
    cout << "Problem with test: on square mesh, perimeter should be " << expectedPerimeter;
    cout << " but is " << perimeter << endl;
    success = false;
  }
  
  int numPRefinements = 5;
  for (int i=1; i<=numPRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    cout << "perimeter: " << perimeter << endl;
    double error = abs(expectedPerimeter - perimeter);
    if (error > tol) {
      success = false;
      cout << "testEdgeLength: On square mesh, error with H1Order = " << i << " exceeds tol " << tol << endl;
      cout << "Error = " << error << endl;
    }
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
  }
  
//  // now, do much the same thing, except with h-refinements:
//  H1Order = 2;
//  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
//  int numHRefinements = 5;
//  for (int i=0; i<=numHRefinements; i++) {
//    perimeter = oneOnBoundary->integrate(mesh);
//    cout << "perimeter: " << perimeter << endl;
//    double error = abs(expectedPerimeter - perimeter);
//    if (error > tol) {
//      success = false;
//      cout << "testEdgeLength: On square mesh, error for h-refinement " << i << " exceeds tol " << tol << endl;
//      cout << "Error = " << error << endl;
//    }
//    // h-refine
//    if (i<numHRefinements) {
//      mesh->hRefine(mesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
//    }
//  }
  
  // now for the real test: swap out the edges for circular arcs.
  ParametricCurvePtr circle = ParametricCurve::circle(radius, meshWidth / 2.0, meshWidth / 2.0);
  
  // to make a more robust test, we would not use knowledge of the way edges and vertices are ordered here...
  typedef pair<int,int> Edge;
  Edge edge0 = make_pair(0,2); // bottom
  Edge edge1 = make_pair(2,3); // right
  Edge edge2 = make_pair(3,1); // top
  Edge edge3 = make_pair(1,0); // left
  
  map< Edge, ParametricCurvePtr > edgeToCurveMap;

  // for now, instead of going for a full circle, just replace one edge
  edgeToCurveMap[edge0] = ParametricCurve::subCurve(circle,  5.0/8.0, 7.0/8.0);
//  edgeToCurveMap[edge1] = ParametricCurve::subCurve(circle, -1.0/8.0, 1.0/8.0); // pretty sure this will work!
//  edgeToCurveMap[edge2] = ParametricCurve::subCurve(circle,  1.0/8.0, 3.0/8.0);
//  edgeToCurveMap[edge3] = ParametricCurve::subCurve(circle,  3.0/8.0, 5.0/8.0);
  
  H1Order = 1;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  mesh->setEdgeToCurveMap(edgeToCurveMap);
  
  double straightEdgePerimeter = meshWidth * 3.0;
  double arcLength = (PI * 2.0 * radius) / 4.0;
  
  double truePerimeter = arcLength + straightEdgePerimeter;
  
  perimeter = oneOnBoundary->integrate(mesh);
  double previousError = 1000;
  
  numPRefinements = 19;
  for (int i=1; i<=numPRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    cout << "perimeter: " << perimeter << endl;
    double impliedPi = (perimeter - straightEdgePerimeter) * 2.0 / radius;
    cout << "For p=" << i << ", implied value of pi: " << impliedPi << endl;
    double error = abs(truePerimeter - perimeter);
    if ((error >= previousError) && (error > tol)) { // non-convergence
      success = false;
      cout << "testEdgeLength: Error with H1Order = " << i << " is greater than with H1Order = " << i - 1 << endl;
      cout << "Current error = " << error << "; previous = " << previousError << endl;
    }
    ostringstream filePath;
    filePath << "/tmp/circularMesh_p" << i << ".dat";
    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    previousError = error;
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
  }
  
//  // now, do much the same thing, except with h-refinements:
//  H1Order = 2;
//  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
//  mesh->setEdgeToCurveMap(edgeToCurveMap);
//  previousError = 1000;
//  numHRefinements = 5;
//  for (int i=0; i<=numHRefinements; i++) {
//    perimeter = oneOnBoundary->integrate(mesh);
//    cout << "perimeter: " << perimeter << endl;
//    double error = abs(truePerimeter - perimeter);
//    if ((error >= previousError) && (error > tol)) { // non-convergence
//      success = false;
//      cout << "testEdgeLength: Error for h-refinement " << i << " is greater than for h-refinement " << i - 1 << endl;
//      cout << "Current error = " << error << "; previous = " << previousError << endl;
//    }
//    ostringstream filePath;
//    filePath << "/tmp/circularMesh_h" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
//    filePath.str("");
//    filePath << "/tmp/circularMesh_h" << i << "_straight_lines.dat";
//    GnuPlotUtil::writeExactMeshSkeleton(filePath.str(), mesh, 2);
//    previousError = error;
//    
//    // h-refine
//    if (i<numHRefinements) {
//      mesh->hRefine(mesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
//    }
//  }
//  
  return success;
}

bool CurvilinearMeshTests::testStraightEdgeMesh() {
  bool success = true;
  
  // to begin, a very simple test: do we compute the correct area for a square?
  FunctionPtr one = Function::constant(1.0);
  
  double width = 1.0;
  double height = 1.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  double trueArea = width * height;
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  

    // make a single-element mesh:
  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
  double approximateArea = one->integrate(mesh);
  
  // sanity check on the test: regular mesh gets the area right:
  double tol = 1e-14;
  double err = abs(trueArea-approximateArea);
  if (err > tol) {
    success = false;
    cout << "Error: even regular mesh (no curves set) gets the area wrong.\n";
  }
  
//  GnuPlotUtil::writeExactMeshSkeleton("/tmp/unitMesh.dat", mesh, 2);
  
  for (int i=0; i<4; i++) {
    H1Order = i+1;
    mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
  
    // now, set curves for each edge:
    map< pair<int, int>, ParametricCurvePtr > edgeToCurveMap;
    
    int cellID = 0; // the only cell
    vector< ParametricCurvePtr > lines = mesh->parametricEdgesForCell(cellID);
    vector< int > vertices = mesh->vertexIndicesForCell(cellID);
    
    for (int i=0; i<vertices.size(); i++) {
      int vertex = vertices[i];
      int nextVertex = vertices[(i+1) % vertices.size()];
      pair< int, int > edge = make_pair(vertex,nextVertex);
      edgeToCurveMap[edge] = lines[i];
    }
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    // now repeat with our straight-edge curves:
    approximateArea = one->integrate(mesh);
    tol = 1e-14;
    err = abs(trueArea-approximateArea);
    if (err > tol) {
      success = false;
      cout << "Error: mesh with straight-edge 'curves' and H1Order " << H1Order;
      cout << " has area " << approximateArea << "; should be " << trueArea << "." << endl;
    }
    
//    ostringstream filePath;
//    filePath << "/tmp/unitMesh" << H1Order << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
  }

  return success;
}

std::string CurvilinearMeshTests::testSuiteName() {
  return "CurvilinearMeshTests";
}