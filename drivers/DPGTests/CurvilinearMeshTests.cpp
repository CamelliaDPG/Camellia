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
#include "BasisSumFunction.h"

#include "GnuPlotUtil.h"

#include "ParametricSurface.h"
#include "StokesFormulation.h"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

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
  if (testH1Projection()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testAutomaticStraightEdgesMatchVertices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  // following test disabled because we never got around to testing anything
  // (mostly just used as a driver to output data to file in course of debugging)
//  setup();
//  if (testPointsRemainInsideElement()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();
  
  setup();
  if (testCylinderMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testTransformationJacobian()) {
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
}

void lineAcrossQuadRefCell(FieldContainer<double> &refPoints, int numPoints, bool horizontal) {
  int pointsInLine = numPoints;
  refPoints.resize(pointsInLine,2);
  for (int i=0; i<pointsInLine; i++) {
    if (horizontal) {
      refPoints(i,0) = 2.0 * (i) / (pointsInLine-1) - 1.0;
      refPoints(i,1) = 0;
    } else {
      refPoints(i,0) = 0;
      refPoints(i,1) = 2.0 * (i) / (pointsInLine-1) - 1.0;
    }
  }
}

bool CurvilinearMeshTests::testCylinderMesh() {
  bool success = true;
  
  int rank = Teuchos::GlobalMPISession::getRank();
  
  FunctionPtr one = Function::constant(1.0);
  
  double width = 30.0;
  double height = 30.0;
  double r = 1.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  double trueArea = width * height - PI * r * r;
  
  int H1Order = 1;
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  MeshPtr mesh = MeshFactory::hemkerMesh(width, height, r, bf, H1Order, pToAdd);
  
  //  GnuPlotUtil::writeExactMeshSkeleton("/tmp/cylinderFlowExactMesh.dat", mesh, 10);
  
  //  double approximateArea = one->integrate(mesh);
  
  cout << setprecision(15);
  //  cout << "Exact area:" << trueArea << endl;
  //  cout << "Approximate area on straight-line mesh: " << approximateArea << endl;
  //
  
  GlobalIndexType numCells = mesh->numElements();
  set<GlobalIndexType> allCells;
  for (GlobalIndexType i=0; i<numCells; i++) {
    allCells.insert(i);
  }
  
  double tol = 1e-10;
  cout << setprecision(15);
  // test p-convergence of mesh area
  double previousError = 1000;
  int numPRefinements = 3;
  for (int i=1; i<=numPRefinements; i++) {
    double approximateArea = one->integrate(mesh,5);
    double impliedPi = (width * height - approximateArea) / (r*r);
    if (rank==0) {
      cout << "For k=" << i << ", implied value of pi: " << impliedPi;
      cout << " (error " << abs(PI-impliedPi) << ")\n";
    }
    //    cout << "Area with H1Order " << H1Order << ": " << approximateArea << endl;
    double error = abs(trueArea - approximateArea);
    if ((error > previousError) && (error > tol)) { // non-convergence
      success = false;
      if (rank==0) {
        cout << "Error with H1Order = " << i << " is greater than with H1Order = " << i - 1 << endl;
        cout << "Current error = " << error << "; previous = " << previousError << endl;
      }
    }
//    ostringstream filePath;
//    filePath << "/tmp/cylinderFlowMesh" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    previousError = error;
    
    // DEBUGGING code
//    if (true) { //((i==3) || (i==4)) {
//      // here, we're getting a negative area for cellID 6
//      // to start, let's visualize the cubature points
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 6);
//      double area = basisCache->getCellMeasures()[0];
//      cout << "area of cellID 6 is " << area << endl;
////      FieldContainer<double> cubaturePoints = basisCache->getPhysicalCubaturePoints();
////      GnuPlotUtil::writeXYPoints("/tmp/cellID6_cubPoints.dat", cubaturePoints);
//      // try drawing a vertical line in the reference element
//      int pointsInLine = 15;
//      FieldContainer<double> refPoints;
//      lineAcrossQuadRefCell(refPoints, pointsInLine, false);
//      basisCache->setRefCellPoints(refPoints);
//      GnuPlotUtil::writeXYPoints("/tmp/cellID6_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
//      // now, a horizontal line
//      lineAcrossQuadRefCell(refPoints, pointsInLine, true);
//      basisCache->setRefCellPoints(refPoints);
//      GnuPlotUtil::writeXYPoints("/tmp/cellID6_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
//    }
    
    
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(allCells);
    }
  }
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::hemkerMesh(width, height, r, bf, H1Order, pToAdd);
  previousError = 1000;
  int numHRefinements = 3;
  for (int i=0; i<=numHRefinements; i++) {
    double approximateArea = one->integrate(mesh);
    double impliedPi = (width * height - approximateArea) / (r*r);
    if (rank==0) {
      cout << "For h-refinement " << i << ", implied value of pi: " << impliedPi;
      cout << " (error " << abs(PI-impliedPi) << ")\n";
    }
    //    cout << "Area with H1Order " << H1Order << ": " << approximateArea << endl;
    double error = abs(trueArea - approximateArea);
    if ((error > previousError) && (error > tol)) { // non-convergence
      success = false;
      if (rank==0) {
        cout << "Error for h-refinement " << i << " is greater than for h-refinement " << i - 1 << endl;
        cout << "Current error = " << error << "; previous = " << previousError << endl;
      }
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

bool CurvilinearMeshTests::testAutomaticStraightEdgesMatchVertices() {
  bool success = true;
  FunctionPtr t = Teuchos::rcp( new Xn(1) );
  
  // for the moment, try just ref to affine (parallelogram) map: 
  FunctionPtr x = 2 * t;
  FunctionPtr y = Function::constant(-1);
  
  // (this is the real one we want to try)
//  FunctionPtr x = 2 * t - 1;
//  FunctionPtr y = x - 1;
  
  // for the moment, try just ref to ref map  
  ParametricCurvePtr bottomCurve = ParametricCurve::curve(x,y);

  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  
  // for the moment, try just ref to affine (parallelogram) map:
  FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) =  0;
  physicalCellNodes(0,0,1) = -1;
  
  physicalCellNodes(0,1,0) =  2;
  physicalCellNodes(0,1,1) = -1;
  
  physicalCellNodes(0,2,0) = 3;
  physicalCellNodes(0,2,1) = 1;
  
  physicalCellNodes(0,3,0) = 1;
  physicalCellNodes(0,3,1) = 1;
  
  // real one commented out below
//  physicalCellNodes(0,0,0) = -1;
//  physicalCellNodes(0,0,1) = -2;
//  
//  physicalCellNodes(0,1,0) = 1;
//  physicalCellNodes(0,1,1) = 0;
//  
//  physicalCellNodes(0,2,0) = 1;
//  physicalCellNodes(0,2,1) = 1;
//  
//  physicalCellNodes(0,3,0) = -1;
//  physicalCellNodes(0,3,1) = 1;
  
  int quadraticOrder = 2;
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf(); // just to specify something
  MeshPtr quadMesh = MeshFactory::quadMesh(bf, quadraticOrder, physicalCellNodes, 0);
  
  int cellID = 0;
  vector<unsigned> vertices = quadMesh->vertexIndicesForCell(cellID);
  pair<int,int> edge = make_pair(vertices[0],vertices[1]);
  map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
  edgeToCurveMap[edge] = bottomCurve;
  
  quadMesh->setEdgeToCurveMap(edgeToCurveMap);
  
  // now, get back the edges for the cell, and check that they interpolate the vertices:
  vector< ParametricCurvePtr > edgeCurves = quadMesh->parametricEdgesForCell(cellID);
  double tol=1e-14;
  for (int vertex=0; vertex<numVertices; vertex++) {
    ParametricCurvePtr curve = edgeCurves[vertex];
    double x0, y0;
    double x1, y1;
    curve->value(0, x0, y0);
    curve->value(1, x1, y1);
    double x0_expected = physicalCellNodes(0,vertex,0);
    double y0_expected = physicalCellNodes(0,vertex,1);
    double x1_expected = physicalCellNodes(0,(vertex+1)%4,0);
    double y1_expected = physicalCellNodes(0,(vertex+1)%4,1);
    if ((abs(x0_expected - x0) > tol) 
        || ((x1_expected - x1) > tol)
        || (abs(y0_expected - y0) > tol) 
        || ((y1_expected - y1) > tol)) {
      success = false;
      cout << "testAutomaticStraightEdgesMatchVertices() failure: edge " << vertex << " does not interpolate vertices.\n";
    }
  }
  
  FunctionPtr transformationFunction = quadMesh->getTransformationFunction();
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(quadMesh, cellID);
  
  // turn off the basisCache transformation function, so we don't transform twice
  basisCache->setTransformationFunction(Function::null());
  FieldContainer<double> values(numCells, numVertices, spaceDim);
  FieldContainer<double> refCellVertices(numVertices,spaceDim);
  
  refCellVertices(0,0) = -1;
  refCellVertices(0,1) = -1;
  refCellVertices(1,0) =  1;
  refCellVertices(1,1) = -1;
  refCellVertices(2,0) =  1;
  refCellVertices(2,1) =  1;
  refCellVertices(3,0) = -1;
  refCellVertices(3,1) =  1;
  
  basisCache->setRefCellPoints(refCellVertices);
  
  transformationFunction->values(values, basisCache);
  
//  { // let's take a quick look to make sure we haven't screwed anything up majorly:
//    ostringstream filePath;
//    filePath << "/tmp/straightEdgeMesh.dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), quadMesh);
//  }
  
  
  double maxDiff = 0;
  if (! fcsAgree(values, physicalCellNodes, tol, maxDiff)) {
    success = false;
    cout << "transformation function for straight-edge mesh does not map vertices to themselves; max difference " << maxDiff << endl;
    cout << "vertices:\n" << physicalCellNodes;
    cout << "transformed vertices:\n" << values;
  }
  
  return success;
}

bool CurvilinearMeshTests::testEdgeLength() {
  bool success = true;
  
  int rank = Teuchos::GlobalMPISession::getRank();
  
  cout << setprecision(15);
  double tol = 1e-14;
  
  // to begin, a very simple test: do we compute the correct perimeter for a square?
  FunctionPtr oneOnBoundary = Function::meshBoundaryCharacteristic();
  
  double radius = 1.0;
  
  double meshWidth = radius * sqrt(2.0);
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  int H1Order = 1;
  
  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  
  
  // first test, before we get into the circular stuff: define a sloped edge
  // whose exact integral we know (and which is exactly representable by our geometry)
  {
    FunctionPtr t = Teuchos::rcp( new Xn(1) );
//    FunctionPtr x = 2 * t - 1;
//    FunctionPtr y = x - 1;
    FunctionPtr x = 2 * t;
    FunctionPtr y = Function::constant(-1);
    
    ParametricCurvePtr bottomCurve = ParametricCurve::curve(x,y);
    
//    FieldContainer<double> physicalCellNodes(1,4,2); // (C,P,D)
//    physicalCellNodes(0,0,0) = -1;
//    physicalCellNodes(0,0,1) = -2;
//    
//    physicalCellNodes(0,1,0) = 1;
//    physicalCellNodes(0,1,1) = 0;
//    
//    physicalCellNodes(0,2,0) = 1;
//    physicalCellNodes(0,2,1) = 1;
//    
//    physicalCellNodes(0,3,0) = -1;
//    physicalCellNodes(0,3,1) = 1;
    
    FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
    physicalCellNodes(0,0,0) =  0;
    physicalCellNodes(0,0,1) = -1;
    
    physicalCellNodes(0,1,0) =  2;
    physicalCellNodes(0,1,1) = -1;
    
    physicalCellNodes(0,2,0) = 3;
    physicalCellNodes(0,2,1) = 1;
    
    physicalCellNodes(0,3,0) = 1;
    physicalCellNodes(0,3,1) = 1;
    
    int quadraticOrder = 2;
    MeshPtr quadMesh = MeshFactory::quadMesh(bf, quadraticOrder, physicalCellNodes, 0);
    
    int cellID = 0;
    vector<unsigned> vertices = quadMesh->vertexIndicesForCell(cellID);
    pair<GlobalIndexType,GlobalIndexType> edge = make_pair(vertices[0],vertices[1]);
    map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    edgeToCurveMap[edge] = bottomCurve;
    
    quadMesh->setEdgeToCurveMap(edgeToCurveMap);

//    { // let's take a quick look to make sure we haven't screwed anything up majorly:
//      ostringstream filePath;
//      filePath << "/tmp/skewedQuadMesh.dat";
//      GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), quadMesh);
//    }
    
    // the length of the sloped edge is 2 sqrt (2)
    // and the other edges have total length of 5:
    double expectedPerimeter = 4 + 2 * sqrt(5);
    
    // since our map from straight edges is the identity,
    // the expected jacobian function everywhere, including along the side, is
    // [  1  0 ]
    // [  0  1 ]
    FunctionPtr expectedTransformation, expectedJacobian;
    {
      expectedTransformation = Function::vectorize(Function::xn(1), Function::yn(1));
      expectedJacobian = expectedTransformation->grad(spaceDim);
    }
    
    for (int hRefinement=0; hRefinement<5; hRefinement++) {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(quadMesh, cellID);
      BasisCachePtr sideCache = basisCache->getSideBasisCache(0);
      
      // need sideCache not to retransform things so we can test the transformationFxn itself:
      // TODO: think through this carefully, and/or try with basisCache to confirm that the interior works
      //       the way we're asking the edge to.  We have reason to think the interior is working...
      
      // I'm unclear on whether we should set basisCache's transformation to null:
      basisCache->setTransformationFunction(Function::null());
      sideCache->setTransformationFunction(Function::null());
      
      int numCells = 1;
      int numPoints = sideCache->getPhysicalCubaturePoints().dimension(1);
      int spaceDim = 2;
      FieldContainer<double> expectedJacobianValues(numCells,numPoints,spaceDim,spaceDim);
      expectedJacobian->values(expectedJacobianValues,sideCache);
      FieldContainer<double> jacobianValues(numCells,numPoints,spaceDim,spaceDim);
      FunctionPtr transformationFxn = quadMesh->getTransformationFunction();
      FunctionPtr transformationJacobian = transformationFxn->grad();
      transformationJacobian->values(jacobianValues,sideCache);
      
      if (! expectedTransformation->equals(transformationFxn, basisCache) ) { // pass
        success = false;
        cout << "expectedTransformation and transformationFxn differ on interior of the element.\n";
        reportFunctionValueDifferences(expectedTransformation, transformationFxn, basisCache, tol);
      }
      
      if (! expectedJacobian->equals(transformationJacobian, basisCache) ) { // fail
        success = false;
        cout << "expectedJacobian and transformationJacobian differ on interior of the element.\n";
        reportFunctionValueDifferences(expectedJacobian, transformationJacobian, basisCache, tol);
      }
      
      double maxDiff = 0;
      
      if (! expectedTransformation->equals(transformationFxn, sideCache)) { // pass
        success = false;
        cout << "testEdgeLength(): expected values don't match transformation function values along parametrically specified edge.\n";
        reportFunctionValueDifferences(expectedTransformation, transformationFxn, sideCache, tol);
      }
      
      if (! fcsAgree(expectedJacobianValues, jacobianValues, tol, maxDiff)) { // fail
        success = false;
        cout << "testEdgeLength(): expected jacobian values don't match transformation function's gradient values along parametrically specified edge.\n";
        reportFunctionValueDifferences(expectedJacobian, transformationJacobian, sideCache, tol);
      }
      
      double perimeter = oneOnBoundary->integrate(quadMesh);
      double err = abs( perimeter - expectedPerimeter );
      if (err > tol) {
        cout << "For h-refinement " << hRefinement << ", edge integral of y=x-1 does not match expected.\n";
        cout << "err = " << err << endl;
        cout << "expected perimeter = " << expectedPerimeter << "; actual = " << perimeter << endl;
      }      
      quadMesh->hRefine(quadMesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
    }
  }
  
  
  // make a single-element mesh:
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  double perimeter = oneOnBoundary->integrate(mesh);
  

  double expectedPerimeter = 4*(meshWidth);
  double err = abs(perimeter - expectedPerimeter);
  if (err > tol) {
    cout << "Problem with test: on square mesh, perimeter should be " << expectedPerimeter;
    cout << " but is " << perimeter << endl;
    success = false;
  }
  
  map< Edge, ParametricCurvePtr > edgeToCurveMap;
  
  int cellID = 0; // the only cell
  vector< ParametricCurvePtr > lines = mesh->parametricEdgesForCell(cellID);
  vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
  
  for (int i=0; i<vertices.size(); i++) {
    int vertex = vertices[i];
    int nextVertex = vertices[(i+1) % vertices.size()];
    pair< int, int > edge = make_pair(vertex,nextVertex);
    edgeToCurveMap[edge] = lines[i];
  }
  
  mesh->setEdgeToCurveMap(edgeToCurveMap);
  
  int numPRefinements = 5;
  for (int i=1; i<=numPRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    //    cout << "perimeter: " << perimeter << endl;
    double error = abs(expectedPerimeter - perimeter);
    if (error > tol) {
      success = false;
      cout << "testEdgeLength: On square mesh (straight 'curves'), error with H1Order = " << i << " exceeds tol " << tol << endl;
      cout << "Error = " << error << endl;
    }
    
//    ostringstream filePath;
//    filePath << "/tmp/squareMesh_p" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
  }
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  int numHRefinements = 3;
  for (int i=0; i<=numHRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    //    cout << "perimeter: " << perimeter << endl;
    double error = abs(expectedPerimeter - perimeter);
    if (error > tol) {
      success = false;
      cout << "testEdgeLength: On square mesh (straight 'curves'), error for h-refinement " << i << " exceeds tol " << tol << endl;
      cout << "Error = " << error << endl;
    }
    // h-refine
    if (i<numHRefinements) {
      mesh->hRefine(mesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
    }
  }
  
  // now for the real test: swap out the edges for circular arcs.
  ParametricCurvePtr circle = ParametricCurve::circle(radius, meshWidth / 2.0, meshWidth / 2.0);
  
  // to make a more robust test, we would not use knowledge of the way edges and vertices are ordered here...
  typedef pair<int,int> Edge;
  Edge edge0 = make_pair(0,2); // bottom
  Edge edge1 = make_pair(2,3); // right
  Edge edge2 = make_pair(3,1); // top
  Edge edge3 = make_pair(1,0); // left
  
  // the full circle, split into 4 edges:
  edgeToCurveMap[edge0] = ParametricCurve::subCurve(circle,  5.0/8.0, 7.0/8.0);
  edgeToCurveMap[edge1] = ParametricCurve::subCurve(circle, -1.0/8.0, 1.0/8.0);
  edgeToCurveMap[edge2] = ParametricCurve::subCurve(circle,  1.0/8.0, 3.0/8.0);
  edgeToCurveMap[edge3] = ParametricCurve::subCurve(circle,  3.0/8.0, 5.0/8.0);
  
  H1Order = 1;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  mesh->setEdgeToCurveMap(edgeToCurveMap);
  
  double straightEdgePerimeter = 0; // = meshWidth * 3.0;
  double arcLength = (PI * 2.0 * radius); // (PI * 2.0 * radius) / 4.0;
  
  double truePerimeter = arcLength + straightEdgePerimeter;
  
  perimeter = oneOnBoundary->integrate(mesh);
  double previousError = 1000;
  
  numPRefinements = 3;
  for (int i=1; i<=numPRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    //    cout << "perimeter: " << perimeter << endl;
    double impliedPi = (perimeter - straightEdgePerimeter) / (2 * radius);
    if (rank==0) {
      cout << "For p=" << i << ", implied value of pi: " << impliedPi << endl;
    }
    double error = abs(truePerimeter - perimeter);
    if ((error >= previousError) && (error > tol)) { // non-convergence
      success = false;
      if (rank==0) {
        cout << "testEdgeLength: Error with H1Order = " << i << " is greater than with H1Order = " << i - 1 << endl;
        cout << "Current error = " << error << "; previous = " << previousError << endl;
      }
    }
//    ostringstream filePath;
//    filePath << "/tmp/circularMesh_p" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    previousError = error;
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
  }
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth, 1, 1);
  mesh->setEdgeToCurveMap(edgeToCurveMap);
  previousError = 1000;
  numHRefinements = 3;
  for (int i=0; i<=numHRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
    
    //    cout << "perimeter: " << perimeter << endl;
    double impliedPi = (perimeter - straightEdgePerimeter) / (2 * radius);
    if (rank==0) {
      cout << "For h-refinement " << i << ", implied value of pi: " << impliedPi << endl;
    }
    
    double error = abs(truePerimeter - perimeter);
    if ((error >= previousError) && (error > tol)) { // non-convergence
      success = false;
      if (rank==0) {
        cout << "testEdgeLength: Error for h-refinement " << i << " is greater than for h-refinement " << i - 1 << endl;
        cout << "Current error = " << error << "; previous = " << previousError << endl;
      }
    }
//    ostringstream filePath;
//    filePath << "/tmp/circularMesh_h" << i << ".dat";
//    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
//    filePath.str("");
//    filePath << "/tmp/circularMesh_h" << i << "_straight_lines.dat";
//    GnuPlotUtil::writeExactMeshSkeleton(filePath.str(), mesh, 2);
    previousError = error;
    
    // h-refine
    if (i<numHRefinements) {
      mesh->hRefine(mesh->getActiveCellIDs(),RefinementPattern::regularRefinementPatternQuad());
    }
  }
  
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
    map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    
    int cellID = 0; // the only cell
    vector< ParametricCurvePtr > lines = mesh->parametricEdgesForCell(cellID);
    vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
    
    for (int i=0; i<vertices.size(); i++) {
      int vertex = vertices[i];
      int nextVertex = vertices[(i+1) % vertices.size()];
      pair< GlobalIndexType, GlobalIndexType > edge = make_pair(vertex,nextVertex);
      edgeToCurveMap[edge] = lines[i];
    }
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    // now repeat with our straight-edge curves:
    approximateArea = one->integrate(mesh);
    tol = 2e-14;
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


bool CurvilinearMeshTests::testH1Projection() {
  bool success = true;
  
  bool useL2 = false; // H1 semi-norm otherwise
  
  // this test sprawls a bit.  It could very reasonably be broken apart into several
  // others; tests of projection, BasisFactory::basisFactory()->sideFieldIndices, etc.
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  BasisPtr quadraticScalarBasis = BasisFactory::basisFactory()->getBasis(2, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  BasisPtr quadraticVectorBasis = BasisFactory::basisFactory()->getBasis(2, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
  
  set<int> scalarEdgeNodes = BasisFactory::basisFactory()->sideFieldIndices(quadraticScalarBasis);
  set<int> vectorEdgeNodes = BasisFactory::basisFactory()->sideFieldIndices(quadraticVectorBasis);
  
  // how many non-edge nodes are there?
  int numMiddleNodesScalar = quadraticScalarBasis->getCardinality() - scalarEdgeNodes.size();
  int numMiddleNodesVector = quadraticVectorBasis->getCardinality() - vectorEdgeNodes.size();
  
  if (numMiddleNodesScalar != 1) {
    cout << "# middle nodes for quadratic scalar basis is " << numMiddleNodesScalar;
    cout << "; expected 1.\n";
    success = false;
    
    cout << "scalarEdgeNodes: ";
    for (set<int>::iterator nodeIt = scalarEdgeNodes.begin(); nodeIt != scalarEdgeNodes.end(); nodeIt++) {
      cout << *nodeIt << ", ";
    }
    cout << endl;
  }
  if (numMiddleNodesVector != 2) {
    cout << "# middle nodes for quadratic vector basis is " << numMiddleNodesVector;
    cout << "; expected 2.\n";
    success = false;
    cout << "vectorEdgeNodes: ";
    for (set<int>::iterator nodeIt = vectorEdgeNodes.begin(); nodeIt != vectorEdgeNodes.end(); nodeIt++) {
      cout << *nodeIt << ", ";
    }
    cout << endl;
  }
  int middleNodeScalar;
  vector<int> middleNodeVector;
  for (int i=0; i<quadraticScalarBasis->getCardinality(); i++) {
    if (scalarEdgeNodes.find(i) == scalarEdgeNodes.end()) {
      middleNodeScalar = i;
    }
  }
  for (int i=0; i<quadraticVectorBasis->getCardinality(); i++) {
    if (vectorEdgeNodes.find(i) == vectorEdgeNodes.end()) {
      middleNodeVector.push_back( i );
    }
  }
  
  FieldContainer<double> dofCoords(quadraticScalarBasis->getCardinality(),2);
  IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >* intrepidBasisWrapper = dynamic_cast< IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >* >(quadraticScalarBasis.get());
  if (!intrepidBasisWrapper) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "compBasis does not appear to be an instance of IntrepidBasisWrapper");
  }
  Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >* intrepidBasis = dynamic_cast< Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >* >(intrepidBasisWrapper->intrepidBasis().get());
  if (!intrepidBasis) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "IntrepidBasisWrapper does not appear to wrap Basis_HGRAD_QUAD_Cn_FEM.");
  }
  intrepidBasis->getDofCoords(dofCoords);
  
  
//  ((Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >*) quadraticScalarBasis.get())->getDofCoords(dofCoords);
  
  // for quadratic basis, we expect the middle node to be at (0,0)
  if ((dofCoords(middleNodeScalar,0) != 0.0) || (dofCoords(middleNodeScalar,1) != 0.0)) {
    cout << "middle node dof coord is (" << dofCoords(middleNodeScalar,0);
    cout << "," << dofCoords(middleNodeScalar,1) << " not (0,0) as expected.\n";
    success = false;
  }
  
  //  cout << "middle node index for quadratic scalar basis:" << middleNodeScalar << endl;
  
  VarFactory varFactory;
  VarPtr v_vector = varFactory.testVar("v", VECTOR_HGRAD);
  IPPtr ip_vector = Teuchos::rcp( new IP );
  if (useL2) {
    ip_vector->addTerm(v_vector);
  } else {
    ip_vector->addTerm(v_vector->grad());
  }
  
  VarPtr v_scalar = varFactory.testVar("v_s", HGRAD);
  IPPtr ip_scalar = Teuchos::rcp( new IP) ;
  if (useL2) {
    ip_scalar->addTerm(v_scalar);
  } else {
    ip_scalar->addTerm(v_scalar->grad());
  }
  
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  
  FunctionPtr fxnScalar = x*x;
  FunctionPtr fxnVector = Function::vectorize(fxnScalar,fxnScalar);
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  MeshPtr mesh = MeshFactory::quadMesh(bf, 2, pToAdd, 1.0, 1.0);
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0, true);
  
  
  double middleNode_norm;
  double rhs_ip;
  {
    // check that the middle node function is what we expect
    FieldContainer<double> scalarMiddleCoefficients(quadraticScalarBasis->getCardinality());
    scalarMiddleCoefficients[middleNodeScalar] = 1.0;
    FunctionPtr middleNodeFunction = NewBasisSumFunction::basisSumFunction(quadraticScalarBasis, scalarMiddleCoefficients);
    FunctionPtr middleNodeFunction_expected = 4 * x * (1-x) * 4 * y * (1-y);
    // per Wolfram Alpha:
    // on unit quad, the L2 norm should be 64/225
    // the H^1 semi-norm should be 256/45
    if (! middleNodeFunction->equals(middleNodeFunction_expected, basisCache)) {
      cout << "middle node function differs from expected.\n";
      success = false;
    }
    if (useL2) {
      middleNode_norm = (middleNodeFunction_expected * middleNodeFunction_expected)->integrate(basisCache);
      rhs_ip = (middleNodeFunction_expected * fxnScalar)->integrate(basisCache);
      // for fxnScalar = x^2, L2 rhs should be ___
    } else {
      middleNode_norm = (middleNodeFunction_expected->dx() * middleNodeFunction_expected->dx()
                         + middleNodeFunction_expected->dy() * middleNodeFunction_expected->dy()
                         )->integrate(basisCache);
      rhs_ip = (middleNodeFunction_expected->dx() * fxnScalar->dx()
                + middleNodeFunction_expected->dy() * fxnScalar->dy()
                )->integrate(basisCache);
      // for fxnScalar = x^2, H1 semi-norm rhs should be -8/9
    }
  }
  
  //  cout << "Middle node norm squared: " << middleNode_norm << endl;
  //  cout << "rhs for middle node: " << rhs_ip << endl;
  
  FieldContainer<double> scalarCoefficients;
  FieldContainer<double> vectorCoefficients;
  Projector::projectFunctionOntoBasis(scalarCoefficients, fxnScalar, quadraticScalarBasis, basisCache, ip_scalar, v_scalar, scalarEdgeNodes);
  Projector::projectFunctionOntoBasis(vectorCoefficients, fxnVector, quadraticVectorBasis, basisCache, ip_vector, v_vector, vectorEdgeNodes);
  
  FieldContainer<double> scalarCoefficients_expected(quadraticScalarBasis->getCardinality());
  scalarCoefficients_expected(middleNodeScalar) = rhs_ip / middleNode_norm;
  double maxDiff = 0;
  double tol = 1e-14;
  if (! fcsAgree(scalarCoefficients_expected, scalarCoefficients, tol, maxDiff)) {
    cout << "projectFunctionOntoBasis doesn't match expected weights for scalar projection onto quadratic middle node.\n";
    cout << "expected:\n" << scalarCoefficients_expected;
    cout << "actual:\n" << scalarCoefficients;
    success = false;
  }
  
  // need to confirm analytically that the following test is reasonable, but I am pretty sure
  FieldContainer<double> vectorCoefficients_expected(quadraticVectorBasis->getCardinality());
  vectorCoefficients_expected(middleNodeVector[0]) = rhs_ip / middleNode_norm;
  vectorCoefficients_expected(middleNodeVector[1]) = rhs_ip / middleNode_norm;
  
  //  cout << "expected solution to projection problem at quadratic middle node:" << rhs_ip / middleNode_norm << endl;
  
  maxDiff = 0;
  if (! fcsAgree(vectorCoefficients_expected, vectorCoefficients, tol, maxDiff)) {
    cout << "projectFunctionOntoBasis doesn't match expected weights for vector projection onto quadratic middle nodes.\n";
    cout << "expected:\n" << vectorCoefficients_expected;
    cout << "actual:\n" << vectorCoefficients;
    success = false;
  }
  
  // TODO: work out a test that will compare projector of scalar versus vector basis and confirm that vector behaves as expected...
  
  double width = 1;
  double height = 1;
  
  // the transfinite interpolant in physical space should be just (x, y)
  
  FunctionPtr tfi = Function::vectorize(x, y);
  
  
  VarPtr v = varFactory.testVar("v", VECTOR_HGRAD);
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v);
  ip->addTerm(v->grad());
  
  for (int H1Order=1; H1Order<5; H1Order++) {
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
    int cellID = 0; // the only cell
    bool testVsTest = true;
    
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
    
    int cubatureEnrichment = mesh->getElement(cellID)->elementType()->testOrderPtr->maxBasisDegree();
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureEnrichment);
    
    FieldContainer<double> basisCoefficients;
    Projector::projectFunctionOntoBasis(basisCoefficients, tfi, basis, basisCache, ip, v);
    
    // flatten basisCoefficients (remove the numCells dimension, which is 1)
    basisCoefficients.resize(basisCoefficients.size());
    FunctionPtr projectedFunction = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients) );
    
    double tol=1e-13;
    if (! projectedFunction->equals(tfi, basisCache, tol) ) {
      success = false;
      cout << "For H1Order " << H1Order << ", ";
      cout << "H^1-projected function does not match original, even though original is in the space.\n";
      
      int numCells = 1;
      int numPoints = basisCache->getRefCellPoints().dimension(0);
      int spaceDim = 2;
      FieldContainer<double> values(numCells,numPoints,spaceDim);
      FieldContainer<double> expected_values(numCells,numPoints,spaceDim);
      tfi->values(expected_values, basisCache);
      projectedFunction->values(values, basisCache);
      reportFunctionValueDifferences(basisCache->getPhysicalCubaturePoints(), expected_values,
                                     values, tol);
    }
    
    VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<> *)basis.get(),false);
    
    // For H1Order > 1, we don't expect that the edge interpolant will match the TFI on the element interior; we expect that only on the edges.
    
    vector< ParametricCurvePtr > curves = mesh->parametricEdgesForCell(cellID);
    ParametricSurfacePtr exactSurface = ParametricSurface::transfiniteInterpolant(curves);
    int numEdges = curves.size();
    
    FieldContainer<double> edgeInterpolantCoefficients;
    ParametricSurface::basisWeightsForEdgeInterpolant(edgeInterpolantCoefficients, vectorBasis, mesh, cellID);
    edgeInterpolantCoefficients.resize(edgeInterpolantCoefficients.size());
    
    // check that the only nonzeros in edgeInterpolantCoefficients belong to edges
    set<int> edgeFieldIndices = BasisFactory::basisFactory()->sideFieldIndices(vectorBasis, true); // true: include the vertices
    for (int i=0; i<edgeInterpolantCoefficients.size(); i++) {
      if (edgeInterpolantCoefficients(i) != 0) {
        if (edgeFieldIndices.find(i) == edgeFieldIndices.end()) {
          // then we have a nonzero weight for a non-edge field index
          cout << "edgeInterpolationCoefficients(" << i << ") = " << edgeInterpolantCoefficients(i);
          cout << ", but " << i << " is not an edge field index.\n";
          success = false;
        }
      }
    }
    
    // check that the edgeFieldIndices all belong to basis functions that are non-zero on some edge
    for (set<int>::iterator edgeFieldIt = edgeFieldIndices.begin(); edgeFieldIt != edgeFieldIndices.end(); edgeFieldIt++) {
      int fieldIndex = *edgeFieldIt;
      FieldContainer<double> edgeBasisFunctionWeights(basis->getCardinality());
      edgeBasisFunctionWeights[fieldIndex] = 1.0;
      bool nonZeroSomewhere = false;
      
      FunctionPtr edgeBasisFunction = NewBasisSumFunction::basisSumFunction(basis, edgeBasisFunctionWeights);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      
      for (int sideIndex=0; sideIndex<numEdges; sideIndex++) {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideIndex);
        if (! edgeBasisFunction->equals(Function::zero(basisRank), sideCache, tol) ) {
          nonZeroSomewhere = true;
          break;
        }
      }
      
      if (! nonZeroSomewhere ) {
        success = false;
        cout << "Field index " << fieldIndex << " is supposed to be an edge field index, but is zero on all edges.\n";
      }
    }
    
    // check that all the non-edgeFieldIndices are zero on all edges
    for (int fieldIndex=0; fieldIndex < basis->getCardinality(); fieldIndex++) {
      if (edgeFieldIndices.find(fieldIndex) != edgeFieldIndices.end()) {
        // edge field index: skip
        continue;
      }
      FieldContainer<double> nonEdgeBasisFunctionWeights(basis->getCardinality());
      nonEdgeBasisFunctionWeights[fieldIndex] = 1.0;
      
      FunctionPtr nonEdgeBasisFunction = NewBasisSumFunction::basisSumFunction(basis, nonEdgeBasisFunctionWeights);
      int basisRank = BasisFactory::basisFactory()->getBasisRank(basis);
      
      for (int sideIndex=0; sideIndex<numEdges; sideIndex++) {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideIndex);
        if (! nonEdgeBasisFunction->equals(Function::zero(basisRank), sideCache, tol) ) {
          success = false;
          cout << "Field index " << fieldIndex << " is not supposed to be an edge field index, but is non-zero on edge " << sideIndex << ".\n";
        }
      }
    }
    
    FunctionPtr edgeFunction = Teuchos::rcp( new NewBasisSumFunction(basis, edgeInterpolantCoefficients) );
    //
    //    VarFactory vf;
    //    VarPtr v = vf.testVar("v", VECTOR_HGRAD);
    //    IPPtr ip = Teuchos::rcp(new IP);
    //    ip->addTerm(v->grad());
    //
    //    Projector::projectFunctionOntoBasis(…)
    
    for (int sideIndex=0; sideIndex<numEdges; sideIndex++) {
      BasisCachePtr sideCache = basisCache->getSideBasisCache(sideIndex);
      if (! edgeFunction->equals(tfi, sideCache, tol) ) {
        success = false;
        cout << "For H1Order " << H1Order << ", ";
        cout << "edge interpolation function does not match original, even though original is in the space.\n";
        
        int numCells = 1;
        int numPoints = sideCache->getRefCellPoints().dimension(0);
        int spaceDim = 2;
        FieldContainer<double> values(numCells,numPoints,spaceDim);
        FieldContainer<double> expected_values(numCells,numPoints,spaceDim);
        tfi->values(expected_values, sideCache);
        edgeFunction->values(values, sideCache);
        reportFunctionValueDifferences(sideCache->getPhysicalCubaturePoints(), expected_values,
                                       values, tol);
      }
      
      // now, it should be the case that the transfinite interpolant (exactSurface) is exactly the same as the edgeFunction
      // along the edges
      if (! edgeFunction->equals(exactSurface, sideCache, tol) ) {
        success = false;
        cout << "For H1Order " << H1Order << ", ";
        cout << "edge interpolation function does not match exactSurface, even though original is in the space.\n";
        
        int numCells = 1;
        int numPoints = sideCache->getRefCellPoints().dimension(0);
        int spaceDim = 2;
        FieldContainer<double> values(numCells,numPoints,spaceDim);
        FieldContainer<double> expected_values(numCells,numPoints,spaceDim);
        edgeFunction->values(expected_values, sideCache);
        exactSurface->values(values, sideCache);
        reportFunctionValueDifferences(sideCache->getPhysicalCubaturePoints(), expected_values,
                                       values, tol);
      }
    }
    
    ParametricSurface::basisWeightsForProjectedInterpolant(basisCoefficients, vectorBasis, mesh, cellID);
    
    // check that the basisCoefficients for the edge functions are the same as the edgeCoefficients above
    for (set<int>::iterator edgeFieldIt = edgeFieldIndices.begin(); edgeFieldIt != edgeFieldIndices.end(); edgeFieldIt++) {
      int fieldIndex = *edgeFieldIt;
      if (basisCoefficients(fieldIndex) != edgeInterpolantCoefficients(fieldIndex) ) {
        cout << "For field index " << fieldIndex << " (an edge field index), ";
        cout << "projection-based interpolant weight is " << basisCoefficients(fieldIndex);
        cout << ", but the edge projection weight is " << edgeInterpolantCoefficients(fieldIndex) << endl;
        success = false;
      }
    }
    
    // compute the expected basis weights:
    FieldContainer<double> expectedCoefficients;
    IPPtr H1 = Teuchos::rcp(new IP); // here we need the full H1, not just the semi-norm
    H1->addTerm(v_vector);
    H1->addTerm(v_vector->grad());
    Projector::projectFunctionOntoBasis(expectedCoefficients, tfi, basis, basisCache, H1, v);
    
    tol = 5e-14;
    maxDiff = 0;
    if (! fcsAgree(expectedCoefficients, basisCoefficients, tol, maxDiff)) {
      success = false;
      cout << "Expected coefficients do not match actual:\n";
      reportFCDifferences(expectedCoefficients, basisCoefficients, tol);
    }
    
    FieldContainer<double> tfiCoefficients;
    Projector::projectFunctionOntoBasis(tfiCoefficients, tfi, basis, basisCache, H1, v);
    
    FieldContainer<double> edgeCoefficients;
    Projector::projectFunctionOntoBasis(edgeCoefficients, edgeFunction, basis, basisCache, H1, v);
    
    FieldContainer<double> projectedDifferenceCoefficients;
    Projector::projectFunctionOntoBasis(projectedDifferenceCoefficients, tfi-edgeFunction, basis, basisCache, H1, v);
    
    //    cout << "tfiCoefficients:\n" << tfiCoefficients;
    //    cout << "edgeCoefficients:\n" << edgeCoefficients;
    //    cout << "projectedDifferenceCoefficients:\n" << projectedDifferenceCoefficients;
    
    //    cout << "edge interpolation coefficients:\n" << edgeInterpolantCoefficients;
    //    cout << "projected tfi coefficients:\n" << expectedCoefficients;
    //    cout << "projection-based interpolant coefficients:\n" << basisCoefficients;
    
    expectedCoefficients.resize(expectedCoefficients.size()); // flatten
    FunctionPtr expectedFunction = NewBasisSumFunction::basisSumFunction(basis, expectedCoefficients);
    if (! expectedFunction->equals(tfi, basisCache, tol) ) {
      cout << "For H1Order " << H1Order << ", ";
      cout << "Problem with test?? expected function does not match tfi.\n";
      success = false;
      int numCells = 1;
      int numPoints = basisCache->getRefCellPoints().dimension(0);
      int spaceDim = 2;
      FieldContainer<double> values(numCells,numPoints,spaceDim);
      FieldContainer<double> expected_values(numCells,numPoints,spaceDim);
      tfi->values(expected_values, basisCache);
      projectedFunction->values(values, basisCache);
      reportFunctionValueDifferences(basisCache->getPhysicalCubaturePoints(), expected_values,
                                     values, tol);
      
    }
    
    projectedFunction = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients) );
    
    if (! projectedFunction->equals(tfi, basisCache, tol) ) {
      success = false;
      cout << "For H1Order " << H1Order << ", ";
      cout << "projection-based interpolation function does not match original, even though original is in the space.\n";
      
      int numCells = 1;
      int numPoints = basisCache->getRefCellPoints().dimension(0);
      int spaceDim = 2;
      FieldContainer<double> values(numCells,numPoints,spaceDim);
      FieldContainer<double> expected_values(numCells,numPoints,spaceDim);
      tfi->values(expected_values, basisCache);
      projectedFunction->values(values, basisCache);
      reportFunctionValueDifferences(basisCache->getPhysicalCubaturePoints(), expected_values,
                                     values, tol);
    }
    
  }
  
  return success;
}

bool CurvilinearMeshTests::testPointsRemainInsideElement() {
  bool success = true;
  
  double width = 5.0;
  double height = 5.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  for (int H1Order=1; H1Order < 5; H1Order++) {
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
    
    ParametricCurvePtr halfCircleTop = ParametricCurve::circularArc(width/2, width/2, height, 0, PI);
    ParametricCurvePtr halfCircleBottom = ParametricCurve::circularArc(width/2, width/2, 0, PI, 0); // PI to 0: from left vertex to right
    
    GlobalIndexType cellID = 0;
    vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
    map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    edgeToCurveMap[make_pair(vertices[0], vertices[1])] = halfCircleBottom;
    edgeToCurveMap[make_pair(vertices[2], vertices[3])] = halfCircleTop;
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    GnuPlotUtil::writeExactMeshSkeleton("/tmp/halfCircles.dat", mesh, 15);
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int pointsInLine = 15;
    FieldContainer<double> refPoints;
    lineAcrossQuadRefCell(refPoints, pointsInLine, false);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/halfCircles_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
    // now, a horizontal line
    lineAcrossQuadRefCell(refPoints, pointsInLine, true);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/halfCircles_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
  }
  
  for (int H1Order=1; H1Order < 5; H1Order++) {
    FieldContainer<double> physicalCellNodes(1,4,2); // (C,P,D)
    physicalCellNodes(0,0,0) = 0;
    physicalCellNodes(0,0,1) = 0;
    
    physicalCellNodes(0,1,0) = 0.75*width;
    physicalCellNodes(0,1,1) = 0;
    
    physicalCellNodes(0,2,0) = width;
    physicalCellNodes(0,2,1) = height;
    
    physicalCellNodes(0,3,0) = 0;
    physicalCellNodes(0,3,1) = height;
    
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, physicalCellNodes, pToAdd);
    
    ParametricCurvePtr halfCircleTop = ParametricCurve::circularArc(width/2, width/2, height, 0, PI);
    
    int cellID = 0;
    vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
    map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    edgeToCurveMap[make_pair(vertices[2], vertices[3])] = halfCircleTop;
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    GnuPlotUtil::writeExactMeshSkeleton("/tmp/oneHalfCircle.dat", mesh, 15);
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int pointsInLine = 15;
    FieldContainer<double> refPoints;
    lineAcrossQuadRefCell(refPoints, pointsInLine, false);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/oneHalfCircle_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
    // now, a horizontal line
    lineAcrossQuadRefCell(refPoints, pointsInLine, true);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/oneHalfCircle_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
  }
  
  
  for (int H1Order=1; H1Order < 5; H1Order++) {
    FieldContainer<double> physicalCellNodes(1,4,2); // (C,P,D)
    
    physicalCellNodes(0,0,0) = 0;
    physicalCellNodes(0,0,1) = height;
    
    physicalCellNodes(0,1,0) = 0;
    physicalCellNodes(0,1,1) = 0;
    
    physicalCellNodes(0,2,0) = width;
    physicalCellNodes(0,2,1) = 0;
    
    physicalCellNodes(0,3,0) = width;
    physicalCellNodes(0,3,1) = width+height;
    
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, physicalCellNodes, pToAdd);
    
    ParametricCurvePtr quarterCircleTop = ParametricCurve::circularArc(width, 0, width+height, 2*PI, 3*PI/2);
    
    GlobalIndexType cellID = 0;
    vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
    map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    edgeToCurveMap[make_pair(vertices[3], vertices[0])] = quarterCircleTop;
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    GnuPlotUtil::writeExactMeshSkeleton("/tmp/quarterCircle.dat", mesh, 15);
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int pointsInLine = 15;
    FieldContainer<double> refPoints;
    lineAcrossQuadRefCell(refPoints, pointsInLine, false);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/quarterCircle_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
    // now, a horizontal line
    lineAcrossQuadRefCell(refPoints, pointsInLine, true);
    basisCache->setRefCellPoints(refPoints);
    GnuPlotUtil::writeXYPoints("/tmp/quarterCircle_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
  }
  
  {
    int start_H1Order = 1;
    double r = 2 * width / 3;
    
    FieldContainer<double> physicalCellNodes(1,4,2); // (C,P,D)
    physicalCellNodes(0,0,0) = 0;
    physicalCellNodes(0,0,1) = 0;
    
    physicalCellNodes(0,1,0) = 3 * r / 2;
    physicalCellNodes(0,1,1) = 0;
    
    physicalCellNodes(0,2,0) = r / sqrt(2);
    physicalCellNodes(0,2,1) = (2.5 - 1/sqrt(2)) * r;
    
    physicalCellNodes(0,3,0) = 0;
    physicalCellNodes(0,3,1) = 1.5 * r;
    
    MeshPtr mesh_pRefined = MeshFactory::quadMesh(bf, start_H1Order, physicalCellNodes, pToAdd);

    {
      int cellID = 0;
      vector< unsigned > vertices = mesh_pRefined->vertexIndicesForCell(cellID);
      map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
      ParametricCurvePtr arcTop = ParametricCurve::circularArc(r, 0, 2.5 * r, 7*PI/4, 3*PI/2);
      edgeToCurveMap[make_pair(vertices[2], vertices[3])] = arcTop;
      mesh_pRefined->setEdgeToCurveMap(edgeToCurveMap);
    }
    
    vector< ParametricCurvePtr > curves1, curves2, curves3;
    // curves1: manual construction
    // curves2: p-refinement
    // curves3: manually set curves after p-refinement
    
    for (int H1Order=1; H1Order < 5; H1Order++) {
      MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, physicalCellNodes, pToAdd);
      
      ParametricCurvePtr arcTop = ParametricCurve::circularArc(r, 0, 2.5 * r, 7*PI/4, 3*PI/2);
      
      GlobalIndexType cellID = 0;
      vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
      map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
      edgeToCurveMap[make_pair(vertices[2], vertices[3])] = arcTop;
      
      mesh->setEdgeToCurveMap(edgeToCurveMap);
      curves1 = mesh->parametricEdgesForCell(cellID);
      
      GnuPlotUtil::writeExactMeshSkeleton("/tmp/hemkerSegment.dat", mesh, 15);
      
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      int pointsInLine = 15;
      FieldContainer<double> refPoints;
      lineAcrossQuadRefCell(refPoints, pointsInLine, false);
      basisCache->setRefCellPoints(refPoints);
      GnuPlotUtil::writeXYPoints("/tmp/hemkerSegment_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
      // now, a horizontal line
      lineAcrossQuadRefCell(refPoints, pointsInLine, true);
      basisCache->setRefCellPoints(refPoints);
      GnuPlotUtil::writeXYPoints("/tmp/hemkerSegment_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
      
      {
        curves2 = mesh_pRefined->parametricEdgesForCell(cellID);
        mesh_pRefined->setEdgeToCurveMap(edgeToCurveMap);
        curves3 = mesh_pRefined->parametricEdgesForCell(cellID);

        // because we've messed with basisCache's refCellPoints, get a new basisCache
        basisCache = BasisCache::basisCacheForCell(mesh_pRefined, cellID);
        
        // now that we have the various curves defined, let's compare them
        for (int i=0; i<curves1.size(); i++) {
          if (! curves1[i]->equals(curves2[i], basisCache) ) {
            cout << "For H1Order " << H1Order << ", ";
            cout << "curves1 and curves2 differ in entry " << i << endl;
            success = false;
          }
          if (! curves2[i]->equals(curves3[i], basisCache) ) {
            cout << "For H1Order " << H1Order << ", ";
            cout << "curves2 and curves3 differ in entry " << i << endl;
            success = false;
          }
        }
        
        // stuff with the second mesh, which is p-refined (trying to tease out how the different paths differ)
        GnuPlotUtil::writeExactMeshSkeleton("/tmp/copyHemkerSegment.dat", mesh_pRefined, 15);
        
        basisCache = BasisCache::basisCacheForCell(mesh_pRefined, cellID);
        int pointsInLine = 15;
        FieldContainer<double> refPoints;
        lineAcrossQuadRefCell(refPoints, pointsInLine, false);
        basisCache->setRefCellPoints(refPoints);
        GnuPlotUtil::writeXYPoints("/tmp/copyHemkerSegment_vertical_line.dat", basisCache->getPhysicalCubaturePoints());
        // now, a horizontal line
        lineAcrossQuadRefCell(refPoints, pointsInLine, true);
        basisCache->setRefCellPoints(refPoints);
        GnuPlotUtil::writeXYPoints("/tmp/copyHemkerSegment_horizontal_line.dat", basisCache->getPhysicalCubaturePoints());
        
        vector<GlobalIndexType> cellIDs;
        cellIDs.push_back(cellID);
        mesh_pRefined->pRefine(cellIDs);
        
      }
    }
  }
  
  return success;
}

bool CurvilinearMeshTests::testTransformationJacobian() {
  bool success = true;
  
  double width = 4.0;
  double height = 3.0;
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();
  
  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  
  // make a single-element mesh:
  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
  
  double tol = 1e-13;
  
  for (int i=0; i<4; i++) {
    H1Order = i+1;
    mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
    int cellID = 0; // the only cell
    
    // add to cubature just as we'll need to do for the 'curvilinear' mesh
    int cubatureEnrichment = mesh->getElement(cellID)->elementType()->testOrderPtr->maxBasisDegree();
    
    // compute jacobians:
    bool testVsTest = false;
    BasisCachePtr standardMeshCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureEnrichment);
    
    // now, set curves for each edge:
    map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > edgeToCurveMap;
    
    vector< ParametricCurvePtr > lines = mesh->parametricEdgesForCell(cellID);
    vector< unsigned > vertices = mesh->vertexIndicesForCell(cellID);
    
    for (int i=0; i<vertices.size(); i++) {
      GlobalIndexType vertex = vertices[i];
      GlobalIndexType nextVertex = vertices[(i+1) % vertices.size()];
      pair< GlobalIndexType, GlobalIndexType > edge = make_pair(vertex,nextVertex);
      edgeToCurveMap[edge] = lines[i];
    }
    
    mesh->setEdgeToCurveMap(edgeToCurveMap);
    
    BasisCachePtr curvilinearMeshCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    int numSides = mesh->getElement(cellID)->numSides();
    for (int sideIndex=-1; sideIndex<numSides; sideIndex++) {
      BasisCachePtr standardCache, curvilinearCache;
      
      if (sideIndex < 0) {
        standardCache = standardMeshCache;
        curvilinearCache = curvilinearMeshCache;
      } else {
        standardCache = standardMeshCache->getSideBasisCache(sideIndex);
        curvilinearCache = curvilinearMeshCache->getSideBasisCache(sideIndex);
      }
      // check that jacobians are equal
      double maxDiff = 0;
      if (!fcsAgree(standardCache->getJacobian(), curvilinearCache->getJacobian(), tol, maxDiff)) {
        success = false;
        cout << "testTransformationJacobian(): standard and 'curvilinear' Jacobians disagree for k=" << H1Order;
        cout << " and sideIndex " << sideIndex;
        if (maxDiff > 0) {
          cout << ", maxDiff " << maxDiff << endl;
        } else {
          cout << "; sizes differ: standard has " << standardCache->getJacobian().size();
          cout << " entries; 'curvilinear' "   << curvilinearCache->getJacobian().size() << endl;
        }
      }
      if (!fcsAgree(standardCache->getJacobianDet(), curvilinearCache->getJacobianDet(), tol, maxDiff)) {
        success = false;
        cout << "testTransformationJacobian(): standard and 'curvilinear' Jacobian determinants disagree for k=" << H1Order;
        cout << " and sideIndex " << sideIndex;
        if (maxDiff > 0) {
          cout << ", maxDiff " << maxDiff << endl;
        } else {
          cout << "; sizes differ: standard has " << standardCache->getJacobianDet().size();
          cout << " entries; 'curvilinear' "   << curvilinearCache->getJacobianDet().size() << endl;
        }
      }
      if (!fcsAgree(standardCache->getJacobianInv(), curvilinearCache->getJacobianInv(), tol, maxDiff)) {
        success = false;
        cout << "testTransformationJacobian(): standard and 'curvilinear' Jacobian inverses disagree for k=" << H1Order;
        cout << " and sideIndex " << sideIndex;
        if (maxDiff > 0) {
          cout << ", maxDiff " << maxDiff << endl;
        } else {
          cout << "; sizes differ: standard has " << standardCache->getJacobianInv().size();
          cout << " entries; 'curvilinear' "   << curvilinearCache->getJacobianInv().size() << endl;
        }
      }
    }
  }
  
  return success;
}