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

const static double PI  = 3.141592653589793238462;

void CurvilinearMeshTests::setup() {
  
}

void CurvilinearMeshTests::teardown() {
  
}

void CurvilinearMeshTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testH1Projection()) {
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
    double impliedPi = (width * height - approximateArea) / (r*r);
    cout << "For k=" << i << ", implied value of pi: " << impliedPi;
    cout << " (error " << abs(PI-impliedPi) << ")\n";
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
    double impliedPi = (width * height - approximateArea) / (r*r);
    cout << "For h-refinement " << i << ", implied value of pi: " << impliedPi;
    cout << " (error " << abs(PI-impliedPi) << ")\n";
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
  
  // to begin, a very simple test: do we compute the correct perimeter for a square?
  FunctionPtr oneOnBoundary = Function::meshBoundaryCharacteristic();
  
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
  
  map< Edge, ParametricCurvePtr > edgeToCurveMap;

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
    
    ostringstream filePath;
    filePath << "/tmp/squareMesh_p" << i << ".dat";
    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    
    // p-refine
    if (i < numPRefinements) {
      mesh->pRefine(mesh->getActiveCellIDs());
    }
  }
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  int numHRefinements = 5;
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
  
  numPRefinements = 5;
  for (int i=1; i<=numPRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
//    cout << "perimeter: " << perimeter << endl;
    double impliedPi = (perimeter - straightEdgePerimeter) / (2 * radius);
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
  
  // now, do much the same thing, except with h-refinements:
  H1Order = 2;
  mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, meshWidth, meshWidth);
  mesh->setEdgeToCurveMap(edgeToCurveMap);
  previousError = 1000;
  numHRefinements = 5;
  for (int i=0; i<=numHRefinements; i++) {
    perimeter = oneOnBoundary->integrate(mesh);
//    cout << "perimeter: " << perimeter << endl;
    double impliedPi = (perimeter - straightEdgePerimeter) / (2 * radius);
    cout << "For h-refinement " << i << ", implied value of pi: " << impliedPi << endl;
    
    double error = abs(truePerimeter - perimeter);
    if ((error >= previousError) && (error > tol)) { // non-convergence
      success = false;
      cout << "testEdgeLength: Error for h-refinement " << i << " is greater than for h-refinement " << i - 1 << endl;
      cout << "Current error = " << error << "; previous = " << previousError << endl;
    }
    ostringstream filePath;
    filePath << "/tmp/circularMesh_h" << i << ".dat";
    GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), mesh);
    filePath.str("");
    filePath << "/tmp/circularMesh_h" << i << "_straight_lines.dat";
    GnuPlotUtil::writeExactMeshSkeleton(filePath.str(), mesh, 2);
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


bool CurvilinearMeshTests::testH1Projection() {
  bool success = true;

  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  BasisPtr quadraticScalarBasis = BasisFactory::getBasis(2, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  BasisPtr quadraticVectorBasis = BasisFactory::getBasis(2, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
  
  set<int> scalarEdgeNodes = BasisFactory::sideFieldIndices(quadraticScalarBasis);
  set<int> vectorEdgeNodes = BasisFactory::sideFieldIndices(quadraticVectorBasis);
  
  FunctionPtr one = Function::constant(1);
  
  // TODO: work out a test that will compare projector of scalar versus vector basis and confirm that vector behaves as expected...
  
  double width = 2;
  double height = 2;
  
  // the transfinite interpolant in physical space should be just (x, y)
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  FunctionPtr tfi = Function::vectorize(x, y);
  
  BilinearFormPtr bf = VGPStokesFormulation(1.0).bf();

  int pToAdd = 0; // 0 so that H1Order itself will govern the order of the approximation
  

  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", VECTOR_HGRAD);
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v);
  ip->addTerm(v->grad());
  
  for (int H1Order=1; H1Order<5; H1Order++) {
    MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, pToAdd, width, height);
    int cellID = 0; // the only cell
    bool testVsTest = true;
    
    BasisPtr basis = BasisFactory::getBasis(H1Order, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
    
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

    VectorBasisPtr vectorBasis = Teuchos::rcp( (Vectorized_Basis<double, FieldContainer<double> > *)basis.get(),false);

    // For H1Order > 1, we don't expect that the edge interpolant will match the TFI on the element interior; we expect that only on the edges.

    vector< ParametricCurvePtr > curves = mesh->parametricEdgesForCell(cellID);
    ParametricSurfacePtr exactSurface = ParametricSurface::transfiniteInterpolant(curves);
    int numEdges = curves.size();
    
    FieldContainer<double> edgeInterpolantCoefficients;
    ParametricSurface::basisWeightsForEdgeInterpolant(edgeInterpolantCoefficients, vectorBasis, mesh, cellID);
    edgeInterpolantCoefficients.resize(edgeInterpolantCoefficients.size());
    
    // check that the only nonzeros in edgeInterpolantCoefficients belong to edges
    set<int> edgeFieldIndices = BasisFactory::sideFieldIndices(vectorBasis, true); // true: include the vertices
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
      int basisRank = BasisFactory::getBasisRank(basis);
      
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
      int basisRank = BasisFactory::getBasisRank(basis);
      
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
    map< pair<int, int>, ParametricCurvePtr > edgeToCurveMap;
    
    vector< ParametricCurvePtr > lines = mesh->parametricEdgesForCell(cellID);
    vector< int > vertices = mesh->vertexIndicesForCell(cellID);
    
    for (int i=0; i<vertices.size(); i++) {
      int vertex = vertices[i];
      int nextVertex = vertices[(i+1) % vertices.size()];
      pair< int, int > edge = make_pair(vertex,nextVertex);
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