#include "VectorizedBasisTestSuite.h"

#include "Mesh.h"
#include "Solution.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "CamelliaConfig.h"

#include <Teuchos_GlobalMPISession.hpp>

#ifdef USE_VTK
#include "SolutionExporter.h"
#endif

#include <string>
using namespace std;

void VectorizedBasisTestSuite::runTests(int &numTestsRun, int &numTestsPassed) {
  numTestsRun++;
  if ( testVectorizedBasisTags() ) {
    numTestsPassed++;
  }
  
  numTestsRun++;
  if ( testVectorizedBasis() ) {
    numTestsPassed++;
  }

  numTestsRun++;
  if ( testPoisson() ) {
    numTestsPassed++;
  }
}

string VectorizedBasisTestSuite::testSuiteName() {
  return "VectorizedBasisTestSuite";
}


bool checkVertexNodalIndicesQuad(BasisPtr basis, vector<int> &vertexIndices) {
  // check that the given indices are exactly the vertex basis functions:
  // a) these are (1,0) or (0,1) at the corresponding vertices
  // b) others are (0,0) at the vertices
  
  int numVertices = 4;
  
  FieldContainer<double> refCellPoints(numVertices,2); // vertices, in order
  refCellPoints(0,0) = -1;
  refCellPoints(0,1) = -1;
  refCellPoints(1,0) =  1;
  refCellPoints(1,1) = -1;
  refCellPoints(2,0) =  1;
  refCellPoints(2,1) =  1;
  refCellPoints(3,0) = -1;
  refCellPoints(3,1) =  1;

  // assume 2D vector basis for now -- we'll throw an exception if not...
  FieldContainer<double> values(basis->getCardinality(), numVertices,2); // F, P, D
  basis->getValues(values, refCellPoints, OPERATOR_VALUE); 
  
  double tol = 1e-14;
  for (int ptIndex=0; ptIndex<numVertices; ptIndex++) {
    int xNodeIndex = vertexIndices[2*ptIndex];
    int yNodeIndex = vertexIndices[2*ptIndex+1];
    for (int fieldIndex=0; fieldIndex<basis->getCardinality(); fieldIndex++) {
      double xValue = values(fieldIndex,ptIndex,0);
      double yValue = values(fieldIndex,ptIndex,1);
      double xExpected = 0, yExpected = 0;
      if (fieldIndex==xNodeIndex) {
        xExpected = 1;
      } else if (fieldIndex==yNodeIndex) {
        yExpected = 1;
      }
      if ((abs(xExpected-xValue) > tol) || (abs(yExpected-yValue)>tol)) {
        return false;
      }
    }
  }
  return true;
}

bool VectorizedBasisTestSuite::testVectorizedBasisTags() {
  bool success = true;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  int numVertices = quad_4.getVertexCount();
  int numComponents = quad_4.getDimension();
  int vertexDim = 0;
  
  for (int polyOrder = 1; polyOrder<10; polyOrder++) {    
    int basisRank;
    BasisPtr hgradBasis =  BasisFactory::getBasis(basisRank,polyOrder,
                                                  quad_4.getKey(), 
                                                  IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    BasisPtr vectorHGradBasis = BasisFactory::getBasis( polyOrder,
                                                       quad_4.getKey(), 
                                                       IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
    vector<int> vertexNodeFieldIndices;
    for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
      for (int comp=0; comp<numComponents; comp++) {
        int vertexNodeFieldIndex = vectorHGradBasis->getDofOrdinal(vertexDim, vertexIndex, comp);
        vertexNodeFieldIndices.push_back(vertexNodeFieldIndex);
      }
    }
    if (!checkVertexNodalIndicesQuad(vectorHGradBasis, vertexNodeFieldIndices) ) {
      success = false;
      cout << "testVectorizedBasisTags: Vertex tags for vectorized HGRAD basis";
      cout << "of order " << polyOrder << " are incorrect.\n";
    }
  }
  
  return success;
}

bool VectorizedBasisTestSuite::testVectorizedBasis() {
  bool success = true;
  
  string myName = "testVectorizedBasis";
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  int polyOrder = 3, numPoints = 5, spaceDim = 2;
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > hgradBasis
  = 
  BasisFactory::getBasis(basisRank,polyOrder,
                         quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  // first test: make a single-component vector basis.  This should agree in every entry with the basis itself, but its field container will have one higher rank...
  Vectorized_Basis<double, FieldContainer<double> > oneComp(hgradBasis, 1);
  
  FieldContainer<double> linePoints(numPoints, spaceDim);
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<spaceDim; j++) {
      linePoints(i,j) = ((double)(i + j)) / (numPoints + spaceDim);
    }
  }
  
  FieldContainer<double> compValues(hgradBasis->getCardinality(),linePoints.dimension(0));
  hgradBasis->getValues(compValues, linePoints, Intrepid::OPERATOR_VALUE);
  
  FieldContainer<double> values(hgradBasis->getCardinality(),linePoints.dimension(0),1); // one component
  oneComp.getValues(values, linePoints, Intrepid::OPERATOR_VALUE);
  
  for (int i=0; i<compValues.size(); i++) {
    double diff = abs(values[i]-compValues[i]);
    if (diff != 0.0) {
      success = false;
      cout << myName << ": one-component vector basis doesn't produce same values as component basis." << endl;
      cout << "difference: " << diff << " in enumerated value " << i << endl;
      cout << "values:\n" << values;
      cout << "compValues:\n" << compValues;
      return success;
    }
  }
  
  vector< Teuchos::RCP< Basis<double, FieldContainer<double> > > > twoComps;
  twoComps.push_back( Teuchos::rcp( new Vectorized_Basis<double, FieldContainer<double> >(hgradBasis, 2) ) );
  twoComps.push_back( BasisFactory::getBasis( polyOrder,
                                             quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) );
  
  
  vector< Teuchos::RCP< Basis<double, FieldContainer<double> > > >::iterator twoCompIt;
  for (twoCompIt = twoComps.begin(); twoCompIt != twoComps.end(); twoCompIt++) {
    Teuchos::RCP< Basis<double, FieldContainer<double> > > twoComp = *twoCompIt;
    
    int componentCardinality = hgradBasis->getCardinality();
    
    if (twoComp->getCardinality() != 2 * hgradBasis->getCardinality() ) {
      success = false;
      cout << myName << ": two-component vector basis cardinality != one-component cardinality * 2." << endl;
      cout << "twoComp->getCardinality(): " << twoComp->getCardinality() << endl;
      cout << "oneComp->getCardinality(): " << oneComp.getCardinality() << endl;
    }
    
    values.resize(twoComp->getCardinality(),linePoints.dimension(0),2); // two components
    twoComp->getValues(values, linePoints, Intrepid::OPERATOR_VALUE);
    for (int basisIndex=0; basisIndex<twoComp->getCardinality(); basisIndex++) {
      for (int k=0; k<numPoints; k++) {
        double xValueExpected = (basisIndex < componentCardinality) ? compValues(basisIndex,k) : 0;
        double xValueActual = values(basisIndex,k,0);
        double yValueExpected = (basisIndex >= componentCardinality) ? compValues(basisIndex - componentCardinality,k) : 0;
        double yValueActual = values(basisIndex,k,1);
        if ( ( abs(xValueActual - xValueExpected) != 0) || ( abs(yValueActual - yValueExpected) != 0) ) {
          success = false;
          cout << myName << ": expected differs from actual\n";
          cout << "component\n" << compValues;
          cout << "vector values:\n" << values;
          return success;
        }
      }
    }
    
    // test the mapping from oneComp dofOrdinal to twoComp:
    Vectorized_Basis<double, FieldContainer<double> >* twoCompAsVectorBasis = (Vectorized_Basis<double, FieldContainer<double> >  *) twoComp.get();
    
    for (int compDofOrdinal=0; compDofOrdinal<oneComp.getCardinality(); compDofOrdinal++) {
      int dofOrdinal_0 = twoCompAsVectorBasis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 0);
      int dofOrdinal_1 = twoCompAsVectorBasis->getDofOrdinalFromComponentDofOrdinal(compDofOrdinal, 1);
      // we expect the lists to be stacked (this is implicit in the test above)
      // dofOrdinal_0 we expect to be == compDofOrdinal
      // dofOrdinal_1 we expect to be == compDofOrdinal + oneComp.getCardinality()
      if (dofOrdinal_0 != compDofOrdinal) {
        success = false;
        cout << "getDofOrdinalFromComponentDofOrdinal() not returning expected value in first component.\n";
      }
      if (dofOrdinal_1 != compDofOrdinal + oneComp.getCardinality()) {
        success = false;
        cout << "getDofOrdinalFromComponentDofOrdinal() not returning expected value in second component.\n";
      }
    }

  }
  return success;
}

bool VectorizedBasisTestSuite::testHGRAD_2D() {
  bool success = true;
  // on a single quad element, evaluate the various operations
  // TODO: cross normal
  // TODO: div
  // TODO: curl
  return success;
}

bool VectorizedBasisTestSuite::testPoisson() {
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr sigma_n = varFactory.fluxVar("\\widehat{\\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("\\sigma", VECTOR_L2);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  bf->addTerm(sigma, tau);
  bf->addTerm(u, tau->div());
  bf->addTerm(-uhat, tau->dot_normal());

  // v terms:
  bf->addTerm( sigma, v->grad() );
  bf->addTerm( -sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  rhs->addTerm( f * v );

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  bc->addDirichlet(uhat, boundary, zero);

  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) = 0.0; // x1
  meshBoundary(0,1) = 0.0; // y1
  meshBoundary(1,0) = 1.0;
  meshBoundary(1,1) = 0.0;
  meshBoundary(2,0) = 1.0;
  meshBoundary(2,1) = 1.0;
  meshBoundary(3,0) = 0.0;
  meshBoundary(3,1) = 1.0;

  int horizontalCells = 1, verticalCells = 1;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      bf, H1Order, H1Order+pToAdd, false);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
#ifdef USE_VTK
  VTKExporter exporter(solution, mesh, varFactory);
#endif

  for (int refIndex=0; refIndex<=4; refIndex++)
  {
    solution->solve(false);
#ifdef USE_VTK
    // output commented out because it's not properly part of the test.
//    stringstream outfile;
//    outfile << "test_" << refIndex;
//    exporter.exportSolution(outfile.str());
#endif

    if (refIndex < 4)
      refinementStrategy.refine(false); // don't print to console
  }
  return success;
}
