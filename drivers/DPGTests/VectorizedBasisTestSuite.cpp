#include "VectorizedBasisTestSuite.h"

#include "Mesh.h"
#include "Solution.h"
#include "InnerProductScratchPad.h"

void VectorizedBasisTestSuite::runTests(int &numTestsRun, int &numTestsPassed) {
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

class EntireBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      return true;
    }
};

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
  bf->addTerm( -sigma, v->grad() );
  bf->addTerm( sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  rhs->addTerm( f * v );

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr boundary = Teuchos::rcp( new EntireBoundary );
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
  solution->solve(false);

  return success;
}
