#include "VectorizedBasisTestSuite.h"

void VectorizedBasisTestSuite::runTests(int &numTestsRun, int &numTestsPassed) {
  numTestsRun++;
  if ( testVectorizedBasis() ) {
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