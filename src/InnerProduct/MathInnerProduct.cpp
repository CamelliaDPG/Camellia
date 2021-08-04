//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#include "Camellia_FieldContainer.hpp"

#include "MathInnerProduct.h"

using namespace Camellia;

void MathInnerProduct::operators(int testID1, int testID2,
                                 vector<Camellia::EOperator> &testOp1,
                                 vector<Camellia::EOperator> &testOp2)
{
  testOp1.clear();
  testOp2.clear();
  if (testID1 == testID2)
  {
    Camellia::EOperator dOperator;
    if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_REAL_SCALAR)
    {
      testOp1.push_back( Camellia::OP_VALUE);
      testOp2.push_back( Camellia::OP_VALUE);
    }
    else
    {
      if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HGRAD)
      {
        dOperator = Camellia::OP_GRAD;
      }
      else if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HDIV)
      {
        dOperator = Camellia::OP_DIV;
      }
      else if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HCURL)
      {
        dOperator = Camellia::OP_CURL;
      }
      else if ( _bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_VECTOR_HGRAD )
      {
        dOperator = Camellia::OP_GRAD; // will the integration routine do the right thing here??
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown test space.");
      }
      testOp1.push_back( Camellia::OP_VALUE);
      testOp1.push_back(dOperator);
      testOp2.push_back( Camellia::OP_VALUE);
      testOp2.push_back(dOperator);
    }
  }
}

void MathInnerProduct::applyInnerProductData(Camellia::FieldContainer<double> &testValues1,
    Camellia::FieldContainer<double> &testValues2,
    int testID1, int testID2, int operatorIndex,
    const Camellia::FieldContainer<double>& physicalPoints)
{
  // empty implementation -- no weights needed...
}
