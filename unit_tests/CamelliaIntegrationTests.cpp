//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_Integrate.hpp"

#include <stdexcept>

using namespace Camellia;

namespace
{
const double tol = 1e-13;

void fillDistinct(Camellia::FieldContainer<double> &fc, const double offset)
{
  for (int i=0; i<fc.size(); i++) fc[i] = offset + 0.125 * (i + 1);
}

int productDimensions(const Teuchos::Array<int> &dims, const int startDim)
{
  int product = 1;
  for (int d=startDim; d<dims.size(); d++) product *= dims[d];
  return product;
}

void verifyRank1(Camellia::FieldContainer<double> &left, Camellia::FieldContainer<double> &right, bool &success, Teuchos::FancyOStream &out)
{
  Camellia::FieldContainer<double> output(left.dimension(0));
  output.initialize(-2.0);
  Camellia::FunctionSpaceTools::integrate(output, left, right, false);

  Teuchos::Array<int> dims;
  left.dimensions(dims);
  const int numCells = dims[0];
  const int dataCount = productDimensions(dims, 1);
  for (int c=0; c<numCells; c++)
  {
    double expected = 0.0;
    for (int n=0; n<dataCount; n++) expected += left[c*dataCount+n] * right[c*dataCount+n];
    TEST_FLOATING_EQUALITY(expected, output(c), tol);
  }

  for (int c=0; c<numCells; c++) output(c) = 7.0 + c;
  Camellia::FunctionSpaceTools::integrate(output, left, right, true);
  for (int c=0; c<numCells; c++)
  {
    double expected = 7.0 + c;
    for (int n=0; n<dataCount; n++) expected += left[c*dataCount+n] * right[c*dataCount+n];
    TEST_FLOATING_EQUALITY(expected, output(c), tol);
  }
}

void verifyRank2(Camellia::FieldContainer<double> &data, Camellia::FieldContainer<double> &fields, bool &success, Teuchos::FancyOStream &out)
{
  Camellia::FieldContainer<double> output(data.dimension(0), fields.dimension(1));
  output.initialize(-3.0);
  Camellia::FunctionSpaceTools::integrate(output, data, fields, false);

  Teuchos::Array<int> dataDims, fieldDims;
  data.dimensions(dataDims);
  fields.dimensions(fieldDims);
  const int numCells = dataDims[0];
  const int numFields = fieldDims[1];
  const int dataCount = productDimensions(dataDims, 1);
  for (int c=0; c<numCells; c++)
  {
    for (int f=0; f<numFields; f++)
    {
      double expected = 0.0;
      for (int n=0; n<dataCount; n++) expected += data[c*dataCount+n] * fields[c*numFields*dataCount + f*dataCount + n];
      TEST_FLOATING_EQUALITY(expected, output(c,f), tol);
    }
  }

  for (int i=0; i<output.size(); i++) output[i] = 11.0 + i;
  Camellia::FunctionSpaceTools::integrate(output, data, fields, true);
  for (int c=0; c<numCells; c++)
  {
    for (int f=0; f<numFields; f++)
    {
      double expected = 11.0 + c*numFields + f;
      for (int n=0; n<dataCount; n++) expected += data[c*dataCount+n] * fields[c*numFields*dataCount + f*dataCount + n];
      TEST_FLOATING_EQUALITY(expected, output(c,f), tol);
    }
  }
}

void verifyRank2ConstantData(Camellia::FieldContainer<double> &data, Camellia::FieldContainer<double> &fields, bool &success, Teuchos::FancyOStream &out)
{
  Camellia::FieldContainer<double> output(data.dimension(0), fields.dimension(1));
  Camellia::FunctionSpaceTools::integrate(output, data, fields, false);

  Teuchos::Array<int> dataDims, fieldDims;
  data.dimensions(dataDims);
  fields.dimensions(fieldDims);
  const int numCells = dataDims[0];
  const int numFields = fieldDims[1];
  const int numPoints = fieldDims[2];
  const int componentCount = productDimensions(dataDims, 2);
  for (int c=0; c<numCells; c++)
  {
    for (int f=0; f<numFields; f++)
    {
      double expected = 0.0;
      for (int p=0; p<numPoints; p++)
      {
        for (int comp=0; comp<componentCount; comp++)
        {
          expected += data[c*componentCount + comp] * fields[c*numFields*numPoints*componentCount + f*numPoints*componentCount + p*componentCount + comp];
        }
      }
      TEST_FLOATING_EQUALITY(expected, output(c,f), tol);
    }
  }
}

void verifyRank3(Camellia::FieldContainer<double> &left, Camellia::FieldContainer<double> &right, bool &success, Teuchos::FancyOStream &out)
{
  Camellia::FieldContainer<double> output(left.dimension(0), left.dimension(1), right.dimension(1));
  output.initialize(-4.0);
  Camellia::FunctionSpaceTools::integrate(output, left, right, false);

  Teuchos::Array<int> leftDims;
  left.dimensions(leftDims);
  const int numCells = left.dimension(0);
  const int numLeftFields = left.dimension(1);
  const int numRightFields = right.dimension(1);
  const int dataCount = productDimensions(leftDims, 2);
  for (int c=0; c<numCells; c++)
  {
    for (int i=0; i<numLeftFields; i++)
    {
      for (int j=0; j<numRightFields; j++)
      {
        double expected = 0.0;
        for (int n=0; n<dataCount; n++) expected += left[c*numLeftFields*dataCount + i*dataCount + n] * right[c*numRightFields*dataCount + j*dataCount + n];
        TEST_FLOATING_EQUALITY(expected, output(c,i,j), tol);
      }
    }
  }

  for (int i=0; i<output.size(); i++) output[i] = 13.0 + i;
  Camellia::FunctionSpaceTools::integrate(output, left, right, true);
  for (int c=0; c<numCells; c++)
  {
    for (int i=0; i<numLeftFields; i++)
    {
      for (int j=0; j<numRightFields; j++)
      {
        double expected = 13.0 + c*numLeftFields*numRightFields + i*numRightFields + j;
        for (int n=0; n<dataCount; n++) expected += left[c*numLeftFields*dataCount + i*dataCount + n] * right[c*numRightFields*dataCount + j*dataCount + n];
        TEST_FLOATING_EQUALITY(expected, output(c,i,j), tol);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, DataDataContractions )
{
  Camellia::FieldContainer<double> scalarLeft(2,3), scalarRight(2,3);
  fillDistinct(scalarLeft, 1.0);
  fillDistinct(scalarRight, 2.0);
  verifyRank1(scalarLeft, scalarRight, success, out);

  Camellia::FieldContainer<double> vectorLeft(2,3,2), vectorRight(2,3,2);
  fillDistinct(vectorLeft, 3.0);
  fillDistinct(vectorRight, 4.0);
  verifyRank1(vectorLeft, vectorRight, success, out);

  Camellia::FieldContainer<double> tensorLeft(2,2,2,3), tensorRight(2,2,2,3);
  fillDistinct(tensorLeft, 5.0);
  fillDistinct(tensorRight, 6.0);
  verifyRank1(tensorLeft, tensorRight, success, out);
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, DataFieldContractions )
{
  Camellia::FieldContainer<double> scalarData(2,4), scalarFields(2,3,4);
  fillDistinct(scalarData, 1.0);
  fillDistinct(scalarFields, 2.0);
  verifyRank2(scalarData, scalarFields, success, out);

  Camellia::FieldContainer<double> vectorData(2,4,2), vectorFields(2,3,4,2);
  fillDistinct(vectorData, 3.0);
  fillDistinct(vectorFields, 4.0);
  verifyRank2(vectorData, vectorFields, success, out);

  Camellia::FieldContainer<double> tensorData(2,3,2,2), tensorFields(2,3,3,2,2);
  fillDistinct(tensorData, 5.0);
  fillDistinct(tensorFields, 6.0);
  verifyRank2(tensorData, tensorFields, success, out);
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, ConstantDataFieldContractions )
{
  Camellia::FieldContainer<double> scalarData(2,1), scalarFields(2,3,4);
  fillDistinct(scalarData, 1.0);
  fillDistinct(scalarFields, 2.0);
  verifyRank2ConstantData(scalarData, scalarFields, success, out);

  Camellia::FieldContainer<double> vectorData(2,1,2), vectorFields(2,3,4,2);
  fillDistinct(vectorData, 3.0);
  fillDistinct(vectorFields, 4.0);
  verifyRank2ConstantData(vectorData, vectorFields, success, out);
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, FieldFieldContractions )
{
  Camellia::FieldContainer<double> scalarLeft(2,2,4), scalarRight(2,3,4);
  fillDistinct(scalarLeft, 1.0);
  fillDistinct(scalarRight, 2.0);
  verifyRank3(scalarLeft, scalarRight, success, out);

  Camellia::FieldContainer<double> vectorLeft(2,2,4,2), vectorRight(2,3,4,2);
  fillDistinct(vectorLeft, 3.0);
  fillDistinct(vectorRight, 4.0);
  verifyRank3(vectorLeft, vectorRight, success, out);

  Camellia::FieldContainer<double> tensorLeft(2,2,3,2,2), tensorRight(2,3,3,2,2);
  fillDistinct(tensorLeft, 5.0);
  fillDistinct(tensorRight, 6.0);
  verifyRank3(tensorLeft, tensorRight, success, out);
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, ScalarMultiplyDataFieldWrapper )
{
  Camellia::FieldContainer<double> data(2,3), fields(2,4,3), output(2,4,3);
  fillDistinct(data, 1.0);
  fillDistinct(fields, 2.0);

  Camellia::FunctionSpaceTools::scalarMultiplyDataField(output, data, fields, false);
  for (int c=0; c<2; c++)
  {
    for (int f=0; f<4; f++)
    {
      for (int p=0; p<3; p++)
      {
        TEST_FLOATING_EQUALITY(data(c,p) * fields(c,f,p), output(c,f,p), tol);
      }
    }
  }

  Camellia::FunctionSpaceTools::scalarMultiplyDataField(output, data, fields, true);
  for (int c=0; c<2; c++)
  {
    for (int f=0; f<4; f++)
    {
      for (int p=0; p<3; p++)
      {
        TEST_FLOATING_EQUALITY(fields(c,f,p) / data(c,p), output(c,f,p), tol);
      }
    }
  }

  Camellia::FunctionSpaceTools::scalarMultiplyDataField(fields, data, fields, false);
  for (int c=0; c<2; c++)
  {
    for (int f=0; f<4; f++)
    {
      for (int p=0; p<3; p++)
      {
        const double original = 2.0 + 0.125 * ((c*4*3 + f*3 + p) + 1);
        TEST_FLOATING_EQUALITY(data(c,p) * original, fields(c,f,p), tol);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, ScalarMultiplyDataDataWrapper )
{
  Camellia::FieldContainer<double> left(2,3), right(2,3,2), output(2,3,2);
  fillDistinct(left, 1.0);
  fillDistinct(right, 2.0);

  Camellia::FunctionSpaceTools::scalarMultiplyDataData(output, left, right, false);
  for (int c=0; c<2; c++)
  {
    for (int p=0; p<3; p++)
    {
      for (int d=0; d<2; d++)
      {
        TEST_FLOATING_EQUALITY(left(c,p) * right(c,p,d), output(c,p,d), tol);
      }
    }
  }

  Camellia::FunctionSpaceTools::scalarMultiplyDataData(output, left, right, true);
  for (int c=0; c<2; c++)
  {
    for (int p=0; p<3; p++)
    {
      for (int d=0; d<2; d++)
      {
        TEST_FLOATING_EQUALITY(right(c,p,d) / left(c,p), output(c,p,d), tol);
      }
    }
  }

  Camellia::FunctionSpaceTools::scalarMultiplyDataData(right, left, right, false);
  for (int c=0; c<2; c++)
  {
    for (int p=0; p<3; p++)
    {
      for (int d=0; d<2; d++)
      {
        const double original = 2.0 + 0.125 * ((c*3*2 + p*2 + d) + 1);
        TEST_FLOATING_EQUALITY(left(c,p) * original, right(c,p,d), tol);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaIntegrate, InvalidDimensionsThrow )
{
  Camellia::FieldContainer<double> output(2,3);
  Camellia::FieldContainer<double> data(2,4);
  Camellia::FieldContainer<double> mismatchedFields(2,3,5);
  bool threw = false;
  try
  {
    Camellia::FunctionSpaceTools::integrate(output, data, mismatchedFields, false);
  }
  catch (const std::invalid_argument &)
  {
    threw = true;
  }
  TEST_EQUALITY(threw, true);
}

} // namespace
