//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Kokkos_Core.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using Scalar = double;

  int product(const std::vector<int> &values)
  {
    int result = 1;
    for (const int value : values) result *= value;
    return result;
  }

  std::vector<int> multiIndex(const int ordinal, const std::vector<int> &extents)
  {
    std::vector<int> index(extents.size(),0);
    int remainder = ordinal;
    for (size_t d=0; d<extents.size(); d++)
    {
      index[d] = remainder % extents[d];
      remainder /= extents[d];
    }
    return index;
  }

  int ordinal(const std::vector<int> &index, const std::vector<int> &extents)
  {
    int value = 0;
    int stride = 1;
    for (size_t d=0; d<extents.size(); d++)
    {
      value += index[d] * stride;
      stride *= extents[d];
    }
    return value;
  }

  struct OneDimensionalOperator
  {
    int fields = 0;
    int points = 0;
    std::vector<Scalar> values; // values[f + fields * p]

    Scalar operator()(const int field, const int point) const
    {
      return values[field + fields * point];
    }
  };

  OneDimensionalOperator makeOperator(const int fields, const int points, const Scalar offset)
  {
    OneDimensionalOperator op;
    op.fields = fields;
    op.points = points;
    op.values.resize(fields * points);
    for (int p=0; p<points; p++)
    {
      for (int f=0; f<fields; f++)
      {
        // Deterministic, non-symmetric-enough values that remain modest in size.
        op.values[f + fields * p] = offset + 0.125 * (f + 1) - 0.0625 * (p + 1) + 0.015625 * (f + 1) * (p + 1);
      }
    }
    return op;
  }

  class SyntheticTensorPAMatrix
  {
    std::vector<OneDimensionalOperator> leftOps_;
    std::vector<OneDimensionalOperator> rightOps_;
    std::vector<Scalar> pointWeights_;
    int numFieldsLeft_ = 0;
    int numFieldsRight_ = 0;
    int numPoints_ = 0;

  public:
    SyntheticTensorPAMatrix(std::vector<OneDimensionalOperator> leftOps,
                            std::vector<OneDimensionalOperator> rightOps,
                            std::vector<Scalar> pointWeights)
    : leftOps_(std::move(leftOps)), rightOps_(std::move(rightOps)), pointWeights_(std::move(pointWeights))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(leftOps_.empty(), std::invalid_argument, "At least one tensor dimension is required");
      TEUCHOS_TEST_FOR_EXCEPTION(leftOps_.size() != rightOps_.size(), std::invalid_argument, "Left/right operator dimensions must agree");

      std::vector<int> leftFieldExtents, rightFieldExtents, pointExtents;
      for (size_t d=0; d<leftOps_.size(); d++)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(leftOps_[d].points != rightOps_[d].points, std::invalid_argument, "Left/right point extents must agree");
        leftFieldExtents.push_back(leftOps_[d].fields);
        rightFieldExtents.push_back(rightOps_[d].fields);
        pointExtents.push_back(leftOps_[d].points);
      }
      numFieldsLeft_  = product(leftFieldExtents);
      numFieldsRight_ = product(rightFieldExtents);
      numPoints_      = product(pointExtents);
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(pointWeights_.size()) != numPoints_, std::invalid_argument, "Point weight count does not match tensor point count");
    }

    int numFieldsLeft() const { return numFieldsLeft_; }
    int numFieldsRight() const { return numFieldsRight_; }
    int numPoints() const { return numPoints_; }
    int spaceDim() const { return static_cast<int>(leftOps_.size()); }

    std::vector<Scalar> apply(const std::vector<Scalar> &input) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(input.size()) != numFieldsRight_, std::invalid_argument, "Input vector has wrong size");

      std::vector<int> rightExtents, leftExtents, pointExtents;
      for (size_t d=0; d<rightOps_.size(); d++)
      {
        rightExtents.push_back(rightOps_[d].fields);
        leftExtents.push_back(leftOps_[d].fields);
        pointExtents.push_back(rightOps_[d].points);
      }

      // Matrix-free tensor-product action: coefficient-to-quadrature, point multiply,
      // then quadrature-to-coefficient.  The assembled matrix is not formed here.
      std::vector<Scalar> pointValues(numPoints_, 0.0);
      for (int pointOrdinal=0; pointOrdinal<numPoints_; pointOrdinal++)
      {
        const auto pointIndex = multiIndex(pointOrdinal, pointExtents);
        Scalar pointValue = 0.0;
        for (int fieldOrdinal=0; fieldOrdinal<numFieldsRight_; fieldOrdinal++)
        {
          const auto fieldIndex = multiIndex(fieldOrdinal, rightExtents);
          Scalar basisValue = 1.0;
          for (int d=0; d<spaceDim(); d++)
          {
            basisValue *= rightOps_[d](fieldIndex[d], pointIndex[d]);
          }
          pointValue += basisValue * input[fieldOrdinal];
        }
        pointValues[pointOrdinal] = pointWeights_[pointOrdinal] * pointValue;
      }

      std::vector<Scalar> output(numFieldsLeft_, 0.0);
      for (int fieldOrdinal=0; fieldOrdinal<numFieldsLeft_; fieldOrdinal++)
      {
        const auto fieldIndex = multiIndex(fieldOrdinal, leftExtents);
        Scalar value = 0.0;
        for (int pointOrdinal=0; pointOrdinal<numPoints_; pointOrdinal++)
        {
          const auto pointIndex = multiIndex(pointOrdinal, pointExtents);
          Scalar basisValue = 1.0;
          for (int d=0; d<spaceDim(); d++)
          {
            basisValue *= leftOps_[d](fieldIndex[d], pointIndex[d]);
          }
          value += basisValue * pointValues[pointOrdinal];
        }
        output[fieldOrdinal] = value;
      }
      return output;
    }

    std::vector<Scalar> assemble() const
    {
      std::vector<Scalar> matrix(numFieldsLeft_ * numFieldsRight_, 0.0);
      for (int j=0; j<numFieldsRight_; j++)
      {
        std::vector<Scalar> e(numFieldsRight_, 0.0);
        e[j] = 1.0;
        const auto column = apply(e);
        for (int i=0; i<numFieldsLeft_; i++)
        {
          matrix[i + numFieldsLeft_ * j] = column[i];
        }
      }
      return matrix;
    }

    std::vector<Scalar> diagonal() const
    {
      const int numDiag = std::min(numFieldsLeft_, numFieldsRight_);
      const auto matrix = assemble();
      std::vector<Scalar> diag(numDiag, 0.0);
      for (int i=0; i<numDiag; i++) diag[i] = matrix[i + numFieldsLeft_ * i];
      return diag;
    }

    long long estimatedApplyFlops() const
    {
      long long coefficientToPoint = 0;
      long long pointToCoefficient = 0;
      long long rightPartial = numFieldsRight_;
      long long leftPartial = numPoints_;
      for (int d=spaceDim()-1; d>=0; d--)
      {
        coefficientToPoint += 2LL * rightOps_[d].points * rightPartial;
        rightPartial = rightPartial / rightOps_[d].fields * rightOps_[d].points;
        pointToCoefficient += 2LL * leftOps_[d].fields * leftPartial;
        leftPartial = leftPartial / leftOps_[d].points * leftOps_[d].fields;
      }
      return coefficientToPoint + numPoints_ + pointToCoefficient;
    }
  };

  SyntheticTensorPAMatrix makeTensorOperator(const int spaceDim, const int fieldsPerDim, const int pointsPerDim,
                                             const bool rectangular = false)
  {
    std::vector<OneDimensionalOperator> leftOps, rightOps;
    for (int d=0; d<spaceDim; d++)
    {
      leftOps.push_back(makeOperator(fieldsPerDim + (rectangular && d == 0 ? 1 : 0), pointsPerDim, 0.25 + 0.1 * d));
      rightOps.push_back(makeOperator(fieldsPerDim, pointsPerDim, 0.5 + 0.075 * d));
    }

    const int numPoints = product(std::vector<int>(spaceDim, pointsPerDim));
    std::vector<Scalar> pointWeights(numPoints, 0.0);
    for (int p=0; p<numPoints; p++) pointWeights[p] = 0.8 + 0.01 * (p % 7);

    return SyntheticTensorPAMatrix(leftOps, rightOps, pointWeights);
  }

  void testApplyMatchesAssembled(const SyntheticTensorPAMatrix &op, Teuchos::FancyOStream &out, bool &success)
  {
    std::vector<Scalar> input(op.numFieldsRight(), 0.0);
    for (int i=0; i<op.numFieldsRight(); i++) input[i] = 0.125 + 0.0625 * (i + 1) + ((i % 2 == 0) ? 0.5 : -0.25);

    const auto actual = op.apply(input);
    const auto matrix = op.assemble();

    for (int i=0; i<op.numFieldsLeft(); i++)
    {
      Scalar expected = 0.0;
      for (int j=0; j<op.numFieldsRight(); j++) expected += matrix[i + op.numFieldsLeft() * j] * input[j];
      const Scalar error = std::abs(actual[i] - expected);
      if (error > 1e-12)
      {
        out << "Tensor matrix-free apply mismatch at row " << i << ": expected " << expected
            << ", actual " << actual[i] << ", error " << error << std::endl;
        success = false;
      }
    }
  }
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, TensorProductApplyMatchesAssembled)
{
  auto op = makeTensorOperator(3, 3, 4);
  testApplyMatchesAssembled(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RectangularTensorProductApplyMatchesAssembled)
{
  auto op = makeTensorOperator(2, 3, 4, true);
  testApplyMatchesAssembled(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, DiagonalMatchesAssembled)
{
  auto op = makeTensorOperator(3, 2, 3);
  const auto matrix = op.assemble();
  const auto diagonal = op.diagonal();

  for (int i=0; i<static_cast<int>(diagonal.size()); i++)
  {
    TEST_FLOATING_EQUALITY(diagonal[i], matrix[i + op.numFieldsLeft() * i], 1e-14);
  }
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, EstimatedApplyFlopsScaleAsTensorContractions)
{
  const int dim = 4;
  const int p = 5;
  const int fieldsPerDim = p + 1;
  const int pointsPerDim = p + 2;
  auto op = makeTensorOperator(dim, fieldsPerDim, pointsPerDim);

  // Each one-dimensional contraction costs O(p) work for each tensor line and
  // there are O(p^d) tensor entries.  This check protects the first-slice target
  // from silently replacing the apply path with O(p^(2d)) assembled matvec work.
  const long long tensorApplyFlops = op.estimatedApplyFlops();
  const long long denseMatvecFlops = 2LL * op.numFieldsLeft() * op.numFieldsRight();

  TEST_ASSERT(tensorApplyFlops < denseMatvecFlops / 4);
}
