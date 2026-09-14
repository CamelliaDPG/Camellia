//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_PAMatrix.hpp"

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
  using Scalar = double;
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using View2D = Kokkos::View<Scalar**,DeviceType>;

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

  View2D makeView(const OneDimensionalOperator &op, const std::string &label)
  {
    View2D view(label, op.fields, op.points);
    auto host = Kokkos::create_mirror_view(view);
    for (int f=0; f<op.fields; f++)
    {
      for (int p=0; p<op.points; p++)
      {
        host(f,p) = op(f,p);
      }
    }
    Kokkos::deep_copy(view, host);
    return view;
  }

  View2D makeWeights(const int cells, const int points, const std::string &label)
  {
    View2D weights(label, cells, points);
    auto host = Kokkos::create_mirror_view(weights);
    for (int c=0; c<cells; c++)
    {
      for (int p=0; p<points; p++)
      {
        host(c,p) = 0.8 + 0.01 * ((p + 3 * c) % 7);
      }
    }
    Kokkos::deep_copy(weights, host);
    return weights;
  }

  Intrepid2::PAMatrix<DeviceType,Scalar>
  makeRealPAMatrix(const int spaceDim, const int fieldsPerDim, const int pointsPerDim,
                   const bool rectangular = false, const int cells = 1)
  {
    std::vector<Intrepid2::Data<Scalar,DeviceType>> leftDataComps, rightDataComps;
    for (int d=0; d<spaceDim; d++)
    {
      const int leftFields = fieldsPerDim + ((rectangular && d == 0) ? 1 : 0);
      const OneDimensionalOperator leftOp  = makeOperator(leftFields,   pointsPerDim, 0.25 + 0.1   * d);
      const OneDimensionalOperator rightOp = makeOperator(fieldsPerDim, pointsPerDim, 0.50 + 0.075 * d);

      leftDataComps.push_back(Intrepid2::Data<Scalar,DeviceType>(makeView(leftOp,  "TensorPAMatrixTests left op")));
      rightDataComps.push_back(Intrepid2::Data<Scalar,DeviceType>(makeView(rightOp, "TensorPAMatrixTests right op")));
    }

    Intrepid2::TensorData<Scalar,DeviceType> leftTensorData(leftDataComps);
    Intrepid2::TensorData<Scalar,DeviceType> rightTensorData(rightDataComps);
    Intrepid2::BasisValues<Scalar,DeviceType> leftBasisValues(leftTensorData);
    Intrepid2::BasisValues<Scalar,DeviceType> rightBasisValues(rightTensorData);

    const int totalPoints = product(std::vector<int>(spaceDim, pointsPerDim));
    View2D weightsView = makeWeights(cells, totalPoints, "TensorPAMatrixTests point weights");
    Kokkos::Array<int,2> extents {cells, totalPoints};
    Kokkos::Array<Intrepid2::DataVariationType,2> variationTypes {Intrepid2::GENERAL, Intrepid2::GENERAL};
    Intrepid2::Data<Scalar,DeviceType> weightsData(weightsView, extents, variationTypes);
    Intrepid2::TensorData<Scalar,DeviceType> cellMeasures(weightsData);

    Intrepid2::TransformedBasisValues<Scalar,DeviceType> leftTransformed(cells, leftBasisValues);
    Intrepid2::TransformedBasisValues<Scalar,DeviceType> rightTransformed(cells, rightBasisValues);

    return Intrepid2::PAMatrix<DeviceType,Scalar>(leftTransformed, cellMeasures, rightTransformed);
  }

  void testApplyMatchesAssembled(Intrepid2::PAMatrix<DeviceType,Scalar> &op, Teuchos::FancyOStream &out, bool &success)
  {
    auto matrixData = op.allocateMatrixStorage();
    op.assemble(matrixData);
    auto matrix = matrixData.getUnderlyingView();
    TEUCHOS_TEST_FOR_EXCEPTION(matrix.rank() != 3, std::invalid_argument, "PAMatrix matrix storage should expose a rank-3 underlying view for GENERAL cell weights");

    const int cells = matrixData.extent_int(0);
    const int rows  = matrixData.extent_int(1);
    const int cols  = matrixData.extent_int(2);

    Intrepid2::ScalarView<Scalar,DeviceType> input("TensorPAMatrixTests input", cells, cols);
    Intrepid2::ScalarView<Scalar,DeviceType> output("TensorPAMatrixTests output", cells, rows);
    Kokkos::deep_copy(input, 0.0);
    auto workspace = op.allocateWorkspace(cells);

    for (int j=0; j<cols; j++)
    {
      auto inputColumn = Kokkos::subview(input, Kokkos::ALL(), j);
      Kokkos::deep_copy(inputColumn, 1.0);
      op.apply(output, input, workspace);

      auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix);
      auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          const Scalar error = std::abs(outputHost(c,i) - matrixHost(c,i,j));
          if (error > 1e-11)
          {
            out << "PAMatrix apply/assembled mismatch at cell " << c << ", row " << i
                << ", column " << j << ": assembled=" << matrixHost(c,i,j)
                << ", apply=" << outputHost(c,i) << ", error=" << error << std::endl;
            success = false;
          }
        }
      }
      Kokkos::deep_copy(inputColumn, 0.0);
    }
  }

  template<class HostInputViewType>
  std::vector<Scalar> explicitTensorApply(const int spaceDim, const int fieldsPerDim, const int pointsPerDim,
                                          const bool rectangular, const int cells,
                                          const HostInputViewType &inputHost)
  {
    std::vector<OneDimensionalOperator> leftOps, rightOps;
    std::vector<int> leftExtents, rightExtents, pointExtents;
    for (int d=0; d<spaceDim; d++)
    {
      const int leftFields = fieldsPerDim + ((rectangular && d == 0) ? 1 : 0);
      leftOps.push_back(makeOperator(leftFields, pointsPerDim, 0.25 + 0.1 * d));
      rightOps.push_back(makeOperator(fieldsPerDim, pointsPerDim, 0.50 + 0.075 * d));
      leftExtents.push_back(leftFields);
      rightExtents.push_back(fieldsPerDim);
      pointExtents.push_back(pointsPerDim);
    }

    const int rows = product(leftExtents);
    const int cols = product(rightExtents);
    const int points = product(pointExtents);
    std::vector<Scalar> output(cells * rows, 0.0);

    for (int c=0; c<cells; c++)
    {
      for (int rowOrdinal=0; rowOrdinal<rows; rowOrdinal++)
      {
        const auto rowIndex = multiIndex(rowOrdinal, leftExtents);
        Scalar rowValue = 0.0;
        for (int colOrdinal=0; colOrdinal<cols; colOrdinal++)
        {
          const auto colIndex = multiIndex(colOrdinal, rightExtents);
          Scalar matrixEntry = 0.0;
          for (int pointOrdinal=0; pointOrdinal<points; pointOrdinal++)
          {
            const auto pointIndex = multiIndex(pointOrdinal, pointExtents);
            Scalar leftValue = 1.0;
            Scalar rightValue = 1.0;
            for (int d=0; d<spaceDim; d++)
            {
              leftValue  *= leftOps[d]( rowIndex[d], pointIndex[d]);
              rightValue *= rightOps[d](colIndex[d], pointIndex[d]);
            }
            const Scalar weight = 0.8 + 0.01 * ((pointOrdinal + 3 * c) % 7);
            matrixEntry += leftValue * weight * rightValue;
          }
          rowValue += matrixEntry * inputHost(c,colOrdinal);
        }
        output[c * rows + rowOrdinal] = rowValue;
      }
    }
    return output;
  }

  void testApplyMatchesExplicitReference(const int spaceDim, const int fieldsPerDim, const int pointsPerDim,
                                         const bool rectangular, const int cells,
                                         Teuchos::FancyOStream &out, bool &success)
  {
    auto op = makeRealPAMatrix(spaceDim, fieldsPerDim, pointsPerDim, rectangular, cells);

    const int cols = product(std::vector<int>(spaceDim, fieldsPerDim));
    std::vector<int> leftExtents(spaceDim, fieldsPerDim);
    if (rectangular && spaceDim > 0) leftExtents[0] = fieldsPerDim + 1;
    const int rows = product(leftExtents);
    Intrepid2::ScalarView<Scalar,DeviceType> input("TensorPAMatrixTests explicit input", cells, cols);
    Intrepid2::ScalarView<Scalar,DeviceType> output("TensorPAMatrixTests explicit output", cells, rows);
    auto inputHost = Kokkos::create_mirror_view(input);
    for (int c=0; c<cells; c++)
    {
      for (int j=0; j<cols; j++) inputHost(c,j) = 0.25 + 0.0625 * ((j + 5 * c) % 19);
    }
    Kokkos::deep_copy(input, inputHost);

    auto workspace = op.allocateWorkspace(cells);
    op.apply(output, input, workspace);

    auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    const auto expected = explicitTensorApply(spaceDim, fieldsPerDim, pointsPerDim, rectangular, cells, inputHost);
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        const Scalar error = std::abs(outputHost(c,i) - expected[c * rows + i]);
        if (error > 1e-10)
        {
          out << "PAMatrix apply/explicit-reference mismatch at cell " << c << ", row " << i
              << ": expected=" << expected[c * rows + i]
              << ", apply=" << outputHost(c,i) << ", error=" << error << std::endl;
          success = false;
        }
      }
    }
  }

  void testAssemble2MatchesAssemble(Intrepid2::PAMatrix<DeviceType,Scalar> &op,
                                    Teuchos::FancyOStream &out, bool &success)
  {
    auto expectedData = op.allocateMatrixStorage();
    auto actualData   = op.allocateMatrixStorage();
    op.assemble(expectedData);
    op.assemble2(actualData);

    TEUCHOS_TEST_FOR_EXCEPTION(expectedData.getUnderlyingViewRank() != 3 || actualData.getUnderlyingViewRank() != 3,
                               std::invalid_argument,
                               "TensorPAMatrixTests expected rank-3 matrix storage for GENERAL cell weights");
    auto expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expectedData.getUnderlyingView3());
    auto actual   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actualData.getUnderlyingView3());

    const int cells = expectedData.extent_int(0);
    const int rows  = expectedData.extent_int(1);
    const int cols  = expectedData.extent_int(2);
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        for (int j=0; j<cols; j++)
        {
          const Scalar error = std::abs(actual(c,i,j) - expected(c,i,j));
          if (error > 1e-10)
          {
            out << "PAMatrix assemble2/assemble mismatch at cell " << c << ", row " << i
                << ", column " << j << ": expected=" << expected(c,i,j)
                << ", actual=" << actual(c,i,j) << ", error=" << error << std::endl;
            success = false;
          }
        }
      }
    }
  }

  void testDiagonalMatchesAssemble(Intrepid2::PAMatrix<DeviceType,Scalar> &op,
                                   Teuchos::FancyOStream &out, bool &success)
  {
    auto matrixData = op.allocateMatrixStorage();
    op.assemble(matrixData);
    auto diagonalData = op.allocateDiagonalStorage();
    op.extractDiagonal(diagonalData);

    TEUCHOS_TEST_FOR_EXCEPTION(matrixData.getUnderlyingViewRank() != 3 || diagonalData.getUnderlyingViewRank() != 2,
                               std::invalid_argument,
                               "TensorPAMatrixTests expected rank-3 matrix and rank-2 diagonal storage for GENERAL cell weights");
    auto matrix   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.getUnderlyingView3());
    auto diagonal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.getUnderlyingView2());

    const int cells = diagonalData.extent_int(0);
    const int fields = diagonalData.extent_int(1);
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<fields; i++)
      {
        const Scalar error = std::abs(diagonal(c,i) - matrix(c,i,i));
        if (error > 1e-10)
        {
          out << "PAMatrix diagonal/assemble mismatch at cell " << c << ", field " << i
              << ": expected=" << matrix(c,i,i)
              << ", actual=" << diagonal(c,i) << ", error=" << error << std::endl;
          success = false;
        }
      }
    }
  }

  long long estimatedTensorApplyFlops(const int spaceDim, const int fieldsPerDim, const int pointsPerDim)
  {
    long long coefficientToPoint = 0;
    long long pointToCoefficient = 0;
    long long rightPartial = 1;
    for (int d=0; d<spaceDim; d++) rightPartial *= fieldsPerDim;
    long long leftPartial = 1;
    for (int d=0; d<spaceDim; d++) leftPartial *= pointsPerDim;

    for (int d=spaceDim-1; d>=0; d--)
    {
      coefficientToPoint += 2LL * pointsPerDim * rightPartial;
      rightPartial = rightPartial / fieldsPerDim * pointsPerDim;
      pointToCoefficient += 2LL * fieldsPerDim * leftPartial;
      leftPartial = leftPartial / pointsPerDim * fieldsPerDim;
    }
    long long points = 1;
    for (int d=0; d<spaceDim; d++) points *= pointsPerDim;
    return coefficientToPoint + points + pointToCoefficient;
  }
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixTensorProductApplyMatchesAssembled)
{
  auto op = makeRealPAMatrix(3, 3, 4);
  testApplyMatchesAssembled(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixRectangularTensorProductApplyMatchesAssembled)
{
  auto op = makeRealPAMatrix(2, 3, 4, true, 2);
  testApplyMatchesAssembled(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixApplyMatchesExplicitTensorReference)
{
  testApplyMatchesExplicitReference(3, 3, 4, false, 2, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixAssemble2MatchesAssemble)
{
  auto op = makeRealPAMatrix(3, 3, 4, false, 2);
  testAssemble2MatchesAssemble(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixRectangularAssemble2MatchesAssemble)
{
  auto op = makeRealPAMatrix(2, 3, 4, true, 2);
  testAssemble2MatchesAssemble(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixAssemble2MultipleColumnBlocksMatchesAssemble)
{
  auto op = makeRealPAMatrix(3, 5, 6, false, 1);
  testAssemble2MatchesAssemble(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixExtractDiagonalMatchesAssembledDiagonal)
{
  auto op = makeRealPAMatrix(3, 3, 4, false, 2);
  testDiagonalMatchesAssemble(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, RealPAMatrixRectangularExtractDiagonalMatchesAssembledDiagonal)
{
  auto op = makeRealPAMatrix(2, 3, 4, true, 2);
  testDiagonalMatchesAssemble(op, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrix, TensorContractionWorkEstimateBeatsDenseMatvec)
{
  const int dim = 4;
  const int p = 5;
  const int fieldsPerDim = p + 1;
  const int pointsPerDim = p + 2;

  // Each one-dimensional contraction costs O(p) work for each tensor line and
  // there are O(p^d) tensor entries.  This check protects the first-slice target
  // from silently replacing the matrix-free path with O(p^(2d)) assembled matvec work.
  const long long tensorApplyFlops = estimatedTensorApplyFlops(dim, fieldsPerDim, pointsPerDim);
  const long long denseMatvecFlops = 2LL * product(std::vector<int>(dim, fieldsPerDim))
                                       * product(std::vector<int>(dim, fieldsPerDim));

  TEST_ASSERT(tensorApplyFlops < denseMatvecFlops / 4);
}
