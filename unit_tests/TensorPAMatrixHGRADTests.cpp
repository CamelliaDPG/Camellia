//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_PAMatrix.hpp"

#include <Intrepid2_CellGeometryTestUtils.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_NodalBasisFamily.hpp>

#include "PAMatrixAssembly.hpp"
#include "StandardAssembly.hpp"

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <cmath>

namespace
{
  using Scalar = double;
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using BasisFamily = Intrepid2::DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;

  template<int spaceDim>
  Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> makeUnitGeometry(const int cellsPerDim)
  {
    Kokkos::Array<Scalar,spaceDim> domainExtents;
    Kokkos::Array<int,spaceDim> gridCellCounts;
    for (int d=0; d<spaceDim; d++)
    {
      domainExtents[d] = 1.0;
      gridCellCounts[d] = cellsPerDim;
    }
    auto uniformGeometry = Intrepid2::uniformCartesianMesh<Scalar,spaceDim,DeviceType>(domainExtents, gridCellCounts);
    const bool copyAffineness = false;
    return Intrepid2::getNodalCellGeometry(uniformGeometry, copyAffineness);
  }

  void compareDenseMatrices(const std::vector<Scalar> &actual,
                            const std::vector<Scalar> &expected,
                            const std::string &label,
                            Teuchos::FancyOStream &out, bool &success)
  {
    Scalar maxError = 0.0;
    Scalar maxExpected = 0.0;
    for (size_t i=0; i<actual.size(); i++)
    {
      maxError = std::max(maxError, std::abs(actual[i] - expected[i]));
      maxExpected = std::max(maxExpected, std::abs(expected[i]));
    }
    const Scalar relativeError = maxError / std::max<Scalar>(1.0, maxExpected);
    if ((maxError > 1e-10) && (relativeError > 1e-10))
    {
      out << label << " mismatch: maxAbsError=" << maxError
          << ", maxRelError=" << relativeError << std::endl;
      success = false;
    }
  }

  void addPAMatrixAssemblyToDense(Intrepid2::PAMatrix<DeviceType,Scalar> &paMatrix,
                                  const Scalar weight,
                                  std::vector<Scalar> &dense,
                                  const bool useSumFactorizedAssembly = false)
  {
    auto paMatrixData = paMatrix.allocateMatrixStorage();
    if (useSumFactorizedAssembly)
      paMatrix.assemble(paMatrixData); // sum-factorized
    else
      referenceAssembly(paMatrix, paMatrixData); // IntegrationTools
    const int cells = paMatrixData.extent_int(0);
    const int rows  = paMatrixData.extent_int(1);
    const int cols  = paMatrixData.extent_int(2);
    if (dense.empty()) dense.assign(static_cast<size_t>(cells * rows * cols), 0.0);

    if (paMatrixData.getUnderlyingViewRank() == 3)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<3>());
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
            dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * paHost(c,i,j);
    }
    else if (paMatrixData.getUnderlyingViewRank() == 2)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<2>());
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
            dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * paHost(i,j);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unexpected PAMatrix storage rank in HGRAD comparison");
    }
  }

  template<class ViewType>
  void addStandardAssemblyToDense(const ViewType &standardMatrix,
                                  const Scalar weight,
                                  std::vector<Scalar> &dense)
  {
    auto standardHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), standardMatrix);
    const int cells = standardHost.extent_int(0);
    const int rows  = standardHost.extent_int(1);
    const int cols  = standardHost.extent_int(2);
    if (dense.empty()) dense.assign(static_cast<size_t>(cells * rows * cols), 0.0);
    for (int c=0; c<cells; c++)
      for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++)
          dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * standardHost(c,i,j);
  }

  template<int spaceDim>
  void comparePAMatrixAssemblyToStandard(const Intrepid2::EOperator op,
                                         Teuchos::FancyOStream &out, bool &success,
                                         const bool useSumFactorizedAssembly = false,
                                         const int p = 2)
  {
    auto geometry = makeUnitGeometry<spaceDim>(1);
    const int worksetSize = 1;

    auto paMatrix = constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
        geometry,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, op,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, op);

    double standardFlops = 0.0;
    double jacobianFlops = 0.0;
    auto standardMatrix = performStandardAssembly<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
        geometry, worksetSize,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, op,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, op,
        standardFlops, jacobianFlops);

    std::vector<Scalar> actual;
    std::vector<Scalar> expected;
    addPAMatrixAssemblyToDense(paMatrix, 1.0, actual, useSumFactorizedAssembly);
    addStandardAssemblyToDense(standardMatrix, 1.0, expected);
    const std::string label = useSumFactorizedAssembly ? "HGRAD PAMatrix assemble/standard assembly"
                                           : "HGRAD PAMatrix/standard assembly";
    compareDenseMatrices(actual, expected, label, out, success);
  }

  template<int spaceDim>
  void comparePAMatrixAssemblyToStandardForPRange(const Intrepid2::EOperator op,
                                                  const int firstP, const int lastP,
                                                  Teuchos::FancyOStream &out, bool &success,
                                                  const bool useSumFactorizedAssembly = false)
  {
    for (int p=firstP; p<=lastP; p++)
    {
      comparePAMatrixAssemblyToStandard<spaceDim>(op, out, success, useSumFactorizedAssembly, p);
    }
  }

  template<int spaceDim>
  void compareWeightedH1GramToStandard(Teuchos::FancyOStream &out, bool &success,
                                        const bool useSumFactorizedAssembly = false,
                                        const int p = 2)
  {
    auto geometry = makeUnitGeometry<spaceDim>(2);
    const int worksetSize = geometry.numCells();
    const Scalar massWeight = 2.0;
    const Scalar stiffnessWeight = 0.5;

    std::vector<Scalar> actual;
    std::vector<Scalar> expected;

    auto paMass = constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
        geometry,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE);
    addPAMatrixAssemblyToDense(paMass, massWeight, actual, useSumFactorizedAssembly);

    double standardFlops = 0.0;
    double jacobianFlops = 0.0;
    auto standardMass = performStandardAssembly<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
        geometry, worksetSize,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE,
        p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE,
        standardFlops, jacobianFlops);
    addStandardAssemblyToDense(standardMass, massWeight, expected);

    for (int d=0; d<spaceDim; d++)
    {
      auto vectorWeight = Teuchos::rcp(new Kokkos::Array<Scalar,spaceDim>());
      for (int e=0; e<spaceDim; e++) (*vectorWeight)[e] = (e == d) ? Scalar(1.0) : Scalar(0.0);

      auto paStiffness = constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
          geometry,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight);
      addPAMatrixAssemblyToDense(paStiffness, stiffnessWeight, actual, useSumFactorizedAssembly);

      auto standardStiffness = performStandardAssembly<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
          geometry, worksetSize,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight,
          standardFlops, jacobianFlops);
      addStandardAssemblyToDense(standardStiffness, stiffnessWeight, expected);
    }

    compareDenseMatrices(actual, expected, "Weighted component-split H1 Gram PAMatrix/standard assembly", out, success);
  }

  template<int spaceDim>
  void compareWeightedH1GramToStandardForPRange(const int firstP, const int lastP,
                                                Teuchos::FancyOStream &out, bool &success,
                                                const bool useSumFactorizedAssembly = false)
  {
    for (int p=firstP; p<=lastP; p++)
    {
      compareWeightedH1GramToStandard<spaceDim>(out, success, useSumFactorizedAssembly, p);
    }
  }
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, MassMatchesStandardIntegration2D)
{
  comparePAMatrixAssemblyToStandard<2>(Intrepid2::OPERATOR_VALUE, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, StiffnessMatchesStandardIntegration2D)
{
  comparePAMatrixAssemblyToStandard<2>(Intrepid2::OPERATOR_GRAD, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, StiffnessMatchesStandardIntegration3D)
{
  comparePAMatrixAssemblyToStandard<3>(Intrepid2::OPERATOR_GRAD, out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, WeightedH1GramComponentSplitMatchesStandardIntegration3D)
{
  compareWeightedH1GramToStandard<3>(out, success);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, MassSumFactorizedAssembleMatchesStandardIntegration1D)
{
  comparePAMatrixAssemblyToStandard<1>(Intrepid2::OPERATOR_VALUE, out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, MassSumFactorizedAssembleMatchesStandardIntegration2D)
{
  comparePAMatrixAssemblyToStandard<2>(Intrepid2::OPERATOR_VALUE, out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, MassSumFactorizedAssembleMatchesStandardIntegration3D)
{
  comparePAMatrixAssemblyToStandard<3>(Intrepid2::OPERATOR_VALUE, out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, StiffnessSumFactorizedAssembleMatchesStandardIntegration2D)
{
  comparePAMatrixAssemblyToStandard<2>(Intrepid2::OPERATOR_GRAD, out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, StiffnessSumFactorizedAssembleMatchesStandardIntegration3D)
{
  comparePAMatrixAssemblyToStandard<3>(Intrepid2::OPERATOR_GRAD, out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, WeightedH1GramComponentSplitSumFactorizedAssembleMatchesStandardIntegration1D)
{
  compareWeightedH1GramToStandard<1>(out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, WeightedH1GramComponentSplitSumFactorizedAssembleMatchesStandardIntegration2D)
{
  compareWeightedH1GramToStandard<2>(out, success, true);
}

TEUCHOS_UNIT_TEST(TensorPAMatrixHGRAD, WeightedH1GramComponentSplitSumFactorizedAssembleMatchesStandardIntegration3D)
{
  compareWeightedH1GramToStandard<3>(out, success, true);
}

// Broader p-range sum-factorized assemble()/standard-reference tests were attempted here, but
// the standard-assembly HGRAD reference path currently fails in OrientationTools
// coefficient-matrix construction before assemble() is meaningfully exercised.
// Keep the stable explicit 1D/2D/3D sum-factorized assemble() checks above until the reference
// path is fixed or a non-OrientationTools p-sweep reference is added.
