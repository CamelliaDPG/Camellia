//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_PAMatrix.hpp"
#include "Intrepid2_PAMatrixDefAdditions.hpp"

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

  template<int spaceDim>
  void comparePAMatrixAssemblyToStandard(const Intrepid2::EOperator op,
                                         Teuchos::FancyOStream &out, bool &success)
  {
    auto geometry = makeUnitGeometry<spaceDim>(1);
    constexpr int p = 2;
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

    auto paMatrixData = paMatrix.allocateMatrixStorage();
    paMatrix.assemble(paMatrixData);

    auto standardHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), standardMatrix);
    const int cells = paMatrixData.extent_int(0);
    const int rows  = paMatrixData.extent_int(1);
    const int cols  = paMatrixData.extent_int(2);

    Scalar maxError = 0.0;
    Scalar maxExpected = 0.0;
    if (paMatrixData.getUnderlyingViewRank() == 3)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<3>());
      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          for (int j=0; j<cols; j++)
          {
            const Scalar expected = standardHost(c,i,j);
            const Scalar actual = paHost(c,i,j);
            maxError = std::max(maxError, std::abs(actual - expected));
            maxExpected = std::max(maxExpected, std::abs(expected));
          }
        }
      }
    }
    else if (paMatrixData.getUnderlyingViewRank() == 2)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<2>());
      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          for (int j=0; j<cols; j++)
          {
            const Scalar expected = standardHost(c,i,j);
            const Scalar actual = paHost(i,j);
            maxError = std::max(maxError, std::abs(actual - expected));
            maxExpected = std::max(maxExpected, std::abs(expected));
          }
        }
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unexpected PAMatrix storage rank in HGRAD comparison");
    }

    const Scalar relativeError = maxError / std::max<Scalar>(1.0, maxExpected);
    if ((maxError > 1e-10) && (relativeError > 1e-10))
    {
      out << "HGRAD PAMatrix/standard assembly mismatch: maxAbsError=" << maxError
          << ", maxRelError=" << relativeError << std::endl;
      success = false;
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
