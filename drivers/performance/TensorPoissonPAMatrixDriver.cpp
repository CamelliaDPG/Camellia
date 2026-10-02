//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Intrepid2_PAMatrix.hpp"

#include "ApplyInverseTpetraOperatorAdapter.h"
#include "BF.h"
#include "BC.h"
#include "Basis.h"
#include "Camellia.h"
#include "Function.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "Solution.h"
#include "TensorPAMatrixTpetraAdapters.hpp"
#include "VarFactory.h"

#include <Intrepid2_CellGeometryTestUtils.hpp>
#include <Intrepid2_DerivedBasisFamily.hpp>
#include <Intrepid2_HierarchicalBasisProlongator.hpp>

#include "TensorBilinearForm.h"
#include "TensorFunction.h"
#include "TensorPAMatrixBridge.h"
#include "Var.h"

#include "Kokkos_Core.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosSolverFactory_Tpetra.hpp>
#include <Ifpack2_Chebyshev.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <Tpetra_Vector.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Camellia;

namespace
{
  using Scalar = double;
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using View2D = Kokkos::View<Scalar**,DeviceType>;
  using PAMatrix = Intrepid2::PAMatrix<DeviceType,Scalar>;
  using TpetraMap = Tpetra::Map<>;
  using TpetraCrsGraph = Tpetra::CrsGraph<typename TpetraMap::local_ordinal_type, typename TpetraMap::global_ordinal_type, typename TpetraMap::node_type>;
  using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, typename TpetraMap::local_ordinal_type, typename TpetraMap::global_ordinal_type, typename TpetraMap::node_type>;
  using TpetraMultiVector = Tpetra::MultiVector<Scalar>;
  using TpetraOperator = Tpetra::Operator<Scalar>;
  using TpetraRowMatrix = Tpetra::RowMatrix<Scalar>;
  using TpetraVector = Tpetra::Vector<Scalar>;

  using OperatorTerm = Camellia::TensorPAMatrixTerm<DeviceType,Scalar>;

  using TpetraAdapters = Camellia::TensorPAMatrixTpetraAdapters;
  using WorkspaceView = TpetraAdapters::WorkspaceView;

  class AnalyticGraphNormSolutionCoefficientEvaluator : public Camellia::TensorSolutionCoefficientEvaluator
  {
    std::string _kind;
  public:
    explicit AnalyticGraphNormSolutionCoefficientEvaluator(const std::string &kind)
    : _kind(kind)
    {}

    void values(int componentOrdinal,
                Camellia::TensorFunction::PointViewHost pointsIn,
                Camellia::TensorFunction::ValueViewHost valuesOut) const override
    {
      const int pointCount = static_cast<int>(pointsIn.extent(0));
      const int pointDim = static_cast<int>(pointsIn.extent(1));
      if (static_cast<int>(valuesOut.extent(0)) != pointCount)
        throw std::invalid_argument("AnalyticGraphNormSolutionCoefficientEvaluator output length mismatch");
      if (pointDim < 4)
        throw std::invalid_argument("AnalyticGraphNormSolutionCoefficientEvaluator requires 4D points ordered (x,t,ux,uy)");

      for (int p=0; p<pointCount; p++)
      {
        const auto physicalCoordinate = [&](const int d)
        {
          return 0.5 * (pointsIn(p,d) + 1.0);
        };
        if (_kind == "velocity-component")
        {
          if ((componentOrdinal < 0) || (componentOrdinal >= pointDim))
            throw std::invalid_argument("AnalyticGraphNormSolutionCoefficientEvaluator velocity component is outside point dimension");
          valuesOut(p) = physicalCoordinate(componentOrdinal);
        }
        else if (_kind == "maxwell-ax-proxy")
        {
          valuesOut(p) = 0.5 * (physicalCoordinate(0) + physicalCoordinate(1));
        }
        else if (_kind == "maxwell-ay-proxy")
        {
          valuesOut(p) = 0.75 * physicalCoordinate(1);
        }
        else
        {
          throw std::invalid_argument("Unknown analytic graph-norm solution coefficient evaluator kind");
        }
      }
    }
  };

  int checkedProduct(const std::vector<int> &values)
  {
    long long result = 1;
    for (const int value : values)
    {
      result *= value;
      if (result > std::numeric_limits<int>::max())
      {
        throw std::overflow_error("tensor extent exceeds int range");
      }
    }
    return static_cast<int>(result);
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

  struct PolyOrderRange
  {
    int first = 1;
    int last = 4;
  };

  PolyOrderRange parsePolyOrderRange(const std::string &rangeText)
  {
    const auto colon = rangeText.find(':');
    if (colon == std::string::npos)
    {
      const int p = std::stoi(rangeText);
      return {p,p};
    }
    return {std::stoi(rangeText.substr(0,colon)), std::stoi(rangeText.substr(colon+1))};
  }

  View2D makeOneDimensionalOperator(const int fields, const int points, const bool derivative, const std::string &label)
  {
    View2D view(label, fields, points);
    auto host = Kokkos::create_mirror_view(view);
    for (int f=0; f<fields; f++)
    {
      for (int p=0; p<points; p++)
      {
        const Scalar xi = (points == 1) ? 0.0 : -1.0 + 2.0 * p / Scalar(points - 1);
        if (derivative)
        {
          // A small deterministic derivative-like table for the synthetic contraction
          // stress-test mode.  The default hgrad mode exercises true Intrepid2 HGRAD
          // basis values, cubature, and affine tensor-product geometry.
          host(f,p) = (f + 1) * std::pow(xi + 1.25, std::max(0, f - 1));
        }
        else
        {
          host(f,p) = std::pow(xi + 1.25, f);
        }
      }
    }
    Kokkos::deep_copy(view, host);
    return view;
  }

  View2D makePointWeights(const int cells, const int points, const std::string &label)
  {
    View2D weights(label, cells, points);
    auto host = Kokkos::create_mirror_view(weights);
    for (int c=0; c<cells; c++)
    {
      for (int p=0; p<points; p++)
      {
        host(c,p) = 1.0 + 0.01 * ((p + 5 * c) % 11);
      }
    }
    Kokkos::deep_copy(weights, host);
    return weights;
  }

  PAMatrix makePAMatrix(const int dim, const int p, const int cells, const int derivativeDirection)
  {
    const int fieldsPerDim = p + 1;
    const int pointsPerDim = p + 2;

    std::vector<Intrepid2::Data<Scalar,DeviceType>> leftDataComps, rightDataComps;
    for (int d=0; d<dim; d++)
    {
      const bool derivative = (d == derivativeDirection);
      leftDataComps.push_back(Intrepid2::Data<Scalar,DeviceType>(
          makeOneDimensionalOperator(fieldsPerDim, pointsPerDim, derivative, "TensorPoissonPAMatrix left op")));
      rightDataComps.push_back(Intrepid2::Data<Scalar,DeviceType>(
          makeOneDimensionalOperator(fieldsPerDim, pointsPerDim, derivative, "TensorPoissonPAMatrix right op")));
    }

    Intrepid2::TensorData<Scalar,DeviceType> leftTensorData(leftDataComps);
    Intrepid2::TensorData<Scalar,DeviceType> rightTensorData(rightDataComps);
    Intrepid2::BasisValues<Scalar,DeviceType> leftBasisValues(leftTensorData);
    Intrepid2::BasisValues<Scalar,DeviceType> rightBasisValues(rightTensorData);

    const int totalPoints = checkedProduct(std::vector<int>(dim, pointsPerDim));
    View2D weightsView = makePointWeights(cells, totalPoints, "TensorPoissonPAMatrix weights");
    Kokkos::Array<int,2> extents {cells, totalPoints};
    Kokkos::Array<Intrepid2::DataVariationType,2> variationTypes {Intrepid2::GENERAL, Intrepid2::GENERAL};
    Intrepid2::Data<Scalar,DeviceType> weightsData(weightsView, extents, variationTypes);
    Intrepid2::TensorData<Scalar,DeviceType> cellMeasures(weightsData);

    Intrepid2::TransformedBasisValues<Scalar,DeviceType> leftTransformed(cells, leftBasisValues);
    Intrepid2::TransformedBasisValues<Scalar,DeviceType> rightTransformed(cells, rightBasisValues);
    return PAMatrix(leftTransformed, cellMeasures, rightTransformed);
  }

  bool operatorIncludesMass(const std::string &operatorChoice)
  {
    return (operatorChoice == "mass") || (operatorChoice == "mass-plus-stiffness") || (operatorChoice == "h1-gram");
  }

  bool operatorIncludesStiffness(const std::string &operatorChoice)
  {
    return (operatorChoice == "stiffness") || (operatorChoice == "mass-plus-stiffness") || (operatorChoice == "h1-gram");
  }

  std::vector<Scalar> makeConstantTestFunctionRhs(const std::vector<OperatorTerm> &terms,
                                                  const std::vector<WorkspaceView> &workspaces,
                                                  const int dim,
                                                  const int p,
                                                  const int cells,
                                                  const int rows,
                                                  const int worksetSize)
  {
    std::vector<Scalar> rhs(static_cast<size_t>(cells * rows), Scalar(0.0));

    const int fieldsPerDim = p + 1;
    std::vector<Scalar> constantCoefficients(static_cast<size_t>(rows), Scalar(0.0));
    const std::vector<int> basisExtents(static_cast<size_t>(dim), fieldsPerDim);

    for (int i=0; i<rows; i++)
    {
      const std::vector<int> basisOrdinal = multiIndex(i, basisExtents);

      bool isVertexTensorBasis = true;
      for (const int componentOrdinal : basisOrdinal)
      {
        if (componentOrdinal > 1)
        {
          isVertexTensorBasis = false;
          break;
        }
      }

      if (isVertexTensorBasis)
      {
        constantCoefficients[static_cast<size_t>(i)] = Scalar(1.0);
      }
    }

    Intrepid2::ScalarView<Scalar,DeviceType> constantInput(
        "TensorPoissonPAMatrix constant coefficient input", cells, rows);
    Intrepid2::ScalarView<Scalar,DeviceType> massOutput(
        "TensorPoissonPAMatrix constant test rhs output", cells, rows);

    auto constantInputHost = Kokkos::create_mirror_view(constantInput);
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        constantInputHost(c,i) = constantCoefficients[static_cast<size_t>(i)];
      }
    }
    Kokkos::deep_copy(constantInput, constantInputHost);

    bool foundMassTerm = false;
    for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
    {
      const std::string &label = terms[termOrdinal].label;
      if ((label.find("mass") == std::string::npos) &&
          (label.find("l2") == std::string::npos))
      {
        continue;
      }

      massOutput = Intrepid2::ScalarView<Scalar,DeviceType>(
          "TensorPoissonPAMatrix constant test rhs output term", cells, rows);

      terms[termOrdinal].matrix.apply(massOutput,
                                      constantInput,
                                      workspaces[termOrdinal],
                                      false,
                                      worksetSize);
      Kokkos::fence();

      auto massOutputHost =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), massOutput);

      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          rhs[static_cast<size_t>(c * rows + i)] += massOutputHost(c,i);
        }
      }

      foundMassTerm = true;
    }

    if (!foundMassTerm)
    {
      throw std::invalid_argument(
          "makeConstantTestFunctionRhs requires an HGRAD mass/L2 term in the operator");
    }

    return rhs;
  }

  std::vector<OperatorTerm> makeSyntheticOperatorTerms(const int dim, const int p, const int cells, const std::string &operatorChoice,
                                                        const Scalar massWeight, const Scalar stiffnessWeight)
  {
    std::vector<OperatorTerm> terms;
    if (operatorIncludesMass(operatorChoice))
    {
      terms.push_back({makePAMatrix(dim, p, cells, -1), massWeight, "mass"});
    }
    if (operatorIncludesStiffness(operatorChoice))
    {
      for (int d=0; d<dim; d++)
      {
        terms.push_back({makePAMatrix(dim, p, cells, d), stiffnessWeight, "stiffness" + std::to_string(d)});
      }
    }
    if (terms.empty())
    {
      throw std::invalid_argument("--operator must be mass, stiffness, mass-plus-stiffness, or h1-gram");
    }
    return terms;
  }

  bool basisModeIsHGRAD(const std::string &basisMode)
  {
    return (basisMode == "hgrad") ||
           (basisMode == "hgrad-hierarchical") ||
           (basisMode == "hgrad-nodal");
  }

  bool inverseAlgorithmUsesPMultigrid(const std::string &algorithm)
  {
    return (algorithm == "matrix-free-gmg-chebyshev-smoother") ||
           (algorithm == "matrix-free-gmg-jacobi-smoother") ||
           (algorithm == "gmres-pmultigrid-chebyshev") ||
           (algorithm == "gmres-pmultigrid-jacobi-chebyshev") ||
           (algorithm == "pmultigrid-chebyshev") ||
           (algorithm == "pmultigrid-jacobi-chebyshev");
  }

  Camellia::TensorBasisFamily TensorBasisFamilyFromMode(
      const std::string &basisMode)
  {
    if ((basisMode == "hgrad") || (basisMode == "hgrad-hierarchical"))
    {
      return Camellia::TensorBasisFamily::Hierarchical;
    }

    if (basisMode == "hgrad-nodal")
    {
      return Camellia::TensorBasisFamily::Nodal;
    }

    throw std::invalid_argument(
        "TensorBasisFamilyFromMode requires hgrad, hgrad-hierarchical, or hgrad-nodal");
  }

  std::string TensorBasisFamilyName(
      const Camellia::TensorBasisFamily basisFamily)
  {
    switch (basisFamily)
    {
      case Camellia::TensorBasisFamily::Hierarchical:
        return "intrepid2-hierarchical";
      case Camellia::TensorBasisFamily::Nodal:
        return "intrepid2-nodal";
      default:
        return "unknown";
    }
  }


  // HGRAD cubature/coefficient folding and PAMatrix construction are provided by TensorPAMatrixBridge.h.

  // HGRAD tensor field variable construction is provided by TensorPAMatrixBridge.h.

  // Tensor descriptor mapping and derivative vector weights are provided by TensorPAMatrixBridge.h.

  // Coefficient decomposition for tensor bridge terms is provided by TensorPAMatrixBridge.h.

  // Tensor-term-to-PAMatrix adapter wrappers are provided by TensorPAMatrixBridge.h.

  template<int spaceDim>
  std::vector<OperatorTerm> makeHGRADOperatorTermsForDim(const int p, const int cellsPerDim, const std::string &operatorChoice,
                                                         const Scalar massWeight, const Scalar stiffnessWeight,
                                                         const Camellia::TensorBasisFamily basisFamily)
  {
    Kokkos::Array<Scalar,spaceDim> origin;
    Kokkos::Array<Scalar,spaceDim> domainExtents;
    Kokkos::Array<int,spaceDim> gridCellCounts;
    for (int d=0; d<spaceDim; d++)
    {
      origin[d] = 0.0;
      domainExtents[d] = 1.0;
      gridCellCounts[d] = cellsPerDim;
    }

    using CellGeometry = Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType>;
    const auto nodeOrdering = (spaceDim <= 3)
                            ? CellGeometry::HYPERCUBE_NODE_ORDER_CLASSIC_SHARDS
                            : CellGeometry::HYPERCUBE_NODE_ORDER_TENSOR;
    auto geometry = CellGeometry(origin, domainExtents, gridCellCounts,
                                 CellGeometry::NO_SUBDIVISION, nodeOrdering);

    std::vector<OperatorTerm> terms;
    if ((operatorChoice == "vlasov-dt") || (operatorChoice == "vlasov-coordinate") || (operatorChoice == "vlasov-linear-x0-x1-d2") || (operatorChoice == "vlasov-coordinate-x1-d3") || (operatorChoice == "vlasov-graph-norm-1d1t2v") || (operatorChoice == "vlasov-graph-norm-solution-coeff-1d1t2v"))
    {
      if (spaceDim < 2) throw std::invalid_argument("--operator=vlasov-dt and --operator=vlasov-coordinate require --dim >= 2 so derivative component 1 is present");
      if ((operatorChoice == "vlasov-linear-x0-x1-d2") && (spaceDim < 3))
        throw std::invalid_argument("--operator=vlasov-linear-x0-x1-d2 requires --dim >= 3 so derivative component 2 is present");
      if ((operatorChoice == "vlasov-coordinate-x1-d3") && (spaceDim < 4))
        throw std::invalid_argument("--operator=vlasov-coordinate-x1-d3 requires --dim >= 4 so derivative component 3 is present");
      if (((operatorChoice == "vlasov-graph-norm-1d1t2v") || (operatorChoice == "vlasov-graph-norm-solution-coeff-1d1t2v")) && (spaceDim != 4))
        throw std::invalid_argument("--operator=vlasov-graph-norm-1d1t2v and --operator=vlasov-graph-norm-solution-coeff-1d1t2v require --dim=4 with component ordering (x,t,ux,uy)");
      auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
      auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
      if ((operatorChoice == "vlasov-graph-norm-1d1t2v") || (operatorChoice == "vlasov-graph-norm-solution-coeff-1d1t2v"))
      {
        const bool useSolutionCoefficientProxies = (operatorChoice == "vlasov-graph-norm-solution-coeff-1d1t2v");
        // Matrix-free viability proxy for the Planar MITL Vlasov test-space graph norm on
        // a tensor cell with component ordering (x,t,ux,uy).  The production relativistic
        // graph norm is ||-w_t - v_x w_x - a_x w_ux - a_y w_uy||^2 + L2WeightGraph ||w||^2.
        // This diagnostic keeps the full expanded derivative-pair structure.  The default
        // operator uses bounded analytic coefficients directly; the solution-coeff variant
        // evaluates the same proxies through TensorFunctionEvaluationContext so the
        // solution-dependent coefficient PAMatrix path is exercised before live Maxwell
        // TensorSolution coefficients are wired in.
        terms.push_back({Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
            geometry, p, Intrepid2::OPERATOR_VALUE, Intrepid2::OPERATOR_VALUE, basisFamily), massWeight, "vlasov_graph_norm_l2"});

        Camellia::TensorFunctionEvaluationContext solutionCoefficientContext;
        Camellia::TensorFunctionEvaluationContext *graphEvaluationContext = nullptr;
        Camellia::TensorFunctionDependencySource vxSource(Camellia::TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "graph_norm_proxy_vx");
        Camellia::TensorFunctionDependencySource axSource(Camellia::TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH, "graph_norm_proxy_ax");
        Camellia::TensorFunctionDependencySource aySource(Camellia::TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH, "graph_norm_proxy_ay");
        if (useSolutionCoefficientProxies)
        {
          solutionCoefficientContext.registerSolutionCoefficientEvaluator(vxSource, Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("velocity-component")));
          solutionCoefficientContext.registerSolutionCoefficientEvaluator(axSource, Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("maxwell-ax-proxy")));
          solutionCoefficientContext.registerSolutionCoefficientEvaluator(aySource, Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("maxwell-ay-proxy")));
          graphEvaluationContext = &solutionCoefficientContext;
        }

        const std::vector<int> derivativeComponents {1, 0, 2, 3};
        const std::vector<std::string> derivativeLabels {"dt", "dx", "dux", "duy"};
        const std::vector<Camellia::TensorFunctionPtr> coefficients = useSolutionCoefficientProxies
          ? std::vector<Camellia::TensorFunctionPtr>{
              Camellia::TensorFunction::constant(-1.0),
              Camellia::TensorFunction::constant(-1.0) * Camellia::TensorFunction::solutionCoefficient1D(2, vxSource),
              Camellia::TensorFunction::constant(-1.0) * Camellia::TensorFunction::solutionCoefficient1D(0, axSource),
              Camellia::TensorFunction::constant(-1.0) * Camellia::TensorFunction::solutionCoefficient1D(1, aySource)
            }
          : std::vector<Camellia::TensorFunctionPtr>{
              Camellia::TensorFunction::constant(-1.0),
              Camellia::TensorFunction::constant(-0.5) * (Camellia::TensorFunction::componentCoordinate(2) + Camellia::TensorFunction::constant(1.0)),
              Camellia::TensorFunction::constant(-0.25) * (Camellia::TensorFunction::componentCoordinate(0) + Camellia::TensorFunction::componentCoordinate(1) + Camellia::TensorFunction::constant(2.0)),
              Camellia::TensorFunction::constant(-0.375) * (Camellia::TensorFunction::componentCoordinate(1) + Camellia::TensorFunction::constant(1.0))
            };

        for (size_t trialComponent=0; trialComponent<derivativeComponents.size(); trialComponent++)
        {
          for (size_t testComponent=0; testComponent<derivativeComponents.size(); testComponent++)
          {
            Camellia::TensorBilinearTerm graphTerm(trialVar,
                                                   testVar,
                                                   Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim, derivativeComponents[trialComponent]),
                                                   Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim, derivativeComponents[testComponent]),
                                                   coefficients[trialComponent] * coefficients[testComponent],
                                                   Camellia::TENSOR_BILINEAR_TERM_VOLUME,
                                                   Camellia::TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());
            const size_t firstNewTerm = terms.size();
            Camellia::appendTensorVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(terms, geometry, p, graphTerm,
                                                             "vlasov_graph_norm_" + derivativeLabels[trialComponent] + "_" + derivativeLabels[testComponent],
                                                             graphEvaluationContext, basisFamily);
            for (size_t termOrdinal=firstNewTerm; termOrdinal<terms.size(); termOrdinal++)
              terms[termOrdinal].weight *= stiffnessWeight;
          }
        }
        return terms;
      }
      if (operatorChoice == "vlasov-coordinate-x1-d3")
      {
        Camellia::TensorBilinearTerm vlasovCoordinateD3Term(trialVar,
                                                            testVar,
                                                            Camellia::TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                                            Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 3),
                                                            Camellia::TensorFunction::constant(-0.75) * Camellia::TensorFunction::componentCoordinate(1),
                                                            Camellia::TENSOR_BILINEAR_TERM_VOLUME,
                                                            Camellia::TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());
        Camellia::appendTensorVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(terms, geometry, p, vlasovCoordinateD3Term,
                                                                                     "vlasov_coordinate_x1_value_test_d3",
                                                                                     nullptr, basisFamily);
        return terms;
      }
      if (operatorChoice == "vlasov-linear-x0-x1-d2")
      {
        Camellia::TensorBilinearTerm vlasovLinearTerm(trialVar,
                                                      testVar,
                                                      Camellia::TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                                      Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 2),
                                                      Camellia::TensorFunction::constant(-0.5) * (Camellia::TensorFunction::componentCoordinate(0) + Camellia::TensorFunction::componentCoordinate(1)),
                                                      Camellia::TENSOR_BILINEAR_TERM_VOLUME,
                                                      Camellia::TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());
        Camellia::appendTensorVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(terms, geometry, p, vlasovLinearTerm,
                                                                                     "vlasov_linear_x0_plus_x1_value_test_d2",
                                                                                     nullptr, basisFamily);
        return terms;
      }

      const auto coefficient = (operatorChoice == "vlasov-coordinate")
                             ? Camellia::TensorFunction::componentCoordinate(0)
                             : Camellia::TensorFunction::constant(-1.0);
      Camellia::TensorBilinearTerm vlasovDtTerm(trialVar,
                                                 testVar,
                                                 Camellia::TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                                 Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 1),
                                                 coefficient,
                                                 Camellia::TENSOR_BILINEAR_TERM_VOLUME,
                                                 Camellia::TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());
      Camellia::appendTensorVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(terms, geometry, p, vlasovDtTerm,
                                                                                   (operatorChoice == "vlasov-coordinate")
                                                                                   ? "vlasov_coordinate_x0_value_test_d1"
                                                                                   : "vlasov_dt_value_test_d1",
                                                                                   nullptr, basisFamily);
      return terms;
    }
    if (operatorIncludesMass(operatorChoice))
    {
      terms.push_back({Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
          geometry, p, Intrepid2::OPERATOR_VALUE, Intrepid2::OPERATOR_VALUE, basisFamily), massWeight, "mass"});
    }
    if (operatorIncludesStiffness(operatorChoice))
    {
      if (operatorChoice == "h1-gram")
      {
        for (int d=0; d<spaceDim; d++)
        {
          auto vectorWeight = Teuchos::rcp(new Kokkos::Array<Scalar,spaceDim>());
          for (int e=0; e<spaceDim; e++) (*vectorWeight)[e] = (e == d) ? Scalar(1.0) : Scalar(0.0);
          
          terms.push_back({Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
              geometry, p,
              Intrepid2::OPERATOR_GRAD, vectorWeight,
              Intrepid2::OPERATOR_GRAD, vectorWeight,
              basisFamily), stiffnessWeight, "stiffness" + std::to_string(d)});
        }
      }
      else
      {
        terms.push_back({Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
            geometry, p, Intrepid2::OPERATOR_GRAD, Intrepid2::OPERATOR_GRAD, basisFamily), stiffnessWeight, "stiffness"});
      }
    }
    if (terms.empty())
    {
      throw std::invalid_argument("--operator must be mass, stiffness, mass-plus-stiffness, h1-gram, vlasov-dt, vlasov-coordinate, vlasov-linear-x0-x1-d2, vlasov-coordinate-x1-d3, vlasov-graph-norm-1d1t2v, or vlasov-graph-norm-solution-coeff-1d1t2v");
    }
    return terms;
  }

  std::vector<OperatorTerm> makeHGRADOperatorTerms(const int dim, const int p, const int cellsPerDim, const std::string &operatorChoice,
                                                   const Scalar massWeight, const Scalar stiffnessWeight,
                                                   const Camellia::TensorBasisFamily basisFamily)
  {
    switch (dim)
    {
      case 1: return makeHGRADOperatorTermsForDim<1>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 2: return makeHGRADOperatorTermsForDim<2>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 3: return makeHGRADOperatorTermsForDim<3>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 4: return makeHGRADOperatorTermsForDim<4>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 5: return makeHGRADOperatorTermsForDim<5>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 6: return makeHGRADOperatorTermsForDim<6>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      case 7: return makeHGRADOperatorTermsForDim<7>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight, basisFamily);
      default:
        throw std::invalid_argument("--basisFamily=hgrad, hgrad-hierarchical, or hgrad-nodal supports --dim=1 through 7");
    }
  }

  std::vector<OperatorTerm> makeOperatorTerms(const int dim, const int p, const int cellsPerDim, const int cells,
                                              const std::string &operatorChoice, const std::string &basisMode,
                                              const Scalar massWeight, const Scalar stiffnessWeight)
  {
    if (basisMode == "synthetic")
    {
      return makeSyntheticOperatorTerms(
          dim, p, cells, operatorChoice, massWeight, stiffnessWeight);
    }

    if (basisModeIsHGRAD(basisMode))
    {
      return makeHGRADOperatorTerms(
          dim,
          p,
          cellsPerDim,
          operatorChoice,
          massWeight,
          stiffnessWeight,
          TensorBasisFamilyFromMode(basisMode));
    }

    throw std::invalid_argument(
        "--basisFamily must be synthetic, hgrad, hgrad-hierarchical, or hgrad-nodal");
  }

  enum class VlasovCamelliaComparisonBlock
  {
    Mass,
    DtDt,
    DtDx,
    DtDux,
    DtDuy,
    DxDt,
    DxDx,
    DxDux,
    DxDuy,
    DuxDt,
    DuxDx,
    DuxDux,
    DuxDuy,
    DuyDt,
    DuyDx,
    DuyDux,
    DuyDuy,
    FullGraphNorm
  };

  bool vlasovCamelliaComparisonBlockDerivativePair(const VlasovCamelliaComparisonBlock block,
                                                   size_t &trialComponent, size_t &testComponent)
  {
    switch (block)
    {
      case VlasovCamelliaComparisonBlock::DtDt:   trialComponent = 0; testComponent = 0; return true;
      case VlasovCamelliaComparisonBlock::DtDx:   trialComponent = 0; testComponent = 1; return true;
      case VlasovCamelliaComparisonBlock::DtDux:  trialComponent = 0; testComponent = 2; return true;
      case VlasovCamelliaComparisonBlock::DtDuy:  trialComponent = 0; testComponent = 3; return true;
      case VlasovCamelliaComparisonBlock::DxDt:   trialComponent = 1; testComponent = 0; return true;
      case VlasovCamelliaComparisonBlock::DxDx:   trialComponent = 1; testComponent = 1; return true;
      case VlasovCamelliaComparisonBlock::DxDux:  trialComponent = 1; testComponent = 2; return true;
      case VlasovCamelliaComparisonBlock::DxDuy:  trialComponent = 1; testComponent = 3; return true;
      case VlasovCamelliaComparisonBlock::DuxDt:  trialComponent = 2; testComponent = 0; return true;
      case VlasovCamelliaComparisonBlock::DuxDx:  trialComponent = 2; testComponent = 1; return true;
      case VlasovCamelliaComparisonBlock::DuxDux: trialComponent = 2; testComponent = 2; return true;
      case VlasovCamelliaComparisonBlock::DuxDuy: trialComponent = 2; testComponent = 3; return true;
      case VlasovCamelliaComparisonBlock::DuyDt:  trialComponent = 3; testComponent = 0; return true;
      case VlasovCamelliaComparisonBlock::DuyDx:  trialComponent = 3; testComponent = 1; return true;
      case VlasovCamelliaComparisonBlock::DuyDux: trialComponent = 3; testComponent = 2; return true;
      case VlasovCamelliaComparisonBlock::DuyDuy: trialComponent = 3; testComponent = 3; return true;
      default: return false;
    }
  }

  struct DenseMatrixComparison
  {
    double maxAbsError = -1.0;
    double maxRelError = -1.0;
    int maxErrorRow = -1;
    int maxErrorCol = -1;
    double matrixFreeAtMaxError = 0.0;
    double camelliaAtMaxError = 0.0;
    double matrixFreeTrace = 0.0;
    double camelliaTrace = 0.0;
    double matrixFreeTotal = 0.0;
    double camelliaTotal = 0.0;
    double maxDiagonalAbsError = 0.0;
    double maxRowSumAbsError = 0.0;
    double matrixFreeSymmetryMaxAbsError = 0.0;
    double camelliaSymmetryMaxAbsError = 0.0;
    double crossTransposeMaxAbsError = 0.0;
    double matrixFreeDiagonalMin = 0.0;
    double matrixFreeDiagonalMax = 0.0;
    double camelliaDiagonalMin = 0.0;
    double camelliaDiagonalMax = 0.0;
    double sortedDiagonalMaxAbsError = 0.0;
    double sortedDiagonalL2Error = 0.0;
    double sortedRowSumMaxAbsError = 0.0;
    double sortedRowSumL2Error = 0.0;
    double sortedColumnSumMaxAbsError = 0.0;
    double sortedColumnSumL2Error = 0.0;
    double greedyRowMatchMaxAbsError = 0.0;
    double greedyRowMatchMeanAbsError = 0.0;
    int greedyRowMatchWorstRow = -1;
    int greedyRowMatchMatchedRow = -1;
    double greedyColumnMatchMaxAbsError = 0.0;
    double greedyColumnMatchMeanAbsError = 0.0;
    int greedyColumnMatchWorstColumn = -1;
    int greedyColumnMatchMatchedColumn = -1;
    int localDofs = 0;
    int globalDofs = 0;
  };

  struct DenseVectorComparison
  {
    int localDofs = 0;
    int globalDofs = 0;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    double l2Error = 0.0;
    double matrixFreeNorm = 0.0;
    double camelliaNorm = 0.0;
    int maxErrorOrdinal = -1;
    double matrixFreeAtMaxError = 0.0;
    double camelliaAtMaxError = 0.0;
  };

  std::vector<Scalar> assembleDenseLocalMatrix(std::vector<OperatorTerm> &terms, const int cells, const int localDofs,
                                               const long long maxAssembledEntries)
  {
    const long long entries = static_cast<long long>(cells) * localDofs * localDofs;
    if (entries > maxAssembledEntries)
      throw std::runtime_error("Camellia matrix comparison local matrix exceeds --maxAssembledEntries");

    std::vector<Scalar> localMatrices(static_cast<size_t>(entries), Scalar(0.0));
    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      term.matrix.assemble(matrixData);
      Kokkos::fence();
      if (matrixData.getUnderlyingViewRank() == 3)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<3>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            for (int j=0; j<localDofs; j++)
              localMatrices[static_cast<size_t>((c * localDofs + i) * localDofs + j)] += term.weight * matrixHost(c,i,j);
      }
      else if (matrixData.getUnderlyingViewRank() == 2)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            for (int j=0; j<localDofs; j++)
              localMatrices[static_cast<size_t>((c * localDofs + i) * localDofs + j)] += term.weight * matrixHost(i,j);
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix storage rank in Camellia matrix comparison");
      }
    }
    return localMatrices;
  }

std::vector<OperatorTerm> makeVlasovCamelliaComparisonMatrixFreeTerms(const int p,
                                                                      const VlasovCamelliaComparisonBlock block,
                                                                      const Scalar massWeight,
                                                                      const bool useSolutionCoefficientProxies)
{
  constexpr int spaceDim = 4;
  Kokkos::Array<Scalar,spaceDim> origin;
  Kokkos::Array<Scalar,spaceDim> domainExtents;
  Kokkos::Array<int,spaceDim> gridCellCounts;
  for (int d=0; d<spaceDim; d++)
  {
    origin[d] = 0.0;
    domainExtents[d] = 1.0;
    gridCellCounts[d] = 1;
  }

  using CellGeometry = Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType>;
  auto geometry = CellGeometry(origin, domainExtents, gridCellCounts,
                               CellGeometry::NO_SUBDIVISION, CellGeometry::HYPERCUBE_NODE_ORDER_TENSOR);

  if (block == VlasovCamelliaComparisonBlock::Mass)
  {
    return std::vector<OperatorTerm>{{Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
        geometry, p, Intrepid2::OPERATOR_VALUE, Intrepid2::OPERATOR_VALUE), massWeight, "camellia_compare_mass"}};
  }

  auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
  auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
  std::vector<OperatorTerm> terms;

  Camellia::TensorFunctionEvaluationContext solutionCoefficientContext;
  Camellia::TensorFunctionEvaluationContext *graphEvaluationContext = nullptr;

  Camellia::TensorFunctionDependencySource vxSource(Camellia::TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION,
                                                    "graph_norm_proxy_vx");
  Camellia::TensorFunctionDependencySource axSource(Camellia::TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH,
                                                    "graph_norm_proxy_ax");
  Camellia::TensorFunctionDependencySource aySource(Camellia::TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH,
                                                    "graph_norm_proxy_ay");

  if (useSolutionCoefficientProxies)
  {
    solutionCoefficientContext.registerSolutionCoefficientEvaluator(
        vxSource,
        Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("velocity-component")));

    solutionCoefficientContext.registerSolutionCoefficientEvaluator(
        axSource,
        Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("maxwell-ax-proxy")));

    solutionCoefficientContext.registerSolutionCoefficientEvaluator(
        aySource,
        Teuchos::rcp(new AnalyticGraphNormSolutionCoefficientEvaluator("maxwell-ay-proxy")));

    graphEvaluationContext = &solutionCoefficientContext;
  }

  const std::vector<int> derivativeComponents {1, 0, 2, 3};
  const std::vector<std::string> derivativeLabels {"dt", "dx", "dux", "duy"};

  const std::vector<Camellia::TensorFunctionPtr> coefficients = useSolutionCoefficientProxies
    ? std::vector<Camellia::TensorFunctionPtr>{
        Camellia::TensorFunction::constant(-1.0),
        Camellia::TensorFunction::constant(-1.0) *
            Camellia::TensorFunction::solutionCoefficient1D(2, vxSource),
        Camellia::TensorFunction::constant(-1.0) *
            Camellia::TensorFunction::solutionCoefficient1D(0, axSource),
        Camellia::TensorFunction::constant(-1.0) *
            Camellia::TensorFunction::solutionCoefficient1D(1, aySource)
      }
    : std::vector<Camellia::TensorFunctionPtr>{
        Camellia::TensorFunction::constant(-1.0),
        // TensorFunction::componentCoordinate() is evaluated on reference tensor points, while
        // the Camellia comparison form below uses Function::xn/yn/zn/tn on the physical unit
        // hypercube.  Shift reference coordinates from [-1,1] to [0,1] before forming the
        // physical velocity/field coefficients.
        Camellia::TensorFunction::constant(-0.5) *
            (Camellia::TensorFunction::componentCoordinate(2) +
             Camellia::TensorFunction::constant(1.0)),
        Camellia::TensorFunction::constant(-0.25) *
            (Camellia::TensorFunction::componentCoordinate(0) +
             Camellia::TensorFunction::componentCoordinate(1) +
             Camellia::TensorFunction::constant(2.0)),
        Camellia::TensorFunction::constant(-0.375) *
            (Camellia::TensorFunction::componentCoordinate(1) +
             Camellia::TensorFunction::constant(1.0))
      };

  auto appendGraphDerivativePair = [&](const size_t trialComponent,
                                       const size_t testComponent,
                                       const std::string &label)
  {
    Camellia::TensorBilinearTerm graphTerm(
        trialVar,
        testVar,
        Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim,
                                                                      derivativeComponents[testComponent]),
        Camellia::TensorComponentOperatorDescriptor::singleDerivative(spaceDim,
                                                                      derivativeComponents[trialComponent]),
        coefficients[trialComponent] * coefficients[testComponent],
        Camellia::TENSOR_BILINEAR_TERM_VOLUME,
        Camellia::TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    Camellia::appendTensorVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(
        terms,
        geometry,
        p,
        graphTerm,
        label,
        graphEvaluationContext);
  };

  size_t trialComponent = 0;
  size_t testComponent = 0;
  if (vlasovCamelliaComparisonBlockDerivativePair(block, trialComponent, testComponent))
  {
    appendGraphDerivativePair(trialComponent,
                              testComponent,
                              "camellia_compare_" +
                                  derivativeLabels[trialComponent] + "_" +
                                  derivativeLabels[testComponent]);
    return terms;
  }

  if (block == VlasovCamelliaComparisonBlock::FullGraphNorm)
  {
    terms.push_back({Camellia::constructHypercubeHGRADPAMatrix<spaceDim>(
        geometry, p, Intrepid2::OPERATOR_VALUE, Intrepid2::OPERATOR_VALUE),
        massWeight,
        "camellia_compare_l2"});

    for (size_t trialComponent=0; trialComponent<derivativeComponents.size(); trialComponent++)
    {
      for (size_t testComponent=0; testComponent<derivativeComponents.size(); testComponent++)
      {
        appendGraphDerivativePair(trialComponent,
                                  testComponent,
                                  "camellia_compare_full_" +
                                      derivativeLabels[trialComponent] + "_" +
                                      derivativeLabels[testComponent]);
      }
    }
    return terms;
  }

  throw std::invalid_argument("Unknown Vlasov Camellia comparison block");
}

  Camellia::SolutionPtr makeCamelliaVlasovGraphGramComparisonSolution(const int p,
                                                                      const Scalar massWeight,
                                                                      const VlasovCamelliaComparisonBlock block)
  {
    const int spaceTimePhaseDim = 4;
    Camellia::VarFactoryPtr vf = Camellia::VarFactory::varFactory();

    Camellia::VarPtr f = vf->fieldVar("f", Camellia::HGRAD_DISC);
    // VarFactory::setBasisFamily() only updates variables that already exist; it does not store
    // a default for variables created later.  Apply the hierarchical basis after creating f so the
    // Camellia reference matrix uses the same Intrepid2 hierarchical tensor HGRAD family as the
    // matrix-free PAMatrix side.
    using IntrepidBasisFamily = Intrepid2::HierarchicalBasisFamily<DeviceType,Scalar,Scalar>;
    auto intrepid2Basis = Intrepid2::getHypercubeBasis_HGRAD<IntrepidBasisFamily>(p, spaceTimePhaseDim);
    auto camelliaBasis = Teuchos::rcp(new Camellia::Intrepid2BasisWrapper<Scalar>(intrepid2Basis,
                                                                                  spaceTimePhaseDim,
                                                                                  0,
                                                                                  Camellia::FUNCTION_SPACE_HGRAD_DISC));
    auto comparisonBasisFamily = Camellia::BasisFamily::singletonBasisFamily(camelliaBasis);
    vf->setBasisFamily(comparisonBasisFamily);
    Camellia::VarFactoryPtr bubnovVF = vf->getBubnovFactory(Camellia::VarFactory::BUBNOV_TRIAL);
    bubnovVF->setBasisFamily(comparisonBasisFamily);
    Camellia::BFPtr bf = Camellia::BF::bf(bubnovVF);
    Camellia::VarPtr w = f;

    if ((block == VlasovCamelliaComparisonBlock::Mass) || (block == VlasovCamelliaComparisonBlock::FullGraphNorm))
      bf->addTerm(massWeight * f, w);

    Camellia::FunctionPtr x  = Camellia::Function::xn(1);
    Camellia::FunctionPtr t  = Camellia::Function::yn(1);
    Camellia::FunctionPtr ux = Camellia::Function::zn(1);
    Camellia::FunctionPtr ax = Camellia::Function::constant(0.5) * (x + t);
    Camellia::FunctionPtr ay = Camellia::Function::constant(0.75) * t;

    const std::vector<Camellia::LinearTermPtr> trialDerivativeTerms {-f->di(2), -ux * f->di(1), -ax * f->di(3), -ay * f->di(4)};
    const std::vector<Camellia::LinearTermPtr> testDerivativeTerms  {-w->di(2), -ux * w->di(1), -ax * w->di(3), -ay * w->di(4)};

    size_t trialComponent = 0;
    size_t testComponent = 0;
    if (vlasovCamelliaComparisonBlockDerivativePair(block, trialComponent, testComponent))
    {
      bf->addTerm(trialDerivativeTerms[trialComponent], testDerivativeTerms[testComponent]);
    }
    else if (block == VlasovCamelliaComparisonBlock::FullGraphNorm)
    {
      Camellia::LinearTermPtr Lf = trialDerivativeTerms[0];
      Camellia::LinearTermPtr Lw = testDerivativeTerms[0];
      for (size_t i=1; i<trialDerivativeTerms.size(); i++)
      {
        Lf = Lf + trialDerivativeTerms[i];
        Lw = Lw + testDerivativeTerms[i];
      }

      bf->addTerm(Lf, Lw);
    }

    std::vector<double> dimensions(spaceTimePhaseDim, 1.0);
    std::vector<int> elementCounts(spaceTimePhaseDim, 1);
    std::vector<double> origin(spaceTimePhaseDim, 0.0);

    Camellia::MeshTopologyPtr meshTopology = Camellia::MeshFactory::hypercubeMeshTopology(dimensions, elementCounts, origin);
    Camellia::MeshPtr mesh = Camellia::MeshFactory::minRuleMesh(meshTopology, bf, p, 0);

    Camellia::RHSPtr rhs = Camellia::RHS::rhs();
    rhs->addTerm(Camellia::Function::constant(1.0) * w);
    Camellia::BCPtr bc = Camellia::BC::bc();

    Camellia::SolutionPtr solution = Camellia::Solution::solution(mesh, bc, rhs, Teuchos::null);
    solution->setUseCondensedSolve(false);
    solution->setZMCsAsGlobalLagrange(false);
    return solution;
  }

  std::vector<Camellia::GlobalIndexType> localToGlobalDofMap(Camellia::MeshPtr mesh, const Camellia::GlobalIndexType cellID,
                                                             const int localDofs)
  {
    std::vector<Camellia::GlobalIndexType> localToGlobal(static_cast<size_t>(localDofs), -1);
    Camellia::FieldContainer<double> localBasisVector(localDofs);
    Camellia::FieldContainer<double> globalVector;
    Camellia::FieldContainer<Camellia::GlobalIndexType> globalDofIndices;
    for (int localDof=0; localDof<localDofs; localDof++)
    {
      for (int i=0; i<localDofs; i++) localBasisVector(i) = 0.0;
      localBasisVector(localDof) = 1.0;
      mesh->interpretLocalData(cellID, localBasisVector, globalVector, globalDofIndices);
      double maxMagnitude = 0.0;
      int maxOrdinal = -1;
      for (int globalOrdinal=0; globalOrdinal<globalVector.dimension(0); globalOrdinal++)
      {
        const double magnitude = std::abs(globalVector(globalOrdinal));
        if (magnitude > maxMagnitude)
        {
          maxMagnitude = magnitude;
          maxOrdinal = globalOrdinal;
        }
      }
      if ((maxOrdinal < 0) || (maxMagnitude < 0.5))
        throw std::runtime_error("Could not determine local-to-global DOF map for Camellia matrix comparison");
      localToGlobal[static_cast<size_t>(localDof)] = globalDofIndices(maxOrdinal);
    }
    return localToGlobal;
  }

  std::vector<Scalar> interpretLocalDenseMatrixIntoCamelliaGlobalOrder(Camellia::MeshPtr mesh,
                                                                        const std::vector<Scalar> &localDense,
                                                                        const int expectedLocalDofs,
                                                                        std::vector<Camellia::GlobalIndexType> &globalDofIndicesVector)
  {
    const auto &cellIDs = mesh->cellIDsInPartition();
    if (cellIDs.size() != 1)
      throw std::runtime_error("Camellia matrix comparison expects exactly one locally-owned cell");
    const Camellia::GlobalIndexType cellID = *cellIDs.begin();
    const int localDofs = mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
    if (localDofs != expectedLocalDofs)
      throw std::runtime_error("Camellia matrix comparison local DOF count does not match matrix-free local DOF count");

    Camellia::FieldContainer<Scalar> localData(localDofs, localDofs);
    for (int row=0; row<localDofs; row++)
      for (int col=0; col<localDofs; col++)
        localData(row,col) = localDense[static_cast<size_t>(row) * localDofs + col];

    Camellia::FieldContainer<Scalar> globalData;
    Camellia::FieldContainer<Camellia::GlobalIndexType> globalDofIndices;
    mesh->interpretLocalData(cellID, localData, globalData, globalDofIndices);

    const int globalDofs = globalDofIndices.dimension(0);
    globalDofIndicesVector.resize(static_cast<size_t>(globalDofs));
    for (int ordinal=0; ordinal<globalDofs; ordinal++) globalDofIndicesVector[static_cast<size_t>(ordinal)] = globalDofIndices(ordinal);

    std::vector<Scalar> dense(static_cast<size_t>(globalDofs) * globalDofs, Scalar(0.0));
    for (int row=0; row<globalDofs; row++)
      for (int col=0; col<globalDofs; col++)
        dense[static_cast<size_t>(row) * globalDofs + col] = globalData(row,col);
    return dense;
  }

  std::vector<Scalar> extractCamelliaDenseMatrixInGlobalOrder(Camellia::SolutionPtr solution,
                                                              const std::vector<Camellia::GlobalIndexType> &globalDofIndices)
  {
    solution->initializeStiffnessAndLoad();
    solution->populateStiffnessAndLoad();
    auto matrix = solution->getStiffnessMatrix();
    const int globalDofs = static_cast<int>(globalDofIndices.size());
    std::vector<Scalar> dense(static_cast<size_t>(globalDofs) * globalDofs, Scalar(0.0));

    const int maxEntries = static_cast<int>(matrix->getLocalMaxNumRowEntries());
    TpetraCrsMatrix::nonconst_global_inds_host_view_type columnIndices("Camellia comparison columns", maxEntries);
    TpetraCrsMatrix::nonconst_values_host_view_type values("Camellia comparison values", maxEntries);
    for (int row=0; row<globalDofs; row++)
    {
      size_t numEntries = 0;
      matrix->getGlobalRowCopy(globalDofIndices[static_cast<size_t>(row)], columnIndices, values, numEntries);
      for (size_t entryOrdinal=0; entryOrdinal<numEntries; entryOrdinal++)
      {
        const auto columnGlobal = columnIndices(entryOrdinal);
        for (int col=0; col<globalDofs; col++)
        {
          if (globalDofIndices[static_cast<size_t>(col)] == columnGlobal)
          {
            dense[static_cast<size_t>(row) * globalDofs + col] = values(entryOrdinal);
            break;
          }
        }
      }
    }
    return dense;
  }

  DenseMatrixComparison compareDenseMatrices(const std::vector<Scalar> &matrixFree, const std::vector<Scalar> &camellia,
                                             const int localDofs, const int globalDofs)
  {
    DenseMatrixComparison comparison;
    comparison.localDofs = localDofs;
    comparison.globalDofs = globalDofs;
    comparison.maxAbsError = 0.0;
    comparison.maxRelError = 0.0;
    if (matrixFree.size() != camellia.size())
      throw std::invalid_argument("Camellia matrix comparison requires equal matrix sizes");
    if (matrixFree.size() != static_cast<size_t>(globalDofs) * globalDofs)
      throw std::invalid_argument("Camellia matrix comparison matrix size does not match globalDofs");

    std::vector<double> matrixFreeDiagonal(static_cast<size_t>(globalDofs), 0.0);
    std::vector<double> camelliaDiagonal(static_cast<size_t>(globalDofs), 0.0);
    std::vector<double> matrixFreeRowSums(static_cast<size_t>(globalDofs), 0.0);
    std::vector<double> camelliaRowSums(static_cast<size_t>(globalDofs), 0.0);
    std::vector<double> matrixFreeColumnSums(static_cast<size_t>(globalDofs), 0.0);
    std::vector<double> camelliaColumnSums(static_cast<size_t>(globalDofs), 0.0);

    for (int row=0; row<globalDofs; row++)
    {
      for (int col=0; col<globalDofs; col++)
      {
        const size_t entryOrdinal = static_cast<size_t>(row) * globalDofs + col;
        const double matrixFreeValue = matrixFree[entryOrdinal];
        const double camelliaValue = camellia[entryOrdinal];
        const double absError = std::abs(matrixFreeValue - camelliaValue);
        const double scale = std::max<double>(1.0, std::max(std::abs(matrixFreeValue), std::abs(camelliaValue)));
        if (absError > comparison.maxAbsError)
        {
          comparison.maxAbsError = absError;
          comparison.maxErrorRow = row;
          comparison.maxErrorCol = col;
          comparison.matrixFreeAtMaxError = matrixFreeValue;
          comparison.camelliaAtMaxError = camelliaValue;
        }
        comparison.maxRelError = std::max(comparison.maxRelError, absError / scale);
        if (row == col)
        {
          comparison.matrixFreeTrace += matrixFreeValue;
          comparison.camelliaTrace += camelliaValue;
          comparison.maxDiagonalAbsError = std::max(comparison.maxDiagonalAbsError, absError);
          matrixFreeDiagonal[static_cast<size_t>(row)] = matrixFreeValue;
          camelliaDiagonal[static_cast<size_t>(row)] = camelliaValue;
        }
        comparison.matrixFreeTotal += matrixFreeValue;
        comparison.camelliaTotal += camelliaValue;
        matrixFreeRowSums[static_cast<size_t>(row)] += matrixFreeValue;
        camelliaRowSums[static_cast<size_t>(row)] += camelliaValue;
        matrixFreeColumnSums[static_cast<size_t>(col)] += matrixFreeValue;
        camelliaColumnSums[static_cast<size_t>(col)] += camelliaValue;
      }
    }

    for (int row=0; row<globalDofs; row++)
      comparison.maxRowSumAbsError = std::max(comparison.maxRowSumAbsError,
                                              std::abs(matrixFreeRowSums[static_cast<size_t>(row)] - camelliaRowSums[static_cast<size_t>(row)]));

    if (globalDofs > 0)
    {
      comparison.matrixFreeDiagonalMin = *std::min_element(matrixFreeDiagonal.begin(), matrixFreeDiagonal.end());
      comparison.matrixFreeDiagonalMax = *std::max_element(matrixFreeDiagonal.begin(), matrixFreeDiagonal.end());
      comparison.camelliaDiagonalMin = *std::min_element(camelliaDiagonal.begin(), camelliaDiagonal.end());
      comparison.camelliaDiagonalMax = *std::max_element(camelliaDiagonal.begin(), camelliaDiagonal.end());
    }

    auto sortedSignatureErrors = [](std::vector<double> left, std::vector<double> right, double &maxAbsError, double &l2Error)
    {
      std::sort(left.begin(), left.end());
      std::sort(right.begin(), right.end());
      maxAbsError = 0.0;
      double squaredError = 0.0;
      for (size_t i=0; i<left.size(); i++)
      {
        const double absError = std::abs(left[i] - right[i]);
        maxAbsError = std::max(maxAbsError, absError);
        squaredError += absError * absError;
      }
      l2Error = std::sqrt(squaredError);
    };
    sortedSignatureErrors(matrixFreeDiagonal, camelliaDiagonal,
                          comparison.sortedDiagonalMaxAbsError, comparison.sortedDiagonalL2Error);
    sortedSignatureErrors(matrixFreeRowSums, camelliaRowSums,
                          comparison.sortedRowSumMaxAbsError, comparison.sortedRowSumL2Error);
    sortedSignatureErrors(matrixFreeColumnSums, camelliaColumnSums,
                          comparison.sortedColumnSumMaxAbsError, comparison.sortedColumnSumL2Error);

    for (int row=0; row<globalDofs; row++)
    {
      for (int col=0; col<globalDofs; col++)
      {
        const size_t entryOrdinal = static_cast<size_t>(row) * globalDofs + col;
        const size_t transposeOrdinal = static_cast<size_t>(col) * globalDofs + row;
        comparison.matrixFreeSymmetryMaxAbsError = std::max(comparison.matrixFreeSymmetryMaxAbsError,
                                                            std::abs(matrixFree[entryOrdinal] - matrixFree[transposeOrdinal]));
        comparison.camelliaSymmetryMaxAbsError = std::max(comparison.camelliaSymmetryMaxAbsError,
                                                          std::abs(camellia[entryOrdinal] - camellia[transposeOrdinal]));
        comparison.crossTransposeMaxAbsError = std::max(comparison.crossTransposeMaxAbsError,
                                                        std::abs(matrixFree[entryOrdinal] - camellia[transposeOrdinal]));
      }
    }

    auto greedyLineMatch = [globalDofs](const std::vector<Scalar> &left, const std::vector<Scalar> &right, const bool matchRows,
                                        double &maxAbsError, double &meanAbsError, int &worstLeftOrdinal, int &matchedRightOrdinal)
    {
      std::vector<int> usedRight(static_cast<size_t>(globalDofs), 0);
      maxAbsError = 0.0;
      meanAbsError = 0.0;
      worstLeftOrdinal = -1;
      matchedRightOrdinal = -1;
      for (int leftOrdinal=0; leftOrdinal<globalDofs; leftOrdinal++)
      {
        double bestError = std::numeric_limits<double>::infinity();
        int bestRightOrdinal = -1;
        for (int rightOrdinal=0; rightOrdinal<globalDofs; rightOrdinal++)
        {
          if (usedRight[static_cast<size_t>(rightOrdinal)] != 0) continue;
          double candidateError = 0.0;
          for (int k=0; k<globalDofs; k++)
          {
            const size_t leftEntry = matchRows ? static_cast<size_t>(leftOrdinal) * globalDofs + k
                                               : static_cast<size_t>(k) * globalDofs + leftOrdinal;
            const size_t rightEntry = matchRows ? static_cast<size_t>(rightOrdinal) * globalDofs + k
                                                : static_cast<size_t>(k) * globalDofs + rightOrdinal;
            candidateError = std::max(candidateError, std::abs(left[leftEntry] - right[rightEntry]));
          }
          if (candidateError < bestError)
          {
            bestError = candidateError;
            bestRightOrdinal = rightOrdinal;
          }
        }
        if (bestRightOrdinal >= 0) usedRight[static_cast<size_t>(bestRightOrdinal)] = 1;
        meanAbsError += bestError;
        if (bestError > maxAbsError)
        {
          maxAbsError = bestError;
          worstLeftOrdinal = leftOrdinal;
          matchedRightOrdinal = bestRightOrdinal;
        }
      }
      if (globalDofs > 0) meanAbsError /= globalDofs;
    };
    greedyLineMatch(matrixFree, camellia, true,
                    comparison.greedyRowMatchMaxAbsError, comparison.greedyRowMatchMeanAbsError,
                    comparison.greedyRowMatchWorstRow, comparison.greedyRowMatchMatchedRow);
    greedyLineMatch(matrixFree, camellia, false,
                    comparison.greedyColumnMatchMaxAbsError, comparison.greedyColumnMatchMeanAbsError,
                    comparison.greedyColumnMatchWorstColumn, comparison.greedyColumnMatchMatchedColumn);

    return comparison;
  }

  DenseMatrixComparison compareCamelliaVlasovBlock(const int p, const VlasovCamelliaComparisonBlock block,
                                                   const Scalar massWeight, const long long maxAssembledEntries,
                                                   const bool useSolutionCoefficientProxies)
  {
    const int cells = 1;
    const int localDofs = checkedProduct(std::vector<int>(4, p + 1));
    auto matrixFreeTerms = makeVlasovCamelliaComparisonMatrixFreeTerms(p, block, massWeight, useSolutionCoefficientProxies);
    const auto matrixFreeDense = assembleDenseLocalMatrix(matrixFreeTerms, cells, localDofs, maxAssembledEntries);

    auto camelliaSolution = makeCamelliaVlasovGraphGramComparisonSolution(p, massWeight, block);
    std::vector<Camellia::GlobalIndexType> globalDofIndices;
    const auto matrixFreeGlobalDense = interpretLocalDenseMatrixIntoCamelliaGlobalOrder(camelliaSolution->mesh(), matrixFreeDense,
                                                                                       localDofs, globalDofIndices);
    const int globalDofs = static_cast<int>(globalDofIndices.size());
    const auto camelliaDense = extractCamelliaDenseMatrixInGlobalOrder(camelliaSolution, globalDofIndices);

    return compareDenseMatrices(matrixFreeGlobalDense, camelliaDense, localDofs, globalDofs);
  }

  std::string camelliaComparisonBlockName(const VlasovCamelliaComparisonBlock block)
  {
    switch (block)
    {
      case VlasovCamelliaComparisonBlock::Mass: return "mass";
      case VlasovCamelliaComparisonBlock::DtDt: return "dt_dt";
      case VlasovCamelliaComparisonBlock::DtDx: return "dt_dx";
      case VlasovCamelliaComparisonBlock::DtDux: return "dt_dux";
      case VlasovCamelliaComparisonBlock::DtDuy: return "dt_duy";
      case VlasovCamelliaComparisonBlock::DxDt: return "dx_dt";
      case VlasovCamelliaComparisonBlock::DxDx: return "dx_dx";
      case VlasovCamelliaComparisonBlock::DxDux: return "dx_dux";
      case VlasovCamelliaComparisonBlock::DxDuy: return "dx_duy";
      case VlasovCamelliaComparisonBlock::DuxDt: return "dux_dt";
      case VlasovCamelliaComparisonBlock::DuxDx: return "dux_dx";
      case VlasovCamelliaComparisonBlock::DuxDux: return "dux_dux";
      case VlasovCamelliaComparisonBlock::DuxDuy: return "dux_duy";
      case VlasovCamelliaComparisonBlock::DuyDt: return "duy_dt";
      case VlasovCamelliaComparisonBlock::DuyDx: return "duy_dx";
      case VlasovCamelliaComparisonBlock::DuyDux: return "duy_dux";
      case VlasovCamelliaComparisonBlock::DuyDuy: return "duy_duy";
      case VlasovCamelliaComparisonBlock::FullGraphNorm: return "full_graph_norm";
    }
    return "unknown";
  }

  std::vector<DenseMatrixComparison> runCamelliaVlasovMatrixComparison(const int p, const Scalar massWeight,
                                                                       const long long maxAssembledEntries,
                                                                       const bool useSolutionCoefficientProxies)
  {
    const std::vector<VlasovCamelliaComparisonBlock> blocks {VlasovCamelliaComparisonBlock::Mass,
                                                            VlasovCamelliaComparisonBlock::DtDt,
                                                            VlasovCamelliaComparisonBlock::DtDx,
                                                            VlasovCamelliaComparisonBlock::DtDux,
                                                            VlasovCamelliaComparisonBlock::DtDuy,
                                                            VlasovCamelliaComparisonBlock::DxDt,
                                                            VlasovCamelliaComparisonBlock::DxDx,
                                                            VlasovCamelliaComparisonBlock::DxDux,
                                                            VlasovCamelliaComparisonBlock::DxDuy,
                                                            VlasovCamelliaComparisonBlock::DuxDt,
                                                            VlasovCamelliaComparisonBlock::DuxDx,
                                                            VlasovCamelliaComparisonBlock::DuxDux,
                                                            VlasovCamelliaComparisonBlock::DuxDuy,
                                                            VlasovCamelliaComparisonBlock::DuyDt,
                                                            VlasovCamelliaComparisonBlock::DuyDx,
                                                            VlasovCamelliaComparisonBlock::DuyDux,
                                                            VlasovCamelliaComparisonBlock::DuyDuy,
                                                            VlasovCamelliaComparisonBlock::FullGraphNorm};
    std::vector<DenseMatrixComparison> comparisons;
    comparisons.reserve(blocks.size());
    int passed = 1;
    std::string firstFailingBlock = "none";
    double worstAbsError = 0.0;
    double worstRelError = 0.0;
    constexpr double tolerance = 1e-10;
    for (const auto block : blocks)
    {
      const auto comparison = compareCamelliaVlasovBlock(p, block, massWeight, maxAssembledEntries, useSolutionCoefficientProxies);
      const std::string blockName = camelliaComparisonBlockName(block);
      const bool blockPassed = (comparison.maxAbsError <= tolerance) || (comparison.maxRelError <= tolerance);
      if (!blockPassed && passed)
      {
        passed = 0;
        firstFailingBlock = blockName;
      }
      worstAbsError = std::max(worstAbsError, comparison.maxAbsError);
      worstRelError = std::max(worstRelError, comparison.maxRelError);
      std::cout << "CAMELLIA_VLASOV_MATRIX_COMPARISON"
                << " block=" << blockName
                << " p=" << p
                << " localDofs=" << comparison.localDofs
                << " globalDofs=" << comparison.globalDofs
                << " maxAbsError=" << std::setprecision(16) << comparison.maxAbsError
                << " maxRelError=" << comparison.maxRelError
                << " maxErrorRow=" << comparison.maxErrorRow
                << " maxErrorCol=" << comparison.maxErrorCol
                << " matrixFreeAtMaxError=" << comparison.matrixFreeAtMaxError
                << " camelliaAtMaxError=" << comparison.camelliaAtMaxError
                << " matrixFreeTrace=" << comparison.matrixFreeTrace
                << " camelliaTrace=" << comparison.camelliaTrace
                << " matrixFreeTotal=" << comparison.matrixFreeTotal
                << " camelliaTotal=" << comparison.camelliaTotal
                << " maxDiagonalAbsError=" << comparison.maxDiagonalAbsError
                << " maxRowSumAbsError=" << comparison.maxRowSumAbsError
                << " matrixFreeSymmetryMaxAbsError=" << comparison.matrixFreeSymmetryMaxAbsError
                << " camelliaSymmetryMaxAbsError=" << comparison.camelliaSymmetryMaxAbsError
                << " crossTransposeMaxAbsError=" << comparison.crossTransposeMaxAbsError
                << " matrixFreeDiagonalMin=" << comparison.matrixFreeDiagonalMin
                << " matrixFreeDiagonalMax=" << comparison.matrixFreeDiagonalMax
                << " camelliaDiagonalMin=" << comparison.camelliaDiagonalMin
                << " camelliaDiagonalMax=" << comparison.camelliaDiagonalMax
                << " sortedDiagonalMaxAbsError=" << comparison.sortedDiagonalMaxAbsError
                << " sortedDiagonalL2Error=" << comparison.sortedDiagonalL2Error
                << " sortedRowSumMaxAbsError=" << comparison.sortedRowSumMaxAbsError
                << " sortedRowSumL2Error=" << comparison.sortedRowSumL2Error
                << " sortedColumnSumMaxAbsError=" << comparison.sortedColumnSumMaxAbsError
                << " sortedColumnSumL2Error=" << comparison.sortedColumnSumL2Error
                << " greedyRowMatchMaxAbsError=" << comparison.greedyRowMatchMaxAbsError
                << " greedyRowMatchMeanAbsError=" << comparison.greedyRowMatchMeanAbsError
                << " greedyRowMatchWorstRow=" << comparison.greedyRowMatchWorstRow
                << " greedyRowMatchMatchedRow=" << comparison.greedyRowMatchMatchedRow
                << " greedyColumnMatchMaxAbsError=" << comparison.greedyColumnMatchMaxAbsError
                << " greedyColumnMatchMeanAbsError=" << comparison.greedyColumnMatchMeanAbsError
                << " greedyColumnMatchWorstColumn=" << comparison.greedyColumnMatchWorstColumn
                << " greedyColumnMatchMatchedColumn=" << comparison.greedyColumnMatchMatchedColumn
                << " tolerance=" << tolerance
                << " passed=" << (blockPassed ? 1 : 0)
                << std::endl;
      comparisons.push_back(comparison);
    }
    std::cout << "CAMELLIA_VLASOV_MATRIX_COMPARISON_SUMMARY"
              << " p=" << p
              << " passed=" << passed
              << " firstFailingBlock=" << firstFailingBlock
              << " worstMaxAbsError=" << std::setprecision(16) << worstAbsError
              << " worstMaxRelError=" << worstRelError
              << " tolerance=" << tolerance
              << std::endl;
    return comparisons;
  }

  void fillInput(Intrepid2::ScalarView<Scalar,DeviceType> input)
  {
    auto host = Kokkos::create_mirror_view(input);
    for (int c=0; c<input.extent_int(0); c++)
    {
      for (int f=0; f<input.extent_int(1); f++)
      {
        host(c,f) = 0.25 + 0.125 * ((f + 3 * c) % 17);
      }
    }
    Kokkos::deep_copy(input, host);
  }

  struct AssemblyTiming
  {
    double totalSeconds = -1.0;
    double massSeconds = -1.0;
    double stiffnessSeconds = -1.0;
  };

  struct FactorTiming
  {
    AssemblyTiming assembly;
    double factorSeconds = -1.0;
    double solveSeconds = -1.0;
    double maxSolveResidual = -1.0;
    double maxSolveRelativeResidual = -1.0;
  };

  struct AssemblyComparison
  {
    double maxAbsError = -1.0;
    double maxRelError = -1.0;
  };

  struct TpetraOperatorComparison
  {
    double maxAbsError = -1.0;
    double maxRelError = -1.0;
    double inverseDiagonalMaxAbsError = -1.0;
    double inverseDiagonalMin = -1.0;
    double inverseDiagonalMax = -1.0;
    double rowMatrixDiagonalMaxAbsError = -1.0;
    double rowMatrixDiagonalMin = -1.0;
    double rowMatrixDiagonalMax = -1.0;
    double rowMatrixRowCopyMaxAbsError = -1.0;
    int rowMatrixRowCopyChecked = 0;
    double rowMatrixRowViewMaxAbsError = -1.0;
    int rowMatrixRowViewChecked = 0;
    long long rowMatrixDenseRowMaterializationEntries = -1;
    long long rowMatrixMaxDenseRowMaterializationEntries = -1;
    double firstApplySeconds = -1.0;
    double avgApplySeconds = -1.0;
    int numVectors = 0;
    int localLength = 0;
    int domainRangeMapSame = 0;
    int rowMatrixRowMapSame = 0;
    int rowMatrixColMapSame = 0;
    std::string adapterPath = "not_requested";
    std::string globalDofSemantics = "not_requested";
    std::string ifpack2RowMatrix = "not_requested";
  };

  struct InverseDiagnosticTiming
  {
    double avgInverseApplySeconds = -1.0;
    double localPMultigridWrapperApplyInfNorm = -1.0;
    double matrixFreeGMGWrapperApplyInfNorm = -1.0;
    double localPMultigridWrapperDifferenceInfNorm = -1.0;
    double localPMultigridWrapperRelativeDifferenceInfNorm = -1.0;
    int inverseIterations = 0;
    int inversePreconditionerDegree = 0;
    double initialInverseAbsError = -1.0;
    double initialInverseRelError = -1.0;
    double maxInverseAbsError = -1.0;
    double maxInverseRelError = -1.0;
    double initialResidualNorm = -1.0;
    double finalResidualNorm = -1.0;
    double residualReduction = -1.0;
    double inverseTolerance = -1.0;
    int inverseConverged = 0;
    std::string inverseStoppedReason = "not_requested";
    std::string inverseRhs = "deterministic";
    double lambdaMinEstimate = -1.0;
    double lambdaMaxEstimate = -1.0;
    double assembledGershgorinLambdaMax = -1.0;
    double powerRawLambdaMaxEstimate = -1.0;
    double tpetraPowerRawLambdaMaxEstimate = -1.0;
    double tpetraPowerLambdaMaxAbsError = -1.0;
    double tpetraPowerLambdaMaxRelError = -1.0;
    double tpetraRowMatrixPowerRawLambdaMaxEstimate = -1.0;
    double tpetraRowMatrixPowerLambdaMaxAbsError = -1.0;
    double tpetraRowMatrixPowerLambdaMaxRelError = -1.0;
    double lambdaMinRawEstimate = -1.0;
    double lambdaMinSafetyFactor = -1.0;
    double lambdaMinEstimateOverRaw = -1.0;
    double lambdaMaxEstimateOverAssembledGershgorin = -1.0;
    double perCellLambdaMinEstimateMin = -1.0;
    double perCellLambdaMinEstimateMax = -1.0;
    double perCellLambdaMaxEstimateMin = -1.0;
    double perCellLambdaMaxEstimateMax = -1.0;
    double perCellLambdaMaxSpread = -1.0;
    int perCellSpectralEstimate = 0;
    int compareTpetraPowerEstimate = 0;
    int compareTpetraRowMatrixPowerEstimate = 0;
    int ifpack2ChebyshevSmoke = 0;
    double ifpack2ChebyshevSeconds = -1.0;
    double ifpack2ChebyshevMaxAbsError = -1.0;
    double ifpack2ChebyshevMaxRelError = -1.0;
    double inverseEigenRatio = -1.0;
    int spectralEstimateIterations = 0;
    double spectralEstimateSeconds = -1.0;
    double lambdaMaxSafetyFactor = -1.0;
    double maxSymmetryAbsError = -1.0;
    double minDiagonalEstimate = -1.0;
    double maxDiagonalEstimate = -1.0;
    double trueLambdaMin = -1.0;
    double trueLambdaMax = -1.0;
    double trueCondition = -1.0;
    double trueLambdaMaxOverAssembledGershgorin = -1.0;
    double trueLambdaMaxOverPowerRaw = -1.0;
    double trueLambdaMinOverLowerEstimate = -1.0;
    int trueEigenDiagnostic = 0;
    std::string trueEigenPath = "not_requested";
    int operatorCheckSamples = 0;
    double operatorActionMaxAbsDiff = -1.0;
    double operatorActionMaxRelDiff = -1.0;
    double operatorBilinearSymmetryAbsError = -1.0;
    double operatorMinRayleighQuotient = -1.0;
    int chebyshevScalarNumEigenvalues = 0;
    int chebyshevScalarNumBelowLower = 0;
    int chebyshevScalarNumInsideInterval = 0;
    int chebyshevScalarNumAboveUpper = 0;
    int chebyshevScalarNumModesAmplified = 0;
    double chebyshevScalarMaxAbsAmplification = -1.0;
    double chebyshevScalarWorstLambda = -1.0;
    int pMultigridLevels = 0;
    int pMultigridFineDofs = 0;
    int pMultigridCoarseDofs = 0;
    int pMultigridChebyshevSmootherIterations = 0;
    double pMultigridChebyshevSmootherLowerFraction = -1.0;
    double pMultigridFineSmootherWeight = -1.0;
    int pMultigridJacobiSmootherIterations = 0;
    int pMultigridCoarseChebyshevIterations = 0;
    double pMultigridCoarseChebyshevLowerFraction = -1.0;
    int pMultigridCoarseSolveCount = 0;
    int matrixFreeGMGStageTrace = 0;
    int matrixFreeGMGStageTraceEntries = 0;
    std::string pMultigridPath = "not_requested";
    std::string inversePath = "not_requested";
    std::string spectralEstimatePath = "not_requested";
    std::string ifpack2ChebyshevPath = "not_requested";
    std::string lambdaMinSource = "not_requested";
    std::string lambdaMaxSource = "not_requested";
    std::string inverseLowerSpectralEstimate = "not_requested";
    std::string inverseInitialGuess = "not_requested";
  };

  void addTermSeconds(AssemblyTiming &timing, const std::string &label, const double seconds)
  {
    if (timing.totalSeconds < 0.0) timing.totalSeconds = 0.0;
    timing.totalSeconds += seconds;
    if (label == "mass")
    {
      if (timing.massSeconds < 0.0) timing.massSeconds = 0.0;
      timing.massSeconds += seconds;
    }
    else if (label.find("stiffness") == 0)
    {
      if (timing.stiffnessSeconds < 0.0) timing.stiffnessSeconds = 0.0;
      timing.stiffnessSeconds += seconds;
    }
  }

  std::string assemble2PathSummary(const std::vector<OperatorTerm> &terms)
  {
    bool sawUnitColumn = false;
    bool sawApplyFallback = false;
    for (const auto &term : terms)
    {
      if (Intrepid2::pamatrixAssemble2UsesUnitColumnPath(term.matrix)) sawUnitColumn = true;
      else sawApplyFallback = true;
    }

    if (sawUnitColumn && sawApplyFallback) return "mixed_staged_unit_column_and_apply_unit_multivector_fallback";
    if (sawUnitColumn) return "staged_sparse_unit_column_right_apply";
    return "apply_unit_multivector_fallback";
  }

  TpetraOperatorComparison compareTpetraOperatorAgainstDirect(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                                                               const Intrepid2::ScalarView<Scalar,DeviceType> &baseInput,
                                                               const int cells, const int localDofs, const int worksetSize,
                                                               const int numApply)
  {
    TpetraOperatorComparison comparison;
    comparison.numVectors = 2;
    const size_t localLength = static_cast<size_t>(cells) * static_cast<size_t>(localDofs);
    comparison.localLength = static_cast<int>(localLength);
    comparison.adapterPath = "local_flat_map_nonproduction_operator";
    comparison.globalDofSemantics = "not_supported";
    comparison.ifpack2RowMatrix = "not_supported";
    auto comm = Teuchos::DefaultComm<int>::getComm();
    auto map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localLength, 0, comm));
    
    auto op = TpetraAdapters::createLocalPAMatrixTpetraOperator(map, terms, workspaces, cells, localDofs, worksetSize);
    
    comparison.domainRangeMapSame = (op->getDomainMap().getRawPtr() == op->getRangeMap().getRawPtr()) ? 1 : 0;
    TpetraMultiVector inverseDiagonal(map, 1);
    TpetraAdapters::fillInverseDiagonalFromTerms(terms,
                                                 cells,
                                                 localDofs,
                                                 inverseDiagonal);
    auto inverseDiagonalHost = inverseDiagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto expectedDiagonal = TpetraAdapters::extractCombinedDiagonal(terms, cells, localDofs);
    comparison.inverseDiagonalMaxAbsError = 0.0;
    comparison.inverseDiagonalMin = std::numeric_limits<Scalar>::max();
    comparison.inverseDiagonalMax = 0.0;
    for (size_t localOrdinal=0; localOrdinal<localLength; localOrdinal++)
    {
      const Scalar expectedInverseDiagonal = Scalar(1.0) / expectedDiagonal[localOrdinal];
      comparison.inverseDiagonalMaxAbsError = std::max(comparison.inverseDiagonalMaxAbsError,
                                                       std::abs(inverseDiagonalHost(localOrdinal,0) - expectedInverseDiagonal));
      comparison.inverseDiagonalMin = std::min(comparison.inverseDiagonalMin, inverseDiagonalHost(localOrdinal,0));
      comparison.inverseDiagonalMax = std::max(comparison.inverseDiagonalMax, inverseDiagonalHost(localOrdinal,0));
    }

    TpetraMultiVector X(map, comparison.numVectors);
    TpetraMultiVector Y(map, comparison.numVectors);
    auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto yHost = Y.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto baseHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), baseInput);
    std::vector<Scalar> yInitial(localLength * comparison.numVectors, Scalar(0.0));
    for (int v=0; v<comparison.numVectors; v++)
    {
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
        {
          const size_t localOrdinal = static_cast<size_t>(c * localDofs + f);
          xHost(localOrdinal, v) = baseHost(c,f) * (Scalar(1.0) + Scalar(0.25) * v);
          yInitial[localOrdinal + localLength * v] = Scalar(0.125) * (Scalar(1.0) + Scalar((localOrdinal + v) % 5));
          yHost(localOrdinal, v) = yInitial[localOrdinal + localLength * v];
        }
    }

    const Scalar alpha = 0.75;
    const Scalar beta = 0.5;
    Kokkos::Timer firstApplyTimer;
    op->apply(X, Y, Teuchos::NO_TRANS, alpha, beta);
    comparison.firstApplySeconds = firstApplyTimer.seconds();
    auto yResultHost = Y.getLocalViewHost(Tpetra::Access::ReadOnly);

    comparison.maxAbsError = 0.0;
    Scalar maxExpected = 0.0;
    Intrepid2::ScalarView<Scalar,DeviceType> input("Tpetra operator compare direct input", cells, localDofs);
    Intrepid2::ScalarView<Scalar,DeviceType> output("Tpetra operator compare direct output", cells, localDofs);
    auto inputHost = Kokkos::create_mirror_view(input);
    for (int v=0; v<comparison.numVectors; v++)
    {
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
          inputHost(c,f) = xHost(static_cast<size_t>(c * localDofs + f), v);
      Kokkos::deep_copy(input, inputHost);
      TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
      auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
        {
          const size_t localOrdinal = static_cast<size_t>(c * localDofs + f);
          const Scalar expected = alpha * outputHost(c,f) + beta * yInitial[localOrdinal + localLength * v];
          comparison.maxAbsError = std::max(comparison.maxAbsError, std::abs(yResultHost(localOrdinal, v) - expected));
          maxExpected = std::max(maxExpected, std::abs(expected));
        }
    }

    Kokkos::Timer repeatApplyTimer;
    for (int applyOrdinal=0; applyOrdinal<numApply; applyOrdinal++)
    {
      op->apply(X, Y, Teuchos::NO_TRANS, Scalar(1.0), Scalar(0.0));
    }
    comparison.avgApplySeconds = repeatApplyTimer.seconds() / numApply;

    comparison.maxRelError = comparison.maxAbsError / std::max<Scalar>(1.0, maxExpected);
    std::cout << "TPETRA_OPERATOR_COMPARE maxAbsError=" << std::setprecision(16) << comparison.maxAbsError
              << " maxRelError=" << comparison.maxRelError
              << " inverseDiagonalMaxAbsError=" << comparison.inverseDiagonalMaxAbsError
              << " inverseDiagonalMin=" << comparison.inverseDiagonalMin
              << " inverseDiagonalMax=" << comparison.inverseDiagonalMax
              << " firstApplySeconds=" << comparison.firstApplySeconds
              << " avgApplySeconds=" << comparison.avgApplySeconds
              << " numVectors=" << comparison.numVectors
              << " localLength=" << comparison.localLength
              << " domainRangeMapSame=" << comparison.domainRangeMapSame
              << " adapterPath=" << comparison.adapterPath
              << " globalDofSemantics=" << comparison.globalDofSemantics
              << " ifpack2RowMatrix=" << comparison.ifpack2RowMatrix
              << " note=local_flat_map_diagnostic_not_global_fe_operator" << std::endl;
    return comparison;
  }

  TpetraOperatorComparison compareTpetraRowMatrixAgainstDirect(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                                                               const Intrepid2::ScalarView<Scalar,DeviceType> &baseInput,
                                                               const int cells, const int localDofs, const int worksetSize,
                                                               const int numApply, const long long maxAssembledEntries)
  {
    TpetraOperatorComparison comparison;
    comparison.numVectors = 2;
    const size_t localLength = static_cast<size_t>(cells) * static_cast<size_t>(localDofs);
    comparison.localLength = static_cast<int>(localLength);
    comparison.adapterPath = "local_flat_map_nonproduction_rowmatrix";
    comparison.globalDofSemantics = "not_supported";
    comparison.ifpack2RowMatrix = "local_diag_copy_only_no_global_fe_semantics";
    auto comm = Teuchos::DefaultComm<int>::getComm();
    auto map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localLength, 0, comm));
    auto rowMatrix = TpetraAdapters::createLocalPAMatrixTpetraRowMatrix(map, terms, workspaces, cells, localDofs, worksetSize, maxAssembledEntries);
    comparison.domainRangeMapSame = (rowMatrix->getDomainMap().getRawPtr() == rowMatrix->getRangeMap().getRawPtr()) ? 1 : 0;
    comparison.rowMatrixRowMapSame = (rowMatrix->getRowMap().getRawPtr() == rowMatrix->getRangeMap().getRawPtr()) ? 1 : 0;
    comparison.rowMatrixColMapSame = (rowMatrix->getColMap().getRawPtr() == rowMatrix->getDomainMap().getRawPtr()) ? 1 : 0;

    TpetraVector diagonal(map);
    rowMatrix->getLocalDiagCopy(diagonal);
    auto diagonalHost = diagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto expectedDiagonal = TpetraAdapters::extractCombinedDiagonal(terms, cells, localDofs);
    comparison.rowMatrixDiagonalMaxAbsError = 0.0;
    comparison.rowMatrixDiagonalMin = std::numeric_limits<Scalar>::max();
    comparison.rowMatrixDiagonalMax = 0.0;
    for (size_t localOrdinal=0; localOrdinal<localLength; localOrdinal++)
    {
      comparison.rowMatrixDiagonalMaxAbsError = std::max(comparison.rowMatrixDiagonalMaxAbsError,
                                                         std::abs(diagonalHost(localOrdinal,0) - expectedDiagonal[localOrdinal]));
      comparison.rowMatrixDiagonalMin = std::min(comparison.rowMatrixDiagonalMin, diagonalHost(localOrdinal,0));
      comparison.rowMatrixDiagonalMax = std::max(comparison.rowMatrixDiagonalMax, diagonalHost(localOrdinal,0));
    }

    const long long entries = static_cast<long long>(cells) * localDofs * localDofs;
    const size_t flatLength = localLength;
    const long long materializedRowEntries = (flatLength > static_cast<size_t>(std::numeric_limits<long long>::max() / std::max<size_t>(size_t(1), flatLength)))
                                          ? std::numeric_limits<long long>::max()
                                          : static_cast<long long>(flatLength * flatLength);
    comparison.rowMatrixDenseRowMaterializationEntries = materializedRowEntries;
    comparison.rowMatrixMaxDenseRowMaterializationEntries = maxAssembledEntries;
    if ((entries <= maxAssembledEntries) && (materializedRowEntries <= maxAssembledEntries))
    {
      std::vector<Scalar> localMatrices(static_cast<size_t>(entries), Scalar(0.0));
      for (auto &term : terms)
      {
        auto matrixData = term.matrix.allocateMatrixStorage();
        term.matrix.assemble(matrixData);
        Kokkos::fence();
        if (matrixData.getUnderlyingViewRank() == 3)
        {
          auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<3>());
          for (int c=0; c<cells; c++)
            for (int i=0; i<localDofs; i++)
              for (int j=0; j<localDofs; j++)
                localMatrices[static_cast<size_t>((c * localDofs + i) * localDofs + j)] += term.weight * matrixHost(c,i,j);
        }
        else if (matrixData.getUnderlyingViewRank() == 2)
        {
          auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<2>());
          for (int c=0; c<cells; c++)
            for (int i=0; i<localDofs; i++)
              for (int j=0; j<localDofs; j++)
                localMatrices[static_cast<size_t>((c * localDofs + i) * localDofs + j)] += term.weight * matrixHost(i,j);
        }
        else
        {
          throw std::invalid_argument("Unexpected PAMatrix storage rank in RowMatrix row-copy diagnostic");
        }
      }

      TpetraRowMatrix::nonconst_local_inds_host_view_type localIndices("rowmatrix diagnostic local indices", localLength);
      TpetraRowMatrix::nonconst_values_host_view_type values("rowmatrix diagnostic row values", localLength);
      comparison.rowMatrixRowCopyMaxAbsError = 0.0;
      comparison.rowMatrixRowCopyChecked = 1;
      for (size_t row=0; row<localLength; row++)
      {
        size_t numEntries = 0;
        rowMatrix->getLocalRowCopy(static_cast<TpetraRowMatrix::local_ordinal_type>(row), localIndices, values, numEntries);
        if (numEntries != localLength) throw std::runtime_error("RowMatrix diagnostic returned unexpected row length");
        const int cell = static_cast<int>(row / static_cast<size_t>(localDofs));
        const int rowInCell = static_cast<int>(row % static_cast<size_t>(localDofs));
        for (size_t column=0; column<localLength; column++)
        {
          if (localIndices(column) != static_cast<TpetraRowMatrix::local_ordinal_type>(column))
            throw std::runtime_error("RowMatrix diagnostic returned unexpected local column ordinal");
          const int columnCell = static_cast<int>(column / static_cast<size_t>(localDofs));
          const int columnInCell = static_cast<int>(column % static_cast<size_t>(localDofs));
          const Scalar expected = (columnCell == cell)
                                ? localMatrices[static_cast<size_t>((cell * localDofs + rowInCell) * localDofs + columnInCell)]
                                : Scalar(0.0);
          comparison.rowMatrixRowCopyMaxAbsError = std::max(comparison.rowMatrixRowCopyMaxAbsError, std::abs(values(column) - expected));
        }
      }

      TpetraRowMatrix::local_inds_host_view_type rowViewIndices;
      TpetraRowMatrix::values_host_view_type rowViewValues;
      comparison.rowMatrixRowViewMaxAbsError = 0.0;
      comparison.rowMatrixRowViewChecked = 1;
      for (size_t row=0; row<localLength; row++)
      {
        rowMatrix->getLocalRowView(static_cast<TpetraRowMatrix::local_ordinal_type>(row), rowViewIndices, rowViewValues);
        if (rowViewIndices.extent(0) != localLength) throw std::runtime_error("RowMatrix diagnostic returned unexpected row-view length");
        if (rowViewValues.extent(0) != localLength) throw std::runtime_error("RowMatrix diagnostic returned unexpected row-view value length");
        const int cell = static_cast<int>(row / static_cast<size_t>(localDofs));
        const int rowInCell = static_cast<int>(row % static_cast<size_t>(localDofs));
        for (size_t column=0; column<localLength; column++)
        {
          if (rowViewIndices(column) != static_cast<TpetraRowMatrix::local_ordinal_type>(column))
            throw std::runtime_error("RowMatrix diagnostic returned unexpected local row-view column ordinal");
          const int columnCell = static_cast<int>(column / static_cast<size_t>(localDofs));
          const int columnInCell = static_cast<int>(column % static_cast<size_t>(localDofs));
          const Scalar expected = (columnCell == cell)
                                ? localMatrices[static_cast<size_t>((cell * localDofs + rowInCell) * localDofs + columnInCell)]
                                : Scalar(0.0);
          comparison.rowMatrixRowViewMaxAbsError = std::max(comparison.rowMatrixRowViewMaxAbsError, std::abs(rowViewValues(column) - expected));
        }
      }
    }
    else
    {
      std::cout << "TPETRA_ROWMATRIX_ROW_COPY_SKIPPED entries=" << entries
                << " materializedRowEntries=" << comparison.rowMatrixDenseRowMaterializationEntries
                << " maxAssembledEntries=" << comparison.rowMatrixMaxDenseRowMaterializationEntries
                << " rowViewChecked=" << comparison.rowMatrixRowViewChecked
                << " note=guarded_tiny_case_only_row_copy_and_view" << std::endl;
    }

    TpetraMultiVector X(map, comparison.numVectors);
    TpetraMultiVector Y(map, comparison.numVectors);
    auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto yHost = Y.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto baseHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), baseInput);
    std::vector<Scalar> yInitial(localLength * comparison.numVectors, Scalar(0.0));
    for (int v=0; v<comparison.numVectors; v++)
    {
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
        {
          const size_t localOrdinal = static_cast<size_t>(c * localDofs + f);
          xHost(localOrdinal, v) = baseHost(c,f) * (Scalar(1.0) + Scalar(0.25) * v);
          yInitial[localOrdinal + localLength * v] = Scalar(0.125) * (Scalar(1.0) + Scalar((localOrdinal + v) % 5));
          yHost(localOrdinal, v) = yInitial[localOrdinal + localLength * v];
        }
    }

    const Scalar alpha = 0.75;
    const Scalar beta = 0.5;
    Kokkos::Timer firstApplyTimer;
    rowMatrix->apply(X, Y, Teuchos::NO_TRANS, alpha, beta);
    comparison.firstApplySeconds = firstApplyTimer.seconds();
    auto yResultHost = Y.getLocalViewHost(Tpetra::Access::ReadOnly);

    comparison.maxAbsError = 0.0;
    Scalar maxExpected = 0.0;
    Intrepid2::ScalarView<Scalar,DeviceType> input("Tpetra rowmatrix compare direct input", cells, localDofs);
    Intrepid2::ScalarView<Scalar,DeviceType> output("Tpetra rowmatrix compare direct output", cells, localDofs);
    auto inputHost = Kokkos::create_mirror_view(input);
    for (int v=0; v<comparison.numVectors; v++)
    {
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
          inputHost(c,f) = xHost(static_cast<size_t>(c * localDofs + f), v);
      Kokkos::deep_copy(input, inputHost);
      TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
      auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
      for (int c=0; c<cells; c++)
        for (int f=0; f<localDofs; f++)
        {
          const size_t localOrdinal = static_cast<size_t>(c * localDofs + f);
          const Scalar expected = alpha * outputHost(c,f) + beta * yInitial[localOrdinal + localLength * v];
          comparison.maxAbsError = std::max(comparison.maxAbsError, std::abs(yResultHost(localOrdinal, v) - expected));
          maxExpected = std::max(maxExpected, std::abs(expected));
        }
    }

    Kokkos::Timer repeatApplyTimer;
    for (int applyOrdinal=0; applyOrdinal<numApply; applyOrdinal++)
    {
      rowMatrix->apply(X, Y, Teuchos::NO_TRANS, Scalar(1.0), Scalar(0.0));
    }
    comparison.avgApplySeconds = repeatApplyTimer.seconds() / numApply;

    comparison.maxRelError = comparison.maxAbsError / std::max<Scalar>(1.0, maxExpected);
    std::cout << "TPETRA_ROWMATRIX_COMPARE maxAbsError=" << std::setprecision(16) << comparison.maxAbsError
              << " maxRelError=" << comparison.maxRelError
              << " diagonalMaxAbsError=" << comparison.rowMatrixDiagonalMaxAbsError
              << " diagonalMin=" << comparison.rowMatrixDiagonalMin
              << " diagonalMax=" << comparison.rowMatrixDiagonalMax
              << " rowCopyMaxAbsError=" << comparison.rowMatrixRowCopyMaxAbsError
              << " rowCopyChecked=" << comparison.rowMatrixRowCopyChecked
              << " rowViewMaxAbsError=" << comparison.rowMatrixRowViewMaxAbsError
              << " rowViewChecked=" << comparison.rowMatrixRowViewChecked
              << " firstApplySeconds=" << comparison.firstApplySeconds
              << " avgApplySeconds=" << comparison.avgApplySeconds
              << " numVectors=" << comparison.numVectors
              << " localLength=" << comparison.localLength
              << " domainRangeMapSame=" << comparison.domainRangeMapSame
              << " rowRangeMapSame=" << comparison.rowMatrixRowMapSame
              << " colDomainMapSame=" << comparison.rowMatrixColMapSame
              << " localNumRows=" << rowMatrix->getLocalNumRows()
              << " localNumCols=" << rowMatrix->getLocalNumCols()
              << " localNumEntries=" << rowMatrix->getLocalNumEntries()
              << " localMaxNumRowEntries=" << rowMatrix->getLocalMaxNumRowEntries()
              << " denseRowMaterializationEntries=" << comparison.rowMatrixDenseRowMaterializationEntries
              << " maxDenseRowMaterializationEntries=" << comparison.rowMatrixMaxDenseRowMaterializationEntries
              << " adapterPath=" << comparison.adapterPath
              << " globalDofSemantics=" << comparison.globalDofSemantics
              << " ifpack2RowMatrix=" << comparison.ifpack2RowMatrix
              << " note=local_flat_rowmatrix_diagnostic_expensive_row_extraction_not_global_fe_operator" << std::endl;
    return comparison;
  }

  bool compareApplyAgainstAssembled(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                                    const Intrepid2::ScalarView<Scalar,DeviceType> &input,
                                    Intrepid2::ScalarView<Scalar,DeviceType> &output, const int worksetSize,
                                    const long long maxAssembledEntries)
  {
    const int cells = input.extent_int(0);
    const int cols  = input.extent_int(1);
    const int rows  = output.extent_int(1);
    const long long entries = static_cast<long long>(cells) * rows * cols;
    if (entries > maxAssembledEntries)
    {
      std::cout << "COMPARE_SKIPPED entries=" << entries << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
      return true;
    }

    TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
    auto inputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input);
    auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    std::vector<Scalar> expected(cells * rows, 0.0);

    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      term.matrix.assemble(matrixData);
      if (matrixData.getUnderlyingViewRank() == 3)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<3>());
        for (int c=0; c<cells; c++)
        {
          for (int i=0; i<rows; i++)
          {
            Scalar value = 0.0;
            for (int j=0; j<cols; j++) value += matrixHost(c,i,j) * inputHost(c,j);
            expected[c * rows + i] += term.weight * value;
          }
        }
      }
      else if (matrixData.getUnderlyingViewRank() == 2)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
        {
          for (int i=0; i<rows; i++)
          {
            Scalar value = 0.0;
            for (int j=0; j<cols; j++) value += matrixHost(i,j) * inputHost(c,j);
            expected[c * rows + i] += term.weight * value;
          }
        }
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix storage rank in apply-vs-assembled compare");
      }
    }

    Scalar maxError = 0.0;
    Scalar maxExpected = 0.0;
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        maxError = std::max(maxError, std::abs(outputHost(c,i) - expected[c * rows + i]));
        maxExpected = std::max(maxExpected, std::abs(expected[c * rows + i]));
      }
    }
    const Scalar relError = maxError / std::max<Scalar>(1.0, maxExpected);
    constexpr Scalar compareAbsTol = 1e-10;
    constexpr Scalar compareRelTol = 1e-10;
    const bool passed = (maxError <= compareAbsTol) || (relError <= compareRelTol);
    std::cout << "COMPARE passed=" << (passed ? 1 : 0)
              << " maxAbsError=" << std::setprecision(16) << maxError
              << " maxRelError=" << relError
              << " absTol=" << compareAbsTol
              << " relTol=" << compareRelTol
              << " note=apply_vs_assembled" << std::endl;
    return passed;
  }

  AssemblyTiming assembleTerms(std::vector<OperatorTerm> &terms, const long long maxAssembledEntries)
  {
    AssemblyTiming timing;
    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      const long long entries = static_cast<long long>(matrixData.extent_int(0))
                              * static_cast<long long>(matrixData.extent_int(1))
                              * static_cast<long long>(matrixData.extent_int(2));
      if (entries > maxAssembledEntries)
      {
        std::cout << "ASSEMBLE_SKIPPED term=" << term.label
                  << " entries=" << entries
                  << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
        return timing;
      }
      Kokkos::Timer timer;
      term.matrix.assemble(matrixData);
      Kokkos::fence();
      addTermSeconds(timing, term.label, timer.seconds());
    }
    return timing;
  }

  AssemblyTiming assemble2Terms(std::vector<OperatorTerm> &terms, const long long maxAssembledEntries)
  {
    AssemblyTiming timing;
    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      const long long entries = static_cast<long long>(matrixData.extent_int(0))
                              * static_cast<long long>(matrixData.extent_int(1))
                              * static_cast<long long>(matrixData.extent_int(2));
      if (entries > maxAssembledEntries)
      {
        std::cout << "ASSEMBLE2_SKIPPED term=" << term.label
                  << " entries=" << entries
                  << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
        return timing;
      }
      Kokkos::Timer timer;
      term.matrix.assemble2(matrixData);
      Kokkos::fence();
      addTermSeconds(timing, term.label, timer.seconds());
    }
    return timing;
  }

  AssemblyComparison compareAssemble2AgainstAssemble(std::vector<OperatorTerm> &terms, const long long maxAssembledEntries)
  {
    AssemblyComparison comparison;
    double maxExpected = 0.0;
    comparison.maxAbsError = 0.0;

    for (auto &term : terms)
    {
      auto expectedData = term.matrix.allocateMatrixStorage();
      auto actualData   = term.matrix.allocateMatrixStorage();
      const long long entries = static_cast<long long>(expectedData.extent_int(0))
                              * static_cast<long long>(expectedData.extent_int(1))
                              * static_cast<long long>(expectedData.extent_int(2));
      if (entries > maxAssembledEntries)
      {
        std::cout << "ASSEMBLY_AGREEMENT_SKIPPED term=" << term.label
                  << " entries=" << entries
                  << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
        comparison.maxAbsError = -1.0;
        comparison.maxRelError = -1.0;
        return comparison;
      }

      term.matrix.assemble(expectedData);
      term.matrix.assemble2(actualData);
      Kokkos::fence();

      if (expectedData.getUnderlyingViewRank() == 3)
      {
        auto expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expectedData.template getUnderlyingView<3>());
        auto actual   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actualData.template getUnderlyingView<3>());
        for (int c=0; c<expectedData.extent_int(0); c++)
          for (int i=0; i<expectedData.extent_int(1); i++)
            for (int j=0; j<expectedData.extent_int(2); j++)
            {
              const double expectedValue = term.weight * expected(c,i,j);
              const double actualValue   = term.weight * actual(c,i,j);
              comparison.maxAbsError = std::max(comparison.maxAbsError, std::abs(actualValue - expectedValue));
              maxExpected = std::max(maxExpected, std::abs(expectedValue));
            }
      }
      else if (expectedData.getUnderlyingViewRank() == 2)
      {
        auto expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expectedData.template getUnderlyingView<2>());
        auto actual   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actualData.template getUnderlyingView<2>());
        for (int i=0; i<expectedData.extent_int(1); i++)
          for (int j=0; j<expectedData.extent_int(2); j++)
          {
            const double expectedValue = term.weight * expected(i,j);
            const double actualValue   = term.weight * actual(i,j);
            comparison.maxAbsError = std::max(comparison.maxAbsError, std::abs(actualValue - expectedValue));
            maxExpected = std::max(maxExpected, std::abs(expectedValue));
          }
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix storage rank in assembly agreement check");
      }
    }

    comparison.maxRelError = comparison.maxAbsError / std::max(1.0, maxExpected);
    std::cout << "ASSEMBLY_AGREEMENT maxAbsError=" << std::setprecision(16) << comparison.maxAbsError
              << " maxRelError=" << comparison.maxRelError << std::endl;
    return comparison;
  }

  struct RepeatedTiming
  {
    double firstSeconds = -1.0;
    double averageSeconds = -1.0;
  };

  int factorSPDMatrixInPlace(std::vector<Scalar> &matrix, const size_t offset, const int size)
  {
    Teuchos::LAPACK<int,Scalar> lapack;
    char uplo = 'L';
    int info = 0;
    // The local matrix is symmetric.  The host vector is indexed row-major by the
    // residual check below, while LAPACK interprets it as column-major; for an SPD
    // matrix these are equivalent up to transpose, and POTRF/POTRS use the same
    // convention consistently for the factor and solves.
    lapack.POTRF(uplo, size, matrix.data() + offset, size, &info);
    return info;
  }

  int solveFactoredSPDMatrix(const std::vector<Scalar> &matrix, const size_t offset, const int size,
                             std::vector<Scalar> &rhs)
  {
    Teuchos::LAPACK<int,Scalar> lapack;
    char uplo = 'L';
    int numRhs = 1;
    int info = 0;
    lapack.POTRS(uplo, size, numRhs, matrix.data() + offset, size, rhs.data(), size, &info);
    return info;
  }

  std::vector<Scalar> computeTrueEigenDiagnostics(const std::vector<Scalar> &localMatrices, const std::vector<Scalar> &diagonal,
                                                  const int cells, const int rows, InverseDiagnosticTiming &timing)
  {
    if ((cells < 1) || (rows < 1)) throw std::invalid_argument("true eigen diagnostic requires positive local extents");
    if (localMatrices.size() != static_cast<size_t>(cells * rows * rows))
      throw std::invalid_argument("true eigen diagnostic local matrix size mismatch");
    if (diagonal.size() != static_cast<size_t>(cells * rows))
      throw std::invalid_argument("true eigen diagnostic diagonal size mismatch");

    Teuchos::LAPACK<int,Scalar> lapack;
    char jobz = 'N';
    char uplo = 'U';
    timing.trueLambdaMin = std::numeric_limits<Scalar>::max();
    timing.trueLambdaMax = -std::numeric_limits<Scalar>::max();
    std::vector<Scalar> allEigenvalues;
    allEigenvalues.reserve(static_cast<size_t>(cells * rows));

    for (int c=0; c<cells; c++)
    {
      std::vector<Scalar> eigenMatrix(static_cast<size_t>(rows * rows), Scalar(0.0));
      std::vector<Scalar> eigenvalues(static_cast<size_t>(rows), Scalar(0.0));
      const size_t cellOffset = static_cast<size_t>(c * rows);
      for (int i=0; i<rows; i++)
      {
        const Scalar di = diagonal[cellOffset + i];
        if (!std::isfinite(di) || (di <= Scalar(0.0)))
          throw std::runtime_error("true eigen diagnostic found nonpositive or nonfinite diagonal entry");
        for (int j=0; j<rows; j++)
        {
          const Scalar dj = diagonal[cellOffset + j];
          if (!std::isfinite(dj) || (dj <= Scalar(0.0)))
            throw std::runtime_error("true eigen diagnostic found nonpositive or nonfinite diagonal entry");
          const Scalar aij = localMatrices[static_cast<size_t>((c * rows + i) * rows + j)];
          const Scalar aji = localMatrices[static_cast<size_t>((c * rows + j) * rows + i)];
          eigenMatrix[static_cast<size_t>(j * rows + i)] = Scalar(0.5) * (aij + aji) / std::sqrt(di * dj);
        }
      }

      Scalar workQuery = Scalar(0.0);
      int lwork = -1;
      int info = 0;
      lapack.SYEV(jobz, uplo, rows, eigenMatrix.data(), rows, eigenvalues.data(), &workQuery, lwork, &info);
      if (info != 0) throw std::runtime_error("true eigen diagnostic LAPACK SYEV workspace query failed with info=" + std::to_string(info));
      lwork = std::max(1, static_cast<int>(workQuery));
      std::vector<Scalar> work(static_cast<size_t>(lwork), Scalar(0.0));
      lapack.SYEV(jobz, uplo, rows, eigenMatrix.data(), rows, eigenvalues.data(), work.data(), lwork, &info);
      if (info != 0) throw std::runtime_error("true eigen diagnostic LAPACK SYEV failed with info=" + std::to_string(info));

      timing.trueLambdaMin = std::min(timing.trueLambdaMin, eigenvalues.front());
      timing.trueLambdaMax = std::max(timing.trueLambdaMax, eigenvalues.back());
      allEigenvalues.insert(allEigenvalues.end(), eigenvalues.begin(), eigenvalues.end());
    }

    if ((timing.trueLambdaMin > Scalar(0.0)) && (timing.trueLambdaMax > Scalar(0.0)))
      timing.trueCondition = timing.trueLambdaMax / timing.trueLambdaMin;
    timing.trueEigenDiagnostic = 1;
    timing.trueEigenPath = "assembled_lapack_syev_symmetric_diagonal_scaling";
    return allEigenvalues;
  }

  int ordinalFromMultiIndex(const std::vector<int> &index, const std::vector<int> &extents)
  {
    if (index.size() != extents.size()) throw std::invalid_argument("ordinalFromMultiIndex rank mismatch");
    int ordinal = 0;
    int stride = 1;
    for (size_t d=0; d<index.size(); d++)
    {
      ordinal += index[d] * stride;
      stride *= extents[d];
    }
    return ordinal;
  }

  struct TensorHGradPTransfer
  {
    int dim = 0;
    int fineP = 0;
    int coarseP = 0;
    int fineRows = 0;
    int coarseRows = 0;
    std::vector<Scalar> prolongation;
    std::vector<Scalar> restriction;
  };

  const TensorHGradPTransfer &cachedTensorHGradPTransfer(const int dim, const int fineP)
  {
    if ((dim < 1) || (fineP < 2)) throw std::invalid_argument("p-multigrid diagnostic requires dim >= 1 and p >= 2");
    using CacheKey = std::pair<int,int>;
    static std::map<CacheKey,TensorHGradPTransfer> cache;
    const CacheKey key(dim, fineP);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    TensorHGradPTransfer transfer;
    transfer.dim = dim;
    transfer.fineP = fineP;
    transfer.coarseP = fineP - 1;
    const std::vector<int> fineExtents(static_cast<size_t>(dim), fineP + 1);
    const std::vector<int> coarseExtents(static_cast<size_t>(dim), transfer.coarseP + 1);
    transfer.fineRows = checkedProduct(fineExtents);
    transfer.coarseRows = checkedProduct(coarseExtents);

    using BasisFamily = Intrepid2::HierarchicalBasisFamily<DeviceType>;
    auto coarseBasis = Intrepid2::getHypercubeBasis_HGRAD<BasisFamily>(transfer.coarseP, dim);
    auto fineBasis = Intrepid2::getHypercubeBasis_HGRAD<BasisFamily>(fineP, dim);
    Intrepid2::HierarchicalBasisProlongator<DeviceType,Scalar> prolongator(coarseBasis, fineBasis);
    if (coarseBasis->getCardinality() != transfer.coarseRows)
      throw std::runtime_error("hierarchical p-multigrid coarse basis cardinality did not match tensor-product row count");
    if (fineBasis->getCardinality() != transfer.fineRows)
      throw std::runtime_error("hierarchical p-multigrid fine basis cardinality did not match tensor-product row count");

    View2D coarseIdentity("TensorHGradPTransfer hierarchical coarse identity", transfer.coarseRows, transfer.coarseRows);
    View2D fineProlongedIdentity("TensorHGradPTransfer hierarchical fine identity", transfer.coarseRows, transfer.fineRows);
    Kokkos::deep_copy(coarseIdentity, Scalar(0.0));
    auto coarseIdentityHost = Kokkos::create_mirror_view(coarseIdentity);
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      coarseIdentityHost(coarseOrdinal, coarseOrdinal) = Scalar(1.0);
    Kokkos::deep_copy(coarseIdentity, coarseIdentityHost);
    prolongator.apply(fineProlongedIdentity, coarseIdentity);
    Kokkos::fence();

    auto fineProlongedIdentityHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fineProlongedIdentity);
    transfer.prolongation.assign(static_cast<size_t>(transfer.fineRows) * transfer.coarseRows, Scalar(0.0));
    transfer.restriction.assign(static_cast<size_t>(transfer.coarseRows) * transfer.fineRows, Scalar(0.0));
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
      {
        const Scalar value = fineProlongedIdentityHost(coarseOrdinal, fineOrdinal);
        transfer.prolongation[static_cast<size_t>(fineOrdinal * transfer.coarseRows + coarseOrdinal)] = value;
        transfer.restriction[static_cast<size_t>(coarseOrdinal * transfer.fineRows + fineOrdinal)] = value;
      }

    auto inserted = cache.emplace(key, std::move(transfer));
    return inserted.first->second;
  }

  void applyTensorHGradPTransferProlongation(const TensorHGradPTransfer &transfer,
                                             const std::vector<Scalar> &coarseVector,
                                             std::vector<Scalar> &fineVector)
  {
    if (static_cast<int>(coarseVector.size()) != transfer.coarseRows) throw std::runtime_error("p-multigrid prolongation coarse vector has incorrect length");
    fineVector.assign(static_cast<size_t>(transfer.fineRows), Scalar(0.0));
    for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
      for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
        fineVector[static_cast<size_t>(fineOrdinal)] += transfer.prolongation[static_cast<size_t>(fineOrdinal * transfer.coarseRows + coarseOrdinal)] * coarseVector[static_cast<size_t>(coarseOrdinal)];
  }

  void applyTensorHGradPTransferRestriction(const TensorHGradPTransfer &transfer,
                                            const std::vector<Scalar> &fineVector,
                                            std::vector<Scalar> &coarseVector)
  {
    if (static_cast<int>(fineVector.size()) != transfer.fineRows) throw std::runtime_error("p-multigrid restriction fine vector has incorrect length");
    coarseVector.assign(static_cast<size_t>(transfer.coarseRows), Scalar(0.0));
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
        coarseVector[static_cast<size_t>(coarseOrdinal)] += transfer.restriction[static_cast<size_t>(coarseOrdinal * transfer.fineRows + fineOrdinal)] * fineVector[static_cast<size_t>(fineOrdinal)];
  }

  class LocalPMultigridTpetraPreconditioner : public TpetraOperator
  {
    Teuchos::RCP<const TpetraMap> map_;
    Teuchos::RCP<const TpetraMap> coarseMap_;
    const TensorHGradPTransfer *transfer_ = nullptr;
    int cells_ = 0;
    int localDofs_ = 0;
    int coarseRows_ = 0;
    int smootherIterations_ = 1;
    Scalar smootherLowerFraction_ = Scalar(0.3);
    Scalar smootherWeight_ = Scalar(1.0);
    std::string smootherName_;
    std::vector<Scalar> fineMatrices_;
    std::vector<Scalar> fineInverseDiagonal_;
    std::vector<Scalar> fineLambdaMaxByCell_;
    std::vector<Scalar> coarseMatrices_;
    std::vector<Scalar> factoredCoarseMatrices_;
    std::string description_;

    static void applyDenseCellMatrix(const std::vector<Scalar> &matrices, const int cellOrdinal,
                                     const int rows, const std::vector<Scalar> &input,
                                     std::vector<Scalar> &output)
    {
      output.assign(static_cast<size_t>(rows), Scalar(0.0));
      const size_t offset = static_cast<size_t>(cellOrdinal) * rows * rows;
      for (int i=0; i<rows; i++)
        for (int j=0; j<rows; j++)
          output[static_cast<size_t>(i)] += matrices[offset + static_cast<size_t>(i * rows + j)] * input[static_cast<size_t>(j)];
    }

    static void runDampedJacobiSmoother(const std::vector<Scalar> &matrices, const std::vector<Scalar> &inverseDiagonal,
                                        const std::vector<Scalar> &lambdaMaxByCell, const int cellOrdinal,
                                        const int rows, const std::vector<Scalar> &rhs,
                                        std::vector<Scalar> &solution, const int iterations)
    {
      std::vector<Scalar> applied(static_cast<size_t>(rows), Scalar(0.0));
      const Scalar lambdaMax = std::max(lambdaMaxByCell[static_cast<size_t>(cellOrdinal)], Scalar(1.0));
      const Scalar omega = Scalar(1.0) / lambdaMax;
      const size_t cellOffset = static_cast<size_t>(cellOrdinal) * rows;
      for (int iteration=0; iteration<iterations; iteration++)
      {
        applyDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
        for (int i=0; i<rows; i++)
        {
          const size_t index = cellOffset + static_cast<size_t>(i);
          solution[static_cast<size_t>(i)] += omega * inverseDiagonal[index] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)]);
        }
      }
    }

    static void runChebyshevSmoother(const std::vector<Scalar> &matrices, const std::vector<Scalar> &inverseDiagonal,
                                     const std::vector<Scalar> &lambdaMaxByCell, const int cellOrdinal,
                                     const int rows, const std::vector<Scalar> &rhs,
                                     std::vector<Scalar> &solution, const int iterations,
                                     const Scalar lowerFraction)
    {
      if (iterations < 1) return;
      std::vector<Scalar> correction(static_cast<size_t>(rows), Scalar(0.0));
      std::vector<Scalar> applied(static_cast<size_t>(rows), Scalar(0.0));
      const Scalar chebyshevLambdaMax = std::max(lambdaMaxByCell[static_cast<size_t>(cellOrdinal)], Scalar(1.0));
      const Scalar chebyshevLambdaMin = std::max(std::numeric_limits<Scalar>::epsilon(), lowerFraction * chebyshevLambdaMax);
      const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
      const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
      const Scalar s1 = ifpackTheta * ifpackDelta;
      Scalar rhok = Scalar(1.0) / s1;
      const size_t cellOffset = static_cast<size_t>(cellOrdinal) * rows;
      applyDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
      for (int i=0; i<rows; i++)
      {
        const size_t index = cellOffset + static_cast<size_t>(i);
        correction[static_cast<size_t>(i)] = (Scalar(1.0) / ifpackTheta) * inverseDiagonal[index] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)]);
        solution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
      }
      for (int iteration=1; iteration<iterations; iteration++)
      {
        applyDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
        const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhok);
        const Scalar dtemp1 = rhokp1 * rhok;
        const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
        rhok = rhokp1;
        for (int i=0; i<rows; i++)
        {
          const size_t index = cellOffset + static_cast<size_t>(i);
          correction[static_cast<size_t>(i)] = dtemp2 * inverseDiagonal[index] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)])
                                             + dtemp1 * correction[static_cast<size_t>(i)];
          solution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
        }
      }
    }

    void smoothFine(const int cellOrdinal, const std::vector<Scalar> &rhs, std::vector<Scalar> &solution) const
    {
      std::vector<Scalar> before;
      if (smootherWeight_ != Scalar(1.0)) before = solution;
      if (smootherName_ == "chebyshev")
        runChebyshevSmoother(fineMatrices_, fineInverseDiagonal_, fineLambdaMaxByCell_, cellOrdinal, localDofs_, rhs, solution, smootherIterations_, smootherLowerFraction_);
      else
        runDampedJacobiSmoother(fineMatrices_, fineInverseDiagonal_, fineLambdaMaxByCell_, cellOrdinal, localDofs_, rhs, solution, smootherIterations_);
      if (smootherWeight_ != Scalar(1.0))
        for (size_t i=0; i<solution.size(); i++) solution[i] = before[i] + smootherWeight_ * (solution[i] - before[i]);
    }
    
    static std::vector<Scalar> extractInverseDiagonalFromBlockDenseMatrices(
        const std::vector<Scalar> &matrices,
        const int cells,
        const int rows)
    {
      const size_t localLength =
          static_cast<size_t>(cells) * static_cast<size_t>(rows);

      const size_t expectedEntries =
          localLength * static_cast<size_t>(rows);

      if (matrices.size() != expectedEntries)
      {
        std::ostringstream message;
        message << "extractInverseDiagonalFromBlockDenseMatrices matrix size mismatch: "
                << "matrices.size()=" << matrices.size()
                << " expected=" << expectedEntries
                << " cells=" << cells
                << " rows=" << rows;
        throw std::invalid_argument(message.str());
      }

      std::vector<Scalar> inverseDiagonal(localLength, Scalar(0.0));

      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          const size_t localOrdinal =
              static_cast<size_t>(c) * static_cast<size_t>(rows) +
              static_cast<size_t>(i);

          const size_t diagonalIndex =
              localOrdinal * static_cast<size_t>(rows) +
              static_cast<size_t>(i);

          const Scalar diagonalValue = matrices[diagonalIndex];

          if (diagonalValue <= std::numeric_limits<Scalar>::epsilon())
          {
            std::ostringstream message;
            message << "extractInverseDiagonalFromBlockDenseMatrices found nonpositive "
                    << "diagonal entry"
                    << " cell=" << c
                    << " localDof=" << i
                    << " value=" << diagonalValue;
            throw std::runtime_error(message.str());
          }

          inverseDiagonal[localOrdinal] = Scalar(1.0) / diagonalValue;
        }
      }

      return inverseDiagonal;
    }

  public:
    LocalPMultigridTpetraPreconditioner(const Teuchos::RCP<const TpetraMap> &map,
                                        const Teuchos::RCP<TpetraOperator> &fineOperator,
                                        const int dim, const int p, const int cells,
                                        const int localDofs, const int smootherIterations,
                                        const std::string &smootherName,
                                        const Scalar smootherLowerFraction, const Scalar smootherWeight)
      : map_(map), transfer_(&cachedTensorHGradPTransfer(dim, p)), cells_(cells), localDofs_(localDofs),
        coarseRows_(transfer_->coarseRows), smootherIterations_(std::max(1, smootherIterations)),
        smootherLowerFraction_(std::max(std::numeric_limits<Scalar>::epsilon(), smootherLowerFraction)),
        smootherWeight_(smootherWeight), smootherName_(smootherName)
    {
      if (map_.is_null()) throw std::invalid_argument("LocalPMultigridTpetraPreconditioner requires a non-null map");
      if (p < 2) throw std::invalid_argument("LocalPMultigridTpetraPreconditioner requires p >= 2");
      if (transfer_->fineRows != localDofs) throw std::runtime_error("LocalPMultigridTpetraPreconditioner fine transfer dimension does not match local dofs");
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if (map_->getLocalNumElements() != localLength)
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner map/local dof length mismatch");
      coarseMap_ = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                              static_cast<size_t>(cells_) * static_cast<size_t>(coarseRows_),
                                              map_->getIndexBase(),
                                              map_->getComm()));

      fineMatrices_.assign(localLength * static_cast<size_t>(localDofs_), Scalar(0.0));
      TpetraMultiVector basisInput(map_, 1);
      TpetraMultiVector basisOutput(map_, 1);
      auto basisInputHost = basisInput.getLocalViewHost(Tpetra::Access::OverwriteAll);
      for (int fineJ=0; fineJ<localDofs_; fineJ++)
      {
        for (size_t localOrdinal=0; localOrdinal<localLength; localOrdinal++) basisInputHost(localOrdinal,0) = Scalar(0.0);
        for (int c=0; c<cells_; c++) basisInputHost(static_cast<size_t>(c * localDofs_ + fineJ),0) = Scalar(1.0);
        fineOperator->apply(basisInput, basisOutput);
        auto basisOutputHost = basisOutput.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (int c=0; c<cells_; c++)
          for (int fineI=0; fineI<localDofs_; fineI++)
            fineMatrices_[static_cast<size_t>((c * localDofs_ + fineI) * localDofs_ + fineJ)] = basisOutputHost(static_cast<size_t>(c * localDofs_ + fineI),0);
      }
      
      fineInverseDiagonal_ = extractInverseDiagonalFromBlockDenseMatrices(fineMatrices_, cells_, localDofs_);

      fineLambdaMaxByCell_.assign(static_cast<size_t>(cells_), Scalar(0.0));
      for (int c=0; c<cells_; c++)
      {
        for (int i=0; i<localDofs_; i++)
        {
          const size_t rowOffset = static_cast<size_t>((c * localDofs_ + i) * localDofs_);
          Scalar scaledRowSum = Scalar(0.0);
          for (int j=0; j<localDofs_; j++) scaledRowSum += std::abs(fineMatrices_[rowOffset + static_cast<size_t>(j)]) * fineInverseDiagonal_[static_cast<size_t>(c * localDofs_ + i)];
          fineLambdaMaxByCell_[static_cast<size_t>(c)] = std::max(fineLambdaMaxByCell_[static_cast<size_t>(c)], scaledRowSum);
        }
      }

      coarseMatrices_.assign(static_cast<size_t>(cells_) * coarseRows_ * coarseRows_, Scalar(0.0));
      std::vector<Scalar> matrixTimesP(static_cast<size_t>(localDofs_ * coarseRows_), Scalar(0.0));
      for (int c=0; c<cells_; c++)
      {
        std::fill(matrixTimesP.begin(), matrixTimesP.end(), Scalar(0.0));
        for (int fineI=0; fineI<localDofs_; fineI++)
          for (int fineJ=0; fineJ<localDofs_; fineJ++)
          {
            const Scalar matrixEntry = fineMatrices_[static_cast<size_t>((c * localDofs_ + fineI) * localDofs_ + fineJ)];
            if (matrixEntry == Scalar(0.0)) continue;
            for (int coarseJ=0; coarseJ<coarseRows_; coarseJ++)
              matrixTimesP[static_cast<size_t>(fineI * coarseRows_ + coarseJ)] += matrixEntry * transfer_->prolongation[static_cast<size_t>(fineJ * coarseRows_ + coarseJ)];
          }
        for (int coarseI=0; coarseI<coarseRows_; coarseI++)
          for (int fineI=0; fineI<localDofs_; fineI++)
          {
            const Scalar restrictionEntry = transfer_->restriction[static_cast<size_t>(coarseI * localDofs_ + fineI)];
            if (restrictionEntry == Scalar(0.0)) continue;
            for (int coarseJ=0; coarseJ<coarseRows_; coarseJ++)
              coarseMatrices_[static_cast<size_t>((c * coarseRows_ + coarseI) * coarseRows_ + coarseJ)] += restrictionEntry * matrixTimesP[static_cast<size_t>(fineI * coarseRows_ + coarseJ)];
          }
      }

      factoredCoarseMatrices_ = coarseMatrices_;
      for (int c=0; c<cells_; c++)
      {
        const int factorInfo = factorSPDMatrixInPlace(factoredCoarseMatrices_, static_cast<size_t>(c) * coarseRows_ * coarseRows_, coarseRows_);
        if (factorInfo != 0) throw std::runtime_error("LocalPMultigridTpetraPreconditioner coarse Cholesky factorization failed with LAPACK info=" + std::to_string(factorInfo));
      }

      std::ostringstream desc;
      desc << "LocalPMultigridTpetraPreconditioner(two_level_v_cycle,dim=" << dim
           << ",fineP=" << p << ",coarseP=" << transfer_->coarseP
           << ",fineRows=" << transfer_->fineRows << ",coarseRows=" << transfer_->coarseRows
           << ",smoother=" << smootherName_ << ",smootherIterations=" << smootherIterations_
           << ",smootherLowerFraction=" << smootherLowerFraction_
           << ",coarse=exact_cholesky)";
      description_ = desc.str();
    }

    Teuchos::RCP<const TpetraMap> coarseMap() const { return coarseMap_; }

    void applyFineOperator(const TpetraMultiVector &X, TpetraMultiVector &Y,
                           Teuchos::ETransp mode = Teuchos::NO_TRANS,
                           Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                           Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const
    {
      if (mode != Teuchos::NO_TRANS) throw std::invalid_argument("LocalPMultigridTpetraPreconditioner fine operator does not support transpose apply");
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((X.getLocalLength() != localLength) || (Y.getLocalLength() != localLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner fine operator local vector length mismatch");
      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner fine operator multivector count mismatch");

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> input(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> output(static_cast<size_t>(localDofs_), Scalar(0.0));
      for (size_t vectorOrdinal=0; vectorOrdinal<X.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t cellOffset = static_cast<size_t>(c) * localDofs_;
          for (int i=0; i<localDofs_; i++) input[static_cast<size_t>(i)] = xHost(cellOffset + static_cast<size_t>(i), vectorOrdinal);
          applyDenseCellMatrix(fineMatrices_, c, localDofs_, input, output);
          for (int i=0; i<localDofs_; i++)
          {
            const size_t localOrdinal = cellOffset + static_cast<size_t>(i);
            yHost(localOrdinal, vectorOrdinal) = alpha * output[static_cast<size_t>(i)] + beta * yHost(localOrdinal, vectorOrdinal);
          }
        }
      }
    }

    void smoothFineLevel(const TpetraMultiVector &RHS, TpetraMultiVector &Solution) const
    {
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((RHS.getLocalLength() != localLength) || (Solution.getLocalLength() != localLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner smoother local vector length mismatch");
      if (RHS.getNumVectors() != Solution.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner smoother multivector count mismatch");

      auto rhsHost = RHS.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto solutionHost = Solution.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> rhs(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> solution(static_cast<size_t>(localDofs_), Scalar(0.0));
      for (size_t vectorOrdinal=0; vectorOrdinal<RHS.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t cellOffset = static_cast<size_t>(c) * localDofs_;
          for (int i=0; i<localDofs_; i++)
          {
            const size_t localOrdinal = cellOffset + static_cast<size_t>(i);
            rhs[static_cast<size_t>(i)] = rhsHost(localOrdinal, vectorOrdinal);
            solution[static_cast<size_t>(i)] = solutionHost(localOrdinal, vectorOrdinal);
          }
          smoothFine(c, rhs, solution);
          for (int i=0; i<localDofs_; i++) solutionHost(cellOffset + static_cast<size_t>(i), vectorOrdinal) = solution[static_cast<size_t>(i)];
        }
      }
    }

    void restrictFineResidual(const TpetraMultiVector &FineResidual, TpetraMultiVector &CoarseResidual) const
    {
      const size_t fineLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      const size_t coarseLength = static_cast<size_t>(cells_) * static_cast<size_t>(coarseRows_);
      if ((FineResidual.getLocalLength() != fineLength) || (CoarseResidual.getLocalLength() != coarseLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner restriction local vector length mismatch");
      if (FineResidual.getNumVectors() != CoarseResidual.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner restriction multivector count mismatch");

      auto fineHost = FineResidual.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto coarseHost = CoarseResidual.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> fine(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> coarse(static_cast<size_t>(coarseRows_), Scalar(0.0));
      for (size_t vectorOrdinal=0; vectorOrdinal<FineResidual.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t fineOffset = static_cast<size_t>(c) * localDofs_;
          const size_t coarseOffset = static_cast<size_t>(c) * coarseRows_;
          for (int i=0; i<localDofs_; i++) fine[static_cast<size_t>(i)] = fineHost(fineOffset + static_cast<size_t>(i), vectorOrdinal);
          applyTensorHGradPTransferRestriction(*transfer_, fine, coarse);
          for (int i=0; i<coarseRows_; i++) coarseHost(coarseOffset + static_cast<size_t>(i), vectorOrdinal) = coarse[static_cast<size_t>(i)];
        }
      }
    }

    void prolongateCoarseCorrection(const TpetraMultiVector &CoarseCorrection, TpetraMultiVector &FineCorrection) const
    {
      const size_t coarseLength = static_cast<size_t>(cells_) * static_cast<size_t>(coarseRows_);
      const size_t fineLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((CoarseCorrection.getLocalLength() != coarseLength) || (FineCorrection.getLocalLength() != fineLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner prolongation local vector length mismatch");
      if (CoarseCorrection.getNumVectors() != FineCorrection.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner prolongation multivector count mismatch");

      auto coarseHost = CoarseCorrection.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto fineHost = FineCorrection.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> coarse(static_cast<size_t>(coarseRows_), Scalar(0.0));
      std::vector<Scalar> fine(static_cast<size_t>(localDofs_), Scalar(0.0));
      for (size_t vectorOrdinal=0; vectorOrdinal<CoarseCorrection.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t coarseOffset = static_cast<size_t>(c) * coarseRows_;
          const size_t fineOffset = static_cast<size_t>(c) * localDofs_;
          for (int i=0; i<coarseRows_; i++) coarse[static_cast<size_t>(i)] = coarseHost(coarseOffset + static_cast<size_t>(i), vectorOrdinal);
          applyTensorHGradPTransferProlongation(*transfer_, coarse, fine);
          for (int i=0; i<localDofs_; i++) fineHost(fineOffset + static_cast<size_t>(i), vectorOrdinal) = fine[static_cast<size_t>(i)];
        }
      }
    }

    void solveCoarseLevel(const TpetraMultiVector &CoarseRHS, TpetraMultiVector &CoarseSolution) const
    {
      const size_t coarseLength = static_cast<size_t>(cells_) * static_cast<size_t>(coarseRows_);
      if ((CoarseRHS.getLocalLength() != coarseLength) || (CoarseSolution.getLocalLength() != coarseLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner coarse solve local vector length mismatch");
      if (CoarseRHS.getNumVectors() != CoarseSolution.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner coarse solve multivector count mismatch");

      auto rhsHost = CoarseRHS.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto solutionHost = CoarseSolution.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> coarse(static_cast<size_t>(coarseRows_), Scalar(0.0));
      for (size_t vectorOrdinal=0; vectorOrdinal<CoarseRHS.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t coarseOffset = static_cast<size_t>(c) * coarseRows_;
          for (int i=0; i<coarseRows_; i++) coarse[static_cast<size_t>(i)] = rhsHost(coarseOffset + static_cast<size_t>(i), vectorOrdinal);
          const int solveInfo = solveFactoredSPDMatrix(factoredCoarseMatrices_, static_cast<size_t>(c) * coarseRows_ * coarseRows_, coarseRows_, coarse);
          if (solveInfo != 0) throw std::runtime_error("LocalPMultigridTpetraPreconditioner coarse callback Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
          for (int i=0; i<coarseRows_; i++) solutionHost(coarseOffset + static_cast<size_t>(i), vectorOrdinal) = coarse[static_cast<size_t>(i)];
        }
      }
    }

    Teuchos::RCP<const TpetraMap> getDomainMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getRangeMap() const override { return map_; }
    bool hasTransposeApply() const override { return false; }

    void apply(const TpetraMultiVector &X, TpetraMultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      if (mode != Teuchos::NO_TRANS) throw std::invalid_argument("LocalPMultigridTpetraPreconditioner does not support transpose apply");
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((X.getLocalLength() != localLength) || (Y.getLocalLength() != localLength))
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner local vector length mismatch");
      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalPMultigridTpetraPreconditioner multivector count mismatch");

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
      std::vector<Scalar> rhs(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> solution(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> applied(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> fineResidual(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> fineCorrection(static_cast<size_t>(localDofs_), Scalar(0.0));
      std::vector<Scalar> coarseCorrection(static_cast<size_t>(coarseRows_), Scalar(0.0));

      for (size_t vectorOrdinal=0; vectorOrdinal<X.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
        {
          const size_t cellOffset = static_cast<size_t>(c) * localDofs_;
          for (int i=0; i<localDofs_; i++) rhs[static_cast<size_t>(i)] = xHost(cellOffset + static_cast<size_t>(i), vectorOrdinal);
          std::fill(solution.begin(), solution.end(), Scalar(0.0));

          smoothFine(c, rhs, solution);
          applyDenseCellMatrix(fineMatrices_, c, localDofs_, solution, applied);
          for (int i=0; i<localDofs_; i++) fineResidual[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)];
          applyTensorHGradPTransferRestriction(*transfer_, fineResidual, coarseCorrection);
          const int solveInfo = solveFactoredSPDMatrix(factoredCoarseMatrices_, static_cast<size_t>(c) * coarseRows_ * coarseRows_, coarseRows_, coarseCorrection);
          if (solveInfo != 0) throw std::runtime_error("LocalPMultigridTpetraPreconditioner coarse Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
          applyTensorHGradPTransferProlongation(*transfer_, coarseCorrection, fineCorrection);
          for (int i=0; i<localDofs_; i++) solution[static_cast<size_t>(i)] += fineCorrection[static_cast<size_t>(i)];
          smoothFine(c, rhs, solution);

          for (int i=0; i<localDofs_; i++)
          {
            const size_t localOrdinal = cellOffset + static_cast<size_t>(i);
            yHost(localOrdinal, vectorOrdinal) = alpha * solution[static_cast<size_t>(i)] + beta * yHost(localOrdinal, vectorOrdinal);
          }
        }
      }
    }

    std::string description() const override { return description_; }

    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const override
    {
      if ((verbLevel == Teuchos::VERB_MEDIUM) || (verbLevel == Teuchos::VERB_HIGH) || (verbLevel == Teuchos::VERB_EXTREME)) out << description() << std::endl;
    }
  };

  struct LocalPMultigridWrapperDiagnostic
  {
    double directInfNorm = -1.0;
    double wrapperInfNorm = -1.0;
    double differenceInfNorm = -1.0;
    double relativeDifferenceInfNorm = -1.0;
  };

  struct MatrixFreeGMGRightPreconditionerBuild
  {
    Teuchos::RCP<TpetraOperator> belosRightPreconditioner;
    Teuchos::RCP<TpetraOperator> directLocalPMultigrid;
    Teuchos::RCP<Camellia::MatrixFreeGMGPreconditioner> matrixFreeGMG;
  };

  Camellia::MatrixFreeGMGPreconditioner::Level
  buildLocalPMultigridMatrixFreeGMGFineLevel(const Teuchos::RCP<const TpetraMap> &map,
                                             const Teuchos::RCP<LocalPMultigridTpetraPreconditioner> &localPMultigrid)
  {
    Camellia::MatrixFreeGMGPreconditioner::Level level;
    level.map = map;
    level.applyOperator = [localPMultigrid](const TpetraMultiVector &X, TpetraMultiVector &Y)
    {
      localPMultigrid->applyFineOperator(X, Y);
    };
    level.preSmooth = [localPMultigrid](const TpetraMultiVector &RHS, TpetraMultiVector &solution)
    {
      localPMultigrid->smoothFineLevel(RHS, solution);
    };
    level.postSmooth = level.preSmooth;
    level.restrictResidual = [localPMultigrid](const TpetraMultiVector &fineResidual, TpetraMultiVector &coarseResidual)
    {
      localPMultigrid->restrictFineResidual(fineResidual, coarseResidual);
    };
    level.prolongateCorrection = [localPMultigrid](const TpetraMultiVector &coarseCorrection, TpetraMultiVector &fineCorrection)
    {
      localPMultigrid->prolongateCoarseCorrection(coarseCorrection, fineCorrection);
    };
    return level;
  }

  Camellia::MatrixFreeGMGPreconditioner::Level
  buildLocalPMultigridMatrixFreeGMGCoarseLevel(const Teuchos::RCP<LocalPMultigridTpetraPreconditioner> &localPMultigrid)
  {
    Camellia::MatrixFreeGMGPreconditioner::Level level;
    level.map = localPMultigrid->coarseMap();
    level.coarseSolve = [localPMultigrid](const TpetraMultiVector &coarseRHS, TpetraMultiVector &coarseSolution)
    {
      localPMultigrid->solveCoarseLevel(coarseRHS, coarseSolution);
    };
    return level;
  }

  MatrixFreeGMGRightPreconditionerBuild
  buildMatrixFreeGMGRightPreconditioner(const Teuchos::RCP<const TpetraMap> &map,
                                        const Teuchos::RCP<TpetraOperator> &fineOperator,
                                        const int dim,
                                        const int p,
                                        const int cells,
                                        const int localDofs,
                                        const int smootherIterations,
                                        const std::string &smootherName,
                                        const Scalar smootherLowerFraction,
                                        const Scalar smootherWeight)
  {
    auto localPMultigrid = Teuchos::rcp(new LocalPMultigridTpetraPreconditioner(map,
                                                                                fineOperator,
                                                                                dim,
                                                                                p,
                                                                                cells,
                                                                                localDofs,
                                                                                smootherIterations,
                                                                                smootherName,
                                                                                smootherLowerFraction,
                                                                                smootherWeight));

    std::vector<Camellia::MatrixFreeGMGPreconditioner::Level> levels;
    levels.push_back(buildLocalPMultigridMatrixFreeGMGFineLevel(map, localPMultigrid));
    levels.push_back(buildLocalPMultigridMatrixFreeGMGCoarseLevel(localPMultigrid));
    Teuchos::RCP<Camellia::MatrixFreeGMGPreconditioner> matrixFreeGMG =
      Teuchos::rcp(new Camellia::MatrixFreeGMGPreconditioner(levels,
                                                             "LocalPMultigridMatrixFreeGMGPreconditioner"));
    auto belosRightPreconditioner = Teuchos::rcp(new Camellia::MatrixFreeGMGApplyInversePreconditioner(matrixFreeGMG,
                                                                                                       "LocalPMultigridMatrixFreeGMGApplyInverseRightPreconditioner"));
    MatrixFreeGMGRightPreconditionerBuild build;
    build.belosRightPreconditioner = Teuchos::rcp_implicit_cast<TpetraOperator>(belosRightPreconditioner);
    build.directLocalPMultigrid = Teuchos::rcp_implicit_cast<TpetraOperator>(localPMultigrid);
    build.matrixFreeGMG = matrixFreeGMG;
    return build;
  }

  DenseVectorComparison compareDenseVectors(const std::vector<Scalar> &matrixFree,
                                            const std::vector<Scalar> &camellia,
                                            const int localDofs,
                                            const int globalDofs)
  {
    if (matrixFree.size() != camellia.size())
      throw std::invalid_argument("RHS comparison requires equal vector sizes");

    if (matrixFree.size() != static_cast<size_t>(globalDofs))
      throw std::invalid_argument("RHS comparison vector size does not match globalDofs");

    DenseVectorComparison comparison;
    comparison.localDofs = localDofs;
    comparison.globalDofs = globalDofs;

    double squaredError = 0.0;
    double matrixFreeSquaredNorm = 0.0;
    double camelliaSquaredNorm = 0.0;

    for (int i=0; i<globalDofs; i++)
    {
      const double matrixFreeValue = matrixFree[static_cast<size_t>(i)];
      const double camelliaValue = camellia[static_cast<size_t>(i)];
      const double error = matrixFreeValue - camelliaValue;
      const double absError = std::abs(error);
      const double scale =
          std::max<double>(1.0, std::max(std::abs(matrixFreeValue), std::abs(camelliaValue)));

      if (absError > comparison.maxAbsError)
      {
        comparison.maxAbsError = absError;
        comparison.maxErrorOrdinal = i;
        comparison.matrixFreeAtMaxError = matrixFreeValue;
        comparison.camelliaAtMaxError = camelliaValue;
      }

      comparison.maxRelError = std::max(comparison.maxRelError, absError / scale);
      squaredError += error * error;
      matrixFreeSquaredNorm += matrixFreeValue * matrixFreeValue;
      camelliaSquaredNorm += camelliaValue * camelliaValue;
    }

    comparison.l2Error = std::sqrt(squaredError);
    comparison.matrixFreeNorm = std::sqrt(matrixFreeSquaredNorm);
    comparison.camelliaNorm = std::sqrt(camelliaSquaredNorm);

    return comparison;
  }

  LocalPMultigridWrapperDiagnostic
  diagnoseLocalPMultigridWrapperEquivalence(const TpetraOperator &directLocalPMultigrid,
                                            const TpetraOperator &wrapperPreconditioner,
                                            const Teuchos::RCP<const TpetraMap> &map)
  {
    TpetraMultiVector rhs(map, 1);
    TpetraMultiVector direct(map, 1);
    TpetraMultiVector wrapped(map, 1);
    TpetraMultiVector difference(map, 1);

    auto rhsView = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (std::size_t i = 0; i < rhs.getLocalLength(); ++i)
      rhsView(i,0) = Scalar(1.0) + Scalar(0.125) * static_cast<Scalar>((i % 7) + 1);

    direct.putScalar(0.0);
    wrapped.putScalar(0.0);
    directLocalPMultigrid.apply(rhs, direct);
    wrapperPreconditioner.apply(rhs, wrapped);

    difference.assign(wrapped);
    difference.update(-1.0, direct, 1.0);

    Teuchos::Array<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> directNorm(1), wrapperNorm(1), diffNorm(1);
    direct.normInf(directNorm());
    wrapped.normInf(wrapperNorm());
    difference.normInf(diffNorm());

    LocalPMultigridWrapperDiagnostic diagnostic;
    diagnostic.directInfNorm = directNorm[0];
    diagnostic.wrapperInfNorm = wrapperNorm[0];
    diagnostic.differenceInfNorm = diffNorm[0];
    diagnostic.relativeDifferenceInfNorm = diffNorm[0] / std::max(1.0, directNorm[0]);
    return diagnostic;
  }

  std::vector<Scalar> extractCamelliaRhsVectorInGlobalOrder(
      Camellia::SolutionPtr solution,
      const std::vector<Camellia::GlobalIndexType> &globalDofIndices)
  {
    solution->initializeStiffnessAndLoad();
    solution->populateStiffnessAndLoad();

    auto rhsVector = solution->getRHSVector();
    if (rhsVector == Teuchos::null)
    {
      throw std::runtime_error("Camellia RHS comparison found null RHS vector");
    }

    auto rhsMap = rhsVector->getMap();
    auto rhsHost = rhsVector->getLocalViewHost(Tpetra::Access::ReadOnly);

    const int globalDofs = static_cast<int>(globalDofIndices.size());
    std::vector<Scalar> rhs(static_cast<size_t>(globalDofs), Scalar(0.0));

    using local_ordinal_type = typename TpetraMap::local_ordinal_type;
    const local_ordinal_type invalidLocalOrdinal =
        Teuchos::OrdinalTraits<local_ordinal_type>::invalid();

    for (int ordinal=0; ordinal<globalDofs; ordinal++)
    {
      const auto globalDof = globalDofIndices[static_cast<size_t>(ordinal)];
      const local_ordinal_type localOrdinal = rhsMap->getLocalElement(globalDof);

      if (localOrdinal == invalidLocalOrdinal)
      {
        std::ostringstream message;
        message << "Camellia RHS comparison could not find local ordinal for global dof "
                << globalDof;
        throw std::runtime_error(message.str());
      }

      rhs[static_cast<size_t>(ordinal)] = rhsHost(localOrdinal, 0);
    }

    return rhs;
  }

  std::vector<Scalar> interpretLocalVectorIntoCamelliaGlobalOrder(
      Camellia::MeshPtr mesh,
      const std::vector<Scalar> &localVector,
      const int expectedLocalDofs,
      std::vector<Camellia::GlobalIndexType> &globalDofIndicesVector)
  {
    const auto &cellIDs = mesh->cellIDsInPartition();
    if (cellIDs.size() != 1)
      throw std::runtime_error("Camellia RHS comparison expects exactly one locally-owned cell");

    const Camellia::GlobalIndexType cellID = *cellIDs.begin();
    const int localDofs = mesh->getElementType(cellID)->trialOrderPtr->totalDofs();

    if (localDofs != expectedLocalDofs)
      throw std::runtime_error("Camellia RHS comparison local DOF count does not match matrix-free local DOF count");

    if (localVector.size() != static_cast<size_t>(localDofs))
      throw std::runtime_error("Camellia RHS comparison local vector size mismatch");

    Camellia::FieldContainer<Scalar> localData(localDofs);
    for (int i=0; i<localDofs; i++)
    {
      localData(i) = localVector[static_cast<size_t>(i)];
    }

    Camellia::FieldContainer<Scalar> globalData;
    Camellia::FieldContainer<Camellia::GlobalIndexType> globalDofIndices;
    mesh->interpretLocalData(cellID, localData, globalData, globalDofIndices);

    const int globalDofs = globalDofIndices.dimension(0);
    globalDofIndicesVector.resize(static_cast<size_t>(globalDofs));

    for (int ordinal=0; ordinal<globalDofs; ordinal++)
    {
      globalDofIndicesVector[static_cast<size_t>(ordinal)] = globalDofIndices(ordinal);
    }

    std::vector<Scalar> globalVector(static_cast<size_t>(globalDofs), Scalar(0.0));
    for (int ordinal=0; ordinal<globalDofs; ordinal++)
    {
      globalVector[static_cast<size_t>(ordinal)] = globalData(ordinal);
    }

    return globalVector;
  }

  InverseDiagnosticTiming inverseDiagnosticTerms(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                                                 const int worksetSize, const int cells, const int rows, const int dim, const int p,
                                                 const long long maxAssembledEntries, const int inverseDiagnosticRequireAssembledReference,
                                                 const int diagnoseLocalPMultigridWrapperEquivalenceFlag,
                                                 const int numInverseIterations, const int pMultigridChebyshevSmootherIterations,
                                                 const Scalar pMultigridChebyshevSmootherLowerFraction,
                                                 const Scalar pMultigridFineSmootherWeight,
                                                 const int matrixFreeGMGStageTrace,
                                                 const int pMultigridJacobiSmootherIterations,
                                                 const int pMultigridCoarseChebyshevIterations,
                                                 const Scalar pMultigridCoarseChebyshevLowerFraction,
                                                 const std::string &inverseDiagnosticAlgorithm,
                                                 const Scalar inverseEigenRatio, const Scalar inverseLambdaMinOverride,
                                                 const Scalar inverseLambdaMaxOverride, const std::string &inverseSpectralEstimate,
                                                 const int inverseSpectralEstimateIterations, const Scalar inversePowerSafetyFactor,
                                                 const std::string &inverseLowerSpectralEstimate, const int inverseLowerSpectralEstimateIterations,
                                                 const Scalar inverseLowerPowerSafetyFactor,
                                                 const int inverseTrueEigenDiagnostic, const int inverseOperatorCheckSamples,
                                                 const std::string &inversePreconditionerMatrixSource,
                                                 const std::string &inverseOperatorSource,
                                                 const int compareTpetraPowerEstimate, const int compareTpetraRowMatrixPowerEstimate,
                                                 const int ifpack2ChebyshevSmoke, const std::string &inverseInitialGuess,
                                                 const std::string &inverseRhs, const int inversePreconditionerDegree,
                                                 const Scalar inverseTolerance)
  {
    const bool useBelosGmres = (inverseDiagnosticAlgorithm == "gmres-chebyshev") ||
                               (inverseDiagnosticAlgorithm == "gmres-jacobi") ||
                               (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ||
                               (inverseDiagnosticAlgorithm == "gmres-pmultigrid-jacobi-chebyshev");

    InverseDiagnosticTiming timing;
    timing.inverseIterations = numInverseIterations;
    timing.inversePreconditionerDegree = inversePreconditionerDegree;
    timing.inverseTolerance = inverseTolerance;
    timing.inverseStoppedReason = useBelosGmres ? "belos_gmres_iteration_limit" : "fixed_iteration_limit";
    if (inverseDiagnosticAlgorithm == "chebyshev")
      timing.inversePath = "local_chebyshev_ifpack2_compatible_pamatrix_apply_host_recurrence";
    else if (inverseDiagnosticAlgorithm == "gmres-chebyshev")
      timing.inversePath = "belos_gmres_right_preconditioned_by_ifpack2_chebyshev_rowmatrix_pamatrix_apply";
    else if (inverseDiagnosticAlgorithm == "gmres-jacobi")
      timing.inversePath = "belos_gmres_right_preconditioned_by_ifpack2_point_jacobi_rowmatrix_pamatrix_apply";
    else if (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev")
      timing.inversePath = "belos_gmres_right_preconditioned_by_tensor_hgrad_pmultigrid_operator_two_level_v_cycle_chebyshev_smoother";
    else if (inverseDiagnosticAlgorithm == "gmres-pmultigrid-jacobi-chebyshev")
      timing.inversePath = "belos_gmres_right_preconditioned_by_tensor_hgrad_pmultigrid_operator_two_level_v_cycle_jacobi_smoother";
    else if (inverseDiagnosticAlgorithm == "matrix-free-gmg-chebyshev-smoother")
      timing.inversePath = "matrix_free_gmg_style_smoother_only_chebyshev_pamatrix_apply";
    else if (inverseDiagnosticAlgorithm == "matrix-free-gmg-jacobi-smoother")
      timing.inversePath = "matrix_free_gmg_style_smoother_only_point_jacobi_pamatrix_apply";
    else
      timing.inversePath = "local_damped_jacobi_richardson_or_pmultigrid_pamatrix_apply";
    timing.spectralEstimatePath = "assembled_reference_gershgorin_preconditioned_upper_bound_with_extracted_diagonal";
    timing.spectralEstimateIterations = inverseSpectralEstimateIterations;
    timing.lambdaMaxSafetyFactor = inversePowerSafetyFactor;
    timing.lambdaMinSafetyFactor = inverseLowerPowerSafetyFactor;
    timing.inverseLowerSpectralEstimate = inverseLowerSpectralEstimate;
    timing.matrixFreeGMGStageTrace = matrixFreeGMGStageTrace;
    timing.inverseRhs = inverseRhs;
    timing.compareTpetraPowerEstimate = compareTpetraPowerEstimate;
    timing.compareTpetraRowMatrixPowerEstimate = compareTpetraRowMatrixPowerEstimate;
    timing.ifpack2ChebyshevSmoke = ifpack2ChebyshevSmoke;
    timing.lambdaMaxSource = "assembled_reference_gershgorin_preconditioned_upper_bound";
    timing.inverseInitialGuess = inverseInitialGuess;
    timing.trueEigenDiagnostic = inverseTrueEigenDiagnostic;
    timing.trueEigenPath = (inverseTrueEigenDiagnostic != 0) ? "assembled_lapack_syev_symmetric_diagonal_scaling" : "not_requested";
    timing.operatorCheckSamples = inverseOperatorCheckSamples;
    if (terms.empty()) return timing;

    const int cols = rows;
    if ((cells < 1) || (rows < 1)) throw std::invalid_argument("inverse diagnostic requires positive local extents");
    const long long entries = static_cast<long long>(cells) * rows * cols;
    const bool useAssembledReference = entries <= maxAssembledEntries;
    if (!useAssembledReference && (inverseDiagnosticRequireAssembledReference != 0))
    {
      std::cout << "INVERSE_DIAGNOSTIC_SKIPPED entries=" << entries
                << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
      timing.inversePath = "skipped_assembled_reference_guard";
      timing.spectralEstimatePath = "not_computed";
      return timing;
    }
    if ((inverseTrueEigenDiagnostic != 0) && !useAssembledReference)
      throw std::invalid_argument("--inverseTrueEigenDiagnostic requires assembled reference; increase --maxAssembledEntries");
    if (!useAssembledReference && (inverseSpectralEstimate != "power") && (inverseSpectralEstimate != "cell-power") && (inverseSpectralEstimate != "user"))
      throw std::invalid_argument("inverse diagnostic without assembled reference requires --inverseSpectralEstimate=power, cell-power, or user");
    if (!useAssembledReference) timing.inversePath += "_no_assembled_reference";

    std::vector<Scalar> localMatrices;
    if (useAssembledReference)
    {
      localMatrices.assign(static_cast<size_t>(entries), 0.0);
      for (auto &term : terms)
      {
        auto matrixData = term.matrix.allocateMatrixStorage();
        term.matrix.assemble(matrixData);
        Kokkos::fence();

        if (matrixData.getUnderlyingViewRank() == 3)
        {
          auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<3>());
          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
              for (int j=0; j<cols; j++)
                localMatrices[static_cast<size_t>((c * rows + i) * cols + j)] += term.weight * matrixHost(c,i,j);
        }
        else if (matrixData.getUnderlyingViewRank() == 2)
        {
          auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<2>());
          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
              for (int j=0; j<cols; j++)
                localMatrices[static_cast<size_t>((c * rows + i) * cols + j)] += term.weight * matrixHost(i,j);
        }
        else
        {
          throw std::invalid_argument("Unexpected PAMatrix storage rank in inverse diagnostic");
        }
      }
    }

    std::vector<Scalar> diagonal(static_cast<size_t>(cells * rows), 0.0);
    for (auto &term : terms)
    {
      auto diagonalData = term.matrix.allocateDiagonalStorage();
      term.matrix.extractDiagonal(diagonalData);
      Kokkos::fence();

      if (diagonalData.getUnderlyingViewRank() == 2)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
            diagonal[static_cast<size_t>(c * rows + i)] += term.weight * diagonalHost(c,i);
      }
      else if (diagonalData.getUnderlyingViewRank() == 1)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<1>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
            diagonal[static_cast<size_t>(c * rows + i)] += term.weight * diagonalHost(i);
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix diagonal storage rank in inverse diagnostic");
      }
    }

    timing.maxSymmetryAbsError = useAssembledReference ? 0.0 : -1.0;
    timing.minDiagonalEstimate = std::numeric_limits<Scalar>::max();
    timing.maxDiagonalEstimate = 0.0;
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        const Scalar diagonalValue = diagonal[static_cast<size_t>(c * rows + i)];
        timing.minDiagonalEstimate = std::min(timing.minDiagonalEstimate, diagonalValue);
        timing.maxDiagonalEstimate = std::max(timing.maxDiagonalEstimate, diagonalValue);
        if (useAssembledReference)
        {
          for (int j=i+1; j<cols; j++)
          {
            const Scalar aij = localMatrices[static_cast<size_t>((c * rows + i) * cols + j)];
            const Scalar aji = localMatrices[static_cast<size_t>((c * rows + j) * cols + i)];
            timing.maxSymmetryAbsError = std::max(timing.maxSymmetryAbsError, std::abs(aij - aji));
          }
        }
      }
    }
    if (timing.minDiagonalEstimate <= std::numeric_limits<Scalar>::epsilon())
      throw std::runtime_error("inverse diagnostic found nonpositive diagonal entry");

    Intrepid2::ScalarView<Scalar,DeviceType> input("TensorPoissonPAMatrix inverse diagnostic input", cells, rows);
    Intrepid2::ScalarView<Scalar,DeviceType> output("TensorPoissonPAMatrix inverse diagnostic output", cells, rows);

    auto applyVector = [&](const std::vector<Scalar> &inputVector, std::vector<Scalar> &outputVector)
    {
      auto inputHost = Kokkos::create_mirror_view(input);
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          inputHost(c,i) = inputVector[static_cast<size_t>(c * rows + i)];
      Kokkos::deep_copy(input, inputHost);
      TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
      auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          outputVector[static_cast<size_t>(c * rows + i)] = outputHost(c,i);
    };

    Scalar assembledGershgorinLambdaMax = -1.0;
    if (useAssembledReference)
    {
      assembledGershgorinLambdaMax = 0.0;
      for (int c=0; c<cells; c++)
      {
        for (int i=0; i<rows; i++)
        {
          const Scalar diagonalValue = diagonal[static_cast<size_t>(c * rows + i)];
          Scalar scaledRowSum = 0.0;
          for (int j=0; j<cols; j++) scaledRowSum += std::abs(localMatrices[static_cast<size_t>((c * rows + i) * cols + j)]) / diagonalValue;
          assembledGershgorinLambdaMax = std::max(assembledGershgorinLambdaMax, scaledRowSum);
        }
      }
    }
    timing.assembledGershgorinLambdaMax = assembledGershgorinLambdaMax;
    std::vector<Scalar> trueEigenvalues;
    if ((inverseTrueEigenDiagnostic != 0) || (inverseSpectralEstimate == "true-eigen") ||
        (inverseLowerSpectralEstimate == "true-eigen") || (inverseLowerSpectralEstimate == "true-eigen-safe"))
    {
      if (!useAssembledReference)
        throw std::invalid_argument("true-eigen diagnostics require assembled reference; increase --maxAssembledEntries");
      trueEigenvalues = computeTrueEigenDiagnostics(localMatrices, diagonal, cells, rows, timing);
    }

    if (inverseOperatorCheckSamples > 0)
    {
      if (!useAssembledReference)
        throw std::invalid_argument("operator sanity checks require assembled reference; increase --maxAssembledEntries");
      timing.operatorActionMaxAbsDiff = 0.0;
      timing.operatorActionMaxRelDiff = 0.0;
      timing.operatorBilinearSymmetryAbsError = 0.0;
      timing.operatorMinRayleighQuotient = std::numeric_limits<Scalar>::max();
      std::vector<Scalar> x(static_cast<size_t>(cells * rows), Scalar(0.0));
      std::vector<Scalar> y(static_cast<size_t>(cells * rows), Scalar(0.0));
      std::vector<Scalar> matrixFreeAx(x.size(), Scalar(0.0));
      std::vector<Scalar> matrixFreeAy(x.size(), Scalar(0.0));
      std::vector<Scalar> assembledAx(x.size(), Scalar(0.0));
      for (int sample=0; sample<inverseOperatorCheckSamples; sample++)
      {
        for (size_t i=0; i<x.size(); i++)
        {
          x[i] = std::sin(Scalar(sample + 1) * Scalar(i + 3) * Scalar(0.37))
               + Scalar(0.5) * std::cos(Scalar(sample + 2) * Scalar(i + 5) * Scalar(0.19));
          y[i] = std::cos(Scalar(sample + 3) * Scalar(i + 7) * Scalar(0.23))
               - Scalar(0.25) * std::sin(Scalar(sample + 4) * Scalar(i + 11) * Scalar(0.31));
        }
        applyVector(x, matrixFreeAx);
        applyVector(y, matrixFreeAy);
        std::fill(assembledAx.begin(), assembledAx.end(), Scalar(0.0));
        for (int c=0; c<cells; c++)
        {
          const size_t cellOffset = static_cast<size_t>(c * rows);
          for (int i=0; i<rows; i++)
            for (int j=0; j<rows; j++)
              assembledAx[cellOffset + i] += localMatrices[static_cast<size_t>((c * rows + i) * rows + j)] * x[cellOffset + j];
        }
        Scalar diffNormSquared = 0.0;
        Scalar assembledNormSquared = 0.0;
        Scalar xNormSquared = 0.0;
        Scalar xTAx = 0.0;
        Scalar xTAy = 0.0;
        Scalar yTAx = 0.0;
        for (size_t i=0; i<x.size(); i++)
        {
          const Scalar diff = matrixFreeAx[i] - assembledAx[i];
          timing.operatorActionMaxAbsDiff = std::max(timing.operatorActionMaxAbsDiff, std::abs(diff));
          diffNormSquared += diff * diff;
          assembledNormSquared += assembledAx[i] * assembledAx[i];
          xNormSquared += x[i] * x[i];
          xTAx += x[i] * matrixFreeAx[i];
          xTAy += x[i] * matrixFreeAy[i];
          yTAx += y[i] * matrixFreeAx[i];
        }
        const Scalar relDiff = std::sqrt(diffNormSquared) / std::max(std::numeric_limits<Scalar>::epsilon(), std::sqrt(assembledNormSquared));
        timing.operatorActionMaxRelDiff = std::max(timing.operatorActionMaxRelDiff, relDiff);
        timing.operatorBilinearSymmetryAbsError = std::max(timing.operatorBilinearSymmetryAbsError, std::abs(xTAy - yTAx));
        timing.operatorMinRayleighQuotient = std::min(timing.operatorMinRayleighQuotient, xTAx / std::max(std::numeric_limits<Scalar>::epsilon(), xNormSquared));
      }
    }

    Scalar lambdaMaxEstimate = 0.0;
    std::vector<Scalar> perCellLambdaMaxEstimates;
    std::vector<Scalar> perCellLambdaMinEstimates;
    Kokkos::Timer spectralEstimateTimer;
    if (inverseSpectralEstimate == "assembled-gershgorin")
    {
      if (!useAssembledReference) throw std::invalid_argument("assembled-gershgorin spectral estimate requires assembled reference");
      lambdaMaxEstimate = assembledGershgorinLambdaMax;
    }
    else if (inverseSpectralEstimate == "true-eigen")
    {
      if ((timing.trueEigenDiagnostic == 0) || !(timing.trueLambdaMax > Scalar(0.0)))
        throw std::invalid_argument("--inverseSpectralEstimate=true-eigen requires positive trueLambdaMax");
      lambdaMaxEstimate = inversePowerSafetyFactor * timing.trueLambdaMax;
      timing.spectralEstimatePath = "assembled_lapack_syev_true_eigen_lambda_max_with_safety_factor";
      timing.lambdaMaxSource = "true_eigen_oracle_lambda_max_with_safety_factor";
      timing.powerRawLambdaMaxEstimate = timing.trueLambdaMax;
    }
    else if (inverseSpectralEstimate == "user")
    {
      if (inverseLambdaMaxOverride <= 0.0) throw std::invalid_argument("--inverseSpectralEstimate=user requires positive --inverseLambdaMax");
      lambdaMaxEstimate = inverseLambdaMaxOverride;
      timing.spectralEstimatePath = "user_lambda_max_with_extracted_diagonal";
      timing.lambdaMaxSource = "user_lambda_max_override";
    }
    else if (inverseSpectralEstimate == "power")
    {
      timing.spectralEstimatePath = "matrix_free_power_ifpack2_left_diagonal_scaling_with_extracted_diagonal";
      timing.lambdaMaxSource = "matrix_free_power_ifpack2_left_diagonal_scaling_with_safety_factor";
      auto runPowerEstimate = [&](const std::string &tpetraAdapter, const int activeCell)
      {
        std::vector<Scalar> vector(static_cast<size_t>(cells * rows), Scalar(0.0));
        std::vector<Scalar> applied(vector.size(), Scalar(0.0));
        std::vector<Scalar> scaledOutput(vector.size(), Scalar(0.0));
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
          {
            const size_t index = static_cast<size_t>(c * rows + i);
            if ((activeCell < 0) || (activeCell == c)) vector[index] = Scalar(1.0) + Scalar((index * 13) % 17) / Scalar(17.0);
          }

        Teuchos::RCP<TpetraMap> map;
        Teuchos::RCP<TpetraOperator> tpetraOperator;
        Teuchos::RCP<TpetraRowMatrix> tpetraRowMatrix;
        Teuchos::RCP<TpetraMultiVector> X;
        Teuchos::RCP<TpetraMultiVector> Y;
        if (tpetraAdapter != "direct")
        {
          auto comm = Teuchos::DefaultComm<int>::getComm();
          map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), vector.size(), 0, comm));
          if (tpetraAdapter == "operator")
            tpetraOperator = TpetraAdapters::createLocalPAMatrixTpetraOperator(map, terms, workspaces, cells, rows, worksetSize);
          else if (tpetraAdapter == "rowmatrix")
            tpetraRowMatrix = TpetraAdapters::createLocalPAMatrixTpetraRowMatrix(map, terms, workspaces, cells, rows, worksetSize, maxAssembledEntries);
          else
            throw std::invalid_argument("unexpected Tpetra adapter in power spectral estimate");
          X = Teuchos::rcp(new TpetraMultiVector(map, 1));
          Y = Teuchos::rcp(new TpetraMultiVector(map, 1));
        }

        Scalar rawEstimate = 0.0;
        for (int iteration=0; iteration<inverseSpectralEstimateIterations; iteration++)
        {
          Scalar inputNorm = 0.0;
          for (const Scalar value : vector) inputNorm += value * value;
          inputNorm = std::sqrt(inputNorm);
          if (inputNorm <= std::numeric_limits<Scalar>::epsilon()) throw std::runtime_error("inverse diagnostic power estimate found zero vector");
          for (Scalar &value : vector) value /= inputNorm;

          if (tpetraAdapter != "direct")
          {
            auto xHost = X->getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t index=0; index<vector.size(); index++) xHost(index,0) = vector[index];
            if (tpetraAdapter == "operator") tpetraOperator->apply(*X, *Y);
            else if (tpetraAdapter == "rowmatrix") tpetraRowMatrix->apply(*X, *Y);
            auto yHost = Y->getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t index=0; index<applied.size(); index++) applied[index] = yHost(index,0);
          }
          else
          {
            applyVector(vector, applied);
          }

          Scalar outputNorm = 0.0;
          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
            {
              const size_t index = static_cast<size_t>(c * rows + i);
              scaledOutput[index] = applied[index] / diagonal[index];
              outputNorm += scaledOutput[index] * scaledOutput[index];
            }
          outputNorm = std::sqrt(outputNorm);
          rawEstimate = outputNorm;
          vector.swap(scaledOutput);
        }
        return rawEstimate;
      };

      lambdaMaxEstimate = runPowerEstimate("direct", -1);
      timing.powerRawLambdaMaxEstimate = lambdaMaxEstimate;
      if (compareTpetraPowerEstimate != 0)
      {
        timing.tpetraPowerRawLambdaMaxEstimate = runPowerEstimate("operator", -1);
        timing.tpetraPowerLambdaMaxAbsError = std::abs(timing.tpetraPowerRawLambdaMaxEstimate - timing.powerRawLambdaMaxEstimate);
        timing.tpetraPowerLambdaMaxRelError = timing.tpetraPowerLambdaMaxAbsError / std::max<Scalar>(Scalar(1.0), std::abs(timing.powerRawLambdaMaxEstimate));
      }
      if (compareTpetraRowMatrixPowerEstimate != 0)
      {
        timing.tpetraRowMatrixPowerRawLambdaMaxEstimate = runPowerEstimate("rowmatrix", -1);
        timing.tpetraRowMatrixPowerLambdaMaxAbsError = std::abs(timing.tpetraRowMatrixPowerRawLambdaMaxEstimate - timing.powerRawLambdaMaxEstimate);
        timing.tpetraRowMatrixPowerLambdaMaxRelError = timing.tpetraRowMatrixPowerLambdaMaxAbsError / std::max<Scalar>(Scalar(1.0), std::abs(timing.powerRawLambdaMaxEstimate));
      }
      // Power iteration is a diagnostic estimate, not a rigorous upper bound; add a
      // user-visible safety margin before using it as the Chebyshev upper endpoint.
      lambdaMaxEstimate *= inversePowerSafetyFactor;
    }
    else if (inverseSpectralEstimate == "cell-power")
    {
      timing.perCellSpectralEstimate = 1;
      timing.spectralEstimatePath = "per_cell_matrix_free_power_ifpack2_left_diagonal_scaling_with_extracted_diagonal";
      timing.lambdaMaxSource = "per_cell_matrix_free_power_ifpack2_left_diagonal_scaling_with_safety_factor";
      auto runCellPowerEstimate = [&](const int activeCell)
      {
        std::vector<Scalar> vector(static_cast<size_t>(cells * rows), Scalar(0.0));
        std::vector<Scalar> applied(vector.size(), Scalar(0.0));
        std::vector<Scalar> scaledOutput(vector.size(), Scalar(0.0));
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
          {
            const size_t index = static_cast<size_t>(c * rows + i);
            if (activeCell == c) vector[index] = Scalar(1.0) + Scalar((index * 13) % 17) / Scalar(17.0);
          }

        Scalar rawEstimate = 0.0;
        for (int iteration=0; iteration<inverseSpectralEstimateIterations; iteration++)
        {
          Scalar inputNorm = 0.0;
          for (const Scalar value : vector) inputNorm += value * value;
          inputNorm = std::sqrt(inputNorm);
          if (inputNorm <= std::numeric_limits<Scalar>::epsilon()) throw std::runtime_error("per-cell inverse diagnostic power estimate found zero vector");
          for (Scalar &value : vector) value /= inputNorm;

          applyVector(vector, applied);
          Scalar outputNorm = 0.0;
          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
            {
              const size_t index = static_cast<size_t>(c * rows + i);
              scaledOutput[index] = (c == activeCell) ? applied[index] / diagonal[index] : Scalar(0.0);
              outputNorm += scaledOutput[index] * scaledOutput[index];
            }
          outputNorm = std::sqrt(outputNorm);
          rawEstimate = outputNorm;
          vector.swap(scaledOutput);
        }
        return rawEstimate;
      };
      perCellLambdaMaxEstimates.assign(static_cast<size_t>(cells), Scalar(0.0));
      timing.perCellLambdaMaxEstimateMin = std::numeric_limits<Scalar>::max();
      timing.perCellLambdaMaxEstimateMax = 0.0;
      for (int c=0; c<cells; c++)
      {
        const Scalar rawCellEstimate = runCellPowerEstimate(c);
        const Scalar safeCellEstimate = rawCellEstimate * inversePowerSafetyFactor;
        if (!std::isfinite(safeCellEstimate) || (safeCellEstimate <= 0.0))
          throw std::runtime_error("per-cell inverse diagnostic found nonpositive or nonfinite spectral upper estimate");
        perCellLambdaMaxEstimates[static_cast<size_t>(c)] = safeCellEstimate;
        timing.perCellLambdaMaxEstimateMin = std::min(timing.perCellLambdaMaxEstimateMin, safeCellEstimate);
        timing.perCellLambdaMaxEstimateMax = std::max(timing.perCellLambdaMaxEstimateMax, safeCellEstimate);
      }
      timing.perCellLambdaMaxSpread = timing.perCellLambdaMaxEstimateMax / std::max<Scalar>(Scalar(1.0), timing.perCellLambdaMaxEstimateMin);
      lambdaMaxEstimate = timing.perCellLambdaMaxEstimateMax;
      timing.powerRawLambdaMaxEstimate = lambdaMaxEstimate;
    }
    else
    {
      throw std::invalid_argument("--inverseSpectralEstimate must be assembled-gershgorin, true-eigen, power, cell-power, or user");
    }
    timing.spectralEstimateSeconds = spectralEstimateTimer.seconds();
    if (!std::isfinite(lambdaMaxEstimate) || (lambdaMaxEstimate <= 0.0)) throw std::runtime_error("inverse diagnostic found nonpositive or nonfinite spectral upper estimate");
    if (inverseLambdaMaxOverride > 0.0)
    {
      lambdaMaxEstimate = inverseLambdaMaxOverride;
      timing.spectralEstimatePath = "user_lambda_max_override_with_extracted_diagonal";
      timing.lambdaMaxSource = "user_lambda_max_override";
    }
    timing.lambdaMaxEstimate = lambdaMaxEstimate;
    if (assembledGershgorinLambdaMax > 0.0)
      timing.lambdaMaxEstimateOverAssembledGershgorin = lambdaMaxEstimate / assembledGershgorinLambdaMax;
    timing.inverseEigenRatio = inverseEigenRatio;
    if (inverseLambdaMinOverride > 0.0)
    {
      timing.lambdaMinEstimate = inverseLambdaMinOverride;
      timing.lambdaMinRawEstimate = inverseLambdaMinOverride;
      timing.lambdaMinEstimateOverRaw = Scalar(1.0);
      timing.lambdaMinSource = "user_lambda_min_override";
    }
    else if (inverseLowerSpectralEstimate == "ratio")
    {
      if (!perCellLambdaMaxEstimates.empty())
      {
        perCellLambdaMinEstimates.assign(perCellLambdaMaxEstimates.size(), Scalar(0.0));
        timing.perCellLambdaMinEstimateMin = std::numeric_limits<Scalar>::max();
        timing.perCellLambdaMinEstimateMax = 0.0;
        for (size_t c=0; c<perCellLambdaMaxEstimates.size(); c++)
        {
          const Scalar cellLambdaMin = perCellLambdaMaxEstimates[c] / inverseEigenRatio;
          perCellLambdaMinEstimates[c] = cellLambdaMin;
          timing.perCellLambdaMinEstimateMin = std::min(timing.perCellLambdaMinEstimateMin, cellLambdaMin);
          timing.perCellLambdaMinEstimateMax = std::max(timing.perCellLambdaMinEstimateMax, cellLambdaMin);
        }
        timing.lambdaMinEstimate = timing.perCellLambdaMinEstimateMin;
        timing.lambdaMinRawEstimate = timing.lambdaMinEstimate;
        timing.lambdaMinEstimateOverRaw = Scalar(1.0);
        timing.lambdaMinSource = "per_cell_ratio_from_lambda_max";
      }
      else
      {
        timing.lambdaMinEstimate = lambdaMaxEstimate / inverseEigenRatio;
        timing.lambdaMinRawEstimate = timing.lambdaMinEstimate;
        timing.lambdaMinEstimateOverRaw = Scalar(1.0);
        timing.lambdaMinSource = "ratio_from_lambda_max";
      }
    }
    else if (inverseLowerSpectralEstimate == "true-eigen")
    {
      if ((timing.trueEigenDiagnostic == 0) || !(timing.trueLambdaMin > Scalar(0.0)))
        throw std::invalid_argument("--inverseLowerSpectralEstimate=true-eigen requires true eigen diagnostics and positive trueLambdaMin");
      timing.lambdaMinEstimate = timing.trueLambdaMin;
      timing.lambdaMinRawEstimate = timing.trueLambdaMin;
      timing.lambdaMinEstimateOverRaw = Scalar(1.0);
      timing.lambdaMinSource = "true_eigen_oracle_lambda_min";
      timing.spectralEstimatePath += "_true_eigen_oracle_lambda_min";
    }
    else if (inverseLowerSpectralEstimate == "true-eigen-safe")
    {
      if ((timing.trueEigenDiagnostic == 0) || !(timing.trueLambdaMin > Scalar(0.0)))
        throw std::invalid_argument("--inverseLowerSpectralEstimate=true-eigen-safe requires true eigen diagnostics and positive trueLambdaMin");
      timing.lambdaMinEstimate = inverseLowerPowerSafetyFactor * timing.trueLambdaMin;
      timing.lambdaMinRawEstimate = timing.trueLambdaMin;
      timing.lambdaMinEstimateOverRaw = timing.lambdaMinEstimate / timing.lambdaMinRawEstimate;
      timing.lambdaMinSource = "true_eigen_oracle_lambda_min_with_safety_factor";
      timing.spectralEstimatePath += "_true_eigen_oracle_lambda_min_with_safety_factor";
    }
    else if (inverseLowerSpectralEstimate == "user")
    {
      throw std::invalid_argument("--inverseLowerSpectralEstimate=user requires positive --inverseLambdaMin");
    }
    else if (inverseLowerSpectralEstimate == "assembled-inverse-iteration")
    {
      if (!useAssembledReference)
        throw std::invalid_argument("--inverseLowerSpectralEstimate=assembled-inverse-iteration requires assembled reference; increase --maxAssembledEntries or use --inverseLowerSpectralEstimate=ratio/user");

      std::vector<Scalar> factoredMatrices = localMatrices;
      for (int c=0; c<cells; c++)
      {
        const int factorInfo = factorSPDMatrixInPlace(factoredMatrices, static_cast<size_t>(c) * rows * cols, rows);
        if (factorInfo != 0) throw std::runtime_error("lower spectral inverse-iteration Cholesky factorization failed with LAPACK info=" + std::to_string(factorInfo));
      }

      std::vector<Scalar> vector(static_cast<size_t>(cells * rows), Scalar(0.0));
      std::vector<Scalar> solved(vector.size(), Scalar(0.0));
      for (size_t i=0; i<vector.size(); i++) vector[i] = Scalar(1.0) + Scalar((i * 7) % 19) / Scalar(19.0);
      Scalar rawLowerEstimate = 0.0;
      for (int iteration=0; iteration<inverseLowerSpectralEstimateIterations; iteration++)
      {
        Scalar inputNorm = 0.0;
        for (const Scalar value : vector) inputNorm += value * value;
        inputNorm = std::sqrt(inputNorm);
        if (inputNorm <= std::numeric_limits<Scalar>::epsilon()) throw std::runtime_error("lower spectral inverse iteration found zero vector");
        for (Scalar &value : vector) value /= inputNorm;

        for (int c=0; c<cells; c++)
        {
          const size_t cellOffset = static_cast<size_t>(c * rows);
          std::vector<Scalar> cellRhs(rows, Scalar(0.0));
          for (int i=0; i<rows; i++) cellRhs[static_cast<size_t>(i)] = std::sqrt(diagonal[cellOffset + i]) * vector[cellOffset + i];
          const int solveInfo = solveFactoredSPDMatrix(factoredMatrices, static_cast<size_t>(c) * rows * cols, rows, cellRhs);
          if (solveInfo != 0) throw std::runtime_error("lower spectral inverse-iteration Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
          for (int i=0; i<rows; i++) solved[cellOffset + i] = std::sqrt(diagonal[cellOffset + i]) * cellRhs[static_cast<size_t>(i)];
        }
        vector.swap(solved);
      }

      Scalar inverseIterationNorm = 0.0;
      for (const Scalar value : vector) inverseIterationNorm += value * value;
      inverseIterationNorm = std::sqrt(inverseIterationNorm);
      if (inverseIterationNorm <= std::numeric_limits<Scalar>::epsilon()) throw std::runtime_error("lower spectral inverse iteration produced zero vector");
      for (Scalar &value : vector) value /= inverseIterationNorm;

      Scalar rayleigh = 0.0;
      for (int c=0; c<cells; c++)
      {
        const size_t cellOffset = static_cast<size_t>(c * rows);
        for (int i=0; i<rows; i++)
        {
          Scalar matrixVectorValue = 0.0;
          for (int j=0; j<cols; j++)
          {
            const Scalar scaledInput = vector[cellOffset + j] / std::sqrt(diagonal[cellOffset + j]);
            matrixVectorValue += localMatrices[static_cast<size_t>((c * rows + i) * cols + j)] * scaledInput;
          }
          const Scalar scaledOutput = matrixVectorValue / std::sqrt(diagonal[cellOffset + i]);
          rayleigh += vector[cellOffset + i] * scaledOutput;
        }
      }
      rawLowerEstimate = rayleigh;
      if (!std::isfinite(rawLowerEstimate) || (rawLowerEstimate <= 0.0))
        throw std::runtime_error("lower spectral inverse iteration found nonpositive or nonfinite Rayleigh estimate");
      timing.lambdaMinRawEstimate = rawLowerEstimate;
      timing.lambdaMinEstimate = inverseLowerPowerSafetyFactor * rawLowerEstimate;
      timing.lambdaMinEstimateOverRaw = timing.lambdaMinEstimate / rawLowerEstimate;
      timing.lambdaMinSource = "assembled_reference_inverse_iteration_lower_bound";
      timing.spectralEstimatePath += "_assembled_inverse_iteration_lambda_min";
    }
    else
    {
      throw std::invalid_argument("--inverseLowerSpectralEstimate must be ratio, true-eigen, true-eigen-safe, user, or assembled-inverse-iteration");
    }
    if ((timing.trueEigenDiagnostic != 0) && (timing.trueLambdaMax > 0.0))
    {
      if (assembledGershgorinLambdaMax > 0.0)
        timing.trueLambdaMaxOverAssembledGershgorin = timing.trueLambdaMax / assembledGershgorinLambdaMax;
      if (timing.powerRawLambdaMaxEstimate > 0.0)
        timing.trueLambdaMaxOverPowerRaw = timing.trueLambdaMax / timing.powerRawLambdaMaxEstimate;
      if ((timing.trueLambdaMin > 0.0) && (timing.lambdaMinEstimate > 0.0))
        timing.trueLambdaMinOverLowerEstimate = timing.trueLambdaMin / timing.lambdaMinEstimate;
    }
    if ((timing.lambdaMinEstimate <= 0.0) || (timing.lambdaMinEstimate >= timing.lambdaMaxEstimate))
      throw std::runtime_error("inverse diagnostic Chebyshev/Richardson spectral bounds must satisfy 0 < lambdaMin < lambdaMax");
    if (inverseLambdaMinOverride > 0.0)
    {
      timing.spectralEstimatePath = (inverseLambdaMaxOverride > 0.0)
                                  ? "user_lambda_min_max_override_with_extracted_diagonal"
                                  : timing.spectralEstimatePath + "_user_lambda_min";
    }

    if (!trueEigenvalues.empty() && (inverseDiagnosticAlgorithm == "chebyshev"))
    {
      timing.chebyshevScalarNumEigenvalues = static_cast<int>(trueEigenvalues.size());
      timing.chebyshevScalarMaxAbsAmplification = 0.0;
      const Scalar chebyshevLambdaMax = timing.lambdaMaxEstimate;
      const Scalar chebyshevLambdaMin = timing.lambdaMinEstimate;
      const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
      const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
      const Scalar s1 = ifpackTheta * ifpackDelta;
      for (const Scalar lambda : trueEigenvalues)
      {
        if (lambda < chebyshevLambdaMin) timing.chebyshevScalarNumBelowLower++;
        else if (lambda > chebyshevLambdaMax) timing.chebyshevScalarNumAboveUpper++;
        else timing.chebyshevScalarNumInsideInterval++;
        Scalar error = Scalar(1.0);
        Scalar correction = Scalar(0.0);
        if (numInverseIterations > 0)
        {
          Scalar rhok = Scalar(1.0) / s1;
          correction = -(lambda / ifpackTheta) * error;
          error += correction;
          for (int iteration=1; iteration<numInverseIterations; iteration++)
          {
            const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhok);
            const Scalar dtemp1 = rhokp1 * rhok;
            const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
            rhok = rhokp1;
            correction = -dtemp2 * lambda * error + dtemp1 * correction;
            error += correction;
          }
        }
        const Scalar amplification = std::abs(error);
        if (amplification > Scalar(1.0) + Scalar(100.0) * std::numeric_limits<Scalar>::epsilon())
          timing.chebyshevScalarNumModesAmplified++;
        if (amplification > timing.chebyshevScalarMaxAbsAmplification)
        {
          timing.chebyshevScalarMaxAbsAmplification = amplification;
          timing.chebyshevScalarWorstLambda = lambda;
        }
      }
    }

    std::vector<Scalar> rhs(static_cast<size_t>(cells * rows), 0.0);
    if (inverseRhs == "deterministic")
    {
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          rhs[static_cast<size_t>(c * rows + i)] = 0.25 + 0.0625 * ((i + 5 * c) % 29);
    }
    else if (inverseRhs == "constant-test-function")
    {
      rhs = makeConstantTestFunctionRhs(terms, workspaces, dim, p, cells, rows, worksetSize);
    }
    else
    {
      throw std::invalid_argument("--inverseRhs must be deterministic or constant-test-function");
    }

    std::vector<Scalar> referenceSolution;
    Scalar referenceNorm = 0.0;
    if (useAssembledReference)
    {
      std::vector<Scalar> factoredMatrices = localMatrices;
      referenceSolution = rhs;
      for (int c=0; c<cells; c++)
      {
        const size_t offset = static_cast<size_t>(c) * rows * cols;
        const int factorInfo = factorSPDMatrixInPlace(factoredMatrices, offset, rows);
        if (factorInfo != 0) throw std::runtime_error("inverse diagnostic Cholesky factorization failed with LAPACK info=" + std::to_string(factorInfo));
        std::vector<Scalar> cellRhs(rhs.begin() + static_cast<size_t>(c * rows),
                                    rhs.begin() + static_cast<size_t>((c + 1) * rows));
        const int solveInfo = solveFactoredSPDMatrix(factoredMatrices, offset, rows, cellRhs);
        if (solveInfo != 0) throw std::runtime_error("inverse diagnostic Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
        for (int i=0; i<rows; i++) referenceSolution[static_cast<size_t>(c * rows + i)] = cellRhs[i];
      }

      for (const Scalar value : referenceSolution) referenceNorm = std::max(referenceNorm, std::abs(value));
    }

    std::vector<Scalar> iterate(static_cast<size_t>(cells * rows), 0.0);
    std::vector<Scalar> hostChebyshevIterate;
    std::vector<Scalar> applied(static_cast<size_t>(cells * rows), 0.0);
    if (inverseInitialGuess == "diagonal")
    {
      for (size_t i=0; i<iterate.size(); i++) iterate[i] = rhs[i] / diagonal[i];
    }
    else if (inverseInitialGuess != "zero")
    {
      throw std::invalid_argument("--inverseInitialGuess must be zero or diagonal");
    }

    auto applyHostVector = [&]()
    {
      applyVector(iterate, applied);
    };

    auto residualNorm = [&]()
    {
      Scalar maxResidual = 0.0;
      for (size_t i=0; i<rhs.size(); i++) maxResidual = std::max(maxResidual, std::abs(rhs[i] - applied[i]));
      return maxResidual;
    };

    applyHostVector();
    timing.initialResidualNorm = residualNorm();
    if (useAssembledReference)
    {
      timing.initialInverseAbsError = 0.0;
      for (size_t i=0; i<iterate.size(); i++)
        timing.initialInverseAbsError = std::max(timing.initialInverseAbsError, std::abs(iterate[i] - referenceSolution[i]));
      timing.initialInverseRelError = timing.initialInverseAbsError / std::max<Scalar>(1.0, referenceNorm);
    }

    auto runIfpack2CompatibleHostChebyshev = [&](std::vector<Scalar> &candidate, const int chebyshevDegree,
                                                 const Scalar chebyshevLowerFractionOfUpper = Scalar(-1.0))
    {
      const size_t localLength = rhs.size();
      if (candidate.size() != localLength)
        throw std::runtime_error("Chebyshev host recurrence candidate has incorrect length");
      if (chebyshevDegree <= 0) return;

      auto lambdaMaxForCell = [&](const int cellOrdinal)
      {
        return perCellLambdaMaxEstimates.empty() ? timing.lambdaMaxEstimate : perCellLambdaMaxEstimates[static_cast<size_t>(cellOrdinal)];
      };
      auto lambdaMinForCell = [&](const int cellOrdinal)
      {
        const Scalar chebyshevLambdaMax = lambdaMaxForCell(cellOrdinal);
        if (chebyshevLowerFractionOfUpper > Scalar(0.0))
          return std::max(std::numeric_limits<Scalar>::epsilon(), chebyshevLowerFractionOfUpper * chebyshevLambdaMax);
        return perCellLambdaMinEstimates.empty() ? timing.lambdaMinEstimate : perCellLambdaMinEstimates[static_cast<size_t>(cellOrdinal)];
      };
      std::vector<Scalar> correction(localLength, Scalar(0.0));
      std::vector<Scalar> candidateApplied(localLength, Scalar(0.0));
      std::vector<Scalar> rhokByCell(static_cast<size_t>(cells), Scalar(0.0));

      applyVector(candidate, candidateApplied);
      for (int c=0; c<cells; c++)
      {
        const Scalar chebyshevLambdaMax = lambdaMaxForCell(c);
        const Scalar chebyshevLambdaMin = lambdaMinForCell(c);
        const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
        const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
        const Scalar s1 = ifpackTheta * ifpackDelta;
        rhokByCell[static_cast<size_t>(c)] = Scalar(1.0) / s1;
        for (int i=0; i<rows; i++)
        {
          const size_t index = static_cast<size_t>(c * rows + i);
          correction[index] = (Scalar(1.0) / ifpackTheta) * (rhs[index] - candidateApplied[index]) / diagonal[index];
          candidate[index] += correction[index];
        }
      }

      for (int iteration=1; iteration<chebyshevDegree; iteration++)
      {
        applyVector(candidate, candidateApplied);
        for (int c=0; c<cells; c++)
        {
          const Scalar chebyshevLambdaMax = lambdaMaxForCell(c);
          const Scalar chebyshevLambdaMin = lambdaMinForCell(c);
          const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
          const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
          const Scalar s1 = ifpackTheta * ifpackDelta;
          const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhokByCell[static_cast<size_t>(c)]);
          const Scalar dtemp1 = rhokp1 * rhokByCell[static_cast<size_t>(c)];
          const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
          rhokByCell[static_cast<size_t>(c)] = rhokp1;
          for (int i=0; i<rows; i++)
          {
            const size_t index = static_cast<size_t>(c * rows + i);
            correction[index] = dtemp2 * (rhs[index] - candidateApplied[index]) / diagonal[index]
                              + dtemp1 * correction[index];
            candidate[index] += correction[index];
          }
        }
      }
    };

    Kokkos::Timer inverseTimer;
    if (inverseDiagnosticAlgorithm == "richardson")
    {
      const Scalar omega = Scalar(1.0) / timing.lambdaMaxEstimate;
      for (int iteration=0; iteration<numInverseIterations; iteration++)
      {
        applyHostVector();
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
          {
            const size_t index = static_cast<size_t>(c * rows + i);
            iterate[index] += omega * (rhs[index] - applied[index]) / diagonal[index];
          }
      }
    }
    else if (inverseDiagnosticAlgorithm == "chebyshev")
    {
      runIfpack2CompatibleHostChebyshev(iterate, numInverseIterations);
      hostChebyshevIterate = iterate;
    }
    else if ((inverseDiagnosticAlgorithm == "matrix-free-gmg-chebyshev-smoother") ||
             (inverseDiagnosticAlgorithm == "matrix-free-gmg-jacobi-smoother"))
    {
      if (inverseDiagnosticAlgorithm == "matrix-free-gmg-chebyshev-smoother")
      {
        if (inversePreconditionerDegree < 1)
          throw std::invalid_argument("--inversePreconditionerDegree must be positive for matrix-free GMG-style Chebyshev smoother diagnostics");
        if (inverseEigenRatio <= Scalar(1.0))
          throw std::invalid_argument("--inverseEigenRatio must be greater than one for matrix-free GMG-style Chebyshev smoother diagnostics");
      }

      for (int iteration=0; iteration<numInverseIterations; iteration++)
      {
        applyHostVector();
        std::vector<Scalar> residual(rhs.size(), Scalar(0.0));
        for (size_t i=0; i<rhs.size(); i++) residual[i] = rhs[i] - applied[i];

        std::vector<Scalar> correction(rhs.size(), Scalar(0.0));
        if (inverseDiagnosticAlgorithm == "matrix-free-gmg-chebyshev-smoother")
        {
          const std::vector<Scalar> savedRhs = rhs;
          rhs = residual;
          runIfpack2CompatibleHostChebyshev(correction, inversePreconditionerDegree, Scalar(1.0) / inverseEigenRatio);
          rhs = savedRhs;
        }
        else
        {
          for (size_t i=0; i<rhs.size(); i++) correction[i] = residual[i] / diagonal[i];
        }

        for (size_t i=0; i<iterate.size(); i++) iterate[i] += correction[i];
        timing.inverseIterations = iteration + 1;

        applyHostVector();
        timing.finalResidualNorm = residualNorm();
        timing.residualReduction = timing.finalResidualNorm / std::max<Scalar>(std::numeric_limits<Scalar>::epsilon(), timing.initialResidualNorm);
        if (timing.residualReduction <= inverseTolerance)
        {
          timing.inverseConverged = 1;
          timing.inverseStoppedReason = "matrix_free_gmg_style_smoother_only_converged";
          break;
        }
      }
    }
    else if (useBelosGmres)
    {
      if (inverseInitialGuess != "zero")
        throw std::invalid_argument("Belos GMRES inverse diagnostics currently support --inverseInitialGuess=zero only");
      if (inversePreconditionerDegree < 1)
        throw std::invalid_argument("--inversePreconditionerDegree must be positive for Belos GMRES Chebyshev diagnostics");

      const size_t localLength = rhs.size();
      auto comm = Teuchos::DefaultComm<int>::getComm();
      auto map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localLength, 0, comm));
      auto matrixFreeOperator = TpetraAdapters::createLocalPAMatrixTpetraOperator(map, terms, workspaces, cells, rows, worksetSize);
      
      Teuchos::RCP<TpetraCrsMatrix> assembledCrsMatrix;

      auto getAssembledCrsMatrix = [&]() -> Teuchos::RCP<TpetraCrsMatrix>
      {
        if (!useAssembledReference)
          throw std::invalid_argument("assembled CRS matrix requested, but assembled reference is unavailable");

        if (assembledCrsMatrix.is_null())
        {
          assembledCrsMatrix =
              TensorPAMatrixTpetraAdapters::buildLocalCrsMatrixFromDense(map,
                                                                         localMatrices,
                                                                         cells,
                                                                         rows);
        }

        return assembledCrsMatrix;
      };
      
      Teuchos::RCP<const TpetraRowMatrix> rowMatrixBase;

      const bool useAssembledCrsPreconditioner =
          (inversePreconditionerMatrixSource == "assembled-crs") ||
          ((inversePreconditionerMatrixSource == "auto") && useAssembledReference);

      const bool usePAMatrixRowMatrixPreconditioner =
          (inversePreconditionerMatrixSource == "matrix-free-rowmatrix") ||
          ((inversePreconditionerMatrixSource == "auto") && !useAssembledReference);

      if (useAssembledCrsPreconditioner)
      {
        if (!useAssembledReference)
          throw std::invalid_argument("--inversePreconditionerMatrixSource=assembled-crs requires assembled reference");

        rowMatrixBase =
            Teuchos::rcp_implicit_cast<const TpetraRowMatrix>(getAssembledCrsMatrix());
      }
      else if (usePAMatrixRowMatrixPreconditioner)
      {
        auto rowMatrix = TpetraAdapters::createLocalPAMatrixTpetraRowMatrix(map, terms, workspaces, cells, rows, worksetSize, maxAssembledEntries);
        rowMatrixBase = Teuchos::rcp_implicit_cast<const TpetraRowMatrix>(rowMatrix);
      }
      else
      {
        throw std::invalid_argument("--inversePreconditionerMatrixSource must be auto, assembled-crs, or matrix-free-rowmatrix");
      }

      Teuchos::RCP<TpetraOperator> preconditionerOperator;
      Teuchos::RCP<Camellia::MatrixFreeGMGPreconditioner> matrixFreeGMGForStageTrace;
      if ((inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ||
          (inverseDiagnosticAlgorithm == "gmres-pmultigrid-jacobi-chebyshev"))
      {
        if (p < 2)
          throw std::invalid_argument("GMRES p-multigrid inverse diagnostics require --polyOrder >= 2");
        const int smootherIterations = (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev")
                                     ? pMultigridChebyshevSmootherIterations
                                     : pMultigridJacobiSmootherIterations;
        const std::string smootherName = (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ? "chebyshev" : "jacobi";
        const TensorHGradPTransfer &transfer = cachedTensorHGradPTransfer(dim, p);
        const MatrixFreeGMGRightPreconditionerBuild preconditionerBuild =
          buildMatrixFreeGMGRightPreconditioner(map, matrixFreeOperator, dim, p, cells, rows, smootherIterations, smootherName, pMultigridChebyshevSmootherLowerFraction, pMultigridFineSmootherWeight);
        matrixFreeGMGForStageTrace = preconditionerBuild.matrixFreeGMG;
        if (matrixFreeGMGStageTrace != 0)
          matrixFreeGMGForStageTrace->setEmitStageTrace(true);
        preconditionerOperator = preconditionerBuild.belosRightPreconditioner;
        if (diagnoseLocalPMultigridWrapperEquivalenceFlag != 0)
        {
          const LocalPMultigridWrapperDiagnostic wrapperDiagnostic =
            diagnoseLocalPMultigridWrapperEquivalence(*preconditionerBuild.directLocalPMultigrid,
                                                      *preconditionerBuild.belosRightPreconditioner,
                                                      map);
          timing.localPMultigridWrapperApplyInfNorm = wrapperDiagnostic.directInfNorm;
          timing.matrixFreeGMGWrapperApplyInfNorm = wrapperDiagnostic.wrapperInfNorm;
          timing.localPMultigridWrapperDifferenceInfNorm = wrapperDiagnostic.differenceInfNorm;
          timing.localPMultigridWrapperRelativeDifferenceInfNorm = wrapperDiagnostic.relativeDifferenceInfNorm;
        }
        timing.pMultigridLevels = 2;
        timing.pMultigridFineDofs = rows;
        timing.pMultigridCoarseDofs = transfer.coarseRows;
        timing.pMultigridChebyshevSmootherIterations = (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ? smootherIterations : 0;
        timing.pMultigridChebyshevSmootherLowerFraction = (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ? pMultigridChebyshevSmootherLowerFraction : -1.0;
        timing.pMultigridFineSmootherWeight = pMultigridFineSmootherWeight;
        timing.pMultigridJacobiSmootherIterations = (inverseDiagnosticAlgorithm == "gmres-pmultigrid-jacobi-chebyshev") ? smootherIterations : 0;
        timing.pMultigridPath = "tpetra_operator_right_preconditioner_cached_tensor_hgrad_transfer_configured_chebyshev_interval";
      }
      else
      {
        Teuchos::RCP<Ifpack2::Preconditioner<Scalar,
                                              typename TpetraMap::local_ordinal_type,
                                              typename TpetraMap::global_ordinal_type,
                                              typename TpetraMap::node_type>> preconditioner;
        Teuchos::ParameterList preconditionerParams;
        if (inverseDiagnosticAlgorithm == "gmres-chebyshev")
        {
          preconditioner = Ifpack2::Factory::create<TpetraRowMatrix>("CHEBYSHEV", rowMatrixBase);
          preconditionerParams.set("chebyshev: ratio eigenvalue", inverseEigenRatio);
          preconditionerParams.set("chebyshev: eigenvalue max iterations", inverseSpectralEstimateIterations);
          preconditionerParams.set("chebyshev: degree", inversePreconditionerDegree);
        }
        else if (inverseDiagnosticAlgorithm == "gmres-jacobi")
        {
          preconditioner = Ifpack2::Factory::create<TpetraRowMatrix>("RELAXATION", rowMatrixBase);
          preconditionerParams.set("relaxation: type", "Jacobi");
          preconditionerParams.set("relaxation: sweeps", 1);
        }
        preconditioner->setParameters(preconditionerParams);
        preconditioner->initialize();
        preconditioner->compute();
        preconditionerOperator = Teuchos::rcp_implicit_cast<TpetraOperator>(preconditioner);
      }

      TpetraMultiVector B(map, 1);
      TpetraMultiVector X(map, 1);
      {
        auto bHost = B.getLocalViewHost(Tpetra::Access::OverwriteAll);
        auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i=0; i<localLength; i++)
        {
          bHost(i,0) = rhs[i];
          xHost(i,0) = Scalar(0.0);
        }
      }

      using MV = TpetraMultiVector;
      using OP = TpetraOperator;
      using BelosProblem = Belos::LinearProblem<Scalar,MV,OP>;
      Teuchos::RCP<TpetraOperator> operatorBase;

      if (inverseOperatorSource == "matrix-free")
      {
        operatorBase = Teuchos::rcp_implicit_cast<TpetraOperator>(matrixFreeOperator);
      }
      else if (inverseOperatorSource == "assembled-crs")
      {
        if (!useAssembledReference)
          throw std::invalid_argument("--inverseOperatorSource=assembled-crs requires assembled reference");

        operatorBase =
            Teuchos::rcp_implicit_cast<TpetraOperator>(getAssembledCrsMatrix());
      }
      else
      {
        throw std::invalid_argument("--inverseOperatorSource must be matrix-free or assembled-crs");
      }
      
      auto problem = Teuchos::rcp(new BelosProblem(operatorBase,
                                                   Teuchos::rcpFromRef(X),
                                                   Teuchos::rcpFromRef(B)));
      problem->setRightPrec(preconditionerOperator);
      problem->setProblem();

      Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
      solverParams->set("Maximum Iterations", numInverseIterations);
      solverParams->set("Convergence Tolerance", inverseTolerance);
      solverParams->set("Residual Scaling", "Norm of RHS");
      solverParams->set("Num Blocks", 200);
      Belos::SolverFactory<Scalar,MV,OP> solverFactory;
      auto solver = solverFactory.create("GMRES", solverParams);
      solver->setProblem(problem);
      const Belos::ReturnType belosResult = solver->solve();
      timing.inverseIterations = solver->getNumIters();
      timing.inverseConverged = (belosResult == Belos::Converged) ? 1 : 0;
      timing.inverseStoppedReason = (timing.inverseConverged != 0) ? "belos_gmres_converged" : "belos_gmres_iteration_limit";

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t i=0; i<localLength; i++) iterate[i] = xHost(i,0);
      if ((matrixFreeGMGStageTrace != 0) && (matrixFreeGMGForStageTrace != Teuchos::null))
      {
        TpetraMultiVector stageTraceOutput(map, 1);
        stageTraceOutput.putScalar(Scalar(0.0));
        matrixFreeGMGForStageTrace->apply(B, stageTraceOutput);

        const auto &trace = matrixFreeGMGForStageTrace->lastStageTrace();
        timing.matrixFreeGMGStageTraceEntries = static_cast<int>(trace.size());
        for (size_t traceIndex=0; traceIndex<trace.size(); traceIndex++)
        {
          const auto &entry = trace[traceIndex];
          std::cout << "MATRIX_FREE_GMG_STAGE_TRACE"
                    << " index=" << traceIndex
                    << " level=" << entry.levelIndex
                    << " stage=" << entry.stageName
                    << " rhsInfNorm=" << std::setprecision(15) << entry.rhsInfNorm
                    << " solutionInfNorm=" << entry.solutionInfNorm
                    << " residualInfNorm=" << entry.residualInfNorm
                    << std::endl;
        }
      }
    }
    else if ((inverseDiagnosticAlgorithm == "pmultigrid-chebyshev") || (inverseDiagnosticAlgorithm == "pmultigrid-jacobi-chebyshev"))
    {
      if (!useAssembledReference)
        throw std::invalid_argument("p-multigrid inverse diagnostics currently require assembled reference; increase --maxAssembledEntries");
      if (p < 2)
        throw std::invalid_argument("p-multigrid inverse diagnostics require --polyOrder >= 2");
      if (inverseDiagnosticAlgorithm == "pmultigrid-chebyshev")
      {
        if (pMultigridChebyshevSmootherIterations < 1)
          throw std::invalid_argument("--pMultigridChebyshevSmootherIterations must be positive");
        if ((pMultigridChebyshevSmootherLowerFraction <= Scalar(0.0)) || (pMultigridChebyshevSmootherLowerFraction >= Scalar(1.0)))
          throw std::invalid_argument("--pMultigridChebyshevSmootherLowerFraction must be in (0,1) for high-frequency Chebyshev smoothing");
      }
      const TensorHGradPTransfer &transfer = cachedTensorHGradPTransfer(dim, p);
      if (transfer.fineRows != rows) throw std::runtime_error("p-multigrid transfer fine dimension does not match local dofs");
      const int coarseRows = transfer.coarseRows;

      std::vector<Scalar> coarseMatrices(static_cast<size_t>(cells * coarseRows * coarseRows), Scalar(0.0));
      std::vector<Scalar> matrixTimesP(static_cast<size_t>(rows * coarseRows), Scalar(0.0));
      for (int c=0; c<cells; c++)
      {
        std::fill(matrixTimesP.begin(), matrixTimesP.end(), Scalar(0.0));
        for (int fineI=0; fineI<rows; fineI++)
          for (int fineJ=0; fineJ<rows; fineJ++)
          {
            const Scalar matrixEntry = localMatrices[static_cast<size_t>((c * rows + fineI) * rows + fineJ)];
            if (matrixEntry == Scalar(0.0)) continue;
            for (int coarseJ=0; coarseJ<coarseRows; coarseJ++)
              matrixTimesP[static_cast<size_t>(fineI * coarseRows + coarseJ)] += matrixEntry * transfer.prolongation[static_cast<size_t>(fineJ * coarseRows + coarseJ)];
          }
        for (int coarseI=0; coarseI<coarseRows; coarseI++)
          for (int fineI=0; fineI<rows; fineI++)
          {
            const Scalar restrictionEntry = transfer.restriction[static_cast<size_t>(coarseI * rows + fineI)];
            if (restrictionEntry == Scalar(0.0)) continue;
            for (int coarseJ=0; coarseJ<coarseRows; coarseJ++)
              coarseMatrices[static_cast<size_t>((c * coarseRows + coarseI) * coarseRows + coarseJ)] += restrictionEntry * matrixTimesP[static_cast<size_t>(fineI * coarseRows + coarseJ)];
          }
      }

      std::vector<Scalar> factoredCoarseMatrices;
      std::vector<Scalar> coarseDiagonal(static_cast<size_t>(cells * coarseRows), Scalar(0.0));
      std::vector<Scalar> coarseLambdaMaxByCell(static_cast<size_t>(cells), Scalar(0.0));
      Scalar fineMatrixMaxAbs = Scalar(0.0);
      Scalar fineMatrixFrobeniusNorm = Scalar(0.0);
      Scalar fineMatrixDiagonalMinAbs = Scalar(-1.0);
      Scalar fineMatrixDiagonalMaxAbs = Scalar(0.0);
      Scalar coarseMatrixMaxAbs = Scalar(0.0);
      Scalar coarseMatrixFrobeniusNorm = Scalar(0.0);
      Scalar coarseMatrixDiagonalMinAbs = Scalar(-1.0);
      Scalar coarseMatrixDiagonalMaxAbs = Scalar(0.0);
      if (inverseDiagnosticAlgorithm == "pmultigrid-chebyshev")
      {
        coarseMatrixMaxAbs = Scalar(0.0);
        Scalar coarseMatrixFrobeniusNormSquared = Scalar(0.0);
        coarseMatrixDiagonalMinAbs = Scalar(-1.0);
        coarseMatrixDiagonalMaxAbs = Scalar(0.0);
        for (int coarseScaleCell=0; coarseScaleCell<cells; coarseScaleCell++)
        {
          const size_t coarseOffset = static_cast<size_t>(coarseScaleCell) * coarseRows * coarseRows;
          for (int i=0; i<coarseRows; i++)
          {
            for (int j=0; j<coarseRows; j++)
            {
              const Scalar entryAbs = std::abs(coarseMatrices[coarseOffset + static_cast<size_t>(i * coarseRows + j)]);
              coarseMatrixMaxAbs = std::max(coarseMatrixMaxAbs, entryAbs);
              coarseMatrixFrobeniusNormSquared += entryAbs * entryAbs;
              if (i == j)
              {
                coarseMatrixDiagonalMaxAbs = std::max(coarseMatrixDiagonalMaxAbs, entryAbs);
                if ((coarseMatrixDiagonalMinAbs < Scalar(0.0)) || (entryAbs < coarseMatrixDiagonalMinAbs)) coarseMatrixDiagonalMinAbs = entryAbs;
              }
            }
          }
        }
        coarseMatrixFrobeniusNorm = std::sqrt(coarseMatrixFrobeniusNormSquared);

        Scalar fineMatrixFrobeniusNormSquared = Scalar(0.0);
        for (size_t entry=0; entry<localMatrices.size(); entry++)
        {
          const Scalar entryAbs = std::abs(localMatrices[entry]);
          fineMatrixMaxAbs = std::max(fineMatrixMaxAbs, entryAbs);
          fineMatrixFrobeniusNormSquared += localMatrices[entry] * localMatrices[entry];
        }
        for (size_t entry=0; entry<diagonal.size(); entry++)
        {
          const Scalar diagonalAbs = std::abs(diagonal[entry]);
          fineMatrixDiagonalMaxAbs = std::max(fineMatrixDiagonalMaxAbs, diagonalAbs);
          if ((fineMatrixDiagonalMinAbs < Scalar(0.0)) || (diagonalAbs < fineMatrixDiagonalMinAbs)) fineMatrixDiagonalMinAbs = diagonalAbs;
        }
        fineMatrixFrobeniusNorm = std::sqrt(fineMatrixFrobeniusNormSquared);

        factoredCoarseMatrices = coarseMatrices;
        for (int c=0; c<cells; c++)
        {
          const int factorInfo = factorSPDMatrixInPlace(factoredCoarseMatrices, static_cast<size_t>(c) * coarseRows * coarseRows, coarseRows);
          if (factorInfo != 0) throw std::runtime_error("p-multigrid diagnostic coarse Cholesky factorization failed with LAPACK info=" + std::to_string(factorInfo));
        }
      }
      else
      {
        for (int c=0; c<cells; c++)
        {
          for (int i=0; i<coarseRows; i++)
          {
            const Scalar diagonalValue = coarseMatrices[static_cast<size_t>((c * coarseRows + i) * coarseRows + i)];
            if (diagonalValue <= std::numeric_limits<Scalar>::epsilon())
              throw std::runtime_error("p-multigrid Chebyshev coarse solve found nonpositive coarse diagonal entry");
            coarseDiagonal[static_cast<size_t>(c * coarseRows + i)] = diagonalValue;
            Scalar scaledRowSum = 0.0;
            for (int j=0; j<coarseRows; j++)
              scaledRowSum += std::abs(coarseMatrices[static_cast<size_t>((c * coarseRows + i) * coarseRows + j)]) / diagonalValue;
            coarseLambdaMaxByCell[static_cast<size_t>(c)] = std::max(coarseLambdaMaxByCell[static_cast<size_t>(c)], scaledRowSum);
          }
        }
      }

      auto runFineJacobiSmoother = [&]()
      {
        for (int sweep=0; sweep<pMultigridJacobiSmootherIterations; sweep++)
        {
          applyHostVector();
          for (int c=0; c<cells; c++)
          {
            const Scalar lambdaMaxForCell = perCellLambdaMaxEstimates.empty() ? timing.lambdaMaxEstimate : perCellLambdaMaxEstimates[static_cast<size_t>(c)];
            const Scalar omega = Scalar(1.0) / lambdaMaxForCell;
            for (int i=0; i<rows; i++)
            {
              const size_t index = static_cast<size_t>(c * rows + i);
              iterate[index] += omega * (rhs[index] - applied[index]) / diagonal[index];
            }
          }
        }
      };

      auto applyCoarseMatrix = [&](const int cellOrdinal, const std::vector<Scalar> &inputVector, std::vector<Scalar> &outputVector)
      {
        outputVector.assign(static_cast<size_t>(coarseRows), Scalar(0.0));
        const size_t offset = static_cast<size_t>(cellOrdinal) * coarseRows * coarseRows;
        for (int i=0; i<coarseRows; i++)
          for (int j=0; j<coarseRows; j++)
            outputVector[static_cast<size_t>(i)] += coarseMatrices[offset + static_cast<size_t>(i * coarseRows + j)] * inputVector[static_cast<size_t>(j)];
      };

      auto runCoarseChebyshev = [&](const int cellOrdinal, const std::vector<Scalar> &coarseRhs, std::vector<Scalar> &coarseSolution)
      {
        coarseSolution.assign(static_cast<size_t>(coarseRows), Scalar(0.0));
        const Scalar chebyshevLambdaMax = coarseLambdaMaxByCell[static_cast<size_t>(cellOrdinal)];
        const Scalar chebyshevLambdaMin = std::max(std::numeric_limits<Scalar>::epsilon(), pMultigridCoarseChebyshevLowerFraction * chebyshevLambdaMax);
        const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
        const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
        const Scalar s1 = ifpackTheta * ifpackDelta;
        Scalar rhok = Scalar(1.0) / s1;
        std::vector<Scalar> correction(static_cast<size_t>(coarseRows), Scalar(0.0));
        std::vector<Scalar> coarseApplied(static_cast<size_t>(coarseRows), Scalar(0.0));
        applyCoarseMatrix(cellOrdinal, coarseSolution, coarseApplied);
        for (int i=0; i<coarseRows; i++)
        {
          const size_t index = static_cast<size_t>(cellOrdinal * coarseRows + i);
          correction[static_cast<size_t>(i)] = (Scalar(1.0) / ifpackTheta) * (coarseRhs[static_cast<size_t>(i)] - coarseApplied[static_cast<size_t>(i)]) / coarseDiagonal[index];
          coarseSolution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
        }
        for (int iteration=1; iteration<pMultigridCoarseChebyshevIterations; iteration++)
        {
          applyCoarseMatrix(cellOrdinal, coarseSolution, coarseApplied);
          const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhok);
          const Scalar dtemp1 = rhokp1 * rhok;
          const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
          rhok = rhokp1;
          for (int i=0; i<coarseRows; i++)
          {
            const size_t index = static_cast<size_t>(cellOrdinal * coarseRows + i);
            correction[static_cast<size_t>(i)] = dtemp2 * (coarseRhs[static_cast<size_t>(i)] - coarseApplied[static_cast<size_t>(i)]) / coarseDiagonal[index]
                                           + dtemp1 * correction[static_cast<size_t>(i)];
            coarseSolution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
          }
        }
      };

      std::vector<Scalar> fineResidual(static_cast<size_t>(rows), Scalar(0.0));
      std::vector<Scalar> fineCorrection(static_cast<size_t>(rows), Scalar(0.0));
      std::vector<Scalar> coarseCorrection(static_cast<size_t>(coarseRows), Scalar(0.0));
      auto maxAbsNorm = [](const std::vector<Scalar> &values)
      {
        Scalar norm = Scalar(0.0);
        for (const Scalar value : values) norm = std::max(norm, std::abs(value));
        return norm;
      };
      if (inverseDiagnosticAlgorithm == "pmultigrid-chebyshev")
      {
        for (int cycle=0; cycle<numInverseIterations; cycle++)
        {
          applyHostVector();
          const Scalar cycleInitialResidualNorm = residualNorm();
          {
            const std::vector<Scalar> beforeSmooth = iterate;
            runIfpack2CompatibleHostChebyshev(iterate, pMultigridChebyshevSmootherIterations, pMultigridChebyshevSmootherLowerFraction);
            if (pMultigridFineSmootherWeight != Scalar(1.0))
              for (size_t i=0; i<iterate.size(); i++) iterate[i] = beforeSmooth[i] + pMultigridFineSmootherWeight * (iterate[i] - beforeSmooth[i]);
          }
          applyHostVector();
          const Scalar residualAfterPreSmooth = residualNorm();
          Scalar restrictedResidualNorm = Scalar(0.0);
          Scalar coarseCorrectionNorm = Scalar(0.0);
          Scalar prolongatedCorrectionNorm = Scalar(0.0);
          Scalar coarseSolveResidualNorm = Scalar(0.0);
          Scalar galerkinActionDefectNorm = Scalar(0.0);
          Scalar prolongationScaleRatio = Scalar(0.0);
          Scalar fineActionScaleRatio = Scalar(0.0);
          Scalar coarseActionScaleRatio = Scalar(0.0);
          Scalar restrictedResidualL2NormSquared = Scalar(0.0);
          Scalar coarseResidualDotCorrection = Scalar(0.0);
          Scalar coarseCorrectionEnergy = Scalar(0.0);
          Scalar fineResidualDotCorrection = Scalar(0.0);
          Scalar fineCorrectionEnergy = Scalar(0.0);
          std::vector<Scalar> coarseAction(static_cast<size_t>(coarseRows), Scalar(0.0));
          std::vector<Scalar> fineAction(static_cast<size_t>(rows), Scalar(0.0));
          std::vector<Scalar> restrictedFineAction(static_cast<size_t>(coarseRows), Scalar(0.0));
          const std::vector<Scalar> iterateBeforeCoarseCorrection = iterate;
          std::vector<Scalar> prolongatedCoarseCorrection(static_cast<size_t>(cells * rows), Scalar(0.0));
          for (int c=0; c<cells; c++)
          {
            const size_t cellOffset = static_cast<size_t>(c * rows);
            for (int i=0; i<rows; i++) fineResidual[static_cast<size_t>(i)] = rhs[cellOffset + i] - applied[cellOffset + i];
            applyTensorHGradPTransferRestriction(transfer, fineResidual, coarseCorrection);
            const std::vector<Scalar> restrictedResidual = coarseCorrection;
            const Scalar restrictedResidualCellNorm = maxAbsNorm(restrictedResidual);
            restrictedResidualNorm = std::max(restrictedResidualNorm, restrictedResidualCellNorm);
            for (int i=0; i<coarseRows; i++) restrictedResidualL2NormSquared += restrictedResidual[static_cast<size_t>(i)] * restrictedResidual[static_cast<size_t>(i)];
            const int solveInfo = solveFactoredSPDMatrix(factoredCoarseMatrices, static_cast<size_t>(c) * coarseRows * coarseRows, coarseRows, coarseCorrection);
            for (int i=0; i<coarseRows; i++) coarseResidualDotCorrection += restrictedResidual[static_cast<size_t>(i)] * coarseCorrection[static_cast<size_t>(i)];
            if (solveInfo != 0) throw std::runtime_error("p-multigrid diagnostic coarse Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
            const Scalar coarseCorrectionCellNorm = maxAbsNorm(coarseCorrection);
            coarseCorrectionNorm = std::max(coarseCorrectionNorm, coarseCorrectionCellNorm);

            coarseAction.assign(static_cast<size_t>(coarseRows), Scalar(0.0));
            const size_t coarseOffset = static_cast<size_t>(c) * coarseRows * coarseRows;
            for (int i=0; i<coarseRows; i++)
              for (int j=0; j<coarseRows; j++)
                coarseAction[static_cast<size_t>(i)] += coarseMatrices[coarseOffset + static_cast<size_t>(i * coarseRows + j)] * coarseCorrection[static_cast<size_t>(j)];
            Scalar coarseSolveResidualCellNorm = Scalar(0.0);
            for (int i=0; i<coarseRows; i++)
              coarseSolveResidualCellNorm = std::max(coarseSolveResidualCellNorm, std::abs(coarseAction[static_cast<size_t>(i)] - restrictedResidual[static_cast<size_t>(i)]));
            coarseSolveResidualNorm = std::max(coarseSolveResidualNorm, coarseSolveResidualCellNorm);
            const Scalar coarseActionNorm = maxAbsNorm(coarseAction);
            if (restrictedResidualCellNorm > Scalar(0.0)) coarseActionScaleRatio = std::max(coarseActionScaleRatio, coarseActionNorm / restrictedResidualCellNorm);
            for (int i=0; i<coarseRows; i++) coarseCorrectionEnergy += coarseCorrection[static_cast<size_t>(i)] * coarseAction[static_cast<size_t>(i)];

            applyTensorHGradPTransferProlongation(transfer, coarseCorrection, fineCorrection);
            const Scalar fineCorrectionNorm = maxAbsNorm(fineCorrection);
            prolongatedCorrectionNorm = std::max(prolongatedCorrectionNorm, fineCorrectionNorm);
            if (coarseCorrectionCellNorm > Scalar(0.0)) prolongationScaleRatio = std::max(prolongationScaleRatio, fineCorrectionNorm / coarseCorrectionCellNorm);

            fineAction.assign(static_cast<size_t>(rows), Scalar(0.0));
            const size_t fineOffset = static_cast<size_t>(c) * rows * rows;
            for (int i=0; i<rows; i++)
              for (int j=0; j<rows; j++)
                fineAction[static_cast<size_t>(i)] += localMatrices[fineOffset + static_cast<size_t>(i * rows + j)] * fineCorrection[static_cast<size_t>(j)];
            const Scalar fineActionNorm = maxAbsNorm(fineAction);
            if (fineCorrectionNorm > Scalar(0.0)) fineActionScaleRatio = std::max(fineActionScaleRatio, fineActionNorm / fineCorrectionNorm);
            for (int i=0; i<rows; i++)
            {
              fineResidualDotCorrection += fineResidual[static_cast<size_t>(i)] * fineCorrection[static_cast<size_t>(i)];
              fineCorrectionEnergy += fineCorrection[static_cast<size_t>(i)] * fineAction[static_cast<size_t>(i)];
            }
            applyTensorHGradPTransferRestriction(transfer, fineAction, restrictedFineAction);
            Scalar galerkinActionDefectCellNorm = Scalar(0.0);
            for (int i=0; i<coarseRows; i++)
              galerkinActionDefectCellNorm = std::max(galerkinActionDefectCellNorm, std::abs(restrictedFineAction[static_cast<size_t>(i)] - coarseAction[static_cast<size_t>(i)]));
            galerkinActionDefectNorm = std::max(galerkinActionDefectNorm, galerkinActionDefectCellNorm);

            for (int i=0; i<rows; i++) prolongatedCoarseCorrection[static_cast<size_t>(cellOffset + i)] += fineCorrection[static_cast<size_t>(i)];
          }
          for (size_t i=0; i<iterate.size(); i++) iterate[i] = iterateBeforeCoarseCorrection[i] + prolongatedCoarseCorrection[i];
          applyHostVector();
          const Scalar residualAfterPlusCoarseCorrection = residualNorm();
          for (size_t i=0; i<iterate.size(); i++) iterate[i] = iterateBeforeCoarseCorrection[i] - prolongatedCoarseCorrection[i];
          applyHostVector();
          const Scalar residualAfterMinusCoarseCorrection = residualNorm();
          for (size_t i=0; i<iterate.size(); i++) iterate[i] = iterateBeforeCoarseCorrection[i] + prolongatedCoarseCorrection[i];
          applyHostVector();
          const Scalar residualAfterCoarseCorrection = residualAfterPlusCoarseCorrection;
          {
            const std::vector<Scalar> beforeSmooth = iterate;
            runIfpack2CompatibleHostChebyshev(iterate, pMultigridChebyshevSmootherIterations, pMultigridChebyshevSmootherLowerFraction);
            if (pMultigridFineSmootherWeight != Scalar(1.0))
              for (size_t i=0; i<iterate.size(); i++) iterate[i] = beforeSmooth[i] + pMultigridFineSmootherWeight * (iterate[i] - beforeSmooth[i]);
          }
          applyHostVector();
          const Scalar residualAfterPostSmooth = residualNorm();
          const Scalar restrictedResidualL2Norm = std::sqrt(restrictedResidualL2NormSquared);
          const Scalar coarsePredictedEnergyReduction = coarseResidualDotCorrection - Scalar(0.5) * coarseCorrectionEnergy;
          const Scalar finePredictedEnergyReduction = fineResidualDotCorrection - Scalar(0.5) * fineCorrectionEnergy;
          if (p == 2)
          {
            std::cout << "VLASOV_GRAPH_NORM_PMULTIGRID_STAGE p=" << p
                      << " cycle=" << (cycle + 1)
                      << " cycles=" << numInverseIterations
                      << " initialResidualNorm=" << std::setprecision(16) << cycleInitialResidualNorm
                      << " residualAfterPreSmooth=" << residualAfterPreSmooth
                      << " restrictedResidualNorm=" << restrictedResidualNorm
                      << " coarseCorrectionNorm=" << coarseCorrectionNorm
                      << " prolongatedCorrectionNorm=" << prolongatedCorrectionNorm
                      << " residualAfterPlusCoarseCorrection=" << residualAfterPlusCoarseCorrection
                      << " residualAfterMinusCoarseCorrection=" << residualAfterMinusCoarseCorrection
                      << " residualAfterCoarseCorrection=" << residualAfterCoarseCorrection
                      << " residualAfterPostSmooth=" << residualAfterPostSmooth
                      << " coarseSolveResidualNorm=" << coarseSolveResidualNorm
                      << " galerkinActionDefectNorm=" << galerkinActionDefectNorm
                      << " restrictedResidualL2Norm=" << restrictedResidualL2Norm
                      << " coarseResidualDotCorrection=" << coarseResidualDotCorrection
                      << " fineResidualDotCorrection=" << fineResidualDotCorrection
                      << " coarseCorrectionEnergy=" << coarseCorrectionEnergy
                      << " fineCorrectionEnergy=" << fineCorrectionEnergy
                      << " coarsePredictedEnergyReduction=" << coarsePredictedEnergyReduction
                      << " finePredictedEnergyReduction=" << finePredictedEnergyReduction
                      << " prolongationScaleRatio=" << prolongationScaleRatio
                      << " fineActionScaleRatio=" << fineActionScaleRatio
                      << " coarseActionScaleRatio=" << coarseActionScaleRatio
                      << " fineMatrixMaxAbs=" << fineMatrixMaxAbs
                      << " fineMatrixFrobeniusNorm=" << fineMatrixFrobeniusNorm
                      << " fineMatrixDiagonalMinAbs=" << fineMatrixDiagonalMinAbs
                      << " fineMatrixDiagonalMaxAbs=" << fineMatrixDiagonalMaxAbs
                      << " coarseMatrixMaxAbs=" << coarseMatrixMaxAbs
                      << " coarseMatrixFrobeniusNorm=" << coarseMatrixFrobeniusNorm
                      << " coarseMatrixDiagonalMinAbs=" << coarseMatrixDiagonalMinAbs
                      << " coarseMatrixDiagonalMaxAbs=" << coarseMatrixDiagonalMaxAbs
                      << " assembledProxyOperatorActionMaxRelDiff=" << timing.operatorActionMaxRelDiff
                      << " assembledProxyOperatorBilinearSymmetryAbsError=" << timing.operatorBilinearSymmetryAbsError
                      << " lambdaMax=" << timing.lambdaMaxEstimate
                      << " smootherIterations=" << pMultigridChebyshevSmootherIterations
                      << " fineSmootherWeight=" << pMultigridFineSmootherWeight
                      << " coarseDofs=" << coarseRows
                      << " residualReduction=" << (residualAfterPostSmooth / std::max<Scalar>(std::numeric_limits<Scalar>::epsilon(), cycleInitialResidualNorm))
                      << std::endl;
          }
        }
        timing.pMultigridLevels = 2;
        timing.pMultigridFineDofs = rows;
        timing.pMultigridCoarseDofs = coarseRows;
        timing.pMultigridChebyshevSmootherIterations = pMultigridChebyshevSmootherIterations;
        timing.pMultigridChebyshevSmootherLowerFraction = pMultigridChebyshevSmootherLowerFraction;
        timing.pMultigridFineSmootherWeight = pMultigridFineSmootherWeight;
        timing.pMultigridCoarseSolveCount = numInverseIterations * cells;
        timing.inversePath = "two_level_pmultigrid_cached_tensor_hgrad_transfer_chebyshev_smoother";
        timing.pMultigridPath = "two_level_cached_tensor_hgrad_hierarchical_restriction_prolongation_galerkin_coarse_exact_solve_high_frequency_chebyshev_pre_post_smoothing_diagnostic";
      }
      else
      {
        struct PMultigridLevel
        {
          int degree = 0;
          int levelRows = 0;
          std::vector<Scalar> matrix;
          std::vector<Scalar> levelDiagonal;
          std::vector<Scalar> lambdaMaxByCell;
        };

        std::vector<PMultigridLevel> levels;
        levels.push_back({p, rows, localMatrices, diagonal, std::vector<Scalar>(static_cast<size_t>(cells), timing.lambdaMaxEstimate)});
        for (int c=0; c<cells; c++)
          if (!perCellLambdaMaxEstimates.empty()) levels.front().lambdaMaxByCell[static_cast<size_t>(c)] = perCellLambdaMaxEstimates[static_cast<size_t>(c)];
        levels.push_back({p - 1, coarseRows, coarseMatrices, coarseDiagonal, coarseLambdaMaxByCell});
        while (levels.back().degree > 1)
        {
          const PMultigridLevel &fineLevel = levels.back();
          const TensorHGradPTransfer &levelTransfer = cachedTensorHGradPTransfer(dim, fineLevel.degree);
          const int levelCoarseRows = levelTransfer.coarseRows;
          PMultigridLevel nextLevel;
          nextLevel.degree = fineLevel.degree - 1;
          nextLevel.levelRows = levelCoarseRows;
          nextLevel.matrix.assign(static_cast<size_t>(cells * levelCoarseRows * levelCoarseRows), Scalar(0.0));
          nextLevel.levelDiagonal.assign(static_cast<size_t>(cells * levelCoarseRows), Scalar(0.0));
          nextLevel.lambdaMaxByCell.assign(static_cast<size_t>(cells), Scalar(0.0));
          std::vector<Scalar> levelMatrixTimesP(static_cast<size_t>(fineLevel.levelRows * levelCoarseRows), Scalar(0.0));
          for (int c=0; c<cells; c++)
          {
            std::fill(levelMatrixTimesP.begin(), levelMatrixTimesP.end(), Scalar(0.0));
            for (int fineI=0; fineI<fineLevel.levelRows; fineI++)
              for (int fineJ=0; fineJ<fineLevel.levelRows; fineJ++)
              {
                const Scalar matrixEntry = fineLevel.matrix[static_cast<size_t>((c * fineLevel.levelRows + fineI) * fineLevel.levelRows + fineJ)];
                if (matrixEntry == Scalar(0.0)) continue;
                for (int coarseJ=0; coarseJ<levelCoarseRows; coarseJ++)
                  levelMatrixTimesP[static_cast<size_t>(fineI * levelCoarseRows + coarseJ)] += matrixEntry * levelTransfer.prolongation[static_cast<size_t>(fineJ * levelCoarseRows + coarseJ)];
              }
            for (int coarseI=0; coarseI<levelCoarseRows; coarseI++)
              for (int fineI=0; fineI<fineLevel.levelRows; fineI++)
              {
                const Scalar restrictionEntry = levelTransfer.restriction[static_cast<size_t>(coarseI * fineLevel.levelRows + fineI)];
                if (restrictionEntry == Scalar(0.0)) continue;
                for (int coarseJ=0; coarseJ<levelCoarseRows; coarseJ++)
                  nextLevel.matrix[static_cast<size_t>((c * levelCoarseRows + coarseI) * levelCoarseRows + coarseJ)] += restrictionEntry * levelMatrixTimesP[static_cast<size_t>(fineI * levelCoarseRows + coarseJ)];
              }
            for (int i=0; i<levelCoarseRows; i++)
            {
              const Scalar diagonalValue = nextLevel.matrix[static_cast<size_t>((c * levelCoarseRows + i) * levelCoarseRows + i)];
              if (diagonalValue <= std::numeric_limits<Scalar>::epsilon())
                throw std::runtime_error("p-multigrid Chebyshev coarse solve found nonpositive recursive coarse diagonal entry");
              nextLevel.levelDiagonal[static_cast<size_t>(c * levelCoarseRows + i)] = diagonalValue;
              Scalar scaledRowSum = 0.0;
              for (int j=0; j<levelCoarseRows; j++)
                scaledRowSum += std::abs(nextLevel.matrix[static_cast<size_t>((c * levelCoarseRows + i) * levelCoarseRows + j)]) / diagonalValue;
              nextLevel.lambdaMaxByCell[static_cast<size_t>(c)] = std::max(nextLevel.lambdaMaxByCell[static_cast<size_t>(c)], scaledRowSum);
            }
          }
          levels.push_back(std::move(nextLevel));
        }

        auto applyLevelMatrix = [&](const PMultigridLevel &level, const int cellOrdinal, const std::vector<Scalar> &inputVector, std::vector<Scalar> &outputVector)
        {
          outputVector.assign(static_cast<size_t>(level.levelRows), Scalar(0.0));
          const size_t offset = static_cast<size_t>(cellOrdinal) * level.levelRows * level.levelRows;
          for (int i=0; i<level.levelRows; i++)
            for (int j=0; j<level.levelRows; j++)
              outputVector[static_cast<size_t>(i)] += level.matrix[offset + static_cast<size_t>(i * level.levelRows + j)] * inputVector[static_cast<size_t>(j)];
        };

        auto jacobiSmoothLevel = [&](const PMultigridLevel &level, const int cellOrdinal, const std::vector<Scalar> &levelRhs, std::vector<Scalar> &levelSolution)
        {
          std::vector<Scalar> levelApplied(static_cast<size_t>(level.levelRows), Scalar(0.0));
          const Scalar omega = Scalar(1.0) / level.lambdaMaxByCell[static_cast<size_t>(cellOrdinal)];
          for (int sweep=0; sweep<pMultigridJacobiSmootherIterations; sweep++)
          {
            applyLevelMatrix(level, cellOrdinal, levelSolution, levelApplied);
            for (int i=0; i<level.levelRows; i++)
            {
              const size_t index = static_cast<size_t>(cellOrdinal * level.levelRows + i);
              levelSolution[static_cast<size_t>(i)] += omega * (levelRhs[static_cast<size_t>(i)] - levelApplied[static_cast<size_t>(i)]) / level.levelDiagonal[index];
            }
          }
        };

        auto coarseChebyshevSolveLevel = [&](const PMultigridLevel &level, const int cellOrdinal, const std::vector<Scalar> &levelRhs, std::vector<Scalar> &levelSolution)
        {
          levelSolution.assign(static_cast<size_t>(level.levelRows), Scalar(0.0));
          const Scalar chebyshevLambdaMax = level.lambdaMaxByCell[static_cast<size_t>(cellOrdinal)];
          const Scalar chebyshevLambdaMin = std::max(std::numeric_limits<Scalar>::epsilon(), pMultigridCoarseChebyshevLowerFraction * chebyshevLambdaMax);
          if (!(chebyshevLambdaMax > chebyshevLambdaMin))
            throw std::runtime_error("p-multigrid Chebyshev coarse solve has invalid spectral interval");
          const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
          const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
          const Scalar s1 = ifpackTheta * ifpackDelta;
          Scalar rhok = Scalar(1.0) / s1;
          std::vector<Scalar> correction(static_cast<size_t>(level.levelRows), Scalar(0.0));
          std::vector<Scalar> levelApplied(static_cast<size_t>(level.levelRows), Scalar(0.0));
          applyLevelMatrix(level, cellOrdinal, levelSolution, levelApplied);
          for (int i=0; i<level.levelRows; i++)
          {
            const size_t index = static_cast<size_t>(cellOrdinal * level.levelRows + i);
            correction[static_cast<size_t>(i)] = (Scalar(1.0) / ifpackTheta) * (levelRhs[static_cast<size_t>(i)] - levelApplied[static_cast<size_t>(i)]) / level.levelDiagonal[index];
            levelSolution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
          }
          for (int iteration=1; iteration<pMultigridCoarseChebyshevIterations; iteration++)
          {
            applyLevelMatrix(level, cellOrdinal, levelSolution, levelApplied);
            const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhok);
            const Scalar dtemp1 = rhokp1 * rhok;
            const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
            rhok = rhokp1;
            for (int i=0; i<level.levelRows; i++)
            {
              const size_t index = static_cast<size_t>(cellOrdinal * level.levelRows + i);
              correction[static_cast<size_t>(i)] = dtemp2 * (levelRhs[static_cast<size_t>(i)] - levelApplied[static_cast<size_t>(i)]) / level.levelDiagonal[index]
                                             + dtemp1 * correction[static_cast<size_t>(i)];
              levelSolution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
            }
          }
        };

        std::function<void(size_t,int,const std::vector<Scalar>&,std::vector<Scalar>&)> vCycle;
        vCycle = [&](const size_t levelOrdinal, const int cellOrdinal, const std::vector<Scalar> &levelRhs, std::vector<Scalar> &levelSolution)
        {
          const PMultigridLevel &level = levels[levelOrdinal];
          levelSolution.assign(static_cast<size_t>(level.levelRows), Scalar(0.0));
          if (levelOrdinal + 1 == levels.size())
          {
            coarseChebyshevSolveLevel(level, cellOrdinal, levelRhs, levelSolution);
            return;
          }

          jacobiSmoothLevel(level, cellOrdinal, levelRhs, levelSolution);
          std::vector<Scalar> levelApplied(static_cast<size_t>(level.levelRows), Scalar(0.0));
          applyLevelMatrix(level, cellOrdinal, levelSolution, levelApplied);
          std::vector<Scalar> levelResidual(static_cast<size_t>(level.levelRows), Scalar(0.0));
          for (int i=0; i<level.levelRows; i++) levelResidual[static_cast<size_t>(i)] = levelRhs[static_cast<size_t>(i)] - levelApplied[static_cast<size_t>(i)];
          const TensorHGradPTransfer &levelTransfer = cachedTensorHGradPTransfer(dim, level.degree);
          std::vector<Scalar> restrictedResidual;
          applyTensorHGradPTransferRestriction(levelTransfer, levelResidual, restrictedResidual);
          std::vector<Scalar> coarseLevelCorrection;
          vCycle(levelOrdinal + 1, cellOrdinal, restrictedResidual, coarseLevelCorrection);
          std::vector<Scalar> levelCorrection;
          applyTensorHGradPTransferProlongation(levelTransfer, coarseLevelCorrection, levelCorrection);
          for (int i=0; i<level.levelRows; i++) levelSolution[static_cast<size_t>(i)] += levelCorrection[static_cast<size_t>(i)];
          jacobiSmoothLevel(level, cellOrdinal, levelRhs, levelSolution);
        };

        for (int cycle=0; cycle<numInverseIterations; cycle++)
        {
          applyHostVector();
          for (int c=0; c<cells; c++)
          {
            const size_t cellOffset = static_cast<size_t>(c * rows);
            for (int i=0; i<rows; i++) fineResidual[static_cast<size_t>(i)] = rhs[cellOffset + i] - applied[cellOffset + i];
            vCycle(0, c, fineResidual, fineCorrection);
            for (int i=0; i<rows; i++) iterate[cellOffset + i] += fineCorrection[static_cast<size_t>(i)];
          }
        }
        timing.pMultigridLevels = static_cast<int>(levels.size());
        timing.pMultigridFineDofs = rows;
        timing.pMultigridCoarseDofs = levels.back().levelRows;
        timing.pMultigridFineSmootherWeight = pMultigridFineSmootherWeight;
        timing.pMultigridJacobiSmootherIterations = pMultigridJacobiSmootherIterations;
        timing.pMultigridCoarseChebyshevIterations = pMultigridCoarseChebyshevIterations;
        timing.pMultigridCoarseChebyshevLowerFraction = pMultigridCoarseChebyshevLowerFraction;
        timing.pMultigridCoarseSolveCount = numInverseIterations * cells;
        timing.inversePath = "recursive_pmultigrid_cached_tensor_hgrad_transfer_jacobi_smoother_chebyshev_coarse";
        timing.pMultigridPath = "recursive_cached_tensor_hgrad_hierarchical_restriction_prolongation_galerkin_hierarchy_diagonal_jacobi_smoothing_chebyshev_coarse_solver_diagnostic";
      }
    }
    else
    {
      throw std::invalid_argument("--inverseDiagnosticAlgorithm must be richardson, chebyshev, gmres-chebyshev, gmres-jacobi, matrix-free-gmg-chebyshev-smoother, matrix-free-gmg-jacobi-smoother, gmres-pmultigrid-chebyshev, gmres-pmultigrid-jacobi-chebyshev, pmultigrid-chebyshev, or pmultigrid-jacobi-chebyshev");
    }
    timing.avgInverseApplySeconds = inverseTimer.seconds() / std::max(1, timing.inverseIterations);

    if (ifpack2ChebyshevSmoke != 0)
    {
      if (inverseDiagnosticAlgorithm != "chebyshev")
        throw std::invalid_argument("--ifpack2ChebyshevSmoke requires --inverseDiagnosticAlgorithm=chebyshev");
      if (inverseInitialGuess != "zero")
        throw std::invalid_argument("--ifpack2ChebyshevSmoke currently supports --inverseInitialGuess=zero only");
      if (hostChebyshevIterate.size() != rhs.size())
        throw std::runtime_error("Ifpack2 Chebyshev smoke did not find a host Chebyshev reference iterate");

      const size_t localLength = rhs.size();
      std::vector<Scalar> ifpackHostReference(localLength, Scalar(0.0));
      runIfpack2CompatibleHostChebyshev(ifpackHostReference, numInverseIterations);

      auto comm = Teuchos::DefaultComm<int>::getComm();
      auto map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localLength, 0, comm));
      auto rowMatrix = TpetraAdapters::createLocalPAMatrixTpetraRowMatrix(map, terms, workspaces, cells, rows, worksetSize, maxAssembledEntries);
      Teuchos::RCP<const TpetraRowMatrix> rowMatrixBase = Teuchos::rcp_implicit_cast<const TpetraRowMatrix>(rowMatrix);
      auto inverseDiagonal = Teuchos::rcp(new TpetraVector(map));
      {
        auto inverseDiagonalHost = inverseDiagonal->getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i=0; i<localLength; i++) inverseDiagonalHost(i,0) = Scalar(1.0) / diagonal[i];
      }

      TpetraMultiVector B(map, 1);
      TpetraMultiVector X(map, 1);
      {
        auto bHost = B.getLocalViewHost(Tpetra::Access::OverwriteAll);
        auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i=0; i<localLength; i++)
        {
          bHost(i,0) = rhs[i];
          xHost(i,0) = Scalar(0.0);
        }
      }

      Teuchos::ParameterList params;
      params.set("chebyshev: degree", numInverseIterations);
      params.set("chebyshev: max eigenvalue", timing.lambdaMaxEstimate);
      params.set("chebyshev: min eigenvalue", timing.lambdaMinEstimate);
      params.set("chebyshev: ratio eigenvalue", timing.lambdaMaxEstimate / timing.lambdaMinEstimate);
      params.set("chebyshev: boost factor", Scalar(1.0));
      params.set("chebyshev: zero starting solution", true);
      params.set("chebyshev: operator inv diagonal", inverseDiagonal);
      params.set("chebyshev: eigenvalue max iterations", 0);
      params.set("chebyshev: use native spmv", false);

      using Ifpack2Chebyshev = Ifpack2::Chebyshev<TpetraRowMatrix>;
      Kokkos::Timer ifpack2Timer;
      Ifpack2Chebyshev chebyshev(rowMatrixBase);
      chebyshev.setParameters(params);
      chebyshev.initialize();
      chebyshev.compute();
      chebyshev.apply(B, X);
      timing.ifpack2ChebyshevSeconds = ifpack2Timer.seconds();

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      Scalar maxReference = 0.0;
      timing.ifpack2ChebyshevMaxAbsError = 0.0;
      for (size_t i=0; i<localLength; i++)
      {
        const Scalar referenceValue = ifpackHostReference[i];
        maxReference = std::max(maxReference, std::abs(referenceValue));
        timing.ifpack2ChebyshevMaxAbsError = std::max(timing.ifpack2ChebyshevMaxAbsError,
                                                       std::abs(xHost(i,0) - referenceValue));
      }
      timing.ifpack2ChebyshevMaxRelError = timing.ifpack2ChebyshevMaxAbsError / std::max<Scalar>(Scalar(1.0), maxReference);
      timing.ifpack2ChebyshevPath = "local_flat_rowmatrix_ifpack2_chebyshev_smoke_against_matching_host_recurrence_not_production_solver";
      std::cout << "IFPACK2_CHEBYSHEV_SMOKE maxAbsError=" << std::setprecision(16) << timing.ifpack2ChebyshevMaxAbsError
                << " maxRelError=" << timing.ifpack2ChebyshevMaxRelError
                << " seconds=" << timing.ifpack2ChebyshevSeconds
                << " iterations=" << numInverseIterations
                << " lambdaMinEstimate=" << timing.lambdaMinEstimate
                << " lambdaMaxEstimate=" << timing.lambdaMaxEstimate
                << " inverseEigenRatio=" << timing.inverseEigenRatio
                << " lambdaMinRawEstimate=" << timing.lambdaMinRawEstimate
                << " lambdaMinSafetyFactor=" << timing.lambdaMinSafetyFactor
                << " lambdaMinEstimateOverRaw=" << timing.lambdaMinEstimateOverRaw
                << " lambdaMinSource=" << timing.lambdaMinSource
                << " lambdaMaxSource=" << timing.lambdaMaxSource
                << " inverseLowerSpectralEstimate=" << timing.inverseLowerSpectralEstimate
                << " inverseLowerSpectralEstimateIterations=" << inverseLowerSpectralEstimateIterations
                << " spectralEstimatePath=" << timing.spectralEstimatePath
                << " diagonalSource=pamatrix_extract_diagonal_true_A_then_inverted_for_ifpack2_parameter"
                << " ifpack2Parameter=chebyshev_operator_inv_diagonal"
                << " adapterPath=local_flat_map_nonproduction_rowmatrix"
                << " globalDofSemantics=not_supported"
                << " rowMatrixSemantics=local_flat_identical_domain_range_row_col_maps"
                << " path=" << timing.ifpack2ChebyshevPath
                << " note=diagnostic_smoke_only_not_global_fe_solver" << std::endl;
    }

    applyHostVector();
    timing.finalResidualNorm = residualNorm();
    timing.residualReduction = timing.finalResidualNorm / std::max<Scalar>(std::numeric_limits<Scalar>::epsilon(), timing.initialResidualNorm);
    if (!useBelosGmres)
    {
      timing.inverseConverged = (timing.residualReduction <= inverseTolerance) ? 1 : 0;
      if (timing.inverseConverged != 0) timing.inverseStoppedReason = "fixed_iterations_met_tolerance";
    }

    if (useAssembledReference)
    {
      timing.maxInverseAbsError = 0.0;
      for (size_t i=0; i<iterate.size(); i++)
        timing.maxInverseAbsError = std::max(timing.maxInverseAbsError, std::abs(iterate[i] - referenceSolution[i]));
      timing.maxInverseRelError = timing.maxInverseAbsError / std::max<Scalar>(1.0, referenceNorm);
    }
    std::cout << "INVERSE_DIAGNOSTIC initialMaxAbsError=" << std::setprecision(16) << timing.initialInverseAbsError
              << " initialMaxRelError=" << timing.initialInverseRelError
              << " maxAbsError=" << timing.maxInverseAbsError
              << " maxRelError=" << timing.maxInverseRelError
              << " initialResidualNorm=" << timing.initialResidualNorm
              << " finalResidualNorm=" << timing.finalResidualNorm
              << " residualReduction=" << timing.residualReduction
              << " inverseTolerance=" << timing.inverseTolerance
              << " inversePreconditionerDegree=" << timing.inversePreconditionerDegree
              << " inverseConverged=" << timing.inverseConverged
              << " inverseStoppedReason=" << timing.inverseStoppedReason
              << " lambdaMinEstimate=" << timing.lambdaMinEstimate
              << " lambdaMaxEstimate=" << timing.lambdaMaxEstimate
              << " assembledGershgorinLambdaMax=" << timing.assembledGershgorinLambdaMax
              << " powerRawLambdaMaxEstimate=" << timing.powerRawLambdaMaxEstimate
              << " tpetraPowerRawLambdaMaxEstimate=" << timing.tpetraPowerRawLambdaMaxEstimate
              << " tpetraPowerLambdaMaxAbsError=" << timing.tpetraPowerLambdaMaxAbsError
              << " tpetraPowerLambdaMaxRelError=" << timing.tpetraPowerLambdaMaxRelError
              << " tpetraRowMatrixPowerRawLambdaMaxEstimate=" << timing.tpetraRowMatrixPowerRawLambdaMaxEstimate
              << " tpetraRowMatrixPowerLambdaMaxAbsError=" << timing.tpetraRowMatrixPowerLambdaMaxAbsError
              << " tpetraRowMatrixPowerLambdaMaxRelError=" << timing.tpetraRowMatrixPowerLambdaMaxRelError
              << " compareTpetraRowMatrixPowerEstimate=" << timing.compareTpetraRowMatrixPowerEstimate
              << " lambdaMinRawEstimate=" << timing.lambdaMinRawEstimate
              << " lambdaMinSafetyFactor=" << timing.lambdaMinSafetyFactor
              << " lambdaMinEstimateOverRaw=" << timing.lambdaMinEstimateOverRaw
              << " compareTpetraPowerEstimate=" << timing.compareTpetraPowerEstimate
              << " lambdaMaxEstimateOverAssembledGershgorin=" << timing.lambdaMaxEstimateOverAssembledGershgorin
              << " inverseEigenRatio=" << timing.inverseEigenRatio
              << " spectralEstimateIterations=" << timing.spectralEstimateIterations
              << " spectralEstimateSeconds=" << timing.spectralEstimateSeconds
              << " lambdaMaxSafetyFactor=" << timing.lambdaMaxSafetyFactor
              << " lambdaMinSource=" << timing.lambdaMinSource
              << " lambdaMaxSource=" << timing.lambdaMaxSource
              << " perCellSpectralEstimate=" << timing.perCellSpectralEstimate
              << " perCellLambdaMinEstimateMin=" << timing.perCellLambdaMinEstimateMin
              << " perCellLambdaMinEstimateMax=" << timing.perCellLambdaMinEstimateMax
              << " perCellLambdaMaxEstimateMin=" << timing.perCellLambdaMaxEstimateMin
              << " perCellLambdaMaxEstimateMax=" << timing.perCellLambdaMaxEstimateMax
              << " perCellLambdaMaxSpread=" << timing.perCellLambdaMaxSpread
              << " inverseLowerSpectralEstimate=" << timing.inverseLowerSpectralEstimate
              << " inverseLowerSpectralEstimateIterations=" << inverseLowerSpectralEstimateIterations
              << " maxSymmetryAbsError=" << timing.maxSymmetryAbsError
              << " minDiagonalEstimate=" << timing.minDiagonalEstimate
              << " maxDiagonalEstimate=" << timing.maxDiagonalEstimate
              << " trueEigenDiagnostic=" << timing.trueEigenDiagnostic
              << " trueLambdaMin=" << timing.trueLambdaMin
              << " trueLambdaMax=" << timing.trueLambdaMax
              << " trueCondition=" << timing.trueCondition
              << " trueLambdaMaxOverAssembledGershgorin=" << timing.trueLambdaMaxOverAssembledGershgorin
              << " trueLambdaMaxOverPowerRaw=" << timing.trueLambdaMaxOverPowerRaw
              << " trueLambdaMinOverLowerEstimate=" << timing.trueLambdaMinOverLowerEstimate
              << " trueEigenPath=" << timing.trueEigenPath
              << " operatorCheckSamples=" << timing.operatorCheckSamples
              << " operatorActionMaxAbsDiff=" << timing.operatorActionMaxAbsDiff
              << " operatorActionMaxRelDiff=" << timing.operatorActionMaxRelDiff
              << " operatorBilinearSymmetryAbsError=" << timing.operatorBilinearSymmetryAbsError
              << " operatorMinRayleighQuotient=" << timing.operatorMinRayleighQuotient
              << " chebyshevScalarNumEigenvalues=" << timing.chebyshevScalarNumEigenvalues
              << " chebyshevScalarNumBelowLower=" << timing.chebyshevScalarNumBelowLower
              << " chebyshevScalarNumInsideInterval=" << timing.chebyshevScalarNumInsideInterval
              << " chebyshevScalarNumAboveUpper=" << timing.chebyshevScalarNumAboveUpper
              << " chebyshevScalarNumModesAmplified=" << timing.chebyshevScalarNumModesAmplified
              << " chebyshevScalarMaxAbsAmplification=" << timing.chebyshevScalarMaxAbsAmplification
              << " chebyshevScalarWorstLambda=" << timing.chebyshevScalarWorstLambda
              << " pMultigridLevels=" << timing.pMultigridLevels
              << " pMultigridFineDofs=" << timing.pMultigridFineDofs
              << " pMultigridCoarseDofs=" << timing.pMultigridCoarseDofs
              << " pMultigridChebyshevSmootherIterations=" << timing.pMultigridChebyshevSmootherIterations
              << " pMultigridChebyshevSmootherLowerFraction=" << timing.pMultigridChebyshevSmootherLowerFraction
              << " pMultigridFineSmootherWeight=" << timing.pMultigridFineSmootherWeight
              << " pMultigridJacobiSmootherIterations=" << timing.pMultigridJacobiSmootherIterations
              << " pMultigridCoarseChebyshevIterations=" << timing.pMultigridCoarseChebyshevIterations
              << " pMultigridCoarseChebyshevLowerFraction=" << timing.pMultigridCoarseChebyshevLowerFraction
              << " pMultigridCoarseSolveCount=" << timing.pMultigridCoarseSolveCount
              << " matrixFreeGMGStageTrace=" << timing.matrixFreeGMGStageTrace
              << " matrixFreeGMGStageTraceEntries=" << timing.matrixFreeGMGStageTraceEntries
              << " pMultigridPath=" << timing.pMultigridPath
              << " iterations=" << timing.inverseIterations
              << " inverseInitialGuess=" << timing.inverseInitialGuess
              << " inverseRhs=" << timing.inverseRhs
              << " inversePath=" << timing.inversePath
              << " spectralEstimatePath=" << timing.spectralEstimatePath << std::endl;
    return timing;
  }

  FactorTiming factorTerms(std::vector<OperatorTerm> &terms, const long long maxAssembledEntries, const int numSolve)
  {
    FactorTiming timing;
    if (terms.empty()) return timing;

    auto firstMatrixData = terms.front().matrix.allocateMatrixStorage();
    const int cells = firstMatrixData.extent_int(0);
    const int rows  = firstMatrixData.extent_int(1);
    const int cols  = firstMatrixData.extent_int(2);
    if (rows != cols) throw std::invalid_argument("local factor timing requires square Gram blocks");
    const long long entries = static_cast<long long>(cells) * rows * cols;
    if (entries > maxAssembledEntries)
    {
      std::cout << "FACTOR_SKIPPED entries=" << entries
                << " maxAssembledEntries=" << maxAssembledEntries << std::endl;
      return timing;
    }

    std::vector<Scalar> localMatrices(static_cast<size_t>(entries), 0.0);
    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      Kokkos::Timer assemblyTimer;
      term.matrix.assemble(matrixData);
      Kokkos::fence();
      addTermSeconds(timing.assembly, term.label, assemblyTimer.seconds());

      if (matrixData.getUnderlyingViewRank() == 3)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<3>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
            for (int j=0; j<cols; j++)
              localMatrices[static_cast<size_t>((c * rows + i) * cols + j)] += term.weight * matrixHost(c,i,j);
      }
      else if (matrixData.getUnderlyingViewRank() == 2)
      {
        auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<rows; i++)
            for (int j=0; j<cols; j++)
              localMatrices[static_cast<size_t>((c * rows + i) * cols + j)] += term.weight * matrixHost(i,j);
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix storage rank in factor timing");
      }
    }

    const std::vector<Scalar> originalMatrices = localMatrices;
    Kokkos::Timer factorTimer;
    for (int c=0; c<cells; c++)
    {
      const int factorInfo = factorSPDMatrixInPlace(localMatrices, static_cast<size_t>(c) * rows * cols, rows);
      if (factorInfo != 0)
      {
        throw std::runtime_error("local H1 Gram Cholesky factorization failed with LAPACK info=" + std::to_string(factorInfo));
      }
    }
    timing.factorSeconds = factorTimer.seconds();

    std::vector<Scalar> rhs(static_cast<size_t>(cells * rows), 0.0);
    for (int c=0; c<cells; c++)
      for (int i=0; i<rows; i++)
        rhs[static_cast<size_t>(c * rows + i)] = 0.125 + 0.03125 * ((i + 7 * c) % 23);

    Scalar maxResidual = 0.0;
    Scalar maxRhs = 0.0;
    for (int c=0; c<cells; c++)
    {
      std::vector<Scalar> solution(rhs.begin() + static_cast<size_t>(c * rows),
                                   rhs.begin() + static_cast<size_t>((c + 1) * rows));
      const int solveInfo = solveFactoredSPDMatrix(localMatrices, static_cast<size_t>(c) * rows * cols, rows, solution);
      if (solveInfo != 0)
      {
        throw std::runtime_error("local H1 Gram Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
      }
      for (int i=0; i<rows; i++)
      {
        Scalar residual = -rhs[static_cast<size_t>(c * rows + i)];
        for (int j=0; j<cols; j++)
        {
          residual += originalMatrices[static_cast<size_t>((c * rows + i) * cols + j)] * solution[j];
        }
        maxResidual = std::max(maxResidual, std::abs(residual));
        maxRhs = std::max(maxRhs, std::abs(rhs[static_cast<size_t>(c * rows + i)]));
      }
    }
    timing.maxSolveResidual = maxResidual;
    timing.maxSolveRelativeResidual = maxResidual / std::max<Scalar>(1.0, maxRhs);
    if ((timing.maxSolveResidual > 1e-8) && (timing.maxSolveRelativeResidual > 1e-8))
    {
      std::cout << "FACTOR_RESIDUAL_WARNING maxSolveResidual=" << std::setprecision(16) << timing.maxSolveResidual
                << " maxSolveRelativeResidual=" << timing.maxSolveRelativeResidual
                << " tolerance=1e-8 note=nonfatal_sweep_diagnostic" << std::endl;
    }

    Kokkos::Timer solveTimer;
    for (int solveOrdinal=0; solveOrdinal<numSolve; solveOrdinal++)
    {
      for (int c=0; c<cells; c++)
      {
        std::vector<Scalar> working(rhs.begin() + static_cast<size_t>(c * rows),
                                    rhs.begin() + static_cast<size_t>((c + 1) * rows));
        const int solveInfo = solveFactoredSPDMatrix(localMatrices, static_cast<size_t>(c * rows * cols), rows, working);
        if (solveInfo != 0)
        {
          throw std::runtime_error("local H1 Gram Cholesky solve failed with LAPACK info=" + std::to_string(solveInfo));
        }
      }
    }
    timing.solveSeconds = solveTimer.seconds() / numSolve;
    return timing;
  }

  RepeatedTiming diagonalTerms(std::vector<OperatorTerm> &terms, const int numDiagonal)
  {
    std::vector<Intrepid2::Data<Scalar,DeviceType>> diagonals;
    diagonals.reserve(terms.size());
    for (auto &term : terms)
    {
      diagonals.push_back(term.matrix.allocateDiagonalStorage());
    }

    RepeatedTiming timing;
    Kokkos::Timer firstTimer;
    for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
    {
      terms[termOrdinal].matrix.extractDiagonal(diagonals[termOrdinal]);
    }
    Kokkos::fence();
    timing.firstSeconds = firstTimer.seconds();

    Kokkos::Timer repeatTimer;
    for (int diagonalOrdinal=0; diagonalOrdinal<numDiagonal; diagonalOrdinal++)
    {
      for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
      {
        terms[termOrdinal].matrix.extractDiagonal(diagonals[termOrdinal]);
      }
    }
    Kokkos::fence();
    timing.averageSeconds = repeatTimer.seconds() / numDiagonal;
    return timing;
  }

  std::string termSummary(const std::vector<OperatorTerm> &terms)
  {
    std::string summary;
    for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
    {
      if (!summary.empty()) summary += ",";
      summary += terms[termOrdinal].label + ":" + std::to_string(terms[termOrdinal].weight);
    }
    return summary;
  }

  bool isDisabledResultToken(const std::string &token)
  {
    const auto equalsPosition = token.find('=');
    if (equalsPosition == std::string::npos) return false;
    const std::string value = token.substr(equalsPosition + 1);
    return (value == "-1") || (value == "not_requested");
  }

  void printCompactKeyValueLine(const std::string &line)
  {
    std::istringstream tokens(line);
    std::string token;
    bool first = true;
    while (tokens >> token)
    {
      if (isDisabledResultToken(token)) continue;
      if (!first) std::cout << ' ';
      std::cout << token;
      first = false;
    }
    std::cout << std::endl;
  }

  double leastSquaresSlope(const std::vector<double> &x, const std::vector<double> &y)
  {
    if (x.size() != y.size() || x.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
    for (size_t i=0; i<x.size(); i++)
    {
      sumX  += x[i];
      sumY  += y[i];
      sumXX += x[i] * x[i];
      sumXY += x[i] * y[i];
    }
    const double n = static_cast<double>(x.size());
    const double denominator = n * sumXX - sumX * sumX;
    return (denominator == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                : (n * sumXY - sumX * sumY) / denominator;
  }

  long long estimatedApplyFlopsPerCellPerTerm(const int dim, const int p)
  {
    const long long fields = p + 1;
    const long long points = p + 2;
    long long coefficientToPoint = 0;
    long long pointToCoefficient = 0;
    long long rightPartial = 1;
    long long leftPartial = 1;
    long long pointCount = 1;
    for (int d=0; d<dim; d++)
    {
      rightPartial *= fields;
      leftPartial *= points;
      pointCount *= points;
    }
    for (int d=dim-1; d>=0; d--)
    {
      coefficientToPoint += 2LL * points * rightPartial;
      rightPartial = rightPartial / fields * points;
      pointToCoefficient += 2LL * fields * leftPartial;
      leftPartial = leftPartial / points * fields;
    }
    return coefficientToPoint + pointCount + pointToCoefficient;
  }

  DenseVectorComparison runCamelliaVlasovRhsComparison(
      const int p,
      const Scalar massWeight,
      const long long maxAssembledEntries,
      const bool useSolutionCoefficientProxies)
  {
    const int dim = 4;
    const int cells = 1;
    const int localDofs = checkedProduct(std::vector<int>(dim, p + 1));
    const int worksetSize = cells;

    auto matrixFreeTerms =
        makeVlasovCamelliaComparisonMatrixFreeTerms(p,
                                                    VlasovCamelliaComparisonBlock::FullGraphNorm,
                                                    massWeight,
                                                    useSolutionCoefficientProxies);

    auto workspaces = TpetraAdapters::allocateWorkspaces(matrixFreeTerms, worksetSize);

    const auto matrixFreeLocalRhs =
        makeConstantTestFunctionRhs(matrixFreeTerms,
                                    workspaces,
                                    dim,
                                    p,
                                    cells,
                                    localDofs,
                                    worksetSize);

    auto camelliaSolution =
        makeCamelliaVlasovGraphGramComparisonSolution(p,
                                                      massWeight,
                                                      VlasovCamelliaComparisonBlock::FullGraphNorm);

    std::vector<Camellia::GlobalIndexType> globalDofIndices;
    const auto matrixFreeGlobalRhs =
        interpretLocalVectorIntoCamelliaGlobalOrder(camelliaSolution->mesh(),
                                                    matrixFreeLocalRhs,
                                                    localDofs,
                                                    globalDofIndices);

    const int globalDofs = static_cast<int>(globalDofIndices.size());

    const auto camelliaRhs =
        extractCamelliaRhsVectorInGlobalOrder(camelliaSolution, globalDofIndices);

    return compareDenseVectors(matrixFreeGlobalRhs, camelliaRhs, localDofs, globalDofs);
  }
}


int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int returnCode = 0;
  try
  {
    int dim = 2;
    int cellsPerDim = 1;
    int numApply = 10;
    int numDiagonal = 10;
    int numSolve = 10;
    int numInverseIterations = 10;
    int inversePreconditionerDegree = 8;
    int inverseSpectralEstimateIterations = 8;
    int inverseLowerSpectralEstimateIterations = 0;
    int inverseDiagnosticRequireAssembledReference = 1;
    int diagnoseLocalPMultigridWrapperEquivalence = 0;
    int compareTpetraPowerEstimate = 0;
    int compareTpetraRowMatrixPowerEstimate = 0;
    int ifpack2ChebyshevSmoke = 0;
    int inverseTrueEigenDiagnostic = 0;
    int worksetSize = 0;
    int checkAssemblyAgreement = 0;
    int camelliaMatrixCompareFailOnMismatch = 1;
    int camelliaMatrixCompareUseSolutionCoeff = 0;
    long long maxAssembledEntries = 200000;
    Scalar massWeight = 1.0;
    Scalar stiffnessWeight = 1.0;
    Scalar inverseEigenRatio = 30.0;
    Scalar inverseLambdaMin = -1.0;
    Scalar inverseLambdaMax = -1.0;
    Scalar inversePowerSafetyFactor = 1.1;
    Scalar inverseLowerPowerSafetyFactor = 0.95;
    Scalar inverseTolerance = 1e-8;
    Scalar inverseMaxResidualReduction = -1.0;
    std::string inversePreconditionerMatrixSource = "auto";
    std::string inverseOperatorSource = "matrix-free";
    int inverseOperatorCheckSamples = 0;
    int printTermWeights = 1;
    Scalar pMultigridChebyshevSmootherLowerFraction = 1.0 / 30.0;
    Scalar pMultigridFineSmootherWeight = 1.0;
    // Match the assembled GMG p=2 study's effective Chebyshev smoother degree by default.
    int pMultigridChebyshevSmootherIterations = 8;
    int pMultigridJacobiSmootherIterations = 2;
    int pMultigridCoarseChebyshevIterations = 8;
    Scalar pMultigridCoarseChebyshevLowerFraction = 0.05;
    int matrixFreeGMGStageTrace = 0;
    std::string polyOrderText = "1:4";
    std::string operatorChoice = "mass-plus-stiffness";
    std::string mode = "apply";
    std::string basisFamily = "hgrad-hierarchical";
    std::string basisMode;
    std::string inverseDiagnosticAlgorithm = "richardson";
    std::string inverseSpectralEstimate = "assembled-gershgorin";
    std::string inverseLowerSpectralEstimate = "ratio";
    std::string inverseInitialGuess = "zero";
    std::string inverseRhs = "deterministic";

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("dim", &dim, "Tensor dimension, 1 through 7.");
    cmdp.setOption("polyOrder", &polyOrderText, "Polynomial order p or inclusive range pMin:pMax.");
    cmdp.setOption("cellsPerDim", &cellsPerDim, "Uniform tensor cell count per dimension.");
    cmdp.setOption("operator", &operatorChoice, "mass, stiffness, mass-plus-stiffness, h1-gram, vlasov-dt, vlasov-coordinate, vlasov-linear-x0-x1-d2, vlasov-coordinate-x1-d3, vlasov-graph-norm-1d1t2v, or vlasov-graph-norm-solution-coeff-1d1t2v.  h1-gram is the weighted mass-plus-stiffness Gram starter case; vlasov-dt is the narrow constant-coefficient HGRAD/HGRAD tensor-volume bridge; vlasov-coordinate exercises the single-component coefficient bridge; vlasov-linear-x0-x1-d2 exercises a two-term separable decomposition of -0.5*(x0+x1)*value(f)*d2(w); vlasov-coordinate-x1-d3 exercises the separable -0.75*x1*value(f)*d3(w) partial term; vlasov-graph-norm-1d1t2v expands a 4D Planar MITL graph-norm proxy for Chebyshev G^{-1} calibration; vlasov-graph-norm-solution-coeff-1d1t2v routes the same proxy through TensorFunctionEvaluationContext solution-coefficient evaluators.");
    cmdp.setOption("basisFamily", &basisFamily,
                   "synthetic, hgrad, hgrad-hierarchical, or hgrad-nodal. "
                   "hgrad is an alias for hgrad-hierarchical. "
                   "hgrad-hierarchical uses Intrepid2::HierarchicalBasisFamily; "
                   "hgrad-nodal uses Intrepid2::DerivedNodalBasisFamily. "
                   "Both HGRAD options use Intrepid2 hypercube HGRAD tensor-product cubature and affine geometry for dim=1..7.");
    cmdp.setOption("basisMode", &basisMode,
                   "Deprecated alias for --basisFamily. If supplied, overrides --basisFamily for backward compatibility.");
    cmdp.setOption("mode", &mode,
                   "apply, assemble, assemble2, assembly-compare, diagonal, compare, factor, "
                   "inverse-diagnostic, tpetra-operator-compare, tpetra-rowmatrix-compare, "
                   "camellia-vlasov-matrix-compare, or camellia-vlasov-rhs-compare.");
    cmdp.setOption("numApply", &numApply, "Number of repeated applies used for steady-state timing in apply/compare modes.");
    cmdp.setOption("numDiagonal", &numDiagonal, "Number of repeated diagonal extractions used for steady-state timing in diagonal mode.");
    cmdp.setOption("numSolve", &numSolve, "Number of repeated local triangular solves used for average timing in factor mode.");
    cmdp.setOption("numInverseIterations", &numInverseIterations, "Number of local Richardson/Chebyshev iterations, p-multigrid V-cycles, or outer GMRES iterations used in inverse-diagnostic mode.");
    cmdp.setOption("inversePreconditionerDegree", &inversePreconditionerDegree, "Chebyshev polynomial degree for --inverseDiagnosticAlgorithm=gmres-chebyshev.");
    cmdp.setOption("pMultigridChebyshevSmootherIterations", &pMultigridChebyshevSmootherIterations, "Chebyshev pre/post smoother iterations for --inverseDiagnosticAlgorithm=pmultigrid-chebyshev.");
    cmdp.setOption("pMultigridChebyshevSmootherLowerFraction", &pMultigridChebyshevSmootherLowerFraction, "Lower endpoint as a fraction of lambdaMax for high-frequency Chebyshev p-multigrid smoothing.");
    cmdp.setOption("pMultigridFineSmootherWeight", &pMultigridFineSmootherWeight, "Scalar multiplier applied to each fine-level p-multigrid smoother correction; default 1.0 matches GMGOperator's default smoother weight and preserves the current matrix-free diagnostic behavior.");
    cmdp.setOption("pMultigridJacobiSmootherIterations", &pMultigridJacobiSmootherIterations, "Diagonal Jacobi pre/post smoother iterations per level for --inverseDiagnosticAlgorithm=pmultigrid-jacobi-chebyshev.");
    cmdp.setOption("pMultigridCoarseChebyshevIterations", &pMultigridCoarseChebyshevIterations, "Chebyshev iterations used as the coarsest p-multigrid solver for --inverseDiagnosticAlgorithm=pmultigrid-jacobi-chebyshev.");
    cmdp.setOption("pMultigridCoarseChebyshevLowerFraction", &pMultigridCoarseChebyshevLowerFraction, "Coarse Chebyshev lower endpoint as a fraction of the coarse diagonal-Jacobi lambdaMax estimate.");
    cmdp.setOption("diagnoseLocalPMultigridWrapperEquivalence", &diagnoseLocalPMultigridWrapperEquivalence, "When nonzero, compare direct local pMG apply with MatrixFreeGMGApplyInversePreconditioner wrapping the same local pMG operator.");
    cmdp.setOption("matrixFreeGMGStageTrace", &matrixFreeGMGStageTrace, "When nonzero, emit explicit per-level MatrixFreeGMGPreconditioner V-cycle stage norm diagnostics for GMRES p-multigrid inverse diagnostics.");
    cmdp.setOption("inverseDiagnosticAlgorithm", &inverseDiagnosticAlgorithm,
                   "richardson, chebyshev, gmres-chebyshev, gmres-jacobi, matrix-free-gmg-chebyshev-smoother, matrix-free-gmg-jacobi-smoother, gmres-pmultigrid-chebyshev, gmres-pmultigrid-jacobi-chebyshev, pmultigrid-chebyshev, or pmultigrid-jacobi-chebyshev inverse diagnostic recurrence.");
    cmdp.setOption("inverseSpectralEstimate", &inverseSpectralEstimate, "assembled-gershgorin, true-eigen, power, cell-power, or user spectral upper-estimate path for inverse diagnostics.");
    cmdp.setOption("inverseSpectralEstimateIterations", &inverseSpectralEstimateIterations, "Iterations for the matrix-free Ifpack2-style D^{-1}A spectral-radius power estimate.");
    cmdp.setOption("inversePowerSafetyFactor", &inversePowerSafetyFactor, "Safety factor applied to matrix-free power or true-eigen lambdaMax estimates before Chebyshev use.");
    cmdp.setOption("inverseLowerSpectralEstimate", &inverseLowerSpectralEstimate, "ratio, true-eigen, true-eigen-safe, user, or assembled-inverse-iteration lower spectral estimate for inverse diagnostics; positive --inverseLambdaMin overrides this choice.");
    cmdp.setOption("inverseLowerSpectralEstimateIterations", &inverseLowerSpectralEstimateIterations, "Iterations for assembled inverse-iteration lower spectral estimate; <=0 reuses --inverseSpectralEstimateIterations.");
    cmdp.setOption("inverseLowerPowerSafetyFactor", &inverseLowerPowerSafetyFactor, "Safety factor applied to assembled inverse-iteration lambdaMin estimates before Chebyshev use.");
    cmdp.setOption("inverseDiagnosticRequireAssembledReference", &inverseDiagnosticRequireAssembledReference, "Set to 0 to allow power-estimated inverse diagnostics without assembled reference/oracle matrices.");
    cmdp.setOption("compareTpetraPowerEstimate", &compareTpetraPowerEstimate, "Set to 1 in inverse-diagnostic power mode to compare direct and local-Tpetra-operator power estimates.");
    cmdp.setOption("compareTpetraRowMatrixPowerEstimate", &compareTpetraRowMatrixPowerEstimate, "Set to 1 in inverse-diagnostic power mode to compare direct and local-flat RowMatrix power estimates.");
    cmdp.setOption("ifpack2ChebyshevSmoke", &ifpack2ChebyshevSmoke, "Set to 1 in inverse-diagnostic Chebyshev mode to compare Ifpack2::Chebyshev on the local-flat RowMatrix against the host recurrence.");
    cmdp.setOption("inverseTrueEigenDiagnostic", &inverseTrueEigenDiagnostic, "Set to 1 to assemble each local diagonally preconditioned block and compute true eigenvalue diagnostics with LAPACK; requires assembled reference.");
    cmdp.setOption("inverseEigenRatio", &inverseEigenRatio, "Chebyshev lambdaMax/lambdaMin ratio used when --inverseLambdaMin is not positive.");
    cmdp.setOption("inverseLambdaMin", &inverseLambdaMin, "Optional positive user override for the Chebyshev/Richardson lower spectral bound.");
    cmdp.setOption("inverseLambdaMax", &inverseLambdaMax, "Optional positive user override for the Chebyshev/Richardson upper spectral bound.");
    cmdp.setOption("inverseInitialGuess", &inverseInitialGuess, "zero or diagonal initial guess for inverse diagnostics.");
    cmdp.setOption("inverseRhs", &inverseRhs, "deterministic or constant-test-function.  constant-test-function matches RHS::addTerm(1*w) in VlasovGramGMGStudyDriver.");
    cmdp.setOption("inverseTolerance", &inverseTolerance, "Relative residual-reduction tolerance reported by inverse diagnostics; fixed iteration count is unchanged.");
    cmdp.setOption("inverseMaxResidualReduction", &inverseMaxResidualReduction, "Optional pass/fail ceiling for inverse-diagnostic final/initial residual reduction; negative disables the check.");
    cmdp.setOption("inverseOperatorCheckSamples", &inverseOperatorCheckSamples, "Deterministic vector samples for assembled-vs-matrix-free action, symmetry, and positivity checks in inverse diagnostics; 0 disables.");
    cmdp.setOption("inversePreconditionerMatrixSource",
                   &inversePreconditionerMatrixSource,
                   "auto, assembled-crs, or matrix-free-rowmatrix. Controls the RowMatrix supplied to Ifpack2 in GMRES inverse diagnostics.");
    cmdp.setOption("inverseOperatorSource",
                   &inverseOperatorSource,
                   "matrix-free or assembled-crs. Controls the operator supplied to Belos in GMRES inverse diagnostics.");
    cmdp.setOption("massWeight", &massWeight, "Scalar coefficient alpha for mass or h1-gram value-value terms.");
    cmdp.setOption("stiffnessWeight", &stiffnessWeight, "Scalar coefficient beta for stiffness or h1-gram grad-grad terms.");
    cmdp.setOption("worksetSize", &worksetSize, "PAMatrix apply workset size; <=0 uses all cells.");
    cmdp.setOption("checkAssemblyAgreement", &checkAssemblyAgreement, "Set to 1 in assembly-compare mode to compare assemble2() entries against standard assemble().");
    cmdp.setOption("camelliaMatrixCompareFailOnMismatch",
                   &camelliaMatrixCompareFailOnMismatch,
                   "Set to 0 to make Camellia Vlasov matrix/RHS comparison modes report mismatches without returning failure.");
    cmdp.setOption("camelliaMatrixCompareUseSolutionCoeff",
                   &camelliaMatrixCompareUseSolutionCoeff,
                   "Set to 1 to compare the solution-coefficient Vlasov graph-norm PAMatrix/RHS path against the same assembled Camellia graph norm.");
    cmdp.setOption("maxAssembledEntries", &maxAssembledEntries, "Maximum local assembled entries allowed in assemble/assemble2/compare/factor modes.");
    cmdp.setOption("printTermWeights", &printTermWeights, "Set to 0 to omit the long termWeights field from TENSOR_POISSON_PAMATRIX_RESULT rows.");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      Camellia::finalize(false);
      Kokkos::finalize();
      return -1;
    }

    if ((dim < 1) || (dim > 7)) throw std::invalid_argument("--dim must be in [1,7]");
    if (cellsPerDim < 1) throw std::invalid_argument("--cellsPerDim must be positive");
    if (numApply < 1) throw std::invalid_argument("--numApply must be positive");
    if (numDiagonal < 1) throw std::invalid_argument("--numDiagonal must be positive");
    if (numSolve < 1) throw std::invalid_argument("--numSolve must be positive");
    if (numInverseIterations < 1) throw std::invalid_argument("--numInverseIterations must be positive");
    if (pMultigridChebyshevSmootherIterations < 1) throw std::invalid_argument("--pMultigridChebyshevSmootherIterations must be positive");
    if (pMultigridJacobiSmootherIterations < 1) throw std::invalid_argument("--pMultigridJacobiSmootherIterations must be positive");
    if (pMultigridCoarseChebyshevIterations < 1) throw std::invalid_argument("--pMultigridCoarseChebyshevIterations must be positive");
    if ((pMultigridCoarseChebyshevLowerFraction <= 0.0) || (pMultigridCoarseChebyshevLowerFraction >= 1.0))
      throw std::invalid_argument("--pMultigridCoarseChebyshevLowerFraction must be in (0,1)");
    const bool inverseDiagnosticNeedsSpectralEstimate = (inverseDiagnosticAlgorithm == "chebyshev") ||
                                                        (inverseDiagnosticAlgorithm == "gmres-chebyshev") ||
                                                        (inverseDiagnosticAlgorithm == "matrix-free-gmg-chebyshev-smoother") ||
                                                        (inverseDiagnosticAlgorithm == "gmres-pmultigrid-chebyshev") ||
                                                        (inverseDiagnosticAlgorithm == "gmres-pmultigrid-jacobi-chebyshev") ||
                                                        (inverseDiagnosticAlgorithm == "pmultigrid-chebyshev") ||
                                                        (inverseDiagnosticAlgorithm == "pmultigrid-jacobi-chebyshev");
    if (inverseDiagnosticNeedsSpectralEstimate && (inverseSpectralEstimateIterations < 1)) throw std::invalid_argument("--inverseSpectralEstimateIterations must be positive");
    if (!inverseDiagnosticNeedsSpectralEstimate && (inverseSpectralEstimateIterations < 1)) inverseSpectralEstimateIterations = 1;
    if (inverseLowerSpectralEstimateIterations <= 0) inverseLowerSpectralEstimateIterations = inverseSpectralEstimateIterations;
    if (inverseLowerSpectralEstimateIterations < 1) throw std::invalid_argument("--inverseLowerSpectralEstimateIterations must be positive after defaulting");
    if (!std::isfinite(inverseLowerPowerSafetyFactor) || (inverseLowerPowerSafetyFactor <= 0.0) || (inverseLowerPowerSafetyFactor > 1.0))
      throw std::invalid_argument("--inverseLowerPowerSafetyFactor must be in (0,1]");
    if ((checkAssemblyAgreement != 0) && (checkAssemblyAgreement != 1)) throw std::invalid_argument("--checkAssemblyAgreement must be 0 or 1");
    if ((camelliaMatrixCompareFailOnMismatch != 0) && (camelliaMatrixCompareFailOnMismatch != 1)) throw std::invalid_argument("--camelliaMatrixCompareFailOnMismatch must be 0 or 1");
    if ((printTermWeights != 0) && (printTermWeights != 1)) throw std::invalid_argument("--printTermWeights must be 0 or 1");
    if ((inverseDiagnosticRequireAssembledReference != 0) && (inverseDiagnosticRequireAssembledReference != 1)) throw std::invalid_argument("--inverseDiagnosticRequireAssembledReference must be 0 or 1");
    if ((compareTpetraPowerEstimate != 0) && (compareTpetraPowerEstimate != 1)) throw std::invalid_argument("--compareTpetraPowerEstimate must be 0 or 1");
    if ((compareTpetraRowMatrixPowerEstimate != 0) && (compareTpetraRowMatrixPowerEstimate != 1)) throw std::invalid_argument("--compareTpetraRowMatrixPowerEstimate must be 0 or 1");
    if ((ifpack2ChebyshevSmoke != 0) && (ifpack2ChebyshevSmoke != 1)) throw std::invalid_argument("--ifpack2ChebyshevSmoke must be 0 or 1");

    if (!basisMode.empty()) basisFamily = basisMode;
    basisMode = basisFamily;

    if ((mode != "apply") && (mode != "assemble") && (mode != "assemble2") &&
        (mode != "assembly-compare") && (mode != "diagonal") &&
        (mode != "compare") && (mode != "factor") &&
        (mode != "inverse-diagnostic") &&
        (mode != "tpetra-operator-compare") &&
        (mode != "tpetra-rowmatrix-compare") &&
        (mode != "camellia-vlasov-matrix-compare") &&
        (mode != "camellia-vlasov-rhs-compare"))
    {
      throw std::invalid_argument(
          "--mode must be apply, assemble, assemble2, assembly-compare, diagonal, "
          "compare, factor, inverse-diagnostic, tpetra-operator-compare, "
          "tpetra-rowmatrix-compare, camellia-vlasov-matrix-compare, or "
          "camellia-vlasov-rhs-compare");
    }
    const bool isCamelliaVlasovComparisonMode =
        (mode == "camellia-vlasov-matrix-compare") ||
        (mode == "camellia-vlasov-rhs-compare");

    if (isCamelliaVlasovComparisonMode &&
        ((dim != 4) || (cellsPerDim != 1) || !basisModeIsHGRAD(basisMode) ||
         (std::abs(stiffnessWeight - 1.0) > 1e-15)))
    {
      throw std::invalid_argument(
          "--mode=camellia-vlasov-matrix-compare and "
          "--mode=camellia-vlasov-rhs-compare require "
          "--dim=4 --cellsPerDim=1 "
          "--basisFamily=hgrad, hgrad-hierarchical, or hgrad-nodal "
          "--stiffnessWeight=1");
    }
    if ((basisMode != "synthetic") && !basisModeIsHGRAD(basisMode))
    {
      throw std::invalid_argument(
          "--basisFamily must be synthetic, hgrad, hgrad-hierarchical, or hgrad-nodal");
    }
    if ((basisMode == "hgrad-nodal") &&
        inverseAlgorithmUsesPMultigrid(inverseDiagnosticAlgorithm))
    {
      throw std::invalid_argument(
          "--basisFamily=hgrad-nodal is not yet supported for p-multigrid inverse "
          "diagnostics because nodal p-prolongation/restriction has not been implemented");
    }
    if (((operatorChoice == "vlasov-dt") || (operatorChoice == "vlasov-coordinate") || (operatorChoice == "vlasov-linear-x0-x1-d2") || (operatorChoice == "vlasov-coordinate-x1-d3") || (operatorChoice == "vlasov-graph-norm-1d1t2v") || (operatorChoice == "vlasov-graph-norm-solution-coeff-1d1t2v")) && !basisModeIsHGRAD(basisMode))
      throw std::invalid_argument(
          "--operator=vlasov-dt, --operator=vlasov-coordinate, "
          "--operator=vlasov-linear-x0-x1-d2, --operator=vlasov-coordinate-x1-d3, "
          "--operator=vlasov-graph-norm-1d1t2v, and "
          "--operator=vlasov-graph-norm-solution-coeff-1d1t2v require "
          "--basisFamily=hgrad, hgrad-hierarchical, or hgrad-nodal");
    if ((inverseDiagnosticAlgorithm != "richardson") && (inverseDiagnosticAlgorithm != "chebyshev") &&
        (inverseDiagnosticAlgorithm != "gmres-chebyshev") &&
        (inverseDiagnosticAlgorithm != "gmres-jacobi") &&
        (inverseDiagnosticAlgorithm != "matrix-free-gmg-chebyshev-smoother") &&
        (inverseDiagnosticAlgorithm != "matrix-free-gmg-jacobi-smoother") &&
        (inverseDiagnosticAlgorithm != "gmres-pmultigrid-chebyshev") &&
        (inverseDiagnosticAlgorithm != "gmres-pmultigrid-jacobi-chebyshev") &&
        (inverseDiagnosticAlgorithm != "pmultigrid-chebyshev") &&
        (inverseDiagnosticAlgorithm != "pmultigrid-jacobi-chebyshev"))
      throw std::invalid_argument("--inverseDiagnosticAlgorithm must be richardson, chebyshev, gmres-chebyshev, gmres-jacobi, matrix-free-gmg-chebyshev-smoother, matrix-free-gmg-jacobi-smoother, gmres-pmultigrid-chebyshev, gmres-pmultigrid-jacobi-chebyshev, pmultigrid-chebyshev, or pmultigrid-jacobi-chebyshev");
    if ((inverseLowerSpectralEstimate != "ratio") && (inverseLowerSpectralEstimate != "true-eigen") && (inverseLowerSpectralEstimate != "true-eigen-safe") && (inverseLowerSpectralEstimate != "user") && (inverseLowerSpectralEstimate != "assembled-inverse-iteration"))
      throw std::invalid_argument("--inverseLowerSpectralEstimate must be ratio, true-eigen, true-eigen-safe, user, or assembled-inverse-iteration");
    if ((inverseInitialGuess != "zero") && (inverseInitialGuess != "diagonal"))
      throw std::invalid_argument("--inverseInitialGuess must be zero or diagonal");
    if ((inverseRhs != "deterministic") && (inverseRhs != "constant-test-function"))
      throw std::invalid_argument("--inverseRhs must be deterministic or constant-test-function");
    if ((ifpack2ChebyshevSmoke != 0) && ((mode != "inverse-diagnostic") || (inverseDiagnosticAlgorithm != "chebyshev")))
      throw std::invalid_argument("--ifpack2ChebyshevSmoke requires --mode=inverse-diagnostic --inverseDiagnosticAlgorithm=chebyshev");
    if ((ifpack2ChebyshevSmoke != 0) && (inverseInitialGuess != "zero"))
      throw std::invalid_argument("--ifpack2ChebyshevSmoke currently supports --inverseInitialGuess=zero only");
    if ((ifpack2ChebyshevSmoke != 0) && (inverseSpectralEstimate == "cell-power"))
      throw std::invalid_argument("--ifpack2ChebyshevSmoke compares against scalar Ifpack2 parameters and is not compatible with per-cell Chebyshev endpoints");
    if ((matrixFreeGMGStageTrace != 0) &&
        ((mode != "inverse-diagnostic") ||
         ((inverseDiagnosticAlgorithm != "gmres-pmultigrid-chebyshev") &&
          (inverseDiagnosticAlgorithm != "gmres-pmultigrid-jacobi-chebyshev"))))
      throw std::invalid_argument("--matrixFreeGMGStageTrace requires --mode=inverse-diagnostic with --inverseDiagnosticAlgorithm=gmres-pmultigrid-chebyshev or gmres-pmultigrid-jacobi-chebyshev");
    if ((compareTpetraRowMatrixPowerEstimate != 0) && (inverseSpectralEstimate != "power"))
      throw std::invalid_argument("--compareTpetraRowMatrixPowerEstimate requires --inverseSpectralEstimate=power");
    if ((inverseSpectralEstimate != "assembled-gershgorin") && (inverseSpectralEstimate != "true-eigen") && (inverseSpectralEstimate != "power") && (inverseSpectralEstimate != "cell-power") && (inverseSpectralEstimate != "user"))
      throw std::invalid_argument("--inverseSpectralEstimate must be assembled-gershgorin, true-eigen, power, cell-power, or user");
    if ((inverseSpectralEstimate == "user") && (inverseLambdaMax <= 0.0))
      throw std::invalid_argument("--inverseSpectralEstimate=user requires positive --inverseLambdaMax");
    if (inverseDiagnosticNeedsSpectralEstimate && (inverseSpectralEstimateIterations < 1)) throw std::invalid_argument("--inverseSpectralEstimateIterations must be positive");
    if (inversePowerSafetyFactor < 1.0) throw std::invalid_argument("--inversePowerSafetyFactor must be at least 1");
    if (inverseEigenRatio <= 1.0) throw std::invalid_argument("--inverseEigenRatio must be greater than 1");
    if ((inverseLambdaMin > 0.0) && (inverseLambdaMax > 0.0) && (inverseLambdaMin >= inverseLambdaMax))
      throw std::invalid_argument("--inverseLambdaMin must be less than --inverseLambdaMax when both are positive");
    if (inverseTolerance <= 0.0) throw std::invalid_argument("--inverseTolerance must be positive");
    if ((inverseMaxResidualReduction == 0.0) || (inverseMaxResidualReduction < -1.0))
      throw std::invalid_argument("--inverseMaxResidualReduction must be positive, or negative to disable the check");
    // --basisFamily=hgrad, hgrad-hierarchical, or hgrad-nodal is supported for --dim=1..7; unsupported dimensions are rejected above and in makeHGRADOperatorTerms().

    const PolyOrderRange pRange = parsePolyOrderRange(polyOrderText);
    if ((pRange.first < 1) || (pRange.last < pRange.first)) throw std::invalid_argument("invalid --polyOrder range");

    const int cells = checkedProduct(std::vector<int>(dim, cellsPerDim));
    if (worksetSize <= 0) worksetSize = cells;
    worksetSize = std::min(worksetSize, cells);

    std::vector<double> logApplyPValues;
    std::vector<double> logApplyTimes;
    std::vector<double> logAssemblyPValues;
    std::vector<double> logAssemblyTimes;
    std::vector<double> logMassAssemblyPValues;
    std::vector<double> logMassAssemblyTimes;
    std::vector<double> logStiffnessAssemblyPValues;
    std::vector<double> logStiffnessAssemblyTimes;
    std::vector<double> logAssemble2PValues;
    std::vector<double> logAssemble2Times;
    std::vector<double> logMassAssemble2PValues;
    std::vector<double> logMassAssemble2Times;
    std::vector<double> logStiffnessAssemble2PValues;
    std::vector<double> logStiffnessAssemble2Times;
    std::vector<double> logDiagonalPValues;
    std::vector<double> logDiagonalTimes;

    std::cout << "TENSOR_POISSON_PAMATRIX_SCHEMA format=space_separated_key_value disabled_fields_omitted=1 disabled_values=-1,not_requested" << std::endl;
    for (int p=pRange.first; p<=pRange.last; p++)
    {
      Kokkos::Timer setupTimer;
      auto terms = makeOperatorTerms(dim, p, cellsPerDim, cells, operatorChoice, basisMode, massWeight, stiffnessWeight);
      const double setupSeconds = setupTimer.seconds();
      const std::string assemble2Path = ((mode == "assemble2") || (mode == "assembly-compare"))
                                      ? assemble2PathSummary(terms)
                                      : "not_requested";

      const int localDofs = checkedProduct(std::vector<int>(dim, p + 1));
      const int pointsPerDim = (basisMode == "hgrad") ? (p + 1) : (p + 2);
      const int localPoints = checkedProduct(std::vector<int>(dim, pointsPerDim));
      auto workspaces = TpetraAdapters::allocateWorkspaces(terms, worksetSize);
      Intrepid2::ScalarView<Scalar,DeviceType> input("TensorPoissonPAMatrix input", cells, localDofs);
      Intrepid2::ScalarView<Scalar,DeviceType> output("TensorPoissonPAMatrix output", cells, localDofs);
      fillInput(input);

      double firstApplySeconds = -1.0;
      double avgApplySeconds = -1.0;
      if ((mode == "apply") || (mode == "compare"))
      {
        firstApplySeconds = TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
        Kokkos::Timer applyTimer;
        for (int applyOrdinal=0; applyOrdinal<numApply; applyOrdinal++)
        {
          TpetraAdapters::applyTerms(terms, workspaces, input, output, worksetSize);
        }
        avgApplySeconds = applyTimer.seconds() / numApply;
      }

      AssemblyTiming assemblyTiming;
      if ((mode == "assemble") || (mode == "compare") || (mode == "assembly-compare"))
      {
        assemblyTiming = assembleTerms(terms, maxAssembledEntries);
      }
      AssemblyTiming assemble2Timing;
      if ((mode == "assemble2") || (mode == "assembly-compare"))
      {
        assemble2Timing = assemble2Terms(terms, maxAssembledEntries);
      }
      FactorTiming factorTiming;
      if (mode == "factor")
      {
        factorTiming = factorTerms(terms, maxAssembledEntries, numSolve);
        assemblyTiming = factorTiming.assembly;
      }
      RepeatedTiming diagonalTiming;
      std::string diagonalPath = "not_requested";
      if (mode == "diagonal")
      {
        diagonalTiming = diagonalTerms(terms, numDiagonal);
        diagonalPath = (basisMode == "synthetic") ? "sum_factorized_synthetic_scalar_tensor_terms"
                                               : "hgrad_pamatrix_direct_or_guarded_fallback";
      }
      InverseDiagnosticTiming inverseDiagnosticTiming;
      if (mode == "inverse-diagnostic")
      {
        inverseDiagnosticTiming = inverseDiagnosticTerms(terms, workspaces, worksetSize, cells, localDofs, dim, p,
                                                        maxAssembledEntries, inverseDiagnosticRequireAssembledReference, diagnoseLocalPMultigridWrapperEquivalence, numInverseIterations,
                                                        pMultigridChebyshevSmootherIterations, pMultigridChebyshevSmootherLowerFraction,
                                                        pMultigridFineSmootherWeight,
                                                        matrixFreeGMGStageTrace,
                                                        pMultigridJacobiSmootherIterations, pMultigridCoarseChebyshevIterations,
                                                        pMultigridCoarseChebyshevLowerFraction,
                                                        inverseDiagnosticAlgorithm, inverseEigenRatio, inverseLambdaMin, inverseLambdaMax,
                                                        inverseSpectralEstimate, inverseSpectralEstimateIterations, inversePowerSafetyFactor,
                                                        inverseLowerSpectralEstimate, inverseLowerSpectralEstimateIterations, inverseLowerPowerSafetyFactor,
                                                        inverseTrueEigenDiagnostic, inverseOperatorCheckSamples,
                                                        inversePreconditionerMatrixSource, inverseOperatorSource,
                                                        compareTpetraPowerEstimate, compareTpetraRowMatrixPowerEstimate,
                                                        ifpack2ChebyshevSmoke, inverseInitialGuess,
                                                        inverseRhs, inversePreconditionerDegree, inverseTolerance);
        if ((compareTpetraPowerEstimate != 0) &&
            (inverseDiagnosticTiming.tpetraPowerLambdaMaxAbsError > 1e-10) &&
            (inverseDiagnosticTiming.tpetraPowerLambdaMaxRelError > 1e-10)) returnCode = 2;
        if ((compareTpetraRowMatrixPowerEstimate != 0) &&
            (inverseDiagnosticTiming.tpetraRowMatrixPowerLambdaMaxAbsError > 1e-10) &&
            (inverseDiagnosticTiming.tpetraRowMatrixPowerLambdaMaxRelError > 1e-10)) returnCode = 2;
        if ((ifpack2ChebyshevSmoke != 0) &&
            (inverseDiagnosticTiming.ifpack2ChebyshevMaxAbsError > 1e-10) &&
            (inverseDiagnosticTiming.ifpack2ChebyshevMaxRelError > 1e-10)) returnCode = 2;
        if ((inverseMaxResidualReduction > 0.0) &&
            (inverseDiagnosticTiming.residualReduction > inverseMaxResidualReduction)) returnCode = 2;
      }

      const long long flopsPerCell = estimatedApplyFlopsPerCellPerTerm(dim, p) * static_cast<long long>(terms.size());
      const double achievedGFlops = (avgApplySeconds > 0.0) ? (flopsPerCell * static_cast<double>(cells) / avgApplySeconds / 1.0e9) : -1.0;
      const double assemble2OverAssembleRatio = ((assemble2Timing.totalSeconds > 0.0) && (assemblyTiming.totalSeconds > 0.0))
                                              ? assemble2Timing.totalSeconds / assemblyTiming.totalSeconds
                                              : -1.0;
      AssemblyComparison assemblyComparison;
      if ((mode == "assembly-compare") && (checkAssemblyAgreement != 0))
      {
        assemblyComparison = compareAssemble2AgainstAssemble(terms, maxAssembledEntries);
        if ((assemblyComparison.maxAbsError > 1e-10) && (assemblyComparison.maxRelError > 1e-10)) returnCode = 2;
      }
      TpetraOperatorComparison tpetraOperatorComparison;
      if (mode == "tpetra-operator-compare")
      {
        tpetraOperatorComparison = compareTpetraOperatorAgainstDirect(terms, workspaces, input, cells, localDofs, worksetSize, numApply);
        if ((tpetraOperatorComparison.maxAbsError > 1e-10) && (tpetraOperatorComparison.maxRelError > 1e-10)) returnCode = 2;
        if (tpetraOperatorComparison.inverseDiagonalMaxAbsError > 1e-12) returnCode = 2;
        if (tpetraOperatorComparison.domainRangeMapSame != 1) returnCode = 2;
      }
      if (mode == "tpetra-rowmatrix-compare")
      {
        tpetraOperatorComparison = compareTpetraRowMatrixAgainstDirect(terms, workspaces, input, cells, localDofs, worksetSize, numApply, maxAssembledEntries);
        if ((tpetraOperatorComparison.maxAbsError > 1e-10) && (tpetraOperatorComparison.maxRelError > 1e-10)) returnCode = 2;
        if (tpetraOperatorComparison.rowMatrixDiagonalMaxAbsError > 1e-12) returnCode = 2;
        if ((tpetraOperatorComparison.rowMatrixRowCopyChecked != 0) &&
            (tpetraOperatorComparison.rowMatrixRowCopyMaxAbsError > 1e-10)) returnCode = 2;
        if ((tpetraOperatorComparison.rowMatrixRowViewChecked != 0) &&
            (tpetraOperatorComparison.rowMatrixRowViewMaxAbsError > 1e-10)) returnCode = 2;
        if (tpetraOperatorComparison.domainRangeMapSame != 1) returnCode = 2;
        if (tpetraOperatorComparison.rowMatrixRowMapSame != 1) returnCode = 2;
        if (tpetraOperatorComparison.rowMatrixColMapSame != 1) returnCode = 2;
      }
      if (mode == "camellia-vlasov-matrix-compare")
      {
        const auto camelliaComparisons =
            runCamelliaVlasovMatrixComparison(p,
                                              massWeight,
                                              maxAssembledEntries,
                                              camelliaMatrixCompareUseSolutionCoeff != 0);
        if (camelliaMatrixCompareFailOnMismatch != 0)
        {
          for (const auto &comparison : camelliaComparisons)
          {
            if ((comparison.maxAbsError > 1e-10) && (comparison.maxRelError > 1e-10)) returnCode = 2;
          }
        }
      }
      if (mode == "camellia-vlasov-rhs-compare")
      {
        const auto rhsComparison =
            runCamelliaVlasovRhsComparison(p,
                                           massWeight,
                                           maxAssembledEntries,
                                           camelliaMatrixCompareUseSolutionCoeff != 0);

        const double tolerance = 1e-10;
        const int passed =
            ((rhsComparison.maxAbsError <= tolerance) ||
             (rhsComparison.maxRelError <= tolerance)) ? 1 : 0;

        std::cout << "CAMELLIA_VLASOV_RHS_COMPARISON"
                  << " p=" << p
                  << " localDofs=" << rhsComparison.localDofs
                  << " globalDofs=" << rhsComparison.globalDofs
                  << " maxAbsError=" << std::setprecision(16) << rhsComparison.maxAbsError
                  << " maxRelError=" << rhsComparison.maxRelError
                  << " l2Error=" << rhsComparison.l2Error
                  << " matrixFreeNorm=" << rhsComparison.matrixFreeNorm
                  << " camelliaNorm=" << rhsComparison.camelliaNorm
                  << " maxErrorOrdinal=" << rhsComparison.maxErrorOrdinal
                  << " matrixFreeAtMaxError=" << rhsComparison.matrixFreeAtMaxError
                  << " camelliaAtMaxError=" << rhsComparison.camelliaAtMaxError
                  << " tolerance=" << tolerance
                  << " passed=" << passed
                  << std::endl;

        if (camelliaMatrixCompareFailOnMismatch != 0 && passed == 0)
        {
          returnCode = 2;
        }
      }

      std::ostringstream resultLine;
      resultLine << "TENSOR_POISSON_PAMATRIX_RESULT"
                << " dim=" << dim
                << " p=" << p
                << " cells=" << cells
                << " localDofs=" << localDofs
                << " localPoints=" << localPoints
                << " terms=" << terms.size();
      if (printTermWeights != 0) resultLine << " termWeights=" << termSummary(terms);
      resultLine << " setupSeconds=" << std::setprecision(8) << setupSeconds
                << " firstApplySeconds=" << firstApplySeconds
                << " avgApplySeconds=" << avgApplySeconds
                << " assemblySeconds=" << assemblyTiming.totalSeconds
                << " massAssemblySeconds=" << assemblyTiming.massSeconds
                << " stiffnessAssemblySeconds=" << assemblyTiming.stiffnessSeconds
                << " assemble2Seconds=" << assemble2Timing.totalSeconds
                << " massAssemble2Seconds=" << assemble2Timing.massSeconds
                << " stiffnessAssemble2Seconds=" << assemble2Timing.stiffnessSeconds
                << " assemble2OverAssembleRatio=" << assemble2OverAssembleRatio
                << " assemblyAgreementMaxAbsError=" << assemblyComparison.maxAbsError
                << " assemblyAgreementMaxRelError=" << assemblyComparison.maxRelError
                << " tpetraOperatorMaxAbsError=" << tpetraOperatorComparison.maxAbsError
                << " tpetraOperatorMaxRelError=" << tpetraOperatorComparison.maxRelError
                << " tpetraInverseDiagonalMaxAbsError=" << tpetraOperatorComparison.inverseDiagonalMaxAbsError
                << " tpetraInverseDiagonalMin=" << tpetraOperatorComparison.inverseDiagonalMin
                << " tpetraInverseDiagonalMax=" << tpetraOperatorComparison.inverseDiagonalMax
                << " tpetraRowMatrixDiagonalMaxAbsError=" << tpetraOperatorComparison.rowMatrixDiagonalMaxAbsError
                << " tpetraRowMatrixDiagonalMin=" << tpetraOperatorComparison.rowMatrixDiagonalMin
                << " tpetraRowMatrixDiagonalMax=" << tpetraOperatorComparison.rowMatrixDiagonalMax
                << " tpetraRowMatrixRowCopyMaxAbsError=" << tpetraOperatorComparison.rowMatrixRowCopyMaxAbsError
                << " tpetraRowMatrixRowCopyChecked=" << tpetraOperatorComparison.rowMatrixRowCopyChecked
                << " tpetraRowMatrixRowViewMaxAbsError=" << tpetraOperatorComparison.rowMatrixRowViewMaxAbsError
                << " tpetraRowMatrixRowViewChecked=" << tpetraOperatorComparison.rowMatrixRowViewChecked
                << " tpetraRowMatrixDenseRowMaterializationEntries=" << tpetraOperatorComparison.rowMatrixDenseRowMaterializationEntries
                << " tpetraRowMatrixMaxDenseRowMaterializationEntries=" << tpetraOperatorComparison.rowMatrixMaxDenseRowMaterializationEntries
                << " tpetraOperatorFirstApplySeconds=" << tpetraOperatorComparison.firstApplySeconds
                << " tpetraOperatorAvgApplySeconds=" << tpetraOperatorComparison.avgApplySeconds
                << " tpetraOperatorNumVectors=" << tpetraOperatorComparison.numVectors
                << " tpetraOperatorLocalLength=" << tpetraOperatorComparison.localLength
                << " tpetraDomainRangeMapSame=" << tpetraOperatorComparison.domainRangeMapSame
                << " tpetraRowMatrixRowMapSame=" << tpetraOperatorComparison.rowMatrixRowMapSame
                << " tpetraRowMatrixColMapSame=" << tpetraOperatorComparison.rowMatrixColMapSame
                << " tpetraAdapterPath=" << tpetraOperatorComparison.adapterPath
                << " tpetraGlobalDofSemantics=" << tpetraOperatorComparison.globalDofSemantics
                << " tpetraIfpack2RowMatrix=" << tpetraOperatorComparison.ifpack2RowMatrix
                << " factorSeconds=" << factorTiming.factorSeconds
                << " avgSolveSeconds=" << factorTiming.solveSeconds
                << " maxSolveResidual=" << factorTiming.maxSolveResidual
                << " maxSolveRelativeResidual=" << factorTiming.maxSolveRelativeResidual
                << " firstDiagonalSeconds=" << diagonalTiming.firstSeconds
                << " avgDiagonalSeconds=" << diagonalTiming.averageSeconds
                << " avgInverseApplySeconds=" << inverseDiagnosticTiming.avgInverseApplySeconds
                << " inverseIterations=" << inverseDiagnosticTiming.inverseIterations
                << " initialInverseAbsError=" << inverseDiagnosticTiming.initialInverseAbsError
                << " initialInverseRelError=" << inverseDiagnosticTiming.initialInverseRelError
                << " maxInverseAbsError=" << inverseDiagnosticTiming.maxInverseAbsError
                << " maxInverseRelError=" << inverseDiagnosticTiming.maxInverseRelError
                << " initialResidualNorm=" << inverseDiagnosticTiming.initialResidualNorm
                << " finalResidualNorm=" << inverseDiagnosticTiming.finalResidualNorm
                << " residualReduction=" << inverseDiagnosticTiming.residualReduction
                << " inverseTolerance=" << inverseDiagnosticTiming.inverseTolerance
                << " inversePreconditionerDegree=" << inverseDiagnosticTiming.inversePreconditionerDegree
                << " inverseConverged=" << inverseDiagnosticTiming.inverseConverged
                << " inverseStoppedReason=" << inverseDiagnosticTiming.inverseStoppedReason
                << " lambdaMinEstimate=" << inverseDiagnosticTiming.lambdaMinEstimate
                << " lambdaMaxEstimate=" << inverseDiagnosticTiming.lambdaMaxEstimate
                << " assembledGershgorinLambdaMax=" << inverseDiagnosticTiming.assembledGershgorinLambdaMax
                << " powerRawLambdaMaxEstimate=" << inverseDiagnosticTiming.powerRawLambdaMaxEstimate
                << " tpetraPowerRawLambdaMaxEstimate=" << inverseDiagnosticTiming.tpetraPowerRawLambdaMaxEstimate
                << " tpetraPowerLambdaMaxAbsError=" << inverseDiagnosticTiming.tpetraPowerLambdaMaxAbsError
                << " tpetraPowerLambdaMaxRelError=" << inverseDiagnosticTiming.tpetraPowerLambdaMaxRelError
                << " tpetraRowMatrixPowerRawLambdaMaxEstimate=" << inverseDiagnosticTiming.tpetraRowMatrixPowerRawLambdaMaxEstimate
                << " tpetraRowMatrixPowerLambdaMaxAbsError=" << inverseDiagnosticTiming.tpetraRowMatrixPowerLambdaMaxAbsError
                << " tpetraRowMatrixPowerLambdaMaxRelError=" << inverseDiagnosticTiming.tpetraRowMatrixPowerLambdaMaxRelError
                << " compareTpetraRowMatrixPowerEstimate=" << inverseDiagnosticTiming.compareTpetraRowMatrixPowerEstimate
                << " lambdaMinRawEstimate=" << inverseDiagnosticTiming.lambdaMinRawEstimate
                << " lambdaMinSafetyFactor=" << inverseDiagnosticTiming.lambdaMinSafetyFactor
                << " lambdaMinEstimateOverRaw=" << inverseDiagnosticTiming.lambdaMinEstimateOverRaw
                << " compareTpetraPowerEstimate=" << inverseDiagnosticTiming.compareTpetraPowerEstimate
                << " ifpack2ChebyshevSmoke=" << inverseDiagnosticTiming.ifpack2ChebyshevSmoke
                << " ifpack2ChebyshevSeconds=" << inverseDiagnosticTiming.ifpack2ChebyshevSeconds
                << " ifpack2ChebyshevMaxAbsError=" << inverseDiagnosticTiming.ifpack2ChebyshevMaxAbsError
                << " ifpack2ChebyshevMaxRelError=" << inverseDiagnosticTiming.ifpack2ChebyshevMaxRelError
                << " ifpack2ChebyshevPath=" << inverseDiagnosticTiming.ifpack2ChebyshevPath
                << " lambdaMaxEstimateOverAssembledGershgorin=" << inverseDiagnosticTiming.lambdaMaxEstimateOverAssembledGershgorin
                << " inverseEigenRatio=" << inverseDiagnosticTiming.inverseEigenRatio
                << " spectralEstimateIterations=" << inverseDiagnosticTiming.spectralEstimateIterations
                << " spectralEstimateSeconds=" << inverseDiagnosticTiming.spectralEstimateSeconds
                << " lambdaMaxSafetyFactor=" << inverseDiagnosticTiming.lambdaMaxSafetyFactor
                << " maxSymmetryAbsError=" << inverseDiagnosticTiming.maxSymmetryAbsError
                << " minDiagonalEstimate=" << inverseDiagnosticTiming.minDiagonalEstimate
                << " maxDiagonalEstimate=" << inverseDiagnosticTiming.maxDiagonalEstimate
                << " trueEigenDiagnostic=" << inverseDiagnosticTiming.trueEigenDiagnostic
                << " trueLambdaMin=" << inverseDiagnosticTiming.trueLambdaMin
                << " trueLambdaMax=" << inverseDiagnosticTiming.trueLambdaMax
                << " trueCondition=" << inverseDiagnosticTiming.trueCondition
                << " trueLambdaMaxOverAssembledGershgorin=" << inverseDiagnosticTiming.trueLambdaMaxOverAssembledGershgorin
                << " trueLambdaMaxOverPowerRaw=" << inverseDiagnosticTiming.trueLambdaMaxOverPowerRaw
                << " trueLambdaMinOverLowerEstimate=" << inverseDiagnosticTiming.trueLambdaMinOverLowerEstimate
                << " trueEigenPath=" << inverseDiagnosticTiming.trueEigenPath
                << " pMultigridLevels=" << inverseDiagnosticTiming.pMultigridLevels
                << " pMultigridFineDofs=" << inverseDiagnosticTiming.pMultigridFineDofs
                << " pMultigridCoarseDofs=" << inverseDiagnosticTiming.pMultigridCoarseDofs
                << " pMultigridChebyshevSmootherIterations=" << inverseDiagnosticTiming.pMultigridChebyshevSmootherIterations
                << " pMultigridChebyshevSmootherLowerFraction=" << inverseDiagnosticTiming.pMultigridChebyshevSmootherLowerFraction
                << " pMultigridFineSmootherWeight=" << inverseDiagnosticTiming.pMultigridFineSmootherWeight
                << " pMultigridJacobiSmootherIterations=" << inverseDiagnosticTiming.pMultigridJacobiSmootherIterations
                << " pMultigridCoarseChebyshevIterations=" << inverseDiagnosticTiming.pMultigridCoarseChebyshevIterations
                << " pMultigridCoarseChebyshevLowerFraction=" << inverseDiagnosticTiming.pMultigridCoarseChebyshevLowerFraction
                << " pMultigridCoarseSolveCount=" << inverseDiagnosticTiming.pMultigridCoarseSolveCount
                << " pMultigridPath=" << inverseDiagnosticTiming.pMultigridPath
                << " estimatedApplyFlopsPerCell=" << flopsPerCell
                << " lambdaMinSource=" << inverseDiagnosticTiming.lambdaMinSource
                << " lambdaMaxSource=" << inverseDiagnosticTiming.lambdaMaxSource
                << " inverseLowerSpectralEstimate=" << inverseDiagnosticTiming.inverseLowerSpectralEstimate
                << " inverseLowerSpectralEstimateIterations=" << inverseLowerSpectralEstimateIterations
                << " inverseInitialGuess=" << inverseDiagnosticTiming.inverseInitialGuess
                << " inverseRhs=" << inverseDiagnosticTiming.inverseRhs
                << " achievedGFlops=" << achievedGFlops
                << " mode=" << mode
                << " operator=" << operatorChoice
                << " basisMode=" << basisMode
                << " diagonalPath=" << diagonalPath
                << " assemble2Path=" << assemble2Path
                << " inversePath=" << inverseDiagnosticTiming.inversePath
                << " spectralEstimatePath=" << inverseDiagnosticTiming.spectralEstimatePath;
      if (diagnoseLocalPMultigridWrapperEquivalence != 0)
        resultLine << " localPMultigridWrapperApplyInfNorm=" << inverseDiagnosticTiming.localPMultigridWrapperApplyInfNorm
                   << " matrixFreeGMGWrapperApplyInfNorm=" << inverseDiagnosticTiming.matrixFreeGMGWrapperApplyInfNorm
                   << " localPMultigridWrapperDifferenceInfNorm=" << inverseDiagnosticTiming.localPMultigridWrapperDifferenceInfNorm
                   << " localPMultigridWrapperRelativeDifferenceInfNorm=" << inverseDiagnosticTiming.localPMultigridWrapperRelativeDifferenceInfNorm;
      printCompactKeyValueLine(resultLine.str());

      const double logP = std::log(static_cast<double>(p + 1));
      if (avgApplySeconds > 0.0)
      {
        logApplyPValues.push_back(logP);
        logApplyTimes.push_back(std::log(avgApplySeconds));
      }
      if (assemblyTiming.totalSeconds > 0.0)
      {
        logAssemblyPValues.push_back(logP);
        logAssemblyTimes.push_back(std::log(assemblyTiming.totalSeconds));
      }
      if (assemblyTiming.massSeconds > 0.0)
      {
        logMassAssemblyPValues.push_back(logP);
        logMassAssemblyTimes.push_back(std::log(assemblyTiming.massSeconds));
      }
      if (assemblyTiming.stiffnessSeconds > 0.0)
      {
        logStiffnessAssemblyPValues.push_back(logP);
        logStiffnessAssemblyTimes.push_back(std::log(assemblyTiming.stiffnessSeconds));
      }
      if (assemble2Timing.totalSeconds > 0.0)
      {
        logAssemble2PValues.push_back(logP);
        logAssemble2Times.push_back(std::log(assemble2Timing.totalSeconds));
      }
      if (assemble2Timing.massSeconds > 0.0)
      {
        logMassAssemble2PValues.push_back(logP);
        logMassAssemble2Times.push_back(std::log(assemble2Timing.massSeconds));
      }
      if (assemble2Timing.stiffnessSeconds > 0.0)
      {
        logStiffnessAssemble2PValues.push_back(logP);
        logStiffnessAssemble2Times.push_back(std::log(assemble2Timing.stiffnessSeconds));
      }
      if (diagonalTiming.averageSeconds > 0.0)
      {
        logDiagonalPValues.push_back(logP);
        logDiagonalTimes.push_back(std::log(diagonalTiming.averageSeconds));
      }

      if (mode == "compare")
      {
        const bool ok = compareApplyAgainstAssembled(terms, workspaces, input, output, worksetSize, maxAssembledEntries);
        if (!ok) returnCode = 2;
      }
    }

    if (logApplyPValues.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE applySlope=" << leastSquaresSlope(logApplyPValues, logApplyTimes)
                << " expectedApplySlope=" << (dim + 1)
                << " samples=" << logApplyTimes.size()
                << " note=uses_log_p_plus_1" << std::endl;
    }
    if (logAssemblyTimes.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE assemblySlope=" << leastSquaresSlope(logAssemblyPValues, logAssemblyTimes)
                << " expectedStandardAssembleSlopeUpper=" << (2 * dim + 1)
                << " expectedSeparableAssembleSlope=" << (2 * dim)
                << " denseFallbackSlope=" << (3 * dim)
                << " samples=" << logAssemblyTimes.size()
                << " note=standard_Intrepid2_assemble_expected_sum_factorized_dense_is_guarded_fallback" << std::endl;
    }
    if (logMassAssemblyTimes.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE massAssemblySlope=" << leastSquaresSlope(logMassAssemblyPValues, logMassAssemblyTimes)
                << " expectedStandardAssembleSlopeUpper=" << (2 * dim + 1)
                << " expectedSeparableAssembleSlope=" << (2 * dim)
                << " denseFallbackSlope=" << (3 * dim)
                << " samples=" << logMassAssemblyTimes.size()
                << " note=term_separated_standard_assemble_timing" << std::endl;
    }
    if (logStiffnessAssemblyTimes.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE stiffnessAssemblySlope=" << leastSquaresSlope(logStiffnessAssemblyPValues, logStiffnessAssemblyTimes)
                << " expectedStandardAssembleSlopeUpper=" << (2 * dim + 1)
                << " expectedSeparableAssembleSlope=" << (2 * dim)
                << " denseFallbackSlope=" << (3 * dim)
                << " samples=" << logStiffnessAssemblyTimes.size()
                << " note=term_separated_standard_assemble_timing" << std::endl;
    }
    if (logAssemble2Times.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE assemble2Slope=" << leastSquaresSlope(logAssemble2PValues, logAssemble2Times)
                << " expectedColumnApplyAssemblySlope=" << (2 * dim + 1)
                << " samples=" << logAssemble2Times.size()
                << " note=assemble2_path_reported_per_result_row" << std::endl;
    }
    if (logMassAssemble2Times.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE massAssemble2Slope=" << leastSquaresSlope(logMassAssemble2PValues, logMassAssemble2Times)
                << " expectedColumnApplyAssemblySlope=" << (2 * dim + 1)
                << " samples=" << logMassAssemble2Times.size()
                << " note=term_separated_assemble2_timing_path_reported_per_result_row" << std::endl;
    }
    if (logStiffnessAssemble2Times.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE stiffnessAssemble2Slope=" << leastSquaresSlope(logStiffnessAssemble2PValues, logStiffnessAssemble2Times)
                << " expectedColumnApplyAssemblySlope=" << (2 * dim + 1)
                << " samples=" << logStiffnessAssemble2Times.size()
                << " note=term_separated_assemble2_timing_path_reported_per_result_row" << std::endl;
    }
    if (logDiagonalTimes.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE diagonalSlope=" << leastSquaresSlope(logDiagonalPValues, logDiagonalTimes)
                << " expectedSumFactorizedDiagonalSlope=" << (dim + 1)
                << " samples=" << logDiagonalTimes.size()
                << " note=uses_log_p_plus_1_and_excludes_diagonal_storage_allocation" << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "TensorPoissonPAMatrixDriver failed: " << e.what() << std::endl;
    returnCode = 1;
  }
  Camellia::finalize(false);
  Kokkos::finalize();
  return returnCode;
}
