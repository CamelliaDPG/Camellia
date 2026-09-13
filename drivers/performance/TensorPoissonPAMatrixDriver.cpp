//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Intrepid2_PAMatrix.hpp"
#include "Intrepid2_PAMatrixDefAdditions.hpp"

#include <Intrepid2_CellGeometryTestUtils.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_NodalBasisFamily.hpp>

#include "PAMatrixAssembly.hpp"

#include "Kokkos_Core.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using Scalar = double;
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using View2D = Kokkos::View<Scalar**,DeviceType>;
  using PAMatrix = Intrepid2::PAMatrix<DeviceType,Scalar>;
  using TpetraMap = Tpetra::Map<>;
  using TpetraMultiVector = Tpetra::MultiVector<Scalar>;
  using TpetraOperator = Tpetra::Operator<Scalar>;

  struct OperatorTerm
  {
    PAMatrix matrix;
    Scalar weight = 1.0;
    std::string label;
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

  template<int spaceDim>
  std::vector<OperatorTerm> makeHGRADOperatorTermsForDim(const int p, const int cellsPerDim, const std::string &operatorChoice,
                                                         const Scalar massWeight, const Scalar stiffnessWeight)
  {
    using BasisFamily = Intrepid2::DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;
    Kokkos::Array<Scalar,spaceDim> domainExtents;
    Kokkos::Array<int,spaceDim> gridCellCounts;
    for (int d=0; d<spaceDim; d++)
    {
      domainExtents[d] = 1.0;
      gridCellCounts[d] = cellsPerDim;
    }
    auto geometry = Intrepid2::uniformCartesianMesh<Scalar,spaceDim,DeviceType>(domainExtents, gridCellCounts);

    std::vector<OperatorTerm> terms;
    if (operatorIncludesMass(operatorChoice))
    {
      terms.push_back({constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
          geometry,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE,
          p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_VALUE), massWeight, "mass"});
    }
    if (operatorIncludesStiffness(operatorChoice))
    {
      if (operatorChoice == "h1-gram")
      {
        for (int d=0; d<spaceDim; d++)
        {
          auto vectorWeight = Teuchos::rcp(new Kokkos::Array<Scalar,spaceDim>());
          for (int e=0; e<spaceDim; e++) (*vectorWeight)[e] = (e == d) ? Scalar(1.0) : Scalar(0.0);
          terms.push_back({constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
              geometry,
              p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight,
              p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD, vectorWeight), stiffnessWeight, "stiffness" + std::to_string(d)});
        }
      }
      else
      {
        terms.push_back({constructPAMatrix<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
            geometry,
            p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD,
            p, Intrepid2::FUNCTION_SPACE_HGRAD, Intrepid2::OPERATOR_GRAD), stiffnessWeight, "stiffness"});
      }
    }
    if (terms.empty())
    {
      throw std::invalid_argument("--operator must be mass, stiffness, mass-plus-stiffness, or h1-gram");
    }
    return terms;
  }

  std::vector<OperatorTerm> makeHGRADOperatorTerms(const int dim, const int p, const int cellsPerDim, const std::string &operatorChoice,
                                                    const Scalar massWeight, const Scalar stiffnessWeight)
  {
    switch (dim)
    {
      case 1: return makeHGRADOperatorTermsForDim<1>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight);
      case 2: return makeHGRADOperatorTermsForDim<2>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight);
      case 3: return makeHGRADOperatorTermsForDim<3>(p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight);
      default:
        throw std::invalid_argument("--basisMode=hgrad currently supports --dim=1,2,3; use --basisMode=synthetic for higher-dimensional contraction stress tests");
    }
  }

  std::vector<OperatorTerm> makeOperatorTerms(const int dim, const int p, const int cellsPerDim, const int cells,
                                              const std::string &operatorChoice, const std::string &basisMode,
                                              const Scalar massWeight, const Scalar stiffnessWeight)
  {
    if (basisMode == "synthetic") return makeSyntheticOperatorTerms(dim, p, cells, operatorChoice, massWeight, stiffnessWeight);
    if (basisMode == "hgrad")     return makeHGRADOperatorTerms(dim, p, cellsPerDim, operatorChoice, massWeight, stiffnessWeight);
    throw std::invalid_argument("--basisMode must be synthetic or hgrad");
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

  using WorkspaceView = Kokkos::View<Scalar*,DeviceType>;

  std::vector<WorkspaceView> allocateWorkspaces(const std::vector<OperatorTerm> &terms, const int worksetSize)
  {
    std::vector<WorkspaceView> workspaces;
    workspaces.reserve(terms.size());
    for (const auto &term : terms)
    {
      workspaces.push_back(term.matrix.allocateWorkspace(worksetSize));
    }
    return workspaces;
  }

  bool allWeightsAreOne(const std::vector<OperatorTerm> &terms)
  {
    for (const auto &term : terms)
    {
      if (term.weight != Scalar(1.0)) return false;
    }
    return true;
  }

  void addScaledOutput(Intrepid2::ScalarView<Scalar,DeviceType> &output,
                       const Intrepid2::ScalarView<Scalar,DeviceType> &termOutput,
                       const Scalar weight)
  {
    auto policy = Kokkos::MDRangePolicy<typename DeviceType::execution_space,Kokkos::Rank<2>>(
        {0,0}, {output.extent_int(0), output.extent_int(1)});
    Kokkos::parallel_for("TensorPoissonPAMatrix add weighted term", policy,
        KOKKOS_LAMBDA(const int c, const int f)
        {
          output(c,f) += weight * termOutput(c,f);
        });
  }

  double applyTerms(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                    const Intrepid2::ScalarView<Scalar,DeviceType> &input,
                    Intrepid2::ScalarView<Scalar,DeviceType> &output, const int worksetSize)
  {
    Kokkos::Timer timer;
    if (allWeightsAreOne(terms))
    {
      bool sumInto = false;
      for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
      {
        terms[termOrdinal].matrix.apply(output, input, workspaces[termOrdinal], sumInto, worksetSize);
        sumInto = true;
      }
    }
    else
    {
      Kokkos::deep_copy(output, Scalar(0.0));
      Intrepid2::ScalarView<Scalar,DeviceType> termOutput("TensorPoissonPAMatrix weighted term output", output.extent_int(0), output.extent_int(1));
      for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
      {
        terms[termOrdinal].matrix.apply(termOutput, input, workspaces[termOrdinal], false, worksetSize);
        addScaledOutput(output, termOutput, terms[termOrdinal].weight);
      }
    }
    Kokkos::fence();
    return timer.seconds();
  }

  std::vector<Scalar> extractCombinedDiagonal(std::vector<OperatorTerm> &terms, const int cells, const int localDofs)
  {
    std::vector<Scalar> diagonal(static_cast<size_t>(cells * localDofs), Scalar(0.0));
    for (auto &term : terms)
    {
      auto diagonalData = term.matrix.allocateDiagonalStorage();
      term.matrix.extractDiagonal(diagonalData);
      Kokkos::fence();

      if (diagonalData.getUnderlyingViewRank() == 2)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            diagonal[static_cast<size_t>(c * localDofs + i)] += term.weight * diagonalHost(c,i);
      }
      else if (diagonalData.getUnderlyingViewRank() == 1)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<1>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            diagonal[static_cast<size_t>(c * localDofs + i)] += term.weight * diagonalHost(i);
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix diagonal storage rank");
      }
    }
    return diagonal;
  }

  class LocalPAMatrixTpetraOperator : public TpetraOperator
  {
    Teuchos::RCP<const TpetraMap> map_;
    mutable std::vector<OperatorTerm> *terms_;
    std::vector<WorkspaceView> workspaces_;
    int cells_ = 0;
    int localDofs_ = 0;
    int worksetSize_ = 0;

  public:
    LocalPAMatrixTpetraOperator(const Teuchos::RCP<const TpetraMap> &map, std::vector<OperatorTerm> &terms,
                                const std::vector<WorkspaceView> &workspaces, const int cells,
                                const int localDofs, const int worksetSize)
      : map_(map), terms_(&terms), workspaces_(workspaces), cells_(cells), localDofs_(localDofs), worksetSize_(worksetSize)
    {}

    Teuchos::RCP<const TpetraMap> getDomainMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getRangeMap() const override { return map_; }
    bool hasTransposeApply() const override { return false; }

    void fillInverseDiagonal(TpetraMultiVector &inverseDiagonal) const
    {
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if (inverseDiagonal.getLocalLength() != localLength)
        throw std::invalid_argument("LocalPAMatrixTpetraOperator inverse diagonal length mismatch");
      auto inverseDiagonalHost = inverseDiagonal.getLocalViewHost(Tpetra::Access::OverwriteAll);
      const auto diagonal = extractCombinedDiagonal(*terms_, cells_, localDofs_);
      for (size_t localOrdinal=0; localOrdinal<localLength; localOrdinal++)
      {
        if (diagonal[localOrdinal] <= std::numeric_limits<Scalar>::epsilon())
          throw std::runtime_error("LocalPAMatrixTpetraOperator found nonpositive diagonal entry");
        for (size_t vectorOrdinal=0; vectorOrdinal<inverseDiagonal.getNumVectors(); vectorOrdinal++)
          inverseDiagonalHost(localOrdinal, vectorOrdinal) = Scalar(1.0) / diagonal[localOrdinal];
      }
    }

    void apply(const TpetraMultiVector &X, TpetraMultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      if (mode != Teuchos::NO_TRANS) throw std::invalid_argument("LocalPAMatrixTpetraOperator does not support transpose apply");
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((X.getLocalLength() != localLength) || (Y.getLocalLength() != localLength))
        throw std::invalid_argument("LocalPAMatrixTpetraOperator local vector length mismatch");
      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalPAMatrixTpetraOperator multivector count mismatch");

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
      Intrepid2::ScalarView<Scalar,DeviceType> input("LocalPAMatrixTpetraOperator input", cells_, localDofs_);
      Intrepid2::ScalarView<Scalar,DeviceType> output("LocalPAMatrixTpetraOperator output", cells_, localDofs_);
      auto inputHost = Kokkos::create_mirror_view(input);

      for (size_t vectorOrdinal=0; vectorOrdinal<X.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
            inputHost(c,f) = xHost(static_cast<size_t>(c * localDofs_ + f), vectorOrdinal);
        Kokkos::deep_copy(input, inputHost);

        applyTerms(*terms_, workspaces_, input, output, worksetSize_);
        auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);

        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
          {
            const size_t localOrdinal = static_cast<size_t>(c * localDofs_ + f);
            yHost(localOrdinal, vectorOrdinal) = alpha * outputHost(c,f) + beta * yHost(localOrdinal, vectorOrdinal);
          }
      }
    }
  };

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
    double firstApplySeconds = -1.0;
    double avgApplySeconds = -1.0;
    int numVectors = 0;
    int localLength = 0;
    int domainRangeMapSame = 0;
    std::string adapterPath = "not_requested";
    std::string globalDofSemantics = "not_requested";
    std::string ifpack2RowMatrix = "not_requested";
  };

  struct InverseDiagnosticTiming
  {
    double avgInverseApplySeconds = -1.0;
    int inverseIterations = 0;
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
    double lambdaMinEstimate = -1.0;
    double lambdaMaxEstimate = -1.0;
    double assembledGershgorinLambdaMax = -1.0;
    double powerRawLambdaMaxEstimate = -1.0;
    double tpetraPowerRawLambdaMaxEstimate = -1.0;
    double tpetraPowerLambdaMaxAbsError = -1.0;
    double tpetraPowerLambdaMaxRelError = -1.0;
    double lambdaMaxEstimateOverAssembledGershgorin = -1.0;
    int compareTpetraPowerEstimate = 0;
    double inverseEigenRatio = -1.0;
    int spectralEstimateIterations = 0;
    double spectralEstimateSeconds = -1.0;
    double lambdaMaxSafetyFactor = -1.0;
    double maxSymmetryAbsError = -1.0;
    double minDiagonalEstimate = -1.0;
    double maxDiagonalEstimate = -1.0;
    std::string inversePath = "not_requested";
    std::string spectralEstimatePath = "not_requested";
    std::string lambdaMinSource = "not_requested";
    std::string lambdaMaxSource = "not_requested";
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

    if (sawUnitColumn && sawApplyFallback) return "mixed_unit_column_and_blocked_apply_fallback";
    if (sawUnitColumn) return "unit_column_fast_path";
    return "blocked_apply_fallback";
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
    LocalPAMatrixTpetraOperator op(map, terms, workspaces, cells, localDofs, worksetSize);
    comparison.domainRangeMapSame = (op.getDomainMap().getRawPtr() == op.getRangeMap().getRawPtr()) ? 1 : 0;
    TpetraMultiVector inverseDiagonal(map, 1);
    op.fillInverseDiagonal(inverseDiagonal);
    auto inverseDiagonalHost = inverseDiagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto expectedDiagonal = extractCombinedDiagonal(terms, cells, localDofs);
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
    op.apply(X, Y, Teuchos::NO_TRANS, alpha, beta);
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
      applyTerms(terms, workspaces, input, output, worksetSize);
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
      op.apply(X, Y, Teuchos::NO_TRANS, Scalar(1.0), Scalar(0.0));
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

    applyTerms(terms, workspaces, input, output, worksetSize);
    auto inputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input);
    auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    std::vector<Scalar> expected(cells * rows, 0.0);

    for (auto &term : terms)
    {
      auto matrixData = term.matrix.allocateMatrixStorage();
      term.matrix.assemble(matrixData);
      auto matrixHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixData.getUnderlyingView());
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
    std::cout << "COMPARE maxAbsError=" << std::setprecision(16) << maxError
              << " maxRelError=" << relError << std::endl;
    return (maxError <= 1e-10) || (relError <= 1e-10);
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

  InverseDiagnosticTiming inverseDiagnosticTerms(std::vector<OperatorTerm> &terms, const std::vector<WorkspaceView> &workspaces,
                                                 const int worksetSize, const int cells, const int rows,
                                                 const long long maxAssembledEntries, const int inverseDiagnosticRequireAssembledReference,
                                                 const int numInverseIterations, const std::string &inverseDiagnosticAlgorithm,
                                                 const Scalar inverseEigenRatio, const Scalar inverseLambdaMinOverride,
                                                 const Scalar inverseLambdaMaxOverride, const std::string &inverseSpectralEstimate,
                                                 const int inverseSpectralEstimateIterations, const Scalar inversePowerSafetyFactor,
                                                 const int compareTpetraPowerEstimate, const std::string &inverseInitialGuess,
                                                 const Scalar inverseTolerance)
  {
    InverseDiagnosticTiming timing;
    timing.inverseIterations = numInverseIterations;
    timing.inverseTolerance = inverseTolerance;
    timing.inverseStoppedReason = "fixed_iteration_limit";
    timing.inversePath = (inverseDiagnosticAlgorithm == "chebyshev")
                       ? "local_chebyshev_pamatrix_apply_host_recurrence"
                       : "local_damped_jacobi_richardson_pamatrix_apply";
    timing.spectralEstimatePath = "assembled_reference_gershgorin_preconditioned_upper_bound_with_extracted_diagonal";
    timing.spectralEstimateIterations = inverseSpectralEstimateIterations;
    timing.lambdaMaxSafetyFactor = inversePowerSafetyFactor;
    timing.compareTpetraPowerEstimate = compareTpetraPowerEstimate;
    timing.lambdaMaxSource = "assembled_reference_gershgorin_preconditioned_upper_bound";
    timing.inverseInitialGuess = inverseInitialGuess;
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
    if (!useAssembledReference && (inverseSpectralEstimate != "power") && (inverseSpectralEstimate != "user"))
      throw std::invalid_argument("inverse diagnostic without assembled reference requires --inverseSpectralEstimate=power or user");
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
      applyTerms(terms, workspaces, input, output, worksetSize);
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

    Scalar lambdaMaxEstimate = 0.0;
    Kokkos::Timer spectralEstimateTimer;
    if (inverseSpectralEstimate == "assembled-gershgorin")
    {
      if (!useAssembledReference) throw std::invalid_argument("assembled-gershgorin spectral estimate requires assembled reference");
      lambdaMaxEstimate = assembledGershgorinLambdaMax;
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
      timing.spectralEstimatePath = "matrix_free_power_symmetric_diagonal_scaling_with_extracted_diagonal";
      timing.lambdaMaxSource = "matrix_free_power_symmetric_diagonal_scaling_with_safety_factor";
      auto runPowerEstimate = [&](const bool useTpetraApply)
      {
        std::vector<Scalar> vector(static_cast<size_t>(cells * rows), Scalar(0.0));
        std::vector<Scalar> scaledInput(vector.size(), Scalar(0.0));
        std::vector<Scalar> applied(vector.size(), Scalar(0.0));
        std::vector<Scalar> scaledOutput(vector.size(), Scalar(0.0));
        for (size_t i=0; i<vector.size(); i++) vector[i] = Scalar(1.0) + Scalar((i * 13) % 17) / Scalar(17.0);

        Teuchos::RCP<TpetraMap> map;
        Teuchos::RCP<LocalPAMatrixTpetraOperator> tpetraOperator;
        Teuchos::RCP<TpetraMultiVector> X;
        Teuchos::RCP<TpetraMultiVector> Y;
        if (useTpetraApply)
        {
          auto comm = Teuchos::DefaultComm<int>::getComm();
          map = Teuchos::rcp(new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), vector.size(), 0, comm));
          tpetraOperator = Teuchos::rcp(new LocalPAMatrixTpetraOperator(map, terms, workspaces, cells, rows, worksetSize));
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

          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
            {
              const size_t index = static_cast<size_t>(c * rows + i);
              scaledInput[index] = vector[index] / std::sqrt(diagonal[index]);
            }

          if (useTpetraApply)
          {
            auto xHost = X->getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t index=0; index<scaledInput.size(); index++) xHost(index,0) = scaledInput[index];
            tpetraOperator->apply(*X, *Y);
            auto yHost = Y->getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t index=0; index<applied.size(); index++) applied[index] = yHost(index,0);
          }
          else
          {
            applyVector(scaledInput, applied);
          }

          Scalar rayleigh = 0.0;
          Scalar outputNorm = 0.0;
          for (int c=0; c<cells; c++)
            for (int i=0; i<rows; i++)
            {
              const size_t index = static_cast<size_t>(c * rows + i);
              scaledOutput[index] = applied[index] / std::sqrt(diagonal[index]);
              rayleigh += vector[index] * scaledOutput[index];
              outputNorm += scaledOutput[index] * scaledOutput[index];
            }
          outputNorm = std::sqrt(outputNorm);
          rawEstimate = std::max(rawEstimate, std::max(std::abs(rayleigh), outputNorm));
          vector.swap(scaledOutput);
        }
        return rawEstimate;
      };

      lambdaMaxEstimate = runPowerEstimate(false);
      timing.powerRawLambdaMaxEstimate = lambdaMaxEstimate;
      if (compareTpetraPowerEstimate != 0)
      {
        timing.tpetraPowerRawLambdaMaxEstimate = runPowerEstimate(true);
        timing.tpetraPowerLambdaMaxAbsError = std::abs(timing.tpetraPowerRawLambdaMaxEstimate - timing.powerRawLambdaMaxEstimate);
        timing.tpetraPowerLambdaMaxRelError = timing.tpetraPowerLambdaMaxAbsError / std::max<Scalar>(Scalar(1.0), std::abs(timing.powerRawLambdaMaxEstimate));
      }
      // Power iteration is a diagnostic estimate, not a rigorous upper bound; add a
      // user-visible safety margin before using it as the Chebyshev upper endpoint.
      lambdaMaxEstimate *= inversePowerSafetyFactor;
    }
    else
    {
      throw std::invalid_argument("--inverseSpectralEstimate must be assembled-gershgorin, power, or user");
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
      timing.lambdaMinSource = "user_lambda_min_override";
    }
    else
    {
      timing.lambdaMinEstimate = lambdaMaxEstimate / inverseEigenRatio;
      timing.lambdaMinSource = "ratio_from_lambda_max";
    }
    if ((timing.lambdaMinEstimate <= 0.0) || (timing.lambdaMinEstimate >= timing.lambdaMaxEstimate))
      throw std::runtime_error("inverse diagnostic Chebyshev/Richardson spectral bounds must satisfy 0 < lambdaMin < lambdaMax");
    if (inverseLambdaMinOverride > 0.0)
    {
      timing.spectralEstimatePath = (inverseLambdaMaxOverride > 0.0)
                                  ? "user_lambda_min_max_override_with_extracted_diagonal"
                                  : timing.spectralEstimatePath + "_user_lambda_min";
    }

    std::vector<Scalar> rhs(static_cast<size_t>(cells * rows), 0.0);
    for (int c=0; c<cells; c++)
      for (int i=0; i<rows; i++)
        rhs[static_cast<size_t>(c * rows + i)] = 0.25 + 0.0625 * ((i + 5 * c) % 29);

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
      const Scalar theta = Scalar(0.5) * (timing.lambdaMaxEstimate + timing.lambdaMinEstimate);
      const Scalar delta = Scalar(0.5) * (timing.lambdaMaxEstimate - timing.lambdaMinEstimate);
      Scalar alpha = Scalar(0.0);
      std::vector<Scalar> direction(iterate.size(), Scalar(0.0));
      for (int iteration=0; iteration<numInverseIterations; iteration++)
      {
        applyHostVector();
        if (iteration == 0)
        {
          alpha = Scalar(1.0) / theta;
          for (size_t i=0; i<iterate.size(); i++) direction[i] = (rhs[i] - applied[i]) / diagonal[i];
        }
        else
        {
          const Scalar beta = std::pow(alpha * delta * Scalar(0.5), 2);
          const Scalar previousAlpha = alpha;
          alpha = Scalar(1.0) / (theta - beta / previousAlpha);
          for (size_t i=0; i<iterate.size(); i++) direction[i] = (rhs[i] - applied[i]) / diagonal[i] + beta * direction[i];
        }
        for (size_t i=0; i<iterate.size(); i++) iterate[i] += alpha * direction[i];
      }
    }
    else
    {
      throw std::invalid_argument("--inverseDiagnosticAlgorithm must be richardson or chebyshev");
    }
    timing.avgInverseApplySeconds = inverseTimer.seconds() / numInverseIterations;

    applyHostVector();
    timing.finalResidualNorm = residualNorm();
    timing.residualReduction = timing.finalResidualNorm / std::max<Scalar>(std::numeric_limits<Scalar>::epsilon(), timing.initialResidualNorm);
    timing.inverseConverged = (timing.residualReduction <= inverseTolerance) ? 1 : 0;
    if (timing.inverseConverged != 0) timing.inverseStoppedReason = "fixed_iterations_met_tolerance";

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
              << " inverseConverged=" << timing.inverseConverged
              << " inverseStoppedReason=" << timing.inverseStoppedReason
              << " lambdaMinEstimate=" << timing.lambdaMinEstimate
              << " lambdaMaxEstimate=" << timing.lambdaMaxEstimate
              << " assembledGershgorinLambdaMax=" << timing.assembledGershgorinLambdaMax
              << " powerRawLambdaMaxEstimate=" << timing.powerRawLambdaMaxEstimate
              << " tpetraPowerRawLambdaMaxEstimate=" << timing.tpetraPowerRawLambdaMaxEstimate
              << " tpetraPowerLambdaMaxAbsError=" << timing.tpetraPowerLambdaMaxAbsError
              << " tpetraPowerLambdaMaxRelError=" << timing.tpetraPowerLambdaMaxRelError
              << " compareTpetraPowerEstimate=" << timing.compareTpetraPowerEstimate
              << " lambdaMaxEstimateOverAssembledGershgorin=" << timing.lambdaMaxEstimateOverAssembledGershgorin
              << " inverseEigenRatio=" << timing.inverseEigenRatio
              << " spectralEstimateIterations=" << timing.spectralEstimateIterations
              << " spectralEstimateSeconds=" << timing.spectralEstimateSeconds
              << " lambdaMaxSafetyFactor=" << timing.lambdaMaxSafetyFactor
              << " lambdaMinSource=" << timing.lambdaMinSource
              << " lambdaMaxSource=" << timing.lambdaMaxSource
              << " maxSymmetryAbsError=" << timing.maxSymmetryAbsError
              << " minDiagonalEstimate=" << timing.minDiagonalEstimate
              << " maxDiagonalEstimate=" << timing.maxDiagonalEstimate
              << " iterations=" << timing.inverseIterations
              << " inverseInitialGuess=" << timing.inverseInitialGuess
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
    int inverseSpectralEstimateIterations = 8;
    int inverseDiagnosticRequireAssembledReference = 1;
    int compareTpetraPowerEstimate = 0;
    int worksetSize = 0;
    int checkAssemblyAgreement = 0;
    long long maxAssembledEntries = 200000;
    Scalar massWeight = 1.0;
    Scalar stiffnessWeight = 1.0;
    Scalar inverseEigenRatio = 30.0;
    Scalar inverseLambdaMin = -1.0;
    Scalar inverseLambdaMax = -1.0;
    Scalar inversePowerSafetyFactor = 1.05;
    Scalar inverseTolerance = 1e-8;
    std::string polyOrderText = "1:4";
    std::string operatorChoice = "mass-plus-stiffness";
    std::string mode = "apply";
    std::string basisMode = "hgrad";
    std::string inverseDiagnosticAlgorithm = "richardson";
    std::string inverseSpectralEstimate = "assembled-gershgorin";
    std::string inverseInitialGuess = "zero";

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("dim", &dim, "Tensor dimension, 1 through 7.");
    cmdp.setOption("polyOrder", &polyOrderText, "Polynomial order p or inclusive range pMin:pMax.");
    cmdp.setOption("cellsPerDim", &cellsPerDim, "Uniform tensor cell count per dimension.");
    cmdp.setOption("operator", &operatorChoice, "mass, stiffness, mass-plus-stiffness, or h1-gram.  h1-gram is the weighted mass-plus-stiffness Gram starter case.");
    cmdp.setOption("basisMode", &basisMode, "synthetic or hgrad.  hgrad uses true Intrepid2 HGRAD basis/cubature/affine geometry and currently supports dim<=3.");
    cmdp.setOption("mode", &mode, "apply, assemble, assemble2, assembly-compare, diagonal, compare, factor, inverse-diagnostic, or tpetra-operator-compare.");
    cmdp.setOption("numApply", &numApply, "Number of repeated applies used for steady-state timing in apply/compare modes.");
    cmdp.setOption("numDiagonal", &numDiagonal, "Number of repeated diagonal extractions used for steady-state timing in diagonal mode.");
    cmdp.setOption("numSolve", &numSolve, "Number of repeated local triangular solves used for average timing in factor mode.");
    cmdp.setOption("numInverseIterations", &numInverseIterations, "Number of local Richardson or Chebyshev iterations used in inverse-diagnostic mode.");
    cmdp.setOption("inverseDiagnosticAlgorithm", &inverseDiagnosticAlgorithm, "richardson or chebyshev local inverse diagnostic recurrence.");
    cmdp.setOption("inverseSpectralEstimate", &inverseSpectralEstimate, "assembled-gershgorin, power, or user spectral upper-estimate path for inverse diagnostics.");
    cmdp.setOption("inverseSpectralEstimateIterations", &inverseSpectralEstimateIterations, "Iterations for the matrix-free power spectral estimate.");
    cmdp.setOption("inversePowerSafetyFactor", &inversePowerSafetyFactor, "Safety factor applied to matrix-free power lambdaMax estimates before Chebyshev use.");
    cmdp.setOption("inverseDiagnosticRequireAssembledReference", &inverseDiagnosticRequireAssembledReference, "Set to 0 to allow power-estimated inverse diagnostics without assembled reference/oracle matrices.");
    cmdp.setOption("compareTpetraPowerEstimate", &compareTpetraPowerEstimate, "Set to 1 in inverse-diagnostic power mode to compare direct and local-Tpetra-operator power estimates.");
    cmdp.setOption("inverseEigenRatio", &inverseEigenRatio, "Chebyshev lambdaMax/lambdaMin ratio used when --inverseLambdaMin is not positive.");
    cmdp.setOption("inverseLambdaMin", &inverseLambdaMin, "Optional positive user override for the Chebyshev/Richardson lower spectral bound.");
    cmdp.setOption("inverseLambdaMax", &inverseLambdaMax, "Optional positive user override for the Chebyshev/Richardson upper spectral bound.");
    cmdp.setOption("inverseInitialGuess", &inverseInitialGuess, "zero or diagonal initial guess for inverse diagnostics.");
    cmdp.setOption("inverseTolerance", &inverseTolerance, "Relative residual-reduction tolerance reported by inverse diagnostics; fixed iteration count is unchanged.");
    cmdp.setOption("massWeight", &massWeight, "Scalar coefficient alpha for mass or h1-gram value-value terms.");
    cmdp.setOption("stiffnessWeight", &stiffnessWeight, "Scalar coefficient beta for stiffness or h1-gram grad-grad terms.");
    cmdp.setOption("worksetSize", &worksetSize, "PAMatrix apply workset size; <=0 uses all cells.");
    cmdp.setOption("checkAssemblyAgreement", &checkAssemblyAgreement, "Set to 1 in assembly-compare mode to compare assemble2() entries against standard assemble().");
    cmdp.setOption("maxAssembledEntries", &maxAssembledEntries, "Maximum local assembled entries allowed in assemble/assemble2/compare/factor modes.");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      Kokkos::finalize();
      return -1;
    }

    if ((dim < 1) || (dim > 7)) throw std::invalid_argument("--dim must be in [1,7]");
    if (cellsPerDim < 1) throw std::invalid_argument("--cellsPerDim must be positive");
    if (numApply < 1) throw std::invalid_argument("--numApply must be positive");
    if (numDiagonal < 1) throw std::invalid_argument("--numDiagonal must be positive");
    if (numSolve < 1) throw std::invalid_argument("--numSolve must be positive");
    if (numInverseIterations < 1) throw std::invalid_argument("--numInverseIterations must be positive");
    if ((checkAssemblyAgreement != 0) && (checkAssemblyAgreement != 1)) throw std::invalid_argument("--checkAssemblyAgreement must be 0 or 1");
    if ((inverseDiagnosticRequireAssembledReference != 0) && (inverseDiagnosticRequireAssembledReference != 1)) throw std::invalid_argument("--inverseDiagnosticRequireAssembledReference must be 0 or 1");
    if ((compareTpetraPowerEstimate != 0) && (compareTpetraPowerEstimate != 1)) throw std::invalid_argument("--compareTpetraPowerEstimate must be 0 or 1");
    if ((mode != "apply") && (mode != "assemble") && (mode != "assemble2") && (mode != "assembly-compare") && (mode != "diagonal") && (mode != "compare") && (mode != "factor") && (mode != "inverse-diagnostic") && (mode != "tpetra-operator-compare"))
      throw std::invalid_argument("--mode must be apply, assemble, assemble2, assembly-compare, diagonal, compare, factor, inverse-diagnostic, or tpetra-operator-compare");
    if ((basisMode != "synthetic") && (basisMode != "hgrad"))
      throw std::invalid_argument("--basisMode must be synthetic or hgrad");
    if ((inverseDiagnosticAlgorithm != "richardson") && (inverseDiagnosticAlgorithm != "chebyshev"))
      throw std::invalid_argument("--inverseDiagnosticAlgorithm must be richardson or chebyshev");
    if ((inverseInitialGuess != "zero") && (inverseInitialGuess != "diagonal"))
      throw std::invalid_argument("--inverseInitialGuess must be zero or diagonal");
    if ((inverseSpectralEstimate != "assembled-gershgorin") && (inverseSpectralEstimate != "power") && (inverseSpectralEstimate != "user"))
      throw std::invalid_argument("--inverseSpectralEstimate must be assembled-gershgorin, power, or user");
    if ((inverseSpectralEstimate == "user") && (inverseLambdaMax <= 0.0))
      throw std::invalid_argument("--inverseSpectralEstimate=user requires positive --inverseLambdaMax");
    if (inverseSpectralEstimateIterations < 1) throw std::invalid_argument("--inverseSpectralEstimateIterations must be positive");
    if (inversePowerSafetyFactor < 1.0) throw std::invalid_argument("--inversePowerSafetyFactor must be at least 1");
    if (inverseEigenRatio <= 1.0) throw std::invalid_argument("--inverseEigenRatio must be greater than 1");
    if ((inverseLambdaMin > 0.0) && (inverseLambdaMax > 0.0) && (inverseLambdaMin >= inverseLambdaMax))
      throw std::invalid_argument("--inverseLambdaMin must be less than --inverseLambdaMax when both are positive");
    if (inverseTolerance <= 0.0) throw std::invalid_argument("--inverseTolerance must be positive");
    if ((basisMode == "hgrad") && (dim > 3))
      throw std::invalid_argument("--basisMode=hgrad currently supports --dim=1,2,3");

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

    std::cout << "TENSOR_POISSON_PAMATRIX_HEADER dim p cells localDofs localPoints terms termWeights setupSeconds firstApplySeconds avgApplySeconds assemblySeconds massAssemblySeconds stiffnessAssemblySeconds assemble2Seconds massAssemble2Seconds stiffnessAssemble2Seconds assemble2OverAssembleRatio assemblyAgreementMaxAbsError assemblyAgreementMaxRelError tpetraOperatorMaxAbsError tpetraOperatorMaxRelError tpetraInverseDiagonalMaxAbsError tpetraInverseDiagonalMin tpetraInverseDiagonalMax tpetraOperatorFirstApplySeconds tpetraOperatorAvgApplySeconds tpetraOperatorNumVectors tpetraOperatorLocalLength tpetraDomainRangeMapSame tpetraAdapterPath tpetraGlobalDofSemantics tpetraIfpack2RowMatrix factorSeconds avgSolveSeconds maxSolveResidual maxSolveRelativeResidual firstDiagonalSeconds avgDiagonalSeconds avgInverseApplySeconds inverseIterations initialInverseAbsError initialInverseRelError maxInverseAbsError maxInverseRelError initialResidualNorm finalResidualNorm residualReduction inverseTolerance inverseConverged inverseStoppedReason lambdaMinEstimate lambdaMaxEstimate assembledGershgorinLambdaMax powerRawLambdaMaxEstimate tpetraPowerRawLambdaMaxEstimate tpetraPowerLambdaMaxAbsError tpetraPowerLambdaMaxRelError compareTpetraPowerEstimate lambdaMaxEstimateOverAssembledGershgorin inverseEigenRatio spectralEstimateIterations spectralEstimateSeconds lambdaMaxSafetyFactor maxSymmetryAbsError minDiagonalEstimate maxDiagonalEstimate estimatedApplyFlopsPerCell lambdaMinSource lambdaMaxSource inverseInitialGuess achievedGFlops mode operator basisMode diagonalPath assemble2Path inversePath spectralEstimatePath" << std::endl;
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
      auto workspaces = allocateWorkspaces(terms, worksetSize);
      Intrepid2::ScalarView<Scalar,DeviceType> input("TensorPoissonPAMatrix input", cells, localDofs);
      Intrepid2::ScalarView<Scalar,DeviceType> output("TensorPoissonPAMatrix output", cells, localDofs);
      fillInput(input);

      double firstApplySeconds = -1.0;
      double avgApplySeconds = -1.0;
      if ((mode == "apply") || (mode == "compare"))
      {
        firstApplySeconds = applyTerms(terms, workspaces, input, output, worksetSize);
        Kokkos::Timer applyTimer;
        for (int applyOrdinal=0; applyOrdinal<numApply; applyOrdinal++)
        {
          applyTerms(terms, workspaces, input, output, worksetSize);
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
        inverseDiagnosticTiming = inverseDiagnosticTerms(terms, workspaces, worksetSize, cells, localDofs,
                                                        maxAssembledEntries, inverseDiagnosticRequireAssembledReference, numInverseIterations,
                                                        inverseDiagnosticAlgorithm, inverseEigenRatio, inverseLambdaMin, inverseLambdaMax,
                                                        inverseSpectralEstimate, inverseSpectralEstimateIterations, inversePowerSafetyFactor,
                                                        compareTpetraPowerEstimate, inverseInitialGuess, inverseTolerance);
        if ((compareTpetraPowerEstimate != 0) &&
            (inverseDiagnosticTiming.tpetraPowerLambdaMaxAbsError > 1e-10) &&
            (inverseDiagnosticTiming.tpetraPowerLambdaMaxRelError > 1e-10)) returnCode = 2;
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

      std::cout << "TENSOR_POISSON_PAMATRIX_RESULT"
                << " dim=" << dim
                << " p=" << p
                << " cells=" << cells
                << " localDofs=" << localDofs
                << " localPoints=" << localPoints
                << " terms=" << terms.size()
                << " termWeights=" << termSummary(terms)
                << " setupSeconds=" << std::setprecision(8) << setupSeconds
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
                << " tpetraOperatorFirstApplySeconds=" << tpetraOperatorComparison.firstApplySeconds
                << " tpetraOperatorAvgApplySeconds=" << tpetraOperatorComparison.avgApplySeconds
                << " tpetraOperatorNumVectors=" << tpetraOperatorComparison.numVectors
                << " tpetraOperatorLocalLength=" << tpetraOperatorComparison.localLength
                << " tpetraDomainRangeMapSame=" << tpetraOperatorComparison.domainRangeMapSame
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
                << " inverseConverged=" << inverseDiagnosticTiming.inverseConverged
                << " inverseStoppedReason=" << inverseDiagnosticTiming.inverseStoppedReason
                << " lambdaMinEstimate=" << inverseDiagnosticTiming.lambdaMinEstimate
                << " lambdaMaxEstimate=" << inverseDiagnosticTiming.lambdaMaxEstimate
                << " assembledGershgorinLambdaMax=" << inverseDiagnosticTiming.assembledGershgorinLambdaMax
                << " powerRawLambdaMaxEstimate=" << inverseDiagnosticTiming.powerRawLambdaMaxEstimate
                << " tpetraPowerRawLambdaMaxEstimate=" << inverseDiagnosticTiming.tpetraPowerRawLambdaMaxEstimate
                << " tpetraPowerLambdaMaxAbsError=" << inverseDiagnosticTiming.tpetraPowerLambdaMaxAbsError
                << " tpetraPowerLambdaMaxRelError=" << inverseDiagnosticTiming.tpetraPowerLambdaMaxRelError
                << " compareTpetraPowerEstimate=" << inverseDiagnosticTiming.compareTpetraPowerEstimate
                << " lambdaMaxEstimateOverAssembledGershgorin=" << inverseDiagnosticTiming.lambdaMaxEstimateOverAssembledGershgorin
                << " inverseEigenRatio=" << inverseDiagnosticTiming.inverseEigenRatio
                << " spectralEstimateIterations=" << inverseDiagnosticTiming.spectralEstimateIterations
                << " spectralEstimateSeconds=" << inverseDiagnosticTiming.spectralEstimateSeconds
                << " lambdaMaxSafetyFactor=" << inverseDiagnosticTiming.lambdaMaxSafetyFactor
                << " maxSymmetryAbsError=" << inverseDiagnosticTiming.maxSymmetryAbsError
                << " minDiagonalEstimate=" << inverseDiagnosticTiming.minDiagonalEstimate
                << " maxDiagonalEstimate=" << inverseDiagnosticTiming.maxDiagonalEstimate
                << " estimatedApplyFlopsPerCell=" << flopsPerCell
                << " lambdaMinSource=" << inverseDiagnosticTiming.lambdaMinSource
                << " lambdaMaxSource=" << inverseDiagnosticTiming.lambdaMaxSource
                << " inverseInitialGuess=" << inverseDiagnosticTiming.inverseInitialGuess
                << " achievedGFlops=" << achievedGFlops
                << " mode=" << mode
                << " operator=" << operatorChoice
                << " basisMode=" << basisMode
                << " diagonalPath=" << diagonalPath
                << " assemble2Path=" << assemble2Path
                << " inversePath=" << inverseDiagnosticTiming.inversePath
                << " spectralEstimatePath=" << inverseDiagnosticTiming.spectralEstimatePath
                << std::endl;

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
  Kokkos::finalize();
  return returnCode;
}
