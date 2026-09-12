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
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_RCP.hpp>

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
    int worksetSize = 0;
    long long maxAssembledEntries = 200000;
    Scalar massWeight = 1.0;
    Scalar stiffnessWeight = 1.0;
    std::string polyOrderText = "1:4";
    std::string operatorChoice = "mass-plus-stiffness";
    std::string mode = "apply";
    std::string basisMode = "hgrad";

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("dim", &dim, "Tensor dimension, 1 through 7.");
    cmdp.setOption("polyOrder", &polyOrderText, "Polynomial order p or inclusive range pMin:pMax.");
    cmdp.setOption("cellsPerDim", &cellsPerDim, "Uniform tensor cell count per dimension.");
    cmdp.setOption("operator", &operatorChoice, "mass, stiffness, mass-plus-stiffness, or h1-gram.  h1-gram is the weighted mass-plus-stiffness Gram starter case.");
    cmdp.setOption("basisMode", &basisMode, "synthetic or hgrad.  hgrad uses true Intrepid2 HGRAD basis/cubature/affine geometry and currently supports dim<=3.");
    cmdp.setOption("mode", &mode, "apply, assemble, assemble2, assembly-compare, diagonal, compare, or factor.");
    cmdp.setOption("numApply", &numApply, "Number of repeated applies used for steady-state timing in apply/compare modes.");
    cmdp.setOption("numDiagonal", &numDiagonal, "Number of repeated diagonal extractions used for steady-state timing in diagonal mode.");
    cmdp.setOption("numSolve", &numSolve, "Number of repeated local triangular solves used for average timing in factor mode.");
    cmdp.setOption("massWeight", &massWeight, "Scalar coefficient alpha for mass or h1-gram value-value terms.");
    cmdp.setOption("stiffnessWeight", &stiffnessWeight, "Scalar coefficient beta for stiffness or h1-gram grad-grad terms.");
    cmdp.setOption("worksetSize", &worksetSize, "PAMatrix apply workset size; <=0 uses all cells.");
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
    if ((mode != "apply") && (mode != "assemble") && (mode != "assemble2") && (mode != "assembly-compare") && (mode != "diagonal") && (mode != "compare") && (mode != "factor"))
      throw std::invalid_argument("--mode must be apply, assemble, assemble2, assembly-compare, diagonal, compare, or factor");
    if ((basisMode != "synthetic") && (basisMode != "hgrad"))
      throw std::invalid_argument("--basisMode must be synthetic or hgrad");
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

    std::cout << "TENSOR_POISSON_PAMATRIX_HEADER dim p cells localDofs localPoints terms termWeights setupSeconds firstApplySeconds avgApplySeconds assemblySeconds massAssemblySeconds stiffnessAssemblySeconds assemble2Seconds massAssemble2Seconds stiffnessAssemble2Seconds assemble2OverAssembleRatio factorSeconds avgSolveSeconds maxSolveResidual maxSolveRelativeResidual firstDiagonalSeconds avgDiagonalSeconds estimatedApplyFlopsPerCell achievedGFlops mode operator basisMode diagonalPath" << std::endl;
    for (int p=pRange.first; p<=pRange.last; p++)
    {
      Kokkos::Timer setupTimer;
      auto terms = makeOperatorTerms(dim, p, cellsPerDim, cells, operatorChoice, basisMode, massWeight, stiffnessWeight);
      const double setupSeconds = setupTimer.seconds();

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

      const long long flopsPerCell = estimatedApplyFlopsPerCellPerTerm(dim, p) * static_cast<long long>(terms.size());
      const double achievedGFlops = (avgApplySeconds > 0.0) ? (flopsPerCell * static_cast<double>(cells) / avgApplySeconds / 1.0e9) : -1.0;
      const double assemble2OverAssembleRatio = ((assemble2Timing.totalSeconds > 0.0) && (assemblyTiming.totalSeconds > 0.0))
                                              ? assemble2Timing.totalSeconds / assemblyTiming.totalSeconds
                                              : -1.0;

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
                << " factorSeconds=" << factorTiming.factorSeconds
                << " avgSolveSeconds=" << factorTiming.solveSeconds
                << " maxSolveResidual=" << factorTiming.maxSolveResidual
                << " maxSolveRelativeResidual=" << factorTiming.maxSolveRelativeResidual
                << " firstDiagonalSeconds=" << diagonalTiming.firstSeconds
                << " avgDiagonalSeconds=" << diagonalTiming.averageSeconds
                << " estimatedApplyFlopsPerCell=" << flopsPerCell
                << " achievedGFlops=" << achievedGFlops
                << " mode=" << mode
                << " operator=" << operatorChoice
                << " basisMode=" << basisMode
                << " diagonalPath=" << diagonalPath
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
                << " note=assemble2_forms_columns_by_repeated_matrix_free_apply" << std::endl;
    }
    if (logMassAssemble2Times.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE massAssemble2Slope=" << leastSquaresSlope(logMassAssemble2PValues, logMassAssemble2Times)
                << " expectedColumnApplyAssemblySlope=" << (2 * dim + 1)
                << " samples=" << logMassAssemble2Times.size()
                << " note=term_separated_column_apply_assembly_timing" << std::endl;
    }
    if (logStiffnessAssemble2Times.size() >= 2)
    {
      std::cout << "TENSOR_POISSON_PAMATRIX_SLOPE stiffnessAssemble2Slope=" << leastSquaresSlope(logStiffnessAssemble2PValues, logStiffnessAssemble2Times)
                << " expectedColumnApplyAssemblySlope=" << (2 * dim + 1)
                << " samples=" << logStiffnessAssemble2Times.size()
                << " note=term_separated_column_apply_assembly_timing" << std::endl;
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
