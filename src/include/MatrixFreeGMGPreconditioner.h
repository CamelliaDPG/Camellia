// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MatrixFreeGMGPreconditioner.h
//  Camellia
//

#ifndef Camellia_MatrixFreeGMGPreconditioner_h
#define Camellia_MatrixFreeGMGPreconditioner_h

#include "TypeDefs.h"

#include "Teuchos_Exceptions.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
  /** \brief Minimal matrix-free geometric multigrid V-cycle preconditioner shell.

      This class is intentionally infrastructure-only: it owns the control flow for
      one V-cycle application, while each level supplies matrix-free operations via
      callbacks.  That makes the V-cycle semantics unit-testable before wiring in
      tensor/PAMatrix smoothers, restriction/prolongation kernels, or assembled
      GMGOperator comparison fixtures.

      One apply() call computes

        Y = beta * Y + alpha * VCycle(X),

      so the object can be used directly as a Belos right preconditioner once the
      level callbacks are backed by production matrix-free kernels.

      Optional stage tracing records the concrete residual/correction stages used
      by GMGOperator study output: initial residual, residual after pre-smoothing,
      restricted residual, coarse correction, residual after coarse correction, and
      residual after post-smoothing.  The trace is intentionally stored as data so
      unit tests and study drivers can compare MatrixFreeGMGPreconditioner and
      GMGOperator stage-by-stage without scraping stdout.
   */
  class MatrixFreeGMGPreconditioner : public TOperator<double>
  {
  public:
    using Scalar = double;
    using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
    using MultiVector = TVector<Scalar>;
    using Operator = TOperator<Scalar>;
    using ConstOperatorPtr = Teuchos::RCP<const Operator>;
    using Operation = std::function<void(const MultiVector &, MultiVector &)>;

    struct StageTraceEntry
    {
      size_t levelIndex = 0;
      std::string stageName;
      Magnitude rhsInfNorm = Magnitude(0.0);
      Magnitude solutionInfNorm = Magnitude(0.0);
      Magnitude residualInfNorm = Magnitude(0.0);
    };

    struct Level
    {
      TMapConstPtr map;

      // Required on non-coarsest levels: A_level * x.
      Operation applyOperator;

      // Required on non-coarsest levels.  Smoothers update the supplied solution
      // approximation in-place for the supplied right-hand side.
      Operation preSmooth;
      Operation postSmooth;

      // Required on non-coarsest levels.
      Operation restrictResidual;
      Operation prolongateCorrection;

      // Required on the coarsest level.
      Operation coarseSolve;
    };

    explicit MatrixFreeGMGPreconditioner(const std::vector<Level> &levels,
                                         const std::string &operatorName = "MatrixFreeGMGPreconditioner")
    : _levels(levels), _operatorName(operatorName), _emitStageTrace(false)
    {
      validateLevels();
    }

    /** \brief Adapt a Tpetra::Operator-style object to a level callback. */
    static Operation operatorApply(const ConstOperatorPtr &op)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(op.is_null(), std::invalid_argument,
                                 "MatrixFreeGMGPreconditioner::operatorApply requires a non-null operator");
      return [op](const MultiVector &input, MultiVector &output)
      {
        op->apply(input, output);
      };
    }

    /** \brief Adapt an approximate inverse/correction operator to an in-place additive smoother. */
    static Operation operatorAdditiveUpdate(const ConstOperatorPtr &op,
                                            const Scalar weight = Teuchos::ScalarTraits<Scalar>::one())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(op.is_null(), std::invalid_argument,
                                 "MatrixFreeGMGPreconditioner::operatorAdditiveUpdate requires a non-null operator");
      return [op, weight](const MultiVector &input, MultiVector &output)
      {
        MultiVector correction(output.getMap(), output.getNumVectors());
        op->apply(input, correction);
        output.update(weight, correction, Teuchos::ScalarTraits<Scalar>::one());
      };
    }

    /** \brief Convenience constructor for a non-coarsest level backed by Tpetra operators. */
    static Level operatorBackedLevel(const TMapConstPtr &map,
                                     const ConstOperatorPtr &levelOperator,
                                     const ConstOperatorPtr &preSmoother,
                                     const ConstOperatorPtr &postSmoother,
                                     const ConstOperatorPtr &restriction,
                                     const ConstOperatorPtr &prolongation)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(levelOperator.is_null(), std::invalid_argument,
                                 "MatrixFreeGMGPreconditioner::operatorBackedLevel requires a non-null level operator");
      Level level;
      level.map = map.is_null() ? levelOperator->getRangeMap() : map;
      level.applyOperator = operatorApply(levelOperator);
      level.preSmooth = operatorAdditiveUpdate(preSmoother);
      level.postSmooth = operatorAdditiveUpdate(postSmoother);
      level.restrictResidual = operatorApply(restriction);
      level.prolongateCorrection = operatorApply(prolongation);
      return level;
    }

    /** \brief Convenience constructor for the coarsest level backed by an operator solve/update. */
    static Level operatorBackedCoarseLevel(const TMapConstPtr &map,
                                           const ConstOperatorPtr &coarseSolver)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(coarseSolver.is_null(), std::invalid_argument,
                                 "MatrixFreeGMGPreconditioner::operatorBackedCoarseLevel requires a non-null coarse solver");

      Level level;
      level.map = map.is_null() ? coarseSolver->getRangeMap() : map;
      level.coarseSolve = operatorApply(coarseSolver);
      return level;
    }

    void setEmitStageTrace(const bool emitStageTrace)
    {
      _emitStageTrace = emitStageTrace;
      if (!emitStageTrace)
        _lastStageTrace.clear();
    }

    bool getEmitStageTrace() const
    {
      return _emitStageTrace;
    }

    const std::vector<StageTraceEntry> &lastStageTrace() const
    {
      return _lastStageTrace;
    }

    void clearStageTrace() const
    {
      _lastStageTrace.clear();
    }

    void apply(const MultiVector &X, MultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta  = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::invalid_argument,
                                 _operatorName << " does not support transpose apply");
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                                 _operatorName << " requires X and Y to have the same number of vectors");

      _lastStageTrace.clear();

      MultiVector vcycleResult(this->getRangeMap(), X.getNumVectors());
      applyLevel(0, X, vcycleResult);
      Y.update(alpha, vcycleResult, beta);
    }

    int ApplyInverse(const MultiVector &X, MultiVector &Y) const
    {
      this->apply(X, Y);
      return 0;
    }

    TMapConstPtr getDomainMap() const override
    {
      return _levels.front().map;
    }

    TMapConstPtr getRangeMap() const override
    {
      return _levels.front().map;
    }

    bool hasTransposeApply() const override
    {
      return false;
    }

  private:
    Magnitude vectorInfNorm(const MultiVector &x) const
    {
      Magnitude norm = Magnitude(0.0);
      for (size_t vectorIndex=0; vectorIndex<x.getNumVectors(); vectorIndex++)
      {
        auto data = x.getData(vectorIndex);
        for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
          norm = std::max(norm, static_cast<Magnitude>(std::abs(data[localOrdinal])));
      }
      return norm;
    }

    void recordStage(const size_t levelIndex, const std::string &stageName,
                     const MultiVector &rhs, const MultiVector &solution,
                     const MultiVector *residual = nullptr) const
    {
      if (!_emitStageTrace)
        return;

      StageTraceEntry entry;
      entry.levelIndex = levelIndex;
      entry.stageName = stageName;
      entry.rhsInfNorm = vectorInfNorm(rhs);
      entry.solutionInfNorm = vectorInfNorm(solution);
      entry.residualInfNorm = residual == nullptr ? Magnitude(0.0) : vectorInfNorm(*residual);
      _lastStageTrace.push_back(entry);
    }

    void computeResidual(const Level &level, const MultiVector &rhs, const MultiVector &solution,
                         MultiVector &operatorSolution, MultiVector &residual) const
    {
      level.applyOperator(solution, operatorSolution);
      residual.update(Teuchos::ScalarTraits<Scalar>::one(), rhs,
                      Teuchos::ScalarTraits<Scalar>::zero());
      residual.update(-Teuchos::ScalarTraits<Scalar>::one(), operatorSolution,
                      Teuchos::ScalarTraits<Scalar>::one());
    }

    void validateLevels() const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_levels.empty(), std::invalid_argument,
                                 _operatorName << " requires at least one level");

      for (size_t levelIndex=0; levelIndex<_levels.size(); levelIndex++)
      {
        const Level &level = _levels[levelIndex];
        TEUCHOS_TEST_FOR_EXCEPTION(level.map.is_null(), std::invalid_argument,
                                   _operatorName << " level " << levelIndex << " has a null map");

        const bool coarsest = (levelIndex + 1 == _levels.size());
        if (coarsest)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(!level.coarseSolve, std::invalid_argument,
                                     _operatorName << " coarsest level requires coarseSolve");
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(!level.applyOperator, std::invalid_argument,
                                     _operatorName << " level " << levelIndex << " requires applyOperator");
          TEUCHOS_TEST_FOR_EXCEPTION(!level.preSmooth, std::invalid_argument,
                                     _operatorName << " level " << levelIndex << " requires preSmooth");
          TEUCHOS_TEST_FOR_EXCEPTION(!level.postSmooth, std::invalid_argument,
                                     _operatorName << " level " << levelIndex << " requires postSmooth");
          TEUCHOS_TEST_FOR_EXCEPTION(!level.restrictResidual, std::invalid_argument,
                                     _operatorName << " level " << levelIndex << " requires restrictResidual");
          TEUCHOS_TEST_FOR_EXCEPTION(!level.prolongateCorrection, std::invalid_argument,
                                     _operatorName << " level " << levelIndex << " requires prolongateCorrection");
        }
      }
    }

    void applyLevel(size_t levelIndex, const MultiVector &rhs, MultiVector &solution) const
    {
      const Level &level = _levels[levelIndex];
      solution.putScalar(Teuchos::ScalarTraits<Scalar>::zero());

      const bool coarsest = (levelIndex + 1 == _levels.size());
      if (coarsest)
      {
        level.coarseSolve(rhs, solution);
        recordStage(levelIndex, "CoarseSolve", rhs, solution);
        return;
      }

      const size_t numVectors = rhs.getNumVectors();
      MultiVector operatorSolution(level.map, numVectors);
      MultiVector residual(level.map, numVectors);
      MultiVector coarseRhs(_levels[levelIndex+1].map, numVectors);
      MultiVector coarseCorrection(_levels[levelIndex+1].map, numVectors);
      MultiVector fineCorrection(level.map, numVectors);

      recordStage(levelIndex, "initialResidualNorm", rhs, solution, &rhs);

      level.preSmooth(rhs, solution);
      computeResidual(level, rhs, solution, operatorSolution, residual);
      recordStage(levelIndex, "residualAfterPreSmooth", rhs, solution, &residual);

      level.restrictResidual(residual, coarseRhs);
      recordStage(levelIndex, "restrictedResidualNorm", residual, coarseRhs, &coarseRhs);

      applyLevel(levelIndex+1, coarseRhs, coarseCorrection);
      recordStage(levelIndex, "coarseCorrectionNorm", coarseRhs, coarseCorrection);

      level.prolongateCorrection(coarseCorrection, fineCorrection);
      recordStage(levelIndex, "prolongateCorrection", coarseCorrection, fineCorrection);

      solution.update(Teuchos::ScalarTraits<Scalar>::one(), fineCorrection,
                      Teuchos::ScalarTraits<Scalar>::one());
      computeResidual(level, rhs, solution, operatorSolution, residual);
      recordStage(levelIndex, "residualAfterCoarseCorrection", rhs, solution, &residual);

      level.postSmooth(rhs, solution);
      computeResidual(level, rhs, solution, operatorSolution, residual);
      recordStage(levelIndex, "residualAfterPostSmooth", rhs, solution, &residual);
    }

    std::vector<Level> _levels;
    std::string _operatorName;
    bool _emitStageTrace;
    mutable std::vector<StageTraceEntry> _lastStageTrace;
  };
}

#endif
