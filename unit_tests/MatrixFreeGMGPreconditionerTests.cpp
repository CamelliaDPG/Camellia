#include "Teuchos_UnitTestHarness.hpp"

#include "MatrixFreeGMGPreconditioner.h"
#include "TypeDefs.h"

#include "Tpetra_Map.hpp"
#include "Teuchos_DefaultComm.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  TMapConstPtr serialMap(const GlobalIndexType numGlobalEntries)
  {
    return Teuchos::rcp(new TMap(numGlobalEntries, 0, Teuchos::DefaultComm<int>::getComm()));
  }

  void fillVector(TVector<double> &x, const double value)
  {
    for (size_t vectorIndex=0; vectorIndex<x.getNumVectors(); vectorIndex++)
    {
      auto data = x.getDataNonConst(vectorIndex);
      for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
        data[localOrdinal] = value;
    }
  }

  double vectorValue(const TVector<double> &x)
  {
    auto data = x.getData(0);
    return data[0];
  }

  void scaleVector(const TVector<double> &x, TVector<double> &y, const double scale)
  {
    for (size_t vectorIndex=0; vectorIndex<x.getNumVectors(); vectorIndex++)
    {
      auto xData = x.getData(vectorIndex);
      auto yData = y.getDataNonConst(vectorIndex);
      for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
        yData[localOrdinal] = scale * xData[localOrdinal];
    }
  }

  void copyVector(const TVector<double> &x, TVector<double> &y)
  {
    scaleVector(x, y, 1.0);
  }

  class ScalingOperator : public MatrixFreeGMGPreconditioner::Operator
  {
  public:
    ScalingOperator(const TMapConstPtr &domainMap, const TMapConstPtr &rangeMap, const double scale)
    : _domainMap(domainMap), _rangeMap(rangeMap), _scale(scale)
    {}

    void apply(const TVector<double> &X, TVector<double> &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               double alpha = Teuchos::ScalarTraits<double>::one(),
               double beta = Teuchos::ScalarTraits<double>::zero()) const override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::invalid_argument,
                                 "ScalingOperator does not support transpose apply");
      TVector<double> scaled(Y.getMap(), Y.getNumVectors());
      scaleVector(X, scaled, _scale);
      Y.update(alpha, scaled, beta);
    }

    TMapConstPtr getDomainMap() const override
    {
      return _domainMap;
    }

    TMapConstPtr getRangeMap() const override
    {
      return _rangeMap;
    }

    bool hasTransposeApply() const override
    {
      return false;
    }

  private:
    TMapConstPtr _domainMap;
    TMapConstPtr _rangeMap;
    double _scale;
  };

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, MatrixFreeGMGPreconditionerOneLevelUsesCoarseSolveAndAlphaBeta )
  {
    auto map = serialMap(3);

    int coarseSolveCount = 0;

    MatrixFreeGMGPreconditioner::Level coarseLevel;
    coarseLevel.map = map;
    coarseLevel.coarseSolve = [&](const TVector<double> &rhs, TVector<double> &solution)
    {
      coarseSolveCount++;
      scaleVector(rhs, solution, 2.0);
    };

    MatrixFreeGMGPreconditioner preconditioner({coarseLevel});

    TVector<double> rhs(map, 2);
    TVector<double> y(map, 2);
    fillVector(rhs, 5.0);
    fillVector(y, 3.0);

    preconditioner.apply(rhs, y, Teuchos::NO_TRANS, 4.0, 7.0);

    TEST_EQUALITY(coarseSolveCount, 1);
    for (size_t vectorIndex=0; vectorIndex<y.getNumVectors(); vectorIndex++)
    {
      auto yData = y.getData(vectorIndex);
      for (size_t localOrdinal=0; localOrdinal<y.getLocalLength(); localOrdinal++)
        TEST_FLOATING_EQUALITY(yData[localOrdinal], 61.0, 1e-14); // 7*3 + 4*(2*5)
    }
    TEST_EQUALITY(preconditioner.lastStageTrace().size(), 0);
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, MatrixFreeGMGPreconditionerTwoLevelVcycleOrdersResidualCorrectionAndPostsmooth )
  {
    auto fineMap = serialMap(1);
    auto coarseMap = serialMap(1);

    std::vector<int> stageOrder;

    MatrixFreeGMGPreconditioner::Level fineLevel;
    fineLevel.map = fineMap;
    fineLevel.preSmooth = [&](const TVector<double> &rhs, TVector<double> &solution)
    {
      stageOrder.push_back(0);
      scaleVector(rhs, solution, 0.25);
    };
    fineLevel.applyOperator = [&](const TVector<double> &x, TVector<double> &Ax)
    {
      stageOrder.push_back(1);
      copyVector(x, Ax);
    };
    fineLevel.restrictResidual = [&](const TVector<double> &fineResidual, TVector<double> &coarseResidual)
    {
      stageOrder.push_back(2);
      copyVector(fineResidual, coarseResidual);
    };
    fineLevel.prolongateCorrection = [&](const TVector<double> &coarseCorrection, TVector<double> &fineCorrection)
    {
      stageOrder.push_back(4);
      copyVector(coarseCorrection, fineCorrection);
    };
    fineLevel.postSmooth = [&](const TVector<double> &, TVector<double> &solution)
    {
      stageOrder.push_back(5);
      solution.scale(0.5);
    };

    MatrixFreeGMGPreconditioner::Level coarseLevel;
    coarseLevel.map = coarseMap;
    coarseLevel.coarseSolve = [&](const TVector<double> &rhs, TVector<double> &solution)
    {
      stageOrder.push_back(3);
      scaleVector(rhs, solution, 2.0);
    };

    MatrixFreeGMGPreconditioner preconditioner({fineLevel, coarseLevel});

    TVector<double> rhs(fineMap, 1);
    TVector<double> y(fineMap, 1);
    fillVector(rhs, 8.0);
    fillVector(y, 0.0);

    preconditioner.apply(rhs, y);

    const std::vector<int> expectedOrder = {0, 1, 2, 3, 4, 1, 5, 1};
    TEST_EQUALITY(stageOrder.size(), expectedOrder.size());
    if (stageOrder.size() == expectedOrder.size())
    {
      for (size_t i=0; i<expectedOrder.size(); i++)
        TEST_EQUALITY(stageOrder[i], expectedOrder[i]);
    }

    auto yData = y.getData(0);
    TEST_FLOATING_EQUALITY(yData[0], 7.0, 1e-14); // 0.5 * (0.25*8 + 2*(8 - 0.25*8))
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, MatrixFreeGMGPreconditionerStageTraceRecordsGMGComparableNorms )
  {
    auto fineMap = serialMap(1);
    auto coarseMap = serialMap(1);

    MatrixFreeGMGPreconditioner::Level fineLevel;
    fineLevel.map = fineMap;
    fineLevel.preSmooth = [](const TVector<double> &rhs, TVector<double> &solution)
    {
      scaleVector(rhs, solution, 0.25);
    };
    fineLevel.applyOperator = [](const TVector<double> &x, TVector<double> &Ax)
    {
      copyVector(x, Ax);
    };
    fineLevel.restrictResidual = [](const TVector<double> &fineResidual, TVector<double> &coarseResidual)
    {
      copyVector(fineResidual, coarseResidual);
    };
    fineLevel.prolongateCorrection = [](const TVector<double> &coarseCorrection, TVector<double> &fineCorrection)
    {
      copyVector(coarseCorrection, fineCorrection);
    };
    fineLevel.postSmooth = [](const TVector<double> &, TVector<double> &solution)
    {
      solution.scale(0.5);
    };

    MatrixFreeGMGPreconditioner::Level coarseLevel;
    coarseLevel.map = coarseMap;
    coarseLevel.coarseSolve = [](const TVector<double> &rhs, TVector<double> &solution)
    {
      scaleVector(rhs, solution, 2.0);
    };

    MatrixFreeGMGPreconditioner preconditioner({fineLevel, coarseLevel});
    preconditioner.setEmitStageTrace(true);

    TVector<double> rhs(fineMap, 1);
    TVector<double> y(fineMap, 1);
    fillVector(rhs, 8.0);
    fillVector(y, 0.0);

    preconditioner.apply(rhs, y);

    const auto &trace = preconditioner.lastStageTrace();
    const std::vector<std::string> expectedNames = {
      "initialResidualNorm",
      "residualAfterPreSmooth",
      "restrictedResidualNorm",
      "CoarseSolve",
      "coarseCorrectionNorm",
      "prolongateCorrection",
      "residualAfterCoarseCorrection",
      "residualAfterPostSmooth"
    };
    TEST_EQUALITY(trace.size(), expectedNames.size());
    if (trace.size() == expectedNames.size())
    {
      for (size_t i=0; i<expectedNames.size(); i++)
        TEST_EQUALITY(trace[i].stageName, expectedNames[i]);

      TEST_EQUALITY(trace[0].levelIndex, 0);
      TEST_EQUALITY(trace[3].levelIndex, 1);
      TEST_FLOATING_EQUALITY(trace[0].residualInfNorm, 8.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[1].solutionInfNorm, 2.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[1].residualInfNorm, 6.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[2].residualInfNorm, 6.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[3].solutionInfNorm, 12.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[4].solutionInfNorm, 12.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[6].solutionInfNorm, 14.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[6].residualInfNorm, 6.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[7].solutionInfNorm, 7.0, 1e-14);
      TEST_FLOATING_EQUALITY(trace[7].residualInfNorm, 1.0, 1e-14);
    }

    preconditioner.setEmitStageTrace(false);
    preconditioner.apply(rhs, y);
    TEST_EQUALITY(preconditioner.lastStageTrace().size(), 0);
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, MatrixFreeGMGPreconditionerOperatorBackedLevelAdaptersDriveVcycle )
  {
    auto fineMap = serialMap(1);
    auto coarseMap = serialMap(1);

    auto fineOperator = Teuchos::rcp(new ScalingOperator(fineMap, fineMap, 1.0));
    auto preSmoother = Teuchos::rcp(new ScalingOperator(fineMap, fineMap, 0.25));
    auto postSmoother = Teuchos::rcp(new ScalingOperator(fineMap, fineMap, -0.5));
    auto restriction = Teuchos::rcp(new ScalingOperator(fineMap, coarseMap, 1.0));
    auto prolongation = Teuchos::rcp(new ScalingOperator(coarseMap, fineMap, 1.0));
    auto coarseSolver = Teuchos::rcp(new ScalingOperator(coarseMap, coarseMap, 2.0));

    auto fineLevel = MatrixFreeGMGPreconditioner::operatorBackedLevel(fineMap,
                                                                      fineOperator,
                                                                      preSmoother,
                                                                      postSmoother,
                                                                      restriction,
                                                                      prolongation);
    auto coarseLevel = MatrixFreeGMGPreconditioner::operatorBackedCoarseLevel(coarseMap, coarseSolver);

    MatrixFreeGMGPreconditioner preconditioner({fineLevel, coarseLevel});

    TVector<double> rhs(fineMap, 1);
    TVector<double> y(fineMap, 1);
    fillVector(rhs, 8.0);
    fillVector(y, 0.0);

    preconditioner.apply(rhs, y);

    TEST_FLOATING_EQUALITY(vectorValue(y), 10.0, 1e-14); // 0.25*8 + 2*(8 - 0.25*8) - 0.5*8
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, MatrixFreeGMGPreconditionerRejectsIncompleteLevel )
  {
    MatrixFreeGMGPreconditioner::Level level;
    level.map = serialMap(1);

    TEST_THROW(MatrixFreeGMGPreconditioner preconditioner({level}), std::invalid_argument);

    MatrixFreeGMGPreconditioner::ConstOperatorPtr nullOperator;
    TEST_THROW(MatrixFreeGMGPreconditioner::operatorApply(nullOperator), std::invalid_argument);
    TEST_THROW(MatrixFreeGMGPreconditioner::operatorAdditiveUpdate(nullOperator), std::invalid_argument);
    TEST_THROW(MatrixFreeGMGPreconditioner::operatorBackedCoarseLevel(Teuchos::null, nullOperator), std::invalid_argument);
  }
}
