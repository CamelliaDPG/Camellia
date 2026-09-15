#include "TensorFunctionEvaluationContext.h"

#include <cmath>

#include "Function.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorFunctionEvaluationContext::TensorFunctionEvaluationContext()
: _lifecycle()
{
}

TensorFunctionEvaluationContext::TensorFunctionEvaluationContext(const TensorFixedPointLifecycle &lifecycle)
: _lifecycle(lifecycle)
{
}

TensorLegacyFunctionSolutionCoefficientEvaluator::TensorLegacyFunctionSolutionCoefficientEvaluator(FunctionPtr function,
                                                                                                   MeshPtr mesh,
                                                                                                   int physicalDimension)
: _function(function),
  _mesh(mesh),
  _physicalDimension(physicalDimension)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_function == Teuchos::null, std::invalid_argument,
                             "TensorLegacyFunctionSolutionCoefficientEvaluator requires a non-null Function.");
  TEUCHOS_TEST_FOR_EXCEPTION((_physicalDimension < 1) || (_physicalDimension > 3), std::invalid_argument,
                             "TensorLegacyFunctionSolutionCoefficientEvaluator currently supports one, two, or three physical coordinates.");
}

void TensorLegacyFunctionSolutionCoefficientEvaluator::values(int componentOrdinal,
                                                              TensorFunction::PointViewHost pointsIn,
                                                              TensorFunction::ValueViewHost valuesOut) const
{
  (void) componentOrdinal;
  const int pointCount = static_cast<int>(pointsIn.extent(0));
  const int pointDim = static_cast<int>(pointsIn.extent(1));
  TEUCHOS_TEST_FOR_EXCEPTION(pointDim < _physicalDimension, std::invalid_argument,
                             "TensorLegacyFunctionSolutionCoefficientEvaluator point dimension is smaller than the requested physical dimension.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(valuesOut.extent(0)) != pointCount, std::invalid_argument,
                             "TensorLegacyFunctionSolutionCoefficientEvaluator values output length does not match point count.");

  for (int p=0; p<pointCount; p++)
  {
    if (_mesh != Teuchos::null)
    {
      switch (_physicalDimension)
      {
        case 1: valuesOut(p) = _function->evaluate(_mesh, pointsIn(p,0)); break;
        case 2: valuesOut(p) = _function->evaluate(_mesh, pointsIn(p,0), pointsIn(p,1)); break;
        case 3: valuesOut(p) = _function->evaluate(_mesh, pointsIn(p,0), pointsIn(p,1), pointsIn(p,2)); break;
      }
    }
    else
    {
      switch (_physicalDimension)
      {
        case 1: valuesOut(p) = _function->evaluate(pointsIn(p,0)); break;
        case 2: valuesOut(p) = _function->evaluate(pointsIn(p,0), pointsIn(p,1)); break;
        case 3: valuesOut(p) = _function->evaluate(pointsIn(p,0), pointsIn(p,1), pointsIn(p,2)); break;
      }
    }
  }
}

const TensorFixedPointLifecycle &TensorFunctionEvaluationContext::lifecycle() const
{
  return _lifecycle;
}

int TensorFunctionEvaluationContext::fixedPointSubstepOrdinal() const
{
  return _fixedPointSubstepOrdinal;
}

long long TensorFunctionEvaluationContext::configurationSolutionEpoch() const
{
  return _configurationSolutionEpoch;
}

std::string TensorFunctionEvaluationContext::sourceKey(const TensorFunctionDependencySource &source)
{
  std::ostringstream os;
  os << static_cast<int>(source.sourceType()) << ":" << source.name();
  return os.str();
}

std::string TensorFunctionEvaluationContext::pointsFingerprint(TensorFunction::PointViewHost pointsIn)
{
  std::ostringstream os;
  os << std::setprecision(17);
  const int pointCount = static_cast<int>(pointsIn.extent(0));
  const int pointDim = static_cast<int>(pointsIn.extent(1));
  os << pointCount << "x" << pointDim;
  for (int p=0; p<pointCount; p++)
  {
    for (int d=0; d<pointDim; d++)
    {
      os << ";" << pointsIn(p,d);
    }
  }
  return os.str();
}

bool TensorFunctionEvaluationContext::directSolutionCoefficientComponent(TensorFunctionPtr function,
                                                                         int &componentOrdinal,
                                                                         TensorFunctionDependencySource &source)
{
  if (function == Teuchos::null) return false;
  if (!function->hasSolutionDependencies()) return false;
  if (function->dependencySources().size() != 1) return false;
  if (!function->isAtomic()) return false;

  const auto &components = function->dependencyScope().componentOrdinals();
  if (components.size() != 1) return false;

  componentOrdinal = *components.begin();
  source = function->dependencySources()[0];
  return true;
}

std::string TensorFunctionEvaluationContext::cacheKey(const TensorFunctionDependencySource &source,
                                                      int componentOrdinal,
                                                      TensorFunction::PointViewHost pointsIn) const
{
  std::ostringstream os;
  os << sourceKey(source) << ":component=" << componentOrdinal
     << ":configurationEpoch=" << _configurationSolutionEpoch;
  if (_lifecycle.cachesAreScopedToFixedPointSubstep())
  {
    os << ":substep=" << _fixedPointSubstepOrdinal;
  }
  os << ":points=" << pointsFingerprint(pointsIn);
  return os.str();
}

void TensorFunctionEvaluationContext::registerSolutionCoefficientEvaluator(const TensorFunctionDependencySource &source,
                                                                           TensorSolutionCoefficientEvaluatorPtr evaluator)
{
  TEUCHOS_TEST_FOR_EXCEPTION(evaluator == Teuchos::null, std::invalid_argument,
                             "TensorFunctionEvaluationContext requires a non-null solution coefficient evaluator.");
  _solutionCoefficientEvaluators[sourceKey(source)] = evaluator;
  clearCoefficientCache();
}

bool TensorFunctionEvaluationContext::hasSolutionCoefficientEvaluator(const TensorFunctionDependencySource &source) const
{
  return _solutionCoefficientEvaluators.find(sourceKey(source)) != _solutionCoefficientEvaluators.end();
}

void TensorFunctionEvaluationContext::beginFixedPointSubstep(int substepOrdinal)
{
  TEUCHOS_TEST_FOR_EXCEPTION(substepOrdinal < 0, std::invalid_argument,
                             "TensorFunctionEvaluationContext fixed-point substep ordinal must be nonnegative.");
  if (_fixedPointSubstepOrdinal != substepOrdinal)
  {
    _fixedPointSubstepOrdinal = substepOrdinal;
    if (_lifecycle.cachesAreScopedToFixedPointSubstep()) clearCoefficientCache();
  }
}

void TensorFunctionEvaluationContext::configurationSolutionChanged()
{
  _configurationSolutionEpoch++;
  if (_lifecycle.invalidateCoefficientCachesWhenConfigurationSolutionChanges()) clearCoefficientCache();
}

void TensorFunctionEvaluationContext::clearCoefficientCache()
{
  _coefficientCache.clear();
}

int TensorFunctionEvaluationContext::cachedCoefficientEntryCount() const
{
  return static_cast<int>(_coefficientCache.size());
}

void TensorFunctionEvaluationContext::evaluateExpression(TensorFunctionPtr function,
                                                         TensorFunction::PointViewHost pointsIn,
                                                         TensorFunction::ValueViewHost valuesOut)
{
  const int pointCount = static_cast<int>(pointsIn.extent(0));

  int componentOrdinal = -1;
  TensorFunctionDependencySource source(TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "unused");
  if (directSolutionCoefficientComponent(function, componentOrdinal, source))
  {
    const std::string evaluatorKey = sourceKey(source);
    auto evaluatorIt = _solutionCoefficientEvaluators.find(evaluatorKey);
    TEUCHOS_TEST_FOR_EXCEPTION(evaluatorIt == _solutionCoefficientEvaluators.end(), std::invalid_argument,
                               "TensorFunctionEvaluationContext has no evaluator registered for the requested solution coefficient source.");

    const std::string key = cacheKey(source, componentOrdinal, pointsIn);
    auto cacheIt = _coefficientCache.find(key);
    if (cacheIt == _coefficientCache.end())
    {
      TensorFunction::ValueViewHost scratch("TensorFunctionEvaluationContext cached coefficient values", pointCount);
      evaluatorIt->second->values(componentOrdinal, pointsIn, scratch);
      std::vector<double> cached(pointCount);
      for (int p=0; p<pointCount; p++) cached[p] = scratch(p);
      cacheIt = _coefficientCache.insert(std::make_pair(key, cached)).first;
    }
    for (int p=0; p<pointCount; p++) valuesOut(p) = cacheIt->second[p];
    return;
  }

  if (!function->hasSolutionDependencies())
  {
    function->values(pointsIn, valuesOut);
    return;
  }

  const auto children = function->childFunctions();
  if (function->isSum())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(children.size() != 2, std::invalid_argument,
                               "TensorFunctionEvaluationContext expected two children for a sum TensorFunction.");
    TensorFunction::ValueViewHost leftValues("TensorFunctionEvaluationContext sum left values", pointCount);
    evaluateExpression(children[0], pointsIn, leftValues);
    evaluateExpression(children[1], pointsIn, valuesOut);
    for (int p=0; p<pointCount; p++) valuesOut(p) += leftValues(p);
    return;
  }

  if (function->isProduct())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(children.size() != 2, std::invalid_argument,
                               "TensorFunctionEvaluationContext expected two children for a product TensorFunction.");
    TensorFunction::ValueViewHost rightValues("TensorFunctionEvaluationContext product right values", pointCount);
    evaluateExpression(children[0], pointsIn, valuesOut);
    evaluateExpression(children[1], pointsIn, rightValues);
    for (int p=0; p<pointCount; p++) valuesOut(p) *= rightValues(p);
    return;
  }

  if (function->isUnaryComposition())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(children.size() != 1, std::invalid_argument,
                               "TensorFunctionEvaluationContext expected one child for a unary TensorFunction.");
    evaluateExpression(children[0], pointsIn, valuesOut);
    switch (function->unaryFunctionType())
    {
      case TENSOR_UNARY_EXP:
        for (int p=0; p<pointCount; p++) valuesOut(p) = std::exp(valuesOut(p));
        return;
      case TENSOR_UNARY_SIN:
        for (int p=0; p<pointCount; p++) valuesOut(p) = std::sin(valuesOut(p));
        return;
      case TENSOR_UNARY_COS:
        for (int p=0; p<pointCount; p++) valuesOut(p) = std::cos(valuesOut(p));
        return;
      case TENSOR_UNARY_SQRT:
        for (int p=0; p<pointCount; p++) valuesOut(p) = std::sqrt(valuesOut(p));
        return;
      case TENSOR_UNARY_RECIPROCAL:
        for (int p=0; p<pointCount; p++) valuesOut(p) = 1.0 / valuesOut(p);
        return;
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "TensorFunctionEvaluationContext does not understand this solution-dependent TensorFunction node; refusing placeholder evaluation.");
}

void TensorFunctionEvaluationContext::values(TensorFunctionPtr function,
                                             TensorFunction::PointViewHost pointsIn,
                                             TensorFunction::ValueViewHost valuesOut)
{
  TEUCHOS_TEST_FOR_EXCEPTION(function == Teuchos::null, std::invalid_argument,
                             "TensorFunctionEvaluationContext cannot evaluate a null TensorFunction.");
  const int pointCount = static_cast<int>(pointsIn.extent(0));
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(valuesOut.extent(0)) != pointCount, std::invalid_argument,
                             "TensorFunctionEvaluationContext valuesOut extent must match pointsIn point count.");

  evaluateExpression(function, pointsIn, valuesOut);
}
