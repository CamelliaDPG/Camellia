#ifndef Camellia_TensorFunctionEvaluationContext_h
#define Camellia_TensorFunctionEvaluationContext_h

#include <map>
#include <string>
#include <vector>

#include "TensorFixedPointLifecycle.h"
#include "TensorFunction.h"

namespace Camellia
{
  class TensorSolutionCoefficientEvaluator
  {
  public:
    virtual ~TensorSolutionCoefficientEvaluator() {}

    virtual void values(int componentOrdinal,
                        TensorFunction::PointViewHost pointsIn,
                        TensorFunction::ValueViewHost valuesOut) const = 0;
  };

  using TensorSolutionCoefficientEvaluatorPtr = Teuchos::RCP<TensorSolutionCoefficientEvaluator>;

  /**
   * \brief TensorSolutionCoefficientEvaluator backed by a production Camellia Function.
   *
   * This adapter is intentionally host-side and scalar for the first Maxwell/Vlasov coupling
   * path.  Mesh-dependent Function subclasses, including Function::solution(), should be
   * constructed with the owning Mesh so evaluation follows the production solution lookup path
   * rather than the test-only coordinate-placeholder evaluator.
   */
  class TensorLegacyFunctionSolutionCoefficientEvaluator : public TensorSolutionCoefficientEvaluator
  {
    FunctionPtr _function;
    MeshPtr _mesh;
    int _physicalDimension;
  public:
    TensorLegacyFunctionSolutionCoefficientEvaluator(FunctionPtr function,
                                                     MeshPtr mesh = Teuchos::null,
                                                     int physicalDimension = 1);

    void values(int componentOrdinal,
                TensorFunction::PointViewHost pointsIn,
                TensorFunction::ValueViewHost valuesOut) const override;
  };

  class TensorFunctionEvaluationContext
  {
    TensorFixedPointLifecycle _lifecycle;
    int _fixedPointSubstepOrdinal = 0;
    long long _configurationSolutionEpoch = 0;
    std::map<std::string, TensorSolutionCoefficientEvaluatorPtr> _solutionCoefficientEvaluators;
    std::map<std::string, std::vector<double>> _coefficientCache;

    static std::string sourceKey(const TensorFunctionDependencySource &source);
    static std::string pointsFingerprint(TensorFunction::PointViewHost pointsIn);
    static bool directSolutionCoefficientComponent(TensorFunctionPtr function, int &componentOrdinal, TensorFunctionDependencySource &source);
    std::string cacheKey(const TensorFunctionDependencySource &source, int componentOrdinal, TensorFunction::PointViewHost pointsIn) const;
    void evaluateExpression(TensorFunctionPtr function,
                            TensorFunction::PointViewHost pointsIn,
                            TensorFunction::ValueViewHost valuesOut);

  public:
    TensorFunctionEvaluationContext();
    explicit TensorFunctionEvaluationContext(const TensorFixedPointLifecycle &lifecycle);

    const TensorFixedPointLifecycle &lifecycle() const;
    int fixedPointSubstepOrdinal() const;
    long long configurationSolutionEpoch() const;

    void registerSolutionCoefficientEvaluator(const TensorFunctionDependencySource &source,
                                              TensorSolutionCoefficientEvaluatorPtr evaluator);
    bool hasSolutionCoefficientEvaluator(const TensorFunctionDependencySource &source) const;

    void beginFixedPointSubstep(int substepOrdinal);
    void configurationSolutionChanged();
    void clearCoefficientCache();
    int cachedCoefficientEntryCount() const;

    void values(TensorFunctionPtr function,
                TensorFunction::PointViewHost pointsIn,
                TensorFunction::ValueViewHost valuesOut);
  };
}

#endif
