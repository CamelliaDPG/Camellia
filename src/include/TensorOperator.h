#ifndef Camellia_TensorOperator_h
#define Camellia_TensorOperator_h

#include <string>
#include <vector>

#include "TensorBilinearForm.h"
#include "TensorGatherScatter.h"

namespace Camellia
{
  /**
   *  \brief Host-side Phase 0 instrumentation for tensor-native operator apply work.
   *
   *  The counters are intentionally lightweight and passive: operator implementations
   *  can add elapsed wall-clock seconds for each stage while callers that do not query
   *  instrumentation continue to see the same apply() behavior.
   */
  class TensorOperatorInstrumentation
  {
  public:
    enum Stage
    {
      BASIS_CACHE_CONSTRUCTION = 0,
      COEFFICIENT_EVALUATION,
      CONTRACTION_1D_CONSTRUCTION,
      LOCAL_ACTION,
      SCATTER_LOCAL_OUTPUT_ACCUMULATION,
      STAGE_COUNT
    };

    struct StageStats
    {
      long long count = 0;
      double seconds = 0.0;
    };

  private:
    StageStats _stageStats[STAGE_COUNT];
    long long _sumFactorizedVolumeTermCount = 0;
    long long _fullTensorVolumeTermCount = 0;
    long long _graphNormSchurApplyCount = 0;
    long long _graphNormSchurActiveTrialDofCount = 0;

  public:
    TensorOperatorInstrumentation()
    {
      reset();
    }

    void reset()
    {
      for (int stageOrdinal=0; stageOrdinal<STAGE_COUNT; stageOrdinal++)
      {
        _stageStats[stageOrdinal].count = 0;
        _stageStats[stageOrdinal].seconds = 0.0;
      }
      _sumFactorizedVolumeTermCount = 0;
      _fullTensorVolumeTermCount = 0;
      _graphNormSchurApplyCount = 0;
      _graphNormSchurActiveTrialDofCount = 0;
    }

    void add(Stage stage,
             double seconds)
    {
      _stageStats[stage].count++;
      _stageStats[stage].seconds += seconds;
    }

    void recordSumFactorizedVolumeTerm()
    {
      _sumFactorizedVolumeTermCount++;
    }

    void recordFullTensorVolumeTerm()
    {
      _fullTensorVolumeTermCount++;
    }

    void recordGraphNormSchurApply()
    {
      _graphNormSchurApplyCount++;
    }

    void recordGraphNormSchurActiveTrialDofs(long long activeTrialDofs)
    {
      _graphNormSchurActiveTrialDofCount += activeTrialDofs;
    }

    StageStats stats(Stage stage) const
    {
      return _stageStats[stage];
    }

    long long count(Stage stage) const
    {
      return _stageStats[stage].count;
    }

    double seconds(Stage stage) const
    {
      return _stageStats[stage].seconds;
    }

    long long sumFactorizedVolumeTermCount() const
    {
      return _sumFactorizedVolumeTermCount;
    }

    long long fullTensorVolumeTermCount() const
    {
      return _fullTensorVolumeTermCount;
    }

    long long graphNormSchurApplyCount() const
    {
      return _graphNormSchurApplyCount;
    }

    long long graphNormSchurActiveTrialDofCount() const
    {
      return _graphNormSchurActiveTrialDofCount;
    }
  };

  /**
   *  \brief Explicit stage labeling for tensor-native operator application.
   *
   *  The initial issue #298 implementation is matrix-free-first. These stage labels make the
   *  intended gather/apply/scatter flow explicit, even before all optimized kernels exist.
   */
  enum TensorApplyStage
  {
    TENSOR_APPLY_STAGE_GATHER,
    TENSOR_APPLY_STAGE_LOCAL_APPLY,
    TENSOR_APPLY_STAGE_SCATTER
  };

  class TensorApplyPipelineDescription
  {
    std::vector<TensorApplyStage> _stages;
  public:
    TensorApplyPipelineDescription();
    TensorApplyPipelineDescription(const std::vector<TensorApplyStage> &stages);

    const std::vector<TensorApplyStage> &stages() const;
    bool containsStage(TensorApplyStage stage) const;

    static TensorApplyPipelineDescription standardMatrixFreePipeline();
  };

  class TensorOperator;
  using TensorOperatorPtr = Teuchos::RCP<TensorOperator>;

  /**
   *  \brief Abstract realized tensor-native operator.
   *
   *  This is the realized side of the descriptor/realization split introduced in issue #298.
   *  The current concrete implementation remains matrix-free-first while allowing explicit
   *  local coefficient ordering choices in the realization path.
   */
  class TensorOperator
  {
  public:
    virtual ~TensorOperator() {}

    virtual TensorOperatorBackend backend() const = 0;
    virtual TensorMeshPtr tensorMesh() const = 0;
    virtual TensorDofLayoutPtr tensorDofLayout() const = 0;
    virtual TensorBilinearFormPtr bilinearForm() const = 0;
    virtual TensorSpaceRole domainRole() const = 0;
    virtual TensorSpaceRole rangeRole() const = 0;
    virtual const std::string &domainVarName() const = 0;
    virtual const std::string &rangeVarName() const = 0;
    virtual IndexType domainSize() const = 0;
    virtual IndexType rangeSize() const = 0;
    virtual TensorApplyPipelineDescription pipelineDescription() const = 0;

    virtual void apply(const std::vector<double> &input,
                       std::vector<double> &output) const = 0;

    virtual void resetInstrumentation() const {}
    virtual TensorOperatorInstrumentation instrumentation() const
    {
      return TensorOperatorInstrumentation();
    }
  };
}

#endif
