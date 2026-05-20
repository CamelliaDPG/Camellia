#ifndef Camellia_TensorOperator_h
#define Camellia_TensorOperator_h

#include <string>
#include <vector>

#include "TensorBilinearForm.h"
#include "TensorGatherScatter.h"

namespace Camellia
{
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
   *  The initial concrete implementation will be matrix-free-first and flat-buffer-first.
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
  };
}

#endif
