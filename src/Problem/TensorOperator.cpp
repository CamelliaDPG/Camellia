#include "TensorOperator.h"
#include "TensorOperatorFactory.h"

#include <algorithm>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  class MatrixFreeTensorOperator : public TensorOperator
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorBilinearFormPtr _bilinearForm;
    TensorSpaceRole _domainRole;
    TensorSpaceRole _rangeRole;
    std::string _domainVarName;
    std::string _rangeVarName;
    TensorGatherScatter _domainGatherScatter;
    TensorGatherScatter _rangeGatherScatter;
    TensorApplyPipelineDescription _pipeline;

    void validateInputVector(const std::vector<double> &input) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != domainSize(), std::invalid_argument,
                                 "MatrixFreeTensorOperator::apply input size does not match domain size.");
    }

    void applyTerm(const TensorBilinearTerm &term,
                   const std::vector<double> &localInput,
                   std::vector<double> &localOutput) const
    {
      const auto &domainLayout = _tensorDofLayout->layout(_domainVarName, _domainRole);
      const auto &rangeLayout  = _tensorDofLayout->layout(_rangeVarName,  _rangeRole);
      const int coupledComponentCount = static_cast<int>(term.coefficient()->dependencyScope().componentOrdinals().size());
      const double coefficientWeight = 1.0 + static_cast<double>(coupledComponentCount);

      const IndexType overlap = std::min(domainLayout.totalDofs(), rangeLayout.totalDofs());
      for (IndexType i=0; i<overlap; i++)
      {
        double value = localInput[i];

        for (int d=0; d<term.componentCount(); d++)
        {
          if (term.trialOperator().takesDerivativeInComponent(d))
          {
            value *= static_cast<double>(domainLayout.componentCardinality(d));
          }
          if (term.testOperator().takesDerivativeInComponent(d))
          {
            value *= static_cast<double>(rangeLayout.componentCardinality(d));
          }
        }

        localOutput[i] += coefficientWeight * value;
      }
    }

  public:
    MatrixFreeTensorOperator(const TensorMeshPtr &tensorMesh,
                             const TensorDofLayoutPtr &tensorDofLayout,
                             const TensorBilinearFormPtr &bilinearForm,
                             TensorSpaceRole domainRole,
                             const std::string &domainVarName,
                             TensorSpaceRole rangeRole,
                             const std::string &rangeVarName)
    : _tensorMesh(tensorMesh),
      _tensorDofLayout(tensorDofLayout),
      _bilinearForm(bilinearForm),
      _domainRole(domainRole),
      _rangeRole(rangeRole),
      _domainVarName(domainVarName),
      _rangeVarName(rangeVarName),
      _domainGatherScatter(tensorMesh, tensorDofLayout, domainRole, domainVarName),
      _rangeGatherScatter(tensorMesh, tensorDofLayout, rangeRole, rangeVarName),
      _pipeline(TensorApplyPipelineDescription::standardMatrixFreePipeline())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorMesh.");
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorDofLayout.");
      TEUCHOS_TEST_FOR_EXCEPTION(_bilinearForm == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorBilinearForm.");
      _bilinearForm->validateAgainst(_tensorMesh);
    }

    TensorOperatorBackend backend() const override
    {
      return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
    }

    TensorMeshPtr tensorMesh() const override
    {
      return _tensorMesh;
    }

    TensorDofLayoutPtr tensorDofLayout() const override
    {
      return _tensorDofLayout;
    }

    TensorBilinearFormPtr bilinearForm() const override
    {
      return _bilinearForm;
    }

    TensorSpaceRole domainRole() const override
    {
      return _domainRole;
    }

    TensorSpaceRole rangeRole() const override
    {
      return _rangeRole;
    }

    const std::string &domainVarName() const override
    {
      return _domainVarName;
    }

    const std::string &rangeVarName() const override
    {
      return _rangeVarName;
    }

    IndexType domainSize() const override
    {
      return _domainGatherScatter.localDofCount();
    }

    IndexType rangeSize() const override
    {
      return _rangeGatherScatter.localDofCount();
    }

    TensorApplyPipelineDescription pipelineDescription() const override
    {
      return _pipeline;
    }

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const override
    {
      validateInputVector(input);

      std::vector<double> localInput;
      _domainGatherScatter.gatherGlobalToLocal(input, localInput, 0);

      std::vector<double> localOutput(rangeSize(), 0.0);
      for (const auto &term : _bilinearForm->terms())
      {
        const bool correctDomain = (term.trialVar()->name() == _domainVarName);
        const bool correctRange  = (term.testVar()->name()  == _rangeVarName);
        if (correctDomain && correctRange)
        {
          applyTerm(term, localInput, localOutput);
        }
      }

      output.assign(rangeSize(), 0.0);
      _rangeGatherScatter.scatterLocalToGlobal(localOutput, output, 0);
    }
  };
}

TensorApplyPipelineDescription::TensorApplyPipelineDescription()
{
}

TensorApplyPipelineDescription::TensorApplyPipelineDescription(const std::vector<TensorApplyStage> &stages)
: _stages(stages)
{
}

const std::vector<TensorApplyStage> &TensorApplyPipelineDescription::stages() const
{
  return _stages;
}

bool TensorApplyPipelineDescription::containsStage(TensorApplyStage stage) const
{
  return _stages.end() != std::find(_stages.begin(), _stages.end(), stage);
}

TensorApplyPipelineDescription TensorApplyPipelineDescription::standardMatrixFreePipeline()
{
  return TensorApplyPipelineDescription({TENSOR_APPLY_STAGE_GATHER,
                                         TENSOR_APPLY_STAGE_LOCAL_APPLY,
                                         TENSOR_APPLY_STAGE_SCATTER});
}

TensorOperatorBackend MatrixFreeTensorOperatorFactory::backend() const
{
  return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
}

bool MatrixFreeTensorOperatorFactory::canBuild(const TensorMeshPtr &tensorMesh,
                                               const TensorDofLayoutPtr &tensorDofLayout,
                                               const TensorBilinearFormPtr &bilinearForm,
                                               TensorSpaceRole domainRole,
                                               const std::string &domainVarName,
                                               TensorSpaceRole rangeRole,
                                               const std::string &rangeVarName) const
{
  if ((tensorMesh == Teuchos::null) || (tensorDofLayout == Teuchos::null) || (bilinearForm == Teuchos::null))
  {
    return false;
  }
  if (!tensorDofLayout->hasLayout(domainVarName, domainRole)) return false;
  if (!tensorDofLayout->hasLayout(rangeVarName, rangeRole)) return false;
  if (!bilinearForm->allTermsSupport(TENSOR_OPERATOR_BACKEND_MATRIX_FREE)) return false;

  try
  {
    bilinearForm->validateAgainst(tensorMesh);
  }
  catch (...)
  {
    return false;
  }

  return true;
}

TensorOperatorPtr MatrixFreeTensorOperatorFactory::build(const TensorMeshPtr &tensorMesh,
                                                         const TensorDofLayoutPtr &tensorDofLayout,
                                                         const TensorBilinearFormPtr &bilinearForm,
                                                         TensorSpaceRole domainRole,
                                                         const std::string &domainVarName,
                                                         TensorSpaceRole rangeRole,
                                                         const std::string &rangeVarName) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!canBuild(tensorMesh, tensorDofLayout, bilinearForm,
                                       domainRole, domainVarName,
                                       rangeRole, rangeVarName),
                             std::invalid_argument,
                             "MatrixFreeTensorOperatorFactory::build called with unsupported inputs.");

  return Teuchos::rcp(new MatrixFreeTensorOperator(tensorMesh,
                                                   tensorDofLayout,
                                                   bilinearForm,
                                                   domainRole,
                                                   domainVarName,
                                                   rangeRole,
                                                   rangeVarName));
}
