#include "TensorOperator.h"
#include "TensorOperatorFactory.h"

#include <algorithm>
#include <stdexcept>

#include "Basis.h"
#include "BasisFamily.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "Intrepid2_CubatureDirectLineGauss.hpp"
#include "TensorFunction.h"
#include "Var.h"
#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  struct LineQuadratureRule
  {
    std::vector<double> points;
    std::vector<double> weights;
  };

  LineQuadratureRule lineQuadratureFromCubatureFactory(int exactnessDegree)
  {
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), exactnessDegree);
    
    const int pointCount = cubature->getNumPoints();
    auto points = cubature->allocateCubaturePoints();
    auto weights = cubature->allocateCubatureWeights();
    
    cubature->getCubature(points, weights);
    
    auto pointsView  = points.getTensorComponent(0);
    auto weightsView = weights.getTensorComponent(0).getUnderlyingView(); // 1D: just one component

    auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pointsView);
    auto weightsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weightsView);

    LineQuadratureRule rule;
    rule.points.resize(pointCount);
    rule.weights.resize(pointCount);
    for (int p=0; p<pointCount; p++)
    {
      rule.points[p] = pointsHost(p,0);
      rule.weights[p] = weightsHost(p);
    }
    return rule;
  }

  class MatrixFreeTensorOperator : public TensorOperator
  {
    struct BasisData1D
    {
      BasisPtr basis;
      std::vector<double> valuesAtQuadrature;      // flattened as (field, point)
      std::vector<double> derivativesAtQuadrature; // flattened as (field, point)
      int fieldCount = 0;
      int pointCount = 0;
    };

    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorBilinearFormPtr _bilinearForm;
    TensorSpaceRole _domainRole;
    TensorSpaceRole _rangeRole;
    std::string _domainVarName;
    std::string _rangeVarName;
    TensorLocalOrdering _domainOrdering;
    TensorLocalOrdering _rangeOrdering;
    TensorGatherScatter _domainGatherScatter;
    TensorGatherScatter _rangeGatherScatter;
    TensorApplyPipelineDescription _pipeline;

    void validateInputVector(const std::vector<double> &input) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != domainSize(), std::invalid_argument,
                                 "MatrixFreeTensorOperator::apply input size does not match domain size.");
    }

    int quadratureDegreeForComponent(int trialPolyOrder,
                                     int testPolyOrder,
                                     bool trialDerivative,
                                     bool testDerivative) const
    {
      int degree = trialPolyOrder + testPolyOrder;
      if (trialDerivative) degree -= 1;
      if (testDerivative) degree -= 1;
      return std::max(1, degree + 1);
    }

    BasisData1D buildBasisDataForComponent(const TensorVarDiscretizationPtr &discretization,
                                           int componentOrdinal,
                                           bool isTest,
                                           const ComponentCellID1D &cellID,
                                           const std::vector<double> &quadraturePoints) const
    {
      BasisData1D basisData;
      auto basisFamily = _tensorMesh->effectiveBasisFamily(discretization);
      VarPtr componentVar = discretization->tensorVar()->componentVar(componentOrdinal);
      const int polyOrder = _tensorMesh->effectivePolyOrder(discretization, cellID, componentOrdinal, isTest);
      basisData.basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(componentVar->space()), polyOrder);

      basisData.fieldCount = basisData.basis->getCardinality();
      basisData.pointCount = static_cast<int>(quadraturePoints.size());
      basisData.valuesAtQuadrature.assign(basisData.fieldCount * basisData.pointCount, 0.0);
      basisData.derivativesAtQuadrature.assign(basisData.fieldCount * basisData.pointCount, 0.0);

      Camellia::FieldContainer<double> refPointContainer(basisData.pointCount, 1);
      for (int p=0; p<basisData.pointCount; p++)
      {
        refPointContainer(p,0) = quadraturePoints[p];
      }

      Camellia::FieldContainer<double> basisValues(basisData.fieldCount, basisData.pointCount);
      basisData.basis->getValues(basisValues, refPointContainer, OP_VALUE);
      for (int f=0; f<basisData.fieldCount; f++)
      {
        for (int p=0; p<basisData.pointCount; p++)
        {
          basisData.valuesAtQuadrature[f * basisData.pointCount + p] = basisValues(f,p);
        }
      }

      Camellia::FieldContainer<double> basisDerivatives(basisData.fieldCount, basisData.pointCount, 1);
      basisData.basis->getValues(basisDerivatives, refPointContainer, OP_GRAD);
      for (int f=0; f<basisData.fieldCount; f++)
      {
        for (int p=0; p<basisData.pointCount; p++)
        {
          basisData.derivativesAtQuadrature[f * basisData.pointCount + p] = basisDerivatives(f,p,0);
        }
      }

      return basisData;
    }

    double basisEvaluationAtPoint(const std::vector<int> &componentBasisOrdinals,
                                  const std::vector<BasisData1D> &basisData,
                                  const TensorComponentOperatorDescriptor &descriptor,
                                  const std::vector<int> &pointOrdinals,
                                  const std::vector<double> &derivativeScales) const
    {
      double value = 1.0;
      const int componentCount = descriptor.componentCount();
      for (int d=0; d<componentCount; d++)
      {
        const int basisOrdinal = componentBasisOrdinals[d];
        const int pointOrdinal = pointOrdinals[d];
        const int flatIndex = basisOrdinal * basisData[d].pointCount + pointOrdinal;
        if (descriptor.takesDerivativeInComponent(d))
        {
          value *= basisData[d].derivativesAtQuadrature[flatIndex] * derivativeScales[d];
        }
        else
        {
          value *= basisData[d].valuesAtQuadrature[flatIndex];
        }
      }
      return value;
    }

    void decodeDomainOrdinal(IndexType ordinal, std::vector<int> &componentBasisOrdinals) const
    {
      _tensorDofLayout->decodeLocalDofOrdinal(_domainVarName, _domainRole, _domainOrdering, ordinal, componentBasisOrdinals);
    }

    void decodeRangeOrdinal(IndexType ordinal, std::vector<int> &componentBasisOrdinals) const
    {
      _tensorDofLayout->decodeLocalDofOrdinal(_rangeVarName, _rangeRole, _rangeOrdering, ordinal, componentBasisOrdinals);
    }

    void applyTerm(const TensorBilinearTerm &term,
                   const std::vector<double> &localInput,
                   std::vector<double> &localOutput) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!term.isVolumeTerm(), std::invalid_argument,
                                 "MatrixFreeTensorOperator currently supports only volume terms.");

      auto trialDiscretization = _tensorMesh->trialDiscretization(_domainVarName);
      auto testDiscretization  = _tensorMesh->testDiscretization(_rangeVarName);
      const auto &domainLayout = _tensorDofLayout->layout(_domainVarName, _domainRole);
      const auto &rangeLayout  = _tensorDofLayout->layout(_rangeVarName,  _rangeRole);
      const auto &activeCellTuples = _tensorMesh->topology()->activeCellTuples();
      const int componentCount = term.componentCount();

      std::vector<int> trialBasisOrdinals(componentCount, 0);
      std::vector<int> testBasisOrdinals(componentCount, 0);
      std::vector<int> pointOrdinals(componentCount, 0);

      for (const auto &cellTuple : activeCellTuples)
      {
        std::vector<BasisData1D> trialBasisData(componentCount);
        std::vector<BasisData1D> testBasisData(componentCount);
        std::vector<int> pointCounts(componentCount, 0);
        std::vector<std::vector<double>> refPoints(componentCount);
        std::vector<std::vector<double>> refWeights(componentCount);
        std::vector<double> measureScales(componentCount, 1.0);
        std::vector<double> derivativeScales(componentCount, 1.0);
        std::vector<double> cellLeft(componentCount, 0.0);
        std::vector<double> cellWidth(componentCount, 1.0);

        for (int d=0; d<componentCount; d++)
        {
          const int trialPolyOrder = _tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
          const int testPolyOrder  = _tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
          const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
          const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
          const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                                    trialDerivative, testDerivative);

          LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
          refPoints[d] = quadrature.points;
          refWeights[d] = quadrature.weights;
          pointCounts[d] = static_cast<int>(quadrature.points.size());

          trialBasisData[d] = buildBasisDataForComponent(trialDiscretization, d, false, cellTuple[d], refPoints[d]);
          testBasisData[d]  = buildBasisDataForComponent(testDiscretization,  d, true,  cellTuple[d], refPoints[d]);

          auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
          cellLeft[d] = componentTopo->cellLeft(cellTuple[d]);
          cellWidth[d] = componentTopo->cellWidth(cellTuple[d]);
          measureScales[d] = 0.5 * cellWidth[d];
          derivativeScales[d] = 2.0 / cellWidth[d];
        }

        int tensorPointCount = 1;
        for (int d=0; d<componentCount; d++) tensorPointCount *= pointCounts[d];

        TensorFunction::PointView physicalPoints("MatrixFreeTensorOperator::physicalPoints", tensorPointCount, componentCount);
        auto physicalPointsHost = Kokkos::create_mirror_view(physicalPoints);
        std::vector<double> quadratureWeights(tensorPointCount, 1.0);

        for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
        {
          int workingOrdinal = flatPointOrdinal;
          double weight = 1.0;
          for (int d=0; d<componentCount; d++)
          {
            pointOrdinals[d] = workingOrdinal % pointCounts[d];
            workingOrdinal /= pointCounts[d];
            const double xi = refPoints[d][pointOrdinals[d]];
            const double xPhysical = cellLeft[d] + 0.5 * (xi + 1.0) * cellWidth[d];
            physicalPointsHost(flatPointOrdinal, d) = xPhysical;
            weight *= refWeights[d][pointOrdinals[d]] * measureScales[d];
          }
          quadratureWeights[flatPointOrdinal] = weight;
        }
        Kokkos::deep_copy(physicalPoints, physicalPointsHost);

        TensorFunction::ValueView coefficientValues("MatrixFreeTensorOperator::coefficientValues", tensorPointCount);
        term.coefficient()->values(physicalPoints, coefficientValues);
        auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

        std::vector<double> trialPointValues(tensorPointCount, 0.0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<domainLayout.totalDofs(); localTrialOrdinal++)
        {
          decodeDomainOrdinal(localTrialOrdinal, trialBasisOrdinals);
          const double trialCoefficient = localInput[localTrialOrdinal];
          if (trialCoefficient == 0.0) continue;

          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double trialValue = basisEvaluationAtPoint(trialBasisOrdinals,
                                                             trialBasisData,
                                                             term.trialOperator(),
                                                             pointOrdinals,
                                                             derivativeScales);
            trialPointValues[flatPointOrdinal] += trialCoefficient * trialValue;
          }
        }

        for (IndexType localTestOrdinal=0; localTestOrdinal<rangeLayout.totalDofs(); localTestOrdinal++)
        {
          decodeRangeOrdinal(localTestOrdinal, testBasisOrdinals);

          double contribution = 0.0;
          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double testValue = basisEvaluationAtPoint(testBasisOrdinals,
                                                            testBasisData,
                                                            term.testOperator(),
                                                            pointOrdinals,
                                                            derivativeScales);
            contribution += quadratureWeights[flatPointOrdinal]
                            * coefficientValuesHost(flatPointOrdinal)
                            * trialPointValues[flatPointOrdinal]
                            * testValue;
          }
          localOutput[localTestOrdinal] += contribution;
        }
      }
    }

  public:
    MatrixFreeTensorOperator(const TensorMeshPtr &tensorMesh,
                             const TensorDofLayoutPtr &tensorDofLayout,
                             const TensorBilinearFormPtr &bilinearForm,
                             TensorSpaceRole domainRole,
                             const std::string &domainVarName,
                             TensorSpaceRole rangeRole,
                             const std::string &rangeVarName,
                             TensorLocalOrdering domainOrdering = TENSOR_VARIABLE_ORDERING,
                             TensorLocalOrdering rangeOrdering = TENSOR_VARIABLE_ORDERING)
    : _tensorMesh(tensorMesh),
      _tensorDofLayout(tensorDofLayout),
      _bilinearForm(bilinearForm),
      _domainRole(domainRole),
      _rangeRole(rangeRole),
      _domainVarName(domainVarName),
      _rangeVarName(rangeVarName),
      _domainOrdering(domainOrdering),
      _rangeOrdering(rangeOrdering),
      _domainGatherScatter(tensorMesh, tensorDofLayout, domainRole, domainVarName, domainOrdering),
      _rangeGatherScatter(tensorMesh, tensorDofLayout, rangeRole, rangeVarName, rangeOrdering),
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
  if (bilinearForm->hasBoundaryTerms()) return false;

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
