#ifndef Camellia_TensorPAMatrixBridge_h
#define Camellia_TensorPAMatrixBridge_h

#include "TensorBilinearForm.h"
#include "TensorFunction.h"
#include "TensorFunctionEvaluationContext.h"
#include "Var.h"

#include <Intrepid2_CellGeometry.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_DerivedBasisFamily.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_HierarchicalBasisFamily.hpp>
#include <Intrepid2_PAMatrix.hpp>

#include <Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
  enum class TensorBasisFamily
  {
    Hierarchical,
    Nodal
  };

  namespace detail
  {
    template<class BasisFamily>
    typename BasisFamily::BasisPtr getHypercubeBasisForPAMatrixBridge(const Intrepid2::EFunctionSpace functionSpace,
                                                                      const int p,
                                                                      const int spaceDim)
    {
      switch (functionSpace)
      {
        case Intrepid2::FUNCTION_SPACE_HGRAD:
          return Intrepid2::getHypercubeBasis_HGRAD<BasisFamily>(p, spaceDim);
        case Intrepid2::FUNCTION_SPACE_HVOL:
          return Intrepid2::getHypercubeBasis_HVOL<BasisFamily>(p, spaceDim);
        default:
          throw std::invalid_argument("Tensor PAMatrix bridge currently routes only HGRAD and scalar HVOL volume bases");
      }
    }

    template<class Scalar, class DeviceType>
    Intrepid2::TransformedBasisValues<Scalar,DeviceType> transformForPAMatrixBridge(const Intrepid2::EFunctionSpace functionSpace,
                                                                                    const Intrepid2::EOperator op,
                                                                                    const Intrepid2::BasisValues<Scalar,DeviceType> &basisValues,
                                                                                    const Intrepid2::Data<Scalar,DeviceType> &jacobian,
                                                                                    const Intrepid2::Data<Scalar,DeviceType> &jacobianInv,
                                                                                    const Intrepid2::Data<Scalar,DeviceType> &jacobianDetInv,
                                                                                    const Intrepid2::Data<Scalar,DeviceType> &jacobianDividedByJacobianDet)
    {
      using FunctionSpaceTools = Intrepid2::FunctionSpaceTools<DeviceType>;
      switch (functionSpace)
      {
        case Intrepid2::FUNCTION_SPACE_HGRAD:
        {
          switch (op)
          {
            case Intrepid2::OPERATOR_VALUE:
              return FunctionSpaceTools::getHGRADtransformVALUE(jacobian.extent_int(0), basisValues);
            case Intrepid2::OPERATOR_GRAD:
              return FunctionSpaceTools::getHGRADtransformGRAD(jacobianInv, basisValues);
            default:
              throw std::invalid_argument("PAMatrix bridge does not support the requested HGRAD operator");
          }
        }
        case Intrepid2::FUNCTION_SPACE_HCURL:
        {
          switch (op)
          {
            case Intrepid2::OPERATOR_VALUE:
              return FunctionSpaceTools::getHCURLtransformVALUE(jacobianInv, basisValues);
            case Intrepid2::OPERATOR_CURL:
            {
              const int spaceDim = jacobian.extent_int(2);
              if (spaceDim == 2)
                return FunctionSpaceTools::getHCURLtransformCURL2D(jacobianDetInv, basisValues);
              return FunctionSpaceTools::getHCURLtransformCURL(jacobianDividedByJacobianDet, basisValues);
            }
            default:
              throw std::invalid_argument("PAMatrix bridge does not support the requested HCURL operator");
          }
        }
        case Intrepid2::FUNCTION_SPACE_HDIV:
        {
          switch (op)
          {
            case Intrepid2::OPERATOR_VALUE:
              return FunctionSpaceTools::getHDIVtransformVALUE(jacobianDividedByJacobianDet, basisValues);
            case Intrepid2::OPERATOR_DIV:
              return FunctionSpaceTools::getHDIVtransformDIV(jacobianDetInv, basisValues);
            default:
              throw std::invalid_argument("PAMatrix bridge does not support the requested HDIV operator");
          }
        }
        case Intrepid2::FUNCTION_SPACE_HVOL:
        {
          switch (op)
          {
            case Intrepid2::OPERATOR_VALUE:
              return FunctionSpaceTools::getHVOLtransformVALUE(jacobianDetInv, basisValues);
            default:
              throw std::invalid_argument("PAMatrix bridge does not support the requested HVOL operator");
          }
        }
        default:
          throw std::invalid_argument("PAMatrix bridge does not support the requested function space");
      }
    }
  }

  /**
   * \brief Minimal tensor-term realization result for the narrow PAMatrix bridge.
   *
   * This intentionally mirrors the performance driver's term list without importing
   * Tpetra or benchmark-specific concepts into the reusable tensor descriptor layer.
   */
  template<class DeviceType, class Scalar>
  struct TensorPAMatrixTerm
  {
    Intrepid2::PAMatrix<DeviceType,Scalar> matrix;
    Scalar weight = Scalar(1.0);
    std::string label;
  };

  /**
   * \brief Convenience constructor for HGRAD tensor field variables used by bridge smokes.
   */
  inline TensorVarPtr makeHGRADTensorFieldVar(const std::string &name, const int componentCount)
  {
    std::vector<VarPtr> components;
    components.reserve(componentCount);
    for (int d=0; d<componentCount; d++)
    {
      components.push_back(Teuchos::rcp(new Var(d, 0, name + "_" + std::to_string(d),
                                                  OP_VALUE, HGRAD, FIELD,
                                                  Teuchos::null, true, UNKNOWN_FS)));
    }
    return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, components));
  }

  /**
   * \brief Convenience constructor for scalar L2/HVOL tensor field variables used by bridge smokes.
   */
  inline TensorVarPtr makeHVOLTensorFieldVar(const std::string &name, const int componentCount)
  {
    std::vector<VarPtr> components;
    components.reserve(componentCount);
    for (int d=0; d<componentCount; d++)
    {
      components.push_back(Teuchos::rcp(new Var(d, 0, name + "_" + std::to_string(d),
                                                  OP_VALUE, L2, FIELD,
                                                  Teuchos::null, true, UNKNOWN_FS)));
    }
    return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, components));
  }

  inline Intrepid2::EFunctionSpace intrepidFunctionSpaceForTensorDescriptor(const TensorComponentOperatorDescriptor &descriptor)
  {
    switch (descriptor.functionSpace())
    {
      case TENSOR_COMPONENT_FUNCTION_SPACE_HGRAD: return Intrepid2::FUNCTION_SPACE_HGRAD;
      case TENSOR_COMPONENT_FUNCTION_SPACE_HCURL: return Intrepid2::FUNCTION_SPACE_HCURL;
      case TENSOR_COMPONENT_FUNCTION_SPACE_HDIV:  return Intrepid2::FUNCTION_SPACE_HDIV;
      case TENSOR_COMPONENT_FUNCTION_SPACE_HVOL:  return Intrepid2::FUNCTION_SPACE_HVOL;
      default: throw std::invalid_argument("PAMatrix bridge does not recognize tensor descriptor function-space metadata");
    }
  }

  inline Intrepid2::EOperator pamatrixOperatorForTensorDescriptor(const TensorComponentOperatorDescriptor &descriptor)
  {
    if (descriptor.isValueOnly()) return Intrepid2::OPERATOR_VALUE;
    if (descriptor.derivativeComponentOrdinals().size() == 1)
    {
      if (descriptor.functionSpace() != TENSOR_COMPONENT_FUNCTION_SPACE_HGRAD)
        throw std::invalid_argument("PAMatrix bridge only supports derivative descriptors for HGRAD variables");
      return Intrepid2::OPERATOR_GRAD;
    }
    throw std::invalid_argument("PAMatrix bridge currently supports only value-only or single-derivative tensor descriptors");
  }

  template<int spaceDim, class Scalar>
  Teuchos::RCP<Kokkos::Array<Scalar,spaceDim>> vectorWeightForSingleDerivative(const TensorComponentOperatorDescriptor &descriptor)
  {
    if (descriptor.derivativeComponentOrdinals().empty()) return Teuchos::null;
    if (descriptor.derivativeComponentOrdinals().size() != 1)
      throw std::invalid_argument("PAMatrix bridge currently supports one derivative component per side");

    const int derivativeComponent = descriptor.derivativeComponentOrdinals()[0];
    if ((derivativeComponent < 0) || (derivativeComponent >= spaceDim))
      throw std::invalid_argument("Tensor derivative component is outside the HGRAD hypercube dimension");

    auto vectorWeight = Teuchos::rcp(new Kokkos::Array<Scalar,spaceDim>());
    for (int d=0; d<spaceDim; d++) (*vectorWeight)[d] = (d == derivativeComponent) ? Scalar(1.0) : Scalar(0.0);
    return vectorWeight;
  }

  template<int spaceDim, class DeviceType, class Scalar>
  Intrepid2::TensorPoints<Scalar,DeviceType> makeHypercubeCubaturePoints(const int degree,
                                                                         Intrepid2::TensorData<Scalar,DeviceType> &tensorCubatureWeights)
  {
    const auto lineTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >());
    auto lineCubature = Intrepid2::DefaultCubatureFactory::create<DeviceType,Scalar,Scalar>(lineTopo, degree);
    auto linePoints = lineCubature->allocateCubaturePoints();
    auto lineWeights = lineCubature->allocateCubatureWeights();
    lineCubature->getCubature(linePoints, lineWeights);

    std::vector<Intrepid2::ScalarView<Scalar,DeviceType>> pointComponents(spaceDim);
    std::vector<Intrepid2::Data<Scalar,DeviceType>> weightComponents(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      pointComponents[d] = linePoints.getTensorComponent(0);
      weightComponents[d] = lineWeights.getTensorComponent(0);
    }
    tensorCubatureWeights = Intrepid2::TensorData<Scalar,DeviceType>(weightComponents);
    return Intrepid2::TensorPoints<Scalar,DeviceType>(pointComponents);
  }

  template<int spaceDim, class DeviceType, class Scalar>
  void injectGeneralTensorCoefficientIntoCubatureWeights(const TensorFunctionPtr &coefficient,
                                                         const Intrepid2::TensorPoints<Scalar,DeviceType> &tensorCubaturePoints,
                                                         Intrepid2::TensorData<Scalar,DeviceType> &tensorCubatureWeights,
                                                         TensorFunctionEvaluationContext *evaluationContext = nullptr)
  {
    const int pointCount = tensorCubaturePoints.extent_int(0);
    if (pointCount != tensorCubatureWeights.extent_int(0))
      throw std::invalid_argument("Tensor cubature points and weights have inconsistent point extents");

    using PointComponentHostView = typename Intrepid2::ScalarView<Scalar,DeviceType>::host_mirror_type;
    using WeightComponentHostView = typename Kokkos::View<Scalar*,DeviceType>::host_mirror_type;
    std::vector<int> componentPointCounts(spaceDim);
    std::vector<PointComponentHostView> pointComponentHosts(spaceDim);
    std::vector<WeightComponentHostView> weightComponentHosts(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      auto componentPoints  = tensorCubaturePoints.getTensorComponent(d);
      auto componentWeights = tensorCubatureWeights.getTensorComponent(d);
      componentPointCounts[d] = tensorCubaturePoints.componentPointCount(d);
      pointComponentHosts[d]  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), componentPoints);
      weightComponentHosts[d] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), componentWeights.getUnderlyingView1());
    }

    Intrepid2::ScalarView<Scalar,DeviceType> weightedPointView("general tensor coefficient weighted cubature", pointCount);
    auto weightedHost = Kokkos::create_mirror_view(weightedPointView);
    TensorFunction::PointViewHost coefficientPoints("general tensor coefficient host points", pointCount, spaceDim);
    std::vector<Scalar> referenceWeights(pointCount, Scalar(1.0));

    std::vector<double> coord(spaceDim, 0.0);
    for (int pointOrdinal=0; pointOrdinal<pointCount; pointOrdinal++)
    {
      int remainingPointOrdinal = pointOrdinal;
      Scalar referenceWeight = Scalar(1.0);
      for (int d=0; d<spaceDim; d++)
      {
        const int componentPointOrdinal = remainingPointOrdinal % componentPointCounts[d];
        remainingPointOrdinal /= componentPointCounts[d];
        coord[d] = pointComponentHosts[d](componentPointOrdinal, 0);
        coefficientPoints(pointOrdinal,d) = coord[d];
        referenceWeight *= weightComponentHosts[d](componentPointOrdinal);
      }
      referenceWeights[pointOrdinal] = referenceWeight;
      if (evaluationContext == nullptr)
        weightedHost(pointOrdinal) = static_cast<Scalar>(referenceWeight * coefficient->value(coord));
    }

    if (evaluationContext != nullptr)
    {
      TensorFunction::ValueViewHost coefficientValues("general tensor coefficient values", pointCount);
      evaluationContext->values(coefficient, coefficientPoints, coefficientValues);
      for (int pointOrdinal=0; pointOrdinal<pointCount; pointOrdinal++)
        weightedHost(pointOrdinal) = static_cast<Scalar>(referenceWeights[pointOrdinal] * coefficientValues(pointOrdinal));
    }
    Kokkos::deep_copy(weightedPointView, weightedHost);

    Kokkos::Array<int,1> extents {pointCount};
    Kokkos::Array<Intrepid2::DataVariationType,1> variationTypes {Intrepid2::GENERAL};
    Intrepid2::Data<Scalar,DeviceType> weightedPointData(weightedPointView, extents, variationTypes);
    tensorCubatureWeights = Intrepid2::TensorData<Scalar,DeviceType>(std::vector<Intrepid2::Data<Scalar,DeviceType>>{weightedPointData});
  }

  template<int spaceDim, class DeviceType, class Scalar>
  Scalar foldSupportedTensorCoefficientIntoCubatureWeights(const TensorFunctionPtr &coefficient,
                                                           const Intrepid2::TensorPoints<Scalar,DeviceType> &tensorCubaturePoints,
                                                           Intrepid2::TensorData<Scalar,DeviceType> &tensorCubatureWeights,
                                                           TensorFunctionEvaluationContext *evaluationContext = nullptr)
  {
    if (coefficient == Teuchos::null) return Scalar(1.0);
    if (coefficient->hasSolutionDependencies())
    {
      if (evaluationContext == nullptr)
        throw std::invalid_argument("PAMatrix bridge requires a TensorFunctionEvaluationContext for solution-dependent tensor coefficients");
      injectGeneralTensorCoefficientIntoCubatureWeights<spaceDim,DeviceType,Scalar>(coefficient,
                                                                                    tensorCubaturePoints,
                                                                                    tensorCubatureWeights,
                                                                                    evaluationContext);
      return Scalar(1.0);
    }

    const auto &dependencyScope = coefficient->dependencyScope();
    if (dependencyScope.isEmpty())
      return static_cast<Scalar>(coefficient->value(std::vector<double>(spaceDim, 0.0)));

    const auto &componentOrdinals = dependencyScope.componentOrdinals();
    if (componentOrdinals.size() != 1)
    {
      injectGeneralTensorCoefficientIntoCubatureWeights<spaceDim,DeviceType,Scalar>(coefficient,
                                                                                    tensorCubaturePoints,
                                                                                    tensorCubatureWeights,
                                                                                    evaluationContext);
      return Scalar(1.0);
    }

    const int coefficientComponent = *componentOrdinals.begin();
    if ((coefficientComponent < 0) || (coefficientComponent >= spaceDim))
      throw std::invalid_argument("Tensor coefficient component is outside the HGRAD hypercube dimension");

    std::vector<Intrepid2::Data<Scalar,DeviceType>> weightedComponents(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      weightedComponents[d] = tensorCubatureWeights.getTensorComponent(d);
    }

    auto componentPoints  = tensorCubaturePoints.getTensorComponent(coefficientComponent);
    auto componentWeights = tensorCubatureWeights.getTensorComponent(coefficientComponent);
    const int componentPointCount = componentPoints.extent_int(0);

    Intrepid2::ScalarView<Scalar,DeviceType> weightedComponentView("tensor coefficient weighted cubature", componentPointCount);
    auto pointsHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), componentPoints);
    auto weightsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), componentWeights.getUnderlyingView1());
    auto weightedHost = Kokkos::create_mirror_view(weightedComponentView);

    std::vector<double> coord(spaceDim, 0.0);
    for (int pointOrdinal=0; pointOrdinal<componentPointCount; pointOrdinal++)
    {
      coord[coefficientComponent] = pointsHost(pointOrdinal, 0);
      weightedHost(pointOrdinal) = static_cast<Scalar>(weightsHost(pointOrdinal) * coefficient->value(coord));
    }
    Kokkos::deep_copy(weightedComponentView, weightedHost);

    Kokkos::Array<int,1> extents {componentPointCount};
    Kokkos::Array<Intrepid2::DataVariationType,1> variationTypes {Intrepid2::GENERAL};
    weightedComponents[coefficientComponent] = Intrepid2::Data<Scalar,DeviceType>(weightedComponentView, extents, variationTypes);
    tensorCubatureWeights = Intrepid2::TensorData<Scalar,DeviceType>(weightedComponents);
    return Scalar(1.0);
  }

  template<class IntrepidBasisFamily, int spaceDim, unsigned long spaceDim2, class DeviceType, class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar> constructHypercubePAMatrix(Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
                                                                    const int p,
                                                                    const Intrepid2::EFunctionSpace functionSpace1,
                                                                    const Intrepid2::EOperator op1,
                                                                    Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight1,
                                                                    const Intrepid2::EFunctionSpace functionSpace2,
                                                                    const Intrepid2::EOperator op2,
                                                                    Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight2,
                                                                    const TensorFunctionPtr &coefficient = Teuchos::null,
                                                                    Scalar *scalarWeight = nullptr,
                                                                    TensorFunctionEvaluationContext *evaluationContext = nullptr)
  {
    using ExecutionSpace = typename DeviceType::execution_space;

    auto basis1 = detail::getHypercubeBasisForPAMatrixBridge<IntrepidBasisFamily>(functionSpace1, p, spaceDim);
    auto basis2 = detail::getHypercubeBasisForPAMatrixBridge<IntrepidBasisFamily>(functionSpace2, p, spaceDim);

    Intrepid2::TensorData<Scalar,DeviceType> tensorCubatureWeights;
    auto tensorCubaturePoints = makeHypercubeCubaturePoints<spaceDim,DeviceType,Scalar>(2 * p, tensorCubatureWeights);
    const Scalar coefficientScalarWeight = foldSupportedTensorCoefficientIntoCubatureWeights<spaceDim,DeviceType,Scalar>(coefficient,
                                                                                                                        tensorCubaturePoints,
                                                                                                                        tensorCubatureWeights,
                                                                                                                        evaluationContext);
    if (scalarWeight != nullptr) *scalarWeight = coefficientScalarWeight;

    auto basis1Values = basis1->allocateBasisValues(tensorCubaturePoints, op1);
    basis1->getValues(basis1Values, tensorCubaturePoints, op1);
    basis1Values.setBasis(basis1);

    auto basis2Values = basis2->allocateBasisValues(tensorCubaturePoints, op2);
    basis2->getValues(basis2Values, tensorCubaturePoints, op2);
    basis2Values.setBasis(basis2);

    const int numCells = geometry.numCells();
    auto jacobian       = geometry.allocateJacobianData(tensorCubaturePoints, 0, numCells);
    auto jacobianDividedByJacobianDet = geometry.allocateJacobianData(tensorCubaturePoints, 0, numCells);
    auto jacobianDet    = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
    auto jacobianDetInv = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
    auto jacobianInv    = Intrepid2::CellTools<DeviceType>::allocateJacobianInv(jacobian);
    auto cellMeasures   = geometry.allocateCellMeasure(jacobianDet, tensorCubatureWeights);

    auto refData = geometry.getJacobianRefData(tensorCubaturePoints);
    geometry.setJacobian(jacobian, tensorCubaturePoints, refData);
    Intrepid2::CellTools<DeviceType>::setJacobianDet         (jacobianDet,    jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianDetInv      (jacobianDetInv, jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianInv         (jacobianInv,    jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianDividedByDet(jacobianDividedByJacobianDet, jacobian, jacobianDetInv);

    auto transformedBasis1Values = detail::transformForPAMatrixBridge(functionSpace1, op1, basis1Values,
                                                                       jacobian, jacobianInv, jacobianDetInv, jacobianDividedByJacobianDet);
    auto transformedBasis2Values = detail::transformForPAMatrixBridge(functionSpace2, op2, basis2Values,
                                                                       jacobian, jacobianInv, jacobianDetInv, jacobianDividedByJacobianDet);

    if (vectorWeight1 != Teuchos::null)
    {
      Intrepid2::ScalarView<Scalar,DeviceType> auView("a_u", spaceDim);
      auto auViewHost = Kokkos::create_mirror(auView);
      for (int d=0; d<spaceDim; d++) auViewHost(d) = (*vectorWeight1)[d];
      Kokkos::deep_copy(auView, auViewHost);

      Kokkos::Array<int,3> extents {numCells, tensorCubaturePoints.extent_int(0), spaceDim};
      Kokkos::Array<Intrepid2::DataVariationType,3> variationTypes {Intrepid2::CONSTANT, Intrepid2::CONSTANT, Intrepid2::GENERAL};
      Intrepid2::Data<Scalar,DeviceType> auData(auView, extents, variationTypes);
      auto uTransform = Intrepid2::Data<Scalar,DeviceType>::allocateMatVecResult(transformedBasis1Values.transform(), auData, true);
      uTransform.storeMatVec(transformedBasis1Values.transform(), auData, true);
      transformedBasis1Values = Intrepid2::TransformedBasisValues<Scalar,DeviceType>(uTransform, basis1Values);
    }

    if (vectorWeight2 != Teuchos::null)
    {
      Intrepid2::ScalarView<Scalar,DeviceType> avView("a_v", spaceDim);
      auto avViewHost = Kokkos::create_mirror(avView);
      for (int d=0; d<spaceDim; d++) avViewHost(d) = (*vectorWeight2)[d];
      Kokkos::deep_copy(avView, avViewHost);

      Kokkos::Array<int,3> extents {numCells, tensorCubaturePoints.extent_int(0), spaceDim};
      Kokkos::Array<Intrepid2::DataVariationType,3> variationTypes {Intrepid2::CONSTANT, Intrepid2::CONSTANT, Intrepid2::GENERAL};
      Intrepid2::Data<Scalar,DeviceType> avData(avView, extents, variationTypes);
      auto vTransform = Intrepid2::Data<Scalar,DeviceType>::allocateMatVecResult(transformedBasis2Values.transform(), avData, true);
      vTransform.storeMatVec(transformedBasis2Values.transform(), avData, true);
      transformedBasis2Values = Intrepid2::TransformedBasisValues<Scalar,DeviceType>(vTransform, basis2Values);
    }

    geometry.computeCellMeasure(cellMeasures, jacobianDet, tensorCubatureWeights);
    ExecutionSpace().fence();

    return Intrepid2::PAMatrix<DeviceType,Scalar>(transformedBasis1Values, cellMeasures, transformedBasis2Values);
  }

  template<int spaceDim, unsigned long spaceDim2, class DeviceType, class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar>
  constructHypercubePAMatrix(
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const Intrepid2::EFunctionSpace functionSpace1,
      const Intrepid2::EOperator op1,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight1,
      const Intrepid2::EFunctionSpace functionSpace2,
      const Intrepid2::EOperator op2,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight2,
      const TensorFunctionPtr &coefficient = Teuchos::null,
      Scalar *scalarWeight = nullptr,
      TensorFunctionEvaluationContext *evaluationContext = nullptr,
      const TensorBasisFamily basisFamily = TensorBasisFamily::Hierarchical)
  {
    switch (basisFamily)
    {
      case TensorBasisFamily::Hierarchical:
      {
        using IntrepidBasisFamily =
            Intrepid2::HierarchicalBasisFamily<DeviceType,Scalar,Scalar>;

        return constructHypercubePAMatrix
            <IntrepidBasisFamily,spaceDim,spaceDim2,DeviceType,Scalar>(
                geometry,
                p,
                functionSpace1,
                op1,
                vectorWeight1,
                functionSpace2,
                op2,
                vectorWeight2,
                coefficient,
                scalarWeight,
                evaluationContext);
      }

      case TensorBasisFamily::Nodal:
      {
        using IntrepidBasisFamily =
            Intrepid2::DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;

        return constructHypercubePAMatrix
            <IntrepidBasisFamily,spaceDim,spaceDim2,DeviceType,Scalar>(
                geometry,
                p,
                functionSpace1,
                op1,
                vectorWeight1,
                functionSpace2,
                op2,
                vectorWeight2,
                coefficient,
                scalarWeight,
                evaluationContext);
      }

      default:
        throw std::invalid_argument("Unknown TensorBasisFamily");
    }
  }

  template<class IntrepidBasisFamily,
           int spaceDim,
           unsigned long spaceDim2,
           class DeviceType,
           class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar>
  constructHypercubeHGRADPAMatrixForBasisFamily(
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const Intrepid2::EOperator op1,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight1,
      const Intrepid2::EOperator op2,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight2,
      const TensorFunctionPtr &coefficient = Teuchos::null,
      Scalar *scalarWeight = nullptr,
      TensorFunctionEvaluationContext *evaluationContext = nullptr)
  {
    using ExecutionSpace = typename DeviceType::execution_space;

    auto basis1 = Intrepid2::getHypercubeBasis_HGRAD<IntrepidBasisFamily>(p, spaceDim);
    auto basis2 = Intrepid2::getHypercubeBasis_HGRAD<IntrepidBasisFamily>(p, spaceDim);

    Intrepid2::TensorData<Scalar,DeviceType> tensorCubatureWeights;
    auto tensorCubaturePoints = makeHypercubeCubaturePoints<spaceDim,DeviceType,Scalar>(2 * p, tensorCubatureWeights);
    const Scalar coefficientScalarWeight = foldSupportedTensorCoefficientIntoCubatureWeights<spaceDim,DeviceType,Scalar>(coefficient,
                                                                                                                        tensorCubaturePoints,
                                                                                                                        tensorCubatureWeights,
                                                                                                                        evaluationContext);
    if (scalarWeight != nullptr) *scalarWeight = coefficientScalarWeight;

    auto basis1Values = basis1->allocateBasisValues(tensorCubaturePoints, op1);
    basis1->getValues(basis1Values, tensorCubaturePoints, op1);
    basis1Values.setBasis(basis1);

    auto basis2Values = basis2->allocateBasisValues(tensorCubaturePoints, op2);
    basis2->getValues(basis2Values, tensorCubaturePoints, op2);
    basis2Values.setBasis(basis2);

    const int numCells = geometry.numCells();
    auto jacobian       = geometry.allocateJacobianData(tensorCubaturePoints, 0, numCells);
    auto jacobianDividedByJacobianDet = geometry.allocateJacobianData(tensorCubaturePoints, 0, numCells);
    auto jacobianDet    = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
    auto jacobianDetInv = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
    auto jacobianInv    = Intrepid2::CellTools<DeviceType>::allocateJacobianInv(jacobian);
    auto cellMeasures   = geometry.allocateCellMeasure(jacobianDet, tensorCubatureWeights);

    auto refData = geometry.getJacobianRefData(tensorCubaturePoints);
    geometry.setJacobian(jacobian, tensorCubaturePoints, refData);
    Intrepid2::CellTools<DeviceType>::setJacobianDet         (jacobianDet,    jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianDetInv      (jacobianDetInv, jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianInv         (jacobianInv,    jacobian);
    Intrepid2::CellTools<DeviceType>::setJacobianDividedByDet(jacobianDividedByJacobianDet, jacobian, jacobianDetInv);

    auto transformedBasis1Values = detail::transformForPAMatrixBridge(Intrepid2::FUNCTION_SPACE_HGRAD, op1, basis1Values,
                                                                       jacobian, jacobianInv, jacobianDetInv, jacobianDividedByJacobianDet);
    auto transformedBasis2Values = detail::transformForPAMatrixBridge(Intrepid2::FUNCTION_SPACE_HGRAD, op2, basis2Values,
                                                                       jacobian, jacobianInv, jacobianDetInv, jacobianDividedByJacobianDet);

    if (vectorWeight1 != Teuchos::null)
    {
      Intrepid2::ScalarView<Scalar,DeviceType> auView("a_u", spaceDim);
      auto auViewHost = Kokkos::create_mirror(auView);
      for (int d=0; d<spaceDim; d++) auViewHost(d) = (*vectorWeight1)[d];
      Kokkos::deep_copy(auView, auViewHost);

      Kokkos::Array<int,3> extents {numCells, tensorCubaturePoints.extent_int(0), spaceDim};
      Kokkos::Array<Intrepid2::DataVariationType,3> variationTypes {Intrepid2::CONSTANT, Intrepid2::CONSTANT, Intrepid2::GENERAL};
      Intrepid2::Data<Scalar,DeviceType> auData(auView, extents, variationTypes);
      auto uTransform = Intrepid2::Data<Scalar,DeviceType>::allocateMatVecResult(transformedBasis1Values.transform(), auData, true);
      uTransform.storeMatVec(transformedBasis1Values.transform(), auData, true);
      transformedBasis1Values = Intrepid2::TransformedBasisValues<Scalar,DeviceType>(uTransform, basis1Values);
    }

    if (vectorWeight2 != Teuchos::null)
    {
      Intrepid2::ScalarView<Scalar,DeviceType> avView("a_v", spaceDim);
      auto avViewHost = Kokkos::create_mirror(avView);
      for (int d=0; d<spaceDim; d++) avViewHost(d) = (*vectorWeight2)[d];
      Kokkos::deep_copy(avView, avViewHost);

      Kokkos::Array<int,3> extents {numCells, tensorCubaturePoints.extent_int(0), spaceDim};
      Kokkos::Array<Intrepid2::DataVariationType,3> variationTypes {Intrepid2::CONSTANT, Intrepid2::CONSTANT, Intrepid2::GENERAL};
      Intrepid2::Data<Scalar,DeviceType> avData(avView, extents, variationTypes);
      auto vTransform = Intrepid2::Data<Scalar,DeviceType>::allocateMatVecResult(transformedBasis2Values.transform(), avData, true);
      vTransform.storeMatVec(transformedBasis2Values.transform(), avData, true);
      transformedBasis2Values = Intrepid2::TransformedBasisValues<Scalar,DeviceType>(vTransform, basis2Values);
    }

    geometry.computeCellMeasure(cellMeasures, jacobianDet, tensorCubatureWeights);
    ExecutionSpace().fence();

    return Intrepid2::PAMatrix<DeviceType,Scalar>(transformedBasis1Values, cellMeasures, transformedBasis2Values);
  }

  template<int spaceDim, unsigned long spaceDim2, class DeviceType, class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar>
  constructHypercubeHGRADPAMatrix(
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const Intrepid2::EOperator op1,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight1,
      const Intrepid2::EOperator op2,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight2,
      const TensorFunctionPtr &coefficient = Teuchos::null,
      Scalar *scalarWeight = nullptr,
      TensorFunctionEvaluationContext *evaluationContext = nullptr,
      const TensorBasisFamily basisFamily = TensorBasisFamily::Hierarchical)
  {
    switch (basisFamily)
    {
      case TensorBasisFamily::Hierarchical:
      {
        using IntrepidBasisFamily =
            Intrepid2::HierarchicalBasisFamily<DeviceType,Scalar,Scalar>;

        return constructHypercubeHGRADPAMatrixForBasisFamily
            <IntrepidBasisFamily,spaceDim,spaceDim2,DeviceType,Scalar>(
                geometry,
                p,
                op1,
                vectorWeight1,
                op2,
                vectorWeight2,
                coefficient,
                scalarWeight,
                evaluationContext);
      }

      case TensorBasisFamily::Nodal:
      {
        using IntrepidBasisFamily =
            Intrepid2::DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;

        return constructHypercubeHGRADPAMatrixForBasisFamily
            <IntrepidBasisFamily,spaceDim,spaceDim2,DeviceType,Scalar>(
                geometry,
                p,
                op1,
                vectorWeight1,
                op2,
                vectorWeight2,
                coefficient,
                scalarWeight,
                evaluationContext);
      }

      default:
        throw std::invalid_argument("Unknown TensorBasisFamily");
    }
  }

  template<int spaceDim, class DeviceType, class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar>
  constructHypercubeHGRADPAMatrix(
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const Intrepid2::EOperator op1,
      const Intrepid2::EOperator op2,
      const TensorBasisFamily basisFamily = TensorBasisFamily::Hierarchical)
  {
    Teuchos::RCP<Kokkos::Array<Scalar,spaceDim>> nullVectorWeight = Teuchos::null;
    return constructHypercubeHGRADPAMatrix<spaceDim,spaceDim,DeviceType,Scalar>(
        geometry,
        p,
        op1,
        nullVectorWeight,
        op2,
        nullVectorWeight,
        Teuchos::null,
        nullptr,
        nullptr,
        basisFamily);
  }

  template<int spaceDim, unsigned long spaceDim2, class DeviceType, class Scalar>
  Intrepid2::PAMatrix<DeviceType,Scalar>
  constructHypercubeHGRADPAMatrix(
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const Intrepid2::EOperator op1,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight1,
      const Intrepid2::EOperator op2,
      Teuchos::RCP<Kokkos::Array<Scalar,spaceDim2>> vectorWeight2,
      const TensorBasisFamily basisFamily)
  {
    Scalar *nullScalarWeight = nullptr;

    return constructHypercubeHGRADPAMatrix<spaceDim,spaceDim2,DeviceType,Scalar>(
        geometry,
        p,
        op1,
        vectorWeight1,
        op2,
        vectorWeight2,
        Teuchos::null,
        nullScalarWeight,
        nullptr,
        basisFamily);
  }

  template<int spaceDim>
  std::vector<TensorFunctionPtr> decomposeTensorCoefficientForPAMatrixBridge(const TensorFunctionPtr &coefficient)
  {
    if (coefficient == Teuchos::null) throw std::invalid_argument("Tensor term requires a coefficient");
    if (coefficient->hasSolutionDependencies())
      return {coefficient};

    const auto &componentOrdinals = coefficient->dependencyScope().componentOrdinals();
    if (componentOrdinals.empty() || (componentOrdinals.size() == 1))
      return {coefficient};

    if (coefficient->isSum())
    {
      std::vector<TensorFunctionPtr> decomposedAddends;
      for (const auto &child : coefficient->childFunctions())
      {
        auto childAddends = decomposeTensorCoefficientForPAMatrixBridge<spaceDim>(child);
        decomposedAddends.insert(decomposedAddends.end(), childAddends.begin(), childAddends.end());
      }
      return decomposedAddends;
    }

    if (coefficient->isProduct())
    {
      const auto children = coefficient->childFunctions();
      if (children.size() == 2)
      {
        const bool leftIsScalar  = children[0]->dependencyScope().isEmpty();
        const bool rightIsScalar = children[1]->dependencyScope().isEmpty();
        if (leftIsScalar || rightIsScalar)
        {
          const auto scalarFactor = leftIsScalar ? children[0] : children[1];
          const auto tensorFactor = leftIsScalar ? children[1] : children[0];
          auto tensorAddends = decomposeTensorCoefficientForPAMatrixBridge<spaceDim>(tensorFactor);
          std::vector<TensorFunctionPtr> scaledAddends;
          scaledAddends.reserve(tensorAddends.size());
          for (const auto &addend : tensorAddends)
          {
            scaledAddends.push_back(scalarFactor * addend);
          }
          return scaledAddends;
        }
      }
    }

    return {coefficient};
  }

  template<int spaceDim, class DeviceType, class Scalar>
  TensorPAMatrixTerm<DeviceType,Scalar> adaptTensorVolumeTermToPAMatrix(Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
                                                                        const int p,
                                                                        const TensorBilinearTerm &term,
                                                                        const std::string &label,
                                                                        TensorFunctionEvaluationContext *evaluationContext = nullptr,
                                                                        const TensorBasisFamily basisFamily = TensorBasisFamily::Hierarchical)
  {
    if (!term.isVolumeTerm()) throw std::invalid_argument("PAMatrix bridge only accepts tensor volume terms");
    if (!term.isPAMatrixCandidate()) throw std::invalid_argument("Tensor term does not declare PAMatrix support");
    if (term.componentCount() != spaceDim) throw std::invalid_argument("Tensor term component count must match HGRAD hypercube dimension");
    if (term.coefficient() == Teuchos::null) throw std::invalid_argument("Tensor term requires a coefficient");

    const auto functionSpace1 = intrepidFunctionSpaceForTensorDescriptor(term.trialOperator());
    const auto functionSpace2 = intrepidFunctionSpaceForTensorDescriptor(term.testOperator());

    const auto & HVOL  = Intrepid2::FUNCTION_SPACE_HVOL;
    const auto & HGRAD = Intrepid2::FUNCTION_SPACE_HGRAD;
    
    const bool supportedTrialSpace = (functionSpace1 == HGRAD) || (functionSpace1 == HVOL);
    const bool supportedTestSpace  = (functionSpace2 == HGRAD) || (functionSpace2 == HVOL);

    if (!supportedTrialSpace || !supportedTestSpace)
    {
      throw std::invalid_argument(
          "adaptTensorHGRADVolumeTermToPAMatrix currently supports HGRAD/HVOL volume terms");
    }

    const auto op1 = pamatrixOperatorForTensorDescriptor(term.trialOperator());
    const auto op2 = pamatrixOperatorForTensorDescriptor(term.testOperator());

    auto vectorWeight1 = vectorWeightForSingleDerivative<spaceDim,Scalar>(term.trialOperator());
    auto vectorWeight2 = vectorWeightForSingleDerivative<spaceDim,Scalar>(term.testOperator());

    Scalar scalarWeight = Scalar(1.0);

    return {constructHypercubePAMatrix<spaceDim,spaceDim,DeviceType,Scalar>(
                geometry,
                p,
                functionSpace1, op1, vectorWeight1,
                functionSpace2, op2, vectorWeight2,
                term.coefficient(), &scalarWeight, evaluationContext, basisFamily),
            scalarWeight,
            label};
  }

  template<int spaceDim, class DeviceType, class Scalar>
  void appendTensorVolumeTermsToPAMatrix(
      std::vector<TensorPAMatrixTerm<DeviceType,Scalar>> &terms,
      Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
      const int p,
      const TensorBilinearTerm &term,
      const std::string &label,
      TensorFunctionEvaluationContext *evaluationContext = nullptr,
      const TensorBasisFamily basisFamily = TensorBasisFamily::Hierarchical)
  {
    const auto coefficientAddends = decomposeTensorCoefficientForPAMatrixBridge<spaceDim>(term.coefficient());
    if (coefficientAddends.size() == 1)
    {
      terms.push_back(adaptTensorVolumeTermToPAMatrix<spaceDim,DeviceType,Scalar>(geometry, p, term, label,
                                                                                       evaluationContext, basisFamily));
      return;
    }

    int addendOrdinal = 0;
    for (const auto &addend : coefficientAddends)
    {
      Camellia::TensorBilinearTerm addendTerm(term.trialVar(),
                                              term.testVar(),
                                              term.trialOperator(),
                                              term.testOperator(),
                                              addend,
                                              term.domain(),
                                              term.capabilities());
      terms.push_back(adaptTensorVolumeTermToPAMatrix<spaceDim,DeviceType,Scalar>(geometry, p, addendTerm,
                                                                                       label + "_addend" + std::to_string(addendOrdinal),
                                                                                       evaluationContext, basisFamily));
      addendOrdinal++;
    }
  }
}

#endif
