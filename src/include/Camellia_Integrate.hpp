//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#ifndef Camellia_Integrate_hpp
#define Camellia_Integrate_hpp

/// \file  Camellia_Integrate.hpp
/// \brief BLAS-backed integrate for Camellia::FieldContainer, replacing
///        Intrepid::FunctionSpaceTools::integrate for Camellia containers.

#include "Camellia_FieldContainer.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_TestForException.hpp"

namespace Camellia {
namespace FunctionSpaceTools {

namespace Impl {

inline int productDimensions(const Teuchos::Array<int> &dimensions, const int startDim)
{
  int product = 1;
  for (int d=startDim; d<dimensions.size(); d++) product *= dimensions[d];
  return product;
}

template<typename Scalar>
void dimensions(const Camellia::FieldContainer<Scalar> &container, Teuchos::Array<int> &dims)
{
  container.dimensions(dims);
}

inline void requireSameDimensions(const Teuchos::Array<int> &leftDims,
                                  const Teuchos::Array<int> &rightDims,
                                  const int startLeft,
                                  const int startRight,
                                  const char *message)
{
  TEUCHOS_TEST_FOR_EXCEPTION(leftDims.size() - startLeft != rightDims.size() - startRight, std::invalid_argument, message);
  for (int d=0; d<leftDims.size() - startLeft; d++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(leftDims[startLeft+d] != rightDims[startRight+d], std::invalid_argument, message);
  }
}

template<typename Scalar, typename OutputViewType, typename InputDataViewType, typename InputFieldsViewType>
void scalarMultiplyDataFieldDirect(OutputViewType outputFields,
                                   const InputDataViewType inputData,
                                   const InputFieldsViewType inputFields,
                                   const bool equalRank,
                                   const bool reciprocal)
{
  const int dataPointCount = inputData.extent_int(1);

  switch (outputFields.rank())
  {
    case 3:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank3_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy3(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p)
          {
            outputFields(c,f,p) = inputFields(c,f,p) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank3_equal", FieldContainer<Scalar>::getMDRangePolicy3(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p)
          {
            outputFields(c,f,p) = inputData(c, p % dataPointCount) * inputFields(c,f,p);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank3_reciprocal", FieldContainer<Scalar>::getMDRangePolicy3(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p)
          {
            outputFields(c,f,p) = inputFields(f,p) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank3", FieldContainer<Scalar>::getMDRangePolicy3(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p)
          {
            outputFields(c,f,p) = inputData(c, p % dataPointCount) * inputFields(f,p);
          });
        }
      }
      break;
    }
    case 4:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<4> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank4_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy4(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1)
          {
            outputFields(c,f,p,d1) = inputFields(c,f,p,d1) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank4_equal", FieldContainer<Scalar>::getMDRangePolicy4(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1)
          {
            outputFields(c,f,p,d1) = inputData(c, p % dataPointCount) * inputFields(c,f,p,d1);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank4_reciprocal", FieldContainer<Scalar>::getMDRangePolicy4(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1)
          {
            outputFields(c,f,p,d1) = inputFields(f,p,d1) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank4", FieldContainer<Scalar>::getMDRangePolicy4(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1)
          {
            outputFields(c,f,p,d1) = inputData(c, p % dataPointCount) * inputFields(f,p,d1);
          });
        }
      }
      break;
    }
    case 5:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<5> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank5_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy5(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputFields(c,f,p,d1,d2) = inputFields(c,f,p,d1,d2) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank5_equal", FieldContainer<Scalar>::getMDRangePolicy5(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputFields(c,f,p,d1,d2) = inputData(c, p % dataPointCount) * inputFields(c,f,p,d1,d2);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank5_reciprocal", FieldContainer<Scalar>::getMDRangePolicy5(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputFields(c,f,p,d1,d2) = inputFields(f,p,d1,d2) / inputData(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataField_rank5", FieldContainer<Scalar>::getMDRangePolicy5(outputFields),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type f, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputFields(c,f,p,d1,d2) = inputData(c, p % dataPointCount) * inputFields(f,p,d1,d2);
          });
        }
      }
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "scalarMultiplyDataFieldDirect: unsupported output rank");
  }
}

template<typename Scalar, typename OutputViewType, typename InputDataLeftViewType, typename InputDataRightViewType>
void scalarMultiplyDataDataDirect(OutputViewType outputData,
                                  const InputDataLeftViewType inputDataLeft,
                                  const InputDataRightViewType inputDataRight,
                                  const bool equalRank,
                                  const bool reciprocal)
{
  const int dataPointCount = inputDataLeft.extent_int(1);

  switch (outputData.rank())
  {
    case 2:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank2_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy2(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p)
          {
            outputData(c,p) = inputDataRight(c,p) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank2_equal", FieldContainer<Scalar>::getMDRangePolicy2(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p)
          {
            outputData(c,p) = inputDataLeft(c, p % dataPointCount) * inputDataRight(c,p);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank2_reciprocal", FieldContainer<Scalar>::getMDRangePolicy2(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p)
          {
            outputData(c,p) = inputDataRight(p) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank2", FieldContainer<Scalar>::getMDRangePolicy2(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p)
          {
            outputData(c,p) = inputDataLeft(c, p % dataPointCount) * inputDataRight(p);
          });
        }
      }
      break;
    }
    case 3:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank3_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy3(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1)
          {
            outputData(c,p,d1) = inputDataRight(c,p,d1) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank3_equal", FieldContainer<Scalar>::getMDRangePolicy3(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1)
          {
            outputData(c,p,d1) = inputDataLeft(c, p % dataPointCount) * inputDataRight(c,p,d1);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank3_reciprocal", FieldContainer<Scalar>::getMDRangePolicy3(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1)
          {
            outputData(c,p,d1) = inputDataRight(p,d1) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank3", FieldContainer<Scalar>::getMDRangePolicy3(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1)
          {
            outputData(c,p,d1) = inputDataLeft(c, p % dataPointCount) * inputDataRight(p,d1);
          });
        }
      }
      break;
    }
    case 4:
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<4> >::array_index_type;
      if (equalRank)
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank4_equal_reciprocal", FieldContainer<Scalar>::getMDRangePolicy4(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputData(c,p,d1,d2) = inputDataRight(c,p,d1,d2) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank4_equal", FieldContainer<Scalar>::getMDRangePolicy4(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputData(c,p,d1,d2) = inputDataLeft(c, p % dataPointCount) * inputDataRight(c,p,d1,d2);
          });
        }
      }
      else
      {
        if (reciprocal)
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank4_reciprocal", FieldContainer<Scalar>::getMDRangePolicy4(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputData(c,p,d1,d2) = inputDataRight(p,d1,d2) / inputDataLeft(c, p % dataPointCount);
          });
        }
        else
        {
          Kokkos::parallel_for("scalarMultiplyDataData_rank4", FieldContainer<Scalar>::getMDRangePolicy4(outputData),
          KOKKOS_LAMBDA (const array_index_type c, const array_index_type p, const array_index_type d1, const array_index_type d2)
          {
            outputData(c,p,d1,d2) = inputDataLeft(c, p % dataPointCount) * inputDataRight(p,d1,d2);
          });
        }
      }
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "scalarMultiplyDataDataDirect: unsupported output rank");
  }
}

} // namespace Impl

template<typename Scalar>
void HGRADtransformVALUE(Camellia::FieldContainer<Scalar>       &output,
                         const Camellia::FieldContainer<Scalar> &input)
{
  const bool outputIsInput = (&output == &input);
  if (outputIsInput)
  {
    auto outputView = output.checkoutDeviceView(true);
    Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HGRADtransformVALUE(outputView, outputView);
    output.checkinDeviceView(true);
  }
  else
  {
    auto outputView = output.checkoutDeviceView(false);
    auto inputView  = input.checkoutDeviceView();
    Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HGRADtransformVALUE(outputView, inputView);
    input.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
}

template<typename Scalar>
void HGRADtransformGRAD(Camellia::FieldContainer<Scalar>       &output,
                        const Camellia::FieldContainer<Scalar> &jacobianInverse,
                        const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView          = output.checkoutDeviceView(false);
  auto jacobianInverseView = jacobianInverse.checkoutDeviceView();
  auto inputView           = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HGRADtransformGRAD(outputView, jacobianInverseView, inputView);
  input.checkinDeviceView(false);
  jacobianInverse.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void HCURLtransformVALUE(Camellia::FieldContainer<Scalar>       &output,
                         const Camellia::FieldContainer<Scalar> &jacobianInverse,
                         const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView          = output.checkoutDeviceView(false);
  auto jacobianInverseView = jacobianInverse.checkoutDeviceView();
  auto inputView           = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HCURLtransformVALUE(outputView, jacobianInverseView, inputView);
  input.checkinDeviceView(false);
  jacobianInverse.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void HCURLtransformCURL(Camellia::FieldContainer<Scalar>       &output,
                        const Camellia::FieldContainer<Scalar> &jacobian,
                        const Camellia::FieldContainer<Scalar> &jacobianDet,
                        const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView      = output.checkoutDeviceView(false);
  auto jacobianView    = jacobian.checkoutDeviceView();
  auto jacobianDetView = jacobianDet.checkoutDeviceView();
  auto inputView       = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HCURLtransformCURL(outputView, jacobianView, jacobianDetView, inputView);
  input.checkinDeviceView(false);
  jacobianDet.checkinDeviceView(false);
  jacobian.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void HDIVtransformVALUE(Camellia::FieldContainer<Scalar>       &output,
                        const Camellia::FieldContainer<Scalar> &jacobian,
                        const Camellia::FieldContainer<Scalar> &jacobianDet,
                        const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView      = output.checkoutDeviceView(false);
  auto jacobianView    = jacobian.checkoutDeviceView();
  auto jacobianDetView = jacobianDet.checkoutDeviceView();
  auto inputView       = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HDIVtransformVALUE(outputView, jacobianView, jacobianDetView, inputView);
  input.checkinDeviceView(false);
  jacobianDet.checkinDeviceView(false);
  jacobian.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void HDIVtransformDIV(Camellia::FieldContainer<Scalar>       &output,
                      const Camellia::FieldContainer<Scalar> &jacobianDet,
                      const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView      = output.checkoutDeviceView(false);
  auto jacobianDetView = jacobianDet.checkoutDeviceView();
  auto inputView       = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HDIVtransformDIV(outputView, jacobianDetView, inputView);
  input.checkinDeviceView(false);
  jacobianDet.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void HVOLtransformVALUE(Camellia::FieldContainer<Scalar>       &output,
                        const Camellia::FieldContainer<Scalar> &jacobianDet,
                        const Camellia::FieldContainer<Scalar> &input)
{
  auto outputView      = output.checkoutDeviceView(false);
  auto jacobianDetView = jacobianDet.checkoutDeviceView();
  auto inputView       = input.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::HVOLtransformVALUE(outputView, jacobianDetView, inputView);
  input.checkinDeviceView(false);
  jacobianDet.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void multiplyMeasure(Camellia::FieldContainer<Scalar>       &output,
                     const Camellia::FieldContainer<Scalar> &inputMeasure,
                     const Camellia::FieldContainer<Scalar> &input)
{
  const bool outputIsInput = (&output == &input);
  auto inputMeasureView = inputMeasure.checkoutDeviceView();
  if (outputIsInput)
  {
    auto outputView = output.checkoutDeviceView(true);
    Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::multiplyMeasure(outputView, inputMeasureView, outputView);
    output.checkinDeviceView(true);
  }
  else
  {
    auto outputView = output.checkoutDeviceView(false);
    auto inputView  = input.checkoutDeviceView();
    Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::multiplyMeasure(outputView, inputMeasureView, inputView);
    input.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
  inputMeasure.checkinDeviceView(false);
}

template<typename Scalar>
void computeCellMeasure(Camellia::FieldContainer<Scalar>       &output,
                        const Camellia::FieldContainer<Scalar> &jacobianDet,
                        const Camellia::FieldContainer<Scalar> &inputWeights)
{
  auto outputView        = output.checkoutDeviceView(false);
  auto jacobianDetView   = jacobianDet.checkoutDeviceView();
  auto inputWeightsView  = inputWeights.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::computeCellMeasure(outputView, jacobianDetView, inputWeightsView);
  inputWeights.checkinDeviceView(false);
  jacobianDet.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

template<typename Scalar>
void scalarMultiplyDataField(Camellia::FieldContainer<Scalar>       &output,
                             const Camellia::FieldContainer<Scalar> &inputData,
                             const Camellia::FieldContainer<Scalar> &inputFields,
                             const bool reciprocal = false)
{
  const bool outputIsInputFields = (&output == &inputFields);
  const bool equalRank = (output.rank() == inputFields.rank());
  auto inputDataView = inputData.checkoutDeviceView();
  if (outputIsInputFields)
  {
    auto outputView = output.checkoutDeviceView(true);
    Impl::scalarMultiplyDataFieldDirect<Scalar>(outputView, inputDataView, outputView, equalRank, reciprocal);
    output.checkinDeviceView(true);
  }
  else
  {
    auto outputView      = output.checkoutDeviceView(false);
    auto inputFieldsView = inputFields.checkoutDeviceView();
    Impl::scalarMultiplyDataFieldDirect<Scalar>(outputView, inputDataView, inputFieldsView, equalRank, reciprocal);
    inputFields.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
  inputData.checkinDeviceView(false);
}

template<typename Scalar>
void scalarMultiplyDataData(Camellia::FieldContainer<Scalar>       &output,
                            const Camellia::FieldContainer<Scalar> &inputDataLeft,
                            const Camellia::FieldContainer<Scalar> &inputDataRight,
                            const bool reciprocal = false)
{
  const bool outputIsLeft  = (&output == &inputDataLeft);
  const bool outputIsRight = (&output == &inputDataRight);
  const bool equalRank = (output.rank() == inputDataRight.rank());
  if (outputIsLeft && outputIsRight)
  {
    auto outputView = output.checkoutDeviceView(true);
    Impl::scalarMultiplyDataDataDirect<Scalar>(outputView, outputView, outputView, equalRank, reciprocal);
    output.checkinDeviceView(true);
  }
  else if (outputIsLeft)
  {
    auto outputView = output.checkoutDeviceView(true);
    auto inputDataRightView = inputDataRight.checkoutDeviceView();
    Impl::scalarMultiplyDataDataDirect<Scalar>(outputView, outputView, inputDataRightView, equalRank, reciprocal);
    inputDataRight.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
  else if (outputIsRight)
  {
    auto outputView = output.checkoutDeviceView(true);
    auto inputDataLeftView = inputDataLeft.checkoutDeviceView();
    Impl::scalarMultiplyDataDataDirect<Scalar>(outputView, inputDataLeftView, outputView, equalRank, reciprocal);
    inputDataLeft.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
  else
  {
    auto outputView = output.checkoutDeviceView(false);
    auto inputDataLeftView  = inputDataLeft.checkoutDeviceView();
    auto inputDataRightView = inputDataRight.checkoutDeviceView();
    Impl::scalarMultiplyDataDataDirect<Scalar>(outputView, inputDataLeftView, inputDataRightView, equalRank, reciprocal);
    inputDataRight.checkinDeviceView(false);
    inputDataLeft.checkinDeviceView(false);
    output.checkinDeviceView(true);
  }
}

template<typename Scalar>
void tensorMultiplyDataData(Camellia::FieldContainer<Scalar>       &output,
                            const Camellia::FieldContainer<Scalar> &inputDataLeft,
                            const Camellia::FieldContainer<Scalar> &inputDataRight,
                            const char transpose = 'N')
{
  auto outputView         = output.checkoutDeviceView(false);
  auto inputDataLeftView  = inputDataLeft.checkoutDeviceView();
  auto inputDataRightView = inputDataRight.checkoutDeviceView();
  Intrepid2::FunctionSpaceTools<Camellia::DeviceType>::tensorMultiplyDataData(outputView, inputDataLeftView, inputDataRightView, transpose);
  inputDataRight.checkinDeviceView(false);
  inputDataLeft.checkinDeviceView(false);
  output.checkinDeviceView(true);
}

/// \brief BLAS-backed integrate for Camellia::FieldContainer.
///
/// Dispatches on output rank to the appropriate contraction:
///   outRank 1: data-data   (DOT per cell)
///   outRank 2: data-field  (GEMM with M=1 per cell)
///   outRank 3: field-field (GEMM per cell)
template<typename Scalar>
void integrate(Camellia::FieldContainer<Scalar>       & outputValues,
               const Camellia::FieldContainer<Scalar> & leftValues,
               const Camellia::FieldContainer<Scalar> & rightValues,
               const bool                               sumInto = false)
{
  const int outRank = outputValues.rank();
  const int lRank   = leftValues.rank();
  const int rRank   = rightValues.rank();

  TEUCHOS_TEST_FOR_EXCEPTION( (outRank < 1) || (outRank > 3), std::invalid_argument,
      ">>> ERROR (Camellia::FunctionSpaceTools::integrate): Output container must have rank 1, 2, or 3.");

  Teuchos::Array<int> outDims, leftDims, rightDims;
  Impl::dimensions(outputValues, outDims);
  Impl::dimensions(leftValues,  leftDims);
  Impl::dimensions(rightValues, rightDims);

  Teuchos::BLAS<int, Scalar> blas;

  if (outRank == 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(lRank < 2, std::invalid_argument, "integrate rank-1 output requires data containers of rank at least 2.");
    TEUCHOS_TEST_FOR_EXCEPTION(rRank != lRank, std::invalid_argument, "integrate rank-1 inputs must have the same rank.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[0] != leftDims[0], std::invalid_argument, "integrate rank-1 output cell count mismatch.");
    Impl::requireSameDimensions(leftDims, rightDims, 0, 0, "integrate rank-1 input dimensions mismatch.");

    const int numCells = leftDims[0];
    const int dotLen   = Impl::productDimensions(leftDims, 1);
    const int incr = 1;
    const Scalar* L = leftValues.data();
    const Scalar* R = rightValues.data();
    Scalar* O = outputValues.data();
    for (int cl = 0; cl < numCells; ++cl)
    {
      const Scalar dot = blas.DOT(dotLen, L + cl * dotLen, incr, R + cl * dotLen, incr);
      if (sumInto) O[cl] += dot;
      else         O[cl]  = dot;
    }
  }
  else if (outRank == 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(lRank < 2, std::invalid_argument, "integrate rank-2 output requires data input of rank at least 2.");
    TEUCHOS_TEST_FOR_EXCEPTION(rRank != lRank + 1, std::invalid_argument, "integrate rank-2 right input rank must equal left rank plus one.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[0] != leftDims[0],  std::invalid_argument, "integrate rank-2 output cell count mismatch.");
    TEUCHOS_TEST_FOR_EXCEPTION(rightDims[0] != leftDims[0], std::invalid_argument, "integrate rank-2 input cell count mismatch.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[1] != rightDims[1], std::invalid_argument, "integrate rank-2 output field count mismatch.");

    const bool constantData = (leftDims[1] == 1) && (rightDims[2] != 1);
    if (constantData)
    {
      Impl::requireSameDimensions(leftDims, rightDims, 2, 3, "integrate rank-2 constant-data dimensions mismatch.");
    }
    else
    {
      Impl::requireSameDimensions(leftDims, rightDims, 1, 2, "integrate rank-2 data dimensions mismatch.");
    }

    const int numCells  = outDims[0];
    const int numFields = outDims[1];
    const Scalar* L = leftValues.data();
    const Scalar* R = rightValues.data();
    Scalar* O = outputValues.data();

    if (constantData)
    {
      const int numPoints = rightDims[2];
      const int componentCount = Impl::productDimensions(leftDims, 2);
      const int skipL = componentCount;
      const int skipR = numFields * numPoints * componentCount;
      const int skipO = numFields;
      for (int cl=0; cl<numCells; cl++)
      {
        for (int field=0; field<numFields; field++)
        {
          Scalar value = sumInto ? O[cl*skipO + field] : Scalar(0);
          for (int pt=0; pt<numPoints; pt++)
          {
            for (int comp=0; comp<componentCount; comp++)
            {
              value += L[cl*skipL + comp] * R[cl*skipR + field*numPoints*componentCount + pt*componentCount + comp];
            }
          }
          O[cl*skipO + field] = value;
        }
      }
    }
    else
    {
      const int numData = Impl::productDimensions(leftDims, 1);
      const int skipL  = numData;
      const int skipR  = numFields * numData;
      const int skipOp = numFields;
      const Scalar alpha(1);
      const Scalar beta = sumInto ? Scalar(1) : Scalar(0);

      for (int cl = 0; cl < numCells; ++cl)
      {
        blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                  1, numFields, numData,
                  alpha, L + cl * skipL,  numData,
                         R + cl * skipR, numData,
                  beta,  O + cl * skipOp, 1);
      }
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(lRank < 3, std::invalid_argument, "integrate rank-3 output requires field inputs of rank at least 3.");
    TEUCHOS_TEST_FOR_EXCEPTION(rRank != lRank, std::invalid_argument, "integrate rank-3 inputs must have the same rank.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[0] != leftDims[0],  std::invalid_argument, "integrate rank-3 output cell count mismatch.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[1] != leftDims[1],  std::invalid_argument, "integrate rank-3 output left-field count mismatch.");
    TEUCHOS_TEST_FOR_EXCEPTION(outDims[2] != rightDims[1], std::invalid_argument, "integrate rank-3 output right-field count mismatch.");
    TEUCHOS_TEST_FOR_EXCEPTION(rightDims[0] != leftDims[0], std::invalid_argument, "integrate rank-3 input cell count mismatch.");
    Impl::requireSameDimensions(leftDims, rightDims, 2, 2, "integrate rank-3 data dimensions mismatch.");

    const int numCells       = outDims[0];
    const int numLeftFields  = outDims[1];
    const int numRightFields = outDims[2];
    const int numData        = Impl::productDimensions(leftDims, 2);

    const int skipL  = numLeftFields  * numData;
    const int skipR  = numRightFields * numData;
    const int skipOp = numLeftFields  * numRightFields;
    const Scalar alpha(1);
    const Scalar beta = sumInto ? Scalar(1) : Scalar(0);

    const Scalar* L = leftValues.data();
    const Scalar* R = rightValues.data();
    Scalar* O = outputValues.data();
    for (int cl = 0; cl < numCells; ++cl)
    {
      blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                numRightFields, numLeftFields, numData,
                alpha, R + cl * skipR, numData,
                       L + cl * skipL, numData,
                beta,  O + cl * skipOp, numRightFields);
    }
  }
}

} // namespace FunctionSpaceTools
} // namespace Camellia

#endif // Camellia_Integrate_hpp
