#include "TensorBilinearFormBuilder.h"

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorBilinearFormBuilder::TensorBilinearFormBuilder(const std::string &name)
: _form(Teuchos::rcp(new TensorBilinearForm(name)))
{
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addValueValueTerm(TensorVarPtr trialVar,
                                                                        TensorVarPtr testVar,
                                                                        TensorFunctionPtr coefficient,
                                                                        const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount()),
                                    TensorComponentOperatorDescriptor::valueOnly(testVar->componentCount()),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addBoundaryValueValueTerm(TensorVarPtr trialVar,
                                                                                TensorVarPtr testVar,
                                                                                TensorFunctionPtr coefficient,
                                                                                const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount()),
                                    TensorComponentOperatorDescriptor::valueOnly(testVar->componentCount()),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_BOUNDARY,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addDirectionalTraceBoundaryValueTerms(const std::vector<TensorVarPtr> &directionalTraceVars,
                                                                                            TensorVarPtr testVar,
                                                                                            const std::vector<TensorFunctionPtr> &coefficients,
                                                                                            const TensorOperatorCapabilitySet &capabilities)
{
  TEUCHOS_TEST_FOR_EXCEPTION(directionalTraceVars.empty(), std::invalid_argument,
                             "TensorBilinearFormBuilder::addDirectionalTraceBoundaryValueTerms requires at least one directional trace variable.");
  TEUCHOS_TEST_FOR_EXCEPTION(directionalTraceVars.size() != coefficients.size(), std::invalid_argument,
                             "TensorBilinearFormBuilder::addDirectionalTraceBoundaryValueTerms requires one coefficient per directional trace variable.");

  for (size_t traceOrdinal=0; traceOrdinal<directionalTraceVars.size(); traceOrdinal++)
  {
    addBoundaryValueValueTerm(directionalTraceVars[traceOrdinal],
                              testVar,
                              coefficients[traceOrdinal],
                              capabilities);
  }
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addDerivativeValueTerm(TensorVarPtr trialVar,
                                                                             TensorVarPtr testVar,
                                                                             int derivativeComponentOrdinal,
                                                                             TensorFunctionPtr coefficient,
                                                                             const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::singleDerivative(trialVar->componentCount(), derivativeComponentOrdinal),
                                    TensorComponentOperatorDescriptor::valueOnly(testVar->componentCount()),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addValueDerivativeTerm(TensorVarPtr trialVar,
                                                                             TensorVarPtr testVar,
                                                                             int derivativeComponentOrdinal,
                                                                             TensorFunctionPtr coefficient,
                                                                             const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount()),
                                    TensorComponentOperatorDescriptor::singleDerivative(testVar->componentCount(), derivativeComponentOrdinal),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addL2ValueDerivativeTerm(TensorVarPtr trialVar,
                                                                                TensorVarPtr testVar,
                                                                                int derivativeComponentOrdinal,
                                                                                TensorFunctionPtr coefficient,
                                                                                const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount(), TENSOR_COMPONENT_FUNCTION_SPACE_HVOL),
                                    TensorComponentOperatorDescriptor::singleDerivative(testVar->componentCount(), derivativeComponentOrdinal),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addValueL2ValueTerm(TensorVarPtr trialVar,
                                                                          TensorVarPtr testVar,
                                                                          TensorFunctionPtr coefficient,
                                                                          const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount()),
                                    TensorComponentOperatorDescriptor::valueOnly(testVar->componentCount(), TENSOR_COMPONENT_FUNCTION_SPACE_HVOL),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addL2ValueL2ValueTerm(TensorVarPtr trialVar,
                                                                            TensorVarPtr testVar,
                                                                            TensorFunctionPtr coefficient,
                                                                            const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(trialVar->componentCount(), TENSOR_COMPONENT_FUNCTION_SPACE_HVOL),
                                    TensorComponentOperatorDescriptor::valueOnly(testVar->componentCount(), TENSOR_COMPONENT_FUNCTION_SPACE_HVOL),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormBuilder &TensorBilinearFormBuilder::addDerivativeDerivativeTerm(TensorVarPtr trialVar,
                                                                                  TensorVarPtr testVar,
                                                                                  int trialDerivativeComponentOrdinal,
                                                                                  int testDerivativeComponentOrdinal,
                                                                                  TensorFunctionPtr coefficient,
                                                                                  const TensorOperatorCapabilitySet &capabilities)
{
  _form->addTerm(TensorBilinearTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::singleDerivative(trialVar->componentCount(), trialDerivativeComponentOrdinal),
                                    TensorComponentOperatorDescriptor::singleDerivative(testVar->componentCount(), testDerivativeComponentOrdinal),
                                    coefficient,
                                    TENSOR_BILINEAR_TERM_VOLUME,
                                    capabilities));
  return *this;
}

TensorBilinearFormPtr TensorBilinearFormBuilder::build() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_form == Teuchos::null, std::invalid_argument,
                             "TensorBilinearFormBuilder::build requires a valid form.");
  return _form;
}
