#ifndef Camellia_TensorBilinearFormBuilder_h
#define Camellia_TensorBilinearFormBuilder_h

#include "TensorBilinearForm.h"

namespace Camellia
{
  /**
   *  \brief Convenience builder for the descriptor-side tensor bilinear-form layer.
   *
   *  The main TensorBilinearForm and TensorBilinearTerm classes remain the canonical descriptor types.
   *  This helper exists only to reduce repetitive boilerplate in common volume-term construction paths.
   */
  class TensorBilinearFormBuilder
  {
    TensorBilinearFormPtr _form;
  public:
    TensorBilinearFormBuilder(const std::string &name = "");

    TensorBilinearFormBuilder &addValueValueTerm(TensorVarPtr trialVar,
                                                 TensorVarPtr testVar,
                                                 TensorFunctionPtr coefficient,
                                                 const TensorOperatorCapabilitySet &capabilities = TensorOperatorCapabilitySet::matrixFreeOnly());

    TensorBilinearFormBuilder &addDerivativeValueTerm(TensorVarPtr trialVar,
                                                      TensorVarPtr testVar,
                                                      int derivativeComponentOrdinal,
                                                      TensorFunctionPtr coefficient,
                                                      const TensorOperatorCapabilitySet &capabilities = TensorOperatorCapabilitySet::matrixFreeOnly());

    TensorBilinearFormBuilder &addValueDerivativeTerm(TensorVarPtr trialVar,
                                                      TensorVarPtr testVar,
                                                      int derivativeComponentOrdinal,
                                                      TensorFunctionPtr coefficient,
                                                      const TensorOperatorCapabilitySet &capabilities = TensorOperatorCapabilitySet::matrixFreeOnly());

    TensorBilinearFormBuilder &addDerivativeDerivativeTerm(TensorVarPtr trialVar,
                                                           TensorVarPtr testVar,
                                                           int trialDerivativeComponentOrdinal,
                                                           int testDerivativeComponentOrdinal,
                                                           TensorFunctionPtr coefficient,
                                                           const TensorOperatorCapabilitySet &capabilities = TensorOperatorCapabilitySet::matrixFreeOnly());

    TensorBilinearFormPtr build() const;
  };
}

#endif
