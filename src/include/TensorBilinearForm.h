#ifndef Camellia_TensorBilinearForm_h
#define Camellia_TensorBilinearForm_h

#include <set>
#include <string>
#include <vector>

#include "TensorFunction.h"
#include "TensorMesh.h"

namespace Camellia
{
  class TensorBilinearForm;
  using TensorBilinearFormPtr = Teuchos::RCP<TensorBilinearForm>;

  /**
   *  \brief Small tensor-native operator vocabulary for the descriptor-first realization path.
   *
   *  The goal in issue #298 is not to expose the full legacy or Intrepid operator universe immediately.
   *  Instead, we define a narrow descriptor vocabulary that captures the operator intent needed by the
   *  first tensor-native realization layer.
   */
  enum TensorComponentOperator
  {
    TENSOR_COMPONENT_OP_VALUE,
    TENSOR_COMPONENT_OP_DERIVATIVE
  };

  /**
   *  \brief Volume-versus-boundary classification for tensor bilinear-form terms.
   */
  enum TensorBilinearTermDomain
  {
    TENSOR_BILINEAR_TERM_VOLUME,
    TENSOR_BILINEAR_TERM_BOUNDARY
  };

  /**
   *  \brief Realization-backend categories anticipated by the tensor-native operator layer.
   */
  enum TensorOperatorBackend
  {
    TENSOR_OPERATOR_BACKEND_MATRIX_FREE,
    TENSOR_OPERATOR_BACKEND_PAMATRIX,
    TENSOR_OPERATOR_BACKEND_PARTIALLY_ASSEMBLED,
    TENSOR_OPERATOR_BACKEND_ASSEMBLED
  };

  /**
   *  \brief Backend-capability metadata carried by each tensor bilinear-form term.
   *
   *  This keeps descriptor-side backend selection explicit without prematurely forcing implementation
   *  of all supported backend classes.
   */
  class TensorOperatorCapabilitySet
  {
    std::set<TensorOperatorBackend> _supportedBackends;
  public:
    TensorOperatorCapabilitySet();
    TensorOperatorCapabilitySet(const std::set<TensorOperatorBackend> &supportedBackends);

    const std::set<TensorOperatorBackend> &supportedBackends() const;
    bool supports(TensorOperatorBackend backend) const;

    static TensorOperatorCapabilitySet matrixFreeOnly();
    static TensorOperatorCapabilitySet matrixFreeAndPAMatrix();
  };

  /**
   *  \brief Per-component operator intent for one trial or test tensor variable within a bilinear term.
   */
  class TensorComponentOperatorDescriptor
  {
    std::vector<TensorComponentOperator> _operators;
    std::vector<int> _derivativeComponentOrdinals;

    void validate() const;
  public:
    TensorComponentOperatorDescriptor();
    TensorComponentOperatorDescriptor(const std::vector<TensorComponentOperator> &operators,
                                      const std::vector<int> &derivativeComponentOrdinals = {});

    int componentCount() const;
    TensorComponentOperator op(int componentOrdinal) const;
    bool isIdentityValueAction() const;
    bool isValueOnly() const;
    bool hasDerivatives() const;
    bool takesDerivativeInComponent(int componentOrdinal) const;
    const std::vector<TensorComponentOperator> &operators() const;
    const std::vector<int> &derivativeComponentOrdinals() const;

    static TensorComponentOperatorDescriptor valueOnly(int componentCount);
    static TensorComponentOperatorDescriptor singleDerivative(int componentCount, int derivativeComponentOrdinal);
  };

  /**
   *  \brief One tensor-native bilinear-form term descriptor.
   *
   *  The current issue scopes this to the subset needed for descriptor-first, matrix-free-first
   *  tensor-native realization. In particular, the term records the trial/test tensor variables,
   *  per-component operator intent, coefficient metadata, domain classification, and backend
   *  capability hints.
   */
  class TensorBilinearTerm
  {
    TensorVarPtr _trialVar;
    TensorVarPtr _testVar;
    TensorComponentOperatorDescriptor _trialOperator;
    TensorComponentOperatorDescriptor _testOperator;
    TensorFunctionPtr _coefficient;
    TensorBilinearTermDomain _domain;
    TensorOperatorCapabilitySet _capabilities;

    void validate() const;
  public:
    TensorBilinearTerm(TensorVarPtr trialVar,
                       TensorVarPtr testVar,
                       const TensorComponentOperatorDescriptor &trialOperator,
                       const TensorComponentOperatorDescriptor &testOperator,
                       TensorFunctionPtr coefficient,
                       TensorBilinearTermDomain domain = TENSOR_BILINEAR_TERM_VOLUME,
                       const TensorOperatorCapabilitySet &capabilities = TensorOperatorCapabilitySet::matrixFreeOnly());

    TensorVarPtr trialVar() const;
    TensorVarPtr testVar() const;
    const TensorComponentOperatorDescriptor &trialOperator() const;
    const TensorComponentOperatorDescriptor &testOperator() const;
    TensorFunctionPtr coefficient() const;
    TensorBilinearTermDomain domain() const;
    const TensorOperatorCapabilitySet &capabilities() const;

    int componentCount() const;
    bool isVolumeTerm() const;
    bool isBoundaryTerm() const;
    bool isMatrixFreeCandidate() const;
    bool isPAMatrixCandidate() const;
    bool couplesMultipleComponents() const;
  };

  /**
   *  \brief Descriptor-side tensor bilinear-form abstraction for issue #298.
   *
   *  TensorBilinearForm is intentionally descriptor-first. It does not itself implement operator
   *  realization, gather/scatter, or basis application. Instead, it owns a list of tensor-native
   *  term descriptors that later factory/realization layers can validate and consume.
   */
  class TensorBilinearForm
  {
    std::string _name;
    std::vector<TensorBilinearTerm> _terms;

  public:
    TensorBilinearForm(const std::string &name = "");

    const std::string &name() const;
    int termCount() const;
    const std::vector<TensorBilinearTerm> &terms() const;
    const TensorBilinearTerm &term(int termOrdinal) const;

    void addTerm(const TensorBilinearTerm &term);

    std::set<std::string> trialVarNames() const;
    std::set<std::string> testVarNames() const;
    std::set<int> coupledComponentOrdinals() const;

    bool hasBoundaryTerms() const;
    bool allTermsSupport(TensorOperatorBackend backend) const;

    void validateAgainst(const TensorMeshPtr &tensorMesh) const;
  };
}

#endif
