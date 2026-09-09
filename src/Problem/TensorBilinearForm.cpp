#include "TensorBilinearForm.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorOperatorCapabilitySet::TensorOperatorCapabilitySet()
{
}

TensorOperatorCapabilitySet::TensorOperatorCapabilitySet(const std::set<TensorOperatorBackend> &supportedBackends)
: _supportedBackends(supportedBackends)
{
}

const std::set<TensorOperatorBackend> &TensorOperatorCapabilitySet::supportedBackends() const
{
  return _supportedBackends;
}

bool TensorOperatorCapabilitySet::supports(TensorOperatorBackend backend) const
{
  return _supportedBackends.find(backend) != _supportedBackends.end();
}

TensorOperatorCapabilitySet TensorOperatorCapabilitySet::matrixFreeOnly()
{
  return TensorOperatorCapabilitySet({TENSOR_OPERATOR_BACKEND_MATRIX_FREE});
}

TensorOperatorCapabilitySet TensorOperatorCapabilitySet::matrixFreeAndPAMatrix()
{
  return TensorOperatorCapabilitySet({TENSOR_OPERATOR_BACKEND_MATRIX_FREE,
                                      TENSOR_OPERATOR_BACKEND_PAMATRIX});
}

void TensorComponentOperatorDescriptor::validate() const
{
  for (int derivativeComponentOrdinal : _derivativeComponentOrdinals)
  {
    TEUCHOS_TEST_FOR_EXCEPTION((derivativeComponentOrdinal < 0) || (derivativeComponentOrdinal >= static_cast<int>(_operators.size())),
                               std::invalid_argument,
                               "TensorComponentOperatorDescriptor derivative component ordinal out of range.");
    TEUCHOS_TEST_FOR_EXCEPTION(_operators[derivativeComponentOrdinal] != TENSOR_COMPONENT_OP_DERIVATIVE,
                               std::invalid_argument,
                               "TensorComponentOperatorDescriptor derivative metadata must agree with operator tags.");
  }
  for (int d=0; d<static_cast<int>(_operators.size()); d++)
  {
    if (_operators[d] == TENSOR_COMPONENT_OP_DERIVATIVE)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_derivativeComponentOrdinals.end() == std::find(_derivativeComponentOrdinals.begin(),
                                                                                  _derivativeComponentOrdinals.end(), d),
                                 std::invalid_argument,
                                 "TensorComponentOperatorDescriptor derivative operator missing from derivative metadata list.");
    }
  }
}

TensorComponentOperatorDescriptor::TensorComponentOperatorDescriptor()
{
}

TensorComponentOperatorDescriptor::TensorComponentOperatorDescriptor(const std::vector<TensorComponentOperator> &operators,
                                                                     const std::vector<int> &derivativeComponentOrdinals)
: _operators(operators), _derivativeComponentOrdinals(derivativeComponentOrdinals)
{
  validate();
}

int TensorComponentOperatorDescriptor::componentCount() const
{
  return static_cast<int>(_operators.size());
}

TensorComponentOperator TensorComponentOperatorDescriptor::op(int componentOrdinal) const
{
  return _operators[componentOrdinal];
}

bool TensorComponentOperatorDescriptor::isIdentityValueAction() const
{
  return isValueOnly();
}

bool TensorComponentOperatorDescriptor::isValueOnly() const
{
  for (auto op : _operators)
  {
    if (op != TENSOR_COMPONENT_OP_VALUE) return false;
  }
  return true;
}

bool TensorComponentOperatorDescriptor::hasDerivatives() const
{
  return !_derivativeComponentOrdinals.empty();
}

bool TensorComponentOperatorDescriptor::takesDerivativeInComponent(int componentOrdinal) const
{
  return _derivativeComponentOrdinals.end() != std::find(_derivativeComponentOrdinals.begin(),
                                                         _derivativeComponentOrdinals.end(),
                                                         componentOrdinal);
}

const std::vector<TensorComponentOperator> &TensorComponentOperatorDescriptor::operators() const
{
  return _operators;
}

const std::vector<int> &TensorComponentOperatorDescriptor::derivativeComponentOrdinals() const
{
  return _derivativeComponentOrdinals;
}

TensorComponentOperatorDescriptor TensorComponentOperatorDescriptor::valueOnly(int componentCount)
{
  return TensorComponentOperatorDescriptor(std::vector<TensorComponentOperator>(componentCount, TENSOR_COMPONENT_OP_VALUE));
}

TensorComponentOperatorDescriptor TensorComponentOperatorDescriptor::singleDerivative(int componentCount, int derivativeComponentOrdinal)
{
  std::vector<TensorComponentOperator> operators(componentCount, TENSOR_COMPONENT_OP_VALUE);
  operators[derivativeComponentOrdinal] = TENSOR_COMPONENT_OP_DERIVATIVE;
  return TensorComponentOperatorDescriptor(operators, {derivativeComponentOrdinal});
}

void TensorBilinearTerm::validate() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_trialVar == Teuchos::null, std::invalid_argument,
                             "TensorBilinearTerm requires a non-null trial TensorVar.");
  TEUCHOS_TEST_FOR_EXCEPTION(_testVar == Teuchos::null, std::invalid_argument,
                             "TensorBilinearTerm requires a non-null test TensorVar.");
  TEUCHOS_TEST_FOR_EXCEPTION(_coefficient == Teuchos::null, std::invalid_argument,
                             "TensorBilinearTerm requires a non-null TensorFunction coefficient.");
  TEUCHOS_TEST_FOR_EXCEPTION(_trialVar->componentCount() != _testVar->componentCount(), std::invalid_argument,
                             "TensorBilinearTerm requires matching trial/test tensor ranks.");
  TEUCHOS_TEST_FOR_EXCEPTION(_trialOperator.componentCount() != _trialVar->componentCount(), std::invalid_argument,
                             "TensorBilinearTerm trial operator rank must match trial variable rank.");
  TEUCHOS_TEST_FOR_EXCEPTION(_testOperator.componentCount() != _testVar->componentCount(), std::invalid_argument,
                             "TensorBilinearTerm test operator rank must match test variable rank.");
  TEUCHOS_TEST_FOR_EXCEPTION(_capabilities.supportedBackends().empty(), std::invalid_argument,
                             "TensorBilinearTerm requires at least one candidate backend capability.");
}

TensorBilinearTerm::TensorBilinearTerm(TensorVarPtr trialVar,
                                       TensorVarPtr testVar,
                                       const TensorComponentOperatorDescriptor &trialOperator,
                                       const TensorComponentOperatorDescriptor &testOperator,
                                       TensorFunctionPtr coefficient,
                                       TensorBilinearTermDomain domain,
                                       const TensorOperatorCapabilitySet &capabilities)
: _trialVar(trialVar),
  _testVar(testVar),
  _trialOperator(trialOperator),
  _testOperator(testOperator),
  _coefficient(coefficient),
  _domain(domain),
  _capabilities(capabilities)
{
  validate();
}

TensorVarPtr TensorBilinearTerm::trialVar() const
{
  return _trialVar;
}

TensorVarPtr TensorBilinearTerm::testVar() const
{
  return _testVar;
}

const TensorComponentOperatorDescriptor &TensorBilinearTerm::trialOperator() const
{
  return _trialOperator;
}

const TensorComponentOperatorDescriptor &TensorBilinearTerm::testOperator() const
{
  return _testOperator;
}

TensorFunctionPtr TensorBilinearTerm::coefficient() const
{
  return _coefficient;
}

TensorBilinearTermDomain TensorBilinearTerm::domain() const
{
  return _domain;
}

const TensorOperatorCapabilitySet &TensorBilinearTerm::capabilities() const
{
  return _capabilities;
}

int TensorBilinearTerm::componentCount() const
{
  return _trialVar->componentCount();
}

bool TensorBilinearTerm::isVolumeTerm() const
{
  return _domain == TENSOR_BILINEAR_TERM_VOLUME;
}

bool TensorBilinearTerm::isBoundaryTerm() const
{
  return _domain == TENSOR_BILINEAR_TERM_BOUNDARY;
}

bool TensorBilinearTerm::isMatrixFreeCandidate() const
{
  return _capabilities.supports(TENSOR_OPERATOR_BACKEND_MATRIX_FREE);
}

bool TensorBilinearTerm::isPAMatrixCandidate() const
{
  return _capabilities.supports(TENSOR_OPERATOR_BACKEND_PAMATRIX);
}

bool TensorBilinearTerm::couplesMultipleComponents() const
{
  return _coefficient->couplingStructure().isFullyCoupled() || _coefficient->couplingStructure().isPartiallyCoupled();
}

TensorBilinearForm::TensorBilinearForm(const std::string &name)
: _name(name)
{
}

const std::string &TensorBilinearForm::name() const
{
  return _name;
}

int TensorBilinearForm::termCount() const
{
  return static_cast<int>(_terms.size());
}

const std::vector<TensorBilinearTerm> &TensorBilinearForm::terms() const
{
  return _terms;
}

const TensorBilinearTerm &TensorBilinearForm::term(int termOrdinal) const
{
  return _terms[termOrdinal];
}

void TensorBilinearForm::addTerm(const TensorBilinearTerm &term)
{
  _terms.push_back(term);
}

std::set<std::string> TensorBilinearForm::trialVarNames() const
{
  std::set<std::string> names;
  for (const auto &term : _terms)
  {
    names.insert(term.trialVar()->name());
  }
  return names;
}

std::set<std::string> TensorBilinearForm::testVarNames() const
{
  std::set<std::string> names;
  for (const auto &term : _terms)
  {
    names.insert(term.testVar()->name());
  }
  return names;
}

std::set<int> TensorBilinearForm::coupledComponentOrdinals() const
{
  std::set<int> coupled;
  for (const auto &term : _terms)
  {
    const auto &scope = term.coefficient()->dependencyScope().componentOrdinals();
    coupled.insert(scope.begin(), scope.end());
  }
  return coupled;
}

bool TensorBilinearForm::hasBoundaryTerms() const
{
  for (const auto &term : _terms)
  {
    if (term.isBoundaryTerm()) return true;
  }
  return false;
}

bool TensorBilinearForm::allTermsSupport(TensorOperatorBackend backend) const
{
  for (const auto &term : _terms)
  {
    if (!term.capabilities().supports(backend)) return false;
  }
  return true;
}

void TensorBilinearForm::validateAgainst(const TensorMeshPtr &tensorMesh) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorBilinearForm::validateAgainst requires a non-null TensorMesh.");

  for (const auto &term : _terms)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!tensorMesh->hasTrialVar(term.trialVar()->name()), std::invalid_argument,
                               "TensorBilinearForm::validateAgainst found an unregistered trial variable.");
    TEUCHOS_TEST_FOR_EXCEPTION(!tensorMesh->hasTestVar(term.testVar()->name()), std::invalid_argument,
                               "TensorBilinearForm::validateAgainst found an unregistered test variable.");
    TEUCHOS_TEST_FOR_EXCEPTION(term.componentCount() != tensorMesh->topology()->componentCount(), std::invalid_argument,
                               "TensorBilinearForm::validateAgainst found a term with rank incompatible with the TensorMesh topology.");
  }
}
