#include "TensorMesh.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorMesh::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_topology == Teuchos::null, std::invalid_argument,
                             "TensorMesh requires a non-null TensorMeshTopology.");
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorVarFactory == Teuchos::null, std::invalid_argument,
                             "TensorMesh requires a non-null TensorVarFactory.");
  TEUCHOS_TEST_FOR_EXCEPTION(_defaultBasisFamily == Teuchos::null, std::invalid_argument,
                             "TensorMesh requires a non-null default BasisFamily.");
  TEUCHOS_TEST_FOR_EXCEPTION(_defaultPolyOrder < 0, std::invalid_argument,
                             "TensorMesh requires nonnegative default polynomial order.");
  TEUCHOS_TEST_FOR_EXCEPTION(_testSpaceEnrichment < 0, std::invalid_argument,
                             "TensorMesh requires nonnegative test-space enrichment.");
}

void TensorMesh::validateDiscretization(const TensorVarDiscretizationPtr &discretization) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(discretization == Teuchos::null, std::invalid_argument,
                             "TensorMesh requires non-null TensorVarDiscretization objects.");
  TEUCHOS_TEST_FOR_EXCEPTION(discretization->tensorVar() == Teuchos::null, std::invalid_argument,
                             "TensorMesh requires TensorVarDiscretization to hold a non-null TensorVar.");
  TEUCHOS_TEST_FOR_EXCEPTION(discretization->componentCount() != _topology->componentCount(), std::invalid_argument,
                             "TensorMesh requires TensorVarDiscretization rank to match TensorMeshTopology component count.");
}

TensorMesh::TensorMesh(Teuchos::RCP<TensorMeshTopology> topology,
                       TensorVarFactoryPtr tensorVarFactory,
                       BasisFamilyPtr defaultBasisFamily,
                       int defaultPolyOrder,
                       int testSpaceEnrichment)
: _topology(topology), _tensorVarFactory(tensorVarFactory), _defaultBasisFamily(defaultBasisFamily),
  _defaultPolyOrder(defaultPolyOrder), _testSpaceEnrichment(testSpaceEnrichment)
{
  validateConstructorInputs();
}

Teuchos::RCP<TensorMeshTopology> TensorMesh::topology() const
{
  return _topology;
}

TensorVarFactoryPtr TensorMesh::tensorVarFactory() const
{
  return _tensorVarFactory;
}

BasisFamilyPtr TensorMesh::defaultBasisFamily() const
{
  return _defaultBasisFamily;
}

int TensorMesh::defaultPolyOrder() const
{
  return _defaultPolyOrder;
}

int TensorMesh::testSpaceEnrichment() const
{
  return _testSpaceEnrichment;
}

void TensorMesh::addTrialDiscretization(TensorVarDiscretizationPtr discretization)
{
  validateDiscretization(discretization);
  const std::string &name = discretization->tensorVar()->name();
  TEUCHOS_TEST_FOR_EXCEPTION(_trialDiscretizations.find(name) != _trialDiscretizations.end(), std::invalid_argument,
                             "TensorMesh already has a trial discretization with the requested name.");
  _tensorVarFactory->addTensorVar(discretization->tensorVar());
  _trialDiscretizations[name] = discretization;
}

void TensorMesh::addTestDiscretization(TensorVarDiscretizationPtr discretization)
{
  validateDiscretization(discretization);
  const std::string &name = discretization->tensorVar()->name();
  TEUCHOS_TEST_FOR_EXCEPTION(_testDiscretizations.find(name) != _testDiscretizations.end(), std::invalid_argument,
                             "TensorMesh already has a test discretization with the requested name.");
  if (!_tensorVarFactory->hasTensorVar(name))
  {
    _tensorVarFactory->addTensorVar(discretization->tensorVar());
  }
  _testDiscretizations[name] = discretization;
}

bool TensorMesh::hasTrialVar(const std::string &name) const
{
  return _trialDiscretizations.find(name) != _trialDiscretizations.end();
}

bool TensorMesh::hasTestVar(const std::string &name) const
{
  return _testDiscretizations.find(name) != _testDiscretizations.end();
}

TensorVarDiscretizationPtr TensorMesh::trialDiscretization(const std::string &name) const
{
  auto entry = _trialDiscretizations.find(name);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _trialDiscretizations.end(), std::invalid_argument,
                             "Unknown trial TensorVar discretization name.");
  return entry->second;
}

TensorVarDiscretizationPtr TensorMesh::testDiscretization(const std::string &name) const
{
  auto entry = _testDiscretizations.find(name);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _testDiscretizations.end(), std::invalid_argument,
                             "Unknown test TensorVar discretization name.");
  return entry->second;
}

std::vector<TensorVarDiscretizationPtr> TensorMesh::trialDiscretizations() const
{
  std::vector<TensorVarDiscretizationPtr> discretizations;
  for (const auto &entry : _trialDiscretizations)
  {
    discretizations.push_back(entry.second);
  }
  return discretizations;
}

std::vector<TensorVarDiscretizationPtr> TensorMesh::testDiscretizations() const
{
  std::vector<TensorVarDiscretizationPtr> discretizations;
  for (const auto &entry : _testDiscretizations)
  {
    discretizations.push_back(entry.second);
  }
  return discretizations;
}

BasisFamilyPtr TensorMesh::effectiveBasisFamily(const TensorVarDiscretizationPtr &discretization) const
{
  validateDiscretization(discretization);
  if (discretization->hasBasisFamilyOverride())
  {
    return discretization->basisFamilyOverride();
  }
  return _defaultBasisFamily;
}

int TensorMesh::effectivePolyOrderDelta(const TensorVarDiscretizationPtr &discretization, int componentOrdinal, bool isTest) const
{
  validateDiscretization(discretization);
  int delta = discretization->componentPolyOrderDelta(componentOrdinal);
  if (isTest)
  {
    delta += _testSpaceEnrichment;
  }
  return delta;
}

int TensorMesh::effectivePolyOrder(const TensorVarDiscretizationPtr &discretization,
                                   const ComponentCellID1D &cellID,
                                   int componentOrdinal,
                                   bool isTest) const
{
  validateDiscretization(discretization);
  const auto componentTopo = _topology->componentMeshTopology(componentOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION(componentTopo == Teuchos::null, std::invalid_argument,
                             "TensorMesh::effectivePolyOrder found null component topology.");
  const int topologyP = componentTopo->cellPolyOrder(cellID);
  const int delta = effectivePolyOrderDelta(discretization, componentOrdinal, isTest);
  return topologyP + _defaultPolyOrder + delta;
}
