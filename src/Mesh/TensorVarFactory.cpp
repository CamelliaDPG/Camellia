#include "TensorVarFactory.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"
#include "Var.h"

using namespace Camellia;

TensorVarFactory::TensorVarFactory()
{
}

void TensorVarFactory::addTensorVar(TensorVarPtr tensorVar)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorVar == Teuchos::null, std::invalid_argument,
                             "TensorVarFactory::addTensorVar requires a non-null TensorVar.");
  TEUCHOS_TEST_FOR_EXCEPTION(hasTensorVar(tensorVar->name()), std::invalid_argument,
                             "TensorVarFactory already has a TensorVar with the requested name.");
  _tensorVars[tensorVar->name()] = tensorVar;
}

TensorVarPtr TensorVarFactory::tensorVar(const std::string &name) const
{
  auto entry = _tensorVars.find(name);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _tensorVars.end(), std::invalid_argument,
                             "Unknown TensorVar name.");
  return entry->second;
}

bool TensorVarFactory::hasTensorVar(const std::string &name) const
{
  return _tensorVars.find(name) != _tensorVars.end();
}

std::vector<TensorVarPtr> TensorVarFactory::tensorVars() const
{
  std::vector<TensorVarPtr> vars;
  for (const auto &entry : _tensorVars)
  {
    vars.push_back(entry.second);
  }
  return vars;
}

std::vector<TensorVarPtr> TensorVarFactory::directionalTraceFamily(const std::string &baseName,
                                                                   const std::vector<std::string> &componentLabels,
                                                                   const std::vector<VarPtr> &componentVars)
{
  TEUCHOS_TEST_FOR_EXCEPTION(baseName == "", std::invalid_argument,
                             "TensorVarFactory::directionalTraceFamily requires a nonempty base name.");
  TEUCHOS_TEST_FOR_EXCEPTION(componentLabels.empty(), std::invalid_argument,
                             "TensorVarFactory::directionalTraceFamily requires at least one component label.");
  TEUCHOS_TEST_FOR_EXCEPTION(componentLabels.size() != componentVars.size(), std::invalid_argument,
                             "TensorVarFactory::directionalTraceFamily requires one component label per component Var.");

  std::vector<TensorVarPtr> directionalVars;
  directionalVars.reserve(componentLabels.size());
  for (int traceComponentOrdinal=0; traceComponentOrdinal<static_cast<int>(componentLabels.size()); traceComponentOrdinal++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(componentLabels[traceComponentOrdinal] == "", std::invalid_argument,
                               "TensorVarFactory::directionalTraceFamily requires nonempty component labels.");

    std::vector<VarPtr> copiedComponentVars;
    copiedComponentVars.reserve(componentVars.size());
    for (const auto &componentVar : componentVars)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(componentVar == Teuchos::null, std::invalid_argument,
                                 "TensorVarFactory::directionalTraceFamily requires non-null component Vars.");
      copiedComponentVars.push_back(Teuchos::rcp(new Var(*componentVar)));
    }

    directionalVars.push_back(Teuchos::rcp(new TensorVar(baseName + "_" + componentLabels[traceComponentOrdinal],
                                                        TENSOR_TRACE,
                                                        copiedComponentVars,
                                                        traceComponentOrdinal)));
  }
  return directionalVars;
}

std::vector<TensorVarPtr> TensorVarFactory::addDirectionalTraceFamily(const std::string &baseName,
                                                                      const std::vector<std::string> &componentLabels,
                                                                      const std::vector<VarPtr> &componentVars)
{
  auto directionalVars = directionalTraceFamily(baseName, componentLabels, componentVars);
  for (const auto &directionalVar : directionalVars)
  {
    addTensorVar(directionalVar);
  }
  return directionalVars;
}
