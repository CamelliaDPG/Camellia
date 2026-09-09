#include "TensorVarFactory.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

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
