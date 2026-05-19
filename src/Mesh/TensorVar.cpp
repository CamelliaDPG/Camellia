#include "TensorVar.h"
#include "Var.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorVar::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_name == "", std::invalid_argument,
                             "TensorVar requires a nonempty name.");
  TEUCHOS_TEST_FOR_EXCEPTION(_componentVars.empty(), std::invalid_argument,
                             "TensorVar requires at least one component Var.");

  for (size_t i=0; i<_componentVars.size(); i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_componentVars[i] == Teuchos::null, std::invalid_argument,
                               "TensorVar requires non-null component Vars.");
    const auto componentVarType = _componentVars[i]->varType();
    const bool isFieldLike = (componentVarType == FIELD);
    const bool isTraceLike = (componentVarType == TRACE) || (componentVarType == FLUX);

    if (_kind == TENSOR_FIELD)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!isFieldLike, std::invalid_argument,
                                 "TensorVar with kind TENSOR_FIELD requires component Vars of FIELD type.");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!isTraceLike, std::invalid_argument,
                                 "TensorVar with kind TENSOR_TRACE requires component Vars of TRACE or FLUX type.");
    }
  }
}

TensorVar::TensorVar(const std::string &name, TensorVarKind kind, const std::vector<VarPtr> &componentVars)
: _name(name), _kind(kind), _componentVars(componentVars)
{
  validateConstructorInputs();
}

const std::string &TensorVar::name() const
{
  return _name;
}

TensorVarKind TensorVar::kind() const
{
  return _kind;
}

bool TensorVar::isField() const
{
  return _kind == TENSOR_FIELD;
}

bool TensorVar::isTrace() const
{
  return _kind == TENSOR_TRACE;
}

int TensorVar::componentCount() const
{
  return static_cast<int>(_componentVars.size());
}

VarPtr TensorVar::componentVar(int componentOrdinal) const
{
  return _componentVars[componentOrdinal];
}

const std::vector<VarPtr> &TensorVar::componentVars() const
{
  return _componentVars;
}

TensorVarPtr TensorVar::copyWithNewName(const std::string &newName) const
{
  std::vector<VarPtr> copiedComponentVars;
  copiedComponentVars.reserve(_componentVars.size());
  for (size_t i=0; i<_componentVars.size(); i++)
  {
    VarPtr oldVar = _componentVars[i];
    VarPtr copiedVar = Teuchos::rcp(new Var(*oldVar));
    copiedComponentVars.push_back(copiedVar);
  }
  return Teuchos::rcp(new TensorVar(newName, _kind, copiedComponentVars));
}
