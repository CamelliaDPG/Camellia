#include "TensorVarDiscretization.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorVarDiscretization::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorVar == Teuchos::null, std::invalid_argument,
                             "TensorVarDiscretization requires a non-null TensorVar.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_componentPolyOrderDeltas.size()) != _tensorVar->componentCount(), std::invalid_argument,
                             "TensorVarDiscretization requires one polynomial-order delta per component.");
}

TensorVarDiscretization::TensorVarDiscretization(TensorVarPtr tensorVar,
                                                 BasisFamilyPtr basisFamilyOverride,
                                                 const std::vector<int> &componentPolyOrderDeltas)
: _tensorVar(tensorVar), _basisFamilyOverride(basisFamilyOverride), _componentPolyOrderDeltas(componentPolyOrderDeltas)
{
  validateConstructorInputs();
}

TensorVarPtr TensorVarDiscretization::tensorVar() const
{
  return _tensorVar;
}

BasisFamilyPtr TensorVarDiscretization::basisFamilyOverride() const
{
  return _basisFamilyOverride;
}

bool TensorVarDiscretization::hasBasisFamilyOverride() const
{
  return _basisFamilyOverride != Teuchos::null;
}

int TensorVarDiscretization::componentCount() const
{
  return static_cast<int>(_componentPolyOrderDeltas.size());
}

int TensorVarDiscretization::componentPolyOrderDelta(int componentOrdinal) const
{
  return _componentPolyOrderDeltas[componentOrdinal];
}

const std::vector<int> &TensorVarDiscretization::componentPolyOrderDeltas() const
{
  return _componentPolyOrderDeltas;
}
