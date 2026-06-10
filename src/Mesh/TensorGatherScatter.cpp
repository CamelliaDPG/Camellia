#include "TensorGatherScatter.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorGatherScatter::TensorGatherScatter(TensorMeshPtr tensorMesh,
                                         TensorDofLayoutPtr tensorDofLayout,
                                         TensorSpaceRole role,
                                         const std::string &varName,
                                         TensorLocalOrdering ordering)
: _tensorMesh(tensorMesh),
  _tensorDofLayout(tensorDofLayout),
  _role(role),
  _varName(varName),
  _ordering(ordering),
  _variableLayout(tensorDofLayout->layout(varName, role))
{
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorGatherScatter requires a non-null TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorGatherScatter requires a non-null TensorDofLayout.");
}

TensorMeshPtr TensorGatherScatter::tensorMesh() const
{
  return _tensorMesh;
}

TensorDofLayoutPtr TensorGatherScatter::tensorDofLayout() const
{
  return _tensorDofLayout;
}

TensorSpaceRole TensorGatherScatter::role() const
{
  return _role;
}

const std::string &TensorGatherScatter::varName() const
{
  return _varName;
}

TensorLocalOrdering TensorGatherScatter::ordering() const
{
  return _ordering;
}

const TensorDofLayout::VariableLayout &TensorGatherScatter::variableLayout() const
{
  return _variableLayout;
}

IndexType TensorGatherScatter::localDofCount(const TensorFactorSpaceShape &shape) const
{
  return _tensorDofLayout->localDofCount(shape, _ordering);
}

void TensorGatherScatter::scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                                               const TensorFactorSpaceShape &shape,
                                               std::vector<double> &globalCoefficients,
                                               IndexType globalOffset) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localCoefficients.size()) != localDofCount(shape), std::invalid_argument,
                             "TensorGatherScatter::scatterLocalToGlobal local coefficient count does not match shape-local dof count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(globalCoefficients.size()) < globalOffset + localDofCount(shape), std::invalid_argument,
                             "TensorGatherScatter::scatterLocalToGlobal global buffer too small for requested offset and exact local dof count.");

  for (IndexType i=0; i<localDofCount(shape); i++)
  {
    globalCoefficients[globalOffset + i] += localCoefficients[i];
  }
}

void TensorGatherScatter::gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                                              const TensorFactorSpaceShape &shape,
                                              std::vector<double> &localCoefficients,
                                              IndexType globalOffset) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(globalCoefficients.size()) < globalOffset + localDofCount(shape), std::invalid_argument,
                             "TensorGatherScatter::gatherGlobalToLocal global buffer too small for requested offset and exact local dof count.");

  localCoefficients.resize(localDofCount(shape));
  for (IndexType i=0; i<localDofCount(shape); i++)
  {
    localCoefficients[i] = globalCoefficients[globalOffset + i];
  }
}
