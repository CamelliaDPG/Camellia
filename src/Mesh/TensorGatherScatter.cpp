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
  _variableLayout(tensorDofLayout->layout(varName, role)),
  _coefficientMap(tensorMesh, tensorDofLayout, role, varName, ordering)
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

const TensorCoefficientMap &TensorGatherScatter::coefficientMap() const
{
  return _coefficientMap;
}

IndexType TensorGatherScatter::coefficientCount() const
{
  return _coefficientMap.coefficientCount();
}

IndexType TensorGatherScatter::localDofCount(const TensorFactorSpaceShape &shape) const
{
  return _tensorDofLayout->localDofCount(shape, _ordering);
}

void TensorGatherScatter::scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                                               const std::vector<IndexType> &globalIndices,
                                               std::vector<double> &globalCoefficients) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localCoefficients.size()) != static_cast<IndexType>(globalIndices.size()), std::invalid_argument,
                             "TensorGatherScatter::scatterLocalToGlobal local coefficient count must match the number of global indices.");

  for (IndexType i=0; i<static_cast<IndexType>(globalIndices.size()); i++)
  {
    const IndexType globalIndex = globalIndices[i];
    TEUCHOS_TEST_FOR_EXCEPTION((globalIndex < 0) || (globalIndex >= static_cast<IndexType>(globalCoefficients.size())), std::invalid_argument,
                               "TensorGatherScatter::scatterLocalToGlobal global index out of range.");
    globalCoefficients[globalIndex] += localCoefficients[i];
  }
}

void TensorGatherScatter::gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                                              const std::vector<IndexType> &globalIndices,
                                              std::vector<double> &localCoefficients) const
{
  localCoefficients.resize(globalIndices.size());
  for (IndexType i=0; i<static_cast<IndexType>(globalIndices.size()); i++)
  {
    const IndexType globalIndex = globalIndices[i];
    TEUCHOS_TEST_FOR_EXCEPTION((globalIndex < 0) || (globalIndex >= static_cast<IndexType>(globalCoefficients.size())), std::invalid_argument,
                               "TensorGatherScatter::gatherGlobalToLocal global index out of range.");
    localCoefficients[i] = globalCoefficients[globalIndex];
  }
}

void TensorGatherScatter::scatterByTraversal(const TensorCoefficientTraversal &traversal,
                                          const std::vector<double> &localCoefficients,
                                          std::vector<double> &globalCoefficients) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localCoefficients.size()) != traversal.localDofCount(), std::invalid_argument,
                             "TensorGatherScatter::scatterByTraversal local coefficient count does not match traversal local dof count.");
  for (IndexType localOrdinal=0; localOrdinal<traversal.localDofCount(); localOrdinal++)
  {
    const IndexType globalIndex = traversal.globalIndexForLocalOrdinal(localOrdinal, _tensorDofLayout);
    TEUCHOS_TEST_FOR_EXCEPTION((globalIndex < 0) || (globalIndex >= static_cast<IndexType>(globalCoefficients.size())), std::invalid_argument,
                               "TensorGatherScatter::scatterByTraversal global index out of range.");
    globalCoefficients[globalIndex] += localCoefficients[localOrdinal];
  }
}

void TensorGatherScatter::gatherByTraversal(const TensorCoefficientTraversal &traversal,
                                            const std::vector<double> &globalCoefficients,
                                            std::vector<double> &localCoefficients) const
{
  localCoefficients.resize(traversal.localDofCount());
  for (IndexType localOrdinal=0; localOrdinal<traversal.localDofCount(); localOrdinal++)
  {
    const IndexType globalIndex = traversal.globalIndexForLocalOrdinal(localOrdinal, _tensorDofLayout);
    TEUCHOS_TEST_FOR_EXCEPTION((globalIndex < 0) || (globalIndex >= static_cast<IndexType>(globalCoefficients.size())), std::invalid_argument,
                               "TensorGatherScatter::gatherByTraversal global index out of range.");
    localCoefficients[localOrdinal] = globalCoefficients[globalIndex];
  }
}

void TensorGatherScatter::scatterLocalToVolumeCell(const std::vector<double> &localCoefficients,
                                                   const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                   std::vector<double> &globalCoefficients) const
{
  scatterByTraversal(_coefficientMap.volumeTraversal(cellTuple), localCoefficients, globalCoefficients);
}

void TensorGatherScatter::gatherVolumeCellToLocal(const std::vector<double> &globalCoefficients,
                                                  const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                  std::vector<double> &localCoefficients) const
{
  gatherByTraversal(_coefficientMap.volumeTraversal(cellTuple), globalCoefficients, localCoefficients);
}

void TensorGatherScatter::scatterLocalToSkeletonEntity(const std::vector<double> &localCoefficients,
                                                       const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                       std::vector<double> &globalCoefficients) const
{
  scatterByTraversal(_coefficientMap.skeletonTraversal(entity), localCoefficients, globalCoefficients);
}

void TensorGatherScatter::gatherSkeletonEntityToLocal(const std::vector<double> &globalCoefficients,
                                                      const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                      std::vector<double> &localCoefficients) const
{
  gatherByTraversal(_coefficientMap.skeletonTraversal(entity), globalCoefficients, localCoefficients);
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

  std::vector<IndexType> globalIndices(localDofCount(shape));
  for (IndexType i=0; i<localDofCount(shape); i++)
  {
    globalIndices[i] = globalOffset + i;
  }
  scatterLocalToGlobal(localCoefficients, globalIndices, globalCoefficients);
}

void TensorGatherScatter::gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                                              const TensorFactorSpaceShape &shape,
                                              std::vector<double> &localCoefficients,
                                              IndexType globalOffset) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(globalCoefficients.size()) < globalOffset + localDofCount(shape), std::invalid_argument,
                             "TensorGatherScatter::gatherGlobalToLocal global buffer too small for requested offset and exact local dof count.");

  std::vector<IndexType> globalIndices(localDofCount(shape));
  for (IndexType i=0; i<localDofCount(shape); i++)
  {
    globalIndices[i] = globalOffset + i;
  }
  gatherGlobalToLocal(globalCoefficients, globalIndices, localCoefficients);
}
