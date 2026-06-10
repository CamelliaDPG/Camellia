#ifndef Camellia_TensorGatherScatter_h
#define Camellia_TensorGatherScatter_h

#include <vector>

#include "TensorDofLayout.h"
#include "TensorCoefficientMap.h"
#include "TensorCoefficientTraversal.h"

namespace Camellia
{
  class TensorGatherScatter;
  using TensorGatherScatterPtr = Teuchos::RCP<TensorGatherScatter>;

  /**
   *  \brief Tensor-native coefficient movement helper based on explicit coefficient index lists.
   *
   *  TensorGatherScatter no longer treats a variable as having one exact fixed local block, nor does it
   *  primarily operate through flat offsets into an ambient per-variable buffer.  Instead, it moves
   *  coefficients through explicit global index lists supplied by TensorCoefficientMap.
   */
  class TensorGatherScatter
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorSpaceRole _role;
    std::string _varName;
    TensorLocalOrdering _ordering;
    TensorDofLayout::VariableLayout _variableLayout;
    TensorCoefficientMap _coefficientMap;

  public:
    TensorGatherScatter(TensorMeshPtr tensorMesh,
                        TensorDofLayoutPtr tensorDofLayout,
                        TensorSpaceRole role,
                        const std::string &varName,
                        TensorLocalOrdering ordering = TENSOR_VARIABLE_ORDERING);

    TensorMeshPtr tensorMesh() const;
    TensorDofLayoutPtr tensorDofLayout() const;
    TensorSpaceRole role() const;
    const std::string &varName() const;
    TensorLocalOrdering ordering() const;
    const TensorDofLayout::VariableLayout &variableLayout() const;
    const TensorCoefficientMap &coefficientMap() const;

    IndexType coefficientCount() const;
    IndexType localDofCount(const TensorFactorSpaceShape &shape) const;

    void scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                              const std::vector<IndexType> &globalIndices,
                              std::vector<double> &globalCoefficients) const;

    void gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                             const std::vector<IndexType> &globalIndices,
                             std::vector<double> &localCoefficients) const;

    void scatterByTraversal(const TensorCoefficientTraversal &traversal,
                            const std::vector<double> &localCoefficients,
                            std::vector<double> &globalCoefficients) const;

    void gatherByTraversal(const TensorCoefficientTraversal &traversal,
                           const std::vector<double> &globalCoefficients,
                           std::vector<double> &localCoefficients) const;

    void scatterLocalToVolumeCell(const std::vector<double> &localCoefficients,
                                  const TensorMeshTopology::TensorCellTuple &cellTuple,
                                  std::vector<double> &globalCoefficients) const;

    void gatherVolumeCellToLocal(const std::vector<double> &globalCoefficients,
                                 const TensorMeshTopology::TensorCellTuple &cellTuple,
                                 std::vector<double> &localCoefficients) const;

    void scatterLocalToSkeletonEntity(const std::vector<double> &localCoefficients,
                                      const TensorMeshTopology::TensorSkeletonEntity &entity,
                                      std::vector<double> &globalCoefficients) const;

    void gatherSkeletonEntityToLocal(const std::vector<double> &globalCoefficients,
                                     const TensorMeshTopology::TensorSkeletonEntity &entity,
                                     std::vector<double> &localCoefficients) const;

    void scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                              const TensorFactorSpaceShape &shape,
                              std::vector<double> &globalCoefficients,
                              IndexType globalOffset = 0) const;

    void gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                             const TensorFactorSpaceShape &shape,
                             std::vector<double> &localCoefficients,
                             IndexType globalOffset = 0) const;
  };
}

#endif
