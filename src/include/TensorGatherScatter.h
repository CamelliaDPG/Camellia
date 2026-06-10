#ifndef Camellia_TensorGatherScatter_h
#define Camellia_TensorGatherScatter_h

#include <vector>

#include "TensorDofLayout.h"

namespace Camellia
{
  class TensorGatherScatter;
  using TensorGatherScatterPtr = Teuchos::RCP<TensorGatherScatter>;

  /**
   *  \brief Tensor-native coefficient movement helper parameterized by exact local factor-space shape.
   *
   *  TensorGatherScatter no longer treats a variable as having one exact fixed local block.  Instead,
   *  it provides flat-buffer movement for a supplied exact local factor-space shape.
   */
  class TensorGatherScatter
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorSpaceRole _role;
    std::string _varName;
    TensorLocalOrdering _ordering;
    TensorDofLayout::VariableLayout _variableLayout;

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

    IndexType localDofCount(const TensorFactorSpaceShape &shape) const;

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
