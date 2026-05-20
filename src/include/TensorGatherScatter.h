#ifndef Camellia_TensorGatherScatter_h
#define Camellia_TensorGatherScatter_h

#include <vector>

#include "TensorDofLayout.h"

namespace Camellia
{
  class TensorGatherScatter;
  using TensorGatherScatterPtr = Teuchos::RCP<TensorGatherScatter>;

  /**
   *  \brief Explicit tensor-native per-variable gather/scatter abstraction.
   *
   *  TensorGatherScatter is the first local/global data-motion operator layer above TensorMesh and
   *  TensorDofLayout. The current implementation supports both variable-centric and cell-centric
   *  local coefficient orderings as same-size alternate flattenings of the same local vector space,
   *  while remaining flat-buffer-first rather than Tpetra-centric.
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

    IndexType localDofCount() const;

    void scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                              std::vector<double> &globalCoefficients,
                              IndexType globalOffset = 0) const;

    void gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                             std::vector<double> &localCoefficients,
                             IndexType globalOffset = 0) const;
  };
}

#endif
