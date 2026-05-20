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
   *  TensorDofLayout. The current implementation is intentionally modest in scope:
   *  \begin{itemize}
   *    \item it is per-variable rather than block-composed across all registered variables,
   *    \item it is parameterized by TensorSpaceRole rather than split into separate trial/test classes,
   *    \item and it is flat-buffer-first rather than Tpetra-centric.
   *  \end{itemize}
   *
   *  The role of this object is to make local/global movement explicit while leaving room for later
   *  adapters and richer vector/container abstractions.
   */
  class TensorGatherScatter
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorSpaceRole _role;
    std::string _varName;
    TensorDofLayout::VariableLayout _variableLayout;

  public:
    TensorGatherScatter(TensorMeshPtr tensorMesh,
                        TensorDofLayoutPtr tensorDofLayout,
                        TensorSpaceRole role,
                        const std::string &varName);

    TensorMeshPtr tensorMesh() const;
    TensorDofLayoutPtr tensorDofLayout() const;
    TensorSpaceRole role() const;
    const std::string &varName() const;
    const TensorDofLayout::VariableLayout &variableLayout() const;

    IndexType localDofCount() const;

    /**
     *  \brief Scatter one local variable block into a flat global buffer.
     *
     *  In the present first draft, the scatter uses a simple offset-based placement into the target buffer.
     *  Later issues may add richer global indexing and adapter layers on top of this abstraction.
     */
    void scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                              std::vector<double> &globalCoefficients,
                              IndexType globalOffset = 0) const;

    /**
     *  \brief Gather one local variable block from a flat global buffer.
     */
    void gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                             std::vector<double> &localCoefficients,
                             IndexType globalOffset = 0) const;
  };
}

#endif
