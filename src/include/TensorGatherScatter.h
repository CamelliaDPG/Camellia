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
   *  This class is the first local/global data-motion operator layer above TensorMesh and TensorDofLayout.
   *  It is intentionally per-variable in the current implementation and is parameterized by TensorSpaceRole
   *  rather than split into separate trial and test class families.
   *
   *  The current implementation is flat-vector-first and local-layout-driven. It uses raw container data
   *  rather than making Tpetra part of the core abstraction, but the API is intended to leave room for
   *  later adapters or wrappers.
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

    void scatterLocalToGlobal(const std::vector<double> &localCoefficients,
                              std::vector<double> &globalCoefficients,
                              IndexType globalOffset = 0) const;

    void gatherGlobalToLocal(const std::vector<double> &globalCoefficients,
                             std::vector<double> &localCoefficients,
                             IndexType globalOffset = 0) const;
  };
}

#endif
