#ifndef Camellia_TensorDofLayout_h
#define Camellia_TensorDofLayout_h

#include <map>
#include <string>
#include <vector>

#include "TensorMesh.h"

namespace Camellia
{
  class TensorDofLayout;
  using TensorDofLayoutPtr = Teuchos::RCP<TensorDofLayout>;

  /**
   *  \brief Selects whether a layout belongs to the trial or test registration set.
   */
  enum TensorSpaceRole
  {
    TENSOR_TRIAL_SPACE,
    TENSOR_TEST_SPACE
  };

  /**
   *  \brief Explicit tensor-native local layout abstraction.
   *
   *  TensorDofLayout provides local coefficient-layout metadata derived from TensorMesh.
   *  It is intentionally local in scope: it defines component cardinalities, component strides,
   *  and deterministic local ordinal maps for registered trial/test discretizations, but does
   *  not attempt to provide global numbering or gather/scatter behavior.
   */
  class TensorDofLayout
  {
  public:
    /**
     *  \brief Concrete local-layout metadata for one registered tensor variable.
     *
     *  The current implementation stores one cardinality per component, one stride per component,
     *  and the resulting tensor-product local dof count. Lower-indexed components vary fastest.
     */
    struct VariableLayout
    {
      std::string _name;
      TensorSpaceRole _role;
      std::vector<int> _componentCardinalities;
      std::vector<IndexType> _componentStrides;
      IndexType _totalDofs = 0;

      VariableLayout() = default;
      VariableLayout(const std::string &name,
                     TensorSpaceRole role,
                     const std::vector<int> &componentCardinalities,
                     const std::vector<IndexType> &componentStrides,
                     IndexType totalDofs)
      : _name(name), _role(role), _componentCardinalities(componentCardinalities),
        _componentStrides(componentStrides), _totalDofs(totalDofs) {}

      const std::string &name() const { return _name; }
      TensorSpaceRole role() const { return _role; }
      int componentCount() const { return static_cast<int>(_componentCardinalities.size()); }
      int componentCardinality(int componentOrdinal) const { return _componentCardinalities[componentOrdinal]; }
      IndexType componentStride(int componentOrdinal) const { return _componentStrides[componentOrdinal]; }
      IndexType totalDofs() const { return _totalDofs; }

      bool operator==(const VariableLayout &other) const
      {
        return (_name == other._name) && (_role == other._role) && (_componentCardinalities == other._componentCardinalities) && (_componentStrides == other._componentStrides) && (_totalDofs == other._totalDofs);
      }
    };

  private:
    TensorMeshPtr _tensorMesh;
    std::map<std::string, VariableLayout> _trialLayouts;
    std::map<std::string, VariableLayout> _testLayouts;

    VariableLayout buildVariableLayout(const TensorVarDiscretizationPtr &discretization, TensorSpaceRole role) const;
  public:
    TensorDofLayout(TensorMeshPtr tensorMesh);

    TensorMeshPtr tensorMesh() const;

    bool hasLayout(const std::string &name, TensorSpaceRole role) const;
    const VariableLayout &layout(const std::string &name, TensorSpaceRole role) const;

    std::vector<VariableLayout> trialLayouts() const;
    std::vector<VariableLayout> testLayouts() const;

    IndexType localDofOrdinal(const std::string &name,
                              TensorSpaceRole role,
                              const std::vector<int> &componentBasisOrdinals) const;
  };
}

#endif
