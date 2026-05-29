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
   *  \brief Explicit local coefficient ordering choices for tensor-native vectors.
   *
   *  These orderings are two views/flattenings of the same local coefficient space.
   *  TENSOR_VARIABLE_ORDERING follows the tensor-variable-local component-stride ordering.
   *  TENSOR_CELL_ORDERING follows a more traditional FE-style traversal order intended to expose
   *  cell-centric locality without changing the underlying local vector size.
   */
  enum TensorLocalOrdering
  {
    TENSOR_VARIABLE_ORDERING,
    TENSOR_CELL_ORDERING
  };

  /**
   *  \brief Per-component support classification for tensor-native local coefficient factors.
   *
   *  Volume variables use basis cardinality factors in every component. Trace variables use a
   *  distinguished skeleton-support factor in one component and basis-cardinality factors in the
   *  remaining components.
   */
  enum TensorComponentSupportKind
  {
    TENSOR_VOLUME_BASIS_SUPPORT,
    TENSOR_TRACE_SKELETON_SUPPORT
  };

  /**
   *  \brief Explicit tensor-native local layout abstraction.
   *
   *  TensorDofLayout provides local coefficient-layout metadata derived from TensorMesh.
   *  It is intentionally local in scope: it defines component cardinalities, component strides,
   *  and deterministic local ordinal maps for registered trial/test discretizations, but does
   *  not attempt to provide global numbering or assembled operator behavior.
   */
  class TensorDofLayout
  {
  public:
    /**
     *  \brief Concrete local-layout metadata for one registered tensor variable.
     *
     *  The current implementation stores one extent per component, one stride per component,
     *  per-component support-kind metadata, and the resulting tensor-product local dof count for
     *  a single variable-local block. Lower-indexed components vary fastest.
     */
    struct VariableLayout
    {
      std::string _name;
      TensorSpaceRole _role;
      std::vector<int> _componentCardinalities;
      std::vector<IndexType> _componentStrides;
      std::vector<TensorComponentSupportKind> _componentSupportKinds;
      int _traceComponentOrdinal = -1;
      IndexType _totalDofs = 0;

      VariableLayout() = default;
      VariableLayout(const std::string &name,
                     TensorSpaceRole role,
                     const std::vector<int> &componentCardinalities,
                     const std::vector<IndexType> &componentStrides,
                     const std::vector<TensorComponentSupportKind> &componentSupportKinds,
                     int traceComponentOrdinal,
                     IndexType totalDofs)
      : _name(name), _role(role), _componentCardinalities(componentCardinalities),
        _componentStrides(componentStrides), _componentSupportKinds(componentSupportKinds),
        _traceComponentOrdinal(traceComponentOrdinal), _totalDofs(totalDofs) {}

      const std::string &name() const { return _name; }
      TensorSpaceRole role() const { return _role; }
      int componentCount() const { return static_cast<int>(_componentCardinalities.size()); }
      int componentCardinality(int componentOrdinal) const { return _componentCardinalities[componentOrdinal]; }
      IndexType componentStride(int componentOrdinal) const { return _componentStrides[componentOrdinal]; }
      TensorComponentSupportKind componentSupportKind(int componentOrdinal) const { return _componentSupportKinds[componentOrdinal]; }
      bool isTraceSupportComponent(int componentOrdinal) const { return componentSupportKind(componentOrdinal) == TENSOR_TRACE_SKELETON_SUPPORT; }
      bool hasTraceComponentOrdinal() const { return _traceComponentOrdinal >= 0; }
      int traceComponentOrdinal() const { return _traceComponentOrdinal; }
      IndexType totalDofs() const { return _totalDofs; }

      bool operator==(const VariableLayout &other) const
      {
        return (_name == other._name)
            && (_role == other._role)
            && (_componentCardinalities == other._componentCardinalities)
            && (_componentStrides == other._componentStrides)
            && (_componentSupportKinds == other._componentSupportKinds)
            && (_traceComponentOrdinal == other._traceComponentOrdinal)
            && (_totalDofs == other._totalDofs);
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

    IndexType variableLocalDofOrdinal(const std::string &name,
                                      TensorSpaceRole role,
                                      const std::vector<int> &componentBasisOrdinals) const;

    IndexType cellLocalDofOrdinal(const std::string &name,
                                  TensorSpaceRole role,
                                  const std::vector<int> &componentBasisOrdinals) const;

    IndexType localDofCount(const std::string &name,
                            TensorSpaceRole role,
                            TensorLocalOrdering ordering) const;

    IndexType localDofOrdinal(const std::string &name,
                              TensorSpaceRole role,
                              const std::vector<int> &componentBasisOrdinals) const;

    IndexType localDofOrdinal(const std::string &name,
                              TensorSpaceRole role,
                              TensorLocalOrdering ordering,
                              const std::vector<int> &componentBasisOrdinals) const;

    void decodeVariableLocalDofOrdinal(const std::string &name,
                                       TensorSpaceRole role,
                                       IndexType ordinal,
                                       std::vector<int> &componentBasisOrdinals) const;

    void decodeCellLocalDofOrdinal(const std::string &name,
                                   TensorSpaceRole role,
                                   IndexType ordinal,
                                   std::vector<int> &componentBasisOrdinals) const;

    void decodeLocalDofOrdinal(const std::string &name,
                               TensorSpaceRole role,
                               TensorLocalOrdering ordering,
                               IndexType ordinal,
                               std::vector<int> &componentBasisOrdinals) const;
  };
}

#endif
