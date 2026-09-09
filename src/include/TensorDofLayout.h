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

  enum TensorSpaceRole
  {
    TENSOR_TRIAL_SPACE,
    TENSOR_TEST_SPACE
  };

  enum TensorLocalOrdering
  {
    TENSOR_VARIABLE_ORDERING,
    TENSOR_CELL_ORDERING
  };

  enum TensorComponentSupportKind
  {
    TENSOR_VOLUME_BASIS_SUPPORT,
    TENSOR_TRACE_SKELETON_SUPPORT
  };

  struct TensorFactorSpaceShape
  {
    std::vector<int> _componentCardinalities;
    std::vector<IndexType> _componentStrides;
    IndexType _totalDofs = 0;

    TensorFactorSpaceShape() = default;
    TensorFactorSpaceShape(const std::vector<int> &componentCardinalities,
                           const std::vector<IndexType> &componentStrides,
                           IndexType totalDofs)
    : _componentCardinalities(componentCardinalities),
      _componentStrides(componentStrides),
      _totalDofs(totalDofs) {}

    int componentCount() const { return static_cast<int>(_componentCardinalities.size()); }
    int componentCardinality(int componentOrdinal) const { return _componentCardinalities[componentOrdinal]; }
    IndexType componentStride(int componentOrdinal) const { return _componentStrides[componentOrdinal]; }
    IndexType totalDofs() const { return _totalDofs; }

    bool operator==(const TensorFactorSpaceShape &other) const
    {
      return (_componentCardinalities == other._componentCardinalities)
          && (_componentStrides == other._componentStrides)
          && (_totalDofs == other._totalDofs);
    }
  };

  struct TensorLocalVolumeContext
  {
    TensorMeshTopology::TensorCellTuple _cellTuple;
    TensorFactorSpaceShape _trialShape;
    TensorFactorSpaceShape _testShape;

    TensorLocalVolumeContext() = default;
    TensorLocalVolumeContext(const TensorMeshTopology::TensorCellTuple &cellTuple,
                             const TensorFactorSpaceShape &trialShape,
                             const TensorFactorSpaceShape &testShape)
    : _cellTuple(cellTuple), _trialShape(trialShape), _testShape(testShape) {}

    const TensorMeshTopology::TensorCellTuple &cellTuple() const { return _cellTuple; }
    const TensorFactorSpaceShape &trialShape() const { return _trialShape; }
    const TensorFactorSpaceShape &testShape() const { return _testShape; }
  };

  struct TensorLocalSkeletonContext
  {
    TensorMeshTopology::TensorSkeletonEntity _entity;
    TensorFactorSpaceShape _trialShape;
    TensorFactorSpaceShape _testShape;

    TensorLocalSkeletonContext() = default;
    TensorLocalSkeletonContext(const TensorMeshTopology::TensorSkeletonEntity &entity,
                               const TensorFactorSpaceShape &trialShape,
                               const TensorFactorSpaceShape &testShape)
    : _entity(entity), _trialShape(trialShape), _testShape(testShape) {}

    const TensorMeshTopology::TensorSkeletonEntity &entity() const { return _entity; }
    const TensorFactorSpaceShape &trialShape() const { return _trialShape; }
    const TensorFactorSpaceShape &testShape() const { return _testShape; }
  };

  class TensorDofLayout
  {
  public:
    struct VariableLayout
    {
      std::string _name;
      TensorSpaceRole _role;
      int _componentCount = 0;
      std::vector<TensorComponentSupportKind> _componentSupportKinds;
      int _traceComponentOrdinal = -1;

      VariableLayout() = default;
      VariableLayout(const std::string &name,
                     TensorSpaceRole role,
                     int componentCount,
                     const std::vector<TensorComponentSupportKind> &componentSupportKinds,
                     int traceComponentOrdinal)
      : _name(name),
        _role(role),
        _componentCount(componentCount),
        _componentSupportKinds(componentSupportKinds),
        _traceComponentOrdinal(traceComponentOrdinal) {}

      const std::string &name() const { return _name; }
      TensorSpaceRole role() const { return _role; }
      int componentCount() const { return _componentCount; }
      TensorComponentSupportKind componentSupportKind(int componentOrdinal) const { return _componentSupportKinds[componentOrdinal]; }
      bool isTraceSupportComponent(int componentOrdinal) const { return componentSupportKind(componentOrdinal) == TENSOR_TRACE_SKELETON_SUPPORT; }
      bool hasTraceComponentOrdinal() const { return _traceComponentOrdinal >= 0; }
      int traceComponentOrdinal() const { return _traceComponentOrdinal; }

      bool operator==(const VariableLayout &other) const
      {
        return (_name == other._name)
            && (_role == other._role)
            && (_componentCount == other._componentCount)
            && (_componentSupportKinds == other._componentSupportKinds)
            && (_traceComponentOrdinal == other._traceComponentOrdinal);
      }
    };

  private:
    TensorMeshPtr _tensorMesh;
    std::map<std::string, VariableLayout> _trialLayouts;
    std::map<std::string, VariableLayout> _testLayouts;

    VariableLayout buildVariableLayout(const TensorVarDiscretizationPtr &discretization, TensorSpaceRole role) const;
    TensorFactorSpaceShape buildFactorSpaceShape(const TensorVarDiscretizationPtr &discretization,
                                                 TensorSpaceRole role,
                                                 const std::vector<ComponentCellID1D> &componentCells,
                                                 int orthogonalTraceComponentOrdinal,
                                                 const std::vector<int> &traceSupportComponentOrdinals) const;
    TensorFactorSpaceShape buildAmbientShape(const std::string &name,
                                             TensorSpaceRole role) const;
  public:
    TensorDofLayout(TensorMeshPtr tensorMesh);

    TensorMeshPtr tensorMesh() const;

    bool hasLayout(const std::string &name, TensorSpaceRole role) const;
    const VariableLayout &layout(const std::string &name, TensorSpaceRole role) const;

    std::vector<VariableLayout> trialLayouts() const;
    std::vector<VariableLayout> testLayouts() const;

    TensorFactorSpaceShape factorSpaceShape(const std::string &name,
                                            TensorSpaceRole role,
                                            const std::vector<ComponentCellID1D> &componentCells,
                                            int orthogonalTraceComponentOrdinal = -1,
                                            const std::vector<int> &traceSupportComponentOrdinals = {}) const;

    TensorFactorSpaceShape ambientShape(const std::string &name,
                                        TensorSpaceRole role) const;

    TensorLocalVolumeContext volumeContext(const std::string &trialVarName,
                                           const std::string &testVarName,
                                           const TensorMeshTopology::TensorCellTuple &cellTuple) const;

    TensorLocalSkeletonContext skeletonContext(const std::string &trialVarName,
                                               const std::string &testVarName,
                                               const TensorMeshTopology::TensorSkeletonEntity &entity) const;

    IndexType variableLocalDofOrdinal(const VariableLayout &layout,
                                      const TensorFactorSpaceShape &shape,
                                      const std::vector<int> &componentBasisOrdinals) const;

    IndexType cellLocalDofOrdinal(const VariableLayout &layout,
                                  const TensorFactorSpaceShape &shape,
                                  const std::vector<int> &componentBasisOrdinals) const;

    IndexType localDofCount(const TensorFactorSpaceShape &shape,
                            TensorLocalOrdering ordering) const;

    IndexType localDofOrdinal(const VariableLayout &layout,
                              const TensorFactorSpaceShape &shape,
                              const std::vector<int> &componentBasisOrdinals) const;

    IndexType localDofOrdinal(const VariableLayout &layout,
                              const TensorFactorSpaceShape &shape,
                              TensorLocalOrdering ordering,
                              const std::vector<int> &componentBasisOrdinals) const;

    void decodeVariableLocalDofOrdinal(const VariableLayout &layout,
                                       const TensorFactorSpaceShape &shape,
                                       IndexType ordinal,
                                       std::vector<int> &componentBasisOrdinals) const;

    void decodeCellLocalDofOrdinal(const VariableLayout &layout,
                                   const TensorFactorSpaceShape &shape,
                                   IndexType ordinal,
                                   std::vector<int> &componentBasisOrdinals) const;

    void decodeLocalDofOrdinal(const VariableLayout &layout,
                               const TensorFactorSpaceShape &shape,
                               TensorLocalOrdering ordering,
                               IndexType ordinal,
                               std::vector<int> &componentBasisOrdinals) const;
  };
}

#endif
