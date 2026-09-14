#ifndef Camellia_TensorOperatorData_h
#define Camellia_TensorOperatorData_h

#include <string>
#include <vector>

#include <Kokkos_DualView.hpp>

#include "TensorCoefficientMap.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "TensorMeshTopology.h"

namespace Camellia
{
  class TensorOperatorData;
  using TensorOperatorDataPtr = Teuchos::RCP<TensorOperatorData>;

  /**
   * \brief Setup-compiled, kernel-facing metadata for tensor-native operators.
   *
   * TensorOperatorData is the first Phase 1 bridge from the descriptive TensorMesh and
   * TensorDofLayout objects to flat integer/real arrays that can be consumed by later
   * sum-factorized assembly and matrix-free kernels.  It intentionally does no numerical
   * assembly; it records stable ordinals, geometry scalings, boundary-side flags, and
   * variable coefficient-space sizes in Kokkos DualViews or compact setup-only records.
   */
  class TensorOperatorData
  {
  public:
    using ComponentCellVerticesDualView = Kokkos::DualView<IndexType**>;
    using ComponentCellMetricDualView  = Kokkos::DualView<double*>;
    using ComponentCellPolyOrderDualView = Kokkos::DualView<int*>;
    using ActiveTensorCellComponentOrdinalsDualView = Kokkos::DualView<IndexType**>;
    using BoundarySideFlagDualView = Kokkos::DualView<int**>;
    using LocalDofCountDualView = Kokkos::DualView<IndexType*>;
    using LocalToGlobalIndexDualView = Kokkos::DualView<IndexType**>;
    using SkeletonEntityOrdinalDualView = Kokkos::DualView<IndexType*>;
    using SkeletonEntityParityDualView = Kokkos::DualView<int*>;

    struct ComponentData
    {
      IndexType _activeCellCount = 0;
      IndexType _vertexCount = 0;
      ComponentMeshTopology1D::VertexCoordinateDualView _vertexCoordinates;
      ComponentMeshTopology1D::CellVerticesDualView _cellVertices;
      ComponentMeshTopology1D::VertexToCellOffsetsDualView _vertexToCellOffsets;
      ComponentMeshTopology1D::VertexToCellOrdinalsDualView _vertexToCellOrdinals;
      ComponentCellMetricDualView _cellWidths;
      ComponentCellMetricDualView _cellJacobians;
      ComponentCellPolyOrderDualView _cellPolyOrders;

      IndexType activeCellCount() const { return _activeCellCount; }
      IndexType vertexCount() const { return _vertexCount; }
      const ComponentMeshTopology1D::VertexCoordinateDualView &vertexCoordinates() const { return _vertexCoordinates; }
      const ComponentMeshTopology1D::CellVerticesDualView &cellVertices() const { return _cellVertices; }
      const ComponentMeshTopology1D::VertexToCellOffsetsDualView &vertexToCellOffsets() const { return _vertexToCellOffsets; }
      const ComponentMeshTopology1D::VertexToCellOrdinalsDualView &vertexToCellOrdinals() const { return _vertexToCellOrdinals; }
      const ComponentCellMetricDualView &cellWidths() const { return _cellWidths; }
      const ComponentCellMetricDualView &cellJacobians() const { return _cellJacobians; }
      const ComponentCellPolyOrderDualView &cellPolyOrders() const { return _cellPolyOrders; }
    };

    struct VariableData
    {
      std::string _name;
      TensorSpaceRole _role = TENSOR_TRIAL_SPACE;
      bool _isTraceVariable = false;
      int _traceComponentOrdinal = -1;
      IndexType _coefficientCount = 0;
      TensorFactorSpaceShape _ambientShape;
      IndexType _maxVolumeLocalDofCount = 0;
      LocalDofCountDualView _volumeLocalDofCounts;
      LocalToGlobalIndexDualView _volumeLocalToGlobalIndices;
      IndexType _skeletonEntityCount = 0;
      IndexType _maxSkeletonLocalDofCount = 0;
      LocalDofCountDualView _skeletonLocalDofCounts;
      LocalToGlobalIndexDualView _skeletonLocalToGlobalIndices;
      SkeletonEntityOrdinalDualView _skeletonEntityVertexOrdinals;
      // For trace/skeleton variables, side parity is the 1D outward-normal sign in the
      // orthogonal component: -1 on the lower/left side and +1 on the upper/right side.
      SkeletonEntityParityDualView _skeletonEntitySideParities;

      const std::string &name() const { return _name; }
      TensorSpaceRole role() const { return _role; }
      bool isTraceVariable() const { return _isTraceVariable; }
      int traceComponentOrdinal() const { return _traceComponentOrdinal; }
      IndexType coefficientCount() const { return _coefficientCount; }
      const TensorFactorSpaceShape &ambientShape() const { return _ambientShape; }
      IndexType maxVolumeLocalDofCount() const { return _maxVolumeLocalDofCount; }
      const LocalDofCountDualView &volumeLocalDofCounts() const { return _volumeLocalDofCounts; }
      const LocalToGlobalIndexDualView &volumeLocalToGlobalIndices() const { return _volumeLocalToGlobalIndices; }
      IndexType skeletonEntityCount() const { return _skeletonEntityCount; }
      IndexType maxSkeletonLocalDofCount() const { return _maxSkeletonLocalDofCount; }
      const LocalDofCountDualView &skeletonLocalDofCounts() const { return _skeletonLocalDofCounts; }
      const LocalToGlobalIndexDualView &skeletonLocalToGlobalIndices() const { return _skeletonLocalToGlobalIndices; }
      const SkeletonEntityOrdinalDualView &skeletonEntityVertexOrdinals() const { return _skeletonEntityVertexOrdinals; }
      const SkeletonEntityParityDualView &skeletonEntitySideParities() const { return _skeletonEntitySideParities; }
      const SkeletonEntityParityDualView &skeletonEntityOutwardNormalSigns() const { return _skeletonEntitySideParities; }
    };

  private:
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    std::vector<ComponentData> _components;
    ActiveTensorCellComponentOrdinalsDualView _activeTensorCellComponentOrdinals;
    BoundarySideFlagDualView _boundarySideFlags;
    std::vector<VariableData> _variables;

    void validateConstructorInputs() const;
    void buildComponentData();
    void buildActiveTensorCellData();
    void buildBoundarySideData();
    void buildVariableData();
    void buildVolumeLocalToGlobalData(VariableData &data,
                                      const TensorCoefficientMap &coefficientMap) const;
    void buildSkeletonLocalToGlobalData(VariableData &data,
                                        const TensorCoefficientMap &coefficientMap) const;
    IndexType variableOrdinal(TensorSpaceRole role,
                              const std::string &varName) const;

  public:
    TensorOperatorData(TensorMeshPtr tensorMesh,
                       TensorDofLayoutPtr tensorDofLayout);

    TensorMeshPtr tensorMesh() const;
    TensorDofLayoutPtr tensorDofLayout() const;

    int componentCount() const;
    IndexType activeCellCount() const;
    IndexType tensorSideCount() const;

    const ComponentData &componentData(int componentOrdinal) const;
    const ActiveTensorCellComponentOrdinalsDualView &activeTensorCellComponentOrdinals() const;
    IndexType activeTensorCellComponentOrdinal(IndexType tensorCellOrdinal,
                                               int componentOrdinal) const;
    const BoundarySideFlagDualView &boundarySideFlags() const;
    bool isBoundarySide(IndexType tensorCellOrdinal,
                        int tensorSideOrdinal) const;

    IndexType variableCount() const;
    const VariableData &variableData(IndexType variableOrdinal) const;
    const VariableData &variableData(TensorSpaceRole role,
                                     const std::string &varName) const;
    IndexType coefficientCount(TensorSpaceRole role,
                               const std::string &varName) const;
    IndexType volumeLocalDofCount(TensorSpaceRole role,
                                  const std::string &varName,
                                  IndexType tensorCellOrdinal) const;
    IndexType volumeLocalToGlobalIndex(TensorSpaceRole role,
                                       const std::string &varName,
                                       IndexType tensorCellOrdinal,
                                       IndexType localOrdinal) const;
    IndexType skeletonEntityCount(TensorSpaceRole role,
                                  const std::string &varName) const;
    IndexType skeletonLocalDofCount(TensorSpaceRole role,
                                    const std::string &varName,
                                    IndexType skeletonEntityOrdinal) const;
    IndexType skeletonLocalToGlobalIndex(TensorSpaceRole role,
                                         const std::string &varName,
                                         IndexType skeletonEntityOrdinal,
                                         IndexType localOrdinal) const;
    int skeletonEntitySideParity(TensorSpaceRole role,
                                 const std::string &varName,
                                 IndexType skeletonEntityOrdinal) const;
    int skeletonEntityOutwardNormalSign(TensorSpaceRole role,
                                        const std::string &varName,
                                        IndexType skeletonEntityOrdinal) const;
  };
}

#endif
