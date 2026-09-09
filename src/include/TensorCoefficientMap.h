#ifndef Camellia_TensorCoefficientMap_h
#define Camellia_TensorCoefficientMap_h

#include <map>
#include <string>
#include <vector>

#include "TensorCoefficientTraversal.h"
#include "TensorDofLayout.h"

namespace Camellia
{
  /**
   *  \brief Explicit tensor-native coefficient indexing support for realized entities.
   *
   *  The current design is intentionally component-factorized rather than tensor-cell-expanded.
   *  For volume variables, global coefficient indices are derived from per-component active-cell /
   *  local-basis coordinates.  For trace variables, the distinguished trace component contributes
   *  a support-vertex coordinate, while retained components contribute per-cell / local-basis
   *  coordinates.  Multiple slice-based tensor skeleton entities therefore share trace coefficients
   *  automatically whenever they correspond to the same underlying trace-support vertex together
   *  with the same retained-component basis tuple.
   */
  class TensorCoefficientMap
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorSpaceRole _role;
    std::string _varName;
    TensorLocalOrdering _ordering;
    TensorDofLayout::VariableLayout _variableLayout;
    IndexType _coefficientCount;

    struct ComponentFactorData
    {
      std::vector<ComponentCellID1D> _activeCells;
      std::vector<int> _cellBasisCardinalities;
      std::vector<IndexType> _cellBasisOffsets;
      IndexType _factorSize = 0;
    };

    std::vector<ComponentFactorData> _componentFactors;
    std::vector<IndexType> _componentStrides;

    struct TraceSupportFactorData
    {
      std::vector<IndexType> _vertexOrdinals;
      std::vector<IndexType> _vertexComponentOrdinals;
      std::map<IndexType, IndexType> _vertexOrdinalsToFactorOrdinals;
      IndexType _factorSize = 0;
    };

    bool _isTraceVariable = false;
    int _traceComponentOrdinal = -1;
    TraceSupportFactorData _traceSupportFactor;

    void validateConstructorInputs() const;
    void buildComponentFactors();
    void buildVolumeMetadata();
    void buildTraceMetadata();

    bool variableIsTraceLike() const;
    int traceComponentOrdinalForVariable() const;
    std::string compatibleOppositeRoleVarName(int traceComponentOrdinal) const;
    std::string compatibleSkeletonTrialVarName(int traceComponentOrdinal) const;
    std::string compatibleSkeletonTestVarName(int traceComponentOrdinal) const;

    ComponentFactorData buildComponentFactorData(int componentOrdinal) const;
    TraceSupportFactorData buildTraceSupportFactorData(int traceComponentOrdinal) const;
    std::vector<ComponentCellID1D> componentCellsForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const;
    ComponentCellID1D incidentOrthogonalCellForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                              const TensorFactorSpaceShape &shape) const;

    IndexType componentCellBasisOffset(int componentOrdinal,
                                       const ComponentCellID1D &cellID) const;
    IndexType componentCoordinate(int componentOrdinal,
                                  const ComponentCellID1D &cellID,
                                  int basisOrdinal) const;
    IndexType traceSupportCoordinate(IndexType vertexOrdinal) const;
    IndexType flattenCoordinates(const std::vector<IndexType> &componentCoordinates) const;

  public:
    TensorCoefficientMap(TensorMeshPtr tensorMesh,
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

    IndexType coefficientCount() const;

    TensorCoefficientTraversal volumeTraversal(const TensorMeshTopology::TensorCellTuple &cellTuple) const;
    TensorCoefficientTraversal skeletonTraversal(const TensorMeshTopology::TensorSkeletonEntity &entity) const;

    IndexType globalIndexForVolumeLocalOrdinal(const TensorMeshTopology::TensorCellTuple &cellTuple,
                                               IndexType localOrdinal) const;
    IndexType globalIndexForSkeletonLocalOrdinal(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                 IndexType localOrdinal) const;

    std::vector<IndexType> indicesForVolumeCell(const TensorMeshTopology::TensorCellTuple &cellTuple) const;
    std::vector<IndexType> indicesForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const;

    std::vector<IndexType> ownedIndicesForRank(int rankOrdinal, int rankCount) const;
    std::pair<IndexType,IndexType> ownedRangeForRank(int rankOrdinal, int rankCount) const;

    std::vector<IndexType> neededIndicesForVolumeCell(const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                      int rankOrdinal,
                                                      int rankCount) const;
    std::vector<IndexType> neededIndicesForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                          int rankOrdinal,
                                                          int rankCount) const;
    std::vector<IndexType> nonownedNeededIndices(const TensorCoefficientTraversal &traversal,
                                                 int rankOrdinal,
                                                 int rankCount) const;
  };
}

#endif
