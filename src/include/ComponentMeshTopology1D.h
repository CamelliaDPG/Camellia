#ifndef Camellia_ComponentMeshTopology1D_h
#define Camellia_ComponentMeshTopology1D_h

#include <map>
#include <vector>
#include <Kokkos_DualView.hpp>

#include "ActiveLeafSet1D.h"
#include "ComponentCellID1D.h"
#include "TypeDefs.h"

namespace Camellia
{
  class ComponentMeshTopology1D
  {
  public:
    using VertexCoordinateDualView = Kokkos::DualView<double*>;
    using CellVerticesDualView = Kokkos::DualView<IndexType**>;
    using VertexToCellOffsetsDualView = Kokkos::DualView<IndexType*>;
    using VertexToCellOrdinalsDualView = Kokkos::DualView<IndexType*>;

  private:
    std::vector<double> _coarseVertices;
    std::vector<ActiveLeafSet1D> _activeLeafSets;
    std::vector<ComponentCellID1D> _activeCells;
    bool _isPeriodic;
    int _defaultPolyOrder;
    std::map<ComponentCellID1D,int> _cellPolyOrders;
    std::map<ComponentCellID1D,IndexType> _activeCellOrdinals;

    VertexCoordinateDualView _activeVertexCoordinates;
    CellVerticesDualView _cellVertices;
    VertexToCellOffsetsDualView _vertexToCellOffsets;
    VertexToCellOrdinalsDualView _vertexToCellOrdinals;
    std::vector<ComponentCellID1D> _vertexToActiveCellsHost;

    void validateConstructorInputs() const;
    void rebuildActiveCells();
    void initializePolyOrders();
    void rebuildVertexIncidence();

  public:
    ComponentMeshTopology1D(const std::vector<double> &coarseVertices,
                            const std::vector<ActiveLeafSet1D> &activeLeafSets,
                            bool isPeriodic = false,
                            int defaultPolyOrder = 1);

    IndexType rootCellCount() const;
    IndexType activeCellCount() const;
    IndexType vertexCount() const;

    bool isPeriodic() const;
    int defaultPolyOrder() const;

    double rootCellLeft(IndexType rootOrdinal) const;
    double rootCellRight(IndexType rootOrdinal) const;

    const ActiveLeafSet1D &activeLeafSet(IndexType rootOrdinal) const;

    const std::vector<ComponentCellID1D> &activeCells() const;
    std::vector<ComponentCellID1D> activeCellsForRoot(IndexType rootOrdinal) const;
    const std::vector<ComponentCellID1D> &activeCellsForVertex(IndexType vertexIndex) const;

    bool containsCell(const ComponentCellID1D &cellID) const;

    int cellPolyOrder(const ComponentCellID1D &cellID) const;
    void setCellPolyOrder(const ComponentCellID1D &cellID, int polyOrder);

    void refineCell(const ComponentCellID1D &cellID);

    IndexType cellVertexIndex(const ComponentCellID1D &cellID, int vertexOrdinal) const;

    const VertexCoordinateDualView &activeVertexCoordinates() const;
    const CellVerticesDualView &cellVertices() const;
    const VertexToCellOffsetsDualView &vertexToCellOffsets() const;
    const VertexToCellOrdinalsDualView &vertexToCellOrdinals() const;

    IndexType vertexToCellOffsetHost(IndexType vertexIndex) const;
    IndexType vertexToCellOrdinalHost(IndexType entryOrdinal) const;

    double cellLeft(const ComponentCellID1D &cellID) const;
    double cellRight(const ComponentCellID1D &cellID) const;
    double cellWidth(const ComponentCellID1D &cellID) const;
  };
}

#endif
