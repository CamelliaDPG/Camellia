#ifndef Camellia_TensorMeshTopology_h
#define Camellia_TensorMeshTopology_h

#include <map>
#include <vector>

#include "ComponentMeshTopology1D.h"

namespace Camellia
{
  class TensorMeshTopology
  {
  public:
    using TensorCellTuple = std::vector<ComponentCellID1D>;

    struct TensorSide
    {
      TensorCellTuple _cellTuple;
      int _componentOrdinal = -1;
      int _sideOrdinalInComponent = -1; // 0 = lower, 1 = upper for 1D components

      TensorSide() = default;
      TensorSide(const TensorCellTuple &cellTuple, int componentOrdinal, int sideOrdinalInComponent)
      : _cellTuple(cellTuple), _componentOrdinal(componentOrdinal), _sideOrdinalInComponent(sideOrdinalInComponent) {}

      const TensorCellTuple &cellTuple() const { return _cellTuple; }
      int componentOrdinal() const { return _componentOrdinal; }
      int sideOrdinalInComponent() const { return _sideOrdinalInComponent; }

      bool operator==(const TensorSide &other) const
      {
        return (_cellTuple == other._cellTuple) && (_componentOrdinal == other._componentOrdinal) && (_sideOrdinalInComponent == other._sideOrdinalInComponent);
      }
    };

    struct TensorSkeletonEntity
    {
      int _orthogonalComponentOrdinal = -1;
      IndexType _vertexOrdinal = -1;
      TensorCellTuple _retainedCellTuple;

      TensorSkeletonEntity() = default;
      TensorSkeletonEntity(int orthogonalComponentOrdinal,
                           IndexType vertexOrdinal,
                           const TensorCellTuple &retainedCellTuple)
      : _orthogonalComponentOrdinal(orthogonalComponentOrdinal),
        _vertexOrdinal(vertexOrdinal),
        _retainedCellTuple(retainedCellTuple) {}

      int orthogonalComponentOrdinal() const { return _orthogonalComponentOrdinal; }
      IndexType vertexOrdinal() const { return _vertexOrdinal; }
      const TensorCellTuple &retainedCellTuple() const { return _retainedCellTuple; }

      bool operator==(const TensorSkeletonEntity &other) const
      {
        return (_orthogonalComponentOrdinal == other._orthogonalComponentOrdinal)
            && (_vertexOrdinal == other._vertexOrdinal)
            && (_retainedCellTuple == other._retainedCellTuple);
      }
    };

    class TensorSkeleton
    {
      int _orthogonalComponentOrdinal = -1;
      std::vector<int> _retainedComponentOrdinals;
      std::vector<IndexType> _vertexOrdinals;
      std::vector<TensorSkeletonEntity> _entities;

    public:
      TensorSkeleton() = default;
      TensorSkeleton(int orthogonalComponentOrdinal,
                     const std::vector<int> &retainedComponentOrdinals,
                     const std::vector<IndexType> &vertexOrdinals,
                     const std::vector<TensorSkeletonEntity> &entities)
      : _orthogonalComponentOrdinal(orthogonalComponentOrdinal),
        _retainedComponentOrdinals(retainedComponentOrdinals),
        _vertexOrdinals(vertexOrdinals),
        _entities(entities) {}

      int orthogonalComponentOrdinal() const { return _orthogonalComponentOrdinal; }
      int retainedComponentCount() const { return static_cast<int>(_retainedComponentOrdinals.size()); }
      const std::vector<int> &retainedComponentOrdinals() const { return _retainedComponentOrdinals; }
      const std::vector<IndexType> &vertexOrdinals() const { return _vertexOrdinals; }
      IndexType entityCount() const { return static_cast<IndexType>(_entities.size()); }
      const std::vector<TensorSkeletonEntity> &entities() const { return _entities; }
      const TensorSkeletonEntity &entity(IndexType ordinal) const { return _entities[ordinal]; }
    };

  private:
    std::vector<Teuchos::RCP<ComponentMeshTopology1D>> _components;
    std::vector<TensorCellTuple> _activeCellTuples;
    std::vector<int> _originalComponentOrdinals;

    void validateConstructorInputs() const;
    void rebuildActiveCellTuples();

  public:
    TensorMeshTopology(const std::vector<Teuchos::RCP<ComponentMeshTopology1D>> &components);
    TensorMeshTopology(const std::vector<Teuchos::RCP<ComponentMeshTopology1D>> &components,
                       const std::vector<TensorCellTuple> &activeCellTuples,
                       const std::vector<int> &originalComponentOrdinals);

    int componentCount() const;
    IndexType activeCellCount() const;

    Teuchos::RCP<ComponentMeshTopology1D> componentMeshTopology(int componentOrdinal) const;
    int originalComponentOrdinal(int componentOrdinal) const;

    const std::vector<TensorCellTuple> &activeCellTuples() const;

    bool containsTensorCell(const TensorCellTuple &cellTuple) const;
    IndexType tensorCellOrdinal(const TensorCellTuple &cellTuple) const;
    TensorCellTuple tensorCellTuple(IndexType ordinal) const;

    std::vector<TensorSide> sidesForTensorCell(const TensorCellTuple &cellTuple) const;
    bool isBoundarySide(const TensorSide &side) const;
    bool sideIsValid(const TensorSide &side) const;

    TensorMeshTopology slice(const std::map<int, IndexType> &fixedComponentVertices) const;

    std::vector<int> retainedComponentOrdinalsForSkeleton(int orthogonalComponentOrdinal) const;
    TensorSkeleton tensorSkeleton(int orthogonalComponentOrdinal) const;
    bool isBoundarySkeletonEntity(const TensorSkeletonEntity &entity) const;
  };

  inline std::ostream &operator<<(std::ostream &os, const TensorMeshTopology::TensorSide &side)
  {
    os << "TensorSide(component=" << side.componentOrdinal() << ",side=" << side.sideOrdinalInComponent() << ",tuple=[";
    const auto &tuple = side.cellTuple();
    for (size_t i=0; i<tuple.size(); i++)
    {
      if (i > 0) os << ",";
      os << tuple[i];
    }
    os << "])";
    return os;
  }

  inline std::ostream &operator<<(std::ostream &os, const TensorMeshTopology::TensorSkeletonEntity &entity)
  {
    os << "TensorSkeletonEntity(component=" << entity.orthogonalComponentOrdinal()
       << ",vertex=" << entity.vertexOrdinal() << ",retainedTuple=[";
    const auto &tuple = entity.retainedCellTuple();
    for (size_t i=0; i<tuple.size(); i++)
    {
      if (i > 0) os << ",";
      os << tuple[i];
    }
    os << "])";
    return os;
  }
}

#endif
