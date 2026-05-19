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
}

#endif
