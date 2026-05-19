#ifndef Camellia_ComponentCellID1D_h
#define Camellia_ComponentCellID1D_h

#include <ostream>

#include "TypeDefs.h"

namespace Camellia
{
  class ComponentCellID1D
  {
    IndexType _rootOrdinal;
    unsigned _level;
    unsigned long long _address;
  public:
    ComponentCellID1D()
    : _rootOrdinal(-1), _level(0), _address(0) {}

    ComponentCellID1D(IndexType rootOrdinal, unsigned level, unsigned long long address)
    : _rootOrdinal(rootOrdinal), _level(level), _address(address) {}

    IndexType rootOrdinal() const { return _rootOrdinal; }
    unsigned level() const { return _level; }
    unsigned long long address() const { return _address; }

    bool isValid() const
    {
      if (_rootOrdinal < 0) return false;
      if (_level >= 8 * sizeof(unsigned long long)) return false;
      const unsigned long long maxAddress = 1ULL << _level;
      return _address < maxAddress;
    }

    ComponentCellID1D parent() const
    {
      if (_level == 0) return ComponentCellID1D();
      return ComponentCellID1D(_rootOrdinal, _level - 1, _address / 2);
    }
    ComponentCellID1D leftChild() const
    {
      return ComponentCellID1D(_rootOrdinal, _level + 1, 2 * _address);
    }
    ComponentCellID1D rightChild() const
    {
      return ComponentCellID1D(_rootOrdinal, _level + 1, 2 * _address + 1);
    }

    bool isAncestorOf(const ComponentCellID1D &other) const
    {
      if (!isValid() || !other.isValid()) return false;
      if (_rootOrdinal != other._rootOrdinal) return false;
      if (_level > other._level) return false;
      const unsigned levelDiff = other._level - _level;
      return _address == (other._address >> levelDiff);
    }

    bool operator==(const ComponentCellID1D &other) const
    {
      return (_rootOrdinal == other._rootOrdinal) && (_level == other._level) && (_address == other._address);
    }
    bool operator!=(const ComponentCellID1D &other) const
    {
      return !(*this == other);
    }
    bool operator<(const ComponentCellID1D &other) const
    {
      if (_rootOrdinal != other._rootOrdinal) return _rootOrdinal < other._rootOrdinal;
      if (_level != other._level) return _level < other._level;
      return _address < other._address;
    }
  };

  inline std::ostream &operator<<(std::ostream &os, const ComponentCellID1D &cellID)
  {
    os << "ComponentCellID1D(" << cellID.rootOrdinal() << "," << cellID.level() << "," << cellID.address() << ")";
    return os;
  }
}

#endif
