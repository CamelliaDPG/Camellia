#ifndef Camellia_ActiveLeafSet1D_h
#define Camellia_ActiveLeafSet1D_h

#include <algorithm>
#include <ostream>
#include <vector>

#include "ComponentCellID1D.h"

namespace Camellia
{
  class ActiveLeafSet1D
  {
  public:
    struct Entry
    {
      unsigned _level;
      unsigned long long _address;

      Entry()
      : _level(0), _address(0) {}

      Entry(unsigned level, unsigned long long address)
      : _level(level), _address(address) {}

      unsigned level() const { return _level; }
      unsigned long long address() const { return _address; }

      bool isValid() const
      {
        if (_level >= 8 * sizeof(unsigned long long)) return false;
        const unsigned long long maxAddress = 1ULL << _level;
        return _address < maxAddress;
      }
      bool operator==(const Entry &other) const
      {
        return (_level == other._level) && (_address == other._address);
      }
      bool operator<(const Entry &other) const
      {
        if (_level == other._level) return _address < other._address;
        const unsigned commonLevel = (_level > other._level) ? _level : other._level;
        const unsigned long long thisStart = _address << (commonLevel - _level);
        const unsigned long long otherStart = other._address << (commonLevel - other._level);
        if (thisStart != otherStart) return thisStart < otherStart;
        if (_level != other._level) return _level < other._level;
        return _address < other._address;
      }
    };

  private:
    std::vector<Entry> _entries;

  public:
    ActiveLeafSet1D() {}
    ActiveLeafSet1D(const std::vector<Entry> &entries)
    : _entries(entries)
    {
      std::sort(_entries.begin(), _entries.end());
    }

    const std::vector<Entry> &entries() const
    {
      return _entries;
    }

    bool isValid() const
    {
      return isExactCover();
    }
    bool isExactCover() const
    {
      if (_entries.empty()) return false;
      unsigned maxLevel = 0;
      for (const auto &entry : _entries)
      {
        if (!entry.isValid()) return false;
        if (entry.level() > maxLevel) maxLevel = entry.level();
      }

      unsigned long long expectedStart = 0;
      for (const auto &entry : _entries)
      {
        const unsigned shift = maxLevel - entry.level();
        const unsigned long long start = entry.address() << shift;
        const unsigned long long end = (entry.address() + 1ULL) << shift;
        if (start != expectedStart) return false;
        expectedStart = end;
      }
      return expectedStart == (1ULL << maxLevel);
    }

    std::vector<ComponentCellID1D> cellIDsForRoot(IndexType rootOrdinal) const
    {
      std::vector<ComponentCellID1D> cellIDs;
      cellIDs.reserve(_entries.size());
      for (const auto &entry : _entries)
      {
        cellIDs.push_back(ComponentCellID1D(rootOrdinal, entry.level(), entry.address()));
      }
      return cellIDs;
    }
  };

  inline std::ostream &operator<<(std::ostream &os, const ActiveLeafSet1D::Entry &entry)
  {
    os << "ActiveLeafSet1D::Entry(" << entry.level() << "," << entry.address() << ")";
    return os;
  }
}

#endif
