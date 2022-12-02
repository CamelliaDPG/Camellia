// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  RangeList.h
//  Camellia
//
//  Created by Nate Roberts on 4/7/16.
//
//

#ifndef Camellia_RangeList_h
#define Camellia_RangeList_h

#include <vector>

namespace Camellia
{
  template<class IntegerType> class RangeListIterator;
  
  template<class IntegerType> class RangeList
  {
    // _rangeBegins and _rangeEnds have the same length
    // _rangeBegins stores the beginning of each range
    // _rangeEnds stores the final value in the range
    // _rangeEnds[i] - _rangeBegins[i] + 1 gives the # of elements in the ith range
    std::vector<IntegerType> _rangeBegins; // in order container
    std::vector<IntegerType> _rangeEnds;
    int _size;
    
    // ! returns the ordinal of the last range whose first element is less than or equal to value
    int rangeOrdinalLowerBound(IntegerType value) const;
    void mergeIfPossible(int entryOrdinal);
  public:
    // ! Constructor
    RangeList();
    
    // ! returns the nth value in the range list
    IntegerType getValue(int n) const;
    
    // ! Inserts the element
    void insert(IntegerType value);
    
    // ! Inserts count contiguous elements starting with firstValue
    void insertRange(IntegerType firstValue, int count);
    
    // ! Removes the element, if it is present in some range in the list
    void remove(IntegerType value);
    
    // ! Returns true if the value is contained in the range
    bool contains(IntegerType value) const;
    
    // ! number of ranges in the range list
    int length() const;
    
    // ! number of elements in the range list
    int size() const;
    
    RangeListIterator<IntegerType> begin();
    RangeListIterator<IntegerType> end();
    
    friend class RangeListIterator<IntegerType>;
  };
  
  
  template<class IntegerType>
  class RangeListIterator
  : public std::iterator<std::forward_iterator_tag, IntegerType>
  {
  private:
    RangeList<IntegerType>* _rangeList;
    int _currentRangeOrdinal;
    IntegerType _currentValue;
    
  public:
    RangeListIterator(RangeList<IntegerType>* rangeList, bool initializeToEnd)
    {
      _rangeList = rangeList;
      if (!initializeToEnd && (rangeList->size() != 0)) // for empty list: begin() is the same as end()
      {
        _currentRangeOrdinal = 0;
        _currentValue = _rangeList->_rangeBegins[_currentRangeOrdinal];
      }
      else
      {
        _currentRangeOrdinal = _rangeList->length();
        _currentValue = -1;
      }
    }
    
    const IntegerType& operator*() const
    {
      return _currentValue;
    }
    
    const IntegerType* operator->() const
    {
      return &_currentValue;
    }
    
    RangeListIterator& operator++() // prefix
    {
      if (_currentValue + 1 <= _rangeList->_rangeEnds[_currentRangeOrdinal])
      {
        _currentValue++;
      }
      else
      {
        _currentRangeOrdinal++;
        const int rangeBeginsSize = _rangeList->_rangeBegins.size();
        if (_currentRangeOrdinal < rangeBeginsSize)
        {
          _currentValue = _rangeList->_rangeBegins[_currentRangeOrdinal];
        }
        else
        {
          _currentValue = -1;
        }
      }
      return *this;
    }
    
    RangeListIterator operator++(int) // postfix
    {
      RangeListIterator result(*this);
      ++(*this);
      return result;
    }
    
    friend bool operator==(RangeListIterator a, RangeListIterator b)
    {
      return (a._rangeList == b._rangeList) && (a._currentValue == b._currentValue) && (a._currentRangeOrdinal == b._currentRangeOrdinal);
    }
    
    friend bool operator!=(RangeListIterator a, RangeListIterator b)
    {
      return !(a == b);
    }
  };
  
}

#include "RangeListDef.h"

#endif
