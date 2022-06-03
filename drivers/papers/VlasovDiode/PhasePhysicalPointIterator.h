//
//  PhasePhysicalPointIterator.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/15/21.
//

#ifndef PhasePhysicalPointIterator_h
#define PhasePhysicalPointIterator_h

#include "PhysicalCellRelativeRefinementInfo.h"

namespace Camellia
{
  class PhasePhysicalPointIterator
  {
  private:
    const std::vector<PhysicalCellRelativeRefinementInfo> &_physicalCellInfo;
    const std::vector< std::pair<int,int> > _pointStartEndPairs; // in the physical basis cache, which points are of interest to the phase space element?
    const bool _reversePointOrder; // used for side ordinal 2 -- on each physical cell, go through the points in reverse order
    int _numPhysicalPoints;
    int _currentCellOrdinal; // index into _physicalCellInfo
    int _pointOrdinalInCurrentCell;
    int _pointOrdinal;
    bool _isValid = true;
  public:
    PhasePhysicalPointIterator(const std::vector<PhysicalCellRelativeRefinementInfo> &physicalCellInfo,
                               const std::vector< std::pair<int,int> > &pointStartEndPairs,
                               const bool reversePointOrder, int numPhysicalPoints)
    :
    _physicalCellInfo(physicalCellInfo),
    _pointStartEndPairs(pointStartEndPairs),
    _reversePointOrder(reversePointOrder),
    _numPhysicalPoints(numPhysicalPoints)
    {
      _currentCellOrdinal = _physicalCellInfo.size()-1; // we go in reverse order from that in _physicalCellInfo
      _pointOrdinalInCurrentCell = -1;
      _pointOrdinal = -1;
      increment();
    }
    
    void increment()
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!_isValid, std::invalid_argument, "Invalid iterator");
      int pointStart = _pointStartEndPairs[_currentCellOrdinal].first;
      int pointEnd   = _pointStartEndPairs[_currentCellOrdinal].second;
      int numPointsInCurrentCell = pointEnd - pointStart;
      if (_pointOrdinalInCurrentCell + 1 < numPointsInCurrentCell)
      {
        _pointOrdinalInCurrentCell++;
      }
      else
      {
        _currentCellOrdinal--; // we go in reverse order from _physicalCellInfo

        if (_currentCellOrdinal < 0)
        {
          _isValid = false;
          return;
        }
        
        // advance until we get to a cell that has some points, or until we get past the first cell
        pointStart = _pointStartEndPairs[_currentCellOrdinal].first;
        pointEnd   = _pointStartEndPairs[_currentCellOrdinal].second;
        numPointsInCurrentCell = pointEnd - pointStart;
        
        while ((_currentCellOrdinal > 0) && (numPointsInCurrentCell == 0))
        {
          _currentCellOrdinal--;
          
          pointStart = _pointStartEndPairs[_currentCellOrdinal].first;
          pointEnd   = _pointStartEndPairs[_currentCellOrdinal].second;
          numPointsInCurrentCell = pointEnd - pointStart;
        }
        
        if (numPointsInCurrentCell == 0) // then we've reached the first cell without finding any further points
        {
          _isValid = false;
        }
        else
        {
          _pointOrdinalInCurrentCell = 0;
        }
      }
      _pointOrdinal++;
    }
    
    bool isValid()
    {
      return _isValid;
    }
    
    int numPhysicalPoints()
    {
      return _numPhysicalPoints;
    }
    
    GlobalIndexType physicalCellID()
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!_isValid, std::invalid_argument, "Invalid iterator");
      return _physicalCellInfo[_currentCellOrdinal].physicalCellID;
    }
    
    int pointOrdinal()
    {
      const int pointOrdinal = _reversePointOrder ? _numPhysicalPoints - _pointOrdinal - 1 : _pointOrdinal;
      TEUCHOS_TEST_FOR_EXCEPTION(pointOrdinal < 0, std::invalid_argument, "pointOrdinal is out of bounds");
      return pointOrdinal;
    }
    
    int pointOrdinalInPhysicalCell()
    {
      return _pointStartEndPairs[_currentCellOrdinal].first + _pointOrdinalInCurrentCell;
    }
  };
}

#endif /* PhasePhysicalPointIterator_h */
