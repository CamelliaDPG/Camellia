//
//  PhasePhysicalMeshMap1D1V.h
//  Camellia
//
//  Created by Roberts, Nathan V on 5/13/20.
//

#ifndef PhasePhysicalMeshMap1D1V_h
#define PhasePhysicalMeshMap1D1V_h

#include "MPIWrapper.h"
#include "RefinementObserver.h"
#include "TypeDefs.h"

namespace Camellia
{
  struct PhysicalCellRelativeRefinementInfo
  {
    GlobalIndexType physicalCellID;
    int unrefinementLevel; // relative to the ancestor that is as wide as the phase space cell
    int uniformDescendantNumber;  // in 1D, these move left to right
    double refSpaceXCoordinateInAncestor() const // left vertex coordinate; ranges from -1 to 1
    {
      int numDescendants = 1 << unrefinementLevel;
      return uniformDescendantNumber * 2.0 / numDescendants - 1.0;
    }
  };

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

  //! This class requires that the phase space mesh be an orthogonal extrusion of the physical mesh.
  //! It also requires that on construction the physical mesh is nowhere coarser than the phase space mesh.
  //! (It provides mechanisms for maintaining this constraint following refinement of the phase space mesh.)
  //! It also requires that the physical (1D) mesh be replicated, as opposed to distributed across MPI ranks.
  class PhasePhysicalMeshMap1D1V : public RefinementObserver {
    MeshPtr _mesh_1D;
    MeshPtr _mesh_2D;
    
    std::map<GlobalIndexType, std::vector<PhysicalCellRelativeRefinementInfo> > _phaseSpaceCellIDToPhysicalCellInfo;
    std::map< GlobalIndexType, std::vector<double> > _physicalCellIDToRefSpaceXCoordInAncestor; // index to vector is ancestor level (0 being immediate parent); the length of vector indicates number of levels of (local) interest
  public:
    PhasePhysicalMeshMap1D1V( MeshPtr mesh_1D, MeshPtr mesh_2D );
    
    const std::vector<PhysicalCellRelativeRefinementInfo> &physicalCellInfo(const GlobalIndexType &phaseSpaceCellID) const;
    std::set<GlobalIndexType> physicalCellIDsForPhaseSpaceCells(const std::set<GlobalIndexType> &phaseSpaceCellIDs) const;
    std::vector<GlobalIndexType> physicalCellIDsForPhaseSpaceCells(const std::vector<GlobalIndexType> &phaseSpaceCellIDs) const;
    
    //! Returns a list of all the unrefinementLevel, uniformDescendantNumbers that appear in _phaseSpaceCellIDToPhysicalCellInfo
    std::set< std::pair<int,int> > referenceSpaceUnrefinementLevelDescendantNumberPairs() const;
    
    int maximumUnrefinementLevel() const;
    
    MeshPtr mesh1D() const;
    MeshPtr mesh2D() const;
    
    PhasePhysicalPointIterator getPhysicalPointIterator(BasisCachePtr phaseSpaceBasisCache, GlobalIndexType phaseSpaceCellID,
                                                        map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > &physicalBasisCachePairs) const;
    
    void rebuildMap();
    
    //! Returns BasisCache, alongside a vector indicating point offsets for each ancestral level.
    std::pair<BasisCachePtr, std::vector<int> > physicalCellBasisCache(const GlobalIndexType physicalCellID, bool testVsTest, int quadratureEnrichment) const;
      
    // RefinementObserver overrides:
    void didHRefine(MeshTopologyPtr phaseSpaceMesh, const set<GlobalIndexType> &phaseSpaceCellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool willRepartitionAndRebuild) override;
    void didHUnrefine(MeshTopologyPtr phaseSpaceMesh, const std::set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild) override;
    void didRepartition(MeshTopologyPtr phaseSpaceMesh) override;
  };
}

#endif /* PhasePhysicalMeshMap1D1V_hpp */
