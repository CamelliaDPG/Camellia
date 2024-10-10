//
//  PhasePhysicalMeshMap.h
//  Camellia
//
//  Created by Roberts, Nathan V on 5/13/20.
//

#ifndef PhasePhysicalMeshMap_h
#define PhasePhysicalMeshMap_h

#include "ExtrudedMeshTopology.h"
#include "MPIWrapper.h"
#include "PhasePhysicalPointIterator.h"
#include "PhysicalCellRelativeRefinementInfo.h"
#include "RefinementObserver.h"
#include "TypeDefs.h"

namespace Camellia
{
  //! This class requires that the phase space mesh be an orthogonal extrusion of the physical mesh.
  //! It also requires that on construction the physical mesh is nowhere coarser than the phase space mesh.
  //! (It provides mechanisms for maintaining this constraint following refinement of the phase space mesh.)
  //! It also requires that the physical (1D) mesh be replicated, as opposed to distributed across MPI ranks.
  class PhasePhysicalMeshMap : public RefinementObserver {
    MeshPtr _physicalMesh;
    MeshPtr _phaseMesh;
    
    ExtrudedMeshTopology* _extrudedMeshTopology; // pointer to the phase-space mesh as an ExtrudedMeshTopology (a convenience)
    
    std::map<GlobalIndexType, std::vector<PhysicalCellRelativeRefinementInfo> > _phaseSpaceCellIDToPhysicalCellInfo;
    std::map< GlobalIndexType, std::vector<double> > _physicalCellIDToRefSpaceXCoordInAncestor; // index to vector is ancestor level (0 being immediate parent); the length of vector indicates number of levels of (local) interest
  public:
    PhasePhysicalMeshMap( MeshPtr physicalMesh, MeshPtr phaseSpaceMesh );
    
    const std::vector<PhysicalCellRelativeRefinementInfo> &physicalCellInfo(const GlobalIndexType &phaseSpaceCellID) const;
    std::set<GlobalIndexType> physicalCellIDsForPhaseSpaceCells(const std::set<GlobalIndexType> &phaseSpaceCellIDs) const;
    std::vector<GlobalIndexType> physicalCellIDsForPhaseSpaceCells(const std::vector<GlobalIndexType> &phaseSpaceCellIDs) const;
    
    //! Returns a list of all the unrefinementLevel, uniformDescendantNumbers that appear in _phaseSpaceCellIDToPhysicalCellInfo
    std::set< std::pair<int,int> > referenceSpaceUnrefinementLevelDescendantNumberPairs() const;
    
    int maximumUnrefinementLevel() const;
    
    MeshPtr physicalMesh() const;
    MeshPtr phaseMesh() const;
    
    PhasePhysicalPointIterator getPhysicalPointIterator(BasisCachePtr phaseSpaceBasisCache, GlobalIndexType phaseSpaceCellID,
                                                        map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > &physicalBasisCachePairs) const;
    
    void rebuildMap();
    
    //! Returns BasisCache, alongside a vector indicating point offsets for each ancestral level.
    std::pair<BasisCachePtr, std::vector<int> > physicalCellBasisCache(const GlobalIndexType physicalCellID, bool testVsTest, int quadratureEnrichment) const;
      
    // RefinementObserver overrides:
    using RefinementObserver::didHRefine;
    void didHRefine(MeshTopologyPtr phaseSpaceMesh, const set<GlobalIndexType> &phaseSpaceCellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool willRepartitionAndRebuild) override;
    using RefinementObserver::didHUnrefine;
    void didHUnrefine(MeshTopologyPtr phaseSpaceMesh, const std::set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild) override;
    void didRepartition(MeshTopologyPtr phaseSpaceMesh) override;
    
    ~PhasePhysicalMeshMap();
  };
}

#endif /* PhasePhysicalMeshMap1D1V_hpp */
