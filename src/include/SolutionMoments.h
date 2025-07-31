//
//  SolutionMoments.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 9/28/21.
//

#ifndef SolutionMoments_hpp
#define SolutionMoments_hpp

#include "SolutionMomentsForPolyOrder.h"
#include "TypeDefs.h"

#include <Intrepid2_HierarchicalBasisProlongator.hpp>

namespace Camellia
{
  class SolutionMoments
  {
    using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
    using Prolongator = Intrepid2::HierarchicalBasisProlongator<DeviceType,double,double>;
    using BasisIntegralView = Camellia::DeviceView<double>;
    int _numExtrusionDimensions;
    BasisPtr _highestOrderExtrudedSpatialBasis;
    
    int _maxMomentNumber;
    
    SolutionPtr _soln; // in phase space, defined on an ExtrudedMeshTopology
    VarPtr _fieldVar;
    MeshPtr _configurationSpaceMesh; // to start, this is only allowed to be defined on the physical MeshTopology from the ExtrudedMeshTopology in _soln, but we will eventually support a more general slice (extrusions of that physical MeshTopology)
    
    std::vector<IndexType> _physicalCellIDs; // sorted list of physical cell IDs, for which moments are stored in _spatialBasisCoefficients
    
    BasisIntegralView _spatialBasisCoefficients; // (M,C,F) container, defined on the rank-owned cells of the physicalMesh; if physicalMesh varies in p, then this is sized to accommodate maximum p in the F index.
    bool _spatialBasisCoefficientsHasBeenCheckedOut = false;
    
    using IndexView = Camellia::DeviceView<IndexType>;
    
    int _survivingExtrusionDimension = -1; // if non-negative, indicates a single dimension which is *not* integrated away
    int _numDimensionsPhysicalMesh;
    
    BasisReconciliation _basisReconciliation;
    
    std::map<int, BasisPtr> _extrudedSpatialBases; // key is deltaP, relative to initialH1Order
    
    std::map<std::pair<int,int>,Teuchos::RCP<Prolongator>> _physicalMeshProlongators; // keys are deltaP values relative to initialH1Order[0] of the physicalMesh, for {coarse basis, fine basis}.  (Fine basis being the actual physical-mesh basis.)
    
    std::map<int,Teuchos::RCP<SolutionMomentsForPolyOrder> > _momentsForDeltaP; // keys are deltaP values
  public:
    SolutionMoments(const int &maxMomentNumber, SolutionPtr phaseSpaceSoln, VarPtr fieldVar, MeshPtr physicalMesh,
                    int survivingExtrusionDimension = -1);
    
    //! Sends all data to MPI rank 0.  Call update() first to ensure that the data sent is up to date.
    void gatherOnRankZero();
    
    //! 1V variant
    FunctionPtr solutionMomentFunction(const int &xMoment);
    
    //! for 2V
    FunctionPtr solutionMomentFunction(const int &xMoment, const int &yMoment);
    
    //! for 3V
    FunctionPtr solutionMomentFunction(const int &xMoment, const int &yMoment, const int &zMoment);
    
    //! \brief Fills values container, with shape (C,P,M), with computed moments (dimension M).
    void valuesForAllMoments(DeviceView<double> values, BasisCachePtr configurationSpaceBasisCache);
    
    //! Returns the mesh on which SolutionMoments is defined — may be the physical mesh, or the physical mesh extruded.
    MeshPtr mesh();
    
    //! Sets the mesh on which SolutionMoments is defined — may be the configuration-space mesh, or the configuration-space mesh extruded.  Must match dimensionally what was set at construction time (i.e., if there was a surviving extrusion dimension, should be the configuration-space mesh extruded…).
    void setMesh(MeshPtr configurationSpaceMesh);
    
    //! returns the cell index in the basisCoefficients container corresponding to the specified cellID, or -1 if the cellID is not owned by this MPI rank.
    int localCellOrdinal(const IndexType &cellID) const;
    
    //! returns the number of velocity-space dimensions (1 for 1V, 2 for 2V, 3 for 3V)
    int numVelocityDims() const;
    
    //! A (M,C,F) container containing the coefficients for locally-owned cells.  See localCellOrdinal().
    const BasisIntegralView basisCoefficients() const;
    
    MeshPtr phaseSpaceMesh();
    
    //! recomputes all moments on (extruded) spatial mesh.  MPI collective method.
    void update();
    
    int maxMomentNumber() const;
    int numMomentEntries() const;
    
    ~SolutionMoments();
    
    // methods to support SolutionMomentFunction:
    //! The basis defined on the physical mesh, possibly once-extruded in a surviving phase dimension.
    BasisPtr extrudedSpatialBasis();
    
    BasisIntegralView getSpatialBasisCoefficients();
    
    //! sorted list of physical cell IDs, for which moments are stored in _moments
    const std::vector<IndexType> & physicalCellIDs() const;
  };
}

#endif /* SolutionMoments_hpp */
