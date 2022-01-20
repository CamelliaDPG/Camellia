//
//  SolutionMoments.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 9/28/21.
//

#ifndef SolutionMoments_hpp
#define SolutionMoments_hpp

#include "Camellia_TensorIndices.h"
#include "CubatureFactory.h"
#include "Function.h"
#include "MPIWrapper.h"
#include "SubExtrudedMeshTopology.h"
#include "TypeDefs.h"

namespace Camellia
{
  namespace MomentIndices
  {
    //! gets the number of distinct moments given the specified moment count.  In 3D, returns 1 for 1, 4 for 2, etc.
    KOKKOS_INLINE_FUNCTION int getNumMomentEntries(const int &momentCount, const int &numDims)
    {
      if (numDims == 3)
      {
        return (momentCount)*(momentCount+1)*(momentCount+2) / 6;
      }
      else if (numDims == 2)
      {
        return (momentCount)*(momentCount+1) / 2;
      }
      else if (numDims == 1)
      {
        return momentCount;
      }
      else
      {
        return -1;
      }
    }
  
    //! returns an index, with k as the fastest-moving, for contiguous moment indexing.  For zeroth moment, momentCount should be 1; for zeroth and first, it shoud be 2, etc.
    KOKKOS_INLINE_FUNCTION int getMomentIndex(const int &xMoment, const int &yMoment, const int &zMoment,
                                              const int &momentCount)
    {
      // TODO: implement a closed-form version of this; for now we do the brute force thing:
      int index = 0;
      for(int i=0; i<momentCount; ++i)
        for(int j=0; j<momentCount-i; ++j)
          for(int k=0; k<momentCount-i-j; ++k)
          {
            if ((i==xMoment) && (j==yMoment) && (k==zMoment))
              return index;
            index++;
          }
      return index;
    }
  
    //! returns an index, with final dimension as the fastest-moving, for contiguous moment indexing.  For zeroth moment, momentCount should be 1; for zeroth and first, it shoud be 2, etc.  This is the 2D variant.
    KOKKOS_INLINE_FUNCTION int getMomentIndex(const int &xMoment, const int &yMoment, const int &momentCount)
    {
      // TODO: implement a closed-form version of this; for now we do the brute force thing:
      int index = 0;
      for(int i=0; i<momentCount; ++i)
        for(int j=0; j<momentCount-i; ++j)
        {
          if ((i==xMoment) && (j==yMoment))
            return index;
          index++;
        }
      return index;
    }
  
    //! returns an index, with final dimension as the fastest-moving, for contiguous moment indexing.  For zeroth moment, momentCount should be 1; for zeroth and first, it shoud be 2, etc.  This is the 1D variant.
    KOKKOS_INLINE_FUNCTION int getMomentIndex(const int &xMoment, const int &momentCount)
    {
      return xMoment;
    }
  }

  class SolutionMoments
  {
    using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
    const int _maxMomentNumber;
    
    BasisPtr _spatialBasis; // _spatialBasis x extrusionBases = _phaseSpaceBasis.
    BasisPtr _phaseSpaceBasis;
    BasisPtr _extrudedSpatialBasis; // if there is a "surviving" extrusion dimension, _extrudedSpatialBasis = spatialBasis * extrusionBasis.  Otherwise, it's just spatialBasis.
    std::vector<BasisPtr> _extrusionBases;
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> _componentBasisCardinalities; // first entry belongs to spatial basis
    
    using BasisIntegralView = Camellia::DeviceView<double>;
    int _numExtrusionDimensions;
    Kokkos::Array<BasisIntegralView,Intrepid2::Parameters::MaxTensorComponents> _extrusionBasisIntegrals; // one entry per extrusion dimension; each has shape (F); each is a *unit reference space* integral: we integrate over [-1,1], and then divide by 2.  That way, we can simply multiply by the physical measure to get the physical integral.
    
    SolutionPtr _soln; // in phase space, defined on an ExtrudedMeshTopology
    VarPtr _fieldVar;
    
    MeshPtr _configurationSpaceMesh; // to start, this is only allowed to be defined on the physical MeshTopology from the ExtrudedMeshTopology in _soln, but we will eventually support a more general slice (extrusions of that physical MeshTopology)
    
    std::vector<IndexType> _physicalCellIDs; // sorted list of physical cell IDs, for which moments are stored in _spatialBasisCoefficients
    
    BasisIntegralView _spatialBasisCoefficients; // (M,C,F) container, defined on the rank-owned cells of the physicalMesh
    bool _spatialBasisCoefficientsHasBeenCheckedOut = false;
    
    using IndexView = Camellia::DeviceView<IndexType>;
    
    int _survivingExtrusionDimension = -1; // if non-negative, indicates a single dimension which is *not* integrated away
    int _numDimensionsPhysicalMesh;
    
    BasisReconciliation _basisReconciliation;
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
    
    //! A (C,F) container containing the coefficients for locally-owned cells.  See localCellOrdinal().
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
