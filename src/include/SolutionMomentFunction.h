//
//  SolutionMomentFunction.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/24/21.
//

#ifndef SolutionMomentFunction_h
#define SolutionMomentFunction_h

#include "Camellia_FieldContainerConversion.hpp"
#include "CubatureFactory.h"
#include "Function.h"
#include "MPIWrapper.h"
#include "SubExtrudedMeshTopology.h"
#include "TensorBasis.h"

namespace Camellia
{
  // TODO: name/place this method somewhere more appropriate; defined here because used in SolutionMomentFunction, and outside SolutionMomentFunction to allow external testing
  KOKKOS_INLINE_FUNCTION void extractNextComponentOrdinal(int &ordinalInComponent, int &remainingTensorOrdinal, const int &componentCardinality)
  {
    ordinalInComponent = remainingTensorOrdinal % componentCardinality;
    remainingTensorOrdinal /= componentCardinality;
  }

  // TODO: name/place this method somewhere more appropriate; defined here because used in SolutionMomentFunction, and outside SolutionMomentFunction to allow external testing
  KOKKOS_INLINE_FUNCTION void setTensorOrdinal(int &tensorOrdinal, const int &spatialOrdinal, const int &spatialCardinality, const int &extrusionFieldOrdinal)
  {
    tensorOrdinal = spatialOrdinal + spatialCardinality * extrusionFieldOrdinal;
  }

  KOKKOS_INLINE_FUNCTION void setTensorOrdinal(int &tensorOrdinal, const Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> &componentOrdinals,
                                               const Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> &componentCardinalities,
                                               const int &numComponents)
  {
    tensorOrdinal = 0;
    for (int d=numComponents-1; d>0; d--)
    {
      tensorOrdinal += componentOrdinals[d];
      tensorOrdinal *= componentCardinalities[d-1];
    }
    tensorOrdinal += componentOrdinals[0];
  }

  class SolutionMomentFunction : public TFunction<double>
  {
    using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
    
    BasisPtr _spatialBasis; // _spatialBasis x extrusionBases = _phaseSpaceBasis.
    BasisPtr _phaseSpaceBasis;
    BasisPtr _extrudedSpatialBasis; // if there is a "surviving" extrusion dimension, _extrudedSpatialBasis = spatialBasis * extrusionBasis.  Otherwise, it's just spatialBasis.
    std::vector<BasisPtr> _extrusionBases;
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> _componentBasisCardinalities; // first entry belongs to spatial basis
    
    using BasisIntegralView = Camellia::DeviceView<double>;
    int _numExtrusionDimensions;
    Kokkos::Array<BasisIntegralView,Intrepid2::Parameters::MaxTensorComponents> _extrusionBasisIntegrals; // one entry per extrusion dimension; each has shape (F); each is a *unit reference space* integral: we integrate over [-1,1], and then divide by 2.  That way, we can simply multiply by the a physical measure to get the physical integral.
    
    SolutionPtr _soln; // in phase space, defined on an ExtrudedMeshTopology
    VarPtr _fieldVar;
    
    MeshPtr _possiblyExtrudedPhysicalMesh; // to start, this is only allowed to be defined on the physical MeshTopology from the ExtrudedMeshTopology in _soln, but we will eventually support a more general slice (extrusions of that physical MeshTopology)
    
    std::vector<IndexType> _physicalCellIDs; // sorted list of physical cell IDs, for which moments are stored in _moments
    BasisIntegralView _moments; // (C,P[,D,…,D]) container, defined on the rank-owned cells of the physicalMesh
    bool _momentsHasBeenCheckedOut = false;
    
    BasisIntegralView _spatialBasisCoefficients; // (C,F) container, defined on the rank-owned cells of the physicalMesh
    bool _spatialBasisCoefficientsHasBeenCheckedOut = false;
    
    using IndexView = Camellia::DeviceView<IndexType>;
    IndexView _physicalCellOrdinalLookup; // resized on updateValues(); recomputed during values(); allows lookup of the BasisCache's cell ordinal in the _spatialBasisCoefficients container…
    bool _physicalCellOrdinalLookupHasBeenCheckedOut = false;
    
    int _survivingExtrusionDimension = -1; // if non-negative, indicates a single dimension which is *not* integrated away
    
    BasisReconciliation _basisReconciliation;
  public:
    SolutionMomentFunction(SolutionPtr phaseSpaceSoln, VarPtr fieldVar, int momentNumber, MeshPtr physicalMesh, int survivingExtrusionDimension = -1);
    
    //! Returns the mesh on which the SolutionMomentFunction is defined — may be the physical mesh, or the physical mesh extruded.
    MeshPtr mesh();
    
    //! returns the cell index in the basisCoefficients container corresponding to the specified cellID, or -1 if the cellID is not owned by this MPI rank.
    int localCellOrdinal(const IndexType &cellID) const;
    
    //! A (C,F) container containing the coefficients for locally-owned cells.  See localCellOrdinal().
    const BasisIntegralView basisCoefficients() const;
    
    //! recomputes coefficients on (extruded) spatial mesh.  MPI collective method.
    void updateValues();
    
    using TFunction<double>::values;
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    ~SolutionMomentFunction();
  };
}

#endif /* SolutionMomentFunction_h */
