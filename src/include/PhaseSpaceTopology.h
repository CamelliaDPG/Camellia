#ifndef Camellia_PhaseSpaceTopology_h
#define Camellia_PhaseSpaceTopology_h

//
//  PhaseSpaceTopology.h
//  Camellia
//
//  What the moment and field-extrusion code needs to know about a phase-space mesh topology: for each phase-space
//  cell, its extent in each velocity (extrusion) direction and the configuration-space cell it lies over.
//  Implemented by ExtrudedMeshTopology and by BoxMeshTopology.
//

#include "TypeDefs.h"

namespace Camellia
{
  class MeshTopologyView;

  class PhaseSpaceTopology
  {
  public:
    virtual ~PhaseSpaceTopology() {}

    //! the number of velocity (extrusion) dimensions
    virtual int phaseSpaceVelocityDimensionCount() const = 0;
    //! the phase-space cell's extent in the specified velocity dimension (numbered from 0)
    virtual void phaseSpaceVelocityExtents(double &v0, double &v1, IndexType phaseSpaceCellID, int velocityDimension) const = 0;
    //! The cell of the configuration-space mesh topology (active or not) whose extent is the phase-space cell's
    //! projection onto configuration space.  The configuration mesh is at least as fine as the phase-space mesh, so the
    //! active configuration cells under the phase-space cell are this cell's descendants.
    virtual IndexType phaseSpaceConfigurationCellID(IndexType phaseSpaceCellID) const = 0;

    //! The PhaseSpaceTopology interface of a mesh topology (or of its base topology), or NULL if it has none.
    static const PhaseSpaceTopology* phaseSpaceTopology(const MeshTopologyView* meshTopology);
  };
}

#endif
