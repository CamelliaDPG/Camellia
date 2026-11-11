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

#include <vector>

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
    //! The configuration-space mesh topology beneath this phase-space mesh.
    virtual MeshTopologyViewPtr phaseSpaceConfigurationTopology() const = 0;
    //! The cell of the configuration-space mesh topology (active or not) whose extent is exactly the phase-space cell's
    //! projection onto configuration space, or -1 if there is none.  (There is always one for an ExtrudedMeshTopology.
    //! On a BoxMeshTopology there is none when the phase-space cell is at different refinement levels in different
    //! configuration directions, e.g. bisected in x but not in t.)
    virtual IndexType phaseSpaceConfigurationCellID(IndexType phaseSpaceCellID) const = 0;
    //! The active configuration-space cells that lie within the phase-space cell's projection onto configuration
    //! space.  The configuration mesh is at least as fine as the phase-space mesh, so these tile the projection.
    //! The default implementation returns the active descendants of phaseSpaceConfigurationCellID().
    virtual std::vector<IndexType> phaseSpaceActiveConfigurationCells(IndexType phaseSpaceCellID) const;

    //! The PhaseSpaceTopology interface of a mesh topology (or of its base topology), or NULL if it has none.
    static const PhaseSpaceTopology* phaseSpaceTopology(const MeshTopologyView* meshTopology);
  };
}

#endif
