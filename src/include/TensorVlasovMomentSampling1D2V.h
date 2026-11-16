// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TensorVlasovMomentSampling1D2V.h
//  Camellia
//

#ifndef Camellia_TensorVlasovMomentSampling1D2V_h
#define Camellia_TensorVlasovMomentSampling1D2V_h

#include "TensorMaxwellRHSMomentData.h"
#include "TypeDefs.h"
#include "Camellia_FieldContainer.hpp"

#include <vector>

namespace Camellia
{
  /**
   * \brief Velocity moments of a 1D2V phase-space Vlasov solution, at the cubature points of the rank-local
   * configuration-space cells.
   *
   * For each configuration cell, the moments are integrals over velocity of f at that cell's (x,t) cubature points,
   * using a Gauss rule of the requested degree on each velocity cell.  The phase-space cells over a configuration cell
   * are found through ExtrudedMeshTopology's component maps, so no point search is required.  Each rank evaluates f
   * only on the phase-space cells it owns, and the per-configuration-point partial moments are then summed across
   * ranks; the phase-space and configuration-space partitions therefore need not agree.
   *
   * Entries are ordered configuration cell (in the order of configurationCellIDs) then cubature point; this matches
   * the layout expected by TensorMaxwell1D2VMomentAdapter::buildRHSMomentData().
   */
  struct TensorVlasovMoments1D2V
  {
    int numConfigurationCells = 0;
    int numPointsPerConfigurationCell = 0;
    int numVelocityPoints = 0;           // total velocity quadrature points over the velocity mesh
    TensorPlanar1D2VSpeciesMoments moments;
  };

  /**
   * \brief Computes density and first velocity moments of the trial variable fVar of phaseSpaceSolution.
   *
   * \param configurationCellIDs   [in] - rank-local configuration-space cells at whose cubature points moments are wanted
   * \param configurationMesh      [in] - the configuration-space (x,t) mesh
   * \param phaseSpaceSolution     [in] - solution on an ExtrudedMeshTopology over configurationMesh's topology, with
   *                                      two velocity extrusions
   * \param fVar                   [in] - the distribution-function trial variable
   * \param velocityQuadratureDegree [in] - Gauss rule degree used in each velocity direction on each velocity cell; 0
   *                                      selects a rule per phase-space cell from that cell's own cubature degree
   *                                      (lowered in a velocity direction whose order is below the cell's highest),
   *                                      which equals defaultVlasovMomentVelocityQuadratureDegree1D2V on meshes of
   *                                      uniform order.  numVelocityPoints is then 0.
   * \param useRelativisticCurrentMoments [in] - if true, first moments are the relativistic currents (weights u/gamma);
   *                                      otherwise, proper-velocity moments (weights u)
   * \param lightSpeed             [in] - light speed used for gamma when useRelativisticCurrentMoments is true
   *
   * Must be called collectively on all ranks of the phase-space mesh communicator.
   */
  TensorVlasovMoments1D2V computeVlasovMomentsByPhaseSpaceCell1D2V(const std::vector<GlobalIndexType> &configurationCellIDs,
                                                                   MeshPtr configurationMesh,
                                                                   SolutionPtr phaseSpaceSolution,
                                                                   VarPtr fVar,
                                                                   int velocityQuadratureDegree,
                                                                   bool useRelativisticCurrentMoments,
                                                                   double lightSpeed);

  /**
   * \brief As above, but at caller-specified points on each configuration cell rather than at its cubature points.
   *
   * \param configurationReferencePoints [in] - (P,2) points on the reference configuration cell; the moments are
   *                                      evaluated at their physical images in each of configurationCellIDs.  This
   *                                      allows the moments of the polynomial f to be sampled densely (e.g., for
   *                                      visualization), rather than only at the points the Maxwell RHS uses.
   *
   * Entries are ordered configuration cell then reference point.  Must be called collectively.
   */
  TensorVlasovMoments1D2V computeVlasovMomentsByPhaseSpaceCell1D2V(const std::vector<GlobalIndexType> &configurationCellIDs,
                                                                   MeshPtr configurationMesh,
                                                                   SolutionPtr phaseSpaceSolution,
                                                                   VarPtr fVar,
                                                                   int velocityQuadratureDegree,
                                                                   bool useRelativisticCurrentMoments,
                                                                   double lightSpeed,
                                                                   const Camellia::FieldContainer<double> &configurationReferencePoints);

  /** \brief Default velocityQuadratureDegree for computeVlasovMomentsByPhaseSpaceCell1D2V(): the maximum cubature degree
   *         over the phase-space cells (which the sampler integrates over), agreed across ranks.
   *
   * A rank may own phase-space cells but no configuration cells, so the degree must not be derived from the
   * configuration cells a rank owns.  Must be called collectively on all ranks of the phase-space mesh communicator.
   */
  int defaultVlasovMomentVelocityQuadratureDegree1D2V(SolutionPtr phaseSpaceSolution);
}

#endif
