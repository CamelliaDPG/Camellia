#include "Teuchos_UnitTestHarness.hpp"

#include "BoxMeshTopology.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "SpatialFilter.h"

#include <cmath>

using namespace Camellia;
using namespace std;

// The relativistic Vlasov-Maxwell 1D2V space-time formulation with its phase-space mesh on a BoxMeshTopology.

namespace
{
  const double X0 = 0.0, XWIDTH = 2.0, T0 = 0.0, TWIDTH = 0.5, V0 = -2.0, VWIDTH = 4.0;
  const double Q = 1.0, M = 1.0, C = 2.0, EPS0 = 0.5;

  //! RAII: use box meshes for the phase space of formulations constructed in this scope
  struct UseBoxMesh
  {
    bool _previous;
    UseBoxMesh(bool value) : _previous(VlasovSpaceTimeFormulation::useBoxMeshForPhaseSpace()) { VlasovSpaceTimeFormulation::setUseBoxMeshForPhaseSpace(value); }
    ~UseBoxMesh() { VlasovSpaceTimeFormulation::setUseBoxMeshForPhaseSpace(_previous); }
  };

  struct Problem
  {
    Teuchos::RCP<ConfigurationSpaceFormulationMaxwell1D2V> configForm;
    Teuchos::RCP<RelativisticVlasovSpaceTimeFormulation> form;
  };

  Problem problem(bool boxMesh, int xElements, int vxElements, int vyElements, int H1Order, FunctionPtr f0, double q = Q)
  {
    UseBoxMesh useBoxMesh(boxMesh);
    const int tElements = 1, delta_k = 1, numUniformRefinements = 0;
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(X0, X0 + XWIDTH, xElements, false);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(T0, TWIDTH, tElements);
    std::vector<double> velocityVertices_x = MeshTools::verticesFor1DIntervalMesh(V0, VWIDTH, vxElements);
    std::vector<double> velocityVertices_y = MeshTools::verticesFor1DIntervalMesh(V0, VWIDTH, vyElements);
    std::vector<double> velocityVertices_z;

    Problem p;
    p.configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, EPS0));
    p.form = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(C, q, M, p.configForm, spatialMeshTopo, temporalVertices,
                                                                     velocityVertices_x, velocityVertices_y, velocityVertices_z,
                                                                     H1Order, delta_k, numUniformRefinements));
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(X0 + XWIDTH);
    auto xLeftRightTag = p.form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, true);
    p.form->addConfigurationSpaceBC(xLeftRightTag, p.form->VE_configurationSpaceTrace(), Function::zero());
    p.form->addConfigurationSpaceBC(xLeftRightTag, p.configForm->A_hat(1), Function::zero());
    p.form->addConfigurationSpaceBC(p.form->tagConfigurationInitialTime(), p.configForm->E_hat(0), Function::zero());
    p.form->setFixedPointMinSubsteps(2);
    p.form->setFixedPointMaxSubsteps(2);
    p.form->setFixedPointTolerance(1e-12);
    p.form->setInitialAndInflowConditions(f0, f0);
    return p;
  }

  //! an initial/inflow distribution that drives a current: not symmetric in u_x
  FunctionPtr drivenDistribution()
  {
    FunctionPtr x = Function::xn(1);
    FunctionPtr ux = MonomialFunction::xi_n(2, 1); // coordinates are (x, t, u_x, u_y)
    return Function::constant(1.0) + 0.1 * x + 0.05 * ux;
  }

  struct Diagnostics
  {
    double fNorm, fEnergyError, ExNorm;
    GlobalIndexType vlasovDofs, vlasovCells, configCells;
  };

  Diagnostics diagnostics(Problem &p)
  {
    Diagnostics d;
    auto vlasovMesh = p.form->vlasovMesh();
    auto configMesh = p.form->configurationSpaceMesh();
    d.fNorm = Function::solution(p.form->f(), p.form->vlasovSolution())->l2norm(vlasovMesh);
    d.fEnergyError = p.form->vlasovSolution()->energyErrorTotal();
    d.ExNorm = Function::solution(p.configForm->E_fieldVar(0), p.form->configurationSpaceSolution())->l2norm(configMesh);
    d.vlasovDofs = vlasovMesh->numGlobalDofs();
    d.vlasovCells = vlasovMesh->numActiveElements();
    d.configCells = configMesh->numActiveElements();
    return d;
  }

  void print(Teuchos::FancyOStream &out, const std::string &label, const Diagnostics &d)
  {
    out << label << ": " << d.vlasovCells << " phase cells, " << d.vlasovDofs << " dofs, " << d.configCells << " config cells; ||f|| = " << d.fNorm
        << ", ||E_x|| = " << d.ExNorm << ", Vlasov energy error = " << d.fEnergyError << endl;
  }

TEUCHOS_UNIT_TEST( BoxVlasov, MatchesExtrudedMesh )
{
  // the same problem with the phase-space mesh on an ExtrudedMeshTopology and on a BoxMeshTopology
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Problem extruded = problem(false, xElements, vxElements, vyElements, H1Order, drivenDistribution());
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
  TEST_ASSERT(dynamic_cast<BoxMeshTopology*>(box.form->vlasovMesh()->getTopology().get()) != NULL);
  extruded.form->solve();
  box.form->solve();
  Diagnostics de = diagnostics(extruded), db = diagnostics(box);
  print(out, "extruded", de);
  print(out, "box     ", db);
  TEST_EQUALITY(de.vlasovCells, db.vlasovCells);
  TEST_EQUALITY(de.vlasovDofs, db.vlasovDofs);
  TEST_COMPARE(de.ExNorm, >, 1e-6); // the problem is not trivial
  TEST_FLOATING_EQUALITY(de.fNorm, db.fNorm, 1e-8);
  TEST_FLOATING_EQUALITY(de.ExNorm, db.ExNorm, 1e-7);
  TEST_FLOATING_EQUALITY(de.fEnergyError, db.fEnergyError, 1e-6);
}

TEUCHOS_UNIT_TEST( BoxVlasov, VelocityOnlyRefinementMatchesFinerVelocityMesh )
{
  // Bisecting every box cell in u_x alone gives the mesh with twice as many u_x root cells; the configuration mesh is untouched.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Problem extrudedFine = problem(false, xElements, 2 * vxElements, vyElements, H1Order, drivenDistribution());
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
  const GlobalIndexType configCellsBefore = box.form->configurationSpaceMesh()->numActiveElements();
  std::map<GlobalIndexType,unsigned> masks;
  const unsigned uxOnly = 1u << 2; // coordinates are (x, t, u_x, u_y)
  for (GlobalIndexType cellID : box.form->vlasovMesh()->getActiveCellIDsGlobal()) masks[cellID] = uxOnly;
  box.form->vlasovMesh()->hRefine(masks);
  TEST_EQUALITY(box.form->configurationSpaceMesh()->numActiveElements(), configCellsBefore);
  box.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution()); // BCs are projected per cell: redo on the new mesh
  extrudedFine.form->solve();
  box.form->solve();
  Diagnostics de = diagnostics(extrudedFine), db = diagnostics(box);
  print(out, "extruded, 4 u_x cells    ", de);
  print(out, "box, 2 u_x cells bisected", db);
  TEST_EQUALITY(de.vlasovCells, db.vlasovCells);
  TEST_EQUALITY(de.vlasovDofs, db.vlasovDofs);
  TEST_FLOATING_EQUALITY(de.fNorm, db.fNorm, 1e-8);
  TEST_FLOATING_EQUALITY(de.ExNorm, db.ExNorm, 1e-7);
}

TEUCHOS_UNIT_TEST( BoxVlasov, LocalVelocityRefinement )
{
  // With zero charge the particles stream freely and f = 1 is a solution.  Refine some cells in u_x only and others in
  // u_y only, so that there are hanging and crossed faces.  On a mesh with hanging faces f = 1 is not reproduced
  // exactly, because the flux (u_x / gamma) f is not a polynomial in u (the extruded mesh shows an error of the same
  // size with hanging nodes: see IsotropicHangingRefinementMatchesExtrudedMesh); it must stay small, and the fields zero.
  const int xElements = 2, vxElements = 2, vyElements = 2, H1Order = 2;
  auto solveAndMeasure = [&](bool refine, GlobalIndexType &cellCount)
  {
    Problem box = problem(true, xElements, vxElements, vyElements, H1Order, Function::constant(1.0), 0.0);
    auto vlasovMesh = box.form->vlasovMesh();
    if (refine)
    {
      std::map<GlobalIndexType,unsigned> masks;
      int ordinal = 0;
      for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal())
      {
        if (ordinal % 3 == 0) masks[cellID] = 1u << 2;      // u_x
        else if (ordinal % 3 == 1) masks[cellID] = 1u << 3; // u_y
        ordinal++;
      }
      vlasovMesh->hRefine(masks);
      box.form->setInitialAndInflowConditions(Function::constant(1.0), Function::constant(1.0));
    }
    box.form->solve();
    cellCount = vlasovMesh->numActiveElements();
    const double ExNorm = Function::solution(box.configForm->E_fieldVar(0), box.form->configurationSpaceSolution())->l2norm(box.form->configurationSpaceMesh());
    TEST_COMPARE(ExNorm, <, 1e-10);
    return (Function::solution(box.form->f(), box.form->vlasovSolution()) - Function::constant(1.0))->l2norm(vlasovMesh);
  };
  GlobalIndexType coarseCells, refinedCells;
  const double coarseError = solveAndMeasure(false, coarseCells);
  const double refinedError = solveAndMeasure(true, refinedCells);
  out << coarseCells << " phase cells: ||f - 1|| = " << coarseError << "; " << refinedCells << " phase cells: ||f - 1|| = " << refinedError << endl;
  TEST_COMPARE(refinedCells, >, coarseCells);
  TEST_COMPARE(coarseError, <, 1e-10);
  TEST_COMPARE(refinedError, <, 1e-2);
}

TEUCHOS_UNIT_TEST( BoxVlasov, SpaceTimeRefinementRefinesConfigurationMesh )
{
  // bisecting a phase-space cell in x and t (together) must refine the configuration cell beneath it
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, Function::constant(1.0));
  auto vlasovMesh = box.form->vlasovMesh();
  auto configMesh = box.form->configurationSpaceMesh();
  const GlobalIndexType configCellsBefore = configMesh->numActiveElements();
  const GlobalIndexType cellID = *vlasovMesh->getActiveCellIDsGlobal().begin();
  vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{cellID, 1u | 2u}});
  TEST_COMPARE(configMesh->numActiveElements(), >=, configCellsBefore + 3);
  // x alone is rejected, for now
  const GlobalIndexType otherCellID = *vlasovMesh->getActiveCellIDsGlobal().rbegin();
  TEST_THROW(vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{otherCellID, 1u}}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( BoxVlasov, NextTimeSlab )
{
  // Two time slabs.  On the box mesh, the first slab is bisected in u_x everywhere; the second slab's mesh starts
  // unrefined and must be refined (in u_x only) to match at the slab interface.  Compare with the extruded mesh that
  // has twice as many u_x cells from the start.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Diagnostics results[2];
  for (int useBox=0; useBox<2; useBox++)
  {
    UseBoxMesh useBoxMesh(useBox == 1); // nextTimeSlab() constructs a formulation
    Problem p = problem(useBox == 1, xElements, (useBox == 1) ? vxElements : 2 * vxElements, vyElements, H1Order, drivenDistribution());
    if (useBox == 1)
    {
      std::map<GlobalIndexType,unsigned> masks;
      for (GlobalIndexType cellID : p.form->vlasovMesh()->getActiveCellIDsGlobal()) masks[cellID] = 1u << 2;
      p.form->vlasovMesh()->hRefine(masks);
      p.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
    }
    p.form->solve();
    auto secondSlab = p.form->nextTimeSlab(TWIDTH, 1);
    TEST_EQUALITY(secondSlab->vlasovMesh()->numActiveElements(), p.form->vlasovMesh()->numActiveElements());
    secondSlab->solve();
    Diagnostics &d = results[useBox];
    d.fNorm = Function::solution(secondSlab->f(), secondSlab->vlasovSolution())->l2norm(secondSlab->vlasovMesh());
    d.ExNorm = Function::solution(p.configForm->E_fieldVar(0), secondSlab->configurationSpaceSolution())->l2norm(secondSlab->configurationSpaceMesh());
    d.fEnergyError = secondSlab->vlasovSolution()->energyErrorTotal();
    d.vlasovDofs = secondSlab->vlasovMesh()->numGlobalDofs();
    d.vlasovCells = secondSlab->vlasovMesh()->numActiveElements();
    d.configCells = secondSlab->configurationSpaceMesh()->numActiveElements();
    print(out, (useBox == 1) ? "second slab, box     " : "second slab, extruded", d);
  }
  TEST_EQUALITY(results[0].vlasovDofs, results[1].vlasovDofs);
  TEST_FLOATING_EQUALITY(results[0].fNorm, results[1].fNorm, 1e-8);
  TEST_FLOATING_EQUALITY(results[0].ExNorm, results[1].ExNorm, 1e-7);
}
TEUCHOS_UNIT_TEST( BoxVlasov, IsotropicHangingRefinementMatchesExtrudedMesh )
{
  // One phase-space cell refined in all four directions (hanging faces in every direction; the configuration mesh is
  // refined beneath it): a mesh that both topologies can represent.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  for (double q : {Q, 0.0})
  {
    FunctionPtr f0 = (q == 0.0) ? FunctionPtr(Function::constant(1.0)) : drivenDistribution();
    Diagnostics results[2];
    double uniformStateError[2];
    for (int useBox=0; useBox<2; useBox++)
    {
      Problem p = problem(useBox == 1, xElements, vxElements, vyElements, H1Order, f0, q);
      auto vlasovMesh = p.form->vlasovMesh();
      // the cell containing a point near the low corner of the domain
      Camellia::FieldContainer<double> point(1, 4);
      point(0,0) = X0 + 0.1 * XWIDTH; point(0,1) = T0 + 0.1 * TWIDTH; point(0,2) = V0 + 0.1 * VWIDTH; point(0,3) = V0 + 0.1 * VWIDTH;
      const GlobalIndexType cellID = vlasovMesh->getTopology()->cellIDsForPoints(point)[0];
      vlasovMesh->hRefine(std::set<GlobalIndexType>{cellID});
      p.form->setInitialAndInflowConditions(f0, f0);
      p.form->solve();
      results[useBox] = diagnostics(p);
      uniformStateError[useBox] = (Function::solution(p.form->f(), p.form->vlasovSolution()) - Function::constant(1.0))->l2norm(vlasovMesh);
      print(out, (useBox == 1) ? "box     " : "extruded", results[useBox]);
      if (q == 0.0) out << "  ||f - 1|| = " << uniformStateError[useBox] << endl;
    }
    TEST_EQUALITY(results[0].vlasovCells, results[1].vlasovCells);
    TEST_EQUALITY(results[0].vlasovDofs, results[1].vlasovDofs);
    TEST_EQUALITY(results[0].configCells, results[1].configCells);
    TEST_FLOATING_EQUALITY(results[0].fNorm, results[1].fNorm, 1e-8);
    TEST_FLOATING_EQUALITY(results[0].fEnergyError + 1.0, results[1].fEnergyError + 1.0, 1e-7);
    if (q != 0.0) { TEST_FLOATING_EQUALITY(results[0].ExNorm, results[1].ExNorm, 1e-7); }
  }
}
TEUCHOS_UNIT_TEST( BoxVlasov, GeometricMultigridMatchesDirectSolve )
{
  // the Vlasov solve on a box mesh with local u_x-only refinement, by multigrid-preconditioned iteration and directly
  const int xElements = 2, vxElements = 2, vyElements = 2, H1Order = 2;
  Diagnostics results[2];
  for (int useGMG=0; useGMG<2; useGMG++)
  {
    Problem box = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
    auto vlasovMesh = box.form->vlasovMesh();
    std::map<GlobalIndexType,unsigned> masks;
    int ordinal = 0;
    for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal())
    {
      if (ordinal++ % 2 == 0) masks[cellID] = 1u << 2; // u_x
    }
    vlasovMesh->hRefine(masks);
    box.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
    box.form->setUseGeometricMultigrid(useGMG == 1);
    box.form->setIterativeSolveTolerance(1e-12);
    box.form->setIterativeSolveMaxIterations(2000);
    box.form->solve();
    results[useGMG] = diagnostics(box);
    print(out, (useGMG == 1) ? "multigrid" : "direct   ", results[useGMG]);
  }
  TEST_FLOATING_EQUALITY(results[0].fNorm, results[1].fNorm, 1e-7);
  TEST_FLOATING_EQUALITY(results[0].ExNorm, results[1].ExNorm, 1e-6);
}
} // namespace
