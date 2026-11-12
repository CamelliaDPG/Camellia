#include "Teuchos_UnitTestHarness.hpp"

#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "SpatialFilter.h"

#include <cmath>
#include <iostream>
#include <sstream>

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

  Problem problem(bool boxMesh, int xElements, int vxElements, int vyElements, int H1Order, FunctionPtr f0, double q = Q, int tElements = 1,
                  EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL, double tWidth = TWIDTH)
  {
    UseBoxMesh useBoxMesh(boxMesh);
    const int delta_k = 1, numUniformRefinements = 0;
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(X0, X0 + XWIDTH, xElements, false);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(T0, tWidth, tElements);
    std::vector<double> velocityVertices_x = MeshTools::verticesFor1DIntervalMesh(V0, VWIDTH, vxElements);
    std::vector<double> velocityVertices_y = MeshTools::verticesFor1DIntervalMesh(V0, VWIDTH, vyElements);
    std::vector<double> velocityVertices_z;

    Problem p;
    p.configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, EPS0));
    p.form = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(C, q, M, p.configForm, spatialMeshTopo, temporalVertices,
                                                                     velocityVertices_x, velocityVertices_y, velocityVertices_z,
                                                                     H1Order, delta_k, numUniformRefinements,
                                                                     RelativisticVlasovSpaceTimeFormulation::L2TestSpaceInnerProduct, 1.0, basisFamily));
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
}

TEUCHOS_UNIT_TEST( BoxVlasov, SingleDirectionSpaceTimeRefinement )
{
  // Bisecting a phase-space cell in x alone, or in t alone, refines the configuration cells beneath it (regularly, so
  // in both directions): the configuration mesh stays at least as fine as the phase-space mesh.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, Function::constant(1.0));
  auto vlasovMesh = box.form->vlasovMesh();
  auto configMesh = box.form->configurationSpaceMesh();
  auto topology = dynamic_cast<BoxMeshTopology*>(vlasovMesh->getTopology().get());
  const std::set<GlobalIndexType> activeCellSet = vlasovMesh->getActiveCellIDsGlobal(); // returned by value: keep one copy
  std::vector<GlobalIndexType> cells(activeCellSet.begin(), activeCellSet.end());
  TEST_EQUALITY(configMesh->numActiveElements(), 2);
  for (GlobalIndexType cellID : cells) { TEST_EQUALITY(topology->phaseSpaceActiveConfigurationCells(cellID).size(), 1); }

  // x only, for a cell over the first configuration cell
  vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{cells.front(), 1u}});
  TEST_EQUALITY(configMesh->numActiveElements(), 5);
  for (IndexType childID : topology->getCell(cells.front())->getChildIndices(vlasovMesh->getTopology()))
  {
    TEST_EQUALITY(topology->phaseSpaceConfigurationCellID(childID), -1);               // half in x, whole in t: not a configuration cell
    TEST_EQUALITY(topology->phaseSpaceActiveConfigurationCells(childID).size(), 2);    // but two of them tile it
  }
  // the other cells over that configuration cell were not refined; four configuration cells tile each
  int cellsOverFourConfigurationCells = 0;
  for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal())
  {
    if (topology->phaseSpaceActiveConfigurationCells(cellID).size() == 4) cellsOverFourConfigurationCells++;
  }
  TEST_EQUALITY(cellsOverFourConfigurationCells, 1);

  // t only, for a cell over the second configuration cell
  vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{cells.back(), 2u}});
  TEST_EQUALITY(configMesh->numActiveElements(), 8);
}

TEUCHOS_UNIT_TEST( BoxVlasov, ConfigurationLookupIsQuiet )
{
  // A phase-space cell whose configuration cell has been refined further: the centroid of its projection is a vertex
  // shared by all the children of that configuration cell.  Looking it up must not go through MeshTopology's point
  // search, which then finds the parent but none of its children and prints a long diagnostic each time.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  // (a slab width that is not dyadic, as in the MITL driver, so that refined t coordinates are rounded)
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, Function::constant(1.0), Q, 1, BASIS_FAMILY_HIERARCHICAL, 0.1);
  auto vlasovMesh = box.form->vlasovMesh();
  auto topology = dynamic_cast<BoxMeshTopology*>(vlasovMesh->getTopology().get());
  const std::set<GlobalIndexType> activeCellSet = vlasovMesh->getActiveCellIDsGlobal();
  std::vector<GlobalIndexType> cells(activeCellSet.begin(), activeCellSet.end());
  // refining one cell in x and t twice refines the configuration cell beneath it twice; the other cells over it are unrefined
  vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{cells.front(), 1u | 2u}});
  const GlobalIndexType child = topology->getCell(cells.front())->getChildIndices(vlasovMesh->getTopology())[0];
  vlasovMesh->hRefine(std::map<GlobalIndexType,unsigned>{{child, 1u | 2u}});

  std::ostringstream captured;
  std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
  std::map<GlobalIndexType,std::pair<IndexType,size_t>> lookups;
  for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal())
  {
    lookups[cellID] = {topology->phaseSpaceConfigurationCellID(cellID), topology->phaseSpaceActiveConfigurationCells(cellID).size()};
  }
  std::cout.rdbuf(previous);
  TEST_EQUALITY(captured.str(), "");

  int cellsOverSevenConfigurationCells = 0;
  for (const auto &entry : lookups)
  {
    TEST_ASSERT(entry.second.first != IndexType(-1)); // all cells are refined isotropically in (x, t)
    // each cell's configuration cell, at its level, has the same extents in (x, t) as the cell
    const auto cellExtents = topology->cellExtents(entry.first);
    const auto configurationExtents = topology->configurationCellExtents(entry.second.first);
    for (int k=0; k<2; k++)
    {
      TEST_FLOATING_EQUALITY(configurationExtents[k].first + 1.0, cellExtents[k].first + 1.0, 1e-14);
      TEST_FLOATING_EQUALITY(configurationExtents[k].second, cellExtents[k].second, 1e-14);
    }
    if (entry.second.second == 7) cellsOverSevenConfigurationCells++; // 3 children + 4 grandchildren
  }
  TEST_EQUALITY(cellsOverSevenConfigurationCells, 1); // the unrefined sibling of cells.front() in velocity
}

TEUCHOS_UNIT_TEST( BoxVlasov, FieldExtrusionOnUnmatchedCells )
{
  // A configuration-space field that is a polynomial of the field degree in (x,t), extruded to a phase-space mesh in
  // which some cells are bisected in x alone and some in t alone, must be reproduced at every phase-space point.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Problem box = problem(true, xElements, vxElements, vyElements, H1Order, Function::constant(1.0));
  auto vlasovMesh = box.form->vlasovMesh();
  const std::set<GlobalIndexType> activeCellSet = vlasovMesh->getActiveCellIDsGlobal(); // returned by value: keep one copy
  std::vector<GlobalIndexType> cells(activeCellSet.begin(), activeCellSet.end());
  std::map<GlobalIndexType,unsigned> masks;
  masks[cells[0]] = 1u;          // x
  masks[cells[1]] = 2u;          // t
  masks[cells[2]] = 1u | 4u;     // x and u_x
  vlasovMesh->hRefine(masks);

  FunctionPtr x = Function::xn(1), t = Function::yn(1);
  FunctionPtr E_exact = Function::constant(1.0) + 2.0 * x - 3.0 * t + 0.5 * x * t;
  auto configSoln = box.form->configurationSpaceSolution();
  VarPtr E_var = box.configForm->E_fieldVar(0);
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[E_var->ID()] = E_exact;
  configSoln->projectOntoMesh(projectionMap);

  auto extrusion = Teuchos::rcp(new PhysicalSolutionExtrusion(configSoln, E_var, vlasovMesh));
  extrusion->updateValues();
  FunctionPtr extrusionFunction = extrusion;
  const double volumeError = (extrusionFunction - E_exact)->l2norm(vlasovMesh);
  out << "L2 error of the extruded field over the phase-space mesh: " << volumeError << endl;
  TEST_COMPARE(volumeError, <, 1e-11);
  int unmatchedCells = 0;
  auto topology = dynamic_cast<BoxMeshTopology*>(vlasovMesh->getTopology().get());
  for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal()) if (topology->phaseSpaceConfigurationCellID(cellID) == -1) unmatchedCells++;
  TEST_COMPARE(unmatchedCells, >=, 6);
}

TEUCHOS_UNIT_TEST( BoxVlasov, MomentsOnUnmatchedCells )
{
  // The same polynomial f on two phase-space meshes over the same configuration mesh: every cell bisected in x alone,
  // and every cell bisected in x and t.  The moments, and so the fields they drive, must agree.
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  double ExNorms[2], ByNorms[2];
  GlobalIndexType configCells[2];
  for (int both=0; both<2; both++)
  {
    Problem box = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
    auto vlasovMesh = box.form->vlasovMesh();
    std::map<GlobalIndexType,unsigned> masks;
    for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal()) masks[cellID] = (both == 1) ? (1u | 2u) : 1u;
    vlasovMesh->hRefine(masks);
    box.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
    std::map<int,FunctionPtr> projectionMap;
    projectionMap[box.form->f()->ID()] = drivenDistribution();
    box.form->vlasovSolution()->projectOntoMesh(projectionMap);
    box.form->configurationSpaceSolve();
    auto configMesh = box.form->configurationSpaceMesh();
    configCells[both] = configMesh->numActiveElements();
    ExNorms[both] = Function::solution(box.configForm->E_fieldVar(0), box.form->configurationSpaceSolution())->l2norm(configMesh);
    ByNorms[both] = Function::solution(box.configForm->B_fieldVar(1), box.form->configurationSpaceSolution())->l2norm(configMesh);
    out << ((both == 1) ? "x and t: " : "x only:  ") << vlasovMesh->numActiveElements() << " phase cells, " << configCells[both]
        << " config cells; ||E_x|| = " << ExNorms[both] << ", ||B_y|| = " << ByNorms[both] << endl;
  }
  TEST_EQUALITY(configCells[0], configCells[1]);
  TEST_COMPARE(ExNorms[0], >, 1e-6);
  TEST_FLOATING_EQUALITY(ExNorms[0], ExNorms[1], 1e-9);
  TEST_FLOATING_EQUALITY(ByNorms[0] + 1.0, ByNorms[1] + 1.0, 1e-9);
}

TEUCHOS_UNIT_TEST( BoxVlasov, SolveWithSingleDirectionSpaceTimeRefinement )
{
  // the coupled solve with every cell bisected in x alone, against every cell bisected in x and t: close, not equal
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 2;
  Diagnostics results[3];
  const unsigned maskChoices[3] = {1u, 2u, 1u | 2u};
  const char* labels[3] = {"x only ", "t only ", "x and t"};
  for (int choice=0; choice<3; choice++)
  {
    Problem box = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
    auto vlasovMesh = box.form->vlasovMesh();
    std::map<GlobalIndexType,unsigned> masks;
    for (GlobalIndexType cellID : vlasovMesh->getActiveCellIDsGlobal()) masks[cellID] = maskChoices[choice];
    vlasovMesh->hRefine(masks);
    box.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
    box.form->solve();
    results[choice] = diagnostics(box);
    print(out, labels[choice], results[choice]);
  }
  // The configuration mesh is the same in all three; the phase-space resolution is not, and the problem is far from
  // resolved on these meshes, so this is only a check that the coupled solves run and land near each other.
  TEST_EQUALITY(results[0].configCells, results[2].configCells);
  TEST_EQUALITY(results[1].configCells, results[2].configCells);
  TEST_FLOATING_EQUALITY(results[0].fNorm, results[2].fNorm, 0.05);
  TEST_FLOATING_EQUALITY(results[1].fNorm, results[2].fNorm, 0.1);
  TEST_FLOATING_EQUALITY(results[0].ExNorm, results[2].ExNorm, 0.05);
  TEST_FLOATING_EQUALITY(results[1].ExNorm, results[2].ExNorm, 0.25);
}

TEUCHOS_UNIT_TEST( BoxVlasov, SingleDirectionSpaceTimeRefinementMatchesExtrudedMesh )
{
  // Free streaming (zero charge), so that the configuration mesh plays no part: every box cell bisected in x alone
  // must give the solution on the extruded mesh with twice as many x cells, and likewise for t.
  const int xElements = 2, tElements = 1, vxElements = 2, vyElements = 1, H1Order = 2;
  FunctionPtr x = Function::xn(1);
  FunctionPtr ux = MonomialFunction::xi_n(2, 1);
  FunctionPtr f0 = Function::constant(1.0) + 0.3 * x + 0.1 * ux; // not reproduced exactly: a real comparison
  for (int direction=0; direction<2; direction++) // 0: x, 1: t
  {
    Problem extruded = problem(false, (direction == 0) ? 2 * xElements : xElements, vxElements, vyElements, H1Order, f0, 0.0,
                               (direction == 1) ? 2 * tElements : tElements);
    Problem box = problem(true, xElements, vxElements, vyElements, H1Order, f0, 0.0, tElements);
    std::map<GlobalIndexType,unsigned> masks;
    for (GlobalIndexType cellID : box.form->vlasovMesh()->getActiveCellIDsGlobal()) masks[cellID] = 1u << direction;
    box.form->vlasovMesh()->hRefine(masks);
    box.form->setInitialAndInflowConditions(f0, f0);
    extruded.form->solve();
    box.form->solve();
    Diagnostics de = diagnostics(extruded), db = diagnostics(box);
    print(out, (direction == 0) ? "extruded, 2x the x cells" : "extruded, 2x the t cells", de);
    print(out, (direction == 0) ? "box, bisected in x      " : "box, bisected in t      ", db);
    TEST_EQUALITY(de.vlasovCells, db.vlasovCells);
    TEST_EQUALITY(de.vlasovDofs, db.vlasovDofs);
    TEST_FLOATING_EQUALITY(de.fNorm, db.fNorm, 1e-9);
    TEST_FLOATING_EQUALITY(de.fEnergyError + 1.0, db.fEnergyError + 1.0, 1e-8);
  }
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
TEUCHOS_UNIT_TEST( BoxVlasov, NextTimeSlabMatchesOrders )
{
  // The first slab starts at H1 order 1 (piecewise constants) with some cells raised; the second slab's cells at the
  // interface take the orders of the cells across it.  (The interface flux is copied coefficient by coefficient, so the
  // orders must agree; they do when the first slab keeps neighbours within one order, as hp-refinement does, since the
  // second slab's interface cells then have the first slab's tangential neighbour relations.)
  UseBoxMesh useBoxMesh(true);
  VlasovSpaceTimeFormulation::setBoxMeshInitialH1Order(1);
  const int xElements = 2, vxElements = 2, vyElements = 2, H1Order = 3;
  Problem p = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
  MeshPtr mesh = p.form->vlasovMesh();
  auto topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  set<GlobalIndexType> raise;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    if (topology->cellExtents(cellID)[2].first < V0 + 0.5 * VWIDTH) raise.insert(cellID); // the cells with u_x below the middle
  }
  mesh->pRefine(raise, 1); // order 2 on half the cells, 1 on the rest (neighbours within one order, as hp keeps them)
  p.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
  p.form->solve();
  auto secondSlab = p.form->nextTimeSlab(TWIDTH, 1);
  VlasovSpaceTimeFormulation::setBoxMeshInitialH1Order(0);
  MeshPtr nextMesh = secondSlab->vlasovMesh();
  auto nextTopology = dynamic_cast<BoxMeshTopology*>(nextMesh->getTopology().get());
  int interfaceCells = 0;
  for (GlobalIndexType cellID : nextMesh->getActiveCellIDsGlobal())
  {
    const auto extents = nextTopology->cellExtents(cellID);
    const int order = nextMesh->globalDofAssignment()->getH1Order(cellID)[0];
    if (std::abs(extents[1].first - TWIDTH) < 1e-12) // on the interface (the second slab's t-lower side)
    {
      interfaceCells++;
      TEST_EQUALITY(order, (extents[2].first < V0 + 0.5 * VWIDTH) ? 2 : 1);
    }
  }
  TEST_EQUALITY(interfaceCells, int(nextMesh->numActiveElements())); // one t cell per slab
  secondSlab->solve();
  const double fNorm = Function::solution(secondSlab->f(), secondSlab->vlasovSolution())->l2norm(nextMesh);
  out << "second slab: " << nextMesh->numActiveElements() << " cells, " << nextMesh->numGlobalDofs() << " dofs; ||f|| = " << fNorm << endl;
  TEST_ASSERT(std::isfinite(fNorm));
  TEST_COMPARE(fNorm, >, 0.0);
}

TEUCHOS_UNIT_TEST( BoxVlasov, NextTimeSlabMatchesAnisotropicOrders )
{
  // the first slab's cells raised in u_x only (half of them) or in x and u_z: the second slab's interface cells take
  // the same orders, direction by direction
  UseBoxMesh useBoxMesh(true);
  VlasovSpaceTimeFormulation::setBoxMeshInitialH1Order(1);
  const int xElements = 2, vxElements = 2, vyElements = 2, H1Order = 3;
  Problem p = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
  MeshPtr mesh = p.form->vlasovMesh();
  auto topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  std::map<GlobalIndexType, std::vector<int>> increments;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    increments[cellID] = (topology->cellExtents(cellID)[2].first < V0 + 0.5 * VWIDTH) ? std::vector<int>{0, 0, 1, 0} : std::vector<int>{1, 0, 0, 1};
  }
  BoxGDA::pRefineInDirections(mesh, increments);
  mesh->repartitionAndRebuild();
  p.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
  p.form->solve();
  auto secondSlab = p.form->nextTimeSlab(TWIDTH, 1);
  VlasovSpaceTimeFormulation::setBoxMeshInitialH1Order(0);
  MeshPtr nextMesh = secondSlab->vlasovMesh();
  auto nextTopology = dynamic_cast<BoxMeshTopology*>(nextMesh->getTopology().get());
  for (GlobalIndexType cellID : nextMesh->getActiveCellIDsGlobal())
  {
    const std::vector<int> expected = (nextTopology->cellExtents(cellID)[2].first < V0 + 0.5 * VWIDTH) ? std::vector<int>{1, 1, 2, 1} : std::vector<int>{2, 1, 1, 2};
    TEST_ASSERT(nextMesh->globalDofAssignment()->getH1Order(cellID) == expected);
  }
  secondSlab->solve();
  const double fNorm = Function::solution(secondSlab->f(), secondSlab->vlasovSolution())->l2norm(nextMesh);
  out << "second slab: " << nextMesh->numActiveElements() << " cells, " << nextMesh->numGlobalDofs() << " dofs; ||f|| = " << fNorm << endl;
  TEST_ASSERT(std::isfinite(fNorm));
}

TEUCHOS_UNIT_TEST( BoxVlasov, NextTimeSlabWithHigherOrderInterface )
{
  // The second slab's interface cells end up of higher order than the cells across from them (here by a pre-refinement
  // raising them in u_x; in practice by the replayed boundary-data mesh or the order halo): the interface trace is
  // transferred by projection.  The second slab's solution must stay close to the one without the extra order.
  UseBoxMesh useBoxMesh(true);
  const int xElements = 2, vxElements = 2, vyElements = 2, H1Order = 2;
  double fNorms[2];
  for (int raise=0; raise<2; raise++)
  {
    Problem p = problem(true, xElements, vxElements, vyElements, H1Order, drivenDistribution());
    if (raise == 1)
    {
      p.form->setNextSlabPreRefinement([](MeshPtr nextMesh)
      {
        std::map<GlobalIndexType, std::vector<int>> increments;
        for (GlobalIndexType cellID : nextMesh->getActiveCellIDsGlobal()) increments[cellID] = {0, 0, 1, 0};
        BoxGDA::pRefineInDirections(nextMesh, increments);
        nextMesh->repartitionAndRebuild();
      });
    }
    p.form->solve();
    auto secondSlab = p.form->nextTimeSlab(TWIDTH, 1);
    secondSlab->solve();
    fNorms[raise] = Function::solution(secondSlab->f(), secondSlab->vlasovSolution())->l2norm(secondSlab->vlasovMesh());
    out << (raise ? "interface cells raised in u_x: " : "matching orders:              ") << secondSlab->vlasovMesh()->numGlobalDofs()
        << " dofs; second slab ||f|| = " << fNorms[raise] << endl;
  }
  TEST_FLOATING_EQUALITY(fNorms[0], fNorms[1], 0.02);
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
TEUCHOS_UNIT_TEST( BoxVlasov, SerendipityMatchesExtrudedMesh )
{
  // serendipity bases at p = 2 (a proper subspace of the tensor-product space), with one cell refined in all four
  // directions so that there are hanging faces: box against extruded
  const int xElements = 2, vxElements = 2, vyElements = 1, H1Order = 3;
  Diagnostics results[2];
  for (int useBox=0; useBox<2; useBox++)
  {
    Problem p = problem(useBox == 1, xElements, vxElements, vyElements, H1Order, drivenDistribution(), Q, 1, BASIS_FAMILY_SERENDIPITY);
    auto vlasovMesh = p.form->vlasovMesh();
    Camellia::FieldContainer<double> point(1, 4);
    point(0,0) = X0 + 0.1 * XWIDTH; point(0,1) = T0 + 0.1 * TWIDTH; point(0,2) = V0 + 0.1 * VWIDTH; point(0,3) = V0 + 0.1 * VWIDTH;
    vlasovMesh->hRefine(std::set<GlobalIndexType>{GlobalIndexType(vlasovMesh->getTopology()->cellIDsForPoints(point)[0])});
    p.form->setInitialAndInflowConditions(drivenDistribution(), drivenDistribution());
    p.form->solve();
    results[useBox] = diagnostics(p);
    print(out, (useBox == 1) ? "box     " : "extruded", results[useBox]);
  }
  TEST_EQUALITY(results[0].vlasovCells, results[1].vlasovCells);
  TEST_EQUALITY(results[0].vlasovDofs, results[1].vlasovDofs);
  TEST_FLOATING_EQUALITY(results[0].fNorm, results[1].fNorm, 1e-8);
  TEST_FLOATING_EQUALITY(results[0].ExNorm, results[1].ExNorm, 1e-7);
  TEST_FLOATING_EQUALITY(results[0].fEnergyError, results[1].fEnergyError, 1e-6);
}
} // namespace
