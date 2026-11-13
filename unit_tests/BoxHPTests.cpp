#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BasisFamily.h"
#include "BoxDPG.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "GMGSolver.h"
#include "MPIWrapper.h"
#include "PolynomialOrder.hpp"
#include "MeshFactory.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include <algorithm>
#include <cmath>
#include <functional>

using namespace Camellia;
using namespace std;

// Variable polynomial order (p-refinement) on box meshes.

namespace
{
  // Ultraweak DPG for steady convection, beta . grad u = f: L2 field u, L2 flux t_n = (beta . n) u.
  struct Convection
  {
    int dim;
    vector<double> beta;
    VarPtr u, t_n, v;
    BFPtr bf;

    Convection(int dim_, bool serendipity = false) : dim(dim_)
    {
      beta = {1.0, 0.5, -0.75, 0.3};
      beta.resize(dim);
      VarFactoryPtr vf = VarFactory::varFactory();
      t_n = vf->fluxVar("t_n_hat");
      u = vf->fieldVar("u", L2);
      v = vf->testVar("v", HGRAD);
      if (serendipity) vf->setBasisFamily(BasisFamily::serendipityBasisFamily());
      bf = Teuchos::rcp(new BF(vf));
      for (int k=0; k<dim; k++) bf->addTerm(-beta[k] * u, v->di(k+1));
      bf->addTerm(t_n, v);
    }

    RHSPtr rhs(BoxDPG::PointFunction f)
    {
      RHSPtr rhs = RHS::rhs();
      rhs->addTerm(BoxDPG::function(f) * v);
      return rhs;
    }

    BCPtr inflowBC(Teuchos::RCP<BoxMeshTopology> topology, BoxDPG::PointFunction uExact)
    {
      BCPtr bc = BC::bc();
      for (int k=0; k<dim; k++)
      {
        for (int upper=0; upper<2; upper++)
        {
          const double beta_n = beta[k] * (upper ? 1.0 : -1.0);
          if (beta_n >= 0) continue;
          const int tagID = 2 * k + upper;
          topology->applyTag(DIRICHLET_SET_TAG_NAME, tagID,
                             [k, upper](int direction, bool isUpper, const BoxMeshTopology::Extents &) { return (direction == k) && (isUpper == bool(upper)); });
          bc->addDirichlet(t_n, tagID, beta_n * BoxDPG::function(uExact));
        }
      }
      return bc;
    }
  };

  //! a polynomial of total degree p (in each variable separately, at most p)
  BoxDPG::PointFunction polynomial(int dim, int p)
  {
    return [=](const double *x)
    {
      double sum = 1;
      for (int k=0; k<dim; k++) sum += (k+1) * std::pow(x[k], p) + 0.25 * x[k];
      return sum;
    };
  }

  BoxDPG::PointFunction polynomialConvectiveDerivative(int dim, int p, vector<double> beta)
  {
    return [=](const double *x)
    {
      double result = 0;
      for (int k=0; k<dim; k++) result += beta[k] * ((k+1) * p * std::pow(x[k], p-1) + 0.25);
      return result;
    };
  }

  //! a box mesh on [0,1]^dim with roots split at 0.6, refined anisotropically, some cells raised by one or two orders
  MeshPtr variableOrderMesh(Convection &form, int H1Order, Teuchos::RCP<BoxMeshTopology> &topology)
  {
    const int dim = form.dim, delta_k = 2;
    topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 2u}, {1, 1u << (dim-1)}});
    mesh->pRefine(set<GlobalIndexType>{2, 3}, 1); // two cells up one order
    const IndexType child = topology->getCell(0)->getChildIndices(topology)[0];
    mesh->pRefine(set<GlobalIndexType>{GlobalIndexType(child)}, 2); // one child up two
    return mesh;
  }
} // namespace

TEUCHOS_UNIT_TEST( BoxHP, VariableOrderMatchesMeshTopology_2D )
{
  // 2 x 2 quads, p-refined in places and h-refined in one, on a MeshTopology (GDAMinimumRule) and on a box mesh
  const int dim = 2, H1Order = 2, delta_k = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::exp(x[0] + 2 * x[1]) + std::sin(3 * x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * (std::exp(x[0] + 2 * x[1]) + 3 * std::cos(3 * x[0])) + beta[1] * 2 * std::exp(x[0] + 2 * x[1]); };
  BCPtr bc = BC::bc();
  FunctionPtr n = Function::normal();
  bc->addDirichlet(form.t_n, SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0),
                   (beta[0] * n->x() + beta[1] * n->y()) * BoxDPG::function(uExact));
  auto cellAt = [](MeshPtr mesh, double x, double y)
  {
    Camellia::FieldContainer<double> point(1, 2);
    point(0,0) = x;
    point(0,1) = y;
    return GlobalIndexType(mesh->getTopology()->cellIDsForPoints(point)[0]);
  };
  // configurations: p only; p everywhere; h and p, with the p-refined cells away from the hanging sides; and a
  // higher-order coarse cell with lower-order children hanging on its side
  for (int configuration=0; configuration<4; configuration++)
  {
    double errors[2], energyErrors[2];
    GlobalIndexType dofCounts[2];
    for (int useBox=0; useBox<2; useBox++)
    {
      MeshPtr mesh;
      if (useBox == 0) mesh = MeshFactory::quadMesh(form.bf, H1Order, delta_k, 1.0, 1.0, 2, 2);
      else mesh = BoxGDA::mesh(Teuchos::rcp(new BoxMeshTopology({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}})), form.bf, H1Order, delta_k);
      if (configuration == 0)
      {
        mesh->pRefine(set<GlobalIndexType>{cellAt(mesh, 0.75, 0.25)}, 1);
        mesh->pRefine(set<GlobalIndexType>{cellAt(mesh, 0.75, 0.75)}, 2);
      }
      else if (configuration == 1)
      {
        mesh->pRefine(mesh->getActiveCellIDsGlobal(), 1);
      }
      else
      {
        mesh->hRefine(set<GlobalIndexType>{cellAt(mesh, 0.25, 0.25)});
        mesh->pRefine(set<GlobalIndexType>{cellAt(mesh, 0.1, 0.1)}, 1); // a child, away from the hanging sides
        mesh->pRefine(set<GlobalIndexType>{cellAt(mesh, 0.75, 0.75)}, 2); // a coarse cell touching the refined one at a corner only
        if (configuration == 3) mesh->pRefine(set<GlobalIndexType>{cellAt(mesh, 0.75, 0.25)}, 1); // children hang on its left side
      }
      SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
      solution->solve();
      errors[useBox] = (Function::solution(form.u, solution) - BoxDPG::function(uExact))->l2norm(mesh);
      energyErrors[useBox] = solution->energyErrorTotal();
      dofCounts[useBox] = mesh->numGlobalDofs();
      out << "configuration " << configuration << ((useBox == 0) ? ", MeshTopology:    " : ", BoxMeshTopology: ") << mesh->numActiveElements()
          << " cells, " << dofCounts[useBox] << " dofs; L2 error " << errors[useBox] << ", energy error " << energyErrors[useBox] << endl;
    }
    if (configuration < 3)
    {
      TEST_EQUALITY(dofCounts[0], dofCounts[1]);
      TEST_FLOATING_EQUALITY(errors[0], errors[1], 1e-6);
      TEST_FLOATING_EQUALITY(energyErrors[0], energyErrors[1], 1e-6);
    }
    else
    {
      // The side x = 0.5, y < 0.5 has the quadratic coarse cell on one side and two linear children on the other.
      // BoxGDA gives it the lowest order among all its cells, so the flux there is linear and single-valued.  Counted by
      // hand: fields 9 + 4 + 4 + 4 (children) + 4 + 9 + 16 = 50; boundary fluxes 7 + 8 + 7 + 6 = 28; interior fluxes
      // 2 + 3 + 2 + 2 + 4 * 2 = 17; 95 in all.  GDAMinimumRule gives the coarse side the coarse cell's degree, which the
      // linear children's side bases cannot hold: one dof more.
      TEST_EQUALITY(dofCounts[1], 95);
      TEST_EQUALITY(dofCounts[0], dofCounts[1] + 1);
    }
  }
}

TEUCHOS_UNIT_TEST( BoxHP, VariableOrderReproducesPolynomial )
{
  // with the lowest order linear, a linear solution is in every cell's space: it must be reproduced exactly
  for (int dim : {3, 4})
  {
    const int H1Order = 2, p = 1;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    Teuchos::RCP<BoxMeshTopology> topology;
    MeshPtr mesh = variableOrderMesh(form, H1Order, topology);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(polynomialConvectiveDerivative(dim, p, form.beta)), form.bf->graphNorm());
    solution->solve();
    const double error = (Function::solution(form.u, solution) - BoxDPG::function(uExact))->l2norm(mesh);
    out << dim << "D: " << mesh->numActiveElements() << " cells, " << mesh->numGlobalDofs() << " dofs; L2 error " << error
        << ", energy error " << solution->energyErrorTotal() << endl;
    TEST_COMPARE(error, <, 1e-10);
    TEST_COMPARE(solution->energyErrorTotal(), <, 1e-9);
  }
}

TEUCHOS_UNIT_TEST( BoxHP, PRefinementProjectsSolution )
{
  // a solution survives p-refinement, and then h-refinement of a p-refined cell, unchanged; children take the parent's order
  for (int dim : {2, 4})
  {
    const int H1Order = 2, p = 1, delta_k = 2;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(polynomialConvectiveDerivative(dim, p, form.beta)), form.bf->graphNorm());
    mesh->registerSolution(solution);
    solution->solve();
    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);

    const GlobalIndexType dofsBefore = mesh->numGlobalDofs();
    mesh->pRefine(set<GlobalIndexType>{0, 1}, 1);
    TEST_EQUALITY(mesh->globalDofAssignment()->getH1Order(0)[0], H1Order + 1);
    TEST_COMPARE(mesh->numGlobalDofs(), >, dofsBefore);
    const double projectedError = error->l2norm(mesh);
    out << dim << "D: L2 error after p-refinement " << projectedError;
    TEST_COMPARE(projectedError, <, 1e-10);

    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u}});
    for (IndexType child : topology->getCell(0)->getChildIndices(topology))
    {
      TEST_EQUALITY(mesh->globalDofAssignment()->getH1Order(child)[0], H1Order + 1);
    }
    out << ", after h-refinement " << error->l2norm(mesh);
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
    solution->solve();
    out << ", after solving again " << error->l2norm(mesh) << endl;
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
    TEST_COMPARE(solution->energyErrorTotal(), <, 1e-9);
  }
}

TEUCHOS_UNIT_TEST( BoxHP, HigherOrderCellsAreUsed )
{
  // a quadratic solution is not in the linear cells' space, but raising every cell to quadratic reproduces it: the
  // p-refined space is the uniform quadratic one
  const int dim = 3, H1Order = 2, p = 2;
  Convection form(dim);
  auto uExact = polynomial(dim, p);
  Teuchos::RCP<BoxMeshTopology> topology;
  MeshPtr mesh = variableOrderMesh(form, H1Order, topology);
  BCPtr bc = form.inflowBC(topology, uExact);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(polynomialConvectiveDerivative(dim, p, form.beta)), form.bf->graphNorm());
  mesh->registerSolution(solution);
  solution->solve();
  FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
  const double mixedError = error->l2norm(mesh);
  set<GlobalIndexType> linearCells;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    if (mesh->globalDofAssignment()->getH1Order(cellID)[0] == H1Order) linearCells.insert(cellID);
  }
  mesh->pRefine(linearCells, 1);
  solution->solve();
  const double raisedError = error->l2norm(mesh);
  out << "mixed orders: L2 error " << mixedError << "; every cell at least quadratic: " << raisedError << endl;
  TEST_COMPARE(mixedError, >, 1e-4);
  TEST_COMPARE(raisedError, <, 1e-10);
}

TEUCHOS_UNIT_TEST( BoxHP, GeometricMultigridOnVariableOrder )
{
  // multigrid-preconditioned CG on a variable-order mesh: the coarse levels are at the lowest order
  for (int dim : {2, 4})
  {
    const int H1Order = 2, p = 1;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    Teuchos::RCP<BoxMeshTopology> topology;
    MeshPtr mesh = variableOrderMesh(form, H1Order, topology);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(polynomialConvectiveDerivative(dim, p, form.beta)), form.bf->graphNorm());
    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, 200, 1e-10, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
    gmgSolver->setAztecOutput(0);
    solution->solve(gmgSolver);
    const double error = (Function::solution(form.u, solution) - BoxDPG::function(uExact))->l2norm(mesh);
    out << dim << "D: " << meshes.size() << " levels, " << mesh->numGlobalDofs() << " dofs; " << gmgSolver->iterationCount()
        << " CG iterations; L2 error " << error << endl;
    TEST_COMPARE(gmgSolver->iterationCount(), <, 100);
    TEST_COMPARE(error, <, 1e-7);
  }
}

namespace
{
  //! the largest difference of order between face neighbors
  int maxNeighborOrderJump(MeshPtr mesh, BoxMeshTopology *topology)
  {
    int maxJump = 0;
    const int sideCount = topology->cellTopology()->getSideCount();
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
    {
      const int order = mesh->globalDofAssignment()->getH1Order(cellID)[0];
      for (int side=0; side<sideCount; side++)
        for (IndexType neighborID : topology->sideNeighbors(cellID, side))
          maxJump = std::max(maxJump, std::abs(order - mesh->globalDofAssignment()->getH1Order(neighborID)[0]));
    }
    return maxJump;
  }

  std::map<GlobalIndexType,unsigned> allDirections(MeshPtr mesh, const set<GlobalIndexType> &cellIDs)
  {
    std::map<GlobalIndexType,unsigned> masks;
    for (GlobalIndexType cellID : cellIDs) masks[cellID] = (1u << mesh->getDimension()) - 1;
    return masks;
  }
}

TEUCHOS_UNIT_TEST( BoxHP, HPRefinementsPreferHUntilMinimumWidth )
{
  // roots of widths 0.6 and 0.4; minimum widths 0.2 in x and 0.3 in y: the 0.6 cells may be bisected once in x and in
  // y, the 0.4 cells once in x only; after that, marked cells are raised in order instead, up to maxH1Order
  Convection form(2);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(2, vector<double>{0.0, 0.6, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
  const vector<double> minWidths = {0.2, 0.3};
  const int maxH1Order = 3;

  auto refinements = BoxGDA::hpRefinements(mesh, allDirections(mesh, mesh->getActiveCellIDsGlobal()), minWidths, maxH1Order);
  TEST_EQUALITY(refinements.hMasks.size(), 4);
  TEST_EQUALITY(refinements.pIncrements.size(), 0);
  for (const auto &[cellID, mask] : refinements.hMasks)
  {
    const auto extents = topology->cellExtents(cellID);
    const bool wideInY = (extents[1].second - extents[1].first) > 0.5;
    TEST_EQUALITY(mask, wideInY ? 3u : 1u);
  }
  BoxGDA::hpRefine(mesh, refinements);
  TEST_EQUALITY(mesh->numActiveElements(), 2 * 4 + 2 * 2 + 0); // two cells wide in y (4 children), two narrow (2)

  // every cell is now at its minimum width in every direction: marking raises the order, once per round, to the maximum
  for (int round=0; round<3; round++)
  {
    refinements = BoxGDA::hpRefinements(mesh, allDirections(mesh, mesh->getActiveCellIDsGlobal()), minWidths, maxH1Order);
    TEST_EQUALITY(refinements.hMasks.size(), 0);
    TEST_EQUALITY(refinements.pIncrements.size(), (round < 2) ? mesh->numActiveElements() : 0);
    BoxGDA::hpRefine(mesh, refinements);
  }
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) TEST_EQUALITY(mesh->globalDofAssignment()->getH1Order(cellID)[0], maxH1Order);
}

TEUCHOS_UNIT_TEST( BoxHP, OrderHaloLimitsJumpsToOne )
{
  // raising one cell repeatedly raises its neighbors, and theirs, so that face neighbors differ by at most one order
  for (int dim : {2, 4})
  {
    Convection form(dim);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.25, 0.5, 0.75, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
    // some anisotropic refinement first, so that neighbors include hanging ones
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u}, {5, 2u}});
    const vector<double> minWidths(dim, 1.0); // no h-refinement
    const GlobalIndexType corner = *mesh->getActiveCellIDsGlobal().begin();
    for (int round=0; round<3; round++)
    {
      auto refinements = BoxGDA::hpRefinements(mesh, {{corner, (1u << dim) - 1}}, minWidths, 10);
      TEST_EQUALITY(refinements.hMasks.size(), 0);
      BoxGDA::hpRefine(mesh, refinements);
    }
    TEST_EQUALITY(mesh->globalDofAssignment()->getH1Order(corner)[0], 4);
    TEST_EQUALITY(maxNeighborOrderJump(mesh, topology.get()), 1);
    int raisedCells = 0;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) if (mesh->globalDofAssignment()->getH1Order(cellID)[0] > 1) raisedCells++;
    out << dim << "D: " << raisedCells << " of " << mesh->numActiveElements() << " cells raised; max neighbor jump "
        << maxNeighborOrderJump(mesh, topology.get()) << endl;
    TEST_COMPARE(raisedCells, >, 1 + (1 << dim) - 1); // more than the corner and its face neighbors: the halo is transitive
  }
}

TEUCHOS_UNIT_TEST( BoxHP, PiecewiseConstantDirectionIndicator )
{
  // on p = 0 cells the direction indicator uses jumps of cell means: a function of x alone is refined in x alone
  const int dim = 3;
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.25, 0.5, 0.75, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
  FunctionPtr f = BoxDPG::function([](const double *x) { return x[0] * x[0] + 0.01 * x[2]; });
  const set<GlobalIndexType> activeCells = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cells(activeCells.begin(), activeCells.end());
  auto masks = BoxGDA::anisotropicRefinementMasks(mesh, f, cells, 0.5);
  TEST_EQUALITY(masks.size(), cells.size());
  for (const auto &[cellID, mask] : masks) TEST_EQUALITY(mask, 1u);
}

TEUCHOS_UNIT_TEST( BoxHP, PiecewiseConstantDirectionIndicatorSeesBoundaryData )
{
  // One cell across the domain in t (as in a time slab): on p = 0 cells there is no jump in t between cells, only
  // against the data at the initial time.  Without that data f, which varies only a little in x, is refined in x;
  // with data at t = 0 far from f's means, in t.
  const int dim = 2; // (x, t)
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>{{0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
  FunctionPtr f = BoxDPG::function([](const double *x) { return 1.0 + 0.1 * x[0]; });
  const set<GlobalIndexType> activeCells = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cells(activeCells.begin(), activeCells.end());
  const int timeDirection = 1;
  auto withoutData = BoxGDA::anisotropicRefinementMasks(mesh, f, cells, 0.5);
  auto withData = BoxGDA::anisotropicRefinementMasks(mesh, f, cells, 0.5, ~0u, {{2 * timeDirection, Function::constant(3.0)}});
  // data equal to f on the face: no jump there, so as without data
  auto withMatchingData = BoxGDA::anisotropicRefinementMasks(mesh, f, cells, 0.5, ~0u, {{2 * timeDirection, f}});
  TEST_EQUALITY(withoutData.size(), cells.size());
  for (GlobalIndexType cellID : cells)
  {
    TEST_EQUALITY(withoutData[cellID], 1u);
    TEST_EQUALITY(withData[cellID], 2u);
    TEST_EQUALITY(withMatchingData[cellID], 1u);
  }
}

TEUCHOS_UNIT_TEST( BoxHP, SolveAfterHPRefinement )
{
  // piecewise constants to start; one round of hp-refinement from the solution's own indicator, then solve by
  // multigrid: a constant is reproduced exactly at every stage
  for (int dim : {2, 4})
  {
    Convection form(dim);
    auto uExact = [](const double *) { return 1.5; };
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.5, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 2);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs([](const double *) { return 0.0; }), form.bf->graphNorm());
    mesh->registerSolution(solution);
    solution->solve();
    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);

    const vector<double> minWidths(dim, 0.25); // the roots may be bisected once in each direction
    for (int round=0; round<2; round++)
    {
      const set<GlobalIndexType> activeCells = mesh->getActiveCellIDsGlobal();
      vector<GlobalIndexType> marked(activeCells.begin(), activeCells.end());
      marked.resize(std::max<size_t>(1, marked.size() / 2));
      auto masks = BoxGDA::anisotropicRefinementMasks(mesh, Function::solution(form.u, solution), marked, 0.5);
      BoxGDA::hpRefine(mesh, BoxGDA::hpRefinements(mesh, masks, minWidths, 3));
      TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
    }
    TEST_COMPARE(maxNeighborOrderJump(mesh, topology.get()), <=, 1);
    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, 200, 1e-10, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
    gmgSolver->setAztecOutput(0);
    solution->solve(gmgSolver);
    int maxOrder = 0;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) maxOrder = std::max(maxOrder, mesh->globalDofAssignment()->getH1Order(cellID)[0]);
    out << dim << "D: " << mesh->numActiveElements() << " cells, " << mesh->numGlobalDofs() << " dofs, max H1 order " << maxOrder << "; "
        << gmgSolver->iterationCount() << " CG iterations; L2 error " << error->l2norm(mesh) << endl;
    TEST_COMPARE(error->l2norm(mesh), <, 1e-8);
  }
}

TEUCHOS_UNIT_TEST( BoxHP, SaturatedCellsAreNotMarked )
{
  // cells at the minimum width in their chosen directions and at the maximum order are reported saturated, and a
  // refinement strategy told so marks the worst of the others instead
  const int dim = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::sin(4 * x[0]) * std::exp(x[1]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * 4 * std::cos(4 * x[0]) * std::exp(x[1]) + beta[1] * std::sin(4 * x[0]) * std::exp(x[1]); };
  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 0.5, 1.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 2, 1);
  BCPtr bc = form.inflowBC(topology, uExact);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
  solution->solve();

  // x widths 0.25, y widths 0.5; minimum widths 0.2: no cell may be bisected in x, every cell in y
  const vector<double> minWidths = {0.2, 0.2};
  const int maxH1Order = 2;
  std::map<GlobalIndexType,unsigned> xOnly, yOnly;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) { xOnly[cellID] = 1u; yOnly[cellID] = 2u; }
  TEST_EQUALITY(BoxGDA::hpSaturatedCells(mesh, xOnly, minWidths, maxH1Order).size(), xOnly.size());
  TEST_EQUALITY(BoxGDA::hpSaturatedCells(mesh, yOnly, minWidths, maxH1Order).size(), 0);
  TEST_EQUALITY(BoxGDA::hpSaturatedCells(mesh, xOnly, minWidths, maxH1Order + 1).size(), 0); // may still be raised

  // a greedy strategy marking cells within 1% of the largest error, told that all but one cell are saturated, marks
  // that one: the threshold is relative to the largest error among the cells that may be refined
  const GlobalIndexType onlyEligible = *mesh->getActiveCellIDsGlobal().rbegin();
  auto refStrategy = Teuchos::rcp(new RefinementStrategy(solution, 0.99));
  std::vector<GlobalIndexType> marked;
  refStrategy->setCellRefiner([&marked](std::vector<GlobalIndexType> &cellIDs) { marked = cellIDs; return 0; });
  refStrategy->setIneligibleCells([&]() {
    std::set<GlobalIndexType> ineligible = mesh->getActiveCellIDsGlobal();
    ineligible.erase(onlyEligible);
    return ineligible;
  });
  refStrategy->refine(false);
  TEST_EQUALITY(marked.size(), 1);
  if (marked.size() == 1) TEST_EQUALITY(marked[0], onlyEligible);
}


TEUCHOS_UNIT_TEST( BoxHP, RefineToContainReproducesMesh )
{
  // a mesh refined anisotropically and raised in places (neighbors within one order, as hp keeps them), recorded and
  // replayed on a fresh mesh with the same roots, comes out the same: the same boxes, each with the same order
  for (int dim : {2, 4})
  {
    Convection form(dim);
    auto makeTopology = [dim]() { return Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.3, 1.0}))); };
    auto sourceTopology = makeTopology();
    MeshPtr source = BoxGDA::mesh(sourceTopology, form.bf, 1, 1);
    source->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u}, {1, 2u | (1u << (dim-1))}});
    const IndexType child = sourceTopology->getCell(0)->getChildIndices(sourceTopology)[1];
    source->hRefine(std::map<GlobalIndexType,unsigned>{{GlobalIndexType(child), 1u}});
    source->pRefine(set<GlobalIndexType>{GlobalIndexType(sourceTopology->getCell(child)->getChildIndices(sourceTopology)[0])}, 1);
    source->pRefine(set<GlobalIndexType>{2}, 1);
    const vector<BoxGDA::CellRecord> records = BoxGDA::cellRecords(source);

    auto targetTopology = makeTopology();
    MeshPtr target = BoxGDA::mesh(targetTopology, form.bf, 1, 1);
    BoxGDA::refineToContain(target, records);
    TEST_EQUALITY(target->numActiveElements(), source->numActiveElements());
    TEST_EQUALITY(target->numGlobalDofs(), source->numGlobalDofs());
    auto key = [dim](const BoxForest::Box &box)
    {
      vector<BoxForest::Coord> k;
      for (int d=0; d<dim; d++) { k.push_back(box.lo[d]); k.push_back(box.level[d]); }
      return k;
    };
    std::map<vector<BoxForest::Coord>,std::vector<int>> sourceCells;
    for (const auto &record : records) sourceCells[key(record.box)] = record.H1Orders;
    int matched = 0;
    for (const auto &record : BoxGDA::cellRecords(target))
    {
      auto entry = sourceCells.find(key(record.box));
      TEST_ASSERT(entry != sourceCells.end());
      if (entry == sourceCells.end()) continue;
      TEST_ASSERT(record.H1Orders == entry->second);
      if (record.H1Orders == entry->second) matched++;
    }
    out << dim << "D: " << target->numActiveElements() << " cells; " << matched << " with the recorded order" << endl;
    TEST_EQUALITY(matched, int(records.size()));
    // replaying again changes nothing
    const GlobalIndexType dofs = target->numGlobalDofs();
    BoxGDA::refineToContain(target, records);
    TEST_EQUALITY(target->numGlobalDofs(), dofs);
  }
}

namespace
{
  //! 1 + x^2 + 0.25 (y + z + w): quadratic in x, linear in the other directions
  double quadraticInX(const double *x)
  {
    return 1 + x[0] * x[0] + 0.25 * (x[1] + x[2] + x[3]);
  }
  BoxDPG::PointFunction quadraticInXDerivative(vector<double> beta)
  {
    return [beta](const double *x) { return beta[0] * 2 * x[0] + 0.25 * (beta[1] + beta[2] + beta[3]); };
  }
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicOrdersReproducePolynomial )
{
  // 4D, linear to start.  Raising every cell's order in x alone makes 1 + x^2 + ... exact; raising some cells further in
  // other directions (giving patches with mixed spaces) keeps it exact.  Raising in x alone costs fewer dofs than
  // raising isotropically.
  const int dim = 4, H1Order = 2, delta_k = 2;
  Convection form(dim);
  Teuchos::RCP<BoxMeshTopology> topology;
  MeshPtr mesh = variableOrderMesh(form, H1Order, topology); // anisotropic h-refinement and some isotropic p-refinement
  BCPtr bc = form.inflowBC(topology, quadraticInX);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(quadraticInXDerivative(form.beta)), form.bf->graphNorm());
  mesh->registerSolution(solution);
  FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(quadraticInX);

  std::map<GlobalIndexType, std::vector<int>> xOnly, someMore;
  int count = 0;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    // cells already at order 3 or more are quadratic in x already
    if (mesh->globalDofAssignment()->getH1Order(cellID)[0] < 3) xOnly[cellID] = {1, 0, 0, 0};
    if (count++ % 3 == 0) someMore[cellID] = {0, 1, 0, 2};
  }
  BoxGDA::pRefineInDirections(mesh, xOnly);
  mesh->repartitionAndRebuild();
  const GlobalIndexType xOnlyDofs = mesh->numGlobalDofs();
  solution->solve();
  const double xOnlyError = error->l2norm(mesh);
  BoxGDA::pRefineInDirections(mesh, someMore);
  mesh->repartitionAndRebuild();
  TEST_COMPARE(error->l2norm(mesh), <, 1e-10); // the projection onto the raised bases is exact
  solution->solve();
  const double mixedError = error->l2norm(mesh);
  out << "raised in x: " << xOnlyDofs << " dofs, L2 error " << xOnlyError << "; some cells raised further in t and u_z: "
      << mesh->numGlobalDofs() << " dofs, L2 error " << mixedError << endl;
  TEST_COMPARE(xOnlyError, <, 1e-10);
  TEST_COMPARE(mixedError, <, 1e-10);

  // the same cells raised isotropically instead
  Teuchos::RCP<BoxMeshTopology> isotropicTopology;
  MeshPtr isotropicMesh = variableOrderMesh(form, H1Order, isotropicTopology);
  set<GlobalIndexType> raise;
  for (const auto &entry : xOnly) raise.insert(entry.first);
  isotropicMesh->pRefine(raise, 1);
  out << "raised isotropically instead: " << isotropicMesh->numGlobalDofs() << " dofs" << endl;
  TEST_COMPARE(xOnlyDofs, <, isotropicMesh->numGlobalDofs());
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicOrdersEqualInEveryDirectionAreIsotropic )
{
  // raising a cell by the same amount in every direction gives it the element type of an isotropic p-refinement
  const int dim = 4;
  Convection form(dim);
  Teuchos::RCP<BoxMeshTopology> topologyA, topologyB;
  MeshPtr a = variableOrderMesh(form, 2, topologyA), b = variableOrderMesh(form, 2, topologyB);
  BoxGDA::pRefineInDirections(a, {{4, {1, 1, 1, 1}}});
  a->repartitionAndRebuild();
  b->pRefine(set<GlobalIndexType>{4}, 1);
  TEST_EQUALITY(a->numGlobalDofs(), b->numGlobalDofs());
  TEST_ASSERT(a->getElementType(4).get() == b->getElementType(4).get() || *a->getElementType(4) == *b->getElementType(4));
  TEST_EQUALITY(a->globalDofAssignment()->getH1Order(4).size(), b->globalDofAssignment()->getH1Order(4).size());
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicOrdersSurviveRefinement )
{
  // a solution survives anisotropic p-refinement and then h-refinement of the raised cell; children take its orders;
  // multigrid then solves on the mixed mesh
  const int dim = 4;
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 2, 2);
  BCPtr bc = form.inflowBC(topology, quadraticInX);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(quadraticInXDerivative(form.beta)), form.bf->graphNorm());
  mesh->registerSolution(solution);
  FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(quadraticInX);
  std::map<GlobalIndexType, std::vector<int>> xOnly;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) xOnly[cellID] = {1, 0, 0, 0};
  BoxGDA::pRefineInDirections(mesh, xOnly);
  mesh->repartitionAndRebuild();
  solution->solve();
  TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
  BoxGDA::pRefineInDirections(mesh, {{0, {0, 0, 1, 0}}});
  mesh->repartitionAndRebuild();
  TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
  const std::vector<int> raised = mesh->globalDofAssignment()->getH1Order(0);
  TEST_EQUALITY(raised.size(), 4);
  if (raised.size() == 4) { TEST_EQUALITY(raised[0], 3); TEST_EQUALITY(raised[1], 2); TEST_EQUALITY(raised[2], 3); TEST_EQUALITY(raised[3], 2); }
  mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u | 4u}});
  TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
  for (IndexType child : topology->getCell(0)->getChildIndices(topology))
  {
    TEST_ASSERT(mesh->globalDofAssignment()->getH1Order(child) == raised);
  }
  vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
  Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, 200, 1e-10, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
  gmgSolver->setAztecOutput(0);
  solution->solve(gmgSolver);
  out << mesh->numActiveElements() << " cells, " << mesh->numGlobalDofs() << " dofs; " << gmgSolver->iterationCount()
      << " CG iterations; L2 error " << error->l2norm(mesh) << endl;
  TEST_COMPARE(gmgSolver->iterationCount(), <, 100);
  TEST_COMPARE(error->l2norm(mesh), <, 1e-7);
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicOrdersWithSerendipityBasis )
{
  // as AnisotropicOrdersReproducePolynomial, with serendipity trial and test spaces (as the MITL driver uses): the
  // anisotropic serendipity spaces Intrepid2 builds are downward closed, so the patch spaces are found and exact
  const int dim = 4, H1Order = 2, delta_k = 2;
  Convection form(dim, true);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
  mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 2u}, {1, 8u}});
  BCPtr bc = form.inflowBC(topology, quadraticInX);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(quadraticInXDerivative(form.beta)), form.bf->graphNorm());
  std::map<GlobalIndexType, std::vector<int>> increments;
  int count = 0;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    increments[cellID] = (count++ % 2 == 0) ? std::vector<int>{1, 0, 0, 0} : std::vector<int>{2, 0, 2, 1};
  }
  BoxGDA::pRefineInDirections(mesh, increments);
  mesh->repartitionAndRebuild();
  solution->solve();
  const double error = (Function::solution(form.u, solution) - BoxDPG::function(quadraticInX))->l2norm(mesh);
  out << mesh->numActiveElements() << " cells, " << mesh->numGlobalDofs() << " dofs; L2 error " << error << endl;
  TEST_COMPARE(error, <, 1e-10);
}


namespace
{
  //! the largest difference of order, in any one direction, between face neighbors
  int maxNeighborOrderJumpInDirections(MeshPtr mesh, BoxMeshTopology *topology)
  {
    const int dim = topology->getDimension();
    auto orders = [&](GlobalIndexType cellID)
    {
      std::vector<int> order = mesh->globalDofAssignment()->getH1Order(cellID);
      if (int(order.size()) == 1) order = std::vector<int>(dim, order[0]); // (not assign(): its value argument would refer into the vector being reassigned)
      return order;
    };
    int maxJump = 0;
    const int sideCount = topology->cellTopology()->getSideCount();
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
    {
      const std::vector<int> mine = orders(cellID);
      for (int side=0; side<sideCount; side++)
        for (IndexType neighborID : topology->sideNeighbors(cellID, side))
        {
          const std::vector<int> theirs = orders(neighborID);
          for (int k=0; k<dim; k++) maxJump = std::max(maxJump, std::abs(mine[k] - theirs[k]));
        }
    }
    return maxJump;
  }
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicHPRefinementsChooseByDirection )
{
  // 4D roots of width 0.25 and 0.75 in every direction; minimum widths 0.2 in x and t, 1 in u_x and u_z.  A cell marked
  // in x and u_x is bisected in x (where wide enough) and raised in u_x (where it may not be bisected); a cell marked
  // in u_z alone is raised in u_z alone.  The halo keeps each direction's orders within one.
  const int dim = 4;
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.25, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
  const vector<double> minWidths = {0.2, 0.2, 1.0, 1.0};
  const int maxH1Order = 4;
  // root 15 is the wide one in every direction (0.75); root 0 the narrow one (0.25)
  std::map<GlobalIndexType,unsigned> masks = {{15, 1u | 4u}, {0, 8u}};
  auto refinements = BoxGDA::hpRefinements(mesh, masks, minWidths, maxH1Order, true);
  TEST_EQUALITY(refinements.pIncrements.size(), 0);
  TEST_EQUALITY(refinements.hMasks.size(), 1);
  TEST_EQUALITY(refinements.hMasks[15], 1u);
  TEST_ASSERT(refinements.pIncrementsInDirections[15] == (std::vector<int>{0, 0, 1, 0}));
  TEST_ASSERT(refinements.pIncrementsInDirections[0] == (std::vector<int>{0, 0, 0, 1}));
  BoxGDA::hpRefine(mesh, refinements);
  for (IndexType child : topology->getCell(15)->getChildIndices(topology))
  {
    TEST_ASSERT(mesh->globalDofAssignment()->getH1Order(child) == (std::vector<int>{1, 1, 2, 1}));
  }
  TEST_ASSERT(mesh->globalDofAssignment()->getH1Order(0) == (std::vector<int>{1, 1, 1, 2}));

  // raising root 0 in u_z three more times spreads, in u_z only, to its neighbors
  for (int round=0; round<3; round++) BoxGDA::hpRefine(mesh, BoxGDA::hpRefinements(mesh, {{0, 8u}}, minWidths, 10, true));
  TEST_ASSERT(mesh->globalDofAssignment()->getH1Order(0) == (std::vector<int>{1, 1, 1, 5}));
  TEST_EQUALITY(maxNeighborOrderJumpInDirections(mesh, topology.get()), 1);
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
  {
    std::vector<int> order = mesh->globalDofAssignment()->getH1Order(cellID);
    if (order.size() == 1) continue; // never raised
    TEST_EQUALITY(order[0], 1); // never raised in x or t
    TEST_EQUALITY(order[1], 1);
  }
  // saturation is per direction: root 0 at the maximum in u_z is saturated for u_z, not for t
  TEST_EQUALITY(BoxGDA::hpSaturatedCells(mesh, {{0, 8u}}, minWidths, 5, true).size(), 1);
  TEST_EQUALITY(BoxGDA::hpSaturatedCells(mesh, {{0, 8u | 2u}}, {0.2, 1.0, 1.0, 1.0}, 5, true).size(), 0);
}

TEUCHOS_UNIT_TEST( BoxHP, DirectionIndicatorWithMixedOrders )
{
  // a cell linear in x and constant in the other directions (H1 orders 2,1,1,1): f = x^3 has high modes in x only, and
  // f = (t - 0.5)^2 has nothing in x and jumps in t: each picks its own direction
  const int dim = 4;
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.25, 0.5, 0.75, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, 1);
  std::map<GlobalIndexType, std::vector<int>> xOnly;
  for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) xOnly[cellID] = {1, 0, 0, 0};
  BoxGDA::pRefineInDirections(mesh, xOnly);
  mesh->repartitionAndRebuild();
  const set<GlobalIndexType> activeCells = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cells(activeCells.begin(), activeCells.end());
  auto masksX = BoxGDA::anisotropicRefinementMasks(mesh, BoxDPG::function([](const double *x) { return x[0] * x[0] * x[0]; }), cells, 0.5);
  auto masksT = BoxGDA::anisotropicRefinementMasks(mesh, BoxDPG::function([](const double *x) { return (x[1] - 0.5) * (x[1] - 0.5) + 2 * x[1]; }), cells, 0.5);
  int xChosen = 0, tChosen = 0;
  for (const auto &[cellID, mask] : masksX) if (mask == 1u) xChosen++;
  for (const auto &[cellID, mask] : masksT) if (mask == 2u) tChosen++;
  out << "x^3: " << xChosen << " of " << cells.size() << " cells choose x alone; t^2: " << tChosen << " choose t alone" << endl;
  TEST_EQUALITY(xChosen, int(cells.size()));
  TEST_EQUALITY(tChosen, int(cells.size()));
}

TEUCHOS_UNIT_TEST( BoxHP, RefineToContainReproducesAnisotropicOrders )
{
  // records carry orders per direction: replaying a mesh with anisotropic orders reproduces them
  const int dim = 4;
  Convection form(dim);
  auto makeTopology = [dim]() { return Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.3, 1.0}))); };
  auto sourceTopology = makeTopology();
  MeshPtr source = BoxGDA::mesh(sourceTopology, form.bf, 1, 1);
  source->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u}, {3, 4u}});
  BoxGDA::pRefineInDirections(source, {{2, {0, 1, 0, 0}}, {5, {1, 0, 1, 1}}});
  source->repartitionAndRebuild();
  const vector<BoxGDA::CellRecord> records = BoxGDA::cellRecords(source);
  MeshPtr target = BoxGDA::mesh(makeTopology(), form.bf, 1, 1);
  BoxGDA::refineToContain(target, records);
  TEST_EQUALITY(target->numActiveElements(), source->numActiveElements());
  TEST_EQUALITY(target->numGlobalDofs(), source->numGlobalDofs());
}

TEUCHOS_UNIT_TEST( BoxHP, TransferSideCoefficientsRoundTrip )
{
  // a flux trace moved to a higher-order cell (as across a slab interface) and back is unchanged: the first transfer
  // is exact, and the second projects back onto the space it came from
  const int dim = 4;
  for (bool serendipity : {false, true})
  {
    Convection form(dim, serendipity);
    auto makeMesh = [&](Teuchos::RCP<BoxMeshTopology> &topology)
    {
      topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.4, 1.0})));
      return BoxGDA::mesh(topology, form.bf, 2, 1);
    };
    Teuchos::RCP<BoxMeshTopology> fromTopology, toTopology;
    MeshPtr from = makeMesh(fromTopology), to = makeMesh(toTopology);
    BoxGDA::pRefineInDirections(to, {{3, {1, 0, 2, 1}}}); // higher in x, u_x, u_z (tangential to t-sides)
    to->repartitionAndRebuild();
    const int cellID = 3, side = fromTopology->sideOrdinalForFace(2 * 1 + 1), varID = form.t_n->ID();
    const int n = from->getElementType(cellID)->trialOrderPtr->getBasisCardinality(varID, side);
    FieldContainer<double> coefficients(n);
    for (int i=0; i<n; i++) coefficients[i] = std::sin(1.0 + 3.7 * i);
    FieldContainer<double> there = BoxGDA::transferSideCoefficients(from, cellID, side, to, cellID, side, varID, varID, coefficients);
    TEST_COMPARE(int(there.size()), >, n);
    FieldContainer<double> back = BoxGDA::transferSideCoefficients(to, cellID, side, from, cellID, side, varID, varID, there);
    double maxDifference = 0;
    for (int i=0; i<n; i++) maxDifference = std::max(maxDifference, std::abs(back[i] - coefficients[i]));
    out << (serendipity ? "serendipity" : "hierarchical") << ": " << n << " -> " << there.size() << " coefficients; round trip max difference " << maxDifference << endl;
    TEST_COMPARE(maxDifference, <, 1e-11);
  }
}

TEUCHOS_UNIT_TEST( BoxHP, AnisotropicSerendipityTracesAreSeenByTheTestSpace )
{
  // With a basis family (here serendipity) and a flux order enhancement, as the Vlasov formulation uses, each side's
  // trace basis must take the orders of the side's own directions.  It once took the cell's whole order vector, which
  // on a Line^4 cell gave the sides normal to u_x and u_z the orders of the sides normal to t: a trace function of too
  // high a degree for the test space, a zero column in B, and a singular local stiffness matrix.
  const int dim = 4, delta_k = 2;
  Convection form(dim, true);
  const std::map<int,int> trialEnhancements = {{form.t_n->ID(), 1}};
  const int sideCount = 2 * dim;
  for (const std::vector<int> &orders : std::vector<std::vector<int>>{{2,1,4,3}, {1,1,1,4}, {4,1,1,1}, {1,4,2,1}, {3,3,3,3}})
  {
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 1, delta_k, Teuchos::null, MPIWrapper::TeuchosCommWorld(), trialEnhancements);
    std::vector<int> increments(dim);
    for (int k=0; k<dim; k++) increments[k] = orders[k] - 1;
    BoxGDA::pRefineInDirections(mesh, {{0, increments}});
    std::vector<int> cellOrders = mesh->globalDofAssignment()->getH1Order(0); // (equal orders fold into one isotropic order)
    if (int(cellOrders.size()) == 1) cellOrders = std::vector<int>(dim, cellOrders[0]);
    TEST_ASSERT(cellOrders == orders);
    ElementTypePtr elemType = mesh->getElementType(0);
    // each side's trace cardinality, against a basis built for that side's own orders (flux enhancement included)
    std::vector<int> enhanced = orders;
    for (int &p : enhanced) p += 1;
    CellTopoPtr cellTopo = topology->cellTopology();
    for (int side=0; side<sideCount; side++)
    {
      std::vector<int> sideOrders = PolynomialOrder::sidePolynomialOrder(enhanced, cellTopo, side);
      for (int &p : sideOrders) p = std::max(p, 1);
      bool isotropic = std::adjacent_find(enhanced.begin(), enhanced.end(), std::not_equal_to<int>()) == enhanced.end();
      BasisPtr expected = form.t_n->getBasis(cellTopo, isotropic ? std::vector<int>{enhanced[0]} : enhanced, side);
      TEST_EQUALITY(elemType->trialOrderPtr->getBasisCardinality(form.t_n->ID(), side), expected->getCardinality());
      BasisPtr direct = BasisFamily::serendipityBasisFamily()->getBasis(cellTopo->getSide(side), std::vector<Camellia::EFunctionSpace>{efsForSpace(form.t_n->space())}, sideOrders);
      TEST_EQUALITY(expected->getCardinality(), direct->getCardinality());
    }
    // no trial basis function is invisible to the test space
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0, true);
    const int trialCount = elemType->trialOrderPtr->totalDofs(), testCount = elemType->testOrderPtr->totalDofs();
    Camellia::FieldContainer<double> B(1, testCount, trialCount);
    form.bf->stiffnessMatrix(B, elemType, basisCache->getCellSideParities(), basisCache);
    int zeroColumns = 0;
    for (int j=0; j<trialCount; j++)
    {
      double maxEntry = 0;
      for (int i=0; i<testCount; i++) maxEntry = std::max(maxEntry, std::abs(B(0,i,j)));
      if (maxEntry < 1e-12) zeroColumns++;
    }
    out << "orders " << orders[0] << orders[1] << orders[2] << orders[3] << ": " << trialCount << " trial dofs, " << zeroColumns << " zero columns in B" << endl;
    TEST_EQUALITY(zeroColumns, 0);
  }
}
