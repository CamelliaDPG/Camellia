#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BoxDPG.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include <cmath>

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

    Convection(int dim_) : dim(dim_)
    {
      beta = {1.0, 0.5, -0.75, 0.3};
      beta.resize(dim);
      VarFactoryPtr vf = VarFactory::varFactory();
      t_n = vf->fluxVar("t_n_hat");
      u = vf->fieldVar("u", L2);
      v = vf->testVar("v", HGRAD);
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
