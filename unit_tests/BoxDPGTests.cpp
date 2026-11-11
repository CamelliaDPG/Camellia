#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BoxDPG.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include <cmath>

using namespace Camellia;
using namespace std;

namespace
{
  // Ultraweak DPG for steady convection, beta . grad u = f, in any dimension: L2 field u, L2 flux t_n = (beta . n) u.
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

    Teuchos::RCP<BoxDPG> discretization(const vector<vector<double>> &rootVertices, int H1Order, BoxDPG::PointFunction uExact, BoxDPG::PointFunction f)
    {
      const int delta_k = 2;
      auto dpg = Teuchos::rcp(new BoxDPG(rootVertices, bf, H1Order, delta_k));
      RHSPtr rhs = RHS::rhs();
      rhs->addTerm(BoxDPG::function(f) * v);
      dpg->setRHS(rhs);
      vector<double> betaCopy = beta;
      auto outwardBeta = [betaCopy](int direction, bool upper) { return betaCopy[direction] * (upper ? 1.0 : -1.0); };
      dpg->addFluxBC(t_n,
                     [outwardBeta](int direction, bool upper) { return outwardBeta(direction, upper) < 0; }, // inflow
                     [outwardBeta, uExact](const double *x, int direction, bool upper) { return outwardBeta(direction, upper) * uExact(x); });
      return dpg;
    }
  };

  // a polynomial of degree p in each coordinate, and its convective derivative
  BoxDPG::PointFunction polynomial(int dim, int p)
  {
    return [=](const double *x)
    {
      double sum = 1, product = 1;
      for (int k=0; k<dim; k++)
      {
        sum += (k+1) * std::pow(x[k], p);
        product *= (1 + 0.5 * x[k]);
      }
      return sum + product;
    };
  }

  BoxDPG::PointFunction polynomialConvectiveDerivative(int dim, int p, vector<double> beta)
  {
    return [=](const double *x)
    {
      double result = 0;
      for (int k=0; k<dim; k++)
      {
        double dProduct = 0.5;
        for (int j=0; j<dim; j++) if (j != k) dProduct *= (1 + 0.5 * x[j]);
        result += beta[k] * ((k+1) * p * std::pow(x[k], p-1) + dProduct);
      }
      return result;
    };
  }

  int crossingPatchCount(const BoxForest::Skeleton &skeleton)
  {
    // patches that are no single cell's face: at least two faces on each side
    int count = 0;
    for (const auto &patch : skeleton.patches)
    {
      if ((patch.minusFaces.size() >= 2) && (patch.plusFaces.size() >= 2)) count++;
    }
    return count;
  }

  void testPolynomialIsReproduced(int dim, int H1Order, Teuchos::FancyOStream &out, bool &success)
  {
    const int p = H1Order - 1;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    auto f = polynomialConvectiveDerivative(dim, p, form.beta);
    vector<vector<double>> rootVertices(dim, vector<double>{0.0, 0.6, 1.0}); // 2^dim roots, not uniform

    auto dpg = form.discretization(rootVertices, H1Order, uExact, f);
    dpg->solve();
    TEST_COMPARE(dpg->L2Error(form.u, uExact), <, 1e-11);

    // roots 0 and 1 are neighbours in direction 0.  Bisect them in different tangential directions, so that their
    // interface is a single patch that is neither cell's face; then add a second level and a two-direction bisection.
    BoxForest &forest = dpg->forest();
    int child = forest.refine(0, 2);
    forest.refine(1, 1u << (dim-1));
    forest.refine(child, 1);
    if (dim >= 3) forest.refine(2, 1 | 4);
    if (dim >= 3) { TEST_COMPARE(crossingPatchCount(dpg->skeleton()), >=, 1); }
    dpg->solve();
    TEST_COMPARE(dpg->L2Error(form.u, uExact), <, 1e-11);
    for (double energyError : dpg->energyErrors()) TEST_COMPARE(energyError, <, 1e-10);
  }

TEUCHOS_UNIT_TEST( BoxDPG, DofCount_Quad_xNextTo_y )
{
  Convection form(2);
  auto zero = [](const double *x) { return 0.0; };
  const int H1Order = 2; // p = 1: 4 field dofs per cell, 2 flux dofs per patch
  auto dpg = form.discretization({{0.0, 1.0, 2.0}, {0.0, 1.0}}, H1Order, zero, zero);
  TEST_EQUALITY(dpg->globalDofCount(), 2 * 4 + 7 * 2);
  dpg->forest().refine(0, 1);
  dpg->forest().refine(1, 2);
  // 4 cells; 9 boundary faces and 3 interior patches (the left|right interface is one patch)
  TEST_EQUALITY(dpg->skeleton().patches.size(), 12);
  TEST_EQUALITY(dpg->globalDofCount(), 4 * 4 + 12 * 2);
}

TEUCHOS_UNIT_TEST( BoxDPG, PolynomialIsReproduced_2D )
{
  testPolynomialIsReproduced(2, 3, out, success);
}

TEUCHOS_UNIT_TEST( BoxDPG, PolynomialIsReproduced_3D )
{
  testPolynomialIsReproduced(3, 3, out, success);
}

TEUCHOS_UNIT_TEST( BoxDPG, PolynomialIsReproduced_4D )
{
  testPolynomialIsReproduced(4, 2, out, success);
}

TEUCHOS_UNIT_TEST( BoxDPG, PolynomialIsReproduced_NonBoxPatch )
{
  // an L-shaped patch (three quarters of a face): the trace there is one polynomial on a union that is not a box
  const int dim = 3, H1Order = 3, p = H1Order - 1;
  Convection form(dim);
  auto uExact = polynomial(dim, p);
  auto f = polynomialConvectiveDerivative(dim, p, form.beta);
  auto dpg = form.discretization({{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}, H1Order, uExact, f);
  BoxForest &forest = dpg->forest();
  int left = forest.refine(0, 4);
  forest.refine(left+1, 2);
  int right = forest.refine(1, 2);
  forest.refine(right+1, 4);
  int nonBoxCount = 0;
  for (const auto &patch : dpg->skeleton().patches) if (!patch.isBox) nonBoxCount++;
  TEST_EQUALITY(nonBoxCount, 1);
  dpg->solve();
  TEST_COMPARE(dpg->L2Error(form.u, uExact), <, 1e-11);
}

TEUCHOS_UNIT_TEST( BoxDPG, PolynomialIsReproduced_4D_VelocityOnlyRefinement )
{
  // (x, t, u_x, u_z): bisect the cells of the first x root in the velocity directions only
  const int dim = 4, H1Order = 2, p = H1Order - 1;
  Convection form(dim);
  auto uExact = polynomial(dim, p);
  auto f = polynomialConvectiveDerivative(dim, p, form.beta);
  auto dpg = form.discretization({{0.0, 0.5, 1.0}, {0.0, 1.0}, {-1.0, 0.0, 1.0}, {-1.0, 1.0}}, H1Order, uExact, f);
  BoxForest &forest = dpg->forest();
  forest.refine(0, 4 | 8);
  forest.refine(2, 4 | 8);
  TEST_EQUALITY(forest.leaves().size(), 4 + 4 + 2);
  dpg->solve();
  TEST_COMPARE(dpg->L2Error(form.u, uExact), <, 1e-11);
}

TEUCHOS_UNIT_TEST( BoxDPG, OneDirectionRefinementConverges_2D )
{
  // u = exp(2y) (1 + x): polynomial in x, so bisecting in y alone should converge at the full rate h^(p+1)
  const int dim = 2, H1Order = 3, p = H1Order - 1;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::exp(2 * x[1]) * (1 + x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * std::exp(2 * x[1]) + beta[1] * 2 * std::exp(2 * x[1]) * (1 + x[0]); };
  auto dpg = form.discretization({{0.0, 1.0}, {0.0, 1.0}}, H1Order, uExact, f);

  vector<double> errors;
  for (int level=0; level<4; level++)
  {
    if (level > 0)
    {
      vector<int> leaves = dpg->forest().leaves();
      for (int leaf : leaves) dpg->forest().refine(leaf, 2); // y only
    }
    dpg->solve();
    errors.push_back(dpg->L2Error(form.u, uExact));
  }
  TEST_EQUALITY(dpg->forest().leaves().size(), 8);
  for (size_t i=1; i<errors.size(); i++)
  {
    const double rate = std::log2(errors[i-1] / errors[i]);
    out << "y-only refinement level " << i << ": L2 error " << errors[i] << ", rate " << rate << endl;
    TEST_COMPARE(rate, >, p + 1 - 0.5);
  }
}
TEUCHOS_UNIT_TEST( BoxDPG, MatchesMeshAndSolution_2D )
{
  // On a mesh both can represent (2 x 2 quads, one refined isotropically: hanging nodes, 1-irregular), BoxDPG and
  // Mesh / GDAMinimumRule / Solution should produce the same discrete problem.
  const int dim = 2, H1Order = 3, delta_k = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::exp(x[0] + 2 * x[1]) + std::sin(3 * x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * (std::exp(x[0] + 2 * x[1]) + 3 * std::cos(3 * x[0])) + beta[1] * 2 * std::exp(x[0] + 2 * x[1]); };

  auto dpg = form.discretization({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}, H1Order, uExact, f);

  MeshPtr mesh = MeshFactory::quadMesh(form.bf, H1Order, delta_k, 1.0, 1.0, 2, 2);
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(BoxDPG::function(f) * form.v);
  BCPtr bc = BC::bc();
  FunctionPtr n = Function::normal();
  FunctionPtr beta_n = beta[0] * n->x() + beta[1] * n->y();
  SpatialFilterPtr inflow = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0); // beta = (1, 0.5)
  bc->addDirichlet(form.t_n, inflow, beta_n * BoxDPG::function(uExact));
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, rhs, form.bf->graphNorm());

  for (int refinement=0; refinement<2; refinement++)
  {
    if (refinement == 1)
    {
      // refine the cell containing (0.25, 0.25) in each
      Camellia::FieldContainer<double> point(1, 2);
      point(0,0) = 0.25;
      point(0,1) = 0.25;
      GlobalIndexType cellID = mesh->getTopology()->cellIDsForPoints(point)[0];
      mesh->hRefine(vector<GlobalIndexType>{cellID});
      dpg->forest().refine(0, 3);
    }
    TEST_EQUALITY(mesh->numActiveElements(), GlobalIndexType(dpg->forest().leaves().size()));
    TEST_EQUALITY(mesh->numGlobalDofs(), dpg->globalDofCount());

    solution->solve();
    dpg->solve();

    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    const double legacyL2Error = error->l2norm(mesh); // same cubature as BoxDPG::L2Error
    const double boxL2Error = dpg->L2Error(form.u, uExact);
    out << "L2 error: Solution " << legacyL2Error << ", BoxDPG " << boxL2Error << endl;
    TEST_FLOATING_EQUALITY(legacyL2Error, boxL2Error, 1e-5);

    double boxEnergyErrorSquared = 0;
    for (double cellError : dpg->energyErrors()) boxEnergyErrorSquared += cellError * cellError;
    const double legacyEnergyError = solution->energyErrorTotal();
    out << "energy error: Solution " << legacyEnergyError << ", BoxDPG " << std::sqrt(boxEnergyErrorSquared) << endl;
    TEST_FLOATING_EQUALITY(legacyEnergyError, std::sqrt(boxEnergyErrorSquared), 1e-7);
  }
}
} // namespace
