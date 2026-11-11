#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BoxDPG.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include <cmath>

using namespace Camellia;
using namespace std;

// Tests of the existing Mesh / Solution machinery running on a BoxMeshTopology with a BoxGDA.

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

    //! Tags each inflow face of the domain (tag ID = 2 * direction + upper) and sets t_n = (beta . n) u there.
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

  //! The anisotropic refinements used by the tests: neighbours bisected in different directions, and a second level.
  template<class Refine>
  void refineAnisotropically(int dim, Refine refine)
  {
    vector<IndexType> children = refine(0, 2);
    refine(1, 1u << (dim-1));
    refine(children[0], 1);
    if (dim >= 3) refine(2, 1 | 4);
  }

  void testSolutionReproducesPolynomial(int dim, int H1Order, Teuchos::FancyOStream &out, bool &success)
  {
    const int p = H1Order - 1, delta_k = 2;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    auto f = polynomialConvectiveDerivative(dim, p, form.beta);
    vector<vector<double>> rootVertices(dim, vector<double>{0.0, 0.6, 1.0});

    auto topology = Teuchos::rcp(new BoxMeshTopology(rootVertices));
    refineAnisotropically(dim, [&](IndexType cell, unsigned mask) { return topology->refineCell(cell, mask); });
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
    solution->solve();

    // the same discretization through BoxDPG
    BoxDPG dpg(rootVertices, form.bf, H1Order, delta_k);
    refineAnisotropically(dim, [&](IndexType cell, unsigned mask)
    {
      int firstChild = dpg.forest().refine(cell, mask);
      return vector<IndexType>{firstChild, firstChild + 1};
    });
    TEST_EQUALITY(mesh->numActiveElements(), GlobalIndexType(dpg.forest().leaves().size()));
    TEST_EQUALITY(mesh->numGlobalDofs(), dpg.globalDofCount());

    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    const double L2Error = error->l2norm(mesh);
    out << dim << "D: " << mesh->numActiveElements() << " cells, " << mesh->numGlobalDofs() << " dofs; L2 error " << L2Error
        << ", energy error " << solution->energyErrorTotal() << endl;
    TEST_COMPARE(L2Error, <, 1e-10);
    TEST_COMPARE(solution->energyErrorTotal(), <, 1e-9);
  }

TEUCHOS_UNIT_TEST( BoxMesh, TopologyBasics )
{
  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 1.0, 3.0}, {0.0, 2.0}}));
  TEST_EQUALITY(topology->getDimension(), 2);
  TEST_EQUALITY(topology->cellCount(), 2);
  TEST_EQUALITY(topology->activeCellCount(), 2);
  vector<IndexType> children = topology->refineCell(1, 2); // bisect the right cell in y
  TEST_EQUALITY(children.size(), 2);
  TEST_EQUALITY(topology->cellCount(), 4);
  TEST_EQUALITY(topology->activeCellCount(), 3);
  TEST_ASSERT(topology->isParent(1));
  TEST_ASSERT(!topology->isParent(children[0]));
  TEST_EQUALITY(topology->getCell(children[1])->getParent()->cellIndex(), 1);
  TEST_EQUALITY(topology->getCell(children[1])->level(), 1);
  TEST_EQUALITY(topology->getCell(1)->getChildIndices(topology).size(), 2);

  vector<double> centroid = topology->getCellCentroid(children[1]);
  TEST_FLOATING_EQUALITY(centroid[0], 2.0, 1e-15);
  TEST_FLOATING_EQUALITY(centroid[1], 1.5, 1e-15);

  Camellia::FieldContainer<double> points(2, 2);
  points(0,0) = 0.5; points(0,1) = 1.9; // in cell 0
  points(1,0) = 2.5; points(1,1) = 1.9; // in the upper child
  vector<IndexType> cellIDs = topology->cellIDsForPoints(points);
  TEST_EQUALITY(cellIDs[0], 0);
  TEST_EQUALITY(cellIDs[1], children[1]);

  // cell 0's right side has the two children as neighbours; its left side is on the boundary
  CellPtr cell0 = topology->getCell(0);
  int boundarySideCount = 0, sidesWithTwoNeighbors = 0;
  for (unsigned side=0; side<cell0->getSideCount(); side++)
  {
    if (cell0->isBoundary(side)) boundarySideCount++;
    if (topology->sideNeighbors(0, side).size() == 2) sidesWithTwoNeighbors++;
  }
  TEST_EQUALITY(boundarySideCount, 3);
  TEST_EQUALITY(sidesWithTwoNeighbors, 1);
  TEST_EQUALITY(cell0->boundarySides().size(), 3);
  TEST_EQUALITY(cell0->getActiveNeighborIndices(topology).size(), 2);

  // parities: +1 on the boundary and on upper faces
  vector<int> parities;
  TEST_ASSERT(topology->getCellSideParities(children[0], parities));
  int negativeCount = 0;
  for (int parity : parities) if (parity == -1) negativeCount++;
  TEST_EQUALITY(negativeCount, 1); // only the x-lower face, shared with cell 0
}

TEUCHOS_UNIT_TEST( BoxMesh, SolutionReproducesPolynomial_2D )
{
  testSolutionReproducesPolynomial(2, 3, out, success);
}

TEUCHOS_UNIT_TEST( BoxMesh, SolutionReproducesPolynomial_3D )
{
  testSolutionReproducesPolynomial(3, 3, out, success);
}

TEUCHOS_UNIT_TEST( BoxMesh, SolutionReproducesPolynomial_4D )
{
  testSolutionReproducesPolynomial(4, 2, out, success);
}

TEUCHOS_UNIT_TEST( BoxMesh, SolutionMatchesMeshTopology_2D )
{
  // The same problem on a BoxMeshTopology and on a MeshTopology (2 x 2 quads, one refined isotropically), both through
  // Solution, with BCs imposed through spatial filters.
  const int dim = 2, H1Order = 3, delta_k = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::exp(x[0] + 2 * x[1]) + std::sin(3 * x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * (std::exp(x[0] + 2 * x[1]) + 3 * std::cos(3 * x[0])) + beta[1] * 2 * std::exp(x[0] + 2 * x[1]); };

  BCPtr bc = BC::bc();
  FunctionPtr n = Function::normal();
  FunctionPtr beta_n = beta[0] * n->x() + beta[1] * n->y();
  SpatialFilterPtr inflow = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0);
  bc->addDirichlet(form.t_n, inflow, beta_n * BoxDPG::function(uExact));

  MeshPtr legacyMesh = MeshFactory::quadMesh(form.bf, H1Order, delta_k, 1.0, 1.0, 2, 2);
  Camellia::FieldContainer<double> point(1, 2);
  point(0,0) = 0.25;
  point(0,1) = 0.25;
  legacyMesh->hRefine(vector<GlobalIndexType>{GlobalIndexType(legacyMesh->getTopology()->cellIDsForPoints(point)[0])});

  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}));
  topology->refineCell(topology->cellIDsForPoints(point)[0], 3);
  MeshPtr boxMesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);

  TEST_EQUALITY(legacyMesh->numActiveElements(), boxMesh->numActiveElements());
  TEST_EQUALITY(legacyMesh->numGlobalDofs(), boxMesh->numGlobalDofs());

  SolutionPtr legacySolution = Solution::solution(form.bf, legacyMesh, bc, form.rhs(f), form.bf->graphNorm());
  SolutionPtr boxSolution = Solution::solution(form.bf, boxMesh, bc, form.rhs(f), form.bf->graphNorm());
  legacySolution->solve();
  boxSolution->solve();

  const double legacyL2Error = (Function::solution(form.u, legacySolution) - BoxDPG::function(uExact))->l2norm(legacyMesh);
  const double boxL2Error = (Function::solution(form.u, boxSolution) - BoxDPG::function(uExact))->l2norm(boxMesh);
  out << "L2 error: MeshTopology " << legacyL2Error << ", BoxMeshTopology " << boxL2Error << endl;
  TEST_FLOATING_EQUALITY(legacyL2Error, boxL2Error, 1e-6);
  TEST_FLOATING_EQUALITY(legacySolution->energyErrorTotal(), boxSolution->energyErrorTotal(), 1e-7);
}
TEUCHOS_UNIT_TEST( BoxMesh, RefineThroughMeshProjectsSolution )
{
  // Solve, refine anisotropically through Mesh::hRefine(), and check that the solution carried onto the children is
  // still the exact polynomial (fields), before and after solving again.
  for (int dim : {2, 4})
  {
    const int H1Order = (dim == 2) ? 3 : 2, p = H1Order - 1, delta_k = 2;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    auto f = polynomialConvectiveDerivative(dim, p, form.beta);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
    mesh->registerSolution(solution);
    solution->solve();
    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);

    const GlobalIndexType dofsBefore = mesh->numGlobalDofs();
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 2u}, {1, 1u << (dim-1)}});
    TEST_EQUALITY(mesh->numActiveElements(), (1 << dim) + 2);
    TEST_COMPARE(mesh->numGlobalDofs(), >, dofsBefore);
    const double projectedError = error->l2norm(mesh);
    out << dim << "D: L2 error of the projected solution after refinement: " << projectedError << endl;
    TEST_COMPARE(projectedError, <, 1e-10);

    // a second level, in a child, then solve again
    vector<IndexType> children = topology->getCell(0)->getChildIndices(topology);
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{GlobalIndexType(children[0]), 1u}});
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
    solution->solve();
    TEST_COMPARE(error->l2norm(mesh), <, 1e-10);
    TEST_COMPARE(solution->energyErrorTotal(), <, 1e-9);
  }
}

TEUCHOS_UNIT_TEST( BoxMesh, IsotropicRefinementThroughMesh )
{
  // Mesh::hRefine(cellIDs) with no direction mask bisects in every direction
  Convection form(3);
  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 2, 2);
  mesh->hRefine(set<GlobalIndexType>{0});
  TEST_EQUALITY(mesh->numActiveElements(), 8);
}
} // namespace
