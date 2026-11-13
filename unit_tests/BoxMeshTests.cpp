#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BasisFamily.h"
#include "BoxDPG.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "RefinementStrategy.h"
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

  //! The anisotropic refinements used by the tests: neighbors bisected in different directions, and a second level.
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

  // cell 0's right side has the two children as neighbors; its left side is on the boundary
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
TEUCHOS_UNIT_TEST( BoxMesh, EnforceOneIrregularityIsDirectional )
{
  // Mesh::enforceOneIrregularity() on a box mesh bisects neighbors only in the direction in which they are too coarse
  Convection form(3);
  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 2, 2);
  vector<IndexType> cells = {13}; // the centre cell
  for (int level=0; level<3; level++)
  {
    std::map<GlobalIndexType,unsigned> masks;
    for (IndexType cell : cells) masks[cell] = 4u; // z only
    mesh->hRefine(masks);
    vector<IndexType> children;
    for (IndexType cell : cells) for (IndexType child : topology->getCell(cell)->getChildIndices(topology)) children.push_back(child);
    cells = children;
  }
  TEST_EQUALITY(mesh->numActiveElements(), 26 + 8);
  mesh->enforceOneIrregularity();
  TEST_EQUALITY(topology->forest().maxTangentialLevelDifference(), 1);
  TEST_EQUALITY(mesh->numActiveElements(), 8 + 4 * 4 + 4 * 2 + 18);
  for (IndexType cellID : topology->getLocallyKnownActiveCellIndices())
  {
    TEST_EQUALITY(int(topology->forest().node(cellID).box.level[0]), 0);
    TEST_EQUALITY(int(topology->forest().node(cellID).box.level[1]), 0);
  }
  TEST_ASSERT(topology->cellsToRefineForRegularity().empty());
}
TEUCHOS_UNIT_TEST( BoxMesh, RefinementStrategyAdaptsBoxMesh )
{
  // The existing energy-error-driven RefinementStrategy on a box mesh (isotropic marking; one-irregularity enforced by
  // the box rule).  A solution with a sharp layer in y: the error should fall as the mesh adapts.
  const int dim = 2, H1Order = 2, delta_k = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::tanh(20 * (x[1] - 0.5)) * (1 + x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x)
  {
    const double th = std::tanh(20 * (x[1] - 0.5));
    return beta[0] * th + beta[1] * 20 * (1 - th * th) * (1 + x[0]);
  };
  auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
  BCPtr bc = form.inflowBC(topology, uExact);
  SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
  mesh->registerSolution(solution);
  RefinementStrategyPtr refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution, 0.3));
  FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);

  solution->solve();
  const double initialError = error->l2norm(mesh, 4);
  const GlobalIndexType initialCells = mesh->numActiveElements();
  double finalError = initialError;
  for (int step=0; step<4; step++)
  {
    refinementStrategy->refine(false);
    solution->solve();
    finalError = error->l2norm(mesh, 4);
    out << "step " << step << ": " << mesh->numActiveElements() << " cells, L2 error " << finalError << endl;
    TEST_COMPARE(topology->forest().maxTangentialLevelDifference(), <=, 1);
  }
  TEST_COMPARE(mesh->numActiveElements(), >, initialCells);
  TEST_COMPARE(finalError, <, 0.5 * initialError);
}
  //! applies y = P x, with x and y the solution vectors of the coarse and fine Solutions
  void prolong(TMatrixPtr<double> P, SolutionPtr coarseSolution, SolutionPtr fineSolution, TVectorPtr<double> &result)
  {
    result = Teuchos::rcp(new TVector<double>(P->getRangeMap(), 1));
    P->apply(*coarseSolution->getLHSVector(), *result);
  }

TEUCHOS_UNIT_TEST( BoxMesh, MultigridHierarchy )
{
  Convection form(3);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(3, vector<double>{0.0, 0.6, 1.0})));
  refineAnisotropically(3, [&](IndexType cell, unsigned mask) { return topology->refineCell(cell, mask); });
  TEST_EQUALITY(topology->refinementDepth(), 2);
  MeshPtr fineMesh = BoxGDA::mesh(topology, form.bf, 3, 2);
  vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(fineMesh, 0, 1);
  TEST_EQUALITY(meshes.size(), 4); // depths 0, 1, 2 at k = 0, then the fine mesh
  TEST_EQUALITY(meshes[0]->numActiveElements(), 8);
  TEST_EQUALITY(meshes[2]->numActiveElements(), fineMesh->numActiveElements());
  TEST_ASSERT(meshes[3].get() == fineMesh.get());
  for (size_t i=0; i+1<meshes.size(); i++)
  {
    TEST_COMPARE(meshes[i]->numActiveElements(), <=, meshes[i+1]->numActiveElements());
    TEST_COMPARE(meshes[i]->numGlobalDofs(), <, meshes[i+1]->numGlobalDofs());
  }
  // every rank's cells in a coarse mesh contain cells that rank owns in the fine mesh
  auto coarseTopology = meshes[0]->getTopology();
  for (GlobalIndexType coarseCellID : meshes[0]->cellIDsInPartition())
  {
    vector<double> centroid = coarseTopology->getCellCentroid(coarseCellID);
    bool ownsSomeFineCell = false;
    for (GlobalIndexType fineCellID : fineMesh->cellIDsInPartition())
    {
      auto extents = topology->cellExtents(fineCellID);
      auto coarseExtents = dynamic_cast<BoxMeshTopology*>(coarseTopology.get())->cellExtents(coarseCellID);
      bool inside = true;
      for (int k=0; k<3; k++) inside = inside && (extents[k].first >= coarseExtents[k].first - 1e-12) && (extents[k].second <= coarseExtents[k].second + 1e-12);
      ownsSomeFineCell = ownsSomeFineCell || inside;
    }
    TEST_ASSERT(ownsSomeFineCell);
  }
}

TEUCHOS_UNIT_TEST( BoxMesh, ProlongationIsExact )
{
  // Solve a problem whose solution is a polynomial of the coarse degree on a coarse and on a fine mesh (both exact).
  // Prolonging the coarse solution vector must give the fine solution vector, wherever the prolongation has a row
  // (everywhere except fluxes on fine faces interior to a coarse cell).
  for (int dim : {2, 3})
  {
    for (int coarseH1Order : {1, 2})
    {
      const int fineH1Order = 3, delta_k = 2, coarseDegree = coarseH1Order - 1;
      Convection form(dim);
      BoxDPG::PointFunction uExact = polynomial(dim, coarseDegree), f = polynomialConvectiveDerivative(dim, coarseDegree, form.beta);
      if (coarseDegree == 0)
      {
        uExact = [](const double *x) { return 3.0; };
        f = [](const double *x) { return 0.0; };
      }
      vector<vector<double>> rootVertices(dim, vector<double>{0.0, 0.6, 1.0});

      auto fineTopology = Teuchos::rcp(new BoxMeshTopology(rootVertices));
      refineAnisotropically(dim, [&](IndexType cell, unsigned mask) { return fineTopology->refineCell(cell, mask); });
      MeshPtr fineMesh = BoxGDA::mesh(fineTopology, form.bf, fineH1Order, delta_k);
      BCPtr fineBC = form.inflowBC(fineTopology, uExact);
      SolutionPtr fineSolution = Solution::solution(form.bf, fineMesh, fineBC, form.rhs(f), form.bf->graphNorm());
      fineSolution->solve();
      TEST_COMPARE((Function::solution(form.u, fineSolution) - BoxDPG::function(uExact))->l2norm(fineMesh), <, 1e-10);

      for (int coarseDepth : {0, 1, 2})
      {
        auto coarseTopology = fineTopology->copyTruncatedAtDepth(coarseDepth);
        MeshPtr coarseMesh = BoxGDA::mesh(coarseTopology, form.bf, coarseH1Order, delta_k);
        BCPtr coarseBC = form.inflowBC(coarseTopology, uExact);
        SolutionPtr coarseSolution = Solution::solution(form.bf, coarseMesh, coarseBC, form.rhs(f), form.bf->graphNorm());
        coarseSolution->solve();
        TEST_COMPARE((Function::solution(form.u, coarseSolution) - BoxDPG::function(uExact))->l2norm(coarseMesh), <, 1e-10);

        BoxGDA* fineGDA = dynamic_cast<BoxGDA*>(fineMesh->globalDofAssignment().get());
        TMatrixPtr<double> P = fineGDA->prolongationOperator(coarseMesh, coarseSolution->getPartitionMap(), fineSolution->getPartitionMap());
        TVectorPtr<double> prolonged;
        prolong(P, coarseSolution, fineSolution, prolonged);

        auto prolongedData = prolonged->getData(0);
        auto fineData = fineSolution->getLHSVector()->getData(0);
        // A row is empty for a fine flux on a face interior to a coarse cell (skip those), and for a fine mode of higher
        // degree than the coarse space (whose exact coefficient is zero).  With the same cells in both meshes (depth 2)
        // only the second kind occurs, so every row can be compared.
        double maxDifference = 0;
        int emptyRows = 0, rows = int(P->getLocalNumRows());
        for (int row=0; row<rows; row++)
        {
          const bool emptyRow = (P->getNumEntriesInLocalRow(row) == 0);
          if (emptyRow) emptyRows++;
          if (emptyRow && (coarseDepth < 2)) continue;
          maxDifference = std::max(maxDifference, std::abs(prolongedData[row] - fineData[row]));
        }
        out << dim << "D, coarse degree " << coarseDegree << ", coarse depth " << coarseDepth << ": " << coarseMesh->numGlobalDofs() << " -> "
            << fineMesh->numGlobalDofs() << " dofs; " << emptyRows << " of " << rows << " rows empty; max difference " << maxDifference << endl;
        TEST_COMPARE(maxDifference, <, 1e-9);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( BoxMesh, GeometricMultigridSolve )
{
  // conjugate gradients preconditioned by a geometric multigrid V-cycle, against the direct solve
  for (int dim : {2, 3, 4})
  {
    const int H1Order = (dim == 4) ? 2 : 3, p = H1Order - 1, delta_k = 2;
    Convection form(dim);
    auto uExact = polynomial(dim, p);
    auto f = polynomialConvectiveDerivative(dim, p, form.beta);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
    refineAnisotropically(dim, [&](IndexType cell, unsigned mask) { return topology->refineCell(cell, mask); });
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());

    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    const int maxIters = 200;
    const double tol = 1e-10;
    Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, maxIters, tol, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
    gmgSolver->setAztecOutput(0);
    solution->solve(gmgSolver);
    const double error = (Function::solution(form.u, solution) - BoxDPG::function(uExact))->l2norm(mesh);
    out << dim << "D: " << meshes.size() << " levels, " << mesh->numGlobalDofs() << " dofs; " << gmgSolver->iterationCount()
        << " CG iterations; L2 error " << error << endl;
    TEST_COMPARE(gmgSolver->iterationCount(), <, 100);
    TEST_COMPARE(error, <, 1e-7);
  }
}
TEUCHOS_UNIT_TEST( BoxMesh, GeometricMultigridMatchesMeshTopology_2D )
{
  // The same problem and the same meshes (2 x 2 quads refined uniformly twice, then one cell once more, so with hanging
  // nodes) on a MeshTopology and on a BoxMeshTopology: multigrid-preconditioned CG should need about as many iterations.
  const int dim = 2, H1Order = 3, delta_k = 2;
  Convection form(dim);
  auto uExact = [](const double *x) { return std::exp(x[0] + 2 * x[1]) + std::sin(3 * x[0]); };
  vector<double> beta = form.beta;
  auto f = [beta](const double *x) { return beta[0] * (std::exp(x[0] + 2 * x[1]) + 3 * std::cos(3 * x[0])) + beta[1] * 2 * std::exp(x[0] + 2 * x[1]); };
  BCPtr bc = BC::bc();
  FunctionPtr n = Function::normal();
  bc->addDirichlet(form.t_n, SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0), (beta[0] * n->x() + beta[1] * n->y()) * BoxDPG::function(uExact));

  Camellia::FieldContainer<double> point(1, 2);
  point(0,0) = 0.3;
  point(0,1) = 0.3;
  int iterationCounts[2];
  double errors[2];
  GlobalIndexType dofCounts[2];
  for (int useBox=0; useBox<2; useBox++)
  {
    MeshPtr mesh;
    if (useBox == 0) mesh = MeshFactory::quadMesh(form.bf, H1Order, delta_k, 1.0, 1.0, 2, 2);
    else mesh = BoxGDA::mesh(Teuchos::rcp(new BoxMeshTopology({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}})), form.bf, H1Order, delta_k);
    mesh->hRefine(mesh->getActiveCellIDsGlobal());
    mesh->hRefine(mesh->getActiveCellIDsGlobal());
    mesh->hRefine(set<GlobalIndexType>{GlobalIndexType(mesh->getTopology()->cellIDsForPoints(point)[0])});
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, 300, 1e-10, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
    gmgSolver->setAztecOutput(0);
    solution->solve(gmgSolver);
    iterationCounts[useBox] = gmgSolver->iterationCount();
    errors[useBox] = (Function::solution(form.u, solution) - BoxDPG::function(uExact))->l2norm(mesh);
    dofCounts[useBox] = mesh->numGlobalDofs();
    out << ((useBox == 0) ? "MeshTopology:    " : "BoxMeshTopology: ") << mesh->numActiveElements() << " cells, " << dofCounts[useBox] << " dofs, "
        << meshes.size() << " levels; " << iterationCounts[useBox] << " CG iterations; L2 error " << errors[useBox] << endl;
  }
  TEST_EQUALITY(dofCounts[0], dofCounts[1]);
  TEST_FLOATING_EQUALITY(errors[0], errors[1], 1e-5);
  TEST_COMPARE(iterationCounts[1], <=, int(1.15 * iterationCounts[0]));
}
TEUCHOS_UNIT_TEST( BoxMesh, SerendipityBasis )
{
  // Serendipity trial and test spaces (polynomials of superlinear degree <= p): the flux on a patch is then a
  // serendipity polynomial.  Polynomial reproduction on anisotropically refined meshes, directly and by multigrid,
  // and exact projection on refinement.
  for (int dim : {3, 4})
  {
    const int H1Order = 4, p = H1Order - 1, delta_k = (dim == 4) ? 1 : 2; // p = 3: S_3 is a proper subspace of Q_3
    Convection form(dim, true);
    // in Camellia's serendipity space of degree p: pure powers up to x_k^p, and the product of two linear factors
    BoxDPG::PointFunction uExact = [dim, p](const double *x)
    {
      double value = 1 + (1 + 0.5 * x[0]) * (1 - 0.25 * x[dim-1]);
      for (int k=0; k<dim; k++) value += (k+1) * std::pow(x[k], p);
      return value;
    };
    vector<double> betaCopy = form.beta;
    BoxDPG::PointFunction f = [dim, p, betaCopy](const double *x)
    {
      double value = betaCopy[0] * 0.5 * (1 - 0.25 * x[dim-1]) + betaCopy[dim-1] * (1 + 0.5 * x[0]) * (-0.25);
      for (int k=0; k<dim; k++) value += betaCopy[k] * (k+1) * p * std::pow(x[k], p-1);
      return value;
    };
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.6, 1.0})));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    // fewer dofs per cell than the full tensor space
    const int fluxCardinality = mesh->getElementType(0)->trialOrderPtr->getBasisCardinality(form.t_n->ID(), 0);
    int fullTensorCardinality = 1;
    for (int k=0; k<dim-1; k++) fullTensorCardinality *= (p + 1);
    TEST_COMPARE(fluxCardinality, <, fullTensorCardinality);

    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
    mesh->registerSolution(solution);
    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    solution->solve();
    TEST_COMPARE(error->l2norm(mesh), <, 1e-9);

    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 2u}, {1, 1u << (dim-1)}});
    const double projectedError = error->l2norm(mesh);
    TEST_COMPARE(projectedError, <, 1e-9);
    solution->solve();
    const double directError = error->l2norm(mesh);
    TEST_COMPARE(directError, <, 1e-9);

    vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    Teuchos::RCP<GMGSolver> gmgSolver = GMGSolver::gmgSolver(solution, true, meshes, 400, 1e-11, GMGOperator::V_CYCLE, Solver::getDirectSolver(true));
    gmgSolver->setAztecOutput(0);
    solution->solve(gmgSolver);
    const double gmgError = error->l2norm(mesh);
    out << dim << "D serendipity p=" << p << ": flux cardinality " << fluxCardinality << " (full tensor " << fullTensorCardinality << "); "
        << mesh->numGlobalDofs() << " dofs; L2 errors: projected " << projectedError << ", direct " << directError << ", multigrid " << gmgError
        << " (" << gmgSolver->iterationCount() << " iterations)" << endl;
    TEST_COMPARE(gmgError, <, 1e-7);
  }
}

TEUCHOS_UNIT_TEST( BoxMesh, AnisotropicRefinementMasks )
{
  const int dim = 3;
  Convection form(dim);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>(dim, vector<double>{0.0, 0.5, 1.0})));
  MeshPtr mesh = BoxGDA::mesh(topology, form.bf, 4, 1); // p = 3: the indicator uses the quadratic and cubic modes
  set<GlobalIndexType> activeCells = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cellIDs(activeCells.begin(), activeCells.end());

  // varies in y only
  FunctionPtr fy = BoxDPG::function([](const double *x) { return std::sin(9 * x[1]); });
  for (const auto &entry : BoxGDA::anisotropicRefinementMasks(mesh, fy, cellIDs, 0.5)) { TEST_EQUALITY(entry.second, 2u); }
  // the same quadratic in x and in z (the linear parts differ from cell to cell, and must not matter)
  FunctionPtr fxz = BoxDPG::function([](const double *x) { return x[0] * x[0] + x[2] * x[2] + 3 * x[1]; });
  auto masks = BoxGDA::anisotropicRefinementMasks(mesh, fxz, cellIDs, 0.5);
  TEST_EQUALITY(masks.size(), cellIDs.size());
  for (const auto &entry : masks) { TEST_EQUALITY(entry.second, 1u | 4u); }
  // strongly in x, weakly in z: z drops out at a high threshold and comes back at a low one
  FunctionPtr fStrongX = BoxDPG::function([](const double *x) { return x[0] * x[0] + 0.05 * x[2] * x[2]; });
  for (const auto &entry : BoxGDA::anisotropicRefinementMasks(mesh, fStrongX, cellIDs, 0.5)) { TEST_EQUALITY(entry.second, 1u); }
  for (const auto &entry : BoxGDA::anisotropicRefinementMasks(mesh, fStrongX, cellIDs, 0.01)) { TEST_EQUALITY(entry.second, 1u | 4u); }
  // restricted to allowed directions
  for (const auto &entry : BoxGDA::anisotropicRefinementMasks(mesh, fxz, cellIDs, 0.5, 4u | 2u)) { TEST_EQUALITY(entry.second, 4u); }
  // linear: nothing in the high modes, so no information, and every allowed direction is chosen
  FunctionPtr linear = BoxDPG::function([](const double *x) { return 1 + x[0] - 2 * x[2]; });
  for (const auto &entry : BoxGDA::anisotropicRefinementMasks(mesh, linear, cellIDs, 0.5)) { TEST_EQUALITY(entry.second, 7u); }

  // tangential to tagged sides: cells on the x-min boundary must not be bisected in x
  topology->applyTag(DIRICHLET_SET_TAG_NAME, 17, [](int direction, bool upper, const BoxMeshTopology::Extents &) { return (direction == 0) && !upper; });
  for (const auto &entry : BoxGDA::masksTangentialToTaggedSides(mesh, DIRICHLET_SET_TAG_NAME, 17, cellIDs))
  {
    const bool onXMin = (topology->cellExtents(entry.first)[0].first == 0.0);
    TEST_EQUALITY(entry.second, onXMin ? (2u | 4u) : 7u);
  }
}

TEUCHOS_UNIT_TEST( BoxMesh, AnisotropicAdaptivity )
{
  // A layer in y: energy-error marking, with each marked cell bisected in the directions in which the solution is least
  // resolved, against the same marking with every marked cell bisected in both directions.
  const int dim = 2, H1Order = 3, delta_k = 2;
  auto uExact = [](const double *x) { return std::tanh(20 * (x[1] - 0.5)) * (1 + x[0]); };
  double errors[2];
  GlobalIndexType cellCounts[2], dofCounts[2];
  int maxXLevel[2];
  for (int anisotropic=0; anisotropic<2; anisotropic++)
  {
    Convection form(dim);
    vector<double> beta = form.beta;
    auto f = [beta](const double *x)
    {
      const double th = std::tanh(20 * (x[1] - 0.5));
      return beta[0] * th + beta[1] * 20 * (1 - th * th) * (1 + x[0]);
    };
    auto topology = Teuchos::rcp(new BoxMeshTopology({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}));
    MeshPtr mesh = BoxGDA::mesh(topology, form.bf, H1Order, delta_k);
    BCPtr bc = form.inflowBC(topology, uExact);
    SolutionPtr solution = Solution::solution(form.bf, mesh, bc, form.rhs(f), form.bf->graphNorm());
    mesh->registerSolution(solution);
    RefinementStrategyPtr refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution, 0.3));
    if (anisotropic)
    {
      FunctionPtr uSolution = Function::solution(form.u, solution);
      refinementStrategy->setCellRefiner([mesh, uSolution](std::vector<GlobalIndexType> &cellIDs) {
        uSolution->importCellData(cellIDs);
        mesh->hRefine(BoxGDA::anisotropicRefinementMasks(mesh, uSolution, cellIDs, 0.5), false);
        return static_cast<int>(cellIDs.size());
      });
    }
    FunctionPtr error = Function::solution(form.u, solution) - BoxDPG::function(uExact);
    solution->solve();
    // isotropic: five marking steps.  anisotropic: as many steps as it takes to reach the isotropic error.
    const int maxSteps = anisotropic ? 20 : 5;
    for (int step=0; step<maxSteps; step++)
    {
      refinementStrategy->refine(false);
      solution->solve();
      if (anisotropic && (error->l2norm(mesh, 4) <= errors[0])) break;
    }
    errors[anisotropic] = error->l2norm(mesh, 4);
    cellCounts[anisotropic] = mesh->numActiveElements();
    dofCounts[anisotropic] = mesh->numGlobalDofs();
    maxXLevel[anisotropic] = 0;
    for (IndexType cellID : topology->getLocallyKnownActiveCellIndices()) maxXLevel[anisotropic] = std::max(maxXLevel[anisotropic], int(topology->forest().node(cellID).box.level[0]));
    out << (anisotropic ? "anisotropic: " : "isotropic:   ") << cellCounts[anisotropic] << " cells, " << dofCounts[anisotropic] << " dofs, max x level "
        << maxXLevel[anisotropic] << ", L2 error " << errors[anisotropic] << endl;
  }
  // the anisotropic mesh reaches the isotropic mesh's error with fewer cells, and with less refinement in x
  TEST_COMPARE(errors[1], <=, errors[0]);
  TEST_COMPARE(cellCounts[1], <, cellCounts[0]);
  TEST_COMPARE(maxXLevel[1], <, maxXLevel[0]);
}
} // namespace
