#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "BasisFamily.h"
#include "BC.h"
#include "BoxCellDPGOperator.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "ElementType.h"
#include "Function.h"
#include "MatrixFreeDPGOperator.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "RHS.h"
#include "SeparableTensorIP.h"
#include "VarFactory.h"

#include <cmath>

using namespace Camellia;
using namespace std;

namespace
{
  //! a function constant on each cell (1 + 10 times the product of the cell's widths), as the separable norm's weights are
  class CellWidthWeight : public TFunction<double>
  {
  public:
    CellWidthWeight() : TFunction<double>(0) {}
    void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const auto &nodes = basisCache->getPhysicalCellNodes();
      for (int c=0; c<values.dimension(0); c++)
      {
        double product = 1;
        for (int k=0; k<nodes.dimension(2); k++)
        {
          double lo = nodes(c,0,k), hi = nodes(c,0,k);
          for (int n=1; n<nodes.dimension(1); n++) { lo = std::min(lo, nodes(c,n,k)); hi = std::max(hi, nodes(c,n,k)); }
          product *= (hi - lo);
        }
        for (int p=0; p<values.dimension(1); p++) values(c,p) = 1.0 + 10.0 * product;
      }
    }
  };

  struct Setup
  {
    MeshPtr mesh;
    BFPtr bf;
    VarPtr v;
    Teuchos::RCP<SeparableTensorIP> ip;
    VarPtr t;
  };

  //! a transport-like ultraweak form on a 4D box mesh (cells of different widths, orders raised in some directions on
  //! every other cell): field u, flux t, test v; x-dependent coefficients; the direction-separable graph norm
  Setup setup(bool serendipityTrial, double constantModeRatio = 1.0)
  {
    const int H1Order = 2, delta_k = 2;
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr u = vf->fieldVar("u", L2);
    VarPtr t = vf->fluxVar("t", L2);
    VarPtr v = vf->testVar("v", HGRAD);
    vf->setTrialBasisFamily(serendipityTrial ? BasisFamily::serendipityBasisFamily() : BasisFamily::basisFamily(BASIS_FAMILY_HIERARCHICAL));
    vf->setTestBasisFamily(BasisFamily::basisFamily(BASIS_FAMILY_HIERARCHICAL));
    BFPtr bf = Teuchos::rcp(new BF(vf));
    FunctionPtr x = Function::xn(1), y = Function::yn(1);
    bf->addTerm(-x * u, v->di(1));
    bf->addTerm(-u, v->di(2));
    bf->addTerm((2.0 + y) * u, v->di(3));
    bf->addTerm(0.5 * u, v->di(4));
    bf->addTerm(3.0 * u, v);
    bf->addTerm(t, v);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>{{0.0, 0.5, 2.0}, {0.0, 0.1}, {-1.0, 0.0, 3.0}, {-2.0, 2.0}}));
    MeshPtr mesh = BoxGDA::mesh(topology, bf, H1Order, delta_k);
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u | 4u}});
    std::map<GlobalIndexType, std::vector<int>> orderIncrements;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) if (cellID % 2 == 0) orderIncrements[cellID] = {1,0,2,0};
    BoxGDA::pRefineInDirections(mesh, orderIncrements);

    auto ip = SeparableTensorIP::ip();
    FunctionPtr weight = Teuchos::rcp(new CellWidthWeight);
    ip->addTerm(v->di(2));
    ip->addTerm(weight * v->di(1));
    ip->addTerm(1.8 * v->di(3));
    ip->addTerm(0.65 * v->di(4));
    ip->addTerm(Function::sqrtFunction(weight * 3.0) * v);
    if (constantModeRatio < 1.0) ip->setConstantModeWeight(v, weight * 3.0, constantModeRatio);
    return {mesh, bf, v, ip, t};
  }
}

TEUCHOS_UNIT_TEST( BoxCellDPGOperator, MatchesAssembledMatrices )
{
  // B u, B^T v and B^T G^{-1} B u by sum factorization agree with the assembled element matrices (to roundoff: the
  // coefficients are polynomial, so both quadratures are exact), on every cell, for serendipity and tensor trial bases.
  const int enrichment = 2;
  for (bool serendipityTrial : {false, true})
  {
    for (double constantModeRatio : {1.0, 1e-2})
    {
      Setup s = setup(serendipityTrial, constantModeRatio);
      BoxCellDPGOperator op(s.bf, s.ip, s.mesh, enrichment);
      RHSPtr rhs = RHS::rhs();
      rhs->addTerm(1.0 * s.v);
      s.bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);
      double maxB = 0, maxBT = 0, maxS = 0;
      int cells = 0;
      for (GlobalIndexType cellID : s.mesh->getActiveCellIDsGlobal())
      {
        std::string reason;
        TEST_ASSERT(op.supports(cellID, &reason));
        if (!op.supports(cellID)) { out << "cell " << cellID << " unsupported: " << reason << endl; continue; }
        cells++;
        ElementTypePtr elemType = s.mesh->getElementType(cellID);
        const int trialCount = elemType->trialOrderPtr->totalDofs(), testCount = elemType->testOrderPtr->totalDofs();
        TEST_EQUALITY(op.trialDofCount(cellID), trialCount);
        TEST_EQUALITY(op.testSize(cellID), testCount);
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(s.mesh, cellID, false, enrichment);
        BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(s.mesh, cellID, true, enrichment);
        Camellia::FieldContainer<double> B(1, testCount, trialCount);
        s.bf->stiffnessMatrix(B, elemType, s.mesh->cellSideParitiesForCell(cellID), basisCache, true, false); // (C, test, trial)
        Camellia::FieldContainer<double> K(1, trialCount, trialCount), load(1, trialCount);
        s.bf->localStiffnessMatrixAndRHS(K, load, s.ip, ipBasisCache, rhs, basisCache);

        vector<double> u(trialCount), w(testCount);
        for (int j=0; j<trialCount; j++) u[j] = sin(1.0 + 3.7 * j);
        for (int i=0; i<testCount; i++) w[i] = cos(0.3 + 1.9 * i); // in the tensor layout
        const vector<int> &testDof = op.testDofIndex(cellID);

        vector<double> Bu(testCount), BTw(trialCount), Su(trialCount);
        op.applyB(cellID, u.data(), Bu.data());
        op.applyBTranspose(cellID, w.data(), BTw.data());
        op.applySchur(cellID, u.data(), Su.data());

        double scaleB = 0, scaleBT = 0, scaleS = 0, errorB = 0, errorBT = 0, errorS = 0;
        for (int t=0; t<testCount; t++)
        {
          double expected = 0;
          for (int j=0; j<trialCount; j++) expected += B(0, testDof[t], j) * u[j];
          errorB = std::max(errorB, abs(expected - Bu[t]));
          scaleB = std::max(scaleB, abs(expected));
        }
        for (int j=0; j<trialCount; j++)
        {
          double expectedBT = 0, expectedS = 0;
          for (int t=0; t<testCount; t++) expectedBT += B(0, testDof[t], j) * w[t];
          for (int i=0; i<trialCount; i++) expectedS += K(0, j, i) * u[i];
          errorBT = std::max(errorBT, abs(expectedBT - BTw[j]));
          scaleBT = std::max(scaleBT, abs(expectedBT));
          errorS = std::max(errorS, abs(expectedS - Su[j]));
          scaleS = std::max(scaleS, abs(expectedS));
        }
        maxB = std::max(maxB, errorB / scaleB);
        maxBT = std::max(maxBT, errorBT / scaleBT);
        maxS = std::max(maxS, errorS / scaleS);
      }
      out << (serendipityTrial ? "serendipity" : "tensor") << " trial, constant-mode ratio " << constantModeRatio << ", " << cells
          << " cells: relative errors B " << maxB << ", B^T " << maxBT << ", B^T G^{-1} B " << maxS << endl;
      TEST_COMPARE(cells, >, 0);
      TEST_COMPARE(maxB, <, 1e-12);
      TEST_COMPARE(maxBT, <, 1e-12);
      TEST_COMPARE(maxS, <, 1e-9);
    }
  }
}

TEUCHOS_UNIT_TEST( BoxCellDPGOperator, GlobalOperatorMatchesSolution )
{
  // The matrix-free global operator applies the matrix Solution assembles, and its load is Solution's load; with all
  // boundary fluxes prescribed, conjugate gradients on the matrix-free system reproduce Solution's direct solve.
  const int enrichment = 2;
  Setup s = setup(true, 1e-2);
  auto local = Teuchos::rcp(new BoxCellDPGOperator(s.bf, s.ip, s.mesh, enrichment));
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm((1.0 + Function::xn(1)) * s.v);
  BCPtr bc = BC::bc();
  bc->addDirichlet(s.t, SpatialFilter::allSpace(), 1.0 + Function::xn(1));
  SolutionPtr solution = Solution::solution(s.bf, s.mesh, bc, rhs, s.ip);
  solution->setCubatureEnrichmentDegree(enrichment);
  s.bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);
  solution->initializeLHSVector();
  solution->initializeStiffnessAndLoad();
  solution->populateStiffnessAndLoad();
  TMatrixPtr<double> A = solution->getStiffnessMatrix();
  TVectorPtr<double> assembledLoad = solution->getRHSVector();
  TMapConstPtr map = A->getRowMap();

  using MV = MatrixFreeDPGOperator::MV;
  MatrixFreeDPGOperator unconstrained(s.mesh, local, map);
  MV x(map, 1), Ax(map, 1), Sx(map, 1), L(map, 1);
  x.randomize();
  A->apply(x, Ax);
  unconstrained.applyUnconstrained(x, Sx);
  unconstrained.load(rhs, L);
  Sx.update(-1.0, Ax, 1.0);
  L.update(-1.0, *assembledLoad, 1.0);
  const double operatorError = Sx.getVector(0)->normInf() / Ax.getVector(0)->normInf();
  const double loadError = L.getVector(0)->normInf() / assembledLoad->getVector(0)->normInf();
  out << map->getGlobalNumElements() << " global dofs: |S x - A x| / |A x| = " << operatorError << ", |L - assembled load| / |assembled load| = "
      << loadError << endl;
  TEST_COMPARE(operatorError, <, 1e-10);
  TEST_COMPARE(loadError, <, 1e-10);

  // the direct solve
  solution->solve();
  TVectorPtr<double> direct = solution->getLHSVector();

  // conjugate gradients on the matrix-free system with the same Dirichlet data
  std::vector<GlobalIndexType> dirichletIndices;
  std::vector<double> dirichletValues;
  MatrixFreeDPGOperator::dirichletData(s.mesh, bc, dirichletIndices, dirichletValues);
  MatrixFreeDPGOperator op(s.mesh, local, map, std::set<GlobalIndexType>(dirichletIndices.begin(), dirichletIndices.end()));
  MV b(map, 1), u(map, 1), r(map, 1), p(map, 1), Ap(map, 1);
  op.dirichletRightHandSide(rhs, dirichletIndices, dirichletValues, b);
  u.putScalar(0.0);
  r = MV(b, Teuchos::Copy);
  p = MV(r, Teuchos::Copy);
  double rr = r.getVector(0)->dot(*r.getVector(0));
  const double bNorm = std::sqrt(rr);
  int iterations = 0;
  for (; iterations < 5000 && std::sqrt(rr) > 1e-13 * bNorm; iterations++)
  {
    op.apply(p, Ap);
    const double alpha = rr / p.getVector(0)->dot(*Ap.getVector(0));
    u.update(alpha, p, 1.0);
    r.update(-alpha, Ap, 1.0);
    const double rrNew = r.getVector(0)->dot(*r.getVector(0));
    p.update(1.0, r, rrNew / rr);
    rr = rrNew;
  }
  MV difference(u, Teuchos::Copy);
  difference.update(-1.0, *direct, 1.0);
  const double solutionError = difference.getVector(0)->normInf() / direct->getVector(0)->normInf();
  out << dirichletIndices.size() << " Dirichlet dofs; CG: " << iterations << " iterations, |u_CG - u_direct| / |u_direct| = " << solutionError << endl;
  TEST_COMPARE(solutionError, <, 1e-8);
}
