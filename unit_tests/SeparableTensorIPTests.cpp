#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "BasisFamily.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "Function.h"
#include "RHS.h"
#include "SeparableGramInverse.h"
#include "SeparableTensorIP.h"
#include "SerialDenseWrapper.h"
#include "VarFactory.h"

#include <cmath>

using namespace Camellia;
using namespace std;

namespace
{
  //! a function constant on each cell (the product of the cell's widths), as the separable norm's weights are
  class CellWidthProduct : public TFunction<double>
  {
  public:
    CellWidthProduct() : TFunction<double>(0) {}
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

  //! a 4D box mesh of cells of different widths (isotropic order, raised by increments on every other cell), and the
  //! direction-separable graph norm's form: per-cell weights on each derivative, and on the value
  struct SeparableSetup
  {
    MeshPtr mesh;
    BFPtr bf;
    VarPtr v;
    Teuchos::RCP<SeparableTensorIP> ip;
  };

  //! constantModeRatio < 1: also weaken the L^2 term on each cell's constant mode (setConstantModeWeight())
  SeparableSetup separableSetup(bool serendipity, const vector<int> &increments, double constantModeRatio = 1.0)
  {
    const int H1Order = 2, delta_k = 2;
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr u = vf->fieldVar("u", L2);
    VarPtr v = vf->testVar("v", HGRAD);
    if (serendipity) vf->setBasisFamily(BasisFamily::serendipityBasisFamily());
    BFPtr bf = Teuchos::rcp(new BF(vf));
    bf->addTerm(u, v);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>{{0.0, 0.5, 2.0}, {0.0, 0.1}, {-1.0, 0.0, 3.0}, {-2.0, 2.0}}));
    MeshPtr mesh = BoxGDA::mesh(topology, bf, H1Order, delta_k);
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u | 4u}});
    std::map<GlobalIndexType, std::vector<int>> orderIncrements;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) if (cellID % 2 == 0) orderIncrements[cellID] = increments;
    BoxGDA::pRefineInDirections(mesh, orderIncrements);

    auto ip = SeparableTensorIP::ip();
    FunctionPtr weight = Teuchos::rcp(new CellWidthProduct);
    ip->addTerm(v->di(2));
    ip->addTerm(weight * v->di(1));
    ip->addTerm(1.8 * v->di(3));
    ip->addTerm(0.65 * v->di(4));
    ip->addTerm(Function::sqrtFunction(weight * 3.0) * v);
    if (constantModeRatio < 1.0) ip->setConstantModeWeight(v, weight * 3.0, constantModeRatio);
    return {mesh, bf, v, ip};
  }

  //! the largest difference, relative to the largest entry, between the Gram matrices from 1D factors and by quadrature
  double tensorVersusQuadrature(bool serendipity, const vector<int> &increments, Teuchos::FancyOStream &out, long long &tensorCells)
  {
    SeparableSetup setup = separableSetup(serendipity, increments);
    MeshPtr mesh = setup.mesh;
    auto ip = setup.ip;

    double maxDifference = 0, maxEntry = 0;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
    {
      DofOrderingPtr testOrdering = mesh->getElementType(cellID)->testOrderPtr;
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, true);
      const int n = testOrdering->totalDofs();
      Camellia::FieldContainer<double> tensor(1, n, n), quadrature(1, n, n);
      ip->setUseTensorAssembly(true);
      ip->computeInnerProductMatrix(tensor, testOrdering, basisCache);
      ip->setUseTensorAssembly(false);
      ip->computeInnerProductMatrix(quadrature, testOrdering, basisCache);
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
        {
          maxDifference = std::max(maxDifference, std::abs(tensor(0,i,j) - quadrature(0,i,j)));
          maxEntry = std::max(maxEntry, std::abs(quadrature(0,i,j)));
        }
    }
    tensorCells = ip->tensorCellCount();
    out << (serendipity ? "serendipity" : "hierarchical") << ", increments " << increments[0] << increments[1] << increments[2] << increments[3]
        << ": " << tensorCells << " cells from 1D factors, " << ip->fallbackCellCount() << " by quadrature; max |difference| / max |entry| = "
        << maxDifference / maxEntry << endl;
    return maxDifference / maxEntry;
  }
}

TEUCHOS_UNIT_TEST( SeparableTensorIP, MatchesQuadrature )
{
  // Box cells of different widths, with isotropic and anisotropic orders, hierarchical and serendipity test bases:
  // the Gram matrix from 1D factors agrees with quadrature to roundoff, and every cell takes the 1D path.
  for (bool serendipity : {false, true})
  {
    for (const vector<int> &increments : vector<vector<int>>{{0,0,0,0}, {2,0,1,0}, {1,1,1,1}})
    {
      long long tensorCells = 0;
      const double relativeDifference = tensorVersusQuadrature(serendipity, increments, out, tensorCells);
      TEST_COMPARE(relativeDifference, <, 1e-12);
      TEST_COMPARE(tensorCells, >, 0);
    }
  }
}

TEUCHOS_UNIT_TEST( SeparableTensorIP, ConstantModeWeight )
{
  // alpha ||v||^2 -> alpha ||v - mean v||^2 + ratio alpha |K| (mean v)^2 on each cell: the constant 1 (which the graph
  // terms do not see) has squared norm alpha |K| without the weakening and ratio alpha |K| with it, on the 1D-factor and
  // quadrature paths alike, and the Gram matrix stays positive definite.
  const int dim = 3, H1Order = 2, delta_k = 1;
  const double ratio = 1e-2;
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u", L2);
  VarPtr v = vf->testVar("v", HGRAD);
  vf->setBasisFamily(BasisFamily::serendipityBasisFamily());
  BFPtr bf = Teuchos::rcp(new BF(vf));
  bf->addTerm(u, v);
  auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>{{0.0, 0.5, 2.0}, {0.0, 0.1}, {-1.0, 0.0, 3.0}}));
  MeshPtr mesh = BoxGDA::mesh(topology, bf, H1Order, delta_k);
  mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u | 4u}});

  FunctionPtr alpha = Teuchos::rcp(new CellWidthProduct); // constant on each cell
  auto makeIP = [&](double constantModeRatio) {
    auto ip = SeparableTensorIP::ip();
    ip->addTerm(v->di(2));
    ip->addTerm(1.5 * v->di(1));
    ip->addTerm(0.65 * v->di(3));
    ip->addTerm(Function::sqrtFunction(alpha) * v);
    if (constantModeRatio < 1.0) ip->setConstantModeWeight(v, alpha, constantModeRatio);
    return ip;
  };
  auto massIP = SeparableTensorIP::ip();
  massIP->addTerm(v);
  for (bool tensorAssembly : {true, false})
  {
    auto baseIP = makeIP(1.0), weakenedIP = makeIP(ratio);
    baseIP->setUseTensorAssembly(tensorAssembly);
    weakenedIP->setUseTensorAssembly(tensorAssembly);
    double maxRelativeDeviation = 0;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
    {
      DofOrderingPtr testOrdering = mesh->getElementType(cellID)->testOrderPtr;
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, true);
      const int n = testOrdering->totalDofs();
      Camellia::FieldContainer<double> M(1, n, n), G(1, n, n), Gweak(1, n, n), m(1, n);
      massIP->computeInnerProductMatrix(M, testOrdering, basisCache);
      baseIP->computeInnerProductMatrix(G, testOrdering, basisCache);
      weakenedIP->computeInnerProductMatrix(Gweak, testOrdering, basisCache);
      (1.0 * v)->integrate(m, testOrdering, basisCache);
      // e: the coefficients of the constant 1 (M e = m, since m_i = int phi_i * 1)
      Camellia::FieldContainer<double> Mcell(n, n), mcell(n, 1), e(n, 1);
      for (int i=0; i<n; i++) { mcell(i,0) = m(0,i); for (int j=0; j<n; j++) Mcell(i,j) = M(0,i,j); }
      const int solveInfo = SerialDenseWrapper::solveSystemMultipleRHS(e, Mcell, mcell);
      TEST_EQUALITY(solveInfo, 0);
      double eGe = 0, eGweake = 0;
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++) { eGe += e(i,0) * G(0,i,j) * e(j,0); eGweake += e(i,0) * Gweak(0,i,j) * e(j,0); }
      Camellia::FieldContainer<double> alphaValues(1, basisCache->getWeightedMeasures().dimension(1));
      alpha->values(alphaValues, basisCache);
      double measure = 0;
      for (int p=0; p<basisCache->getWeightedMeasures().dimension(1); p++) measure += basisCache->getWeightedMeasures()(0,p);
      const double alphaMeasure = alphaValues(0,0) * measure;
      maxRelativeDeviation = std::max(maxRelativeDeviation, std::abs(eGe - alphaMeasure) / alphaMeasure);
      maxRelativeDeviation = std::max(maxRelativeDeviation, std::abs(eGweake - ratio * alphaMeasure) / (ratio * alphaMeasure));
      // positive definite
      Camellia::FieldContainer<double> Gcell(n, n), rhs(n, 1);
      for (int i=0; i<n; i++) { rhs(i,0) = 1.0; for (int j=0; j<n; j++) Gcell(i,j) = Gweak(0,i,j); }
      const int choleskyInfo = SerialDenseWrapper::solveSPDSystemLAPACKCholesky(rhs, Gcell); // (TEST_EQUALITY evaluates its arguments twice)
      TEST_EQUALITY(choleskyInfo, 0);
    }
    out << (tensorAssembly ? "1D factors" : "quadrature") << ": max relative deviation of the constant's norms " << maxRelativeDeviation << endl;
    TEST_COMPARE(maxRelativeDeviation, <, 1e-10);
  }
}

namespace
{
  //! reference mass and stiffness matrices on [-1,1] of the hierarchical H^1 basis of degree k (vertex functions, then
  //! integrated Legendre polynomials), by Gauss-Legendre quadrature exact for the products
  void hierarchicalMatrices(int k, vector<double> &M, vector<double> &K)
  {
    const int n = k + 1, Q = k + 3;
    auto legendre = [](int degree, double x, vector<double> &P, vector<double> &dP)
    {
      P.assign(degree + 1, 0.0); dP.assign(degree + 1, 0.0);
      P[0] = 1.0;
      if (degree > 0) P[1] = x;
      for (int i=2; i<=degree; i++) P[i] = ((2*i - 1) * x * P[i-1] - (i - 1) * P[i-2]) / i;
      for (int i=1; i<=degree; i++) dP[i] = i * P[i-1] + x * dP[i-1];
    };
    vector<double> P, dP, x(Q), w(Q);
    for (int i=0; i<Q; i++)
    {
      double z = cos(M_PI * (i + 0.75) / (Q + 0.5));
      for (int iteration=0; iteration<100; iteration++)
      {
        legendre(Q, z, P, dP);
        const double dz = P[Q] / dP[Q];
        z -= dz;
        if (abs(dz) < 1e-16) break;
      }
      legendre(Q, z, P, dP);
      x[i] = z;
      w[i] = 2.0 / ((1 - z * z) * dP[Q] * dP[Q]);
    }
    M.assign(n * n, 0.0); K.assign(n * n, 0.0);
    vector<double> value(n), derivative(n);
    for (int q=0; q<Q; q++)
    {
      legendre(k + 1, x[q], P, dP);
      value[0] = (1 - x[q]) / 2; derivative[0] = -0.5;
      value[1] = (1 + x[q]) / 2; derivative[1] =  0.5;
      for (int i=2; i<=k; i++)
      {
        value[i] = (P[i] - P[i-2]) / (2*i - 1);
        derivative[i] = (dP[i] - dP[i-2]) / (2*i - 1);
      }
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
        {
          M[i*n+j] += w[q] * value[i] * value[j];
          K[i*n+j] += w[q] * derivative[i] * derivative[j];
        }
    }
  }

  //! the dense Kronecker product of row-major factors, first direction fastest
  vector<double> kroneckerProduct(const vector<vector<double>> &factors, const vector<int> &n)
  {
    int N = 1;
    for (int nk : n) N *= nk;
    vector<double> product(size_t(N) * N);
    for (int I=0; I<N; I++)
      for (int J=0; J<N; J++)
      {
        int remainderI = I, remainderJ = J;
        double entry = 1.0;
        for (size_t k=0; k<n.size(); k++)
        {
          const int a = remainderI % n[k], b = remainderJ % n[k];
          remainderI /= n[k]; remainderJ /= n[k];
          entry *= factors[k][a * n[k] + b];
        }
        product[size_t(I) * N + J] = entry;
      }
    return product;
  }
}

TEUCHOS_UNIT_TEST( SeparableGramInverse, MatchesDenseKroneckerSum )
{
  // G = J (a M (x) ... (x) M + sum_k b_k M (x) .. K_k .. (x) M) on isotropic and anisotropic tensor spaces, with a zero
  // and a large derivative weight: apply() agrees with the dense matrix, and applyInverse() is a backward-stable solve.
  for (const vector<int> &degrees : vector<vector<int>>{{4,4,4,4}, {4,2,5,3}, {3,3}, {2,4,3}})
  {
    const int d = degrees.size();
    vector<vector<double>> M(d), K(d);
    vector<int> n(d);
    for (int k=0; k<d; k++)
    {
      hierarchicalMatrices(degrees[k], M[k], K[k]);
      n[k] = degrees[k] + 1;
    }
    SeparableGramInverse inverse(M, K);
    const int N = inverse.size();
    TEST_COMPARE(inverse.factorResidual(), <, 1e-12);

    SeparableGramInverse::Scales scales;
    scales.jacobian = 0.37;
    scales.valueWeight = 123.0;
    for (int k=0; k<d; k++) scales.derivativeWeights.push_back((k == 0) ? 4e5 : ((k == 1) ? 0.0 : 2.5 * (k + 1)));

    vector<double> G = kroneckerProduct(M, n);
    for (double &entry : G) entry *= scales.valueWeight;
    for (int k=0; k<d; k++)
    {
      vector<vector<double>> factors = M;
      factors[k] = K[k];
      vector<double> term = kroneckerProduct(factors, n);
      for (size_t i=0; i<G.size(); i++) G[i] += scales.derivativeWeights[k] * term[i];
    }
    for (double &entry : G) entry *= scales.jacobian;
    double normG = 0;
    for (int i=0; i<N; i++)
    {
      double rowSum = 0;
      for (int j=0; j<N; j++) rowSum += abs(G[size_t(i)*N+j]);
      normG = std::max(normG, rowSum);
    }

    const int vectorCount = 3;
    vector<double> x(size_t(N) * vectorCount);
    for (size_t i=0; i<x.size(); i++) x[i] = sin(1.0 + 7.3 * i);
    vector<double> Gx = x, y = x;
    inverse.apply(scales, Gx.data(), vectorCount);
    inverse.applyInverse(scales, y.data(), vectorCount);
    double applyError = 0, applyScale = 0, backwardError = 0, yNorm = 0;
    for (double value : y) yNorm = std::max(yNorm, abs(value));
    for (int v=0; v<vectorCount; v++)
      for (int i=0; i<N; i++)
      {
        double denseGx = 0, denseGy = 0;
        for (int j=0; j<N; j++)
        {
          denseGx += G[size_t(i)*N+j] * x[size_t(v)*N+j];
          denseGy += G[size_t(i)*N+j] * y[size_t(v)*N+j];
        }
        applyError = std::max(applyError, abs(denseGx - Gx[size_t(v)*N+i]));
        applyScale = std::max(applyScale, abs(denseGx));
        backwardError = std::max(backwardError, abs(denseGy - x[size_t(v)*N+i]) / (normG * yNorm));
      }
    out << "N = " << N << ": |apply - G x| / |G x| = " << applyError / applyScale << ", backward error of applyInverse = " << backwardError << endl;
    TEST_COMPARE(applyError / applyScale, <, 1e-12);
    TEST_COMPARE(backwardError, <, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( SeparableGramInverse, InverseInequalityConstants )
{
  // the largest generalized eigenvalue of (K, M) on P_k is the sharp 1D inverse-inequality constant Lambda_k (Table 1 of
  // the separable-graph-norm notes); it does not depend on the basis
  const vector<double> expected = {3.000, 15.000, 42.531, 95.059, 184.726, 326.151, 536.374, 834.862};
  for (int k=1; k<=8; k++)
  {
    vector<double> M, K;
    hierarchicalMatrices(k, M, K);
    SeparableGramInverse inverse({M}, {K});
    TEST_FLOATING_EQUALITY(inverse.eigenvalues(0).back(), expected[k-1], 1e-4);
    TEST_COMPARE(abs(inverse.eigenvalues(0).front()), <, 1e-10);
  }
}

TEUCHOS_UNIT_TEST( SeparableTensorIP, FastInverseInvertsGramMatrix )
{
  // On full tensor-product test bases, applyInverseInnerProductMatrix() applied to the rows of the (symmetric) Gram matrix
  // gives the identity, on every cell -- also with the constant-mode weight, whose rank-one correction it inverts by
  // Sherman-Morrison; on serendipity test bases it declines.
  for (double constantModeRatio : {1.0, 1e-2})
  for (bool serendipity : {false, true})
  {
    for (const vector<int> &increments : vector<vector<int>>{{0,0,0,0}, {2,0,1,0}})
    {
      SeparableSetup setup = separableSetup(serendipity, increments, constantModeRatio);
      double maxError = 0;
      int declined = 0, cells = 0;
      for (GlobalIndexType cellID : setup.mesh->getActiveCellIDsGlobal())
      {
        DofOrderingPtr testOrdering = setup.mesh->getElementType(cellID)->testOrderPtr;
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(setup.mesh, cellID, true);
        const int n = testOrdering->totalDofs();
        Camellia::FieldContainer<double> gram(1, n, n);
        setup.ip->computeInnerProductMatrix(gram, testOrdering, basisCache);
        Camellia::FieldContainer<double> product = gram; // row r: g_r, then x_r = G^{-1} g_r (should be e_r)
        cells++;
        if (!setup.ip->applyInverseInnerProductMatrix(product, testOrdering, basisCache))
        {
          declined++;
          bool unchanged = true;
          for (int i=0; i<n; i++)
            for (int j=0; j<n; j++) unchanged = unchanged && (product(0,i,j) == gram(0,i,j));
          TEST_ASSERT(unchanged);
          continue;
        }
        // normwise backward error max_r |G x_r - g_r| / (|G| |x_r|): G can be ill-conditioned in the hierarchical basis
        double normG = 0;
        for (int i=0; i<n; i++)
        {
          double rowSum = 0;
          for (int j=0; j<n; j++) rowSum += abs(gram(0,i,j));
          normG = std::max(normG, rowSum);
        }
        for (int r=0; r<n; r++)
        {
          double xNorm = 0, residual = 0;
          for (int j=0; j<n; j++) xNorm = std::max(xNorm, abs(product(0,r,j)));
          for (int i=0; i<n; i++)
          {
            double Gx = 0;
            for (int j=0; j<n; j++) Gx += gram(0,i,j) * product(0,r,j);
            residual = std::max(residual, abs(Gx - gram(0,r,i)));
          }
          maxError = std::max(maxError, residual / (normG * xNorm));
        }
      }
      out << (serendipity ? "serendipity" : "hierarchical") << ", increments " << increments[0] << increments[1] << increments[2]
          << increments[3] << ", constant-mode ratio " << constantModeRatio << ": " << cells - declined << " of " << cells << " cells inverted; backward error of G^{-1} = " << maxError << endl;
      if (serendipity)
      {
        TEST_EQUALITY(declined, cells);
      }
      else
      {
        TEST_EQUALITY(declined, 0);
        TEST_COMPARE(maxError, <, 1e-14);
        TEST_COMPARE(setup.ip->inverseCellCount(), >=, cells);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( SeparableTensorIP, LocalStiffnessFastInverseMatchesCholesky )
{
  // BF's factored-Cholesky local solve takes the fast inverse when the inner product offers it: the local stiffness
  // matrices and load vectors agree with the Cholesky path to roundoff, with and without the constant-mode weight.
  for (double constantModeRatio : {1.0, 1e-2})
  {
    SeparableSetup setup = separableSetup(false, {1,0,2,0}, constantModeRatio);
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(1.0 * setup.v + Function::xn(1) * setup.v->di(3));
    setup.bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);
    double maxDifference = 0, maxEntry = 0;
    for (GlobalIndexType cellID : setup.mesh->getActiveCellIDsGlobal())
    {
      ElementTypePtr elemType = setup.mesh->getElementType(cellID);
      const int trialCount = elemType->trialOrderPtr->totalDofs();
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(setup.mesh, cellID);
      BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(setup.mesh, cellID, true);
      Camellia::FieldContainer<double> stiffnessFast(1, trialCount, trialCount), rhsFast(1, trialCount);
      Camellia::FieldContainer<double> stiffnessCholesky(1, trialCount, trialCount), rhsCholesky(1, trialCount);
      const long long invertedBefore = setup.ip->inverseCellCount();
      setup.ip->setUseFastInverse(true);
      setup.bf->localStiffnessMatrixAndRHS(stiffnessFast, rhsFast, setup.ip, ipBasisCache, rhs, basisCache);
      TEST_COMPARE(setup.ip->inverseCellCount(), >, invertedBefore);
      setup.ip->setUseFastInverse(false);
      setup.bf->localStiffnessMatrixAndRHS(stiffnessCholesky, rhsCholesky, setup.ip, ipBasisCache, rhs, basisCache);
      for (int i=0; i<trialCount; i++)
      {
        for (int j=0; j<trialCount; j++)
        {
          maxDifference = std::max(maxDifference, abs(stiffnessFast(0,i,j) - stiffnessCholesky(0,i,j)));
          maxEntry = std::max(maxEntry, abs(stiffnessCholesky(0,i,j)));
        }
        maxDifference = std::max(maxDifference, abs(rhsFast(0,i) - rhsCholesky(0,i)));
        maxEntry = std::max(maxEntry, abs(rhsCholesky(0,i)));
      }
    }
    out << "constant-mode ratio " << constantModeRatio << ": max |fast - Cholesky| / max |entry| = " << maxDifference / maxEntry << endl;
    // (the two paths differ by the forward error of the solves, up to cond(G) times roundoff)
    TEST_COMPARE(maxDifference / maxEntry, <, 1e-8);
  }
}
