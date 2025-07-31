//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  QuickTests.cpp
//
//  Created by Nate Roberts on 5/10/2022.
//
//

/*
 The idea here is just to have a driver that we can add experimental code to
 to try something out in a quick-and-dirty fashion.  E.g. the PrintMatrices
 test below just prints out some matrices associated with Appendix A in our
 journal-submitted manuscript,
 "Combining DPG in space with DPG time-marching scheme for the transient advection-reaction equation."
 
 */

#include "Camellia.h"

#include "MaxwellianFunction.h"

// #include "ActivationFunction.hpp"
//#include "GNN.hpp"
//#include "MeshPhysicalQuadrature.hpp"
//#include "NNOptimizer.hpp"
#include "VlasovBackwardEulerFormulation.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

//! Compute (n choose k)
int choose(const int &n, const int &k)
{
  int result = 1;
  if (n - k < k)
  {
    return choose(n, n-k);
  }
  else
  {
    for (int i=1; i<=k; i++)
    {
      result *= (n - k + i);
    }
    for (int i=2; i<=k; i++)
    {
      result /= i;
    }
  }
  return result;
}

//TEUCHOS_UNIT_TEST( GNN, AinsworthExample1 )
//{
//  // function fitting example: first four terms of Fourier series for square wave
//  FunctionPtr f;
//  {
//    FunctionPtr sin_x     = Teuchos::rcp(new Sin_ax(1.0));
//    FunctionPtr sin_3pi_x = Teuchos::rcp(new Sin_ax(3.0 * M_PI));
//    FunctionPtr sin_5pi_x = Teuchos::rcp(new Sin_ax(5.0 * M_PI));
//    FunctionPtr sin_7pi_x = Teuchos::rcp(new Sin_ax(7.0 * M_PI));
//    f = sin_x + sin_3pi_x / 3.0 + sin_5pi_x / 5.0 + sin_7pi_x / 7.0;
//  }
//  const int spaceDim = 1;
//  
//  VarFactoryPtr vf = VarFactory::varFactory();
//  auto u = vf->testVar("u", HGRAD);
//  
//  RHSPtr rhs = RHS::rhs();
//  rhs->addTerm(f * u);
//  
//  IPPtr ip = IP::ip();
//  ip->addTerm(u); // (u,v)_L^2
//  
//  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(0.0, 1.0, 20); // 20 elements in initial mesh
//  
//  auto quadrature = Teuchos::rcp( new MeshPhysicalQuadrature(meshTopo) );
//  
//  double quadratureRelTol = 1e-6;
//  double quadratureAbsTol = 1e-14;
//  
//  auto gnn = Teuchos::rcp(new GNN(spaceDim, ip, rhs, {u}));
//  
//  // let's try the one with a fixed width of n_i = 100
//  int numGalerkinIterations = 6;
//  for (int i=0; i<numGalerkinIterations; i++)
//  {
//    const int width = 100;
//    const double tol = 1e-6;
//    const double alpha = 0.01; // learning rate for this problem is not specified by Ainsworth and Dong
//    
//    const double beta_i = 1. + 3 * (i-1);
//    auto sigma = ActivationFunction::tanhFunction(beta_i);
//    
//    DeviceView<double> W = Allocation<double>::allocateDeviceView({width,spaceDim}, "W");
//    DeviceView<double> b = Allocation<double>::allocateDeviceView({width}, "b");
//    
//    auto WHost = Kokkos::create_mirror_view(W);
//    auto bHost = Kokkos::create_mirror_view(b);
//    for (int j=0; j<width; j++)
//    {
//      WHost(j,0) = 1.0;
//      bHost(j)   = -j * (1.0 / width);
//    }
//    
//    Kokkos::deep_copy(W,WHost);
//    Kokkos::deep_copy(b,bHost);
//    
//    auto slnn = Teuchos::rcp( new SingleLayerNN(spaceDim, sigma, W, b));
//    
//    auto optimizer = NNOptimizer::gradientDescent(gnn, slnn, alpha); // TODO: switch to adam, once we've implemented
//    
//    int maxEpochsPerOptimizeCall = 1;
//    int maxEpochs = 1000;
//    for (int j=0; j<maxEpochs; j++)
//    {
//      optimizer->optimize(tol, maxEpochsPerOptimizeCall, quadrature, quadratureRelTol, quadratureAbsTol);
//      
//      double loss = optimizer->loss();
//      
//      cout << "epoch " << j << ", loss: " << loss << std::endl;
//      
//      if (loss < tol)
//        break;
//    }
//    gnn->augmentBasis(slnn);
//    gnn->solve(quadrature, quadratureRelTol, quadratureAbsTol);
//    
//    auto u_soln = gnn->solutionFunction(u);
//    auto errFxn = u_soln - f;
//    
//    double err_l2_squared = quadrature->adaptivelyIntegrateOverVolume(errFxn * errFxn, quadratureRelTol, quadratureAbsTol);
//    double err_l2 = sqrt(err_l2_squared);
//    cout << "Galerkin iteration i, L^2 error: " << err_l2 << std::endl;
//  }
//  
//}

TEUCHOS_UNIT_TEST( Hexahedron, PrintSides )
{
  auto meshTopo = MeshFactory::rectilinearMeshTopology({1,1,1}, {1,1,1});
  
  meshTopo->printAllEntities();
}

#include <sys/stat.h>
#include <sys/types.h>

#include <cfenv>        // for std::fenv_t
#include <cmath>        // for sqrt()
#include <csignal>      // for signal()
#include <fenv.h>       // for fegetenv(), fesetenv()

void fpe_signal_handler(int /*signal*/) {
  std::cerr << "Floating point exception!\n";
  exit(1);
}

TEUCHOS_UNIT_TEST( Ternary, AssignParens )
{
  std::fenv_t env;
  fegetenv(&env);
  env.__fpcr = env.__fpcr | __fpcr_trap_invalid | __fpcr_trap_divbyzero | __fpcr_trap_overflow;
  fesetenv(&env);
  signal(SIGFPE, fpe_signal_handler);
  
  double actual = 0.0;
  double desired = 0.0;
  double correction;
  
  correction = ((desired == 0) ? 1.0 : actual/desired);
  
  TEST_EQUALITY(1, correction);
}

TEUCHOS_UNIT_TEST( TimeMarchingMatrices, PrintMatrices )
{
  VarFactoryPtr vf = VarFactory::varFactory();
  auto u = vf->fieldVar("u");
  auto v = vf->testVar("v", HGRAD);
  
  IPPtr ip = IP::ip();
  ip->addTerm(v);
  ip->addTerm(v->dx());
  
  auto bf_K = BF::bf(vf);
  auto bf_M = BF::bf(vf);
  
  bf_K->addTerm(u, v->dx());
  bf_M->addTerm(u, v);
  
  auto meshTopo = MeshFactory::intervalMeshTopology(0, 1, 1);
  
  const int H1Order = 1;
  const int delta_k = 1;
  
  auto mesh_K = MeshFactory::minRuleMesh(meshTopo, bf_K, H1Order, delta_k);
  auto mesh_M = MeshFactory::minRuleMesh(meshTopo, bf_M, H1Order, delta_k);
  
  const int numElements = 1;
  GlobalIndexType cellID = 0;
  auto elemType = mesh_K->getElementType(cellID);
  FieldContainer<double> K(numElements,elemType->testOrderPtr->totalDofs(),elemType->trialOrderPtr->totalDofs());
  FieldContainer<double> M(numElements,elemType->testOrderPtr->totalDofs(),elemType->trialOrderPtr->totalDofs());
  FieldContainer<double> G(numElements,elemType->testOrderPtr->totalDofs(),elemType->testOrderPtr->totalDofs());
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh_K, cellID);
  bf_K->stiffnessMatrix(K, elemType, basisCache->getCellSideParities(), basisCache);
  bf_M->stiffnessMatrix(M, elemType, basisCache->getCellSideParities(), basisCache);
  ip->computeInnerProductMatrix(G, elemType->testOrderPtr, basisCache);
  
  using std::cout;
  cout << "K:\n" << K;
  cout << "M:\n" << M;
  cout << "G:\n" << G;
}

//template<typename U, typename V>
//constexpr bool same = std::is_same_v<U, V>;
//
//using type1 = std::basic_string<char>;
//using type2 = std::basic_string<char, std::char_traits<char>, std::allocator<char>>;
//
//static_assert(same<type1,type2>, "type1 and type2 should match!");

template<typename U, typename V>
constexpr bool same = std::is_same_v<U, V>;

using string1 = std::basic_string<char, std::char_traits<char>, std::allocator<char> >;
using string2 = std::basic_string<char>;
using string3 = __remove_cv(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >);

static_assert(same<string1,string2>, "string1 and string2 should match!");
static_assert(same<string1,string3>, "string1 and string3 should match!");

using int1 = int;
using int2 = __remove_cv(int);

static_assert(same<int1,int2>, "int1 and int2 should match!");

using map1 = std::map<std::pair<__remove_cv(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >), __remove_cv(int)>, Kokkos::View<double****, DeviceType> >;
using map2 = std::map<std::pair<string3, int2>, Kokkos::View<double****, DeviceType> >;
using map3 = std::map<std::pair<std::basic_string<char>, int>, Kokkos::View<double****, DeviceType> >;

static_assert(same<map1,map2>, "map1 and map2 should match!");
static_assert(same<map1,map3>, "map1 and map3 should match!");

TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, GramSemiGraphOutput)
{
  const int xElements = 4;
  const int vElements = 4;
  const int H1Order   = 1;
  
  const bool usePeriodicBCs = false;
  const bool useHomogeneousPotentialBCs = true;
  const bool useCrankNicolson = true;

  const double q = 1;
  double eps0    = 1;
  double m       = 1;
  
  const int delta_k = 2;
  const double x0 = 0.0;
  const double xWidth = 1.0;
  const double v0 = -1.0;
  const double vWidth = 2.0;
  const double dt = 0.001;
  
  auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
  auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
  std::vector<double> vxVertices;
  double vx = v0;
  for (int i=0; i<=vElements; i++)
  {
    vx = v0 + (i * vWidth) / vElements;
    vxVertices.push_back(vx);
  }
  const bool useSerendipity = false;
  const int maxMomentNumber = 0;
  VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber,
                                      useCrankNicolson);
  form.setRunQuietly(true);
  
  auto vlasovBF = form.vlasovMesh()->bilinearForm();
  
  auto semiGraphIP = vlasovBF->graphNorm(1.0);
  
  auto elementType0 = form.vlasovMesh()->getElement(0)->elementType();
  auto basisCache_cell0 = BasisCache::basisCacheForCell(form.vlasovMesh(), 0);
  int numTestDofs = elementType0->testOrderPtr->totalDofs();
  FieldContainer<double> gramMatrix(1,numTestDofs,numTestDofs);
  semiGraphIP->computeInnerProductMatrix(gramMatrix, elementType0->testOrderPtr, basisCache_cell0);
  
  gramMatrix.resize(numTestDofs,numTestDofs);
  cout << "numTestDofs: " << numTestDofs << std::endl;
  SerialDenseWrapper::writeMatrixToMatlabFile("/tmp/G.dat", gramMatrix);
}

TEUCHOS_UNIT_TEST( HCURLCounts, Counts )
{
  for (int p=1; p<10; p++)
  {
    int total = 0;
    for (int ijk_sum = 2; ijk_sum <= p-1; ijk_sum++)
    {
      for (int i=0; i < ijk_sum; i++)
      {
        for (int j=0; j < ijk_sum-i; j++)
        {
          const int k = ijk_sum - i - j;
          total++;
        }
      }
    }
    cout << p << "\t" << total << endl;
    
//    const int expectedCardinality = choose(p+1, 3) - 1;
    // equivalently:
    const int expectedCardinality = (p > 2) ? (p-1)*p*(p+1)/6 - 1 : 0;
    
    TEST_EQUALITY(total, expectedCardinality);
  }
}

TEUCHOS_UNIT_TEST( SerendipityCounts, Counts )
{
  for (int d=1; d<8; d++)
  {
    std::array<int, 8> cardinalities;
    for (int p=1; p<8; p++)
    {
      // closed form for H^1 -- see Juno et al. doi:10.1016/j.jcp.2017.10.009, Eqn. (83).
      int i_max = std::min(d,p/2);
      int n = 0;
      for (int i = 0; i <= i_max; i++)
      {
        int d_choose_i = choose(d, i);
        int p_minus_i_choose_i = choose(p - i, i);
        int two_to_the_d_minus_i = 1 << (d-i);
        n += two_to_the_d_minus_i * d_choose_i * p_minus_i_choose_i;
      }
      cardinalities[p-1] = n;
    }
    std::cout << d << " & ";
    for (int p=1; p<8; p++)
    {
      std::cout << cardinalities[p-1];
      if (p < 7)
      {
        std::cout << " & ";
      }
    }
    std::cout << "\\\\\n";
  }
}

TEUCHOS_UNIT_TEST( WedgeInfo, Print )
{
  auto wedge = CellTopology::wedge();
  vector<double> v0 {0,0,0};
  vector<double> v1 {1,0,0};
  vector<double> v2 {0,1,0};
  vector<double> v3 {0,0,1};
  vector<double> v4 {1,0,1};
  vector<double> v5 {0,1,1};
  vector<vector<double>> vertices {v0,v1,v2,v3,v4,v5};
  vector<IndexType> element0Vertices {0,1,2,3,4,5};
  vector<vector<IndexType>> elementVertices { element0Vertices };
  vector<CellTopoPtr> cellTopos {wedge};
  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
  
  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(meshGeometry) );
  
  meshTopo->printAllEntities();
}

TEUCHOS_UNIT_TEST( SyntheticEmissionData, Print )
{
  // output some non-dimensional emission data
  double u0 = 1.0;
  double sigma_squared = 0.25;
  double n = 1.0;
  auto f_M = MaxwellianFunction::maxwellian(1, 1, {u0}, sigma_squared, n);
  
  // non-dimensionalization also applies to temporal range: go from 0 to 1.
  double t = 0;
  double t_final = 1.0;
  double t_ramp  = 0.25; // ramp up to full emission between t = 0 and t = 0.25
  const int numTimeSteps = 100;
  double v_min   = -1.0;
  double v_width = 2.0;
  const int v_divisions = 100;
  
  std::cout << std::setprecision(16);
  
  for (int timeStep = 0; timeStep <= numTimeSteps; timeStep++)
  {
    t = timeStep / (double) numTimeSteps;
    const double rampMultiplier = (t < t_ramp) ? t / t_ramp : 1.0;
    for (int vOrdinal = 0; vOrdinal<=v_divisions; vOrdinal++)
    {
      const double x = 0.0; // maxwellian should be independent of x
      const double v = v_min + (v_width * vOrdinal) / v_divisions;
      double f    = f_M->evaluate(x, v);
      double f_em = rampMultiplier * f;
      
      std::cout << t << "\t" << v << "\t" << f_em << "\n";
    }
  }
}
