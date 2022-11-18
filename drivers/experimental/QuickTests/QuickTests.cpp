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
