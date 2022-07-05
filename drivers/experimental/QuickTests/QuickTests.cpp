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

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

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
