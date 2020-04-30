//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "QuadraturePointFunction.h"

#include "MeshFactory.h"
#include "PoissonFormulation.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

MeshPtr getMesh(const int xElements, const int yElements)
{
  // set up a dummy bilinear form; we just want some mesh object...
  int spaceDim = 2;
  bool useConformingTraces = true;
  PoissonFormulation poissonForm = PoissonFormulation(spaceDim, useConformingTraces);
  
  const int H1Order = 1;
  const int delta_k = 1;
  const double xWidth = 2.0;
  const double yWidth = 2.0;
  MeshPtr mesh = MeshFactory::quadMeshMinRule(poissonForm.bf(), H1Order, delta_k, xWidth, yWidth, xElements, yElements);
  
  return mesh;
}

TEUCHOS_UNIT_TEST( QuadraturePointFunction, PrecomputedMatchesDynamicallyComputed )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  
  FunctionPtr f = x + y;
  
  const int xElements = 5;
  const int yElements = 5;
  auto mesh = getMesh(xElements, yElements);
  
  const int quadratureEnrichment = 0;
  bool setValuesNow = true;
  FunctionPtr f_precomputed = QuadraturePointFunction::precomputedFunction(mesh, quadratureEnrichment, f, setValuesNow);
  
  FunctionPtr diff = f-f_precomputed;
  auto err_l2 = diff->l2norm(mesh);
  
  double tol = 1e-14;
  TEST_COMPARE(err_l2, <, tol);
}

TEUCHOS_UNIT_TEST( QuadraturePointFunction, BoundaryValuesWork )
{
  FunctionPtr meshSkeleton = Function::meshSkeletonCharacteristic();
  
  const int xElements = 5;
  const int yElements = 5;
  auto mesh = getMesh(xElements, yElements);
  
  const int quadratureEnrichment = 0;
  const bool setValuesNow = true;
  FunctionPtr meshSkeletonPrecomputed = QuadraturePointFunction::precomputedFunction(mesh, quadratureEnrichment, meshSkeleton, setValuesNow);
  
  FunctionPtr diff = meshSkeleton-meshSkeletonPrecomputed;
  auto err_l2 = diff->l2norm(mesh);
  
  double tol = 1e-14;
  TEST_COMPARE(err_l2, <, tol);
}
