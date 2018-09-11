//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  IPTests
//  Camellia
//
//  Created by Nate Roberts on 9/11/18.
//
//

#include "IP.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RieszRep.h"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
TEUCHOS_UNIT_TEST( IP, ImposeBCs )
{
  // a test for the new feature in which we can impose (homogeneous) BCs on the test space
  // We take the RieszRep test that uses the L^2 norm on the test space, and modify it to
  // impose a zero condition on the mesh boundary.  We then test that the computed Riesz representation
  // is zero on the mesh boundary.
  int spaceDim = 2;
  bool conformingTraces = true;

  PoissonFormulation form(spaceDim,conformingTraces);
  BFPtr bf = form.bf();

  IPPtr ip = bf->l2Norm();

  FunctionPtr weight = Function::xn(1);
  LinearTermPtr lt = weight * form.v();

  int H1Order = 1;
  // make a unit square mesh:
  vector<double> dimensions(2,1.0);
  vector<int> elementCounts(2,1);
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);

  RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, lt) );

  ip->imposeHomogeneousBCs(form.v(), SpatialFilter::allSpace());
  
  rieszRep->computeRieszRep();

  FunctionPtr meshBoundaryRestriction = Function::meshBoundaryCharacteristic();
  
  FunctionPtr repFxnOnBoundary = RieszRep::repFunction(form.v(), rieszRep) * meshBoundaryRestriction;

  FunctionPtr expectedRepFxnOnBoundary = Function::zero() * meshBoundaryRestriction;

  double err = (repFxnOnBoundary - expectedRepFxnOnBoundary)->l2norm(mesh);

  double tol = 1e-14;
  TEST_COMPARE(err,<,tol);
}
} // namespace
