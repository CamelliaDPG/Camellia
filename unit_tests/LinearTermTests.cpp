//
//  LinearTermTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/14/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Function.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace {
  MeshPtr singleElementSpaceTimeMesh(int spaceDim, int H1Order)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    int delta_k = 1;
    vector<int> H1OrderVector(2);
    H1OrderVector[0] = H1Order;
    H1OrderVector[1] = H1Order;
    MeshPtr mesh = Teuchos::rcp( new Mesh(spaceTimeMeshTopo, form.bf(), H1OrderVector, delta_k) ) ;
    return mesh;
  }

  void testSpaceTimeNonzeroTimeDerivative(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // here, we simply test that v->dt() gives something nonzero

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    VarPtr v = form.v();
    FunctionPtr f = Function::xn(1);

    LinearTermPtr lt = 1.0 * v->dt();

    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
    double norm = lt->computeNorm(form.bf()->graphNorm(), mesh); // should be > 0

//    cout << "spaceDim, " << spaceDim << "; norm " << norm << endl;

    TEST_COMPARE(norm, >, 1e-14);
  }

  void testSpaceTimeIntegrationByPartsInTime(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // we consider df/dt where f = x.
    // Integrating by parts 0 = (df/dt, v) = (-f, v->dt()) + < f, v * n->t() >

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    VarPtr v = form.v();
    FunctionPtr f = Function::xn(1);

    FunctionPtr n_xt = Function::normalSpaceTime();

    LinearTermPtr lt = -f * v->dt() + (f * v) * n_xt->t();

    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
    double norm = lt->computeNorm(form.bf()->graphNorm(), mesh); // should be 0

    TEST_COMPARE(norm, <, 1e-14);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_1D )
  {
    testSpaceTimeIntegrationByPartsInTime(1,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_2D )
  {
    testSpaceTimeIntegrationByPartsInTime(2,out,success);
  }

//  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_3D )
//  {
//    testSpaceTimeIntegrationByPartsInTime(3,out,success);
//  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_1D )
  {
    testSpaceTimeNonzeroTimeDerivative(1,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_2D )
  {
    testSpaceTimeNonzeroTimeDerivative(2,out,success);
  }

//  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_3D )
//  {
//    testSpaceTimeNonzeroTimeDerivative(3,out,success);
//  }
} // namespace
