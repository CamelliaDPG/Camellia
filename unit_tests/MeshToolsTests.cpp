//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MeshToolsTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/16/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "BF.h"
#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "IP.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MPIWrapper.h"
#include "RHS.h"
#include "SimpleFunction.h"
#include "SpatialFilter.h"
#include "Solution.h"
#include "VarFactory.h"

#include <algorithm>

using namespace Camellia;
using namespace Intrepid;
using namespace std;

namespace
{
class Cone_U0 : public SimpleFunction<double>
{
  double _r; // cone radius
  double _h; // height
  double _x0, _y0; // center
  bool _usePeriodicData; // if true, for x > 0.5 we set x = x-1; similarly for y
public:
  Cone_U0(double x0 = 0, double y0 = 0.25, double r = 0.1, double h = 1.0, bool usePeriodicData = true)
  {
    _x0 = x0;
    _y0 = y0;
    _r = r;
    _h = h;
    _usePeriodicData = usePeriodicData;
  }
  double value(double x, double y)
  {
    if (_usePeriodicData)
    {
      if (x > 0.5)
      {
        x = x - 1;
      }
      if (y > 0.5) y = y - 1;
    }
    double d = sqrt( (x-_x0) * (x-_x0) + (y-_y0) * (y-_y0) );
    double u = std::max(0.0, _h * (1 - d/_r));

//      if (u != 0)
//        cout << "u(" << x << "," << y << ") = " << u << endl;

    return u;
  }
};

class InflowFilterForClockwisePlanarRotation : public SpatialFilter
{
  double _xLeft, _yBottom, _xRight, _yTop;
  double _xMiddle, _yMiddle;
public:
  InflowFilterForClockwisePlanarRotation(double leftBoundary_x, double rightBoundary_x,
                                         double bottomBoundary_y, double topBoundary_y,
                                         double rotationCenter_x, double rotationCenter_y)
  {
    _xLeft = leftBoundary_x;
    _yBottom = bottomBoundary_y;
    _xRight = rightBoundary_x;
    _yTop = topBoundary_y;
    _xMiddle = rotationCenter_x;
    _yMiddle = rotationCenter_y;
  }
  bool matchesPoint(double x, double y, double z)
  {
    double tol = 1e-14;
    bool inflow;
    if (abs(x-_xLeft)<tol)
    {
      inflow = (y > _yMiddle);
    }
    else if (abs(x-_xRight)<tol)
    {
      inflow = (y < _yMiddle);
    }
    else if (abs(y-_yBottom)<tol)
    {
      inflow = (x < _xMiddle);
    }
    else if (abs(y-_yTop)<tol)
    {
      inflow = (x > _xMiddle);
    }
    else
    {
      inflow = false; // not a boundary point at all...
    }
    return inflow;
  }
};

TEUCHOS_UNIT_TEST( MeshTools, MeshSlice_Polynomial )
{
  // Mesh slicing test with exact polynomial data
  MPIWrapper::CommWorld()->Barrier();
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y2 = Function::yn(2);
  FunctionPtr t = Function::zn(1);
  FunctionPtr u0 = x * y2 + t * x;

  // evaluate u0 at t=0, t=1:
  FunctionPtr u0_t0 = x * y2;
  FunctionPtr u0_t1 = x * y2 + x;

  int k = 2;
  int delta_k = 0; // Projection is exact, and we're not actually solving...
  int H1Order = k + 1;

  VarFactoryPtr varFactory = VarFactory::varFactory();
  // traces:
  VarPtr qHat = varFactory->fluxVar("\\widehat{q}");

  // fields:
  VarPtr u = varFactory->fieldVar("u", L2);

  // test functions:
  VarPtr v = varFactory->testVar("v", HGRAD);

  // bf does not really matter here (not solving); bf is used for the varFactory (the discretization)
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  bf->addTerm( u, v);
  bf->addTerm(qHat, v);

  // for this test, just make a single-element mesh whose geometry hugs the initial data
  double width = 2.0, height = 2.0;
  int numCells = 2, numTimeCells = 2;
  int horizontalCells = numCells, verticalCells = numCells;
  int depthCells = numTimeCells;
  double x0 = 0.0;
  double y0 = 0.0;
  double t0 = 0.0;

  const static double PI  = 3.141592653589793238462;
  double totalTime = 2.0 * PI; // for this test, make sure t=1 is in the domain

  vector<double> dimensions;
  dimensions.push_back(width);
  dimensions.push_back(height);
  dimensions.push_back(totalTime);

  vector<int> elementCounts(3);
  elementCounts[0] = horizontalCells;
  elementCounts[1] = verticalCells;
  elementCounts[2] = depthCells;

  vector<double> origin(3);
  origin[0] = x0;
  origin[1] = y0;
  origin[2] = t0;

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, origin);

  FunctionPtr sideParity = Function::sideParity();

  IPPtr ip;
  ip = bf->graphNorm();

  BCPtr bc = BC::bc();
  bc->addDirichlet(qHat, SpatialFilter::matchingZ(t0), u0);

  MeshPtr initialMesh = mesh;

  double tZero = 0.0, tOne = 1.0;
  map<GlobalIndexType,GlobalIndexType> cellMap_t0, cellMap_t1;
  MeshPtr meshSlice_t0 = MeshTools::timeSliceMesh(initialMesh, tZero, cellMap_t0, H1Order);
  MeshPtr meshSlice_t1 = MeshTools::timeSliceMesh(initialMesh, tOne,  cellMap_t1, H1Order);

//    int rank = Teuchos::GlobalMPISession::getRank();
//    ostringstream rankStr;
//    rankStr << "rank " << rank << " cellMap_t0";
//    Camellia::print(rankStr.str(), cellMap_t0);
//    rankStr.str("");
//    rankStr << "rank " << rank << " cellMap_t1";
//    Camellia::print(rankStr.str(), cellMap_t1);

  SolutionPtr soln = Solution::solution(mesh,bc,RHS::rhs(), ip);

  // project u0 onto the whole spacetime mesh (i.e. it'll look like the initial value is a steady solution)
  std::map<int, FunctionPtr> functionMap;
  functionMap[u->ID()] = u0;
  const int solutionOrdinal = 0;
  soln->projectOntoMesh(functionMap, solutionOrdinal);

  FunctionPtr u_spacetime = Function::solution(u, soln);

  FunctionPtr sliceFunction_t0 = MeshTools::timeSliceFunction(mesh, cellMap_t0, u_spacetime, tZero);
  FunctionPtr sliceFunction_t1 = MeshTools::timeSliceFunction(mesh, cellMap_t1, u_spacetime, tOne);

  double tol = 1e-14;
  // expectation is that on the slice mesh, the sliceFunction matches u0
  double diff_l2_t0 = (u0_t0 - sliceFunction_t0)->l2norm(meshSlice_t0);
  TEST_COMPARE(diff_l2_t0, <, tol);

  double diff_l2_t1 = (u0_t1 - sliceFunction_t1)->l2norm(meshSlice_t1);
  TEST_COMPARE(diff_l2_t1, <, tol);
}

TEUCHOS_UNIT_TEST( MeshTools, MeshSliceTimeZero_Cone )
{
  // trying out mesh slicing, just confirming that the slice at time zero matches prescribed initial condition

  // doing this for the convecting cone problem, one of the early test cases for space-time meshes

  double cone_x0 = 0, cone_y0 = 0.25, r = 0.1, h = 1.0;
  FunctionPtr u0 = Teuchos::rcp( new Cone_U0(cone_x0, cone_y0, r, h, false) );

  int k = 2;
  int delta_k = 5; // because the projection isn't exact, and this will enhance the cubature degree
  int H1Order = k + 1;

  VarFactoryPtr varFactory = VarFactory::varFactory();
  // traces:
  VarPtr qHat = varFactory->fluxVar("\\widehat{q}");

  // fields:
  VarPtr u = varFactory->fieldVar("u", L2);

  // test functions:
  VarPtr v = varFactory->testVar("v", HGRAD);

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);

  FunctionPtr c = Function::vectorize(y-0.5, 0.5-x, Function::constant(1.0));
  FunctionPtr n = Function::normal();

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  bf->addTerm( u, c * v->grad());
  bf->addTerm(qHat, v);

  // for this test, just make a single-element mesh whose geometry hugs the initial data
  double width = 2.0 * r, height = 2.0 * r;
  int numCells = 1, numTimeCells = 1;
  int horizontalCells = numCells, verticalCells = numCells;
  int depthCells = numTimeCells;
  double x0 = cone_x0 - r;
  double y0 = cone_y0 - r;
  double t0 = 0;

  const static double PI  = 3.141592653589793238462;
  double totalTime = 2.0 * PI;
  // want the number of grid points in temporal direction to be about 2000.  The temporal length is 2 * PI
  int numTimeSlabs = (int) 2000 / k;
  double timeLengthPerSlab = totalTime / numTimeSlabs;

  SpatialFilterPtr inflowFilter  = Teuchos::rcp( new InflowFilterForClockwisePlanarRotation (x0,x0+width,y0,y0+height,0.5,0.5));

  vector<double> dimensions;
  dimensions.push_back(width);
  dimensions.push_back(height);
  dimensions.push_back(timeLengthPerSlab);

  vector<int> elementCounts(3);
  elementCounts[0] = horizontalCells;
  elementCounts[1] = verticalCells;
  elementCounts[2] = depthCells;

  vector<double> origin(3);
  origin[0] = x0;
  origin[1] = y0;
  origin[2] = t0;

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, origin);

  FunctionPtr sideParity = Function::sideParity();

  IPPtr ip;
  ip = bf->graphNorm();

  double tol = 1e-2; // can't expect exact agreement because cone is not polynomial!
  // sanity checks on u0:
  // - at center of cone, u0 = h:
  TEST_FLOATING_EQUALITY(h, u0->evaluate(cone_x0, cone_y0), tol);
  // - at r/2 away from center of cone, u0 = h/2
  TEST_FLOATING_EQUALITY(h / 2, u0->evaluate(cone_x0 + r / 2, cone_y0), tol);
  // - at r + epsilon away from center of cone, u0 = 0
  TEST_ASSERT(abs(u0->evaluate(cone_x0, cone_y0 + r + 1e-15)) < tol);

  // check that the same things hold when we use a 3rd argument (t0=0)
  // - at center of cone, u0 = h:
  TEST_FLOATING_EQUALITY(h, u0->evaluate(cone_x0, cone_y0, t0), tol);
  // - at r/2 away from center of cone, u0 = h/2
  TEST_FLOATING_EQUALITY(h / 2, u0->evaluate(cone_x0 + r / 2, cone_y0, t0), tol);
  // - at r + epsilon away from center of cone, u0 = 0
  TEST_ASSERT(abs(u0->evaluate(cone_x0, cone_y0 + r + 1e-15, t0)) < tol);

  BCPtr bc = BC::bc();
  bc->addDirichlet(qHat, inflowFilter, Function::zero()); // zero BCs enforced at the inflow boundary.
  bc->addDirichlet(qHat, SpatialFilter::matchingZ(t0), u0);

  MeshPtr initialMesh = mesh;

  map<GlobalIndexType,GlobalIndexType> cellMap;
  MeshPtr meshSlice = MeshTools::timeSliceMesh(initialMesh, 0, cellMap, H1Order);

  SolutionPtr soln = Solution::solution(mesh,bc,RHS::rhs(), ip);

  // project u0 onto the whole spacetime mesh (i.e. it'll look like the initial value is a steady solution)
  std::map<int, FunctionPtr> functionMap;
  functionMap[u->ID()] = u0;
  const int solutionOrdinal = 0;
  soln->projectOntoMesh(functionMap, solutionOrdinal);

  FunctionPtr u_spacetime = Function::solution(u, soln);

  double l2_u_spacetime = u_spacetime->l2norm(mesh);
  TEST_ASSERT(l2_u_spacetime > 0);

  // it looks like u_spacetime is 0:
  // TODO: add tests against Projector, for 3D / space-time meshes...

  double tZero = 0.0;
  FunctionPtr sliceFunction = MeshTools::timeSliceFunction(mesh, cellMap, u_spacetime, tZero);

//    cout << "u_spacetime(cone_x0,cone_y0,t0) = " << u_spacetime->evaluate(mesh,cone_x0,cone_y0,t0) << endl;
//    cout << "u0(cone_x0,cone_y0) = " << u0->evaluate(meshSlice, cone_x0, cone_y0) << endl;
//    cout << "sliceFunction(cone_x0,cone_y0) = " << sliceFunction->evaluate(meshSlice, cone_x0, cone_y0) << endl;

  // expectation is that on the slice mesh, the sliceFunction matches u0
  double diff_l2 = (u0 - sliceFunction)->l2norm(meshSlice);

//    HDF5Exporter exporter(meshSlice, "MeshToolsTests", "/tmp");
//
//    vector<FunctionPtr> functions;
//    functions.push_back(Function::xn());
//    functions.push_back(u0);
//    functions.push_back(sliceFunction);
//    functions.push_back(u0 - sliceFunction);
//
//    vector<string> names;
//    names.push_back("x");
//    names.push_back("u_0");
//    names.push_back("sliceFunction");
//    names.push_back("diff");
//
//    exporter.exportFunction(functions, names);

  TEST_COMPARE(diff_l2, <, tol);
}
} // namespace
