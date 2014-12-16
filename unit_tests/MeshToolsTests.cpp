//
//  MeshToolsTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/16/14.
//
//

// the following copied from the Teuchos Unit tests documentation.  Just trying it out.

#include "Teuchos_UnitTestHarness.hpp"

#include "Function.h"
#include "VarFactory.h"

#include "SpatialFilter.h"
#include "BF.h"
#include "IP.h"
#include "BC.h"

#include "Solution.h"

#include "MeshTools.h"
#include "Mesh.h"
#include "MeshFactory.h"

#include "CamelliaDebugUtility.h"

#include "HDF5Exporter.h"

namespace {
  class Cone_U0 : public SimpleFunction {
    double _r; // cone radius
    double _h; // height
    double _x0, _y0; // center
    bool _usePeriodicData; // if true, for x > 0.5 we set x = x-1; similarly for y
  public:
    Cone_U0(double x0 = 0, double y0 = 0.25, double r = 0.1, double h = 1.0, bool usePeriodicData = true) {
      _x0 = x0;
      _y0 = y0;
      _r = r;
      _h = h;
      _usePeriodicData = usePeriodicData;
    }
    double value(double x, double y) {
      if (_usePeriodicData) {
        if (x > 0.5) {
          x = x - 1;
        }
        if (y > 0.5) y = y - 1;
      }
      double d = sqrt( (x-_x0) * (x-_x0) + (y-_y0) * (y-_y0) );
      double u = max(0.0, _h * (1 - d/_r));
      
      return u;
    }
  };
  
  class InflowFilterForClockwisePlanarRotation : public SpatialFilter {
    double _xLeft, _yBottom, _xRight, _yTop;
    double _xMiddle, _yMiddle;
  public:
    InflowFilterForClockwisePlanarRotation(double leftBoundary_x, double rightBoundary_x,
                                           double bottomBoundary_y, double topBoundary_y,
                                           double rotationCenter_x, double rotationCenter_y) {
      _xLeft = leftBoundary_x;
      _yBottom = bottomBoundary_y;
      _xRight = rightBoundary_x;
      _yTop = topBoundary_y;
      _xMiddle = rotationCenter_x;
      _yMiddle = rotationCenter_y;
    }
    bool matchesPoint(double x, double y, double z) {
      double tol = 1e-14;
      bool inflow;
      if (abs(x-_xLeft)<tol) {
        inflow = (y > _yMiddle);
      } else if (abs(x-_xRight)<tol) {
        inflow = (y < _yMiddle);
      } else if (abs(y-_yBottom)<tol) {
        inflow = (x < _xMiddle);
      } else if (abs(y-_yTop)<tol) {
        inflow = (x > _xMiddle);
      } else {
        inflow = false; // not a boundary point at all...
      }
      return inflow;
    }
  };
  
  TEUCHOS_UNIT_TEST( MeshTools, MeshSliceTimeZero )
  {
    // trying out mesh slicing, just confirming that the slice at time zero matches prescribed initial condition
    
    // doing this for the convecting cone problem, one of the early test cases for space-time meshes
    
    int k = 4;
    int delta_k = 0; // since we're just doing projections, and this will make for a trial x trial element...
    int H1Order = k + 1;
    
    VarFactory varFactory;
    // traces:
    VarPtr qHat = varFactory.fluxVar("\\widehat{q}");
    
    // fields:
    VarPtr u = varFactory.fieldVar("u", L2);
    
    // test functions:
    VarPtr v = varFactory.testVar("v", HGRAD);
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    
    FunctionPtr c = Function::vectorize(y-0.5, 0.5-x, Function::constant(1.0));
    FunctionPtr n = Function::normal();
    
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    
    bf->addTerm( u, c * v->grad());
    bf->addTerm(qHat, v);
    
    double width = 2.0, height = 2.0;
    int numCells = 2, numTimeCells = 1;
    int horizontalCells = numCells, verticalCells = numCells;
    int depthCells = numTimeCells;
    double x0 = -0.5; double y0 = -0.5;
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
    
    FunctionPtr u0 = Teuchos::rcp( new Cone_U0(0.0, 0.25, 0.1, 1.0, false) );
    
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
    soln->projectOntoMesh(functionMap);
    
    FunctionPtr u_spacetime = Function::solution(u, soln);
    
    double tZero = 0.0;
    FunctionPtr sliceFunction = MeshTools::timeSliceFunction(mesh, cellMap, u_spacetime, tZero);
    
    // expectation is that on the slice mesh, the sliceFunction matches u0
    double diff_l2 = (u0 - sliceFunction)->l2norm(meshSlice);
    
    double tol = 1e-15;
    TEST_COMPARE(diff_l2, <, tol);
  }
} // namespace