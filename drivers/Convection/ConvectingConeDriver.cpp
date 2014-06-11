#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Solver.h"

#include "CamelliaConfig.h"

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

class Cone_U0 : public SimpleFunction {
  double _r; // cone radius
  double _h; // height
  double _x0, _y0; // center
public:
  Cone_U0(double x0 = 0, double y0 = 0.25, double r = 0.1, double h = 1.0) {
    _x0 = x0;
    _y0 = y0;
    _r = r;
    _h = h;
  }
  double value(double x, double y) {
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
  bool matchesPoint(double x, double y) {
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

int main(int argc, char *argv[]) {
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  bool useCondensedSolve = true; // condensed solve not yet compatible with minimum rule meshes
  
  int numGridPoints = 32; // in x,y -- idea is to keep the overall order of approximation constant
  int k = 4; // poly order for u
  double theta = 0.5;
  int numTimeSteps = 2000;
  int numCells = -1; // in x, y (-1 so we can set a default if unset from the command line.)
  int numFrames = 50;
  int delta_k = 2;   // test space enrichment: should be 2 for 2D
  bool convertSolutionsToVTK = false; // when true assumes we've already run with precisely the same options, except without VTK support (so we have a bunch of .soln files)
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("numCells",&numCells,"number of cells in x and y directions");
  cmdp.setOption("theta",&theta,"theta weight for time-stepping");
  cmdp.setOption("numTimeSteps",&numTimeSteps,"number of time steps");
  cmdp.setOption("numFrames",&numFrames,"number of frames for export");
  
  cmdp.setOption("convertPreComputedSolutionsToVTK", "computeSolutions", &convertSolutionsToVTK);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  bool saveSolutionFiles = true;
  
  if (numCells==-1) numCells = numGridPoints / k;
  
  if (rank==0) {
    cout << "solving on " << numCells << " x " << numCells << " mesh " << "of order " << k << ".\n";
  }
  
  set<int> timeStepsToExport;
  timeStepsToExport.insert(numTimeSteps);
  
  int timeStepsPerFrame = numTimeSteps / (numFrames - 1);
  if (timeStepsPerFrame==0) timeStepsPerFrame = 1;
  for (int n=0; n<numTimeSteps; n += timeStepsPerFrame) {
    timeStepsToExport.insert(n);
  }
  
  int H1Order = k + 1;
  
  const static double PI  = 3.141592653589793238462;

  double dt = 2 * PI / numTimeSteps;
  
  VarFactory varFactory;
  // traces:
  VarPtr qHat = varFactory.fluxVar("\\widehat{q}");
 
  // fields:
  VarPtr u = varFactory.fieldVar("u", L2);

  // test functions:
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  
  FunctionPtr c = Function::vectorize(y-0.5, 0.5-x);
//  FunctionPtr c = Function::vectorize(y, x);
  FunctionPtr n = Function::normal();
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  bf->addTerm(u / dt, v);
  bf->addTerm(- theta * u, c * v->grad());
//  bf->addTerm(theta * u_hat, (c * n) * v);
  bf->addTerm(qHat, v);
  
  double width = 2.0, height = 2.0;
  int horizontalCells = numCells, verticalCells = numCells;
  double x0 = -0.5; double y0 = -0.5;
  
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, delta_k, width, height,
                                       horizontalCells, verticalCells, false, x0, y0);
  
  FunctionPtr u0 = Teuchos::rcp( new Cone_U0 );
  
  RHSPtr initialRHS = RHS::rhs();
  initialRHS->addTerm(u0 / dt * v);
  initialRHS->addTerm((1-theta) * u0 * c * v->grad());
  
  BCPtr bc = BC::bc();
  
  SpatialFilterPtr inflowFilter = Teuchos::rcp( new InflowFilterForClockwisePlanarRotation(x0,x0+width,y0,y0+height,0.5,0.5));
//  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
  bc->addDirichlet(qHat, inflowFilter, Function::zero()); // zero BCs enforced at the inflow boundary.
  
  IPPtr ip;
  ip = Teuchos::rcp( new IP );
  ip->addTerm(v);
  ip->addTerm(c * v->grad());
//  ip = bf->graphNorm();
  
  // create two Solution objects; we'll switch between these for time steps
  SolutionPtr soln0 = Solution::solution(mesh, bc, initialRHS, ip);
  soln0->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln0 = Function::solution(u, soln0);
  FunctionPtr qHat_soln0 = Function::solution(qHat, soln0);
  
  RHSPtr rhs1 = RHS::rhs();
  rhs1->addTerm(u_soln0 / dt * v);
  rhs1->addTerm((1-theta) * u_soln0 * c * v->grad());
  
  SolutionPtr soln1 = Solution::solution(mesh, bc, rhs1, ip);
  soln1->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln1 = Function::solution(u, soln1);
  FunctionPtr qHat_soln1 = Function::solution(qHat, soln1);
  
  RHSPtr rhs2 = RHS::rhs(); // after the first solve on soln0, we'll swap out initialRHS for rhs2
  rhs2->addTerm(u_soln1 / dt * v);
  rhs2->addTerm((1-theta) * u_soln1 * c * v->grad());
  
  Teuchos::RCP<Solver> solver;
  
#ifdef USE_MUMPS
  solver = Teuchos::rcp( new MumpsSolver );
#else
  solver = Teuchos::rcp( new KluSolver );
#endif

//  double energyErrorSum = 0;

  ostringstream filePrefix;
  filePrefix << "convectingCone_k" << k << "_t";
  int frameNumber = 0;
  
#ifdef USE_VTK
  VTKExporter soln0Exporter(soln0,mesh,varFactory);
  VTKExporter soln1Exporter(soln1,mesh,varFactory);
#endif

  if (convertSolutionsToVTK) {
#ifdef USE_VTK
    if (rank==0) {
      for (int frameNumber=0; frameNumber<=numFrames; frameNumber++) {
        ostringstream filename;
        filename << filePrefix.str() << frameNumber << ".soln";
        soln0->readFromFile(filename.str());
        filename.str("");
        filename << filePrefix.str() << frameNumber;
        soln0Exporter.exportFields(filename.str());
      }
    }
#else
    if (rank==0) cout << "Driver was built without USE_VTK defined.  This must be defined to convert solution files to VTK files.\n";
#endif
    exit(0);
  }
  
  if (timeStepsToExport.find(0) != timeStepsToExport.end()) {
    map<int,FunctionPtr> solnMap;
    solnMap[u->ID()] = u0; // project field variables
    if (rank==0) cout << "About to project initial solution onto mesh.\n";
    soln0->projectOntoMesh(solnMap);
    if (rank==0) cout << "...projected initial solution onto mesh.\n";
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
    if (rank==0) cout << "About to export initial solution.\n";
#ifdef USE_VTK
    if (rank==0) soln0Exporter.exportFields(filename.str());
#endif
    if (saveSolutionFiles) {
      if (rank==0) {
        filename << ".soln";
        soln0->writeToFile(filename.str());
        cout << endl << "wrote " << filename.str() << endl;
      }
    }
    if (rank==0) cout << "...exported initial solution.\n";
  }
  
  if (rank==0) cout << "About to solve initial time step.\n";
  // first time step:
  soln0->setReportTimingResults(true); // added to gain insight into why MPI blocks in some cases on the server...
  if (useCondensedSolve) soln0->condensedSolve(solver);
  else soln0->solve(solver);
  soln0->setReportTimingResults(false);
//  energyErrorSum += soln0->energyErrorTotal();
  soln0->setRHS(rhs2);
  if (rank==0) cout << "Solved initial time step.\n";
  
  if (timeStepsToExport.find(1) != timeStepsToExport.end()) {
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
#ifdef USE_VTK
    if (rank==0) soln0Exporter.exportFields(filename.str());
#endif
    if (saveSolutionFiles) {
      if (rank==0) {
        filename << ".soln";
        soln0->writeToFile(filename.str());
        cout << endl << "wrote " << filename.str() << endl;
      }
    }
  }
    
  bool reportTimings = false;
  
  for (int n=1; n<numTimeSteps; n++) {
    bool odd = (n%2)==1;
    SolutionPtr soln_n = odd ? soln1 : soln0;
    if (useCondensedSolve) soln_n->solve(solver);
    else soln_n->solve(solver);
    if (reportTimings) {
      if (rank==0) cout << "time step " << n << ", timing report:\n";
      soln_n->reportTimings();
    }
    if (rank==0) {
      cout << "\x1B[2K"; // Erase the entire current line.
      cout << "\x1B[0E"; // Move to the beginning of the current line.
      cout << "Solved time step: " << n;
      flush(cout);
    }
    if (timeStepsToExport.find(n+1)!=timeStepsToExport.end()) {
      ostringstream filename;
      filename << filePrefix.str() << frameNumber++;
#ifdef USE_VTK
      if (rank==0) {
        if (odd) {
          soln1Exporter.exportFields(filename.str());
        } else {
          soln0Exporter.exportFields(filename.str());
        }
      }
#endif
      if (saveSolutionFiles) {
        if (rank==0) {
          filename << ".soln";
          if (odd) {
            soln1->writeToFile(filename.str());
          } else {
            soln0->writeToFile(filename.str());
          }
          cout << "wrote " << filename.str() << endl;
        }
      }
    }
//    energyErrorSum += soln_n->energyErrorTotal();
  }

//  if (rank==0) cout << "energy error, sum over all time steps: " << energyErrorSum << endl;
  
  return 0;
}