#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RefinementStrategy.h"
#include "SimpleFunction.h"
#include "Solver.h"
#include "TypeDefs.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;

class InitialConditions : public SimpleFunction<double>
{
  double _r0;  // width control
public:
  InitialConditions(double r = 0.15)
  {
    _r0 = r;
  }
  double value(double x, double y)
  {
    const bool slottedCylinder = true;
    const bool cone = true;
    const bool hump = true;
    
    const static double PI  = 3.141592653589793238462;

    double r_cyld = sqrt((x-0.5)*(x-0.5)+(y-0.75)*(y-0.75))/_r0;
    double r_cone = sqrt((x-0.5)*(x-0.5)+(y-0.25)*(y-0.25))/_r0;
    double r_hump = sqrt((x-0.25)*(x-0.25)+(y-0.5)*(y-0.5))/_r0;
    
    // slotted cylinder
    if(slottedCylinder) {
      if(r_cyld<=1.0 && (fabs(x-0.5)>=0.025 || y>=0.85))
        return 1.0;
    }
    
    // cone
    if(cone) {
      if(r_cone<=1.0)
        return 1.0-r_cone;
    }
    
    // hump
    if(hump) {
      if(r_hump<=1.0)
        return 0.25*(1.0+cos(PI*r_hump));
    }
    
    // if we get here, no match: return 0
    return 0.0;
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  bool useCondensedSolve = true;

  int numGridPoints = 128; // in x,y -- idea is to keep the overall order of approximation constant
  int k = 1; // poly order for u
  double theta = 0.5;
  int numTimeSteps = 2000;
  int numCells = -1; // in x, y (-1 so we can set a default if unset from the command line.)
  int numFrames = 50;
  int delta_k = 2;   // test space enrichment: should be 2 for 2D
  double revolutionFraction = 1.0;

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("numCells",&numCells,"number of cells in x and y directions");
  cmdp.setOption("revolutionFraction", &revolutionFraction, "how many revolutions to solve for");
  cmdp.setOption("theta",&theta,"theta weight for time-stepping");
  cmdp.setOption("numTimeSteps",&numTimeSteps,"number of time steps");
  cmdp.setOption("numFrames",&numFrames,"number of frames for export");

  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve, "use static condensation to reduce the size of the global solve");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (numCells==-1) numCells = numGridPoints / k;

  if (rank==0)
  {
    cout << "solving on " << numCells << " x " << numCells << " mesh " << "of order " << k << ".\n";
  }

  set<int> timeStepsToExport;
  timeStepsToExport.insert(numTimeSteps);

  int timeStepsPerFrame = numTimeSteps / (numFrames - 1);
  if (timeStepsPerFrame==0) timeStepsPerFrame = 1;
  for (int n=0; n<numTimeSteps; n += timeStepsPerFrame)
  {
    timeStepsToExport.insert(n);
  }

  int H1Order = k + 1;

  const static double PI  = 3.141592653589793238462;

  double dt = revolutionFraction * 2. * PI / numTimeSteps;

  VarFactoryPtr varFactory = VarFactory::varFactory();
  // traces:
  VarPtr qHat = varFactory->fluxVar("\\widehat{q}");

  // fields:
  VarPtr u = varFactory->fieldVar("u", L2);

  // test functions:
  VarPtr v = varFactory->testVar("v", HGRAD);

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);

  FunctionPtr c = Function::vectorize(0.5 - y, x - 0.5);
  FunctionPtr n = Function::normal();

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  bf->addTerm(u / dt, v);
  bf->addTerm(- theta * u, c * v->grad());
  bf->addTerm(qHat, v);

  int horizontalCells = numCells, verticalCells = numCells;

  BCPtr bc = BC::bc();
  bc->addDirichlet(qHat, SpatialFilter::allSpace(), Function::zero());

  double width = 1.0, height = 1.0;
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, width, height,
                 horizontalCells, verticalCells);

  FunctionPtr u0 = Teuchos::rcp( new InitialConditions(0.15) );

  ostringstream initialDataDirName;
  initialDataDirName << "convectingConeInitialData_k" << k;
  HDF5Exporter::exportFunction("./", initialDataDirName.str(), u0, mesh);
  
  if (rank == 0) std::cout << "Exported initial data to " << initialDataDirName.str() << std::endl;
  
  RHSPtr initialRHS = RHS::rhs();
  initialRHS->addTerm(u0 / dt * v);
  initialRHS->addTerm((1-theta) * u0 * c * v->grad());

  IPPtr ip;
  ip = bf->graphNorm();

  // create two Solution objects; we'll switch between these for time steps
  SolutionPtr soln0 = Solution::solution(mesh, bc, initialRHS, ip);
  soln0->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln0 = Function::solution(u, soln0);
  FunctionPtr qHat_soln0 = Function::solution(qHat, soln0, false);

  RHSPtr rhs1 = RHS::rhs();
  rhs1->addTerm(u_soln0 / dt * v);
  rhs1->addTerm((1-theta) * u_soln0 * c * v->grad());

  SolutionPtr soln1 = Solution::solution(mesh, bc, rhs1, ip);
  soln1->setCubatureEnrichmentDegree(5);
  FunctionPtr u_soln1 = Function::solution(u, soln1);
  FunctionPtr qHat_soln1 = Function::solution(qHat, soln1, false);

  RHSPtr rhs2 = RHS::rhs(); // after the first solve on soln0, we'll swap out initialRHS for rhs2
  rhs2->addTerm(u_soln1 / dt * v);
  rhs2->addTerm((1-theta) * u_soln1 * c * v->grad());

  Teuchos::RCP<Solver> solver = Solver::getDirectSolver();

  ostringstream filePrefix;
  filePrefix << "convectingCone_k" << k << "_t";
  int frameNumber = 0;

  ostringstream dir_name;
  dir_name << "convectingCone_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());

  if (timeStepsToExport.find(0) != timeStepsToExport.end())
  {
    map<int,FunctionPtr> solnMap;
    solnMap[u->ID()] = u0; // project field variables
    if (rank==0) cout << "About to project initial solution onto mesh.\n";
    int solnOrdinal = 0;
    soln0->projectOntoMesh(solnMap, solnOrdinal);
    if (rank==0) cout << "...projected initial solution onto mesh.\n";
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
    if (rank==0) cout << "About to export initial solution.\n";
    exporter.exportSolution(soln0,0);
  }

  if (rank==0) cout << "About to solve initial time step.\n";
  // first time step:
  soln0->setReportTimingResults(true); // added to gain insight into why MPI blocks in some cases on the server...
  if (useCondensedSolve) soln0->condensedSolve(solver);
  else soln0->solve(solver);
  soln0->setReportTimingResults(false);
  soln0->setRHS(rhs2);
  if (rank==0) cout << "Solved initial time step.\n";

  if (timeStepsToExport.find(1) != timeStepsToExport.end())
  {
    ostringstream filename;
    filename << filePrefix.str() << frameNumber++;
    exporter.exportSolution(soln0);
  }

  bool reportTimings = false;

  for (int n=1; n<numTimeSteps; n++)
  {
    bool odd = (n%2)==1;
    SolutionPtr soln_n = odd ? soln1 : soln0;
    if (useCondensedSolve) soln_n->solve(solver);
    else soln_n->solve(solver);
    if (reportTimings)
    {
      if (rank==0) cout << "time step " << n << ", timing report:\n";
      soln_n->reportTimings();
    }
    if (rank==0)
    {
      cout << "\x1B[2K"; // Erase the entire current line.
      cout << "\x1B[0E"; // Move to the beginning of the current line.
      cout << "Completed time step: " << n+1 << "/" << numTimeSteps;
      flush(cout);
    }
    if (timeStepsToExport.find(n+1)!=timeStepsToExport.end())
    {
      ostringstream filename;
      filename << filePrefix.str() << frameNumber++;
      double t = n * dt;
      if (odd)
      {
        exporter.exportSolution(soln1, t);
      }
      else
      {
        exporter.exportSolution(soln0, t);
      }
    }
  }
  
  if (rank == 0) cout << "\nExported solution to " << dir_name.str() << endl;
  
  return 0;
}
