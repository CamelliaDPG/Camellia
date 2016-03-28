#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "SimpleFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PolarizedFunction.h"
#include "PenaltyConstraints.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"

using namespace Camellia;
using namespace std;

class Exp_ay2 : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ay2(double a) : _a(a) {}
  double value(double x, double y)
  {
    return exp(_a*y*y);
  }
};

class Log_ay2b : public SimpleFunction<double>
{
  double _a;
  double _b;
public:
  Log_ay2b(double a, double b) : _a(a), _b(b) {}
  double value(double x, double y)
  {
    return log(_a*y*y+_b);
  }
};

// class TimeRamp : public SimpleFunction<double>
// {
//   FunctionPtr _time;
//   double _timeScale;
//   double getTimeValue()
//   {
//     ParameterFunction* timeParamFxn = dynamic_cast<ParameterFunction*>(_time.get());
//     SimpleFunction<double>* timeFxn = dynamic_cast<SimpleFunction<double>*>(timeParamFxn->getValue().get());
//     return timeFxn->value(0);
//   }
// public:
//   TimeRamp(FunctionPtr timeConstantParamFxn, double timeScale)
//   {
//     _time = timeConstantParamFxn;
//     _timeScale = timeScale;
//   }
//   double value(double x)
//   {
//     double t = getTimeValue();
//     if (t >= _timeScale)
//     {
//       return 1.0;
//     }
//     else
//     {
//       return t / _timeScale;
//     }
//   }
// };

void setDirectSolver(CompressibleNavierStokesFormulation &form)
{
  Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
  form.setSolver(coarseSolver);
}

void setGMGSolver(CompressibleNavierStokesFormulation &form, vector<MeshPtr> &meshesCoarseToFine,
                                     int cgMaxIters, double cgTol, bool useCondensedSolve)
{
  Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true);
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(form.solutionIncrement(), meshesCoarseToFine, cgMaxIters, cgTol,
                                                                  GMGOperator::V_CYCLE, coarseSolver, useCondensedSolve) );
  gmgSolver->setAztecOutput(0);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
  form.setSolver(gmgSolver);
}

double computeL2Error(CompressibleNavierStokesFormulation &form, FunctionPtr rho_exact, FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr T_exact, MeshPtr mesh)
{
  // FunctionPtr D1_exact = form.mu()*u_exact->x()->grad();
  // FunctionPtr D2_exact = form.mu()*u_exact->y()->grad();

  int spaceDim = form.spaceDim();

  // double l2Error = 0;
  double rho_l2 = 0, u1_l2 = 0, u2_l2 = 0, T_l2 = 0;
  FunctionPtr rho_soln, u1_soln, u2_soln, T_soln,
              rho_diff, u1_diff, u2_diff, T_diff,
              rho_sqr, u1_sqr, u2_sqr, T_sqr;
  rho_soln = Function::solution(form.rho(), form.solution());
  u1_soln = Function::solution(form.u(1), form.solution());
  if (spaceDim == 2)
    u2_soln = Function::solution(form.u(2), form.solution());
  T_soln = Function::solution(form.T(), form.solution());
  rho_diff = rho_soln - rho_exact;
  rho_sqr = rho_diff*rho_diff;
  rho_l2 = rho_sqr->integrate(mesh, 10);
  if (spaceDim == 1)
  {
    u1_diff = u1_soln - u1_exact;
    u1_sqr = u1_diff*u1_diff;
    u1_l2 = u1_sqr->integrate(mesh, 10);
  }
  else if (spaceDim == 2)
  {
    u2_diff = u2_soln - u2_exact;
    u2_sqr = u2_diff*u2_diff;
    u2_l2 = u2_sqr->integrate(mesh, 10);
  }
  // T_diff = T_soln - T_exact;
  // T_sqr = T_diff*T_diff;
  // T_l2 = T_sqr->integrate(mesh, 10);
  // return sqrt(rho_l2+u1_l2+u2_l2+T_l2);
  return sqrt(rho_l2+u1_l2+u2_l2);
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  // Set Parameters
  string problemName = "Trivial";
  cmdp.setOption("problem", &problemName, "LidDriven, HemkerCylinder");
  bool steady = true;
  cmdp.setOption("steady", "unsteady", &steady, "steady");
  bool timeStepping = false;
  cmdp.setOption("timestep", "newton", &timeStepping, "timetep, newton");
  double dt = 0.1;
  cmdp.setOption("dt", &dt, "timestep size");
  string outputDir = ".";
  cmdp.setOption("outputDir", &outputDir, "output directory");
  int spaceDim = 2;
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  double Re = 1e2;
  cmdp.setOption("Re", &Re, "Re");
  double coarseRe = 1e2;
  cmdp.setOption("coarseRe", &coarseRe, "coarse Re");
  bool rampRe = false;
  cmdp.setOption("rampRe", "constantRe", &rampRe, "ramp Re to final value");
  string norm = "Graph";
  cmdp.setOption("norm", &norm, "norm");
  bool useDirectSolver = false; // false has an issue during GMGOperator::setFineStiffnessMatrix's call to GMGOperator::computeCoarseStiffnessMatrix().  I'm not yet clear on the nature of this isssue.
  cmdp.setOption("useDirectSolver", "useIterativeSolver", &useDirectSolver, "use direct solver");
  bool useCondensedSolve = true;
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  bool useSPDSolver = false;
  cmdp.setOption("useSPDSolver", "useQRSolver", &useSPDSolver);
  bool useCG = true;
  cmdp.setOption("useCG", "useGMRES", &useCG);
  bool useConformingTraces = true;
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  int polyOrder = 1, delta_k = 3;
  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
  cmdp.setOption("delta_k",&delta_k,"polynomial enrichment for test functions");
  int polyOrderCoarse = 1;
  double cgTol = 1e-10;
  cmdp.setOption("cgTol", &cgTol, "iterative solver tolerance");
  int cgMaxIters = 2000;
  cmdp.setOption("cgMaxIters", &cgMaxIters, "maximum number of iterations for linear solver");
  double nlTol = 1e-6;
  cmdp.setOption("nlTol", &nlTol, "Newton iteration tolerance");
  int nlMaxIters = 10;
  cmdp.setOption("nlMaxIters", &nlMaxIters, "maximum number of iterations for Newton solve");
  int numRefs = 10;
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  int startRef = 0;
  cmdp.setOption("startRef",&startRef,"starting ref count");
  bool exportHDF5 = false;
  cmdp.setOption("exportHDF5", "skipHDF5", &exportHDF5, "export solution to HDF5");
  bool saveSolution = false;
  cmdp.setOption("saveSolution", "skipSave", &saveSolution, "save solution and mesh");
  bool loadSolution = false;
  cmdp.setOption("loadSolution", "skipSave", &loadSolution, "load solution and mesh");
  bool computeL2 = false;
  cmdp.setOption("computeL2Error", "skipL2Error", &computeL2, "compute L2 Error");
  string tag="";
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  ostringstream exportName;
  if (steady)
    exportName << "Steady";
  else
    exportName << "Transient";
  exportName << problemName << spaceDim << "D" << "_Re" << Re << "_" << norm << "_k" << polyOrder;// << "_" << solverChoice;// << "_" << multigridStrategyString;
  if (tag != "")
    exportName << "_" << tag;

  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");
  totalTimer->start(true);

  // Construct Mesh
  MeshTopologyPtr meshTopo;
  MeshGeometryPtr meshGeometry = Teuchos::null;
  double gamma = 1.4;
  double Cv = 1;
  double rho_inf, u_inf, T_inf;
  int rampIncrement = 4;
  double pi = atan(1)*4;
  if (problemName == "Trivial")
  {
    int meshWidth = 2;
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,-1.0);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "CompressibleTaylorGreen")
  {
    T_inf = 1e6;
    int meshWidth = 2;
    vector<double> dims(spaceDim,2*pi);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = pi;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "SimpleShock")
  {
    int meshWidth = 4;
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,-0.5);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 0.25;
      int temporalDivisions = 1;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "Carter")
  {
    double U_inf = 1;
    T_inf = 1;
    double M_inf = 3;
    Cv = (U_inf*U_inf)/(M_inf*M_inf*gamma*(gamma-1)*T_inf);
    int meshWidth = 4;
    vector<double> dims(spaceDim,2.0);
    dims[1] = 1;
    vector<int> numElements(spaceDim,meshWidth);
    numElements[1] = 2;
    vector<double> x0(spaceDim, -1);
    x0[1] = 0;

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 4;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "Sod")
  {
    int meshWidth = 4;
    vector<double> dims(spaceDim,1.0);
    if (spaceDim == 2)
      dims[1] = 0.25;
    vector<int> numElements(spaceDim,meshWidth);
    if (spaceDim == 2)
      numElements[1] = 1;
    vector<double> x0(spaceDim,0);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 0.2;
      int temporalDivisions = 1;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "Noh")
  {
    rampIncrement = 0;
    gamma = 5./3;
    rho_inf = 1;
    u_inf = 1;
    T_inf = 0;
    Cv = 1;
    int meshWidth = 2;
    vector<double> dims(spaceDim,1);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,-1);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "Piston")
  {
    gamma = 5./3;
    rho_inf = 1;
    u_inf = 0;
    T_inf = 0;
    Cv = 1;

    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Line<2> >(), tensorialDegree);
    vector<double> v00 = {0,0};
    vector<double> v10 = {1,0};
    vector<double> v01 = {.5,.5};
    vector<double> v11 = {1,.5};
    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    vector<unsigned> spaceTimeLine1VertexList;
    spaceTimeLine1VertexList.push_back(0);
    spaceTimeLine1VertexList.push_back(1);
    spaceTimeLine1VertexList.push_back(2);
    spaceTimeLine1VertexList.push_back(3);
    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    meshTopo = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    // int meshWidth = 2;
    // vector<double> dims(spaceDim,1);
    // vector<int> numElements(spaceDim,meshWidth);
    // vector<double> x0(spaceDim,-1);
    // meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    // if (!steady)
    // {
    //   double t0 = 0;
    //   double t1 = 1;
    //   int temporalDivisions = 2;
    //   meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    // }
  }
  if (problemName == "Sedov")
  {
    rho_inf = 1;
    u_inf = 0;
    T_inf = 0;
    Cv = 1;
    int meshWidth = 4;
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = .25;
      int temporalDivisions = 1;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "TriplePoint")
  {
    rho_inf = 1;
    u_inf = 0;
    T_inf = 1;
    Cv = 1;
    int meshWidth = 7;
    vector<double> dims(spaceDim,7.0);
    dims[1] = 3;
    vector<int> numElements(spaceDim,meshWidth);
    numElements[1] = 3;
    vector<double> x0(spaceDim,0);

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 5;
      int temporalDivisions = 5;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  if (problemName == "RayleighTaylor")
  {
    rho_inf = 1;
    u_inf = 0;
    T_inf = 1;
    Cv = 1;
    vector<double> dims(spaceDim,.5);
    dims[1] = 2;
    vector<int> numElements(spaceDim,1);
    numElements[1] = 4;
    vector<double> x0(spaceDim,0);
    x0[1] = -1;

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 3;
      int temporalDivisions = 3;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  double R = Cv*(gamma-1);
  double formRe = Re;
  if (rampRe)
  {
    // formRe = coarseRe;
    double mu = max(1./Re, min(1./coarseRe,1./pow(2,startRef+rampIncrement)));
    formRe = 1./mu;
  }

  Teuchos::ParameterList nsParameters;
  if (steady)
    nsParameters = CompressibleNavierStokesFormulation::steadyFormulation(spaceDim, formRe, useConformingTraces, meshTopo, polyOrder, delta_k).getConstructorParameters();
  else
    nsParameters = CompressibleNavierStokesFormulation::spaceTimeFormulation(spaceDim, formRe, useConformingTraces, meshTopo, polyOrder, polyOrder, delta_k).getConstructorParameters();
  if (timeStepping)
    nsParameters = CompressibleNavierStokesFormulation::timeSteppingFormulation(spaceDim, formRe, useConformingTraces, meshTopo, polyOrder, polyOrder, delta_k).getConstructorParameters();

  if (loadSolution)
    nsParameters.set("savedSolutionAndMeshPrefix",outputDir+"/"+exportName.str());
  // nsParameters.set("neglectFluxesOnRHS", false);
  nsParameters.set("problemName", problemName);
  nsParameters.set("norm", norm);
  nsParameters.set("dt", dt);
  nsParameters.set("Cv", Cv);
  nsParameters.set("gamma", gamma);
  nsParameters.set("rhoInit", 1.);
  nsParameters.set("u1Init", 1.);
  nsParameters.set("u2Init", 0.);
  nsParameters.set("TInit", 1.);
  if (problemName == "Sod")
  {
    nsParameters.set("u1Init", 0.);
  }
  if (problemName == "Noh")
  {
    nsParameters.set("u1Init", 1.);
    nsParameters.set("TInit", 1.);
  }
  if (problemName == "Piston")
  {
    nsParameters.set("u1Init", 0.);
    nsParameters.set("TInit", 0.);
  }
  if (problemName == "Sedov")
  {
    nsParameters.set("u1Init", 0.);
    nsParameters.set("u2Init", 0.);
  }
  if (problemName == "TriplePoint")
  {
    nsParameters.set("u1Init", 0.);
    nsParameters.set("u2Init", 0.);
  }
  if (problemName == "RayleighTaylor")
  {
    nsParameters.set("u1Init", 0.);
    nsParameters.set("u2Init", 0.);
  }
  CompressibleNavierStokesFormulation form(meshTopo, nsParameters);

  if (useSPDSolver)
    form.bf()->setUseSPDSolveForOptimalTestFunctions(true);

  // form.refine();

  // form.setIP( norm );

  form.solutionIncrement()->setUseCondensedSolve(useCondensedSolve);

  MeshPtr mesh = form.solutionIncrement()->mesh();
  // if (problemName == "Cylinder")
  //   preprocessHemkerMesh(mesh, steady, 1);
  // if (meshGeometry != Teuchos::null)
  //   mesh->setEdgeToCurveMap(meshGeometry->edgeToCurveMap());

  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
  int numberOfMeshesForMultigrid = meshesCoarseToFine.size();

  // VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
  // VarPtr tn1_hat = form.tn_hat(1), tn2_hat = form.tn_hat(2);

  FunctionPtr rho_exact, u1_exact, u2_exact, u3_exact, T_exact, p_exact, u_exact;
  Teuchos::RCP<ParameterFunction> refParamFunc = ParameterFunction::parameterFunction(64);
  FunctionPtr refFunc = refParamFunc;
  if (problemName == "Trivial")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-1);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(0);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-1);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(0);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);

    // if (spaceDim == 1)
    // {
      rho_exact = Function::constant(1);
      u1_exact = Function::constant(1);
      u2_exact = Function::constant(1);
      T_exact = Function::constant(1);
    // }
    // else
    // {
    //   FunctionPtr exp1lambdax = Teuchos::rcp(new Exp_ax(Re));
    //   rho_exact = Function::constant(1)-exp1lambdax;
    //   u1_exact = Function::constant(1)-exp1lambdax;
    //   u2_exact = Function::constant(1)-exp1lambdax;
    //   T_exact = Function::constant(1)-exp1lambdax;
    // }
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    switch (spaceDim)
    {
      case 1:
        form.addMassFluxCondition(        leftX, rho_exact, u_exact, T_exact);
        form.addVelocityTraceCondition(   leftX, one);
        form.addTemperatureTraceCondition(leftX, one);
        form.addVelocityTraceCondition(   rightX, one);
        form.addTemperatureTraceCondition(rightX, one);
        break;
      case 2:
        // form.addMassFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addMomentumFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addEnergyFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addVelocityTraceCondition(SpatialFilter::allSpace(), ones);
        // form.addTemperatureTraceCondition(SpatialFilter::allSpace(), one);
        form.addMassFluxCondition(        leftX, rho_exact, u_exact, T_exact);
        form.addMassFluxCondition(        leftY, rho_exact, u_exact, T_exact);
        // form.addVelocityTraceCondition(   leftX, u_exact);
        // form.addVelocityTraceCondition(   leftY, u_exact);
        // form.addVelocityTraceCondition(   rightX, u_exact);
        // form.addVelocityTraceCondition(   rightY, u_exact);
        // form.addTemperatureTraceCondition(leftX, T_exact);
        // form.addTemperatureTraceCondition(leftY, T_exact);
        // form.addTemperatureTraceCondition(rightX, T_exact);
        // form.addTemperatureTraceCondition(rightY, T_exact);
        form.addMomentumFluxCondition(    leftX, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    leftY, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    rightX, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    rightY, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      leftX, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      leftY, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      rightX, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      rightY, rho_exact, u_exact, T_exact);
        break;
      case 3:
        break;
    }
    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      SpatialFilterPtr t1  = SpatialFilter::matchingT(1);
      form.addMassFluxCondition(    t0,    rho_exact, u_exact, T_exact);
      form.addMomentumFluxCondition(t0,    rho_exact, u_exact, T_exact);
      form.addEnergyFluxCondition(  t0,    rho_exact, u_exact, T_exact);
      // form.addVelocityTraceCondition(t1, ones);
      // form.addTemperatureTraceCondition(t1, one);
    }
  }
  if (problemName == "CompressibleTaylorGreen")
  {
    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);
    FunctionPtr temporalDecay = Teuchos::rcp(new Exp_at(-2./Re));
    FunctionPtr sinX = Teuchos::rcp(new Sin_x());
    FunctionPtr cosX = Teuchos::rcp(new Cos_x());
    FunctionPtr sinY = Teuchos::rcp(new Sin_y());
    FunctionPtr cosY = Teuchos::rcp(new Cos_y());
    FunctionPtr cos2X = Teuchos::rcp(new Cos_ax(2));
    FunctionPtr cos2Y = Teuchos::rcp(new Cos_ay(2));

    rho_exact = one;
    T_exact = 1./.4*0.25*(cos2X+cos2Y)*temporalDecay*temporalDecay;
    u1_exact = sinX*cosY*temporalDecay;
    u2_exact = -cosX*sinY*temporalDecay;

    FunctionPtr dTdt_exact = 1./.4*.25*(cos2X+cos2Y)*4./Re*temporalDecay*temporalDecay;
    FunctionPtr duudt_exact = -4./Re*(cosY*cosY*sinX*sinX+cosX*cosX*sinY*sinY)*temporalDecay*temporalDecay;
    FunctionPtr D11_exact = 1./Re*u1_exact->dx();
    FunctionPtr D12_exact = 1./Re*u1_exact->dy();
    FunctionPtr D21_exact = 1./Re*u2_exact->dx();
    FunctionPtr D22_exact = 1./Re*u2_exact->dy();
    FunctionPtr q1_exact = -form.Cp()/(Re*form.Pr())*T_exact->dx();
    FunctionPtr q2_exact = -form.Cp()/(Re*form.Pr())*T_exact->dy();

    FunctionPtr u_exact = Function::vectorize(u1_exact,u2_exact);

    RHSPtr rhs = form.rhs();
    VarPtr ve = form.ve();
    rhs->addTerm(-Cv*dTdt_exact * ve);
    rhs->addTerm(-0.5*duudt_exact * ve);
    rhs->addTerm(-(Cv*u1_exact*T_exact)->dx() * ve);
    rhs->addTerm(-(0.5*(u1_exact*u1_exact+u2_exact*u2_exact)*u1_exact)->dx() * ve);
    rhs->addTerm(-(R*u1_exact*T_exact)->dx() * ve);
    rhs->addTerm(-(Cv*u2_exact*T_exact)->dy() * ve);
    rhs->addTerm(-(0.5*(u1_exact*u1_exact+u2_exact*u2_exact)*u2_exact)->dy() * ve);
    rhs->addTerm(-(R*u2_exact*T_exact)->dy() * ve);
    rhs->addTerm(-(q1_exact)->dx() * ve);
    rhs->addTerm(-(q2_exact)->dy() * ve);
    rhs->addTerm(-(-(D11_exact+D11_exact-2./3*(D11_exact+D22_exact))*u1_exact)->dx() * ve);
    rhs->addTerm(-(-(D12_exact+D21_exact)*u2_exact)->dx() * ve);
    rhs->addTerm(-(-(D21_exact+D12_exact)*u1_exact)->dy() * ve);
    rhs->addTerm(-(-(D22_exact+D22_exact-2./3*(D11_exact+D22_exact))*u2_exact)->dy() * ve);

    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(2*pi);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(2*pi);

    form.addTemperatureTraceCondition(leftX,  T_exact);
    form.addTemperatureTraceCondition(leftY,  T_exact);
    form.addTemperatureTraceCondition(rightX, T_exact);
    form.addTemperatureTraceCondition(rightY, T_exact);
    form.addVelocityTraceCondition(leftX,  u_exact);
    form.addVelocityTraceCondition(leftY,  u_exact);
    form.addVelocityTraceCondition(rightX, u_exact);
    form.addVelocityTraceCondition(rightY, u_exact);
    form.addMassFluxCondition(leftX,  zero);
    form.addMassFluxCondition(leftY,  zero);
    form.addMassFluxCondition(rightX, zero);
    form.addMassFluxCondition(rightY, zero);
    // form.addEnergyFluxCondition(leftX,  zero);
    // form.addEnergyFluxCondition(leftY,  zero);
    // form.addEnergyFluxCondition(rightX, zero);
    // form.addEnergyFluxCondition(rightY, zero);

    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      form.addMassFluxCondition(    t0,    rho_exact, u_exact, T_exact);
      form.addMomentumFluxCondition(t0,    rho_exact, u_exact, T_exact);
      form.addEnergyFluxCondition(  t0,    rho_exact, u_exact, T_exact);
    }
  }
  if (problemName == "SimpleShock")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-0.5);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(0.5);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-0.5);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(0.5);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);

    if (spaceDim == 1)
    {
      rho_exact = 2*one - Function::heaviside(0);
      u1_exact = Function::constant(0);
      // u1_exact = Function::constant(1) - Function::heaviside(0);
      T_exact = Function::constant(1);
    }
    else
    {
      // rho_exact = 2*one - Function::heaviside(0);
      rho_exact = 2*one - Function::heavisideY(0);
      u1_exact = Function::constant(0);
      // u1_exact = Function::constant(1) - Function::heaviside(0);
      u2_exact = Function::constant(0);
      T_exact = Function::constant(1);
    }
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      form.addMassFluxCondition(    t0, rho_exact, u_exact, T_exact);
      form.addMomentumFluxCondition(t0, rho_exact, u_exact, T_exact);
      form.addEnergyFluxCondition(  t0, rho_exact, u_exact, T_exact);
    }
    switch (spaceDim)
    {
      case 1:
        form.addMassFluxCondition(        leftX, rho_exact, u_exact, T_exact);
        form.addVelocityTraceCondition(   leftX, u_exact);
        form.addTemperatureTraceCondition(leftX, T_exact);
        form.addVelocityTraceCondition(   rightX, u_exact);
        form.addTemperatureTraceCondition(rightX, T_exact);
        break;
      case 2:
        // form.addMassFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addMomentumFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addEnergyFluxCondition(SpatialFilter::allSpace(), one, ones, one);
        // form.addVelocityTraceCondition(SpatialFilter::allSpace(), ones);
        // form.addTemperatureTraceCondition(SpatialFilter::allSpace(), one);
        form.addMassFluxCondition(        leftX, rho_exact, u_exact, T_exact);
        form.addMassFluxCondition(        leftY, rho_exact, u_exact, T_exact);
        // form.addVelocityTraceCondition(   leftX, u_exact);
        // form.addVelocityTraceCondition(   leftY, u_exact);
        // form.addVelocityTraceCondition(   rightX, u_exact);
        // form.addVelocityTraceCondition(   rightY, u_exact);
        // form.addTemperatureTraceCondition(leftX, T_exact);
        // form.addTemperatureTraceCondition(leftY, T_exact);
        // form.addTemperatureTraceCondition(rightX, T_exact);
        // form.addTemperatureTraceCondition(rightY, T_exact);
        form.addMomentumFluxCondition(    leftX, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    leftY, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    rightX, rho_exact, u_exact, T_exact);
        form.addMomentumFluxCondition(    rightY, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      leftX, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      leftY, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      rightX, rho_exact, u_exact, T_exact);
        form.addEnergyFluxCondition(      rightY, rho_exact, u_exact, T_exact);
        break;
      case 3:
        break;
    }
  }
  if (problemName == "Carter")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-1);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    SpatialFilterPtr bottom = SpatialFilter::matchingY(0);
    SpatialFilterPtr top    = SpatialFilter::matchingY(1);
    SpatialFilterPtr xGreater0 = SpatialFilter::greaterThanX(0);
    SpatialFilterPtr xLess0 = SpatialFilter::lessThanX(0);
    SpatialFilterPtr bottomFree = SpatialFilter::intersectionFilter(bottom, xLess0);
    SpatialFilterPtr plate = SpatialFilter::intersectionFilter(bottom, xGreater0);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);

    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "spaceDim must be 2");
    rho_exact = one;
    u1_exact = one;
    u2_exact = zero;
    T_exact = one;
    u_exact = Function::vectorize(u1_exact,u2_exact);

    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      // form.addMassFluxCondition(    t0, rho_exact, u_exact, T_exact);
      // form.addMomentumFluxCondition(t0, rho_exact, u_exact, T_exact);
      // form.addEnergyFluxCondition(  t0, rho_exact, u_exact, T_exact);
      form.addMassFluxCondition(    t0, -one);
      form.addXMomentumFluxCondition(t0, -one);
      form.addYMomentumFluxCondition(t0, zero);
      form.addEnergyFluxCondition(  t0, -(Cv+.5)*one);
    }
    // form.addMassFluxCondition(     leftX, -one);
    // form.addXMomentumFluxCondition(leftX, -(1+R)*one);
    // form.addYMomentumFluxCondition(leftX, zero);
    // form.addEnergyFluxCondition(   leftX, -(Cv+0.5+R)*one);
    // form.addMassFluxCondition(          leftX, rho_exact, u_exact, T_exact);
    // form.addMomentumFluxCondition(      leftX, rho_exact, u_exact, T_exact);
    // form.addEnergyFluxCondition(        leftX, rho_exact, u_exact, T_exact);
    form.addMassFluxCondition(        leftX, -one);
    // form.addXMomentumFluxCondition(leftX, -(1+R)*one);
    // form.addYMomentumFluxCondition(leftX, zero);
    // form.addEnergyFluxCondition(   leftX, -(Cv+0.5+R)*one);
    form.addVelocityTraceCondition(   leftX, onezero);
    form.addTemperatureTraceCondition(leftX, one);

    // form.addMassFluxCondition(        rightX, one);
    // form.addVelocityTraceCondition(   rightX, onezero);
    // form.addTemperatureTraceCondition(rightX, one);

    // form.addMassFluxCondition(        bottom, zero);
    // form.addVelocityTraceCondition(   bottom, onezero);
    // form.addTemperatureTraceCondition(bottom, one);

    // form.addMassFluxCondition(        top, zero);
    // form.addVelocityTraceCondition(   top, onezero);
    // form.addTemperatureTraceCondition(top, one);

    form.addXMomentumFluxCondition( bottomFree, zero);
    form.addEnergyFluxCondition(    bottomFree, zero);
    form.addYVelocityTraceCondition(bottomFree, zero);
    // form.addXMomentumFluxCondition( bottom, zero);
    // form.addEnergyFluxCondition(    bottom, zero);
    // form.addYVelocityTraceCondition(bottom, zero);
    form.addXMomentumFluxCondition( top, zero);
    form.addEnergyFluxCondition(    top, zero);
    form.addYVelocityTraceCondition(top, zero);

    // form.addMassFluxCondition(          leftX, rho_exact, u_exact, T_exact);
    // form.addMomentumFluxCondition(      leftX, rho_exact, u_exact, T_exact);
    // form.addEnergyFluxCondition(        leftX, rho_exact, u_exact, T_exact);
    // form.addXMomentumFluxCondition( bottomFree, rho_exact, u_exact, T_exact);
    // form.addEnergyFluxCondition(    bottomFree, rho_exact, u_exact, T_exact);
    // form.addYVelocityTraceCondition(bottomFree, u2_exact);
    // form.addXMomentumFluxCondition(        top, rho_exact, u_exact, T_exact);
    // form.addEnergyFluxCondition(           top, rho_exact, u_exact, T_exact);
    // form.addYVelocityTraceCondition(       top, u2_exact);

    form.addVelocityTraceCondition(plate, zeros);
    form.addTemperatureTraceCondition(plate, 2.8*one);
  }
  if (problemName == "Sod")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(0.25);

    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);

    rho_exact = one - 0.875*Function::heaviside(0.5);
    u1_exact = zero;
    u2_exact = zero;
    p_exact = one - 0.9*Function::heaviside(0.5);
    T_exact = 1./.4*p_exact/rho_exact;
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      form.addMassFluxCondition(    t0, rho_exact, u_exact, T_exact);
      form.addMomentumFluxCondition(t0, rho_exact, u_exact, T_exact);
      form.addEnergyFluxCondition(  t0, rho_exact, u_exact, T_exact);
    }
    switch (spaceDim)
    {
      case 1:
        form.addMassFluxCondition(        leftX, rho_exact*u1_exact);
        form.addVelocityTraceCondition(   leftX, u_exact);
        form.addTemperatureTraceCondition(leftX, T_exact);
        form.addMassFluxCondition(        rightX, rho_exact*u1_exact);
        form.addVelocityTraceCondition(   rightX, u_exact);
        form.addTemperatureTraceCondition(rightX, T_exact);
        break;
      case 2:
        form.addMassFluxCondition(        leftX, rho_exact*u1_exact);
        form.addVelocityTraceCondition(   leftX, u_exact);
        form.addTemperatureTraceCondition(leftX, T_exact);
        form.addMassFluxCondition(        rightX, rho_exact*u1_exact);
        form.addVelocityTraceCondition(   rightX, u_exact);
        form.addTemperatureTraceCondition(rightX, T_exact);
        form.addMassFluxCondition(        leftY, rho_exact*u2_exact);
        form.addVelocityTraceCondition(   leftY, u_exact);
        form.addTemperatureTraceCondition(leftY, T_exact);
        form.addMassFluxCondition(        rightY, rho_exact*u2_exact);
        form.addVelocityTraceCondition(   rightY, u_exact);
        form.addTemperatureTraceCondition(rightY, T_exact);
        // form.addMassFluxCondition(        leftX, rho_exact, u_exact, T_exact);
        // form.addMassFluxCondition(        leftY, rho_exact, u_exact, T_exact);
        // form.addMomentumFluxCondition(    leftX, rho_exact, u_exact, T_exact);
        // form.addMomentumFluxCondition(    leftY, rho_exact, u_exact, T_exact);
        // form.addMomentumFluxCondition(    rightX, rho_exact, u_exact, T_exact);
        // form.addMomentumFluxCondition(    rightY, rho_exact, u_exact, T_exact);
        // form.addEnergyFluxCondition(      leftX, rho_exact, u_exact, T_exact);
        // form.addEnergyFluxCondition(      leftY, rho_exact, u_exact, T_exact);
        // form.addEnergyFluxCondition(      rightX, rho_exact, u_exact, T_exact);
        // form.addEnergyFluxCondition(      rightY, rho_exact, u_exact, T_exact);
        break;
      case 3:
        break;
    }
  }
  if (problemName == "Noh")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-1);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(0);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-1);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(0);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);
    FunctionPtr cos_y = Teuchos::rcp(new Cos_ay(1));
    FunctionPtr sin_y = Teuchos::rcp(new Sin_ay(1));
    FunctionPtr cos_theta = Teuchos::rcp( new PolarizedFunction<double>( cos_y ) );
    FunctionPtr sin_theta = Teuchos::rcp( new PolarizedFunction<double>( sin_y ) );

    if (spaceDim == 1)
    {
      rho_exact = one;
      u1_exact = one;
      // T_exact = T_inf*one;
      T_exact = zero;
    }
    else
    {
      rho_exact = one;
      u1_exact = -cos_theta;
      u2_exact = -sin_theta;
      // T_exact = T_inf*one;
      T_exact = zero;
    }
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    switch (spaceDim)
    {
      case 1:
        form.addMassFluxCondition(        leftX, -one);
        form.addXMomentumFluxCondition(   leftX, -(u1_exact+rho_exact*R*T_exact));
        form.addEnergyFluxCondition(      leftX, -(Cv*T_exact+.5+R*T_exact)*one);
        form.addVelocityTraceCondition(   rightX, zero);
        form.addMassFluxCondition(        rightX, zero);
        form.addEnergyFluxCondition(      rightX, zero);
        if (!steady)
        {
          form.addMassFluxCondition(     t0, -one);
          form.addXMomentumFluxCondition(t0, -u1_exact);
          form.addEnergyFluxCondition(   t0, -(Cv*T_exact+0.5)*one);
        }
        break;
      case 2:
        form.addMassFluxCondition(        leftX, -u1_exact);
        form.addXMomentumFluxCondition(   leftX, -(u1_exact*u1_exact+R*T_exact));
        form.addYMomentumFluxCondition(   leftX, -(u1_exact*u2_exact));
        form.addEnergyFluxCondition(      leftX, -(u1_exact*(Cv*T_exact+0.5+R*T_exact)));

        form.addMassFluxCondition(        leftY, -u2_exact);
        form.addXMomentumFluxCondition(   leftY, -(u2_exact*u1_exact));
        form.addYMomentumFluxCondition(   leftY, -(u2_exact*u2_exact+R*T_exact));
        form.addEnergyFluxCondition(      leftY, -(u2_exact*(Cv*T_exact+0.5+R*T_exact)));

        form.addXVelocityTraceCondition(  rightX, zero);
        form.addMassFluxCondition(        rightX, zero);
        form.addYMomentumFluxCondition(   rightX, zero);
        form.addEnergyFluxCondition(      rightX, zero);

        form.addYVelocityTraceCondition(  rightY, zero);
        form.addMassFluxCondition(        rightY, zero);
        form.addXMomentumFluxCondition(   rightY, zero);
        form.addEnergyFluxCondition(      rightY, zero);
        if (!steady)
        {
          form.addMassFluxCondition(     t0, -one);
          form.addXMomentumFluxCondition(t0, -u1_exact);
          form.addYMomentumFluxCondition(t0, -u2_exact);
          form.addEnergyFluxCondition(   t0, -(Cv*T_exact+0.5)*one);
        }
        break;
      case 3:
        break;
    }
  }
  if (problemName == "Piston")
  {
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp( new PenaltyConstraints );
    // SpatialFilterPtr leftX  = SpatialFilter::matchingX(-1);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    // SpatialFilterPtr leftY  = SpatialFilter::matchingY(-1);
    // SpatialFilterPtr rightY = SpatialFilter::matchingY(0);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
    SpatialFilterPtr t1  = SpatialFilter::matchingT(.5);
    SpatialFilterPtr leftX  = SpatialFilter::negatedFilter(rightX & t0 & t1);

    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);
    VarPtr te = form.te();
    VarPtr tm1 = form.tm(1);

    // if (spaceDim == 1)
    // {
    //   rho_exact = one;
    //   u1_exact = one;
    //   // T_exact = T_inf*one;
    //   T_exact = zero;
    // }
    // else
    // {
    //   rho_exact = one;
    //   u1_exact = -cos_theta;
    //   u2_exact = -sin_theta;
    //   // T_exact = T_inf*one;
    //   T_exact = zero;
    // }
    // FunctionPtr u_exact;
    // if (spaceDim == 1)
    //   u_exact = u1_exact;
    // else
    //   u_exact = Function::vectorize(u1_exact,u2_exact);

    switch (spaceDim)
    {
      case 1:
        form.addVelocityTraceCondition(   leftX, zero);
        form.addMassFluxCondition(        leftX, zero);
        pc->addConstraint(one*tm1 - one*te == zero, leftX);
        form.addVelocityTraceCondition(   rightX, zero);
        form.addMassFluxCondition(        rightX, zero);
        form.addEnergyFluxCondition(      rightX, zero);
        if (!steady)
        {
          form.addMassFluxCondition(     t0, -one);
          form.addXMomentumFluxCondition(t0, zero);
          form.addEnergyFluxCondition(   t0, zero);
        }
        break;
      // case 2:
      //   form.addMassFluxCondition(        leftX, -u1_exact);
      //   form.addXMomentumFluxCondition(   leftX, -(u1_exact*u1_exact+R*T_exact));
      //   form.addYMomentumFluxCondition(   leftX, -(u1_exact*u2_exact));
      //   form.addEnergyFluxCondition(      leftX, -(u1_exact*(Cv*T_exact+0.5+R*T_exact)));

      //   form.addMassFluxCondition(        leftY, -u2_exact);
      //   form.addXMomentumFluxCondition(   leftY, -(u2_exact*u1_exact));
      //   form.addYMomentumFluxCondition(   leftY, -(u2_exact*u2_exact+R*T_exact));
      //   form.addEnergyFluxCondition(      leftY, -(u2_exact*(Cv*T_exact+0.5+R*T_exact)));

      //   form.addXVelocityTraceCondition(  rightX, zero);
      //   form.addMassFluxCondition(        rightX, zero);
      //   form.addYMomentumFluxCondition(   rightX, zero);
      //   form.addEnergyFluxCondition(      rightX, zero);

      //   form.addYVelocityTraceCondition(  rightY, zero);
      //   form.addMassFluxCondition(        rightY, zero);
      //   form.addXMomentumFluxCondition(   rightY, zero);
      //   form.addEnergyFluxCondition(      rightY, zero);
      //   if (!steady)
      //   {
      //     form.addMassFluxCondition(     t0, -one);
      //     form.addXMomentumFluxCondition(t0, -u1_exact);
      //     form.addYMomentumFluxCondition(t0, -u2_exact);
      //     form.addEnergyFluxCondition(   t0, -(Cv*T_exact+0.5)*one);
      //   }
      //   break;
      case 3:
        break;
    }
    form.solutionIncrement()->setFilter(pc);
  }
  if (problemName == "Sedov")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(1);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);

    if (spaceDim == 1)
    {
      rho_exact = one;
      u1_exact = zero;
      FunctionPtr x = Function::xn(1);
      T_exact = Function::max(2*refFunc*(1-refFunc*x), zero);
      // T_exact = 1./width*(one-Function::heaviside(width));
    }
    else
    {
      rho_exact = one;
      u1_exact = zero;
      u2_exact = zero;
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      T_exact = Function::max(2*refFunc*(1-refFunc*x), zero)*Function::max(2*refFunc*(1-refFunc*y), zero);
      // double width = 1./16;
      // T_exact = 1./(width*width)*(one-Function::heaviside(width))*(one-Function::heavisideY(width));
    }
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    switch (spaceDim)
    {
      case 1:
        form.addMassFluxCondition(       rightX, zero);
        form.addXMomentumFluxCondition(  rightX, zero);
        form.addEnergyFluxCondition(     rightX, zero);
        form.addVelocityTraceCondition(   leftX, zero);
        form.addMassFluxCondition(        leftX, zero);
        form.addEnergyFluxCondition(      leftX, zero);
        if (!steady)
        {
          form.addMassFluxCondition(     t0, -rho_exact);
          form.addXMomentumFluxCondition(t0, -u1_exact);
          form.addEnergyFluxCondition(   t0, -Cv*T_exact);
        }
        break;
      case 2:
        form.addMassFluxCondition(       rightX, zero);
        form.addXMomentumFluxCondition(  rightX, zero);
        form.addYMomentumFluxCondition(  rightX, zero);
        form.addEnergyFluxCondition(     rightX, zero);

        form.addMassFluxCondition(       rightY, zero);
        form.addXMomentumFluxCondition(  rightY, zero);
        form.addYMomentumFluxCondition(  rightY, zero);
        form.addEnergyFluxCondition(     rightY, zero);

        form.addXVelocityTraceCondition(  leftX, zero);
        form.addMassFluxCondition(        leftX, zero);
        form.addYMomentumFluxCondition(   leftX, zero);
        form.addEnergyFluxCondition(      leftX, zero);

        form.addYVelocityTraceCondition(  leftY, zero);
        form.addMassFluxCondition(        leftY, zero);
        form.addXMomentumFluxCondition(   leftY, zero);
        form.addEnergyFluxCondition(      leftY, zero);
        if (!steady)
        {
          form.addMassFluxCondition(     t0, -rho_exact);
          form.addXMomentumFluxCondition(t0, -u1_exact);
          form.addYMomentumFluxCondition(t0, -u2_exact);
          form.addEnergyFluxCondition(   t0, -Cv*T_exact);
        }
        break;
      case 3:
        break;
    }
  }
  if (problemName == "TriplePoint")
  {
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(7);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(3);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    FunctionPtr zero = Function::zero();
    FunctionPtr zeros = Function::vectorize(zero, zero);
    FunctionPtr one = Function::constant(1);
    FunctionPtr onezero = Function::vectorize(one, zero);
    FunctionPtr ones = Function::vectorize(one, one);

    rho_exact = one - (1-0.125)*Function::heaviside(1)*Function::heavisideY(1.5);
    u1_exact = zero;
    u2_exact = zero;
    p_exact = one - (1-0.1)*Function::heaviside(1);
    T_exact = 1./.4*p_exact/rho_exact;
    // T_exact = one - (1-0.1)*Function::heaviside(1);
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    // form.addMassFluxCondition(       rightX, zero);
    // form.addXMomentumFluxCondition(  rightX, zero);
    // form.addYMomentumFluxCondition(  rightX, zero);
    // form.addEnergyFluxCondition(     rightX, zero);

    // form.addMassFluxCondition(       rightY, zero);
    // form.addXMomentumFluxCondition(  rightY, zero);
    // form.addYMomentumFluxCondition(  rightY, zero);
    // form.addEnergyFluxCondition(     rightY, zero);

    form.addXVelocityTraceCondition(  leftX, zero);
    form.addMassFluxCondition(        leftX, zero);
    form.addYMomentumFluxCondition(   leftX, zero);
    form.addEnergyFluxCondition(      leftX, zero);

    form.addYVelocityTraceCondition(  leftY, zero);
    form.addMassFluxCondition(        leftY, zero);
    form.addXMomentumFluxCondition(   leftY, zero);
    form.addEnergyFluxCondition(      leftY, zero);

    form.addXVelocityTraceCondition( rightX, zero);
    form.addMassFluxCondition(       rightX, zero);
    form.addYMomentumFluxCondition(  rightX, zero);
    form.addEnergyFluxCondition(     rightX, zero);

    form.addYVelocityTraceCondition( rightY, zero);
    form.addMassFluxCondition(       rightY, zero);
    form.addXMomentumFluxCondition(  rightY, zero);
    form.addEnergyFluxCondition(     rightY, zero);
    if (!steady)
    {
      form.addMassFluxCondition(     t0, -rho_exact);
      form.addXMomentumFluxCondition(t0, -u1_exact);
      form.addYMomentumFluxCondition(t0, -u2_exact);
      form.addEnergyFluxCondition(   t0, -Cv*T_exact);
    }
  }
  if (problemName == "RayleighTaylor")
  {
    RHSPtr rhs = form.rhs();
    VarPtr vm2 = form.vm(2);
    FunctionPtr rho_prev = Function::solution(form.rho(), form.solution());
    double g = -1;
    rhs->addTerm(g*rho_prev * vm2);

    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(.5);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-2);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(2);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);

    double beta = 20;
    double rho1 = 1;
    double rho2 = 2;
    FunctionPtr atan_betay = Teuchos::rcp(new ArcTan_ay(beta));
    double u0 = 0.02;
    FunctionPtr exp_m2piy2 = Teuchos::rcp(new Exp_ay2(-2*pi));
    FunctionPtr cos_2pix = Teuchos::rcp(new Cos_ax(2*pi));
    FunctionPtr sin_2pix = Teuchos::rcp(new Sin_ax(2*pi));
    FunctionPtr y = Function::yn(1);
    double C = 4 + (1.5*2+2./pi*atan(beta*2)-1./(2*pi*beta)*log(beta*beta*4+1));
    // cout << "C = " << C << endl;
    FunctionPtr log_b2y21 = Teuchos::rcp(new Log_ay2b(beta*beta,1));
    p_exact = g*((rho1+rho2)/2.*y + (rho2-rho1)/pi*(atan_betay*y-1./(2*beta)*log_b2y21))+C*one;

    rho_exact = (rho1+rho2)/2.*one + (rho2-rho1)/pi*atan_betay;
    u1_exact = u0*exp_m2piy2*2*y*sin_2pix;
    u2_exact = u0*exp_m2piy2*2*y*cos_2pix;
    T_exact = 1./.4*p_exact/rho_exact;
    FunctionPtr u_exact;
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

    form.addXVelocityTraceCondition(  leftX, zero);
    form.addMassFluxCondition(        leftX, zero);
    form.addYMomentumFluxCondition(   leftX, zero);
    form.addEnergyFluxCondition(      leftX, zero);

    form.addYVelocityTraceCondition(  leftY, zero);
    form.addMassFluxCondition(        leftY, zero);
    form.addXMomentumFluxCondition(   leftY, zero);
    form.addEnergyFluxCondition(      leftY, zero);

    form.addXVelocityTraceCondition( rightX, zero);
    form.addMassFluxCondition(       rightX, zero);
    form.addYMomentumFluxCondition(  rightX, zero);
    form.addEnergyFluxCondition(     rightX, zero);

    form.addYVelocityTraceCondition( rightY, zero);
    form.addMassFluxCondition(       rightY, zero);
    form.addXMomentumFluxCondition(  rightY, zero);
    form.addEnergyFluxCondition(     rightY, zero);
    if (!steady)
    {
      form.addMassFluxCondition(     t0, -rho_exact);
      form.addXMomentumFluxCondition(t0, -u1_exact);
      form.addYMomentumFluxCondition(t0, -u2_exact);
      form.addEnergyFluxCondition(   t0, -Cv*T_exact);
    }
  }

  string dataFileLocation;
  dataFileLocation = outputDir+"/"+exportName.str()+".txt";
  ofstream dataFile(dataFileLocation);

  Teuchos::RCP<HDF5Exporter> exporter, energyErrorExporter;
  if (exportHDF5)
  {
    exporter = Teuchos::rcp(new HDF5Exporter(mesh, exportName.str(), outputDir));
    // exportName << "_energyError";
    // energyErrorExporter = Teuchos::rcp(new HDF5Exporter(mesh, exportName.str(), outputDir));
  }

  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;

  cout << setprecision(2) << scientific;

  solverTime->start(true);
  int totalIterationCount = 0;
  int timeStep = 0;
  int maxTimeSteps = 100;
  double timeRes = 1;
  double timeTol = 1e-6;
  if (startRef == 0)
  {
    while (timeStep < maxTimeSteps && timeRes > timeTol)
    {
      while ((l2NormOfIncrement > nlTol) && (stepNumber < nlMaxIters))
      {
        if (useDirectSolver)
          setDirectSolver(form);
        else
          setGMGSolver(form, meshesCoarseToFine, cgMaxIters, cgTol, useCondensedSolve);

        double alpha = form.solveAndAccumulate();
        int solveCode = form.getSolveCode();
        if (solveCode != 0)
        {
          if (rank==0) cout << "Solve not completed correctly, repeating Newton iteration" << endl;
          stepNumber--;
        }
        l2NormOfIncrement = form.L2NormSolutionIncrement();
        stepNumber++;

        if (rank==0) cout << stepNumber << ". alpha = " << alpha << " L^2 norm of increment: " << l2NormOfIncrement;

        if (!useDirectSolver)
        {
          Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(dynamic_cast<GMGSolver*>(form.getSolver().get()), false);
          gmgSolver->setUseConjugateGradient(useCG);
          gmgSolver->setComputeConditionNumberEstimate(false);
          int iterationCount = gmgSolver->iterationCount();
          totalIterationCount += iterationCount;
          if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
        }
        else
        {
          if (rank==0) cout << endl;
        }

        if (alpha < 1e-2)
          break;
      }
      if (timeStepping)
      {
        timeRes = form.timeResidual();
        form.solutionPreviousTimeStep()->setSolution(form.solution());
        // if (exportHDF5)
        // {
        //   exporter->exportSolution(form.solution(), timeStep);
        //   // energyErrorExporter->exportFunction(energyErrorFunction, "energy error", refNumber);
        // }
        if (rank==0) cout << timeStep << ". time residual = " << timeRes << endl;
        timeStep++;
        l2NormOfIncrement = 1;
        stepNumber = 0;
      }
      else
        timeRes = 0;
    }
  }
  // form.clearSolutionIncrement(); // need to clear before evaluating energy error
  double solveTime = solverTime->stop();

  // FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.solutionIncrement());
  // if (rank==0) dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "elapsed\t" << "iterations\t " << endl;

  if (exportHDF5)
  {
    exporter->exportSolution(form.solution(), startRef);
    // energyErrorExporter->exportFunction(energyErrorFunction, "energy error", 0);
  }
  if (saveSolution)
    form.save(outputDir+"/"+exportName.str());

  double energyError = form.solutionIncrement()->energyErrorTotal();
  double l2Error = 0;
  if (computeL2)
    l2Error = computeL2Error(form, rho_exact, u1_exact, u2_exact, T_exact, mesh);
  int globalDofs = mesh->globalDofCount();
  if (rank==0) cout << "Refinement: " << startRef
                    << " \tElements: " << mesh->numActiveElements()
                    << " \tDOFs: " << mesh->numGlobalDofs()
                    << " \tEnergy Error: " << energyError
                    << " \tL2 Error: " << l2Error
                    << " \tSolve Time: " << solveTime
                    << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
                    << " \tIteration Count: " << totalIterationCount
                    << endl;
  if (rank==0) dataFile << 0
           << " " << mesh->numActiveElements()
           << " " << mesh->numGlobalDofs()
           << " " << energyError
           << " " << l2Error
           << " " << solveTime
           << " " << totalTimer->totalElapsedTime(true)
           << " " << totalIterationCount
           << " " << endl;

  bool truncateMultigridMeshes = false; // for getting a "fair" sense of how iteration counts vary with h.

  double tol = 1e-5;
  int refNumber = startRef;
  while (refNumber < numRefs)
  {
    refNumber++;
    form.refine();

    if (rank==0) cout << " ****** Refinement " << refNumber << " ****** " << endl;

    if (rampRe)
    {
      double mu = max(1./Re, min(1./coarseRe,1./pow(2,refNumber+rampIncrement)));
      form.setmu(mu);
      double refVal = min(64.,max(pow(2.,2+refNumber),64.));
      refParamFunc->setValue(refVal);
      if (rank==0) cout << " Mesh Re = " << 1./mu << endl;
      // if (rank==0) cout << " ref value = " << refVal << endl;
    }

    meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
    // truncate meshesCoarseToFine to get a "fair" iteration count measure

    if (truncateMultigridMeshes)
    {
      while (meshesCoarseToFine.size() > numberOfMeshesForMultigrid)
        meshesCoarseToFine.erase(meshesCoarseToFine.begin());
    }

    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    solverTime->start(true);
    totalIterationCount = 0;
    timeStep = 0;
    timeRes = 1;
    while (timeStep < maxTimeSteps && timeRes > timeTol)
    {
      while ((l2NormOfIncrement > nlTol) && (stepNumber < nlMaxIters))
      {
        if (!useDirectSolver)
          setGMGSolver(form, meshesCoarseToFine, cgMaxIters, cgTol, useCondensedSolve);
        else
          setDirectSolver(form);

        double alpha = form.solveAndAccumulate();
        l2NormOfIncrement = form.L2NormSolutionIncrement();
        stepNumber++;

        if (rank==0) cout << stepNumber << ". alpha = " << alpha << " L^2 norm of increment: " << l2NormOfIncrement;

        if (!useDirectSolver)
        {
          Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(dynamic_cast<GMGSolver*>(form.getSolver().get()), false);
          gmgSolver->setUseConjugateGradient(useCG);
          gmgSolver->setComputeConditionNumberEstimate(false);
          int iterationCount = gmgSolver->iterationCount();
          totalIterationCount += iterationCount;
          if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
        }
        else
        {
          if (rank==0) cout << endl;
        }

        if (alpha < 1e-2)
          break;
      }
      if (timeStepping)
      {
        timeRes = form.timeResidual();
        form.solutionPreviousTimeStep()->setSolution(form.solution());
        if (rank==0) cout << timeStep << ". time residual = " << timeRes << endl;
        timeStep++;
        l2NormOfIncrement = 1;
        stepNumber = 0;

        // if (exportHDF5)
        // {
        //   exporter->exportSolution(form.solution(), refNumber*maxTimeSteps+timeStep);
        //   // energyErrorExporter->exportFunction(energyErrorFunction, "energy error", refNumber);
        // }

        form.solutionPreviousTimeStep()->setSolution(form.solution());
      }
      else
        timeRes = 0;
    }

    // form.clearSolutionIncrement(); // need to clear before evaluating energy error
    energyError = form.solutionIncrement()->energyErrorTotal();
    if (computeL2)
      l2Error = computeL2Error(form, rho_exact, u1_exact, u2_exact, T_exact, mesh);

    solveTime = solverTime->stop();

    if (rank==0) cout << "Refinement: " << refNumber
      << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << " \tEnergy Error: " << energyError
        << " \tL2 Error: " << l2Error
        << " \tSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << " \tIteration Count: " << totalIterationCount
        << endl;
    if (rank==0) dataFile << refNumber
      << " " << mesh->numActiveElements()
        << " " << mesh->numGlobalDofs()
        << " " << energyError
        << " " << l2Error
        << " " << solveTime
        << " " << totalTimer->totalElapsedTime(true)
        << " " << totalIterationCount
        << " " << endl;

    if (exportHDF5)
    {
      exporter->exportSolution(form.solution(), refNumber);
      // energyErrorExporter->exportFunction(energyErrorFunction, "energy error", refNumber);
    }
    if (saveSolution)
      form.save(outputDir+"/"+exportName.str());
  }

  if (rank==0) dataFile.close();

  return 0;
}
