#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "SimpleFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PenaltyConstraints.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"

using namespace Camellia;
using namespace std;

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

// double computeL2Error(CompressibleNavierStokesFormulation &form, FunctionPtr u_exact, MeshPtr mesh, double Re)
// {
//   FunctionPtr sigma1_exact = 1./Re*u_exact->x()->grad();
//   FunctionPtr sigma2_exact = 1./Re*u_exact->y()->grad();
//
//   // double l2Error = 0;
//   double u1_l2 = 0, u2_l2, sigma11_l2 = 0, sigma12_l2 = 0, sigma21_l2 = 0, sigma22_l2 = 0;
//   FunctionPtr u1_soln, u2_soln, sigma11_soln, sigma12_soln, sigma21_soln, sigma22_soln,
//               u1_diff, u2_diff, sigma11_diff, sigma12_diff, sigma21_diff, sigma22_diff,
//               u1_sqr, u2_sqr, sigma11_sqr, sigma12_sqr, sigma21_sqr, sigma22_sqr;
//   u1_soln = Function::solution(form.u(1), form.solution());
//   u2_soln = Function::solution(form.u(2), form.solution());
//   sigma11_soln = Function::solution(form.sigma(1,1), form.solution());
//   sigma12_soln = Function::solution(form.sigma(1,2), form.solution());
//   sigma21_soln = Function::solution(form.sigma(2,1), form.solution());
//   sigma22_soln = Function::solution(form.sigma(2,2), form.solution());
//   u1_diff = u1_soln - u_exact->x();
//   u2_diff = u2_soln - u_exact->y();
//   sigma11_diff = sigma11_soln - sigma1_exact->x();
//   sigma12_diff = sigma12_soln - sigma1_exact->y();
//   sigma21_diff = sigma21_soln - sigma2_exact->x();
//   sigma22_diff = sigma22_soln - sigma2_exact->y();
//   u1_sqr = u1_diff*u1_diff;
//   u2_sqr = u2_diff*u2_diff;
//   sigma11_sqr = sigma11_diff*sigma11_diff;
//   sigma12_sqr = sigma12_diff*sigma12_diff;
//   sigma21_sqr = sigma21_diff*sigma21_diff;
//   sigma22_sqr = sigma22_diff*sigma22_diff;
//   u1_l2 = u1_sqr->integrate(mesh, 10);
//   u2_l2 = u2_sqr->integrate(mesh, 10);
//   sigma11_l2 = sigma11_sqr->integrate(mesh, 10);
//   sigma12_l2 = sigma12_sqr->integrate(mesh, 10);
//   sigma21_l2 = sigma21_sqr->integrate(mesh, 10);
//   sigma22_l2 = sigma22_sqr->integrate(mesh, 10);
//   return sqrt(u1_l2+sigma11_l2+sigma12_l2+sigma21_l2+sigma22_l2);
//   // l2Error = sqrt(u_l2);
// }

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
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
  string outputDir = ".";
  cmdp.setOption("outputDir", &outputDir, "output directory");
  int spaceDim = 2;
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  double Re = 1e2;
  cmdp.setOption("Re", &Re, "Re");
  string norm = "Graph";
  cmdp.setOption("norm", &norm, "norm");
  bool useDirectSolver = false; // false has an issue during GMGOperator::setFineStiffnessMatrix's call to GMGOperator::computeCoarseStiffnessMatrix().  I'm not yet clear on the nature of this isssue.
  cmdp.setOption("useDirectSolver", "useIterativeSolver", &useDirectSolver, "use direct solver");
  bool useCondensedSolve = false;
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  bool useConformingTraces = false;
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  int polyOrder = 2, delta_k = 2;
  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
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
  bool exportHDF5 = false;
  cmdp.setOption("exportHDF5", "skipHDF5", &exportHDF5, "export solution to HDF5");
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

  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");
  totalTimer->start(true);

  // Construct Mesh
  MeshTopologyPtr meshTopo;
  MeshGeometryPtr meshGeometry = Teuchos::null;
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

  Teuchos::ParameterList nsParameters;
  if (steady)
    nsParameters = CompressibleNavierStokesFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, delta_k).getConstructorParameters();
  else
    nsParameters = CompressibleNavierStokesFormulation::spaceTimeFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, polyOrder, delta_k).getConstructorParameters();

  // nsParameters.set("neglectFluxesOnRHS", false);
  CompressibleNavierStokesFormulation form(meshTopo, nsParameters);

  // form.refine();

  form.setIP( norm );

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

  FunctionPtr rho_exact, u1_exact, u2_exact, u3_exact, T_exact;
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

    FunctionPtr rho_exact, u1_exact, u2_exact, T_exact;
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
    FunctionPtr u_exact;
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

    FunctionPtr rho_exact, u1_exact, u2_exact, T_exact;
    if (spaceDim == 1)
    {
      rho_exact = one + Function::heaviside(0);
      u1_exact = Function::constant(0);
      T_exact = Function::constant(1);
    }
    else
    {
      rho_exact = Function::constant(1);
      u1_exact = Function::constant(0);
      u2_exact = Function::constant(0);
      T_exact = Function::constant(1);
    }
    FunctionPtr u_exact;
    if (spaceDim == 1)
      u_exact = u1_exact;
    else
      u_exact = Function::vectorize(u1_exact,u2_exact);

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
    if (!steady)
    {
      SpatialFilterPtr t0  = SpatialFilter::matchingT(0);
      form.addMassFluxCondition(    t0, rho_exact, u_exact, T_exact);
      form.addMomentumFluxCondition(t0, rho_exact, u_exact, T_exact);
      form.addEnergyFluxCondition(  t0, rho_exact, u_exact, T_exact);
    }
  }

  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;

  cout << setprecision(2) << scientific;

  solverTime->start(true);
  int totalIterationCount = 0;
  while ((l2NormOfIncrement > nlTol) && (stepNumber < nlMaxIters))
  {
    if (useDirectSolver)
      setDirectSolver(form);
    else
      setGMGSolver(form, meshesCoarseToFine, cgMaxIters, cgTol, useCondensedSolve);

    form.solveAndAccumulate();
    l2NormOfIncrement = form.L2NormSolutionIncrement();
    stepNumber++;

    if (rank==0) cout << stepNumber << ". L^2 norm of increment: " << l2NormOfIncrement;

    if (!useDirectSolver)
    {
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(dynamic_cast<GMGSolver*>(form.getSolver().get()), false);
      int iterationCount = gmgSolver->iterationCount();
      totalIterationCount += iterationCount;
      if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
    }
    else
    {
      if (rank==0) cout << endl;
    }
  }
  form.clearSolutionIncrement(); // need to clear before evaluating energy error
  double solveTime = solverTime->stop();

  FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.solutionIncrement());

  ostringstream exportName;
  if (steady)
    exportName << "Steady";
  else
    exportName << "Transient";
  exportName << problemName << spaceDim << "D" << "_Re" << Re << "_" << norm << "_k" << polyOrder;// << "_" << solverChoice;// << "_" << multigridStrategyString;
  if (tag != "")
    exportName << "_" << tag;

  string dataFileLocation;
  dataFileLocation = outputDir+"/"+exportName.str()+".txt";
  ofstream dataFile(dataFileLocation);
  // if (rank==0) dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "elapsed\t" << "iterations\t " << endl;

  Teuchos::RCP<HDF5Exporter> exporter, energyErrorExporter;
  if (exportHDF5)
  {
    exporter = Teuchos::rcp(new HDF5Exporter(mesh, exportName.str(), outputDir));
    exporter->exportSolution(form.solution(), 0);

    exportName << "_energyError";
    energyErrorExporter = Teuchos::rcp(new HDF5Exporter(mesh, exportName.str(), outputDir));
    energyErrorExporter->exportFunction(energyErrorFunction, "energy error", 0);
  }
  // HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), outputDir);

  double energyError = form.solutionIncrement()->energyErrorTotal();
  double l2Error = 0;
  // if (computeL2)
  //   l2Error = computeL2Error(form, u_exact, mesh, Re);
  int globalDofs = mesh->globalDofCount();
  if (rank==0) cout << "Refinement: " << 0
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

  bool truncateMultigridMeshes = true; // for getting a "fair" sense of how iteration counts vary with h.

  double tol = 1e-5;
  int refNumber = 0;
  while ((energyError > tol) && (refNumber < numRefs))
  {
    refNumber++;
    form.refine();

    if (rank==0) cout << " ****** Refinement " << refNumber << " ****** " << endl;

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
    while ((l2NormOfIncrement > nlTol) && (stepNumber < nlMaxIters))
    {
      if (!useDirectSolver)
        setGMGSolver(form, meshesCoarseToFine, cgMaxIters, cgTol, useCondensedSolve);
      else
        setDirectSolver(form);

      form.solveAndAccumulate();
      l2NormOfIncrement = form.L2NormSolutionIncrement();
      stepNumber++;

      if (rank==0) cout << stepNumber << ". L^2 norm of increment: " << l2NormOfIncrement;

      if (!useDirectSolver)
      {
        Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(dynamic_cast<GMGSolver*>(form.getSolver().get()), false);
        int iterationCount = gmgSolver->iterationCount();
        totalIterationCount += iterationCount;
        if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
      }
      else
      {
        if (rank==0) cout << endl;
      }
    }

    form.clearSolutionIncrement(); // need to clear before evaluating energy error
    energyError = form.solutionIncrement()->energyErrorTotal();
    // if (computeL2)
    //   l2Error = computeL2Error(form, u_exact, mesh, Re);

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
      energyErrorExporter->exportFunction(energyErrorFunction, "energy error", refNumber);
    }

  }

  if (rank==0) dataFile.close();

  return 0;
}
