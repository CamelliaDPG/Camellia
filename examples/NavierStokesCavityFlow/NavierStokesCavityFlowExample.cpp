#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class RampBoundaryFunction_U1 : public SimpleFunction<double>
{
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps)
  {
    _eps = eps;
  }
  double value(double x, double y)
  {
    double tol = 1e-14;
    if (abs(y-1.0) < tol)   // top boundary
    {
      if ( (abs(x) < _eps) )   // top left
      {
        return x / _eps;
      }
      else if ( abs(1.0-x) < _eps)     // top right
      {
        return (1.0-x) / _eps;
      }
      else     // top middle
      {
        return 1;
      }
    }
    else     // not top boundary: 0.0
    {
      return 0.0;
    }
  }
  double value(double x, double y, double z)
  {
    // bilinear interpolation with ramp of width _eps around top edges
    double tol = 1e-14;
    if (abs(y-1.0) <tol)
    {
      double xFactor = 1.0;
      double zFactor = 1.0;
      if ( (abs(x) < _eps) )   // top left
      {
        xFactor = x / _eps;
      }
      else if ( abs(1.0-x) < _eps)     // top right
      {
        xFactor = (1.0-x) / _eps;
      }
      if ( (abs(z) < _eps) )   // top back
      {
        zFactor = z / _eps;
      }
      else if ( abs(1.0-z) < _eps)     // top front
      {
        zFactor = (1.0-z) / _eps;
      }
      return xFactor * zFactor;
    }
    else
    {
      return 0.0;
    }
  }
};

class TimeRamp : public SimpleFunction<double>
{
  FunctionPtr _time;
  double _timeScale;
  double getTimeValue()
  {
    ParameterFunction* timeParamFxn = dynamic_cast<ParameterFunction*>(_time.get());
    SimpleFunction<double>* timeFxn = dynamic_cast<SimpleFunction<double>*>(timeParamFxn->getValue().get());
    return timeFxn->value(0);
  }
public:
  TimeRamp(FunctionPtr timeConstantParamFxn, double timeScale)
  {
    _time = timeConstantParamFxn;
    _timeScale = timeScale;
  }
  double value(double x)
  {
    double t = getTimeValue();
    if (t >= _timeScale)
    {
      return 1.0;
    }
    else
    {
      return t / _timeScale;
    }
  }
};

using namespace std;

void setDirectSolver(NavierStokesVGPFormulation &form)
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

void setGMGSolver(NavierStokesVGPFormulation &form, vector<MeshPtr> &meshesCoarseToFine,
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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  int spaceDim = 2;

  double eps = 1.0 / 64.0;

  bool useDirectSolver = false;
  bool useCondensedSolve = false;
  double Re = 1e3;

  int meshWidth = 2;
  vector<double> dims(spaceDim,1.0);
  vector<int> numElements(spaceDim,meshWidth);
  vector<double> x0(spaceDim,0.0);

  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  int polyOrder = 4, delta_k = 2;

  bool useConformingTraces = false;
  NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, delta_k);
  form.addPointPressureCondition({0.5,0.5});
  form.solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
  
  int polyOrderCoarse = 1;
  double cgTol = 1e-6;
  int cgMaxIters = 2000;
  MeshPtr mesh = form.solutionIncrement()->mesh();
  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
  
  int numberOfMeshesForMultigrid = meshesCoarseToFine.size();

  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);

  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
  SpatialFilterPtr notTopBoundary = SpatialFilter::negatedFilter(topBoundary);
  form.addWallCondition(notTopBoundary);

  FunctionPtr u1_topRamp = Teuchos::rcp( new RampBoundaryFunction_U1(eps) );
  FunctionPtr u_topRamp;
  FunctionPtr zero = Function::zero();
  if (spaceDim == 2)
  {
    u_topRamp = Function::vectorize(u1_topRamp,zero);
  }
  else
  {
    u_topRamp = Function::vectorize(u1_topRamp,zero,zero);
  }
  form.addInflowCondition(topBoundary, u_topRamp);

  double nonlinearThreshold = 1e-3;
  int maxNonlinearIterations = 10;
  int maxRefinements = 10;
  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;
  
  cout << setprecision(2) << scientific;
  
  while ((l2NormOfIncrement > nonlinearThreshold) && (stepNumber < maxNonlinearIterations))
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
      if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
    }
    else
    {
      if (rank==0) cout << endl;
    }
  }
  
  FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.solutionIncrement());
  
  string outputDir = ".";
  ostringstream exportName;
  exportName << "navierStokesCavityFlowSolution_Re" << Re;
  HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), outputDir);

  exportName << "_energyError";
  HDF5Exporter energyErrorExporter(form.solution()->mesh(), exportName.str(), outputDir);
  
  exporter.exportSolution(form.solution(), 0);
  energyErrorExporter.exportFunction({energyErrorFunction}, {"energy error"}, 0);
  
  double energyError = form.solutionIncrement()->energyErrorTotal();
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->getTopology()->getActiveCellIndices().size();
  if (rank==0) cout << "Initial energy error: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;

  bool truncateMultigridMeshes = false; // true aims to get a "fair" sense of how iteration counts vary with h.
  
  double tol = 1e-5;
  int refNumber = 0;
  do
  {
    refNumber++;
    form.refine();
    
    // update nonlinear threshold according to last energyError:
    nonlinearThreshold = 1e-3 * energyError;
    
    // update cgTol, too:
    cgTol = 1e-6 * energyError;
    
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
    while ((l2NormOfIncrement > nonlinearThreshold) && (stepNumber < maxNonlinearIterations))
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
        if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
      }
      else
      {
        if (rank==0) cout << endl;
      }
    }

    exporter.exportSolution(form.solution(), refNumber);
    energyErrorExporter.exportFunction({energyErrorFunction}, {"energy error"}, refNumber);

    energyError = form.solutionIncrement()->energyErrorTotal();
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->getTopology()->getActiveCellIndices().size();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  }
  while ((energyError > tol) && (refNumber < maxRefinements));

  FunctionPtr u1_steady = Function::solution(form.u(1), form.solution());
  if (rank==0) cout << "u1(0.5, 0.5) = " << u1_steady->evaluate(0.5, 0.5) << endl;

  // now solve for the stream function on the fine mesh:
  if (spaceDim == 2)
  {
    form.streamSolution()->solve();
    HDF5Exporter steadyStreamExporter(form.streamSolution()->mesh(), "navierStokesSteadyCavityFlowStreamSolution", outputDir);
    steadyStreamExporter.exportSolution(form.streamSolution());
  }

//  /*   Now that we have a fine mesh, try the same problem, but transient, starting with a zero initial
//   *   state, and with boundary conditions that "ramp up" in time (and which also are zero at time 0).
//   *   We expect to recover the steady solution.
//   */
//
//  double totalTime = 3;
//  double dt = 0.1;
//  int numTimeSteps = ceil(totalTime / dt);
//  StokesVGPFormulation transientForm(spaceDim, useConformingTraces, mu, true, dt);
//
//  FunctionPtr t = transientForm.getTimeFunction();
//  FunctionPtr timeRamp = Teuchos::rcp(new TimeRamp(t,1.0));
//
//  transientForm.initializeSolution(meshTopo, polyOrder, delta_k);
//  transientForm.addZeroMeanPressureCondition();
//  transientForm.addWallCondition(notTopBoundary);
//  transientForm.addInflowCondition(topBoundary, timeRamp * u_topRamp);
//
//  MeshPtr transientMesh = transientForm.solution()->mesh();
//  HDF5Exporter transientExporter(transientMesh, "stokesTransientSolution", ".");
//
//  for (int timeStep=0; timeStep<numTimeSteps; timeStep++) {
//    transientForm.solve();
//    double L2_step = transientForm.L2NormOfTimeStep();
//    transientExporter.exportSolution(transientForm.solution(),transientForm.getTime());
//
//    transientForm.takeTimeStep();
//    if (rank==0) cout << "time step " << timeStep << " completed (L^2 norm of difference from prev: " << L2_step << ").\n";
//  }
//
//  { // trying the same thing as below, but computing it differently:
//    FunctionPtr  p_transient = Function::solution( transientForm.p(), transientForm.solution());
//    FunctionPtr u1_transient = Function::solution(transientForm.u(1), transientForm.solution());
//    FunctionPtr u2_transient = Function::solution(transientForm.u(2), transientForm.solution());
//    FunctionPtr  p_steady = Function::solution( form.p(), form.solution());
//    FunctionPtr u1_steady = Function::solution(form.u(1), form.solution());
//    FunctionPtr u2_steady = Function::solution(form.u(2), form.solution());
//
//    FunctionPtr squaredDiff = (p_transient-p_steady) * (p_transient-p_steady) + (u1_transient-u1_steady) * (u1_transient-u1_steady) + (u2_transient - u2_steady) * (u2_transient - u2_steady);
//    double valSquared = squaredDiff->integrate(form.solution()->mesh());
//    if (rank==0) cout << "L^2 norm of difference between converged transient and steady state solution (computed differently): " << sqrt(valSquared) << endl;
//
//    FunctionPtr p_diff_squared  =   (p_transient-p_steady) *   (p_transient-p_steady);
//    FunctionPtr u1_diff_squared = (u1_transient-u1_steady) * (u1_transient-u1_steady);
//    FunctionPtr u2_diff_squared = (u2_transient-u2_steady) * (u2_transient-u2_steady);
//
//    double p_diff_L2 = sqrt(p_diff_squared->integrate(form.solution()->mesh()));
//    double u1_diff_L2 = sqrt(u1_diff_squared->integrate(form.solution()->mesh()));
//    double u2_diff_L2 = sqrt(u2_diff_squared->integrate(form.solution()->mesh()));
//
//    if (rank==0) cout << "L^2 norm (computed differently) for p: " << p_diff_L2 << endl;
//    if (rank==0) cout << "L^2 norm (computed differently) for u1: " << u1_diff_L2 << endl;
//    if (rank==0) cout << "L^2 norm (computed differently) for u2: " << u2_diff_L2 << endl;
//  }
//
//  // by this point, we should have recovered something close to the steady solution.  Did we?
//  SolutionPtr transientSolution = transientForm.solution();
//  transientSolution->addSolution(form.solution(), -1.0);
//
//  double u1_diff_L2 = sqrt(transientSolution->L2NormOfSolutionGlobal(form.u(1)->ID()));
//  double u2_diff_L2 = sqrt(transientSolution->L2NormOfSolutionGlobal(form.u(2)->ID()));
//  double p_diff_L2 = sqrt(transientSolution->L2NormOfSolutionGlobal(form.p()->ID()));
//
//  double diff_L2 = sqrt(u1_diff_L2 * u1_diff_L2 + u2_diff_L2 * u2_diff_L2 + p_diff_L2 * p_diff_L2);
//
//  if (rank==0) cout << "L^2 norm of difference between converged transient and steady state solution: " << diff_L2 << endl;
//
//  if (rank==0) cout << "L^2 norm for p: " << p_diff_L2 << endl;
//  if (rank==0) cout << "L^2 norm for u1: " << u1_diff_L2 << endl;
//  if (rank==0) cout << "L^2 norm for u2: " << u2_diff_L2 << endl;

  return 0;
}