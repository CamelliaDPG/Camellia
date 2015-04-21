#include "Teuchos_GlobalMPISession.hpp"
#include "NavierStokesVGPFormulation.h"
#include "Function.h"
#include "MeshFactory.h"
#include "HDF5Exporter.h"

using namespace Camellia;

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class RampBoundaryFunction_U1 : public SimpleFunction {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    double tol = 1e-14;
    if (abs(y-1.0) < tol) { // top boundary
      if ( (abs(x) < _eps) ) { // top left
        return x / _eps;
      } else if ( abs(1.0-x) < _eps) { // top right
        return (1.0-x) / _eps;
      } else { // top middle
        return 1;
      }
    } else { // not top boundary: 0.0
      return 0.0;
    }
  }
  double value(double x, double y, double z) {
    // bilinear interpolation with ramp of width _eps around top edges
    double tol = 1e-14;
    if (abs(y-1.0) <tol) {
      double xFactor = 1.0;
      double zFactor = 1.0;
      if ( (abs(x) < _eps) ) { // top left
        xFactor = x / _eps;
      } else if ( abs(1.0-x) < _eps) { // top right
        xFactor = (1.0-x) / _eps;
      }
      if ( (abs(z) < _eps) ) { // top back
        zFactor = z / _eps;
      } else if ( abs(1.0-z) < _eps) { // top front
        zFactor = (1.0-z) / _eps;
      }
      return xFactor * zFactor;
    } else {
      return 0.0;
    }
  }
};

class TimeRamp : public SimpleFunction {
  FunctionPtr _time;
  double _timeScale;
  double getTimeValue() {
    ParameterFunction* timeParamFxn = dynamic_cast<ParameterFunction*>(_time.get());
    SimpleFunction* timeFxn = dynamic_cast<SimpleFunction*>(timeParamFxn->getValue().get());
    return timeFxn->value(0);
  }
public:
  TimeRamp(FunctionPtr timeConstantParamFxn, double timeScale) {
    _time = timeConstantParamFxn;
    _timeScale = timeScale;
  }
  double value(double x) {
    double t = getTimeValue();
    if (t >= _timeScale) {
      return 1.0;
    } else {
      return t / _timeScale;
    }
  }
};

using namespace std;

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  int spaceDim = 2;
  
  double eps = 1.0 / 64.0;
  
  bool useConformingTraces = true;
  double Re = 1000.0;
  
  vector<double> dims(spaceDim,1.0);
  vector<int> numElements(spaceDim,2);
  vector<double> x0(spaceDim,0.0);
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
  
  int polyOrder = 3, delta_k = 1;
  
  NavierStokesVGPFormulation form(meshTopo, Re, polyOrder);
  
  form.addZeroMeanPressureCondition();

  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
  
  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
  SpatialFilterPtr notTopBoundary = SpatialFilter::negatedFilter(topBoundary);
  form.addWallCondition(notTopBoundary);
  
  FunctionPtr u1_topRamp = Teuchos::rcp( new RampBoundaryFunction_U1(eps) );
  FunctionPtr u_topRamp;
  FunctionPtr zero = Function::zero();
  if (spaceDim == 2) {
    u_topRamp = Function::vectorize(u1_topRamp,zero);
  } else {
    u_topRamp = Function::vectorize(u1_topRamp,zero,zero);
  }
  form.addInflowCondition(topBoundary, u_topRamp);
  
  double nonlinearThreshold = 1e-3;
  double maxNonlinearIterations = 10;
  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;
  while ((l2NormOfIncrement > nonlinearThreshold) && (stepNumber < maxNonlinearIterations)) {
    form.solveAndAccumulate();
    l2NormOfIncrement = form.L2NormSolutionIncrement();
    stepNumber++;
    cout << "L^2 norm of increment: " << l2NormOfIncrement << endl;
  }

  MeshPtr mesh = form.solution()->mesh();
  string outputDir = "/tmp";
  HDF5Exporter exporter(mesh, "navierStokesInitialSolution", outputDir);
  exporter.exportSolution(form.solution());

  double energyError = form.solutionIncrement()->energyErrorTotal();
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->getTopology()->activeCellCount();
  if (rank==0) cout << "Initial energy error: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;

  double tol = 1e-2;
  int refNumber = 0;
  do {
    refNumber++;
    form.refine();
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    while ((l2NormOfIncrement > nonlinearThreshold) && (stepNumber < maxNonlinearIterations)) {
      form.solveAndAccumulate();
      l2NormOfIncrement = form.L2NormSolutionIncrement();
      stepNumber++;
      cout << "Refinement " << refNumber << ", L^2 norm of increment: " << l2NormOfIncrement << endl;
    }
    
    ostringstream exportName;
    exportName << "navierStokesCavityFlowSolution_ref" << refNumber + 1;
    HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), outputDir);
    exporter.exportSolution(form.solution());
    
    energyError = form.solutionIncrement()->energyErrorTotal();
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->getTopology()->activeCellCount();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  } while (energyError > tol);
  
  FunctionPtr u1_steady = Function::solution(form.u(1), form.solution());
  cout << "u1(0.5, 0.5) = " << u1_steady->evaluate(0.5, 0.5) << endl;
  
  // now solve for the stream function on the fine mesh:
  form.streamSolution()->solve();
  HDF5Exporter steadyStreamExporter(form.streamSolution()->mesh(), "navierStokesSteadyCavityFlowStreamSolution", outputDir);
  steadyStreamExporter.exportSolution(form.streamSolution());
  
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