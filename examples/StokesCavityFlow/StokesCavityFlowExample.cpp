#include "Teuchos_GlobalMPISession.hpp"

#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "TimeSteppingConstants.h"

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

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class SpaceTimeRampBoundaryFunction_U1 : public SimpleFunction<double>
{
  double _eps; // ramp width
public:
  SpaceTimeRampBoundaryFunction_U1(double eps)
  {
    _eps = eps;
  }
  double value(double x, double y, double t)
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
  double value(double x, double y, double z, double t)
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

// TODO: add this Function to the core Camellia library.
class TimeRampSpaceTime : public SimpleFunction<double>
{
  double _timeScale;
public:
  TimeRampSpaceTime(double timeScale)
  {
    _timeScale = timeScale;
  }
  double value(double x, double t)
  {
    if (t >= _timeScale)
    {
      return 1.0;
    }
    else
    {
      return t / _timeScale;
    }
  }
  double value(double x, double y, double t)
  {
    if (t >= _timeScale)
    {
      return 1.0;
    }
    else
    {
      return t / _timeScale;
    }
  }
  double value(double x, double y, double z, double t)
  {
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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  int spaceDim = 2;

  double eps = 1.0 / 64.0;

  bool useConformingTraces = true;
  double mu = 1.0;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);

  vector<double> dims(spaceDim,1.0);
  vector<int> numElements(spaceDim,2);
  vector<double> x0(spaceDim,0.0);

  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  int polyOrder = 2, delta_k = 1;

  form.initializeSolution(meshTopo, polyOrder, delta_k);
  form.addZeroMeanPressureCondition();

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

  form.solve();

  MeshPtr mesh = form.solution()->mesh();
  double energyError = form.solution()->energyErrorTotal();
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->getTopology()->getActiveCellIndices().size();
  if (rank==0) cout << "Initial energy error: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;

  HDF5Exporter exporter(mesh, "stokesInitialSolution", ".");
  exporter.exportSolution(form.solution());

  double tol = 8e-1;
  int refNumber = 0;
  do
  {
    refNumber++;
    form.refine();
    form.solve();

    ostringstream exportName;
    exportName << "stokesCavityFlowSolution_ref" << refNumber;
    HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), ".");
    exporter.exportSolution(form.solution());

    energyError = form.solution()->energyErrorTotal();
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->getTopology()->getActiveCellIndices().size();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  }
  while (energyError > tol);

  // save the formulation for possible later reloading:
  string savePrefix = "stokesExample";
  if (rank==0) cout << "Saving to " << savePrefix << endl;
  form.save(savePrefix);
  if (rank==0) cout << "...saved.\n";

  // try loading:
  if (rank==0) cout << "loading saved solution...\n";
  form.initializeSolution(savePrefix, polyOrder, delta_k);
  if (rank==0) cout << "...loaded.\n";
  FunctionPtr u1_steady = Function::solution(form.u(1), form.solution());
  double centerValue = u1_steady->evaluate(form.solution()->mesh(), 0.5, 0.5); // MUST call this on every MPI rank -- MPI collective method.
  if (rank==0) cout << "u1(0.5, 0.5) = " << centerValue << endl;

  // now solve for the stream function on the fine mesh:
  form.streamSolution()->solve();
  HDF5Exporter steadyStreamExporter(form.streamSolution()->mesh(), "stokesSteadyCavityFlowStreamSolution", ".");
  steadyStreamExporter.exportSolution(form.streamSolution());

  /*   Now that we have a fine mesh, try the same problem, but transient, starting with a zero initial
   *   state, and with boundary conditions that "ramp up" in time (and which also are zero at time 0).
   *   We expect to recover the steady solution.
   */

  double totalTime = 3;
  double dt = 0.5;
  int numTimeSteps = ceil(totalTime / dt);
  StokesVGPFormulation transientForm = StokesVGPFormulation::timeSteppingFormulation(spaceDim, mu, dt, useConformingTraces, BACKWARD_EULER);

  FunctionPtr t = transientForm.getTimeFunction();
  FunctionPtr timeRamp = Teuchos::rcp(new TimeRamp(t,1.0));

  transientForm.initializeSolution(meshTopo, polyOrder, delta_k);
  transientForm.addZeroMeanPressureCondition();
  transientForm.addWallCondition(notTopBoundary);
  transientForm.addInflowCondition(topBoundary, timeRamp * u_topRamp);

  MeshPtr transientMesh = transientForm.solution()->mesh();
  HDF5Exporter transientExporter(transientMesh, "stokesTransientSolution", ".");

  for (int timeStep=0; timeStep<numTimeSteps; timeStep++)
  {
    transientForm.solve();
    double L2_step = transientForm.relativeL2NormOfTimeStep();
    transientExporter.exportSolution(transientForm.solution(),transientForm.getTime());

    transientForm.takeTimeStep();
    if (rank==0) cout << "time step " << timeStep << " completed (L^2 norm of difference from prev: " << L2_step << ").\n";
  }

  // by this point, we should have recovered something close to the steady solution.  Did we?
  SolutionPtr transientSolution = transientForm.solution();
  transientSolution->addSolution(form.solution(), -1.0);

  double u1_diff_L2 = transientSolution->L2NormOfSolutionGlobal(form.u(1)->ID());
  double u2_diff_L2 = transientSolution->L2NormOfSolutionGlobal(form.u(2)->ID());
  double p_diff_L2 = transientSolution->L2NormOfSolutionGlobal(form.p()->ID());

  double diff_L2 = sqrt(u1_diff_L2 * u1_diff_L2 + u2_diff_L2 * u2_diff_L2 + p_diff_L2 * p_diff_L2);

  if (rank==0) cout << "L^2 norm of difference between converged transient and steady state solution: " << diff_L2 << endl;

  if (rank==0) cout << "L^2 norm for p: " << p_diff_L2 << endl;
  if (rank==0) cout << "L^2 norm for u1: " << u1_diff_L2 << endl;
  if (rank==0) cout << "L^2 norm for u2: " << u2_diff_L2 << endl;

  // finally, let's try a space-time formulation:
  // Conforming traces aren't yet supported for space-time elements.  The issue has to do with
  // certain assumptions made in GDAMinimumRule -- basically it assumes that traces will be defined on each
  // side of the element, which isn't true for space-time.  For various reasons this is not an issue when
  // the only continuity being enforced is through the sides.
  useConformingTraces = false;
  StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);

  double t0 = 0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, totalTime, (int)totalTime);
  spaceTimeForm.initializeSolution(spaceTimeMeshTopo, polyOrder, delta_k);
  
  // redefine the time ramp:
  timeRamp = Teuchos::rcp(new TimeRampSpaceTime(1.0));

  spaceTimeForm.addZeroInitialCondition(t0);
  SpatialFilterPtr initialTime = SpatialFilter::matchingT(t0);
  SpatialFilterPtr notInitialTime = SpatialFilter::negatedFilter(initialTime);
  topBoundary = SpatialFilter::matchingY(1) & notInitialTime;
  notTopBoundary = (!SpatialFilter::matchingY(1)) & notInitialTime;
  spaceTimeForm.addWallCondition(notTopBoundary);
  FunctionPtr u1_topRampSpaceTime = Teuchos::rcp(new SpaceTimeRampBoundaryFunction_U1(eps));
  FunctionPtr u_topRampSpaceTime;
  if (spaceDim == 2)
  {
    u_topRampSpaceTime = Function::vectorize(u1_topRampSpaceTime,zero);
  }
  else
  {
    u_topRampSpaceTime = Function::vectorize(u1_topRampSpaceTime,zero,zero);
  }

  spaceTimeForm.addInflowCondition(topBoundary, timeRamp * u_topRampSpaceTime);
  spaceTimeForm.addPointPressureCondition({0.5,1.0});

  MeshPtr spaceTimeMesh = spaceTimeForm.solution()->mesh();
  spaceTimeForm.solve();
  
  energyError = spaceTimeForm.solution()->energyErrorTotal();
  globalDofs = spaceTimeMesh->globalDofCount();
  activeElements = spaceTimeMesh->getTopology()->getActiveCellIndices().size();
  if (rank==0) cout << "Initial energy error for space-time mesh: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  
  HDF5Exporter spaceTimeExporter(spaceTimeMesh, "stokesSpaceTimeCavityFlow", ".");
  spaceTimeExporter.exportSolution(spaceTimeForm.solution(), 0);
  
  tol = 8e-1;
  refNumber = 0;
  do
  {
    refNumber++;
    spaceTimeForm.refine();
    spaceTimeForm.solve();
    
    spaceTimeExporter.exportSolution(spaceTimeForm.solution(), refNumber);
    
    energyError = spaceTimeForm.solution()->energyErrorTotal();
    globalDofs = spaceTimeMesh->globalDofCount();
    activeElements = spaceTimeMesh->getTopology()->getActiveCellIndices().size();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  }
  while (energyError > tol);
  
  return 0;
}