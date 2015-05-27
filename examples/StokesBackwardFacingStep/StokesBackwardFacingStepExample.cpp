#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PreviousSolutionFunction.h"
#include "StokesVGPFormulation.h"

using namespace Camellia;

class TimeRamp : public SimpleFunction
{
  FunctionPtr _time;
  double _timeScale;
  double getTimeValue()
  {
    ParameterFunction* timeParamFxn = dynamic_cast<ParameterFunction*>(_time.get());
    SimpleFunction* timeFxn = dynamic_cast<SimpleFunction*>(timeParamFxn->getValue().get());
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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  int spaceDim = 2;

  bool useConformingTraces = false;
  double mu = 1.0;
  StokesVGPFormulation form(spaceDim, useConformingTraces, mu);

  // set up a mesh topology for the region east of the step: 2 x 2 elements, bottom left corner at (4,0)
  static double X_LEFT = 0.0;
  static double X_STEP = 4.0;
  static double X_OUTFLOW = 8.0;
  static double WIDTH_EAST_REGION = X_OUTFLOW - X_STEP;
  static double Y_BOTTOM = 0.0;
  static double Y_TOP = 2.0;
  vector<double> dims(spaceDim);
  dims[0] = WIDTH_EAST_REGION;
  dims[1] = Y_TOP;
  vector<int> numElements(spaceDim);
  numElements[0] = ceil(WIDTH_EAST_REGION);
  numElements[1] = 2;
  vector<double> x0(spaceDim);
  x0[0] = X_STEP;
  x0[1] = Y_BOTTOM;

  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  // add approximately unit-width elements west of the region just defined:
  static const double Y_STEP = (Y_TOP - Y_BOTTOM) / 2.0;
  int numWestElements = ceil( (X_STEP - X_LEFT) );
  double westElemWidth = (X_STEP - X_LEFT) / numWestElements;
  for (int i=0; i<numWestElements; i++)
  {
    // vertices are defined in counter-clockwise order:
    vector<double> x1(spaceDim), x2(spaceDim), x3(spaceDim);
    x0[0] = X_LEFT + i * westElemWidth;
    x0[1] = Y_STEP;
    x1[0] = X_LEFT + (i+1) * westElemWidth;
    x1[1] = Y_STEP;
    x2[0] = X_LEFT + (i+1) * westElemWidth;
    x2[1] = Y_TOP;
    x3[0] = X_LEFT + i * westElemWidth;
    x3[1] = Y_TOP;
    vector< vector<double> > vertices(4);
    vertices[0] = x0;
    vertices[1] = x1;
    vertices[2] = x2;
    vertices[3] = x3;
    meshTopo->addCell(CellTopology::quad(), vertices);
  }

  int polyOrder = 3, delta_k = 1;

  form.initializeSolution(meshTopo, polyOrder, delta_k);

  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);

  SpatialFilterPtr inflow = SpatialFilter::matchingX(X_LEFT);
  SpatialFilterPtr outflow = SpatialFilter::matchingX(X_OUTFLOW);
  // wherever is not inflow or outflow is a wall:
  SpatialFilterPtr wall = SpatialFilter::negatedFilter(outflow | inflow);

  FunctionPtr y = Function::yn(1);
  FunctionPtr u1_inflow = -6 * (y-2.) * (y-1.);

  FunctionPtr u_inflow;
  FunctionPtr zero = Function::zero();
  if (spaceDim == 2)
  {
    u_inflow = Function::vectorize(u1_inflow,zero);
  }
  else
  {
    u_inflow = Function::vectorize(u1_inflow,zero,zero);
  }

  form.addInflowCondition(inflow, u_inflow);
  form.addWallCondition(wall);
  form.addOutflowCondition(outflow);

  form.solve();

  MeshPtr mesh = form.solution()->mesh();
  double energyError = form.solution()->energyErrorTotal();
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->getTopology()->activeCellCount();
  if (rank==0) cout << "Initial energy error: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;

  HDF5Exporter exporter(mesh, "stokesSteadyBackwardStep", ".");
  exporter.exportSolution(form.solution());

  double tol = 0.01;
  int refNumber = 0;
  int maxRefs = 5;
  do
  {
    refNumber++;
    form.refine();
    form.solve();

    ostringstream exportName;
    exportName << "stokesSteadyBackwardStep_ref" << refNumber;
    HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), ".");
    exporter.exportSolution(form.solution());

    energyError = form.solution()->energyErrorTotal();
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->getTopology()->activeCellCount();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  }
  while ((energyError > tol) && (refNumber < maxRefs));

  // now solve for the stream function on the fine mesh:
  form.streamSolution()->solve();
  HDF5Exporter steadyStreamExporter(form.streamSolution()->mesh(), "stokesSteadyBackwardStepStreamSolution", ".");
  steadyStreamExporter.exportSolution(form.streamSolution());

  VarPtr u1 = form.u(1), u2 = form.u(2);

  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(form.solution(), - u1->dy() + u2->dx()) );

  HDF5Exporter exporterVorticity(form.solution()->mesh(), "stokesSteadyBackwardStepVorticity", ".");
  exporterVorticity.exportFunction(vorticity, "vorticity");

  /*   Now that we have a fine mesh, try the same problem, but transient, starting with a zero initial
   *   state, and with boundary conditions that "ramp up" in time (and which also are zero at time 0).
   *   We expect to recover the steady solution.
   */

  double totalTime = 2;
  double dt = 1e-1;
  int numTimeSteps = ceil(totalTime / dt);
  StokesVGPFormulation transientForm(spaceDim, useConformingTraces, mu, true, dt);

  FunctionPtr t = transientForm.getTimeFunction();
  FunctionPtr timeRamp = Teuchos::rcp(new TimeRamp(t,1.0));

  transientForm.initializeSolution(meshTopo, polyOrder, delta_k);

  transientForm.addWallCondition(wall);
  transientForm.addOutflowCondition(outflow);

  bool debugging = true;
  if (debugging)
  {
    // DEBUGGING: try projecting the steady pressure onto the previous time step
    //            (for this to work, need to dispense with the time ramp)
    transientForm.addInflowCondition(inflow, u_inflow);
    transientForm.solutionPreviousTimeStep()->addSolution(form.solution(), 1.0);

    cout << "DEBUGGING: starting transient solution at the steady solution.";
  }
  else
  {
    transientForm.addInflowCondition(inflow, timeRamp * u_inflow);
  }

  MeshPtr transientMesh = transientForm.solution()->mesh();
  HDF5Exporter transientExporter(transientMesh, "stokesTransientBackwardStep", ".");


  for (int timeStep=0; timeStep<numTimeSteps; timeStep++)
  {
    transientForm.solve();
    double L2_step = transientForm.L2NormOfTimeStep();
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

  if (rank==0)
  {
    cout << "L^2 norm of differences between converged transient and steady state solution:\n";
    cout << "u1:" << setw(10) << u1_diff_L2 << endl;
    cout << "u2:" << setw(10) << u2_diff_L2 << endl;
    cout << "p: " << setw(10) <<  p_diff_L2 << endl;
    cout << "combined:" << setw(10) << diff_L2 << endl;
  }

  return 0;
}