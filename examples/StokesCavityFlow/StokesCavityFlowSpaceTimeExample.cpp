#include "Teuchos_GlobalMPISession.hpp"

#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"

#include "EpetraExt_RowMatrixOut.h"

using namespace Camellia;

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class SpaceTimeRampBoundaryFunction_U1 : public SimpleFunction<double>
{
  double _eps; // ramp width (or height)
  bool _useVerticalBCs;
  int _polyOrder;
public:
  SpaceTimeRampBoundaryFunction_U1(double eps, bool useVerticalBCs, int polyOrder)
  {
    _eps = eps;
    _useVerticalBCs = useVerticalBCs;
    _polyOrder = polyOrder;
  }
  double value(double x, double y, double t)
  {
    double tol = 1e-14;
    if (! _useVerticalBCs)
    {
      if (abs(y-1.0) < tol)   // top boundary
      {
        if ( (abs(x) < _eps) )   // top left
        {
          double x_scaled = x / _eps;
          return pow(x_scaled, _polyOrder);
        }
        else if ( abs(1.0-x) < _eps)     // top right
        {
          double x_scaled = (1 - x) / _eps;
          return pow(x_scaled, _polyOrder);
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
    else
    {
      if (y >= 1 - _eps)
      {
        double y_scaled = (y - (1-_eps)) / _eps;
        return pow(y_scaled,_polyOrder);
      }
      else
      {
        return 0;
      }
    }
  }
  double value(double x, double y, double z, double t)
  {
    // bilinear interpolation with ramp of width _eps around top edges
    double tol = 1e-14;
    if (! _useVerticalBCs)
    {
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
    else
    {
      if (y >= 1 - _eps)
      {
        double y_scaled = (y - (1-_eps)) / _eps;
        return pow(y_scaled,_polyOrder);
      }
      else
      {
        return 0;
      }
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
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  int spaceDim = 2;
  double eps = 1.0 / 64.0;

  bool useCondensedSolve = true;
  bool useConformingTraces = true;
  double mu = 1.0;

  int polyOrder = 2, delta_k = 1, temporalPolyOrder=2;
  double t_0 = 0, t_final = 3;
  
  bool writeStiffnessMatrixAndExit = false;
  int numTimeElementsInitialMesh = -1;
  int spatialMeshWidth = 2;

  double energyErrorTarget = 1e-1;
  int maxRefinements = 4;
  
  int maxIterations = 2000;
  double cgTol = 1e-6;
  int azOutput = 100;
  bool solveIteratively = true;
  
  bool useTimeRamp = true;
  bool verticalBCInterpolation = true; // Paul Fischer recommends this: corresponds to the physical gap between lid and box
  
  bool includeVelocityTracesInFluxTerm = true; // officially the better way to do things, but I like setting this to false
  
  cmdp.setOption("useConformingTraces", "useNonconformingTraces", &useConformingTraces);
  cmdp.setOption("useCondensedSolve", "dontUseCondensedSolve", &useCondensedSolve);
  cmdp.setOption("useTimeRamp", "dontUseTimeRamp", &useTimeRamp);
  cmdp.setOption("errorTarget", &energyErrorTarget);
  cmdp.setOption("maxRefs", &maxRefinements);
  cmdp.setOption("mu", &mu);
  cmdp.setOption("rampWidthForLidBC", &eps);
  cmdp.setOption("spaceDim", &spaceDim);
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("temporalPolyOrder", &temporalPolyOrder);
  cmdp.setOption("writeStiffnessMatrixAndExit", "solveNormally", &writeStiffnessMatrixAndExit);
  cmdp.setOption("solveIteratively", "solveDirectly", &solveIteratively);
  cmdp.setOption("azOutput",&azOutput, "When using iterative solver, level of output (0 for none)");
  cmdp.setOption("maxIters", &maxIterations, "when using iterative solver, maximum # of iterations to converge");
  cmdp.setOption("cgTol", &cgTol, "when using iterative solver, tolerance for conjugate gradient convergence");
  cmdp.setOption("verticalBCInterpolation","horizontalBCInterpolation", &verticalBCInterpolation);
  cmdp.setOption("includeVelocityTracesInFluxTerm", "dontIncludeVelocityTracesInFluxTerm", &includeVelocityTracesInFluxTerm);
  cmdp.setOption("timeElementsInitial", &numTimeElementsInitialMesh, "# elements in time direction in initial mesh (default: 1 per unit time)");
  cmdp.setOption("spatialMeshWidth", &spatialMeshWidth, "# elements in each spatial direction in initial mesh");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  vector<double> dims(spaceDim,1.0);
  vector<int> numElements(spaceDim,spatialMeshWidth);
  vector<double> x0(spaceDim,0.0);
  
  StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces, includeVelocityTracesInFluxTerm);

  // create a mesh topology corresponding to the spatial elements you'd like
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  // create a space-time mesh topology with one element per unit time, if user did not specify the number
  int numTimeElements = (numTimeElementsInitialMesh == -1) ? (int)(t_final-t_0) : numTimeElementsInitialMesh;

  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t_0, t_final, numTimeElements);
  spaceTimeForm.initializeSolution(spaceTimeMeshTopo, polyOrder, delta_k, Teuchos::null, temporalPolyOrder);
  
  // define a time ramp - a function defined as:
  //   t   for t in [0,1]
  //   1   for t > 1
  FunctionPtr timeRamp;
  if (useTimeRamp)
    timeRamp = Teuchos::rcp(new TimeRampSpaceTime(1.0));
  else
    timeRamp = Function::constant(1.0);

  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
  SpatialFilterPtr sides;
  SpatialFilterPtr bottom = SpatialFilter::matchingY(0.0);;
  if (spaceDim == 2)
  {
    sides = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingX(1.0);
  }
  else if (spaceDim == 3)
  {
    sides= SpatialFilter::matchingX(0.0) | SpatialFilter::matchingX(1.0) | SpatialFilter::matchingZ(0.0) | SpatialFilter::matchingZ(1.0);
  }

  FunctionPtr zero = Function::zero();
  FunctionPtr u1_top, u1_side;
  if (verticalBCInterpolation)
  {
    u1_top = Function::constant(1.0);
    u1_side = Teuchos::rcp(new SpaceTimeRampBoundaryFunction_U1(eps,verticalBCInterpolation,polyOrder));
  }
  else
  {
    u1_top = Teuchos::rcp(new SpaceTimeRampBoundaryFunction_U1(eps,verticalBCInterpolation,polyOrder));
    u1_side = zero;
  }
  
  vector<FunctionPtr> u_top_vector, u_side_vector;
  FunctionPtr u_top, u_side;
  if (spaceDim == 2)
  {
    u_top_vector = {u1_top,zero};
    u_side_vector = {u1_side,zero};
  }
  else
  {
    u_top_vector = {u1_top,zero,zero};
    u_side_vector = {u1_side,zero,zero};
  }
  
  u_top = Function::vectorize(u_top_vector);
  u_side = Function::vectorize(u_side_vector);

  if (useTimeRamp)
  {
    spaceTimeForm.addZeroInitialCondition(t_0);
  }
  else
  {
    if (verticalBCInterpolation)
      spaceTimeForm.addInitialCondition(t_0, u_side_vector);
    else
      spaceTimeForm.addInitialCondition(t_0, u_top_vector);
  }
  
  spaceTimeForm.addWallCondition(bottom);
  spaceTimeForm.addInflowCondition(sides, timeRamp * u_side);
  spaceTimeForm.addInflowCondition(topBoundary, timeRamp * u_top);
  if (spatialMeshWidth > 1)
    spaceTimeForm.addPointPressureCondition({0.5,0.5});
  else
    spaceTimeForm.addPointPressureCondition({0.0,0.0});
  
  if (writeStiffnessMatrixAndExit)
  {
    SolutionPtr solution = spaceTimeForm.solution();
    solution->initializeLHSVector();
    solution->initializeStiffnessAndLoad();
    solution->populateStiffnessAndLoad();
    
    int numFieldDofs = solution->mesh()->numFieldDofs();
    int numFluxDofs = solution->mesh()->numFluxDofs();
    
    if (rank==0) cout << "# field dofs: " << numFieldDofs << endl;
    if (rank==0) cout << "# flux/trace dofs: " << numFluxDofs << endl;
    
    Teuchos::RCP<Epetra_CrsMatrix> A = solution->getStiffnessMatrix();
    string fileName = "A.dat";
    if (rank==0) cout << "writing stiffness matrix to " << fileName << ".\n";
    EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(),*A, NULL, NULL, false);
    
    return 0;
  }
  spaceTimeForm.solution()->setUseCondensedSolve(useCondensedSolve);
  if (solveIteratively)
    spaceTimeForm.solveIteratively(maxIterations, cgTol, azOutput, false);
  else
    spaceTimeForm.solve();
  
  double energyError = spaceTimeForm.solution()->energyErrorTotal();
  MeshPtr spaceTimeMesh = spaceTimeForm.solution()->mesh();
  int globalDofs = spaceTimeMesh->globalDofCount();
  int activeElements = spaceTimeMesh->getTopology()->getActiveCellIndices().size();
  if (rank==0) cout << "Initial energy error for space-time mesh: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  
  HDF5Exporter spaceTimeExporter(spaceTimeMesh, "stokesSpaceTimeCavityFlow", ".");
  spaceTimeExporter.exportSolution(spaceTimeForm.solution(), 0);
  
  HDF5Exporter spaceTimeExporterTimeVelocityTraces(spaceTimeMesh, "stokesSpaceTimeCavityFlowTimeVelocityTraces", ".");
  
  FunctionPtr n_t = Function::normalSpaceTime()->t();
  FunctionPtr sideParity = Function::sideParity();
  FunctionPtr u1_temporal = Function::solution(spaceTimeForm.tn_hat(1), spaceTimeForm.solution()) * n_t * sideParity;
  FunctionPtr u2_temporal = Function::solution(spaceTimeForm.tn_hat(2), spaceTimeForm.solution()) * n_t * sideParity;
  
  if (includeVelocityTracesInFluxTerm)
  {
    spaceTimeExporterTimeVelocityTraces.exportFunction({u1_temporal,u2_temporal}, {"u1_hat - temporal", "u2_hat - temporal"}, 0);
  }
  
  int refNumber = 0;
  while ((energyError > energyErrorTarget) && (refNumber < maxRefinements))
  {
    refNumber++;
    spaceTimeForm.refine();
    if (solveIteratively)
      spaceTimeForm.solveIteratively(maxIterations, cgTol, azOutput, false);
    else
      spaceTimeForm.solve();
    
    spaceTimeExporter.exportSolution(spaceTimeForm.solution(), refNumber);
    if (includeVelocityTracesInFluxTerm)
    {
      spaceTimeExporterTimeVelocityTraces.exportFunction({u1_temporal,u2_temporal}, {"u1_hat - temporal", "u2_hat - temporal"}, refNumber);
    }
    
    energyError = spaceTimeForm.solution()->energyErrorTotal();
    globalDofs = spaceTimeMesh->globalDofCount();
    activeElements = spaceTimeMesh->getTopology()->getActiveCellIndices().size();
    if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  }
  
  return 0;
}