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

  bool useConformingTraces = true;
  double mu = 1.0;

  int polyOrder = 2, delta_k = 1;
  double t_0 = 0, t_final = 3;
  
  bool writeStiffnessMatrixAndExit = false;
  int numTimeElementsInitialMesh = -1;
  int spatialMeshWidth = 2;

  double energyErrorTarget = 1e-1;
  int maxRefinements = 4;
  
  cmdp.setOption("useConformingTraces", "useNonconformingTraces", &useConformingTraces);
  cmdp.setOption("errorTarget", &energyErrorTarget);
  cmdp.setOption("maxRefs", &maxRefinements);
  cmdp.setOption("mu", &mu);
  cmdp.setOption("rampWidthForLidBC", &eps);
  cmdp.setOption("spaceDim", &spaceDim);
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("writeStiffnessMatrixAndExit", "solveNormally", &writeStiffnessMatrixAndExit);
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
  
  StokesVGPFormulation spaceTimeForm = StokesVGPFormulation::spaceTimeFormulation(spaceDim, mu, useConformingTraces);

  // create a mesh topology corresponding to the spatial elements you'd like
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  // create a space-time mesh topology with one element per unit time, if user did not specify the number
  int numTimeElements = (numTimeElementsInitialMesh == -1) ? (int)(t_final-t_0) : numTimeElementsInitialMesh;

  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t_0, t_final, numTimeElements);
  spaceTimeForm.initializeSolution(spaceTimeMeshTopo, polyOrder, delta_k);
  
  // define a time ramp - a function defined as:
  //   t   for t in [0,1]
  //   1   for t > 1
  FunctionPtr timeRamp = Teuchos::rcp(new TimeRampSpaceTime(1.0));

  SpatialFilterPtr initialTime = SpatialFilter::matchingT(t_0);
  SpatialFilterPtr finalTime = SpatialFilter::matchingT(t_final);
  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
  SpatialFilterPtr sideAndBottomWalls;
  if (spaceDim == 2)
  {
    sideAndBottomWalls = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingX(1.0) | SpatialFilter::matchingY(0.0);
  }
  else if (spaceDim == 3)
  {
    sideAndBottomWalls = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingX(1.0) | SpatialFilter::matchingY(0.0) | SpatialFilter::matchingZ(0.0) | SpatialFilter::matchingZ(1.0);
  }

  FunctionPtr zero = Function::zero();
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

  spaceTimeForm.addZeroInitialCondition(t_0);
  spaceTimeForm.addWallCondition(sideAndBottomWalls);
  spaceTimeForm.addInflowCondition(topBoundary, timeRamp * u_topRampSpaceTime);
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
  spaceTimeForm.solve();
  
  double energyError = spaceTimeForm.solution()->energyErrorTotal();
  MeshPtr spaceTimeMesh = spaceTimeForm.solution()->mesh();
  int globalDofs = spaceTimeMesh->globalDofCount();
  int activeElements = spaceTimeMesh->getTopology()->getActiveCellIndices().size();
  if (rank==0) cout << "Initial energy error for space-time mesh: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  
  HDF5Exporter spaceTimeExporter(spaceTimeMesh, "stokesSpaceTimeCavityFlow", ".");
  spaceTimeExporter.exportSolution(spaceTimeForm.solution(), 0);
  
  int refNumber = 0;
  while ((energyError > energyErrorTarget) && (refNumber < maxRefinements))
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
  
  return 0;
}