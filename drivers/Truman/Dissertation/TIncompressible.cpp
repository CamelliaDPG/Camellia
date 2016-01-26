#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "SimpleFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PenaltyConstraints.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"

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

class NearCylinder : public SpatialFilter
{
  double _enlarged_radius;
public:
  NearCylinder(double radius)
  {
    double enlargement_factor = 1.1*sqrt(2);
    _enlarged_radius = radius * enlargement_factor;
  }
  bool matchesPoint(double x, double y)
  {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z)
  {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

void preprocessHemkerMesh(MeshPtr hemkerMeshNoCurves, bool steady, double tFinal)
{
  double radius = 0.5;
  bool enforceOneIrregularity = true;

  Intrepid::FieldContainer<double> horizontalBandPoints(6,hemkerMeshNoCurves->getDimension());
  // ESE band
  horizontalBandPoints(0,0) =   radius * 3;
  horizontalBandPoints(0,1) = - radius / 2;
  // ENE band
  horizontalBandPoints(1,0) =   radius * 3;
  horizontalBandPoints(1,1) =   radius / 2;
  // WSW band
  horizontalBandPoints(2,0) = - radius * 3;
  horizontalBandPoints(2,1) = - radius / 2;
  // WNW band
  horizontalBandPoints(3,0) = - radius * 3;
  horizontalBandPoints(3,1) =   radius / 2;
  // the bigger, fatter guys in the corners count as horizontal bands (because that's the direction of their anisotropy)
  // NE big element
  horizontalBandPoints(4,0) = radius * 3;
  horizontalBandPoints(4,1) = radius * 3;
  // SE big element
  horizontalBandPoints(5,0) =   radius * 3;
  horizontalBandPoints(5,1) = - radius * 3;

  Intrepid::FieldContainer<double> verticalBandPoints(4,hemkerMeshNoCurves->getDimension());
  // NNE band
  verticalBandPoints(0,0) =   radius / 2;
  verticalBandPoints(0,1) =   radius * 3;
  // NNW band
  verticalBandPoints(1,0) = - radius / 2;
  verticalBandPoints(1,1) =   radius * 3;
  // SSE band
  verticalBandPoints(2,0) =   radius / 2;
  verticalBandPoints(2,1) = - radius * 3;
  // SSE band
  verticalBandPoints(3,0) = - radius / 2;
  verticalBandPoints(3,1) = - radius * 3;

  if (!steady)
  {
    // TODO: (for Truman) consider what happens if _numSlabs != 1
    double temporalMidpoint = (0 + tFinal) / 2.0;
    int d_time = hemkerMeshNoCurves->getDimension() - 1;
    int numHorizontalPoints = horizontalBandPoints.dimension(1);
    for (int pointOrdinal=0; pointOrdinal<numHorizontalPoints; pointOrdinal++)
    {
      horizontalBandPoints(pointOrdinal,d_time) = temporalMidpoint;
    }
    int numVerticalPoints = verticalBandPoints.dimension(1);
    for (int pointOrdinal=0; pointOrdinal<numVerticalPoints; pointOrdinal++)
    {
      verticalBandPoints(pointOrdinal,d_time) = temporalMidpoint;
    }
  }

  vector< GlobalIndexType > horizontalBandCellIDs = hemkerMeshNoCurves->cellIDsForPoints(horizontalBandPoints, false);
  vector< GlobalIndexType > verticalBandCellIDs = hemkerMeshNoCurves->cellIDsForPoints(verticalBandPoints, false);

  // check results
  for (GlobalIndexType cellID : horizontalBandCellIDs)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellID == -1, std::invalid_argument, "horizontal band cell not found!");
  }
  for (GlobalIndexType cellID : verticalBandCellIDs)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellID == -1, std::invalid_argument, "vertical band cell not found!");
  }

  RefinementPatternPtr verticalCut, horizontalCut;
  Intrepid::FieldContainer<double> vertices;

  if (!steady)
  {
    verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuadTimeExtruded();
    horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuadTimeExtruded();
    vertices.resize(8,3);
  }
  else
  {
    verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
    horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuad();
    vertices.resize(4,2);
  }

  // horizontal bands want vertical cuts, and vice versa
  for (vector<GlobalIndexType>::iterator cellIDIt = horizontalBandCellIDs.begin();
      cellIDIt != horizontalBandCellIDs.end(); cellIDIt++)
  {
    int cellID = *cellIDIt;
    // cout << "Refining cell " << cellID << endl;
    //    cout << "Identified cell " << cellID << " as a horizontal band.\n";
    // work out what the current aspect ratio is
    hemkerMeshNoCurves->verticesForCell(vertices, cellID);
    //    cout << "vertices for cell " << cellID << ":\n" << vertices;
    // here, we use knowledge of the implementation of the hemker mesh generation:
    // we know that the first edges are always horizontal...
    double xDiff = abs(vertices(1,0)-vertices(0,0));
    double yDiff = abs(vertices(2,1)-vertices(1,1));

    //    cout << "xDiff: " << xDiff << endl;
    //    cout << "yDiff: " << yDiff << endl;

    set<GlobalIndexType> cellIDsToRefine;
    cellIDsToRefine.insert(cellID);
    double aspect = xDiff / yDiff;
    while (aspect > 2.0)
    {
      //      cout << "aspect ratio: " << aspect << endl;
      hemkerMeshNoCurves->hRefine(cellIDsToRefine, verticalCut);

      // the next set of cellIDsToRefine are the children of the ones just refined
      set<GlobalIndexType> childCellIDs;
      for (set<GlobalIndexType>::iterator refinedCellIDIt = cellIDsToRefine.begin();
          refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++)
      {
        int refinedCellID = *refinedCellIDIt;
        set<GlobalIndexType> refinedCellChildren = hemkerMeshNoCurves->getTopology()->getCell(refinedCellID)->getDescendants(hemkerMeshNoCurves->getTopology());
        childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
      }

      cellIDsToRefine = childCellIDs;
      aspect /= 2;
    }
  }

  // horizontal bands want vertical cuts, and vice versa
  for (vector<GlobalIndexType>::iterator cellIDIt = verticalBandCellIDs.begin();
      cellIDIt != verticalBandCellIDs.end(); cellIDIt++)
  {
    int cellID = *cellIDIt;
    // cout << "Refining cell " << cellID << endl;
    //    cout << "Identified cell " << cellID << " as a vertical band.\n";
    // work out what the current aspect ratio is
    hemkerMeshNoCurves->verticesForCell(vertices, cellID);
    // here, we use knowledge of the implementation of the hemker mesh generation:
    // we know that the first edges are always horizontal...
    double xDiff = abs(vertices(1,0)-vertices(0,0));
    double yDiff = abs(vertices(2,1)-vertices(1,1));

    set<GlobalIndexType> cellIDsToRefine;
    cellIDsToRefine.insert(cellID);
    double aspect = yDiff / xDiff;
    while (aspect > 2.0)
    {
      hemkerMeshNoCurves->hRefine(cellIDsToRefine, horizontalCut);

      // the next set of cellIDsToRefine are the children of the ones just refined
      set<GlobalIndexType> childCellIDs;
      for (set<GlobalIndexType>::iterator refinedCellIDIt = cellIDsToRefine.begin();
          refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++)
      {
        int refinedCellID = *refinedCellIDIt;
        set<GlobalIndexType> refinedCellChildren = hemkerMeshNoCurves->getTopology()->getCell(refinedCellID)->getDescendants(hemkerMeshNoCurves->getTopology());
        childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
      }

      cellIDsToRefine = childCellIDs;
      aspect /= 2;
    }
  }
  if (enforceOneIrregularity)
    hemkerMeshNoCurves->enforceOneIrregularity();
}

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

double computeL2Error(NavierStokesVGPFormulation &form, FunctionPtr u_exact, MeshPtr mesh, double Re)
{
  FunctionPtr sigma1_exact = 1./Re*u_exact->x()->grad();
  FunctionPtr sigma2_exact = 1./Re*u_exact->y()->grad();

  // double l2Error = 0;
  double u1_l2 = 0, u2_l2, sigma11_l2 = 0, sigma12_l2 = 0, sigma21_l2 = 0, sigma22_l2 = 0;
  FunctionPtr u1_soln, u2_soln, sigma11_soln, sigma12_soln, sigma21_soln, sigma22_soln,
              u1_diff, u2_diff, sigma11_diff, sigma12_diff, sigma21_diff, sigma22_diff,
              u1_sqr, u2_sqr, sigma11_sqr, sigma12_sqr, sigma21_sqr, sigma22_sqr;
  u1_soln = Function::solution(form.u(1), form.solution());
  u2_soln = Function::solution(form.u(2), form.solution());
  sigma11_soln = Function::solution(form.sigma(1,1), form.solution());
  sigma12_soln = Function::solution(form.sigma(1,2), form.solution());
  sigma21_soln = Function::solution(form.sigma(2,1), form.solution());
  sigma22_soln = Function::solution(form.sigma(2,2), form.solution());
  u1_diff = u1_soln - u_exact->x();
  u2_diff = u2_soln - u_exact->y();
  sigma11_diff = sigma11_soln - sigma1_exact->x();
  sigma12_diff = sigma12_soln - sigma1_exact->y();
  sigma21_diff = sigma21_soln - sigma2_exact->x();
  sigma22_diff = sigma22_soln - sigma2_exact->y();
  u1_sqr = u1_diff*u1_diff;
  u2_sqr = u2_diff*u2_diff;
  sigma11_sqr = sigma11_diff*sigma11_diff;
  sigma12_sqr = sigma12_diff*sigma12_diff;
  sigma21_sqr = sigma21_diff*sigma21_diff;
  sigma22_sqr = sigma22_diff*sigma22_diff;
  u1_l2 = u1_sqr->integrate(mesh, 10);
  u2_l2 = u2_sqr->integrate(mesh, 10);
  sigma11_l2 = sigma11_sqr->integrate(mesh, 10);
  sigma12_l2 = sigma12_sqr->integrate(mesh, 10);
  sigma21_l2 = sigma21_sqr->integrate(mesh, 10);
  sigma22_l2 = sigma22_sqr->integrate(mesh, 10);
  return sqrt(u1_l2+sigma11_l2+sigma12_l2+sigma21_l2+sigma22_l2);
  // l2Error = sqrt(u_l2);
}

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
  double cgTol = 1e-6;
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

  vector<double> pressureConstraintPoint;

  // Construct Mesh
  MeshTopologyPtr meshTopo;
  MeshGeometryPtr meshGeometry = Teuchos::null;
  if (problemName == "Trivial")
  {
    int meshWidth = 2;
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0.0);

    pressureConstraintPoint = {0.5,0.5};

    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  else if (problemName == "LidDriven")
  {
    int meshWidth = 2;
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0.0);
    pressureConstraintPoint = {0.5,0.5};
    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  else if (problemName == "Kovasznay")
  {
    vector<double> x0;
    vector<double> dims;
    vector<int> numElements;
    x0.push_back(-.5);
    x0.push_back(-.5);
    dims.push_back(1.5);
    dims.push_back(2.0);
    numElements.push_back(3);
    numElements.push_back(4);
    pressureConstraintPoint = {0,0};
    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  else if (problemName == "TaylorGreen")
  {
    double pi = atan(1)*4;
    vector<double> x0;
    vector<double> dims;
    vector<int> numElements;
    x0.push_back(0);
    x0.push_back(0);
    dims.push_back(2*pi);
    dims.push_back(2*pi);
    numElements.push_back(2);
    numElements.push_back(2);
    pressureConstraintPoint = x0;
    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    if (!steady)
    {
      double t0 = 0;
      double t1 = 1;
      int temporalDivisions = 2;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }
  else if (problemName == "Cylinder")
  {
    double radius = 0.5;
    double xLeft = -3;
    double xRight = 9;
    double meshHeight = 9;
    double embeddedSideLength = 3 * radius;
    meshGeometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, -meshHeight/2, meshHeight/2, radius, embeddedSideLength);
    meshTopo = Teuchos::rcp(new MeshTopology(meshGeometry));
    vector<double> x0;
    x0.push_back(xLeft);
    x0.push_back(0);
    pressureConstraintPoint = x0;
    if (!steady)
    {
      double t0 = 0;
      double t1 = 4;
      int temporalDivisions = 4;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
    }
  }

  Teuchos::ParameterList nsParameters;
  if (steady)
    nsParameters = NavierStokesVGPFormulation::steadyConservationFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, delta_k).getConstructorParameters();
  else
    nsParameters = NavierStokesVGPFormulation::spaceTimeConservationFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, polyOrder, delta_k).getConstructorParameters();

  nsParameters.set("neglectFluxesOnRHS", false);
  NavierStokesVGPFormulation form(meshTopo, nsParameters);

  form.setIP( norm );

  form.solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
  if (problemName == "Trivial")
    form.addPointPressureCondition(pressureConstraintPoint);
  if (problemName == "LidDriven")
    form.addPointPressureCondition(pressureConstraintPoint);
  if (problemName == "Kovasznay")
    form.addPointPressureCondition(pressureConstraintPoint);
  if (problemName == "TaylorGreen")
    form.addPointPressureCondition(pressureConstraintPoint);

  MeshPtr mesh = form.solutionIncrement()->mesh();
  // if (problemName == "Cylinder")
  //   preprocessHemkerMesh(mesh, steady, 1);
  if (meshGeometry != Teuchos::null)
    mesh->setEdgeToCurveMap(meshGeometry->edgeToCurveMap());

  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
  int numberOfMeshesForMultigrid = meshesCoarseToFine.size();

  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);
  VarPtr tn1_hat = form.tn_hat(1), tn2_hat = form.tn_hat(2);

  FunctionPtr u1_exact = Function::zero();
  FunctionPtr u2_exact = Function::zero();
  FunctionPtr u_exact = Function::vectorize(u1_exact,u2_exact);
  if (problemName == "Trivial")
  {
    BCPtr bc = form.solutionIncrement()->bc();

    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(1);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    u1_exact = Function::constant(1);
    u2_exact = Function::zero();
    u_exact = Function::vectorize(u1_exact,u2_exact);
    form.addInflowCondition(leftX,  u_exact);
    form.addInflowCondition(rightX, u_exact);
    form.addInflowCondition(leftY,  u_exact);
    form.addInflowCondition(rightY, u_exact);

    if (!steady)
      form.addInitialCondition(0, u_exact);
      // form.addFluxCondition(t0, -u_exact);
  }
  else if (problemName == "LidDriven")
  {
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
    SpatialFilterPtr notTopBoundary = SpatialFilter::negatedFilter(topBoundary);
    form.addWallCondition(notTopBoundary);

    double eps = 1.0 / 64.0;
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
  }
  else if (problemName == "Kovasznay")
  {
    BCPtr bc = form.solutionIncrement()->bc();

    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-.5);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(1);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-.5);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(1.5);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    double pi = atan(1)*4;
    double lambda = Re/2-sqrt(Re*Re/4+4*pi*pi);
    FunctionPtr explambdaX = Teuchos::rcp(new Exp_ax(lambda));
    FunctionPtr cos2piY = Teuchos::rcp(new Cos_ay(2*pi));
    FunctionPtr sin2piY = Teuchos::rcp(new Sin_ay(2*pi));
    u1_exact = 1 - explambdaX*cos2piY;
    u2_exact = lambda/(2*pi)*explambdaX*sin2piY;

    u_exact = Function::vectorize(u1_exact,u2_exact);
    form.addInflowCondition(leftX,  u_exact);
    form.addInflowCondition(rightX, u_exact);
    form.addInflowCondition(leftY,  u_exact);
    form.addInflowCondition(rightY, u_exact);

    if (!steady)
      form.addInitialCondition(0, u_exact);
      // form.addFluxCondition(t0, -u_exact);
  }
  else if (problemName == "TaylorGreen")
  {
    BCPtr bc = form.solutionIncrement()->bc();

    double pi = atan(1)*4;
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(0);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(2*pi);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(0);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(2*pi);
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    FunctionPtr temporalDecay = Teuchos::rcp(new Exp_at(-2./Re));
    FunctionPtr sinX = Teuchos::rcp(new Sin_x());
    FunctionPtr cosX = Teuchos::rcp(new Cos_x());
    FunctionPtr sinY = Teuchos::rcp(new Sin_y());
    FunctionPtr cosY = Teuchos::rcp(new Cos_y());
    u1_exact = sinX*cosY*temporalDecay;
    u2_exact = -cosX*sinY*temporalDecay;

    u_exact = Function::vectorize(u1_exact,u2_exact);
    form.addInflowCondition(leftX,  u_exact);
    form.addInflowCondition(rightX, u_exact);
    form.addInflowCondition(leftY,  u_exact);
    form.addInflowCondition(rightY, u_exact);

    if (!steady)
      form.addInitialCondition(0, u_exact);
      // form.addFluxCondition(t0, -u_exact);
  }
  else if (problemName == "Cylinder")
  {
    BCPtr bc = form.solutionIncrement()->bc();

    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1);
    SpatialFilterPtr leftX  = SpatialFilter::matchingX(-3);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(9);
    SpatialFilterPtr leftY  = SpatialFilter::matchingY(-4.5);
    SpatialFilterPtr rightY = SpatialFilter::matchingY(4.5);
    SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(0.5) );
    SpatialFilterPtr t0  = SpatialFilter::matchingT(0);

    double pi = atan(1)*4;
    double meshHeight = 9;
    FunctionPtr decay = Teuchos::rcp(new Exp_at(-10));
    FunctionPtr perturbation = Teuchos::rcp(new Sin_ay(2*pi/meshHeight));
    // FunctionPtr perturbed = Function::constant(1) + 0.01*decay*perturbation;
    // FunctionPtr r = PolarizedFunction::r();
    // FunctionPtr cos_theta = PolarizedFunction::cos_theta;
    // FunctionPtr sin_theta = PolarizedFunction::sin_theta;
    // FunctionPtr ur = (one-0.5*0.5/(r*r))*cos_theta;
    // FunctionPtr utheta = -(one+0.5*0.5/(r*r))*sin_theta;
    FunctionPtr x1 = Function::xn(1);
    FunctionPtr y1 = Function::yn(1);
    FunctionPtr x2 = Function::xn(2);
    FunctionPtr y2 = Function::yn(2);
    double R2 = 0.25;
    if (steady)
    {
      // u1_exact = Function::constant(1);
      u1_exact = (R2*(y2-x2)+(x2+y2)*(x2+y2))/((x2+y2)*(x2+y2));
      u2_exact = -(2*R2*x1*y1)/((x2+y2)*(x2+y2));
    }
    else
    {
      // u1_exact = Function::min(Function::tn(1),Function::constant(1))*perturbed;
      u1_exact = (R2*(y2-x2)+(x2+y2)*(x2+y2))/((x2+y2)*(x2+y2)) + 0.1*decay*perturbation;
      u2_exact = -(2*R2*x1*y1)/((x2+y2)*(x2+y2));
    }
    // u2_exact = zero;

    u_exact = Function::vectorize(u1_exact,u2_exact);
    form.addInflowCondition(leftX,  u_exact);
    form.addWallCondition(nearCylinder);

    // define traction components in terms of field variables
    FunctionPtr n = Function::normal();
    VarPtr sigma11 = form.sigma(1,1);
    VarPtr sigma12 = form.sigma(1,2);
    VarPtr sigma21 = form.sigma(2,1);
    VarPtr sigma22 = form.sigma(2,2);
    VarPtr p = form.p();
    LinearTermPtr t1 = n->x() * (2 * sigma11 - p) + n->y() * (sigma12 + sigma21);
    LinearTermPtr t2 = n->x() * (sigma12 + sigma21) + n->y() * (2 * sigma22 - p);

    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(t1==zero, rightX);
    pc->addConstraint(t2==zero, rightX);
    pc->addConstraint(t1==zero, leftY);
    pc->addConstraint(t2==zero, leftY);
    pc->addConstraint(t1==zero, rightY);
    pc->addConstraint(t2==zero, rightY);

    form.solutionIncrement()->setFilter(pc);

    if (!steady)
      form.addInitialCondition(0, u_exact);
  }

  map<int, FunctionPtr> initialGuess;
  initialGuess[form.u(1)->ID()] = u1_exact;
  initialGuess[form.u(2)->ID()] = u2_exact;
  form.solution()->projectOntoMesh(initialGuess);

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
  exportName << problemName << "_Re" << Re << "_" << norm << "_k" << polyOrder;// << "_" << solverChoice;// << "_" << multigridStrategyString;
  if (tag != "")
    exportName << "_" << tag;

  string dataFileLocation;
  dataFileLocation = outputDir+"/"+exportName.str()+".txt";
  ofstream dataFile(dataFileLocation);
  // if (rank==0) dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "elapsed\t" << "iterations\t " << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  if (exportHDF5)
  {
    exporter = Teuchos::rcp(new HDF5Exporter(mesh, exportName.str(), outputDir));
    exporter->exportSolution(form.solution(), 0);
  }
  // HDF5Exporter exporter(form.solution()->mesh(), exportName.str(), outputDir);

  // exportName << "_energyError";
  // HDF5Exporter energyErrorExporter(form.solution()->mesh(), exportName.str(), outputDir);

  // energyErrorExporter.exportFunction(energyErrorFunction, "energy error", 0);

  double energyError = form.solutionIncrement()->energyErrorTotal();
  double l2Error = 0;
  if (computeL2)
    l2Error = computeL2Error(form, u_exact, mesh, Re);
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
  do
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
    if (computeL2)
      l2Error = computeL2Error(form, u_exact, mesh, Re);

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
      exporter->exportSolution(form.solution(), refNumber);
    // energyErrorExporter.exportFunction(energyErrorFunction, "energy error", refNumber);

  }
  while ((energyError > tol) && (refNumber < numRefs));

  if (rank==0) dataFile.close();

  return 0;
}
