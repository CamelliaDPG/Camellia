//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "ErrorIndicator.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "OldroydBFormulationUW.h"
#include "H1ProjectionFormulation.h"
// #include "StokesVGPFormulation.h"
// #include "NavierStokesVGPFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PreviousSolutionFunction.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"


#include "CamelliaDebugUtility.h"


using namespace Camellia;

class TopLidBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
};

// class LeftHemkerBoundary : public SpatialFilter
// {
// public:
//   bool matchesPoint(double x, double y)
//   {
//     double tol = 1e-14;
//     return (abs(x-0.0) < tol);
//   }
// };

// class RightHemkerBoundary : public SpatialFilter
// {
// public:
//   bool matchesPoint(double x, double y)
//   {
//     double tol = 1e-14;
//     return (abs(x-6.0) < tol);
//   }
// };

// class LeftRightHemkerBoundary : public SpatialFilter
// {
// public:
//   bool matchesPoint(double x, double y)
//   {
//     double tol = 1e-14;
//     return ((abs(x-0.0) < tol) || (abs(x-6.0) < tol));
//   }
// };

class CylinderBoundary : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundary(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 5e-1; // be generous b/c dealing with parametric curve
    if ( abs(x) > _radius + 1e-12 )
    // added this exception for the half-space geometries
    {
      return false;
    }
    else {
      return ( sqrt(x*x+y*y) < _radius+tol );
    }
  }
};

// class HalfCylinderBoundary : public SpatialFilter
// {
//   double _radius;
// public:
//   HalfCylinderBoundary(double radius)
//   {
//     _radius = radius;
//   }
//   // CylinderBoundary(double radius) : _radius(radius) {}
//   bool matchesPoint(double x, double y)
//   {
//     double tol = 5e-1; // be generous b/c dealing with parametric curve
//     if ( abs(x) > _radius + 1e-12 ) {
//       return false;
//     }
//     else {
//       return ( sqrt(x*x+y*y) < _radius+tol );
//     }
//   }
// };

class CylinderBoundaryGenerous : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundaryGenerous(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-1; // be generous b/c dealing with parametric curve
    return (sqrt(x*x+y*y) < _radius+tol);
  }
};

class CylinderBoundaryExtreme : public SpatialFilter
{
  double _radius;
public:
  CylinderBoundaryExtreme(double radius)
  {
    _radius = radius;
  }
  // CylinderBoundary(double radius) : _radius(radius) {}
  bool matchesPoint(double x, double y)
  {
    double tol = 5e-10; // be generous b/c dealing with parametric curve
    return (sqrt(x*x+y*y) < _radius+tol);
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction<double> {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    if ( (abs(x) < _eps) ) {   // top left
      return x / _eps;
    }
    else if ( abs(1.0-x) < _eps) {     // top right
      return (1.0-x) / _eps;
    }
    else {     // top middle
      return 1;
    }
  }
};

class ParabolicInflowFunction_U1 : public SimpleFunction<double> {
  double _height; // height of channel
public:
  ParabolicInflowFunction_U1(double height) {
    _height = height;
  }
  double value(double x, double y) {
    return 3.0/2.0*(1.0-pow(y/_height,2)); // chosen so that AVERAGE inflow velocity is 1.0
    // return (1.0-pow(y/_height,2));
    // return 3.0/8.0*(4.0-pow(y,2));
    // return (16.0*y-pow(y,2))/64.0;
  }
};

class ParabolicInflowFunction_Tun : public SimpleFunction<double> {
  double _i, _j; // index numbers
  double _height; // height of channel
  double _muP; // polymeric viscosity
  double _lambda; // relaxation time
public:
  ParabolicInflowFunction_Tun(double height, double muP, double lambda, int i, int j) {
    _i = i;
    _j = j;
    _height = height;
    _muP = muP;
    _lambda = lambda;
  }
  double value(double x, double y) {
    if (_i == 1 && _j == 1)
      // return -(1.0-pow(y/_height,2))*(8.0*_muP*_lambda*pow(y/(_height*_height),2));
      return -3.0/2.0*(1.0-pow(y/_height,2))*(18.0*_muP*_lambda*pow(y/(_height*_height),2));
    else if ((_i == 1 && _j == 2) || (_i == 2 && _j == 1))
      // return -(1.0-pow(y/_height,2))*(-2.0*_muP*y/(_height*_height));
      return -3.0/2.0*(1.0-pow(y/_height,2))*(-3.0*_muP*y/(_height*_height));
    else if (_i == 2 && _j == 2)
      return 0.0;
    else
      cout << "ERROR: Indices not currently supported\n";
    return Teuchos::null;
  }
};

int sgn(double val) {
  if (val > 0) return  1;
  if (val < 0) return -1;
  return 0;
}

template <typename Scalar>
class DragOrientedErrorIndicator : public ErrorIndicator
{
  FunctionPtr _d_eta;
public:
  DragOrientedErrorIndicator(MeshPtr mesh, FunctionPtr d_eta) : ErrorIndicator(mesh)
  {
    _d_eta = d_eta;
  }

  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {
    _localErrorMeasures.clear();
    const set<GlobalIndexType>* myCells = &_mesh->cellIDsInPartition();
    for (auto cellID : *myCells)
    {
      _localErrorMeasures[cellID] = abs(_d_eta->integrate(cellID, _mesh));
    }
  }
};

template <typename Scalar>
class BoundaryOrientedErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  SpatialFilterPtr _spatialFilter;
public:
  BoundaryOrientedErrorIndicator(SolutionPtr soln, SpatialFilterPtr spatialFilter) : ErrorIndicator(soln->mesh())
  {
    _solution = soln;
    _spatialFilter = spatialFilter;
  }

  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {

    const map<GlobalIndexType, double>* rankLocalEnergyError = &_solution->rankLocalEnergyError();
    // square roots have already been taken
    bool energyErrorIsSquared = false;

    _localErrorMeasures.clear();
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      double error = energyErrorIsSquared ? sqrt(entry.second) : entry.second;
      _localErrorMeasures[cellID] = error;
    }

    // calculate max
    double localMax;
    double globalMax;
    for (auto measureEntry : _localErrorMeasures) {
      double cellError = measureEntry.second;
      localMax = max(localMax,cellError);
    }
    _mesh->Comm()->MaxAll(&localMax, &globalMax, 1);

    // add loop through elements to refine elements on the cylinder boundary
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      vector< vector<double> > verticesOfElement = _mesh->verticesForCell(cellID);
      for (int vertexIndex=0; vertexIndex<verticesOfElement.size(); vertexIndex++)
      {
        if (_spatialFilter->matchesPoint(verticesOfElement[vertexIndex][0], verticesOfElement[vertexIndex][1]))
        {
          _localErrorMeasures[cellID] = globalMax;
        }
      }
    }

  }
};


int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  //////////////////////////////////////////////////////////////////////
  ///////////////////////  COMMAND LINE PARAMETERS  ////////////////////
  //////////////////////////////////////////////////////////////////////
  string formulation = "OldroydB";
  // string formulation = "NavierStokes";
  string problemChoice = "LidDriven";
  double rho = 1;
  double lambda = 1;
  double muS = 1; // solvent viscosity
  double muP = 1; // polymeric viscosity
  double alpha = 0;
  // int spaceDim = 2;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  int numXElems = 2;
  int numYElems = 2;
  bool stokesOnly = false;
  bool enforceLocalConservation = false;
  bool useConformingTraces = true;
  string solverChoice = "KLU";
  string multigridStrategyString = "V-cycle";
  bool useCondensedSolve = false;
  bool useConjugateGradient = true;
  bool logFineOperator = false;
  double solverTolerance = 1e-8;
  int maxNonlinearIterations = 25;
  int enrichDegree = 0;
  double nonlinearTolerance = 1e-8;
  // double minNonlinearTolerance = 10*solverTolerance;
  // double minNonlinearTolerance = 5e-7;
  double minNonlinearTolerance = 1e-4;
  // double minNonlinearTolerance = 4e-5;
  int maxLinearIterations = 1000;
  // bool computeL2Error = false;
  bool exportSolution = false;
  bool useLineSearch = false;
  string norm = "Graph";
  string errorIndicator = "Energy";
  string outputDir = ".";
  string tag="";
  cmdp.setOption("formulation", &formulation, "OldroydB, NavierStokes");
  cmdp.setOption("problem", &problemChoice, "LidDriven, HemkerCylinder, HalfHemker, Benchmark");
  cmdp.setOption("rho", &rho, "rho");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("muS", &muS, "muS");
  cmdp.setOption("muP", &muP, "muP");
  cmdp.setOption("alpha", &alpha, "alpha");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("numYElems",&numYElems,"number of elements in y direction");
  cmdp.setOption("stokesOnly", "NavierStokesOnly", &stokesOnly, "couple only with Stokes, not Navier-Stokes");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy,CylinderBoundary,DragOriented");
  cmdp.setOption("enforceLocalConservation", "noLocalConservation", &enforceLocalConservation, "enforce local conservation principles at the element level");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("enrichCubature", &enrichDegree, "enrichment of quadrature");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  // cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("useLineSearch", "dontUseLineSearch", &useLineSearch, "compute increment with line search");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);

  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr n    = Function::normal();

  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////

  ///////////////////////  SET PROBLEM PARAMETERS  /////////////////////
  Teuchos::ParameterList parameters;
  parameters.set("spaceDim", 2);
  parameters.set("spatialPolyOrder", k);
  parameters.set("delta_k", delta_k);
  parameters.set("norm", norm);
  parameters.set("enforceLocalConservation",enforceLocalConservation);
  parameters.set("useConformingTraces", useConformingTraces);
  parameters.set("stokesOnly",stokesOnly);
  parameters.set("useConservationFormulation",false);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("rho", rho);
  parameters.set("lambda", lambda);
  parameters.set("muS", muS);
  parameters.set("muP", muP);
  parameters.set("alpha", alpha);


  //////////////////////  DECLARE EXACT SOLUTION  //////////////////////
  // FunctionPtr u_exact = Function::constant(1) - 2*Function::xn(1);


  ///////////////////////////  DECLARE MESH  ///////////////////////////

  MeshGeometryPtr spatialMeshGeom;
  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;


  double xLeft, xRight, height, yMax, cylinderRadius;
  if (problemChoice == "LidDriven")
  {
    // LID-DRIVEN CAVITY FLOW
    double x0 = 0.0, y0 = 0.0;
    double width = 1.0;
    height = 1.0;
    int horizontalCells = 2, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else if (problemChoice == "HemkerCylinder")
  {
    // FLOW PAST A CYLINDER
    xLeft = -10.0, xRight = 20.0;
    height = 16.0;
    cylinderRadius = 1.0;
    MeshGeometryPtr HemkerGeometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, height, cylinderRadius);
    map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = HemkerGeometry->edgeToCurveMap();
    globalEdgeToCurveMap = map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr >(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
    spatialMeshTopo = Teuchos::rcp( new MeshTopology(HemkerGeometry) );
    spatialMeshTopo->setEdgeToCurveMap(globalEdgeToCurveMap, Teuchos::null);
  }
  else if (problemChoice == "HalfHemker")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    xLeft = -15.0, xRight = 15.0;
    cylinderRadius = 1.0;
    yMax = 2.0*cylinderRadius;
    // MeshGeometryPtr spatialMeshGeom = MeshFactory::halfHemkerGeometry(xLeft,xRight,yMax,cylinderRadius);
    MeshGeometryPtr spatialMeshGeom = MeshFactory::halfConfinedCylinderGeometry(cylinderRadius);
    map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = spatialMeshGeom->edgeToCurveMap();
    globalEdgeToCurveMap = map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr >(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
    spatialMeshTopo = Teuchos::rcp( new MeshTopology(spatialMeshGeom) );
    spatialMeshTopo->setEdgeToCurveMap(globalEdgeToCurveMap, Teuchos::null);
  }
  else if (problemChoice == "Benchmark")
  {
    // CONFINED CYLINDER BENCHMARK PROBLEM
    xLeft = -15.0, xRight = 15.0;
    cylinderRadius = 1.0;
    yMax = 2.0*cylinderRadius;
    spatialMeshGeom = MeshFactory::confinedCylinderGeometry(cylinderRadius);
    map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = spatialMeshGeom->edgeToCurveMap();
    globalEdgeToCurveMap = map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr >(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
    spatialMeshTopo = Teuchos::rcp( new MeshTopology(spatialMeshGeom) );
    spatialMeshTopo->setEdgeToCurveMap(globalEdgeToCurveMap, Teuchos::null);
  }
  else if (problemChoice == "Test 1")
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  ///////////
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);
  OldroydBFormulationUW form(spatialMeshTopo, parameters);
  ///////////
  MeshPtr mesh = form.solutionIncrement()->mesh();
  if (globalEdgeToCurveMap.size() > 0)
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }

  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr solutionIncrement = form.solutionIncrement();
  SolutionPtr solutionBackground = form.solution();

  if (enrichDegree > 0)
  {
    // THIS DOESN'T APPEAR TO WORK
    if (commRank == 0)
      cout << "enriching cubature by " << enrichDegree << endl;
    solutionIncrement->setCubatureEnrichmentDegree(enrichDegree);
    solutionBackground->setCubatureEnrichmentDegree(enrichDegree);
  }

  ///////////////////////////  DECLARE BC'S  ///////////////////////////
  BCPtr bc = form.solutionIncrement()->bc();
  VarPtr u1hat, u2hat, p;
  u1hat = form.u_hat(1);
  u2hat = form.u_hat(2);
  p     = form.p();

  if (problemChoice == "LidDriven")
  {
    // LID-DRIVEN CAVITY FLOW
    SpatialFilterPtr topBoundary = Teuchos::rcp( new TopLidBoundary );
    SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);

    //   top boundary:
    FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(1.0/64) );
    bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
    bc->addDirichlet(u2hat, topBoundary, zero);

    //   everywhere else:
    bc->addDirichlet(u1hat, otherBoundary, zero);
    bc->addDirichlet(u2hat, otherBoundary, zero);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else if (problemChoice == "HemkerCylinder")
  {
    // FLOW PAST A CYLINDER
    // SpatialFilterPtr leftBoundary = Teuchos::rcp( new LeftHemkerBoundary );
    SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
    SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(height/2);
    SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(-height/2);
    SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));
    // SpatialFilterPtr rightBoundary = Teuchos::rcp( new RightHemkerBoundary );
    // SpatialFilterPtr leftRightBoundary = Teuchos::rcp( new LeftRightHemkerBoundary );
    // SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(leftRightBoundary);

    // inflow on left boundary
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(height/2.0) );
    // TFunctionPtr<double> u1_inflowFunction = one;
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height/2.0, muP, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height/2.0, muP, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height/2.0, muP, lambda, 2, 2) );

    TFunctionPtr<double> u = Function::vectorize(u1_inflowFunction,u2_inflowFunction);

    form.addInflowCondition(leftBoundary, u);
    form.addInflowViscoelasticStress(leftBoundary, T11un_inflowFunction, T12un_inflowFunction, T22un_inflowFunction);

    // top+bottom
    // form.addOutflowCondition(topBoundary, false);
    // form.addOutflowCondition(bottomBoundary, false);
    form.addWallCondition(topBoundary);
    form.addWallCondition(bottomBoundary);


    // outflow on right boundary
    // form.addOutflowCondition(rightBoundary, true); // true to impose zero traction by penalty (TODO)
    form.addOutflowCondition(rightBoundary, height/2.0, muP, lambda, false); // false for zero flux variable

    // no slip on cylinder
    form.addWallCondition(cylinderBoundary);

    // cout << "ERROR: Problem type not currently supported. Returning null.\n";
    // return Teuchos::null;
  }
    else if (problemChoice == "HalfHemker")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
    SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(yMax);
    SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(0.0);
    SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));

    // inflow on left boundary
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(yMax) );
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 2, 2) );

    TFunctionPtr<double> u = Function::vectorize(u1_inflowFunction,u2_inflowFunction);

    form.addInflowCondition(leftBoundary, u);
    form.addInflowViscoelasticStress(leftBoundary, T11un_inflowFunction, T12un_inflowFunction, T22un_inflowFunction);

    // top+bottom
    form.addWallCondition(topBoundary);
    form.addSymmetryCondition(bottomBoundary);

    // outflow on right boundary
    // form.addOutflowCondition(rightBoundary, true); // true to impose zero traction by penalty (TODO)
    // form.addOutflowCondition(rightBoundary, yMax, muP, lambda, false); // false for zero flux variable
    form.addInflowCondition(rightBoundary, u);

    // no slip on cylinder
    // do this before symmetry condition in case of overlap of spatial filter
    form.addWallCondition(cylinderBoundary);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else if (problemChoice == "Benchmark")
  {
    // CONFINED CYLINDER exploiting geometric symmetry
    SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
    SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
    SpatialFilterPtr topBoundary = SpatialFilter::matchingY(yMax);
    SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(-yMax);
    SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));


    // UPDATE THIS FOR WHEN LAMBDA CHANGES

    // inflow on left boundary
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(yMax) );
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(yMax, muP, lambda, 2, 2) );

    TFunctionPtr<double> u = Function::vectorize(u1_inflowFunction,u2_inflowFunction);

    form.addInflowCondition(leftBoundary, u);
    form.addInflowViscoelasticStress(leftBoundary, T11un_inflowFunction, T12un_inflowFunction, T22un_inflowFunction);

    // top+bottom
    form.addWallCondition(topBoundary);
    form.addWallCondition(bottomBoundary);

    // outflow on right boundary
    // form.addOutflowCondition(rightBoundary, true); // true to impose zero traction by penalty (TODO)
    // form.addOutflowCondition(rightBoundary, yMax, muP, lambda, false); // false for zero flux variable
    form.addInflowCondition(rightBoundary, u);

    // no slip on cylinder
    form.addWallCondition(cylinderBoundary);

    //   zero-mean constraint
    bc->addZeroMeanConstraint(p);
  }
  else if (problemChoice == "Test 1")
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  //////////////////////////////////////////////////////////////////////
  ///////////////////////////////  SOLVE  //////////////////////////////
  //////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Teuchos::Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");
  // RefinementStrategyPtr refStrategy = form.getRefinementStrategy();

  if (commRank == 0)
    Solver::printAvailableSolversReport();
  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#ifdef HAVE_AMESOS_MUMPS
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
  int azOutput = 50; // print residual every 20 CG iterations

  GMGOperator::MultigridStrategy multigridStrategy;
  if (multigridStrategyString == "Two-level")
  {
    multigridStrategy = GMGOperator::TWO_LEVEL;
  }
  else if (multigridStrategyString == "W-cycle")
  {
    multigridStrategy = GMGOperator::W_CYCLE;
  }
  else if (multigridStrategyString == "V-cycle")
  {
    multigridStrategy = GMGOperator::V_CYCLE;
  }
  else if (multigridStrategyString == "Full-V")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_V;
  }
  else if (multigridStrategyString == "Full-W")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_W;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized multigrid strategy");
  }


  // choose local normal matrix calculation algorithm
  BFPtr bf = form.bf();
  // bf->setOptimalTestSolver(TBF<>::CHOLESKY);
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);

  double errorTol = 0.001;
  double errorRef = 0.0;
  double lambdaInitial = lambda;
  double lambdaMax = lambdaInitial;
  double delta_lambda = 0.1;


  ostringstream solnName;
  solnName << "OldroydB" << "_" << norm << "_k" << k << "_" << solverChoice;
  if (solverChoice[0] == 'G')
    solnName << "_" << multigridStrategyString;
  if (tag != "")
    solnName << "_" << tag;
  if (stokesOnly)
    solnName << "_Stokes";
  else
    solnName << "_NavierStokes";
  solnName << "_lambda_" << lambdaInitial;

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";

  ofstream dataFile(dataFileLocation);
  dataFile << "lambda\t "
           << "ref\t "
           << "elements\t "
           << "dofs\t "
           << "energy\t "
           << "solvetime\t "
           << "elapsed\t "
           << "iterations\t "
           << "drag coefficient (field)\t "
           << "drag coefficient (traction)\t "
           << "y-direction force\t "
           << "drag error estimate\t "
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));


  // string dataFileLocation;
  // if (exportSolution)
  // {
  //   // append initial parameter value to solution name (for storing all data)
  //   ostringstream dataSolnName;
  //   dataSolnName << solnName.str() << "_lambda" << lambdaInitial;
  //   dataFileLocation = outputDir+"/"+dataSolnName.str()+"/"+solnName.str()+".txt";
  // }
  // else
  //   dataFileLocation = outputDir+"/"+solnName.str()+".txt";
  // ofstream dataFile(dataFileLocation);
  // dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "elapsed\t" << "iterations\t " << endl;

  while (lambda <= lambdaMax)
  {
    for (int refIndex=0; refIndex <= numRefs; refIndex++)
    {
      // Teuchos::RCP<HDF5Exporter> exporter;
      // if (exportSolution)
      // {
      //   // append parameter value to solution name
      //   ostringstream fullSolnName;
      //   fullSolnName << solnName.str() << "_lambda" << lambda;
      //   exporter = Teuchos::rcp(new HDF5Exporter(mesh,fullSolnName.str(), outputDir));
      // }

      Teuchos::RCP<GMGSolver> gmgSolver;
      if (solverChoice[0] == 'G')
      {
        bool reuseFactorization = true;
        SolverPtr coarseSolver = Solver::getDirectSolver(reuseFactorization);
        int kCoarse = 1;
        vector<MeshPtr> meshSequence = GMGSolver::meshesForMultigrid(mesh, kCoarse, delta_k);
        // for (int i=0; i < meshSequence.size(); i++)
        // {
        //   if (commRank == 0)
        //     cout << meshSequence[i]->numGlobalDofs() << endl;
        // }
        while (meshSequence[0]->numGlobalDofs() < 100000 && meshSequence.size() > 2)
          meshSequence.erase(meshSequence.begin());
        gmgSolver = Teuchos::rcp(new GMGSolver(solutionIncrement, meshSequence, maxLinearIterations, solverTolerance, multigridStrategy, coarseSolver, useCondensedSolve));
        gmgSolver->setUseConjugateGradient(useConjugateGradient);
        int azOutput = 20; // print residual every 20 CG iterations
        gmgSolver->setAztecOutput(azOutput);
        gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");

        if (solverChoice == "GMG-Direct")
          gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::Direct);
        if (solverChoice == "GMG-ILU")
          gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::ILU);
        if (solverChoice == "GMG-IC")
          gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::IC);
        form.setSolver(gmgSolver);
      }
      else if (solverChoice != "SuperLUDist" && refIndex == 0 && lambda == lambdaInitial)
        form.setSolver(solvers[solverChoice]);


      solverTime->start(true);

      ////////////////////////////////////////////////////////////////////
      //    Solve and accumulate solution
      ////////////////////////////////////////////////////////////////////
      // double s = 1.0;
      double s;
      int iterCount = 0;
      int iterationCount = 0;
      double l2Update = 1e10;
      double l2UpdateInitial = l2Update;
      while (l2Update > nonlinearTolerance*l2UpdateInitial && iterCount < maxNonlinearIterations && l2Update > minNonlinearTolerance)
      {
        if (solverChoice[0] == 'G')
        {
          // solutionIncrement->solve(gmgSolver);
          // form.solveAndAccumulate(s);
          form.solveForIncrement();
          iterationCount += gmgSolver->iterationCount();
        }
        else
          // form.solveAndAccumulate(s);
          form.solveForIncrement();
          // solutionIncrement->condensedSolve(solvers[solverChoice]);

        // ////////////////////////////////////////////////////////////////////
        // //    line search to minimize residual in search direction
        // ////////////////////////////////////////////////////////////////////
        // int sMax = 2;
        // int lineSearchMaxIt = 5;
        // double LStol = 0.5; // as taken from Matthies + Strang
        // //
        // double s0 = 0.0;
        // double s1 = 1.0;
        // // G_i = R(u_i) . delta_u
        // //     = R(u_0 + s_i * delta_u) . delta_u
        // double G_init = form.computeG(0);
        // double G0 = G_init;
        // double G1 = form.computeG(1);

        // // if (commRank == 0)
        // //   cout << "G0 =\t " << G0 << " \t\tG1 =\t " << G1 << endl;
        
        // // find interval about which G changes sign
        // while (sgn(G0)*sgn(G1) > 0 && s1 < sMax)
        // {
        //     s0 = s1;
        //     s1 = 2*s1;
        //     G0 = G1;
            
        //     // compute G1 =  R(u_0+s1*delta_u) . delta
        //     G1 = form.computeG(s1);
        // }

        // // find zero of G using this cool Illinois algorithm
        // s = s1;
        // double G = G1;
        // int i=0;
        // while (i <= lineSearchMaxIt && sgn(G1)*sgn(G0) < 0 && ( abs(G) > LStol*abs(G_init) || abs(s0-s1) > LStol*0.5*(s0+s1) ) )
        // {
        //     ++i;
            
        //     s = s1-G1*(s1-s0)/(G1-G0);

        //     // compute G1 =  R(u_0+s*delta_u) . delta
        //     G1 = form.computeG(s);

        //     if ((sgn(G)*sgn(G1)) > 0)
        //     {
        //         G0 = 0.5*G0;
        //     }
        //     else
        //     {
        //         s0 = s1;
        //         G0 = G1;
        //     }
        //     s1 = s;
        //     G1 = G;
        // }

        ////////////////////////////////////////////////////////////////////
        //    alternative line search to test Sylvester's criteria
        ////////////////////////////////////////////////////////////////////
        s=1;
        if (useLineSearch)
        {
          int posEnrich = 5; // amount of enriching of grid points on which to ensure positivity
          double lineSearchFactor = .75;
          double eps = .001; // arbitrary
          TFunctionPtr<double> conf11 = Function::constant(1.0) + (lambda/muP)*Function::solution(form.T(1,1),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(1,1),solutionIncrement);
          TFunctionPtr<double> conf22 = Function::constant(1.0) + (lambda/muP)*Function::solution(form.T(2,2),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(2,2),solutionIncrement);
          TFunctionPtr<double> conf12 = (lambda/muP)*Function::solution(form.T(1,2),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(1,2),solutionIncrement);
          TFunctionPtr<double> detConf = conf11*conf22 - conf12*conf12;
          bool conf11Positive = (conf11 - Function::constant(eps))->isPositive(mesh,posEnrich);
          bool detConfPositive = (detConf - Function::constant(eps))->isPositive(mesh,posEnrich);
          int iter = 0;
          int maxIter = 3;
          while (!(conf11Positive && detConfPositive) && iter < maxIter)
          {
            s = s*lineSearchFactor;
            conf11 = Function::constant(1.0) + (lambda/muP)*Function::solution(form.T(1,1),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(1,1),solutionIncrement);
            conf22 = Function::constant(1.0) + (lambda/muP)*Function::solution(form.T(2,2),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(2,2),solutionIncrement);
            conf12 = (lambda/muP)*Function::solution(form.T(1,2),solutionBackground) + (lambda/muP)*s*Function::solution(form.T(1,2),solutionIncrement);

            detConf = conf11*conf22 - conf12*conf12;
            conf11Positive = conf11->isPositive(mesh,posEnrich);
            detConfPositive = detConf->isPositive(mesh,posEnrich);
            iter++;
          }
        }

        // Accumulate solution
        form.accumulate(s);

        ////////////////////////////////////////////////////////////////////
        // Compute L2 norm of update and increment counters
        ////////////////////////////////////////////////////////////////////
        l2Update = form.L2NormSolutionIncrement();

        if (commRank == 0)
          cout << "Nonlinear Update:\t " << l2Update << " \t\tLine search distance:\t " << s << endl;

        if (iterCount == 0)
          l2UpdateInitial = l2Update;

        // if (l2Update < 1e-12)
        //   break;

        iterCount++;
      }

      double solveTime = solverTime->stop();

      double energyError = solutionIncrement->energyErrorTotal();
      // double l2Error = 0;
      // if (computeL2Error)
      // {
      //   FunctionPtr u_soln;
      //   u_soln = Function::solution(form.u(), solutionBackground);
      //   FunctionPtr u_diff = u_soln - u_exact;
      //   FunctionPtr u_sqr = u_diff*u_diff;
      //   double u_l2;
      //   u_l2 = u_sqr->integrate(mesh, 10);
      //   l2Error = sqrt(u_l2);
      // }
      // compute drag coefficient if Hemker problem
      double fieldDragCoefficient = 0.0;
      double fluxDragCoefficient = 0.0;
      double verticalForce = 0.0;
      double dragError = 0.0;
      if (problemChoice == "HemkerCylinder" || problemChoice == "HalfHemker" || problemChoice == "Benchmark")
      {
        SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius) );
        // SpatialFilterPtr cylinderBoundary;
        // if ( problemChoice == "HalfHemker" )
        // {
        //   cylinderBoundary = Teuchos::rcp( new HalfCylinderBoundary(cylinderRadius));
        // }
        // else
        // {
        //   cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));
        // }

        TFunctionPtr<double> boundaryRestriction = Function::meshBoundaryCharacteristic();

        // traction computed from field variables
        TFunctionPtr<double> n = TFunction<double>::normal();
        // L = muS*du/dx
        LinearTermPtr f_lt = - form.p()*n->x() + 2.0*form.L(1,1)*n->x() + form.T(1,1)*n->x()
                             + form.L(1,2)*n->y() + form.L(2,1)*n->y() + form.T(1,2)*n->y();
        TFunctionPtr<double> fieldTraction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, -f_lt ) );
        fieldTraction_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( fieldTraction_x*boundaryRestriction, cylinderBoundary) );
        fieldDragCoefficient = fieldTraction_x->integrate(mesh);

        // traction computed from flux
        TFunctionPtr<double> fluxTraction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(1)) );
        fluxTraction_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( fluxTraction_x*boundaryRestriction, cylinderBoundary) );
        fluxDragCoefficient = fluxTraction_x->integrate(mesh);

        // TFunctionPtr<double> dF_D = Teuchos::rcp( new SpatiallyFilteredFunction<double>( Function::constant(1.0)*boundaryRestriction,cylinderBoundary));

        // dragCoefficient = F_D;
        // 2/3 is the average inflow velocity
        // dragCoefficient = F_D;// * (3.0/2.0);

        // compute force in y-direction (from flux)
        TFunctionPtr<double> traction_y = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(2)) );
        traction_y = Teuchos::rcp( new SpatiallyFilteredFunction<double>( traction_y*boundaryRestriction, cylinderBoundary) );
        verticalForce = traction_y->integrate(mesh);

        // estimate error in drag coefficient
        // // drag error ~ (e_1,psi_{v})_{1/2} = (1, psi_{v1})_{1/2}
        // TRieszRepPtr<double> rieszResidual = form.rieszResidual(Teuchos::null);
        // rieszResidual->computeRieszRep();
        // TFunctionPtr<double> psi_v1 =  Teuchos::rcp( new RepFunction<double>(form.v(1), rieszResidual) );
        // dF_D = Teuchos::rcp( new SpatiallyFilteredFunction<double>( psi_v1*boundaryRestriction, cylinderBoundary) );
        // dragError = dF_D->integrate(mesh);
        // // dragError = (dF_D*dF_D)->integrate(mesh);
        // dragError = abs(dragError);
        // // dragError = sqrt(dragError);

        dragError = ((fieldTraction_x-fluxTraction_x)*(fieldTraction_x-fluxTraction_x))->integrate(mesh);
        dragError = sqrt(2.0*M_PI)*sqrt(dragError);

        if (problemChoice == "HalfHemker")
        {
          fieldDragCoefficient = 2.0*fieldDragCoefficient;
          fluxDragCoefficient  = 2.0*fluxDragCoefficient;
          dragError = 2.0*dragError;
        }
        // else if (problemChoice == "Benchmark")
        // {
        //   2/3 is the average inflow velocity
        //   fieldDragCoefficient = 3.0/2.0*fieldDragCoefficient;
        //   fluxDragCoefficient = 3.0/2.0*fluxDragCoefficient;
        //   dragError = 3.0/2.0*dragError;
        // }
      }

      if (commRank == 0)
      {
        cout << setprecision(8) 
          << "Lambda: " << lambda
          << " \nRefinement: " << refIndex
          << " \tElements: " << mesh->numActiveElements()
          << " \tDOFs: " << mesh->numGlobalDofs()
          << " \tEnergy Error: " << energyError
          // << " \tL2 Error: " << l2Error
          << " \nSolve Time: " << solveTime
          << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
          << " \tIteration Count: " << iterationCount
          << " \nDrag Coefficient (from field): " << fieldDragCoefficient
          << " \tDrag Coefficient (from traction): " << fluxDragCoefficient
          << " \ty-direction Force : " << verticalForce
          << " \tDrag error estimate : " << dragError
          // << " \nLine search distance: " << s
          << endl;
        dataFile << setprecision(8)
          << lambda
          << " " << refIndex
          << " " << mesh->numActiveElements()
          << " " << mesh->numGlobalDofs()
          << " " << energyError
          // << " " << l2Error
          << " " << solveTime
          << " " << totalTimer->totalElapsedTime(true)
          << " " << iterationCount
          << " " << fieldDragCoefficient
          << " " << fluxDragCoefficient
          << " " << verticalForce
          << " " << dragError
          // << " " << s
          << endl;
      }

      // if (refIndex == 0 && lambda == lambdaInitial)
      if (dragError > errorRef)
        errorRef = dragError;

      if (exportSolution)
      {
        exporter->exportSolution(solutionBackground, refIndex);
        // exporter->exportSolution(solutionIncrement, refIndex);
        
        // output mesh with GnuPlotUtil
        ostringstream meshExportName;
        meshExportName << outputDir << "/" << solnName.str() << "/" << "ref" << refIndex << "_mesh";
        // int numPointsPerEdge = 3;
        bool labelCells = false;
        string meshColor = "black";
        GnuPlotUtil::writeComputationalMeshSkeleton(meshExportName.str(), mesh, labelCells, meshColor);
      }

      // if ((energyError < errorRef*errorTol && iterCount < maxNonlinearIterations-1) || energyError < 1e-8 )
      // if (energyError < errorRef*errorTol || energyError < 1e-8 )
      // if (dragError < errorRef*errorTol || energyError < 1e-8 )
      //   break;

      if (refIndex != numRefs)
      {
        ///////////////////  CHOOSE REFINEMENT STRATEGY  ////////////////////
        // Can also set whether to use h or p adaptivity here
        if (errorIndicator == "Energy")
        {
          form.refine();
        }
        else if (errorIndicator == "CylinderBoundary")
        {
          if (problemChoice == "Benchmark" || problemChoice == "HalfHemker")
          {
            // SpatialFilterPtr cylinderBoundaryGenerous = Teuchos::rcp( new CylinderBoundaryGenerous(cylinderRadius));
            SpatialFilterPtr cylinderBoundaryExtreme = Teuchos::rcp( new CylinderBoundaryExtreme(cylinderRadius));
            double energyThreshold = 0.2;
            // ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryGenerous) );
            ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryExtreme) );
            RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, energyThreshold) );
            refStrategy->refine();
          }
          else
          {
            cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
            return Teuchos::null;
          }
        }
        else if (errorIndicator == "DragOriented")
        {

          // define spatial filters for the H1-projection problems
          SpatialFilterPtr cylinderBoundary, topBoundary, bottomBoundary, leftBoundary, 
  rightBoundary;
          if (problemChoice == "Benchmark" || problemChoice == "HalfHemker")
          {
            cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));
            topBoundary = SpatialFilter::matchingY(yMax);
            if (problemChoice == "Benchmark")
            {
              bottomBoundary = SpatialFilter::matchingY(-yMax);
            }
            else
            {
              bottomBoundary = SpatialFilter::matchingY(0.0);
            }
            leftBoundary = SpatialFilter::matchingX(xLeft);
            rightBoundary = SpatialFilter::matchingX(xRight);
          }
          else
          {
            cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
            return Teuchos::null;
          }
          
          // Set up Galerkin H1-projection problem(s)
          bool H1ProjectionConformingTraces = true; // no difference for primal/continuous formulations
          int spaceDim = 2;
          double lengthScale = 1.0;
          H1ProjectionFormulation formPhi(spaceDim, H1ProjectionConformingTraces, H1ProjectionFormulation::CONTINUOUS_GALERKIN, lengthScale);
          VarPtr phi = formPhi.phi();
          BFPtr phiBF = formPhi.bf();
          MeshTopologyPtr phiMeshTopo = spatialMeshTopo->deepCopy();
          vector<int> H1Order = {k+1+delta_k}; // = order of the test space velocity
          int phiTestEnrichment = 0;
          MeshPtr phiMesh = Teuchos::rcp( new Mesh(phiMeshTopo, phiBF, H1Order, phiTestEnrichment) ) ;
          phiMesh->registerObserver(mesh);
          IPPtr phiIP = Teuchos::null;
          RHSPtr phiRHS = RHS::rhs();
          // BCPtr qBC = BC::bc(); // necessary for scaling factor
          // BCPtr phiBC = BC::bc(); // necessary for scaling factor
          BCPtr phiPlusBC = BC::bc();
          BCPtr phiMinusBC = BC::bc();

          // compute error representation function from solution increment
          bool excludeBoundaryTerms = false;
          LinearTermPtr residual = solutionIncrement->rhs()->linearTerm() - bf->testFunctional(solutionIncrement,excludeBoundaryTerms);
          RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, solutionIncrement->ip(), residual));
          rieszResidual->computeRieszRep();
          // extract the component of psi corresponding to the first test velocity component
          FunctionPtr psi_v1 =  Teuchos::rcp( new RepFunction<double>(form.v(1), rieszResidual) );


          // compute scaling factor
          double scale; // skipping this for now since it doesn't vastly affect the outcome
          // // use phiPlus for scale of q
          // qBC->addDirichlet(phi, cylinderBoundary, one);
          // qBC->addDirichlet(phi, topBoundary, zero);
          // qBC->addDirichlet(phi, bottomBoundary, zero);
          // SolutionPtr qSolution = Solution::solution(phiBF, phiMesh, qBC, phiRHS, phiIP);
          // qSolution->solve();
          // FunctionPtr extension = TFunction<double>::solution(phi, qSolution);
          // FunctionPtr dx_extension = extension->dx();
          // FunctionPtr dy_extension = extension->dy();
          // FunctionPtr d_extensionNorm = extension * extension + (lengthScale * lengthScale) * (dx_extension * dx_extension + dy_extension * dy_extension);
          // double extensionNorm = d_extensionNorm->integrate(phiMesh);
          // double qExtensionNorm = sqrt(extensionNorm);
          // // use phiPlus for scale of psi_v1
          // phiBC->addDirichlet(phi, cylinderBoundary, psi_v1);
          // phiBC->addDirichlet(phi, topBoundary, psi_v1);
          // phiBC->addDirichlet(phi, bottomBoundary, psi_v1);
          // SolutionPtr phiSolution = Solution::solution(phiBF, phiMesh, phiBC, phiRHS, phiIP);
          // phiSolution->solve();
          // extension = TFunction<double>::solution(phi, phiSolution);
          // dx_extension = extension->dx();
          // dy_extension = extension->dy();
          // d_extensionNorm = extension * extension + (lengthScale * lengthScale) * (dx_extension * dx_extension + dy_extension * dy_extension);
          // extensionNorm = d_extensionNorm->integrate(phiMesh);
          // double psiExtensionNorm = sqrt(extensionNorm);

          // scale = psiExtensionNorm / qExtensionNorm;
          // cout << "scale = " << scale << endl;
          scale = 1.0;


          // drag error ~ (e_1,psi_{v})_{1/2} = (1, psi_{v1})_{1/2}
          phiPlusBC->addDirichlet(phi, cylinderBoundary, 1.0/scale*psi_v1+scale*one);
          phiPlusBC->addDirichlet(phi, topBoundary, 1.0/scale*psi_v1);
          phiPlusBC->addDirichlet(phi, bottomBoundary, 1.0/scale*psi_v1);
          phiPlusBC->addDirichlet(phi, leftBoundary, 1.0/scale*psi_v1);
          phiPlusBC->addDirichlet(phi, rightBoundary, 1.0/scale*psi_v1);

          phiMinusBC->addDirichlet(phi, cylinderBoundary, 1.0/scale*psi_v1-scale*one);
          phiMinusBC->addDirichlet(phi, topBoundary, 1.0/scale*psi_v1);
          phiMinusBC->addDirichlet(phi, bottomBoundary, 1.0/scale*psi_v1);
          phiMinusBC->addDirichlet(phi, leftBoundary, 1.0/scale*psi_v1);
          phiMinusBC->addDirichlet(phi, rightBoundary, 1.0/scale*psi_v1);


          // if (problemChoice == "Benchmark")
          // {
          //   phiPlusBC->addDirichlet(phi, bottomBoundary, 1.0/scale*psi_v1);
          //   phiMinusBC->addDirichlet(phi, bottomBoundary, 1.0/scale*psi_v1);
          // }

          SolutionPtr phiPlusSolution = Solution::solution(phiBF, phiMesh, phiPlusBC, phiRHS, phiIP);
          SolutionPtr phiMinusSolution = Solution::solution(phiBF, phiMesh, phiMinusBC, phiRHS, phiIP);

          int outputLevel = 1;
          phiPlusSolution->setWarnAboutDiscontinuousBCs(outputLevel);
          phiPlusSolution->solve();
          phiMinusSolution->setWarnAboutDiscontinuousBCs(outputLevel);
          phiMinusSolution->solve();


          FunctionPtr phiP_fxn = TFunction<double>::solution(phi, phiPlusSolution);
          FunctionPtr dx_phiP_fxn = phiP_fxn->dx();
          FunctionPtr dy_phiP_fxn = phiP_fxn->dy();
          FunctionPtr d_eta = phiP_fxn * phiP_fxn + (lengthScale * lengthScale) * (dx_phiP_fxn * dx_phiP_fxn + dy_phiP_fxn * dy_phiP_fxn);

          FunctionPtr phiM_fxn = TFunction<double>::solution(phi, phiMinusSolution);
          FunctionPtr dx_phiM_fxn = phiM_fxn->dx();
          FunctionPtr dy_phiM_fxn = phiM_fxn->dy();
          d_eta = d_eta - phiM_fxn * phiM_fxn - (lengthScale * lengthScale) * (dx_phiM_fxn * dx_phiM_fxn + dy_phiM_fxn * dy_phiM_fxn);

          double energyThreshold = 0.01;
          ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new DragOrientedErrorIndicator<double>(phiMesh, d_eta) );
          RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, energyThreshold) );

          refStrategy->refine();


          // just refines based on energy error for debugging
          // form.refine();
        }
      }
    }
    lambda += delta_lambda;
    form.setLambda(lambda);
  }
  dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
