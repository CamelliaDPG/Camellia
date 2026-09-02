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

#include "BF.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "RefinementStrategy.h"
#include "ErrorIndicator.h"
#include "EnergyErrorIndicator.h"
#include "GMGSolver.h"
#include "OldroydBFormulationUWReduced.h"
// #include "OldroydBFormulationUW.h"
// #include "H1ProjectionFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PreviousSolutionFunction.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include "MPIWrapper.h"


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
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    const int numPoints = points.extent(0);
    for (int i=0; i<numPoints; i++)
    {
      pointsMatch(i) = matchesPoint(points(i,0), points(i,1));
    }
  }
};

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
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    const int numPoints = points.extent(0);
    for (int i=0; i<numPoints; i++)
    {
      pointsMatch(i) = matchesPoint(points(i,0), points(i,1));
    }
  }
};

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
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    const int numPoints = points.extent(0);
    for (int i=0; i<numPoints; i++)
    {
      pointsMatch(i) = matchesPoint(points(i,0), points(i,1));
    }
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction<double> {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) : SimpleFunction<double>("RampBoundaryFunction_U1", 2) {
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
  ParabolicInflowFunction_U1(double height)
  : 
  SimpleFunction<double>("ParabolicInflowFunction_U1", 2)
  {
    _height = height;
  }
  double value(double x, double y) {
    return 3.0/2.0*(1.0-pow(y/_height,2)); // chosen so that AVERAGE inflow velocity is 1.0
  }
};

class ParabolicInflowFunction_Tun : public SimpleFunction<double> {
  double _i, _j; // index numbers
  double _height; // height of channel
  double _muP; // polymeric viscosity
  double _lambda; // relaxation time
public:
  ParabolicInflowFunction_Tun(double height, double muP, double lambda, int i, int j)
  : 
  SimpleFunction<double>("ParabolicInflowFunction_Tun", 2)
  {
    _i = i;
    _j = j;
    _height = height;
    _muP = muP;
    _lambda = lambda;
  }
  double value(double x, double y) {
    if (_i == 1 && _j == 1)
      return -3.0/2.0*(1.0-pow(y/_height,2))*(18.0*_muP*_lambda*pow(y/(_height*_height),2));
    else if ((_i == 1 && _j == 2) || (_i == 2 && _j == 1))
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
    globalMax = MPIWrapper::max(_mesh->Comm(), localMax);

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

template <typename Scalar>
class GoalOrientedErrorIndicatorOden : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  std::map<GlobalIndexType, double> _jumpMap_total;
  int _cubatureDegreeEnrichment;
public:
  GoalOrientedErrorIndicatorOden(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, std::map<GlobalIndexType, double> jumpMap_total, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
  {
    _solution = soln;
    _dualSolnResidualFunction = dualSolnResidualFunction;
    _jumpMap_total = jumpMap_total;
    _cubatureDegreeEnrichment = cubatureDegreeEnrichment;
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
      double residual = energyErrorIsSquared ? sqrt(entry.second) : entry.second;
      double dualresidual = _dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment);
      dualresidual = sqrt(dualresidual + _jumpMap_total[cellID]);
      _localErrorMeasures[cellID] = residual*dualresidual;
    }

    // calculate max
    double localMax;
    double globalMax;
    for (auto measureEntry : _localErrorMeasures) {
      double cellError = measureEntry.second;
      localMax = max(localMax,cellError);
    }
    globalMax = MPIWrapper::max(_mesh->Comm(), localMax);
  }
};

template <typename Scalar>
class GoalOrientedErrorIndicatorBecker : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  std::map<GlobalIndexType, double> _jumpMap_total;
  int _cubatureDegreeEnrichment;
public:
  GoalOrientedErrorIndicatorBecker(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, std::map<GlobalIndexType, double> jumpMap_total, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
  {
    _solution = soln;
    _dualSolnResidualFunction = dualSolnResidualFunction;
    _jumpMap_total = jumpMap_total;
    _cubatureDegreeEnrichment = cubatureDegreeEnrichment;
  }

  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {

    const map<GlobalIndexType, double>* rankLocalEnergyError = &_solution->rankLocalEnergyError();

    double localResidualSquared = 0;
    double localDualResidualSquared = 0;
    double localCombinedResidualSquared = 0;
    for (auto entry : *rankLocalEnergyError) {
      GlobalIndexType cellID = entry.first;
      double residualSquared = entry.second * entry.second;
      double dualResidualSquared = _dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment) + _jumpMap_total[cellID];
      localResidualSquared += residualSquared;
      localDualResidualSquared += dualResidualSquared;
      localCombinedResidualSquared += residualSquared + dualResidualSquared;
    }
    double globalResidualSquared;
    double globalDualResidualSquared;
    double globalCombinedResidualSquared;
    globalResidualSquared = MPIWrapper::sum(_mesh->Comm(), localResidualSquared);
    globalDualResidualSquared = MPIWrapper::sum(_mesh->Comm(), localDualResidualSquared);
    globalCombinedResidualSquared = MPIWrapper::sum(_mesh->Comm(), localCombinedResidualSquared);

    _localErrorMeasures.clear();
    for (auto entry : *rankLocalEnergyError)
    {
      GlobalIndexType cellID = entry.first;
      double residualSquared = entry.second * entry.second;
      double dualResidualSquared = _dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment) + _jumpMap_total[cellID];

      _localErrorMeasures[cellID] = (globalDualResidualSquared * residualSquared + globalResidualSquared * dualResidualSquared) / globalCombinedResidualSquared ;
    }

    // calculate max
    double localMax;
    double globalMax;
    for (auto measureEntry : _localErrorMeasures) {
      double cellError = measureEntry.second;
      localMax = max(localMax,cellError);
    }
    globalMax = MPIWrapper::max(_mesh->Comm(), localMax);
  }
};


int main(int argc, char *argv[])
{

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  auto Comm = MPIWrapper::CommWorld();

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm->Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  //////////////////////////////////////////////////////////////////////
  ///////////////////////  COMMAND LINE PARAMETERS  ////////////////////
  //////////////////////////////////////////////////////////////////////
  string problemChoice = "LidDriven";
  int spaceDim = 2;
  double rho = 1;
  double lambda = 1;
  double muS = 1; // solvent viscosity
  double muP = 1; // polymeric viscosity
  double alpha = 0;
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
  double nonlinearTolerance = 1e-8;
  // double minNonlinearTolerance = 10*solverTolerance;
  // double minNonlinearTolerance = 5e-7;
  double minNonlinearTolerance = 1e-4;
  // double minNonlinearTolerance = 4e-5;
  int maxLinearIterations = 1000;
  // bool computeL2Error = false;
  bool exportSolution = false;
  bool includeJumps = false;
  bool jumpTermEdgeScaling = false;
  string norm = "Graph";
  string errorIndicator = "Energy";
  string outputDir = ".";
  string tag="";
  cmdp.setOption("problem", &problemChoice, "LidDriven, HalfHemker, Hemker");
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
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy,CylinderBoundary,GoalOrientedDragCoeff");
  cmdp.setOption("enforceLocalConservation", "noLocalConservation", &enforceLocalConservation, "enforce local conservation principles at the element level");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS, Pardiso");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  // cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("includeJumps", "skipJumps", &includeJumps, "include jump terms in DPG* estimator");
  cmdp.setOption("edgeScaling", "interiorScaling", &jumpTermEdgeScaling, "weight by edge length for jump terms");
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
  parameters.set("spaceDim", spaceDim);
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
  parameters.set("\\hat{(T\\otimes u)_{n_{11}}}-polyOrderAdjustment", 1);
  parameters.set("\\hat{(T\\otimes u)_{n_{12}}}-polyOrderAdjustment", 1);
  parameters.set("\\hat{(T\\otimes u)_{n_{22}}}-polyOrderAdjustment", 1);


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
  else if (problemChoice == "Hemker")
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
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  ///////////
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);
  OldroydBFormulationUWReduced form(spatialMeshTopo, parameters);
  // OldroydBFormulationUW form(spatialMeshTopo, parameters);
  ///////////
  MeshPtr mesh = form.solutionIncrement()->mesh();
  if (globalEdgeToCurveMap.size() > 0)
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }


  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr solutionIncrement = form.solutionIncrement();
  SolutionPtr solutionBackground = form.solution();


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
  else if (problemChoice == "Hemker")
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
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }

  //////////////////////////////////////////////////////////////////////
  ///////////////////////////////  SOLVE  //////////////////////////////
  //////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Teuchos::Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");

  if (commRank == 0)
    Solver::printAvailableSolversReport();
  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#if defined(HAVE_AMESOS_MUMPS) && defined(HAVE_MPI)
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
#ifdef HAVE_AMESOS_PARDISO_MKL
  solvers["Pardiso"] = Solver::getSolver(Solver::Pardiso, true);
#endif

  // choose local normal equation matrix calculation algorithm
  BFPtr bf = form.bf();
  // bf->setOptimalTestSolver(TBF<>::CHOLESKY);
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);

  ostringstream solnName;
  solnName << "GoalOrientedOldroydB" << "_" << norm << "_k" << k << "_dk" << delta_k << "_" << solverChoice;
  if (solverChoice[0] == 'G')
    solnName << "_" << multigridStrategyString;
  if (tag != "")
    solnName << "_" << tag;
  if (stokesOnly)
    solnName << "_Stokes";
  else
    solnName << "_NavierStokes";
  solnName << "_lambda_" << lambda;

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
           << "dofs - BCdofs\t "
           << "energy\t "
           << "solvetime\t "
           << "elapsed\t "
           << "iterations\t "
           << "drag coefficient (field)\t "
           << "drag coefficient (traction)\t "
           << "y-direction force\t "
           << "drag error estimate\t "
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter, functionExporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));
  string exporterName = "_DualSolutions";
  exporterName = solnName.str() + exporterName;
  functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));

  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {

    if (solverChoice != "SuperLUDist" && refIndex == 0)
      form.setSolver(solvers[solverChoice]);


    solverTime->start(true);

    ////////////////////////////////////////////////////////////////////
    //    Solve and accumulate solution
    ////////////////////////////////////////////////////////////////////
    int iterCount = 0;
    int iterationCount = 0;
    double l2Update = 1e10;
    double l2UpdateInitial = l2Update;
    while (l2Update > nonlinearTolerance*l2UpdateInitial && iterCount < maxNonlinearIterations && l2Update > minNonlinearTolerance)
    {      
      form.solveForIncrement();

      // Accumulate solution
      form.accumulate();

      ////////////////////////////////////////////////////////////////////
      // Compute L2 norm of update and increment counters
      ////////////////////////////////////////////////////////////////////
      l2Update = form.L2NormSolutionIncrement();

      if (commRank == 0)
        cout << "Nonlinear Update:\t " << l2Update << endl;

      if (iterCount == 0)
        l2UpdateInitial = l2Update;

      iterCount++;
    }

    double solveTime = solverTime->stop();

    double energyError = solutionIncrement->energyErrorTotal();

    // compute drag coefficient if Hemker problem
    double fieldDragCoefficient = 0.0;
    double fluxDragCoefficient = 0.0;
    double verticalForce = 0.0;
    double dragError = 0.0;
    if (problemChoice == "HalfHemker" || problemChoice == "Hemker")
    {
      SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius) );

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

      // compute force in y-direction (from flux)
      TFunctionPtr<double> traction_y = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(2)) );
      traction_y = Teuchos::rcp( new SpatiallyFilteredFunction<double>( traction_y*boundaryRestriction, cylinderBoundary) );
      verticalForce = traction_y->integrate(mesh);

      dragError = ((fieldTraction_x-fluxTraction_x)*(fieldTraction_x-fluxTraction_x))->integrate(mesh);
      dragError = sqrt(2.0*M_PI)*sqrt(dragError);

      if (problemChoice == "HalfHemker")
      {
        fieldDragCoefficient = 2.0*fieldDragCoefficient;
        fluxDragCoefficient  = 2.0*fluxDragCoefficient;
        dragError = 2.0*dragError;
      }
    }

    if (commRank == 0)
    {
      cout << setprecision(8) 
        << "Lambda: " << lambda
        << " \nRefinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << " \tDOFs - BCDOFs: " << mesh->numGlobalDofs() - solutionIncrement->getBCDofCount()
        << " \tEnergy Error: " << energyError
        << " \nSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << " \tIteration Count: " << iterationCount
        << " \nDrag Coefficient (from field): " << fieldDragCoefficient
        << " \tDrag Coefficient (from traction): " << fluxDragCoefficient
        << " \ty-direction Force : " << verticalForce
        << " \tDrag error estimate : " << dragError
        << endl;
      dataFile << setprecision(8)
        << lambda
        << " " << refIndex
        << " " << mesh->numActiveElements()
        << " " << mesh->numGlobalDofs()
        << " " << mesh->numGlobalDofs() - solutionIncrement->getBCDofCount()
        << " " << energyError
        << " " << solveTime
        << " " << totalTimer->totalElapsedTime(true)
        << " " << iterationCount
        << " " << fieldDragCoefficient
        << " " << fluxDragCoefficient
        << " " << verticalForce
        << " " << dragError
        << endl;
    }

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

    if (refIndex != numRefs)
    {
      ///////////////////  CHOOSE REFINEMENT STRATEGY  ////////////////////
      if (errorIndicator == "Energy")
      {
        // RefinementStrategyPtr refStrategy = form.getRefinementStrategy();
        // refStrategy->setEnforceOneIrregularity(false);
        // refStrategy->refine();
        form.refine();
      }
      if (errorIndicator == "hpEnergy")
      {
        ///////
        // Mark elements based on energy error
        ///////
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new EnergyErrorIndicator<double>(solutionIncrement) );
        double refThreshold = 0.2;
        RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
               
        vector<GlobalIndexType> localCellsToRefine;
        refStrategy->getCellsAboveErrorThreshhold(localCellsToRefine); 
        double totalError = errorIndicator->totalError();

        FunctionPtr T_11_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(1,1)) );
        FunctionPtr T_12_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(1,2)) );
        FunctionPtr T_22_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(2,2)) );

        bool weightBySideMeasure = false;
        int cubatureDegreeEnrichment = delta_k;
        std::map<GlobalIndexType, double> jumpMap_T11 = T_11_prvs->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_T12 = T_12_prvs->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_T22 = T_22_prvs->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);

        std::map<GlobalIndexType, double> jumpMap_T;
        double localMax_jumpT = 0;
        for (GlobalIndexType cellID : localCellsToRefine)
        {
          jumpMap_T[cellID] = jumpMap_T11[cellID] + jumpMap_T12[cellID] + jumpMap_T22[cellID];
          localMax_jumpT = max(localMax_jumpT,jumpMap_T[cellID]);
        }
        double globalMax_jumpT;
        globalMax_jumpT = MPIWrapper::max(mesh->Comm(), localMax_jumpT);



        vector<GlobalIndexType> myCellsToHRefine;
        vector<GlobalIndexType> myCellsToHxRefine;
        vector<GlobalIndexType> myCellsToHyRefine;
        vector<GlobalIndexType> myCellsToPRefine;

        double hpThreshold = 0.1;
        for (GlobalIndexType cellID : localCellsToRefine)
        {
          if (jumpMap_T[cellID] > hpThreshold*globalMax_jumpT)
          {
            myCellsToHRefine.push_back(cellID);
            // myCellsToHxRefine.push_back(cellID);
            // myCellsToHyRefine.push_back(cellID);
          }
          else
          {
            myCellsToPRefine.push_back(cellID);
          }
        }

        // isotropic h-ref
        GlobalIndexTypeToCast numCellsToHRefine = 0;
        GlobalIndexTypeToCast myNumCellsToHRefine = myCellsToHRefine.size();
        numCellsToHRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToHRefine);
        GlobalIndexTypeToCast myCellOrdinalOffset = MPIWrapper::scanSum(mesh->Comm(), static_cast<int>(myNumCellsToHRefine)) - myNumCellsToHRefine;
        
        vector<GlobalIndexTypeToCast> globalCellsToHRefineVector(numCellsToHRefine, 0);
        for (int i=0; i<myNumCellsToHRefine; i++)
        {
          globalCellsToHRefineVector[myCellOrdinalOffset + i] = myCellsToHRefine[i];
        }
        vector<GlobalIndexTypeToCast> gatheredCellsToHRefineVector(numCellsToHRefine, 0);
        MPIWrapper::entryWiseSum(mesh->Comm(), globalCellsToHRefineVector);
        gatheredCellsToHRefineVector = globalCellsToHRefineVector;

        vector<GlobalIndexType> cellsToHRefine(gatheredCellsToHRefineVector.begin(),gatheredCellsToHRefineVector.end());;

        // anisotropic (x) h-ref
        GlobalIndexTypeToCast numCellsToHxRefine = 0;
        GlobalIndexTypeToCast myNumCellsToHxRefine = myCellsToHxRefine.size();
        numCellsToHxRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToHxRefine);
        myCellOrdinalOffset = MPIWrapper::scanSum(mesh->Comm(), static_cast<int>(myNumCellsToHxRefine)) - myNumCellsToHxRefine;
        
        vector<GlobalIndexTypeToCast> globalCellsToHxRefineVector(numCellsToHxRefine, 0);
        for (int i=0; i<myNumCellsToHxRefine; i++)
        {
          globalCellsToHxRefineVector[myCellOrdinalOffset + i] = myCellsToHxRefine[i];
        }
        vector<GlobalIndexTypeToCast> gatheredCellsToHxRefineVector(numCellsToHxRefine, 0);
        MPIWrapper::entryWiseSum(mesh->Comm(), globalCellsToHxRefineVector);
        gatheredCellsToHxRefineVector = globalCellsToHxRefineVector;

        vector<GlobalIndexType> cellsToHxRefine(gatheredCellsToHxRefineVector.begin(),gatheredCellsToHxRefineVector.end());;

        // anisotropic (y) h-ref
        GlobalIndexTypeToCast numCellsToHyRefine = 0;
        GlobalIndexTypeToCast myNumCellsToHyRefine = myCellsToHyRefine.size();
        numCellsToHyRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToHyRefine);
        myCellOrdinalOffset = MPIWrapper::scanSum(mesh->Comm(), static_cast<int>(myNumCellsToHyRefine)) - myNumCellsToHyRefine;
        
        vector<GlobalIndexTypeToCast> globalCellsToHyRefineVector(numCellsToHyRefine, 0);
        for (int i=0; i<myNumCellsToHyRefine; i++)
        {
          globalCellsToHyRefineVector[myCellOrdinalOffset + i] = myCellsToHyRefine[i];
        }
        vector<GlobalIndexTypeToCast> gatheredCellsToHyRefineVector(numCellsToHyRefine, 0);
        MPIWrapper::entryWiseSum(mesh->Comm(), globalCellsToHyRefineVector);
        gatheredCellsToHyRefineVector = globalCellsToHyRefineVector;

        vector<GlobalIndexType> cellsToHyRefine(gatheredCellsToHyRefineVector.begin(),gatheredCellsToHyRefineVector.end());;
        
        GlobalIndexTypeToCast numCellsToPRefine = 0;
        GlobalIndexTypeToCast myNumCellsToPRefine = myCellsToPRefine.size();
        numCellsToPRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToPRefine);

        // isotropic p-ref
        vector<GlobalIndexType> cellsToPRefine;
        if (numCellsToPRefine > 0)
        {
          myCellOrdinalOffset = MPIWrapper::scanSum(mesh->Comm(), static_cast<int>(myNumCellsToPRefine)) - myNumCellsToPRefine;

          vector<GlobalIndexTypeToCast> globalCellsToPRefineVector(numCellsToPRefine, 0);
          for (int i=0; i<myNumCellsToPRefine; i++)
          {
            globalCellsToPRefineVector[myCellOrdinalOffset + i] = myCellsToPRefine[i];
          }
          vector<GlobalIndexTypeToCast> gatheredCellsToPRefineVector(numCellsToPRefine, 0);
          MPIWrapper::entryWiseSum(mesh->Comm(), globalCellsToPRefineVector);
          gatheredCellsToPRefineVector = globalCellsToPRefineVector;

          cellsToPRefine = vector<GlobalIndexType>(gatheredCellsToPRefineVector.begin(),gatheredCellsToPRefineVector.end());
        }
        
        std::sort(cellsToHRefine.begin(), cellsToHRefine.end());
        std::sort(cellsToHxRefine.begin(), cellsToHxRefine.end());
        std::sort(cellsToHyRefine.begin(), cellsToHyRefine.end());
        std::sort(cellsToPRefine.begin(), cellsToPRefine.end());
        
        bool repartitionAndRebuild = false;
        int pToAdd = 1;

        mesh->hRefine(cellsToHRefine, repartitionAndRebuild);
        mesh->hRefine(cellsToHxRefine, RefinementPattern::xAnisotropicRefinementPatternQuad(), repartitionAndRebuild);
        mesh->hRefine(cellsToHyRefine, RefinementPattern::yAnisotropicRefinementPatternQuad(), repartitionAndRebuild);
        set<GlobalIndexType> cellSet(cellsToPRefine.begin(), cellsToPRefine.end());
        mesh->pRefine(cellSet, pToAdd, repartitionAndRebuild);

        mesh->enforceOneIrregularity(repartitionAndRebuild);

        // now, repartition and rebuild:
        mesh->repartitionAndRebuild();
      }
      else if (errorIndicator == "CylinderBoundary")
      {
        if (problemChoice == "HalfHemker" || problemChoice == "Hemker")
        {
          // SpatialFilterPtr cylinderBoundaryGenerous = Teuchos::rcp( new CylinderBoundaryGenerous(cylinderRadius));
          // ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryGenerous) );
          SpatialFilterPtr cylinderBoundaryExtreme = Teuchos::rcp( new CylinderBoundaryExtreme(cylinderRadius));
          ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new BoundaryOrientedErrorIndicator<double>(solutionIncrement, cylinderBoundaryExtreme) );
          double refThreshold = 0.2;
          RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
          refStrategy->refine();
        }
        else
        {
          cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
          return Teuchos::null;
        }
      }
      else if (errorIndicator == "GoalOrientedDragCoeff")
      {
        if (problemChoice != "HalfHemker" && problemChoice != "Hemker")
        {
          cout << "ERROR: Error indicator type not currently supported for this mesh. Returning null.\n";
          return Teuchos::null;
        }

        // CONFINED CYLINDER exploiting geometric symmetry
        SpatialFilterPtr leftBoundary = SpatialFilter::matchingX(xLeft);
        SpatialFilterPtr rightBoundary = SpatialFilter::matchingX(xRight);
        SpatialFilterPtr topBoundary = SpatialFilter::matchingY(yMax);
        SpatialFilterPtr bottomBoundary = SpatialFilter::matchingY(0.0);
        SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));

        // define goal functional
        FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
        // FunctionPtr cylinderNormal_x, cylinderNormal_y;
        // cylinderNormal_x = Teuchos::rcp( new SpatiallyFilteredFunction<double>( n->x()*boundaryRestriction, cylinderBoundary) );
        // cylinderNormal_y = Teuchos::rcp( new SpatiallyFilteredFunction<double>( n->y()*boundaryRestriction, cylinderBoundary) );
        FunctionPtr v1_0 = Teuchos::rcp( new SpatiallyFilteredFunction<double>( 1.0 * boundaryRestriction, cylinderBoundary) );

        LinearTermPtr g_functional = v1_0 * form.sigman_hat(1);
        // g_functional = cylinderNormal_x * form.sigman_hat(1) + cylinderNormal_y * form.sigman_hat(2);

        // set second RHS
        solutionIncrement->setGoalOrientedRHS(g_functional);

        // solve again
        form.solveForIncrement();

        // construct DPG* solution
        bool excludeBoundaryTerms = false;
        LinearTermPtr influence = form.bf()->testFunctional(solutionIncrement,excludeBoundaryTerms);
        RieszRepPtr dualSoln = Teuchos::rcp(new RieszRep(mesh, form.bf()->graphNorm(), influence));
        dualSoln->computeRieszRep();

        FunctionPtr dualSoln_q, dualSoln_v1, dualSoln_v2, dualSoln_M1, dualSoln_M2, dualSoln_S11, dualSoln_S12, dualSoln_S22;
        // dualSoln_q   =  Teuchos::rcp( new RepFunction<double>(form.q(), dualSoln) );
        dualSoln_v1  =  Teuchos::rcp( new RepFunction<double>(form.v(1), dualSoln) );
        dualSoln_v2  =  Teuchos::rcp( new RepFunction<double>(form.v(2), dualSoln) );
        dualSoln_M1  =  Teuchos::rcp( new RepFunction<double>(form.M(1), dualSoln) );
        dualSoln_M2  =  Teuchos::rcp( new RepFunction<double>(form.M(2), dualSoln) );
        dualSoln_S11 =  Teuchos::rcp( new RepFunction<double>(form.S(1,1), dualSoln) );
        dualSoln_S12 =  Teuchos::rcp( new RepFunction<double>(form.S(1,2), dualSoln) );
        dualSoln_S22 =  Teuchos::rcp( new RepFunction<double>(form.S(2,2), dualSoln) );

        // construct DPG* residual
        map<int, FunctionPtr > opDualSoln = bf()->applyAdjointOperatorDPGstar(dualSoln);
        FunctionPtr dualSolnResFxn = zero;
        for ( auto opDualSolnComponent  : opDualSoln  )
        {
          FunctionPtr f = opDualSolnComponent.second;
          dualSolnResFxn = dualSolnResFxn + f * f;
        }

        // evaluate all jump terms        
        SpatialFilterPtr filter_v1 = SpatialFilter::negatedFilter(bottomBoundary);
        SpatialFilterPtr filter_M1 = cylinderBoundary | leftBoundary | topBoundary | rightBoundary;
        filter_M1 = SpatialFilter::negatedFilter(filter_M1);
        SpatialFilterPtr filter_S11 = cylinderBoundary | leftBoundary | topBoundary | bottomBoundary;
        filter_S11 = SpatialFilter::negatedFilter(filter_S11);
        SpatialFilterPtr filter_S12 = cylinderBoundary | leftBoundary | topBoundary;
        filter_S12 = SpatialFilter::negatedFilter(filter_S12);


        FunctionPtr v1_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_v1 - v1_0, filter_v1) );
        // SpatialFilterPtr notBottomBoundary = SpatialFilter::greaterThanY(0.0);
        // FunctionPtr v1_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_v1 - v1_0, notBottomBoundary) );
        FunctionPtr v2_minusBC = dualSoln_v2;
        FunctionPtr gradv1 = dualSoln_v1->dx()*n->y() - dualSoln_v1->dy()*n->x();
        FunctionPtr gradv2 = dualSoln_v2->dx()*n->y() - dualSoln_v2->dy()*n->x();
        FunctionPtr gradv1_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( gradv1, filter_v1) );
        FunctionPtr gradv2_minusBC = (1.0 - boundaryRestriction) * gradv2;
        FunctionPtr M1_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_M1, filter_M1) );
        FunctionPtr M2_minusBC = (1.0 - boundaryRestriction) * dualSoln_M2;
        FunctionPtr S11_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_S11, filter_S11) );
        FunctionPtr S12_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_S12, filter_S12) );
        FunctionPtr S22_minusBC = Teuchos::rcp( new SpatiallyFilteredFunction<double>( dualSoln_S22, filter_S11) );
        // FunctionPtr gradS11 = dualSoln_S11->dx()*n->y() - dualSoln_S11->dy()*n->x();
        // FunctionPtr gradS12 = dualSoln_S12->dx()*n->y() - dualSoln_S12->dy()*n->x();
        // FunctionPtr gradS22 = dualSoln_S22->dx()*n->y() - dualSoln_S22->dy()*n->x();
        // FunctionPtr gradS11_minusBC = (1.0 - boundaryRestriction) * gradS11;
        // FunctionPtr gradS12_minusBC = (1.0 - boundaryRestriction) * gradS12;
        // FunctionPtr gradS22_minusBC = (1.0 - boundaryRestriction) * gradS22;

        bool weightBySideMeasure = false;
        if (jumpTermEdgeScaling == true)
          weightBySideMeasure = true;
        int cubatureDegreeEnrichment = delta_k;
        std::map<GlobalIndexType, double> jumpMap_v1 = v1_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_v2 = v2_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_M1 = M1_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_M2 = M2_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_S11 = S11_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_S12 = S12_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_S22 = S22_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);

        std::map<GlobalIndexType, double> jumpMap_gradv1 = gradv1_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        std::map<GlobalIndexType, double> jumpMap_gradv2 = gradv2_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        // std::map<GlobalIndexType, double> jumpMap_gradS11 = gradS11_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        // std::map<GlobalIndexType, double> jumpMap_gradS12 = gradS12_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);
        // std::map<GlobalIndexType, double> jumpMap_gradS22 = gradS22_minusBC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureDegreeEnrichment);

        // construct weights for jump terms
        FunctionPtr u_1_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.u(1)) );
        FunctionPtr u_2_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.u(2)) );
        FunctionPtr L_11_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.L(1,1)) );
        FunctionPtr L_12_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.L(1,2)) );
        FunctionPtr L_21_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.L(2,1)) );
        FunctionPtr L_22_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.L(2,2)) );
        FunctionPtr T_11_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(1,1)) );
        FunctionPtr T_12_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(1,2)) );
        FunctionPtr T_22_prvs = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.T(2,2)) );

        double u_1_max = u_1_prvs->linfinitynorm(mesh);
        double u_2_max = u_2_prvs->linfinitynorm(mesh);
        double L_11_max = L_11_prvs->linfinitynorm(mesh);
        double L_12_max = L_12_prvs->linfinitynorm(mesh);
        double L_21_max = L_21_prvs->linfinitynorm(mesh);
        double L_22_max = L_22_prvs->linfinitynorm(mesh);
        double T_11_max = T_11_prvs->linfinitynorm(mesh);
        double T_12_max = T_12_prvs->linfinitynorm(mesh);
        double T_22_max = T_22_prvs->linfinitynorm(mesh);

        double u_max = std::max(u_1_max,u_2_max);
        double L1_max = std::max(L_11_max,L_12_max);
        double L2_max = std::max(L_21_max,L_22_max);
        double L_max = std::max(L1_max,L2_max);
        double T_max = std::max(T_11_max,T_12_max);
        T_max = std::max(T_max,T_22_max);

        // weight/scale jump terms
        std::map<GlobalIndexType, double> jumpMap_total;
        const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
        double jump_v1=0.0, jump_v2=0.0;
        double jump_M1=0.0, jump_M2=0.0;
        double jump_S11=0.0, jump_S12=0.0, jump_S22=0.0;
        for (auto cellID: myCellIDs)
        {
          double jump_v1_tmp, jump_v2_tmp;
          double jump_M1_tmp, jump_M2_tmp;
          double jump_S11_tmp, jump_S12_tmp, jump_S22_tmp;
          if (weightBySideMeasure)
          {
            jump_v1_tmp  = jumpMap_v1[cellID] + jumpMap_gradv1[cellID];
            jump_v2_tmp  = jumpMap_v2[cellID] + jumpMap_gradv2[cellID];
            jump_M1_tmp  = jumpMap_M1[cellID];
            jump_M2_tmp  = jumpMap_M2[cellID];
            jump_S11_tmp = jumpMap_S11[cellID];
            jump_S12_tmp = jumpMap_S12[cellID];
            jump_S22_tmp = jumpMap_S22[cellID];
          }
          else
          {
            double vol = mesh->getCellMeasure(cellID);
            double h = pow(vol, 1.0 / spaceDim);
            jump_v1_tmp  = pow(h, -1.0)*jumpMap_v1[cellID];
            jump_v2_tmp  = pow(h, -1.0)*jumpMap_v2[cellID];
            jump_M1_tmp  = h*jumpMap_M1[cellID];
            jump_M2_tmp  = h*jumpMap_M2[cellID];
            jump_S11_tmp = jumpMap_S11[cellID];
            jump_S12_tmp = jumpMap_S12[cellID];
            jump_S22_tmp = jumpMap_S22[cellID];
          }
          jump_v1  += jump_v1_tmp;
          jump_v2  += jump_v2_tmp;
          jump_M1  += jump_M1_tmp;
          jump_M2  += jump_M2_tmp;
          jump_S11 += jump_S11_tmp;
          jump_S12 += jump_S12_tmp;
          jump_S22 += jump_S22_tmp;
          if (includeJumps == true)
            jumpMap_total[cellID] = (jump_v1_tmp + jump_v2_tmp)/(u_max*u_max) + (jump_M1_tmp + jump_M2_tmp)/(L_max*L_max) + (jump_S11_tmp + jump_S12_tmp + jump_S22_tmp)/(T_max*T_max);
            // jumpMap_total[cellID] = jump_v1 + jump_v2 + jump_M1 + jump_M2 + jump_S11 + jump_S12 + jump_S22;
          else
            jumpMap_total[cellID] = 0.0;
        }
        jump_v1 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_v1));
        jump_v2 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_v2));
        jump_M1 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_M1));
        jump_M2 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_M2));
        jump_S11 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_S11));
        jump_S12 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_S12));
        jump_S22 = sqrt(MPIWrapper::sum(mesh->Comm(), jump_S22));

        // print outputs
        double dualSolnRes = sqrt(dualSolnResFxn->l1norm(mesh, cubatureDegreeEnrichment));
        if (commRank == 0)
        {
          cout << setprecision(8) 
            << "\nDPG* residual: " << dualSolnRes
            << "\njump in v1:    " << jump_v1
            << "\njump in v2:    " << jump_v2
            << "\njump in M1:    " << jump_M1
            << "\njump in M2:    " << jump_M2
            << "\njump in S11:   " << jump_S11
            << "\njump in S12:   " << jump_S12
            << "\njump in S22:   " << jump_S22
            << "\nu_1_max:       " << u_1_max
            << "\nu_2_max:       " << u_2_max
            << "\nL_11_max:      " << L_11_max
            << "\nL_12_max:      " << L_12_max
            << "\nL_21_max:      " << L_21_max
            << "\nL_22_max:      " << L_22_max
            << "\nT_11_max:      " << T_11_max
            << "\nT_12_max:      " << T_12_max
            << "\nT_22_max:      " << T_22_max
            << endl;
        }

        // export DPG* solution
        vector<FunctionPtr> functionsToExport = {dualSoln_v1, dualSoln_v2, dualSoln_M1, dualSoln_M2, dualSoln_S11, dualSoln_S12, dualSoln_S22, dualSolnResFxn};
        // vector<FunctionPtr> functionsToExport = {dualSoln_q, dualSoln_v1, dualSoln_v2, dualSoln_M1, dualSoln_M2, dualSoln_S11, dualSoln_S12, dualSoln_S22, dualSolnResFxn};
        vector<string> functionsToExportNames = {"dual_q", "dual_v1", "dual_v2", "dual_M1", "dual_M2", "dual_S11", "dual_S12", "dual_S22", "dualSolnResFxn"};

        if (exportSolution)
        {
          int numLinearPointsPlotting = max(k,15);
          functionExporter->exportFunction(functionsToExport, functionsToExportNames, refIndex, numLinearPointsPlotting);
        }

        // remove second RHS
        solutionIncrement->setGoalOrientedRHS(Teuchos::null);

        // refine
        form.solveForIncrement();
        form.accumulate();
        // form.refine();

        double refThreshold = 0.2;
        // if (refIndex == 0)
        //   refThreshold = 0.0;          
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new GoalOrientedErrorIndicatorBecker<double>(solutionIncrement, dualSolnResFxn, jumpMap_total, cubatureDegreeEnrichment) );
        // ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new GoalOrientedErrorIndicatorOden<double>(solutionIncrement, dualSolnResFxn, jumpMap_total, cubatureDegreeEnrichment) );
        RefinementStrategyPtr refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
        refStrategy->refine();

      }
    }
  }
  dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
