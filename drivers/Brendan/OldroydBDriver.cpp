#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "GMGSolver.h"
// #include "OldroydBFormulation.h"
#include "OldroydBFormulation2.h"
// #include "StokesVGPFormulation.h"
// #include "NavierStokesVGPFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PreviousSolutionFunction.h"

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
  double _height; // ramp width
public:
  ParabolicInflowFunction_U1(double height) {
    _height = height;
  }
  double value(double x, double y) {
    return (1-pow(2.0*y/_height,2));
    // return (16.0*y-pow(y,2))/64.0;
  }
};

class ParabolicInflowFunction_Tun : public SimpleFunction<double> {
  double _i, _j; // index numbers
  double _height; // ramp width
  double _muS; // polymeric viscosity
  double _lambda; // relaxation time
public:
  ParabolicInflowFunction_Tun(double height, double muS, double lambda, int i, int j) {
    _i = i;
    _j = j;
    _height = height;
    _muS = muS;
    _lambda = lambda;
  }
  double value(double x, double y) {
    if (_i == 1 && _j == 1)
      return (1-pow(2.0*y/_height,2))*(8.0*_muS*_lambda*pow(y/(_height*_height/4.0),2));
    else if ((_i == 1 && _j == 2) || (_i == 2 && _j == 1))
      return (1-pow(2.0*y/_height,2))*(-2.0*_muS*y/(_height*_height/4.0));
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
  // double rho = 1;
  double lambda = 1;
  double muS = 1; // solvent viscsity
  double muP = 1; // polymeric viscosity
  double alpha = 0;
  // int spaceDim = 2;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  int numXElems = 2;
  int numYElems = 2;
  bool useConformingTraces = true;
  string solverChoice = "KLU";
  string multigridStrategyString = "V-cycle";
  bool useCondensedSolve = false;
  bool useConjugateGradient = true;
  bool logFineOperator = false;
  double solverTolerance = 1e-10;
  int maxNonlinearIterations = 20;
  double nonlinearTolerance = 1e-5;
  int maxLinearIterations = 10000;
  // bool computeL2Error = false;
  bool exportSolution = false;
  string norm = "Graph";
  string outputDir = ".";
  string tag="";
  cmdp.setOption("formulation", &formulation, "OldroydB, NavierStokes");
  cmdp.setOption("problem", &problemChoice, "LidDriven, HemkerCylinder");
  // cmdp.setOption("rho", &rho, "rho");
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
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLU, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full, or Two-level");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  // cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
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
  parameters.set("useConformingTraces", useConformingTraces);
  parameters.set("useConservationFormulation",false);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  // if (formulation == "NavierStokes")
  // {
  //   parameters.set("mu", muS);
  // }
  // else
  // {
    // parameters.set("rho", rho);
    parameters.set("lambda", lambda);
    parameters.set("muS", muS);
    parameters.set("muP", muP);
    parameters.set("alpha", alpha);
  // }


  //////////////////////  DECLARE EXACT SOLUTION  //////////////////////
  // FunctionPtr u_exact = Function::constant(1) - 2*Function::xn(1);


  ///////////////////////////  DECLARE MESH  ///////////////////////////

  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;

  double xLeft, xRight, height, cylinderRadius;
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



  // form = NULL;
  // if (formulation == "NavierStokes")
  // {
  //   NavierStokesVGPFormulation form(spatialMeshTopo, parameters);
  // }
  // else
  // {
    // OldroydBFormulation form(spatialMeshTopo, parameters);
  // }
  // NavierStokesVGPFormulation form(spatialMeshTopo, parameters);
  // OldroydBFormulation form(spatialMeshTopo, parameters);
  OldroydBFormulation2 form(spatialMeshTopo, parameters);
  // StokesVGPFormulation form(spatialMeshTopo, parameters);



  MeshPtr mesh = form.solutionIncrement()->mesh();
  if (globalEdgeToCurveMap.size() > 0)
  {
    mesh->setEdgeToCurveMap(globalEdgeToCurveMap);
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
    TFunctionPtr<double> u1_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_U1(height) );
    // TFunctionPtr<double> u1_inflowFunction = one;
    TFunctionPtr<double> u2_inflowFunction = zero;

    TFunctionPtr<double> T11un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height, muS, lambda, 1, 1) );
    TFunctionPtr<double> T12un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height, muS, lambda, 1, 2) );
    TFunctionPtr<double> T22un_inflowFunction = Teuchos::rcp( new ParabolicInflowFunction_Tun(height, muS, lambda, 2, 2) );

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
    form.addOutflowCondition(rightBoundary, false); // false for zero flux variable

    // no slip on cylinder
    form.addWallCondition(cylinderBoundary);

    // cout << "ERROR: Problem type not currently supported. Returning null.\n";
    // return Teuchos::null;
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

  ostringstream solnName;
  solnName << "OldroydB" << "_" << norm << "_k" << k << "_" << solverChoice;// << "_" << multigridStrategyString;
  if (solverChoice[0] == 'G')
    solnName << "_" << multigridStrategyString;
  if (tag != "")
    solnName << "_" << tag;

  // RefinementStrategyPtr refStrategy = form.getRefinementStrategy();
  Teuchos::RCP<HDF5Exporter> exporter;
  if (exportSolution)
    exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));

  Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");

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
  bool useStaticCondensation = true;
  int azOutput = 20; // print residual every 20 CG iterations

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

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";
  ofstream dataFile(dataFileLocation);
  dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "elapsed\t" << "iterations\t " << endl;

  double energyTol = 0.01;
  double energyErrorInitial;
  double lambdaInitial = lambda;
  double lambdaMax = 10.0*lambda;
  double delta_lambda = 0.5*lambda;
  while (lambda <= lambdaMax)
  {
    for (int refIndex=0; refIndex <= numRefs; refIndex++)
    {
      solverTime->start(true);
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
        while (meshSequence[0]->numGlobalDofs() < 2000 && meshSequence.size() > 2)
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
        // soln->solve(gmgSolver);
      }
      // else
      //   soln->condensedSolve(solvers[solverChoice]);

      ////////////////////////////////////////////////////////////////////
      //    line search to minimize residual in search direction
      ////////////////////////////////////////////////////////////////////
      int sMax = 16;
      int lineSearchMaxIt = 5;
      double LStol = 0.5; // as taken from Matthies + Strang
      //
      double s0 = 0.0;
      double s1 = 1.0;
      // G_i = R(u_i) . delta_u
      //     = R(u_0 + s_i * delta_u) . delta_u
      double G_init = form.computeG(0);
      double G0 = G_init;
      double G1 = form.computeG(1);
      
      // find interval about which G changes sign
      while (sgn(G0)*sgn(G1) > 0 && s1 < sMax)
      {
          s0 = s1;
          s1 = 2*s1;
          G0 = G1;
          
          // compute G1 =  R(u_0+s1*delta_u) . delta
          G1 = form.computeG(s1);
      }

      // find zero of G using this cool Illinois algorithm
      double s = s1;
      double G = G1;
      int i=0;
      while (i <= lineSearchMaxIt && sgn(G1)*sgn(G0) < 0 && ( abs(G) > LStol*abs(G_init) || abs(s0-s1) > LStol*0.5*(s0+s1)))
      {
          ++i;
          
          s = s1-G1*(s1-s0)/(G1-G0);

          // compute G1 =  R(u_0+s*delta_u) . delta
          G1 = form.computeG(s);

          if ((sgn(G)*sgn(G1)) > 0)
          {
              G0 = 0.5*G0;
          }
          else
          {
              s0 = s1;
              G0 = G1;
          }
          s1 = s;
          G1 = G;
      }

      ////////////////////////////////////////////////////////////////////
      //    Solve and accumulate solution
      ////////////////////////////////////////////////////////////////////
      // double s = 1.0;
      int iterCount = 0;
      int iterationCount = 0;
      double l2Update = 1e10;
      double l2UpdateInitial = l2Update;
      while (l2Update > nonlinearTolerance*l2UpdateInitial && iterCount < maxNonlinearIterations)
      {
        if (solverChoice[0] == 'G')
        {
          // solutionIncrement->solve(gmgSolver);
          form.solveAndAccumulate(s);
          iterationCount += gmgSolver->iterationCount();
        }
        else
          form.solveAndAccumulate(s);
          // solutionIncrement->condensedSolve(solvers[solverChoice]);

        // Compute L2 norm of update
        l2Update = form.L2NormSolutionIncrement();

        if (commRank == 0)
          cout << "Nonlinear Update:\t " << l2Update << endl;

        if (iterCount == 0)
          l2UpdateInitial = l2Update;

        if (l2Update < 1e-12)
          break;

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
      if (commRank == 0)
      {
        // compute drag coefficient if Hemker problem
        double dragCoefficient = 0.0;
        double verticalForce = 0.0;
        if (problemChoice == "HemkerCylinder")
        {
          SpatialFilterPtr cylinderBoundary = Teuchos::rcp( new CylinderBoundary(cylinderRadius));

          TFunctionPtr<double> boundaryRestriction = Function::meshBoundaryCharacteristic();

          // TFunctionPtr<double> traction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(1)));

          TFunctionPtr<double> n = TFunction<double>::normal();

          // L = muS*du/dx
          LinearTermPtr f_lt = - form.p()*n->x() + 2.0*form.L(1,1)*n->x() + form.T(1,1)*n->x()
                               + form.L(1,2)*n->y() + form.L(2,1)*n->y() + form.T(1,2)*n->y();

          TFunctionPtr<double> traction_x = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, -f_lt ) );





          
          // TFunctionPtr<double> dF_D = Teuchos::rcp( new SpatiallyFilteredFunction<double>( Function::constant(1.0)*boundaryRestriction,cylinderBoundary));
          TFunctionPtr<double> dF_D = Teuchos::rcp( new SpatiallyFilteredFunction<double>( traction_x*boundaryRestriction,cylinderBoundary));
          double F_D = dF_D->integrate(solutionBackground->mesh());
          // double F_D = 1.0;

          // dragCoefficient = F_D;
          // 2/3 is the average inflow velocity
          dragCoefficient = F_D;// * (3.0/2.0);

          // compute force in y-direction
          TFunctionPtr<double> traction_y = Teuchos::rcp( new PreviousSolutionFunction<double>(solutionBackground, form.sigman_hat(2)));
          dF_D = Teuchos::rcp( new SpatiallyFilteredFunction<double>( traction_y*boundaryRestriction,cylinderBoundary));
          F_D = dF_D->integrate(solutionBackground->mesh());

          verticalForce = F_D;
        }
        cout << "Refinement: " << refIndex
          << " \tElements: " << mesh->numActiveElements()
          << " \tDOFs: " << mesh->numGlobalDofs()
          << " \tEnergy Error: " << energyError
          // << " \tL2 Error: " << l2Error
          << " \tSolve Time: " << solveTime
          << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
          << " \tIteration Count: " << iterationCount
          << " \tDrag Coefficient: " << dragCoefficient
          << " \ty-direction Force : " << verticalForce
          << " \tLambda: " << lambda
          << endl;
        dataFile << refIndex
          << " " << mesh->numActiveElements()
          << " " << mesh->numGlobalDofs()
          << " " << energyError
          // << " " << l2Error
          << " " << solveTime
          << " " << totalTimer->totalElapsedTime(true)
          << " " << iterationCount
          << " " << dragCoefficient
          << " " << verticalForce
          << " " << lambda
          << endl;
        if (refIndex == 0 && lambda == lambdaInitial)
        {
          energyErrorInitial = energyError;
        }
      }

      if (exportSolution)
        exporter->exportSolution(solutionBackground, refIndex);
        // exporter->exportSolution(solutionIncrement, refIndex);

      if (energyError < energyErrorInitial*energyTol && iterCount < maxNonlinearIterations-1 )
        break;

      if (refIndex != numRefs)
        form.refine();
        // refStrategy->refine();

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
