//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  PoissonGoalOrientedDPGDriver.cpp
//  Driver for goal-oriented adaptive mesh refinement for Poisson's BVP
//  Camellia
//
//  Created by Brendan Keith, December 2017.

#include "math.h"

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
#include "H1ProjectionFormulation.h"
#include "PoissonFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "RieszRep.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include "MPIWrapper.h"
#include "GnuPlotUtil.h"

#include "CamelliaDebugUtility.h"

using namespace Camellia;

template <typename Scalar>
class DPGstarErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  int _cubatureDegreeEnrichment;
  std::map<GlobalIndexType, double> _jumpMap_total;
public:
  DPGstarErrorIndicator(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, std::map<GlobalIndexType, double> jumpMap_total, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
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
      double dualresidual = _dualSolnResidualFunction->integrate(cellID, _solution->mesh(), _cubatureDegreeEnrichment);
      _localErrorMeasures[cellID] = dualresidual + _jumpMap_total[cellID];
      // dualresidual += _jumpMap_total[cellID];
      // _localErrorMeasures[cellID] = sqrt(dualresidual);
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
class GoalOrientedErrorIndicator : public ErrorIndicator
{
  SolutionPtr _solution;
  FunctionPtr _dualSolnResidualFunction;
  std::map<GlobalIndexType, double> _jumpMap_total;
  int _cubatureDegreeEnrichment;
public:
  GoalOrientedErrorIndicator(SolutionPtr soln, FunctionPtr dualSolnResidualFunction, std::map<GlobalIndexType, double> jumpMap_total, int cubatureDegreeEnrichment) : ErrorIndicator(soln->mesh())
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
  string formulation = "Poisson";
  string problemChoice = "SquareDomain";
  string formulationChoice = "ULTRAWEAK";
  int spaceDim = 2;
  int numRefs = 3;
  int k = 2, delta_k = 2;
  string norm = "Graph";
  string errorIndicator = "Uniform";
  bool useConformingTraces = true;
  bool enrichTrial = false;
  bool liftedBC = false;
  bool jumpTermEdgeScaling = true;
  int liftChoice = 2;
  string solverChoice = "KLU";
  bool exportSolution = false;
  string outputDir = ".";
  string tag="";
  cmdp.setOption("formulation", &formulation, "Poisson");
  cmdp.setOption("problem", &problemChoice, "SquareDomain, RectangularDomain");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("formulationChoice", &formulationChoice, "ULTRAWEAK");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variables");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("norm", &norm, "norm: Graph, Naive, qopt");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy, Uniform, GoalOriented, DPGstar");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("enrichTrial", "classictrial", &enrichTrial, "use enriched u-variable");
  cmdp.setOption("liftedBC", "naive BC", &liftedBC, "lifted BC");
  cmdp.setOption("edgeScaling", "interiorScaling", &jumpTermEdgeScaling, "weight by edge length for jump terms");
  cmdp.setOption("liftChoice",&liftChoice,"choice of lifting function");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("outputDir", &outputDir, "output directory");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (commRank == 0)
  {
    // for now, just print out the error indicator choice
    cout << "Selected options:\n";
    cout << " - Error Indicator: " << errorIndicator << endl;
    cout << endl << endl;
  }
  
  
  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);



  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one    = Function::constant(1);
  FunctionPtr zero   = Function::zero();
  FunctionPtr x      = Function::xn(1);
  FunctionPtr y      = Function::yn(1);
  FunctionPtr n      = Function::normal();
  // FunctionPtr heavi1 = Function::heaviside(1);
  const static double PI  = 3.141592653589793238462;
  FunctionPtr sin_pix = Teuchos::rcp( new Sin_ax(PI) );
  FunctionPtr sin_piy = Teuchos::rcp( new Sin_ay(PI) );
  FunctionPtr cos_pix = Teuchos::rcp( new Cos_ax(PI) );
  FunctionPtr cos_piy = Teuchos::rcp( new Cos_ay(PI) );
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );



  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////

  //////////////////////  DECLARE MESH TOPOLOGY  ///////////////////////
  // MeshGeometryPtr spatialMeshGeom;
  MeshTopologyPtr spatialMeshTopo;
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap;
  double width, height;
  if (problemChoice == "SquareDomain")
  {
    double x0 = 0.0, y0 = 0.0;
    width = 1.0;
    height = 1.0;
    // int horizontalCells = 2, verticalCells = 2;
    int horizontalCells = 1, verticalCells = 1;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else if (problemChoice == "RectangularDomain")
  {
    double x0 = 0.0, y0 = 0.0;
    width = 4.0;
    height = 1.0;
    int horizontalCells = 8, verticalCells = 2;
    spatialMeshTopo =  MeshFactory::quadMeshTopology(width, height, horizontalCells, verticalCells,
                                                                     false, x0, y0);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  ///////////////////////  DECLARE BILINEAR FORM  //////////////////////
  PoissonFormulation form(spaceDim, useConformingTraces);

  BFPtr bf = form.bf();
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);


  //////////////////////  DECLARE TRIAL FUNCTIONS //////////////////////
  VarPtr u, sigma1, sigma2, u_hat, sigma_n_hat;
  u           = form.u();
  sigma1      = form.sigma(1);
  sigma2      = form.sigma(2);
  u_hat       = form.u_hat(); 
  sigma_n_hat = form.sigma_n_hat();

  map<int, int> trialOrderEnhancements;
  if (enrichTrial)
    trialOrderEnhancements[u->ID()] = 1;


  //////////////////////  DECLARE TEST FUNCTIONS ///////////////////////
  VarPtr v, tau;
  v   = form.v();
  tau = form.tau();


  ///////////////////////////  DECLARE MESH  ///////////////////////////
  vector<int> H1Order = {k + 1};
  int testEnrichment = delta_k;
  MeshPtr mesh = Teuchos::rcp( new Mesh(spatialMeshTopo, bf, H1Order, testEnrichment, trialOrderEnhancements) ) ;
  if (globalEdgeToCurveMap.size() > 0) // only necessary if geometry is curvilinear
  {
    spatialMeshTopo->initializeTransformationFunction(mesh);
  }


  ///////////////////////  DECLARE INNER PRODUCT ///////////////////////
  // weights for custom quasi-optimal norm
  map<int,double> trialWeights; // leave empty for unit weights (default)
  map<int,double> testL2Weights;
//    testL2Weights[v->ID()] = 1.0;
//    testL2Weights[tau->ID()] = 1.0 / epsilon;
  // trialWeights[u->ID()] = 1.0;
  // trialWeights[sigma->ID()] = 1.0;
  testL2Weights[v->ID()] = 0.0;
  testL2Weights[tau->ID()] = 1.0;
  // custom norm (again bc other way has a bug)
  IPPtr qopt_ip = Teuchos::rcp(new IP());
  qopt_ip->addTerm(v->grad()-tau);
  qopt_ip->addTerm(tau->div());
  qopt_ip->addTerm(0.1*v);
  // qopt_ip->addTerm(tau);

  map<string, IPPtr> poissonIPs;
  poissonIPs["Graph"] = bf->graphNorm();
  poissonIPs["Naive"] = bf->naiveNorm(spaceDim);
  // poissonIPs["qopt"] = qopt_ip;
  poissonIPs["qopt"] = bf->graphNorm(0.1);
  // poissonIPs["qopt"] = bf->graphNorm(trialWeights, testL2Weights);
  IPPtr ip = poissonIPs[norm];

  // ip->printInteractions();


  //////////////////////  DECLARE EXACT SOLUTIONS  /////////////////////
  FunctionPtr u_exact, v_exact;
  FunctionPtr xx = x/width, yy = y/height;
  // u_exact = zero;
  // u_exact = one;
  // u_exact = x * x + 2 * x * y;
  // u_exact = x * x * x * y + 2 * x * y * y;
  u_exact = sin_pix * sin_piy;
  // u_exact = xx * (1.0 - xx) * (xx/4.0 + (1.0 - 4.0*xx)*(1.0 - 4.0*xx) ) * yy * (1.0 - yy) * (yy/4.0 + (1.0 - 4.0*yy)*(1.0 - 4.0*yy) );

  xx = (width-x)/width;
  yy = (height-y)/height;
  // v_exact = zero;
  // v_exact = one;
  // v_exact = x * x + 2 * x * y;
  // v_exact = xx * (1.0 - xx) * (xx/4.0 + (1.0 - 4.0*xx)*(1.0 - 4.0*xx) ) * yy * (1.0 - yy) * (yy/4.0 + (1.0 - 4.0*yy)*(1.0 - 4.0*yy) );
  // v_exact = x * x * x * y + 2 * x * y * y;
  // v_exact = x * (1.0 - x) * y * (1.0 - y);
  v_exact = sin_pix * sin_piy;
  // v_exact = sin_pix * sin_piy + one;
  // v_exact = sin_pix * sin_piy + x*x;
  // v_exact = cos_pix * cos_piy;


  ////////////////////////////  DECLARE RHS  ///////////////////////////
  FunctionPtr f;
  f = u_exact->dx()->dx() + u_exact->dy()->dy();
  // RHSPtr rhs = form.rhs(one);
  RHSPtr rhs = form.rhs(f);


  ////////////////////  DECLARE BOUNDARY CONDITIONS ////////////////////
  BCPtr bc = BC::bc();
  if (problemChoice == "SquareDomain" || problemChoice == "RectangularDomain")
  {
    SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
    SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
    SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(width);
    SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(height);
    bc->addDirichlet(u_hat, x_equals_zero, u_exact);
    bc->addDirichlet(u_hat, y_equals_zero, u_exact);
    bc->addDirichlet(u_hat, x_equals_one, u_exact);
    bc->addDirichlet(u_hat, y_equals_one, u_exact);
    // bc->addDirichlet(sigma_n_hat, x_equals_zero, zero);
    // bc->addDirichlet(sigma_n_hat, y_equals_zero, zero);
    // bc->addDirichlet(sigma_n_hat, x_equals_one,  zero);
    // bc->addDirichlet(sigma_n_hat, y_equals_one,  zero);
  }
  else
  {
    cout << "ERROR: Problem type not currently supported. Returning null.\n";
    return Teuchos::null;
  }


  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);


  //////////////////////  DECLARE GOAL FUNCTIONAL //////////////////////
  LinearTermPtr g_functional;
  FunctionPtr g_u, g_sigma_x, g_sigma_y, g_sigma_hat, g_u_hat;
  g_u = v_exact->dx()->dx() + v_exact->dy()->dy(); // field type
  g_functional = g_u * u;


  // bool liftedBC = false;
  FunctionPtr v_bdry = v_exact;
  // FunctionPtr v_lift = v_exact;
  // if (liftedBC)
  // {
  //   // LIFTED Dirichlet boundary data
  //   if (liftChoice == 1)
  //     v_lift = one;
  //     // v_lift = x*x;
  //   else if (liftChoice == 2)
  //     v_lift = 1 - x * (1 - x) * y * (1 - y);
  //   // else if (liftChoice == 3)
  //   //   v_lift = 1 - x * (1 - x*x) * y * (1 - y*y);
  //   // else if (liftChoice == 4)
  //   //   v_lift = 1 - x * (1 - x) * y * (1 - y) * x * x * x * x;
  //   else
  //   {
  //     cout << "ERROR: not a supported lift.\n";
  //     return Teuchos::null;
  //   }
  //   bool overrideTypeCheck = true;
  //   g_functional->addTerm( v_lift->dx() * sigma->x(), overrideTypeCheck); // add flux type to field type
  //   g_functional->addTerm( v_lift->dy() * sigma->y(), overrideTypeCheck); // add flux type to field type
  //   g_sigma_hat = zero;
  // }
  // else
  // {
    FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
    g_sigma_hat = v_exact * boundaryRestriction; // Dirichlet boundary data
    g_u_hat = v_exact->grad()*n * boundaryRestriction; // Dirichlet boundary data
    bool overrideTypeCheck = true;
    g_functional->addTerm( g_sigma_hat * sigma_n_hat, overrideTypeCheck); // add flux type to field type
    // g_functional->addTerm( g_u_hat * u_hat, overrideTypeCheck); // add flux type to field type
  // }

  // g_u = one - Function::heaviside(1);
  // g_functional = g_u * u;
  soln->setGoalOrientedRHS(g_functional);

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
#ifdef HAVE_AMESOS_MUMPS
  solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif

  ostringstream solnName;
  solnName << "GoalOrientedDPG" << "_" << norm << "_k" << k << "_" << solverChoice;
  if (tag != "")
    solnName << "_" << tag;
  solnName << "_" << formulation;

  string dataFileLocation;
  if (exportSolution)
    dataFileLocation = outputDir+"/"+solnName.str()+"/"+solnName.str()+".txt";
  else
    dataFileLocation = outputDir+"/"+solnName.str()+".txt";  

  ofstream dataFile(dataFileLocation);
  dataFile << setw(4)  << " Ref"
           << setw(11) << "Elements"
           << setw(10) << "DOFs"
           << setw(16) << "DPGresidual"
           << setw(9)  << "Rate"
           << setw(13) << "DPGerror"
           << setw(9) << "Rate"
           << setw(13) << "superconv"
           << setw(9) << "Rate"
           << setw(16) << "DPG*residual"
           << setw(9)  << "Rate"
           << setw(13) << "DPG*error"
           << setw(9) << "Rate"
           << setw(13) << "superconv"
           << setw(9) << "Rate"
           // << setw(12) << "Solvetime"
           // << setw(11) << "Elapsed"
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  Teuchos::RCP<HDF5Exporter> functionExporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));
  string exporterName = "_DualSolutions";
  exporterName = solnName.str() + exporterName;
  functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));

  SolverPtr solver = solvers[solverChoice];
  double energyErrorPrvs, solnErrorPrvs, solnErrorL2Prvs, dualSolnErrorPrvs, dualSolnErrorL2Prvs, dualSolnResidualPrvs, jump_v_Prvs, jump_tau_Prvs, outputErrorPrvs, numDofsPrvs;
  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
    solverTime->start(true);

    soln->solve(solver);
    double solveTime = solverTime->stop();

    Camellia::FieldContainer<GlobalIndexType> bcGlobalIndicesFC;
    Camellia::FieldContainer<double> bcGlobalValuesFC;
      
    mesh->boundary().bcsToImpose(bcGlobalIndicesFC,bcGlobalValuesFC,*soln->bc(), soln->getDofInterpreter().get());

    set<GlobalIndexType> uniqueIDs;
    for (int i=0; i<bcGlobalIndicesFC.size(); i++)
    {
      uniqueIDs.insert(bcGlobalIndicesFC[i]);
    }
    int numBCDOFs = uniqueIDs.size();

    // cout << "numBCDOFS = " << numBCDOFs << endl;
    double numDofs = mesh->numGlobalDofs() - mesh->numFieldDofs() - numBCDOFs;


    // compute error rep function / influence function
    bool excludeBoundaryTerms = false;
    LinearTermPtr residual = rhs()->linearTerm() - bf->testFunctional(soln,excludeBoundaryTerms);
    LinearTermPtr influence = bf->testFunctional(soln,excludeBoundaryTerms,INFLUENCE_SOLUTION);
    RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
    RieszRepPtr dualSoln = Teuchos::rcp(new RieszRep(mesh, ip, influence));
    rieszResidual->computeRieszRep();
    dualSoln->computeRieszRep();

    // extract the test functions
    FunctionPtr psi_v, psi_tau, dualSoln_v, dualSoln_tau;
    psi_v =  Teuchos::rcp( new RepFunction<double>(form.v(), rieszResidual) );
    psi_tau =  Teuchos::rcp( new RepFunction<double>(form.tau(), rieszResidual) );
    dualSoln_v =  Teuchos::rcp( new RepFunction<double>(form.v(), dualSoln) );
    dualSoln_tau =  Teuchos::rcp( new RepFunction<double>(form.tau(), dualSoln) );
    // if (liftedBC)
    // {
    //   dualSoln_v = dualSoln_v + v_lift;
    //   // dualSoln_tau = dualSoln_tau - v_lift->grad();
    // }

    // compute error in DPG solution
    FunctionPtr soln_u = Function::solution(u, soln);
    FunctionPtr soln_sigma = Function::vectorize(Function::solution(sigma1, soln), Function::solution(sigma2, soln));
    FunctionPtr e_u   = u_exact - soln_u;
    FunctionPtr e_sigma_x = u_exact->dx() - soln_sigma->x();
    FunctionPtr e_sigma_y = u_exact->dy() - soln_sigma->y();
    FunctionPtr solnErrorFunction = e_u*e_u + e_sigma_x*e_sigma_x + e_sigma_y*e_sigma_y;
    double solnError = sqrt(solnErrorFunction->l1norm(mesh));
    double solnErrorL2 = e_u->l2norm(mesh);
    double energyError = soln->energyErrorTotal();

    // compute error in DPG* solution
    FunctionPtr e_v   = v_exact - dualSoln_v;
    FunctionPtr e_v_dx   = v_exact->dx() - dualSoln_v->dx();
    FunctionPtr e_v_dy   = v_exact->dy() - dualSoln_v->dy();
    FunctionPtr e_tau_x = v_exact->dx() - dualSoln_tau->x();
    FunctionPtr e_tau_y = v_exact->dy() - dualSoln_tau->y();
    FunctionPtr e_tau_div = v_exact->dx()->dx() + v_exact->dy()->dy() - dualSoln_tau->div();
    FunctionPtr dualSolnErrorFunction = e_v_dx*e_v_dx + e_v_dy*e_v_dy + e_tau_x*e_tau_x + e_tau_y*e_tau_y + e_tau_div*e_tau_div;
    // FunctionPtr dualSolnErrorFunction = e_v*e_v + e_v_dx*e_v_dx + e_v_dy*e_v_dy + e_tau_x*e_tau_x + e_tau_y*e_tau_y + e_tau_div*e_tau_div;
    int cubatureDegreeEnrichment = delta_k+1;
    double dualSolnError = sqrt(dualSolnErrorFunction->l1norm(mesh,cubatureDegreeEnrichment));
    double dualSolnErrorL2 = e_v->l2norm(mesh,cubatureDegreeEnrichment);

    // compute residual in DPG* solution
    FunctionPtr res1 = dualSoln_tau->div() - g_u;
    FunctionPtr res2 = dualSoln_v->dx() - dualSoln_tau->x();  
    FunctionPtr res3 = dualSoln_v->dy() - dualSoln_tau->y();
    FunctionPtr dualSolnResFxn = res1*res1 + res2*res2 + res3*res3;

    map<int, FunctionPtr > opDualSoln = bf()->applyAdjointOperatorDPGstar(dualSoln);

    FunctionPtr res1_2 = opDualSoln[u->ID()] - g_u;
    FunctionPtr res2_2 = Function::vectorize(opDualSoln[sigma1->ID()], opDualSoln[sigma2->ID()]);

    FunctionPtr dualSolnResidualFunction = res1_2*res1_2 + res2_2*res2_2;

    bool weightBySideMeasure = false;
    if (jumpTermEdgeScaling == true)
      weightBySideMeasure = true;
    int cubatureEnrichmentDegree = delta_k;
    FunctionPtr v_minus_BC = dualSoln_v - g_sigma_hat;
    // FunctionPtr Dt_v_minus_BC = dualSoln_v-> - g_sigma_hat;
    FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
    FunctionPtr tauDotNml_minus_BC = dualSoln_tau * n;
    tauDotNml_minus_BC = (1.0 - boundaryRestriction) * tauDotNml_minus_BC;
    std::map<GlobalIndexType, double> jumpMap_v = v_minus_BC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureEnrichmentDegree);
    std::map<GlobalIndexType, double> jumpMap_gradv;
    if (weightBySideMeasure)
    {
      FunctionPtr gradv = dualSoln_v->dx()*n->y() - dualSoln_v->dy()*n->x();
      FunctionPtr gradv_BC = v_bdry->dx()*n->y() - v_bdry->dy()*n->x();
      gradv_BC = gradv_BC*boundaryRestriction;
      FunctionPtr gradv_minus_BC = gradv - gradv_BC;
      jumpMap_gradv = gradv_minus_BC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureEnrichmentDegree, Function::SUM);
    }
    std::map<GlobalIndexType, double> jumpMap_tau = tauDotNml_minus_BC->squaredL2NormOfJumps(mesh, weightBySideMeasure, cubatureEnrichmentDegree, Function::SUM);

    std::map<GlobalIndexType, double> jumpMap_total;
    const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
    double jump_v=0.0;
    double jump_tau=0.0;
    for (auto cellID: myCellIDs)
    {
      double jump_v_tmp, jump_tau_tmp;
      if (weightBySideMeasure)
      {
        jump_v_tmp = jumpMap_v[cellID] + jumpMap_gradv[cellID];
        jump_tau_tmp = jumpMap_tau[cellID];
      }
      else
      {
        double vol = mesh->getCellMeasure(cellID);
        double h = pow(vol, 1.0 / spaceDim);
        jump_v_tmp = pow(h, -1.0)*jumpMap_v[cellID];
        jump_tau_tmp = h*jumpMap_tau[cellID];
      }
      jump_v += jump_v_tmp;
      jump_tau += jump_tau_tmp;
      jumpMap_total[cellID] = jump_v_tmp + jump_tau_tmp;
    }
    jump_v = MPIWrapper::sum(mesh->Comm(), jump_v);
    jump_tau = MPIWrapper::sum(mesh->Comm(), jump_tau);

    // double dualSolnResidual = sqrt(dualSolnResidualFunction->l1norm(mesh, cubatureDegreeEnrichment));
    double dualSolnResidual = sqrt(dualSolnResidualFunction->l1norm(mesh, cubatureDegreeEnrichment) + jump_v + jump_tau);

    jump_v = sqrt(jump_v);
    jump_tau = sqrt(jump_tau);

    vector<FunctionPtr> functionsToExport = {psi_v, psi_tau, dualSoln_v, dualSoln_tau, dualSolnResidualFunction, dualSolnResFxn};
    vector<string> functionsToExportNames = {"psi_v", "psi_tau", "dual_v", "dual_tau", "dualSolnResidualFunction", "dualSolnResFxn"};

    // compute error in output (QOI)
    FunctionPtr outputError_function = g_u * (u_exact - soln_u);
    double outputError = abs(outputError_function->integrate(mesh, cubatureDegreeEnrichment));

    // compute rates
    double solnErrorRate = 0;
    double solnErrorL2Rate = 0;
    double energyRate = 0;
    double dualSolnErrorRate = 0;
    double dualSolnErrorL2Rate = 0;
    double dualSolnResidualRate = 0;
    double jump_v_Rate = 0;
    double jump_tau_Rate = 0;
    double outputErrorRate = 0;
    if (refIndex != 0)
    {
      double denom = log(numDofsPrvs/numDofs);
      solnErrorRate =-spaceDim*log(solnErrorPrvs/solnError)/denom;
      solnErrorL2Rate =-spaceDim*log(solnErrorL2Prvs/solnErrorL2)/denom;
      energyRate =-spaceDim*log(energyErrorPrvs/energyError)/denom;
      dualSolnErrorRate =-spaceDim*log(dualSolnErrorPrvs/dualSolnError)/denom;
      dualSolnErrorL2Rate =-spaceDim*log(dualSolnErrorL2Prvs/dualSolnErrorL2)/denom;
      dualSolnResidualRate =-spaceDim*log(dualSolnResidualPrvs/dualSolnResidual)/denom;
      jump_v_Rate =-spaceDim*log(jump_v_Prvs/jump_v)/denom;
      jump_tau_Rate =-spaceDim*log(jump_tau_Prvs/jump_tau)/denom;
      outputErrorRate =-spaceDim*log(outputErrorPrvs/outputError)/denom;
    }

    if (commRank == 0)
    {
      cout << setprecision(8) 
        << " \n\nRefinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << numDofs
        << setprecision(4)
        << " \tSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        << endl;
      cout << setprecision(4) 
        << " \nDPG Residual:        " << energyError
        << " \tRate: " << energyRate
        << " \nDPG Error (total):   " << solnError
        << " \tRate: " << solnErrorRate
        << " \nDPG Error (u-comp):  " << solnErrorL2
        << " \tRate: " << solnErrorL2Rate
        << endl;
      cout << setprecision(4) 
        << " \nDPG* Residual:       " << dualSolnResidual
        << " \tRate: " << dualSolnResidualRate
        << " \nDPG* jump in v:      " << jump_v
        << " \tRate: " << jump_v_Rate
        << " \nDPG* jump in tau:    " << jump_tau
        << " \tRate: " << jump_tau_Rate
        << " \nDPG* Error: (total): " << dualSolnError
        << " \tRate: " << dualSolnErrorRate
        << " \nDPG* Error (v-comp): " << dualSolnErrorL2
        << " \tRate: " << dualSolnErrorL2Rate
        << endl;
      cout << setprecision(4) 
        << " \nQOI Error:          " << outputError
        << " \tRate: " << outputErrorRate
        << endl;
      dataFile << setprecision(8)
        << setw(4) << refIndex
        << setw(3) << " "
        << setw(6) << mesh->numActiveElements()
        << setw(3) << " "
        << setw(9) << numDofs
        << setprecision(4)
        << setw(7) << " "
        << setw(9) << energyError
        << setw(9) << energyRate
        << setw(4) << " "
        << setw(9) << solnError
        << setw(9) << solnErrorRate
        << setw(4) << " "
        << setw(9) << solnErrorL2
        << setw(9) << solnErrorL2Rate
        << setw(7) << " "
        << setw(9) << dualSolnResidual
        << setw(9) << dualSolnResidualRate
        << setw(4) << " "
        << setw(9) << dualSolnError
        << setw(9) << dualSolnErrorRate
        << setw(4) << " "
        << setw(9) << dualSolnErrorL2
        << setw(9) << dualSolnErrorL2Rate
        // << setw(4) << " "
        // << setw(8) << solveTime
        // << setw(3) << " "
        // << setw(8) << totalTimer->totalElapsedTime(true)
        // << " " << dualSolnError
        // << " " << dualSolnErrorRate
        << endl;
    }

    solnErrorPrvs = solnError;
    solnErrorL2Prvs = solnErrorL2;
    energyErrorPrvs = energyError;
    dualSolnErrorPrvs = dualSolnError;
    dualSolnErrorL2Prvs = dualSolnErrorL2;
    dualSolnResidualPrvs = dualSolnResidual;
    jump_v_Prvs = jump_v;
    jump_tau_Prvs = jump_tau;
    outputErrorPrvs = outputError;
    numDofsPrvs = numDofs;

    if (exportSolution)
    {
      exporter->exportSolution(soln, refIndex);
      int numLinearPointsPlotting = max(k,15);
      functionExporter->exportFunction(functionsToExport, functionsToExportNames, refIndex, numLinearPointsPlotting);

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
      double energyThreshold = 0.5;
      RefinementStrategyPtr refStrategy;
      if (errorIndicator == "Uniform")
      {
        energyThreshold = 0;
        refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
      }
      else if (errorIndicator == "Energy")
      {
        refStrategy = Teuchos::rcp( new RefinementStrategy(ErrorIndicator::energyErrorIndicator(soln), energyThreshold) );
      }
      else if (errorIndicator == "GoalOriented")
      {
        double refThreshold = 0.25;
        // int cubatureDegreeEnrichment = delta_k;
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new GoalOrientedErrorIndicator<double>(soln, dualSolnResidualFunction, jumpMap_total, cubatureDegreeEnrichment) );
        refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
      }
      else if (errorIndicator == "DPGstar")
      {
        double refThreshold = 0.25;
        // int cubatureDegreeEnrichment = delta_k;
        ErrorIndicatorPtr errorIndicator = Teuchos::rcp( new DPGstarErrorIndicator<double>(soln, dualSolnResidualFunction, jumpMap_total, cubatureDegreeEnrichment) );
        refStrategy = Teuchos::rcp( new TRefinementStrategy<double>(errorIndicator, refThreshold) );
      } 
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized refinement strategy");
      }

      refStrategy->refine();
    } 

  }

  // dataFile.close();
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "\n\nTotal time = " << totalTime << endl;

  return 0;
}
