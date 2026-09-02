//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  AcousticsFormulationDriver.cpp
//  Driver for Acoustics with and without impedance BCs
//  Camellia
//
//  Created by Brendan Keith, November 2018.

#include "math.h"

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
#include "ErrorIndicator.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "RefinementStrategy.h"

#include "AcousticsFormulation.h"
// #include "PoissonFormulation.h"
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
  string formulation = "Acoustics";
  string problemChoice = "SquareDomain";
  string formulationChoice = "ULTRAWEAK";
  int spaceDim = 2;
  int numRefs = 3;
  int k = 2, delta_k = 2;
  double omega = 0.8;
  string norm = "Graph";
  string errorIndicator = "Uniform";
  bool useConformingTraces = true;
  bool impedance = false;
  bool enrichTrial = false;
  string solverChoice = "KLU";
  bool exportSolution = false;
  string outputDir = ".";
  string tag="";
  // cmdp.setOption("formulation", &formulation, "Acoustics");
  // cmdp.setOption("problem", &problemChoice, "SquareDomain, RectangularDomain");
  // cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  // cmdp.setOption("formulationChoice", &formulationChoice, "ULTRAWEAK");
  cmdp.setOption("omega", &omega, "omega");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variables");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("norm", &norm, "norm: Graph, Naive, qopt");
  cmdp.setOption("errorIndicator", &errorIndicator, "Energy, Uniform");
  // cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLUDist, MUMPS");
  cmdp.setOption("impedance", "hardBCs", &impedance, "impedance BCs");
  cmdp.setOption("enrichTrial", "standardTrial", &enrichTrial, "enrich pressure variable field by one poly order");
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
  AcousticsFormulation form(spaceDim, useConformingTraces, AcousticsFormulation::ULTRAWEAK, omega);

  BFPtr bf = form.bf();
  bf->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);


  //////////////////////  DECLARE TRIAL FUNCTIONS //////////////////////
  VarPtr p1,p2,u1,u2, p1_hat,p2_hat,u1_n_hat,u2_n_hat;
  p1       = form.p(1);
  p2       = form.p(2);
  u1       = form.u(1);
  u2       = form.u(2);
  p1_hat   = form.p_hat(1); 
  p2_hat   = form.p_hat(2); 
  u1_n_hat = form.u_n_hat(1);
  u2_n_hat = form.u_n_hat(2);

  map<int, int> trialOrderEnhancements;
  if (enrichTrial)
  {
    trialOrderEnhancements[p1->ID()] = 1;
    trialOrderEnhancements[p2->ID()] = 1;
  }


  //////////////////////  DECLARE TEST FUNCTIONS ///////////////////////
  VarPtr q1,q2,v1,v2;
  q1  = form.q(1);
  q2  = form.q(2);
  v1  = form.v(1);
  v2  = form.v(2);


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
  // IPPtr qopt_ip = Teuchos::rcp(new IP());
  // qopt_ip->addTerm(v->grad()-tau);
  // qopt_ip->addTerm(tau->div());
  // qopt_ip->addTerm(0.1*v);
  // qopt_ip->addTerm(tau);

  map<string, IPPtr> poissonIPs;
  poissonIPs["Graph"] = bf->graphNorm();
  poissonIPs["Naive"] = bf->naiveNorm(spaceDim);
  // poissonIPs["qopt"] = bf->graphNorm(0.1);
  IPPtr ip = poissonIPs[norm];

  // ip->printInteractions();


  //////////////////////  DECLARE EXACT SOLUTIONS  /////////////////////
  FunctionPtr p1_exact, p2_exact;
  FunctionPtr xx = x/width, yy = y/height;
  // p1_exact = zero;
  // p1_exact = one;
  // p1_exact = x * x + 2 * x * y;
  // p1_exact = x * x * x * y + 2 * x * y * y;
  p1_exact = sin_pix * sin_piy;
  // p1_exact = xx * (1.0 - xx) * (xx/4.0 + (1.0 - 4.0*xx)*(1.0 - 4.0*xx) ) * yy * (1.0 - yy) * (yy/4.0 + (1.0 - 4.0*yy)*(1.0 - 4.0*yy) );
  p2_exact = one;
  // p2_exact = sin_pix * sin_piy;


  ////////////////////////////  DECLARE RHS  ///////////////////////////
  FunctionPtr f1,f2;
  f1 = (1.0/omega) * p1_exact->dx()->dx() + (1.0/omega) * p1_exact->dy()->dy() + omega * p1_exact;
  f2 = (1.0/omega) * p2_exact->dx()->dx() + (1.0/omega) * p2_exact->dy()->dy() + omega * p2_exact;
  RHSPtr rhs = form.rhs();
  rhs->addTerm(f1 * q1);
  rhs->addTerm(f2 * q2);


  ////////////////////  DECLARE BOUNDARY CONDITIONS ////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(width);
  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(height);
  if (impedance == false)
  {
    if (problemChoice == "SquareDomain" || problemChoice == "RectangularDomain")
    {
      SpatialFilterPtr boundary = x_equals_zero | y_equals_zero | x_equals_one | y_equals_one;
      bc->addDirichlet(p1_hat, boundary, p1_exact);
      bc->addDirichlet(p2_hat, boundary, p2_exact);
    }
    else
    {
      cout << "ERROR: Problem type not currently supported. Returning null.\n";
      return Teuchos::null;
    }
  }
  else
  {
    if (problemChoice == "SquareDomain" || problemChoice == "RectangularDomain")
    {
      // SpatialFilterPtr boundary = SpatialFilter::allSpace();
      SpatialFilterPtr boundary = x_equals_zero | y_equals_zero | x_equals_one | y_equals_one;
      bc->addDirichlet(u1_n_hat, boundary, zero);
      bc->addDirichlet(u2_n_hat, boundary, zero);

      FunctionPtr g1,g2;
      g1 = p1_exact + (1.0/omega) * p2_exact->grad() * n;
      g2 = p2_exact - (1.0/omega) * p1_exact->grad() * n;
      form.addImpedanceBCs(boundary, g1, g2);
      // form.addImpedanceBCs(boundary);
    }
    else
    {
      cout << "ERROR: Problem type not currently supported. Returning null.\n";
      return Teuchos::null;
    }
  }

  /////////////////////  DECLARE SOLUTION POINTERS /////////////////////
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);

  // if (impedance == true)
  // {
  //   SpatialFilterPtr boundary = y_equals_one;
  //   // SpatialFilterPtr boundary = SpatialFilter::allSpace();
  //   FunctionPtr g1,g2;
  //   g1 = p1_exact + (1.0/omega) * p2_exact->grad() * n;
  //   g2 = p2_exact - (1.0/omega) * p1_exact->grad() * n;
  //   form.addImpedanceBCs(soln, boundary, g1, g2);
  //   // form.addImpedanceBCs(soln, boundary);
  // }

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
  solnName << "Acoustics" << "_" << norm << "_k" << k << "_" << solverChoice;
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
           // << setw(13) << "superconv"
           // << setw(9) << "Rate"
           << setw(12) << "Solvetime"
           << setw(11) << "Elapsed"
           << endl;

  Teuchos::RCP<HDF5Exporter> exporter;
  Teuchos::RCP<HDF5Exporter> functionExporter;
  exporter = Teuchos::rcp(new HDF5Exporter(mesh,solnName.str(), outputDir));
  // string exporterName = "_DualSolutions";
  // exporterName = solnName.str() + exporterName;
  // functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));

  SolverPtr solver = solvers[solverChoice];
  double energyErrorPrvs, solnErrorPrvs, solnErrorL2Prvs, numDofsPrvs;
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
    const bool overrideMeshCheck = false; // testFunctional() default for third argument
    const int solutionOrdinal = 1; // solution corresponding to second RHS
    LinearTermPtr residual = rhs()->linearTerm() - bf->testFunctional(soln,excludeBoundaryTerms);
    RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
    rieszResidual->computeRieszRep();

    // compute error in DPG solution
    FunctionPtr soln_p1 = Function::solution(p1, soln);
    FunctionPtr soln_p2 = Function::solution(p2, soln);
    FunctionPtr soln_u1 = Function::solution(u1, soln);
    FunctionPtr soln_u2 = Function::solution(u2, soln);
    FunctionPtr e_p1   = p1_exact - soln_p1;
    FunctionPtr e_p2   = p2_exact - soln_p2;
    FunctionPtr e_u1_x = p2_exact->dx() + omega * soln_u1->x();
    FunctionPtr e_u2_x = p1_exact->dx() - omega * soln_u2->x();
    FunctionPtr e_u1_y = p2_exact->dy() + omega * soln_u1->y();
    FunctionPtr e_u2_y = p1_exact->dy() - omega * soln_u2->y();
    FunctionPtr solnErrorFunction = e_p1*e_p1 + e_p2*e_p2 + e_u1_x*e_u1_x + e_u1_y*e_u1_y + e_u2_x*e_u2_x + e_u2_y*e_u2_y;
    FunctionPtr solnErrorL2Function = e_p1*e_p1 + e_p2*e_p2;
    double solnError = sqrt(solnErrorFunction->l1norm(mesh));
    double solnErrorL2 = sqrt(solnErrorL2Function->l1norm(mesh));
    // double solnErrorL2 = e_p1->l2norm(mesh);
    double energyError = soln->energyErrorTotal();

    // compute rates
    double solnErrorRate = 0;
    double solnErrorL2Rate = 0;
    double energyRate = 0;
    if (refIndex != 0)
    {
      double denom = log(numDofsPrvs/numDofs);
      solnErrorRate =-spaceDim*log(solnErrorPrvs/solnError)/denom;
      solnErrorL2Rate =-spaceDim*log(solnErrorL2Prvs/solnErrorL2)/denom;
      energyRate =-spaceDim*log(energyErrorPrvs/energyError)/denom;
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
        << " \nDPG Error (pressure-comp):  " << solnErrorL2
        << " \tRate: " << solnErrorL2Rate
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
        << setw(4) << " "
        << setw(8) << solveTime
        << setw(3) << " "
        << setw(8) << totalTimer->totalElapsedTime(true)
        << endl;
    }

    solnErrorPrvs = solnError;
    solnErrorL2Prvs = solnErrorL2;
    energyErrorPrvs = energyError;
    numDofsPrvs = numDofs;

    if (exportSolution)
    {
      exporter->exportSolution(soln, refIndex);
      int numLinearPointsPlotting = max(k,15);
      // functionExporter->exportFunction(functionsToExport, functionsToExportNames, refIndex, numLinearPointsPlotting);

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
