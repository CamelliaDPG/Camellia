/*
 *  NavierStokesStudy.cpp
 *
 *  Created by Nathan Roberts on 11/9/15.
 
 The idea here is simply to do the same thing as the old NavierStokesStudy, substituting the new
 NavierStokesVGPFormulation for the old formulation implementation.  These are intended to be identical
 mathematically, so any differences should be investigated...
 
 *
 */


#include "choice.hpp"
#include "mpi_choice.hpp"

#include "HConvergenceStudy.h"

#include "InnerProductScratchPad.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "CGSolver.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "NavierStokesFormulation.h"

#include "NavierStokesVGPFormulation.h"

#include "StokesVGPFormulation.h"

#include "MeshUtilities.h"

#include "DataIO.h"

#include "ParameterFunction.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
  int spaceDim = 2;
#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  int minPolyOrder = args.Input<int>("--minPolyOrder", "L^2 (field) minimum polynomial order",0);
  int maxPolyOrder = args.Input<int>("--maxPolyOrder", "L^2 (field) maximum polynomial order",1);
  int minLogElements = args.Input<int>("--minLogElements", "base 2 log of the minimum number of elements in one mesh direction", 0);
  int maxLogElements = args.Input<int>("--maxLogElements", "base 2 log of the maximum number of elements in one mesh direction", 4);
  double Re = args.Input<double>("--Re", "Reynolds number", 40);
//  bool outputStiffnessMatrix = args.Input<bool>("--writeFinalStiffnessToDisk", "write the final stiffness matrix to disk.", false);
  bool computeMaxConditionNumber = args.Input<bool>("--computeMaxConditionNumber", "compute the maximum Gram matrix condition number for final mesh.", false);
  int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
  double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 1e-12);
  string normChoice = args.Input<string>("--norm", "norm choice: graph, compliantGraph, stokesGraph, or stokesCompliantGraph", "graph");

  bool useCondensedSolve = args.Input<bool>("--useCondensedSolve", "use static condensation", true);

  double dt = args.Input<double>("--timeStep", "time step (0 for none)", 0);

  double zmcRho = args.Input<double>("--zmcRho", "zero-mean constraint rho (stabilization parameter)", -1);

//  string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
//  string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "");

  args.Process();

  int pToAdd = 2; // for optimal test function approximation
  bool computeRelativeErrors = true; // we'll say false when one of the exact solution components is 0
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);

  // parse args:
  bool useTriangles = false, useGraphNorm = false, useCompliantNorm = false, useStokesCompliantNorm = false, useStokesGraphNorm = false;

  if (normChoice=="graph")
  {
    useGraphNorm = true;
  }
  else if (normChoice=="compliantGraph")
  {
    useCompliantNorm = true;
  }
  else if (normChoice=="stokesGraph")
  {
    useStokesGraphNorm = true;
  }
  else if (normChoice=="stokesCompliantGraph")
  {
    useStokesCompliantNorm = true;
  }
  else
  {
    if (rank==0) cout << "unknown norm choice.  Exiting.\n";
    exit(-1);
  }

  bool artificialTimeStepping = (dt > 0);

  if (rank == 0)
  {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "norm = " << normChoice << endl;
  }

  // define Kovasznay domain:
  FieldContainer<double> quadPointsKovasznay(4,2);
  // domain from Cockburn Kanschat for Stokes:
//  quadPointsKovasznay(0,0) = -0.5; // x1
//  quadPointsKovasznay(0,1) =  0.0; // y1
//  quadPointsKovasznay(1,0) =  1.5;
//  quadPointsKovasznay(1,1) =  0.0;
//  quadPointsKovasznay(2,0) =  1.5;
//  quadPointsKovasznay(2,1) =  2.0;
//  quadPointsKovasznay(3,0) = -0.5;
//  quadPointsKovasznay(3,1) =  2.0;
  
  string formulationTypeStr = "vgp";

  FunctionPtr u1_exact, u2_exact, p_exact;

  int numCellsFineMesh = 20; // for computing a zero-mean pressure
  int H1OrderFineMesh = 5;

  StokesVGPFormulation stokesForm = StokesVGPFormulation::steadyFormulation(spaceDim, 1.0/Re, useEnrichedTraces);
  
  VarPtr u1_vgp = stokesForm.u(1);
  VarPtr u2_vgp = stokesForm.u(2);
  VarPtr sigma11_vgp = stokesForm.sigma(1,1);
  VarPtr sigma12_vgp = stokesForm.sigma(1,2);
  VarPtr sigma21_vgp = stokesForm.sigma(2,1);
  VarPtr sigma22_vgp = stokesForm.sigma(2,2);
  VarPtr p_vgp = stokesForm.p();
  
  VarPtr u1hat_vgp = stokesForm.u_hat(1);
  VarPtr u2hat_vgp = stokesForm.u_hat(2);

  VarPtr v1_vgp = stokesForm.v(1);
  VarPtr v2_vgp = stokesForm.v(2);
  
  StokesVGPFormulation stokesFormFineMesh = StokesVGPFormulation::steadyFormulation(spaceDim, 1.0/Re, useEnrichedTraces);
  MeshTopologyPtr fineMeshTopo = MeshFactory::rectilinearMeshTopology({2.0,2.0}, {numCellsFineMesh, numCellsFineMesh}, {-0.5,0.0});
  stokesFormFineMesh.initializeSolution(fineMeshTopo, H1OrderFineMesh-1);
  MeshPtr fineMesh = stokesFormFineMesh.solution()->mesh();
  NavierStokesFormulation::setKovasznay(Re, fineMesh, u1_exact, u2_exact, p_exact);
  
  FunctionPtr u_exact = Function::vectorize({u1_exact,u2_exact});
  FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunctionSteady(spaceDim, Re, u_exact, p_exact);
  
//  if (rank==0) cout << "forcingFunction: " << forcingFunction->displayString() << endl;
  
  Teuchos::RCP<ExactSolution<double>> exactSolution = Teuchos::rcp(new ExactSolution<double>());
  exactSolution->setSolutionFunction(u1_vgp, u1_exact);
  exactSolution->setSolutionFunction(u2_vgp, u2_exact);
  exactSolution->setSolutionFunction(p_vgp, p_exact);
  exactSolution->setSolutionFunction(sigma11_vgp, (1.0/Re) * u1_exact->dx());
  exactSolution->setSolutionFunction(sigma12_vgp, (1.0/Re) * u1_exact->dy());
  exactSolution->setSolutionFunction(sigma21_vgp, (1.0/Re) * u2_exact->dx());
  exactSolution->setSolutionFunction(sigma22_vgp, (1.0/Re) * u2_exact->dy());
  
  map< string, string > convergenceDataForMATLAB; // key: field file name

  for (int polyOrder = minPolyOrder; polyOrder <= maxPolyOrder; polyOrder++)
  {
    int H1Order = polyOrder + 1;

    int numCells1D = pow(2.0,minLogElements);

    if (rank==0)
    {
      cout << "L^2 order: " << polyOrder << endl;
      cout << "Re = " << Re << endl;
    }

    int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.

    vector< NavierStokesVGPFormulation > navierStokesForms;
    do
    {
      MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({2.0,2.0}, {numCells1D, numCells1D}, {-0.5,0.0});
      NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useEnrichedTraces, meshTopo,
                                                                                      H1Order-1, pToAdd);
      
      form.addInflowCondition(SpatialFilter::allSpace(), u_exact);
      form.setForcingFunction(forcingFunction);
      form.addZeroMeanPressureCondition();
      
      form.solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
      form.solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);

      form.solution()->setZeroMeanConstraintRho(zmcRho);
      form.solutionIncrement()->setZeroMeanConstraintRho(zmcRho);
      
//      form.solution()->setUseCondensedSolve(useCondensedSolve);
      form.solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
      
      FunctionPtr dt_inv;

      if (artificialTimeStepping)
      {
        //    // LHS gets u_inc / dt:
        BFPtr bf = form.bf();
        dt_inv = ParameterFunction::parameterFunction(1.0 / dt); //Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
        bf->addTerm(-dt_inv * u1_vgp, v1_vgp);
        bf->addTerm(-dt_inv * u2_vgp, v2_vgp);
        form.solution()->setIP( bf->graphNorm() ); // graph norm has changed...
        form.solutionIncrement()->setIP( bf->graphNorm() ); // graph norm has changed...
      }
      else
      {
        dt_inv = Function::zero();
      }

      navierStokesForms.push_back(form);
      if ( useCompliantNorm )
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option");
        // problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm(dt_inv));
      }
      else if (useStokesCompliantNorm)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option");
//        VGPStokesFormulation stokesForm(1.0); // pretend Re = 1 in the graph norm
//        problem.setIP(stokesForm.scaleCompliantGraphNorm());
      }
      else if (useStokesGraphNorm)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option");
//        VGPStokesFormulation stokesForm(1.0); // pretend Re = 1 in the graph norm
//        problem.setIP(stokesForm.graphNorm());
      }
      else if (! useGraphNorm )
      {
        // then use the naive:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option");
//        problem.setIP(problem.bf()->naiveNorm(spaceDim));
      }
      if (rank==0)
      {
        cout << numCells1D << " x " << numCells1D << ": " << form.solution()->mesh()->numGlobalDofs() << " dofs " << endl;
      }
      numCells1D *= 2;
    }
    while (pow(2.0,maxLogElements) >= numCells1D);
    
    bool neglectFluxesOnRHS = false; // right now, this is hard-coded into NavierStokesVGPFormulation
    RHSPtr rhsForSolve = navierStokesForms[0].rhs(forcingFunction, neglectFluxesOnRHS);
    
    // note that rhs and bilinearForm aren't really going to be right here, since they
    // involve a background flow which varies over the various problems...
    HConvergenceStudy study(exactSolution,
                            navierStokesForms[0].bf(),
                            rhsForSolve,
                            navierStokesForms[0].solution()->bc(),
                            navierStokesForms[0].bf()->graphNorm(),
                            minLogElements, maxLogElements,
                            H1Order, pToAdd, false, useTriangles, false);
    study.setReportRelativeErrors(computeRelativeErrors);
    study.setCubatureDegreeForExact(kovasznayCubatureEnrichment);

    vector< SolutionPtr > solutions;
    numCells1D = pow(2.0,minLogElements);
    for (auto form : navierStokesForms)
    {
      SolutionPtr solnIncrement = form.solutionIncrement();
      FunctionPtr u1_incr = Function::solution(u1_vgp, solnIncrement);
      FunctionPtr u2_incr = Function::solution(u2_vgp, solnIncrement);
      FunctionPtr sigma11_incr = Function::solution(sigma11_vgp, solnIncrement);
      FunctionPtr sigma12_incr = Function::solution(sigma12_vgp, solnIncrement);
      FunctionPtr sigma21_incr = Function::solution(sigma21_vgp, solnIncrement);
      FunctionPtr sigma22_incr = Function::solution(sigma22_vgp, solnIncrement);
      FunctionPtr p_incr = Function::solution(p_vgp, solnIncrement);

      FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
                            + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
                            + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;
      MeshPtr mesh = form.solution()->mesh();

//      if (rank==0) cout << "mesh->bilinearForm(): " << mesh->bilinearForm()->displayString() << endl;
//      if (rank==0) cout << "form.bf(): " << form.bf()->displayString() << endl;
//      
//      if (rank==0) cout << "RHS: " << solnIncrement->rhs()->linearTerm()->displayString() << endl;
      
      do
      {
        form.solveAndAccumulate(); // problem->iterate(useLineSearch, useCondensedSolve);

//        LinearTermPtr rhsLT = problem->backgroundFlow()->rhs()->linearTerm();
//        RieszRep rieszRep(problem->backgroundFlow()->mesh(), problem->backgroundFlow()->ip(), rhsLT);
//        rieszRep.computeRieszRep();
//        double costFunction = rieszRep.getNorm();
        double incr_norm = sqrt(l2_incr->integrate(mesh));

        if (rank==0)
        {
          cout << setprecision(6) << scientific;
          cout << "\x1B[2K"; // Erase the entire current line.
          cout << "\x1B[0E"; // Move to the beginning of the current line.
          cout << "Iteration: " << form.nonlinearIterationCount() << "; L^2(incr) = " << incr_norm;
          flush(cout);
//          cout << setprecision(6) << scientific;
//          cout << "Took " << weight << "-weighted step for " << numCells1D;
//          cout << " x " << numCells1D << " mesh: " << problem->iterationCount();
//          cout << setprecision(6) << fixed;
//          cout << " iterations; cost function " << costFunction << endl;
        }
      }
      while ((sqrt(l2_incr->integrate(mesh)) > minL2Increment ) && (form.nonlinearIterationCount() < maxIters));

      if (rank==0) cout << endl;

      solutions.push_back( form.solution() );

      // set the IP to the naive norm for clearer comparison with the best approximation energy error
//      problem->backgroundFlow()->setIP(problem->bf()->naiveNorm());

//      double energyError = problem->backgroundFlow()->energyErrorTotal();
//      if (rank==0) {
//        cout << setprecision(6) << fixed;
//        cout << numCells1D << " x " << numCells1D << ": " << problem->iterationCount();
//        cout << " iterations; actual energy error " << energyError << endl;
//      }
      numCells1D *= 2;
    }

    study.setSolutions(solutions);


    for (int i=0; i<=maxLogElements-minLogElements; i++)
    {
      SolutionPtr bestApproximation = study.bestApproximations()[i];

      // use solution's IP so that they're comparable
      IPPtr ip = navierStokesForms[i].solutionIncrement()->ip();
      LinearTermPtr rhsLT = navierStokesForms[i].rhs(forcingFunction, false)->linearTerm(); // false: don't exclude fluxes and traces
      RieszRep rieszRep(bestApproximation->mesh(), ip, rhsLT);
      rieszRep.computeRieszRep();

      double bestCostFunction = rieszRep.getNorm();
      if (rank==0)
        cout << "best energy error (measured according to the actual solution's test space IP): " << bestCostFunction << endl;
    }

    map< int, double > energyNormWeights;
    if (useCompliantNorm)
    {
      energyNormWeights[u1_vgp->ID()] = 1.0; // should be 1/h
      energyNormWeights[u2_vgp->ID()] = 1.0; // should be 1/h
      energyNormWeights[sigma11_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma12_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma21_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma22_vgp->ID()] = Re; // 1/mu
      if (Re < 1)   // assuming we're using the experimental small Re thing
      {
        energyNormWeights[p_vgp->ID()] = Re;
      }
      else
      {
        energyNormWeights[p_vgp->ID()] = 1.0;
      }
    }
    else
    {
      energyNormWeights[u1_vgp->ID()] = 1.0;
      energyNormWeights[u2_vgp->ID()] = 1.0;
      energyNormWeights[sigma11_vgp->ID()] = 1.0;
      energyNormWeights[sigma12_vgp->ID()] = 1.0;
      energyNormWeights[sigma21_vgp->ID()] = 1.0;
      energyNormWeights[sigma22_vgp->ID()] = 1.0;
      energyNormWeights[p_vgp->ID()] = 1.0;
    }
    vector<double> bestEnergy = study.weightedL2Error(energyNormWeights,true);
    vector<double> solnEnergy = study.weightedL2Error(energyNormWeights,false);

    map<int, double> velocityWeights;
    velocityWeights[u1_vgp->ID()] = 1.0;
    velocityWeights[u2_vgp->ID()] = 1.0;
    vector<double> bestVelocityError = study.weightedL2Error(velocityWeights,true);
    vector<double> solnVelocityError = study.weightedL2Error(velocityWeights,false);

    map<int, double> pressureWeight;
    pressureWeight[p_vgp->ID()] = 1.0;
    vector<double> bestPressureError = study.weightedL2Error(pressureWeight,true);
    vector<double> solnPressureError = study.weightedL2Error(pressureWeight,false);

    if (rank==0)
    {
      cout << setw(25);
      cout << "Solution Energy Error:" << setw(25) << "Best Energy Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++)
      {
        cout << setw(25) << solnEnergy[i] << setw(25) << bestEnergy[i] << endl;
      }
      cout << setw(25);
      cout << "Solution Velocity Error:" << setw(25) << "Best Velocity Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++)
      {
        cout << setw(25) << solnVelocityError[i] << setw(25) << bestVelocityError[i] << endl;
      }
      cout << setw(25);
      cout << "Solution Pressure Error:" << setw(25) << "Best Pressure Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++)
      {
        cout << setw(25) << solnPressureError[i] << setw(25) << bestPressureError[i] << endl;
      }

      vector< string > tableHeaders;
      vector< vector<double> > dataTable;
      vector< double > meshWidths;
      for (int i=minLogElements; i<=maxLogElements; i++)
      {
        double width = pow(2.0,i);
        meshWidths.push_back(width);
      }

      tableHeaders.push_back("mesh_width");
      dataTable.push_back(meshWidths);
      tableHeaders.push_back("soln_energy_error");
      dataTable.push_back(solnEnergy);
      tableHeaders.push_back("best_energy_error");
      dataTable.push_back(bestEnergy);

      tableHeaders.push_back("soln_velocity_error");
      dataTable.push_back(solnVelocityError);
      tableHeaders.push_back("best_velocity_error");
      dataTable.push_back(bestVelocityError);

      tableHeaders.push_back("soln_pressure_error");
      dataTable.push_back(solnPressureError);
      tableHeaders.push_back("best_pressure_error");
      dataTable.push_back(bestPressureError);

      ostringstream fileNameStream;
      fileNameStream << "nsStudy_Re" << Re << "k" << polyOrder << "_results.dat";

      DataIO::outputTableToFile(tableHeaders,dataTable,fileNameStream.str());
    }

    /*
     // corr. ID == -1 if there isn't one
     int NONE = -1;
     fieldIDs.clear();
     correspondingTraceIDs.clear();
     fileFriendlyNames.clear();
     fieldIDs.push_back(u1->ID());
     fileFriendlyNames.push_back("u1");
     correspondingTraceIDs.push_back(u1hat->ID());
     fieldIDs.push_back(u2->ID());
     fileFriendlyNames.push_back("u2");
     correspondingTraceIDs.push_back(u2hat->ID());
     fieldIDs.push_back(p->ID());
     fileFriendlyNames.push_back("pressure");
     correspondingTraceIDs.push_back(NONE);
     fieldIDs.push_back(sigma11->ID());
     fileFriendlyNames.push_back("sigma11");
     correspondingTraceIDs.push_back(NONE);
     fieldIDs.push_back(sigma12->ID());
     fileFriendlyNames.push_back("sigma12");
     correspondingTraceIDs.push_back(NONE);
     fieldIDs.push_back(sigma21->ID());
     fileFriendlyNames.push_back("sigma21");
     correspondingTraceIDs.push_back(NONE);
     fieldIDs.push_back(sigma22->ID());
     fileFriendlyNames.push_back("sigma22");
     correspondingTraceIDs.push_back(NONE);
     */
    
    if (rank == 0)
    {
      cout << study.TeXErrorRateTable();
      vector<int> primaryVariables = {u1_vgp->ID(), u2_vgp->ID(), p_vgp->ID()};
      vector<int> fieldIDs = {u1_vgp->ID(), u2_vgp->ID(), p_vgp->ID(),
        sigma11_vgp->ID(), sigma12_vgp->ID(), sigma21_vgp->ID(), sigma22_vgp->ID()};
      vector<int> traceIDs = {u1hat_vgp->ID(), u2hat_vgp->ID(), -1, -1, -1, -1, -1};
      vector<string> fieldFileNames = {"u1","u2","pressure","sigma11","sigma12","sigma21","sigma22"};
      cout << "******** Best Approximation comparison: ********\n";
      cout << study.TeXBestApproximationComparisonTable(primaryVariables);

      ostringstream filePathPrefix;
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_velpressure";
      study.TeXBestApproximationComparisonTable(primaryVariables,filePathPrefix.str());
      filePathPrefix.str("");
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_all";
      study.TeXBestApproximationComparisonTable(fieldIDs);

      for (int i=0; i<fieldIDs.size(); i++)
      {
        int fieldID = fieldIDs[i];
        int traceID = traceIDs[i];
        string fieldName = fieldFileNames[i];
        ostringstream filePathPrefix;
        filePathPrefix << "navierStokes/" << fieldName << "_p" << polyOrder;
        bool writeMATLABplotData = false;
        study.writeToFiles(filePathPrefix.str(),fieldID,traceID, writeMATLABplotData);
      }

      for (int i=0; i<primaryVariables.size(); i++)
      {
        string convData = study.convergenceDataMATLAB(primaryVariables[i], minPolyOrder);
        cout << convData;
        convergenceDataForMATLAB[fieldFileNames[i]] += convData;
      }

      filePathPrefix.str("");
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_numDofs";
      cout << study.TeXNumGlobalDofsTable();
    }
    if (computeMaxConditionNumber)
    {
      for (int i=minLogElements; i<=maxLogElements; i++)
      {
        SolutionPtr soln = study.getSolution(i);
        ostringstream fileNameStream;
        fileNameStream << "nsStudy_maxConditionIPMatrix_" << i << ".dat";
        IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(soln->ip().get()), false );
        bool jacobiScalingTrue = true;
        double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, soln->mesh(), jacobiScalingTrue, fileNameStream.str());
        if (rank==0)
        {
          cout << "max Gram matrix condition number estimate for logElements " << i << ": "  << maxConditionNumber << endl;
          cout << "putative worst-conditioned Gram matrix written to: " << fileNameStream.str() << "." << endl;
        }
      }
    }
  }
  if (rank==0)
  {
    ostringstream filePathPrefix;
    filePathPrefix << "navierStokes/" << formulationTypeStr << "_";
    for (map<string,string>::iterator convIt = convergenceDataForMATLAB.begin(); convIt != convergenceDataForMATLAB.end(); convIt++)
    {
      string fileName = convIt->first + ".m";
      string data = convIt->second;
      fileName = filePathPrefix.str() + fileName;
      ofstream fout(fileName.c_str());
      fout << data;
      fout.close();
    }
  }

}