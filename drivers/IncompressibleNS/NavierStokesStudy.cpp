/*
 *  NavierStokesStudy.cpp
 *
 *  Created by Nathan Roberts on 11/15/12.
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

#include "MeshUtilities.h"

#include "DataIO.h"

#include "ParameterFunction.h"

using namespace std;

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
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
  bool longDoubleGramInversion = args.Input<bool>("--longDoubleGramInversion", "use long double Cholesky factorization for Gram matrix", false);
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
  bool useLineSearch = false;
  bool computeRelativeErrors = true; // we'll say false when one of the exact solution components is 0
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::basisFactory()->setUseEnrichedTraces(useEnrichedTraces);
  
  // parse args:
  bool useTriangles = false, useGraphNorm = false, useCompliantNorm = false, useStokesCompliantNorm = false, useStokesGraphNorm = false;
  
  if (normChoice=="graph") {
    useGraphNorm = true;
  } else if (normChoice=="compliantGraph") {
    useCompliantNorm = true;
  } else if (normChoice=="stokesGraph") {
    useStokesGraphNorm = true;
  } else if (normChoice=="stokesCompliantGraph") {
    useStokesCompliantNorm = true;
  } else {
    if (rank==0) cout << "unknown norm choice.  Exiting.\n";
    exit(-1);
  }
  
  bool artificialTimeStepping = (dt > 0);
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "norm = " << normChoice << endl;
    cout << "longDoubleGramInversion = "  << (longDoubleGramInversion ? "true" : "false") << "\n";
  }
  
  // define Kovasznay domain:
  FieldContainer<double> quadPointsKovasznay(4,2);
  // domain from Cockburn Kanschat for Stokes:
  quadPointsKovasznay(0,0) = -0.5; // x1
  quadPointsKovasznay(0,1) =  0.0; // y1
  quadPointsKovasznay(1,0) =  1.5;
  quadPointsKovasznay(1,1) =  0.0;
  quadPointsKovasznay(2,0) =  1.5;
  quadPointsKovasznay(2,1) =  2.0;
  quadPointsKovasznay(3,0) = -0.5;
  quadPointsKovasznay(3,1) =  2.0;
  
  // Domain from Evans Hughes for Navier-Stokes:
//  quadPointsKovasznay(0,0) =  0.0; // x1
//  quadPointsKovasznay(0,1) = -0.5; // y1
//  quadPointsKovasznay(1,0) =  1.0;
//  quadPointsKovasznay(1,1) = -0.5;
//  quadPointsKovasznay(2,0) =  1.0;
//  quadPointsKovasznay(2,1) =  0.5;
//  quadPointsKovasznay(3,0) =  0.0;
//  quadPointsKovasznay(3,1) =  0.5;

//  double Re = 10.0;  // Cockburn Kanschat Stokes
//  double Re = 40.0; // Evans Hughes Navier-Stokes
//  double Re = 1000.0;
  
  string formulationTypeStr = "vgp";
  
  FunctionPtr u1_exact, u2_exact, p_exact;
  
  int numCellsFineMesh = 20; // for computing a zero-mean pressure
  int H1OrderFineMesh = 5;

  FunctionPtr zero = Function::zero();
  VGPNavierStokesProblem zeroProblem = VGPNavierStokesProblem(Re, quadPointsKovasznay,
                                                              numCellsFineMesh, numCellsFineMesh,
                                                              H1OrderFineMesh, pToAdd,
                                                              zero, zero, zero, useCompliantNorm || useStokesCompliantNorm);
  
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1_vgp = varFactory.fieldVar(VGP_U1_S);
  VarPtr u2_vgp = varFactory.fieldVar(VGP_U2_S);
  VarPtr sigma11_vgp = varFactory.fieldVar(VGP_SIGMA11_S);
  VarPtr sigma12_vgp = varFactory.fieldVar(VGP_SIGMA12_S);
  VarPtr sigma21_vgp = varFactory.fieldVar(VGP_SIGMA21_S);
  VarPtr sigma22_vgp = varFactory.fieldVar(VGP_SIGMA22_S);
  VarPtr p_vgp = varFactory.fieldVar(VGP_P_S);
  
  
  VarPtr v1_vgp = varFactory.testVar(VGP_V1_S, HGRAD);
  VarPtr v2_vgp = varFactory.testVar(VGP_V2_S, HGRAD);

//  if (rank==0) {
//    cout << "bilinear form with zero background flow:\n";
//    zeroProblem.bf()->printTrialTestInteractions();
//  }
  
  VGPStokesFormulation stokesForm(1/Re);
  
  NavierStokesFormulation::setKovasznay(Re, zeroProblem.mesh(), u1_exact, u2_exact, p_exact);

  map< string, string > convergenceDataForMATLAB; // key: field file name
  
  for (int polyOrder = minPolyOrder; polyOrder <= maxPolyOrder; polyOrder++) {
    int H1Order = polyOrder + 1;
    
    int numCells1D = pow(2.0,minLogElements);

    if (rank==0) {
      cout << "L^2 order: " << polyOrder << endl;
      cout << "Re = " << Re << endl;
    }
    
    int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.

    vector< VGPNavierStokesProblem > problems;
    do {
      VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re,quadPointsKovasznay,
                                                              numCells1D,numCells1D,
                                                              H1Order, pToAdd,
                                                              u1_exact, u2_exact, p_exact, useCompliantNorm || useStokesCompliantNorm);
      
      problem.bf()->setUseExtendedPrecisionSolveForOptimalTestFunctions(longDoubleGramInversion);
      
      problem.backgroundFlow()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
      problem.solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
      
      problem.backgroundFlow()->setZeroMeanConstraintRho(zmcRho);
      problem.solutionIncrement()->setZeroMeanConstraintRho(zmcRho);
      
      FunctionPtr dt_inv;
      
      if (artificialTimeStepping) {
        //    // LHS gets u_inc / dt:
        BFPtr bf = problem.bf();
        dt_inv = ParameterFunction::parameterFunction(1.0 / dt); //Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
        bf->addTerm(-dt_inv * u1_vgp, v1_vgp);
        bf->addTerm(-dt_inv * u2_vgp, v2_vgp);
        problem.setIP( bf->graphNorm() ); // graph norm has changed...
      } else {
        dt_inv = Function::zero();
      }
      
      problems.push_back(problem);
      if ( useCompliantNorm ) {
        problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm(dt_inv));
      } else if (useStokesCompliantNorm) {
        VGPStokesFormulation stokesForm(1.0); // pretend Re = 1 in the graph norm
        problem.setIP(stokesForm.scaleCompliantGraphNorm());
      } else if (useStokesGraphNorm) {
        VGPStokesFormulation stokesForm(1.0); // pretend Re = 1 in the graph norm
        problem.setIP(stokesForm.graphNorm());
      } else if (! useGraphNorm ) {
        // then use the naive:
        problem.setIP(problem.bf()->naiveNorm());
      }
      if (rank==0) {
        cout << numCells1D << " x " << numCells1D << ": " << problem.mesh()->numGlobalDofs() << " dofs " << endl;
      }
      numCells1D *= 2;
    } while (pow(2.0,maxLogElements) >= numCells1D);
    
    // note that rhs and bilinearForm aren't really going to be right here, since they
    // involve a background flow which varies over the various problems...
    HConvergenceStudy study(problems[0].exactSolution(),
                            problems[0].mesh()->bilinearForm(),
                            problems[0].exactSolution()->rhs(),
                            problems[0].backgroundFlow()->bc(),
                            problems[0].bf()->graphNorm(),
                            minLogElements, maxLogElements, 
                            H1Order, pToAdd, false, useTriangles, false);
    study.setReportRelativeErrors(computeRelativeErrors);
    study.setCubatureDegreeForExact(kovasznayCubatureEnrichment);
    
    vector< SolutionPtr > solutions;
    numCells1D = pow(2.0,minLogElements);
    for (vector< VGPNavierStokesProblem >::iterator problem = problems.begin();
         problem != problems.end(); problem++) {
      SolutionPtr solnIncrement = problem->solutionIncrement();
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
      double weight = 1.0;
      do {
        weight = problem->iterate(useLineSearch, useCondensedSolve);
        
        LinearTermPtr rhsLT = problem->backgroundFlow()->rhs()->linearTerm();
        RieszRep rieszRep(problem->backgroundFlow()->mesh(), problem->backgroundFlow()->ip(), rhsLT);
        rieszRep.computeRieszRep();
        double costFunction = rieszRep.getNorm();
        double incr_norm = sqrt(l2_incr->integrate(problem->mesh()));
        
        if (rank==0) {
          cout << setprecision(6) << scientific;
          cout << "\x1B[2K"; // Erase the entire current line.
          cout << "\x1B[0E"; // Move to the beginning of the current line.
          cout << "Iteration: " << problem->iterationCount() << "; L^2(incr) = " << incr_norm;
          flush(cout);
//          cout << setprecision(6) << scientific;
//          cout << "Took " << weight << "-weighted step for " << numCells1D;
//          cout << " x " << numCells1D << " mesh: " << problem->iterationCount();
//          cout << setprecision(6) << fixed;
//          cout << " iterations; cost function " << costFunction << endl;
        }
      } while ((sqrt(l2_incr->integrate(problem->mesh())) > minL2Increment ) && (problem->iterationCount() < maxIters) && (weight != 0));
      
      if (rank==0) cout << endl;
      
      solutions.push_back( problem->backgroundFlow() );
      
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


    for (int i=0; i<=maxLogElements-minLogElements; i++) {
      SolutionPtr bestApproximation = study.bestApproximations()[i];
      VGPNavierStokesFormulation nsFormBest = VGPNavierStokesFormulation(Re, bestApproximation);
      SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
      Teuchos::RCP<ExactSolution> exact = nsFormBest.exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
//      bestApproximation->setIP( nsFormBest.bf()->naiveNorm() );
//      bestApproximation->setRHS( exact->rhs() );
      
      // use backgroundFlow's IP so that they're comparable
      Teuchos::RCP<DPGInnerProduct> ip = problems[i].backgroundFlow()->ip();
      LinearTermPtr rhsLT = exact->rhs()->linearTerm();
      RieszRep rieszRep(bestApproximation->mesh(), ip, rhsLT);
      rieszRep.computeRieszRep();
            
      double bestCostFunction = rieszRep.getNorm();
      if (rank==0)
        cout << "best energy error (measured according to the actual solution's test space IP): " << bestCostFunction << endl;
    }
    
    map< int, double > energyNormWeights;
    if (useCompliantNorm) {
      energyNormWeights[u1_vgp->ID()] = 1.0; // should be 1/h
      energyNormWeights[u2_vgp->ID()] = 1.0; // should be 1/h
      energyNormWeights[sigma11_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma12_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma21_vgp->ID()] = Re; // 1/mu
      energyNormWeights[sigma22_vgp->ID()] = Re; // 1/mu
      if (Re < 1) { // assuming we're using the experimental small Re thing
        energyNormWeights[p_vgp->ID()] = Re;
      } else {
        energyNormWeights[p_vgp->ID()] = 1.0;
      }
    } else {
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
    
    if (rank==0) {
      cout << setw(25);
      cout << "Solution Energy Error:" << setw(25) << "Best Energy Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++) {
        cout << setw(25) << solnEnergy[i] << setw(25) << bestEnergy[i] << endl;
      }
      cout << setw(25);
      cout << "Solution Velocity Error:" << setw(25) << "Best Velocity Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++) {
        cout << setw(25) << solnVelocityError[i] << setw(25) << bestVelocityError[i] << endl;
      }
      cout << setw(25);
      cout << "Solution Pressure Error:" << setw(25) << "Best Pressure Error:" << endl;
      cout << scientific << setprecision(1);
      for (int i=0; i<bestEnergy.size(); i++) {
        cout << setw(25) << solnPressureError[i] << setw(25) << bestPressureError[i] << endl;
      }
      
      vector< string > tableHeaders;
      vector< vector<double> > dataTable;
      vector< double > meshWidths;
      for (int i=minLogElements; i<=maxLogElements; i++) {
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
  
    if (rank == 0) {
      cout << study.TeXErrorRateTable();
      vector<int> primaryVariables;
      stokesForm.primaryTrialIDs(primaryVariables);
      vector<int> fieldIDs,traceIDs;
      vector<string> fieldFileNames;
      stokesForm.trialIDs(fieldIDs,traceIDs,fieldFileNames);
      cout << "******** Best Approximation comparison: ********\n";
      cout << study.TeXBestApproximationComparisonTable(primaryVariables);
      
      ostringstream filePathPrefix;
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_velpressure";
      study.TeXBestApproximationComparisonTable(primaryVariables,filePathPrefix.str());
      filePathPrefix.str("");
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_all";
      study.TeXBestApproximationComparisonTable(fieldIDs); 

      for (int i=0; i<fieldIDs.size(); i++) {
        int fieldID = fieldIDs[i];
        int traceID = traceIDs[i];
        string fieldName = fieldFileNames[i];
        ostringstream filePathPrefix;
        filePathPrefix << "navierStokes/" << fieldName << "_p" << polyOrder;
        bool writeMATLABplotData = false;
        study.writeToFiles(filePathPrefix.str(),fieldID,traceID, writeMATLABplotData);
      }
      
      for (int i=0; i<primaryVariables.size(); i++) {
        string convData = study.convergenceDataMATLAB(primaryVariables[i], minPolyOrder);
        cout << convData;
        convergenceDataForMATLAB[fieldFileNames[i]] += convData;
      }
      
      filePathPrefix.str("");
      filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_numDofs";
      cout << study.TeXNumGlobalDofsTable();
    }
    if (computeMaxConditionNumber) {
      for (int i=minLogElements; i<=maxLogElements; i++) {
        SolutionPtr soln = study.getSolution(i);
        ostringstream fileNameStream;
        fileNameStream << "nsStudy_maxConditionIPMatrix_" << i << ".dat";
        IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(soln->ip().get()), false );
        bool jacobiScalingTrue = true;
        double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, soln->mesh(), jacobiScalingTrue, fileNameStream.str());
        if (rank==0) {
          cout << "max Gram matrix condition number estimate for logElements " << i << ": "  << maxConditionNumber << endl;
          cout << "putative worst-conditioned Gram matrix written to: " << fileNameStream.str() << "." << endl;
        }
      }
    }
  }
  if (rank==0) {
    ostringstream filePathPrefix;
    filePathPrefix << "navierStokes/" << formulationTypeStr << "_";
    for (map<string,string>::iterator convIt = convergenceDataForMATLAB.begin(); convIt != convergenceDataForMATLAB.end(); convIt++) {
      string fileName = convIt->first + ".m";
      string data = convIt->second;
      fileName = filePathPrefix.str() + fileName;
      ofstream fout(fileName.c_str());
      fout << data;
      fout.close();
    }
  }
  
}