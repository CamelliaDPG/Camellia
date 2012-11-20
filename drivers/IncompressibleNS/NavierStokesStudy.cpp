/*
 *  NavierStokesStudy.cpp
 *
 *  Created by Nathan Roberts on 11/15/12.
 *
 */

#include "HConvergenceStudy.h"

#include "InnerProductScratchPad.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "CGSolver.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "NavierStokesFormulation.h"

using namespace std;

int main(int argc, char *argv[]) {
  int rank = 0, numProcs = 1;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
  numProcs=mpiSession.getNProc();
#else
#endif
  int pToAdd = 2; // for optimal test function approximation
  bool computeRelativeErrors = true; // we'll say false when one of the exact solution components is 0
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::setUseEnrichedTraces(useEnrichedTraces);
  
  // parse args:
  bool useTriangles = false, useOptimalNorm = true;
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
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
  
  // Domain from Evans Hughes for Navier-Stokes:
  quadPointsKovasznay(0,0) =  0.0; // x1
  quadPointsKovasznay(0,1) = -0.5; // y1
  quadPointsKovasznay(1,0) =  1.0;
  quadPointsKovasznay(1,1) = -0.5;
  quadPointsKovasznay(2,0) =  1.0;
  quadPointsKovasznay(2,1) =  0.5;
  quadPointsKovasznay(3,0) =  0.0;
  quadPointsKovasznay(3,1) =  0.5;

//  double Re = 10.0;  // Cockburn Kanschat Stokes
  double Re = 40.0; // Evans Hughes Navier-Stokes
//  double Re = 1000.0;
  
  FunctionPtr u1_exact, u2_exact, p_exact;
  
  int numCellsFineMesh = 20; // for computing a zero-mean pressure
  int H1OrderFineMesh = 5;
  int H1Order = 3;
  
  int polyOrder = H1Order - 1;
  if (rank==0) {
    cout << "L^2 order: " << polyOrder << endl;
    cout << "Re = " << Re << endl;
  }
  
  int kovasznayCubatureEnrichment = H1Order - 1 + 5;
  FunctionPtr zero = Function::zero();
  VGPNavierStokesProblem zeroProblem = VGPNavierStokesProblem(Re, quadPointsKovasznay,
                                                              numCellsFineMesh, numCellsFineMesh,
                                                              H1OrderFineMesh, pToAdd,
                                                              zero, zero, zero);
  
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1_vgp = varFactory.fieldVar(VGP_U1_S);
  VarPtr u2_vgp = varFactory.fieldVar(VGP_U2_S);
  VarPtr sigma11_vgp = varFactory.fieldVar(VGP_SIGMA11_S);
  VarPtr sigma12_vgp = varFactory.fieldVar(VGP_SIGMA12_S);
  VarPtr sigma21_vgp = varFactory.fieldVar(VGP_SIGMA21_S);
  VarPtr sigma22_vgp = varFactory.fieldVar(VGP_SIGMA22_S);
  VarPtr p_vgp = varFactory.fieldVar(VGP_P_S);
    
  VGPStokesFormulation stokesForm(1/Re);
  
  NavierStokesFormulation::setKovasznay(Re, zeroProblem.mesh(), u1_exact, u2_exact, p_exact);

  int minLogElements = 0;
  int maxLogElements = 3;
  
  int numCells1D = pow(2.0,minLogElements);
  int maxIters = 25; // max nonlinear steps
  double minL2Increment = 1e-12;
  vector< VGPNavierStokesProblem > problems;
  do {
    VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re,quadPointsKovasznay,
                                                            numCells1D,numCells1D,
                                                            H1Order, pToAdd,
                                                            u1_exact, u2_exact, p_exact);
    problem.backgroundFlow()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    problem.solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    problems.push_back(problem);
    if ( !useOptimalNorm ) {
      // then use the naive:
      problem.setIP(problem.bf()->naiveNorm());
    }
    if (rank==0) {
      cout << numCells1D << " x " << numCells1D << ": " << problem.mesh()->numGlobalDofs() << " dofs " << endl;
    }
    numCells1D *= 2;
  } while (pow(2.0,maxLogElements) >= numCells1D);
  
  HConvergenceStudy study(problems[0].exactSolution(),
                          problems[0].mesh()->bilinearForm(),
                          problems[0].exactSolution()->rhs(),
                          problems[0].backgroundFlow()->bc(),
                          problems[0].bf()->graphNorm(),
                          minLogElements, maxLogElements, 
                          H1Order, pToAdd, false, useTriangles, false);
  study.setReportRelativeErrors(computeRelativeErrors);
  
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
    do {
      problem->iterate();
    } while ((sqrt(l2_incr->integrate(problem->mesh())) > minL2Increment ) && (problem->iterationCount() < maxIters));
    solutions.push_back( problem->backgroundFlow() );
    if (rank==0) {
      cout << numCells1D << " x " << numCells1D << ": " << problem->iterationCount() << " iterations " << endl;
    }
    numCells1D *= 2;
  }
  
  study.setSolutions(solutions);
  string formulationTypeStr = "vgp";
  
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

    // for now, not interested in plots, etc. of individual variables.
    for (int i=0; i<fieldIDs.size(); i++) {
      int fieldID = fieldIDs[i];
      int traceID = traceIDs[i];
      string fieldName = fieldFileNames[i];
      ostringstream filePathPrefix;
      filePathPrefix << "navierStokes/" << fieldName << "_p" << polyOrder;
      study.writeToFiles(filePathPrefix.str(),fieldID,traceID);
    }
    
    for (int i=0; i<primaryVariables.size(); i++) {
      cout << study.convergenceDataMATLAB(primaryVariables[i]);  
    }
    
    filePathPrefix.str("");
    filePathPrefix << "navierStokes/" << formulationTypeStr << "_p" << polyOrder << "_numDofs";
    cout << study.TeXNumGlobalDofsTable();
  }
}