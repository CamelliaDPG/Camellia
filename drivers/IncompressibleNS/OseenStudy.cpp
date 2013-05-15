/*
 *  OseenStudy.cpp
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

#include "OseenFormulations.h"

#include "NavierStokesFormulation.h"

#include "MeshUtilities.h"

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
  bool useGraphNorm = true;
  
  int minPolyOrder = args.Input<int>("--minPolyOrder", "L^2 (field) minimum polynomial order",0);
  int maxPolyOrder = args.Input<int>("--maxPolyOrder", "L^2 (field) maximum polynomial order",1);
  int minLogElements = args.Input<int>("--minLogElements", "base 2 log of the minimum number of elements in one mesh direction", 0);
  int maxLogElements = args.Input<int>("--maxLogElements", "base 2 log of the maximum number of elements in one mesh direction", 4);
  double Re = args.Input<double>("--Re", "Reynolds number", 40);
  bool longDoubleGramInversion = args.Input<bool>("--longDoubleGramInversion", "use long double Cholesky factorization for Gram matrix", false);
//  bool outputStiffnessMatrix = args.Input<bool>("--writeFinalStiffnessToDisk", "write the final stiffness matrix to disk.", false);
  bool computeMaxConditionNumber = args.Input<bool>("--computeMaxConditionNumber", "compute the maximum Gram matrix condition number for final mesh.", false);
  bool useCompliantNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' norm", !useGraphNorm);
  
  useGraphNorm = !useCompliantNorm;
  
  try {
    args.Process();
  } catch ( choice::ArgException& e )
  {
    exit(1);
  }
  
  int pToAdd = 2; // for optimal test function approximation
  bool computeRelativeErrors = true; // we'll say false when one of the exact solution components is 0
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  BasisFactory::setUseEnrichedTraces(useEnrichedTraces);
  bool scaleSigmaByMu = true;
  
  bool useTriangles = false;
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useGraphNorm = "  << (useGraphNorm ? "true" : "false") << "\n";
    cout << "useCompliantNorm = "  << (useCompliantNorm ? "true" : "false") << "\n";
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

//  // symmetric domain to make it simple to construct zero-mean pressure (choose any odd function)
//  quadPointsKovasznay(0,0) = -1.0; // x1
//  quadPointsKovasznay(0,1) = -1.0; // y1
//  quadPointsKovasznay(1,0) =  1.0;
//  quadPointsKovasznay(1,1) = -1.0;
//  quadPointsKovasznay(2,0) =  1.0;
//  quadPointsKovasznay(2,1) =  1.0;
//  quadPointsKovasznay(3,0) = -1.0;
//  quadPointsKovasznay(3,1) =  1.0;
  
//  double Re = 10.0;  // Cockburn Kanschat Stokes
//  double Re = 40.0; // Evans Hughes Navier-Stokes
//  double Re = 1000.0;
  
  string formulationTypeStr = "vgp";
  
  FunctionPtr u1_exact, u2_exact, p_exact;
  
  int numCellsFineMesh = 20; // for computing a zero-mean pressure
  int H1OrderFineMesh = 5;

  FunctionPtr zero = Function::zero();
  VGPOseenProblem zeroProblem = VGPOseenProblem(Re, quadPointsKovasznay,
                                                numCellsFineMesh, numCellsFineMesh,
                                                H1OrderFineMesh, pToAdd,
                                                zero, zero, zero, useCompliantNorm, scaleSigmaByMu);
  
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1_vgp = varFactory.fieldVar(VGP_U1_S);
  VarPtr u2_vgp = varFactory.fieldVar(VGP_U2_S);
  VarPtr sigma11_vgp = varFactory.fieldVar(VGP_SIGMA11_S);
  VarPtr sigma12_vgp = varFactory.fieldVar(VGP_SIGMA12_S);
  VarPtr sigma21_vgp = varFactory.fieldVar(VGP_SIGMA21_S);
  VarPtr sigma22_vgp = varFactory.fieldVar(VGP_SIGMA22_S);
  VarPtr p_vgp = varFactory.fieldVar(VGP_P_S);
    
  VGPStokesFormulation stokesForm(1/Re,false,true);
  
  NavierStokesFormulation::setKovasznay(Re, zeroProblem.mesh(), u1_exact, u2_exact, p_exact);

  // test:
//  u1_exact = Function::xn(1); // Function::xn(2); // 2 * Function::xn(1) * Function::yn(1); //Function::xn(1);
//  u2_exact = -Function::yn(1); // -2 * Function::xn(1) * Function::yn(1); //- Function::yn(2); //-Function::yn(1);
//  p_exact  = Function::zero(); //Function::xn(5); // odd function
//  computeRelativeErrors = false;
  
  map< string, string > convergenceDataForMATLAB; // key: field file name
  
  for (int polyOrder = minPolyOrder; polyOrder <= maxPolyOrder; polyOrder++) {
    int H1Order = polyOrder + 1;
    
    int numCells1D = pow(2.0,minLogElements);

    if (rank==0) {
      cout << "L^2 order: " << polyOrder << endl;
      cout << "Re = " << Re << endl;
    }
    
    int kovasznayCubatureEnrichment = 10;

    vector< VGPOseenProblem > problems;
    do {
      VGPOseenProblem problem = VGPOseenProblem(Re, quadPointsKovasznay,
                                                numCells1D, numCells1D,
                                                H1Order, pToAdd,
                                                u1_exact, u2_exact, p_exact, useCompliantNorm, scaleSigmaByMu);
      
//      cout << "problem.bf():\n";
//      problem.bf()->printTrialTestInteractions();
      
      problem.bf()->setUseExtendedPrecisionSolveForOptimalTestFunctions(longDoubleGramInversion);
      
      problem.solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
      problems.push_back(problem);
      if ( useCompliantNorm ) {
        problem.setIP(problem.vgpOseenFormulation()->scaleCompliantGraphNorm());
      } else if (! useGraphNorm ) {
        // then use the naive:
        problem.setIP(problem.bf()->naiveNorm());
      }
      if (rank==0) {
        cout << numCells1D << " x " << numCells1D << ": " << problem.mesh()->numGlobalDofs() << " dofs " << endl;
      }
      numCells1D *= 2;
    } while (pow(2.0,maxLogElements) >= numCells1D);
    
    /*{ // DEBUGGING CODE:
      FunctionPtr u1_exact_fxn = problems[0].exactSolution()->exactFunctions().find(u1_vgp->ID())->second;
      FunctionPtr u2_exact_fxn = problems[0].exactSolution()->exactFunctions().find(u2_vgp->ID())->second;
      FunctionPtr p_exact_fxn = problems[0].exactSolution()->exactFunctions().find(p_vgp->ID())->second;
      
      double u1_err = (u1_exact_fxn - u1_exact)->l2norm(problems[0].mesh());
      double u2_err = (u2_exact_fxn - u2_exact)->l2norm(problems[0].mesh());
      double p_err = (p_exact_fxn - p_exact)->l2norm(problems[0].mesh());
      
      cout << "u1 err: " << u1_err << endl;
      cout << "u2 err: " << u2_err << endl;
      cout << "p err: "  << p_err << endl;
      
      FunctionPtr sigma11_exact = problems[0].exactSolution()->exactFunctions().find(sigma11_vgp->ID())->second;
      FunctionPtr sigma12_exact = problems[0].exactSolution()->exactFunctions().find(sigma12_vgp->ID())->second;
      FunctionPtr sigma21_exact = problems[0].exactSolution()->exactFunctions().find(sigma21_vgp->ID())->second;
      FunctionPtr sigma22_exact = problems[0].exactSolution()->exactFunctions().find(sigma22_vgp->ID())->second;
      
      double sigma11_err = (sigma11_exact - 1.0 / Re)->l2norm(problems[0].mesh());
      double sigma12_err = ( sigma12_exact )->l2norm(problems[0].mesh());
      double sigma21_err = ( sigma21_exact )->l2norm(problems[0].mesh());
      double sigma22_err = ( sigma22_exact + 1.0 / Re)->l2norm(problems[0].mesh());
      
      cout << "sigma11 err: " << sigma11_err << endl;
      cout << "sigma12 err: " << sigma12_err << endl;
      cout << "sigma21 err: " << sigma21_err << endl;
      cout << "sigma22 err: " << sigma22_err << endl;
      
      RHSEasy* rhs = dynamic_cast< RHSEasy* >( problems[0].exactSolution()->rhs().get() );
      LinearTermPtr rhsLT = rhs->linearTerm();
      cout << "rhsLT has " << rhsLT->summands().size() << " summands.\n";
      cout << "rhsLT: " << rhsLT->displayString() << endl;
      if (rhsLT->isZero()) {
        cout << "RHS is identically zero.\n";
      } else {
        cout << "RHS is not identically zero.\n";
        FunctionPtr f1 = rhsLT->summands()[0].first;
        double tol = 1e-12;
        FunctionPtr x = Function::xn(1);
        FunctionPtr y = Function::yn(1);
        double l2norm = (f1+x)->l2norm(problems[0].mesh());
        if ( l2norm < tol) {
          cout << "f1 = -x.\n";
        } else {
          cout << "(f1 + x), l2 norm: " << l2norm << endl;
        }
        FunctionPtr f2 = rhsLT->summands()[1].first;
        l2norm = (f2+y)->l2norm(problems[0].mesh());
        if ( l2norm < tol) {
          cout << "f2 = -y.\n";
        } else {
          cout << "(f2 + y), l2 norm: " << l2norm << endl;
        }
      }
    }*/
    
    HConvergenceStudy study(problems[0].exactSolution(),
                            problems[0].mesh()->bilinearForm(),
                            problems[0].exactSolution()->rhs(),
                            problems[0].solution()->bc(),
                            problems[0].bf()->graphNorm(),
                            minLogElements, maxLogElements, 
                            H1Order, pToAdd, false, useTriangles, false);
    study.setReportRelativeErrors(computeRelativeErrors);
    study.setCubatureDegreeForExact(kovasznayCubatureEnrichment);
    
    vector< SolutionPtr > solutions;
    numCells1D = pow(2.0,minLogElements);
    for (vector< VGPOseenProblem >::iterator problem = problems.begin();
         problem != problems.end(); problem++) {
      
      SolutionPtr soln = problem->solution();
      soln->solve();
      solutions.push_back( soln );
      
      numCells1D *= 2;
    }
    
    study.setSolutions(solutions);

//    for (int i=0; i<=maxLogElements-minLogElements; i++) {
//      SolutionPtr bestApproximation = study.bestApproximations()[i];
//      VGPOseenFormulation nsFormBest = VGPOseenFormulation(Re, bestApproximation);
//      SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
//      Teuchos::RCP<ExactSolution> exact = nsFormBest.exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
////      bestApproximation->setIP( nsFormBest.bf()->naiveNorm() );
////      bestApproximation->setRHS( exact->rhs() );
//      
//      // use backgroundFlow's IP so that they're comparable
//      Teuchos::RCP<DPGInnerProduct> ip = problems[i].backgroundFlow()->ip();
//      LinearTermPtr rhsLT = ((RHSEasy*) exact->rhs().get())->linearTerm();
//      RieszRep rieszRep(bestApproximation->mesh(), ip, rhsLT);
//      rieszRep.computeRieszRep();
//            
//      double bestCostFunction = rieszRep.getNorm();
//      if (rank==0)
//        cout << "best energy error (measured according to the actual solution's test space IP): " << bestCostFunction << endl;
//    }
    
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
      filePathPrefix << "oseen/" << formulationTypeStr << "_p" << polyOrder << "_velpressure";
      study.TeXBestApproximationComparisonTable(primaryVariables,filePathPrefix.str());
      filePathPrefix.str("");
      filePathPrefix << "oseen/" << formulationTypeStr << "_p" << polyOrder << "_all";
      study.TeXBestApproximationComparisonTable(fieldIDs); 

      for (int i=0; i<fieldIDs.size(); i++) {
        int fieldID = fieldIDs[i];
        int traceID = traceIDs[i];
        string fieldName = fieldFileNames[i];
        ostringstream filePathPrefix;
        filePathPrefix << "oseen/" << fieldName << "_p" << polyOrder;
        bool writeMATLABplotData = true;
        study.writeToFiles(filePathPrefix.str(),fieldID,traceID, writeMATLABplotData);
      }
      
      for (int i=0; i<primaryVariables.size(); i++) {
        string convData = study.convergenceDataMATLAB(primaryVariables[i], minPolyOrder);
        cout << convData;
        convergenceDataForMATLAB[fieldFileNames[i]] += convData;
      }
      
      filePathPrefix.str("");
      filePathPrefix << "oseen/" << formulationTypeStr << "_p" << polyOrder << "_numDofs";
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
    map< int, double > energyNormWeights;
    energyNormWeights[u1_vgp->ID()] = 1.0; // should be 1/h
    energyNormWeights[u2_vgp->ID()] = 1.0; // should be 1/h
    energyNormWeights[sigma11_vgp->ID()] = Re; // 1/mu
    energyNormWeights[sigma12_vgp->ID()] = Re; // 1/mu
    energyNormWeights[sigma21_vgp->ID()] = Re; // 1/mu
    energyNormWeights[sigma22_vgp->ID()] = Re; // 1/mu
    energyNormWeights[p_vgp->ID()] = 1.0;
    vector<double> bestEnergy = study.weightedL2Error(energyNormWeights,true);
    vector<double> solnEnergy = study.weightedL2Error(energyNormWeights,false);
    cout << "Solution Energy Error: " << setw(30) << "Best Energy Error:" << endl;
    cout << scientific << setprecision(1);
    for (int i=0; i<bestEnergy.size(); i++) {
      cout << solnEnergy[i] << setw(30) << bestEnergy[i] << endl;
    }
  }
  if (rank==0) {
    ostringstream filePathPrefix;
    filePathPrefix << "oseen/" << formulationTypeStr << "_";
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