/*
 *  StokesStudy.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "StokesStudy.h"
#include "StokesManufacturedSolution.h"
#include "StokesBilinearForm.h"

#include "HConvergenceStudy.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "StokesManufacturedSolution.h"
#include "StokesVVPBilinearForm.h"
#include "StokesMathBilinearForm.h"

#include "InnerProductScratchPad.h"

#include "../MultiOrderStudy/MultiOrderStudy.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "CGSolver.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "StokesFormulations.h"

using namespace std;

class EntireBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x+1.0) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y+1.0) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class Cos_ay : public SimpleFunction {
  double _a;
public:
  Cos_ay(double a);
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  
  string displayString();
};

class Sin_ay : public SimpleFunction {
  double _a;
public:
  Sin_ay(double a) {
    _a = a;
  }
  double value(double x, double y) {
    return sin( _a * y);
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return _a * (FunctionPtr) Teuchos::rcp(new Cos_ay(_a));
  }
  string displayString() {
    ostringstream ss;
    ss << "\\sin( " << _a << " y )";
    return ss.str();
  }
};

Cos_ay::Cos_ay(double a) {
  _a = a;
}
double Cos_ay::value(double x, double y) {
  return cos( _a * y );
}
FunctionPtr Cos_ay::dx() {
  return Function::zero();
}
FunctionPtr Cos_ay::dy() {
  return -_a * (FunctionPtr) Teuchos::rcp(new Sin_ay(_a));
}

string Cos_ay::displayString() {
  ostringstream ss;
  ss << "\\cos( " << _a << " y )";
  return ss.str();
}

class Xp : public SimpleFunction { // x^p, for x >= 0
  double _p;
public:
  Xp(double p) {
    _p = p;
  }
  double value(double x, double y) {
    if (x < 0) {
      cout << "calling pow(x," << _p << ") for x < 0: x=" << x << endl;
    }
    return pow(x,_p);
  }
  FunctionPtr dx() {
    return _p * (FunctionPtr) Teuchos::rcp( new Xp(_p-1) );
  }
  FunctionPtr dy() {
    return Function::zero();
  }
  string displayString() {
    ostringstream ss;
    ss << "x^{" << _p << "}";
    return ss.str();
  }
};

void parseArgs(int argc, char *argv[], int &polyOrder, int &minLogElements, int &maxLogElements,
               StokesFormulationChoice &formulationType,
               bool &useTriangles, bool &useMultiOrder, bool &useOptimalNorm, string &formulationTypeStr) {
  polyOrder = 2; minLogElements = 0; maxLogElements = 4;
  
  // set up defaults:
  useTriangles = false;
  useOptimalNorm = true; 
  formulationType = VSP;
  formulationTypeStr = "original conforming";
  string multiOrderStudyType = "multiOrderQuad";
  
  useMultiOrder = false;
  
  string normChoice = "opt";
  
  /* Usage:
   Multi-Order, naive norm:
     StokesStudy "multiOrder{Tri|Quad}" formulationTypeStr
   Single Order, original conforming, optimal norm, quad:
     StokesStudy polyOrder minLogElements maxLogElements
   Single Order, original conforming, quad:
   StokesStudy normChoice polyOrder minLogElements maxLogElements
   Single Order, quad:
     StokesStudy normChoice formulationTypeStr polyOrder minLogElements maxLogElements
   Single Order, quad:
     StokesStudy normChoice formulationTypeStr polyOrder minLogElements maxLogElements {"quad"|"tri"}
   
   where:
   formulationTypeStr = {"original conforming"|"nonConforming"|"vvp"|"math"|"dev"}
   normChoice = {"opt"|"naive"}
   
   */
  
  if (argc == 3) {
    string multiOrderStudyType = argv[1];
    formulationTypeStr = argv[2];
    if (multiOrderStudyType == "multiOrderTri") {
      useTriangles = true;
    } else {
      useTriangles = false;
    }
    useMultiOrder = true;
    useOptimalNorm = false; // using naive norm for paper
    polyOrder = 1;
  } else if (argc == 4) {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
  } else if (argc == 5) {
    normChoice = argv[1];
    polyOrder = atoi(argv[2]);
    minLogElements = atoi(argv[3]);
    maxLogElements = atoi(argv[4]);
  } else if (argc == 6) {
    normChoice = argv[1];

    formulationTypeStr = argv[2];
    polyOrder = atoi(argv[3]);
    minLogElements = atoi(argv[4]);
    maxLogElements = atoi(argv[5]);
  } else if (argc == 7) {
    normChoice = argv[1];
    formulationTypeStr = argv[2];
    polyOrder = atoi(argv[3]);
    minLogElements = atoi(argv[4]);
    maxLogElements = atoi(argv[5]);
    string elementTypeStr = argv[6];
    if (elementTypeStr == "tri") {
      useTriangles = true;
    } else if (elementTypeStr == "quad") {
      useTriangles = false;
    } // otherwise, just use whatever was defined above
  }
  if (normChoice == "naive") {
    useOptimalNorm = false; // otherwise, use naive
  }
  if (formulationTypeStr == "vgp") {
    formulationType = VGP;
  } else if (formulationTypeStr == "vvp") {
    formulationType = VVP;
  } else if (formulationTypeStr == "dds") {
    formulationType = DDS;
  } else if (formulationTypeStr == "ddsp") {
    formulationType = DDSP;
  } else {
    formulationType = VSP;
    formulationTypeStr = "vsp";
  }
}

void LShapedDomain(vector<FieldContainer<double> > &vertices, vector< vector<int> > &elementVertices, bool useTriangles) {
  // builds a domain for (-1,1)^2 \ (0,1) x (-1,0)
  // points start in the lower left and proceed clockwise around the domain
  FieldContainer<double> p1(2);
  FieldContainer<double> p2(2);
  FieldContainer<double> p3(2);
  FieldContainer<double> p4(2);
  FieldContainer<double> p5(2);
  FieldContainer<double> p6(2);
  FieldContainer<double> p7(2);
  FieldContainer<double> p8(2);
  
  p1(0) = -1.0; p1(1) = -1.0;
  p2(0) = -1.0; p2(1) =  0.0;
  p3(0) = -1.0; p3(1) =  1.0;
  p4(0) =  0.0; p4(1) =  1.0;
  p5(0) =  1.0; p5(1) =  1.0;
  p6(0) =  1.0; p6(1) =  0.0;
  p7(0) =  0.0; p7(1) =  0.0;
  p8(0) =  0.0; p8(1) = -1.0;
  
  vertices.push_back(p1);
  vertices.push_back(p2);
  vertices.push_back(p3);
  vertices.push_back(p4);
  vertices.push_back(p5);
  vertices.push_back(p6);
  vertices.push_back(p7);
  vertices.push_back(p8);
  
  if (useTriangles) {
    vector<int> element;
    element.push_back(0);
    element.push_back(7);
    element.push_back(6);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(6);
    element.push_back(1);
    element.push_back(0);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(1);
    element.push_back(6);
    element.push_back(2);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(2);
    element.push_back(6);
    element.push_back(3);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(6);
    element.push_back(4);
    element.push_back(3);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(6);
    element.push_back(5);
    element.push_back(4);
    elementVertices.push_back(element);
  } else {
    vector<int> element;
    element.push_back(0);
    element.push_back(7);
    element.push_back(6);
    element.push_back(1);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(1);
    element.push_back(6);
    element.push_back(3);
    element.push_back(2);
    elementVertices.push_back(element);
    element.clear();
    
    element.push_back(6);
    element.push_back(5);
    element.push_back(4);
    element.push_back(3);
    elementVertices.push_back(element);
  }
}

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
  int pToAdd = 1; // for optimal test function approximation
  bool computeRelativeErrors = true; // we'll say false when one of the exact solution components is 0
  bool useHDGManufacturedSolution = true;
  bool useKanschatSoln = false;
  bool useCG = false;
  bool useEnrichedTraces = true; // enriched traces are the right choice, mathematically speaking
  double cgTol = 1e-8;
  int cgMaxIt = 400000;
  Teuchos::RCP<Solver> cgSolver = Teuchos::rcp( new CGSolver(cgMaxIt, cgTol) );

  if (useHDGManufacturedSolution && useKanschatSoln ) {
    cout << "Error: cannot use both HDG and Kanschat solution simultaneously!\n";
    exit(1);
  }
  
  BasisFactory::setUseEnrichedTraces(useEnrichedTraces);
  
  // parse args:
  int polyOrder, minLogElements, maxLogElements;
  bool useTriangles, useOptimalNorm, useMultiOrder;
  
  StokesFormulationChoice formulationType;
  string formulationTypeStr;
  parseArgs(argc, argv, polyOrder, minLogElements, maxLogElements, formulationType, useTriangles,
            useMultiOrder, useOptimalNorm, formulationTypeStr);

  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "formulationType = " << formulationTypeStr                  << "\n";
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
    cout << "useHDGManufacturedSolution = "  << (useHDGManufacturedSolution ? "true" : "false") << "\n";
  }
  
  double mu = 1.0;
  Teuchos::RCP<StokesFormulation> stokesForm;
  
  switch (formulationType) {
    case DDS:
      stokesForm = Teuchos::rcp(new DDSStokesFormulation(mu));
      break;        
    case DDSP:
      stokesForm = Teuchos::rcp(new DDSPStokesFormulation(mu));
      break;        
    case VGP:
      stokesForm = Teuchos::rcp(new VGPStokesFormulation(mu));
      break;
    case VVP:
      stokesForm = Teuchos::rcp(new VVPStokesFormulation(mu));
      break;
    case VSP:
      stokesForm = Teuchos::rcp(new VSPStokesFormulation(mu));
      break;
    default:
      break;
  }
  
  Teuchos::RCP<ExactSolution> mySolution;
  if (! stokesForm.get() ) {
    cout << "\n\n ERROR: stokesForm undefined!!\n\n";
    exit(1);
  } else {
    // trying out the new ExactSolution features:
    FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
    FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
    FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
    
    FunctionPtr x = Teuchos::rcp ( new Xn(1) );
    FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
    FunctionPtr y2 = Teuchos::rcp( new Yn(2) );
    FunctionPtr y = Teuchos::rcp( new Yn(1) );
    
    FunctionPtr u1_exact, u2_exact, p_exact;
    
    if (useKanschatSoln) {
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y;
      p_exact = 2.0 * exp_x * sin_y;
    } else if (useHDGManufacturedSolution) {
      const double PI  = 3.141592653589793238462;
      const double lambda = 0.54448373678246;
      const double omega = 3.0 * PI / 2.0;
      const double lambda_plus = lambda + 1.0;
      const double lambda_minus = lambda - 1.0;
      const double omega_lambda = omega * lambda;
      
      FunctionPtr sin_lambda_plus_y  = Teuchos::rcp( new Sin_ay( lambda_plus ) );
      FunctionPtr sin_lambda_minus_y = Teuchos::rcp( new Sin_ay( lambda_minus) );
      
      FunctionPtr cos_lambda_plus_y  = Teuchos::rcp( new Cos_ay( lambda_plus ) );
      FunctionPtr cos_lambda_minus_y = Teuchos::rcp( new Cos_ay( lambda_minus) );
      
      FunctionPtr phi_y = (cos(omega_lambda) / lambda_plus)  * sin_lambda_plus_y  - cos_lambda_plus_y
                        - (cos(omega_lambda) / lambda_minus) * sin_lambda_minus_y + cos_lambda_minus_y;
      
      Teuchos::RCP<PolarizedFunction> phi_theta = Teuchos::rcp( new PolarizedFunction( phi_y ) );
      FunctionPtr phi_theta_prime = phi_theta->dtheta();
      FunctionPtr phi_theta_triple_prime = phi_theta->dtheta()->dtheta()->dtheta();
      
      FunctionPtr x_to_lambda = Teuchos::rcp( new Xp(lambda) );
      FunctionPtr x_to_lambda_minus = Teuchos::rcp( new Xp(lambda_minus) );
      FunctionPtr r_to_lambda = Teuchos::rcp( new PolarizedFunction( x_to_lambda ) );
      FunctionPtr r_to_lambda_minus = Teuchos::rcp( new PolarizedFunction( x_to_lambda_minus ) );
      
      FunctionPtr cos_theta = Teuchos::rcp( new PolarizedFunction(Teuchos::rcp( new Cos_y ) ) );
      FunctionPtr sin_theta = Teuchos::rcp( new PolarizedFunction(Teuchos::rcp( new Sin_y ) ) );
      
      u1_exact = r_to_lambda * (  lambda_plus * sin_theta * (FunctionPtr) phi_theta + cos_theta * phi_theta_prime );
      u2_exact = r_to_lambda * ( -lambda_plus * cos_theta * (FunctionPtr) phi_theta + sin_theta * phi_theta_prime );
      p_exact = -r_to_lambda_minus * ( (lambda_plus * lambda_plus) * phi_theta_prime + phi_theta_triple_prime) / lambda_minus;
    } else {
      computeRelativeErrors = false;
      u1_exact = Function::zero();
      u2_exact = Function::zero();
      p_exact = y; // odd function: zero mean on our domain
    }
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary );
    BFPtr stokesBF = stokesForm->bf();
    if (rank==0)
      stokesBF->printTrialTestInteractions();
    
    mySolution = stokesForm->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
  }
  
  Teuchos::RCP<DPGInnerProduct> ip;
  if (useOptimalNorm) {
    if ( stokesForm.get() ) {
      ip = stokesForm->graphNorm();
    } else {
      ip = Teuchos::rcp( new OptimalInnerProduct( mySolution->bilinearForm() ) );
    }
  } else {
    ip = Teuchos::rcp( new   MathInnerProduct( mySolution->bilinearForm() ) );
  }
  
  if (rank==0) 
    ip->printInteractions();
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  int u1_trialID, u2_trialID, p_trialID;
  int u1_traceID, u2_traceID;
  if ( stokesForm.get() ) {
    // don't define IDs: we don't need them
  } else if (formulationType == VVP) {
    u1_trialID = StokesVVPBilinearForm::U1;
    u2_trialID = StokesVVPBilinearForm::U2;
    p_trialID = StokesVVPBilinearForm::P;
    u1_traceID = -1; // no velocity traces available in VVP formulation
    u2_traceID = -1;
  } else {
    u1_trialID = StokesBilinearForm::U1;
    u2_trialID = StokesBilinearForm::U2;
    p_trialID =  StokesBilinearForm::P;    
    u1_traceID = StokesBilinearForm::U1_HAT;
    u2_traceID = StokesBilinearForm::U2_HAT;
  }
  
  if ( !useMultiOrder ) {
    HConvergenceStudy study(mySolution,
                            mySolution->bilinearForm(),
                            mySolution->ExactSolution::rhs(),
                            mySolution->bc(), ip,  
                            minLogElements, maxLogElements, 
                            polyOrder+1, pToAdd, false, useTriangles, false);
    study.setReportRelativeErrors(computeRelativeErrors);
    if (useCG) study.setSolver(cgSolver);
    
    if (! useHDGManufacturedSolution) {
      study.solve(quadPoints);
    } else {
      // L-shaped domain
      vector<FieldContainer<double> > vertices;
      vector< vector<int> > elementVertices;
      LShapedDomain(vertices, elementVertices, useTriangles);
      study.solve(vertices,elementVertices);
    }
    
    if (rank == 0) {

      cout << study.TeXErrorRateTable();
      vector<int> primaryVariables;
      stokesForm->primaryTrialIDs(primaryVariables);
      vector<int> fieldIDs,traceIDs;
      vector<string> fieldFileNames;
      stokesForm->trialIDs(fieldIDs,traceIDs,fieldFileNames);
      cout << "******** Best Approximation comparison: ********\n";
      cout << study.TeXBestApproximationComparisonTable(primaryVariables);
      
      ostringstream filePathPrefix;
      filePathPrefix << "stokes/" << formulationTypeStr << "_p" << polyOrder << "_velpressure";
      study.TeXBestApproximationComparisonTable(primaryVariables,filePathPrefix.str());
      filePathPrefix.str("");
      filePathPrefix << "stokes/" << formulationTypeStr << "_p" << polyOrder << "_all";
      study.TeXBestApproximationComparisonTable(fieldIDs); 

      // for now, not interested in plots, etc. of individual variables.
      for (int i=0; i<fieldIDs.size(); i++) {
        int fieldID = fieldIDs[i];
        int traceID = traceIDs[i];
        string fieldName = fieldFileNames[i];
        ostringstream filePathPrefix;
        filePathPrefix << "stokes/" << fieldName << "_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),fieldID,traceID);
      }
      
      for (int i=0; i<primaryVariables.size(); i++) {
        cout << study.convergenceDataMATLAB(primaryVariables[i]);  
      }
      
      filePathPrefix.str("");
      filePathPrefix << "stokes/" << formulationTypeStr << "_p" << polyOrder << "_numDofs";
      cout << study.TeXNumGlobalDofsTable();
    }
  } else {
    cout << "Generating mixed-order 16x16 mesh" << endl;
    Teuchos::RCP<Mesh> mesh = MultiOrderStudy::makeMultiOrderMesh16x16(quadPoints,
                                                                       mySolution->bilinearForm(),
                                                                       polyOrder+1, pToAdd,
                                                                       useTriangles);
    
    Solution solution(mesh, mySolution->bc(), mySolution->ExactSolution::rhs(), ip);
    solution.solve();
    int cubDegree = 15; // for error computations
    double  pError = mySolution->L2NormOfError(solution, p_trialID,  cubDegree);
    double u1Error = mySolution->L2NormOfError(solution, u1_trialID, cubDegree);
    double u2Error = mySolution->L2NormOfError(solution, u2_trialID, cubDegree);
    
    string meshType = (useTriangles) ? "triangular" : "quad";
    
    cout << "Multi-order, 16x16 " << meshType << " mesh, pressure error: "  <<  pError << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, u1 error: "        << u1Error << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, u2 error: "        << u2Error << endl;
  }
}