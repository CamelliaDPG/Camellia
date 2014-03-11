//
//  StokesBackwardFacingStep.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "StokesFormulation.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "MeshFactory.h"

#include "choice.hpp"
#include "mpi_choice.hpp"

#include "RieszRep.h"

#include "SolutionExporter.h"

using namespace std;

MeshPtr mesh;

double integralOverMesh(LinearTermPtr testTerm, VarPtr testVar, FunctionPtr fxnToSubstitute) {
  map<int, FunctionPtr > varAsFunction;
  varAsFunction[testVar->ID()] = fxnToSubstitute;
  
  FunctionPtr substituteOnBoundary = testTerm->evaluate(varAsFunction, true);
  FunctionPtr substituteOnInterior = testTerm->evaluate(varAsFunction, false);
  double integral = substituteOnBoundary->integrate(mesh);
  integral += substituteOnInterior->integrate(mesh);
  return integral;
}

int main(int argc, char *argv[]) {
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
  
#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  
  int polyOrder, pToAdd;
  try {
    // read args:
    polyOrder = args.Input<int>("--polyOrder", "L^2 (field) polynomial order");
    pToAdd = args.Input<int>("--delta_p", "delta p for test enrichment", 2);
    args.Process();
  } catch ( choice::ArgException& e )
  {
    exit(0);
  }
  
  int H1Order = polyOrder + 1;
  
  bool useCompliantGraphNorm = false;   // weights to improve conditioning of the local problems
  bool useExtendedPrecisionForOptimalTestInversion = false;

  /////////////////////////// "VGP_CONFORMING" VERSION ///////////////////////

  // fluxes and traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // fields for SGP:
  VarPtr phi, p, sigma11, sigma12, sigma21, sigma22;
  // fields specific to VGP:
  VarPtr u1, u2;
  
  BFPtr stokesBF;
  IPPtr qoptIP;
  
  double mu = 1;
  
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  VarPtr tau1,tau2,v1,v2,q;
  VarFactory varFactory;
  tau1 = varFactory.testVar("\\tau_1", HDIV);
  tau2 = varFactory.testVar("\\tau_2", HDIV);
  v1 = varFactory.testVar("v_1", HGRAD);
  v2 = varFactory.testVar("v_2", HGRAD);
  q = varFactory.testVar("q", HGRAD);
  
  u1hat = varFactory.traceVar("\\widehat{u}_1");
  u2hat = varFactory.traceVar("\\widehat{u}_2");
  
  t1n = varFactory.fluxVar("\\widehat{t_{1n}}");
  t2n = varFactory.fluxVar("\\widehat{t_{2n}}");
  if (!useCompliantGraphNorm) {
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
  } else {
    u1 = varFactory.fieldVar("u_1", HGRAD);
    u2 = varFactory.fieldVar("u_2", HGRAD);
  }
  sigma11 = varFactory.fieldVar("\\sigma_11");
  sigma12 = varFactory.fieldVar("\\sigma_12");
  sigma21 = varFactory.fieldVar("\\sigma_21");
  sigma22 = varFactory.fieldVar("\\sigma_22");
  p = varFactory.fieldVar("p");
  
  stokesBF = Teuchos::rcp( new BF(varFactory) );  
  // tau1 terms:
  stokesBF->addTerm(u1,tau1->div());
  stokesBF->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
  stokesBF->addTerm(sigma12,tau1->y());
  stokesBF->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBF->addTerm(u2, tau2->div());
  stokesBF->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
  stokesBF->addTerm(sigma22,tau2->y());
  stokesBF->addTerm(-u2hat, tau2->dot_normal());
  
  // v1:
  stokesBF->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
  stokesBF->addTerm(mu * sigma12,v1->dy());
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( -t1n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
  stokesBF->addTerm(mu * sigma22,v2->dy());
  stokesBF->addTerm( -p, v2->dy());
  stokesBF->addTerm( -t2n, v2);
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  stokesBF->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
  
  if (rank==0)
    stokesBF->printTrialTestInteractions();
  
  stokesBF->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);

  mesh = MeshFactory::quadMesh(stokesBF, H1Order, pToAdd);
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  
  ////////////////////   CREATE RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs(); // zero for now...
  
  IPPtr ip;
  
  qoptIP = Teuchos::rcp(new IP());
      
  if (useCompliantGraphNorm) {
    qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
    qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
    qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
    qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
    qoptIP->addTerm( mu * v1->dx() + mu * v2->dy() );   // pressure
    qoptIP->addTerm( h * tau1->div() - h * q->dx() );   // u1
    qoptIP->addTerm( h * tau2->div() - h * q->dy());    // u2
    
    qoptIP->addTerm( (mu / h) * v1 );
    qoptIP->addTerm( (mu / h) * v2 );
    qoptIP->addTerm( q );
    qoptIP->addTerm( tau1 );
    qoptIP->addTerm( tau2 );
  } else { // standard graph norm, then
    qoptIP = stokesBF->graphNorm();
  }

  ip = qoptIP;
  
  if (rank==0) 
    ip->printInteractions();
  
  // aim is just to answer one simple question:
  // have I figured out a trial-space preimage for optimal test function (q=1, tau=0, v=0)?
  
  SolutionPtr soln = Teuchos::rcp(new Solution(mesh));
  
  FunctionPtr x = Function::xn();
  FunctionPtr y = Function::yn();
  
  // u1 = u1_hat = x / 2
  FunctionPtr u1_exact = x / 2;
  
  // u2 = u2_hat = y / 2
  FunctionPtr u2_exact = y / 2;
  
  // sigma = 0.5 * I
  FunctionPtr sigma11_exact = Function::constant(0.5);
  FunctionPtr sigma22_exact = Function::constant(0.5);
  
  // tn_hat = 0.5 * n
  FunctionPtr n = Function::normal();
  FunctionPtr t1n_exact = n->x() / 2;
  FunctionPtr t2n_exact = n->y() / 2;
  
  map<int, FunctionPtr > exact_soln;
  exact_soln[u1->ID()] = u1_exact;
  exact_soln[u1hat->ID()] = u1_exact;
  exact_soln[u2->ID()] = u2_exact;
  exact_soln[u2hat->ID()] = u2_exact;
  exact_soln[sigma11->ID()] = sigma11_exact;
  exact_soln[sigma22->ID()] = sigma22_exact;
  exact_soln[t1n->ID()] = t1n_exact;
  exact_soln[t2n->ID()] = t2n_exact;
  
  exact_soln[p->ID()] = Function::zero();
  exact_soln[sigma12->ID()] = Function::zero();
  exact_soln[sigma21->ID()] = Function::zero();
  
  soln->projectOntoMesh(exact_soln);
  
  LinearTermPtr soln_functional = stokesBF->testFunctional(soln);
  
  RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, soln_functional) );
  
  rieszRep->computeRieszRep();
  
  // get test functions:
  FunctionPtr q_fxn = Teuchos::rcp( new RepFunction(q, rieszRep) );
  FunctionPtr v1_fxn = Teuchos::rcp( new RepFunction(v1, rieszRep) );
  FunctionPtr v2_fxn = Teuchos::rcp( new RepFunction(v2, rieszRep) );
  FunctionPtr tau1_fxn = Teuchos::rcp( new RepFunction(tau1, rieszRep) );
  FunctionPtr tau2_fxn = Teuchos::rcp( new RepFunction(tau2, rieszRep) );
  
  cout << "L2 norm of (q-1) : " << (q_fxn - 1)->l2norm(mesh) << endl;
  cout << "L2 norm of (v1) : " << (v1_fxn)->l2norm(mesh) << endl;
  cout << "L2 norm of (v2) : " << (v2_fxn)->l2norm(mesh) << endl;
  cout << "L2 norm of (tau1) : " << (tau1_fxn)->l2norm(mesh) << endl;
  cout << "L2 norm of (tau2) : " << (tau2_fxn)->l2norm(mesh) << endl;
  
  VTKExporter exporter(soln, mesh, varFactory);
  exporter.exportSolution("conservationPreimage", H1Order*2);

  cout << "Checking that the soln_functional is what I expect:\n";
  
  FunctionPtr xyVector = Function::vectorize(x, y);
  
  cout << "With v1 = x, integral: " << integralOverMesh(soln_functional, v1, x) << endl;
  cout << "With v2 = y, integral: " << integralOverMesh(soln_functional, v2, y) << endl;
  cout << "With tau1=(x,y), integral: " << integralOverMesh(soln_functional, tau1, xyVector) << endl;
  cout << "With tau2=(x,y), integral: " << integralOverMesh(soln_functional, tau2, xyVector) << endl;
  cout << "With q   =x, integral: " << integralOverMesh(soln_functional, q, x) << endl;
  
  cout << "(Expect 0s all around, except for q, where we expect (1,x) == 0.5.)\n";
  return 0;
}
