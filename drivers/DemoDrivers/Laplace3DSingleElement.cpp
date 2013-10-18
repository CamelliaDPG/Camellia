//
//  Laplace3DSingleElement.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 10/18/13.
//
//

/*
   The basic idea with this driver is just to exercise the part of Camellia's DPG apparatus
   that doesn't depend on either Solution or Mesh.  We do the bits that would depend on Solution
   and Mesh "manually" -- we use a DofOrdering to manage the dof indices, and invert the stiffness matrix
   with some serial matrix solver.
 */

#include "InnerProductScratchPad.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "ExactSolution.h"

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank = mpiSession.getRank();
  
  int minPolyOrder = 1;
  int maxPolyOrder = 5;
  int pToAdd = 2;
  double sideLength = 10;

  if (rank==0) {
    cout << "minPolyOrder: " << minPolyOrder << "\n";
    cout << "maxPolyOrder: " << maxPolyOrder << "\n";
    cout << "pToAdd: " << pToAdd << "\n";
  }
  
  VarFactory vf;
  // trial variables:
  VarPtr phi = vf.fieldVar("\\phi");
  VarPtr psi1 = vf.fieldVar("\\psi_{1}");
  VarPtr psi2 = vf.fieldVar("\\psi_{2}");
  VarPtr psi3 = vf.fieldVar("\\psi_{3}");
  VarPtr phi_hat = vf.traceVar("\\widehat{\\phi}");
  VarPtr psi_hat_n = vf.fluxVar("\\widehat{\\psi}_n");
  // test variables
  VarPtr q = vf.testVar("q", HDIV);
  VarPtr v = vf.testVar("v", HGRAD);
  // bilinear form
  BFPtr bf = Teuchos::rcp( new BF(vf) );
  bf->addTerm(-phi, q->div());
  bf->addTerm(-psi1, q->x());
  bf->addTerm(-psi2, q->y());
  bf->addTerm(-psi3, q->z());
  bf->addTerm(phi_hat, q->dot_normal());
  
  bf->addTerm(-psi1, v->dx());
  bf->addTerm(-psi2, v->dy());
  bf->addTerm(-psi3, v->dz());
  bf->addTerm(psi_hat_n, v);
  
  // define exact solution functions
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact = x * y * z;
  FunctionPtr psi1_exact = phi_exact->dx();
  FunctionPtr psi2_exact = phi_exact->dy();
  FunctionPtr psi3_exact = phi_exact->dz();
  
  // set up BCs
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);
  
  // RHS
  Teuchos::RCP< RHSEasy > rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  rhs->addTerm(f * v);
  
  // exact solution object
  Teuchos::RCP<ExactSolution> exactSolution = Teuchos::rcp( new ExactSolution(bf,bc,rhs) );
  exactSolution->setSolutionFunction(phi, phi_exact);
  exactSolution->setSolutionFunction(psi1, psi1_exact );
  exactSolution->setSolutionFunction(psi2, psi2_exact );
  exactSolution->setSolutionFunction(psi3, psi3_exact );
  
  // inner product
  IPPtr ip = bf->graphNorm();
  
  if (rank==0) {
    cout << "Laplace bilinear form:\n";
    bf->printTrialTestInteractions();
  }
  
  for (int polyOrder=minPolyOrder; polyOrder<=maxPolyOrder; polyOrder++) {
    FieldContainer<double> cubePoints(8,3);
    int ptIndex = 0;
    for (int i=0; i<2; i++) {
      double xValue = (i==0) ? -sideLength : sideLength;
      for (int j=0; j<2; j++) {
        double yValue = (j==0) ? -sideLength : sideLength;
        for (int k=0; k<2; k++) {
          double zValue = (k==0) ? -sideLength : sideLength;
          cubePoints(ptIndex,0) = xValue;
          cubePoints(ptIndex,1) = yValue;
          cubePoints(ptIndex,2) = zValue;
          ptIndex++;
        }
      }
    }
  }
  
  // Steps to solve:
  // 1. Define trial discretization
  // 2. Define test discretization
  // 3. Create ElementType for cube
  // 4. Create BasisCache for ElementType
  // 5. Mimic/copy Solution's treatment of stiffness and optimal test functions.
  // 6. Apply BCs
  // 7. Invert stiffness (using SerialDenseMatrixUtility, say)
  // 8. Create BasisSumFunctions to represent solution
  // 9. Check solution.

}