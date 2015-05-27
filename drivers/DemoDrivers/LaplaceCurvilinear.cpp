//
//  LaplaceCurvilinear.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "HConvergenceStudy.h"
#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"

#include "MeshFactory.h"

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank = mpiSession.getRank();

  int minPolyOrder = 1;
  int maxPolyOrder = 5;
  int pToAdd = 2;
  int minLogElements = 0;
  int maxLogElements = 3;
  bool useTriangles = false;
  double width = 10;
  double height = 10;
  double radius = 1;

  if (rank==0)
  {
    cout << "minPolyOrder: " << minPolyOrder << "\n";
    cout << "maxPolyOrder: " << maxPolyOrder << "\n";
    cout << "pToAdd: " << pToAdd << "\n";
  }

  VarFactoryPtr vf = VarFactory::varFactory();
  // trial variables:
  VarPtr phi = vf->fieldVar("\\phi");
  VarPtr psi1 = vf->fieldVar("\\psi_{1}");
  VarPtr psi2 = vf->fieldVar("\\psi_{2}");
  VarPtr phi_hat = vf->traceVar("\\widehat{\\phi}");
  VarPtr psi_hat_n = vf->fluxVar("\\widehat{\\psi}_n");
  // test variables
  VarPtr q = vf->testVar("q", HDIV);
  VarPtr v = vf->testVar("v", HGRAD);
  // bilinear form
  BFPtr bf = Teuchos::rcp( new BF(vf) );
  bf->addTerm(-phi, q->div());
  bf->addTerm(-psi1, q->x());
  bf->addTerm(-psi2, q->y());
  bf->addTerm(phi_hat, q->dot_normal());

  bf->addTerm(-psi1, v->dx());
  bf->addTerm(-psi2, v->dy());
  bf->addTerm(psi_hat_n, v);

  // define exact solution functions
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  FunctionPtr phi_exact = sin_x * sin_y;
  FunctionPtr psi1_exact = phi_exact->dx();
  FunctionPtr psi2_exact = phi_exact->dy();

  // set up BCs
  BCPtr bc = BC::bc();
  bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);

  // RHS
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  rhs->addTerm(f * v);

  // exact solution object
  Teuchos::RCP<ExactSolution> exactSolution = Teuchos::rcp( new ExactSolution(bf,bc,rhs) );
  exactSolution->setSolutionFunction(phi, phi_exact);
  exactSolution->setSolutionFunction(psi1, psi1_exact );
  exactSolution->setSolutionFunction(psi2, psi2_exact );

  // inner product
  IPPtr ip = bf->graphNorm();

  if (rank==0)
  {
    cout << "Laplace bilinear form:\n";
    bf->printTrialTestInteractions();
  }

  for (int polyOrder=minPolyOrder; polyOrder<=maxPolyOrder; polyOrder++)
  {
    HConvergenceStudy study(exactSolution,
                            bf, rhs, bc, ip,
                            minLogElements, maxLogElements,
                            polyOrder+1, pToAdd, false, useTriangles, false);

    bool useHemkerMesh = true;
    if (useHemkerMesh)
    {
      study.solve(MeshFactory::hemkerGeometry(width, height, radius));
    }
    else
    {
      if (rank==0)
        cout << "TEST: just using a quad mesh\n;";
      // just a quad
      FieldContainer<double> quadPoints(4,2);
      quadPoints(0,0) = -width / 2;
      quadPoints(0,1) = -height / 2;
      quadPoints(1,0) =  width / 2;
      quadPoints(1,1) = -height / 2;
      quadPoints(2,0) =  width / 2;
      quadPoints(2,1) =  height / 2;
      quadPoints(3,0) = -width / 2;
      quadPoints(3,1
                ) =  height / 2;

      study.solve(quadPoints);
    }

    if (rank==0)
    {
      cout << study.TeXErrorRateTable();
      cout << "******** Best Approximation comparison: ********\n";
      vector<int> primaryVariables;
      primaryVariables.push_back(phi->ID());
      primaryVariables.push_back(psi1->ID());
      primaryVariables.push_back(psi2->ID());
      cout << study.TeXBestApproximationComparisonTable(primaryVariables);

      for (int i=minLogElements; i<=maxLogElements; i++)
      {
        ostringstream filePath;
        filePath << "/tmp/hemkerMeshLevel" << i << ".dat";
        GnuPlotUtil::writeComputationalMeshSkeleton(filePath.str(), study.getSolution(i)->mesh());
      }
    }
  }
}
