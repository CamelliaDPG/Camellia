/*
 *  StokesStudyForLeszek.cpp
 *
 *  Created by Nathan Roberts on 1/11/14.
 *  Original version © 2014 Nathan V. Roberts.
 *
 */

#include "HConvergenceStudy.h"
#include "InnerProductScratchPad.h"
#include "GnuPlotUtil.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>

using namespace std;

class NorthOrEastBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0) < tol); // East
    bool yMatch = (abs(y-1.0) < tol); // North
    return xMatch || yMatch;
  }
};

class SouthOrWestBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol); // West
    bool yMatch = (abs(y) < tol); // South
    return xMatch || yMatch;
  }
};

int main(int argc, char *argv[]) {
  int rank = 0, numProcs = 1;
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
#endif
  rank = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  int pToAdd = 2; // for optimal test function approximation
  bool computeRelativeErrors = false; // we'll say false when one of the exact solution components is 0
  
  int polyOrder = 0;
  int minLogElements = 0;
  int maxLogElements = 6;
  
  bool useTriangles = true;
  
  bool useConformingTraces = true;
  
  if (rank == 0) {
    cout << "polyOrder = " << polyOrder << endl;
    cout << "pToAdd = " << pToAdd << endl;
    cout << "minLogElements = " << minLogElements << endl;
    cout << "maxLogElements = " << maxLogElements << endl;
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useConformingTraces = " << (useConformingTraces ? "true" : "false") << endl;
  }
  
  VarFactory varFactory;
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_{11}");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_{12}");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_{22}");
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr t1 = varFactory.fluxVar("t_1");
  VarPtr t2 = varFactory.fluxVar("t_2");
  
  VarPtr tau11 = varFactory.testVar("\\tau_{11}", HGRAD);
  VarPtr tau12 = varFactory.testVar("\\tau_{12}", HGRAD);
  VarPtr tau22 = varFactory.testVar("\\tau_{22}", HGRAD);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  bf->addTerm(0.5 * sigma11 - 0.5 * sigma22, tau11);
  bf->addTerm(u1, tau11->dx());
  
  bf->addTerm(2 * sigma12, tau12);
  bf->addTerm(u1, tau12->dy());
  bf->addTerm(u2, tau12->dx());
  
  bf->addTerm(0.5 * sigma22 - 0.5 * sigma11, tau22);
  bf->addTerm(u2, tau22->dy());
  
  bf->addTerm(-sigma11, v1->dx());
  bf->addTerm(-sigma12, v1->dy());
  
  bf->addTerm(-sigma12, v2->dx());
  bf->addTerm(-sigma22, v2->dy());
  
  FunctionPtr n = Function::normal();
  FunctionPtr n1 = n->x();
  FunctionPtr n2 = n->y();
  
  bf->addTerm(-u1hat, tau11 * n1 + tau12 * n2);
  bf->addTerm(-u2hat, tau12 * n1 + tau22 * n2);
  
  bf->addTerm(t1, v1);
  bf->addTerm(t2, v2);
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr zero = Function::zero();
  
  FunctionPtr u1_exact = zero;
  FunctionPtr u2_exact = zero;
  FunctionPtr sigma11_exact = zero;
  FunctionPtr sigma12_exact = x * x;
  FunctionPtr sigma22_exact = zero;
  
  FunctionPtr t1_exact = sigma11_exact * n1 + sigma12_exact * n2;
  FunctionPtr t2_exact = sigma12_exact * n1 + sigma22_exact * n2;
  
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr northEast = Teuchos::rcp( new NorthOrEastBoundary );
  SpatialFilterPtr southWest = Teuchos::rcp( new SouthOrWestBoundary );
  
  bc->addDirichlet(u1hat, southWest, u1_exact);
  bc->addDirichlet(u2hat, southWest, u2_exact);
//  bc->addDirichlet(u1hat, northEast, u1_exact);
//  bc->addDirichlet(u2hat, northEast, u2_exact);
  
  bc->addDirichlet(t1, northEast, t1_exact);
  bc->addDirichlet(t2, northEast, t2_exact);

  int H1OrderOfExactSolution = 3;
  
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp(new RHSEasy );
  FunctionPtr f1 = 0.5 * sigma11_exact - 0.5 * sigma22_exact - u1_exact->dx();
  FunctionPtr f2 = 2 * sigma12_exact - u1_exact->dy() - u2_exact->dx();
  FunctionPtr f3 = 0.5 * sigma22_exact - 0.5 * sigma11_exact - u2_exact->dy();
  FunctionPtr f4 = sigma11_exact->dx() + sigma12_exact->dy();
  FunctionPtr f5 = sigma12_exact->dx() + sigma22_exact->dy();
  rhs->addTerm(f1 * tau11 + f2 * tau12 + f3 * tau22 + f4 * v1 + f5 * v2);
  
  Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(bf,bc,rhs, H1OrderOfExactSolution ) );
  
  mySolution->setSolutionFunction(u1, u1_exact);
  mySolution->setSolutionFunction(u2, u2_exact);
  mySolution->setSolutionFunction(sigma11, sigma11_exact);
  mySolution->setSolutionFunction(sigma12, sigma12_exact);
  mySolution->setSolutionFunction(sigma22, sigma22_exact);
  
  double beta = 0.1; // weight for L2 terms in graph norm
  IPPtr graphNorm = bf->graphNorm(beta);
  
  if (rank==0)
    graphNorm->printInteractions();

  
  FieldContainer<double> quadPoints(4,2); // NOTE: quadPoints unused for HDGSingular (there, we set the mesh more manually)
  
  quadPoints(0,0) = 0; // x1
  quadPoints(0,1) = 0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0;
  quadPoints(3,1) = 1.0;
  
  HConvergenceStudy study(mySolution,
                          mySolution->bilinearForm(),
                          mySolution->ExactSolution::rhs(),
                          bc, graphNorm,
                          minLogElements, maxLogElements,
                          polyOrder+1, pToAdd, false, useTriangles, false);
  study.setUseCondensedSolve(false);
  study.setReportRelativeErrors(computeRelativeErrors);
  
//  study.setWriteGlobalStiffnessToDisk(true, "stokeStudyForLeszek");
  
  int maxTestDegree = polyOrder + 1 + pToAdd;
  int cubatureDegreeInMesh = polyOrder + maxTestDegree;
  
  if (useTriangles) {
    int INTREPID_CUBATURE_TRI_DEFAULT_MAX_ENUM = 20;
    int cubEnrichment = INTREPID_CUBATURE_TRI_DEFAULT_MAX_ENUM - cubatureDegreeInMesh;
    study.setCubatureDegreeForExact(cubEnrichment);
  }
  
  study.solve(quadPoints,useConformingTraces);

  // test that the computed loads agree with the exact solution sigma11 = x:
  // f1, f3, f4 = 0; f2 = 2x; f5 = 1
  
  MeshPtr minMesh = study.getSolution(minLogElements)->mesh();
  double f1Integral = f1->integrate(minMesh);
  double f2Integral = (f2-2*x)->integrate(minMesh);
  double f3Integral = f3->integrate(minMesh);
  double f4Integral = f4->integrate(minMesh);
  double f5Integral = (f5-Function::constant(1.0))->integrate(minMesh);
  if (rank == 0) {
    cout << "Errors in load on minimal mesh:\n";
    cout << "f1: " << f1Integral << endl;
    cout << "f2: " << f2Integral << endl;
    cout << "f3: " << f3Integral << endl;
    cout << "f4: " << f4Integral << endl;
    cout << "f5: " << f5Integral << endl;
  }
  
  vector<int> primaryVariables;
  primaryVariables.push_back(u1->ID());
  primaryVariables.push_back(u2->ID());
  primaryVariables.push_back(sigma11->ID());
  primaryVariables.push_back(sigma12->ID());
  primaryVariables.push_back(sigma22->ID());
  
  if (rank == 0) {
    cout << study.TeXErrorRateTable();
    vector<int> fieldIDs,traceIDs;
    cout << "******** Best Approximation comparison: ********\n";
    cout << study.TeXBestApproximationComparisonTable(primaryVariables);
    
    ostringstream filePathPrefix;
    filePathPrefix << "stokes/" << "leszek_p" << polyOrder;
    study.TeXBestApproximationComparisonTable(primaryVariables,filePathPrefix.str());
  }
  
  if (rank==0) {
    GnuPlotUtil::writeComputationalMeshSkeleton("stokesStudyFinalMesh", study.getSolution(maxLogElements)->mesh(), false);
  }
  
#ifdef USE_VTK
  {
    SolutionPtr firstSolution = study.getSolution(minLogElements);
    VTKExporter exporter(firstSolution, firstSolution->mesh(), varFactory);
    exporter.exportSolution("stokesStudyLeszekFirstSolution", (polyOrder+1)*2);
    if (rank==0) {
      GnuPlotUtil::writeComputationalMeshSkeleton("stokesStudyInitialMesh", firstSolution->mesh(), false);
    }
  }
  {
    SolutionPtr lastSolution = study.getSolution(maxLogElements);
    VTKExporter exporter(lastSolution, lastSolution->mesh(), varFactory);
    exporter.exportSolution("stokesStudyLeszekFinalSolution", (polyOrder+1)*2);
    if (rank==0) {
      GnuPlotUtil::writeComputationalMeshSkeleton("stokesStudyFinalMesh", lastSolution->mesh(), false);
    }
  }
#endif
}