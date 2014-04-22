#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    if ( (abs(x) < _eps) ) { // top left
      return x / _eps;
    } else if ( abs(1.0-x) < _eps) { // top right
      return (1.0-x) / _eps;
    } else { // top middle
      return 1;
    }
  }
};

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();
  
  bool useMinRule = true;
  bool useMumps = false;
  
  VarFactory varFactory;
  // traces:
//  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
//  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1", L2); // switched to L2 just to isolate/debug
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2", L2);
  cout << "WARNING/NOTE: for debugging purposes, temporarily switching traces to use L^2 discretizations (i.e. they are not conforming, and they are of lower order than they should be).\n";
  VarPtr t1_n = varFactory.fluxVar("\\widehat{t}_{1n}");
  VarPtr t2_n = varFactory.fluxVar("\\widehat{t}_{2n}");
  // fields:
  VarPtr u1 = varFactory.fieldVar("u_1", L2);
  VarPtr u2 = varFactory.fieldVar("u_2", L2);
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2", VECTOR_L2);
  VarPtr p = varFactory.fieldVar("p");
  // test functions:
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr v1 = varFactory.testVar("v1", HGRAD);        // v_1
  VarPtr v2 = varFactory.testVar("v2", HGRAD);        // v_2
  VarPtr q = varFactory.testVar("q", HGRAD);          // q
  
  BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );
  double mu = 1.0; // viscosity
  // tau1 terms:
  stokesBF->addTerm(u1, tau1->div());
  stokesBF->addTerm(sigma1, tau1); // (sigma1, tau1)
  stokesBF->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBF->addTerm(u2, tau2->div());
  stokesBF->addTerm(sigma2, tau2);
  stokesBF->addTerm(-u2hat, tau2->dot_normal());
  
  // v1:
  stokesBF->addTerm(mu * sigma1, v1->grad()); // (mu sigma1, grad v1)
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( t1_n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma2, v2->grad()); // (mu sigma2, grad v2)
  stokesBF->addTerm( - p, v2->dy());
  stokesBF->addTerm( t2_n, v2);
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  FunctionPtr n = Function::normal();
  stokesBF->addTerm(u1hat * n->x() + u2hat * n->y(), q);

  int k = 2; // poly order for field variables
  int H1Order = k + 1;
  int delta_k = 2;   // test space enrichment
  double width = 1.0, height = 1.0;
  int horizontalCells = 2, verticalCells = 1;
  MeshPtr mesh = useMinRule ? MeshFactory::quadMeshMinRule(stokesBF, H1Order, delta_k, width, height,
                                                           horizontalCells, verticalCells)
                            : MeshFactory::quadMesh(stokesBF, H1Order, delta_k, width, height,
                                                    horizontalCells, verticalCells);
  
  RHSPtr rhs = RHS::rhs(); // zero
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr topBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);
  
  // top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(1.0/2) );
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);
  
  // everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);
  
  bc->addZeroMeanConstraint(p);

  IPPtr graphNorm = stokesBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  double energyThreshold = 0.20;
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  refinementStrategy.setReportPerCellErrors(true);
  
  Teuchos::RCP<Solver> solver;
  if (useMumps) {
    solver = Teuchos::rcp( new MumpsSolver );
  } else {
    solver = Teuchos::rcp( new KluSolver );
  }
  solution->solve(solver);
  int refCount = 3;
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    double energyError = solution->energyErrorTotal();
    if (rank==0) {
      cout << "Before refinement " << refIndex << ", energy error = " << energyError;
      cout << " (using " << mesh->numFluxDofs() << " trace degrees of freedom)." << endl;
    }
    refinementStrategy.refine(rank==0);
    solution->solve(solver);
  }
  double energyErrorTotal = solution->energyErrorTotal();
  
  FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y()) );
  double netMassFlux = massFlux->integrate(mesh); // integrate over the mesh skeleton

  if (rank==0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numFluxDofs() << " trace dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "Net mass flux: " << netMassFlux << endl;
  }
  
  VTKExporter solnExporter(solution,mesh,varFactory);
  solnExporter.exportSolution("stokesCavityFlowSolution");
  
  GnuPlotUtil::writeComputationalMeshSkeleton("cavityFlowRefinedMesh", mesh);
  
  return 0;
}