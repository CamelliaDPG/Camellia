//  VirtualConfusionDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"

#include "Solution.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "Virtual.h"

class EntireBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    return true;
  }
};

class UnitSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class InflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) ;
    bool yMatch = (abs(y) < tol) ;
    return xMatch || yMatch;
  }
};

class OutflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0) < tol);
    bool yMatch = (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

// boundary value for u
class U0 : public Function {
public:
  U0() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        // solution with a boundary layer (section 5.2 in DPG Part II)
        // for x = 1, y = 1: u = 0
        if ( ( abs(x-1.0) < tol ) || (abs(y-1.0) < tol ) ) {
          values(cellIndex,ptIndex) = 0;
        } else if ( abs(x) < tol ) { // for x=0: u = 1 - y
          values(cellIndex,ptIndex) = 1.0 - y;
        } else { // for y=0: u=1-x
          values(cellIndex,ptIndex) = 1.0 - x;
        }
        
      }
    }
  }
};

class Beta : public Function {
public:
  Beta() : Function(1) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d = 0; d < spaceDim; d++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex,0) = y;
          values(cellIndex,ptIndex,0) = -x;
        }
      }
    }
  }
};

string fileNameForRefinement(string fileName, int refinementNumber) {
  ostringstream fileNameStream;
  fileNameStream << fileName << "_r" << refinementNumber << ".dat";
  return fileNameStream.str();
}

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();

  bool enforceOneIrregularity = true;
  int numRefs = 10;
  
  int k = 2, delta_k = 2;
  
  // problem parameters:
  double eps = 1;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("eps", &eps, "epsilon");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  int H1Order = k+1;
  
  if (rank==0) {
    cout << "eps = " << eps << endl;
    cout << "numRefs = " << numRefs << endl;
    cout << "p = " << H1Order-1 << endl;
  }
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("\\sigma", HDIV_DISC);
  
  FunctionPtr n = Function::normal();
  FunctionPtr h = Function::h();
  
  VarPtr uhat = varFactory.traceVar("\\widehat{u}", eps * u);
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}", beta_const * n * u - sigma * n);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma / -eps, tau);
  confusionBF->addTerm(-u, tau->div());
  confusionBF->addTerm(uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma - beta_const * u, v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
//  qoptIP->addTerm((eps / h) * tau->div() - beta_const * v->grad() );
//  qoptIP->addTerm( tau / h + v->grad() );
//  qoptIP->addTerm( v / h );
  qoptIP->addTerm( tau->div() - beta_const * v->grad() );
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( v );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  bc->addDirichlet(uhat, outflowBoundary, u0);
  bc->addDirichlet(uhat, inflowBoundary, u0);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // create a pointer to a new, single-element mesh on the unit square:
  Teuchos::RCP<Mesh> standardMesh = MeshFactory::quadMeshMinRule(confusionBF, H1Order, delta_k);
  
  Virtual vem(delta_k);
  vem.addAssociation(v, u, uhat);
  vem.addAssociation(tau, sigma, beta_n_u_minus_sigma_n);

  // strong field term belonging to the v equation
  LinearTermPtr A_v = sigma->div() - beta_const * u->grad();
  vem.addEquation(A_v, beta_n_u_minus_sigma_n, v);
  
  // strong field term belonging to the tau equation
  LinearTermPtr A_tau = sigma / -eps + u->grad();
  vem.addEquation(A_tau, uhat, tau);
  
  // test norm stuff
  // term belonging to u:
  LinearTermPtr A_u = eps * tau->div() - beta_const * v->grad();
  // boundary term for IBP of A_u:
  LinearTermPtr C_u = eps * n * tau - (beta_const * n) * v;
  vem.addTestNormTerm(A_u, C_u);
  
  // term belonging to sigma:
  LinearTermPtr A_sigma  = tau + v->grad();
  LinearTermPtr C_sigma  = v * n;
  vem.addTestNormTerm(A_sigma, C_sigma);

  // stabilization term (since otherwise v enters the norm only through its gradient)
  LinearTermPtr A_stab = 1.0 * v;
  LinearTermPtr C_stab = Teuchos::rcp( new LinearTerm ); // zero
  vem.addTestNormTerm(A_stab, C_stab);
  
  BFPtr vemConfusionBF = Virtual::virtualBF(vem,varFactory);
  Teuchos::RCP<Mesh> vemMesh = MeshFactory::quadMeshMinRule(vemConfusionBF, H1Order, delta_k);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> standardSoln = Teuchos::rcp( new Solution(standardMesh, bc, rhs, qoptIP) );
  Teuchos::RCP<Solution> vemSoln = Teuchos::rcp( new Solution(vemMesh, bc, rhs, qoptIP) );
  
  double energyThreshold = 0.2; // for mesh refinements
  
  if (rank==0) {
    cout << "using RieszRep-based refinement strategy.\n";
  }
  Teuchos::RCP<RefinementStrategy> refinementStrategy, refinementStrategyVEM;
  LinearTermPtr residual = confusionBF->testFunctional(standardSoln) - rhs->linearTerm();
  refinementStrategy = Teuchos::rcp( new RefinementStrategy( standardMesh, residual, qoptIP, energyThreshold ) );
  
  refinementStrategy->setReportPerCellErrors(true);
  refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);

  LinearTermPtr residualVEM = confusionBF->testFunctional(vemSoln) - rhs->linearTerm();
  refinementStrategyVEM = Teuchos::rcp( new RefinementStrategy( vemMesh, residualVEM, qoptIP, energyThreshold ) );
  
  refinementStrategyVEM->setReportPerCellErrors(true);
  refinementStrategyVEM->setEnforceOneIrregularity(enforceOneIrregularity);

  for (int refIndex=0; refIndex<numRefs; refIndex++) {
    if (rank==0) cout << "Solving after " << refIndex << " refinements.\n";
    standardSoln->solve();
    vemSoln->solve();
    if (refIndex == numRefs-1) { // write out second-to-last mesh
      if (rank==0) {
        GnuPlotUtil::writeComputationalMeshSkeleton("standardConfusionMesh", standardMesh, true);
        GnuPlotUtil::writeComputationalMeshSkeleton("vemConfusionMesh", vemMesh, true);
      }
    }
    refinementStrategy->refine(rank==0); // print to console on rank 0
    refinementStrategyVEM->refine(rank==0);
  }
  // one more solve on the final refined mesh:
  standardSoln->solve();
  vemSoln->solve();
  
#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "standardConfusion";
  HDF5Exporter exporter(standardMesh,dir_name.str());
  exporter.exportSolution(standardSoln, varFactory, 0);
  if (rank==0) cout << "wrote solution to " << dir_name.str() << endl;
  dir_name.str("");
  dir_name << "vemConfusion";
  HDF5Exporter vemExporter(vemMesh,dir_name.str());
  vemExporter.exportSolution(vemSoln, varFactory, 0);
  if (rank==0) cout << "wrote vem solution to " << dir_name.str() << endl;
#endif
  
  return 0;
}
