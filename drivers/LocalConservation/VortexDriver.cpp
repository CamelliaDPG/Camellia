//  ConfusionDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

// Nate's addition because he doesn't want to Truman's build system:
#include "CamelliaConfig.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CheckConservation.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class ZeroMeanScaling : public hFunction {
  public:
  double value(double x, double y, double h) {
    return 1.0/(h*h);
  }
};

class InflowSquareBoundary : public SpatialFilter {
  FunctionPtr _beta;
  public:
  InflowSquareBoundary(FunctionPtr beta){
    _beta = beta;
  }
  // bool matchesPoint(double x, double y) {
  //   return true;
  // }
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());    
    const FieldContainer<double> *normals = &(basisCache->getSideNormals());
    int numCells = (*points).dimension(0);
    int numPoints = (*points).dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints,2);
    _beta->values(beta_pts,basisCache);

    double tol=1e-14;
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double n1 = (*normals)(cellIndex,ptIndex,0);
        double n2 = (*normals)(cellIndex,ptIndex,1);
        double beta_n = beta_pts(cellIndex,ptIndex,0)*n1 + beta_pts(cellIndex,ptIndex,1)*n2 ;
        // cout << "beta = (" << beta_pts(cellIndex,ptIndex,0)<<", "<<
        //   beta_pts(cellIndex,ptIndex,1)<<"), n = ("<<n1<<", "<<n2<<")"<<endl;
        pointsMatch(cellIndex,ptIndex) = false;
        if (beta_n < 0){
          pointsMatch(cellIndex,ptIndex) = true;
          somePointMatches = true;
        }
      }
    }
    return somePointMatches;
  }
};

// inflow values for u
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
          double r = sqrt(x*x + y*y); 
          values(cellIndex,ptIndex) = (r-1.0)/(sqrt(2)-1);
          // values(cellIndex,ptIndex) = 1.0;
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
          values(cellIndex,ptIndex,0) = -y;
          values(cellIndex,ptIndex,1) = x;
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  // Required arguments
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  bool enforceLocalConservation = args.Input<bool>("--conserve", "enforce local conservation");
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = robust\n    2 = modified robust");

  // Optional arguments (have defaults)
  double epsilon = args.Input("--epsilon", "diffusion parameter", 1e-4);
  bool zeroL2 = args.Input("--zeroL2", "take L2 term on v in robust norm to zero", false);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("u_hat");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("t_hat");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
  
  FunctionPtr beta = Teuchos::rcp(new Beta());
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  bf->addTerm(sigma / epsilon, tau);
  bf->addTerm(u, tau->div());
  bf->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  bf->addTerm( sigma, v->grad() );
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  if (norm == 0)
  {
    ip = bf->graphNorm();
    FunctionPtr h2_scaling = Teuchos::rcp( new ZeroMeanScaling ); 
    ip->addZeroMeanTerm( h2_scaling*v );
  }
  // Robust norm
  else if (norm == 1)
  {
    // robust test norm
    FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
    FunctionPtr h2_scaling = Teuchos::rcp( new ZeroMeanScaling ); 
    if (!zeroL2)
      ip->addTerm( v );
    ip->addTerm( sqrt(epsilon) * v->grad() );
    // Weight these two terms for inflow
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->div() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
    if (zeroL2)
      ip->addZeroMeanTerm( h2_scaling*v );
  }
  // Modified robust norm
  else if (norm == 2)
  {
    // robust test norm
    FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
    FunctionPtr h2_scaling = Teuchos::rcp( new ZeroMeanScaling ); 
    // FunctionPtr ip_weight = Teuchos::rcp( new IPWeight() );
    if (!zeroL2)
      ip->addTerm( v );
    ip->addTerm( sqrt(epsilon) * v->grad() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->div() - beta*v->grad() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
    if (zeroL2)
      ip->addZeroMeanTerm( h2_scaling*v );
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  // SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary );
  // bc->addDirichlet(uhat, outflowBoundary, u0);
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u0);
  // bc->addDirichlet(uhat, inflowBoundary, u0);
  
  // Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  // pc->addConstraint(uhat==u0,inflowBoundary);

  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) = -1.0; // x1
  meshBoundary(0,1) = -1.0; // y1
  meshBoundary(1,0) =  1.0;
  meshBoundary(1,1) = -1.0;
  meshBoundary(2,0) =  1.0;
  meshBoundary(2,1) =  1.0;
  meshBoundary(3,0) = -1.0;
  meshBoundary(3,1) =  1.0;
  
  int horizontalCells = 4, verticalCells = 4;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd, false);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);
  ofstream errOut;
  if (commRank == 0)
    errOut.open("vortex_err.txt");
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->solve(false);

    double energy_error = solution->energyErrorTotal();
    if (commRank==0){
      stringstream outfile;
      outfile << "vortex_" << refIndex;
      exporter.exportSolution(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;

      errOut << mesh->numGlobalDofs() << " " << energy_error << " "
        << fluxImbalances[0] << " " << fluxImbalances[1] << " " << fluxImbalances[2] << endl;
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }
  if (commRank == 0)
    errOut.close();
  
  return 0;
}
