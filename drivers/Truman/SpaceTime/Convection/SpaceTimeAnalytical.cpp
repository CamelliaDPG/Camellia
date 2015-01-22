//  SimpleConvection.cpp
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"
#include "CheckConservation.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

int H1Order = 3, pToAdd = 2;
double pi = 2.0*acos(0.0);

class IPScaling : public hFunction {
  double _epsilon;
  public:
  IPScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    return min(1./sqrt(_epsilon),1./h);
  }
};

// class EpsilonScaling : public hFunction {
//   double _epsilon;
//   public:
//   EpsilonScaling(double epsilon) {
//     _epsilon = epsilon;
//   }
//   double value(double x, double y, double h) {
//     double scaling = min(_epsilon/(h*h), 1.0);
//     // since this is used in inner product term a like (a,a), take square root
//     return sqrt(scaling);
//   }
// };

class ConstantXBoundary : public SpatialFilter {
   private:
      double val;
   public:
      ConstantXBoundary(double val): val(val) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-val) < tol);
      }
};

class ConstantYBoundary : public SpatialFilter {
   private:
      double val;
   public:
      ConstantYBoundary(double val): val(val) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(y-val) < tol);
      }
};

class UExact : public Function {
  double _eps;
  int _returnID;
  public:
  UExact(double eps) : Function(0) {
    _eps = eps;
    _returnID = 0;
  }
  UExact(double eps,int trialID) : Function(0) {
    _eps = eps;
    _returnID = trialID;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {

    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double t = (*points)(cellIndex,ptIndex,1);

        double l = 3;
        double lambda1 = (-1.+sqrt(1.-4.*_eps*l))/(-2*_eps);
        double lambda2 = (-1.-sqrt(1.-4.*_eps*l))/(-2*_eps);

        double u = exp(-l*t)*(exp(lambda1*(x-1))-exp(lambda2*(x-1)));
        double sigma = _eps*exp(-l*t)*(lambda1*exp(lambda1*(x-1))-lambda2*exp(lambda2*(x-1)));
        if (_returnID==0){
          values(cellIndex,ptIndex) = u;
        }
        else if (_returnID==1){
          values(cellIndex,ptIndex) = sigma;
        }
      }
    }
  }
};

enum NormType
{
  Graph,
  Robust,
  RobustL2Scaling,
  SpaceTimeRobust,
  SpaceTimeRobustL2Scaling,
  DecoupledRobust,
  MinMaxDecoupled,
  MinMaxL2Scaling,
  NSDecoupled,
  MinNSDecoupled,
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

  // Optional arguments (have defaults)
  string normString = args.Input<string>("--norm", "test norm", "Robust");
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  double epsilon = args.Input<double>("--epsilon", "diffusion parameter");
  args.Process();

  map<string, NormType> stringToNorm;
  stringToNorm["Graph"] = Graph;
  stringToNorm["Robust"] = Robust;
  stringToNorm["RobustL2Scaling"] = RobustL2Scaling;
  stringToNorm["SpaceTimeRobust"] = SpaceTimeRobust;
  stringToNorm["SpaceTimeRobustL2Scaling"] = SpaceTimeRobustL2Scaling;
  stringToNorm["DecoupledRobust"] = DecoupledRobust;
  stringToNorm["MinMaxDecoupled"] = MinMaxDecoupled;
  stringToNorm["MinMaxL2Scaling"] = MinMaxL2Scaling;
  stringToNorm["NSDecoupled"] = NSDecoupled;
  stringToNorm["MinNSDecoupled"] = MinNSDecoupled;
  NormType norm = stringToNorm[normString];

  int numX = 4;
  int numY = 4;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  // VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", L2);
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  double xmin = 0.0;
  double xmax = 1.0;
  double tmax = 1.0;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  tmax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  tmax;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, numX, numY,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // tau terms:
  bf->addTerm( sigma/epsilon, tau );
  bf->addTerm( u, tau->dx() );
  bf->addTerm( -uhat, tau->times_normal_x() );

  // v terms:
  bf->addTerm( -u, v->dx() );
  bf->addTerm( sigma, v->dx() );
  bf->addTerm( -u, v->dy() );
  bf->addTerm( fhat, v);

  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  // FunctionPtr f = Teuchos::rcp( new Forcing );
  // rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  // FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) );
  FunctionPtr ip_scaling = Teuchos::rcp( new IPScaling(epsilon) );
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  switch (norm)
  {
    // Automatic graph norm
    // Not robust
    case Graph:
    ip = bf->graphNorm();
    break;

    // Original Coupled Robust
    // Actually Robust
    case Robust:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( v->dx() );
    ip->addTerm( v );
    break;

    // Original Coupled Robust with full L2 scaling
    // 
    case RobustL2Scaling:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( v->dx() );
    ip->addTerm( sqrt(epsilon)*ip_scaling*v );
    break;

    // Space-time gradient
    // Robust
    case SpaceTimeRobust:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( v );
    break;

    // Space-time gradient with full L2 scaling
    case SpaceTimeRobustL2Scaling:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( sqrt(epsilon)*ip_scaling*v );
    break;

    // Decoupled space-time gradient
    // Almost robust
    case DecoupledRobust:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() );
    ip->addTerm( v );
    break;

    // Decoupled min max scaling
    case MinMaxDecoupled:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( 1./ip_scaling * v->dx() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() );
    ip->addTerm( v );
    break;

    // Min max scaling with full L2 scaling
    case MinMaxL2Scaling:
    ip->addTerm( ip_scaling * tau );
    ip->addTerm( 1./ip_scaling * v->dx() );
    ip->addTerm( tau->dx() - beta * v->grad() );
    ip->addTerm( v->dx() );
    ip->addTerm( sqrt(epsilon)*ip_scaling*v );
    break;

    // NS Decoupled
    // Not Robust
    case NSDecoupled:
    ip->addTerm( 1./epsilon * tau );
    ip->addTerm( v->dx() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() );
    ip->addTerm( v );
    break;

    // NS Decoupled with rescaled constitutive
    case MinNSDecoupled:
    ip->addTerm( ip_scaling*ip_scaling * tau );
    ip->addTerm( v->dx() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() );
    ip->addTerm( v );
    break;

    default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid inner product");
  }

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(0) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(tmax) );
  FunctionPtr u_exact = Teuchos::rcp( new UExact(epsilon, 0) );
  FunctionPtr sigma_exact = Teuchos::rcp( new UExact(epsilon, 1) );
  bc->addDirichlet(uhat, right, zero);
  bc->addDirichlet(fhat, left, -u_exact+sigma_exact);
  bc->addDirichlet(fhat, bottom, -u_exact);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);

  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "analytical_conv_" << epsilon <<".txt";
  if (commRank == 0)
    convOut.open(convOutFile.str().c_str());

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
     solution->solve(false);

     FunctionPtr u_soln = Function::solution(u, solution);
     FunctionPtr sigma_soln = Function::solution(sigma, solution);
     FunctionPtr u_diff = (u_soln - u_exact);
     FunctionPtr u_sqr = u_diff*u_diff;
     FunctionPtr sigma_diff = (sigma_soln - sigma_exact);
     FunctionPtr sigma_sqr = sigma_diff*sigma_diff;
     double L2_error_u = u_sqr->integrate(mesh, 5);
     double L2_error_sigma = sigma_sqr->integrate(mesh, 5);
     double L2_error = sqrt(L2_error_u + L2_error_sigma);
     double energy_error = solution->energyErrorTotal();

     if (commRank==0){
      stringstream outfile;
      stringstream errfile;
      outfile << "analytical_" << refIndex;
      errfile << "analytical_error_" << refIndex;
      // exporter.exportSolution(outfile.str());
      // exporter.exportFunction(u_diff, errfile.str());
      // exporter.exportFunction(u_exact, "analytical_exact");

      // Check local conservation
      // FunctionPtr flux = Function::solution(fhat, solution);
      // Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, mesh, 0);
      // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
      // << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;

      // convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << " "
      // << fluxImbalances[0] << " " << fluxImbalances[1] << " " << fluxImbalances[2] << endl;
      convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << endl;
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }
  // if (commRank == 0)
  //   convOut.close();

  return 0;
}

