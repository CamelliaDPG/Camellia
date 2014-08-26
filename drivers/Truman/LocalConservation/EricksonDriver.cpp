//  ConfusionDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 7/12/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CheckConservation.h"
#include "HDF5Exporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);
double epsilon;

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
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

class LeftBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(x) < tol);
  }
};

class RightBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(x-1.0) < tol);
  }
};

class TopBottomBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool topMatch = (abs(y-1.0) < tol);
    bool bottomMatch = (abs(y) < tol);
    return topMatch || bottomMatch;
  }
};

// boundary value for u
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
        double y = (*points)(cellIndex,ptIndex,1);

        double C0 = 0.0;// average of u0
        double u = C0;
        double u_x = 0.0;
        double u_y = 0.0;  	
        bool useDiscontinuous = false; // use discontinuous soln
        int numTerms = 20;
        if (!useDiscontinuous)
          numTerms = 1;
        for (int n = 1;n<numTerms+1;n++){

          double lambda = n*n*pi*pi*_eps;
          double d = sqrt(1.0+4.0*_eps*lambda);
          double r1 = (1.0+d)/(2.0*_eps);
          double r2 = (1.0-d)/(2.0*_eps);

          double Cn = 0.0;            
          if (!useDiscontinuous){
            if (n==1){
              Cn = 1.0; // first term only
            } 	  
          }else{
            // discontinuous hat 
            Cn = -1 + cos(n*pi/2)+.5*n*pi*sin(n*pi/2) + sin(n*pi/4)*(n*pi*cos(n*pi/4)-2*sin(3*n*pi/4));
            Cn /= (n*pi);
            Cn /= (n*pi);    
          }


          // normal stress outflow
          double Xbottom;
          double Xtop;
          double dXtop;
          // wall, zero outflow
          Xtop = (exp(r2*(x-1))-exp(r1*(x-1)));
          Xbottom = (exp(-r2)-exp(-r1));
          dXtop = (exp(r2*(x-1))*r2-exp(r1*(x-1))*r1);    

          double X = Xtop/Xbottom;
          double dX = dXtop/Xbottom;
          double Y = Cn*cos(n*pi*y);
          double dY = -Cn*n*pi*sin(n*pi*y);

          u += X*Y;
          u_x += _eps * dX*Y;
          u_y += _eps * X*dY;
        }
        if (_returnID==0){
          values(cellIndex,ptIndex) = u;
        }
        else if (_returnID==1){
          values(cellIndex,ptIndex) = u_x;
        }
        else if (_returnID==2){
          values(cellIndex,ptIndex) = u_y;
        }
      }
    }
  }
};
// class UExact : public Function {
//   public:
//     UExact() : Function(0) {}
//     void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
//       int numCells = values.dimension(0);
//       int numPoints = values.dimension(1);
//       double lambda_n = pi*pi*epsilon;
//       double r1 = (1+sqrt(1+4*epsilon*lambda_n))/(2*epsilon);
//       double r2 = (1-sqrt(1+4*epsilon*lambda_n))/(2*epsilon);
// 
//       const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
//       double tol=1e-14;
//       for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//         for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//           double x = (*points)(cellIndex,ptIndex,0);
//           double y = (*points)(cellIndex,ptIndex,1);
//           values(cellIndex, ptIndex) = (exp(r2*(x-1)-exp(r1*(x-1))))
//             *cos(pi*y)/(r1*exp(-r2)-r2*exp(-r1));
//         }
//       }
//     }
// };

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
  epsilon = args.Input<double>("--epsilon", "diffusion parameter");
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  bool enforceLocalConservation = args.Input<bool>("--conserve", "enforce local conservation");
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = robust\n    2 = modified robust");

  // Optional arguments (have defaults)
  bool zeroL2 = args.Input("--zeroL2", "take L2 term on v in robust norm to zero", true);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("fhat");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
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
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  SpatialFilterPtr rBoundary = Teuchos::rcp( new RightBoundary );
  SpatialFilterPtr tbBoundary = Teuchos::rcp( new TopBottomBoundary );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  FunctionPtr u_exact = Teuchos::rcp( new UExact(epsilon, 0) );
  FunctionPtr sx_exact = Teuchos::rcp( new UExact(epsilon, 1) );
  FunctionPtr sy_exact = Teuchos::rcp( new UExact(epsilon, 2) );
  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;
  FunctionPtr sigma_exact = sx_exact*e1 + sy_exact*e2;
  FunctionPtr u_r = Function::zero();
  bc->addDirichlet(beta_n_u_minus_sigma_n, lBoundary, beta*n*u_exact-sigma_exact*n);
  bc->addDirichlet(beta_n_u_minus_sigma_n, tbBoundary, beta*n*u_exact-sigma_exact*n);
  bc->addDirichlet(uhat, rBoundary, u_r);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) = 0.0; // x1
  meshBoundary(0,1) = 0.0; // y1
  meshBoundary(1,0) = 1.0;
  meshBoundary(1,1) = 0.0;
  meshBoundary(2,0) = 1.0;
  meshBoundary(2,1) = 1.0;
  meshBoundary(3,0) = 0.0;
  meshBoundary(3,1) = 1.0;
  
  int H1Order = 3, pToAdd = 2;
  int horizontalCells = 2, verticalCells = 2;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.3; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  // VTKExporter exporter(solution, mesh, varFactory);
  HDF5Exporter exporter(mesh, "Erikkson");

  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "erickson_conv_" << epsilon <<".txt";
  if (commRank == 0)
    convOut.open(convOutFile.str().c_str());
  // u_exact->writeValuesToMATLABFile(mesh, "u_exact.m");
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->solve(false);


    FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
    FunctionPtr sigma_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma) );
    FunctionPtr u_diff = (u_soln - u_exact);
    FunctionPtr u_sqr = u_diff*u_diff;
    FunctionPtr sigma_diff = (sigma_soln - sigma_exact);
    FunctionPtr sigma_sqr = sigma_diff*sigma_diff;
    double L2_error_u = u_sqr->integrate(mesh, 4);
    double L2_error_sigma = sigma_sqr->integrate(mesh, 4);
    double L2_error = sqrt(L2_error_u + L2_error_sigma);
    double energy_error = solution->energyErrorTotal();

    if (commRank==0){
      stringstream outfile;
      stringstream errfile;
      outfile << "erickson_" << refIndex;
      errfile << "erickson_error_" << refIndex;
      exporter.exportSolution(solution, varFactory, refIndex, 2, cellIDToSubdivision(mesh, 4));
      // exporter.exportSolution(outfile.str());
      // exporter.exportFunction(u_diff, errfile.str());
      // exporter.exportFunction(u_exact, "erickson_exact");
      // solution->writeToVTK(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, mesh, 0);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;

      convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << " "
        << fluxImbalances[0] << " " << fluxImbalances[1] << " " << fluxImbalances[2] << endl;
    }

    if (refIndex < numRefs)
    {
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
      // refinementStrategy.hRefineUniformly(mesh);
    }
  }
  if (commRank == 0)
    convOut.close();
  
  return 0;
}
