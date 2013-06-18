//  ConfusionDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

// Nate's addition because he doesn't want to Truman's build system:
//#define Camellia_MeshDir string("/Users/nroberts/Documents/Camellia/meshes/")
#include "CamelliaConfig.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CheckConservation.h"
#include "SolutionExporter.h"
#include "StandardAssembler.h"
#include "SerialDenseWrapper.h"

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

class InflowBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x) < tol) ;
      bool yMatch = (abs(y) < tol) ;
      return xMatch || yMatch;
    }
};

class OutflowBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x-1.0) < tol);
      bool yMatch = (abs(y-1.0) < tol);
      return xMatch || yMatch;
    }
};

// boundary value for u
class SourceTerm : public Function {
  public:
    SourceTerm() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      double tol=1e-14;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (y > 2*x)
            values(cellIndex, ptIndex) = 1;
          else
            values(cellIndex, ptIndex) = -1;
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
  double epsilon = args.Input("--epsilon", "diffusion parameter", 1e-3);
  bool zeroL2 = args.Input("--zeroL2", "take L2 term on v in robust norm to zero", false);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("\\sigma", VECTOR_L2);

  vector<double> beta;
  beta.push_back(.5/sqrt(1.25));
  beta.push_back(1/sqrt(1.25));

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
  FunctionPtr f = Teuchos::rcp( new SourceTerm );
  rhs->addTerm( f * v ); 

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowBoundary );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, zero);

  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  bc->addDirichlet(uhat, outflowBoundary, zero);


  // Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  // pc->addConstraint(uhat==u0,inflowBoundary);

  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // Teuchos::RCP<Mesh> mesh = Mesh::readMsh("quad.msh", bf, H1Order, pToAdd);
  // Teuchos::RCP<Mesh> mesh = Mesh::readTriangle(Camellia_MeshDir + "Quad/quad.1", bf, H1Order, pToAdd);
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

  int horizontalCells = 4, verticalCells = 4;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      bf, H1Order, H1Order+pToAdd, false);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == f);
  }

  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  StandardAssembler assembler(solution);
  VTKExporter exporter(solution, mesh, varFactory);
  ofstream errOut;
  if (commRank == 0)
    errOut.open("discontinuous_err.txt");

  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    // // Loop through elements and check conditioning
    // vector<ElementPtr> elems = mesh->activeElements();
    // for (vector<ElementPtr>::iterator it = elems.begin(); it != elems.end(); ++it)
    // {
    //   ElementPtr elem = *it;
    //   FieldContainer<double> ipMatrix = assembler.getIPMatrix(elem);
    //   double cellMeasure = mesh->getCellMeasure(elem->cellID());
    //   double condNumber = SerialDenseWrapper::getMatrixConditionNumber(ipMatrix);
    //   cout << cellMeasure << " " << condNumber << endl;
    // }
    // cout << endl;

    // Solve
    solution->solve(false);

    double energy_error = solution->energyErrorTotal();
    if (commRank==0){
      stringstream outfile;
      outfile << "discontinuous_" << refIndex;
      exporter.exportSolution(outfile.str());
      // solution->writeToVTK(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, f, varFactory, mesh, 0);
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
