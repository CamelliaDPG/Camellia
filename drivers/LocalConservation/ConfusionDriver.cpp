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
#include "VTKExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif


double epsilon = 1e-2;
double numRefs = 0;

bool enforceLocalConservation = true;

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

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
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

  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  // confusionBF->addTerm(sigma1 / epsilon, tau->x());
  // confusionBF->addTerm(sigma2 / epsilon, tau->y());
  confusionBF->addTerm(sigma / epsilon, tau);
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());

  // v terms:
  // confusionBF->addTerm( sigma1, v->dx() );
  // confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( sigma, v->grad() );
  confusionBF->addTerm( beta_const * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // mathematician's norm
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(tau);
  mathIP->addTerm(tau->div());

  mathIP->addTerm(v);
  mathIP->addTerm(v->grad());

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  qoptIP->addTerm( v );
  qoptIP->addTerm( tau / epsilon+ v->grad() );
  qoptIP->addTerm( beta_const * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  // robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon) * v->grad() );
  robIP->addTerm( beta_const * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( ip_scaling/sqrt(epsilon) * tau );
  if (enforceLocalConservation)
    robIP->addZeroMeanTerm( v );

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  bc->addDirichlet(uhat, outflowBoundary, u0);

  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta_const*n*u0);

  // Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  // pc->addConstraint(uhat==u0,inflowBoundary);

  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // Teuchos::RCP<Mesh> mesh = Mesh::readMsh("quad.msh", confusionBF, H1Order, pToAdd);
  // Teuchos::RCP<Mesh> mesh = Mesh::readTriangle(Camellia_MeshDir + "Quad/quad.1", confusionBF, H1Order, pToAdd);
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

  int horizontalCells = 1, verticalCells = 1;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      confusionBF, H1Order, H1Order+pToAdd, false);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }

  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);

  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->solve(false);

    if (rank==0){
      stringstream outfile;
      outfile << "confusion_" << refIndex;
      exporter.exportSolution(outfile.str());
      // solution->writeToVTK(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(rank==0); // print to console on rank 0
  }

  return 0;
}
