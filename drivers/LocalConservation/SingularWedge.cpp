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

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif


bool enforceLocalConservation = false;
double epsilon = 1e-1;
double numRefs = 20;
double halfwidth = 0.5;

typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<BF> BFPtr;

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

class Inflow: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x+halfwidth) < tol) ;
      return xMatch;
    }
};

class Outflow: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x-halfwidth) < tol) ;
      return xMatch;
    }
};

class LeadingWedge: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = abs(1./halfwidth*x-y) < tol;
      return match;
    }
};

class TrailingWedge: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = abs(1./halfwidth*x+y) < tol;
      return match;
    }
};

class Top: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool yMatch = (abs(y-halfwidth) < tol) ;
      return yMatch;
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
  VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("fhat");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("sigma1");
  VarPtr sigma2 = varFactory.fieldVar("sigma2");
  

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / epsilon, tau->x());
  confusionBF->addTerm(sigma2 / epsilon, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  if (!enforceLocalConservation)
    robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon) * v->grad() );
  robIP->addTerm( beta * v->grad() );
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
  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp( new PenaltyConstraints );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  SpatialFilterPtr inflow = Teuchos::rcp( new Inflow );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflow, zero);

  SpatialFilterPtr leadingWedge = Teuchos::rcp( new LeadingWedge );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  bc->addDirichlet(uhat, leadingWedge, one);

  SpatialFilterPtr trailingWedge = Teuchos::rcp( new TrailingWedge );
  // bc->addDirichlet(beta_n_u_minus_sigma_n, trailingWedge, beta*n*one);
  bc->addDirichlet(uhat, trailingWedge, one);

  SpatialFilterPtr top = Teuchos::rcp( new Top );
  // bc->addDirichlet(uhat, top, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, top, zero);

  SpatialFilterPtr outflow = Teuchos::rcp( new Outflow );
  pc->addConstraint(beta*uhat->times_normal() - beta_n_u_minus_sigma_n == zero, outflow);
  
  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // define nodes for mesh
  vector< FieldContainer<double> > vertices;
  FieldContainer<double> pt(2);
  vector< vector<int> > elementIndices;
  vector<int> q(4);
  vector<int> t(3);

  pt(0) = -halfwidth; pt(1) = -1;
  vertices.push_back(pt);
  pt(0) =  0;         pt(1) =  0;
  vertices.push_back(pt);
  pt(0) =  halfwidth; pt(1) = -1;
  vertices.push_back(pt);
  pt(0) =  halfwidth; pt(1) =  0;
  vertices.push_back(pt);
  pt(0) =  halfwidth; pt(1) =  halfwidth;
  vertices.push_back(pt);
  pt(0) =  0;         pt(1) =  halfwidth;
  vertices.push_back(pt);
  pt(0) = -halfwidth; pt(1) =  halfwidth;
  vertices.push_back(pt);
  pt(0) = -halfwidth; pt(1) =  0;
  vertices.push_back(pt);

  t[0] = 0; t[1] = 1; t[2] = 7;
  elementIndices.push_back(t);
  t[0] = 1; t[1] = 2; t[2] = 3;
  elementIndices.push_back(t);
  q[0] = 1; q[1] = 3; q[2] = 4; q[3] = 5;
  elementIndices.push_back(q);
  q[0] = 7; q[1] = 1; q[2] = 5; q[3] = 6;
  elementIndices.push_back(q);

  Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, confusionBF, H1Order, pToAdd) );  
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->solve(false);

    if (rank==0){
      stringstream outfile;
      outfile << "singularwedge_" << refIndex;
      solution->writeToVTK(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
    }

    if (refIndex < numRefs)
    {
      // refinementStrategy.refine(rank==0); // print to console on rank 0
      vector<int> cellsToRefine;
      vector<int> cells_h;
      vector<int> cells_p;
      refinementStrategy.getCellsAboveErrorThreshhold(cellsToRefine);
      for (int i=0; i < cellsToRefine.size(); i++)
        if (sqrt(mesh->getCellMeasure(cellsToRefine[i])) < 5e-5)
        {
          int pOrder = mesh->cellPolyOrder(cellsToRefine[i]);
          if (pOrder < 8)
            cells_p.push_back(cellsToRefine[i]);
          else
            cout << "Reached cell size and polynomial order limits" << endl;
          //   cells_h.push_back(cellsToRefine[i]);
        }
        else
          cells_h.push_back(cellsToRefine[i]);
      refinementStrategy.pRefineCells(mesh, cells_p);
      refinementStrategy.hRefineCells(mesh, cells_h);
    }
  }
  
  return 0;
}
