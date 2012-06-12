#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

bool enforceLocalConservation = false;

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
    double scaling = min(sqrt(_epsilon)/ h, 1.0);
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

class InflowLshapeTop : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol);
    bool yMatch = ((y>.5) && (y<1.0));
    return xMatch && yMatch;
  }
};

class InflowLshapeBottom : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-.5)<tol);
    bool yMatch = ((y>0.0) && (y<.5));
    return xMatch && yMatch;
  }
};

class LshapeBottom1 : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = ((x>0.0)&&(x<.5));
    bool yMatch = (abs(y-.5)<tol);
    return xMatch && yMatch;
  }
};
class LshapeBottom2 : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = ((x>.5)&&(x<1.0));
    bool yMatch = (abs(y)<tol);
    return xMatch && yMatch;
  }
};

class LshapeTop : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = ((x>0.0)&&(x<1.0));
    bool yMatch = (abs(y-1.0)<tol);
    return xMatch && yMatch;
  }
};

class LshapeOutflow : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0)<tol);
    bool yMatch = ((y>0.0)&&(y<1.0));
    return xMatch && yMatch;
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
	values(cellIndex,ptIndex) = 0.0;  
      }
    }
  }
};


class ParabolicProfile : public Function {
public:
  ParabolicProfile() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	
	values(cellIndex,ptIndex) = 5.0*(.5-y)*(y-1.0);
      }
    }
  }
};

class LinearProfile : public Function {
public:
  LinearProfile() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	
	values(cellIndex,ptIndex) = y;
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
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  vector<double> beta_const;
  beta_const.push_back(1.0);
  beta_const.push_back(0.0);
//  FunctionPtr beta = Teuchos::rcp(new Beta());
  
  double eps = 1e-2;
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
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
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta_const * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) ); 
  robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta_const * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( ip_scaling/sqrt(eps) * tau );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  //  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary ); 

  SpatialFilterPtr inflowTop = Teuchos::rcp(new InflowLshapeTop);
  SpatialFilterPtr inflowBot = Teuchos::rcp(new InflowLshapeBottom);
  SpatialFilterPtr LshapeBot1 = Teuchos::rcp(new LshapeBottom1);
  SpatialFilterPtr LshapeBot2 = Teuchos::rcp(new LshapeBottom2);
  SpatialFilterPtr Top = Teuchos::rcp(new LshapeTop);
  SpatialFilterPtr Out = Teuchos::rcp(new LshapeOutflow);

  FunctionPtr u0 = Teuchos::rcp( new U0 );
  bc->addDirichlet(uhat, LshapeBot1, u0);
  bc->addDirichlet(uhat, LshapeBot2, u0);
  bc->addDirichlet(uhat, Top, u0);
  bc->addDirichlet(uhat, Out, u0);

  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  //  bc->addDirichlet(uhat, inflowBot, u0);
  FunctionPtr u0Top = Teuchos::rcp(new ParabolicProfile);
  FunctionPtr u0Bot = Teuchos::rcp(new LinearProfile);
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowTop, beta_const*n*u0Top);
  //  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBot, beta_const*n*u0Bot);
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBot, beta_const*n*zero);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 2, pToAdd = 2;
  /*
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  int horizontalCells = 1, verticalCells = 1;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
  */

  Teuchos::RCP<Mesh> mesh;
  // L-shaped domain for double ramp problem
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A(0) = 0.0; A(1) = 0.5;
  B(0) = 0.0; B(1) = 1.0;
  C(0) = 0.5; C(1) = 1.0;
  D(0) = 1.0; D(1) = 1.0;
  E(0) = 1.0; E(1) = 0.5;
  F(0) = 1.0; F(1) = 0.0;
  G(0) = 0.5; G(1) = 0.0;
  H(0) = 0.5; H(1) = 0.5;
  vector<FieldContainer<double> > vertices;
  vertices.push_back(A); int A_index = 0;
  vertices.push_back(B); int B_index = 1;
  vertices.push_back(C); int C_index = 2;
  vertices.push_back(D); int D_index = 3;
  vertices.push_back(E); int E_index = 4;
  vertices.push_back(F); int F_index = 5;
  vertices.push_back(G); int G_index = 6;
  vertices.push_back(H); int H_index = 7;
  vector< vector<int> > elementVertices;
  vector<int> el1, el2, el3, el4, el5;
  // left patch:
  el1.push_back(A_index); el1.push_back(H_index); el1.push_back(C_index); el1.push_back(B_index);
  // top right:
  el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
  // bottom right:
  el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);

  elementVertices.push_back(el1);
  elementVertices.push_back(el2);
  elementVertices.push_back(el3);
  
  mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, confusionBF, H1Order, pToAdd) );  
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  int numRefs = 8;
    
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  // one more solve on the final refined mesh:
  solution->solve(false);
  
  if (rank==0){
    solution->writeFieldsToFile(u->ID(), "u.m");
    solution->writeFluxesToFile(uhat->ID(), "uhat.dat");
    
    cout << "wrote files: u.m, uhat.dat\n";
  }
  
  return 0;
}
