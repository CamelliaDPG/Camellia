//  BendDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

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
bool highLiftAirfoil = true;
double epsilon = 1e-2;
int numRefs = 10;
int num1DPts = 5;

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

class LeftBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(x+1) < tol);
  }
};

class RightBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool rightMatch = (abs(x-9) < tol);
    return rightMatch;
  }
};

class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(y-1) < tol);
  }
};

class BottomBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(y+1) < tol);
  }
};

class AirfoilInflowBoundary : public SpatialFilter {
private:
  FunctionPtr _beta;
public:
  AirfoilInflowBoundary(FunctionPtr b) : SpatialFilter(), _beta(b) {}
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) 
  {
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());    
    const FieldContainer<double> *normals = &(basisCache->getSideNormals());
    int numCells = (*points).dimension(0);
    int numPoints = (*points).dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints,2);
    _beta->values(beta_pts,basisCache);

    double tol = 1e-3;
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double n1 = (*normals)(cellIndex,ptIndex,0);
        double n2 = (*normals)(cellIndex,ptIndex,1);
        double beta_n = beta_pts(cellIndex,ptIndex,0)*n1 + beta_pts(cellIndex,ptIndex,1)*n2 ;
        pointsMatch(cellIndex,ptIndex) = false;
        if (abs((x-.5)*(x-.5)+y*y) < 0.75+tol && beta_n < 0)
        {
          pointsMatch(cellIndex,ptIndex) = true;
          somePointMatches = true;
        }
      }
    }
    return somePointMatches;
  }
};

class AirfoilOutflowBoundary : public SpatialFilter {
private:
  FunctionPtr _beta;
public:
  AirfoilOutflowBoundary(FunctionPtr b) : SpatialFilter(), _beta(b) {}
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) 
  {
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());    
    const FieldContainer<double> *normals = &(basisCache->getSideNormals());
    int numCells = (*points).dimension(0);
    int numPoints = (*points).dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints,2);
    _beta->values(beta_pts,basisCache);

    double tol = 1e-3;
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double n1 = (*normals)(cellIndex,ptIndex,0);
        double n2 = (*normals)(cellIndex,ptIndex,1);
        double beta_n = beta_pts(cellIndex,ptIndex,0)*n1 + beta_pts(cellIndex,ptIndex,1)*n2 ;
        pointsMatch(cellIndex,ptIndex) = false;
        if (abs((x-.5)*(x-.5)+y*y) < 0.75+tol && beta_n >= 0)
        // if (abs((x-.5)*(x-.5)+y*y) < 0.75+tol)
        {
          pointsMatch(cellIndex,ptIndex) = true;
          somePointMatches = true;
        }
      }
    }
    return somePointMatches;
  }
};

class ZeroBC : public Function {
  public:
    ZeroBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

// boundary value for sigma_n
class OneBC : public Function {
  public:
    OneBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = 1;
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
          values(cellIndex,ptIndex,0) = 1;
          values(cellIndex,ptIndex,1) = .25;
        }
      }
    }
  }
};

class IPWeight : public Function {
  public:
    IPWeight() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      double a = 2;

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          // if (x > 0 && abs(y) < 1+1e-3 && x < a)
          //   values(cellIndex, ptIndex) = epsilon + (x-sqrt(1-y*y))/(a-sqrt(1-y*y));
          // if (x > 0 && sqrt(x*x+y*y) < a)
          // {
          //   double dr = sqrt(x*x+y*y) - 1;
          //   values(cellIndex, ptIndex) = epsilon + dr/(a-1);
          // }
          // else
            values(cellIndex, ptIndex) = 1;
        }
      }
    }
};

int main(int argc, char *argv[]) {
  // Process command line arguments
  if (argc > 1)
    numRefs = atof(argv[1]);
  if (argc > 2)
    epsilon = atof(argv[2]);
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
  VarPtr sigma1 = varFactory.fieldVar("sigma_x");
  VarPtr sigma2 = varFactory.fieldVar("sigma_y");
  
  FunctionPtr beta = Teuchos::rcp(new Beta());
  
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
  // mathematician's norm
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(tau);
  mathIP->addTerm(tau->div());

  mathIP->addTerm(v);
  mathIP->addTerm(v->grad());

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  qoptIP->addTerm( v );
  qoptIP->addTerm( tau / epsilon + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  if (!enforceLocalConservation)
    robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon) * v->grad() );
  // Weight these two terms for inflow
  FunctionPtr ip_weight = Teuchos::rcp( new IPWeight() );
  robIP->addTerm( ip_weight * beta * v->grad() );
  robIP->addTerm( ip_weight * tau->div() );
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
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  SpatialFilterPtr tBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr bBoundary = Teuchos::rcp( new BottomBoundary );
  SpatialFilterPtr rBoundary = Teuchos::rcp( new RightBoundary );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  SpatialFilterPtr airfoilInflowBoundary = Teuchos::rcp( new AirfoilInflowBoundary(beta) );
  SpatialFilterPtr airfoilOutflowBoundary = Teuchos::rcp( new AirfoilOutflowBoundary(beta) );
  FunctionPtr u0 = Teuchos::rcp( new ZeroBC );
  FunctionPtr u1 = Teuchos::rcp( new OneBC );
  bc->addDirichlet(beta_n_u_minus_sigma_n, lBoundary, u0);
  bc->addDirichlet(beta_n_u_minus_sigma_n, bBoundary, u0);
  // bc->addDirichlet(uhat, airfoilInflowBoundary, u1);
  // bc->addDirichlet(uhat, tBoundary, u0);

  bc->addDirichlet(beta_n_u_minus_sigma_n, airfoilInflowBoundary, beta*n*u1);
  bc->addDirichlet(uhat, airfoilOutflowBoundary, u1);

  // pc->addConstraint(beta*uhat->times_normal() - beta_n_u_minus_sigma_n == u0, rBoundary);
  // pc->addConstraint(beta*uhat->times_normal() - beta_n_u_minus_sigma_n == u0, tBoundary);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 3, pToAdd = 2;
  Teuchos::RCP<Mesh> mesh;
  if (highLiftAirfoil)
    mesh = Mesh::readTriangle(Camellia_MeshDir+"HighLift/HighLift.1", confusionBF, H1Order, pToAdd);
  else
    mesh = Mesh::readTriangle(Camellia_MeshDir+"NACA0012/NACA0012.1", confusionBF, H1Order, pToAdd);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, mathIP) );
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    solution->solve(false);

    if (rank == 0)
    {
      stringstream outfile;
      if (highLiftAirfoil)
        outfile << "highlift_" << refIndex;
      else
        outfile << "naca0012_" << refIndex;
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
      // Try pseudo-hp adaptive
      vector<int> cellsToRefine;
      vector<int> cells_h;
      vector<int> cells_p;
      refinementStrategy.getCellsAboveErrorThreshhold(cellsToRefine);
      for (int i=0; i < cellsToRefine.size(); i++)
        if (sqrt(mesh->getCellMeasure(cellsToRefine[i])) < epsilon)
        {
          int pOrder = mesh->cellPolyOrder(cellsToRefine[i]);
          if (pOrder < 8)
            cells_p.push_back(cellsToRefine[i]);
          else
            cells_h.push_back(cellsToRefine[i]);
        }
        else
          cells_h.push_back(cellsToRefine[i]);
      refinementStrategy.pRefineCells(mesh, cells_p);
      refinementStrategy.hRefineCells(mesh, cells_h);
    }
  }
  
  return 0;
}
