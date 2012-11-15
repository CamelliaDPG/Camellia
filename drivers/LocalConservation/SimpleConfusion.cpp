//  ConfusionDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

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

bool enforceLocalConservation = true;
bool steady = false;
double epsilon = 1e-1;
double dt = 0.25;
int numTimeSteps = 20; // max time steps
double halfWidth = 1;
int numRefs = 0;
int H1Order = 3, pToAdd = 2;

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

class ScalarParamFunction : public Function {
  double _a;
  public:
  ScalarParamFunction(double a) : Function(0){
    _a = a;
  }
  void set_param(double a){
    _a = a;
  }
  double get_param(){
    return _a;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(_a);
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
    return (abs(x-4.0) < tol);
  }
};

class TopBottomBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool topMatch = (abs(y-2.0) < tol);
    bool bottomMatch = (abs(y+2.0) < tol);
    return topMatch || bottomMatch;
  }
};

// boundary value for u
class ZeroBC : public Function {
  public:
    ZeroBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      double tol=1e-14;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

// boundary value for sigma_n
class InletBC : public Function {
  public:
    InletBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(y) <= halfWidth)
            values(cellIndex, ptIndex) = -1.0*(1.0-y*y);
          else
            values(cellIndex, ptIndex) = 0;
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
          values(cellIndex,ptIndex,1) = 0;
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
  // Process command line arguments
  if (argc > 1)
    epsilon = atof(argv[1]);
  if (argc > 2)
    numRefs = atof(argv[2]);
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  // FunctionPtr beta = Teuchos::rcp(new Beta());

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
  
  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) =  0.0; // x1
  meshBoundary(0,1) = -2.0; // y1
  meshBoundary(1,0) =  4.0;
  meshBoundary(1,1) = -2.0;
  meshBoundary(2,0) =  4.0;
  meshBoundary(2,1) =  2.0;
  meshBoundary(3,0) =  0.0;
  meshBoundary(3,1) =  2.0;

  int horizontalCells = 8, verticalCells = 8;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr flowResidual = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  FunctionPtr u_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u) );
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    

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

  if (!steady)
  {
    confusionBF->addTerm( u, invDt*v );
    rhs->addTerm( u_prev_time * invDt * v );
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
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
  if (!enforceLocalConservation || !steady)
    robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( ip_scaling/sqrt(epsilon) * tau );
  if (!steady)
    robIP->addTerm( invDt * v );
  if (enforceLocalConservation && steady)
    robIP->addZeroMeanTerm( v );

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  SpatialFilterPtr rBoundary = Teuchos::rcp( new RightBoundary );
  SpatialFilterPtr tbBoundary = Teuchos::rcp( new TopBottomBoundary );
  FunctionPtr u1 = Teuchos::rcp( new InletBC );
  FunctionPtr u0 = Teuchos::rcp( new ZeroBC );
  bc->addDirichlet(beta_n_u_minus_sigma_n, lBoundary, u1);
  bc->addDirichlet(beta_n_u_minus_sigma_n, tbBoundary, u0);
  bc->addDirichlet(uhat, rBoundary, u0);

  // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  // solution->setFilter(pc);

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(prevTimeFlow);
  mesh->registerSolution(flowResidual);

  // ==================== SET INITIAL GUESS ==========================
  double u_free = 0.0;
  double sigma1_free = 0.0;
  double sigma2_free = 0.0;
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()]      = Teuchos::rcp( new ConstantScalarFunction(u_free) );
  functionMap[sigma1->ID()] = Teuchos::rcp( new ConstantScalarFunction(sigma1_free) );
  functionMap[sigma2->ID()] = Teuchos::rcp( new ConstantScalarFunction(sigma2_free) );

  prevTimeFlow->projectOntoMesh(functionMap);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    if (steady)
    {
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
    }
    else
    {
      // FunctionPtr parity = Teuchos::rcp<Function>( new SideParityFunction );
      // LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(parity, beta_n_u_minus_sigma_n) );
      LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(1.0, beta_n_u_minus_sigma_n) );
      LinearTermPtr sourcePart = Teuchos::rcp<LinearTerm>( new LinearTerm(invDt, u) );
      conservedQuantity->addTerm(sourcePart, true);
      solution->lagrangeConstraints()->addConstraint(conservedQuantity == u_prev_time * invDt);
    }
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    if (steady)
    {
      solution->solve(false);

      if (rank == 0)
      {
        stringstream outfile;
        outfile << "SimpleConfusion_" << refIndex;
        solution->writeToVTK(outfile.str(), 5);

        // Check local conservation
        FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
        FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
        Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
        cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
          << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
      }
    }
    else
    {
      int timestepCount = 0;
      double time_tol = 1e-8;
      double L2_time_residual = 1e9;
      while((L2_time_residual > time_tol) && (timestepCount < numTimeSteps))
      {
        solution->solve(false);
        // Subtract solutions to get residual
        flowResidual->setSolution(solution);
        flowResidual->addSolution(prevTimeFlow, -1.0);       
        double L2u = flowResidual->L2NormOfSolutionGlobal(u->ID());
        double L2sigma1 = flowResidual->L2NormOfSolutionGlobal(sigma1->ID());
        double L2sigma2 = flowResidual->L2NormOfSolutionGlobal(sigma2->ID());
        L2_time_residual = sqrt(L2u*L2u + L2sigma1*L2sigma1 + L2sigma2*L2sigma2);
        cout << endl << "Timestep: " << timestepCount << ", dt = " << dt << ", Time residual = " << L2_time_residual << endl;    	

        if (rank == 0)
        {
          stringstream outfile;
          outfile << "TransientConfusion_" << refIndex << "-" << timestepCount;
          solution->writeToVTK(outfile.str(), 5);

          // Check local conservation
          FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
          FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(flowResidual, u) );
          source = invDt * source;
          Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, source, varFactory, mesh);
          cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
            << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
        }

        prevTimeFlow->setSolution(solution); // reset previous time solution to current time sol
        timestepCount++;
      }
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(rank==0); // print to console on rank 0
    cout << endl;
  }
  
  return 0;
}
