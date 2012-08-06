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

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

double epsilon = 1e-2;
double numRefs = 6;

bool enforceLocalConservation = true;

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

class MassFluxParity : public Function 
{
  private:
    FunctionPtr _massFlux;
    Teuchos::RCP<Mesh> _mesh;
  public:
    MassFluxParity(FunctionPtr massFlux, Teuchos::RCP<Mesh> mesh ) : Function(0), 
    _massFlux(massFlux), _mesh(mesh) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      vector<int> cellIDs = basisCache->cellIDs();
      int sideIndex = basisCache->getSideIndex();
      _massFlux->values(values, basisCache);
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        FieldContainer<double> parities = _mesh->cellSideParitiesForCell(cellIDs[cellIndex]);
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          // values(cellIndex, ptIndex) *= parities(sideIndex);
        }
      }
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
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
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
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon
) ); 
  // robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon
) * v->grad() );
  robIP->addTerm( beta_const * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( ip_scaling/sqrt(epsilon
) * tau );
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
  Teuchos::RCP<Mesh> mesh = Mesh::readMsh("quad.msh", confusionBF, H1Order, pToAdd);
  // define nodes for mesh
  // FieldContainer<double> meshBoundary(4,2);
  // 
  // meshBoundary(0,0) = 0.0; // x1
  // meshBoundary(0,1) = 0.0; // y1
  // meshBoundary(1,0) = 1.0;
  // meshBoundary(1,1) = 0.0;
  // meshBoundary(2,0) = 1.0;
  // meshBoundary(2,1) = 1.0;
  // meshBoundary(3,0) = 0.0;
  // meshBoundary(3,1) = 1.0;
  // 
  // int horizontalCells = 1, verticalCells = 1;
  // 
  // // create a pointer to a new mesh:
  // Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
  //                                               confusionBF, H1Order, H1Order+pToAdd, true);
  
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
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  // one more solve on the final refined mesh:
  solution->solve(false);

  // Check conservation by testing against one
  VarPtr testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  // Create a fake bilinear form for the testing
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  // Define our mass flux
  FunctionPtr massFluxVal = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
  FunctionPtr massFlux = Teuchos::rcp( new MassFluxParity(massFluxVal, mesh) );
  LinearTermPtr massFluxTerm = massFlux * testOne;

  Teuchos::RCP<shards::CellTopology> quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(fakeBF);
  int fakeTestOrder = H1Order;
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);
  
  int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
  vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
  map<int, double> massFluxIntegral; // cellID -> integral
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
    vector<int> cellIDs;
    for (int i=0; i<elems.size(); i++) {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh) );
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
    massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
    }
    // find the largest:
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
    }
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }
  
  
  // Print results from processor with rank 0
  if (rank==0){
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;

    stringstream outfile;
    outfile << "confusion" << epsilon;
    solution->writeFieldsToVTK(outfile.str(), 3);
  }
  
  return 0;
}
