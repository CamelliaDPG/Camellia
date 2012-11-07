#include "ConfusionProblem.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "Solution.h"
#include "Mesh.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
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

////////////////////   DECLARE VARIABLES   ///////////////////////
void ConfusionProblem::defineVariables()
{
  // define test variables
  tau = varFactory.testVar("\\tau", HDIV);
  v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  uhat = varFactory.traceVar("\\widehat{u}");
  beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  u = varFactory.fieldVar("u");
  sigma1 = varFactory.fieldVar("\\sigma_1");
  sigma2 = varFactory.fieldVar("\\sigma_2");
}

////////////////////   DEFINE BILINEAR FORM   ///////////////////////
void ConfusionProblem::defineBilinearForm(vector<double> beta)
{
  confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / epsilon, tau->x());
  confusionBF->addTerm(sigma2 / epsilon, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta* u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
}

////////////////////   DEFINE BILINEAR FORM   ///////////////////////
void ConfusionProblem::defineBilinearForm(FunctionPtr beta)
{
  confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / epsilon, tau->x());
  confusionBF->addTerm(sigma2 / epsilon, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta* u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
}

////////////////////   DEFINE INNER PRODUCT   ///////////////////////
void ConfusionProblem::defineInnerProduct(vector<double> beta)
{
  ip = confusionBF->graphNorm();
}
void ConfusionProblem::defineInnerProduct(FunctionPtr beta)
{
  ip = confusionBF->graphNorm();
}

////////////////////   SPECIFY RHS   ///////////////////////
void ConfusionProblem::defineRightHandSide()
{
  rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
}

////////////////////   SOLVE & REFINE   ///////////////////////
void ConfusionProblem::solveSteady(int argc, char *argv[], string filename, double energyThreshold)
{
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif

  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  RefinementStrategy refinementStrategy( solution, energyThreshold );

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    solution->solve(false);
    if (rank==0)
    {
      if (checkLocalConservation)
      {
        FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
        FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
        Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero);
        cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
          << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
      }
      if (filename != "")
      {
        stringstream outfile;
        outfile << filename << "_" << refIndex;
        solution->writeToVTK(outfile.str(), 5);
      }
    }
    if (refIndex < numRefs)
    {
      if (rank==0){
        cout << "Performing refinement number " << refIndex << endl;
      }     
      refinementStrategy.refine(rank==0); // print to console on rank 0
    }
  }
}

void ConfusionProblem::setMathIP()
{
  // mathematician's norm
  ip = Teuchos::rcp(new IP());
  ip->addTerm(tau);
  ip->addTerm(tau->div());

  ip->addTerm(v);
  ip->addTerm(v->grad());
}

void ConfusionProblem::setRobustIP(vector<double> beta)
{
  // Includes L2 term on v
  ip = Teuchos::rcp(new IP());
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  ip->addTerm( ip_scaling * v );
  ip->addTerm( sqrt(epsilon) * v->grad() );
  ip->addTerm( beta * v->grad() );
  ip->addTerm( tau->div() );
  ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
}

void ConfusionProblem::setRobustIP(FunctionPtr beta)
{
  // Includes L2 term on v
  ip = Teuchos::rcp(new IP());
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  ip->addTerm( ip_scaling * v );
  ip->addTerm( sqrt(epsilon) * v->grad() );
  ip->addTerm( beta * v->grad() );
  ip->addTerm( tau->div() );
  ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
}

void ConfusionProblem::setRobustZeroMeanIP(vector<double> beta)
{
  // Excludes L2 term on v and adds zero mean term
  ip = Teuchos::rcp(new IP());
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  ip->addTerm( sqrt(epsilon) * v->grad() );
  ip->addTerm( beta * v->grad() );
  ip->addTerm( tau->div() );
  ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
  ip->addZeroMeanTerm( v );
}

void ConfusionProblem::setRobustZeroMeanIP(FunctionPtr beta)
{
  // Excludes L2 term on v and adds zero mean term
  ip = Teuchos::rcp(new IP());
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  ip->addTerm( sqrt(epsilon) * v->grad() );
  ip->addTerm( beta * v->grad() );
  ip->addTerm( tau->div() );
  ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
  ip->addZeroMeanTerm( v );
}

Teuchos::Tuple<double, 3> ConfusionProblem::checkConservation(FunctionPtr flux, FunctionPtr source)
{
  // Check conservation by testing against one
  VarPtr testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  // Create a fake bilinear form for the testing
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  // Define our mass flux
  LinearTermPtr fluxTerm = flux * testOne;
  LinearTermPtr sourceTerm = source * testOne;
  // LinearTermPtr massFluxTerm = volumeChange;
  // massFluxTerm->addTerm(surfaceFlux);
  //TODO: Evaluate these separately then compare

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
    // FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
    FieldContainer<double> surfaceIntegrals(elems.size(),testOrdering->totalDofs());
    FieldContainer<double> volumeIntegrals(elems.size(),testOrdering->totalDofs());
    // massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    fluxTerm->integrate(surfaceIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    sourceTerm->integrate(volumeIntegrals,testOrdering,basisCache,true); // true: force side evaluation
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = surfaceIntegrals(i,testOneIndex) + volumeIntegrals(i,testOneIndex);
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

  Teuchos::Tuple<double, 3> fluxImbalances = Teuchos::tuple(maxMassFluxIntegral, totalMassFlux, totalAbsMassFlux);

  return fluxImbalances;
}
