//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

using namespace Camellia;
using namespace std;

class LidVelocity : public SimpleFunction<double>
{
  double _eps; // interpolation width
public:
  LidVelocity(double eps)
  :
  SimpleFunction<double>("ch03-StokesCavity-LidVelocity", 2)
  {
    _eps = eps;
  }
  double value(double x, double y)
  {
    if (abs(x) < _eps)
    {
      return x / _eps;         // top left
    }
    else if (abs(1.0-x) < _eps)
    {
      return (1.0-x) / _eps;   // top right
    }
    else
    {
      return 1;                // top middle
    }
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  VarFactoryPtr vf = VarFactory::varFactory();
  
  // field variables:
  VarPtr u = vf->fieldVar("u", VECTOR_L2);
  VarPtr p = vf->fieldVar("p", L2);
  VarPtr sigma1 = vf->fieldVar("sigma_1", VECTOR_L2);
  VarPtr sigma2 = vf->fieldVar("sigma_2", VECTOR_L2);
  
  // trace and flux variables:
  VarPtr u1_hat = vf->traceVar("u1_hat", HGRAD);
  VarPtr u2_hat = vf->traceVar("u2_hat", HGRAD);
  VarPtr tn1_hat = vf->fluxVar("tn1_hat", L2);
  VarPtr tn2_hat = vf->fluxVar("tn2_hat", L2);
  
  // test variables:
  VarPtr q = vf->testVar("q", HGRAD);
  VarPtr v1 = vf->testVar("v_1", HGRAD);
  VarPtr v2 = vf->testVar("v_2", HGRAD);
  VarPtr tau1 = vf->testVar("tau_1", HDIV);
  VarPtr tau2 = vf->testVar("tau_2", HDIV);
  
  // create BF object:
  BFPtr stokesBF = BF::bf(vf);
  
  // get a normal function (will be useful in a moment):
  FunctionPtr n = Function::normal();
  double mu = 1.0; // unit viscosity
  
  // add terms for v1:
  stokesBF->addTerm(sigma1, v1->grad());
  stokesBF->addTerm(-p, v1->dx());
  stokesBF->addTerm(-tn1_hat, v1);
  
  // add terms for v2:
  stokesBF->addTerm(sigma2, v2->grad());
  stokesBF->addTerm(-p, v2->dy());
  stokesBF->addTerm(-tn2_hat, v2);
  
  // add terms for q:
  stokesBF->addTerm(u, q->grad());
  stokesBF->addTerm(-u1_hat * n->x() - u2_hat * n->y(), q);
  
  // add terms for tau1:
  stokesBF->addTerm(u->x(), tau1->div());
  stokesBF->addTerm((1.0/mu) * sigma1, tau1);
  stokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // add terms for tau2:
  stokesBF->addTerm(u->y(), tau2->div());
  stokesBF->addTerm((1.0/mu) * sigma2, tau2);
  stokesBF->addTerm(-u2_hat, tau2->dot_normal());

  vector<double> dims = {1.0,1.0}; // domain dimensions
  vector<int> meshDims = {2,2};  // 2x2 initial mesh
  vector<double> x0 = {0.0,0.0}; // lower-left corner at origin
  
  MeshTopologyPtr meshTopo;
  meshTopo = MeshFactory::rectilinearMeshTopology(dims,
                                                  meshDims, x0);
  int H1Order = 5;
  int delta_k = 2;
  MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, stokesBF,
                                          H1Order, delta_k);
  BCPtr bc = BC::bc();

  double eps = 1.0 / 64.0;
  FunctionPtr lidVelocity = Teuchos::rcp(new LidVelocity(eps));

  SpatialFilterPtr lid  = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr wall = !lid; // not lid --> wall

  FunctionPtr zero = Function::constant(0); // for wall velocity
  bc->addDirichlet(u1_hat, lid, lidVelocity);
  bc->addDirichlet(u1_hat, wall, zero);

  bc->addDirichlet(u2_hat, lid | wall, zero);

  vector<double> center = {0.5,0.5};
  double p_value = 0;
  bc->addSpatialPointBC(p->ID(), p_value, center);
  
  RHSPtr rhs = RHS::rhs();
  
  IPPtr ip = stokesBF->graphNorm();
  SolutionPtr soln = Solution::solution(stokesBF,mesh,bc,rhs,ip);
  
  soln->solve();
  
  int refNumber = 0;
  HDF5Exporter exporter(mesh,"stokes-cavity-flow");
  int numSubdivisions = 30; // coarse mesh -> more subdivisions
  exporter.exportSolution(soln, refNumber, numSubdivisions);

  double threshold = 0.2; // relative energy error threshold
  RefinementStrategyPtr refStrategy =
   RefinementStrategy::energyErrorRefinementStrategy(soln,
                                                     threshold);
  
  int numRefinements = 8;
  bool printToConsole = true;
  for (refNumber = 1; refNumber < numRefinements; refNumber++)
  {
    refStrategy->refine(printToConsole);
    soln->solve();
    exporter.exportSolution(soln, refNumber, numSubdivisions);
  }
  // report final energy error:
  double energyError = soln->energyErrorTotal();
  cout << "Final energy error: " << energyError << endl;
  
  return 0;
}
