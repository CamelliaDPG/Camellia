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
  SimpleFunction<double>("ch04-NavierStokesCavity-LidVelocity", 2)
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
  int rank = MPIWrapper::CommWorld()->MyPID();
  
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
  
  // create Stokes BF object:
  BFPtr stokesBF = BF::bf(vf);
  
  // get a normal function (will be useful in a moment):
  FunctionPtr n = Function::normal();
  double Re = 1e2;
  
  // add terms for v1:
  stokesBF->addTerm(sigma1, v1->grad());
  stokesBF->addTerm(-p, v1->dx());
  stokesBF->addTerm(-tn1_hat, v1);
  
  // add terms for v2:
  stokesBF->addTerm(sigma2, v2->grad());
  stokesBF->addTerm(-p, v2->dy());
  stokesBF->addTerm(-tn2_hat, v2);
  
  // add terms for q:
  stokesBF->addTerm(-u, q->grad());
  stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  
  // add terms for tau1:
  stokesBF->addTerm(u->x(), tau1->div());
  stokesBF->addTerm(Re * sigma1, tau1);
  stokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // add terms for tau2:
  stokesBF->addTerm(u->y(), tau2->div());
  stokesBF->addTerm(Re * sigma2, tau2);
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
  
  SolutionPtr backFlow = Solution::solution(stokesBF,mesh);
  mesh->registerSolution(backFlow);
  
  FunctionPtr u_prev = Function::solution(u,backFlow);
  FunctionPtr sigma1_prev = Function::solution(sigma1,backFlow);
  FunctionPtr sigma2_prev = Function::solution(sigma2,backFlow);
  
  BFPtr nsBF = stokesBF->copy();
  nsBF->addTerm(Re * u_prev * sigma1, v1);
  nsBF->addTerm(Re * u_prev * sigma2, v2);
  
  nsBF->addTerm(Re * sigma1_prev * u, v1);
  nsBF->addTerm(Re * sigma2_prev * u, v2);
  
  double eps = 1.0 / 64.0;
  FunctionPtr lidVelocity = Teuchos::rcp(new LidVelocity(eps));

  SpatialFilterPtr lid  = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr wall = !lid; // not lid --> wall

  FunctionPtr zero = Function::constant(0); // for wall velocity

  // for the non-zero velocity BC, need to impose on the
  // *difference* between previously imposed velocity and
  // the velocity we want (because we'll accumulate, and
  // also because on the coarse meshes we won't get
  // the BCs exactly).
  
  FunctionPtr u1_hat_prev = Function::solution(u1_hat, backFlow);
  
  BCPtr bc = BC::bc();
  bc->addDirichlet(u1_hat, lid, lidVelocity - u1_hat_prev);
  bc->addDirichlet(u1_hat, wall, zero);
  bc->addDirichlet(u2_hat, lid | wall, zero);
  
  RHSPtr rhs = RHS::rhs();
  LinearTermPtr stokesTerm = stokesBF->testFunctional(backFlow);
  rhs->addTerm(-stokesTerm);
  rhs->addTerm(-Re * u_prev * sigma1_prev * v1);
  rhs->addTerm(-Re * u_prev * sigma2_prev * v2);
  
  vector<double> center = {0.5,0.5};
  double p_value = 0;
  bc->addSpatialPointBC(p->ID(), p_value, center);
  
  IPPtr ip = nsBF->graphNorm();
  SolutionPtr soln = Solution::solution(nsBF,mesh,bc,rhs,ip);
  
  FunctionPtr u_incr = Function::solution(u,soln);
  FunctionPtr p_incr = Function::solution(p,soln);
  FunctionPtr sigma1_incr = Function::solution(sigma1,soln);
  FunctionPtr sigma2_incr = Function::solution(sigma2,soln);
  FunctionPtr l2_squared = u_incr * u_incr + p_incr * p_incr
    + sigma1_incr * sigma1_incr + sigma2_incr * sigma2_incr;
  
  FunctionPtr p_prev = Function::solution(p, backFlow);
  FunctionPtr l2_backFlow_squared = u_prev * u_prev
    + p_prev * p_prev
    + sigma1_prev * sigma1_prev + sigma2_prev * sigma2_prev;

  double l2_incr = 0;
  
  int refNumber = 0;
  HDF5Exporter exporter(mesh,"navier-stokes-cavity");
  int numSubdivisions = 30; // coarse mesh -> more subdivisions

  double newtonThreshold = 1e-3;
  double energyThreshold = 0.2;
  RefinementStrategyPtr refStrategy =
  RefinementStrategy::energyErrorRefinementStrategy(soln,
                                                    energyThreshold);
  int numRefinements = 8;
  bool printToConsole = true;
  double energyError, l2_soln;
  for (refNumber = 0; refNumber <= numRefinements; refNumber++)
  {
    do
    {
      soln->solve();
      l2_incr = sqrt( l2_squared->integrate(mesh) );
      if (rank==0) cout << "L^2(increment): " << l2_incr << endl;
      // add increment with unit weight:
      backFlow->addSolution(soln, 1.0);
    }
    while (l2_incr > newtonThreshold);
    
    exporter.exportSolution(backFlow, refNumber, numSubdivisions);
    
    refStrategy->refine(printToConsole);
    energyError = refStrategy->getEnergyError(refNumber);
    l2_soln = sqrt(l2_backFlow_squared->integrate(mesh));
    // update threshold:
    newtonThreshold = 1e-3 * energyError / l2_soln;
    cout << "L^2(soln): " << l2_soln << endl;
    cout << "Newton threshold: " << newtonThreshold << endl;
  }
  energyError = soln->energyErrorTotal();
  cout << "Final energy error: " << energyError << endl;
  
  return 0;
}
