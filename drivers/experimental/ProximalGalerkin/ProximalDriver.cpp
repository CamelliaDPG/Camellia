//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    double alpha1 = 0.5;
    int numElements = 2;
    double x0 = -1.0;
    double x1 = 1.0;
    int polyOrder = 1;
        
    cmdp.setOption("meshWidth", &numElements );
    cmdp.setOption("polyOrder", &polyOrder );
    cmdp.setOption("alpha1", &alpha1);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    auto alphaParameter = ParameterFunction::parameterFunction(alpha1);
    alphaParameter->setName("alpha");
    FunctionPtr alphaFxn = alphaParameter; // cast to FunctionPtr for operator overload below
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto u   = vf->fieldVar("u",HGRAD);
    auto psi = vf->fieldVar("psi",HGRAD);
    
    auto v   = vf->testVar("v",HGRAD);
    auto phi = vf->testVar("phi",HGRAD);
    
    BFPtr bf = BF::bf(vf);
    
    bf->addTerm(alphaFxn * u->grad(), v->grad());
    alphaParameter->setName("alpha");
    bf->addTerm(psi, v);
    bf->addTerm(u, phi);
    
    MeshTopologyPtr meshTopo  = MeshFactory::intervalMeshTopology(x0, x1, numElements);
    
    const int pToAddTest = 0; // Bubnov-Galerkin
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, bf->varFactory(), polyOrder, pToAddTest) );
    mesh->setBilinearForm(bf);
    
    RHSPtr rhs = RHS::rhs();
    BCPtr bc = BC::bc();
    
    SpatialFilterPtr xLeftSF  = SpatialFilter::lessThanOrEqualToX(x0);
    SpatialFilterPtr xRightSF = SpatialFilter::greaterThanOrEqualToX(x1);
    
    bc->addDirichlet(u, xLeftSF, Function::zero());
    bc->addDirichlet(u, xRightSF, Function::constant(pow(x1,4)));
    
    Teuchos::RCP<HDF5Exporter> viz_soln, viz_incr;
    viz_soln = Teuchos::rcp( new HDF5Exporter (mesh, "proximalSolution", "/tmp"));
    viz_incr = Teuchos::rcp( new HDF5Exporter (mesh, "proximalIncrement", "/tmp"));
    
    SolutionPtr soln      = Solution::solution(mesh);
    SolutionPtr soln_prev = Solution::solution(mesh);
    SolutionPtr soln_incr = Solution::solution(mesh,bc,rhs); // this is the one we actually solve
    auto psi_prev = Function::solution(psi, soln_prev);
    auto psi_soln = Function::solution(psi, soln);
    auto psi_incr = Function::solution(psi, soln_incr);
    
    auto u_prev = Function::solution(u, soln_prev);
    auto u_incr = Function::solution(u, soln_incr);
    auto u_soln = Function::solution(u, soln);
    
    auto psi_soln_weak_ref = Teuchos::rcp(psi_soln.get(), false); // avoid circular dependency
    FunctionPtr exp_psi_soln = Exp<double>::exp(psi_soln_weak_ref);
    
    bf->addTerm(-exp_psi_soln * psi, phi);
    
    FunctionPtr H = Function::heaviside(0);
    FunctionPtr f = -12.0 * H * Function::xn(2);
    FunctionPtr u_exact = H * Function::xn(4);
    
    rhs->addTerm((alphaFxn * f + psi_prev - psi_soln_weak_ref) * v);
    rhs->addTerm(exp_psi_soln * phi);
    
    double tolNewton = 1e-1;
    double tolExit   = 1e-10;
    
    soln->initializeLHSVector();
    
    const double r = 1.5;
    const double q = 1.5;
    double alpha = alpha1;
    
    int t = 0;
    
    for (int k=1; k<20; k++)
    {
      cout << "k: " << k << endl;
      soln_prev->setSolution(soln);
      for (int n=0; n<20; n++)
      {
        cout << "  n: " << n << endl;
        soln_incr->solve();
        viz_incr->exportSolution(soln_incr,     t);

        double uw_diff = (u_incr - u_soln)->l2norm(mesh);
        std::cout << "    uw_diff: " << uw_diff << std::endl;
        soln->addReplaceSolution(soln_incr, 1.0, {psi->ID()}, {u->ID()});
        viz_soln->exportSolution(soln, t);
        t++;
        if (uw_diff < tolNewton) break;
      }
      double uk_diff = (u_soln - u_prev)->l2norm(mesh);
      double u_err   = (u_soln - u_exact)->l2norm(mesh);
      tolNewton = uk_diff;
      std::cout << "  uk_diff: " << uk_diff << std::endl;
      std::cout << "  u_err:   " << u_err   << endl;
      if (uk_diff < tolExit) break;
      
      alpha = min(max(alpha1,pow(r,pow(q,k-1))-alpha),1e10);
      cout << "alpha: " << alpha << endl;
      alphaParameter->setValue(alpha);
    }
  }
  
  Camellia::finalize(false);
  
  return 0;
}
