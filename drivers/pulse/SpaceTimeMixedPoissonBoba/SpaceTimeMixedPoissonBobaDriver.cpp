//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include "BOBA/boba.hpp"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  {
    int spaceDim = 1;
    int numElements_x = 2;
    int numElements_t = 2;
    
    vector<vector<double>> domainDim(spaceDim+1,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int polyOrder = 2;
    int pToAdd = spaceDim;
    
    std::string modeString = "single"; // single run; other options are "h" and "p" for h and p convergence tests
    
    cmdp.setOption("xElements", &numElements_x );
    cmdp.setOption("tElements", &numElements_t );
    cmdp.setOption("polyOrder", &polyOrder );
    cmdp.setOption("deltaP", &pToAdd);
    cmdp.setOption("x0", &domainDim[0][0] );
    cmdp.setOption("x1", &domainDim[0][1] );
    cmdp.setOption("t0", &domainDim[1][0] );
    cmdp.setOption("t1", &domainDim[1][1] );
    cmdp.setOption("mode", &modeString, "single, h, or p");
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    std::vector<int> numElements_x_series, numElements_t_series;
    std::vector<int> polyOrder_series;
    if (modeString == "single")
    {
      numElements_x_series.push_back(numElements_x);
      numElements_t_series.push_back(numElements_t);
      polyOrder_series.push_back(polyOrder);
    }
    else if (modeString == "h")
    {
      const int numRefinements = 5;
      int elementMultiplier = 1;
      for (int i=0; i<numRefinements; i++)
      {
        numElements_x_series.push_back(numElements_x * elementMultiplier);
        numElements_t_series.push_back(numElements_t * elementMultiplier);
        polyOrder_series.push_back(polyOrder);
        elementMultiplier *= 2;
      }
    }
    else if (modeString == "p")
    {
      const int numRefinements = 5;
      for (int i=0; i<numRefinements; i++)
      {
        numElements_x_series.push_back(numElements_x);
        numElements_t_series.push_back(numElements_t);
        polyOrder_series.push_back(polyOrder + i);
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported mode");
    }
    
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr E_hat     = vf->fluxVarSpaceOnly("E_hat", HGRAD);
    VarPtr phi_hat   = vf->fluxVarSpaceOnly("phi_hat", HGRAD);
    VarPtr E         = vf->fieldVar("E", HGRAD_DISC);
    VarPtr phi       = vf->fieldVar("phi", HGRAD_DISC);
    VarPtr eps_tau   = vf->fieldVar("eps_tau", HGRAD_DISC);
    VarPtr eps_q     = vf->fieldVar("eps_q", HGRAD_DISC);
    
    // test variables are discretized the same way as trial, so can be represented by the same VarPtrs; we relabel them for clarity:
    auto tau           = eps_tau;
    auto q             = eps_q;
    auto delta_E_hat   = E_hat;
    auto delta_phi_hat = phi_hat;
    auto delta_E       = E;
    auto delta_phi     = phi;
    
    auto bubnovVF = vf->getBubnovFactory(Camellia::VarFactory::BUBNOV_TRIAL);
    
    auto poissonBF = BF::bf(bubnovVF);
    
    // B:
    poissonBF->addTerm(     -E, tau->dx());
    poissonBF->addTerm(  E_hat, tau);
    poissonBF->addTerm(      E, q);
    poissonBF->addTerm(   -phi, q->dx());
    poissonBF->addTerm(phi_hat, q);
    
    // B^T
    poissonBF->addTerm(eps_tau->dx(), -delta_E);
    poissonBF->addTerm(          tau,  delta_E_hat);
    poissonBF->addTerm(            q,  delta_E);
    poissonBF->addTerm(      q->dx(), -delta_phi);
    poissonBF->addTerm(            q,  delta_phi_hat);
    
    // G:
    poissonBF->addTerm( eps_tau->dx() + eps_q, tau->dx() + q );
    poissonBF->addTerm(           eps_q->dx(), q->dx()       );
    poissonBF->addTerm(               eps_tau, tau           );
    poissonBF->addTerm(                 eps_q, q             );
    
    std::map<int,int> polyOrderEnhancements;
    // nominally, E and phi are in L^2, and E_hat and phi_hat are traces of H^1.
    // so the natural thing is for the traces to be discretized with order 1 higher
    // and the test functions to be discretized with order 1 + pToAdd higher
    polyOrderEnhancements[E_hat->ID()]   = 1;
    polyOrderEnhancements[phi_hat->ID()] = 1;
    polyOrderEnhancements[eps_q->ID()]   = 1 + pToAdd;
    polyOrderEnhancements[eps_tau->ID()] = 1 + pToAdd;
    MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
    Teuchos_CommPtr nullComm = Teuchos::null;
    const bool allowPruning = false;  // to keep inter-mesh geometry determinations from getting too complex, don't allow the mesh topology to be pruned to keep only MPI-local geometry.
    
    BCPtr  bc  = BC::bc();
    RHSPtr rhs = RHS::rhs();
    
    FunctionPtr x = Function::xn(1); // x^1
    FunctionPtr t = Function::tn(1); // t^1
    
    FunctionPtr sin_pi_x_over_2 = TrigFunctions<double>::sin(0.5 * M_PI * x);
    FunctionPtr cos_2_pi_t      = TrigFunctions<double>::cos(2.0 * M_PI * t);
    
    FunctionPtr phi_exact = -4. / (M_PI * M_PI) * sin_pi_x_over_2 * cos_2_pi_t;
    FunctionPtr phi_xx    = phi_exact->dx()->dx();
    rhs->addTerm(-phi_xx * tau);
    
    // impose boundary conditions on the potential along the whole (spatial) mesh boundary ( phi_hat is only defined on the spatial part of the boundary):
    bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);
    
    int numRuns = int(numElements_x_series.size());
    for (int i=0; i<numRuns; i++)
    {
      const int polyOrder_i     = polyOrder_series[i];
      const int numElements_x_i = numElements_x_series[i];
      const int numElements_t_i = numElements_t_series[i];
      std::cout << "Running with polyOrder " << polyOrder_i << " on " << numElements_x_i << " x " << numElements_t_i << " mesh.\n";
      
      auto spatialMeshTopo  = MeshFactory::intervalMeshTopology(domainDim[0][0], domainDim[0][1], numElements_x_i);
      auto extrudedMeshTopo = MeshFactory::extrudedMeshTopology(spatialMeshTopo, {domainDim[1][0]}, {domainDim[1][1]}, {numElements_t_i});
      
      MeshPtr mesh = Teuchos::rcp( new Mesh (extrudedMeshTopo, poissonBF, polyOrder_i, 0, polyOrderEnhancements, polyOrderEnhancements, nullPartitionPolicy, nullComm, allowPruning) );
      
//      ::boba::matrix<::boba::host_space> my_matrix;
//      my_matrix.read_from_file("/tmp/K.dat");
//      my_matrix.print();
//      my_matrix.write_to_file("/tmp/K2.dat");
      
      SolutionPtr soln = Solution::solution(poissonBF, mesh, bc, rhs);
      soln->setWriteMatrixToFile(true, "/tmp/K.dat");
      
      soln->solve();
      
      // compute L^2 error in phi:
      FunctionPtr phi_soln = Function::solution(phi, soln);
      double l2_err = (phi_soln - phi_exact)->l2norm(mesh);
      
      double exact_l2_norm = phi_exact->l2norm(mesh);
      
      std::cout << "absolute L^2 error: " << l2_err << std::endl;
      std::cout << "relative L^2 error: " << l2_err / exact_l2_norm << "\n\n";
    }
  }
  Camellia::finalize(false);
  
  return 0;
}
