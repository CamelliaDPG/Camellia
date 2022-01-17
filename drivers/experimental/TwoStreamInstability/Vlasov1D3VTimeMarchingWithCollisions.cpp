//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "Boltzmann1D3VFormulation.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "HDF5Exporter.h"
#include "MaxwellianFunction.h"
#include "PhaseSpaceVisualization.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include "Intrepid2_Cubature.hpp"

#include "MLCollisionFunction.h"

/**
  This driver is a simple demonstration of running a 1D3V, time-marching Vlasov solve with collisions based on a machine-learned collision operator using mlc.
 */

using namespace Camellia;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  mpiSession.barrier();
  
  bool success = true;
  {
    bool implicit = true;
    double u0     = 3.0; // initial drift velocity
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  0.0; // pure vlasov mode: electric field will always be 0
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  20;
    
    const Kokkos::Array<double,3> v0 {-10.0, -10.0, -10.0};
    const Kokkos::Array<double,3> vDomainWidths {20, 20, 20};
        
    //  const bool useAdaptivity = false; // adaptivity not supported; we'd need to do something to keep 1D3V and 1D1V meshes in sync.
    
    // relative tolerance: require 1D1V and 1D3V solutions to be within 1e-8 of each other, in terms of |E|^2 norm.
    // We also require the f_xvx solution to agree to this tolerance.  The E tolerance is the harder one to hit, since
    // |E|^2 can be pretty close to zero.
    // The agreement should hold for arbitrary meshes (coarse and fine ones) and time step sizes: mathematically, 1D3V
    // and 1D1V should be identical for this problem.  The only differences should be due to roundoff and solver stopping criteria.
    double tol = 1e-8;
    
    int xElements = 2;
    int vElements_x = 2;
    int delta_k   = 2;
    double dt_max = 1e-1;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int numTimeSteps = 10;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    
    std::string modelDir = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vMeshWidth", &vElements_x);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("u0", &u0);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("modelDir", &modelDir);
  //  cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    const int vElements_y = vElements_x, vElements_z = vElements_x;
    
    const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
    
    const int rank = mpiSession.getRank();
    
    using FormulationChoice1D3V = Camellia::Boltzmann1D3VFormulation::Boltzmann1D3VFormulationChoice;
    FormulationChoice1D3V formulationChoice1D3V;
    if (implicit)
    {
      if (useL2Trace)
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_IMPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_IMPLICIT_H1_TRACE;
      }
    }
    else
    {
      if (useL2Trace)
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_EXPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_EXPLICIT_H1_TRACE;
      }
    }
    
    ERKTableauPtr erkTableau;
    if (!implicit)
    {
      erkTableau = ERKTableau::ssprk3();
    }
    
    double dt = dt_max;
    
    const bool usePeriodicBCs = true;
    const bool enforceQuasiNeutrality = true;
    Boltzmann1D3VFormulation form1D3V(formulationChoice1D3V, q, eps0, m, dt, x0, xWidth, xElements, v0,    vDomainWidths,    vMeshWidths,    H1Order, delta_k, usePeriodicBCs, erkTableau);
    
    // setup a Maxwellian for initial condition -- should result in a near-0 collisional response (I think)
    const int firstVelocityDimensionZeroBased = 1;
    const int numVelocityDims = 3;
    std::vector<double> u {1,1,1};
    const double sigmaSquared = 1.0e3;
    const double n = 1.0;
    FunctionPtr f0 = MaxwellianFunction::maxwellian(firstVelocityDimensionZeroBased, numVelocityDims, u, sigmaSquared, n);
    
    form1D3V.setInitialConditions(f0);
    
    std::ostringstream suffix;
    if (implicit)
    {
      suffix << "Implicit";
    }
    if (useL2Trace)
    {
      suffix << "_L2Trace";
    }
    else
    {
      suffix << "_H1Trace";
    }
    suffix << "_u0" << u0;
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_" << xElements << "x" << vElements_x << "x" << vElements_y << "x" << vElements_z;
    
    std::string problemPrefix = "ZeroInitialConditions";
    
    std::ostringstream boltzmannSS1D3V;
    boltzmannSS1D3V << problemPrefix <<"Boltzmann1D3V" << suffix.str();
    
    std::ostringstream gaussSS1D3V;
    gaussSS1D3V << problemPrefix << "Gauss1D3V" << suffix.str();
    
    const std::string directoryPath = ".";
    
    PhaseSpaceVisualization phaseSpaceViz1D3V(form1D3V.boltzmannSolution(), form1D3V.f(), boltzmannSS1D3V.str(), ".");
    HDF5Exporter gaussViz1D3V(form1D3V.gaussMesh(), gaussSS1D3V.str(), directoryPath);
    
    double t = 0.0;
    gaussViz1D3V.exportSolution(form1D3V.gaussSolution(),t);
    phaseSpaceViz1D3V.exportTimeStep(t);
    
    FunctionPtr f_soln1D3V = Function::solution(form1D3V.f(), form1D3V.boltzmannSolution());
    
    auto subMeshTopo1D3V_xvx = SubExtrudedMeshTopology::subExtrudedMeshTopology(form1D3V.boltzmannMesh()->getTopology(), 0);
    MeshPtr subMesh1D3V_xvx = MeshFactory::minRuleMesh(subMeshTopo1D3V_xvx, form1D3V.boltzmannMesh()->bilinearForm(), H1Order, delta_k, form1D3V.boltzmannMesh()->Comm(), false);
    
    int maxMomentNumber = MLCollisionFunction::maxMomentNumber(modelDir);
    
    // TODO: incorporate this solnMoments into form1D3V so that it can be automatically updated after each solve, and so that we aren't redundantly computing moments multiple times.
    Teuchos::RCP<SolutionMoments> solnMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, form1D3V.boltzmannSolution(), form1D3V.f(), form1D3V.gaussMesh()));
    
    auto f_soln = Function::solution(form1D3V.f(), form1D3V.boltzmannSolution());
    
    Teuchos::RCP<MLCollisionFunction> mlCollisionFunction = Teuchos::rcp(new MLCollisionFunction(f_soln, solnMoments, modelDir) );
    
    auto w = form1D3V.w(); // test function for Vlasov
    LinearTermPtr collisionTerm = mlCollisionFunction * w;
    form1D3V.boltzmannSolution()->rhs()->addTerm(collisionTerm);
    
    solnMoments->update(); // since we need solnMoments on the RHS, it will be evaluated even in first time step.  This sets it up for evaluation.
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      form1D3V.takeTimeStep();
      t += dt;
      
      // now that we've solved, update solnMoments
      solnMoments->update();
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        gaussViz1D3V.exportSolution(form1D3V.gaussSolution(),t);
        phaseSpaceViz1D3V.exportTimeStep(t);

        if (rank == 0)
        {
          std::cout << "Completed time step " << timeStepNumber << " (t = " << t << ")" << std::endl;
        }
      }
    }
  }
  
  bool printAllocationReport = false; // (rank == 0);
  
  Camellia::finalize(printAllocationReport);
  Kokkos::finalize();
  
  if (success)
    return 0;
  else
    return -1;
}
