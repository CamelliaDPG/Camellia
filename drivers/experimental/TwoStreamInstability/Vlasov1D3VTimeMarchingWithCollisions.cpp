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
    
    double q      =  0.0; // pure vlasov mode: electric field will always be 0
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  1.0;
    
    double vDomainWidth = 20;
        
    double tol = 1e-8;
    
    int xElements = 2;
    int vElements_x = 4; // need enough elements to avoid having nonzero velocity near the velocity boundary (2 is not enough; 4 seems to be)
    int delta_k   = 2;
    double dt_max = 1e-4;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int numTimeSteps = 10;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    
    std::string modelDir = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vMeshWidth", &vElements_x);
    cmdp.setOption("vDomainWidth", &vDomainWidth);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
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
    
    const double vMin = -vDomainWidth/2.;
    
    const Kokkos::Array<double,3> v0 {vMin, vMin, vMin};
    const Kokkos::Array<double,3> vDomainWidths {vDomainWidth, vDomainWidth, vDomainWidth};
    
    const int vElements_y = vElements_x, vElements_z = vElements_x;
    
    const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
    
    const int rank = mpiSession.getRank();
    
    bool enforceQuasiNeutrality = false;
    
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
    
    
    // setup a Maxwellian for initial condition -- should result in a near-0 collisional response (I think)
    const int firstVelocityDimensionZeroBased = 1;
    const int numVelocityDims = 3;
    std::vector<double> u {0,0,0};
    const double sigmaSquared = 1.0;
    const double n = 1.0;
    FunctionPtr f0 = MaxwellianFunction::maxwellian(firstVelocityDimensionZeroBased, numVelocityDims, u, sigmaSquared, n);
    
    // scaling factors:
    const double n0_scaling = 1.e21;
    const double L0_scaling = 5000;
    const double mass = 6.6335209e-26;
    const double kB = 1.602e-19; // for plasmas, we want to use electron volts (eV) rather than Kelvin -- that's why "Boltzmann's constant" is set to to the charge of an electron.
    const double T0 = 20;
    const double vth = std::sqrt(kB * T0 / mass);
    const double u0_scaling = vth;
    
    if (rank == 0)
    {
      std::cout << "dt = " << dt << std::endl;
      std::cout << "Scalings:\n";
      std::cout << "n0 = " << n0_scaling << std::endl;
      std::cout << "L0 = " << L0_scaling << std::endl;
      std::cout << "u0 = " << u0_scaling << std::endl;
    }
    
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
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_" << xElements << "x" << vElements_x << "x" << vElements_y << "x" << vElements_z;
    
    std::string problemPrefix = "MaxwellianInitialConditions";
    
    std::ostringstream boltzmannSS1D3V;
    boltzmannSS1D3V << problemPrefix <<"Boltzmann1D3V" << suffix.str();
    
    std::ostringstream gaussSS1D3V;
    gaussSS1D3V << problemPrefix << "Gauss1D3V" << suffix.str();
    
    const std::string directoryPath = ".";
    
    int maxMomentNumber = MLCollisionFunction::maxMomentNumber(modelDir);
    
    const bool usePeriodicBCs = true;
    Boltzmann1D3VFormulation form1D3V(formulationChoice1D3V, q, eps0, m, dt, x0, xWidth, xElements, v0, vDomainWidths, vMeshWidths, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, maxMomentNumber);
    
    form1D3V.setInitialConditions(f0);
    
    Teuchos::RCP<SolutionMoments> solnMoments = form1D3V.solutionMoments();
    
    auto density  = solnMoments->solutionMomentFunction(0,0,0);
    auto momentum_x = solnMoments->solutionMomentFunction(1,0,0);
    auto momentum_y = solnMoments->solutionMomentFunction(0,1,0);
    auto momentum_z = solnMoments->solutionMomentFunction(0,0,1);
    
    auto velocity_x = momentum_x / density;
    auto velocity_y = momentum_y / density;
    auto velocity_z = momentum_z / density;
    
    PhaseSpaceVisualization phaseSpaceViz1D3V(form1D3V.boltzmannSolution(), form1D3V.f(), boltzmannSS1D3V.str(), ".");
    HDF5Exporter gaussViz1D3V(form1D3V.gaussMesh(), gaussSS1D3V.str(), directoryPath);
    
    solnMoments->update(); // since we need solnMoments on the RHS, it will be evaluated even in first time step.  This sets it up for evaluation.
    double t = 0.0;
    gaussViz1D3V.exportSolution(form1D3V.gaussSolution(), {density,velocity_x,velocity_y,velocity_z}, {"n_e","v_x","v_y","v_z"},t);
    phaseSpaceViz1D3V.exportTimeStep(t);
    
    FunctionPtr f_soln1D3V = Function::solution(form1D3V.f(), form1D3V.boltzmannSolution());
    
    auto subMeshTopo1D3V_xvx = SubExtrudedMeshTopology::subExtrudedMeshTopology(form1D3V.boltzmannMesh()->getTopology(), 0);
    MeshPtr subMesh1D3V_xvx = MeshFactory::minRuleMesh(subMeshTopo1D3V_xvx, form1D3V.boltzmannMesh()->bilinearForm(), H1Order, delta_k, form1D3V.boltzmannMesh()->Comm(), false);
    
    auto f_soln = Function::solution(form1D3V.f(), form1D3V.boltzmannSolution());
    
    // TODO: measure the error in the moments we provide to MLCollisionFunction (we know the exact Maxwellian moments, so we can compute the error)
    // TODO: try providing exact values to MLCollisionFunction; how does it behave?
    
    Teuchos::RCP<MLCollisionFunction> mlCollisionFunction = Teuchos::rcp(new MLCollisionFunction(n0_scaling, L0_scaling, u0_scaling, f_soln, solnMoments, modelDir) );
    
    auto w = form1D3V.w(); // test function for Vlasov
    LinearTermPtr collisionTerm = mlCollisionFunction * w;
    form1D3V.boltzmannSolution()->rhs()->addTerm(collisionTerm);
    
    // DEBUGGING: create a second Solution object onto which we can project the mlCollisionFunction, and then take moments of that
    auto collisionRepresentation = Solution::solution(form1D3V.boltzmannSolution()->bf(),form1D3V.boltzmannMesh());
    std::map<int,FunctionPtr> collisionMap;
    collisionMap[form1D3V.f()->ID()] = mlCollisionFunction;
    collisionRepresentation->projectOntoMesh(collisionMap);
    
    PhaseSpaceVisualization phaseSpaceVizCollisions(collisionRepresentation, form1D3V.f(), "CollisionDebugViz", ".");
    phaseSpaceVizCollisions.exportTimeStep(0);
    
    std::cout << "Exported CollisionDebugViz.\n";
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      form1D3V.takeTimeStep();
      t += dt;
      
      // now that we've solved, update solnMoments
      solnMoments->update();
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        gaussViz1D3V.exportSolution(form1D3V.gaussSolution(), {density,velocity_x,velocity_y,velocity_z}, {"n_e","v_x","v_y","v_z"},t);
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
