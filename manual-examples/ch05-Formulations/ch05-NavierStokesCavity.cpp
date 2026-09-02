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
  SimpleFunction<double>("ch05-NavierStokesCavity-LidVelocity", 2)
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
  
  double Re = 1e2;
  
  int spaceDim = 2;
  vector<double> dims = {1.0, 1.0};
  vector<int> numElements = {2,2};
  vector<double> x0 = {0,0};
  
  MeshTopologyPtr
    meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
  
  int polyOrder = 4, delta_k = 2;
  
  bool useConformingTraces = true;
  NavierStokesVGPFormulation form
    = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                    meshTopo, polyOrder, delta_k);
  
  double eps = 1.0 / 64.0;
  FunctionPtr lidVelocity_x = Teuchos::rcp(new LidVelocity(eps));
  FunctionPtr lidVelocity = Function::vectorize(lidVelocity_x, Function::zero());
  
  SpatialFilterPtr lid  = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr wall = !lid; // not lid --> wall

  form.addInflowCondition(lid, lidVelocity);
  form.addWallCondition(wall);
  form.addPointPressureCondition({0.5,0.5});
  
  double l2_incr = 0;
  
  int refNumber = 0;
  MeshPtr mesh = form.solution()->mesh();
  HDF5Exporter exporter(mesh,"ch05-navier-stokes-cavity");
  int numSubdivisions = 30; // coarse mesh -> more subdivisions

  double newtonThreshold = 1e-3;
  double energyThreshold = 0.2;
  auto refStrategy = form.getRefinementStrategy();
  refStrategy->setRelativeErrorThreshold(energyThreshold);
  
  int numRefinements = 8;
  bool printToConsole = true;
  double energyError, l2_soln;
  for (refNumber = 0; refNumber <= numRefinements; refNumber++)
  {
    do
    {
      form.solveAndAccumulate();
      l2_incr = form.L2NormSolutionIncrement();
      if (rank==0) cout << "L^2(increment): " << l2_incr << endl;
    }
    while (l2_incr > newtonThreshold);
    
    exporter.exportSolution(form.solution(), refNumber, numSubdivisions);
    
    if (refNumber < numRefinements)
    {
      form.refine(printToConsole);
      energyError = refStrategy->getEnergyError(refNumber);
    }
    else
    {
      energyError = refStrategy->computeTotalEnergyError();
    }
    l2_soln = form.L2NormSolution();
    // update threshold:
    newtonThreshold = 1e-3 * energyError / l2_soln;
    if (rank==0) cout << "L^2(soln): " << l2_soln << endl;
    if (rank==0) cout << "Newton threshold: " << newtonThreshold << endl;
  }
  if (rank==0) cout << "Final energy error: " << energyError << endl;
  
  return 0;
}
