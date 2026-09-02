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
  SimpleFunction<double>("ch05-StokesCavity-LidVelocity", 2)
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
  
  int spaceDim = 2; // 3D also supported
  double mu = 1.0;
  bool conformingTraces = true;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);

  vector<double> dims = {1.0,1.0}; // domain dimensions
  vector<int> meshDims = {2,2};    // 2x2 initial mesh
  vector<double> x0 = {0.0,0.0};   // lower-left corner at origin
  
  MeshTopologyPtr meshTopo;
  meshTopo = MeshFactory::rectilinearMeshTopology(dims,
                                                  meshDims, x0);
  int polyOrder = 4;
  int delta_k = 2;
  FunctionPtr f = Function::zero(1); // vector zero
  form.initializeSolution(meshTopo, polyOrder, delta_k, f);
  
  double eps = 1.0 / 64.0;
  FunctionPtr lidVelocity_x = Teuchos::rcp(new LidVelocity(eps));
  FunctionPtr lidVelocity = Function::vectorize(lidVelocity_x, Function::zero());

  SpatialFilterPtr lid  = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr wall = !lid; // not lid --> wall
  
  form.addInflowCondition(lid, lidVelocity);
  form.addWallCondition(wall);
  form.addPointPressureCondition({0.5,0.5});

  form.solve();
  
  int refNumber = 0;
  SolutionPtr soln = form.solution();
  SolutionPtr streamSoln = form.streamSolution();
  VarPtr phi = form.streamPhi();
  FunctionPtr phiSoln = Function::solution(phi, streamSoln);
  
  HDF5Exporter exporter(soln->mesh(),"ch05-stokes-cavity-flow");
  HDF5Exporter streamExport(streamSoln->mesh(),"ch05-stokes-stream");
  
  int numSubdivisions = 30; // coarse mesh -> more subdivisions
  exporter.exportSolution(soln, refNumber, numSubdivisions);
  
  streamSoln->solve();
  streamExport.exportFunction({phiSoln}, {"phi"},
                              refNumber, numSubdivisions);
  
  double threshold = 0.2; // relative energy error threshold
  auto refStrategy = form.getRefinementStrategy();
  refStrategy->setRelativeErrorThreshold(threshold);
  
  int numRefinements = 8;
  bool printToConsole = true;
  for (refNumber = 1; refNumber < numRefinements; refNumber++)
  {
    form.refine(printToConsole);
    form.solve();
    exporter.exportSolution(soln, refNumber, numSubdivisions);
    streamSoln->solve();
    streamExport.exportFunction({phiSoln}, {"phi"},
                                refNumber, numSubdivisions);
  }
  // report final energy error:
  double energyError = soln->energyErrorTotal();
  cout << "Final energy error: " << energyError << endl;
  
  return 0;
}
