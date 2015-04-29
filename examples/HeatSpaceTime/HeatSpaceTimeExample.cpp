#include "Teuchos_GlobalMPISession.hpp"
#include "SpaceTimeHeatFormulation.h"
#include "ExpFunction.h"
#include "Function.h"
#include "MeshFactory.h"
#include "HDF5Exporter.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace std;

const static double PI  = 3.141592653589793238462;

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  int spaceDim = 1;
  double eps = 1e-2;
  bool useConformingTraces = true;
  
  FunctionPtr cos_2pi_x = Teuchos::rcp( new Cos_ax(2 * PI) );
  double lambda = -4.0 * PI * PI * eps;
  FunctionPtr exp_lambda_t = Teuchos::rcp( new Exp_at(lambda) );
  
  FunctionPtr u_exact = cos_2pi_x * exp_lambda_t;
  FunctionPtr sigma_exact = eps * u_exact->dx();

  double h = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo;
  {
    int elementDiameter = (int) ceil(1.0 / h);
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,elementDiameter);
    vector<double> x0(spaceDim,0.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    double t0 = 0.0, t1 = 1.0;
    spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, elementDiameter);
  }
  
  int polyOrder = 2, delta_k = 2;
  
  SpaceTimeHeatFormulation form(spaceDim, eps, useConformingTraces);
  form.initializeSolution(spaceTimeMeshTopo, polyOrder, delta_k);
  
  VarPtr u_hat = form.u_hat();
  VarPtr sigma_n_hat = form.sigma_n_hat();
  BCPtr bc = form.solution()->bc();
  SpatialFilterPtr tInitial = SpatialFilter::matchingT(0);
  bc->addDirichlet(u_hat, tInitial, u_exact);
  SpatialFilterPtr xLeft = SpatialFilter::matchingX(0);
  SpatialFilterPtr xRight = SpatialFilter::matchingX(1);
  bc->addDirichlet(sigma_n_hat, xLeft | xRight, Function::zero());
  
  MeshPtr mesh = form.solution()->mesh();
  string outputDir = ".";
  HDF5Exporter exporter(mesh, "heat space-time", outputDir);
  HDF5Exporter uErrorExporter(mesh, "heat space-time u error", outputDir);

  int hWidth = 10, energyErrorWidth = 20, uErrorWidth = 30;
  if (rank==0) cout << setw(hWidth) << "h" << setw(energyErrorWidth) << "Energy Error" << setw(uErrorWidth) << "u error (L^2)" << endl;
  
  int cubEnrichment = 3;
  int numRefinements = 3;
  // do some refinements:
  for (int refinement=0; refinement<numRefinements+1; refinement++) {
    form.solve();
    
    exporter.exportSolution(form.solution(), refinement);
    
    double energyError = form.solution()->energyErrorTotal();
    int globalDofs = mesh->globalDofCount();
    int activeElements = mesh->getTopology()->activeCellCount();

    FunctionPtr u = Function::solution(form.u(), form.solution());
    double u_error_L2 = (u_exact - u)->l2norm(mesh, cubEnrichment);
    
    uErrorExporter.exportFunction(u_exact - u, "u_err", refinement);
    
    if (rank==0) cout << setw(hWidth) << h << setw(energyErrorWidth) << energyError << setw(uErrorWidth) << u_error_L2 << endl;

    if (refinement<numRefinements)
    {
      mesh->hRefine(mesh->getActiveCellIDs());
      h /= 2.0;
    }
  }
  
  return 0;
}