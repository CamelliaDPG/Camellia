//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  PoissonGLVisExample.cpp
//
//  Adaptive ultraweak Poisson on the unit square, exported for GLVis viewing.
//

#include "Camellia.h"

#include "ExpFunction.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);

  {
    Teuchos::CommandLineProcessor cmdp(false,true);

    int polyOrder = 3;
    int numRefinements = 4;
    string outputDir = "poissonGLVis";
    bool overwriteOutput = false;

    cmdp.setOption("polyOrder", &polyOrder);
    cmdp.setOption("numRefinements", &numRefinements);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("overwriteOutput", "noOverwriteOutput", &overwriteOutput);

    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      Camellia::finalize(false);
      Kokkos::finalize();
      return -1;
    }

    int rank = Teuchos::GlobalMPISession::getRank();

    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim, conformingTraces, PoissonFormulation::ULTRAWEAK);
    BFPtr bf = form.bf();

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    double a = 100.0;
    FunctionPtr r2 = (x - 0.7) * (x - 0.7) + (y - 0.7) * (y - 0.7);
    FunctionPtr u_exact = Exp<double>::exp(-a * r2);
    FunctionPtr forcing = (4.0 * a * a * r2 - 4.0 * a) * u_exact;

    int delta_k = spaceDim;
    int H1Order = polyOrder + 1;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, 1.0, 1.0, 4, 4);

    RHSPtr rhs = form.rhs(forcing);
    IPPtr ip = bf->graphNorm();
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_exact);

    SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);

    double energyThreshold = 0.2;
    RefinementStrategyPtr refStrategy = RefinementStrategy::energyErrorRefinementStrategy(solution, energyThreshold);

    FunctionPtr u_soln = Function::solution(form.u(), solution);
    int quadratureEnrichmentL2 = 10;

    GLVisExporter exporter(mesh, outputDir, ".", overwriteOutput);

    for (int refNumber = 0; refNumber <= numRefinements; refNumber++)
    {
      if (refNumber > 0) refStrategy->refine();
      solution->solve();

      double energyError = solution->energyErrorTotal();
      double errL2 = (u_soln - u_exact)->l2norm(mesh, quadratureEnrichmentL2);
      if (rank == 0)
      {
        cout << "refinement " << refNumber << ": " << mesh->numActiveElements() << " elements, "
             << mesh->numGlobalDofs() << " dofs, energy error " << energyError
             << ", u L2 error " << errL2 << endl;
      }
      exporter.exportSolution(solution, refNumber);
    }

    if (rank == 0)
    {
      cout << "view a refinement level N (0.." << numRefinements << ") with:\n";
      cout << "  glvis -m " << exporter.outputDirPath() << "/stepN.mesh -g " << exporter.outputDirPath() << "/stepN_u.gf\n";
      int numProcs = Teuchos::GlobalMPISession::getNProc();
      if (numProcs > 1) cout << "  (MPI run: add -np " << numProcs << ")\n";
    }
  }

  Camellia::finalize(false);
  Kokkos::finalize();
  return 0;
}
