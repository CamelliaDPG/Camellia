//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  HeatGLVisExample.cpp
//
//  Transient 2D heat on the unit square, ultraweak DPG with backward Euler
//  time stepping, exported for GLVis viewing.
//

#include "Camellia.h"

#include "ExpFunction.h"
#include "HeatFormulation.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include <iomanip>

using namespace Camellia;
using namespace std;

const double PI = 3.141592653589793238462;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);

  {
    Teuchos::CommandLineProcessor cmdp(false,true);

    int polyOrder = 4;
    int numTimeSteps = 20;
    double dt = 0.005;
    int meshWidth = 4;
    string outputDir = "heatGLVis";
    bool overwriteOutput = false;

    cmdp.setOption("polyOrder", &polyOrder);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("dt", &dt);
    cmdp.setOption("meshWidth", &meshWidth);
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
    double theta = 1.0;
    bool conformingTraces = true;
    HeatFormulation form(spaceDim, conformingTraces, dt, theta, HeatFormulation::ULTRAWEAK);
    VarPtr u = form.u();
    VarPtr v = form.v();
    BFPtr bf = form.bf();

    ParameterFunctionPtr t = ParameterFunction::parameterFunction(0.0);
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u_init = TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::sin(PI * y);
    FunctionPtr u_exact = Exp<double>::exp(-2.0 * PI * PI * FunctionPtr(t)) * u_init;

    IPPtr ip = bf->graphNorm();
    BCPtr bc = BC::bc();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), Function::zero());

    int delta_k = spaceDim;
    int H1Order = polyOrder + 1;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, 1.0, 1.0, meshWidth, meshWidth);

    RHSPtr rhs = RHS::rhs();
    SolutionPtr soln      = Solution::solution(bf, mesh, bc, rhs, ip);
    SolutionPtr soln_prev = Solution::solution(bf, mesh, bc, rhs, ip);
    FunctionPtr u_curr = Function::solution(u, soln);
    FunctionPtr u_prev = Function::solution(u, soln_prev);
    rhs->addTerm((u_prev / dt) * v);

    map<int, FunctionPtr> initValueMap;
    initValueMap[u->ID()] = u_init;
    initValueMap[form.sigma()->ID()] = u_init->grad(spaceDim);
    soln_prev->projectOntoMesh(initValueMap, 0);

    int quadratureEnrichmentL2 = 10;

    GLVisExporter exporter(mesh, outputDir, ".", overwriteOutput);
    exporter.exportSolution(soln_prev, 0.0);

    for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
    {
      double time = (timeStepNumber + 1) * dt;
      t->setValue(time);

      soln->solve();
      soln_prev->setSolution(soln);

      double errL2 = (u_curr - u_exact)->l2norm(mesh, quadratureEnrichmentL2);
      if (rank == 0)
      {
        cout << "step " << setw(3) << timeStepNumber + 1 << ", t = " << time
             << ": u L2 error " << errL2 << endl;
      }
      exporter.exportSolution(soln, time);
    }

    if (rank == 0)
    {
      cout << "view a time step N (0.." << numTimeSteps << ") with:\n";
      cout << "  glvis -m " << exporter.outputDirPath() << "/stepN.mesh -g " << exporter.outputDirPath() << "/stepN_u.gf\n";
      int numProcs = Teuchos::GlobalMPISession::getNProc();
      if (numProcs > 1) cout << "  (MPI run: add -np " << numProcs << ")\n";
    }
  }

  Camellia::finalize(false);
  Kokkos::finalize();
  return 0;
}
