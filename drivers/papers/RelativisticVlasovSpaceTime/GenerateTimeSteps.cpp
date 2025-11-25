//
// © 2016-2025 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "HDF5Exporter.h"
#include "MeshFactory.h"

using namespace Camellia;


int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
  {
    mpiSession.barrier();

    double startTime = 0.0;
    double   endTime = 1.0;
    int numTimeSteps = 100;
    std::string outputDir = ".";
    std::string vizName = "time-steps";
    
    cmdp.setOption("startTime",    &startTime,    "start time");
    cmdp.setOption("endTime",      &endTime,      "end time");
    cmdp.setOption("numTimeSteps", &numTimeSteps, "number of time steps");
    cmdp.setOption("outputDir", &outputDir, "output directory");
    
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }
    
    VarFactoryPtr vf = VarFactory::varFactory();
    auto zeroVar = vf->fieldVar("zero");
    
    BFPtr bf = BF::bf(vf);
    
    const double x0 = 0.0;
    const double x1 = 1.0;
    
    const int numElements = 1;
    const int  p = 1;
    const int dp = 0;
    auto mesh = MeshFactory::intervalMesh(bf, x0, x1, numElements, p, dp);
    
    SolutionPtr soln = Solution::solution(bf, mesh);
    
    auto zeroFxn = Function::zero();
    std::map<int,FunctionPtr> zeroMap;
    zeroMap[zeroVar->ID()] = zeroFxn;
    
    soln->projectOntoMesh(zeroMap);
    
    auto timeStepsHDF5 = Teuchos::rcp(new HDF5Exporter(mesh, vizName, outputDir));
    
    double  t = startTime;
    double dt = (endTime - startTime) / numTimeSteps;
    
    timeStepsHDF5->exportSolution(soln, t);
    
    for (int i=0; i<numTimeSteps; i++)
    {
      t += dt;
      timeStepsHDF5->exportSolution(soln, t);
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
