//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Function.h"
#include "GLVisExporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"
#include "Solution.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace Camellia;
using namespace std;

namespace
{
  bool fileExists(const string &path)
  {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
  }

  string readFile(const string &path)
  {
    ifstream in(path.c_str());
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
  }

  void removeGLVisDir(const string &dir)
  {
    remove((dir + "/step0.mesh").c_str());
    remove((dir + "/step0_trace.mesh").c_str());
    remove((dir + "/step0_x.gf").c_str());
    remove((dir + "/step0_velocity.gf").c_str());
    remove((dir + "/step0_u.gf").c_str());
    remove((dir + "/step0_trace_u_hat.gf").c_str());
    remove((dir + "/step0_trace_sigma_n_hat.gf").c_str());
    remove((dir + "/index.txt").c_str());
    rmdir(dir.c_str());
  }

  bool contains(const string &haystack, const string &needle)
  {
    return haystack.find(needle) != string::npos;
  }

  MeshPtr simpleQuadMesh()
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim, conformingTraces);
    int H1Order = 2;
    int delta_k = spaceDim;
    return MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, 1.0, 1.0, 1, 1);
  }

  TEUCHOS_UNIT_TEST( GLVisExporter, FunctionFiles )
  {
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const int rank = comm->getRank();
    const string dir = "/tmp/CamelliaGLVisExporterFunctionTest";

    if (rank == 0) removeGLVisDir(dir);
    comm->barrier();

    MeshPtr mesh = simpleQuadMesh();
    GLVisExporter exporter(mesh, "CamelliaGLVisExporterFunctionTest", "/tmp", true);
    exporter.setGatherToRankZero(true);
    exporter.setMinimumDegree(2);

    FunctionPtr x = Function::xn(1);
    FunctionPtr velocity = Function::vectorize(Function::xn(1), Function::yn(1));
    exporter.exportFunction(vector<FunctionPtr>{x, velocity}, vector<string>{"x", "velocity"}, 0.25);
    comm->barrier();

    if (rank == 0)
    {
      TEST_ASSERT(fileExists(dir + "/step0.mesh"));
      TEST_ASSERT(fileExists(dir + "/step0_x.gf"));
      TEST_ASSERT(fileExists(dir + "/step0_velocity.gf"));
      TEST_ASSERT(fileExists(dir + "/index.txt"));

      string meshFile = readFile(dir + "/step0.mesh");
      string scalarFile = readFile(dir + "/step0_x.gf");
      string vectorFile = readFile(dir + "/step0_velocity.gf");
      string indexFile = readFile(dir + "/index.txt");

      TEST_ASSERT(contains(meshFile, "MFEM mesh v1.0"));
      TEST_ASSERT(contains(meshFile, "dimension\n2"));
      TEST_ASSERT(contains(meshFile, "elements\n1"));
      TEST_ASSERT(contains(meshFile, "vertices\n4\n2"));

      TEST_ASSERT(contains(scalarFile, "FiniteElementCollection: L2_T1_2D_P2"));
      TEST_ASSERT(contains(scalarFile, "VDim: 1"));
      TEST_ASSERT(contains(scalarFile, "\n0.5\n"));

      TEST_ASSERT(contains(vectorFile, "FiniteElementCollection: L2_T1_2D_P2"));
      TEST_ASSERT(contains(vectorFile, "VDim: 2"));

      TEST_ASSERT(contains(indexFile, "step 0 time 0.25 degree 2"));
      TEST_ASSERT(contains(indexFile, "vars x velocity"));

      removeGLVisDir(dir);
    }
    comm->barrier();
  }

  TEUCHOS_UNIT_TEST( GLVisExporter, SolutionFiles )
  {
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const int rank = comm->getRank();
    const string dir = "/tmp/CamelliaGLVisExporterSolutionTest";

    if (rank == 0) removeGLVisDir(dir);
    comm->barrier();

    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim, conformingTraces);
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), 2, spaceDim, 1.0, 1.0, 1, 1);
    SolutionPtr solution = Solution::solution(mesh);
    map<int, FunctionPtr> solutionMap;
    solutionMap[form.u()->ID()] = Function::xn(1);
    solutionMap[form.u_hat()->ID()] = Function::yn(1);
    solutionMap[form.sigma_n_hat()->ID()] = Function::normal()->x();
    solution->projectOntoMesh(solutionMap);

    GLVisExporter exporter(mesh, "CamelliaGLVisExporterSolutionTest", "/tmp", true);
    exporter.setGatherToRankZero(true);
    exporter.exportSolution(solution, 1.5);
    comm->barrier();

    if (rank == 0)
    {
      TEST_ASSERT(fileExists(dir + "/step0.mesh"));
      TEST_ASSERT(fileExists(dir + "/step0_trace.mesh"));
      TEST_ASSERT(fileExists(dir + "/step0_u.gf"));
      TEST_ASSERT(fileExists(dir + "/step0_trace_u_hat.gf"));
      TEST_ASSERT(fileExists(dir + "/step0_trace_sigma_n_hat.gf"));
      TEST_ASSERT(fileExists(dir + "/index.txt"));

      string gfFile = readFile(dir + "/step0_u.gf");
      string traceMeshFile = readFile(dir + "/step0_trace.mesh");
      string traceFile = readFile(dir + "/step0_trace_u_hat.gf");
      string fluxFile = readFile(dir + "/step0_trace_sigma_n_hat.gf");
      string indexFile = readFile(dir + "/index.txt");
      TEST_ASSERT(contains(gfFile, "FiniteElementSpace"));
      TEST_ASSERT(contains(gfFile, "VDim: 1"));
      TEST_ASSERT(contains(traceMeshFile, "dimension\n1"));
      TEST_ASSERT(contains(traceMeshFile, "elements\n4"));
      TEST_ASSERT(contains(traceMeshFile, "vertices\n8\n2"));
      TEST_ASSERT(contains(traceFile, "FiniteElementCollection: L2_T1_1D_P2"));
      TEST_ASSERT(contains(traceFile, "VDim: 1"));
      TEST_ASSERT(contains(fluxFile, "FiniteElementCollection: L2_T1_1D_P2"));
      TEST_ASSERT(contains(fluxFile, "VDim: 1"));
      TEST_ASSERT(contains(indexFile, "step 0 time 1.5"));
      TEST_ASSERT(contains(indexFile, "vars"));
      TEST_ASSERT(contains(indexFile, "u"));
      TEST_ASSERT(contains(indexFile, "step 0 trace time 1.5"));
      TEST_ASSERT(contains(indexFile, "u_hat"));
      TEST_ASSERT(contains(indexFile, "sigma_n_hat"));

      removeGLVisDir(dir);
    }
    comm->barrier();
  }

  TEUCHOS_UNIT_TEST( GLVisExporter, AvoidsDirectoryConflictsByDefault )
  {
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const int rank = comm->getRank();
    const string baseDir = "/tmp/CamelliaGLVisExporterConflictTest";
    const string secondDir = baseDir + "-2";

    if (rank == 0)
    {
      removeGLVisDir(baseDir);
      removeGLVisDir(secondDir);
      mkdir(baseDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH | S_IROTH);
    }
    comm->barrier();

    MeshPtr mesh = simpleQuadMesh();
    GLVisExporter exporter(mesh, "CamelliaGLVisExporterConflictTest", "/tmp");
    comm->barrier();

    if (rank == 0)
    {
      TEST_EQUALITY(exporter.outputDirName(), "CamelliaGLVisExporterConflictTest-2");
      TEST_ASSERT(fileExists(secondDir));
      removeGLVisDir(baseDir);
      removeGLVisDir(secondDir);
    }
    comm->barrier();
  }

  TEUCHOS_UNIT_TEST( GLVisExporter, RejectsUnsupportedExports )
  {
    Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
    const int rank = comm->getRank();
    const string dir = "/tmp/CamelliaGLVisExporterRejectTest";

    if (rank == 0) removeGLVisDir(dir);
    comm->barrier();

    MeshPtr mesh = simpleQuadMesh();
    GLVisExporter exporter(mesh, "CamelliaGLVisExporterRejectTest", "/tmp", true);

    bool timeSlabThrew = false;
    try
    {
      exporter.exportTimeSlab(Function::xn(1), "x");
    }
    catch (const std::invalid_argument &)
    {
      timeSlabThrew = true;
    }
    TEST_ASSERT(timeSlabThrew);

    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim, conformingTraces);
    bool divideIntoTriangles = true;
    MeshPtr triMesh = MeshFactory::quadMeshMinRule(form.bf(), 2, spaceDim, 1.0, 1.0, 1, 1, divideIntoTriangles);
    GLVisExporter triangleExporter(triMesh, "CamelliaGLVisExporterRejectTest", "/tmp", true);

    bool triangleThrew = false;
    try
    {
      triangleExporter.exportFunction(Function::xn(1), "x");
    }
    catch (const std::invalid_argument &)
    {
      triangleThrew = true;
    }
    TEST_ASSERT(triangleThrew);

    if (rank == 0) removeGLVisDir(dir);
    comm->barrier();
  }
}
