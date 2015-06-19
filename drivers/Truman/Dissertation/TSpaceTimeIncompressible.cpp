#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Amesos_config.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "GMGSolver.h"
#include "SpaceTimeIncompressibleFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace Camellia;

class IncompressibleProblem
{
  protected:
    FunctionPtr _u1_exact;
    FunctionPtr _u2_exact;
    FunctionPtr _sigma1_exact;
    FunctionPtr _sigma2_exact;
    vector<double> _x0;
    vector<double> _dimensions;
    vector<int> _elementCounts;
    double _tInit;
    double _tFinal;
    int _numSlabs = 1;
    int _currentStep = 0;
    bool _steady;
  public:
    LinearTermPtr forcingTerm = Teuchos::null;
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1) = 0;
    virtual void setBCs(SpaceTimeIncompressibleFormulationPtr form) = 0;
    virtual double computeL2Error(SpaceTimeIncompressibleFormulationPtr form, SolutionPtr solutionBackground) = 0;
    int numSlabs() { return _numSlabs; }
    int currentStep() { return _currentStep; }
    void advanceStep() { _currentStep++; }
    double stepSize() { return (_tFinal-_tInit)/_numSlabs; }
    double currentT0() { return stepSize()*_currentStep; }
    double currentT1() { return stepSize()*(_currentStep+1); }
};

class AnalyticalIncompressibleProblem : public IncompressibleProblem
{
  protected:
    map<int, FunctionPtr> _exactMap;
  public:
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1)
    {
      MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(_dimensions, _elementCounts, _x0);
      MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, currentT0(), currentT1());
      if (_steady)
        return spatialMeshTopo;
      else
        return spaceTimeMeshTopo;
    }

    void initializeExactMap(SpaceTimeIncompressibleFormulationPtr form)
    {
      _exactMap[form->u(1)->ID()] = _u1_exact;
      _exactMap[form->u(2)->ID()] = _u2_exact;
      _exactMap[form->sigma(1,1)->ID()] = _sigma1_exact->x();
      _exactMap[form->sigma(1,2)->ID()] = _sigma1_exact->y();
      _exactMap[form->sigma(2,1)->ID()] = _sigma2_exact->x();
      _exactMap[form->sigma(2,2)->ID()] = _sigma2_exact->y();
      _exactMap[form->uhat(1)->ID()] = form->uhat(1)->termTraced()->evaluate(_exactMap);
      _exactMap[form->uhat(2)->ID()] = form->uhat(2)->termTraced()->evaluate(_exactMap);
    }

    void projectExactSolution(SolutionPtr solution)
    {
      solution->projectOntoMesh(_exactMap);
    }

    virtual void setBCs(SpaceTimeIncompressibleFormulationPtr form)
    {
      initializeExactMap(form);

      BCPtr bc = form->solutionUpdate()->bc();
      SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
      SpatialFilterPtr leftX  = SpatialFilter::matchingX(_x0[0]);
      SpatialFilterPtr rightX = SpatialFilter::matchingX(_x0[0]+_dimensions[0]);
      SpatialFilterPtr leftY  = SpatialFilter::matchingY(_x0[1]);
      SpatialFilterPtr rightY = SpatialFilter::matchingY(_x0[1]+_dimensions[1]);
      bc->addDirichlet(form->uhat(1), leftX,    _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), leftX,    _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), rightX,   _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), rightX,   _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), leftY,    _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), leftY,    _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), rightY,   _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), rightY,   _exactMap[form->uhat(2)->ID()]);
      if (!_steady)
      {
        bc->addDirichlet(form->tmhat(1),initTime,-_exactMap[form->uhat(1)->ID()]);
        bc->addDirichlet(form->tmhat(2),initTime,-_exactMap[form->uhat(2)->ID()]);
      }
    }
    double computeL2Error(SpaceTimeIncompressibleFormulationPtr form, SolutionPtr solutionBackground)
    {
      FunctionPtr u1_soln, u2_soln, sigma11_soln, sigma12_soln, sigma21_soln, sigma22_soln,
                  u1_diff, u2_diff, sigma11_diff, sigma12_diff, sigma21_diff, sigma22_diff,
                  u1_sqr, u2_sqr, sigma11_sqr, sigma12_sqr, sigma21_sqr, sigma22_sqr;
      u1_soln = Function::solution(form->u(1), solutionBackground);
      u2_soln = Function::solution(form->u(2), solutionBackground);
      sigma11_soln = Function::solution(form->sigma(1,1), solutionBackground);
      sigma12_soln = Function::solution(form->sigma(1,2), solutionBackground);
      sigma21_soln = Function::solution(form->sigma(2,1), solutionBackground);
      sigma22_soln = Function::solution(form->sigma(2,2), solutionBackground);
      u1_diff = u1_soln - _u1_exact;
      u2_diff = u2_soln - _u2_exact;
      sigma11_diff = sigma11_soln - _sigma1_exact->x();
      sigma12_diff = sigma12_soln - _sigma1_exact->y();
      sigma21_diff = sigma21_soln - _sigma2_exact->x();
      sigma22_diff = sigma22_soln - _sigma2_exact->y();
      u1_sqr = u1_diff*u1_diff;
      u2_sqr = u2_diff*u2_diff;
      sigma11_sqr = sigma11_diff*sigma11_diff;
      sigma12_sqr = sigma12_diff*sigma12_diff;
      sigma21_sqr = sigma21_diff*sigma21_diff;
      sigma22_sqr = sigma22_diff*sigma22_diff;
      double u1_l2, u2_l2, sigma11_l2, sigma12_l2, sigma21_l2, sigma22_l2;
      u1_l2 = u1_sqr->integrate(solutionBackground->mesh(), 5);
      u2_l2 = u2_sqr->integrate(solutionBackground->mesh(), 5);
      sigma11_l2 = sigma11_sqr->integrate(solutionBackground->mesh(), 5);
      sigma12_l2 = sigma12_sqr->integrate(solutionBackground->mesh(), 5);
      sigma21_l2 = sigma21_sqr->integrate(solutionBackground->mesh(), 5);
      sigma22_l2 = sigma22_sqr->integrate(solutionBackground->mesh(), 5);
      double l2Error = sqrt(u1_l2+u2_l2+sigma11_l2+sigma12_l2+sigma21_l2+sigma22_l2);
      return l2Error;
    }
};

class KovasznayProblem : public AnalyticalIncompressibleProblem
{
  private:
  public:
    KovasznayProblem(bool steady, double Re)
    {
      _steady = steady;
      // problemName = "Kovasznay";
      double pi = atan(1)*4;
      double lambda = Re/2-sqrt(Re*Re/4+4*pi*pi);
      FunctionPtr explambdaX = Teuchos::rcp(new Exp_ax(lambda));
      FunctionPtr cos2piY = Teuchos::rcp(new Cos_ay(2*pi));
      FunctionPtr sin2piY = Teuchos::rcp(new Sin_ay(2*pi));
      _u1_exact = 1 - explambdaX*cos2piY;
      _u2_exact = lambda/(2*pi)*explambdaX*sin2piY;
      _sigma1_exact = 1./Re*_u1_exact->grad();
      _sigma2_exact = 1./Re*_u2_exact->grad();

      _x0.push_back(-.5);
      _x0.push_back(-.5);
      _dimensions.push_back(1.5);
      _dimensions.push_back(2.0);
      _elementCounts.push_back(3);
      _elementCounts.push_back(4);
      _tInit = 0.0;
      _tFinal = 0.25;
    }
};

class TaylorGreenProblem : public AnalyticalIncompressibleProblem
{
  private:
  public:
    TaylorGreenProblem(bool steady, double Re, int numXElems=2, int numSlabs=1)
    {
      _steady = steady;
      // problemName = "Kovasznay";
      double pi = atan(1)*4;
      FunctionPtr temporalDecay = Teuchos::rcp(new Exp_at(-2./Re));
      FunctionPtr sinX = Teuchos::rcp(new Sin_x());
      FunctionPtr cosX = Teuchos::rcp(new Cos_x());
      FunctionPtr sinY = Teuchos::rcp(new Sin_y());
      FunctionPtr cosY = Teuchos::rcp(new Cos_y());
      _u1_exact = sinX*cosY*temporalDecay;
      _u2_exact = -cosX*sinY*temporalDecay;
      _sigma1_exact = 1./Re*_u1_exact->grad();
      _sigma2_exact = 1./Re*_u2_exact->grad();

      _x0.push_back(0);
      _x0.push_back(0);
      _dimensions.push_back(2*pi);
      _dimensions.push_back(2*pi);
      _elementCounts.push_back(numXElems);
      _elementCounts.push_back(numXElems);
      _tInit = 0.0;
      _tFinal = 1.0;
      _numSlabs = numSlabs;
    }
};

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  // if (commRank == 0)
  // {
  //   int i = 0;
  //   char hostname[256];
  //   gethostname(hostname, sizeof(hostname));
  //   printf("PID %d on %s ready of attach\n", getpid(), hostname);
  //   fflush(stdout);
  //   while (0 == i)
  //     sleep(5);
  // }
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  // problem parameters:
  int spaceDim = 2;
  double Re = 40;
  bool steady = false;
  string problemChoice = "TaylorGreen";
  int numRefs = 1;
  int p = 2, delta_p = 2;
  int numXElems = 1;
  int numTElems = 1;
  int numSlabs = 1;
  bool useConformingTraces = false;
  string solverChoice = "KLU";
  double solverTolerance = 1e-8;
  double nonlinearTolerance = 1e-5;
  int maxLinearIterations = 1000;
  int maxNonlinearIterations = 20;
  bool computeL2Error = false;
  bool exportSolution = false;
  bool saveSolution = false;
  bool loadSolution = false;
  int loadRef = 0;
  int loadDirRef = 0;
  string norm = "Graph";
  string rootDir = ".";
  string tag="";
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("Re", &Re, "Re");
  cmdp.setOption("steady", "transient", &steady, "use steady incompressible Navier-Stokes");
  cmdp.setOption("problem", &problemChoice, "Kovasznay, TaylorGreen");
  cmdp.setOption("polyOrder",&p,"polynomial order for field variable u");
  cmdp.setOption("delta_p", &delta_p, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("numTElems",&numTElems,"number of elements in t direction");
  cmdp.setOption("numSlabs",&numSlabs,"number of time slabs to use");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLU, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");
  cmdp.setOption("nonlinearTolerance", &nonlinearTolerance, "nonlinear solver tolerance");
  cmdp.setOption("maxLinearIterations", &maxLinearIterations, "maximum number of iterations for linear solver");
  cmdp.setOption("maxNonlinearIterations", &maxNonlinearIterations, "maximum number of iterations for Newton solver");
  cmdp.setOption("outputDir", &rootDir, "output directory");
  cmdp.setOption("computeL2Error", "skipL2Error", &computeL2Error, "compute L2 error");
  cmdp.setOption("exportSolution", "skipExport", &exportSolution, "export solution to HDF5");
  cmdp.setOption("saveSolution", "skipSave", &saveSolution, "save mesh and solution to HDF5");
  cmdp.setOption("loadSolution", "skipLoad", &loadSolution, "load mesh and solution from HDF5");
  cmdp.setOption("loadRef", &loadRef, "load refinement number");
  cmdp.setOption("loadDirRef", &loadDirRef, "which refinement directory to load from");
  cmdp.setOption("tag", &tag, "output tag");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  map<string, Teuchos::RCP<IncompressibleProblem>> problems;
  problems["Kovasznay"] = Teuchos::rcp(new KovasznayProblem(steady, Re));
  problems["TaylorGreen"] = Teuchos::rcp(new TaylorGreenProblem(steady, Re, numXElems, numSlabs));
  Teuchos::RCP<IncompressibleProblem> problem = problems.at(problemChoice);

  // if (commRank == 0)
  // {
  //   Solver::printAvailableSolversReport();
  //   cout << endl;
  // }
  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);

  for (; problem->currentStep() < problem->numSlabs(); problem->advanceStep())
  {
    if (problem->numSlabs() > 1 && commRank == 0 && !steady)
      cout << "Solving time slab [" << problem->currentT0() << ", " << problem->currentT1() << "]" << endl;

    ostringstream problemName;
    problemName << problemChoice << spaceDim << "D_slab" << problem->currentStep() << "_" << norm << "_" << Re << "_p" << p << "_" << solverChoice;
    if (tag != "")
      problemName << "_" << tag;
    ostringstream saveDir;
    saveDir << problemName.str() << "_ref" << loadRef;

    int success = mkdir((rootDir+"/"+saveDir.str()).c_str(), S_IRWXU | S_IRWXG);

    string dataFileLocation = rootDir + "/" + saveDir.str() + "/" + saveDir.str() + ".data";
    string exportName = saveDir.str();

    ostringstream loadDir;
    loadDir << problemName.str() << "_ref" << loadDirRef;
    string loadFilePrefix = "";
    if (loadSolution)
    {
      loadFilePrefix = rootDir + "/" + loadDir.str() + "/" + saveDir.str();
      if (commRank == 0) cout << "Loading previous solution " << loadFilePrefix << endl;
    }
    // ostringstream saveDir;
    // saveDir << problemName.str() << "_ref" << loadRef;
    string saveFilePrefix = rootDir + "/" + saveDir.str() + "/" + problemName.str();
    if (saveSolution && commRank == 0) cout << "Saving to " << saveFilePrefix << endl;

    SpaceTimeIncompressibleFormulationPtr form = Teuchos::rcp(new SpaceTimeIncompressibleFormulation(spaceDim, steady, 1./Re,
          useConformingTraces, problem->meshTopology(numTElems), p, delta_p, norm, problem->forcingTerm, loadFilePrefix));

    MeshPtr mesh = form->solutionUpdate()->mesh();
    MeshPtr k0Mesh = Teuchos::rcp( new Mesh (mesh->getTopology()->deepCopy(), form->bf(), 1, delta_p) );
    mesh->registerObserver(k0Mesh);

    // Set up boundary conditions
    problem->setBCs(form);

    // Set up solution
    SolutionPtr solutionUpdate = form->solutionUpdate();
    SolutionPtr solutionBackground = form->solutionBackground();
    // dynamic_cast<AnalyticalIncompressibleProblem*>(problem.get())->projectExactSolution(solutionBackground);

    RefinementStrategyPtr refStrategy = form->getRefinementStrategy();
    Teuchos::RCP<HDF5Exporter> exporter;
    if (exportSolution)
      exporter = Teuchos::rcp(new HDF5Exporter(mesh,exportName, rootDir));

    Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");
    map<string, SolverPtr> solvers;
    solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
    solvers["SuperLUDist"] = Solver::getSolver(Solver::SuperLUDist, true);
#endif
#ifdef HAVE_AMESOS_MUMPS
    solvers["MUMPS"] = Solver::getSolver(Solver::MUMPS, true);
#endif
    bool useStaticCondensation = false;
    int azOutput = 20; // print residual every 20 CG iterations
    ofstream dataFile(dataFileLocation);
    dataFile << "ref\t " << "elements\t " << "dofs\t " << "energy\t " << "l2\t " << "solvetime\t" << "iterations\t " << endl;
    for (int refIndex=loadRef; refIndex <= numRefs; refIndex++)
    {
      double l2Update = 1e10;
      int iterCount = 0;
      solverTime->start(true);
      while (l2Update > nonlinearTolerance && iterCount < maxNonlinearIterations)
      {
        Teuchos::RCP<GMGSolver> gmgSolver;
        if (solverChoice[0] == 'G')
        {
          gmgSolver = Teuchos::rcp( new GMGSolver(solutionUpdate, k0Mesh, maxLinearIterations, solverTolerance, Solver::getDirectSolver(true), useStaticCondensation));
          gmgSolver->setAztecOutput(azOutput);
          if (solverChoice == "GMG-Direct")
            gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::Direct);
          if (solverChoice == "GMG-ILU")
            gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::ILU);
          if (solverChoice == "GMG-IC")
            gmgSolver->gmgOperator()->setSchwarzFactorizationType(GMGOperator::IC);
          solutionUpdate->solve(gmgSolver);
        }
        else
          solutionUpdate->condensedSolve(solvers[solverChoice]);

        // Compute L2 norm of update
        double u1L2Update = solutionUpdate->L2NormOfSolutionGlobal(form->u(1)->ID());
        double u2L2Update = solutionUpdate->L2NormOfSolutionGlobal(form->u(2)->ID());
        l2Update = sqrt(u1L2Update*u1L2Update + u2L2Update*u2L2Update);
        if (commRank == 0)
          cout << "Nonlinear Update:\t " << l2Update << endl;

        form->updateSolution();
        iterCount++;
      }
      double solveTime = solverTime->stop();

      double energyError = solutionUpdate->energyErrorTotal();
      double l2Error = 0;
      if (computeL2Error)
      {
        l2Error = problem->computeL2Error(form, solutionBackground);
      }
      if (commRank == 0)
      {
        cout << "Refinement: " << refIndex
          << " \tElements: " << mesh->numActiveElements()
          << " \tDOFs: " << mesh->numGlobalDofs()
          << " \tEnergy Error: " << energyError
          << " \tL2 Error: " << l2Error
          << " \tSolve Time: " << solveTime
          // << " \tIteration Count: " << iterationCount
          << endl;
        dataFile << refIndex
          << " " << mesh->numActiveElements()
          << " " << mesh->numGlobalDofs()
          << " " << energyError
          << " " << l2Error
          << " " << solveTime
          // << " " << iterationCount
          << endl;
      }

      if (exportSolution)
        exporter->exportSolution(solutionBackground, refIndex);

      if (saveSolution)
      {
        ostringstream saveFile;
        saveFile << saveFilePrefix << "_ref" << refIndex;
        form->save(saveFile.str());
      }

      if (refIndex != numRefs)
        refStrategy->refine();
    }
    dataFile.close();
  }
  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
