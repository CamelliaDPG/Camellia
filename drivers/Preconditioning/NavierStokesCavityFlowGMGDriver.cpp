//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaDebugUtility.h"
#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "NavierStokesVGPFormulation.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "SuperLUDistSolver.h"
#include "TimeSteppingConstants.h"

using namespace Camellia;

void setDirectSolver(NavierStokesVGPFormulation &form)
{
  Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
  form.setSolver(coarseSolver);
}

void setGMGSolver(NavierStokesVGPFormulation &form, vector<MeshPtr> &meshesCoarseToFine,
                  int cgMaxIters, double cgTol, bool useCondensedSolve)
{
  Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true);
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(form.solutionIncrement(), meshesCoarseToFine, cgMaxIters, cgTol,
                                                                  GMGOperator::V_CYCLE, coarseSolver, useCondensedSolve) );
  gmgSolver->setAztecOutput(0);
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
  form.setSolver(gmgSolver);
}

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class RampBoundaryFunction_U1 : public SimpleFunction<double>
{
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps)
  {
    _eps = eps;
  }
  double value(double x, double y)
  {
    double tol = 1e-14;
    if (abs(y-1.0) < tol)   // top boundary
    {
      if ( (abs(x) < _eps) )   // top left
      {
        return x / _eps;
      }
      else if ( abs(1.0-x) < _eps)     // top right
      {
        return (1.0-x) / _eps;
      }
      else     // top middle
      {
        return 1;
      }
    }
    else     // not top boundary: 0.0
    {
      return 0.0;
    }
  }
  double value(double x, double y, double z)
  {
    // bilinear interpolation with ramp of width _eps around top edges
    double tol = 1e-14;
    if (abs(y-1.0) <tol)
    {
      double xFactor = 1.0;
      double zFactor = 1.0;
      if ( (abs(x) < _eps) )   // top left
      {
        xFactor = x / _eps;
      }
      else if ( abs(1.0-x) < _eps)     // top right
      {
        xFactor = (1.0-x) / _eps;
      }
      if ( (abs(z) < _eps) )   // top back
      {
        zFactor = z / _eps;
      }
      else if ( abs(1.0-z) < _eps)     // top front
      {
        zFactor = (1.0-z) / _eps;
      }
      return xFactor * zFactor;
    }
    else
    {
      return 0.0;
    }
  }
};

class TimeRamp : public SimpleFunction<double>
{
  FunctionPtr _time;
  double _timeScale;
  double getTimeValue()
  {
    ParameterFunction* timeParamFxn = dynamic_cast<ParameterFunction*>(_time.get());
    SimpleFunction<double>* timeFxn = dynamic_cast<SimpleFunction<double>*>(timeParamFxn->getValue().get());
    return timeFxn->value(0);
  }
public:
  TimeRamp(FunctionPtr timeConstantParamFxn, double timeScale)
  {
    _time = timeConstantParamFxn;
    _timeScale = timeScale;
  }
  double value(double x)
  {
    double t = getTimeValue();
    if (t >= _timeScale)
    {
      return 1.0;
    }
    else
    {
      return t / _timeScale;
    }
  }
};

using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  int spaceDim = 2;
  double eps = 1.0 / 64.0;

  bool useDirectSolver = false;
  
  bool useConformingTraces = true;
  double mu = 1.0;

  int polyOrder = 4, delta_k = 2;
  int meshWidth = 2;
  int polyOrderCoarse = 1;
  
  bool useCondensedSolve = true;
  int maxCGIters = 10000;
  double cgTol = 1e-6;
  
  bool useFixedNewtonTolerance = true;
  double fixedNewtonTolerance = 1e-6;
  double initialNewtonTolerance = 1e-4;
  int maxNewtonSteps = 10;
  
  double Re = 100;
  double energyThreshold = 0.2;
  
  bool refineUniformly = false;
  bool printRefinementsOnRankZero = false;
  bool enhanceFieldsForH1TracesWhenConforming = false;
  bool outputSolution = false;
  
  int azOutput = 0;
  
  int maxRefinements = 8;
  
  cmdp.setOption("numCells",&meshWidth,"mesh width");
  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("outputSolution", "dontOutputSolution", &outputSolution, "Write solution at each refinement step");
  
  cmdp.setOption("eps",&eps,"width of the 'ramp' between 0 and 1 for lid BCs");
  cmdp.setOption("energyThreshold", &energyThreshold, "energy threshold (fraction of max element error) for refinements");
  
  cmdp.setOption("useFixedNewtonTolerance", "useAdaptiveNewtonTolerance", &useFixedNewtonTolerance);
  cmdp.setOption("fixedNewtonTolerance",&fixedNewtonTolerance,"fixed Newton stopping criterion (L^2 norm of field variables)");
  cmdp.setOption("initialNewtonTolerance",&initialNewtonTolerance,"initial Newton stopping criterion for adaptive Newton tolerance (L^2 norm of field variables)");
  
  cmdp.setOption("refineUniformly", "refineAdaptively", &refineUniformly);
  cmdp.setOption("verboseRefinements", "silentRefinements", &printRefinementsOnRankZero);
  cmdp.setOption("Re", &Re, "Reynolds number");
  cmdp.setOption("azOutput", &azOutput);
  
//  cmdp.setOption("coarsePolyOrder", &k_coarse, "polynomial order for field variables on coarse grid");
  
//  cmdp.setOption("coarseSolver", &coarseSolverChoiceString, "coarse solver choice: KLU, MUMPS, SuperLUDist, SimpleML");
  
//  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  
//  cmdp.setOption("azOutput", &azOutput);
  
//  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full-V, Full-W, or Two-level");
//  cmdp.setOption("numGrids", &numGrids, "Number of grid levels to use (-1 means all).");
  
  
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &useConformingTraces);
  cmdp.setOption("enhanceFieldsForH1TracesWhenConforming", "equalOrderFieldsForH1TracesWhenConforming", &enhanceFieldsForH1TracesWhenConforming);
  
  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (2 or 3)");
  
  cmdp.setOption("maxIterations", &maxCGIters, "maximum number of CG iterations");
  cmdp.setOption("cgTol", &cgTol, "CG convergence tolerance");
  
  cmdp.setOption("maxSteps", &maxNewtonSteps, "maximum number of Newton steps");
  
  cmdp.setOption("maxRefs", &maxRefinements, "maximum number of adaptive refinements");
//  cmdp.setOption("refTol", &refTol, "energy error tolerance for refinements");
  
//  cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");
  cmdp.setOption("useDirect", "useIterative", &useDirectSolver);
  
//  cmdp.setOption("useDiagonalSchwarzWeighting","dontUseDiagonalSchwarzWeighting",&useDiagonalSchwarzWeighting);
//  cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");
  
//  cmdp.setOption("writeOpToFile", "dontWriteOpToFile", &writeOpToFile);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  if ((polyOrder == 0) && (polyOrderCoarse != 0))
  {
    if (rank == 0) cout << "polyOrder = 0; setting polyOrderCoarse to 0, too.\n";
    polyOrderCoarse = 0;
  }
  
  bool printRefinementsToConsole = printRefinementsOnRankZero ? (rank==0) : false;
  
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  if (enhanceFieldsForH1TracesWhenConforming && useConformingTraces)
  {
    for (int i=1; i<=spaceDim; i++)
    {
      string u_i_name = StokesVGPFormulation::u_name(i);
      parameters.set(u_i_name + "-polyOrderAdjustment", 1);
    }
  }
  
  vector<double> dims(spaceDim,1.0);
  vector<int> numElements(spaceDim,meshWidth);
  vector<double> x0(spaceDim,0.0);
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
  NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, polyOrder, delta_k);
  form.addPointPressureCondition({0.5,0.5});
  form.solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
  form.getRefinementStrategy()->setRelativeErrorThreshold(energyThreshold);
 
  
  MeshPtr mesh = form.solutionIncrement()->mesh();
  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
  
  bool excludeFluxesAndTracesFromRieszRepFunctionals = true;
  LinearTermPtr backgroundSolnFunctional = form.bf()->testFunctional(form.solution(), excludeFluxesAndTracesFromRieszRepFunctionals);
  RieszRep solnRieszRep(form.solution()->mesh(), form.solutionIncrement()->ip(), backgroundSolnFunctional);

  FunctionPtr zeroVector = Function::zero(1);
  RHSPtr rhsForResidual = form.rhs(zeroVector, false); // false : don't exclude fluxes and traces from the RHS

//  LinearTermPtr energyErrorFunctional = rhsForResidual->linearTerm(); // after we've accumulated the solution increment, the RHS has the entire residual.
//  RieszRepPtr energyErrorRieszRep = Teuchos::rcp( new RieszRep(form.solution()->mesh(), form.solutionIncrement()->ip(), energyErrorFunctional) );
  FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(form.solutionIncrement()) );
  
//  FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(energyErrorRieszRep) );
  
  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);

  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1);
  SpatialFilterPtr notTopBoundary = SpatialFilter::negatedFilter(topBoundary);
  form.addWallCondition(notTopBoundary);

  FunctionPtr u1_topRamp = Teuchos::rcp( new RampBoundaryFunction_U1(eps) );
  FunctionPtr u_topRamp;
  FunctionPtr zero = Function::zero();
  if (spaceDim == 2)
  {
    u_topRamp = Function::vectorize(u1_topRamp,zero);
  }
  else
  {
    u_topRamp = Function::vectorize(u1_topRamp,zero,zero);
  }
  form.addInflowCondition(topBoundary, u_topRamp);

  vector<vector<int>> iterationCounts;
  vector<int> elementCounts;
  vector<double> hMins, hMaxes;
  vector<double> energyErrors;
  
  double nonlinearThreshold = useFixedNewtonTolerance ? fixedNewtonTolerance : initialNewtonTolerance;
  int refNumber = 0;
  
  cout << setprecision(2) << scientific;
  auto nonlinearSolve = [&form, &rank, &meshesCoarseToFine, &cgTol, &useCondensedSolve, &useDirectSolver,
                         &iterationCounts, &nonlinearThreshold, &initialNewtonTolerance, &maxNewtonSteps,
                         &maxCGIters, &solnRieszRep] () -> void
  {
    double l2NormOfIncrement = nonlinearThreshold + 1000;
    int stepNumber = 0;
    
    vector<int> iterationCountsThisStep;
    
    while ((l2NormOfIncrement > nonlinearThreshold) && (stepNumber < maxNewtonSteps))
    {
      if (useDirectSolver)
        setDirectSolver(form);
      else
        setGMGSolver(form, meshesCoarseToFine, maxCGIters, cgTol, useCondensedSolve);
      
      form.solveAndAccumulate();
      l2NormOfIncrement = form.L2NormSolutionIncrement();
      stepNumber++;
      
      if (rank==0) cout << stepNumber << ". L^2 norm of increment: " << l2NormOfIncrement;
      
      if (!useDirectSolver)
      {
        Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(dynamic_cast<GMGSolver*>(form.getSolver().get()), false);
        int iterationCount = gmgSolver->iterationCount();
        iterationCountsThisStep.push_back(iterationCount);
        if (rank==0) cout << " (" << iterationCount << " GMG iterations)\n";
      }
      else
      {
        iterationCountsThisStep.push_back(-1);
        if (rank==0) cout << endl;
      }
    }
    iterationCounts.push_back(iterationCountsThisStep);
  };
  
  nonlinearSolve();
  
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->getTopology()->activeCellCount();
  elementCounts.push_back(activeElements);
  
  double hMax, hMin, hRatio;
  
  auto updateHValues = [mesh, &hMin, &hMax, &hRatio, spaceDim] () -> void
  {
    FunctionPtr h = Function::h();
    const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
    double my_hMax = 0, my_hMin = 1e10;
    for (GlobalIndexType cellID : *myCells)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      int numCells = 1;
      int numPoints = basisCache->getRefCellPointsFC().dimension(1);
      Camellia::FieldContainer<double> values(numCells,numPoints);
      h->values(values, basisCache);
      my_hMax = max(values[0],my_hMax);
      my_hMin = min(values[0],my_hMin);
    }
    mesh->Comm()->MaxAll(&my_hMax, &hMax, 1);
    mesh->Comm()->MinAll(&my_hMin, &hMin, 1);
    hRatio = hMax / hMin;
  };
  
  updateHValues();
  hMins.push_back(hMin);
  hMaxes.push_back(hMax);
//  energyErrors.push_back(energyError);

  if (rank==0) cout << "Initial mesh has " << activeElements << " elements and " << globalDofs << " global dofs, ";
  if (rank==0) cout << "with hMax/hMin = " << hRatio << ")\n";

  ostringstream exportName;
  exportName << "navierStokesCavityGMGSolution_Re" << Re << "_k" << polyOrder << "_deltak" << delta_k;
  
  HDF5Exporter exporter(mesh, exportName.str());
  if (outputSolution)
  {
    exporter.exportSolution(form.solution(),0);
  }
  
  ostringstream energyExportName;
  energyExportName << exportName.str() << "_energyError";
  HDF5Exporter energyExporter(mesh, energyExportName.str());
  string energyErrorName = "energy error ";
  vector<FunctionPtr> errorFunctions = {energyErrorFunction};
  vector<string> errorFunctionNames = {"energy error"};
  
//  map<int,VarPtr> testVars = form.solutionIncrement()->mesh()->bilinearForm()->varFactory()->testVars();
//  for (auto entry : testVars)
//  {
//    VarPtr testVar = entry.second;
//    string errFxnName = energyErrorName + testVar->name();
//    FunctionPtr repFxn = Teuchos::rcp( new RepFunction<double>(testVar, energyErrorRieszRep) );
//    errorFunctionNames.push_back(errFxnName);
//    errorFunctions.push_back(repFxn);
//  }
  
//  energyErrorRieszRep->computeRieszRep(polyOrder);
  if (outputSolution)
  {
    energyExporter.exportFunction(errorFunctions, errorFunctionNames,0);
  }
  
  do
  {
    refNumber++;
    if (!refineUniformly)
    {
      solnRieszRep.computeRieszRep(polyOrder); // enrich by polyOrder (integrate exactly)
      double solnNorm = solnRieszRep.getNorm();
      
      form.refine(rank==0); // print to console on rank 0
    
      double energyError = form.getRefinementStrategy()->getEnergyError(refNumber-1);
      
      energyErrors.push_back(energyError / solnNorm);
      
      if (rank == 0) cout << "solnNorm: " << solnNorm << "; abs. energyError: " << energyError << "; rel. error: " << energyError / solnNorm << endl;
      // update nonlinear threshold if we're doing that
      if (!useFixedNewtonTolerance)
      {
        double newThreshold = initialNewtonTolerance * (energyError / solnNorm);
        nonlinearThreshold = newThreshold; //min(nonlinearThreshold, newThreshold);
        if (newThreshold == nonlinearThreshold)
        {
          if (rank == 0) cout << "Updated nonlinear threshold to " << nonlinearThreshold << endl;
        }
      }
    }
    else
    {
      form.refineUniformly();
    }
    
    updateHValues();
    hMins.push_back(hMin);
    hMaxes.push_back(hMax);
  
    meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, polyOrderCoarse, delta_k);
    
    nonlinearSolve();
    
    if (outputSolution)
    {
      exporter.exportSolution(form.solution(), refNumber);
    
//    energyErrorRieszRep->computeRieszRep(polyOrder);
      energyExporter.exportFunction(errorFunctions, errorFunctionNames,refNumber);
    }
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->getTopology()->activeCellCount();
    
    if (rank==0) cout << "Refinement " << refNumber << " mesh has " << activeElements << " elements and " << globalDofs << " global dofs, ";
    if (rank==0) cout << "with hMax/hMin = " << hRatio << ".\n";
    elementCounts.push_back(activeElements);
  }
  while (refNumber < maxRefinements);

  // compute last energy error:
  solnRieszRep.computeRieszRep(polyOrder); // enrich by polyOrder (integrate exactly)
  double solnNorm = solnRieszRep.getNorm();
  double energyError = form.solutionIncrement()->energyErrorTotal(); //energyErrorRieszRep->getNorm();
  energyErrors.push_back(energyError / solnNorm);
  
  if (rank == 0)
  {
    ostringstream tout;
    
    vector<int> colWidths = {20,20,20,20,20,20,20,20,20,20};

    // Ref. \# & $h_{\rm max}$ & $h_{\rm min}$ & $\frac{h_{\rm max}}{h_{\rm min}}$ & Elements & Energy Error
    
    tout << setw(colWidths[0]) << "Ref. \\#"; //
    tout << setw(colWidths[1]) << "& $h_{\\rm max}$";
    tout << setw(colWidths[2]) << "& $h_{\\rm min}$";
    tout << setw(colWidths[3]) << "& $\\frac{h_{\\rm max}}{h_{\\rm min}}$";
    tout << setw(colWidths[4]) << "& Elements";
    tout << setw(colWidths[5]) << "& Energy Err.";
    tout << setw(colWidths[6]) << "& Nonlinear Steps";
    tout << setw(colWidths[7]) << "& Total Iterations";
    tout << setw(colWidths[8]) << "& Per Step\\\\\n";
    
    int numRefs = iterationCounts.size()-1;
    for (int i=0; i<=numRefs; i++)
    {
      tout << setw(colWidths[0]) << i;
      
      tout << setw(colWidths[1]) << "&" << "1/" << (int)(1/hMins[i]);
      tout << setw(colWidths[2]) << "&" << "1/" << (int)(1/hMaxes[i]);
      tout << setw(colWidths[3]) << "&" << (int)(hMaxes[i]/hMins[i]);
      tout << setw(colWidths[4]) << "&" << elementCounts[i];
      
      tout << setprecision(2);
      tout << std::scientific;
      tout << setw(colWidths[5]) << "&" << energyErrors[i];
      
      tout << setprecision(6);
      tout.unsetf(ios_base::floatfield);
      
      int numSteps = iterationCounts[i].size();
      int totalIterations = 0;
      for (int count : iterationCounts[i])
      {
        totalIterations += count;
      }
      int averageIterations = round(((double)totalIterations) / (double) numSteps);
      
      tout << setw(colWidths[6]) << "&" << numSteps;
      tout << setw(colWidths[7]) << "&" << totalIterations;
      tout << setw(colWidths[8]) << "&" << averageIterations;
      tout << "\\\\\n";
    }
    
    ostringstream resultsExportName;
    resultsExportName << exportName.str() << "_results.txt";
    ofstream fout(resultsExportName.str().c_str());
    fout << tout.str();
    fout.close();
    
    cout << tout.str();
  }
  
  // gather mesh topologies used in meshesForMultigrid, so we can output them using GnuPlotUtil
  vector<MeshTopologyPtr> meshTopos;
  for (MeshPtr mesh : meshesCoarseToFine)
  {
    MeshTopologyPtr meshTopo = mesh->getTopology()->getGatheredCopy();
    meshTopos.push_back(meshTopo);
  }
  
  // now, on rank 0, output:
  if (rank == 0)
  {
    for (int i=0; i<meshTopos.size(); i++)
    {
      ostringstream meshExportName;
      meshExportName << exportName.str() << "_mesh" << i;
      int numPointsPerEdge = 2;
      bool labelCells = false;
      string meshColor = "black";
      GnuPlotUtil::writeExactMeshSkeleton(meshExportName.str(), meshTopos[i].get(), numPointsPerEdge, labelCells, meshColor);
    }
  }
  
  return 0;
}
