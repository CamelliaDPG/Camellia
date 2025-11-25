//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_GlobalMPISession.hpp"

#include "Camellia.h"
#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "TimeSteppingConstants.h"

using namespace Camellia;

// this Function will work for both 2D and 3D cavity flow top BC (matching y = 1)
class RampBoundaryFunction_U1 : public SimpleFunction<double>
{
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps)
  :
  SimpleFunction<double>("RampBoundaryFunction_U1", 
                         VARIES_IN_DIMENSION_0 | VARIES_IN_DIMENSION_1 | VARIES_IN_DIMENSION_2)
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
  :
  SimpleFunction<double>("TimeRamp", VARIES_IN_NO_DIMENSION)
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
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  {
    int rank = Teuchos::GlobalMPISession::getRank();
    
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
    
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    int spaceDim = 2;
    double eps = 1.0 / 64.0;
    
    bool useConformingTraces = true;
    double mu = 1.0;
    
    int polyOrder = 4, delta_k = 2;
    int meshWidth = 2;
    
    bool useCondensedSolve = true;
    int maxIters = 1000;
    double cgTol = 1e-6;
    
    bool refineUniformly = false;
    bool printRefinementsOnRankZero = false;
    bool enhanceFieldsForH1TracesWhenConforming = false;
    
    int azOutput = 0;
    
    double refTol = 1e-2;
    int maxRefinements = 12;
    bool clearRefinedSolutions = false;
    
    cmdp.setOption("numCells",&meshWidth,"mesh width");
    cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
    cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
    cmdp.setOption("refineUniformly", "refineAdaptively", &refineUniformly);
    cmdp.setOption("verboseRefinements", "silentRefinements", &printRefinementsOnRankZero);
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
    
    cmdp.setOption("maxIterations", &maxIters, "maximum number of CG iterations");
    cmdp.setOption("cgTol", &cgTol, "CG convergence tolerance");
    
    cmdp.setOption("maxRefs", &maxRefinements, "maximum number of adaptive refinements");
    cmdp.setOption("refTol", &refTol, "energy error tolerance for refinements");
    
    //  cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");
    //  cmdp.setOption("solveDirectly", "solveIteratively", &solveDirectly);
    
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
    
    StokesVGPFormulation form(parameters);
    
    vector<double> dims(spaceDim,1.0);
    vector<int> numElements(spaceDim,meshWidth);
    vector<double> x0(spaceDim,0.0);
    
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    form.initializeSolution(meshTopo, polyOrder, delta_k);
    form.addPointPressureCondition();
    form.solution()->setUseCondensedSolve(useCondensedSolve);
    
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
    
    vector<int> iterationCounts, iterationCountsZeroGuess;
    vector<int> elementCounts;
    vector<double> hMins, hMaxes;
    vector<double> energyErrors;
    
    int iterationCount = form.solveIteratively(maxIters, cgTol);
    iterationCounts.push_back(iterationCount);
    iterationCountsZeroGuess.push_back(iterationCount);
    
    MeshPtr mesh = form.solution()->mesh();
    double energyError = form.solution()->energyErrorTotal();
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
      hMax = MPIWrapper::max(mesh->Comm(), my_hMax);
      hMin = MPIWrapper::min(mesh->Comm(), my_hMin);
      hRatio = hMax / hMin;
    };
    
    updateHValues();
    hMins.push_back(hMin);
    hMaxes.push_back(hMax);
    energyErrors.push_back(energyError);
    
    if (rank==0) cout << "Initial energy error: " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs, ";
    if (rank==0) cout << "with hMax/hMin = " << hRatio << ");  iteration count " << iterationCount << endl;
    
    HDF5Exporter exporter(mesh, "stokesCavityGMGSolution");
    exporter.exportSolution(form.solution(),0);
    
    vector<double> refStrategyEnergyErrors;
    
    int refNumber = 0;
    do
    {
      refNumber++;
      if (!refineUniformly)
      {
        form.refine(printRefinementsToConsole);
        refStrategyEnergyErrors.push_back(form.getRefinementStrategy()->getEnergyError(refNumber-1));
      }
      else
        form.refineUniformly();
      
      updateHValues();
      hMins.push_back(hMin);
      hMaxes.push_back(hMax);
      
      iterationCount = form.solveIteratively(maxIters, cgTol, azOutput);
      
      exporter.exportSolution(form.solution(), refNumber);
      
      energyError = form.solution()->energyErrorTotal();
      globalDofs = mesh->globalDofCount();
      activeElements = mesh->getTopology()->activeCellCount();
      energyErrors.push_back(energyError);
      
      if (rank==0) cout << "Energy error for refinement " << refNumber << ": " << energyError;
      if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs, ";
      if (rank==0) cout << "with hMax/hMin = " << hRatio << ");  iteration count " << iterationCount << endl;
      iterationCounts.push_back(iterationCount);
      elementCounts.push_back(activeElements);
      
      // again, but with zero initial guess:
      form.solution()->clear();
      form.solution()->initializeLHSVector();
      int iterationCountZeroGuess = form.solveIteratively(maxIters, cgTol, azOutput);
      iterationCountsZeroGuess.push_back(iterationCountZeroGuess);
    }
    while ((energyError > refTol) && (refNumber < maxRefinements));
    
    ostringstream exportName;
    exportName << "stokesCavityGMGSolution_k" << polyOrder << "_deltak" << delta_k << "_maxRefs" << maxRefinements;
    
    if (rank == 0)
    {
      ostringstream tout;
      
      vector<int> colWidths = {20,20,20,20,20,20,20,20,20};
      
      // Ref. \# & $h_{\rm max}$ & $h_{\rm min}$ & $\frac{h_{\rm max}}{h_{\rm min}}$ & Elements & Energy Error
      
      tout << setw(colWidths[0]) << "Ref. \\#"; //
      tout << setw(colWidths[1]) << "& $h_{\\rm max}$";
      tout << setw(colWidths[2]) << "& $h_{\\rm min}$";
      tout << setw(colWidths[3]) << "& $\\frac{h_{\\rm max}}{h_{\\rm min}}$";
      tout << setw(colWidths[4]) << "& Elements";
      tout << setw(colWidths[5]) << "& Energy Err.";
      tout << setw(colWidths[6]) << "& Iterations";
      tout << setw(colWidths[7]) << "& w/Zero Init. Guess\\\\\n";
      
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
        
        tout << setw(colWidths[6]) << "&" << iterationCounts[i];
        
        tout << setw(colWidths[7]) << "&" << iterationCountsZeroGuess[i];
        tout << "\\\\\n";
      }
      
      ostringstream resultsExportName;
      resultsExportName << exportName.str() << "_results.txt";
      ofstream fout(resultsExportName.str().c_str());
      fout << tout.str();
      fout.close();
      
      cout << tout.str();
    }
    
    {
      // mesh visualization output
      // gather mesh topologies used in meshesForMultigrid, so we can output them using GnuPlotUtil
      
      
      vector<MeshTopologyPtr> meshTopos;
      int kCoarse = 1;
      vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, kCoarse, 1);
      int i = 0;
      for (MeshPtr mesh : meshesCoarseToFine)
      {
        MeshTopologyPtr meshTopo = mesh->getTopology()->getGatheredCopy();
        meshTopos.push_back(meshTopo);
        
        // check the polyOrder:
        int H1Order = -1;
        
        const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
        if (myCells->size() > 0)
        {
          GlobalIndexType myCellID = *myCells->begin();
          H1Order = mesh->globalDofAssignment()->getH1Order(myCellID)[0];
        }
        if (rank == 0) cout << "H1Order for mesh " << i++ << ": " << H1Order << endl;
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
    }
  }
  Camellia::finalize(false);
  
  return 0;
}
