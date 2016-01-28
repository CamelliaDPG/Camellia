#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "SpatiallyFilteredFunction.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"

#include "Intrepid_FunctionSpaceTools.hpp"

/*
 
 Applying Camellia with *DPG* to the same DG example problem in DGAdvectionExample.
 
 See
 
 https://dealii.org/developer/doxygen/deal.II/step_12.html
 
 for a deal.II tutorial discussion of this example in the DG context.
 
 */

using namespace Camellia;
using namespace std;
using namespace Intrepid;

// beta: 1/|x| (-x2, x1)

class BetaX : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * -y;
    else
      return 0;
  }
};

class BetaY : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * x;
    else
      return 0;
  }
};

class U_Exact_CCW : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    // 1 inside the circle of radius 0.5 centered at the origin
    const static double r = 0.5;
    if (x*x + y*y <= r*r)
      return 1.0;
    else
      return 0;
  }
};

class ElementWiseIntegralFunction : public TFunction<double>
{
  FunctionPtr _fxn;
public:
  ElementWiseIntegralFunction(FunctionPtr f)
  {
    _fxn = f;
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      GlobalIndexType cellID = basisCache->cellIDs()[cellOrdinal];
      BasisCachePtr basisCacheForIntegral = BasisCache::basisCacheForCell(basisCache->mesh(), cellID);
    
      FieldContainer<double> cellIntegrals(1);
      _fxn->integrate(cellIntegrals, basisCacheForIntegral);
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIntegrals(0);
      }
    }
  }
};

// DPG has a built-in error measurement (the energy norm), but we use the below for consistency
// with the DGAdvectionExample:
//void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells,
//                                 vector<double> &gradient_l2_norm, double weightWithPowerOfH);

enum RefinementMode
{
  FIXED_REF_COUNT,
  L2_ERR_TOL
};

// most of these are cumulative "maximum" timings taken from Solution, with the exception of timeRefinements
static double timeLocalStiffness = 0, timeOtherAssembly = 0, timeSolve = 0, timeRefinements = 0;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int polyOrder = 1;
  int horizontalElements = 2, verticalElements = 2;
  int pToAddTest = 2; // using at least spaceDim seems to be important for pure convection
  int numRefinements = -1; // prefer using L^2 tolerance
  double l2tol = 5e-2;
  int spaceDim = 2;
  bool useEnergyNormForRefinements = false;
  bool useGraphNorm = true;
  bool useCondensedSolve = false;
  bool exportVisualization = false; // I think visualization might be segfaulting on finer meshes??
  bool enforceOneIrregularity = true;
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("delta_k", &pToAddTest);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("verticalElements", &verticalElements);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("errTol", &l2tol);
  cmdp.setOption("useEnergyError", "useGradientIndicator", &useEnergyNormForRefinements);
  cmdp.setOption("useGraphNorm", "useNaiveNorm", &useGraphNorm);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("exportVisualization", "dontExportVisualization", &exportVisualization);
  cmdp.setOption("enforceOneIrregularity", "dontEnforceOneIrregularity", &enforceOneIrregularity);
  
  string convectiveDirectionChoice = "CCW"; // counter-clockwise, the default.  Other options: left, right, up, down
  cmdp.setOption("convectiveDirection", &convectiveDirectionChoice);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  RefinementMode refMode = (numRefinements==-1) ? L2_ERR_TOL : FIXED_REF_COUNT;
  
  FunctionPtr beta_x, beta_y;
  
  SpatialFilterPtr unitInflow, zeroInflow;
  
  FunctionPtr u_exact;
  
  if (convectiveDirectionChoice == "CCW")
  {
    beta_x = Teuchos::rcp( new BetaX );
    beta_y = Teuchos::rcp( new BetaY );
    
    // set g = 1 on [0,0.5] x {0}
    unitInflow = SpatialFilter::matchingY(0) & (! SpatialFilter::greaterThanX(0.5));
    
    // set g = 0 on {1} x [0,1], (0.5,1.0] x {0}
    zeroInflow = SpatialFilter::matchingX(1) | (SpatialFilter::matchingY(0) & SpatialFilter::greaterThanX(0.5));
    
    u_exact = Teuchos::rcp(new U_Exact_CCW);
  }
  else if (convectiveDirectionChoice == "left")
  {
    beta_x = Function::constant(-1);
    beta_y = Function::zero();
    unitInflow = SpatialFilter::matchingX(1.0) & SpatialFilter::lessThanY(0.5);
    zeroInflow = SpatialFilter::matchingX(1.0) & !SpatialFilter::lessThanY(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanY(0.5)) );
  }
  else if (convectiveDirectionChoice == "right")
  {
    beta_x = Function::constant(1.0);
    beta_y = Function::constant(0.0);
    
    unitInflow = SpatialFilter::matchingX(0.0) & SpatialFilter::lessThanY(0.5); // | SpatialFilter::matchingY(0.0);
    zeroInflow = SpatialFilter::matchingX(0.0) & !SpatialFilter::lessThanY(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanY(0.5)) );
  }
  else if (convectiveDirectionChoice == "up")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(1);
    
    unitInflow = SpatialFilter::matchingY(0.0);
    zeroInflow = !SpatialFilter::allSpace();
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::allSpace()) );
  }
  else if (convectiveDirectionChoice == "down")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(-1);
    
    unitInflow = SpatialFilter::matchingY(1.0) & SpatialFilter::lessThanX(0.5);
    zeroInflow = SpatialFilter::matchingY(1.0) & !SpatialFilter::lessThanX(0.5);
    
    u_exact = Teuchos::rcp( new SpatiallyFilteredFunction<double>(Function::constant(1.0), SpatialFilter::lessThanX(0.5)) );
  }
  else
  {
    if (rank==0) cout << "convective direction " << convectiveDirectionChoice << " is not a supported option.\n";
  }
  
  FunctionPtr beta = Function::vectorize(beta_x, beta_y);
    
  VarFactoryPtr vf = VarFactory::varFactory();
  
  VarPtr u = vf->fieldVar("u");
  FunctionPtr n = Function::normal();
  FunctionPtr parity = Function::sideParity();
  VarPtr u_n = vf->fluxVar("u_n", u * beta * n * parity, L2);
  VarPtr v = vf->testVar("v", HGRAD);
  
  BFPtr bf = BF::bf(vf);
  bf->addTerm(-u, beta * v->grad());
  bf->addTerm(u_n, v);
  
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_n, unitInflow, Function::constant(1.0) * beta * n * parity);
  bc->addDirichlet(u_n, zeroInflow, Function::zero() * beta * n * parity);
  
  /******* Define the mesh ********/
  // solve on [0,1]^2 with 8x8 initial elements
  double width = 1.0, height = 1.0;
  bool divideIntoTriangles = false;
  double x0 = 0.0, y0 = 0.0;
  int H1Order = polyOrder + 1; // polyOrder refers to the order of the fields, which here are L^2
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAddTest,
                                              width, height,
                                              horizontalElements, verticalElements,
                                              divideIntoTriangles, x0, y0);
  RHSPtr rhs = RHS::rhs(); // zero forcing
  
  IPPtr ip;
  if (useGraphNorm)
  {
    ip = bf->graphNorm();
  }
  else
  {
    ip = bf->naiveNorm(spaceDim);
  }
  
  Epetra_Time totalTimer(Comm);
  
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  
  SolverPtr solver = Solver::getDirectSolver();
  soln->setUseCondensedSolve(useCondensedSolve);
  
  int solveSuccess = soln->solve(solver);
  if (solveSuccess != 0)
    if (rank ==0) cout << "solve returned with error code " << solveSuccess << endl;
  
  timeSolve += soln->maxTimeSolve();
  timeLocalStiffness += soln->maxTimeLocalStiffness();
  timeOtherAssembly += soln->maxTimeGlobalAssembly();
  
  ostringstream report;
  report << "elems\tdofs\ttrace_dofs\terr\ttimeRefinement\ttimeAssembly\ttimeSolve\ttimeTotal\n";
  
  ostringstream name;
  name << "DPGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_" << numRefinements << "refs";
  if (useEnergyNormForRefinements)
    name << "_energyErrorIndicator";
  else
    name << "_gradientErrorIndicator";
  if (!enforceOneIrregularity)
    name << "_irregular";
  if (useGraphNorm)
    name << "_graphNorm";
  else
    name << "_naiveNorm";
  HDF5Exporter exporter(mesh, name.str(), ".");
  
  int refNumber = 0;
  if (exportVisualization)
    exporter.exportSolution(soln,refNumber);
  
  FunctionPtr u_soln = Function::solution(u, soln);
  FunctionPtr u_err = u_soln - u_exact;
  
  int numElements = mesh->getActiveCellIDs().size();
  GlobalIndexType dofCount = mesh->numGlobalDofs();
  GlobalIndexType traceCount = mesh->numFluxDofs();

  int DG_cubatureEnrichment = 10;
  // polynomial orders for DPG are relative to H^1 order, leading to +1 on test and trial here;
  // then we also enrich the test space:
  int cubatureEnrichment = DG_cubatureEnrichment - 2 - pToAddTest;
  
  double err = u_err->l2norm(mesh, cubatureEnrichment);
  
  if (rank == 0)
  {
    cout << "Initial mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
    cout << "L^2 error = " << err << ".\n";
  }
  
  double timeThisTotal = totalTimer.ElapsedTime();
  Comm.MaxAll(&timeThisTotal, &timeThisTotal, 1);
  
  double timeThisSolve = soln->maxTimeSolve();
  double timeThisAssembly = soln->maxTimeLocalStiffness() + soln->maxTimeGlobalAssembly();
  double timeThisRefinement = 0;
  
  report << numElements << "\t" << dofCount << "\t" << traceCount << "\t" << err;
  report << "\t" << timeThisRefinement << "\t" << timeThisAssembly << "\t" <<  timeThisSolve;
  report << "\t" << timeThisTotal << "\n";
  
  auto keepRefining = [refMode, &refNumber, numRefinements, &err, l2tol] () -> bool
  {
    if (refMode == FIXED_REF_COUNT)
      return (refNumber < numRefinements);
    else
      return (err > l2tol);
  };
  
  Epetra_Time refinementTimer(Comm);
  
  while (keepRefining())
  {
    refNumber++;
    
    refinementTimer.ResetStartTime();
    totalTimer.ResetStartTime();
    
    double hPower = 1.0 + spaceDim / 2.0;
    
    set<GlobalIndexType> cellIDSet = mesh->getActiveCellIDs();
    set<GlobalIndexType> myCellIDSet = mesh->cellIDsInPartition();
    vector<GlobalIndexType> myCellIDs(myCellIDSet.begin(),myCellIDSet.end());
    
    vector<GlobalIndexType> cellIDs(cellIDSet.begin(),cellIDSet.end());
  
    vector<double> globalErrorIndicatorValues(cellIDs.size(),0);
    
    if (useEnergyNormForRefinements)
    {
      const map<GlobalIndexType,double>* energyError = &soln->globalEnergyError();
      for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
      {
        GlobalIndexType cellID = cellIDs[cellOrdinal];
        globalErrorIndicatorValues[cellOrdinal] = energyError->find(cellID)->second;
//        cout << "energy error for cell " << cellID << ": " << errorIndicatorValues[cellOrdinal] << endl;
      }
    }
    else
    {
      vector<double> myErrorIndicatorValues;
      RefinementStrategy::computeApproximateGradients(soln, u, myCellIDs, myErrorIndicatorValues, hPower);
      // now fill in the global error indicator values
      int myCellOrdinal = 0;
      for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
      {
        GlobalIndexType cellID = cellIDs[cellOrdinal];
        if (myCellIDSet.find(cellID) != myCellIDSet.end())
        {
          globalErrorIndicatorValues[cellOrdinal] = myErrorIndicatorValues[myCellOrdinal];
          myCellOrdinal++;
        }
      }
      MPIWrapper::entryWiseSum<double>(Comm,globalErrorIndicatorValues);
    }
    
    // refine the top 30% of cells
    int numCellsToRefine = 0.3 * cellIDs.size();
    int numCellsNotToRefine = (cellIDs.size()-numCellsToRefine);
    vector<double> globalErrorIndicatorValuesCopy = globalErrorIndicatorValues;
    std::nth_element(globalErrorIndicatorValues.begin(),globalErrorIndicatorValues.begin() + numCellsNotToRefine,
                     globalErrorIndicatorValues.end());
    
    double threshold = globalErrorIndicatorValues[numCellsNotToRefine-1];
    vector<GlobalIndexType> cellsToRefine;
    for (int i=0; i<cellIDs.size(); i++)
    {
      if (globalErrorIndicatorValuesCopy[i] > threshold)
      {
        cellsToRefine.push_back(cellIDs[i]);
      }
    }
    mesh->hRefine(cellsToRefine, false);
    if (enforceOneIrregularity)
      mesh->enforceOneIrregularity(false);
    mesh->repartitionAndRebuild();
    
    double timeThisRefinement = refinementTimer.ElapsedTime();
    Comm.MaxAll(&timeThisRefinement, &timeThisRefinement, 1);
    timeRefinements += timeThisRefinement;
    
    int numElements = mesh->getActiveCellIDs().size();
    GlobalIndexType dofCount = mesh->numGlobalDofs();
    GlobalIndexType traceCount = mesh->numFluxDofs();
    
    int solveSuccess = soln->solve(solver);
    if (solveSuccess != 0)
      if (rank==0) cout << "solve returned with error code " << solveSuccess << endl;

    timeSolve += soln->maxTimeSolve();
    timeLocalStiffness += soln->maxTimeLocalStiffness();
    timeOtherAssembly += soln->maxTimeGlobalAssembly();
    
    err = u_err->l2norm(mesh, cubatureEnrichment);
    
    double timeThisSolve = soln->maxTimeSolve();
    double timeThisAssembly = soln->maxTimeLocalStiffness() + soln->maxTimeGlobalAssembly();
    
    if (rank==0)
    {
      cout << "Ref. " << refNumber << " mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
      cout << "L^2 error = " << err << ".\n";
    }
    
    timeThisTotal = totalTimer.ElapsedTime();
    Comm.MaxAll(&timeThisTotal, &timeThisTotal, 1);
    
    report << numElements << "\t" << dofCount << "\t" << traceCount << "\t" << err;
    report << "\t" << timeThisRefinement << "\t" << timeThisAssembly << "\t" <<  timeThisSolve;
    report << "\t" << timeThisTotal << "\n";
    
    if (exportVisualization)
      exporter.exportSolution(soln,refNumber);
  }
  
  ostringstream reportTitle;
  reportTitle << "DPGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_";
  if (numRefinements > 0)
    reportTitle << numRefinements << "refs";
  else
    reportTitle << "tol" << l2tol;
  if (useEnergyNormForRefinements)
    reportTitle << "_energyErrorIndicator";
  else
    reportTitle << "_gradientErrorIndicator";
  if (!enforceOneIrregularity)
    reportTitle << "_irregular";
  if (useGraphNorm)
    reportTitle << "_graphNorm";
  else
    reportTitle << "_naiveNorm";
  reportTitle << "_" << numProcs << "ranks.dat";
  
  if (rank == 0)
  {
    ofstream fout(reportTitle.str().c_str());
    fout << report.str();
    fout.close();
    cout << "Wrote results to " << reportTitle.str() << ".\n";
    
    cout << "Timings:\n";
    cout << "compute local stiffness: " << timeLocalStiffness << " secs.\n";
    cout << "other assembly:          " << timeOtherAssembly << " secs.\n";
    cout << "solve:                   " << timeSolve << " secs.\n";
    cout << "refine:                  " << timeRefinements << " secs.\n";
  }
  
  return 0;
}