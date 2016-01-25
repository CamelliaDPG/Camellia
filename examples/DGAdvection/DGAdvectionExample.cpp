#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CubatureFactory.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "SpatiallyFilteredFunction.h"
#include "UpwindIndicatorFunction.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#include "Epetra_Time.h"

#include "Intrepid_FunctionSpaceTools.hpp"

/*
 
 An experimental effort to use Camellia for a DG problem.
 
 See
 
 https://dealii.org/developer/doxygen/deal.II/step_12.html
 
 for a deal.II tutorial discussion of this example.
 
 */

using namespace Camellia;
using namespace Intrepid;
using namespace std;

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

// method to apply the upwinding terms to soln's stiffness matrix:
//void applyUpwinding(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta);
void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells, vector<double> &gradient_l2_norm, double weightWithPowerOfH = 0);
void solveAndExport(SolutionPtr soln, HDF5Exporter exporter, int refNumber, bool exportVisualization);

// (implementations below main)

enum RefinementMode
{
  FIXED_REF_COUNT,
  L2_ERR_TOL
};

// some static cumulative timing variables:
static double timeApplyUpwinding = 0, timeOtherAssembly = 0, timeSolve = 0, timeRefinements = 0;

// static timing of the last run
static double timeThisApplyUpwinding = 0, timeThisOtherAssembly = 0, timeThisSolve = 0, timeThisRefinement = 0;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int spaceDim = 2;
  
  int polyOrder = 1;
  int horizontalElements = 2, verticalElements = 2;
  int numRefinements = -1; // prefer using L^2 tolerance
  double l2tol = 1e-2;
  bool exportVisualization = false; // I think visualization might be segfaulting on finer meshes??
  bool enforceOneIrregularity = true;
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("verticalElements", &verticalElements);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("errTol", &l2tol);
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
  
  VarPtr u = vf->fieldVar("u",HGRAD_DISC);
  VarPtr v = vf->testVar("v", HGRAD_DISC);
  
  FunctionPtr n = Function::normal();
  
  BFPtr bf = BF::bf(vf);
  bf->addTerm(-u, beta * v->grad());
  
  // term arising from integration by parts on the whole domain; we want to restrict to the domain boundary
  FunctionPtr boundaryIndicator = Function::meshBoundaryCharacteristic();
  bf->addTerm(u, v * beta * n * boundaryIndicator);
  
  // try the new DG term thing:
  FunctionPtr minus = UpwindIndicatorFunction::minus(beta);
  bf->addJumpTerm(u * minus, v * beta * n);
  
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u, unitInflow, Function::constant(1.0));
  bc->addDirichlet(u, zeroInflow, Function::zero());
  
  /******* Define the mesh ********/
  // solve on [0,1]^2 with 8x8 initial elements
  double width = 1.0, height = 1.0;
  bool divideIntoTriangles = false;
  double x0 = 0.0, y0 = 0.0;
  int pToAddTest = 0; // Bubnov-Galerkin!
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, polyOrder, pToAddTest,
                                              width, height, horizontalElements, verticalElements,
                                              divideIntoTriangles, x0, y0);
  
  RHSPtr rhs = RHS::rhs(); // zero forcing
  Epetra_Time totalTimer(Comm);
  SolutionPtr soln = Solution::solution(mesh, bc, rhs);

  ostringstream report;
  report << "elems\tdofs\ttrace_dofs\terr\ttimeRefinement\ttimeAssembly\ttimeSolve\ttimeTotal\n";
  
  ostringstream name;
  name << "DGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_" << numRefinements << "refs";
  HDF5Exporter exporter(mesh, name.str(), ".");
  
  name << "_mesh";
  HDF5Exporter mesh_exporter(mesh, name.str(), ".");
  FunctionPtr meshFunction = Function::meshSkeletonCharacteristic();
  mesh_exporter.exportFunction(meshFunction,"mesh",0);

  FunctionPtr u_soln = Function::solution(u, soln);
  FunctionPtr u_err = u_soln - u_exact;
  
  int numElements = mesh->getActiveCellIDs().size();
  GlobalIndexType dofCount = mesh->numGlobalDofs();
  
  int cubatureEnrichment = 10;
  int refNumber = 0;
  solveAndExport(soln, exporter, refNumber, exportVisualization);
  
  double err = u_err->l2norm(mesh, cubatureEnrichment);

  if (rank==0)
  {
    cout << "Initial mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
    cout << "L^2 error = " << err << ".\n";
  }
  
  double timeThisTotal = totalTimer.ElapsedTime();
  Comm.MaxAll(&timeThisTotal, &timeThisTotal, 1);
  
  double timeThisAssembly = timeThisOtherAssembly + timeThisApplyUpwinding;
  Comm.MaxAll(&timeThisRefinement, &timeThisRefinement, 1);
  timeRefinements += timeThisRefinement;
  
  int traceCount = 0;
  
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
    vector<GlobalIndexType> cellIDs(cellIDSet.begin(),cellIDSet.end());
    set<GlobalIndexType> myCellIDSet = mesh->cellIDsInPartition();
    vector<GlobalIndexType> myCellIDs(myCellIDSet.begin(),myCellIDSet.end());
    
    vector<double> globalErrorIndicatorValues(cellIDs.size(),0);
    
    vector<double> myErrorIndicatorValues;
    computeApproximateGradients(soln, u, myCellIDs, myErrorIndicatorValues, hPower);
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
    
    // refine the top 30% of cells
    int numCellsToRefine = 0.3 * cellIDs.size();
    int numCellsNotToRefine = (cellIDs.size()-numCellsToRefine);
    vector<double> globalErrorIndicatorValuesCopy = globalErrorIndicatorValues;
    std::nth_element(globalErrorIndicatorValues.begin(),globalErrorIndicatorValues.begin() + numCellsNotToRefine, globalErrorIndicatorValues.end());
    
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
    
    timeThisRefinement = refinementTimer.ElapsedTime();
    
    int numElements = mesh->getActiveCellIDs().size();
    GlobalIndexType dofCount = mesh->numGlobalDofs();
    
    if (exportVisualization)
      mesh_exporter.exportFunction(meshFunction,"mesh",refNumber);
    
    solveAndExport(soln, exporter, refNumber, exportVisualization);
    
    err = u_err->l2norm(mesh, cubatureEnrichment);
    
    timeThisTotal = totalTimer.ElapsedTime();
    Comm.MaxAll(&timeThisTotal, &timeThisTotal, 1);
    
    Comm.MaxAll(&timeThisRefinement, &timeThisRefinement, 1);
    timeRefinements += timeThisRefinement;
    timeThisAssembly = timeThisOtherAssembly + timeThisApplyUpwinding;
    
    report << numElements << "\t" << dofCount << "\t" << traceCount << "\t" << err;
    report << "\t" << timeThisRefinement << "\t" << timeThisAssembly << "\t" <<  timeThisSolve;
    report << "\t" << timeThisTotal << "\n";
    
    if (rank==0)
    {
      cout << "Ref. " << refNumber << " mesh has " << numElements << " active elements and " << dofCount << " degrees of freedom; ";
      cout << "L^2 error = " << err << ".\n";
    }
  }
  
  ostringstream reportTitle;
  reportTitle << "DGAdvection_" << convectiveDirectionChoice << "_k" << polyOrder << "_";
  if (refMode == FIXED_REF_COUNT)
    reportTitle << numRefinements << "refs";
  else
    reportTitle << "tol" << l2tol;
  if (!enforceOneIrregularity)
    reportTitle << "_irregular";
  reportTitle << "_" << numProcs << "ranks.dat";
  
  if (rank==0)
  {
    ofstream fout(reportTitle.str().c_str());
    fout << report.str();
    fout.close();
    cout << "Wrote results to " << reportTitle.str() << ".\n";
    
    cout << "Timings:\n";
    cout << "apply upwinding: " << timeApplyUpwinding << " secs.\n";
    cout << "other assembly : " << timeOtherAssembly << " secs.\n";
    cout << "solve:           " << timeSolve << " secs.\n";
    cout << "refine:          " << timeRefinements << " secs.\n";
  }

  return 0;
}

void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells,
                                 vector<double> &gradient_l2_norm, double weightWithPowerOfH)
{
  // imitates https://dealii.org/developer/doxygen/deal.II/namespaceDerivativeApproximation.html

  // we require that u be a scalar field variable
  TEUCHOS_TEST_FOR_EXCEPTION(u->rank() != 0, std::invalid_argument, "u must be a scalar variable");
  TEUCHOS_TEST_FOR_EXCEPTION(u->varType() != FIELD, std::invalid_argument, "u must be a field variable");
  
  int cellCount = cells.size();
  gradient_l2_norm.resize(cells.size());
  
  int onePoint = 1;
  MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology();
  int spaceDim = meshTopo->getDimension();

  set<GlobalIndexType> cellsAndNeighborsSet;
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    cellsAndNeighborsSet.insert(cells[cellOrdinal]);
    CellPtr cell = meshTopo->getCell(cells[cellOrdinal]);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    cellsAndNeighborsSet.insert(neighborIDs.begin(),neighborIDs.end());
  }
  vector<GlobalIndexType> cellsAndNeighbors(cellsAndNeighborsSet.begin(),cellsAndNeighborsSet.end());
  
  // get any off-rank solution data we may need:
  soln->importSolutionForOffRankCells(cellsAndNeighborsSet);
  
  int cellsAndNeighborsCount = cellsAndNeighbors.size();
  
  FieldContainer<double> cellValues(cellsAndNeighborsCount,onePoint); // values at cell centers
  FieldContainer<double> cellCenters(cellsAndNeighborsCount,spaceDim);
  FieldContainer<double> cellDiameter(cellsAndNeighborsCount,onePoint); // h-values
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForTopology;
  
  FunctionPtr hFunction = Function::h();
  FunctionPtr solnFunction = Function::solution(u, soln);
  Teuchos::Array<int> cellValueDim;
  cellValueDim.push_back(1);
  cellValueDim.push_back(1);
  
  map<GlobalIndexType,int> cellIDToOrdinal; // lookup table for value access
  
  // setup: compute cell centers, and solution values at those points
  for (int cellOrdinal=0; cellOrdinal<cellsAndNeighborsCount; cellOrdinal++)
  {
    GlobalIndexType cellID = cellsAndNeighbors[cellOrdinal];
    cellIDToOrdinal[cellID] = cellOrdinal;
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    if (basisCacheForTopology.find(cellTopo->getKey()) == basisCacheForTopology.end())
    {
      FieldContainer<double> centroid(onePoint,spaceDim);
      int nodeCount = cellTopo->getNodeCount();
      FieldContainer<double> cellNodes(nodeCount,spaceDim);
      CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
      for (int node=0; node<nodeCount; node++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          centroid(0,d) += cellNodes(node,d);
        }
      }
      for (int d=0; d<spaceDim; d++)
      {
        centroid(0,d) /= nodeCount;
      }
      basisCacheForTopology[cellTopo->getKey()] = BasisCache::basisCacheForReferenceCell(cellTopo, 0); // 0 cubature degree
      basisCacheForTopology[cellTopo->getKey()]->setRefCellPoints(centroid);
      basisCacheForTopology[cellTopo->getKey()]->setMesh(soln->mesh());
    }
    BasisCachePtr basisCache = basisCacheForTopology[cellTopo->getKey()];
    basisCache->setPhysicalCellNodes(soln->mesh()->physicalCellNodesForCell(cellID), {cellID}, false);
    
    FieldContainer<double> cellValue(cellValueDim,&cellValues(cellOrdinal,0));
    solnFunction->values(cellValue, basisCache);
    for (int d=0; d<spaceDim; d++)
    {
      cellCenters(cellOrdinal,d) = basisCache->getPhysicalCubaturePoints()(0,0,d);
    }
    if (weightWithPowerOfH != 0)
    {
      cellDiameter(cellOrdinal,0) = soln->mesh()->getCellMeasure(cellID);
    }
  }
  
  // now compute the gradients requested
  FieldContainer<double> Y(spaceDim,spaceDim); // the matrix we'll invert to compute the gradient
  FieldContainer<double> b(spaceDim); // RHS for matrix problem
  FieldContainer<double> grad(spaceDim); // LHS for matrix problem
  vector<double> distanceVector(spaceDim);
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    GlobalIndexType cellID = cells[cellOrdinal];
    CellPtr cell = meshTopo->getCell(cellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[cellID];
    double myValue = cellValues(myOrdinalInCellAndNeighbors,0);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      int neighborOrdinalInCellAndNeighbors = cellIDToOrdinal[neighborID];
      double neighborValue = cellValues(neighborOrdinalInCellAndNeighbors,0);
      
      double dist_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        distanceVector[d] = cellCenters(neighborOrdinalInCellAndNeighbors,d) - cellCenters(myOrdinalInCellAndNeighbors,d);
        dist_squared += distanceVector[d] * distanceVector[d];
      }
      
      for (int d1=0; d1<spaceDim; d1++)
      {
        b(d1) += distanceVector[d1] * (neighborValue - myValue) / dist_squared;
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += distanceVector[d1] * distanceVector[d2] / dist_squared;
        }
      }
    }
    SerialDenseWrapper::solveSystem(grad, Y, b);
    double l2_value_squared = 0;
    for (int d=0; d<spaceDim; d++)
    {
      l2_value_squared += grad(d) * grad(d);
    }
    if (weightWithPowerOfH == 0)
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared);
    }
    else
    {
      gradient_l2_norm[cellOrdinal] = sqrt(l2_value_squared) * pow(cellDiameter(myOrdinalInCellAndNeighbors,0), weightWithPowerOfH);
    }
  }
}

void solveAndExport(SolutionPtr soln, HDF5Exporter exporter, int refNumber, bool exportVisualization)
{
  SolverPtr solver = Solver::getDirectSolver();

  soln->solve(solver);
  
  timeSolve += soln->maxTimeSolve();
  timeApplyUpwinding += soln->maxTimeApplyJumpTerms();
  timeOtherAssembly += soln->maxTimeGlobalAssembly() + soln->maxTimeLocalStiffness();

  if (exportVisualization)
    exporter.exportSolution(soln,refNumber);
}