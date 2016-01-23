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

class UpwindIndicator : public TFunction<double>
{
  vector<FunctionPtr> _beta;
  bool _upwind;
  vector<FieldContainer<double>> _valuesBuffers;
public:
  UpwindIndicator(FunctionPtr beta, bool upwind)
  {
    // upwind = true  means this is the DG '-' operator  (beta * n > 0)
    // upwind = false means this is the DG '+' operator  (beta * n < 0)

    _beta.push_back(beta->x());
    if (beta->y() != Teuchos::null)
    {
      _beta.push_back(beta->y());
      if (beta->z() != Teuchos::null)
        _beta.push_back(beta->z());
    }
    _upwind = upwind;
    _valuesBuffers.resize(_beta.size());
  }
  
  bool isZero(BasisCachePtr basisCache)
  {
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getSpaceDim();
    
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != _beta.size(), std::invalid_argument, "spaceDim and length of beta do not match");
    
    for (int d=0; d<spaceDim; d++)
    {
      _valuesBuffers[d].resize(numCells,numPoints);
      _beta[d]->values(_valuesBuffers[d],basisCache);
    }
  
    const Intrepid::FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());

    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double value = 0;
        for (int d=0; d<spaceDim; d++)
        {
          value += _valuesBuffers[d](cellOrdinal,pointOrdinal) * (*sideNormals)(cellOrdinal,pointOrdinal,d);
        }
        if ((_upwind && (value > 0)) || (!_upwind && (value < 0)))
        {
          return false;
        }
      }
    }
    return true;
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getSpaceDim();
    
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != _beta.size(), std::invalid_argument, "spaceDim and length of beta do not match");
    
    for (int d=0; d<spaceDim; d++)
    {
      _valuesBuffers[d].resize(numCells,numPoints);
      _beta[d]->values(_valuesBuffers[d],basisCache);
    }
    const Intrepid::FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double value = 0;
        for (int d=0; d<spaceDim; d++)
        {
          value += _valuesBuffers[d](cellOrdinal,pointOrdinal) * (*sideNormals)(cellOrdinal,pointOrdinal,d);
        }
        if ((_upwind && (value > 0)) || (!_upwind && (value < 0)))
        {
          values(cellOrdinal,pointOrdinal) = 1;
        }
        else
        {
          values(cellOrdinal,pointOrdinal) = 0;
        }

      }
    }
  }
  
  static FunctionPtr upwindIndicator(FunctionPtr beta, bool upwind)
  {
    return Teuchos::rcp( new UpwindIndicator(beta, upwind) );
  }
};

// method to apply the upwinding terms to soln's stiffness matrix:
void applyUpwinding(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta);
void computeApproximateGradients(SolutionPtr soln, VarPtr u, const vector<GlobalIndexType> &cells, vector<double> &gradient_l2_norm, double weightWithPowerOfH = 0);
void solveAndExport(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta, HDF5Exporter exporter, int refNumber, bool exportVisualization);

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
  solveAndExport(soln, u, v, beta, exporter, refNumber, exportVisualization);
  
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
    
    solveAndExport(soln, u, v, beta, exporter, refNumber, exportVisualization);
    
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
  if (numRefinements > 0)
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

void applyUpwinding(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta)
{
  // accumulate any inter-element DG terms
  // in the present case, there is just one such term: the advective upwinding jump term:
  //   < u^{-}, beta * [v n] >,
  // which is integrated over the interior faces.

  /*
   We do the integration elementwise; on each face of each element, we decide whether the
   element "owns" the face, so that the term is only integrated once, and only on the side
   with finer quadrature, in the case of a locally refined mesh.
   */
  
  MeshPtr mesh = soln->mesh();
  Epetra_FECrsMatrix* stiffnessMatrix = dynamic_cast<Epetra_FECrsMatrix*>(soln->getStiffnessMatrix().get());
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  set<GlobalIndexType> myCellIDs = mesh->cellIDsInPartition();
  int sideDim = meshTopo->getDimension() - 1;
  
  FieldContainer<double> emptyRefPointsVolume(0,meshTopo->getDimension()); // (P,D)
  FieldContainer<double> emptyRefPointsSide(0,sideDim); // (P,D)
  FieldContainer<double> emptyCubWeights(1,0); // (C,P)
  
  map<pair<CellTopologyKey,int>, FieldContainer<double>> cubPointsForSideTopo;
  map<pair<CellTopologyKey,int>, FieldContainer<double>> cubWeightsForSideTopo;
  
  CubatureFactory cubFactory;
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForVolumeTopo; // used for "my" cells
  map<CellTopologyKey,BasisCachePtr> basisCacheForNeighborVolumeTopo; // used for neighbor cells
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnVolumeTopo;
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnNeighborVolumeTopo; // these can have permuted cubature points (i.e. we need to set them every time, so we can't share with basisCacheForSideOnVolumeTopo, which tries to avoid this)
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForReferenceCellTopo;
  
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    ElementTypePtr elemType = mesh->getElementType(cellID);
    DofOrderingPtr trialOrder = elemType->trialOrderPtr;
    BasisPtr uBasis = trialOrder->getBasis(u->ID());
    BasisPtr vBasis = trialOrder->getBasis(v->ID()); // this will be identical to uBasis
    int myPolyOrder = uBasis->getDegree();
    int sideCount = cell->getSideCount();
    
    vector<GlobalIndexType> myGlobalDofs_u_native = mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, u->ID());
    
//    cout << "cell ID " << cellID;
//    print("myGlobalDofs_u", myGlobalDofs_u);
    
    // we'll use this basisCache unless
//    BasisCachePtr cellBasisCacheVolume = BasisCache::basisCacheForCell(mesh, cellID);
    
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
    BasisCachePtr cellBasisCacheVolume;
    if (basisCacheForVolumeTopo.find(cellTopo->getKey()) == basisCacheForVolumeTopo.end())
    {
      basisCacheForVolumeTopo[cellTopo->getKey()] = Teuchos::rcp( new BasisCache(physicalCellNodes, cellTopo,
                                                                                 emptyRefPointsVolume, emptyCubWeights) );
    }
    
    cellBasisCacheVolume = basisCacheForVolumeTopo[cellTopo->getKey()];
    cellBasisCacheVolume->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
    
    cellBasisCacheVolume->setCellIDs({cellID});
    cellBasisCacheVolume->setMesh(mesh);
    
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      // we'll filter this down to match only the side dofs, so we need to do this inside the sideOrdinal loop:
      vector<GlobalIndexTypeToCast> myGlobalDofs_u(myGlobalDofs_u_native.begin(),myGlobalDofs_u_native.end());
      
      pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
      GlobalIndexType neighborCellID = neighborInfo.first;
      unsigned mySideOrdinalInNeighbor = neighborInfo.second;
      if (activeCellIDs.find(neighborCellID) == activeCellIDs.end())
      {
        // no active neigbor on this side: either this is not an interior face (neighborCellID == -1),
        // or the neighbor is refined and therefore inactive.  If the latter, then the neighbor's
        // descendants will collectively "own" this side.
        continue;
      }
      
      // Finally, we need to check whether the neighbor is a "peer" in terms of h-refinements.
      // If so, we use the cellID to break the tie of ownership; lower cellID owns the face.
      CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
      bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
      if (neighborIsPeer && (cellID > neighborCellID))
      {
        // neighbor wins the tie-breaker
        continue;
      }
      
      // if we get here, we own the face and should compute its contribution.
      // determine global dof indices:
      vector<GlobalIndexType> neighborGlobalDofs_u_native = mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(neighborCellID, u->ID());
      vector<GlobalIndexTypeToCast> neighborGlobalDofs_u(neighborGlobalDofs_u_native.begin(),neighborGlobalDofs_u_native.end());
      
      // figure out what the cubature degree should be
      DofOrderingPtr neighborTrialOrder = mesh->getElementType(neighborCellID)->trialOrderPtr;
      BasisPtr uBasisNeighbor = neighborTrialOrder->getBasis(u->ID());
      BasisPtr vBasisNeighbor = neighborTrialOrder->getBasis(v->ID());
      int neighborPolyOrder = uBasisNeighbor->getDegree();
      
      int cubaturePolyOrder = 2 * max(neighborPolyOrder, myPolyOrder);
      
      // set up side basis cache
      CellTopoPtr mySideTopo = cellTopo->getSide(sideOrdinal); // for non-peers, this is the descendant cell topo
      
      pair<int,CellTopologyKey> sideCacheKey{sideOrdinal,cellTopo->getKey()};
      if (basisCacheForSideOnVolumeTopo.find(sideCacheKey) == basisCacheForSideOnVolumeTopo.end())
      {
        basisCacheForSideOnVolumeTopo[sideCacheKey] = Teuchos::rcp( new BasisCache(sideOrdinal, cellBasisCacheVolume,
                                                                                   emptyRefPointsSide, emptyCubWeights, -1));
      }
      BasisCachePtr cellBasisCacheSide = basisCacheForSideOnVolumeTopo[sideCacheKey];
      
      pair<CellTopologyKey,int> cubKey{mySideTopo->getKey(),cubaturePolyOrder};
      if (cubWeightsForSideTopo.find(cubKey) == cubWeightsForSideTopo.end())
      {
        int cubDegree = cubKey.second;
        if (sideDim > 0)
        {
          Teuchos::RCP<Cubature<double> > sideCub;
          if (cubDegree >= 0)
            sideCub = cubFactory.create(mySideTopo, cubDegree);
          
          int numCubPointsSide;
          
          if (sideCub != Teuchos::null)
            numCubPointsSide = sideCub->getNumPoints();
          else
            numCubPointsSide = 0;
          
          FieldContainer<double> cubPoints(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
          FieldContainer<double> cubWeights(numCubPointsSide);
          if (numCubPointsSide > 0)
            sideCub->getCubature(cubPoints, cubWeights);
          cubPointsForSideTopo[cubKey] = cubPoints;
          cubWeightsForSideTopo[cubKey] = cubWeights;
        }
        else
        {
          int numCubPointsSide = 1;
          FieldContainer<double> cubPoints(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
          FieldContainer<double> cubWeights(numCubPointsSide);
          
          cubPoints.initialize(0.0);
          cubWeights.initialize(1.0);
          cubPointsForSideTopo[cubKey] = cubPoints;
          cubWeightsForSideTopo[cubKey] = cubWeights;
        }
      }
      if (cellBasisCacheSide->cubatureDegree() != cubaturePolyOrder)
      {
        cellBasisCacheSide->setRefCellPoints(cubPointsForSideTopo[cubKey], cubWeightsForSideTopo[cubKey], cubaturePolyOrder, false);
      }
      cellBasisCacheSide->setPhysicalCellNodes(cellBasisCacheVolume->getPhysicalCellNodes(), {cellID}, false);
      
      FunctionPtr minus_indicator = UpwindIndicator::upwindIndicator(beta, true);
      LinearTermPtr u_minus = u * minus_indicator;
      
      int numCells = 1;
      int numFields = uBasis->getCardinality();
      int numPoints = cellBasisCacheSide->getRefCellPoints().dimension(0);
      Intrepid::FieldContainer<double> u_values(numCells, numFields, numPoints);
      u_minus->values(u_values, u->ID(), uBasis, cellBasisCacheSide);
      
      // we integrate against the jump term beta * [v n]
      // this is simply the sum of beta * (v n) from cell and neighbor's point of view
      
      FunctionPtr n = Function::normal();
      LinearTermPtr jumpTerm = v * beta * n;
      Intrepid::FieldContainer<double> jump_values(numCells, numFields, numPoints);
      jumpTerm->values(jump_values, v->ID(), vBasis, cellBasisCacheSide);

      // filter to include only those members of uBasis that have support on the side
      // (it might be nice to have LinearTerm::values() support this directly, via a basisDofOrdinal container argument...)
      auto filterSideBasisValues = [] (BasisPtr basis, int sideOrdinal,
                                       FieldContainer<double> &values) -> void
      {
        set<int> mySideBasisDofOrdinals = basis->dofOrdinalsForSide(sideOrdinal);
        int numFilteredFields = mySideBasisDofOrdinals.size();
        int numCells = values.dimension(0);
        int numPoints = values.dimension(2);
        FieldContainer<double> filteredValues(numCells,numFilteredFields,numPoints);
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          int filteredDofOrdinal = 0;
          for (int basisDofOrdinal : mySideBasisDofOrdinals)
          {
            for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
            {
              filteredValues(cellOrdinal,filteredDofOrdinal,pointOrdinal) = values(cellOrdinal,basisDofOrdinal,pointOrdinal);
            }
            filteredDofOrdinal++;
          }
        }
        values = filteredValues;
      };
      
      auto filterGlobalDofOrdinals = [] (BasisPtr basis, int sideOrdinal,
                                       vector<GlobalIndexTypeToCast> &globalDofOrdinals) -> void
      {
        set<int> mySideBasisDofOrdinals = basis->dofOrdinalsForSide(sideOrdinal);
        vector<GlobalIndexTypeToCast> filteredGlobalDofOrdinals;
        for (int basisDofOrdinal : mySideBasisDofOrdinals)
        {
          filteredGlobalDofOrdinals.push_back(globalDofOrdinals[basisDofOrdinal]);
        }
        globalDofOrdinals = filteredGlobalDofOrdinals;
      };
      
      filterSideBasisValues(uBasis,sideOrdinal,u_values);
      filterSideBasisValues(vBasis,sideOrdinal,jump_values);
      filterGlobalDofOrdinals(uBasis,sideOrdinal,myGlobalDofs_u);
      numFields = myGlobalDofs_u.size();
      
      // Now the geometrically challenging bit: we need to line up the physical points in
      // the cellBasisCacheSide with those in a BasisCache for the neighbor cell
      
      CellTopoPtr neighborTopo = neighbor->topology();
      CellTopoPtr sideTopo = neighborTopo->getSide(mySideOrdinalInNeighbor); // for non-peers, this is my ancestor's cell topo
      int nodeCount = sideTopo->getNodeCount();
      
      unsigned permutationFromMeToNeighbor;
      Intrepid::FieldContainer<double> myRefPoints = cellBasisCacheSide->getRefCellPoints();
      
      if (!neighborIsPeer) // then we have some refinements relative to neighbor
      {
        /*******   Map my ref points to my ancestor ******/
        pair<GlobalIndexType,unsigned> ancestorInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
        GlobalIndexType ancestorCellIndex = ancestorInfo.first;
        unsigned ancestorSideOrdinal = ancestorInfo.second;
        
        RefinementBranch refinementBranch = cell->refinementBranchForSide(sideOrdinal, meshTopo);
        RefinementBranch sideRefinementBranch = RefinementPattern::sideRefinementBranch(refinementBranch, ancestorSideOrdinal);
        FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(sideRefinementBranch);
        
        cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
        BasisCachePtr ancestralBasisCache = Teuchos::rcp(new BasisCache(cellNodes,sideTopo,cubaturePolyOrder,false)); // false: don't create side cache too
        
        ancestralBasisCache->setRefCellPoints(myRefPoints, emptyCubWeights, cubaturePolyOrder, true);
        
        // now, the "physical" points in ancestral cache are the ones we want
        myRefPoints = ancestralBasisCache->getPhysicalCubaturePoints();
        myRefPoints.resize(myRefPoints.dimension(1),myRefPoints.dimension(2)); // strip cell dimension
        
        /*******  Determine ancestor's permutation of the side relative to neighbor ******/
        CellPtr ancestor = meshTopo->getCell(ancestorCellIndex);
        vector<IndexType> ancestorSideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
        
        CellTopoPtr sideTopo = ancestor->topology()->getSide(ancestorSideOrdinal);
        nodeCount = sideTopo->getNodeCount();
        
        for (int node=0; node<nodeCount; node++)
        {
          int nodeInCell = cellTopo->getNodeMap(sideDim, sideOrdinal, node);
          ancestorSideNodes.push_back(ancestor->vertices()[nodeInCell]);
          int nodeInNeighborCell = neighborTopo->getNodeMap(sideDim, mySideOrdinalInNeighbor, node);
          neighborSideNodes.push_back(neighbor->vertices()[nodeInNeighborCell]);
        }
        // now, we want to know what permutation of the side topology takes us from my order to neighbor's
        // TODO: make sure I'm not going the wrong direction here; it's easy to get confused.
        permutationFromMeToNeighbor = CamelliaCellTools::permutationMatchingOrder(sideTopo, ancestorSideNodes, neighborSideNodes);
      }
      else
      {
        nodeCount = cellTopo->getSide(sideOrdinal)->getNodeCount();

        vector<IndexType> mySideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
        for (int node=0; node<nodeCount; node++)
        {
          int nodeInCell = cellTopo->getNodeMap(sideDim, sideOrdinal, node);
          mySideNodes.push_back(cell->vertices()[nodeInCell]);
          int nodeInNeighborCell = neighborTopo->getNodeMap(sideDim, mySideOrdinalInNeighbor, node);
          neighborSideNodes.push_back(neighbor->vertices()[nodeInNeighborCell]);
        }
        // now, we want to know what permutation of the side topology takes us from my order to neighbor's
        // TODO: make sure I'm not going the wrong direction here; it's easy to get confused.
        permutationFromMeToNeighbor = CamelliaCellTools::permutationMatchingOrder(sideTopo, mySideNodes, neighborSideNodes);
      }
      
      Intrepid::FieldContainer<double> permutedRefNodes(nodeCount,sideDim);
      CamelliaCellTools::refCellNodesForTopology(permutedRefNodes, sideTopo, permutationFromMeToNeighbor);
      permutedRefNodes.resize(1,nodeCount,sideDim); // add cell dimension to make this a "physical" node container
      if (basisCacheForReferenceCellTopo.find(sideTopo->getKey()) == basisCacheForReferenceCellTopo.end())
      {
        basisCacheForReferenceCellTopo[sideTopo->getKey()] = BasisCache::basisCacheForReferenceCell(sideTopo, -1);
      }
      BasisCachePtr referenceBasisCache = basisCacheForReferenceCellTopo[sideTopo->getKey()];
      referenceBasisCache->setRefCellPoints(myRefPoints,emptyCubWeights,cubaturePolyOrder,false);
      std::vector<GlobalIndexType> cellIDs = {0}; // unused
      referenceBasisCache->setPhysicalCellNodes(permutedRefNodes, cellIDs, false);
      // now, the "physical" points are the ones we should use as ref points for the neighbor
      Intrepid::FieldContainer<double> neighborRefCellPoints = referenceBasisCache->getPhysicalCubaturePoints();
      neighborRefCellPoints.resize(numPoints,sideDim); // strip cell dimension to convert to a "reference" point container
      
      FieldContainer<double> neighborCellNodes = mesh->physicalCellNodesForCell(neighborCellID);
      if (basisCacheForNeighborVolumeTopo.find(neighborTopo->getKey()) == basisCacheForNeighborVolumeTopo.end())
      {
        basisCacheForNeighborVolumeTopo[neighborTopo->getKey()] = Teuchos::rcp( new BasisCache(neighborCellNodes, neighborTopo,
                                                                                       emptyRefPointsVolume, emptyCubWeights) );
      }
      BasisCachePtr neighborVolumeCache = basisCacheForNeighborVolumeTopo[neighborTopo->getKey()];
      neighborVolumeCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
      
      pair<int,CellTopologyKey> neighborSideCacheKey{mySideOrdinalInNeighbor,neighborTopo->getKey()};
      if (basisCacheForSideOnNeighborVolumeTopo.find(neighborSideCacheKey) == basisCacheForSideOnNeighborVolumeTopo.end())
      {
        basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey]
        = Teuchos::rcp( new BasisCache(mySideOrdinalInNeighbor, neighborVolumeCache, emptyRefPointsSide, emptyCubWeights, -1));
      }
      BasisCachePtr neighborSideCache = basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey];
      neighborSideCache->setRefCellPoints(neighborRefCellPoints, emptyCubWeights, cubaturePolyOrder, false);
      neighborSideCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
      {
        // Sanity check that the physical points agree:
        double tol = 1e-15;
        Intrepid::FieldContainer<double> myPhysicalPoints = cellBasisCacheSide->getPhysicalCubaturePoints();
        Intrepid::FieldContainer<double> neighborPhysicalPoints = neighborSideCache->getPhysicalCubaturePoints();
        
        bool pointsMatch = (myPhysicalPoints.size() == neighborPhysicalPoints.size()); // true unless we find a point that doesn't match
        if (pointsMatch)
        {
          for (int i=0; i<myPhysicalPoints.size(); i++)
          {
            double diff = abs(myPhysicalPoints[i]-neighborPhysicalPoints[i]);
            if (diff > tol)
            {
              pointsMatch = false;
              break;
            }
          }
        }
        
        if (!pointsMatch)
        {
          cout << "ERROR: pointsMatch is false.\n";
          cout << "myPhysicalPoints:\n" << myPhysicalPoints;
          cout << "neighborPhysicalPoints:\n" << neighborPhysicalPoints;
        }
      }

      int numNeighborFields = vBasisNeighbor->getCardinality();
      
      Intrepid::FieldContainer<double> neighbor_u_values(numCells, numNeighborFields, numPoints);
      u_minus->values(neighbor_u_values, u->ID(), uBasis, neighborSideCache);
      
      Intrepid::FieldContainer<double> neighbor_jump_values(numCells, numNeighborFields, numPoints);
      jumpTerm->values(neighbor_jump_values, v->ID(), vBasisNeighbor, neighborSideCache);
      
      int neighborSideOrdinal = neighborSideCache->getSideIndex();
      filterSideBasisValues(uBasisNeighbor,neighborSideOrdinal,neighbor_u_values);
      filterSideBasisValues(vBasisNeighbor,neighborSideOrdinal,neighbor_jump_values);
      filterGlobalDofOrdinals(uBasisNeighbor,neighborSideOrdinal,neighborGlobalDofs_u);
      numNeighborFields = neighborGlobalDofs_u.size();
      
      // weight u_values containers using cubature weights defined in cellBasisCacheSide:
      Intrepid::FunctionSpaceTools::multiplyMeasure<double>(u_values, cellBasisCacheSide->getWeightedMeasures(), u_values);
      Intrepid::FunctionSpaceTools::multiplyMeasure<double>(neighbor_u_values, cellBasisCacheSide->getWeightedMeasures(), neighbor_u_values);
      
      // now, we compute four integrals (test, trial) and insert into global stiffness
      // define a lambda function for insertion
      auto insertValues = [stiffnessMatrix] (vector<int> &rowDofOrdinals, vector<int> &colDofOrdinals,
                                             Intrepid::FieldContainer<double> &values) -> void
      {
        // values container is (cell, test, trial)
        int rowCount = rowDofOrdinals.size();
        int colCount = colDofOrdinals.size();
        
        //        stiffnessMatrix->InsertGlobalValues(rowCount,&rowDofOrdinals[0],colCount,&colDofOrdinals[0],&values(0,0,0),
        //                                            Epetra_FECrsMatrix::ROW_MAJOR); // COL_MAJOR is the right thing, actually, but for some reason does not work...
        
        // because I don't trust Epetra in terms of the format, let's insert values one at a time
        for (int i=0; i<rowCount; i++)
        {
          for (int j=0; j<colCount; j++)
          {
            // rows in FieldContainer correspond to trial variables, cols to test
            // in stiffness matrix, it's the opposite.
            if (values(0,i,j) != 0) // skip 0's, which I believe Epetra should ignore anyway
            {
//              cout << "Inserting (" << rowDofOrdinals[i] << "," << colDofOrdinals[j] << ") = ";
//              cout << values(0,i,j) << endl;
              stiffnessMatrix->InsertGlobalValues(1,&rowDofOrdinals[i],1,&colDofOrdinals[j],&values(0,i,j));
            }
          }
        }
      };
      
      auto hasNonzeros = [] (Intrepid::FieldContainer<double> &values, double tol) -> bool
      {
        for (int i=0; i<values.size(); i++)
        {
          if (abs(values[i]) > tol) return true;
        }
        return false;
      };

      if (hasNonzeros(jump_values,0) && hasNonzeros(u_values,0))
      {
        Intrepid::FieldContainer<double> integralValues_me_me(numCells,numFields,numFields);
        Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_me,jump_values,u_values,Intrepid::COMP_BLAS);
        insertValues(myGlobalDofs_u,myGlobalDofs_u,integralValues_me_me);
      }
      
      if (hasNonzeros(neighbor_jump_values,0) && hasNonzeros(u_values,0))
      {
        Intrepid::FieldContainer<double> integralValues_neighbor_me(numCells,numNeighborFields,numFields);
        Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_me,neighbor_jump_values,u_values,Intrepid::COMP_BLAS);
        insertValues(neighborGlobalDofs_u,myGlobalDofs_u,integralValues_neighbor_me);
      }

      if (hasNonzeros(jump_values,0) && hasNonzeros(neighbor_u_values,0))
      {
        Intrepid::FieldContainer<double> integralValues_me_neighbor(numCells,numFields,numNeighborFields);
        Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_neighbor,jump_values,neighbor_u_values,Intrepid::COMP_BLAS);
        insertValues(myGlobalDofs_u,neighborGlobalDofs_u,integralValues_me_neighbor);
      }
      
      if (hasNonzeros(neighbor_jump_values,0) && hasNonzeros(neighbor_u_values,0))
      {
        Intrepid::FieldContainer<double> integralValues_neighbor_neighbor(numCells,numNeighborFields,numNeighborFields);
        Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_neighbor,neighbor_jump_values,neighbor_u_values,Intrepid::COMP_BLAS);
        insertValues(neighborGlobalDofs_u,neighborGlobalDofs_u,integralValues_neighbor_neighbor);
      }
    }
  }
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

void solveAndExport(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta, HDF5Exporter exporter, int refNumber,
                    bool exportVisualization)
{
  SolverPtr solver = Solver::getDirectSolver();
  
  soln->initializeLHSVector();
  soln->initializeStiffnessAndLoad();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time timer(Comm);
  // this is where we want to accumulate any inter-element DG terms
  applyUpwinding(soln, u, v, beta);

  timeThisApplyUpwinding = timer.ElapsedTime();
  
  // it is important to do the above accumulation before BCs are imposed, which they
  // will be in populateStiffnessAndLoad().
  
  timer.ResetStartTime();
  soln->populateStiffnessAndLoad();
  timeThisOtherAssembly = timer.ElapsedTime();
  
  soln->setProblem(solver);
  
  timer.ResetStartTime();
  int solveSuccess = soln->solveWithPrepopulatedStiffnessAndLoad(solver);
  
  timeThisSolve = timer.ElapsedTime();
  
  if (solveSuccess != 0)
    cout << "solve returned with error code " << solveSuccess << endl;
  
  soln->importSolution(); // determines element-local solution coefficients from the global solution vector
  
  if (exportVisualization)
    exporter.exportSolution(soln,refNumber);
  
  // accumulate timings:
  Comm.MaxAll(&timeThisOtherAssembly, &timeThisOtherAssembly, 1);
  Comm.MaxAll(&timeThisApplyUpwinding, &timeThisApplyUpwinding, 1);
  Comm.MaxAll(&timeThisSolve, &timeThisSolve, 1);

  timeOtherAssembly += timeThisOtherAssembly;
  timeSolve += timeThisSolve;
  timeApplyUpwinding += timeThisApplyUpwinding;
}