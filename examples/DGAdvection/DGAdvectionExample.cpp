#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"

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

class UpwindIndicator : public TFunction<double>
{
  FunctionPtr _beta;
  bool _upwind;
public:
  UpwindIndicator(FunctionPtr beta, bool upwind)
  {
    // upwind = true  means this is the DG '-' operator  (beta * n > 0)
    // upwind = false means this is the DG '+' operator  (beta * n < 0)
    _beta = beta;
    _upwind = upwind;
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    FunctionPtr n = Function::normal();
    (_beta * n)->values(values, basisCache);
    for (int i=0; i<values.size(); i++)
    {
      if (_upwind)
      {
        values[i] = (values[i] > 0) ? 1.0 : 0.0;
      }
      else
      {
        values[i] = (values[i] < 0) ? 1.0 : 0.0;
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
void solveAndExport(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta, HDF5Exporter exporter, int refNumber);

// (implementations below main)

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int polyOrder = 1;
  int horizontalElements = 2, verticalElements = 2;
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("verticalElements", &verticalElements);
  
  string convectiveDirectionChoice = "CCW"; // counter-clockwise, the default.  Other options: left, right, up, down
  cmdp.setOption("convectiveDirection", &convectiveDirectionChoice);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  FunctionPtr beta_x, beta_y;
  
  SpatialFilterPtr unitInflow, zeroInflow;
  
  if (convectiveDirectionChoice == "CCW")
  {
    beta_x = Teuchos::rcp( new BetaX );
    beta_y = Teuchos::rcp( new BetaY );
    
    // set g = 1 on [0,0.5] x {0}
    unitInflow = SpatialFilter::matchingY(0) & (! SpatialFilter::greaterThanX(0.5));
    
    // set g = 0 on {1} x [0,1], (0.5,1.0] x {0}
    zeroInflow = SpatialFilter::matchingX(1) | (SpatialFilter::matchingY(0) & SpatialFilter::greaterThanX(0.5));
  }
  else if (convectiveDirectionChoice == "left")
  {
    beta_x = Function::constant(-1);
    beta_y = Function::zero();
    unitInflow = SpatialFilter::matchingX(1.0) & SpatialFilter::lessThanY(0.5);
    zeroInflow = SpatialFilter::matchingX(1.0) & !SpatialFilter::lessThanY(0.5);
  }
  else if (convectiveDirectionChoice == "right")
  {
    beta_x = Function::constant(1.0);
    beta_y = Function::constant(0.0);
    
    unitInflow = SpatialFilter::matchingX(0.0) & SpatialFilter::lessThanY(0.5); // | SpatialFilter::matchingY(0.0);
    zeroInflow = SpatialFilter::matchingX(0.0) & !SpatialFilter::lessThanY(0.5);
  }
  else if (convectiveDirectionChoice == "up")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(1);
    
    unitInflow = SpatialFilter::matchingY(0.0);
    zeroInflow = !SpatialFilter::allSpace();
  }
  else if (convectiveDirectionChoice == "down")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(-1);
    
    unitInflow = SpatialFilter::matchingY(1.0) & SpatialFilter::lessThanX(0.5);
    zeroInflow = SpatialFilter::matchingY(1.0) & !SpatialFilter::lessThanX(0.5);
  }
  else
  {
    cout << "convective direction " << convectiveDirectionChoice << " is not a supported option.\n";
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
  SolutionPtr soln = Solution::solution(mesh, bc, rhs);

  ostringstream name;
  name << "DGAdvectionExample_" << convectiveDirectionChoice;
  HDF5Exporter exporter(mesh, name.str(), ".");

  solveAndExport(soln, u, v, beta, exporter, 0);

  // test code to see if h-refinements work:
  mesh->hRefine(set<GlobalIndexType>{0});
  solveAndExport(soln, u, v, beta, exporter, 1);
  
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
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    ElementTypePtr elemType = mesh->getElementType(cellID);
    DofOrderingPtr trialOrder = elemType->trialOrderPtr;
    BasisPtr uBasis = trialOrder->getBasis(u->ID());
    BasisPtr vBasis = trialOrder->getBasis(v->ID()); // this will be identical to uBasis
    int myPolyOrder = uBasis->getDegree();
    int sideCount = cell->getSideCount();
    
    vector<GlobalIndexType> myGlobalDofs_u_native = mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, u->ID());
    vector<GlobalIndexTypeToCast> myGlobalDofs_u(myGlobalDofs_u_native.begin(),myGlobalDofs_u_native.end());
    
//    cout << "cell ID " << cellID;
//    print("myGlobalDofs_u", myGlobalDofs_u);
    
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
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
      
//      print("neighborGlobalDofs_u", neighborGlobalDofs_u);
      
      // figure out what the cubature degree should be
      DofOrderingPtr neighborTrialOrder = mesh->getElementType(neighborCellID)->trialOrderPtr;
      BasisPtr uBasisNeighbor = neighborTrialOrder->getBasis(u->ID());
      BasisPtr vBasisNeighbor = neighborTrialOrder->getBasis(v->ID());
      int neighborPolyOrder = uBasisNeighbor->getDegree();
      
      int cubaturePolyOrder = 2 * max(neighborPolyOrder, myPolyOrder);
      int myCubatureEnrichment = cubaturePolyOrder - 2 * myPolyOrder;
      
      BasisCachePtr cellBasisCacheVolume = BasisCache::basisCacheForCell(mesh, cellID, false, myCubatureEnrichment);
      BasisCachePtr cellBasisCacheSide = cellBasisCacheVolume->getSideBasisCache(sideOrdinal);
      
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
      
      // Now the geometrically challenging bit: we need to line up the physical points in
      // the cellBasisCacheSide with those in a BasisCache for the neighbor cell
      
      // For the moment, we make the assumption neighbor *is* a peer in terms of the h-refinements
      // We can revisit this later; not too great a change is necessary, really; see particularly
      // RefinementPattern::mapRefCellPointsToAncestor().  The strategy would be to use this on "my"
      // side, and then use the permutation stuff below to transform into neighbor's reference space.
      // (MAY EVEN BE ABLE TO USE BasisCache::basisCacheForRefinedReferenceCell() to make this simpler still.)
      
      // On the peer assumption, the only transformation necessary is a possible permutation of the
      // side vertices.
      
      CellTopoPtr cellTopo = cell->topology();
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
        BasisCachePtr ancestralBasisCache = Teuchos::rcp(new BasisCache(cellNodes,sideTopo,cubaturePolyOrder*2,false)); // false: don't create side cache too
        
        ancestralBasisCache->setRefCellPoints(myRefPoints);
        
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
      BasisCachePtr referenceBasisCache = BasisCache::basisCacheForReferenceCell(sideTopo, 1);
      referenceBasisCache->setRefCellPoints(myRefPoints);
      std::vector<GlobalIndexType> cellIDs = {0}; // unused
      referenceBasisCache->setPhysicalCellNodes(permutedRefNodes, cellIDs, false);
      // now, the "physical" points are the ones we should use as ref points for the neighbor
      Intrepid::FieldContainer<double> neighborRefCellPoints = referenceBasisCache->getPhysicalCubaturePoints();
      neighborRefCellPoints.resize(numPoints,sideDim); // strip cell dimension to convert to a "reference" point container
      
      BasisCachePtr neighborVolumeCache = BasisCache::basisCacheForCell(mesh, neighborCellID);
      BasisCachePtr neighborSideCache = neighborVolumeCache->getSideBasisCache(mySideOrdinalInNeighbor);
      
      neighborSideCache->setRefCellPoints(neighborRefCellPoints);
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
          cout << "ERROR: pointsMatch is false.\n";
      }

      int numNeighborFields = vBasisNeighbor->getCardinality();
      
      Intrepid::FieldContainer<double> neighbor_u_values(numCells, numNeighborFields, numPoints);
      u_minus->values(neighbor_u_values, u->ID(), uBasis, neighborSideCache);
      
      Intrepid::FieldContainer<double> neighbor_jump_values(numCells, numNeighborFields, numPoints);
      jumpTerm->values(neighbor_jump_values, v->ID(), vBasisNeighbor, neighborSideCache);
      
      // weight u_values containers using cubature weights defined in cellBasisCacheSide:
      Intrepid::FunctionSpaceTools::multiplyMeasure<double>(u_values, cellBasisCacheSide->getWeightedMeasures(), u_values);
      Intrepid::FunctionSpaceTools::multiplyMeasure<double>(neighbor_u_values, cellBasisCacheSide->getWeightedMeasures(), neighbor_u_values);
      
      // now, we compute four integrals (test, trial)
      Intrepid::FieldContainer<double> integralValues_me_me(numCells,numFields,numFields);
      Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_me,jump_values,u_values,Intrepid::COMP_BLAS);
//      cout << "integralValues_me_me:\n" << integralValues_me_me;
      
      Intrepid::FieldContainer<double> integralValues_neighbor_me(numCells,numNeighborFields,numFields);
      Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_me,neighbor_jump_values,u_values,Intrepid::COMP_BLAS);
//      cout << "integralValues_me_neighbor:\n" << integralValues_me_neighbor;
      
      Intrepid::FieldContainer<double> integralValues_me_neighbor(numCells,numFields,numNeighborFields);
      Intrepid::FunctionSpaceTools::integrate<double>(integralValues_me_neighbor,jump_values,neighbor_u_values,Intrepid::COMP_BLAS);
//      cout << "integralValues_neighbor_me:\n" << integralValues_neighbor_me;
      
      Intrepid::FieldContainer<double> integralValues_neighbor_neighbor(numCells,numNeighborFields,numNeighborFields);
      Intrepid::FunctionSpaceTools::integrate<double>(integralValues_neighbor_neighbor,neighbor_jump_values,neighbor_u_values,Intrepid::COMP_BLAS);
//      cout << "integralValues_neighbor_neighbor:\n" << integralValues_neighbor_neighbor;

      // because I don't trust Epetra in terms of the format, let's insert values one at a time
      // define lambda for this
      auto insertValues = [stiffnessMatrix] (vector<int> &rowDofOrdinals, vector<int> &colDofOrdinals,
                                             Intrepid::FieldContainer<double> &values) -> void
      {
        // values container is (cell, test, trial)
        int rowCount = rowDofOrdinals.size();
        int colCount = colDofOrdinals.size();
        for (int i=0; i<rowCount; i++)
        {
          for (int j=0; j<colCount; j++)
          {
            // rows in FieldContainer correspond to trial variables, cols to test
            // in stiffness matrix, it's the opposite.
            if (values(0,i,j) != 0) // skip 0's, which I believe Epetra should ignore anyway
              stiffnessMatrix->InsertGlobalValues(1,&rowDofOrdinals[i],1,&colDofOrdinals[j],&values(0,i,j));
          }
        }
      };

      insertValues(myGlobalDofs_u,myGlobalDofs_u,integralValues_me_me);
      insertValues(myGlobalDofs_u,neighborGlobalDofs_u,integralValues_me_neighbor);
      insertValues(neighborGlobalDofs_u,myGlobalDofs_u,integralValues_neighbor_me);
      insertValues(neighborGlobalDofs_u,neighborGlobalDofs_u,integralValues_neighbor_neighbor);
    }
  }
}

void solveAndExport(SolutionPtr soln, VarPtr u, VarPtr v, FunctionPtr beta, HDF5Exporter exporter, int refNumber)
{
  SolverPtr solver = Solver::getDirectSolver();
  
  soln->initializeLHSVector();
  soln->initializeStiffnessAndLoad();
  
  // this is where we want to accumulate any inter-element DG terms
  applyUpwinding(soln, u, v, beta);
  
  // it is important to do the above accumulation before BCs are imposed, which they
  // will be in populateStiffnessAndLoad().
  soln->populateStiffnessAndLoad();
  soln->setProblem(solver);
  
  int solveSuccess = soln->solveWithPrepopulatedStiffnessAndLoad(solver);
  
  if (solveSuccess != 0)
    cout << "solve returned with error code " << solveSuccess << endl;
  
  soln->importSolution(); // determines element-local solution coefficients from the global solution vector
  
  exporter.exportSolution(soln,refNumber);
}