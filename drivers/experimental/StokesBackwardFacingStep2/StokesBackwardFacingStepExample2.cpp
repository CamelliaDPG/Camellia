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
#include "ParameterFunction.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"

using namespace Camellia;

using namespace std;

static const double X_LEFT = 0.0;
static const double X_STEP = 2.0;
static const double X_OUTFLOW = 10.0;
static const double WIDTH_EAST_REGION = X_OUTFLOW - X_STEP;
static const double Y_BOTTOM = 0.0;
static const double Y_TOP = 1.0;
static const double Y_STEP = (Y_TOP - Y_BOTTOM) / 2.0;

bool cellMatchesX(MeshPtr mesh, double x, GlobalIndexType cellID)
{
  auto cellVertexIndices = mesh->getTopology()->getCell(cellID)->vertices();
  double max_x = X_LEFT, min_x = X_OUTFLOW;
  double max_y = Y_BOTTOM, min_y = Y_TOP;
  for (auto cellVertexIndex : cellVertexIndices)
  {
    auto vertex = mesh->getTopology()->getVertex(cellVertexIndex);
//    cout << "vertex: (" << vertex[0] << "," << vertex[1] << ")\n";
    max_x = max(vertex[0],max_x);
    min_x = min(vertex[0],min_x);
    max_y = max(vertex[1],max_y);
    min_y = min(vertex[1],min_y);
  }
  if ((x >= min_x) && (x <= max_x))
  {
    // candidate matches
    return true;
  }
  else
  {
    return false;
  }
}

std::vector<GlobalIndexType> cellsMatchingX(MeshPtr mesh, double x)
{
  // compared to the implementation below, this uses knowledge of mesh layout, which means we need to probe a lot less
  set<GlobalIndexType> matchingCellIDSet;
  
  auto meshTopo = mesh->getTopology();
  
  int spaceDim = mesh->getDimension();
  Camellia::FieldContainer<double> testPoint(1,spaceDim);
  double yBottom = (x <= X_STEP) ? Y_STEP : Y_BOTTOM;
  testPoint(0,0) = x;
  testPoint(0,1) = (yBottom + Y_TOP) / 2.0;
  vector<GlobalIndexType> matchingCellIDs = mesh->cellIDsForPoints(testPoint);
//  print("test point cell IDs", matchingCellIDs);
  CellPtr cell = meshTopo->getCell(matchingCellIDs[0]);
  
  matchingCellIDSet.insert(matchingCellIDs[0]);
  
  int NORTH_SIDE = 2;
  int SOUTH_SIDE = 0;
  CellPtr northNeighbor = cell->getNeighbor(NORTH_SIDE, meshTopo);
  while (northNeighbor != Teuchos::null)
  {
    if (northNeighbor->isParent(meshTopo))
    {
      auto candidatePairs = northNeighbor->getDescendantsForSide(SOUTH_SIDE, meshTopo, true); // true: leaf nodes only
      for (auto candidatePair : candidatePairs)
      {
        auto cellID = candidatePair.first;
        if (cellMatchesX(mesh, x, cellID))
        {
          northNeighbor = meshTopo->getCell(cellID);
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(northNeighbor->isParent(meshTopo), std::invalid_argument, "no descendant matched x, apparently");
    }
    matchingCellIDSet.insert(northNeighbor->cellIndex());
    northNeighbor = northNeighbor->getNeighbor(NORTH_SIDE, meshTopo);
  }
  CellPtr southNeighbor = cell->getNeighbor(SOUTH_SIDE, meshTopo);
  while (southNeighbor != Teuchos::null)
  {
    if (southNeighbor->isParent(meshTopo))
    {
      auto candidatePairs = southNeighbor->getDescendantsForSide(NORTH_SIDE, meshTopo, true); // true: leaf nodes only
      for (auto candidatePair : candidatePairs)
      {
        auto cellID = candidatePair.first;
        if (cellMatchesX(mesh, x, cellID))
        {
          southNeighbor = meshTopo->getCell(cellID);
        }
      }
    }
    matchingCellIDSet.insert(southNeighbor->cellIndex());
    southNeighbor = southNeighbor->getNeighbor(SOUTH_SIDE, meshTopo);
  }
  
  vector<GlobalIndexType> uniqueMatchingCellIDs(matchingCellIDSet.begin(),matchingCellIDSet.end());
  
//  print("cellIDs for x", uniqueMatchingCellIDs);
  return uniqueMatchingCellIDs;
}

std::vector<GlobalIndexType> cellsMatchingX(MeshPtr mesh, double x, int verticalTestPoints)
{
  int spaceDim = mesh->getDimension();
  Camellia::FieldContainer<double> testPoints(verticalTestPoints,spaceDim);
  double yBottom = (x <= X_STEP) ? Y_STEP : Y_BOTTOM;
  double height  = Y_TOP - yBottom;
  for (int pointOrdinal=0; pointOrdinal<verticalTestPoints; pointOrdinal++)
  {
    testPoints(pointOrdinal,0) = x;
    testPoints(pointOrdinal,1) = yBottom + (pointOrdinal + 1) * height / (verticalTestPoints + 1);
  }
  vector<GlobalIndexType> matchingCellIDs = mesh->cellIDsForPoints(testPoints);
  set<GlobalIndexType> matchingCellIDSet(matchingCellIDs.begin(),matchingCellIDs.end());
  vector<GlobalIndexType> uniqueMatchingCellIDs(matchingCellIDSet.begin(),matchingCellIDSet.end());
  
  print("cellIDs for x", uniqueMatchingCellIDs);
  return uniqueMatchingCellIDs;
}

double lineIntegral(MeshPtr mesh, double x, FunctionPtr f, int numRefinementsPerformed)
{
  double probeDistance = pow(0.5, numRefinementsPerformed+1); // initial mesh has uniform h of 0.5; probeDistance is the smallest possible h after the given number of refinements
  // west of the step, mesh has height 0.5
  // east of the step, mesh has height 1.0
//  int numTestPoints = (x <= X_STEP) ? ceil(0.5 / probeDistance) : ceil(1.0 / probeDistance);
  
  double xProbeLocation = (x == X_LEFT) ? 0.0 : x - probeDistance / 2.0; // if x is not on the left boundary, then probe a bit to the left of x to hit cell interiors
  auto cells = cellsMatchingX(mesh, xProbeLocation); // cellsMatchingX(mesh, xProbeLocation, numTestPoints);
 
  // all elements in our mesh have same type (poly order + topology)
  auto elemType = mesh->getElementType(0);
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh) );
  bool createSideCache = true;
  Camellia::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodes(cells);
  basisCache->setPhysicalCellNodes(physicalCellNodes, cells, createSideCache);
  
  // use knowledge of mesh layout: we want the east side of matching cells, unless we're on the left side of the mesh
  int WEST_SIDE = 3;
  int EAST_SIDE = 1;
  int sideOrdinal = (x == X_LEFT) ? WEST_SIDE : EAST_SIDE;
  auto basisCacheForIntegral = basisCache->getSideBasisCache(sideOrdinal);
  return f->integrate(basisCacheForIntegral);
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  int spaceDim = 2;

  bool useConformingTraces = false;
  double mu = 1.0;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);

  // imitate Truman's thesis
  // set up a mesh topology for the region east of the step: 2 x 2 elements, bottom left corner at (2,0)
  
  vector<double> dims(spaceDim);
  dims[0] = WIDTH_EAST_REGION;
  dims[1] = Y_TOP;
  
  vector<int> numElements(spaceDim);
  numElements[0] = ceil(WIDTH_EAST_REGION / Y_STEP); // make elements (roughly) square
  numElements[1] = 2;
  vector<double> x0(spaceDim);
  x0[0] = X_STEP;
  x0[1] = Y_BOTTOM;

  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);

  // add approximately unit-width elements west of the region just defined:
  int numWestElements = ceil( (X_STEP - X_LEFT) / Y_STEP ) ; // make elements square
  double westElemWidth = (X_STEP - X_LEFT) / numWestElements;
  for (int i=0; i<numWestElements; i++)
  {
    // vertices are defined in counter-clockwise order:
    vector<double> x1(spaceDim), x2(spaceDim), x3(spaceDim);
    x0[0] = X_LEFT + i * westElemWidth;
    x0[1] = Y_STEP;
    x1[0] = X_LEFT + (i+1) * westElemWidth;
    x1[1] = Y_STEP;
    x2[0] = X_LEFT + (i+1) * westElemWidth;
    x2[1] = Y_TOP;
    x3[0] = X_LEFT + i * westElemWidth;
    x3[1] = Y_TOP;
    vector< vector<double> > vertices(4);
    vertices[0] = x0;
    vertices[1] = x1;
    vertices[2] = x2;
    vertices[3] = x3;
    meshTopo->addCell(CellTopology::quad(), vertices);
  }

  int polyOrder = 4, delta_k = 1;

  form.initializeSolution(meshTopo, polyOrder, delta_k);

  VarPtr u1_hat = form.u_hat(1), u2_hat = form.u_hat(2);

  SpatialFilterPtr inflow = SpatialFilter::matchingX(X_LEFT);
  SpatialFilterPtr outflow = SpatialFilter::matchingX(X_OUTFLOW);
  // wherever is not inflow or outflow is a wall:
  SpatialFilterPtr wall = SpatialFilter::negatedFilter(outflow | inflow);

  FunctionPtr y = Function::yn(1);
  FunctionPtr u1_inflow = 8. * (y-0.5) * (1.-y);

  FunctionPtr u_inflow;
  FunctionPtr zero = Function::zero();
  FunctionPtr u_outflow; // used if we impose Dirichlet conditions on outflow (not the best practice)
  if (spaceDim == 2)
  {
    u_inflow = Function::vectorize(u1_inflow,zero);
    u_outflow = Function::vectorize(y * (1.-y), zero);
  }
  else
  {
    u_inflow = Function::vectorize(u1_inflow,zero,zero);
    u_outflow = Function::vectorize(y * (1.-y), zero, zero);
  }

  form.addInflowCondition(inflow, u_inflow);
  form.addWallCondition(wall);
  bool usePhysicalTractions = false; // just use t_n
  form.addOutflowCondition(outflow, usePhysicalTractions);
//  form.addInflowCondition(outflow, u_outflow);
  
  bool useQWeightedGraphNorm = false;
  
  if (useQWeightedGraphNorm)
  {
    double q_beta = 1e-2;
    cout << "SETTING IP to GRAPH NORM with q beta=" << q_beta << ".\n";
    auto bf = form.bf();
    
    vector<int> testVarIDs = bf->varFactory()->testIDs();
    map<int, double> trialVarWeights; // can leave empty
    map<int, double> testL2Weights;
    for (int testVarID : testVarIDs)
    {
      testL2Weights[testVarID] = 1.0; // default to 1.0
    }
    auto q = form.q(); // test function for continuity equation
    testL2Weights[q->ID()] = q_beta;
    form.solution()->setIP(bf->graphNorm(trialVarWeights,testL2Weights));
  }
  
  form.solve();
  
  MeshPtr mesh = form.solution()->mesh();
  
  int refNumber = 0;
  
  std::vector<int> elementCounts;
  std::vector<int> dofCounts;
  std::vector<double> massLossPercentages; // at step for now; can add more points later
  
  HDF5Exporter solutionExporter(mesh, "stokesSteadyBackwardStep", ".");
  solutionExporter.exportSolution(form.solution(), refNumber);
  
  auto u_hat_x = Function::solution(form.u_hat(1), form.solution());
  
  double inflowIntegral = lineIntegral(mesh, X_LEFT, u_hat_x, refNumber);
  cout << "inflow integral = " << inflowIntegral << endl;
  
  double stepIntegral = lineIntegral(mesh, X_STEP, u_hat_x, refNumber);
  cout << "step integral = " << stepIntegral << endl;
  
  double percentageMassLossAtStep = (inflowIntegral - stepIntegral) / inflowIntegral * 100.0;
  cout << "mass loss at step: " << percentageMassLossAtStep << "%\n";
  
  auto q = form.q(); // test function for continuity equation
  auto standardRHS = form.solution()->rhs();
  
  RHSPtr qLoadedRHS = RHS::rhs();
  qLoadedRHS->addTerm(standardRHS->linearTermCopy());
  qLoadedRHS->addTerm(1.0 * q); // With this, we solve for the trial-space preimage of optimal test function q=1.
  
  form.solution()->setRHS(qLoadedRHS);
  
  LinearTermPtr conservationResidual = form.solution()->rhs()->linearTerm() - form.bf()->testFunctional(form.solution(),false); // false: don't exclude boundary terms
  
  double energyThreshold = 0.99;
  auto massConservationRefinementStrategy = Teuchos::rcp( new RefinementStrategy( form.solution()->mesh(), conservationResidual, form.solution()->ip(), energyThreshold ) );
  
  form.solve();

  double energyError = form.solution()->energyErrorTotal();
  int globalDofs = mesh->globalDofCount();
  int activeElements = mesh->numActiveElements();
  if (rank==0) cout << "Initial energy error: " << energyError;
  if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;
  
  elementCounts.push_back(activeElements);
  dofCounts.push_back(globalDofs);
  massLossPercentages.push_back(percentageMassLossAtStep);
  
  HDF5Exporter massConservationLoadExporter(mesh, "mass conservation load solution", ".");
  massConservationLoadExporter.exportSolution(form.solution(), refNumber);
  
  HDF5Exporter massConservationLoadEnergyErrorExporter(form.solution()->mesh(), "massConservationLoadEnergyError", ".");
  
  FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.solution());
  massConservationLoadEnergyErrorExporter.exportFunction(vector<FunctionPtr>{energyErrorFunction}, vector<string>{"energy error"}, refNumber);

  double tol = 1e-4;
  
  int maxRefs = 40;
  do
  {
    refNumber++;
    massConservationRefinementStrategy->refine();
    form.solve();

    massConservationLoadExporter.exportSolution(form.solution(),refNumber);

    energyError = form.solution()->energyErrorTotal();
    globalDofs = mesh->globalDofCount();
    activeElements = mesh->numActiveElements();
//    double massFlux = form.absoluteMassFlux();
//    if (rank==0) cout << "Sum of absolute value of element-wise mass flux across mesh: " << massFlux << endl;
    if (rank==0) cout << "Mass conservation load energy error for refinement " << refNumber << ": " << energyError;
    if (rank==0) cout << " (mesh has " << activeElements << " elements and " << globalDofs << " global dofs)." << endl;

    massConservationLoadEnergyErrorExporter.exportFunction(vector<FunctionPtr>{energyErrorFunction}, vector<string>{"energy error"}, refNumber);
    
    // switch RHSes
    form.solution()->setRHS(standardRHS);
    form.solve();
    solutionExporter.exportSolution(form.solution(),refNumber);
    
    energyError = form.solution()->energyErrorTotal();
    if (rank==0) cout << "Energy error for standard solution: " << energyError << endl;
    
    inflowIntegral = lineIntegral(mesh, X_LEFT, u_hat_x, refNumber);
    cout << "inflow integral = " << inflowIntegral << endl;
    
    stepIntegral = lineIntegral(mesh, X_STEP, u_hat_x, refNumber);
    cout << "step integral = " << stepIntegral << endl;
    
    percentageMassLossAtStep = (inflowIntegral - stepIntegral) / inflowIntegral * 100.0;
    cout << "mass loss at step: " << percentageMassLossAtStep << "%\n";
    
    elementCounts.push_back(activeElements);
    dofCounts.push_back(globalDofs);
    massLossPercentages.push_back(percentageMassLossAtStep);
    
    // restore the mass conservation load, and solve again (need to do this because the solution is used in refinement at the top of this loop)
    form.solution()->setRHS(qLoadedRHS);
    form.solve();
    
    if (energyError > 1e2) // probably we're encountering Gram matrix ill-conditioning in this case; so stop
    {
      cout << "energy error (for standard solution) exceeds 1e2; likely this is due to ill-conditioning on small elements.  Stopping...\n";
      break;
    }
  }
  while ((energyError > tol) && (refNumber < maxRefs));

  
  form.solution()->setRHS(standardRHS);
  form.solve();
  massConservationLoadExporter.exportSolution(form.solution(),maxRefs); // "final" solution; a solution to the original problem
  
  double massFlux = form.absoluteMassFlux();
  if (rank==0) cout << "Sum of absolute value of element-wise mass flux across mesh: " << massFlux << endl;
  energyError = form.solution()->energyErrorTotal();
  if (rank==0) cout << "Energy error for final solution: " << energyError << endl;
  
  GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/stokesBackwardStepMesh", mesh, true, "black", "Refined mesh", 1);
  
  if (rank == 0)
  {
    cout << "FINAL REPORT:\n";
    int elemFormatWidth = 10;
    int dofFormatWidth = 15;
    int massLossFormatWidth = 15;
    cout << setw(elemFormatWidth) << "Elements" << setw(dofFormatWidth) << "Dofs" << setw(massLossFormatWidth) << "Mass Loss %" << endl;
    for (int i=0; i<elementCounts.size(); i++)
    {
      cout << setw(elemFormatWidth) << elementCounts[i] << setw(dofFormatWidth) << dofCounts[i] << setw(massLossFormatWidth) << massLossPercentages[i] << endl;
    }
  }
  
  return 0;
}
