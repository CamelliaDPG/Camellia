//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BasisSumFunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/29/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "Function.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TensorBasis.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
MeshPtr singleCellSpaceTimeMesh(CellTopoPtr spaceTopo, vector<int> H1Order)
{
  CellTopoPtr timeTopo = CellTopology::line();
  int tensorialDegree = 1;
  CellTopoPtr tensorTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);

  Camellia::FieldContainer<double> timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
  double t0 = 0, t1 = 1;
  // time will go from t0 to t1:
  timeNodes(0,0) = t0;
  timeNodes(1,0) = t1;

  Camellia::FieldContainer<double> spaceNodes(spaceTopo->getNodeCount(), spaceTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(spaceNodes, spaceTopo);
  Camellia::FieldContainer<double> tensorNodesFC(tensorTopo->getNodeCount(), tensorTopo->getDimension());
  tensorTopo->initializeNodes({spaceNodes, timeNodes}, tensorNodesFC);

  vector<vector<double> > tensorNodes;
  CamelliaCellTools::pointsVectorFromFC(tensorNodes, tensorNodesFC);

  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(tensorTopo->getDimension()) );

  meshTopo->addCell(tensorTopo, tensorNodes);

  double epsilon = 1.0;
  SpaceTimeHeatFormulation form(spaceTopo->getDimension(), epsilon, true);

  int delta_k = 1;
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) );

  return mesh;
}

void testConstantBasisSpaceTime(CellTopoPtr spaceTopo, bool testSides, Teuchos::FancyOStream &out, bool &success)
{
  // a TensorBasis made up of nodal components should itself be nodal.
  // therefore, using constant coefficients we should get a constant function
  const static double CONST_VALUE = 0.5;

  FunctionPtr f = Function::constant(CONST_VALUE);

  int spacePolyOrder = 3;
  int spaceH1Order = spacePolyOrder + 1;
  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  BasisPtr spaceBasis = basisFactory->getBasis(spaceH1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);

  int timePolyOrder = 2;
  int timeH1Order = timePolyOrder + 1;
  BasisPtr timeBasis = basisFactory->getBasis(timeH1Order, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);

  typedef Camellia::TensorBasis<double, Intrepid::FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spaceBasis, timeBasis) );

  Camellia::FieldContainer<double> basisCoefficients(tensorBasis->getCardinality());
  basisCoefficients.initialize(CONST_VALUE);

  FunctionPtr basisSumFunction = BasisSumFunction::basisSumFunction(tensorBasis, basisCoefficients);
  vector<int> H1Order = {spaceH1Order, timeH1Order};

  MeshPtr spaceTimeMesh = singleCellSpaceTimeMesh(spaceTopo, H1Order);

  double tol = 1e-14;
  if (!testSides)
  {
    double l2Err = (f - basisSumFunction)->l2norm(spaceTimeMesh);
    TEST_COMPARE(l2Err, <, tol);
  }
  else
  {
    FunctionPtr restrictor = Function::meshSkeletonCharacteristic();
    double l2Err = (f * restrictor - basisSumFunction * restrictor)->l2norm(spaceTimeMesh);
    TEST_COMPARE(l2Err, <, tol);
  }
}

void testOverridingBasisCache(Teuchos::FancyOStream &out, bool &success)
{
  MPIWrapper::CommWorld()->Barrier();
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  int H1Order = 2;
  vector<int> elemCounts = {1,1};
  
  GlobalIndexType rootCellID = 0;
  
  const int numSides = 1;
  
  for (int rootSideIndex = 0; rootSideIndex<numSides; rootSideIndex++)
  {
    // create a non-distributed mesh (will be replicated on each MPI rank): we are not testing anything MPI-related here…
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, elemCounts, H1Order, 0, {0.,0.},
                                                map<int,int>(), map<int,int>(), MPIWrapper::CommSerial());

    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    mesh->registerSolution(solution);
    
    FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
    map<int, FunctionPtr> solutionMap;
  //  solutionMap[form.u()->ID()] = Function::constant(1.0);
  //  solutionMap[form.sigma()->ID()] = Function::constant({1.0,0.0});
    solutionMap[form.u_hat()->ID()] = Function::constant(1.0) * traceFxn;
    FunctionPtr n_parity = Function::normal() * Function::sideParity();
    solutionMap[form.sigma_n_hat()->ID()] = Function::constant({0.0,1.0}) * n_parity;
    
    solution->projectOntoCell(solutionMap, rootCellID, rootSideIndex);
    
    // get solution coefficients:
    auto elemType = solution->mesh()->getElementType(rootCellID);
    BasisPtr   u_hat_basis = elemType->trialOrderPtr->getBasis(form.u_hat()->ID(),       rootSideIndex);
    BasisPtr sigma_n_basis = elemType->trialOrderPtr->getBasis(form.sigma_n_hat()->ID(), rootSideIndex);
    FieldContainer<double> basisCoefficients_u_hat(u_hat_basis->getCardinality());
    FieldContainer<double> basisCoefficients_sigma_n(sigma_n_basis->getCardinality());
    solution->getBasisCoefficientsForCell(basisCoefficients_u_hat,   form.u_hat()->ID(),       rootCellID, STANDARD_SOLUTION, 0, rootSideIndex);
    solution->getBasisCoefficientsForCell(basisCoefficients_sigma_n, form.sigma_n_hat()->ID(), rootCellID, STANDARD_SOLUTION, 0, rootSideIndex);
    
    BasisCachePtr rootCellBasisCache     = BasisCache::basisCacheForCell(mesh, rootCellID);
    BasisCachePtr rootCellSideBasisCache = rootCellBasisCache->getSideBasisCache(rootSideIndex);
    
    // now, refine uniformly:
    RefinementStrategy::hRefineUniformly(mesh);
    
//    out << "sigma_n, basis coefficients on root cell:\n";
//    for (int basisOrdinal=0; basisOrdinal<sigma_n_basis->getCardinality(); basisOrdinal++)
//    {
//      out << basisOrdinal << ": " << basisCoefficients_sigma_n(basisOrdinal) << std::endl;
//    }
    
    const bool boundaryValueOnly = true;
    BasisSumFunction     u_bsf(  u_hat_basis, basisCoefficients_u_hat,   rootCellSideBasisCache, OP_VALUE, boundaryValueOnly);
    BasisSumFunction sigma_bsf(sigma_n_basis, basisCoefficients_sigma_n, rootCellSideBasisCache, OP_VALUE, boundaryValueOnly);
    
    double tol = 1e-14;
    
    FunctionPtr   u_hat_exact = solutionMap[form.u_hat()->ID()];
    FunctionPtr sigma_n_exact = solutionMap[form.sigma_n_hat()->ID()];
    
    // as a sanity check, let's first confirm that the actual values agree with expected on the root basis cache
    {
      const int numCells = 1;
      const int cellOrdinal = 0;
      auto numPoints = rootCellSideBasisCache->getRefCellPointsFC().dimension(0);
      
      FieldContainer<double> u_actual_values(numCells,numPoints);
      FieldContainer<double> u_exact_values(numCells,numPoints);
      u_bsf.values(u_actual_values, rootCellSideBasisCache);
      u_hat_exact->values(u_exact_values, rootCellSideBasisCache);
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double  exact_value = u_exact_values(cellOrdinal,pointOrdinal);
        double actual_value = u_actual_values(cellOrdinal,pointOrdinal);
        TEST_FLOATING_EQUALITY(exact_value, actual_value, tol);
      }
      
      FieldContainer<double> sigma_actual_values(numCells,numPoints);
      FieldContainer<double> sigma_exact_values(numCells,numPoints);
      sigma_bsf.values(sigma_actual_values, rootCellSideBasisCache);
      sigma_n_exact->values(sigma_exact_values, rootCellSideBasisCache);
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double  exact_value = sigma_exact_values(cellOrdinal,pointOrdinal);
        double actual_value = sigma_actual_values(cellOrdinal,pointOrdinal);
        TEST_FLOATING_EQUALITY(exact_value, actual_value, tol);
      }
    }
    
    auto rootCell = mesh->getTopology()->getCell(rootCellID);
    
    auto childrenInfo = rootCell->childrenForSide(rootSideIndex);
    for (auto childInfo : childrenInfo)
    {
      auto childID   = childInfo.first;
      auto childSide = childInfo.second;
      
      auto childVolumeCache = BasisCache::basisCacheForCell(mesh, childID);
      auto childSideCache   = childVolumeCache->getSideBasisCache(childSide);
      
      int numCells = 1;
      auto numPoints = childSideCache->getRefCellPointsFC().dimension(0);
      
      FieldContainer<double> u_actual_values(numCells,numPoints);
      FieldContainer<double> u_exact_values(numCells,numPoints);
      u_bsf.values(u_actual_values, childSideCache);
      u_hat_exact->values(u_exact_values, childSideCache);
      
      const int cellOrdinal = 0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double  exact_value = u_exact_values(cellOrdinal,pointOrdinal);
        double actual_value = u_actual_values(cellOrdinal,pointOrdinal);
        TEST_FLOATING_EQUALITY(exact_value, actual_value, tol);
      }
      
      FieldContainer<double> sigma_actual_values(numCells,numPoints);
      FieldContainer<double> sigma_exact_values(numCells,numPoints);
      sigma_bsf.values(sigma_actual_values, childSideCache);
      sigma_n_exact->values(sigma_exact_values, childSideCache);
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double  exact_value = sigma_exact_values(cellOrdinal,pointOrdinal);
        double actual_value = sigma_actual_values(cellOrdinal,pointOrdinal);
        TEST_FLOATING_EQUALITY(exact_value, actual_value, tol);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( BasisSumFunction, OverridingBasisCache )
{
  testOverridingBasisCache(out, success);
}

TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisLine )
{
  CellTopoPtr spaceTopo = CellTopology::line();
  bool testSides = false;
  testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
}

TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisQuad )
{
  CellTopoPtr spaceTopo = CellTopology::quad();
  bool testSides = false;
  testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
}

TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisLineSides )
{
  CellTopoPtr spaceTopo = CellTopology::line();
  bool testSides = true;
  testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
}

TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisQuadSides )
{
  CellTopoPtr spaceTopo = CellTopology::quad();
  bool testSides = true;
  testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
}
} // namespace
