//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RefinementStrategyTests
//  Camellia
//
//  Created by Nate Roberts on 2/16/15.
//
//

#include "RefinementStrategy.h"

#include "GlobalDofAssignment.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RieszRep.h"

using namespace Camellia;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  class TestErrorIndicator : public ErrorIndicator
  {
  public:
    TestErrorIndicator(MeshPtr mesh, std::map<GlobalIndexType,double> &localErrorMeasures)
    :
    ErrorIndicator(mesh)
    {
      this->_localErrorMeasures = localErrorMeasures;
    }
    
    virtual void measureError() override
    {
      // no-op; _localErrorMeasures was set at construction.
    }
  };

  TEUCHOS_UNIT_TEST( RefinementStrategy, ElementBudget )
  {
    // tests using the new "element budget" feature, which unrefines as needed to maintain a requested element count
    
    int elementBudget = 5; // limit to 5 elements
    
    int spaceDim = 2;
    bool conformingTraces = true;

    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();

    int H1Order = 1;
    // make a mesh of two unit squares:
    vector<double> dimensions {2.0,1.0};
    vector<int> elementCounts {2,1};
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);
    
    TEST_EQUALITY(2, mesh->numActiveElements());
    
    set<GlobalIndexType> cellsToRefine {0};
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    // the refined initial mesh should have 5 elements
    TEST_EQUALITY(elementBudget, mesh->numActiveElements());
    
    // to define a synthetic error representation, we just use the function error indicator; the L^2 norm of this
    // function, interpreted as error, is used to drive refinements
    // we just want some function that prefers to refine in cell 1 (right-most); the function x will serve this purpose.
    auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(mesh, Function::xn(1), 0);
    
    double refinementThreshold = 0.99; // set so that only cell 1 gets marked for refinement
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    refStrategy->setMaxElements(elementBudget);
    
    bool unrefine = true;
    bool printToConsole = false;
    refStrategy->refineAndUnrefine(unrefine, printToConsole);
    
    // check that cell 1 was in fact refined:
    auto meshTopo = mesh->getTopology();
    if (meshTopo->isValidCellIndex(1))
    {
      CellPtr cell1 = meshTopo->getCell(1);
      TEST_EQUALITY(true, cell1->isParent(meshTopo));
    }
    // check that cell 0 was unrefined:
    if (meshTopo->isValidCellIndex(0))
    {
      CellPtr cell0 = meshTopo->getCell(0);
      TEST_EQUALITY(false, cell0->isParent(meshTopo));
    }
    // check that the element budget is exactly satisfied:
    TEST_EQUALITY(elementBudget, mesh->numActiveElements());
  }

  TEUCHOS_UNIT_TEST( RefinementStrategy, GetNorm )
  {
    int spaceDim = 2;
    bool conformingTraces = true;

    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();

    IPPtr ip = bf->l2Norm();

    FunctionPtr weight = Function::xn(1);
    LinearTermPtr lt = weight * form.v();
    // Riesz rep should be simply (weight * v) (since the ip is (v,v) ).

    int H1Order = 1;
    // make a unit square mesh:
    vector<double> dimensions {1.0,1.0};
    vector<int> elementCounts {1, 1};
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);

    RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, lt) );

    rieszRep->computeRieszRep();

    FunctionPtr repFxn = RieszRep::repFunction(form.v(), rieszRep);

    FunctionPtr expectedRepFxn = weight;

    double err = (repFxn - expectedRepFxn)->l2norm(mesh);

    double tol = 1e-14;
    TEST_COMPARE(err,<,tol);

    double expectedNorm = weight->l2norm(mesh);
    double actualNorm = rieszRep->getNorm();

    TEST_FLOATING_EQUALITY(expectedNorm,actualNorm, tol);

    // Now that we have our RieszRep setup and sanity checks done, create RefinementStrategy:

    double relativeEnergyThreshold = 0.2; // arbitrary
    RefinementStrategy refStrategy( mesh, lt, ip, relativeEnergyThreshold);

    refStrategy.refine();

    int refinementNumber = 0;
    double actualEnergyError = refStrategy.getEnergyError(refinementNumber);

    TEST_FLOATING_EQUALITY(expectedNorm,actualEnergyError, tol);
  }

  TEUCHOS_UNIT_TEST( RefinementStrategy, GreedyHUnrefinement )
  {
    // some basic tests of h-unrefinement
    
    int spaceDim = 2;
    bool conformingTraces = true;

    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();

    int H1Order = 1;
    // make a mesh of two unit squares:
    vector<double> dimensions {2.0,1.0};
    vector<int> elementCounts {2,1};
    MeshPtr mesh = MeshFactory::rectilinearMeshNoPruning(bf, dimensions, elementCounts, H1Order);
    
    // 2 elements to start
    const int expectedInitialElements = elementCounts[0] * elementCounts[1];
    TEST_EQUALITY(expectedInitialElements, mesh->numActiveElements());
    
    set<GlobalIndexType> cellsToRefine {0};
    auto refPattern = RefinementPattern::regularRefinementPatternQuad();
    mesh->hRefine(cellsToRefine, refPattern);
    
    // the refined initial mesh should have 5 elements (2 initial + 4 children - 1 inactive parent)
    const int numCellsToRefine = cellsToRefine.size();
    int expectedActiveElementCount = numCellsToRefine * (refPattern->numChildren() - 1) + expectedInitialElements;
    TEST_EQUALITY(expectedActiveElementCount, mesh->numActiveElements());
    
    const double unrefinementThreshold = .001;
    
    // define a synthetic error on the mesh
    std::map<GlobalIndexType,double> localCellError;
    const double maxError = 1.0;
    auto myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      auto cell = mesh->getTopology()->getCell(myCellID);
      // set all the cells from the initial mesh that weren't refined to have error maxError
      if (cell->getParent() == Teuchos::null)
      {
        localCellError[myCellID] = maxError;
      }
      else
      {
        // set the refined cell children to have error just below unrefinementThreshold divided by the number of children:
        const int numChildren = refPattern->numChildren();
        const double childError = maxError * unrefinementThreshold * (1. - 1e-8) / numChildren;
        localCellError[myCellID] = childError;
      }
    }
    
    ErrorIndicatorPtr errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    double refinementThreshold = 1.1; // arbitrary; unused
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    refStrategy->greedyHUnrefinement(unrefinementThreshold);
    
    // test that we again have the expectedInitialElements
    TEST_EQUALITY(expectedInitialElements, mesh->numActiveElements());
    
    // Test 2: repeat, but now give the refined element enough error such that it should not be unrefined
    mesh->hRefine(cellsToRefine, refPattern);
    expectedActiveElementCount = numCellsToRefine * (refPattern->numChildren() - 1) + expectedInitialElements;
    TEST_EQUALITY(expectedActiveElementCount, mesh->numActiveElements());
    
    // redefine error:
    localCellError.clear();
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      auto cell = mesh->getTopology()->getCell(myCellID);
      // set all the cells from the initial mesh that weren't refined to have error maxError
      if (cell->getParent() == Teuchos::null)
      {
        localCellError[myCellID] = maxError;
      }
      else
      {
        // set the refined cell children to have error just above unrefinementThreshold divided by the number of children:
        const int numChildren = refPattern->numChildren();
        const double childError = maxError * unrefinementThreshold * (1. + 1e-8) / numChildren;
        localCellError[myCellID] = childError;
      }
    }
    
    errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    refinementThreshold = 1.1; // arbitrary; unused
    refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    refStrategy->greedyHUnrefinement(unrefinementThreshold);
    
    // test that we still have the expectedActiveElementCount from just after the refinement
    TEST_EQUALITY(expectedActiveElementCount, mesh->numActiveElements());
    
    // TODO: test with a case in which unrefinement would violate 1-irregularity
  }

  TEUCHOS_UNIT_TEST( RefinementStrategy, GreedyPUnrefinement )
  {
    // some basic tests of p-unrefinement
    
    int spaceDim = 2;
    bool conformingTraces = true;

    PoissonFormulation form(spaceDim,conformingTraces);
    BFPtr bf = form.bf();

    int H1Order = 1; // lowest order until we refine
    // make a mesh of 2x2 unit squares:
    vector<double> dimensions {2.0,2.0};
    vector<int> elementCounts {2,2};
    MeshPtr mesh = MeshFactory::rectilinearMeshNoPruning(bf, dimensions, elementCounts, H1Order);
    
    // 2 elements to start
    const int expectedInitialElements = elementCounts[0] * elementCounts[1];
    TEST_EQUALITY(expectedInitialElements, mesh->numActiveElements());
    
    set<GlobalIndexType> cellsToRefine {0,1};
    mesh->pRefine(cellsToRefine);
        
    const double unrefinementThreshold = .001;
    
    // Test 1: unrefinements are triggered and valid
    // define a synthetic error on the mesh
    std::map<GlobalIndexType,double> localCellError;
    const double maxError = 1.0;
    auto myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      auto pForCell = mesh->globalDofAssignment()->getH1Order(myCellID)[0];
      // set all the cells from the initial mesh that weren't refined to have error maxError
      if (pForCell == H1Order)
      {
        localCellError[myCellID] = maxError;
      }
      else
      {
        // set the high-order cells to have error just below unrefinementThreshold
        const double err = maxError * unrefinementThreshold * (1. - 1e-8);
        localCellError[myCellID] = err;
      }
    }
    
    ErrorIndicatorPtr errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    double refinementThreshold = 1.1; // arbitrary; unused
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    // test that maxH1Order is H1Order + 1, and min is H1Order
    TEST_EQUALITY(H1Order+1, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order,   mesh->globalDofAssignment()->getMinH1Order());
    
    refStrategy->greedyPUnrefinement(unrefinementThreshold);
    
    // test that maxH1Order and minH1Order are H1Order again
    TEST_EQUALITY(H1Order, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order, mesh->globalDofAssignment()->getMinH1Order());
    
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      TEST_EQUALITY(H1Order, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
    }
    
    // Test 2: unrefinements are not triggered
    mesh->pRefine(cellsToRefine);
        
    // redefine error
    localCellError.clear();
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      auto pForCell = mesh->globalDofAssignment()->getH1Order(myCellID)[0];
      // set all the cells that weren't refined to have error maxError
      if (pForCell == H1Order)
      {
        localCellError[myCellID] = maxError;
      }
      else
      {
        // set the higher-order cells to have error just above unrefinementThreshold
        const double err = maxError * unrefinementThreshold * (1. + 1e-8);
        localCellError[myCellID] = err;
      }
    }
    
    errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    refinementThreshold = 1.1; // arbitrary; unused
    refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    // test that maxH1Order is H1Order + 1, and min is H1Order
    TEST_EQUALITY(H1Order+1, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order,   mesh->globalDofAssignment()->getMinH1Order());
    
    refStrategy->greedyPUnrefinement(unrefinementThreshold);
    
    // test that maxH1Order and minH1Order have not changed
    TEST_EQUALITY(H1Order+1, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order,   mesh->globalDofAssignment()->getMinH1Order());
    
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      // refined cells should remain refined
      if (cellsToRefine.find(myCellID) == cellsToRefine.end())
      {
        TEST_EQUALITY(H1Order, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
      }
      else
      {
        TEST_EQUALITY(H1Order+1, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
      }
    }
        
    // Test 3: start with higher-order mesh, then lower order in lowerOrderCells.  Trigger unrefinement in lowerOrderCells; this should be excluded because of the requirement that neighbors not differ by more than order 1.
    mesh = MeshFactory::rectilinearMeshNoPruning(bf, dimensions, elementCounts, H1Order+3);
    
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMinH1Order());
    
    set<GlobalIndexType> lowerOrderCells {0,1};
    mesh->pRefine(lowerOrderCells, -1);
    
    for (auto cellID : lowerOrderCells)
    {
      TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getH1Order(cellID)[0]);
    }
    
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getMinH1Order());
    
    // now set up a further unrefinement of the low-order cells.  This should be excluded by the smoothness criterion (neighbors should not differ by more than 1 order)
    localCellError.clear();
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      if (lowerOrderCells.find(myCellID) != lowerOrderCells.end())
      {
        // set the low-order cells to have zero error
        const double err = 0.0;
        localCellError[myCellID] = err;
      }
      else
      {
        localCellError[myCellID] = maxError;
      }
    }
    errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    refinementThreshold = 1.1; // arbitrary; unused
    refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    refStrategy->greedyPUnrefinement(unrefinementThreshold);
    
    // test that maxH1Order and minH1Order have not changed
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getMinH1Order());
    
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      // lower-order cells should still have order H1Order+2; higher-order cells H1Order+3
      if (lowerOrderCells.find(myCellID) == lowerOrderCells.end())
      {
        TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
      }
      else
      {
        TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
      }
    }
    
    out << "Test 4: allowable unrefinement from order " << H1Order+3 << " to " << H1Order+2 << std::endl;
    // Test 4: similar test, but now set up such that the unrefinement is allowed, by making the *high-order* cells be the ones with zero error.
    mesh = MeshFactory::rectilinearMeshNoPruning(bf, dimensions, elementCounts, H1Order+3);
    
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMinH1Order());
    
    mesh->pRefine(lowerOrderCells, -1);
    
    for (auto cellID : lowerOrderCells)
    {
      TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getH1Order(cellID)[0]);
    }
    
    TEST_EQUALITY(H1Order+3, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getMinH1Order());
    
    // now set up a further unrefinement of the low-order cells.  This should be excluded by the smoothness criterion (neighbors should not differ by more than 1 order)
    localCellError.clear();
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      if (lowerOrderCells.find(myCellID) == lowerOrderCells.end())
      {
        // set the high-order cells to have zero error
        const double err = 0.0;
        localCellError[myCellID] = err;
      }
      else
      {
        localCellError[myCellID] = maxError;
      }
    }
    errorIndicator = Teuchos::rcp(new TestErrorIndicator(mesh, localCellError) );
    
    refinementThreshold = 1.1; // arbitrary; unused
    refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThreshold));
    
    refStrategy->greedyPUnrefinement(unrefinementThreshold);
    // test that maxH1Order and minH1Order are now both H1Order+2
    TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getMaxH1Order());
    TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getMinH1Order());
    
    myCells = mesh->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      // all cells should now have order H1Order+2
      TEST_EQUALITY(H1Order+2, mesh->globalDofAssignment()->getH1Order(myCellID)[0]);
    }
  }
} // namespace
