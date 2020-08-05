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

#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RieszRep.h"

using namespace Camellia;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
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
} // namespace
