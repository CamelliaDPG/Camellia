//
//  BasisReconciliationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/22/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Var.h"
#include "VarFactory.h"
#include "LinearTerm.h"
#include "BasisFactory.h"
#include "CellTopology.h"

#include "CamelliaCellTools.h"

#include "BasisReconciliation.h"

#include "BasisCache.h"

#include "SerialDenseWrapper.h"

namespace {
  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced )
  {
    VarFactory vf;
    VarPtr u = vf.fieldVar("u");
    LinearTermPtr termTraced = 3.0 * u;
    VarPtr u_hat = vf.traceVar("\\widehat{u}", termTraced);

    // in what follows, the fine basis belongs to the trace variable and the coarse to the field
    
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;  // when the coarse domain is a side, this can be non-zero, but fields are always in the volume, so this will always be 0
    unsigned coarseDomainOrdinalInRefinementRoot = 0; // again, since coarse domain is for a field variable, this will always be 0
    unsigned coarseSubcellPermutation = 0;            // not sure if this will ever be nontrivial permutation in practice; we don't test anything else here
    
    // 1D tests
    int H1Order = 2;
    CellTopoPtr lineTopo = CellTopology::line();
    
    // we use HGRAD here because we want to be able to ask for basis ordinal for vertex, e.g. (and HVOL would hide this)
    BasisPtr lineBasisQuadratic = BasisFactory::basisFactory()->getBasis(H1Order, lineTopo->getShardsTopology().getKey(), Camellia::FUNCTION_SPACE_HGRAD);
    
    CellTopoPtr pointTopo = CellTopology::point();
    BasisPtr pointBasis = BasisFactory::basisFactory()->getBasis(1, pointTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    TEST_EQUALITY(pointBasis->getCardinality(), 1); // sanity test

    // first, simple test: for a field variable on an unrefined line, compute the weights for a trace of that variable at the left vertex
    
    // expect weights to be nodal for the vertex (i.e. 1 at the field basis ordinal corresponding to the vertex, and 0 elsewhere)
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinementLine = RefinementPattern::noRefinementPattern(lineTopo);
    noRefinements.push_back( make_pair(noRefinementLine.get(), 0) );
  
    RefinementBranch oneRefinement;
    RefinementPatternPtr regularRefinementLine = RefinementPattern::regularRefinementPatternLine();
    oneRefinement.push_back( make_pair(regularRefinementLine.get(), 1) ); // 1: choose the child to the right
    
    vector<RefinementBranch> refinementBranches;
    refinementBranches.push_back(noRefinements);
    refinementBranches.push_back(oneRefinement);

    FieldContainer<double> lineRefNodes(lineTopo->getVertexCount(), lineTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(lineRefNodes, lineTopo);
    
    BasisCachePtr lineBasisCache = BasisCache::basisCacheForReferenceCell(lineTopo, 1);
    
    for (int i=0; i< refinementBranches.size(); i++) {
      RefinementBranch refBranch = refinementBranches[i];
      
      for (int fineVertexOrdinal=0; fineVertexOrdinal <= 1; fineVertexOrdinal++) {
        int fineSubcellOrdinalInFineDomain = 0;
        
        int numPoints = 1;
        FieldContainer<double> vertexPointInLeaf(numPoints, lineTopo->getDimension());
        for (int d=0; d < lineTopo->getDimension(); d++) {
          vertexPointInLeaf(0,d) = lineRefNodes(fineVertexOrdinal,d);
        }
        
        FieldContainer<double> vertexPointInAncestor(numPoints, lineTopo->getDimension());
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, vertexPointInLeaf, vertexPointInAncestor);
        
        lineBasisCache->setRefCellPoints(vertexPointInAncestor);
        
        SubBasisReconciliationWeights weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, u->ID(), pointTopo->getDimension(),
                                                                                                            pointBasis, fineSubcellOrdinalInFineDomain, refBranch, fineVertexOrdinal,
                                                                                                            lineTopo->getDimension(), lineBasisQuadratic,
                                                                                                            coarseSubcellOrdinalInCoarseDomain,
                                                                                                            coarseDomainOrdinalInRefinementRoot,
                                                                                                            coarseSubcellPermutation);
        // fine basis is the point basis (the trace); coarse is the line basis (the field)
        
        TEST_EQUALITY(weights.fineOrdinals.size(), 1);
        
        int coarseOrdinalInWeights = 0; // iterate over this
        
        double tol = 1e-15; // for floating equality
        
        int oneCell = 1;
        FieldContainer<double> coarseValuesExpected(oneCell,lineBasisQuadratic->getCardinality(),numPoints);
        termTraced->values(coarseValuesExpected, u->ID(), lineBasisQuadratic, lineBasisCache);
        
        FieldContainer<double> fineValues(pointBasis->getCardinality(),numPoints);
        fineValues[0] = 1.0; // pointBasis is identically 1.0
        
        FieldContainer<double> coarseValuesActual(weights.coarseOrdinals.size(), pointBasis->getCardinality());
        SerialDenseWrapper::multiply(coarseValuesActual, weights.weights, fineValues, 'T', 'N');

        int pointOrdinal  = 0;
        for (int coarseOrdinal=0; coarseOrdinal < lineBasisQuadratic->getCardinality(); coarseOrdinal++) {
          double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);
          
          double actualValue;
          if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end()) {
            actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
            coarseOrdinalInWeights++;
          } else {
            actualValue = 0.0;
          }
          
          if (abs(expectedValue > tol) ) {
            TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
          } else {
            TEST_ASSERT( abs(actualValue) < tol );
          }
        }
      }
    }
  }
} // namespace