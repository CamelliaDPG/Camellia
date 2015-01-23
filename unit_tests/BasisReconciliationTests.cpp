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

#include "BasisReconciliation.h"

namespace {
  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced )
  {
    VarFactory vf;
    VarPtr u = vf.fieldVar("u");
    LinearTermPtr termTraced = 1.0 * u;
    VarPtr u_hat = vf.traceVar("\\widehat{u}", termTraced);
    
    // 1D tests
    int H1Order = 2;
    CellTopoPtr lineTopo = CellTopology::line();
    
    BasisPtr lineBasisQuadratic = BasisFactory::basisFactory()->getBasis(H1Order, lineTopo->getShardsTopology().getKey(), Camellia::FUNCTION_SPACE_HGRAD);
    
    CellTopoPtr pointTopo = CellTopology::point();
    BasisPtr pointBasis = BasisFactory::basisFactory()->getBasis(1, pointTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinementLine = RefinementPattern::noRefinementPattern(lineTopo);
    noRefinements.push_back( make_pair(noRefinementLine.get(), 0) );
    int fineNodeOrdinal = 0;
  
    SubBasisReconciliationWeights weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, u->ID(), pointTopo->getDimension(),
                                                                                                        pointBasis, 0, noRefinements, fineNodeOrdinal,
                                                                                                        lineTopo->getDimension(), lineBasisQuadratic, 0, 0, 0);
    
    
  }
} // namespace