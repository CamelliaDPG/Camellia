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
  TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_Vertex)
  {
    int numPoints = 1;
    int spaceDim  = 0;
    FieldContainer<double> fineSubcellPoints(numPoints, spaceDim);
    
    FieldContainer<double> coarseSubcellPoints;
    
    unsigned fineSubcellDimension = 0; // vertex
    unsigned fineSubcellOrdinalInFineDomain = 1; // vertex ordinal 1 in *fine domain*
    
    unsigned fineDomainDim = 1; // fine domain is 1D
    unsigned fineDomainOrdinalInRefinementLeaf = 1; // side 1

    CellTopoPtr volumeTopo = CellTopology::quad();
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
    noRefinements.push_back( make_pair(noRefinement.get(), 0) );
    
    unsigned coarseSubcellDimension = 2; // volume
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;
    unsigned coarseDomainDim = 2;
    unsigned coarseDomainOrdinalInRefinementRoot = 0;
    unsigned coarseSubcellPermutation = 0;
    
    BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
                                                             fineDomainDim, fineDomainOrdinalInRefinementLeaf, noRefinements,
                                                             coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
                                                             coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
    
    // we have no refinements, so basically what we're doing is picking vertex ordinal 1 from side ordinal 1 of the quad
    // should be mapped to (1,1)
    
    double tol = 1e-15;
    double expected_x = 1.0;
    double expected_y = 1.0;
    
    double actual_x = coarseSubcellPoints(0,0);
    double actual_y = coarseSubcellPoints(0,1);
    
    TEST_FLOATING_EQUALITY(actual_x, expected_x, tol);
    TEST_FLOATING_EQUALITY(actual_y, expected_y, tol);
  }
  
  TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_Edge)
  {
    // map two points on a "fine" edge to the coarse volume
    
    int numPoints = 2;
    int fineDim  = 1;
    FieldContainer<double> fineSubcellPoints(numPoints, fineDim);
    fineSubcellPoints(0,0) = -0.25;
    fineSubcellPoints(1,0) = 0.5;
    
    int coarseDim = 2;
    FieldContainer<double> expectedCoarseSubcellPoints(numPoints,coarseDim);
    expectedCoarseSubcellPoints(0,0) = -1.0;
    expectedCoarseSubcellPoints(0,1) = -fineSubcellPoints(0,0);
    expectedCoarseSubcellPoints(1,0) = -1.0;
    expectedCoarseSubcellPoints(1,1) = -fineSubcellPoints(1,0);
    
    FieldContainer<double> coarseSubcellPoints;
    
    unsigned fineSubcellDimension = fineDim; // edge
    unsigned fineSubcellOrdinalInFineDomain = 0; // edge
    
    unsigned fineDomainDim = fineDim; // fine domain is 1D
    unsigned fineDomainOrdinalInRefinementLeaf = 3; // side 3
    
    CellTopoPtr volumeTopo = CellTopology::quad();
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
    noRefinements.push_back( make_pair(noRefinement.get(), 0) );
    
    unsigned coarseSubcellDimension = 2; // volume
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;
    unsigned coarseDomainDim = 2;
    unsigned coarseDomainOrdinalInRefinementRoot = 0;
    unsigned coarseSubcellPermutation = 0;
    
    BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
                                                             fineDomainDim, fineDomainOrdinalInRefinementLeaf, noRefinements,
                                                             coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
                                                             coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
    
    // we have no refinements, so basically what we're doing is picking vertex ordinal 1 from side ordinal 1 of the quad
    // should be mapped to (1,1)
    
    double tol = 1e-15;
    
    TEST_COMPARE_FLOATING_ARRAYS(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
  }
  
  void equispacedPoints(int numPoints1D, CellTopoPtr cellTopo, FieldContainer<double> &points) {
    if (cellTopo->getDimension() == 1) {
      // compute some equispaced points on the reference line:
      points.resize(numPoints1D, cellTopo->getDimension());
      for (int pointOrdinal=0; pointOrdinal < numPoints1D; pointOrdinal++) {
        int d = 0;
        points(pointOrdinal,d) = -1.0 + pointOrdinal * (2.0 / (numPoints1D - 1));
      }
    } else if (cellTopo->getKey() == CellTopology::quad()->getKey()) {
      points.resize(numPoints1D * numPoints1D, cellTopo->getDimension());
      
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled CellTopology");
    }
  }
  
  void termTracedTest(Teuchos::FancyOStream &out, bool &success, CellTopoPtr volumeTopo, VarType traceOrFluxType) {
    int spaceDim = volumeTopo->getDimension();
    
    VarFactory vf;
    VarPtr fieldVar, traceVar;
    if ((traceOrFluxType != FLUX) && (traceOrFluxType != TRACE)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "type must be flux or trace");
    } else if (spaceDim==1) {
      fieldVar = vf.fieldVar("psi", L2);
      FunctionPtr n = Function::normal_1D();
      FunctionPtr parity = Function::sideParity();
      LinearTermPtr fluxTermTraced = 3.0 * n * parity * fieldVar;
      traceVar = vf.fluxVar("\\widehat{\\psi}_n", fluxTermTraced);
    } else if (traceOrFluxType==FLUX) {
      fieldVar = vf.fieldVar("psi", VECTOR_L2);
      FunctionPtr n = Function::normal();
      FunctionPtr parity = Function::sideParity();
      LinearTermPtr fluxTermTraced = 3.0 * n * parity * fieldVar;
      traceVar = vf.fluxVar("\\widehat{\\psi}_n", fluxTermTraced);
    } else {
      fieldVar = vf.fieldVar("u");
      LinearTermPtr termTraced = 3.0 * fieldVar;
      traceVar = vf.traceVar("\\widehat{u}", termTraced);
    }
    
    // in what follows, the fine basis belongs to the trace variable and the coarse to the field
    
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;  // when the coarse domain is a side, this can be non-zero, but fields are always in the volume, so this will always be 0
    unsigned coarseDomainOrdinalInRefinementRoot = 0; // again, since coarse domain is for a field variable, this will always be 0
    unsigned coarseSubcellPermutation = 0;            // not sure if this will ever be nontrivial permutation in practice; we don't test anything else here
    
    int H1Order = 1;
    int numPoints1D = 5;
    
    BasisPtr volumeBasis;
    if (fieldVar->rank() == 0)
      volumeBasis = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_HVOL);
    else
      volumeBasis = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_VECTOR_HVOL);
        
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
    noRefinements.push_back( make_pair(noRefinement.get(), 0) );
    
    RefinementBranch oneRefinement;
    RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
    oneRefinement.push_back( make_pair(regularRefinement.get(), 1) ); // 1: select child ordinal 1
    
    vector<RefinementBranch> refinementBranches;
    refinementBranches.push_back(noRefinements);
    refinementBranches.push_back(oneRefinement);
    
    FieldContainer<double> volumeRefNodes(volumeTopo->getVertexCount(), volumeTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(volumeRefNodes, volumeTopo);
    
    bool createSideCache = true;
    BasisCachePtr volumeBasisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache);
    
    for (int i=0; i< refinementBranches.size(); i++) {
      RefinementBranch refBranch = refinementBranches[i];
      
      out << "***** Refinement Type Number " << i << " *****\n";
      
      for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++) {
        FieldContainer<double> tracePointsSideReferenceSpace;
        CellTopoPtr sideTopo = volumeTopo->getSubcell(spaceDim-1, traceSideOrdinal);
        equispacedPoints(numPoints1D, sideTopo, tracePointsSideReferenceSpace);
        int numPoints = tracePointsSideReferenceSpace.dimension(0);
        
        BasisPtr traceBasis = BasisFactory::basisFactory()->getBasis(H1Order, sideTopo, Camellia::FUNCTION_SPACE_HVOL);

        out << "\n\n*****      Side Ordinal " << traceSideOrdinal << "      *****\n\n\n";
        
        BasisCachePtr traceBasisCache = volumeBasisCache->getSideBasisCache(traceSideOrdinal);
        traceBasisCache->setRefCellPoints(tracePointsSideReferenceSpace);
        
        //        out << "tracePointsSideReferenceSpace:\n" << tracePointsSideReferenceSpace;
        
        FieldContainer<double> tracePointsFineVolume(numPoints, volumeTopo->getDimension());
        
        CamelliaCellTools::mapToReferenceSubcell(tracePointsFineVolume, tracePointsSideReferenceSpace, sideTopo->getDimension(), traceSideOrdinal, volumeTopo);
        
        FieldContainer<double> pointsCoarseVolume(numPoints, volumeTopo->getDimension());
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, tracePointsFineVolume, pointsCoarseVolume);
        
        //        out << "pointsCoarseVolume:\n" << pointsCoarseVolume;
        
        volumeBasisCache->setRefCellPoints(pointsCoarseVolume);
        
        int oneCell = 1;
        FieldContainer<double> fakeParities(oneCell,volumeTopo->getSideCount());
        fakeParities.initialize(1.0);
        BasisCachePtr fakeSideVolumeCache = BasisCache::fakeSideCache(traceSideOrdinal, volumeBasisCache, pointsCoarseVolume,
                                                                      traceBasisCache->getSideNormals(), fakeParities);
        
        int fineSubcellOrdinalInFineDomain = 0; // since the side *is* both domain and subcell, it's necessarily ordinal 0 in the domain
        SubBasisReconciliationWeights weights;
        weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(traceVar->termTraced(), fieldVar->ID(), sideTopo->getDimension(),
                                                                              traceBasis, fineSubcellOrdinalInFineDomain, refBranch, traceSideOrdinal,
                                                                              volumeTopo->getDimension(), volumeBasis,
                                                                              coarseSubcellOrdinalInCoarseDomain,
                                                                              coarseDomainOrdinalInRefinementRoot,
                                                                              coarseSubcellPermutation);
        out << "weights:\n" << weights.weights;
        
        // fine basis is the line basis (the trace); coarse is the quad basis (the field)
        double tol = 1e-14; // for floating equality
        
        FieldContainer<double> coarseValuesExpected(oneCell,volumeBasis->getCardinality(),numPoints);
        traceVar->termTraced()->values(coarseValuesExpected, fieldVar->ID(), volumeBasis, fakeSideVolumeCache);
        
        out << "\ncoarseValuesExpected:\n" << coarseValuesExpected;
        
        FieldContainer<double> fineValues(oneCell,traceBasis->getCardinality(),numPoints);
        (1.0 * traceVar)->values(fineValues, traceVar->ID(), traceBasis, traceBasisCache);
        fineValues.resize(traceBasis->getCardinality(),numPoints); // strip cell dimension
        
        out << "fineValues:\n" << fineValues;
        
        FieldContainer<double> coarseValuesActual(weights.coarseOrdinals.size(), numPoints);
        SerialDenseWrapper::multiply(coarseValuesActual, weights.weights, fineValues, 'T', 'N');
        
        out << "coarseValuesActual:\n" << coarseValuesActual;
        
        for (int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++) {
          int coarseOrdinalInWeights = 0;
          for (int coarseOrdinal=0; coarseOrdinal < volumeBasis->getCardinality(); coarseOrdinal++) {
            double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);
            
            double actualValue;
            if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end()) {
              actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
              coarseOrdinalInWeights++;
            } else {
              actualValue = 0.0;
            }
            
            if ( abs(expectedValue) > tol ) {
              TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
            } else {
              TEST_ASSERT( abs(actualValue) < tol );
              
              if (abs(actualValue) >= tol) {
                out << "coarseOrdinal " << coarseOrdinal << ", point " << pointOrdinal << " on side " << traceSideOrdinal << ", actualValue = " << actualValue << endl;
              }
            }
          }
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_1D )
  {
    // TODO: rewrite this to use termTracedTest(), as in TermTraced_2D tests, below
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
  
  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D )
  {
    CellTopoPtr quadTopo = CellTopology::quad();
    termTracedTest(out,success,quadTopo,TRACE);
  }

  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D_Flux )
  {
    CellTopoPtr quadTopo = CellTopology::quad();
    termTracedTest(out,success,quadTopo,FLUX);
  }

  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_3D_Hexahedron)
  {
    // TODO: rewrite this to use termTracedTest(), as in TermTraced_2D tests, above
    // TODO: add Hexahedron flux test
    VarFactory vf;
    VarPtr u = vf.fieldVar("u");
    LinearTermPtr termTraced = 3.0 * u;
    VarPtr u_hat = vf.traceVar("\\widehat{u}", termTraced);
    
    // TODO: do flux tests...
    //    LinearTermPtr fluxTermTraced = 3.0 * u * n;
    //    VarPtr u_n = vf.traceVar("\\widehat{u}", termTraced);
    
    // in what follows, the fine basis belongs to the trace variable and the coarse to the field
    
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;  // when the coarse domain is a side, this can be non-zero, but fields are always in the volume, so this will always be 0
    unsigned coarseDomainOrdinalInRefinementRoot = 0; // again, since coarse domain is for a field variable, this will always be 0
    unsigned coarseSubcellPermutation = 0;            // not sure if this will ever be nontrivial permutation in practice; we don't test anything else here
    
    // 3D tests
    int H1Order = 2;
    CellTopoPtr volumeTopo = CellTopology::hexahedron();
    
    BasisPtr volumeBasisQuadratic = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    CellTopoPtr sideTopo = CellTopology::quad();
    BasisPtr traceBasis = BasisFactory::basisFactory()->getBasis(H1Order, sideTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
    noRefinements.push_back( make_pair(noRefinement.get(), 0) );
    
    RefinementBranch oneRefinement;
    RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
    oneRefinement.push_back( make_pair(regularRefinement.get(), 1) ); // 1: select child ordinal 1
    
    vector<RefinementBranch> refinementBranches;
    refinementBranches.push_back(noRefinements);
    refinementBranches.push_back(oneRefinement);
    
    FieldContainer<double> volumeRefNodes(volumeTopo->getVertexCount(), volumeTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(volumeRefNodes, volumeTopo);
    
    bool createSideCache = true;
    BasisCachePtr volumeBasisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache);
    
    for (int i=0; i< refinementBranches.size(); i++) {
      RefinementBranch refBranch = refinementBranches[i];
      
      out << "***** Refinement Type Number " << i << " *****\n";
      
      // compute some equispaced points on the reference quad:
      int numPoints_x = 5, numPoints_y = 5;
      FieldContainer<double> tracePointsSideReferenceSpace(numPoints_x * numPoints_y, sideTopo->getDimension());
      int pointOrdinal = 0;
      for (int pointOrdinal_x=0; pointOrdinal_x < numPoints_x; pointOrdinal_x++) {
        double x = -1.0 + pointOrdinal_x * (2.0 / (numPoints_x - 1));
        for (int pointOrdinal_y=0; pointOrdinal_y < numPoints_y; pointOrdinal_y++, pointOrdinal++) {
          double y = -1.0 + pointOrdinal_y * (2.0 / (numPoints_y - 1));
          tracePointsSideReferenceSpace(pointOrdinal,0) = x;
          tracePointsSideReferenceSpace(pointOrdinal,1) = y;
        }
      }
      
      int numPoints = numPoints_x * numPoints_y;
      
      for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++) {
        //      for (int traceSideOrdinal=1; traceSideOrdinal <= 1; traceSideOrdinal++) {
        out << "\n\n*****      Side Ordinal " << traceSideOrdinal << "      *****\n\n\n";
        
        BasisCachePtr traceBasisCache = volumeBasisCache->getSideBasisCache(traceSideOrdinal);
        traceBasisCache->setRefCellPoints(tracePointsSideReferenceSpace);
        
        //        out << "tracePointsSideReferenceSpace:\n" << tracePointsSideReferenceSpace;
        
        FieldContainer<double> tracePointsFineVolume(numPoints, volumeTopo->getDimension());
        
        CamelliaCellTools::mapToReferenceSubcell(tracePointsFineVolume, tracePointsSideReferenceSpace, sideTopo->getDimension(), traceSideOrdinal, volumeTopo);
        
        FieldContainer<double> pointsCoarseVolume(numPoints, volumeTopo->getDimension());
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, tracePointsFineVolume, pointsCoarseVolume);
        
        //        out << "pointsCoarseVolume:\n" << pointsCoarseVolume;
        
        volumeBasisCache->setRefCellPoints(pointsCoarseVolume);
        
        int fineSubcellOrdinalInFineDomain = 0; // since the side *is* both domain and subcell, it's necessarily ordinal 0 in the domain
        SubBasisReconciliationWeights weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, u->ID(), sideTopo->getDimension(),
                                                                                                            traceBasis, fineSubcellOrdinalInFineDomain, refBranch, traceSideOrdinal,
                                                                                                            volumeTopo->getDimension(), volumeBasisQuadratic,
                                                                                                            coarseSubcellOrdinalInCoarseDomain,
                                                                                                            coarseDomainOrdinalInRefinementRoot,
                                                                                                            coarseSubcellPermutation);
        // fine basis is the point basis (the trace); coarse is the line basis (the field)
        double tol = 1e-14; // for floating equality
        
        int oneCell = 1;
        FieldContainer<double> coarseValuesExpected(oneCell,volumeBasisQuadratic->getCardinality(),numPoints);
        termTraced->values(coarseValuesExpected, u->ID(), volumeBasisQuadratic, volumeBasisCache);
        
        //        out << "coarseValuesExpected:\n" << coarseValuesExpected;
        
        FieldContainer<double> fineValues(oneCell,traceBasis->getCardinality(),numPoints);
        (1.0 * u_hat)->values(fineValues, u_hat->ID(), traceBasis, traceBasisCache);
        fineValues.resize(traceBasis->getCardinality(),numPoints); // strip cell dimension
        
        //        out << "fineValues:\n" << fineValues;
        
        FieldContainer<double> coarseValuesActual(weights.coarseOrdinals.size(), numPoints);
        SerialDenseWrapper::multiply(coarseValuesActual, weights.weights, fineValues, 'T', 'N');
        
        //        out << "coarseValuesActual:\n" << coarseValuesActual;
        
        for (int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++) {
          int coarseOrdinalInWeights = 0;
          for (int coarseOrdinal=0; coarseOrdinal < volumeBasisQuadratic->getCardinality(); coarseOrdinal++) {
            double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);
            
            double actualValue;
            if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end()) {
              actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
              coarseOrdinalInWeights++;
            } else {
              actualValue = 0.0;
            }
            
            if ( abs(expectedValue) > tol ) {
              TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
            } else {
              TEST_ASSERT( abs(actualValue) < tol );
              
              if (abs(actualValue) >= tol) {
                out << "coarseOrdinal " << coarseOrdinal << ", point " << pointOrdinal << " on side " << traceSideOrdinal << ", actualValue = " << actualValue << endl;
              }
            }
          }
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_3D_Tetrahedron)
  {
    // TODO: rewrite this to use termTracedTest(), as in TermTraced_2D tests, above
    // TODO: add Tetrahedron flux test
    VarFactory vf;
    VarPtr u = vf.fieldVar("u");
    LinearTermPtr termTraced = 3.0 * u;
    VarPtr u_hat = vf.traceVar("\\widehat{u}", termTraced);
    
    // TODO: do flux tests...
    //    LinearTermPtr fluxTermTraced = 3.0 * u * n;
    //    VarPtr u_n = vf.traceVar("\\widehat{u}", termTraced);
    
    // in what follows, the fine basis belongs to the trace variable and the coarse to the field
    
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;  // when the coarse domain is a side, this can be non-zero, but fields are always in the volume, so this will always be 0
    unsigned coarseDomainOrdinalInRefinementRoot = 0; // again, since coarse domain is for a field variable, this will always be 0
    unsigned coarseSubcellPermutation = 0;            // not sure if this will ever be nontrivial permutation in practice; we don't test anything else here
    
    // 3D tests
    int H1Order = 2;
    CellTopoPtr volumeTopo = CellTopology::tetrahedron();
    
    BasisPtr volumeBasisQuadratic = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    CellTopoPtr sideTopo = CellTopology::triangle();
    BasisPtr traceBasis = BasisFactory::basisFactory()->getBasis(H1Order, sideTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    RefinementBranch noRefinements;
    RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
    noRefinements.push_back( make_pair(noRefinement.get(), 0) );

    // Once we have regular refinement patterns for tetrahedra, we can uncomment the following
//    RefinementBranch oneRefinement;
//    RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
//    oneRefinement.push_back( make_pair(regularRefinement.get(), 1) ); // 1: select child ordinal 1
    
    vector<RefinementBranch> refinementBranches;
    refinementBranches.push_back(noRefinements);
//    refinementBranches.push_back(oneRefinement); // wait until we have refinements for tets
    
    FieldContainer<double> volumeRefNodes(volumeTopo->getVertexCount(), volumeTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(volumeRefNodes, volumeTopo);
    
    bool createSideCache = true;
    BasisCachePtr volumeBasisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache);
    
    for (int i=0; i< refinementBranches.size(); i++) {
      RefinementBranch refBranch = refinementBranches[i];
      
      out << "***** Refinement Type Number " << i << " *****\n";
      
      // compute some equispaced points on the reference triangle:
      int numPoints_x = 5, numPoints_y = 5;
      FieldContainer<double> tracePointsSideReferenceSpace(numPoints_x * numPoints_y, sideTopo->getDimension());
      int pointOrdinal = 0;
      for (int pointOrdinal_x=0; pointOrdinal_x < numPoints_x; pointOrdinal_x++) {
        double x = 0.0 + pointOrdinal_x * (1.0 / (numPoints_x - 1));
        for (int pointOrdinal_y=0; pointOrdinal_y < numPoints_y; pointOrdinal_y++, pointOrdinal++) {
          double y = 0.0 + pointOrdinal_y * (1.0 / (numPoints_y - 1));
          
          // (x,y) lies in the unit quad.  Divide one coordinate by 2 to get into the ref. triangle...
          if ((pointOrdinal % 2) == 0) {
            tracePointsSideReferenceSpace(pointOrdinal,0) = x;
            tracePointsSideReferenceSpace(pointOrdinal,1) = y / 2.0;
          } else {
            tracePointsSideReferenceSpace(pointOrdinal,0) = x / 2.0;
            tracePointsSideReferenceSpace(pointOrdinal,1) = y;
          }
        }
      }
      
      int numPoints = numPoints_x * numPoints_y;
      
      for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++) {
        //      for (int traceSideOrdinal=1; traceSideOrdinal <= 1; traceSideOrdinal++) {
        out << "\n\n*****      Side Ordinal " << traceSideOrdinal << "      *****\n\n\n";
        
        BasisCachePtr traceBasisCache = volumeBasisCache->getSideBasisCache(traceSideOrdinal);
        traceBasisCache->setRefCellPoints(tracePointsSideReferenceSpace);
        
        //        out << "tracePointsSideReferenceSpace:\n" << tracePointsSideReferenceSpace;
        
        FieldContainer<double> tracePointsFineVolume(numPoints, volumeTopo->getDimension());
        
        CamelliaCellTools::mapToReferenceSubcell(tracePointsFineVolume, tracePointsSideReferenceSpace, sideTopo->getDimension(), traceSideOrdinal, volumeTopo);
        
        FieldContainer<double> pointsCoarseVolume(numPoints, volumeTopo->getDimension());
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, tracePointsFineVolume, pointsCoarseVolume);
        
        //        out << "pointsCoarseVolume:\n" << pointsCoarseVolume;
        
        volumeBasisCache->setRefCellPoints(pointsCoarseVolume);
        
        int fineSubcellOrdinalInFineDomain = 0; // since the side *is* both domain and subcell, it's necessarily ordinal 0 in the domain
        SubBasisReconciliationWeights weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, u->ID(), sideTopo->getDimension(),
                                                                                                            traceBasis, fineSubcellOrdinalInFineDomain, refBranch, traceSideOrdinal,
                                                                                                            volumeTopo->getDimension(), volumeBasisQuadratic,
                                                                                                            coarseSubcellOrdinalInCoarseDomain,
                                                                                                            coarseDomainOrdinalInRefinementRoot,
                                                                                                            coarseSubcellPermutation);
        // fine basis is the point basis (the trace); coarse is the line basis (the field)
        double tol = 1e-13; // for floating equality
        
        int oneCell = 1;
        FieldContainer<double> coarseValuesExpected(oneCell,volumeBasisQuadratic->getCardinality(),numPoints);
        termTraced->values(coarseValuesExpected, u->ID(), volumeBasisQuadratic, volumeBasisCache);
        
        //        out << "coarseValuesExpected:\n" << coarseValuesExpected;
        
        FieldContainer<double> fineValues(oneCell,traceBasis->getCardinality(),numPoints);
        (1.0 * u_hat)->values(fineValues, u_hat->ID(), traceBasis, traceBasisCache);
        fineValues.resize(traceBasis->getCardinality(),numPoints); // strip cell dimension
        
        //        out << "fineValues:\n" << fineValues;
        
        FieldContainer<double> coarseValuesActual(weights.coarseOrdinals.size(), numPoints);
        SerialDenseWrapper::multiply(coarseValuesActual, weights.weights, fineValues, 'T', 'N');
        
        //        out << "coarseValuesActual:\n" << coarseValuesActual;
        
        for (int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++) {
          int coarseOrdinalInWeights = 0;
          for (int coarseOrdinal=0; coarseOrdinal < volumeBasisQuadratic->getCardinality(); coarseOrdinal++) {
            double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);
            
            double actualValue;
            if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end()) {
              actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
              coarseOrdinalInWeights++;
            } else {
              actualValue = 0.0;
            }
            
            if ( abs(expectedValue) > tol ) {
              TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
            } else {
              TEST_ASSERT( abs(actualValue) < tol );
              
              if (abs(actualValue) >= tol) {
                out << "coarseOrdinal " << coarseOrdinal << ", point " << pointOrdinal << " on side " << traceSideOrdinal << ", actualValue = " << actualValue << endl;
              }
            }
          }
        }
      }
    }
  }
} // namespace