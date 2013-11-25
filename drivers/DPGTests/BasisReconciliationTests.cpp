//
//  BasisReconciliationTests.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#include "BasisReconciliationTests.h"

#include "CamelliaCellTools.h"
#include "doubleBasisConstruction.h"

#include "SerialDenseWrapper.h"

void BasisReconciliationTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testP()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testH()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

void BasisReconciliationTests::setup() {
  
}
void BasisReconciliationTests::teardown() {
  
}

bool BasisReconciliationTests::testH() {
  bool success = true;
  
  cout << "WARNING: BasisReconciliationTests::testH() not yet implemented.\n";
  
  return success;
}

FieldContainer<double> interpretValues(FieldContainer<double> &fineValues, FieldContainer<double> &weights) {
  int fineCount = weights.dimension(0);   // number of fine ordinals
  int coarseCount = weights.dimension(1); // number of coarse ordinals
  Teuchos::Array<int> dim;
  fineValues.dimensions(dim);
  dim[0] = coarseCount; // the shape we want for the final container is now in dim
  int valueCount = fineValues.size() / fineCount;
  FieldContainer<double> result(coarseCount, valueCount);
  fineValues.resize(fineCount,valueCount);
  SerialDenseWrapper::multiply(result, weights, fineValues, 'T', 'N');
  result.resize(dim);
  dim[0] = fineCount;
  fineValues.resize(dim);
  return result;
}

void interpretSideValues(SubBasisReconciliationWeights weights, const FieldContainer<double> &fineValues, const FieldContainer<double> &coarseValues,
                         FieldContainer<double> &interpretedFineValues, FieldContainer<double> &filteredCoarseValues) {
  int fineCount = weights.fineOrdinals.size();
  int coarseCount = weights.coarseOrdinals.size();

  // resize output containers appropriately
  Teuchos::Array<int> dim;
  fineValues.dimensions(dim);
  dim[0] = coarseCount; // the shape we want for the final container is now in dim
  filteredCoarseValues.resize(dim);
  dim[0] = fineCount;
  interpretedFineValues.resize(dim);
  
  
}

FieldContainer<double> cubaturePoints(shards::CellTopology cellTopo, int cubatureDegree, unsigned nodePermutation) {
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(cellTopo, cubatureDegree, false) );
  FieldContainer<double> permutedCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedCellNodes, cellTopo, nodePermutation);
  permutedCellNodes.resize(1, permutedCellNodes.dimension(0), permutedCellNodes.dimension(1));
  basisCache->setPhysicalCellNodes(permutedCellNodes, vector<int>(), false);
  FieldContainer<double> permutedCubaturePoints = basisCache->getPhysicalCubaturePoints();
  permutedCubaturePoints.resize(permutedCubaturePoints.dimension(1), permutedCubaturePoints.dimension(2));
  return permutedCubaturePoints;
}

FieldContainer<double> basisValuesAtPoints(BasisPtr basis, const FieldContainer<double> &pointsOnRefCell) {
  shards::CellTopology cellTopo = basis->domainTopology();
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(cellTopo, basis->getDegree(), false) );
  FieldContainer<double> refCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
  basisCache->setRefCellPoints(pointsOnRefCell);
  refCellNodes.resize(1,cellTopo.getNodeCount(),cellTopo.getDimension());
  basisCache->setPhysicalCellNodes(refCellNodes, vector<int>(), false);
  return *(basisCache->getValues(basis, OP_VALUE).get());
}

FieldContainer<double> basisValuesAtSidePoints(BasisPtr basis, unsigned sideIndex, const FieldContainer<double> &pointsOnSideRefCell) {
  shards::CellTopology cellTopo = basis->domainTopology();
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(cellTopo, basis->getDegree(), true) );
  basisCache->getSideBasisCache(sideIndex)->setRefCellPoints(pointsOnSideRefCell);
  
  return *(basisCache->getSideBasisCache(sideIndex)->getValues(basis, OP_VALUE, true).get());
}

bool BasisReconciliationTests::pConstraintWholeBasisSubTest(BasisPtr fineBasis, BasisPtr coarseBasis) {
  bool success = true;
  
  // first question: does BasisReconciliation run to completion?
  BasisReconciliation br;
  FieldContainer<double> weights = br.constrainedWeights(fineBasis, coarseBasis);
  
  //  cout << "BasisReconciliation: computed weights when matching whole bases.\n";
  
  FieldContainer<double> points = cubaturePoints(fineBasis->domainTopology(), 5, 0);
  FieldContainer<double> fineBasisValues = basisValuesAtPoints(fineBasis, points);
  FieldContainer<double> coarseBasisValues = basisValuesAtPoints(coarseBasis, points);
  
  FieldContainer<double> interpretedFineBasisValues = interpretValues(fineBasisValues, weights);
  
  double maxDiff;
  double tol = 1e-14;
  if ( !fcsAgree(coarseBasisValues, interpretedFineBasisValues, tol, maxDiff) ) {
    success = false;
    cout << "FAILURE: BasisReconciliation's interpreted fine basis values do not match coarse values on quad.\n";
    cout << "points:\n" << points;
    cout << "weights:\n" << weights;
    cout << "fineValues:\n" << fineBasisValues;
    cout << "coarseBasisValues:\n" << coarseBasisValues;
    cout << "interpretedFineBasisValues:\n" << interpretedFineBasisValues;
  }
  return success;
}

//unsigned matchTopologies(FieldContainer<double> &topo1Nodes, shards::CellTopology &topo1, unsigned topo1SideIndex,
//                         FieldContainer<double> &topo2Nodes, shards::CellTopology &topo2, unsigned topo2SideIndex) {
//  switch (topo1.getKey()) {
//    case shards::Line<2>::key:
//      switch (topo2.getKey()) {
//          case shards::Line<2>::key:
//          
//          break;
//      }
//      break;
//  }
//}

unsigned vertexPermutation(shards::CellTopology &fineTopo, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                           shards::CellTopology &coarseTopo, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes) {
  int d = fineTopo.getDimension();
  shards::CellTopology sideTopo = fineTopo.getBaseCellTopologyData(d-1, fineSideIndex);

  // a brute force search for a matching permutation going from fineTopo's view of the side to coarseTopo's
  int permutationCount = sideTopo.getNodePermutationCount();
  double tol = 1e-14;
  for (unsigned permutation=0; permutation<permutationCount; permutation++) {
    bool matches = true;
    for (unsigned sideNode=0; sideNode<sideTopo.getNodeCount(); sideNode++) {
      unsigned fineNode = fineTopo.getNodeMap(d-1, fineSideIndex, sideNode);
      unsigned permutedSideNode = sideTopo.getNodePermutation(permutation, sideNode);
      unsigned putativeCoarseNode = coarseTopo.getNodeMap(d-1, coarseSideIndex, permutedSideNode);
      for (int dim=0; dim<d; dim++) {
        if (abs(fineCellNodes(fineNode,dim) - coarseCellNodes(putativeCoarseNode,dim)) > tol) {
          // not a match
          matches = false;
          break;
        }
      }
      if (matches == false) break;
    }
    if (matches) {
      return permutation;
    }
  }
  cout << "Matching permutation not found.\n";
  cout << "fine side index: " << fineSideIndex << endl;
  cout << "fine nodes:\n" << fineCellNodes;
  cout << "coarse side index: " << coarseSideIndex << endl;
  cout << "coarse nodes:\n" << coarseCellNodes;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matching permutation not found");
}

FieldContainer<double> permutedSidePoints(shards::CellTopology &sideTopo, FieldContainer<double> &pointsRefCell, unsigned permutation) {
  FieldContainer<double> sideNodes (sideTopo.getNodeCount(), sideTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(sideNodes, sideTopo, permutation);
  
  unsigned cubDegree = 1;
  unsigned oneCell = 1;
  sideNodes.resize(oneCell,sideNodes.dimension(0),sideNodes.dimension(1));
  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideTopo, cubDegree, false) );
  sideCache->setRefCellPoints(pointsRefCell);
  sideCache->setPhysicalCellNodes(sideNodes, vector<int>(), false);
  
  FieldContainer<double> permutedPoints = sideCache->getPhysicalCubaturePoints();
  permutedPoints.resize(permutedPoints.dimension(1), permutedPoints.dimension(2));
  return permutedPoints;
}

bool BasisReconciliationTests::pConstraintSideBasisSubTest(BasisPtr fineBasis, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                                                           BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes) {
  bool success = true;
  
  BasisReconciliation br;

  int d = fineBasis->domainTopology().getDimension();
  
  // want to figure out a set of physical cell nodes that corresponds to this combination
  shards::CellTopology coarseTopo = coarseBasis->domainTopology();
  shards::CellTopology fineTopo = fineBasis->domainTopology();
  
  shards::CellTopology sideTopo = fineTopo.getBaseCellTopologyData(d-1, fineSideIndex);
  
  unsigned permutation = vertexPermutation(fineTopo, fineSideIndex, fineCellNodes, coarseTopo, coarseSideIndex, coarseCellNodes);

  SubBasisReconciliationWeights weights = br.constrainedWeights(fineBasis, fineSideIndex, coarseBasis, coarseSideIndex, permutation);
  
  FieldContainer<double> sidePointsFine = cubaturePoints(sideTopo, 5, 0);
  FieldContainer<double> sidePointsCoarse = permutedSidePoints(sideTopo,sidePointsFine,permutation);
  
  int cubDegree = fineBasis->getDegree() * 2;
  BasisCachePtr fineCache = Teuchos::rcp( new BasisCache(fineCellNodes, fineTopo, cubDegree, true) );
  BasisCachePtr coarseCache = Teuchos::rcp( new BasisCache(coarseCellNodes, coarseTopo, cubDegree, true) );
  
  fineCache->getSideBasisCache(fineSideIndex)->setRefCellPoints(sidePointsFine);
  coarseCache->getSideBasisCache(coarseSideIndex)->setRefCellPoints(sidePointsCoarse);
  
  FieldContainer<double> finePoints = fineCache->getSideBasisCache(fineSideIndex)->getSideRefCellPointsInVolumeCoordinates();
  FieldContainer<double> coarsePoints = coarseCache->getSideBasisCache(coarseSideIndex)->getSideRefCellPointsInVolumeCoordinates();
  
  // finePoints and coarsePoints should match
  double maxDiff;
  double tol = 1e-14;
  if ( !fcsAgree(finePoints, coarsePoints, tol, maxDiff) ) {
    cout << "TEST failure in pConstraintSideBasisSubTest: points on fine and coarse topology do not match on side.\n";
    success = false;
    return success;
  }
  
  FieldContainer<double> fineBasisValues = basisValuesAtPoints(fineBasis, finePoints);
  FieldContainer<double> coarseBasisValues = basisValuesAtPoints(coarseBasis, coarsePoints);
  
//  FieldContainer<double> interpretedFineBasisValues = interpretValues(fineBasisValues, weights);
//  
//  double maxDiff;
//  double tol = 1e-14;
//  if ( !fcsAgree(coarseBasisValues, interpretedFineBasisValues, tol, maxDiff) ) {
//    success = false;
//    cout << "FAILURE: BasisReconciliation's interpreted fine basis values do not match coarse values on quad.\n";
//    cout << "points:\n" << points;
//    cout << "weights:\n" << weights.weights;
//    cout << "fineValues:\n" << fineBasisValues;
//    cout << "coarseBasisValues:\n" << coarseBasisValues;
//    cout << "interpretedFineBasisValues:\n" << interpretedFineBasisValues;
//  }
  return success;
}

bool BasisReconciliationTests::testP() {
  bool success = true;
  
  int fineOrder = 5;
  int coarseOrder = 3;
  
  vector< pair< BasisPtr, BasisPtr > > basisPairsToCheck;
  
  BasisPtr fineBasis = Camellia::intrepidLineHGRAD(fineOrder);
  BasisPtr coarseBasis = Camellia::intrepidLineHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );
  
  fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );
  
  fineBasis = Camellia::intrepidQuadHDIV(fineOrder);
  coarseBasis = Camellia::intrepidQuadHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );
  
  fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );
  
  fineBasis = Camellia::intrepidHexHDIV(fineOrder);
  coarseBasis = Camellia::intrepidHexHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );
  
  for (vector< pair< BasisPtr, BasisPtr > >::iterator bpIt = basisPairsToCheck.begin(); bpIt != basisPairsToCheck.end(); bpIt++) {
    fineBasis = bpIt->first;
    coarseBasis = bpIt->second;
    if (! pConstraintWholeBasisSubTest(fineBasis, coarseBasis)) {
      success = false;
    }
  }
  
  // try it with sides
  BasisReconciliation br;
  SubBasisReconciliationWeights sideWeights = br.constrainedWeights(fineBasis, 0, coarseBasis, 0, 0);
  
  cout << "sideWeights: \n" << sideWeights.weights;
  
  cout << "BasisReconciliation: computed weights when matching sides with the identity permutation.\n";
  cout << "(still need to check that the weights are correct!)\n";
  
  return success;
}