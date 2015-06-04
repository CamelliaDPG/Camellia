//
//  BasisReconciliationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/22/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisReconciliation.h"
#include "CamelliaCellTools.h"
#include "CamelliaTestingHelpers.h"
#include "CellTopology.h"
#include "doubleBasisConstruction.h"
#include "LinearTerm.h"
#include "MeshFactory.h"
#include "SerialDenseWrapper.h"
#include "Var.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
// copied from DPGTests's BasisReconciliationTests
struct hSideTest
{
  BasisPtr fineBasis;
  int fineSideIndex; // ancestral side index
  FieldContainer<double> fineCellNodes; // these are actually the ancestral nodes
  BasisPtr coarseBasis;
  int coarseSideIndex;
  FieldContainer<double> coarseCellNodes;

  RefinementBranch volumeRefinements; // how the ancestor topology of fineCellNodes has been refined to get to the descendant of interest...
};

// copied from DPGTests's BasisReconciliationTests
struct pSideTest
{
  BasisPtr fineBasis;
  int fineSideIndex;
  FieldContainer<double> fineCellNodes;
  BasisPtr coarseBasis;
  int coarseSideIndex;
  FieldContainer<double> coarseCellNodes;
};

// copied from DPGTests's BasisReconciliationTests
void addCellDimensionToFC(FieldContainer<double> &fc)
{
  Teuchos::Array<int> dim;
  int oneCell = 1;
  fc.dimensions(dim);
  dim.insert(dim.begin(), oneCell);
  fc.resize(dim);
}

// copied from DPGTests's BasisReconciliationTests
void stripCellDimensionFromFC(FieldContainer<double> &fc)
{
  Teuchos::Array<int> dim;
  int oneCell = 1;
  fc.dimensions(dim);
  if (dim[0] != oneCell)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "'dummy' cell dimension isn't unit-sized!");
  }
  dim.remove(0);
  fc.resize(dim);
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> cubaturePoints(shards::CellTopology cellTopo, int cubatureDegree, unsigned nodePermutation)
{
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(cellTopo, cubatureDegree, false) );
  FieldContainer<double> permutedCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedCellNodes, cellTopo, nodePermutation);
  permutedCellNodes.resize(1, permutedCellNodes.dimension(0), permutedCellNodes.dimension(1));
  basisCache->setPhysicalCellNodes(permutedCellNodes, vector<GlobalIndexType>(), false);
  FieldContainer<double> permutedCubaturePoints = basisCache->getPhysicalCubaturePoints();
  permutedCubaturePoints.resize(permutedCubaturePoints.dimension(1), permutedCubaturePoints.dimension(2));
  return permutedCubaturePoints;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> cubaturePoints(shards::CellTopology fineCellTopo, int cubatureDegree,
                                      shards::CellTopology coarseCellTopo, unsigned nodePermutation, RefinementBranch &refinements)
{
  BasisCachePtr fineBasisCache = Teuchos::rcp( new BasisCache(fineCellTopo, cubatureDegree, false) );
  FieldContainer<double> coarseCellNodes(coarseCellTopo.getNodeCount(),coarseCellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseCellNodes, coarseCellTopo, nodePermutation);

  FieldContainer<double> fineCellNodesInCoarseCell = RefinementPattern::descendantNodes(refinements,coarseCellNodes);
  fineCellNodesInCoarseCell.resize(1,fineCellNodesInCoarseCell.dimension(0),fineCellNodesInCoarseCell.dimension(1));

  fineBasisCache->setPhysicalCellNodes(fineCellNodesInCoarseCell, vector<GlobalIndexType>(), false);
  FieldContainer<double> fineCubPoints = fineBasisCache->getPhysicalCubaturePoints();
  fineCubPoints.resize(fineCubPoints.dimension(1), fineCubPoints.dimension(2));

  return fineCubPoints;
}

// copied from DPGTests's BasisReconciliationTests
RefinementBranch demoRefinementsOnSide(shards::CellTopology cellTopo, unsigned sideIndex, int numRefinements)
{
  RefinementBranch demoRefinements;
  for (int refIndex = 0; refIndex < numRefinements; refIndex++)
  {
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo.getKey());
    vector< pair<unsigned, unsigned> > childrenForSide = refPattern->childrenForSides()[sideIndex];
    // pick one randomly
    unsigned indexOfChildInSide = rand() % childrenForSide.size();
    unsigned childIndex = childrenForSide[indexOfChildInSide].first;

    sideIndex = childrenForSide[indexOfChildInSide].second; // the side child shares with parent...
    demoRefinements.push_back(make_pair(refPattern.get(),childIndex));
  }
  return demoRefinements;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> filterValues(FieldContainer<double> &basisValues, set<unsigned> &dofOrdinalFilter, bool includesCellDimension)
{
  Teuchos::Array<int> dim;
  basisValues.dimensions(dim); // dimensions are ordered (C,F,P[,D,…]) if there is a cell dimension, (F,P[,D,…]) otherwise
  int fieldDimOrdinal = includesCellDimension ? 1 : 0;
  dim[fieldDimOrdinal] = dofOrdinalFilter.size();
  FieldContainer<double> filteredValues(dim);
  if (dofOrdinalFilter.size() == 0) return filteredValues; // empty container
  int valuesPerField = filteredValues.size() / dofOrdinalFilter.size();
  double *filteredValue = &filteredValues[0];
  for (set<unsigned>::iterator dofOrdinalIt=dofOrdinalFilter.begin(); dofOrdinalIt != dofOrdinalFilter.end(); dofOrdinalIt++)
  {
    unsigned dofOrdinal = *dofOrdinalIt;
    for (int i=0; i<valuesPerField; i++)
    {
      *filteredValue = basisValues[dofOrdinal * valuesPerField + i];
      filteredValue++;
    }
  }
  return filteredValues;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> interpretValues(FieldContainer<double> &fineValues, FieldContainer<double> &weights)
{
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

// copied from DPGTests's BasisReconciliationTests
void interpretSideValues(SubBasisReconciliationWeights weights, const FieldContainer<double> &fineValues, const FieldContainer<double> &coarseValues,
                         FieldContainer<double> &interpretedFineValues, FieldContainer<double> &filteredCoarseValues)
{
  int fineCount = weights.fineOrdinals.size();
  int coarseCount = weights.coarseOrdinals.size();

  // resize output containers appropriately
  Teuchos::Array<int> dim;
  fineValues.dimensions(dim);
  dim[0] = coarseCount; // the shape we want for the final container is now in dim
  filteredCoarseValues.resize(dim);
  interpretedFineValues.resize(dim);

  dim[0] = fineCount;
  FieldContainer<double> filteredFineValues(dim);
  int numValuesPerBasisFunction = filteredFineValues.size() / fineCount;
  int filteredIndex = 0;
  for (set<int>::iterator fineIt = weights.fineOrdinals.begin(); fineIt != weights.fineOrdinals.end(); fineIt++)
  {
    int fineIndex = *fineIt;
    const double *fineValue = &fineValues[fineIndex * numValuesPerBasisFunction];
    double * filteredFineValue = &filteredFineValues[filteredIndex * numValuesPerBasisFunction];
    for (int i=0; i<numValuesPerBasisFunction; i++)
    {
      *filteredFineValue++ = *fineValue++;
    }
    filteredIndex++;
  }

  interpretedFineValues = interpretValues(filteredFineValues, weights.weights);

  filteredIndex = 0;
  for (set<int>::iterator coarseIt = weights.coarseOrdinals.begin(); coarseIt != weights.coarseOrdinals.end(); coarseIt++)
  {
    int coarseIndex = *coarseIt;
    const double *coarseValue = &coarseValues[coarseIndex * numValuesPerBasisFunction];
    double * filteredCoarseValue = &filteredCoarseValues[filteredIndex * numValuesPerBasisFunction];
    for (int i=0; i<numValuesPerBasisFunction; i++)
    {
      *filteredCoarseValue++ = *coarseValue++;
    }
    filteredIndex++;
  }

}

// copied from DPGTests's BasisReconciliationTests
RefinementBranch makeRefinementBranch( RefinementPatternPtr refPattern, vector<unsigned> childChoices )
{
  // make branch with the same refinement pattern all the way down...
  RefinementBranch branch;
  for (vector<unsigned>::iterator childIndexIt = childChoices.begin(); childIndexIt != childChoices.end(); childIndexIt++)
  {
    branch.push_back( make_pair(refPattern.get(), *childIndexIt));
  }
  return branch;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> permutedSidePoints(shards::CellTopology &sideTopo, FieldContainer<double> &pointsRefCell, unsigned permutation)
{
  FieldContainer<double> sideNodes (sideTopo.getNodeCount(), sideTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(sideNodes, sideTopo, permutation);

  unsigned cubDegree = 1;
  unsigned oneCell = 1;
  sideNodes.resize(oneCell,sideNodes.dimension(0),sideNodes.dimension(1));
  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideTopo, cubDegree, false) );
  sideCache->setRefCellPoints(pointsRefCell);
  sideCache->setPhysicalCellNodes(sideNodes, vector<GlobalIndexType>(), false);

  FieldContainer<double> permutedPoints = sideCache->getPhysicalCubaturePoints();
  permutedPoints.resize(permutedPoints.dimension(1), permutedPoints.dimension(2));
  return permutedPoints;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> translateQuad(const FieldContainer<double> &quad, double x, double y)
{
  vector<double> translations;
  translations.push_back(x);
  translations.push_back(y);
  FieldContainer<double> translatedQuad = quad;
  for (int node=0; node < translatedQuad.dimension(0); node++)
  {
    for (int d = 0; d < translatedQuad.dimension(1); d++)
    {
      translatedQuad(node,d) += translations[d];
    }
  }
  return translatedQuad;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> translateHex(const FieldContainer<double> &hex, double x, double y, double z)
{
  vector<double> translations;
  translations.push_back(x);
  translations.push_back(y);
  translations.push_back(z);
  FieldContainer<double> translatedHex = hex;
  for (int node=0; node < translatedHex.dimension(0); node++)
  {
    for (int d = 0; d < translatedHex.dimension(1); d++)
    {
      translatedHex(node,d) += translations[d];
    }
  }
  return translatedHex;
}

// copied from DPGTests's BasisReconciliationTests
FieldContainer<double> transformedBasisValuesAtPoints(BasisPtr basis, const FieldContainer<double> &pointsOnFineCell, RefinementBranch refinements)
{
  CellTopoPtr cellTopo = basis->domainTopology();
  FieldContainer<double> refCellNodes;
  refCellNodes.resize(cellTopo->getNodeCount(),cellTopo->getDimension());
  if (refinements.size() == 0)
  {
    CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
  }
  else
  {
    CellTopoPtr coarseCellTopo = refinements[0].first->parentTopology();
    FieldContainer<double> coarseCellNodes(coarseCellTopo->getNodeCount(),coarseCellTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(coarseCellNodes, coarseCellTopo);
    refCellNodes = RefinementPattern::descendantNodes(refinements,coarseCellNodes);
  }
  refCellNodes.resize(1,cellTopo->getNodeCount(),cellTopo->getDimension());

  int dummyCubatureDegree = 1;
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(cellTopo, dummyCubatureDegree, false) );
  basisCache->setRefCellPoints(pointsOnFineCell);
  basisCache->setPhysicalCellNodes(refCellNodes, vector<GlobalIndexType>(), false);
  return *(basisCache->getTransformedValues(basis, OP_VALUE).get());
}

// copied from DPGTests's BasisReconciliationTests
RefinementBranch determineSideRefinements(RefinementBranch volumeRefinements, unsigned sideIndex)
{
  RefinementBranch sideRefinements;
  CellTopoPtr volumeTopo = volumeRefinements[0].first->parentTopology();
  unsigned sideDim = volumeTopo->getDimension() - 1;
  for (int refIndex=0; refIndex<volumeRefinements.size(); refIndex++)
  {
    RefinementPattern* refPattern = volumeRefinements[refIndex].first;
    unsigned volumeBranchChild = volumeRefinements[refIndex].second;
    RefinementPattern* sideRefPattern = refPattern->patternForSubcell(sideDim, sideIndex).get();

    int sideBranchChild = -1;
    for (int sideChildIndex = 0; sideChildIndex < sideRefPattern->numChildren(); sideChildIndex++)
    {
      if (refPattern->mapSideChildIndex(sideIndex, sideChildIndex) == volumeBranchChild)
      {
        sideBranchChild = sideChildIndex;
      }
    }
    if (sideBranchChild == -1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Did not find child");
    }

    sideRefinements.push_back(make_pair(sideRefPattern,sideBranchChild));
  }
  return sideRefinements;
}

void testHConstraintInternalBasis(BasisPtr fineBasis, RefinementBranch &refinements, BasisPtr coarseBasis, Teuchos::FancyOStream &out, bool &success)
{
  // copied from DPGTests's BasisReconciliationTests
  BasisReconciliation br;
  unsigned permutation = 0; // for now, just assuming the identity permutation.  TODO: try other permutations.
  SubBasisReconciliationWeights weights = br.constrainedWeights(fineBasis, refinements, coarseBasis, permutation);

  //  cout << "BasisReconciliation: computed weights when matching whole bases.\n";

  if ( (coarseBasis->domainTopology()->getTensorialDegree() > 0) || (fineBasis->domainTopology()->getTensorialDegree() > 0) )
  {
    out << "ERROR: hConstraintInternalBasisSubTest() does not support tensorial degree > 0.\n";
    success = false;
    return;
  }

  FieldContainer<double> finePoints = cubaturePoints(fineBasis->domainTopology()->getShardsTopology(), 1, 0);
  FieldContainer<double> coarsePoints = cubaturePoints(fineBasis->domainTopology()->getShardsTopology(), 1, coarseBasis->domainTopology()->getShardsTopology(), 0, refinements);

  FieldContainer<double> fineBasisValues = transformedBasisValuesAtPoints(fineBasis, finePoints, refinements);
  RefinementBranch noRefinements;
  FieldContainer<double> coarseBasisValues = transformedBasisValuesAtPoints(coarseBasis, coarsePoints, noRefinements);

  stripCellDimensionFromFC(fineBasisValues);
  stripCellDimensionFromFC(coarseBasisValues);

  FieldContainer<double> interpretedFineBasisValues, filteredCoarseValues;
  interpretSideValues(weights, fineBasisValues, coarseBasisValues, interpretedFineBasisValues, filteredCoarseValues);

  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(filteredCoarseValues, interpretedFineBasisValues, tol);
}

// copied from DPGTests's BasisReconciliationTests
unsigned vertexPermutation(shards::CellTopology &fineTopo, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                           shards::CellTopology &coarseTopo, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes)
{
  int d = fineTopo.getDimension();
  shards::CellTopology sideTopo = fineTopo.getBaseCellTopologyData(d-1, fineSideIndex);

  // a brute force search for a matching permutation going from fineTopo's view of the side to coarseTopo's
  int permutationCount = sideTopo.getNodePermutationCount();
  double tol = 1e-14;
  for (unsigned permutation=0; permutation<permutationCount; permutation++)
  {
    bool matches = true;
    for (unsigned sideNode=0; sideNode<sideTopo.getNodeCount(); sideNode++)
    {
      unsigned fineNode = fineTopo.getNodeMap(d-1, fineSideIndex, sideNode);
      unsigned permutedSideNode = sideTopo.getNodePermutation(permutation, sideNode);
      unsigned putativeCoarseNode = coarseTopo.getNodeMap(d-1, coarseSideIndex, permutedSideNode);
      for (int dim=0; dim<d; dim++)
      {
        if (abs(fineCellNodes(0,fineNode,dim) - coarseCellNodes(0,putativeCoarseNode,dim)) > tol)
        {
          // not a match
          matches = false;
          break;
        }
      }
      if (matches == false) break;
    }
    if (matches)
    {
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

void testHConstraintSideBasis(BasisPtr fineBasis, unsigned fineAncestralSideIndex, FieldContainer<double> &fineCellAncestralNodes,
                              RefinementBranch &volumeRefinements,
                              BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes,
                              Teuchos::FancyOStream &out, bool &success)
{
  // copied from DPGTests's BasisReconciliationTests

  BasisReconciliation br;

  RefinementBranch sideRefinements = determineSideRefinements(volumeRefinements, fineAncestralSideIndex);

  FieldContainer<double> fineCellNodes = RefinementPattern::descendantNodes(volumeRefinements, fineCellAncestralNodes);

  int spaceDim = fineBasis->domainTopology()->getDimension();
  int sideDim = spaceDim - 1;

  // want to figure out a set of physical cell nodes that corresponds to this combination
  CellTopoPtr coarseTopo = coarseBasis->domainTopology();
  CellTopoPtr fineTopo = fineBasis->domainTopology();

  if ( (coarseTopo->getTensorialDegree() > 0) || (fineTopo->getTensorialDegree() > 0) )
  {
    out << "ERROR: hConstraintSideBasisSubTest() does not support tensorial degree > 0.\n";
    success = false;
  }

  CellTopoPtr ancestralTopo = volumeRefinements[0].first->parentTopology();

  // figure out fineSideIndex
  unsigned fineSideIndex = fineAncestralSideIndex;
  for (int refIndex=0; refIndex<volumeRefinements.size(); refIndex++)
  {
    RefinementPattern* refPattern = volumeRefinements[refIndex].first;
    unsigned childIndex = volumeRefinements[refIndex].second;
    vector< pair<unsigned, unsigned> > childrenForSide = refPattern->childrenForSides()[fineSideIndex];
    for (vector< pair<unsigned, unsigned> >::iterator entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++)
    {
      if (entryIt->first == childIndex)
      {
        fineSideIndex = entryIt->second;
      }
    }
  }

  CellTopoPtr fineSideTopo = fineTopo->getSubcell(sideDim, fineSideIndex);
  CellTopoPtr coarseSideTopo = coarseTopo->getSubcell(sideDim, coarseSideIndex);

  int oneCell = 1;
  fineCellAncestralNodes.resize(oneCell, fineCellAncestralNodes.dimension(0), fineCellAncestralNodes.dimension(1));
  fineCellNodes.resize(oneCell, fineCellNodes.dimension(0), fineCellNodes.dimension(1));
  coarseCellNodes.resize(oneCell, coarseCellNodes.dimension(0), coarseCellNodes.dimension(1));

  shards::CellTopology shardsAncestralTopo = ancestralTopo->getShardsTopology();
  shards::CellTopology shardsCoarseTopo = coarseTopo->getShardsTopology();
  unsigned permutation = vertexPermutation(shardsAncestralTopo, fineAncestralSideIndex, fineCellAncestralNodes, shardsCoarseTopo, coarseSideIndex, coarseCellNodes);

  SubBasisReconciliationWeights weights = br.constrainedWeights(sideDim, fineBasis, fineSideIndex, volumeRefinements, coarseBasis, coarseSideIndex, permutation);

  shards::CellTopology shardsFineSideTopo = fineSideTopo->getShardsTopology();
  shards::CellTopology shardsCoarseSideTopo = coarseSideTopo->getShardsTopology();

  int cubDegree = 1;
  FieldContainer<double> sidePointsFine = cubaturePoints(shardsFineSideTopo, cubDegree, 0);
  FieldContainer<double> sidePointsCoarse = cubaturePoints(shardsFineSideTopo, cubDegree, shardsCoarseSideTopo, permutation, sideRefinements);

  cubDegree = fineBasis->getDegree() * 2;

  shards::CellTopology shardsFineTopo = fineTopo->getShardsTopology();

  BasisCachePtr fineCache = Teuchos::rcp( new BasisCache(fineCellNodes, shardsFineTopo, cubDegree, true) );
  BasisCachePtr coarseCache = Teuchos::rcp( new BasisCache(coarseCellNodes, shardsCoarseTopo, cubDegree, true) );

  fineCache->getSideBasisCache(fineSideIndex)->setRefCellPoints(sidePointsFine);
  coarseCache->getSideBasisCache(coarseSideIndex)->setRefCellPoints(sidePointsCoarse);

  FieldContainer<double> finePointsPhysical = fineCache->getSideBasisCache(fineSideIndex)->getPhysicalCubaturePoints();
  FieldContainer<double> coarsePointsPhysical = coarseCache->getSideBasisCache(coarseSideIndex)->getPhysicalCubaturePoints();

  FieldContainer<double> finePoints = fineCache->getSideBasisCache(fineSideIndex)->getSideRefCellPointsInVolumeCoordinates();
  FieldContainer<double> coarsePoints = coarseCache->getSideBasisCache(coarseSideIndex)->getSideRefCellPointsInVolumeCoordinates();

  // finePoints and coarsePoints should match
  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(finePointsPhysical, coarsePointsPhysical, tol);

  bool useVolumeCoordinates = true;
  FieldContainer<double> fineBasisValues = *fineCache->getSideBasisCache(fineSideIndex)->getTransformedValues(fineBasis, OP_VALUE, useVolumeCoordinates);
  FieldContainer<double> coarseBasisValues = *coarseCache->getSideBasisCache(coarseSideIndex)->getTransformedValues(coarseBasis, OP_VALUE, useVolumeCoordinates); // basisValuesAtPoints(coarseBasis, coarsePoints);
  stripCellDimensionFromFC(fineBasisValues);
  stripCellDimensionFromFC(coarseBasisValues);

  FieldContainer<double> interpretedFineBasisValues, filteredCoarseBasisValues;
  interpretSideValues(weights, fineBasisValues, coarseBasisValues, interpretedFineBasisValues, filteredCoarseBasisValues);

  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(filteredCoarseBasisValues, interpretedFineBasisValues, tol);
}

void testPConstraintInternalBasis(BasisPtr fineBasis, BasisPtr coarseBasis, Teuchos::FancyOStream &out, bool &success)
{
  // copied from DPGTests's BasisReconciliationTests

  BasisReconciliation br;
  unsigned shardsPermutationCount = fineBasis->domainTopology()->getShardsTopology().getNodePermutationCount(); // shards doesn't provide 3D permutations.

  if ( (coarseBasis->domainTopology()->getTensorialDegree() > 0) || (fineBasis->domainTopology()->getTensorialDegree() > 0) )
  {
    out << "ERROR: pConstraintInternalBasisSubTest() does not support tensorial degree > 0.\n";
    success = false;
    return;
  }

  for (unsigned permutation = 0; permutation < shardsPermutationCount; permutation++)
  {
    SubBasisReconciliationWeights weights = br.constrainedWeights(fineBasis, coarseBasis, permutation);

    //  cout << "BasisReconciliation: computed weights when matching whole bases.\n";

    FieldContainer<double> points = cubaturePoints(fineBasis->domainTopology()->getShardsTopology(), 5, 0);
    RefinementBranch noRefinements;
    FieldContainer<double> fineBasisValues = transformedBasisValuesAtPoints(fineBasis, points, noRefinements);
    FieldContainer<double> coarseBasisPoints = cubaturePoints(fineBasis->domainTopology()->getShardsTopology(), 5, permutation);
    FieldContainer<double> coarseBasisValues = transformedBasisValuesAtPoints(coarseBasis, coarseBasisPoints, noRefinements);

    stripCellDimensionFromFC(fineBasisValues);
    stripCellDimensionFromFC(coarseBasisValues);

    FieldContainer<double> interpretedFineBasisValues;
    interpretSideValues(weights, fineBasisValues, coarseBasisValues, interpretedFineBasisValues, coarseBasisValues);

    double tol = 1e-13;
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseBasisValues, interpretedFineBasisValues, tol);
  }
}

void testPConstraintSideBasis(BasisPtr fineBasis, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                              BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes,
                              Teuchos::FancyOStream &out, bool &success)
{
  BasisReconciliation br;

  int d = fineBasis->domainTopology()->getDimension();

  int oneCell = 1;
  fineCellNodes.resize(oneCell, fineCellNodes.dimension(0), fineCellNodes.dimension(1));
  coarseCellNodes.resize(oneCell, coarseCellNodes.dimension(0), coarseCellNodes.dimension(1));

  if ( (coarseBasis->domainTopology()->getTensorialDegree() > 0) || (fineBasis->domainTopology()->getTensorialDegree() > 0) )
  {
    out << "ERROR: pConstraintSideBasisSubTest() does not support tensorial degree > 0.\n";
    success = false;
    return;
  }

  // want to figure out a set of physical cell nodes that corresponds to this combination
  shards::CellTopology coarseTopo = coarseBasis->domainTopology()->getShardsTopology();
  shards::CellTopology fineTopo = fineBasis->domainTopology()->getShardsTopology();

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

  FieldContainer<double> finePointsPhysical = fineCache->getSideBasisCache(fineSideIndex)->getPhysicalCubaturePoints();
  FieldContainer<double> coarsePointsPhysical = coarseCache->getSideBasisCache(coarseSideIndex)->getPhysicalCubaturePoints();

  FieldContainer<double> finePoints = fineCache->getSideBasisCache(fineSideIndex)->getSideRefCellPointsInVolumeCoordinates();
  FieldContainer<double> coarsePoints = coarseCache->getSideBasisCache(coarseSideIndex)->getSideRefCellPointsInVolumeCoordinates();

  // finePoints and coarsePoints should match
  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(finePointsPhysical, coarsePointsPhysical, tol);

  RefinementBranch noRefinements;

  FieldContainer<double> fineBasisValues = transformedBasisValuesAtPoints(fineBasis, finePoints, noRefinements);
  FieldContainer<double> coarseBasisValues = transformedBasisValuesAtPoints(coarseBasis, coarsePoints, noRefinements);

  stripCellDimensionFromFC(fineBasisValues);
  stripCellDimensionFromFC(coarseBasisValues);

  FieldContainer<double> interpretedFineBasisValues, filteredCoarseBasisValues;
  interpretSideValues(weights, fineBasisValues, coarseBasisValues, interpretedFineBasisValues, filteredCoarseBasisValues);

  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(filteredCoarseBasisValues, interpretedFineBasisValues, tol);
}

void testHGRADTrace(MeshTopologyPtr meshTopo, int polyOrder, Teuchos::FancyOStream &out, bool &success)
{
  Camellia::EFunctionSpace fsSpace = Camellia::FUNCTION_SPACE_HGRAD;
  Camellia::EFunctionSpace fsTime = Camellia::FUNCTION_SPACE_HGRAD;
  vector<int> H1Orders = {polyOrder,polyOrder};

  int spaceDim = meshTopo->getSpaceDim();
  int sideDim = spaceDim - 1;
  int sideCount = meshTopo->getEntityCount(sideDim);

  vector<BasisPtr> basisForSideEntity(sideCount);
  for (int sideEntityIndex=0; sideEntityIndex<sideCount; sideEntityIndex++)
  {
    CellTopoPtr sideTopo = meshTopo->getEntityTopology(sideDim, sideEntityIndex);
    basisForSideEntity[sideEntityIndex] = BasisFactory::basisFactory()->getBasis(H1Orders, sideTopo, fsSpace, fsTime);
  }

//    meshTopo->printAllEntities();

  int cubatureDegree = polyOrder*2;

  set<IndexType> cellIDs = meshTopo->getActiveCellIndices();

  BasisReconciliation br;
  for (IndexType cellID1 : cellIDs)
  {
    CellPtr cell1 = meshTopo->getCell(cellID1);
    CellTopoPtr cellTopo1 = cell1->topology();
    RefinementPatternPtr trivialRefinement = RefinementPattern::noRefinementPattern(cellTopo1);
    RefinementBranch trivialRefBranch = {{trivialRefinement.get(),0}};

    for (int sideOrdinal1=0; sideOrdinal1<cellTopo1->getSideCount(); sideOrdinal1++)
    {
      CellTopoPtr sideTopo1 = cellTopo1->getSide(sideOrdinal1);
      IndexType sideEntityIndex1 = cell1->entityIndex(sideDim, sideOrdinal1);
      for (int d=0; d<=sideDim; d++)
      {
        int subcellCount = sideTopo1->getSubcellCount(d);
        for (unsigned subcellOrdinalSide1=0; subcellOrdinalSide1<subcellCount; subcellOrdinalSide1++)
        {
          unsigned subcellOrdinalCell1 = CamelliaCellTools::subcellOrdinalMap(cellTopo1, sideDim, sideOrdinal1, d, subcellOrdinalSide1);
          IndexType entityIndex = cell1->entityIndex(d, subcellOrdinalCell1); //meshTopo->getSubEntityIndex(sideDim, sideEntityIndex1, d, subcellOrdinalSide1);
          CellTopoPtr entityTopo = meshTopo->getEntityTopology(d, entityIndex);

          // get the permutation that goes from meshTopo's canonical node ordering to cell1's sideOrdinal1's ordering of the subcell nodes.
          unsigned canonicalToCell1Subcell = cell1->subcellPermutation(d, subcellOrdinalCell1);
          unsigned cell1SubcellToCanonical = CamelliaCellTools::permutationInverse(entityTopo, canonicalToCell1Subcell);

          auto sidesForEntity = meshTopo->getSidesContainingEntity(d, entityIndex);
          for (auto sideEntityIndex2 : sidesForEntity)
          {
            set< pair<IndexType, unsigned> > cellsForSide2 = meshTopo->getCellsContainingEntity(sideDim, sideEntityIndex2);

            for (pair<IndexType, unsigned> cellPair2 : cellsForSide2)
            {
              IndexType cellID2 = cellPair2.first;
              unsigned sideOrdinal2 = cellPair2.second;
              if ((cellID2 == cellID1) && (sideOrdinal1 == sideOrdinal2)) continue;

              CellPtr cell2 = meshTopo->getCell(cellID2);
              CellTopoPtr sideTopo2 = meshTopo->getEntityTopology(sideDim, sideEntityIndex2);

              int subcellOrdinalCell2 = cell2->findSubcellOrdinal(d, entityIndex);
              int subcellOrdinalSide2 = cell2->findSubcellOrdinalInSide(d, entityIndex, sideOrdinal2);
              // get the permutation that goes from meshTopo's canonical node ordering to cell1's sideOrdinal1's ordering of the subcell nodes.
              unsigned canonicalToCell2Subcell = cell2->subcellPermutation(d, subcellOrdinalCell2);
              unsigned subcellPermutationCell1ToCell2 = CamelliaCellTools::permutationComposition(entityTopo,
                  cell1SubcellToCanonical,
                  canonicalToCell2Subcell);

              BasisPtr basis1 = basisForSideEntity[sideEntityIndex1];
              BasisPtr basis2 = basisForSideEntity[sideEntityIndex2];

              SubBasisReconciliationWeights weights;
              weights = br.computeConstrainedWeights(d, basis1, subcellOrdinalSide1,
                                                     trivialRefBranch, sideOrdinal1,
                                                     cell2->topology(),
                                                     d, basis2, subcellOrdinalSide2,
                                                     sideOrdinal2, subcellPermutationCell1ToCell2);

              // now, we want to try computing some values with the two bases, and check that basis1 (the "finer") weighted
              // with weights actually sums up to match basis2 on the subcell that we've matched.

              CellTopoPtr subcellTopo1 = sideTopo1->getSubcell(d, subcellOrdinalSide1);
              CellTopoPtr subcellTopo2 = sideTopo2->getSubcell(d, subcellOrdinalSide2);

              // if guard added to suppress output when this comparison succeeds (which is always, at present)
              if (subcellTopo1->getKey() != subcellTopo2->getKey())
              {
                TEST_EQUALITY(subcellTopo1->getKey(), subcellTopo2->getKey());
              }

              // lazy way to get the cubature points for subcell:
              BasisCachePtr subcellCache = BasisCache::basisCacheForReferenceCell(subcellTopo1, cubatureDegree);
              FieldContainer<double> subcellCubPoints1 = subcellCache->getRefCellPoints();
              int numPoints = subcellCubPoints1.dimension(0);

              FieldContainer<double> subcellPointsInParent1(numPoints,sideTopo1->getDimension());
              CamelliaCellTools::mapToReferenceSubcell(subcellPointsInParent1, subcellCubPoints1, d, subcellOrdinalSide1, sideTopo1);


              unsigned canonicalToSide1Subcell = cell1->sideSubcellPermutation(sideOrdinal1, d, subcellOrdinalSide1);
              unsigned side1SubcellToCanonical = CamelliaCellTools::permutationInverse(entityTopo, canonicalToSide1Subcell);
              unsigned canonicalToSide2Subcell = cell2->sideSubcellPermutation(sideOrdinal2, d, subcellOrdinalSide2);
              unsigned subcellPermutationSide1ToSide2 = CamelliaCellTools::permutationComposition(subcellTopo1,
                  side1SubcellToCanonical,
                  canonicalToSide2Subcell);

              FieldContainer<double> subcellCubPoints2(subcellCubPoints1.dimension(0), subcellCubPoints1.dimension(1));
              CamelliaCellTools::permutedReferenceCellPoints(subcellTopo1, subcellPermutationSide1ToSide2, subcellCubPoints1, subcellCubPoints2);

              FieldContainer<double> subcellPointsInParent2(numPoints,sideTopo2->getDimension());
              CamelliaCellTools::mapToReferenceSubcell(subcellPointsInParent2, subcellCubPoints2, d, subcellOrdinalSide2, sideTopo2);

              // sanity check on the mapped points: map from the sides to physical space and make sure they're the same
              bool createSideCaches = true;
              FieldContainer<double> physicalCellNodes1(cellTopo1->getNodeCount(), cellTopo1->getDimension());
              meshTopo->verticesForCell(physicalCellNodes1, cellID1);
              physicalCellNodes1.resize(1,physicalCellNodes1.dimension(0),physicalCellNodes1.dimension(1));
              BasisCachePtr basisCacheCell1 = BasisCache::basisCacheForCellTopology(cellTopo1, cubatureDegree,
                                              physicalCellNodes1, createSideCaches);
              BasisCachePtr sideCache1 = basisCacheCell1->getSideBasisCache(sideOrdinal1);
              sideCache1->setRefCellPoints(subcellPointsInParent1);

              FieldContainer<double> physicalCellNodes2(cell2->topology()->getNodeCount(), cell2->topology()->getDimension());
              meshTopo->verticesForCell(physicalCellNodes2, cell2->cellIndex());
              physicalCellNodes2.resize(1,physicalCellNodes2.dimension(0),physicalCellNodes2.dimension(1));
              BasisCachePtr basisCacheCell2 = BasisCache::basisCacheForCellTopology(cell2->topology(), cubatureDegree,
                                              physicalCellNodes2, createSideCaches);
              BasisCachePtr sideCache2 = basisCacheCell2->getSideBasisCache(sideOrdinal2);
              sideCache2->setRefCellPoints(subcellPointsInParent2);

              bool oldSuccess = success;
              success = true;
              TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(sideCache1->getPhysicalCubaturePoints(), sideCache2->getPhysicalCubaturePoints(), 1e-14);
              bool justFailed = !success;
              success = success && oldSuccess;
              if (justFailed)
              {
                out << "sideCache1->getPhysicalCubaturePoints():\n" << sideCache1->getPhysicalCubaturePoints();
                out << "sideCache2->getPhysicalCubaturePoints():\n" << sideCache2->getPhysicalCubaturePoints();
              }

              FieldContainer<double> values1 = *sideCache1->getTransformedValues(basis1, OP_VALUE);
              FieldContainer<double> values2 = *sideCache2->getTransformedValues(basis2, OP_VALUE);
//
//                // multiply by the determinant of the Jacobian at each point:
//                FunctionSpaceTools::multiplyMeasure<double>(values1, sideCache1->getJacobianDet(), values1);
//                FunctionSpaceTools::multiplyMeasure<double>(values2, sideCache2->getJacobianDet(), values2);
//
              FieldContainer<double> values1Filtered(weights.fineOrdinals.size(),numPoints);
              int i=0;
              for (auto fineOrdinal : weights.fineOrdinals)
              {
                for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
                {
                  values1Filtered(i,ptOrdinal) = values1(0,fineOrdinal,ptOrdinal);
                }
                i++;
              }

              FieldContainer<double> values2Filtered(weights.coarseOrdinals.size(),numPoints);
              i=0;
              for (auto coarseOrdinal : weights.coarseOrdinals)
              {
                for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
                {
                  values2Filtered(i,ptOrdinal) = values2(0,coarseOrdinal,ptOrdinal);
                }
                i++;
              }

              //            out << "Weights to represent side2 = sideOrdinal " << side2 << " subcell " << subcellOrdinalSide2;
              //            out << " of dimension " << d << " in terms of basis on side1 = sideOrdinal " << side1;
              //            out << " subcell " << subcellOrdinalSide1 << ":\n" << weights.weights;

              FieldContainer<double> weightedValues1(weights.coarseOrdinals.size(),numPoints);
              weightedValues1.initialize(0.0);
              for (int i=0; i<weights.fineOrdinals.size(); i++)
              {
                for (int j=0; j<weights.coarseOrdinals.size(); j++)
                {
                  for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
                  {
                    weightedValues1(j,ptOrdinal) += weights.weights(i,j) * values1Filtered(i,ptOrdinal);
                    //                  out << "weightedValues1(" << j << "," << ptOrdinal << ") += " << weights.weights(i,j) * values1Filtered(i,ptOrdinal) << endl;
                  }
                }
              }

              double tol = 1e-14;
              // allow detection of local failure, and print out info just for things that fail
              oldSuccess = success;
              success = true;
              TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(values2Filtered, weightedValues1, tol);
              justFailed = !success;
              // restore global success
              success = oldSuccess && success;

              if (justFailed)
              {
                out << "Failure while checking BasisReconciliation for " << CamelliaCellTools::entityTypeString(subcellTopo1->getDimension());
                out << " with entity index " << entityIndex << " on cell " << cellID1 << ", side ordinal " << sideOrdinal1;
                out << " (subcell ordinal " << subcellOrdinalSide1;
                out << ") and cell " << cellID2 << ", side ordinal " << sideOrdinal2;
                out <<  " (subcell ordinal " << subcellOrdinalSide2 << ") -- subcellPermutation = " << subcellPermutationSide1ToSide2 << endl;

                out << "subcellCubPoints1:\n" << subcellCubPoints1;
                out << "subcellCubPoints2:\n" << subcellCubPoints2;

                out << "subcellPointsInParent1:\n" << subcellPointsInParent1;
                out << "subcellPointsInParent2:\n" << subcellPointsInParent2;

//                  out << "sideCache1->getJacobianDet():\n" << sideCache1->getJacobianDet();
//                  out << "sideCache2->getJacobianDet():\n" << sideCache2->getJacobianDet();

                out << "cell1 side1 permutation: " << cell1->subcellPermutation(sideDim, sideOrdinal1) << endl;
                out << "cell2 side2 permutation: " << cell2->subcellPermutation(sideDim, sideOrdinal2) << endl;

                vector<IndexType> side1Vertices = meshTopo->getEntityVertexIndices(sideDim, sideEntityIndex1);
                out << "side1Vertices: {";
                for (int i=0; i<side1Vertices.size(); i++)
                {
                  out << side1Vertices[i];
                  if (i < side1Vertices.size()-1) out << ",";
                  else out << "}\n";
                }

                vector<IndexType> side2Vertices = meshTopo->getEntityVertexIndices(sideDim, sideEntityIndex2);
                out << "side2Vertices: {";
                for (int i=0; i<side2Vertices.size(); i++)
                {
                  out << side2Vertices[i];
                  if (i < side2Vertices.size()-1) out << ",";
                  else out << "}\n";
                }

                out << "values1Filtered:\n" << values1Filtered;
                out << "weights.weights:\n" << weights.weights;
                out << "values2Filtered:\n" << values2Filtered;
                out << "weightedValues1:\n" << weightedValues1;

                BasisCachePtr side1Cache, side2Cache;
                br.setupFineAndCoarseBasisCachesForReconciliation(side1Cache, side2Cache,
                    d, basis1, subcellOrdinalSide1,
                    trivialRefBranch, sideOrdinal1,
                    cell2->topology(),
                    d, basis2, subcellOrdinalSide2,
                    sideOrdinal2, subcellPermutationCell1ToCell2);
                out << "side1Cache->getRefCellPoints():\n" << side1Cache->getRefCellPoints();
                out << "side2Cache->getRefCellPoints():\n" << side2Cache->getRefCellPoints();
              }
            }
          }
        }
      }
    }
  }
}

void testHGRADTrace(CellTopoPtr volumeTopo, int polyOrder, Teuchos::FancyOStream &out, bool &success)
{
  MeshTopologyPtr meshTopo = Teuchos::rcp(new MeshTopology(volumeTopo->getDimension()));
  FieldContainer<double> refCellNodes(volumeTopo->getNodeCount(), volumeTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, volumeTopo);
  vector<vector<double>> cellNodesVector;
  CamelliaCellTools::pointsVectorFromFC(cellNodesVector, refCellNodes);
  meshTopo->addCell(volumeTopo, cellNodesVector);
  testHGRADTrace(meshTopo, polyOrder, out, success);
}

void testWeightedBasesAgree(FieldContainer<double> &fineBasisValues, FieldContainer<double> &coarseBasisValues,
                            SubBasisReconciliationWeights &weights,
                            Teuchos::FancyOStream &out, bool &success)
{
  int numPoints = fineBasisValues.dimension(2);
  FieldContainer<double> values1Filtered(weights.fineOrdinals.size(),numPoints);
  int i=0;
  for (auto fineOrdinal : weights.fineOrdinals)
  {
    for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      values1Filtered(i,ptOrdinal) = fineBasisValues(0,fineOrdinal,ptOrdinal);
    }
    i++;
  }

  FieldContainer<double> values2Filtered(weights.coarseOrdinals.size(),numPoints);
  i=0;
  for (auto coarseOrdinal : weights.coarseOrdinals)
  {
    for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      values2Filtered(i,ptOrdinal) = coarseBasisValues(0,coarseOrdinal,ptOrdinal);
    }
    i++;
  }

  FieldContainer<double> weightedValues1(weights.coarseOrdinals.size(),numPoints);
  weightedValues1.initialize(0.0);
  for (int i=0; i<weights.fineOrdinals.size(); i++)
  {
    for (int j=0; j<weights.coarseOrdinals.size(); j++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        weightedValues1(j,ptOrdinal) += weights.weights(i,j) * values1Filtered(i,ptOrdinal);
      }
    }
  }

  double tol = 1e-14;
  // allow detection of local failure, and print out info just for things that fail
  bool oldSuccess = success;
  success = true;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(values2Filtered, weightedValues1, tol);

  if (!success)
  {
    out << "values1Filtered:\n" << values1Filtered;
    out << "weights.weights:\n" << weights.weights;
    out << "values2Filtered:\n" << values2Filtered;
    out << "weightedValues1:\n" << weightedValues1;
  }

  // restore global success
  success = oldSuccess && success;
}

void testHGRADVolumeNoHangingNodes(MeshTopologyPtr meshTopo, int polyOrder, Teuchos::FancyOStream &out, bool &success)
{
  Camellia::EFunctionSpace fsSpace = Camellia::FUNCTION_SPACE_HGRAD;
  Camellia::EFunctionSpace fsTime = Camellia::FUNCTION_SPACE_HGRAD;
  vector<int> H1Orders = {polyOrder,polyOrder};

  int spaceDim = meshTopo->getSpaceDim();
  int sideDim = spaceDim - 1;

  set<IndexType> cellIndices = meshTopo->getActiveCellIndices();

  map<IndexType, BasisPtr> basisForCell;
  for (IndexType cellIndex : cellIndices)
  {
    CellTopoPtr cellTopo = meshTopo->getCell(cellIndex)->topology();
    basisForCell[cellIndex] = BasisFactory::basisFactory()->getBasis(H1Orders, cellTopo, fsSpace, fsTime);
  }

//    meshTopo->printAllEntities();

  int cubatureDegree = polyOrder*2;

  set<IndexType> cellIDs = meshTopo->getActiveCellIndices();

  BasisReconciliation br;
  for (IndexType cellID1 : cellIDs)
  {
    CellPtr cell1 = meshTopo->getCell(cellID1);
    CellTopoPtr cellTopo1 = cell1->topology();
    RefinementPatternPtr trivialRefinement = RefinementPattern::noRefinementPattern(cellTopo1);
    RefinementBranch trivialRefBranch = {{trivialRefinement.get(),0}};

    for (int d=0; d<=sideDim; d++)
    {
      int subcellCount = cellTopo1->getSubcellCount(d);
      for (unsigned scord1=0; scord1<subcellCount; scord1++)
      {
        IndexType entityIndex = cell1->entityIndex(d, scord1);
        CellTopoPtr entityTopo = meshTopo->getEntityTopology(d, entityIndex);

        // get the permutation that goes from meshTopo's canonical node ordering to cell1's sideOrdinal1's ordering of the subcell nodes.
        unsigned canonicalToSubcell1 = cell1->subcellPermutation(d,scord1);
        unsigned subcell1ToCanonical = CamelliaCellTools::permutationInverse(entityTopo, canonicalToSubcell1);

        set< pair<IndexType, unsigned> > cellsForEntity = meshTopo->getCellsContainingEntity(d, entityIndex);

        for (pair<IndexType, unsigned> cellEntry : cellsForEntity)
        {
          IndexType cellID2 = cellEntry.first;
          //            unsigned sideOrdinal2 = cellEntry.second;
          if (cellID2 == cellID1) continue;

          CellPtr cell2 = meshTopo->getCell(cellID2);
          CellTopoPtr cellTopo2 = cell2->topology();
          unsigned scord2 = cell2->findSubcellOrdinal(d,entityIndex);

          // get the permutation that goes from meshTopo's canonical node ordering to cell1's sideOrdinal1's ordering of the subcell nodes.
          unsigned canonicalToSubcell2 = cell2->subcellPermutation(d, scord2);
          unsigned subcellPermutation1To2 = CamelliaCellTools::permutationComposition(entityTopo, subcell1ToCanonical, canonicalToSubcell2);

          BasisPtr basis1 = basisForCell[cellID1];
          BasisPtr basis2 = basisForCell[cellID2];

          SubBasisReconciliationWeights weights;
          unsigned volumeOrdinal = 0;
          weights = br.computeConstrainedWeights(d, basis1, scord1,
                                                 trivialRefBranch, volumeOrdinal,
                                                 cell2->topology(),
                                                 d, basis2, scord2,
                                                 volumeOrdinal, subcellPermutation1To2);

          // now, we want to try computing some values with the two bases, and check that basis1 (the "finer") weighted
          // with weights actually sums up to match basis2 on the subcell that we've matched.

          CellTopoPtr subcellTopo1 = cellTopo1->getSubcell(d, scord1);
          CellTopoPtr subcellTopo2 = cellTopo2->getSubcell(d, scord2);

          // if guard added to suppress output when this comparison succeeds (which is always, at present)
          if (subcellTopo1->getKey() != subcellTopo2->getKey())
          {
            TEST_EQUALITY(subcellTopo1->getKey(), subcellTopo2->getKey());
          }

          // lazy way to get the cubature points for subcell:
          BasisCachePtr subcellCache = BasisCache::basisCacheForReferenceCell(subcellTopo1, cubatureDegree);
          FieldContainer<double> subcellCubPoints1 = subcellCache->getRefCellPoints();
          int numPoints = subcellCubPoints1.dimension(0);

          FieldContainer<double> subcellPointsInRefCell1(numPoints,cellTopo1->getDimension());
          CamelliaCellTools::mapToReferenceSubcell(subcellPointsInRefCell1, subcellCubPoints1, d, scord1, cellTopo1);

          FieldContainer<double> subcellCubPoints2(subcellCubPoints1.dimension(0), subcellCubPoints1.dimension(1));
          CamelliaCellTools::permutedReferenceCellPoints(subcellTopo1, subcellPermutation1To2, subcellCubPoints1, subcellCubPoints2);

          FieldContainer<double> subcellPointsInRefCell2(numPoints,cellTopo2->getDimension());
          CamelliaCellTools::mapToReferenceSubcell(subcellPointsInRefCell2, subcellCubPoints2, d, scord2, cellTopo2);

          // sanity check on the mapped points: map from the sides to physical space and make sure they're the same
          bool createSideCaches = false;
          FieldContainer<double> physicalCellNodes1(cellTopo1->getNodeCount(), cellTopo1->getDimension());
          meshTopo->verticesForCell(physicalCellNodes1, cellID1);
          physicalCellNodes1.resize(1,physicalCellNodes1.dimension(0),physicalCellNodes1.dimension(1));
          BasisCachePtr basisCache1 = BasisCache::basisCacheForCellTopology(cellTopo1, cubatureDegree,
                                      physicalCellNodes1, createSideCaches);
          basisCache1->setRefCellPoints(subcellPointsInRefCell1);

          FieldContainer<double> physicalCellNodes2(cellTopo2->getNodeCount(), cellTopo2->getDimension());
          meshTopo->verticesForCell(physicalCellNodes2, cell2->cellIndex());
          physicalCellNodes2.resize(1,physicalCellNodes2.dimension(0),physicalCellNodes2.dimension(1));
          BasisCachePtr basisCache2 = BasisCache::basisCacheForCellTopology(cellTopo2, cubatureDegree,
                                      physicalCellNodes2, createSideCaches);
          basisCache2->setRefCellPoints(subcellPointsInRefCell2);

          TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(basisCache1->getPhysicalCubaturePoints(), basisCache2->getPhysicalCubaturePoints(), 1e-14);

          FieldContainer<double> values1 = *basisCache1->getTransformedValues(basis1, OP_VALUE);
          FieldContainer<double> values2 = *basisCache2->getTransformedValues(basis2, OP_VALUE);

          // allow detection of local failure, and print out info just for things that fail
          bool oldSuccess = success;
          success = true;
          testWeightedBasesAgree(values1, values2, weights, out, success);

          if (!success)
          {
            out << "Failure while checking BasisReconciliation for " << CamelliaCellTools::entityTypeString(subcellTopo1->getDimension());
            out << " with entity index " << entityIndex << " on cell " << cellID1;
            out << " (subcell ordinal " << scord1;
            out << ") and cell " << cellID2;
            out <<  " (subcell ordinal " << scord2 << ") -- subcellPermutation = " << subcellPermutation1To2 << endl;

            out << "subcellCubPoints1:\n" << subcellCubPoints1;
            out << "subcellCubPoints2:\n" << subcellCubPoints2;

            out << "subcellPointsInParent1:\n" << subcellPointsInRefCell1;
            out << "subcellPointsInParent2:\n" << subcellPointsInRefCell2;

            out << "cell1 subcell permutation: " << cell1->subcellPermutation(d, scord1) << endl;
            out << "cell2 subcell permutation: " << cell2->subcellPermutation(d, scord2) << endl;

            vector<IndexType> scVertices = meshTopo->getEntityVertexIndices(d, entityIndex);
            out << "subcell vertices: {";
            for (int i=0; i<scVertices.size(); i++)
            {
              out << scVertices[i];
              if (i < scVertices.size()-1) out << ",";
              else out << "}\n";
            }
          }

          // restore global success
          success = oldSuccess && success;
        }
      }
    }
  }
}

void testMap(const FieldContainer<double> &expectedCoarseSubcellPoints,
             const FieldContainer<double> &fineSubcellPoints,
             unsigned fineSubcellDimension,
             unsigned fineSubcellOrdinalInFineDomain, unsigned fineDomainDim,
             unsigned fineDomainOrdinalInRefinementLeaf,
             RefinementBranch refBranch,
             CellTopoPtr volumeTopo,
             unsigned coarseSubcellDimension, unsigned coarseSubcellOrdinalInCoarseDomain,
             unsigned coarseDomainDim, unsigned coarseDomainOrdinalInRefinementRoot,
             unsigned coarseSubcellPermutation,
             Teuchos::FancyOStream &out, bool &success)
{
  // map two points on a fine edge to the coarse volume
  int coarseDim = expectedCoarseSubcellPoints.dimension(1);
  int numPoints = fineSubcellPoints.dimension(0);
  FieldContainer<double> coarseSubcellPoints(numPoints,coarseDim);

  BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
      fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);

  out << "coarseSubcellPoints:\n" << coarseSubcellPoints;
  out << "expectedCoarseSubcellPoints:\n" << expectedCoarseSubcellPoints;

  double tol = 1e-15;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, h_Slow )
{
  // copied from DPGTests's BasisReconciliationTests::testH()
  int fineOrder = 1;
  int coarseOrder = 1;

  RefinementBranch lineRefinementsZero = makeRefinementBranch(RefinementPattern::regularRefinementPatternLine(), vector<unsigned>(4,0));
  vector<unsigned> alternatingZeroOne;
  alternatingZeroOne.push_back(0);
  alternatingZeroOne.push_back(1);
  alternatingZeroOne.push_back(0);
  alternatingZeroOne.push_back(1);
  RefinementBranch lineRefinementsAlternating = makeRefinementBranch(RefinementPattern::regularRefinementPatternLine(), alternatingZeroOne);

  RefinementBranch quadRefinementsZero = makeRefinementBranch(RefinementPattern::regularRefinementPatternQuad(), vector<unsigned>(4,0));
  vector<unsigned> zeroOneTwoThree;
  zeroOneTwoThree.push_back(0);
  zeroOneTwoThree.push_back(1);
  zeroOneTwoThree.push_back(2);
  zeroOneTwoThree.push_back(3);
  RefinementBranch quadRefinementsVarious = makeRefinementBranch(RefinementPattern::regularRefinementPatternQuad(), zeroOneTwoThree);

  RefinementBranch hexRefinementsZero = makeRefinementBranch(RefinementPattern::regularRefinementPatternHexahedron(), vector<unsigned>(4,0));
  RefinementBranch hexRefinementsVarious = makeRefinementBranch(RefinementPattern::regularRefinementPatternHexahedron(), zeroOneTwoThree);

  vector< pair< pair< BasisPtr, BasisPtr >, RefinementBranch > > basisPairsToCheck;

  BasisPtr fineBasis = Camellia::intrepidLineHGRAD(fineOrder);
  BasisPtr coarseBasis = Camellia::intrepidLineHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), lineRefinementsZero ) );
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), lineRefinementsAlternating ) );

  fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), quadRefinementsZero ) );
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), quadRefinementsVarious ) );

  fineBasis = Camellia::intrepidQuadHDIV(fineOrder);
  coarseBasis = Camellia::intrepidQuadHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), quadRefinementsZero ) );
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), quadRefinementsVarious ) );

  fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), hexRefinementsZero ) );
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), hexRefinementsVarious ) );

  fineBasis = Camellia::intrepidHexHDIV(fineOrder);
  coarseBasis = Camellia::intrepidHexHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), hexRefinementsZero ) );
  basisPairsToCheck.push_back( make_pair(make_pair(fineBasis, coarseBasis), hexRefinementsVarious ) );

  for (auto basisPair : basisPairsToCheck)
  {
    fineBasis = basisPair.first.first;
    coarseBasis = basisPair.first.second;
    RefinementBranch refinements = basisPair.second;
    testHConstraintInternalBasis(fineBasis, refinements, coarseBasis, out, success);
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, hSide )
{
  // copied from DPGTests's BasisReconciliationTests::testHSide()

  int fineOrder = 1;
  int coarseOrder = 1;

  shards::CellTopology line = shards::getCellTopologyData< shards::Line<2> >();
  shards::CellTopology quad = shards::getCellTopologyData< shards::Quadrilateral<4> >();
  shards::CellTopology hex = shards::getCellTopologyData< shards::Hexahedron<8> >();

  double width = 2;
  double height = 2;
  FieldContainer<double> centeredQuad(4,2);
  centeredQuad(0,0) = -width / 2;
  centeredQuad(0,1) = -height / 2;
  centeredQuad(1,0) = width / 2;
  centeredQuad(1,1) = -height / 2;
  centeredQuad(2,0) = width / 2;
  centeredQuad(2,1) = height / 2;
  centeredQuad(3,0) = -width / 2;
  centeredQuad(3,1) = height / 2;

  FieldContainer<double> eastQuad = translateQuad(centeredQuad, width, 0);
  FieldContainer<double> westQuad = translateQuad(centeredQuad, -width, 0);
  FieldContainer<double> southQuad = translateQuad(centeredQuad, -height, 0);
  FieldContainer<double> northQuad = translateQuad(centeredQuad, height, 0);

  hSideTest test;

  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;

  vector< hSideTest > sideTests;

  test.fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = EAST;
  test.coarseCellNodes = eastQuad;
  test.coarseSideIndex = WEST;

  test.volumeRefinements = demoRefinementsOnSide(quad, EAST, 1);
  sideTests.push_back(test);

  test.volumeRefinements = demoRefinementsOnSide(quad, EAST, 3);
  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westQuad;
  test.coarseSideIndex = EAST;

  test.volumeRefinements = demoRefinementsOnSide(quad, WEST, 1);
  sideTests.push_back(test);
  test.volumeRefinements = demoRefinementsOnSide(quad, WEST, 3);
  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidQuadHDIV(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHDIV(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westQuad;
  test.coarseSideIndex = EAST;

  test.volumeRefinements = demoRefinementsOnSide(quad, WEST, 1);
  sideTests.push_back(test);
  test.volumeRefinements = demoRefinementsOnSide(quad, WEST, 3);
  sideTests.push_back(test);


  /* ************** HEXES ************* */

  // redefine NORTH, SOUTH, EAST and WEST:
  NORTH = 2;
  SOUTH = 0;
  EAST = 1;
  WEST = 3;
  int FRONT = 4; // FRONT is -z direction
  int BACK = 5; // BACK is +z direction

  FieldContainer<double> centeredHex(8,3);
  double depth = 8;
  centeredHex(0,0) = -width / 2;
  centeredHex(0,1) = -height / 2;
  centeredHex(0,2) = -depth / 2;

  centeredHex(1,0) =  width / 2;
  centeredHex(1,1) = -height / 2;
  centeredHex(1,2) = -depth / 2;

  centeredHex(2,0) =  width / 2;
  centeredHex(2,1) =  height / 2;
  centeredHex(2,2) = -depth / 2;

  centeredHex(3,0) = -width / 2;
  centeredHex(3,1) =  height / 2;
  centeredHex(3,2) = -depth / 2;

  centeredHex(4,0) = -width / 2;
  centeredHex(4,1) = -height / 2;
  centeredHex(4,2) =  depth / 2;

  centeredHex(5,0) =  width / 2;
  centeredHex(5,1) = -height / 2;
  centeredHex(5,2) =  depth / 2;

  centeredHex(6,0) =  width / 2;
  centeredHex(6,1) =  height / 2;
  centeredHex(6,2) =  depth / 2;

  centeredHex(7,0) = -width / 2;
  centeredHex(7,1) =  height / 2;
  centeredHex(7,2) =  depth / 2;

  FieldContainer<double> eastHex = translateHex(centeredHex, width, 0, 0);
  FieldContainer<double> westHex = translateHex(centeredHex, -width, 0, 0);
  FieldContainer<double> northHex = translateHex(centeredHex, 0, height, 0);
  FieldContainer<double> southHex = translateHex(centeredHex, 0, -height, 0);
  FieldContainer<double> frontHex = translateHex(centeredHex, 0, 0, -depth);
  FieldContainer<double> backHex = translateHex(centeredHex, 0, 0, depth);

  test.fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westHex;
  test.coarseSideIndex = EAST;
  test.volumeRefinements = demoRefinementsOnSide(hex, WEST, 1);
  sideTests.push_back(test);

  test.volumeRefinements = demoRefinementsOnSide(hex, WEST, 3);
  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = BACK;
  test.coarseCellNodes = backHex;
  test.coarseSideIndex = FRONT;

  test.volumeRefinements = demoRefinementsOnSide(hex, BACK, 1);
  sideTests.push_back(test);
  test.volumeRefinements = demoRefinementsOnSide(hex, BACK, 3);
  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidHexHDIV(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHDIV(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westHex;
  test.coarseSideIndex = EAST;

  test.volumeRefinements = demoRefinementsOnSide(hex, WEST, 1);
  sideTests.push_back(test);
  test.volumeRefinements = demoRefinementsOnSide(hex, WEST, 3);
  sideTests.push_back(test);

  for (auto test : sideTests)
  {
    testHConstraintSideBasis(test.fineBasis, test.fineSideIndex, test.fineCellNodes,
                             test.volumeRefinements,
                             test.coarseBasis, test.coarseSideIndex, test.coarseCellNodes,
                             out, success);
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Line )
{
  // this should pass trivially: no sides share any subcells in a line topology
  CellTopoPtr volumeTopo = CellTopology::line();
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Line_x_Line )
{
  CellTopoPtr volumeTopo = CellTopology::cellTopology(CellTopology::line(), 1);
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Quad )
{
  CellTopoPtr volumeTopo = CellTopology::quad();
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_TwoQuadCW )
{
  // a mesh with two quads, both oriented clockwise
  CellTopoPtr volumeTopo = CellTopology::quad();
  int polyOrder = 2;
  vector<vector<double>> vertices1 = {{0,0},{0,1},{1,1},{1,0}};
  vector<vector<double>> vertices2 = {{1,0},{1,1},{2,1},{2,0}};
  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(volumeTopo->getDimension()));
  meshTopo->addCell(volumeTopo, vertices1);
  meshTopo->addCell(volumeTopo, vertices2);
  testHGRADTrace(meshTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_TwoQuadCCW )
{
  // a mesh with two quads, both oriented counter-clockwise
  CellTopoPtr volumeTopo = CellTopology::quad();
  int polyOrder = 2;
  vector<vector<double>> vertices1 = {{0,0},{1,0},{1,1},{0,1}};
  vector<vector<double>> vertices2 = {{1,0},{2,0},{2,1},{1,1}};
  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(volumeTopo->getDimension()));
  meshTopo->addCell(volumeTopo, vertices1);
  meshTopo->addCell(volumeTopo, vertices2);
  testHGRADTrace(meshTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_TwoQuadMixed )
{
  // a mesh with two quads, with one oriented counter-clockwise and the other clockwise
  CellTopoPtr volumeTopo = CellTopology::quad();
  int polyOrder = 2;
  vector<vector<double>> vertices1 = {{0,0},{1,0},{1,1},{0,1}};
  vector<vector<double>> vertices2 = {{1,0},{1,1},{2,1},{2,0}};
  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(volumeTopo->getDimension()));
  meshTopo->addCell(volumeTopo, vertices1);
  meshTopo->addCell(volumeTopo, vertices2);
  testHGRADTrace(meshTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Quad_x_Line_Slow )
{
  CellTopoPtr volumeTopo = CellTopology::cellTopology(CellTopology::quad(), 1);
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Triangle_x_Line_Slow )
{
  CellTopoPtr volumeTopo = CellTopology::cellTopology(CellTopology::triangle(), 1);
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Triangle )
{
  CellTopoPtr volumeTopo = CellTopology::triangle();
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Hexahedron_x_Line_Slow )
{
  CellTopoPtr volumeTopo = CellTopology::cellTopology(CellTopology::hexahedron(), 1);
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Hexahedron_Slow )
{
  CellTopoPtr volumeTopo = CellTopology::hexahedron();
  int polyOrder = 1;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

// There's a missing implementation for permutations of Tetrahedra; should be fairly simple to fill in.
// For now, we disable this test.
//  TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Tetrahedron_x_Line )
//  {
//    CellTopoPtr volumeTopo = CellTopology::cellTopology(CellTopology::tetrahedron(), 1);
//    int polyOrder = 2;
//    testHGRADTrace(volumeTopo, polyOrder, out, success);
//  }

TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADTrace_Tetrahedron )
{
  CellTopoPtr volumeTopo = CellTopology::tetrahedron();
  int polyOrder = 2;
  testHGRADTrace(volumeTopo, polyOrder, out, success);
}

// ! Tests BasisReconciliation as it would be used in a continuous Galerkin mesh made up of lines.
TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADVolume_Line )
{
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({1.0}, {2});
  int polyOrder = 2;
  testHGRADVolumeNoHangingNodes(meshTopo, polyOrder, out, success);
}

// ! Tests BasisReconciliation as it would be used in a continuous Galerkin mesh made up of quadrilaterals.
TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADVolume_Quad )
{
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({1.0,1.0}, {2,2});
  int polyOrder = 2;
  testHGRADVolumeNoHangingNodes(meshTopo, polyOrder, out, success);
}

// ! Tests BasisReconciliation as it would be used in a continuous Galerkin mesh made up of hexahedra.
TEUCHOS_UNIT_TEST( BasisReconciliation, HGRADVolume_Hexahedron )
{
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({1,1,1}, {1,2,1});
  int polyOrder = 2;
  testHGRADVolumeNoHangingNodes(meshTopo, polyOrder, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, InternalSubcellOrdinals )
{
  // Copied from DPGTests's BasisReconciliationTests::testInternalSubcellOrdinals()
  BasisPtr fineBasis = Camellia::intrepidLineHGRAD(2); // quadratic

  RefinementBranch emptyBranch;

  set<unsigned> internalDofOrdinals = BasisReconciliation::internalDofOrdinalsForFinerBasis(fineBasis, emptyBranch);
  if (internalDofOrdinals.size() != 1)
  {
    out << "BasisReconciliationTests: test failure.  Unrefined quadratic H^1 basis should have 1 internal dof ordinal, but has " << internalDofOrdinals.size() << endl;
    success = false;
  }

  RefinementBranch oneRefinement;
  oneRefinement.push_back(make_pair(RefinementPattern::regularRefinementPatternLine().get(), 0));

  internalDofOrdinals = BasisReconciliation::internalDofOrdinalsForFinerBasis(fineBasis, oneRefinement);
  // now, one vertex should lie inside the neighboring/constraining basis: two dof ordinals should now be interior
  if (internalDofOrdinals.size() != 2)
  {
    out << "BasisReconciliationTests: test failure.  Once-refined quadratic H^1 basis should have 2 internal dof ordinals, but has " << internalDofOrdinals.size() << endl;
    success = false;
  }

  RefinementBranch twoRefinements = oneRefinement;
  twoRefinements.push_back(make_pair(RefinementPattern::regularRefinementPatternLine().get(), 1)); // now the whole edge is interior to the constraining (ancestral) edge

  internalDofOrdinals = BasisReconciliation::internalDofOrdinalsForFinerBasis(fineBasis, twoRefinements);
  // now, both vertices should lie inside the neighboring/constraining basis: three dof ordinals should now be interior
  if (internalDofOrdinals.size() != 3)
  {
    out << "BasisReconciliationTests: test failure.  Twice-refined quadratic H^1 basis should have 3 internal dof ordinals, but has " << internalDofOrdinals.size() << endl;
    success = false;
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_1DLine_TwoRefinements)
{
  // refine line twice.  Let the fine domain be a line (volume topo), choosing the left branch at first refinement,
  // right at second.  Subcell is the domain itself, but the points we choose are interior, and not symmetric.
  // the points should thus be scaled by 1/4, and translated -0.25

  int numPoints = 2;
  int spaceDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, spaceDim);
  fineSubcellPoints(0,0) = -0.5;
  fineSubcellPoints(0,1) =  1.0;

  FieldContainer<double> expectedCoarseSubcellPoints(numPoints,spaceDim);
  for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
  {
    expectedCoarseSubcellPoints(ptOrdinal,0) = fineSubcellPoints(ptOrdinal,0) / 4.0 - 0.25;
  }

  FieldContainer<double> coarseSubcellPoints;

  unsigned fineSubcellDimension = 1; // line
  unsigned fineSubcellOrdinalInFineDomain = 0; // only one

  unsigned fineDomainDim = 1; // fine domain is 1D (a volume)
  unsigned fineDomainOrdinalInRefinementLeaf = 0; // the only one

  CellTopoPtr volumeTopo = CellTopology::line();

  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
  RefinementBranch twoRefinements = {{oneRefinement.get(),0},{oneRefinement.get(),1}}; // left, then right

  unsigned coarseSubcellDimension = 1; // volume
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 1;
  unsigned coarseDomainOrdinalInRefinementRoot = 0;
  unsigned coarseSubcellPermutation = 0;

  BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
      fineDomainDim, fineDomainOrdinalInRefinementLeaf, twoRefinements,
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
  double tol = 1e-15;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(expectedCoarseSubcellPoints, coarseSubcellPoints, tol);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_1DVertex_TwoRefinementsFineDomainIsLine)
{
  // refine line twice.  Let the fine domain be a line (volume topo), choosing the left branch at first refinement,
  // right at second.  Subcell is the left of two vertices -- this should be the x = -0.5 point.

  int numPoints = 1;
  int spaceDim  = 0;
  FieldContainer<double> fineSubcellPoints(numPoints, spaceDim);

  FieldContainer<double> coarseSubcellPoints;

  unsigned fineSubcellDimension = 0; // vertex
  unsigned fineSubcellOrdinalInFineDomain = 0; // left vertex

  unsigned fineDomainDim = 1; // fine domain is 1D (a volume)
  unsigned fineDomainOrdinalInRefinementLeaf = 0; // the only one

  CellTopoPtr volumeTopo = CellTopology::line();

  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
  RefinementBranch twoRefinements = {{oneRefinement.get(),0},{oneRefinement.get(),1}}; // left, then right

  unsigned coarseSubcellDimension = 1; // volume
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 1;
  unsigned coarseDomainOrdinalInRefinementRoot = 0;
  unsigned coarseSubcellPermutation = 0;

  BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
      fineDomainDim, fineDomainOrdinalInRefinementLeaf, twoRefinements,
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
  double tol = 1e-15;
  double expected_x = -0.5;

  double actual_x = coarseSubcellPoints(0,0);

  TEST_FLOATING_EQUALITY(actual_x, expected_x, tol);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_1DVertex_TwoRefinementsFineDomainIsVertex)
{
  // refine line twice.  Let the fine domain be a vertex.  Choose the right branch at first refinement,
  // left at second.  Fine domain is the right of two vertices -- this should be the x = +0.5 point.

  int numPoints = 1;
  int spaceDim  = 0;
  FieldContainer<double> fineSubcellPoints(numPoints, spaceDim);

  FieldContainer<double> coarseSubcellPoints;

  unsigned fineSubcellDimension = 0; // vertex
  unsigned fineSubcellOrdinalInFineDomain = 0; // left vertex

  unsigned fineDomainDim = 0; // fine domain is a point (a "side")
  unsigned fineDomainOrdinalInRefinementLeaf = 1; // right vertex in fine cell

  CellTopoPtr volumeTopo = CellTopology::line();

  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
  RefinementBranch twoRefinements = {{oneRefinement.get(),1},{oneRefinement.get(),0}}; // right, then left

  unsigned coarseSubcellDimension = 1; // volume
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 1;
  unsigned coarseDomainOrdinalInRefinementRoot = 0;
  unsigned coarseSubcellPermutation = 0;

  BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
      fineDomainDim, fineDomainOrdinalInRefinementLeaf, twoRefinements,
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);


  double tol = 1e-15;
  double expected_x = 0.5;

  double actual_x = coarseSubcellPoints(0,0);
  TEST_FLOATING_EQUALITY(actual_x, expected_x, tol);
}

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
      volumeTopo,
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
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);

  // we have no refinements, so basically what we're doing is picking vertex ordinal 1 from side ordinal 1 of the quad
  // should be mapped to (1,1)

  double tol = 1e-15;

  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_Volume)
{
  // map two points on a fine edge to the coarse volume

  int numPoints = 2;
  int fineDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, fineDim);
  fineSubcellPoints(0,0) = -0.25;
  fineSubcellPoints(1,0) = 0.5;

  int coarseDim = 2;
  FieldContainer<double> expectedCoarseSubcellPoints(numPoints,coarseDim);
  expectedCoarseSubcellPoints(0,0) = 0.0;
  expectedCoarseSubcellPoints(0,1) = 0.5 * fineSubcellPoints(0,0) - 0.5;
  expectedCoarseSubcellPoints(1,0) = 0.0;
  expectedCoarseSubcellPoints(1,1) = 0.5 * fineSubcellPoints(1,0) - 0.5;

  FieldContainer<double> coarseSubcellPoints;

  unsigned fineSubcellDimension = fineDim; // edge
  unsigned fineSubcellOrdinalInFineDomain = 0; // edge

  unsigned fineDomainDim = fineDim; // fine domain is 1D
  unsigned fineDomainOrdinalInRefinementLeaf = 1; // side 1

  CellTopoPtr volumeTopo = CellTopology::quad();

  RefinementBranch refBranch;
  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPatternQuad();
  refBranch.push_back( make_pair(oneRefinement.get(), 0) );

  unsigned coarseSubcellDimension = 2; // volume
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 2;
  unsigned coarseDomainOrdinalInRefinementRoot = 0;
  unsigned coarseSubcellPermutation = 0;

  BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
      fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
      volumeTopo,
      coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);

  out << "coarseSubcellPoints:\n" << coarseSubcellPoints;
  out << "expectedCoarseSubcellPoints:\n" << expectedCoarseSubcellPoints;

  double tol = 1e-15;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
}
  
  TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToPermutedCoarseVolume)
  {
    /*
     
     Take a twice-refined mesh like this:
    
     _________________________________
     |               |               |
     |               |               |
     |               |               |
     |               |               |
     |               |               |
     |               |               |
     |               |               |
     |_______________|_______________|
     |               |       |       |
     |               |       |       |
     |               |       |       |
     |               |_______|_______|
     |               |       |       |
     |               |       |       |
     |               |   A   |       |
     |_______________|_______|_______|
     
     We are interested in the west side of cell A, in particular in the case
     when we are mapping points on this edge to a coarse quad that has been permuted
     (so that it's clockwise rather than counter-clockwise; this is permutation 4 for the
     quadrilateral).
     
     TODO: Rewrite this to be a slice of a 3D mesh, so that the permuted case is allowed.
     
     */
    
    // UNPERMUTED CASE:
    int numPoints = 2;
    int fineDomainDim  = 2;
    int fineSubcellDim = 1; // edge
    FieldContainer<double> fineSubcellPoints(numPoints, fineSubcellDim);
    fineSubcellPoints(0,0) = -0.25;
    fineSubcellPoints(1,0) = 0.5;
    
    int coarseDim = 2;
    FieldContainer<double> expectedCoarseSubcellPoints(numPoints,coarseDim);
    expectedCoarseSubcellPoints(0,0) = 0.0;
    expectedCoarseSubcellPoints(0,1) = -0.25 * fineSubcellPoints(0,0) - 0.75;
    expectedCoarseSubcellPoints(1,0) = 0.0;
    expectedCoarseSubcellPoints(1,1) = -0.25 * fineSubcellPoints(1,0) - 0.75;
    
    FieldContainer<double> coarseSubcellPoints;
    
    unsigned fineSubcellOrdinalInFineDomain = 3; // edge 3 in the fine domain
    unsigned fineDomainOrdinalInRefinementLeaf = 0; // side 0
    
    CellTopoPtr volumeTopo = CellTopology::hexahedron();
    
    RefinementBranch refBranch;
    RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPatternHexahedron();
    refBranch.push_back( make_pair(oneRefinement.get(), 1) );
    refBranch.push_back( make_pair(oneRefinement.get(), 4) );
    
    unsigned coarseSubcellDimension = 2; // face
    unsigned coarseSubcellOrdinalInCoarseDomain = 0;
    unsigned coarseDomainDim = 2;
    unsigned coarseDomainOrdinalInRefinementRoot = 0; // bottom face
    unsigned coarseSubcellPermutation = 0;
    
    BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDim, fineSubcellOrdinalInFineDomain,
                                                            fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
                                                            volumeTopo,
                                                            coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
                                                            coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
    
    out << "UNPERMUTED CASE\n";
    out << "coarseSubcellPoints:\n" << coarseSubcellPoints;
    out << "expectedCoarseSubcellPoints:\n" << expectedCoarseSubcellPoints;
    
    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
    
    // PERMUTED CASE:
    // exactly the same as above, except now the coarse domain is permuted.  This means we expect x and y to be swapped:
    expectedCoarseSubcellPoints(0,0) = -0.25 * fineSubcellPoints(0,0) - 0.75;
    expectedCoarseSubcellPoints(0,1) = 0.0;
    expectedCoarseSubcellPoints(1,0) = -0.25 * fineSubcellPoints(1,0) - 0.75;
    expectedCoarseSubcellPoints(1,1) = 0.0;
    
    coarseSubcellDimension = 2; // face
    coarseSubcellOrdinalInCoarseDomain = 0;
    coarseDomainDim = 2;
    coarseDomainOrdinalInRefinementRoot = 5; // top face
    coarseSubcellPermutation = 4; // would be better to determine the correct permutation programmatically.
    BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseSubcellPoints, fineSubcellPoints, fineSubcellDim, fineSubcellOrdinalInFineDomain,
                                                            fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
                                                            volumeTopo,
                                                            coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
                                                            coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation);
    out << "PERMUTED CASE\n";
    out << "coarseSubcellPoints:\n" << coarseSubcellPoints;
    out << "expectedCoarseSubcellPoints:\n" << expectedCoarseSubcellPoints;
    
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseSubcellPoints, expectedCoarseSubcellPoints, tol);
  }

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseDomain_HexahedronHangingEdge)
{
  // map points on child 1, side 2, edge 3 to the parent's side 1.  In parent reference space, this edge goes from
  // (1,0,-1) to (1,0,0), but it has orientation opposite that in the child's side.  This is a particular
  // case where GDAMinimumRule is failing, and my suspicion is that it has to do with the orientation of that
  // edge.

  int numPoints = 3;
  int fineDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, fineDim);
  fineSubcellPoints(0,0) = -1.0;
  fineSubcellPoints(1,0) =  0.0;
  fineSubcellPoints(2,0) =  1.0;

  int coarseDim = 2;
  FieldContainer<double> expectedCoarseDomainPoints(numPoints,coarseDim);
  for (int coarsePointOrdinal=0; coarsePointOrdinal<numPoints; coarsePointOrdinal++)
  {
    int finePointOrdinal = numPoints - 1 - coarsePointOrdinal; // account for permutation
    expectedCoarseDomainPoints(coarsePointOrdinal,0) = 0.0;
    expectedCoarseDomainPoints(coarsePointOrdinal,1) = 0.5 * (fineSubcellPoints(finePointOrdinal,0) - 1.0);
  }

  unsigned fineSubcellDimension = fineDim; // edge
  unsigned fineSubcellOrdinalInFineDomain = 3; // edge

  unsigned fineDomainDim = 2; // fine domain is 2D
  unsigned fineDomainOrdinalInRefinementLeaf = 2; // side 2

  CellTopoPtr volumeTopo = CellTopology::hexahedron();

  RefinementBranch refBranch;
  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPatternHexahedron();
  refBranch.push_back( make_pair(oneRefinement.get(), 1) ); // child ordinal 1

  unsigned coarseSubcellDimension = 2; // side
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 2;
  unsigned coarseDomainOrdinalInRefinementRoot = 1;
  unsigned coarseSubcellPermutation = 0;

  testMap(expectedCoarseDomainPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
          fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
          volumeTopo,
          coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
          coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseDomain_HexahedronSameSideEdge)
{
  /*
   For an unrefined hexahedron with fine = coarse subcell, but for which the side orientation disagrees with that
   of volume, confirm that fineSubcellPoints and coarseSubcellPoints are identical.

   Side 2, edge 3 of the hexahedron has such an opposite orientation (this is edge 10, from vertex 2 to vertex 6 in hexahedron).
   */

  int numPoints = 3;
  int fineDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, fineDim);
  fineSubcellPoints(0,0) = -1.0;
  fineSubcellPoints(1,0) =  0.0;
  fineSubcellPoints(2,0) =  1.0;


  unsigned fineSubcellDimension = fineDim; // edge
  unsigned fineSubcellOrdinalInFineDomain = 3; // edge

  unsigned fineDomainDim = 2; // fine domain is 2D
  unsigned fineDomainOrdinalInRefinementLeaf = 2; // side 2

  CellTopoPtr volumeTopo = CellTopology::hexahedron();
  CellTopoPtr sideTopo = volumeTopo->getSide(fineDomainOrdinalInRefinementLeaf);

  RefinementBranch refBranch;
  RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
  refBranch.push_back( make_pair(noRefinement.get(), 0) ); // child ordinal 0

  unsigned coarseSubcellDimension = fineSubcellDimension;
  unsigned coarseSubcellOrdinalInCoarseDomain = fineSubcellOrdinalInFineDomain;
  unsigned coarseDomainDim = fineDomainDim;
  unsigned coarseDomainOrdinalInRefinementRoot = fineDomainOrdinalInRefinementLeaf;
  unsigned coarseSubcellPermutation = 0;

  FieldContainer<double> expectedCoarseDomainPoints(numPoints,coarseDomainDim);
  // the coarse domain points expected are the fine subcell points mapped to the fine/coarse reference domain
  CamelliaCellTools::mapToReferenceSubcell(expectedCoarseDomainPoints, fineSubcellPoints,
      fineSubcellDimension, fineSubcellOrdinalInFineDomain, sideTopo);

  testMap(expectedCoarseDomainPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
          fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
          volumeTopo,
          coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
          coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation, out, success);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseDomain_HexahedronSideEdges)
{
  /*
   This test runs through the faces of the hexahedron, and for each considers its edges.  The present
   face is considered the "fine" domain, and the neighboring face along the edge is the coarse domain.
   The coarse and fine subcells of interest are the shared edge.

   We check the mapping by using a BasisCache defined on the volume; we check that the physical points
   for both fine and coarse sides agree.
   */

  int numPoints = 3;
  int edgeDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, edgeDim);
  fineSubcellPoints(0,0) = -1.0;
  fineSubcellPoints(1,0) =  0.0;
  fineSubcellPoints(2,0) =  1.0;

  unsigned fineSubcellDimension = edgeDim; // edge
  unsigned fineDomainDim = 2; // fine domain is 2D
  unsigned coarseDomainDim = fineDomainDim;
  unsigned coarseSubcellDimension = edgeDim; // edge

  CellTopoPtr volumeTopo = CellTopology::hexahedron();
  RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
  RefinementBranch refBranch = {{noRefinement.get(), 0}}; // child ordinal 0

  // we use MeshTopology to find the other side that shares a given edge:
  MeshTopologyPtr meshTopo = noRefinement->refinementMeshTopology();
  IndexType cellIndex = 0;
  CellPtr cell = meshTopo->getCell(cellIndex);
  int sideDim = fineDomainDim;
  bool createSideCache = true;
  BasisCachePtr cellBasisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache); // 1: arbitrary cubature degree
  double tol = 1e-15;

  for (int fineSideOrdinal=0; fineSideOrdinal<volumeTopo->getSideCount(); fineSideOrdinal++)
  {
    CellTopoPtr sideTopo = volumeTopo->getSide(fineSideOrdinal);
    IndexType fineSideEntityIndex = cell->entityIndex(sideDim, fineSideOrdinal);
    for (int fineEdgeOrdinal=0; fineEdgeOrdinal<sideTopo->getEdgeCount(); fineEdgeOrdinal++)
    {
      CellTopoPtr edgeTopo = sideTopo->getSubcell(edgeDim, fineEdgeOrdinal);
      int edgeOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(volumeTopo, sideDim, fineSideOrdinal,
                              edgeDim, fineEdgeOrdinal);
      IndexType edgeEntityIndex = cell->entityIndex(edgeDim, edgeOrdinalInCell);
      vector<IndexType> sidesForEdge = meshTopo->getSidesContainingEntity(edgeDim, edgeEntityIndex);
      if (sidesForEdge.size() != 2)
      {
        out << "Failure: sidesForEdge has unexpected size.\n";
        success = false;
        // quit early if this happens
        return;
      }
      IndexType coarseSideEntityIndex = (sidesForEdge[0] == fineSideEntityIndex) ? sidesForEdge[1] : sidesForEdge[0];
      int coarseSideOrdinal = cell->findSubcellOrdinal(sideDim, coarseSideEntityIndex);
      int coarseEdgeOrdinal = cell->findSubcellOrdinalInSide(edgeDim, edgeEntityIndex, coarseSideOrdinal);

      unsigned fineSubcellOrdinalInFineDomain = fineEdgeOrdinal;
      unsigned coarseSubcellOrdinalInCoarseDomain = coarseEdgeOrdinal;
      unsigned coarseDomainOrdinalInRefinementRoot = coarseSideOrdinal;
      unsigned fineDomainOrdinalInRefinementLeaf = fineSideOrdinal;

      unsigned coarseSubcellPermutation = 0; // since fine ancestral cell and coarse cell are the same, 0.

      FieldContainer<double> coarseDomainPoints(numPoints,coarseDomainDim);
      BasisReconciliation::mapFineSubcellPointsToCoarseDomain(coarseDomainPoints, fineSubcellPoints,
          fineSubcellDimension, fineSubcellOrdinalInFineDomain,
          fineDomainDim, fineDomainOrdinalInRefinementLeaf,
          refBranch, volumeTopo,
          coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
          coarseDomainDim, coarseDomainOrdinalInRefinementRoot,
          coarseSubcellPermutation);

      // Now that we've mapped the *subcell* points, map them into their respective side BasisCaches,
      // and check that the physical points mapped are the same.
      BasisCachePtr fineCache = cellBasisCache->getSideBasisCache(fineSideOrdinal);
      BasisCachePtr coarseCache = cellBasisCache->getSideBasisCache(coarseSideOrdinal);

      FieldContainer<double> fineDomainPoints(numPoints,fineDomainDim);
      CamelliaCellTools::mapToReferenceSubcell(fineDomainPoints, fineSubcellPoints, edgeDim, fineEdgeOrdinal, sideTopo);
      fineCache->setRefCellPoints(fineDomainPoints);

      coarseCache->setRefCellPoints(coarseDomainPoints);

      FieldContainer<double> finePhysicalPoints = fineCache->getPhysicalCubaturePoints();
      FieldContainer<double> coarsePhysicalPoints = coarseCache->getPhysicalCubaturePoints();

      out << "Mapping side ordinal " << fineSideOrdinal << ", edge " << fineEdgeOrdinal;
      out << " to side ordinal " << coarseSideOrdinal << ", edge " << coarseEdgeOrdinal;
      out << " (permutation = " << coarseSubcellPermutation << ")\n";

      out << "fine subcell points:\n" << fineSubcellPoints;
      out << "fine domain points:\n" << fineDomainPoints;

      out << "coarse domain points:\n" << coarseDomainPoints;

      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(finePhysicalPoints, coarsePhysicalPoints, tol);
    }
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, MapFineSubcellPointsToCoarseSubcell_InteriorTriangleSideToTriangleVolume)
{
  // map two points on edge of interior triangle to the parent
  // we take the 0 edge on the interior triangle, which extends from (0.5,0) to (0.5,0.5) in parent

  int numPoints = 2;
  int fineDim  = 1;
  FieldContainer<double> fineSubcellPoints(numPoints, fineDim);
  fineSubcellPoints(0,0) = -0.25;
  fineSubcellPoints(1,0) = 0.5;

  int coarseDim = 2;
  FieldContainer<double> expectedCoarseSubcellPoints(numPoints,coarseDim);
  expectedCoarseSubcellPoints(0,0) = 0.5;
  expectedCoarseSubcellPoints(0,1) = 0.25 * fineSubcellPoints(0,0) + 0.25;
  expectedCoarseSubcellPoints(1,0) = 0.5;
  expectedCoarseSubcellPoints(1,1) = 0.25 * fineSubcellPoints(1,0) + 0.25;

  unsigned fineSubcellDimension = fineDim; // edge
  unsigned fineSubcellOrdinalInFineDomain = 0; // edge

  unsigned fineDomainDim = fineDim; // fine domain is 1D
  unsigned fineDomainOrdinalInRefinementLeaf = 0; // side 0

  CellTopoPtr volumeTopo = CellTopology::triangle();

  RefinementBranch refBranch;
  RefinementPatternPtr oneRefinement = RefinementPattern::regularRefinementPatternTriangle();
  refBranch.push_back( make_pair(oneRefinement.get(), 1) ); // child ordinal 1 is the interior triangle

  unsigned coarseSubcellDimension = 2; // volume
  unsigned coarseSubcellOrdinalInCoarseDomain = 0;
  unsigned coarseDomainDim = 2;
  unsigned coarseDomainOrdinalInRefinementRoot = 0;
  unsigned coarseSubcellPermutation = 0;

  testMap(expectedCoarseSubcellPoints, fineSubcellPoints, fineSubcellDimension, fineSubcellOrdinalInFineDomain,
          fineDomainDim, fineDomainOrdinalInRefinementLeaf, refBranch,
          volumeTopo,
          coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
          coarseDomainDim, coarseDomainOrdinalInRefinementRoot, coarseSubcellPermutation, out, success);
}

TEUCHOS_UNIT_TEST(BasisReconciliation, p)
{
  // copied from DPGTests's BasisReconciliationTests::testP()
  int fineOrder = 5;
  int coarseOrder = 3;

  vector< pair< BasisPtr, BasisPtr > > basisPairsToCheck;

  BasisPtr fineBasis = Camellia::intrepidLineHGRAD(fineOrder);
  BasisPtr coarseBasis = Camellia::intrepidLineHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );

  fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );

  //  cout << "WARNING: commented out tests in BasisReconciliationTests::testP() for HDIV basis tests that fail.\n";
  fineBasis = Camellia::intrepidQuadHDIV(fineOrder);
  coarseBasis = Camellia::intrepidQuadHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );

  fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );

  fineBasis = Camellia::intrepidHexHDIV(fineOrder);
  coarseBasis = Camellia::intrepidHexHDIV(coarseOrder);
  basisPairsToCheck.push_back( make_pair(fineBasis, coarseBasis) );

  for (vector< pair< BasisPtr, BasisPtr > >::iterator bpIt = basisPairsToCheck.begin(); bpIt != basisPairsToCheck.end(); bpIt++)
  {
    fineBasis = bpIt->first;
    coarseBasis = bpIt->second;
    testPConstraintInternalBasis(fineBasis, coarseBasis, out, success);
  }
}

TEUCHOS_UNIT_TEST(BasisReconciliation, pSide)
{
  // Copied from DPGTests's BasisReconciliationTests::testPSide()

  int fineOrder = 3;
  int coarseOrder = 3;

  double width = 2;
  double height = 4;
  FieldContainer<double> centeredQuad(4,2);
  centeredQuad(0,0) = -width / 2;
  centeredQuad(0,1) = -height / 2;
  centeredQuad(1,0) = width / 2;
  centeredQuad(1,1) = -height / 2;
  centeredQuad(2,0) = width / 2;
  centeredQuad(2,1) = height / 2;
  centeredQuad(3,0) = -width / 2;
  centeredQuad(3,1) = height / 2;

  FieldContainer<double> eastQuad = translateQuad(centeredQuad, width, 0);
  FieldContainer<double> westQuad = translateQuad(centeredQuad, -width, 0);
  FieldContainer<double> southQuad = translateQuad(centeredQuad, -height, 0);
  FieldContainer<double> northQuad = translateQuad(centeredQuad, height, 0);

  pSideTest test;

  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;

  vector< pSideTest > sideTests;

  test.fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = EAST;
  test.coarseCellNodes = eastQuad;
  test.coarseSideIndex = WEST;

  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westQuad;
  test.coarseSideIndex = EAST;

  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidQuadHDIV(fineOrder);
  test.coarseBasis = Camellia::intrepidQuadHDIV(coarseOrder);
  test.fineCellNodes = centeredQuad;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westQuad;
  test.coarseSideIndex = EAST;

  sideTests.push_back(test);

  /* ************** HEXES ************* */

  // redefine NORTH, SOUTH, EAST and WEST:
  NORTH = 2;
  SOUTH = 0;
  EAST = 1;
  WEST = 3;
  int FRONT = 4; // FRONT is -z direction
  int BACK = 5; // BACK is +z direction

  FieldContainer<double> centeredHex(8,3);
  double depth = 8;
  centeredHex(0,0) = -width / 2;
  centeredHex(0,1) = -height / 2;
  centeredHex(0,2) = -depth / 2;

  centeredHex(1,0) =  width / 2;
  centeredHex(1,1) = -height / 2;
  centeredHex(1,2) = -depth / 2;

  centeredHex(2,0) =  width / 2;
  centeredHex(2,1) =  height / 2;
  centeredHex(2,2) = -depth / 2;

  centeredHex(3,0) = -width / 2;
  centeredHex(3,1) =  height / 2;
  centeredHex(3,2) = -depth / 2;

  centeredHex(4,0) = -width / 2;
  centeredHex(4,1) = -height / 2;
  centeredHex(4,2) =  depth / 2;

  centeredHex(5,0) =  width / 2;
  centeredHex(5,1) = -height / 2;
  centeredHex(5,2) =  depth / 2;

  centeredHex(6,0) =  width / 2;
  centeredHex(6,1) =  height / 2;
  centeredHex(6,2) =  depth / 2;

  centeredHex(7,0) = -width / 2;
  centeredHex(7,1) =  height / 2;
  centeredHex(7,2) =  depth / 2;

  FieldContainer<double> eastHex = translateHex(centeredHex, width, 0, 0);
  FieldContainer<double> westHex = translateHex(centeredHex, -width, 0, 0);
  FieldContainer<double> northHex = translateHex(centeredHex, 0, height, 0);
  FieldContainer<double> southHex = translateHex(centeredHex, 0, -height, 0);
  FieldContainer<double> frontHex = translateHex(centeredHex, 0, 0, -depth);
  FieldContainer<double> backHex = translateHex(centeredHex, 0, 0, depth);

  test.fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westHex;
  test.coarseSideIndex = EAST;

  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidHexHGRAD(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHGRAD(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = BACK;
  test.coarseCellNodes = backHex;
  test.coarseSideIndex = FRONT;

  sideTests.push_back(test);

  test.fineBasis = Camellia::intrepidHexHDIV(fineOrder);
  test.coarseBasis = Camellia::intrepidHexHDIV(coarseOrder);
  test.fineCellNodes = centeredHex;
  test.fineSideIndex = WEST;
  test.coarseCellNodes = westHex;
  test.coarseSideIndex = EAST;

  sideTests.push_back(test);

  for (vector< pSideTest >::iterator testIt = sideTests.begin(); testIt != sideTests.end(); testIt++)
  {
    test = *testIt;

    testPConstraintSideBasis(test.fineBasis, test.fineSideIndex, test.fineCellNodes,
                             test.coarseBasis, test.coarseSideIndex, test.coarseCellNodes,
                             out, success);
  }
}

void equispacedPoints(int numPoints1D, CellTopoPtr cellTopo, FieldContainer<double> &points)
{
  if (cellTopo->getDimension() == 0)
  {
    points.resize(1,1);
  }
  else if (cellTopo->getDimension() == 1)
  {
    // compute some equispaced points on the reference line:
    points.resize(numPoints1D, cellTopo->getDimension());
    for (int pointOrdinal=0; pointOrdinal < numPoints1D; pointOrdinal++)
    {
      int d = 0;
      points(pointOrdinal,d) = -1.0 + pointOrdinal * (2.0 / (numPoints1D - 1));
    }
  }
  else if (cellTopo->getKey() == CellTopology::quad()->getKey())
  {
    points.resize(numPoints1D * numPoints1D, cellTopo->getDimension());
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled CellTopology");
  }
}

void termTracedTest(Teuchos::FancyOStream &out, bool &success, CellTopoPtr volumeTopo, VarType traceOrFluxType)
{
  int spaceDim = volumeTopo->getDimension();

  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr fieldVar, traceVar;
  if ((traceOrFluxType != FLUX) && (traceOrFluxType != TRACE))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "type must be flux or trace");
  }
  else if (spaceDim==1)
  {
    fieldVar = vf->fieldVar("psi", L2);
    FunctionPtr n = Function::normal_1D();
    FunctionPtr parity = Function::sideParity();
    LinearTermPtr fluxTermTraced = 3.0 * n * parity * fieldVar;
    traceVar = vf->fluxVar("\\widehat{\\psi}_n", fluxTermTraced);
  }
  else if (traceOrFluxType==FLUX)
  {
    fieldVar = vf->fieldVar("psi", VECTOR_L2);
    FunctionPtr n = Function::normal();
    FunctionPtr parity = Function::sideParity();
    LinearTermPtr fluxTermTraced = 3.0 * n * parity * fieldVar;
    traceVar = vf->fluxVar("\\widehat{\\psi}_n", fluxTermTraced);
  }
  else
  {
    fieldVar = vf->fieldVar("u");
    LinearTermPtr termTraced = 3.0 * fieldVar;
    traceVar = vf->traceVar("\\widehat{u}", termTraced);
  }

  // in what follows, the fine basis belongs to the trace variable and the coarse to the field

  unsigned coarseSubcellOrdinalInCoarseDomain = 0;  // when the coarse domain is a side, this can be non-zero, but fields are always in the volume, so this will always be 0
  unsigned coarseDomainOrdinalInRefinementRoot = 0; // again, since coarse domain is for a field variable, this will always be 0
  unsigned coarseSubcellPermutation = 0;            // not sure if this will ever be nontrivial permutation in practice; we don't test anything else here

  int H1Order = 2;
  int numPoints1D = 5;

  BasisPtr volumeBasis;
  if (fieldVar->rank() == 0)
    volumeBasis = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_HVOL);
  else
    volumeBasis = BasisFactory::basisFactory()->getBasis(H1Order, volumeTopo, Camellia::FUNCTION_SPACE_VECTOR_HVOL);

  RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(volumeTopo);
  RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);

  RefinementBranch noRefinements = {{noRefinement.get(), 0}};
  RefinementBranch oneRefinement = {{regularRefinement.get(), 1}}; // 1: select child ordinal 1
//    RefinementBranch twoRefinements = {{regularRefinement.get(), 1}, {regularRefinement.get(),0}}; // select child 1 in first refinement, 0 in second

//    vector<RefinementBranch> refinementBranches = {noRefinements,oneRefinement,twoRefinements};

  vector<RefinementBranch> refinementBranches = {noRefinements,oneRefinement};

  FieldContainer<double> volumeRefNodes(volumeTopo->getVertexCount(), volumeTopo->getDimension());

  CamelliaCellTools::refCellNodesForTopology(volumeRefNodes, volumeTopo);

  bool createSideCache = true;
  BasisCachePtr volumeBasisCache = BasisCache::basisCacheForReferenceCell(volumeTopo, 1, createSideCache);

  for (int i=0; i< refinementBranches.size(); i++)
  {
    RefinementBranch refBranch = refinementBranches[i];

    FieldContainer<double> refinedNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refBranch);
    addCellDimensionToFC(refinedNodes);
    volumeBasisCache->setPhysicalCellNodes(refinedNodes, {}, createSideCache);

    out << "***** Refinement Type Number " << i << " *****\n";

    for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++)
    {
      FieldContainer<double> tracePointsSideReferenceSpace;
      CellTopoPtr sideTopo = volumeTopo->getSubcell(spaceDim-1, traceSideOrdinal);
      equispacedPoints(numPoints1D, sideTopo, tracePointsSideReferenceSpace);
      int numPoints = tracePointsSideReferenceSpace.dimension(0);

      // choose trace basis in much the way that we would in actual applications:
      Camellia::EFunctionSpace traceFS = (traceOrFluxType==TRACE) ? Camellia::FUNCTION_SPACE_HGRAD : Camellia::FUNCTION_SPACE_HVOL;
      BasisPtr traceBasis = BasisFactory::basisFactory()->getBasis(H1Order, sideTopo, traceFS);

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
      weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(traceVar->termTraced(), fieldVar->ID(),
                sideTopo->getDimension(),
                traceBasis, fineSubcellOrdinalInFineDomain, refBranch,
                traceSideOrdinal,
                volumeTopo,
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

      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(coarseValuesExpected, coarseValuesActual, tol);
    }
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_1D_new )
{
  CellTopoPtr lineTopo = CellTopology::line();
  termTracedTest(out,success,lineTopo,TRACE);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_1D )
{
  // TODO: rewrite this to use termTracedTest(), as in TermTraced_2D tests, below
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u");
  LinearTermPtr termTraced = 3.0 * u;
  VarPtr u_hat = vf->traceVar("\\widehat{u}", termTraced);

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

  for (int i=0; i< refinementBranches.size(); i++)
  {
    RefinementBranch refBranch = refinementBranches[i];

    for (int fineVertexOrdinal=0; fineVertexOrdinal <= 1; fineVertexOrdinal++)
    {
      int fineSubcellOrdinalInFineDomain = 0;

      int numPoints = 1;
      FieldContainer<double> vertexPointInLeaf(numPoints, lineTopo->getDimension());
      for (int d=0; d < lineTopo->getDimension(); d++)
      {
        vertexPointInLeaf(0,d) = lineRefNodes(fineVertexOrdinal,d);
      }

      FieldContainer<double> vertexPointInAncestor(numPoints, lineTopo->getDimension());
      RefinementPattern::mapRefCellPointsToAncestor(refBranch, vertexPointInLeaf, vertexPointInAncestor);

      lineBasisCache->setRefCellPoints(vertexPointInAncestor);

      SubBasisReconciliationWeights weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, u->ID(), pointTopo->getDimension(),
                                              pointBasis, fineSubcellOrdinalInFineDomain, refBranch, fineVertexOrdinal,
                                              lineTopo,
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
      for (int coarseOrdinal=0; coarseOrdinal < lineBasisQuadratic->getCardinality(); coarseOrdinal++)
      {
        double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);

        double actualValue;
        if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end())
        {
          actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
          coarseOrdinalInWeights++;
        }
        else
        {
          actualValue = 0.0;
        }

        if (abs(expectedValue) > tol )
        {
          TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
        }
        else
        {
          TEST_ASSERT( abs(actualValue) < tol );
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D_Quad )
{
  CellTopoPtr quadTopo = CellTopology::quad();
  termTracedTest(out,success,quadTopo,TRACE);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D_Quad_Flux )
{
  CellTopoPtr quadTopo = CellTopology::quad();
  termTracedTest(out,success,quadTopo,FLUX);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D_Triangle )
{
  CellTopoPtr triangleTopo = CellTopology::triangle();
  termTracedTest(out,success,triangleTopo,TRACE);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_2D_Triangle_Flux )
{
  CellTopoPtr triangleTopo = CellTopology::triangle();
  termTracedTest(out,success,triangleTopo,FLUX);
}

TEUCHOS_UNIT_TEST( BasisReconciliation, TermTraced_3D_Hexahedron_Slow)
{
  // TODO: rewrite this to use termTracedTest(), as in TermTraced_2D tests, above
  // TODO: add Hexahedron flux test
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u");
  LinearTermPtr termTraced = 3.0 * u;
  VarPtr u_hat = vf->traceVar("\\widehat{u}", termTraced);

  // TODO: do flux tests...
  //    LinearTermPtr fluxTermTraced = 3.0 * u * n;
  //    VarPtr u_n = vf->traceVar("\\widehat{u}", termTraced);

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

  for (int i=0; i< refinementBranches.size(); i++)
  {
    RefinementBranch refBranch = refinementBranches[i];

    out << "***** Refinement Type Number " << i << " *****\n";

    // compute some equispaced points on the reference quad:
    int numPoints_x = 5, numPoints_y = 5;
    FieldContainer<double> tracePointsSideReferenceSpace(numPoints_x * numPoints_y, sideTopo->getDimension());
    int pointOrdinal = 0;
    for (int pointOrdinal_x=0; pointOrdinal_x < numPoints_x; pointOrdinal_x++)
    {
      double x = -1.0 + pointOrdinal_x * (2.0 / (numPoints_x - 1));
      for (int pointOrdinal_y=0; pointOrdinal_y < numPoints_y; pointOrdinal_y++, pointOrdinal++)
      {
        double y = -1.0 + pointOrdinal_y * (2.0 / (numPoints_y - 1));
        tracePointsSideReferenceSpace(pointOrdinal,0) = x;
        tracePointsSideReferenceSpace(pointOrdinal,1) = y;
      }
    }

    int numPoints = numPoints_x * numPoints_y;

    for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++)
    {
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
                                              volumeTopo,
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

      for (int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++)
      {
        int coarseOrdinalInWeights = 0;
        for (int coarseOrdinal=0; coarseOrdinal < volumeBasisQuadratic->getCardinality(); coarseOrdinal++)
        {
          double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);

          double actualValue;
          if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end())
          {
            actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
            coarseOrdinalInWeights++;
          }
          else
          {
            actualValue = 0.0;
          }

          if ( abs(expectedValue) > tol )
          {
            TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
          }
          else
          {
            TEST_ASSERT( abs(actualValue) < tol );

            if (abs(actualValue) >= tol)
            {
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
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u");
  LinearTermPtr termTraced = 3.0 * u;
  VarPtr u_hat = vf->traceVar("\\widehat{u}", termTraced);

  // TODO: do flux tests...
  //    LinearTermPtr fluxTermTraced = 3.0 * u * n;
  //    VarPtr u_n = vf->traceVar("\\widehat{u}", termTraced);

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

  for (int i=0; i< refinementBranches.size(); i++)
  {
    RefinementBranch refBranch = refinementBranches[i];

    out << "***** Refinement Type Number " << i << " *****\n";

    // compute some equispaced points on the reference triangle:
    int numPoints_x = 5, numPoints_y = 5;
    FieldContainer<double> tracePointsSideReferenceSpace(numPoints_x * numPoints_y, sideTopo->getDimension());
    int pointOrdinal = 0;
    for (int pointOrdinal_x=0; pointOrdinal_x < numPoints_x; pointOrdinal_x++)
    {
      double x = 0.0 + pointOrdinal_x * (1.0 / (numPoints_x - 1));
      for (int pointOrdinal_y=0; pointOrdinal_y < numPoints_y; pointOrdinal_y++, pointOrdinal++)
      {
        double y = 0.0 + pointOrdinal_y * (1.0 / (numPoints_y - 1));

        // (x,y) lies in the unit quad.  Divide one coordinate by 2 to get into the ref. triangle...
        if ((pointOrdinal % 2) == 0)
        {
          tracePointsSideReferenceSpace(pointOrdinal,0) = x;
          tracePointsSideReferenceSpace(pointOrdinal,1) = y / 2.0;
        }
        else
        {
          tracePointsSideReferenceSpace(pointOrdinal,0) = x / 2.0;
          tracePointsSideReferenceSpace(pointOrdinal,1) = y;
        }
      }
    }

    int numPoints = numPoints_x * numPoints_y;

    for (int traceSideOrdinal=0; traceSideOrdinal < volumeTopo->getSideCount(); traceSideOrdinal++)
    {
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
                                              volumeTopo,
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

      for (int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++)
      {
        int coarseOrdinalInWeights = 0;
        for (int coarseOrdinal=0; coarseOrdinal < volumeBasisQuadratic->getCardinality(); coarseOrdinal++)
        {
          double expectedValue = coarseValuesExpected(0,coarseOrdinal,pointOrdinal);

          double actualValue;
          if (weights.coarseOrdinals.find(coarseOrdinal) != weights.coarseOrdinals.end())
          {
            actualValue = coarseValuesActual(coarseOrdinalInWeights,pointOrdinal);
            coarseOrdinalInWeights++;
          }
          else
          {
            actualValue = 0.0;
          }

          if ( abs(expectedValue) > tol )
          {
            TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
          }
          else
          {
            TEST_ASSERT( abs(actualValue) < tol );

            if (abs(actualValue) >= tol)
            {
              out << "coarseOrdinal " << coarseOrdinal << ", point " << pointOrdinal << " on side " << traceSideOrdinal << ", actualValue = " << actualValue << endl;
            }
          }
        }
      }
    }
  }
}
} // namespace
