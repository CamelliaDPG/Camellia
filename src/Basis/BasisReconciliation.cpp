//
//  BasisReconciliation.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#include "BasisReconciliation.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h" // includes print() methods
#include "CellTopology.h"
#include "SerialDenseMatrixUtility.h"
#include "SerialDenseWrapper.h"
#include "TensorBasis.h"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"

using namespace Intrepid;
using namespace Camellia;

void sizeFCForBasisValues(FieldContainer<double> &fc, BasisPtr basis, int numPoints, bool includeCellDimension = false, int numBasisFieldsToInclude = -1)
{
  // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
  Teuchos::Array<int> dim;
  if (includeCellDimension)
  {
    dim.push_back(1);
  }
  if (numBasisFieldsToInclude == -1)
  {
    dim.push_back(basis->getCardinality()); // F
  }
  else
  {
    dim.push_back(numBasisFieldsToInclude);
  }
  dim.push_back(numPoints); // P
  for (int d=0; d<basis->rangeRank(); d++)
  {
    dim.push_back(basis->rangeDimension()); // D
  }
  fc.resize(dim);
}

void sizeFCForTermTracedValues(FieldContainer<double> &fc, LinearTermPtr termTraced, BasisPtr basis, int numPoints, bool includeCellDimension = false, int numBasisFieldsToInclude = -1)
{
  // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
  Teuchos::Array<int> dim;
  if (includeCellDimension)
  {
    dim.push_back(1);
  }
  if (numBasisFieldsToInclude == -1)
  {
    dim.push_back(basis->getCardinality()); // F
  }
  else
  {
    dim.push_back(numBasisFieldsToInclude);
  }
  dim.push_back(numPoints); // P
  for (int d=0; d < termTraced->rank(); d++)
  {
    dim.push_back(basis->rangeDimension()); // D
  }
  fc.resize(dim);
}

void filterFCValues(FieldContainer<double> &filteredFC, const FieldContainer<double> &fc, set<int> ordinals, int basisCardinality)
{
  // we use pointer arithmetic here, which doesn't allow Intrepid's safety checks, for two reasons:
  // 1. It's easier to manage in a way that's independent of the rank of the basis
  // 2. It's faster.
  int numEntriesPerBasisField = fc.size() / basisCardinality;
  int filteredFCIndex = 0;
  for (set<int>::iterator ordinalIt = ordinals.begin(); ordinalIt != ordinals.end(); ordinalIt++)
  {
    int ordinal = *ordinalIt;
    const double *fcEntry = &fc[ordinal * numEntriesPerBasisField];
    double *filteredEntry = &filteredFC[ filteredFCIndex * numEntriesPerBasisField ];
    for (int i=0; i<numEntriesPerBasisField; i++)
    {
      *filteredEntry = *fcEntry;
      filteredEntry++;
      fcEntry++;
    }

    filteredFCIndex++;
  }
}

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
  return -1; // just for compilers that would otherwise warn that we're missing a return value...
}

SubBasisReconciliationWeights BasisReconciliation::composedSubBasisReconciliationWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights)
{
  // first render aWeights and bWeights compatible by intersecting aWeights.coarseOrdinals with bWeights.fineOrdinals
  set<int> aColOrdinalsToInclude, bRowOrdinalsToInclude;
  vector<int> aCoarseOrdinalsVector(aWeights.coarseOrdinals.begin(),aWeights.coarseOrdinals.end());
  for (int j=0; j<aCoarseOrdinalsVector.size(); j++)
  {
    int aCoarseBasisOrdinal = aCoarseOrdinalsVector[j];
    if (bWeights.fineOrdinals.find(aCoarseBasisOrdinal) != bWeights.fineOrdinals.end())
    {
      aColOrdinalsToInclude.insert(j);
    }
  }
  vector<int> bFineOrdinalsVector(bWeights.fineOrdinals.begin(), bWeights.fineOrdinals.end());
  for (int i=0; i<bFineOrdinalsVector.size(); i++)
  {
    int bFineBasisOrdinal = bFineOrdinalsVector[i];
    if (aWeights.coarseOrdinals.find(bFineBasisOrdinal) != aWeights.coarseOrdinals.end())
    {
      bRowOrdinalsToInclude.insert(i);
    }
  }

  set<int> aAllRows, bAllCols;
  for (int i=0; i<aWeights.fineOrdinals.size(); i++)
  {
    aAllRows.insert(i);
  }
  for (int j=0; j<bWeights.coarseOrdinals.size(); j++)
  {
    bAllCols.insert(j);
  }

//  if (aWeights.coarseOrdinals.size() != bWeights.fineOrdinals.size()) {
//    cout << "DEBUGGING: aWeights and bWeights have different coarse/fine sizes before filter to intersect.\n";
//  }
//
//  cout << "aWeights, bWeights before filtering to intersect:\n";
//
//  Camellia::print("a: fine ordinals", aWeights.fineOrdinals);
//  Camellia::print("a: coarse ordinals", aWeights.coarseOrdinals);
//  cout << "a: weights:\n" << aWeights.weights;
//
//  Camellia::print("b: fine ordinals", bWeights.fineOrdinals);
//  Camellia::print("b: coarse ordinals", bWeights.coarseOrdinals);
//  cout << "b: weights:\n" << bWeights.weights;

  aWeights = filterToInclude(aAllRows, aColOrdinalsToInclude, aWeights);
  bWeights = filterToInclude(bRowOrdinalsToInclude, bAllCols, bWeights);

//  cout << "aWeights, bWeights after filtering to intersect:\n";
//
//  Camellia::print("a: fine ordinals", aWeights.fineOrdinals);
//  Camellia::print("a: coarse ordinals", aWeights.coarseOrdinals);
//  cout << "a: weights:\n" << aWeights.weights;
//
//  Camellia::print("b: fine ordinals", bWeights.fineOrdinals);
//  Camellia::print("b: coarse ordinals", bWeights.coarseOrdinals);
//  cout << "b: weights:\n" << bWeights.weights;

  if (aWeights.coarseOrdinals.size() != bWeights.fineOrdinals.size())
  {
    cout << "aWeights and bWeights are incompatible...\n";

    Camellia::print("a: fine ordinals", aWeights.fineOrdinals);
    Camellia::print("a: coarse ordinals", aWeights.coarseOrdinals);
    cout << "a: weights:\n" << aWeights.weights;

    Camellia::print("b: fine ordinals", bWeights.fineOrdinals);
    Camellia::print("b: coarse ordinals", bWeights.coarseOrdinals);
    cout << "b: weights:\n" << bWeights.weights;

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "aWeights and bWeights are incompatible...");
  }
  SubBasisReconciliationWeights cWeights;
  cWeights.weights = FieldContainer<double>(aWeights.fineOrdinals.size(), bWeights.coarseOrdinals.size());
  if (cWeights.weights.size() != 0)
  {
    FieldContainer<double> aMatrix = aWeights.weights;
    FieldContainer<double> bMatrix = bWeights.weights;
    SerialDenseWrapper::multiply(cWeights.weights, aWeights.weights, bWeights.weights);
  }
  cWeights.fineOrdinals = aWeights.fineOrdinals;
  cWeights.coarseOrdinals = bWeights.coarseOrdinals;
  return filterOutZeroRowsAndColumns(cWeights);
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(unsigned subcellDimension,
                                                                             BasisPtr finerBasis,
                                                                             unsigned finerBasisSubcellOrdinal,
                                                                             RefinementBranch &refinements,
                                                                             BasisPtr coarserBasis,
                                                                             unsigned coarserBasisSubcellOrdinal,
                                                                             unsigned vertexNodePermutation)
{
  SubBasisReconciliationWeights weights;

  // use the functionSpace to determine what continuities should be enforced:
  Camellia::EFunctionSpace fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");

  if (fs==Camellia::FUNCTION_SPACE_REAL_SCALAR)
  {
    if (((finerBasisSubcellOrdinal==0) && (coarserBasisSubcellOrdinal==0)) && (subcellDimension==0) && (refinements.size()==0)
        && (finerBasis->getCardinality()==1) && (coarserBasis->getCardinality()==1))
    {
      // then we're just matching a single scalar with another
      set<int> zeroSet;
      zeroSet.insert(0);
      weights.coarseOrdinals = zeroSet;
      weights.fineOrdinals = zeroSet;
      weights.weights.resize(1, 1);
      weights.weights.initialize(1.0);
      return weights;
    }
    else
    {
      cout << "Encountered basis with function space FUNCTION_SPACE_REAL_SCALAR with unsupported arguments.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Encountered basis with function space FUNCTION_SPACE_REAL_SCALAR with unsupported arguments");
    }
  }

  int spaceDim = finerBasis->domainTopology()->getDimension();

  // figure out ancestralSubcellOrdinal
  unsigned ancestralSubcellOrdinal = RefinementPattern::ancestralSubcellOrdinal(refinements, subcellDimension, finerBasisSubcellOrdinal);

  if (ancestralSubcellOrdinal == -1)
  {
    cout << "ancestralSubcellOrdinal not found!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestralSubcellOrdinal not found!");
  }

  //  set<unsigned> fineOrdinalsUnsigned = internalDofOrdinalsForFinerBasis(finerBasis, refinements, subcellDimension, finerBasisSubcellOrdinal);
  //  weights.fineOrdinals.insert(fineOrdinalsUnsigned.begin(),fineOrdinalsUnsigned.end());

  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(subcellDimension, finerBasisSubcellOrdinal, 0);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(subcellDimension,coarserBasisSubcellOrdinal,0);

  if (weights.fineOrdinals.size() == 0)
  {
    if (weights.coarseOrdinals.size() != 0)
    {
      cout << "fineOrdinals determined to be empty when coarseOrdinals is not!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fineOrdinals determined to be empty when coarseOrdinals is not!");
    }
    weights.weights.resize(0, 0);
    return weights;
  }

  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself (this is overkill; we could likely get away with less by reasoning about the cardinality of fineOrdinals)

  // determine cubature points as seen by the fine basis
  CellTopoPtr fineTopo = finerBasis->domainTopology();
  CellTopoPtr fineSubcellTopo = fineTopo->getSubcell(subcellDimension, finerBasisSubcellOrdinal);

  CellTopoPtr coarseTopo = coarserBasis->domainTopology();

  FieldContainer<double> fineTopoRefNodes(fineTopo->getVertexCount(), fineTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(fineTopoRefNodes, fineTopo);
  FieldContainer<double> coarseTopoRefNodes(coarseTopo->getVertexCount(), coarseTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseTopoRefNodes, coarseTopo);

  FieldContainer<double> coarseVolumeCubaturePoints;
  FieldContainer<double> fineVolumeCubaturePoints; // points at which the fine basis will be evaluated
  FieldContainer<double> cubatureWeights;

  if (subcellDimension > 0)
  {
    RefinementBranch subcellRefinements = RefinementPattern::subcellRefinementBranch(refinements, subcellDimension, ancestralSubcellOrdinal);
    FieldContainer<double> fineSubcellRefNodes;
    if (subcellRefinements.size() > 0)
    {
      fineSubcellRefNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(subcellRefinements);
    }
    else
    {
      fineSubcellRefNodes.resize(fineSubcellTopo->getVertexCount(),fineSubcellTopo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(fineSubcellRefNodes, fineSubcellTopo);
    }
    fineSubcellRefNodes.resize(1,fineSubcellRefNodes.dimension(0),fineSubcellRefNodes.dimension(1));
//    fineSubcellCache->setPhysicalCellNodes(fineSubcellRefNodes, vector<GlobalIndexType>(), false);

    BasisCachePtr fineSubcellCache = BasisCache::basisCacheForCellTopology(fineSubcellTopo, cubDegree, fineSubcellRefNodes);
//    BasisCachePtr fineSubcellCache = Teuchos::rcp( new BasisCache(fineSubcellTopo, cubDegree, false) );
    int numPoints = fineSubcellCache->getRefCellPoints().dimension(0);
    fineVolumeCubaturePoints.resize(numPoints,spaceDim);
    FieldContainer<double> fineSubcellCubaturePoints = fineSubcellCache->getRefCellPoints();
    cubatureWeights = fineSubcellCache->getCubatureWeights();
    if (subcellDimension == spaceDim)
    {
      fineVolumeCubaturePoints = fineSubcellCubaturePoints;
    }
    else
    {
      CamelliaCellTools::mapToReferenceSubcell(fineVolumeCubaturePoints, fineSubcellCubaturePoints, subcellDimension,
                                               finerBasisSubcellOrdinal, fineTopo);
    }

    // now, fineSubcellCache's physicalCubaturePoints are exactly the ones in the ancestral subcell
    FieldContainer<double> ancestralSubcellCubaturePoints = fineSubcellCache->getPhysicalCubaturePoints();
    ancestralSubcellCubaturePoints.resize(ancestralSubcellCubaturePoints.dimension(1), ancestralSubcellCubaturePoints.dimension(2)); // get rid of cellIndex dimension

    // now, we need first to map the ancestralSubcellCubaturePoints to the subcell as seen by the coarse neigbor, then to map those points into the coarse volume topology

    CellTopoPtr ancestralSubcellTopo = (subcellRefinements.size() > 0) ? subcellRefinements[0].first->parentTopology() : fineSubcellTopo;
    BasisCachePtr ancestralSubcellCache = Teuchos::rcp( new BasisCache(ancestralSubcellTopo, cubDegree, false) );
    ancestralSubcellCache->setRefCellPoints(ancestralSubcellCubaturePoints);

    FieldContainer<double> coarseSubcellNodes(ancestralSubcellTopo->getNodeCount(), subcellDimension);
    // when you set physical cell nodes according to the coarse-to-fine permutation, then the reference-to-physical map
    // is fine-to-coarse (which is what we want).  Because the vertexNodePermutation is fine-to-coarse, we want its inverse:
    unsigned permutationInverse = CamelliaCellTools::permutationInverse(ancestralSubcellTopo, vertexNodePermutation);
    CamelliaCellTools::refCellNodesForTopology(coarseSubcellNodes, ancestralSubcellTopo, permutationInverse);
    coarseSubcellNodes.resize(1,coarseSubcellNodes.dimension(0),coarseSubcellNodes.dimension(1));

    ancestralSubcellCache->setPhysicalCellNodes(coarseSubcellNodes, vector<GlobalIndexType>(), false);
    FieldContainer<double> coarseSubcellCubaturePoints = ancestralSubcellCache->getPhysicalCubaturePoints();
    coarseSubcellCubaturePoints.resize(coarseSubcellCubaturePoints.dimension(1),coarseSubcellCubaturePoints.dimension(2));

    coarseVolumeCubaturePoints.resize(numPoints,spaceDim); // points at which the coarse basis will be evaluated
    if (subcellDimension == spaceDim)
    {
      coarseVolumeCubaturePoints = coarseSubcellCubaturePoints;
    }
    else
    {
      CamelliaCellTools::mapToReferenceSubcell(coarseVolumeCubaturePoints, coarseSubcellCubaturePoints,
          subcellDimension, coarserBasisSubcellOrdinal, coarseTopo);
    }
  }
  else     // subcellDimension == 0 --> vertex
  {
    fineVolumeCubaturePoints.resize(1,spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      fineVolumeCubaturePoints(0,d) = fineTopoRefNodes(finerBasisSubcellOrdinal,d);
    }
    cubatureWeights.resize(1);
    cubatureWeights(0) = 1.0;

    coarseVolumeCubaturePoints.resize(1,spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      coarseVolumeCubaturePoints(0,d) = coarseTopoRefNodes(coarserBasisSubcellOrdinal,d);
    }
  }

  BasisCachePtr fineVolumeCache;
  if (refinements.size() > 0)
  {
    FieldContainer<double> fineVolumeNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinements);
    fineVolumeNodes.resize(1, fineVolumeNodes.dimension(0), fineVolumeNodes.dimension(1));
    fineVolumeCache = BasisCache::basisCacheForCellTopology(fineTopo, cubDegree, fineVolumeNodes);
  }
  else
  {
    fineTopoRefNodes.resize(1,fineTopoRefNodes.dimension(0),fineTopoRefNodes.dimension(1));
    fineVolumeCache = BasisCache::basisCacheForCellTopology(fineTopo, cubDegree, fineTopoRefNodes);
    fineTopoRefNodes.resize(fineTopoRefNodes.dimension(1),fineTopoRefNodes.dimension(2));
  }
  fineVolumeCache->setRefCellPoints(fineVolumeCubaturePoints, cubatureWeights);

  coarseTopoRefNodes.resize(1,coarseTopoRefNodes.dimension(0),coarseTopoRefNodes.dimension(1));
  BasisCachePtr coarseVolumeCache = BasisCache::basisCacheForCellTopology(coarseTopo, cubDegree, coarseTopoRefNodes);
  coarseTopoRefNodes.resize(coarseTopoRefNodes.dimension(1),coarseTopoRefNodes.dimension(2));
  coarseVolumeCache->setRefCellPoints(coarseVolumeCubaturePoints);
  
  int numPoints = cubatureWeights.size();

  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineVolumeCache->getTransformedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineVolumeCache->getTransformedWeightedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseVolumeCache->getTransformedValues(coarserBasis, OP_VALUE, false);

  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());

  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());

//  cout << "fineBasisValues:\n" << *finerBasisValues;
//  cout << "fineBasisValuesFiltered:\n" << fineBasisValuesFiltered;
//  cout << "finerBasisValuesWeighted:\n" << *finerBasisValuesWeighted;
//  cout << "fineBasisValuesFilteredWeighted:\n" << fineBasisValuesFilteredWeighted;
//  cout << "coarserBasisValues:\n" << *coarserBasisValues;
//  cout << "coarserBasisValuesFiltered:\n" << coarserBasisValuesFiltered;

  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());

  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_BLAS);
  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_BLAS);

  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));

//  cout << "lhsValues:\n" << lhsValues;
//  cout << "rhsValues:\n" << rhsValues;
  
  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());

//  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  
  int err = SerialDenseWrapper::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  if (err != 0)
  {
    cout << "ERROR: solve for BasisReconciliation returned error code " << err << endl;
    cout << "lhsValues:\n" << lhsValues;
    cout << "rhsValues:\n" << rhsValues;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solve for BasisReconciliation returned error code.");
  }
  
//  cout << weights.weights;

  return weights;
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(unsigned fineSubcellDimension,
    BasisPtr finerBasis, unsigned fineSubcellOrdinalInFineDomain,
    RefinementBranch &fineCellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
    unsigned fineDomainOrdinalInRefinementLeaf,
    CellTopoPtr coarseCellTopo,
    unsigned coarseSubcellDimension,
    BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
    unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
    unsigned coarseSubcellPermutation)
{
  SubBasisReconciliationWeights weights;

  // use the functionSpace to determine what continuities should be enforced:
  Camellia::EFunctionSpace fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");

  int domainDim = finerBasis->domainTopology()->getDimension();
  if (domainDim != coarserBasis->domainTopology()->getDimension())
  {
    cout << "dimensions of finerBasis and coarserBasis domains do not match!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "dimensions of finerBasis and coarserBasis domains do not match!");
  }

  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(fineSubcellDimension, fineSubcellOrdinalInFineDomain, 0);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain, 0);

  if (weights.fineOrdinals.size() == 0)
  {
    if (weights.coarseOrdinals.size() != 0)
    {
      cout << "fineOrdinals determined to be empty when coarseOrdinals is not!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fineOrdinals determined to be empty when coarseOrdinals is not!");
    }
    weights.weights.resize(0, 0);
    return weights;
  }

  BasisCachePtr fineDomainCache, coarseDomainCache;

  setupFineAndCoarseBasisCachesForReconciliation(fineDomainCache, coarseDomainCache,
      fineSubcellDimension, finerBasis, fineSubcellOrdinalInFineDomain,
      fineCellRefinementBranch, fineDomainOrdinalInRefinementLeaf,
      coarseCellTopo, coarseSubcellDimension, coarserBasis,
      coarseSubcellOrdinalInCoarseDomain, coarseDomainOrdinalInCoarseCellTopo, coarseSubcellPermutation);

//  cout << "fineDomainCache->getPhysicalCubaturePoints():\n" << fineDomainCache->getPhysicalCubaturePoints();
//  cout << "coarseDomainCache->getPhysicalCubaturePoints():\n" << coarseDomainCache->getPhysicalCubaturePoints();

//  cout << "fineDomainCache->getRefCellPoints():\n" << fineDomainCache->getRefCellPoints();
//  cout << "coarseDomainCache->getRefCellPoints():\n" << coarseDomainCache->getRefCellPoints();

//  cout << "fineDomainCache->getPhysicalCellNodes():\n" << fineDomainCache->getPhysicalCellNodes();
//  cout << "coarseDomainCache->getPhysicalCellNodes():\n" << coarseDomainCache->getPhysicalCellNodes();

  int numPoints = fineDomainCache->getPhysicalCubaturePoints().dimension(1); // (C,P,D): 1 is the points dimension

  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineDomainCache->getTransformedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineDomainCache->getTransformedWeightedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseDomainCache->getTransformedValues(coarserBasis, OP_VALUE, false);

  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());

  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());

  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());

  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_BLAS);
  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_BLAS);

  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));

  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());

//  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  
  int err = SerialDenseWrapper::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  if (err != 0)
  {
    cout << "ERROR: solve for BasisReconciliation returned error code " << err << endl;
    cout << "lhsValues:\n" << lhsValues;
    cout << "rhsValues:\n" << rhsValues;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solve for BasisReconciliation returned error code.");
  }

  return weights;
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeightsForTermTraced(LinearTermPtr termTraced, int fieldID,
    unsigned fineSubcellDimension,
    BasisPtr finerBasis,
    unsigned fineSubcellOrdinalInFineDomain,
    RefinementBranch &cellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
    unsigned fineDomainOrdinalInRefinementLeaf,
    CellTopoPtr coarseCellTopo,
    unsigned coarseSubcellDimension,
    BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
    unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
    unsigned coarseSubcellPermutation)
{
  SubBasisReconciliationWeights weights;

  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(fineSubcellDimension, fineSubcellOrdinalInFineDomain, 0);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain, 0);

  if (weights.fineOrdinals.size() == 0)
  {
    if (weights.coarseOrdinals.size() != 0)
    {
      cout << "fineOrdinals determined to be empty when coarseOrdinals is not!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fineOrdinals determined to be empty when coarseOrdinals is not!");
    }
    weights.weights.resize(0, 0);
    return weights;
  }

  BasisCachePtr fineDomainCache, coarseDomainCache;

  setupFineAndCoarseBasisCachesForReconciliation(fineDomainCache, coarseDomainCache,
      fineSubcellDimension, finerBasis, fineSubcellOrdinalInFineDomain,
      cellRefinementBranch, fineDomainOrdinalInRefinementLeaf, coarseCellTopo,
      coarseSubcellDimension, coarserBasis, coarseSubcellOrdinalInCoarseDomain,
      coarseDomainOrdinalInCoarseCellTopo, coarseSubcellPermutation);

  int numPoints = fineDomainCache->getRefCellPoints().dimension(0); // (P,D): 0 is the points dimension

  // set up a dummy DofOrdering for the coarse domain:
  CellTopoPtr coarseDomainTopo = coarserBasis->domainTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering(coarseDomainTopo));

  dofOrderPtr->addEntry(fieldID,coarserBasis, coarserBasis->rangeRank());

  FieldContainer<double> coarserBasisValues;
  if (termTraced->rank() == 0)
  {
    int oneCell = 1;
    coarserBasisValues.resize(oneCell, coarserBasis->getCardinality(),numPoints);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "termTraced reconciliation does not yet support rank > 0 terms...");
  }
//  termTraced->values(coarserBasisValues, fieldID, coarserBasis, coarseDomainCache);

  // TODO: consider this change that might make for a cleaner implementation:
  // - Make BasisCache::fakeSideCache always (or by default) set up cellSideParities that are 1.0
  int oneCell = 1;
  int numSides = fineDomainCache->cellTopology()->getSideCount();
  FieldContainer<double> cellSideParities(oneCell,numSides);
  cellSideParities.initialize(1.0);
  BasisCachePtr fakeSideCache = BasisCache::fakeSideCache(fineDomainCache->getSideIndex(), coarseDomainCache,
                                coarseDomainCache->getRefCellPoints(),
                                fineDomainCache->getSideNormals(), cellSideParities);
  termTraced->values(coarserBasisValues, fieldID, coarserBasis, fakeSideCache);

  // reshape to get rid of cell dimension:
  if (termTraced->rank() == 0)
  {
    coarserBasisValues.resize(coarserBasisValues.dimension(1), coarserBasisValues.dimension(2));
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "termTraced reconciliation does not yet support rank > 0 terms...");
  }

  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineDomainCache->getTransformedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineDomainCache->getTransformedWeightedValues(finerBasis, OP_VALUE, false);

  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForTermTracedValues(coarserBasisValuesFiltered, termTraced, coarserBasis, numPoints, true, weights.coarseOrdinals.size());

  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, coarserBasisValues, weights.coarseOrdinals, coarserBasis->getCardinality());

  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());

  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_BLAS);
  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_BLAS);

  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));

  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());

//  cout << "lhsValues:\n" << lhsValues;
//  cout << "rhsValues:\n" << rhsValues;

//  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  
  int err = SerialDenseWrapper::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  if (err != 0)
  {
    cout << "ERROR: solve for BasisReconciliation returned error code " << err << endl;
    cout << "lhsValues:\n" << lhsValues;
    cout << "rhsValues:\n" << rhsValues;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solve for BasisReconciliation returned error code.");
  }
  return weights;
}


const SubBasisReconciliationWeights& BasisReconciliation::constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, Permutation vertexNodePermutation)
{
  unsigned domainDimension = finerBasis->domainTopology()->getDimension();
  RefinementBranch noRefinements;

  return constrainedWeights(domainDimension, finerBasis, 0, noRefinements, coarserBasis, 0, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideOrdinal, BasisPtr coarserBasis, int coarserBasisSideOrdinal,
    Permutation vertexNodePermutation)
{
  unsigned domainDimension = finerBasis->domainTopology()->getDimension();
  unsigned sideDimension = domainDimension - 1;
  RefinementBranch noRefinements;

  return constrainedWeights(sideDimension, finerBasis, finerBasisSideOrdinal, noRefinements, coarserBasis, coarserBasisSideOrdinal, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, Permutation vertexNodePermutation)
{
  unsigned domainDimension = finerBasis->domainTopology()->getDimension();

  return constrainedWeights(domainDimension, finerBasis, 0, refinements, coarserBasis, 0, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideOrdinal, RefinementBranch &volumeRefinements,
    BasisPtr coarserBasis, int coarserBasisSideOrdinal, unsigned vertexNodePermutation)   // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...
{

  unsigned domainDimension = finerBasis->domainTopology()->getDimension();
  unsigned sideDimension = domainDimension - 1;

  return constrainedWeights(sideDimension, finerBasis, finerBasisSideOrdinal, volumeRefinements, coarserBasis, finerBasisSideOrdinal, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(unsigned subcellDimension,
                                                                              BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
                                                                              RefinementBranch &refinements,
                                                                              BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
                                                                              unsigned vertexNodePermutation)
{

  SubcellBasisRestriction fineBasisRestriction = make_pair(finerBasis.get(), make_pair(subcellDimension, finerBasisSubcellOrdinal) );
  SubcellBasisRestriction coarseBasisRestriction = make_pair(coarserBasis.get(), make_pair(subcellDimension, coarserBasisSubcellOrdinal) );
  SubcellRefinedBasisPair refinedBasisPair = make_pair(make_pair(fineBasisRestriction, coarseBasisRestriction), refinements);

  pair< SubcellRefinedBasisPair, Permutation> cacheKey = make_pair(refinedBasisPair, vertexNodePermutation);

  if (_subcellReconcilationWeights.find(cacheKey) == _subcellReconcilationWeights.end())
  {
    _subcellReconcilationWeights[cacheKey] = computeConstrainedWeights(subcellDimension, finerBasis, finerBasisSubcellOrdinal, refinements,
                                                                       coarserBasis, coarserBasisSubcellOrdinal, vertexNodePermutation);
    // unnecessary copy added for debugging purposes:
//    SubBasisReconciliationWeights weights = computeConstrainedWeights(subcellDimension, finerBasis, finerBasisSubcellOrdinal, refinements,
//                                                                      coarserBasis, coarserBasisSubcellOrdinal, vertexNodePermutation);
//    cout << "weights.weights:\n" << weights.weights;
//    _subcellReconcilationWeights[cacheKey] = weights;
  }

  return _subcellReconcilationWeights[cacheKey];
}

FieldContainer<double> BasisReconciliation::filterBasisValues(const FieldContainer<double> &basisValues, set<int> &filter)
{
  Teuchos::Array<int> dim;
  basisValues.dimensions(dim);
  int basisCardinality = dim[0];
  dim[0] = filter.size();
  FieldContainer<double> filteredValues(dim);

  if (filter.size() == 0)   // empty container
  {
    return filteredValues;
  }

  // apply filter:
  double* filteredValue = &filteredValues[0];
  unsigned valuesPerDof = basisValues.size() / basisCardinality;
  for (set<int>::iterator filteredDofOrdinalIt = filter.begin();
       filteredDofOrdinalIt != filter.end(); filteredDofOrdinalIt++)
  {
    unsigned filteredDofOrdinal = *filteredDofOrdinalIt;
    for (int i=0; i<valuesPerDof; i++)
    {
      *filteredValue = basisValues[filteredDofOrdinal*valuesPerDof + i];
      filteredValue++;
    }
  }
  return filteredValues;
}

SubBasisReconciliationWeights BasisReconciliation::filterOutZeroRowsAndColumns(SubBasisReconciliationWeights &weights)
{
  // we also will filter out zero rows and columns (and while we're at it, round to zero/one for any values that are very close to those):
  const double maxZeroTol = 1e-14;
  // first, look for zero rows:
  set<int> rowsToInclude;
  vector<int> fineOrdinalsVector(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
  set<int> filteredFineOrdinals;

  for (int i=0; i<weights.weights.dimension(0); i++)
  {
    bool nonZeroFound = false;
    for (int j=0; j<weights.weights.dimension(1); j++)
    {
      double value = weights.weights(i,j);
      if (abs(value) < maxZeroTol)
      {
        value = 0;
      }
      else if (abs(value-1.0) < maxZeroTol)
      {
        value = 1;
        nonZeroFound = true;
      }
      else
      {
        nonZeroFound = true;
      }
      weights.weights(i,j) = value;
    }
    if (nonZeroFound)
    {
      rowsToInclude.insert(i);
      filteredFineOrdinals.insert(fineOrdinalsVector[i]);
    }
  }
  // now, look for zero cols:
  set<int> colsToInclude;
  vector<int> coarseOrdinalsVector(weights.coarseOrdinals.begin(),weights.coarseOrdinals.end());
  set<int> filteredCoarseOrdinals;
  for (int j=0; j<weights.weights.dimension(1); j++)
  {
    bool nonZeroFound = false;
    for (int i=0; i<weights.weights.dimension(0); i++)
    {
      if (weights.weights(i,j) != 0)   // by virtue of the rounding above, any zeros will be exact
      {
        nonZeroFound = true;
      }
    }
    if (nonZeroFound)
    {
      colsToInclude.insert(j);
      filteredCoarseOrdinals.insert(coarseOrdinalsVector[j]);
    }
  }

  return filterToInclude(rowsToInclude, colsToInclude, weights);
}

SubBasisReconciliationWeights BasisReconciliation::filterToInclude(set<int> &rowOrdinals, set<int> &colOrdinals, SubBasisReconciliationWeights &weights)
{
  vector<int> coarseOrdinalsVector(weights.coarseOrdinals.begin(),weights.coarseOrdinals.end());
  vector<int> fineOrdinalsVector(weights.fineOrdinals.begin(), weights.fineOrdinals.end());

  SubBasisReconciliationWeights filteredWeights;

  for (set<int>::iterator rowIt=rowOrdinals.begin(); rowIt != rowOrdinals.end(); rowIt++)
  {
    filteredWeights.fineOrdinals.insert(fineOrdinalsVector[*rowIt]);
  }

  for (set<int>::iterator colIt=colOrdinals.begin(); colIt != colOrdinals.end(); colIt++)
  {
    filteredWeights.coarseOrdinals.insert(coarseOrdinalsVector[*colIt]);
  }

  filteredWeights.weights = FieldContainer<double>(filteredWeights.fineOrdinals.size(), filteredWeights.coarseOrdinals.size());

  int filtered_i = 0;
  for (set<int>::iterator rowOrdinalIt = rowOrdinals.begin(); rowOrdinalIt != rowOrdinals.end(); rowOrdinalIt++)
  {
    int i = *rowOrdinalIt;
    int filtered_j = 0;
    for (set<int>::iterator colOrdinalIt = colOrdinals.begin(); colOrdinalIt != colOrdinals.end(); colOrdinalIt++)
    {
      int j = *colOrdinalIt;
      filteredWeights.weights(filtered_i,filtered_j) = weights.weights(i,j);
      filtered_j++;
    }
    filtered_i++;
  }

  return filteredWeights;
}

set<int> BasisReconciliation::interiorDofOrdinalsForBasis(BasisPtr basis)
{
  // if L2, we include all dofs, not just the interior ones
  bool isL2 = (basis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL) || (basis->functionSpace() == Camellia::FUNCTION_SPACE_VECTOR_HVOL);
  int spaceDim = basis->domainTopology()->getDimension();
  set<int> interiorDofOrdinals = isL2 ? basis->dofOrdinalsForSubcells(spaceDim, true) : basis->dofOrdinalsForInterior();
  if (interiorDofOrdinals.size() == 0)
  {
    cout << "Empty interiorDofOrdinals ";
    if (isL2)
      cout << "(L^2 basis).\n";
    else
      cout << "(non-L^2 basis).\n";
  }
  return interiorDofOrdinals;
}

set<unsigned> BasisReconciliation::internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements)
{
  // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  set<unsigned> internalDofOrdinals;
  unsigned spaceDim = finerBasis->domainTopology()->getDimension();

  if (refinements.size() == 0)
  {
    set<int> internalDofOrdinalsInt = BasisReconciliation::interiorDofOrdinalsForBasis(finerBasis);
    internalDofOrdinals.insert(internalDofOrdinalsInt.begin(),internalDofOrdinalsInt.end());
    return internalDofOrdinals;
  }

  bool isL2 = (finerBasis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL) || (finerBasis->functionSpace() == Camellia::FUNCTION_SPACE_VECTOR_HVOL);

  if (isL2)
  {
    // L2 basis, so everything is interior:
    set<int> dofOrdinalsInt = finerBasis->dofOrdinalsForSubcells(spaceDim,true);
    internalDofOrdinals.insert(dofOrdinalsInt.begin(),dofOrdinalsInt.end());
    return internalDofOrdinals;
  }

  map<unsigned, set<unsigned> > internalSubcellOrdinals = RefinementPattern::getInternalSubcellOrdinals(refinements); // might like to cache this result (there is a repeated, inefficient call inside this method)

  for (int scDim = 0; scDim < spaceDim; scDim++)
  {
    set<unsigned> internalOrdinals = internalSubcellOrdinals[scDim];

    for (set<unsigned>::iterator scOrdinalIt = internalOrdinals.begin(); scOrdinalIt != internalOrdinals.end(); scOrdinalIt++)
    {
      unsigned scOrdinal = *scOrdinalIt;
      set<int> scInternalDofs = finerBasis->dofOrdinalsForSubcell(scDim, scOrdinal, scDim);
      internalDofOrdinals.insert(scInternalDofs.begin(),scInternalDofs.end());
    }
  }
  set<int> cellInternalDofs = finerBasis->dofOrdinalsForSubcell(spaceDim, 0, spaceDim);
  internalDofOrdinals.insert(cellInternalDofs.begin(),cellInternalDofs.end());

  return internalDofOrdinals;
}

set<unsigned> BasisReconciliation::internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements, unsigned subcdim, unsigned subcord)
{
  // subcord is in the fine cell (as opposed to the ancestor)
  unsigned spaceDim = finerBasis->domainTopology()->getDimension();
  if (subcdim == spaceDim)
  {
    if (subcord != 0)
    {
      cout << "ERROR: subcord out of range.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subcord out of range");
    }
    return internalDofOrdinalsForFinerBasis(finerBasis, refinements);
  }

  // determine ancestral subcord:
  unsigned ancestralSubcord = RefinementPattern::ancestralSubcellOrdinal(refinements, subcdim, subcord);
//  unsigned ancestralSubcord = subcord;
//  for (int i=refinements.size()-1; i>=0; i--) {
//    RefinementPattern* refPattern = refinements[i].first;
//    unsigned childOrdinal = refinements[i].second;
//    ancestralSubcord = refPattern->mapSubcellOrdinalFromChildToParent(childOrdinal, subcdim, ancestralSubcord);
//  }

  RefinementBranch subcellRefinementBranch = RefinementPattern::subcellRefinementBranch(refinements, subcdim, ancestralSubcord);

  map< unsigned, set<unsigned> > internalSubSubcellOrdinals;
  if (subcellRefinementBranch.size() > 0)
  {
    internalSubSubcellOrdinals = RefinementPattern::getInternalSubcellOrdinals(subcellRefinementBranch); // might like to cache this result (there is a repeated, inefficient call inside this method)
  }
  else
  {
    // if there aren't any refinements, then there aren't any internal sub-subcells...
  }

  // these ordinals are relative to the fine subcell; we need to map them to the fine cell

  map< unsigned, set<unsigned> > internalSubcellOrdinals;
//  for (int d=0; d<subcdim; d++) {
  for (map< unsigned, set<unsigned> >::iterator entryIt = internalSubSubcellOrdinals.begin(); entryIt != internalSubSubcellOrdinals.end(); entryIt++)
  {
    unsigned d = entryIt->first;
    set<unsigned> subSubcellOrdinals = entryIt->second;
    set<unsigned> subcellOrdinals;
    for (set<unsigned>::iterator ordIt=subSubcellOrdinals.begin(); ordIt != subSubcellOrdinals.end(); ordIt++)
    {
      unsigned subcellOrdinal = CamelliaCellTools::subcellOrdinalMap(finerBasis->domainTopology(), subcdim, subcord, d, *ordIt);
      subcellOrdinals.insert(subcellOrdinal);
    }
    internalSubcellOrdinals[d] = subcellOrdinals;
  }

  set<unsigned> internalDofOrdinals;
  // first, add all dofs internal to the fine subcell:
  set<int> fineSubcellInternalDofs = finerBasis->dofOrdinalsForSubcell(subcdim, subcord);
  internalDofOrdinals.insert(fineSubcellInternalDofs.begin(),fineSubcellInternalDofs.end());
  for (int d=0; d<subcdim; d++)
  {
    set<unsigned> subcellOrdinals = internalSubcellOrdinals[d];
    for (set<unsigned>::iterator ordIt=subcellOrdinals.begin(); ordIt != subcellOrdinals.end(); ordIt++)
    {
      unsigned subcellOrdinal = *ordIt;
      set<int> subcellDofs = finerBasis->dofOrdinalsForSubcell(d, subcellOrdinal);
      internalDofOrdinals.insert(subcellDofs.begin(),subcellDofs.end());
    }
  }
  return internalDofOrdinals;
}

unsigned BasisReconciliation::minimumSubcellDimension(BasisPtr basis)
{
  // use the functionSpace to determine what continuities should be enforced:
  Camellia::EFunctionSpace fs = basis->functionSpace();

  if (fs == FUNCTION_SPACE_UNKNOWN)
  {
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    TensorBasis* tensorBasis = dynamic_cast<TensorBasis*>(basis.get());

    if (tensorBasis != NULL)
    {
      // TODO: consider whether the below is in fact the right formula.
      // then the minimum subcell dimension is the sum of the minimum for the two components
      BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
      BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
      return minimumSubcellDimension(spatialBasis) + minimumSubcellDimension(temporalBasis);
    }
    cout << "ERROR: Unhandled functionSpace in BasisReconciliation.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
  }

  int d = basis->domainTopology()->getDimension();
  int minSubcellDimension = d-1;
  switch (fs)
  {
  case Camellia::FUNCTION_SPACE_HGRAD:
  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
  case Camellia::FUNCTION_SPACE_TENSOR_HGRAD:
    minSubcellDimension = 0; // vertices
    break;
  case Camellia::FUNCTION_SPACE_HCURL:
    minSubcellDimension = 1; // edges
    break;
  case Camellia::FUNCTION_SPACE_HDIV:
  case Camellia::FUNCTION_SPACE_HDIV_FREE:
    minSubcellDimension = d-1; // faces in 3D, edges in 2D.  (Unsure if this is right in 4D)
    break;
  case Camellia::FUNCTION_SPACE_HVOL:
  case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
  case Camellia::FUNCTION_SPACE_HDIV_DISC:
  case Camellia::FUNCTION_SPACE_HGRAD_DISC:
  case Camellia::FUNCTION_SPACE_HCURL_DISC:
  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC:
    minSubcellDimension = d; // i.e. no continuities enforced
    break;
  case Camellia::FUNCTION_SPACE_REAL_SCALAR:
    minSubcellDimension = 0;
    break;
  default:
    cout << "ERROR: Unhandled functionSpace in BasisReconciliation.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
    break;
  }
  return minSubcellDimension;
}

void BasisReconciliation::mapFineSubcellPointsToCoarseDomain(FieldContainer<double> &coarseDomainPoints,
    const FieldContainer<double> &fineSubcellPoints,
    unsigned fineSubcellDimension,
    unsigned fineSubcellOrdinalInFineDomain,
    unsigned fineDomainDim,
    unsigned fineDomainOrdinalInRefinementLeaf,
    RefinementBranch &fineCellRefinementBranch,
    CellTopoPtr coarseCellTopo,
    unsigned coarseSubcellDimension,
    unsigned coarseSubcellOrdinalInCoarseDomain,
    unsigned coarseDomainDim,
    unsigned coarseDomainOrdinalInCoarseCellTopo,
    unsigned coarseSubcellPermutation)
{
  if (fineCellRefinementBranch.size() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellRefinementBranch must have at least one refinement!");
  }

  RefinementPattern* leafRefPattern = fineCellRefinementBranch[fineCellRefinementBranch.size()-1].first;
  unsigned leafChildOrdinal = fineCellRefinementBranch[fineCellRefinementBranch.size()-1].second;
  CellTopoPtr fineCellTopo = leafRefPattern->childTopology(leafChildOrdinal);
  CellTopoPtr fineDomainTopo = fineCellTopo->getSubcell(fineDomainDim, fineDomainOrdinalInRefinementLeaf);
  CellTopoPtr fineSubcellTopo = fineDomainTopo->getSubcell(fineSubcellDimension, fineSubcellOrdinalInFineDomain);
  CellTopoPtr coarseDomainTopo = coarseCellTopo->getSubcell(coarseDomainDim, coarseDomainOrdinalInCoarseCellTopo);
  CellTopoPtr coarseSubcellTopo = coarseDomainTopo->getSubcell(coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain);
  int numPoints = fineSubcellPoints.dimension(0);

  unsigned fineSubcellOrdinalInFineCell = CamelliaCellTools::subcellOrdinalMap(fineCellTopo, fineDomainDim, fineDomainOrdinalInRefinementLeaf,
                                          fineSubcellDimension, fineSubcellOrdinalInFineDomain);

  // fineSubcellPoints are understood in the subcell orientation on the fine domain.  When we use genRefBranch below, we use the subcell in
  // the *cell*, which may have a different orientation.  Therefore, we need to determine the permutation that maps from the fine domain's
  // view to that of the cell, and use that to permute fineSubcellPoints.

  unsigned permutationFromFineDomainToCell = CamelliaCellTools::permutationFromSubsubcellToParent(fineCellTopo, fineDomainDim, fineDomainOrdinalInRefinementLeaf,
      fineSubcellDimension, fineSubcellOrdinalInFineDomain);

  FieldContainer<double> fineSubcellPermutedPoints;
  if (permutationFromFineDomainToCell == 0)
  {
    fineSubcellPermutedPoints = fineSubcellPoints; // we could avoid this copy if we made fineSubcellPermutedPoints an RCP, say.  Not sure it matters much, though.
  }
  else
  {
    fineSubcellPermutedPoints.resize(fineSubcellPoints.dimension(0),fineSubcellPoints.dimension(1));
    CamelliaCellTools::permutedReferenceCellPoints(fineSubcellTopo, permutationFromFineDomainToCell, fineSubcellPoints, fineSubcellPermutedPoints);
  }

  GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(fineCellRefinementBranch,
      fineSubcellDimension, fineSubcellOrdinalInFineCell);
  unsigned fineSubcellAncestralDimension = genRefBranch[0].rootDimension;
  unsigned fineSubcellAncestralOrdinalInCell = genRefBranch[0].previousTierSubcellOrdinal;
  CellTopoPtr fineCellAncestor = fineCellRefinementBranch[0].first->parentTopology();

  if (coarseSubcellDimension > fineSubcellAncestralDimension)
  {
    // in this case, the refinement branch doesn't take us all the way to the coarse subcell; therefore,
    // we assume that the coarse cell is identical to the ancestral cell in the refinement branch.  Here,
    // we check two consequences of this assumption:
    TEUCHOS_TEST_FOR_EXCEPTION(fineCellAncestor->getKey() != coarseCellTopo->getKey(), std::invalid_argument, "When refinement branch does not end in coarse subcell, coarse cell must be identical to the fine cell ancestor!");
    TEUCHOS_TEST_FOR_EXCEPTION(coarseSubcellPermutation != 0, std::invalid_argument, "When refinement branch does not end in coarse subcell, coarse cell must be identical to the fine cell ancestor!");
  }

  if (fineDomainDim < fineSubcellAncestralDimension)
  {
    // since the coarseSubcellPermutation argument tells us about the permutation relative to the fine domain's
    // view of the subcell, and we are now using the *cell* which contains the fine domain to view the subcell,
    // we need to tweak the permutation.

    // MAYBE so.  I'm not certain.
    // Here's my feeling: if we are mapping to something that's higher-dimensional than the fine domain, then any "orientation" of the subcell should
    // be entirely captured by the refinement branch -- I believe this amounts to saying that we won't do any "termTraced" type stuff for neighbors,
    // only for ancestors.  This makes sense to me.  Given that, for now anyway I'm going to suggest that having coarseSubcellPermutation be nonzero
    // is an error condition:

    TEUCHOS_TEST_FOR_EXCEPTION(coarseSubcellPermutation != 0, std::invalid_argument, "for trace-to-volume maps, permutation must be the identity!");
  }

  // we want to map fineSubcellPermutedPoints to the ancestor of the fine subcell.  genRefBranch currently ends in the
  // fine *cell* ancestor.  We therefore rewrite the root-level entry to match the fine subcell ancestor:
  genRefBranch[0].previousTierTopo = fineCellAncestor->getSubcell(genRefBranch[0].rootDimension,
                                                                  genRefBranch[0].previousTierSubcellOrdinal);
  genRefBranch[0].previousTierSubcellOrdinal = 0;
  genRefBranch[0].previousTierSubcellPermutation = 0;

  FieldContainer<double> mappedFineSubcellPoints;
  RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, fineSubcellPermutedPoints, mappedFineSubcellPoints);

  if (coarseSubcellDimension > fineSubcellAncestralDimension) // (we handle the (fineSubcellAncestralDimension == 0) case above)
  {
    // then we need to determine the subcellAncestor's ordinal in the coarse subcell topology, and map the points

    // the canonical use case for this is when we map a trace term to volume
    // when this happens, the understanding is that the coarse cell is identified with the root of the
    // fineCellRefinementBranch.  Therefore, fineRootCellTopo and coarseCellTopo should be identical topologies:
    TEUCHOS_TEST_FOR_EXCEPTION(fineCellTopo != coarseCellTopo, std::invalid_argument, "When fine subcell ancestor is not identical to coarse subcell, then the coarse cell must be the root of the refinement branch.");

    // subcellAncestor has dimension and ordinal in the *volume*
    unsigned coarseSubcellOrdinalInRootVolume = CamelliaCellTools::subcellOrdinalMap(coarseCellTopo,
        coarseDomainDim, coarseDomainOrdinalInCoarseCellTopo,
        coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain);
    unsigned ancestralSubcellOrdinalInCoarseSubcell = CamelliaCellTools::subcellReverseOrdinalMap(coarseCellTopo, coarseSubcellDimension,
        coarseSubcellOrdinalInRootVolume,
        fineSubcellAncestralDimension, fineSubcellAncestralOrdinalInCell);
    if (fineSubcellAncestralDimension > 0)
    {
      // copy
      FieldContainer<double> ancestralSubcellPoints = mappedFineSubcellPoints;
      // resize mappedFineSubcellPoints for the coarseSubcellDimension:
      mappedFineSubcellPoints.resize(mappedFineSubcellPoints.dimension(0), coarseSubcellDimension);
      // map
      CamelliaCellTools::mapToReferenceSubcell(mappedFineSubcellPoints, ancestralSubcellPoints,
          fineSubcellAncestralDimension, ancestralSubcellOrdinalInCoarseSubcell, coarseSubcellTopo);
    }
    else
    {
      CellTopoPtr topo = coarseSubcellTopo;
      FieldContainer<double> topoRefNodes(topo->getVertexCount(),topo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(topoRefNodes, topo);
      mappedFineSubcellPoints.resize(1, topo->getDimension());
      for (int d=0; d < topo->getDimension(); d++)
      {
        mappedFineSubcellPoints(0,d) = topoRefNodes(ancestralSubcellOrdinalInCoarseSubcell,d);
      }
    }
  }

  FieldContainer<double> coarseSubcellPoints;

  if (coarseSubcellDimension == 0)
  {
    coarseSubcellPoints = mappedFineSubcellPoints; // not real sure we need to do this...
  }
  else
  {
    // when you set physical cell nodes according to the coarse-to-fine permutation, then the reference-to-physical map
    // is fine-to-coarse (which is what we want).  Because the coarseSubcellPermutation is fine-to-coarse, we want its inverse:
    unsigned ancestralSubcellPermutation = CamelliaCellTools::permutationInverse(coarseSubcellTopo, coarseSubcellPermutation);

    coarseSubcellPoints.resize(numPoints,coarseSubcellDimension);
    CamelliaCellTools::permutedReferenceCellPoints(coarseSubcellTopo, ancestralSubcellPermutation, mappedFineSubcellPoints, coarseSubcellPoints);

    // coarseSubcellPoints are understood in the subcell orientation on the coarse domain.  When we use genRefBranch as above, we map to the subcell in
    // the *cell*, which may have a different orientation.  Therefore, we need to determine the permutation that maps from the cell's view
    // to that of the coarse domain, and use that to permute coarseSubcellPoints.
    unsigned permutationFromCoarseDomainToCell = CamelliaCellTools::permutationFromSubsubcellToParent(coarseCellTopo, coarseDomainDim, coarseDomainOrdinalInCoarseCellTopo,
        coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain);
    if (permutationFromCoarseDomainToCell != 0)
    {
      unsigned permutationFromCellToCoarseDomain = CamelliaCellTools::permutationInverse(coarseSubcellTopo, permutationFromCoarseDomainToCell);
      FieldContainer<double> coarseSubcellPointsCopy = coarseSubcellPoints;
      CamelliaCellTools::permutedReferenceCellPoints(coarseSubcellTopo, permutationFromCellToCoarseDomain, coarseSubcellPointsCopy, coarseSubcellPoints);
    }
    // TODO: combine the two permutations above.  I think this should be represented by:
//    CamelliaCellTools::permutationComposition(coarseSubcellTopo, ancestralSubcellPermutation, permutationFromCellToCoarseDomain);
  }

  if (coarseDomainDim == coarseSubcellDimension)
  {
    // if fineSubcellAncestralDimension is the same as the domain dimension, then the ancestral subcell is exactly the coarse domain
    coarseDomainPoints = coarseSubcellPoints;
  }
  else
  {
    // If fineSubcellAncestralDimension is *NOT* the same as the domain dimension, then the ancestral subcell is a subcell of the coarse domain

    coarseDomainPoints.resize(mappedFineSubcellPoints.dimension(0), coarseDomainDim);
    CamelliaCellTools::mapToReferenceSubcell(coarseDomainPoints, coarseSubcellPoints, fineSubcellAncestralDimension,
        coarseSubcellOrdinalInCoarseDomain, coarseDomainTopo);
  }
}

void BasisReconciliation::setupFineAndCoarseBasisCachesForReconciliation(BasisCachePtr &fineDomainCache,
                                                                         BasisCachePtr &coarseDomainCache,
                                                                         unsigned fineSubcellDimension,
                                                                         BasisPtr finerBasis,
                                                                         unsigned fineSubcellOrdinalInFineDomain,
                                                                         RefinementBranch &fineCellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
                                                                         unsigned fineDomainOrdinalInRefinementLeaf,
                                                                         CellTopoPtr coarseCellTopo,
                                                                         unsigned coarseSubcellDimension,
                                                                         BasisPtr coarserBasis, unsigned coarseSubcellOrdinalInCoarseDomain,
                                                                         unsigned coarseDomainOrdinalInCoarseCellTopo, // we use the coarserBasis's domain topology to determine the domain's space dimension
                                                                         unsigned coarseSubcellPermutation)
{
  int fineDomainDim = finerBasis->domainTopology()->getDimension();
  int coarseDomainDim = coarserBasis->domainTopology()->getDimension();

  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself (this is overkill; we could likely get away with less by reasoning about the cardinality of fineOrdinals)

  // determine cubature points as seen by the fine basis
  CellTopoPtr fineTopo = finerBasis->domainTopology();
  CellTopoPtr fineSubcellTopo = fineTopo->getSubcell(fineSubcellDimension, fineSubcellOrdinalInFineDomain);

  CellTopoPtr coarseTopo = coarserBasis->domainTopology();
  CellTopoPtr coarseSubcellTopo = coarseTopo->getSubcell(coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain);

  FieldContainer<double> fineTopoRefNodes(fineTopo->getVertexCount(), fineTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(fineTopoRefNodes, fineTopo);
  FieldContainer<double> coarseTopoRefNodes(coarseTopo->getVertexCount(), coarseTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseTopoRefNodes, coarseTopo);

  FieldContainer<double> coarseDomainPoints; // these are as seen from the coarse cell's neighbor -- we use the coarseDomainPermutation together with the physicalCellNodes on the coarse domain to convert...
  FieldContainer<double> fineDomainPoints; // points at which the fine basis will be evaluated
  FieldContainer<double> cubatureWeightsFineSubcell; // allows us to integrate over the fine subcell even when domain is higher-dimensioned

  coarseDomainCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false));

  if (fineCellRefinementBranch.size() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellRefinementBranch must have at least one refinement!");
  }

  FieldContainer<double> leafCellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(fineCellRefinementBranch);
  CellTopoPtr leafCellTopo = RefinementPattern::descendantTopology(fineCellRefinementBranch);

  FieldContainer<double> subcellCubaturePoints;
  int numPoints;

  if (fineSubcellDimension > 0)
  {
    BasisCachePtr fineSubcellCache = Teuchos::rcp( new BasisCache(fineSubcellTopo, cubDegree, false) );
    numPoints = fineSubcellCache->getRefCellPoints().dimension(0);
    fineDomainPoints.resize(numPoints,fineDomainDim);
    subcellCubaturePoints = fineSubcellCache->getRefCellPoints();
    cubatureWeightsFineSubcell = fineSubcellCache->getCubatureWeights();
    if (fineSubcellDimension == fineDomainDim)
    {
      fineDomainPoints = subcellCubaturePoints;
    }
    else
    {
      CamelliaCellTools::mapToReferenceSubcell(fineDomainPoints, subcellCubaturePoints, fineSubcellDimension,
          fineSubcellOrdinalInFineDomain, fineTopo);
    }
  }
  else     // subcellDimension == 0 --> vertex
  {
    numPoints = 1;
    subcellCubaturePoints.resize(numPoints,1);
    fineDomainPoints.resize(numPoints,fineDomainDim);
    for (int d=0; d<fineDomainDim; d++)
    {
      fineDomainPoints(0,d) = fineTopoRefNodes(fineSubcellOrdinalInFineDomain,d);
    }
    cubatureWeightsFineSubcell.resize(1);
    cubatureWeightsFineSubcell(0) = 1.0;
  }

  mapFineSubcellPointsToCoarseDomain(coarseDomainPoints, subcellCubaturePoints,
                                     fineSubcellDimension, fineSubcellOrdinalInFineDomain,
                                     fineDomainDim, fineDomainOrdinalInRefinementLeaf,
                                     fineCellRefinementBranch, coarseCellTopo,
                                     coarseSubcellDimension, coarseSubcellOrdinalInCoarseDomain,
                                     coarseDomainDim, coarseDomainOrdinalInCoarseCellTopo,
                                     coarseSubcellPermutation);

//  cout << "coarseDomainPoints:\n" << coarseDomainPoints;

//  fineDomainCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) ); // false: don't create side cache
//  leafDomainNodes.resize(1,leafDomainNodes.dimension(0),leafDomainNodes.dimension(1));
//  fineDomainCache->setPhysicalCellNodes(leafDomainNodes, vector<GlobalIndexType>(), false);

  BasisCachePtr fineCellCache = Teuchos::rcp( new BasisCache(leafCellTopo, cubDegree, false) ); // false: don't create side cache
  leafCellNodes.resize(1,leafCellNodes.dimension(0),leafCellNodes.dimension(1));
  fineCellCache->setPhysicalCellNodes(leafCellNodes, {}, false);

  if (leafCellTopo->getDimension() > fineDomainDim)
  {
    if (leafCellTopo->getDimension() != fineDomainDim + 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basis domains that are not either on cells or cell sides are unsupported.");
    }
    // need to use the following constructor so that the side cache will have a strong RCP to the volume one.
    // (BasisCache::createSideCaches() uses weak RCPs from side cache to volume to avoid circular references.)
    fineDomainCache = BasisCache::sideBasisCache(fineCellCache, fineDomainOrdinalInRefinementLeaf);
    fineDomainCache->setPhysicalCellNodes(leafCellNodes, {}, false);
  }
  else
  {
    fineDomainCache = fineCellCache;
  }
  fineDomainCache->setRefCellPoints(fineDomainPoints, cubatureWeightsFineSubcell);
  coarseDomainCache->setRefCellPoints(coarseDomainPoints);

  coarseTopoRefNodes.resize(1,coarseTopoRefNodes.dimension(0),coarseTopoRefNodes.dimension(1));
  coarseDomainCache->setPhysicalCellNodes(coarseTopoRefNodes, vector<GlobalIndexType>(), false);
}

SubBasisReconciliationWeights BasisReconciliation::sumWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights)
{
  SubBasisReconciliationWeights weights;
  // union the ordinals in the two weights:
  weights.fineOrdinals.insert(aWeights.fineOrdinals.begin(), aWeights.fineOrdinals.end());
  weights.fineOrdinals.insert(bWeights.fineOrdinals.begin(), bWeights.fineOrdinals.end());

  weights.coarseOrdinals.insert(aWeights.coarseOrdinals.begin(), aWeights.coarseOrdinals.end());
  weights.coarseOrdinals.insert(bWeights.coarseOrdinals.begin(), bWeights.coarseOrdinals.end());

  // size the weights FieldContainer appropriately:
  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());

  map<int, int> aFineOrdinal, bFineOrdinal;  // lookups: key is the basis ordinal; value is the ordinal in the aWeights/bWeights container
  int aPos = 0, bPos = 0;
  for (set<int>::iterator allFineOrdinalIt = weights.fineOrdinals.begin(); allFineOrdinalIt != weights.fineOrdinals.end(); allFineOrdinalIt++)
  {
    int fineOrdinal = *allFineOrdinalIt;
    if (aWeights.fineOrdinals.find(fineOrdinal) != aWeights.fineOrdinals.end())
    {
      aFineOrdinal[fineOrdinal] = aPos++;
    }
    else
    {
      aFineOrdinal[fineOrdinal] = -1;
    }
    if (bWeights.fineOrdinals.find(fineOrdinal) != bWeights.fineOrdinals.end())
    {
      bFineOrdinal[fineOrdinal] = bPos++;
    }
    else
    {
      bFineOrdinal[fineOrdinal] = -1;
    }
  }
  map<int, int> aCoarseOrdinal, bCoarseOrdinal;  // lookups: key is the basis ordinal; value is the ordinal in the aWeights/bWeights container
  aPos = 0;
  bPos = 0;
  for (set<int>::iterator allCoarseOrdinalIt = weights.coarseOrdinals.begin(); allCoarseOrdinalIt != weights.coarseOrdinals.end(); allCoarseOrdinalIt++)
  {
    int coarseOrdinal = *allCoarseOrdinalIt;
    if (aWeights.coarseOrdinals.find(coarseOrdinal) != aWeights.coarseOrdinals.end())
    {
      aCoarseOrdinal[coarseOrdinal] = aPos++;
    }
    else
    {
      aCoarseOrdinal[coarseOrdinal] = -1;
    }
    if (bWeights.coarseOrdinals.find(coarseOrdinal) != bWeights.coarseOrdinals.end())
    {
      bCoarseOrdinal[coarseOrdinal] = bPos++;
    }
    else
    {
      bCoarseOrdinal[coarseOrdinal] = -1;
    }
  }
  int i=0;
  for (set<int>::iterator allFineOrdinalIt = weights.fineOrdinals.begin(); allFineOrdinalIt != weights.fineOrdinals.end(); allFineOrdinalIt++)
  {
    int j=0;
    int aFineOrd = aFineOrdinal[*allFineOrdinalIt];
    int bFineOrd = bFineOrdinal[*allFineOrdinalIt];
    for (set<int>::iterator allCoarseOrdinalIt = weights.coarseOrdinals.begin(); allCoarseOrdinalIt != weights.coarseOrdinals.end(); allCoarseOrdinalIt++)
    {
      int aCoarseOrd = aCoarseOrdinal[*allCoarseOrdinalIt];
      int bCoarseOrd = bCoarseOrdinal[*allCoarseOrdinalIt];

      if ((aFineOrd != -1) && (aCoarseOrd != -1))
      {
        weights.weights(i,j) += aWeights.weights(aFineOrd,aCoarseOrd);
      }

      if ((bFineOrd != -1) && (bFineOrd != -1))
      {
        weights.weights(i,j) += bWeights.weights(bFineOrd,bCoarseOrd);
      }

      j++;
    }
    i++;
  }
  return weights;
}

SubBasisReconciliationWeights BasisReconciliation::weightsForCoarseSubcell(SubBasisReconciliationWeights &weights, BasisPtr constrainingBasis, unsigned subcdim, unsigned subcord, bool includeSubsubcells)
{
  int minSubcellDimension = includeSubsubcells ? 0 : subcdim;
  set<int> coarseDofsForSubcell = constrainingBasis->dofOrdinalsForSubcell(subcdim, subcord, minSubcellDimension);
  set<int> coarseDofsToInclude; // will be the intersection of coarseDofsForSubcell and weights.coarseOrdinals
  vector<unsigned> columnOrdinalsToInclude;
  int columnOrdinal = 0;
  for (set<int>::iterator coarseOrdinalIt = weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++)
  {
    int coarseOrdinal = *coarseOrdinalIt;
    if (coarseDofsForSubcell.find(coarseOrdinal) != coarseDofsForSubcell.end())
    {
      columnOrdinalsToInclude.push_back(columnOrdinal);
      coarseDofsToInclude.insert(coarseOrdinal);
    }
    columnOrdinal++;
  }

  SubBasisReconciliationWeights filteredWeights;

  filteredWeights.coarseOrdinals = coarseDofsToInclude;
  filteredWeights.fineOrdinals = weights.fineOrdinals;

  filteredWeights.weights = FieldContainer<double>(weights.fineOrdinals.size(), columnOrdinalsToInclude.size());

  for (int i=0; i<filteredWeights.weights.dimension(0); i++)
  {
    for (int j=0; j<filteredWeights.weights.dimension(1); j++)
    {
      double value = weights.weights(i,columnOrdinalsToInclude[j]);
      filteredWeights.weights(i,j) = value;
    }
  }

//  cout << "weights before filtering:\n";
//  Camellia::print("fine ordinals", weights.fineOrdinals);
//  Camellia::print("coarse ordinals", weights.coarseOrdinals);
//  cout << "weights:\n" << weights.weights;
//
//  cout << "filtered weights for coarse subcell " << subcord << " of dimension " << subcdim << endl;
//  Camellia::print("filtered fine ordinals", filteredWeights.fineOrdinals);
//  Camellia::print("filtered coarse ordinals", filteredWeights.coarseOrdinals);
//  cout << "filtered weights:\n" << filteredWeights.weights;

  return filterOutZeroRowsAndColumns(filteredWeights);
}