//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <iostream>

#include "MeshTestUtility.h"

#include "BasisCache.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "CamelliaCellTools.h"
#include "CubatureFactory.h"
#include "GDAMinimumRule.h"
#include "MPIWrapper.h"

using namespace Camellia;
using namespace std;

bool MeshTestUtility::constraintIsConsistent(MeshTopologyViewPtr meshTopo, AnnotatedEntity &constrainingEntity,
                                             int d, IndexType constrainedEntityIndex, bool requireConstrainingEntityBelongToActiveCell)
{
  int sideDim = meshTopo->getDimension() - 1;
  CellPtr constrainingCell = meshTopo->getCell(constrainingEntity.cellID);
  int constrainingSubcordInCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(),
                                                                       sideDim, constrainingEntity.sideOrdinal,
                                                                       constrainingEntity.dimension, constrainingEntity.subcellOrdinal);
  
  IndexType constrainingEntityIndex = constrainingCell->entityIndex(constrainingEntity.dimension, constrainingSubcordInCell);
  bool isAncestor = meshTopo->entityIsGeneralizedAncestor(constrainingEntity.dimension, constrainingEntityIndex, d, constrainedEntityIndex);
  
  if (!isAncestor) return false;
  
  if (requireConstrainingEntityBelongToActiveCell)
  {
    set<pair<IndexType,unsigned>> cellEntries = meshTopo->getCellsContainingEntity(constrainingEntity.dimension, constrainingEntityIndex);
    auto activeCells = &meshTopo->getLocallyKnownActiveCellIndices();
    for (auto cellEntry : cellEntries)
    {
      IndexType cellID = cellEntry.first;
      if (activeCells->find(cellID) != activeCells->end()) return true; // found an active cell, and isAncestor is true
    }
    return false;
  }
  
  return isAncestor;
}

bool MeshTestUtility::checkConstraintConsistency(MeshPtr meshMinimumRule)
{
  MeshTopologyViewPtr meshTopo = meshMinimumRule->getTopology();
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(meshMinimumRule->globalDofAssignment().get());
  
  bool consistent = true;
  
  for (IndexType cellID : meshMinimumRule->cellIDsInPartition())
  {
    CellConstraints constraints = minRule->getCellConstraints(cellID);
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    
    for (int d=0; d<cellTopo->getDimension(); d++)
    {
      int scCount = cellTopo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        IndexType entityIndex = cell->entityIndex(d, scord);
        
        AnnotatedEntity constrainingEntity = constraints.subcellConstraints[d][scord];
        bool isConsistent = constraintIsConsistent(meshTopo, constrainingEntity, d, entityIndex, true);
        
        if (!isConsistent)
        {
          cout << "Failed consistency test for standard constraints on cell " << cellID << ", " << CamelliaCellTools::entityTypeString(d) << " " << scord << endl;
          
          consistent = false;
          break;
        }
        
        // now, check space-only constraints (for space-time meshes), if these are defined for this subcell
        if (constraints.spatialSliceConstraints != Teuchos::null)
        {
          AnnotatedEntity constrainingEntityForSpatialSlice = constraints.spatialSliceConstraints->subcellConstraints[d][scord];
          if (constrainingEntityForSpatialSlice.cellID != -1) {
            isConsistent = constraintIsConsistent(meshTopo, constrainingEntityForSpatialSlice, d, entityIndex, true);
            
          }
          if (!isConsistent)
          {
            cout << "Failed consistency test for spatial slice on cell " << cellID << ", " << CamelliaCellTools::entityTypeString(d) << " " << scord << endl;
            
            consistent = false;
            break;
          }
        }
      }
      if (!consistent) break;
    }
    if (!consistent) break;
  }
  return consistent;
}

bool MeshTestUtility::checkLocalGlobalConsistency(MeshPtr mesh, double tol)
{
  bool success = true;

  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();

  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();

  int numGlobalDofs = gda->globalDofCount();
  Camellia::FieldContainer<double> globalCoefficients(numGlobalDofs);
  for (int i=0; i<numGlobalDofs; i++)
  {
    globalCoefficients(i) = 2*i + 1; // arbitrary cofficients
  }
  Camellia::FieldContainer<double> globalCoefficientsExpected = globalCoefficients;
  Camellia::FieldContainer<double> globalCoefficientsActual(numGlobalDofs);
  Camellia::FieldContainer<double> localCoefficients;

  using GO = TMap::global_ordinal_type;
  TMapPtr map = Teuchos::rcp(new TMap(static_cast<Tpetra::global_size_t>(numGlobalDofs), static_cast<GO>(0), MPIWrapper::TeuchosCommSerial()));
  TVectorPtr<double> globalCoefficientsVector = Teuchos::rcp(new TVector<double>(map, 1));
  auto globalCoefficientsVectorView = globalCoefficientsVector->getLocalViewHost(Tpetra::Access::ReadWrite);
  for (int i=0; i<numGlobalDofs; i++)
  {
    globalCoefficientsVectorView(i,0) = globalCoefficients(i);
  }

  for (GlobalIndexType cellID : cellIDs)
  {
    gda->interpretGlobalCoefficients(cellID, localCoefficients, globalCoefficientsVector);
    Camellia::FieldContainer<GlobalIndexType> globalDofIndices;
    Camellia::FieldContainer<double> globalCoefficientsForCell;

    DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    set<int> varIDs = trialOrder->getVarIDs();

    for (set<int>::iterator varIt=varIDs.begin(); varIt != varIDs.end(); varIt++)
    {
      int varID = *varIt;
      const bool isVolumeVar = trialOrder->isVolumeVar(varID);
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(varID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;

        BasisPtr basis;
        if (isVolumeVar)
        {
          basis = trialOrder->getBasis(varID);
        }
        else
        {
          basis = trialOrder->getBasis(varID,sideOrdinal);
        }

        Camellia::FieldContainer<double> basisCoefficients(basis->getCardinality());

        for (int dofOrdinal=0; dofOrdinal<basis->getCardinality(); dofOrdinal++)
        {
          int localDofIndex;
          if (isVolumeVar)
          {
            localDofIndex = trialOrder->getDofIndex(varID, dofOrdinal);
          }
          else
          {
            localDofIndex = trialOrder->getDofIndex(varID, dofOrdinal, sideOrdinal);
          }

          basisCoefficients(dofOrdinal) = localCoefficients(localDofIndex);
        }

        gda->interpretLocalBasisCoefficients(cellID, varID, sideOrdinal,
                                             basisCoefficients, globalCoefficientsForCell, globalDofIndices);

        //        if ( (cellID==2) && (sideOrdinal==1) && (varID==0) ) {
        //          cout << "DEBUGGING: (cellID==2) && (sideOrdinal==1).\n";
        //          cout << "globalDofIndices:\n" << globalDofIndices;
        //          cout << "globalCoefficientsForCell:\n" << globalCoefficientsForCell;
        //          cout << "basisCoefficients:\n" << basisCoefficients;
        //        }

        for (int dofOrdinal=0; dofOrdinal < globalDofIndices.size(); dofOrdinal++)
        {
          GlobalIndexType dofIndex = globalDofIndices(dofOrdinal);
          globalCoefficientsActual(dofIndex) = globalCoefficientsForCell(dofOrdinal);

          double diff = abs(globalCoefficientsForCell(dofOrdinal) - globalCoefficientsExpected(dofIndex)) / globalCoefficientsExpected(dofIndex);
          if (diff > tol)
          {
            cout << "In mapping for cell " << cellID << " and var " << varID << " on side " << sideOrdinal << ", ";
            cout << "expected coefficient for global dof index " << dofIndex << " to be " << globalCoefficientsExpected(dofIndex);
            cout << ", but was " << globalCoefficientsForCell(dofOrdinal);
            cout << " (relative diff = " << diff << "; tol = " << tol << ")\n";
            success = false;
          }
        }
      }
    }
  }

  //  double maxDiff;
  //  if (TestSuite::fcsAgree(globalCoefficientsActual, globalCoefficientsExpected, tol, maxDiff)) {
  //    //    cout << "global data actual and expected AGREE; max difference is " << maxDiff << endl;
  //    //    cout << "globalCoefficientsActual:\n" << globalCoefficientsActual;
  //  } else {
  //    cout << "global data actual and expected DISAGREE; max difference is " << maxDiff << endl;
  //    success = false;
  //    cout << "Expected:\n" << globalCoefficientsExpected;
  //    cout << "Actual:\n" << globalCoefficientsActual;
  //  }

  return success;
}

bool MeshTestUtility::determineRefTestPointsForNeighbors(MeshTopologyViewPtr meshTopo, CellPtr fineCell, unsigned int sideOrdinal,
    Camellia::FieldContainer<double> &fineSideRefPoints, Camellia::FieldContainer<double> &fineCellRefPoints,
    Camellia::FieldContainer<double> &coarseSideRefPoints, Camellia::FieldContainer<double> &coarseCellRefPoints)
{
  unsigned spaceDim = meshTopo->getDimension();
  unsigned sideDim = spaceDim - 1;

  if (spaceDim == 1)
  {
    fineSideRefPoints.resize(0,0);
    coarseSideRefPoints.resize(0,0);
    fineCellRefPoints.resize(1,1);
    coarseCellRefPoints.resize(1,1);

    Camellia::FieldContainer<double> lineRefNodes(2,1);
    CellTopoPtr line = CellTopology::line();

    CamelliaCellTools::refCellNodesForTopology(lineRefNodes, line);

    fineCellRefPoints[0] = lineRefNodes[sideOrdinal];
    unsigned neighborSideOrdinal = fineCell->getNeighborInfo(sideOrdinal, meshTopo).second;
    if (neighborSideOrdinal != INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
    {
      coarseCellRefPoints[0] = lineRefNodes[neighborSideOrdinal];
      return true;
    }
    else
    {
      return false;
    }
  }
  pair<GlobalIndexType, unsigned> neighborInfo = fineCell->getNeighborInfo(sideOrdinal, meshTopo);
  if (neighborInfo.first == -1)
  {
    // boundary
    return false;
  }
  CellPtr neighborCell = meshTopo->getCell(neighborInfo.first);
  if (neighborCell->isParent(meshTopo))
  {
    return false; // fineCell isn't the finer of the two...
  }

  CellTopoPtr fineSideTopo = fineCell->topology()->getSubcell(sideDim, sideOrdinal);

  CubatureFactory cubFactory;
  int cubDegree = 4; // fairly arbitrary choice: enough to get a decent number of test points...
  
  using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
  using Cubature = Intrepid2::Cubature<DeviceType,double,double>;
  Teuchos::RCP< Cubature > fineSideTopoCub = cubFactory.create(fineSideTopo, cubDegree);

  int numCubPoints = fineSideTopoCub->getNumPoints();

  auto tensorCubatureWeights = fineSideTopoCub->allocateCubatureWeights();
  auto tensorCubaturePoints = fineSideTopoCub->allocateCubaturePoints(); // we neglect these
  
  fineSideTopoCub->getCubature(tensorCubaturePoints, tensorCubatureWeights);
  
  Camellia::FieldContainer<double> cubPoints  = FieldContainerConversion<double>::fromTensorPoints(tensorCubaturePoints);
  
  Camellia::FieldContainer<double> sideRefCellNodes(fineSideTopo->getNodeCount(),sideDim);
  CamelliaCellTools::refCellNodesForTopology(sideRefCellNodes, fineSideTopo);

  int numTestPoints = numCubPoints + fineSideTopo->getNodeCount();

  Camellia::FieldContainer<double> testPoints(numTestPoints, sideDim);
  for (int ptOrdinal=0; ptOrdinal<testPoints.dimension(0); ptOrdinal++)
  {
    if (ptOrdinal<fineSideTopo->getNodeCount())
    {
      for (int d=0; d<sideDim; d++)
      {
        testPoints(ptOrdinal,d) = sideRefCellNodes(ptOrdinal,d);
      }
    }
    else
    {
      for (int d=0; d<sideDim; d++)
      {
        testPoints(ptOrdinal,d) = cubPoints(ptOrdinal-fineSideTopo->getNodeCount(),d);
      }
    }
  }

  fineSideRefPoints = testPoints;
  fineCellRefPoints.resize(numTestPoints, spaceDim);

  CamelliaCellTools::mapToReferenceSubcell(fineCellRefPoints, testPoints, sideDim, sideOrdinal, fineCell->topology());

  CellTopoPtr coarseSideTopo = neighborCell->topology()->getSubcell(sideDim, neighborInfo.second);

  unsigned fineSideAncestorPermutation = fineCell->ancestralPermutationForSubcell(sideDim, sideOrdinal, meshTopo);
  unsigned coarseSidePermutation = neighborCell->subcellPermutation(sideDim, neighborInfo.second);

  unsigned coarseSideAncestorPermutationInverse = CamelliaCellTools::permutationInverse(coarseSideTopo, coarseSidePermutation);

  unsigned composedPermutation = CamelliaCellTools::permutationComposition(coarseSideTopo, coarseSideAncestorPermutationInverse, fineSideAncestorPermutation); // goes from coarse ordering to fine.

  RefinementBranch fineRefBranch = fineCell->refinementBranchForSide(sideOrdinal, meshTopo);

  Camellia::FieldContainer<double> fineSideNodes(fineSideTopo->getNodeCount(), sideDim);  // relative to the ancestral cell whose neighbor is compatible
  if (fineRefBranch.size() == 0)
  {
    CamelliaCellTools::refCellNodesForTopology(fineSideNodes, coarseSideTopo, composedPermutation);
  }
  else
  {
    Camellia::FieldContainer<double> ancestralSideNodes(coarseSideTopo->getNodeCount(), sideDim);
    CamelliaCellTools::refCellNodesForTopology(ancestralSideNodes, coarseSideTopo, composedPermutation);

    RefinementBranch fineSideRefBranch = RefinementPattern::sideRefinementBranch(fineRefBranch, sideOrdinal);
    fineSideNodes = RefinementPattern::descendantNodes(fineSideRefBranch, ancestralSideNodes);
  }

  BasisCachePtr sideTopoCache = Teuchos::rcp( new BasisCache(fineSideTopo, 1, false) );
  sideTopoCache->setRefCellPointsFC(testPoints);

  // add cell dimension
  fineSideNodes.resize(1,fineSideNodes.dimension(0), fineSideNodes.dimension(1));
  sideTopoCache->setPhysicalCellNodes(fineSideNodes, vector<GlobalIndexType>(), false);
  coarseSideRefPoints = sideTopoCache->getPhysicalCubaturePointsFC();

  // strip off the cell dimension:
  coarseSideRefPoints.resize(coarseSideRefPoints.dimension(1),coarseSideRefPoints.dimension(2));

  coarseCellRefPoints.resize(numTestPoints,spaceDim);
  CamelliaCellTools::mapToReferenceSubcell(coarseCellRefPoints, coarseSideRefPoints, sideDim, neighborInfo.second, neighborCell->topology());

  return true; // containers filled....
}

bool MeshTestUtility::fcsAgree(const Camellia::FieldContainer<double> &fc1, const Camellia::FieldContainer<double> &fc2, double tol, double &maxDiff)
{
  if (fc1.size() != fc2.size())
  {
    maxDiff = -1.0; // a signal something's wrong…
    return false;
  }
  maxDiff = 0.0;
  for (int i=0; i<fc1.size(); i++)
  {
    maxDiff = std::max(maxDiff, abs(fc1[i] - fc2[i]));
  }
  return (maxDiff <= tol);
}

bool MeshTestUtility::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh)
{
  using GO = TMap::global_ordinal_type;
  int numGlobalDofs = mesh->globalDofCount();
  TMapPtr map = Teuchos::rcp(new TMap(static_cast<Tpetra::global_size_t>(numGlobalDofs), static_cast<GO>(0), MPIWrapper::TeuchosCommSerial()));
  TVectorPtr<double> testSolutionCoefficients = Teuchos::rcp(new TVector<double>(map, 1));
  auto testSolutionCoefficientsView = testSolutionCoefficients->getLocalViewHost(Tpetra::Access::ReadWrite);
  for (int i=0; i<numGlobalDofs; i++)
  {
    testSolutionCoefficientsView(i,0) =  0.1 * i;
  }

  return neighborBasesAgreeOnSides(mesh, testSolutionCoefficients);
}

bool MeshTestUtility::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, TVectorPtr<double> globalSolutionCoefficients)
{
  bool success = true;
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  int spaceDim = meshTopo->getDimension();

  set<GlobalIndexType> myCellIDs = mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);

    BasisCachePtr fineCellBasisCache = BasisCache::basisCacheForCell(mesh, cellID);
    ElementTypePtr fineElemType = mesh->getElementType(cellID);
    DofOrderingPtr fineElemTrialOrder = fineElemType->trialOrderPtr;

    Camellia::FieldContainer<double> fineSolutionCoefficients(fineElemTrialOrder->totalDofs());
    mesh->globalDofAssignment()->interpretGlobalCoefficients(cellID, fineSolutionCoefficients, globalSolutionCoefficients);
//    if ((cellID==0) || (cellID==2)) {
//      cout << "MeshTestUtility: local coefficients for cell " << cellID << ":\n" << fineSolutionCoefficients;
//    }

    unsigned sideCount = cell->getSideCount();
    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      Camellia::FieldContainer<double> fineSideRefPoints, fineCellRefPoints, coarseSideRefPoints, coarseCellRefPoints;
      bool hasCoarserNeighbor = determineRefTestPointsForNeighbors(meshTopo, cell, sideOrdinal, fineSideRefPoints, fineCellRefPoints, coarseSideRefPoints, coarseCellRefPoints);
      if (!hasCoarserNeighbor) continue;

      pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);

      CellPtr neighborCell = meshTopo->getCell(neighborInfo.first);

      unsigned numTestPoints = coarseCellRefPoints.dimension(0);

      //        cout << "testing neighbor agreement between cell " << cellID << " and " << neighborCell->cellIndex() << endl;

      // if we get here, the cell has a neighbor on this side, and is at least as fine as that neighbor.

      BasisCachePtr fineSideBasisCache = fineCellBasisCache->getSideBasisCache(sideOrdinal);

      fineCellBasisCache->setRefCellPointsFC(fineCellRefPoints);

      BasisCachePtr coarseCellBasisCache = BasisCache::basisCacheForCell(mesh, neighborInfo.first);
      BasisCachePtr coarseSideBasisCache = coarseCellBasisCache->getSideBasisCache(neighborInfo.second);

      coarseCellBasisCache->setRefCellPointsFC(coarseCellRefPoints);

      bool hasSidePoints = (fineSideRefPoints.size() > 0);
      if (hasSidePoints)
      {
        fineSideBasisCache->setRefCellPointsFC(fineSideRefPoints);
        coarseSideBasisCache->setRefCellPointsFC(coarseSideRefPoints);
      }

      // sanity check: do physical points match?

      Camellia::FieldContainer<double> fineCellPhysicalCubaturePoints = fineCellBasisCache->getPhysicalCubaturePointsFC();
      Camellia::FieldContainer<double> coarseCellPhysicalCubaturePoints = coarseCellBasisCache->getPhysicalCubaturePointsFC();

      double tol = 1e-13;
      double maxDiff = 0;
      if (! fcsAgree(coarseCellPhysicalCubaturePoints, fineCellPhysicalCubaturePoints, tol, maxDiff) )
      {
        cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine and coarse cell cubature points do not agree.\n";
        success = false;
        continue;
      }

      if (hasSidePoints)
      {
        Camellia::FieldContainer<double> fineSidePhysicalCubaturePoints = fineSideBasisCache->getPhysicalCubaturePointsFC();
        Camellia::FieldContainer<double> coarseSidePhysicalCubaturePoints = coarseSideBasisCache->getPhysicalCubaturePointsFC();

        double tol = 1e-14;
        double maxDiff = 0;
        if (! fcsAgree(fineSidePhysicalCubaturePoints, fineCellPhysicalCubaturePoints, tol, maxDiff) )
        {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine side and cell cubature points do not agree.\n";
          success = false;
          continue;
        }
        if (! fcsAgree(coarseSidePhysicalCubaturePoints, coarseCellPhysicalCubaturePoints, tol, maxDiff) )
        {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: coarse side and cell cubature points do not agree.\n";
          success = false;
          continue;
        }
        if (! fcsAgree(coarseSidePhysicalCubaturePoints, fineSidePhysicalCubaturePoints, tol, maxDiff) )
        {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine and coarse side cubature points do not agree.\n";
          success = false;
          continue;
        }
      }
      
      GlobalIndexType neighborCellID = neighborInfo.first;
      if (myCellIDs.find(neighborCellID) == myCellIDs.end() )
      {
        // non-local neighbor; it's probably OK to skip testing this one -- we
        // more or less cover this case when we run the same test in serial.
        // (It'd be kind of a pain to do this the right way, asking neighbor's owner to
        //  compute...)
        continue;
      }

      ElementTypePtr coarseElementType = mesh->getElementType(neighborCellID);
      DofOrderingPtr coarseElemTrialOrder = coarseElementType->trialOrderPtr;

      Camellia::FieldContainer<double> coarseSolutionCoefficients(coarseElemTrialOrder->totalDofs());
      mesh->globalDofAssignment()->interpretGlobalCoefficients(neighborInfo.first, coarseSolutionCoefficients, globalSolutionCoefficients);

      set<int> varIDs = fineElemTrialOrder->getVarIDs();
      for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++)
      {
        int varID = *varIt;
//        cout << "MeshTestUtility: varID " << varID << ":\n";
        bool isTraceVar = mesh->bilinearForm()->isFluxOrTrace(varID);
        BasisPtr fineBasis, coarseBasis;
        BasisCachePtr fineCache, coarseCache;
        if (isTraceVar)
        {
          if (! hasSidePoints) continue; // then nothing to do for traces
          fineBasis = fineElemTrialOrder->getBasis(varID, sideOrdinal);
          coarseBasis = coarseElemTrialOrder->getBasis(varID, neighborInfo.second);
          fineCache = fineSideBasisCache;
          coarseCache = coarseSideBasisCache;
        }
        else
        {
          fineBasis = fineElemTrialOrder->getBasis(varID);
          coarseBasis = coarseElemTrialOrder->getBasis(varID);
          fineCache = fineCellBasisCache;
          coarseCache = coarseCellBasisCache;

          Camellia::EFunctionSpace fs = fineBasis->functionSpace();
          if ((fs == Camellia::FUNCTION_SPACE_HVOL)
              || (fs == Camellia::FUNCTION_SPACE_VECTOR_HVOL)
              || (fs == Camellia::FUNCTION_SPACE_TENSOR_HVOL))
          {
            // volume L^2 basis: no continuities expected...
            continue;
          }
        }
        Camellia::FieldContainer<double> localValuesFine = *fineCache->getTransformedValues(fineBasis, OP_VALUE);
        Camellia::FieldContainer<double> localValuesCoarse = *coarseCache->getTransformedValues(coarseBasis, OP_VALUE);

        bool scalarValued = (localValuesFine.rank() == 3);

        // the following used if vector or tensor-valued:
        Teuchos::Array<int> valueDim;
        unsigned componentsPerValue = 1;
        Camellia::FieldContainer<double> valueContainer; // just used for enumeration computation...
        if (!scalarValued)
        {
          localValuesFine.dimensions(valueDim);
          // clear first three:
          valueDim.erase(valueDim.begin());
          valueDim.erase(valueDim.begin());
          valueDim.erase(valueDim.begin());
          valueContainer.resize(valueDim);
          componentsPerValue = valueContainer.size();
        }

        //          if (localValuesFine.rank() != 3) {
        //            cout << "WARNING: MeshTestUtility::neighborBasesAgreeOnSides() only supports scalar-valued bases right now.  Skipping check for varID " << varID << endl;
        //            continue;
        //          }

        Camellia::FieldContainer<double> localPointValuesFine(fineElemTrialOrder->totalDofs());
        Camellia::FieldContainer<double> localPointValuesCoarse(coarseElemTrialOrder->totalDofs());

        for (int valueComponentOrdinal=0; valueComponentOrdinal<componentsPerValue; valueComponentOrdinal++)
        {
          Teuchos::Array<int> valueMultiIndex(valueContainer.rank());

          if (!scalarValued)
            valueContainer.getMultiIndex(valueMultiIndex, valueComponentOrdinal);

          Teuchos::Array<int> localValuesMultiIndex(localValuesFine.rank());

          for (int r=0; r<valueMultiIndex.size(); r++)
          {
            localValuesMultiIndex[r+3] = valueMultiIndex[r];
          }

          for (int ptOrdinal=0; ptOrdinal<numTestPoints; ptOrdinal++)
          {
            localPointValuesCoarse.initialize(0);
            localPointValuesFine.initialize(0);
            localValuesMultiIndex[2] = ptOrdinal;

            double fineSolutionValue = 0, coarseSolutionValue = 0;

            for (int basisOrdinal=0; basisOrdinal < fineBasis->getCardinality(); basisOrdinal++)
            {
              int fineDofIndex;
              if (isTraceVar)
                fineDofIndex = fineElemTrialOrder->getDofIndex(varID, basisOrdinal, sideOrdinal);
              else
                fineDofIndex = fineElemTrialOrder->getDofIndex(varID, basisOrdinal);
              if (scalarValued)
              {
                localPointValuesFine(fineDofIndex) = localValuesFine(0,basisOrdinal,ptOrdinal);
              }
              else
              {
                localValuesMultiIndex[1] = basisOrdinal;
                localPointValuesFine(fineDofIndex) = localValuesFine.getValue(localValuesMultiIndex);
              }

              fineSolutionValue += fineSolutionCoefficients(fineDofIndex) * localPointValuesFine(fineDofIndex);
            }
            for (int basisOrdinal=0; basisOrdinal < coarseBasis->getCardinality(); basisOrdinal++)
            {
              int coarseDofIndex;
              if (isTraceVar)
                coarseDofIndex = coarseElemTrialOrder->getDofIndex(varID, basisOrdinal, neighborInfo.second);
              else
                coarseDofIndex = coarseElemTrialOrder->getDofIndex(varID, basisOrdinal);
              if (scalarValued)
              {
                localPointValuesCoarse(coarseDofIndex) = localValuesCoarse(0,basisOrdinal,ptOrdinal);
              }
              else
              {
                localValuesMultiIndex[1] = basisOrdinal;
                localPointValuesCoarse(coarseDofIndex) = localValuesCoarse.getValue(localValuesMultiIndex);
              }
              coarseSolutionValue += coarseSolutionCoefficients(coarseDofIndex) * localPointValuesCoarse(coarseDofIndex);
            }

            if (abs(coarseSolutionValue - fineSolutionValue) > 1e-13)
            {
              success = false;
              cout << "coarseSolutionValue (" << coarseSolutionValue << ") and fineSolutionValue (" << fineSolutionValue << ") differ by " << abs(coarseSolutionValue - fineSolutionValue);
              cout << " at point " << ptOrdinal << " for varID " << varID << ".  ";
              cout << "This may be an indication that something is amiss with the global-to-local map.\n";
            }
            else
            {
//              // DEBUGGING:
//              cout << "solution value at point (";
//              for (int d=0; d<spaceDim-1; d++) {
//                cout << fineSidePhysicalCubaturePoints(0,ptOrdinal,d) << ", ";
//              }
//              cout << fineSidePhysicalCubaturePoints(0,ptOrdinal,spaceDim-1) << "): ";
//              cout << coarseSolutionValue << endl;
            }

            Camellia::FieldContainer<double> globalValuesFromFine, globalValuesFromCoarse;
            Camellia::FieldContainer<GlobalIndexType> globalDofIndicesFromFine, globalDofIndicesFromCoarse;

            mesh->globalDofAssignment()->interpretLocalData(cellID, localPointValuesFine, globalValuesFromFine, globalDofIndicesFromFine);
            mesh->globalDofAssignment()->interpretLocalData(neighborInfo.first, localPointValuesCoarse, globalValuesFromCoarse, globalDofIndicesFromCoarse);

            std::map<GlobalIndexType, double> fineValuesMap;
            std::map<GlobalIndexType, double> coarseValuesMap;

            for (int i=0; i<globalDofIndicesFromCoarse.size(); i++)
            {
              GlobalIndexType globalDofIndex = globalDofIndicesFromCoarse[i];
              coarseValuesMap[globalDofIndex] = globalValuesFromCoarse[i];
            }

            double maxDiff = 0;
            for (int i=0; i<globalDofIndicesFromFine.size(); i++)
            {
              GlobalIndexType globalDofIndex = globalDofIndicesFromFine[i];
              fineValuesMap[globalDofIndex] = globalValuesFromFine[i];

              double diff = abs( fineValuesMap[globalDofIndex] - coarseValuesMap[globalDofIndex]);
              maxDiff = std::max(diff, maxDiff);
              if (diff > tol)
              {
                success = false;
                cout << "interpreted fine and coarse disagree at point (";
                for (int d=0; d<spaceDim; d++)
                {
                  cout << fineCellPhysicalCubaturePoints(0,ptOrdinal,d);
                  if (d==spaceDim-1)
                    cout <<  ").\n";
                  else
                    cout << ", ";
                }
              }
            }
            if (maxDiff > tol)
            {
              cout << "maxDiff: " << maxDiff << endl;
              cout << "globalValuesFromFine:\n" << globalValuesFromFine;
              cout << "globalValuesFromCoarse:\n" << globalValuesFromCoarse;

              cout << "globalDofIndicesFromFine:\n" << globalDofIndicesFromFine;
              cout <<  "globalDofIndicesFromCoarse:\n" << globalDofIndicesFromCoarse;

              continue; // only worth testing further if we passed the above
            }
          }
        }
      }
    }
  }

//  cout << "Completed neighborBasesAgreeOnSides.\n";
  return success;
}

