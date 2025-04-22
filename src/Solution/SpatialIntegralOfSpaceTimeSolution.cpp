//
//  SpatialIntegralOfSpaceTimeSolution.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 9/28/21.
//

#include "SpatialIntegralOfSpaceTimeSolution.h"

#include "BasisReconciliation.h"
#include "Camellia_TensorIndices.h"
#include "CubatureFactory.h"
#include "ExtrudedMeshTopology.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshBasisIntegrals.h"
#include "MPIWrapper.h"
#include "Solution.h"

#include "doubleBasisConstruction.h"
#include "TensorBasis.h"

#include "CamelliaDebugUtility.h"

#include <Intrepid2_TensorBasis.hpp>

using namespace Camellia;

SpatialIntegralOfSpaceTimeSolution::SpatialIntegralOfSpaceTimeSolution(SolutionPtr spaceTimeSoln, VarPtr fieldVar, FunctionPtr spatialWeight, int quadratureEnrichment)
{  
  auto elemType = spaceTimeSoln->mesh()->globalDofAssignment()->getSampleElementType();
  auto cellTopo = elemType->cellTopoPtr;
  
  auto trialOrdering  = elemType->trialOrderPtr;
  auto spaceTimeBasis = trialOrdering->getBasis(fieldVar->ID());
  
  using Intrepid2Basis = Intrepid2::Basis<DeviceType,double,double>;
  using TensorBasis = Intrepid2::Basis_TensorBasis<Intrepid2Basis>;
  auto tensorBasis = dynamic_cast<TensorBasis*>(spaceTimeBasis->intrepid2Basis().get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(tensorBasis == NULL, std::invalid_argument, "Basis must be either a TensorBasis");
  
  BasisPtr spatialBasis, temporalBasis;
  
  ::Camellia::Space fs_spatial;
  ::Camellia::Space fs_temporal;
  
  if (tensorBasis != NULL)
  {
    auto basisComponents = tensorBasis->getTensorBasisComponents();
    
    TEUCHOS_TEST_FOR_EXCEPTION(basisComponents.size() < 2, std::invalid_argument, "basis must have at least two tensorial components");
    
    int temporalComponentIndex = basisComponents.size() - 1; // temporal basis is the final component
    // set spatialBasis
    {
      auto spatialBasisIntrepid2 = basisComponents[0];
      ::Camellia::EFunctionSpace fs = fromIntrepid2FunctionSpace(spatialBasisIntrepid2->getFunctionSpace());
      
      fs_spatial = spaceForEFS(fs);
      
      for (int i=1; i<temporalComponentIndex; i++)
      {
        auto extrusionIntrepid2 = basisComponents[i];
        spatialBasisIntrepid2 = Teuchos::rcp( new TensorBasis(spatialBasisIntrepid2, extrusionIntrepid2, spatialBasisIntrepid2->getFunctionSpace()) );
      }
      spatialBasis = Teuchos::rcp(new Intrepid2BasisWrapper<>(spatialBasisIntrepid2, spatialBasisIntrepid2->getDomainDimension(), 0, fs) );
    }
    // set temporalBasis
    auto intrepid2Basis = basisComponents[temporalComponentIndex];
    ::Camellia::EFunctionSpace fs = fromIntrepid2FunctionSpace(intrepid2Basis->getFunctionSpace());
    temporalBasis = Teuchos::rcp(new Intrepid2BasisWrapper<>(intrepid2Basis, 1, 0, fs) );
    
    fs_temporal = spaceForEFS(fs);
  }
  
  bool spaceTimeMeshAllowsPruning = spaceTimeSoln->mesh()->globalDofAssignment()->allowMeshTopologyPruning();
  TEUCHOS_TEST_FOR_EXCEPTION(spaceTimeMeshAllowsPruning, std::invalid_argument, "SpatialIntegralOfSpaceTimeSolution does not support distributed mesh geometry: space-time mesh should disallow pruning");
  
  ExtrudedMeshTopology* spaceTimeExtrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(spaceTimeSoln->mesh()->getTopology().get());
  
  auto spatialMeshTopology  = spaceTimeExtrudedMeshTopology->physicalMeshTopology();
  auto temporalMeshTopology = spaceTimeExtrudedMeshTopology->extrusionMeshTopology(0);
  
  VarFactoryPtr vf_spatial  = VarFactory::varFactory();
  VarFactoryPtr vf_temporal = VarFactory::varFactory();
  
  // create spatial, temporal field vars; these will be associated with the integrated variable on the one hand, and the output solution variable on the other.
  auto u_spatial  = vf_spatial->fieldVar (fieldVar->name() + "_x", fs_spatial);
  auto u_temporal = vf_temporal->fieldVar(fieldVar->name() + "_t", fs_temporal);
  
  // create some dummy test variables (Bubnov-Galerkin, but there's not even a bf associated with this; we'll never use the test space)
  auto v_spatial  =  vf_spatial->testVar("v_x", fs_spatial);
  auto v_temporal = vf_temporal->testVar("v_t", fs_temporal);
  
  vf_spatial->setBasisFamily( BasisFamily::singletonBasisFamily( spatialBasis));
  vf_temporal->setBasisFamily(BasisFamily::singletonBasisFamily(temporalBasis));
  
  int H1Order_spatial  = ( fs_spatial == L2) ?  spatialBasis->getDegree() + 1 :  spatialBasis->getDegree();
  int H1Order_temporal = (fs_temporal == L2) ? temporalBasis->getDegree() + 1 : temporalBasis->getDegree();
  
  int pToAddTest = 0;
  
  map<int,int> p_enhancements; // empty
  bool allowPruning = false; // we don't support distributed geometry…
  MeshPtr mesh_spatial  = Teuchos::rcp( new Mesh( spatialMeshTopology,  vf_spatial,  H1Order_spatial, pToAddTest, p_enhancements, p_enhancements, Teuchos::null, Teuchos::null, allowPruning) );
  MeshPtr mesh_temporal = Teuchos::rcp( new Mesh(temporalMeshTopology, vf_temporal, H1Order_temporal, pToAddTest, p_enhancements, p_enhancements, Teuchos::null, Teuchos::null, allowPruning) );
  
  // create the solution object where we'll store the integrated function:
  _temporalSolution = Solution::solution(mesh_temporal);
  _temporalFunction = Function::solution(u_temporal, _temporalSolution);
  _temporalMesh     = mesh_temporal;
  
  auto &cellIDs_spaceTime = spaceTimeSoln->mesh()->cellIDsInPartition();
  
  std::vector<GlobalIndexType>  cellIDs_spatial_ordered; //  spatial cellIDs in the order they occur in the phase-space mesh (in general, contains duplicates).
  std::vector<GlobalIndexType> cellIDs_temporal_ordered; // temporal cellIDs in the order they occur in the phase-space mesh (in general, contains duplicates).
  
  // track temporal cellIDs to which our space-time cells will contribute
  std::set<GlobalIndexType> activeTemporalCellIDs;
  
  for (const auto & cellID_spaceTime : cellIDs_spaceTime)
  {
    const auto & cellID_spatial = spaceTimeExtrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(cellID_spaceTime);
    const auto &cellID_temporal = spaceTimeExtrudedMeshTopology->extrusionCellIDForPhaseSpaceCellID(cellID_spaceTime, 0);
    cellIDs_spatial_ordered .push_back( cellID_spatial);
    cellIDs_temporal_ordered.push_back(cellID_temporal);
    
    auto temporalCell = temporalMeshTopology->getCell(cellID_temporal);
    if (temporalCell->isParent(temporalMeshTopology))
    {
      auto descendants = temporalCell->getDescendants(temporalMeshTopology, true); // true: leaf nodes (active cells) only
      for (const auto &activeCellID : descendants)
      {
        activeTemporalCellIDs.insert(activeCellID);
      }
    }
    else
    {
      activeTemporalCellIDs.insert(cellID_temporal);
    }
  }
  std::set<GlobalIndexType> cellIDs_spatial(cellIDs_spatial_ordered.begin(),cellIDs_spatial_ordered.end()); // sort and de-duplicate.
  
  MeshBasisIntegrals meshBasisIntegrals(mesh_spatial, spatialBasis, spatialWeight, quadratureEnrichment, cellIDs_spatial);
  
  const int numCells = cellIDs_spaceTime.size();
  const int temporalBasisCardinality = temporalBasis->getCardinality();
  const int numSpatialSolutions = temporalBasisCardinality;
  const int numSpatialFields    = spatialBasis->getCardinality();
    
  DeviceView<double> spatialCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numSpatialSolutions,numSpatialFields},"SpatialIntegralOfSpaceTimeSolution spatialCoefficients"); // (C,N,F)
  DeviceView<double> solutionIntegrals = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numSpatialSolutions},"SpatialIntegralOfSpaceTimeSolution solutionIntegrals"); // (C,N)
  
  // copy from spaceTimeSoln into spatialCoefficients
  FieldContainer<double> basisCoefficients;
  int spaceTimeCellOrdinal = 0;
  for (const auto & cellID_spaceTime : cellIDs_spaceTime)
  {
    spaceTimeSoln->getBasisCoefficientsForCell(basisCoefficients, fieldVar->ID(), cellID_spaceTime, STANDARD_SOLUTION, 0);
    auto basisCoefficientsDeviceView = basisCoefficients.checkoutDeviceView();
    
    auto rangePolicy = Camellia::FieldContainer<double>::getMDRangePolicy<2>({numSpatialSolutions,numSpatialFields});
    Kokkos::parallel_for("copy into spatialCoefficients", rangePolicy,
    KOKKOS_LAMBDA (const int solutionOrdinal, const int spatialFieldOrdinal) {
      int spaceTimeFieldOrdinal;
      TensorIndices::setTensorOrdinal(spaceTimeFieldOrdinal, spatialFieldOrdinal, numSpatialFields, solutionOrdinal);
      spatialCoefficients(spaceTimeCellOrdinal, solutionOrdinal, spatialFieldOrdinal) = basisCoefficientsDeviceView(spaceTimeFieldOrdinal);
    });
    
    Kokkos::fence(); // pretty sure we need this fence to ensure basisCoefficientsDeviceView has the right data; there may be a better way...
    
    spaceTimeCellOrdinal++;
    basisCoefficients.checkinDeviceView(false);
  }
  
  meshBasisIntegrals.solutionIntegral(solutionIntegrals, spatialCoefficients, cellIDs_spatial_ordered);
  
  auto solutionIntegralsHost = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), solutionIntegrals);
  
  // For each active temporal cell corresponding to owned space-time cell, compute entry-wise (N) coefficient sum.
  // For each inactive temporal cell corresponding to owned space-time cell, compute its contribution to its active descendents, using BasisReconciliation (see SolutionMoments.cpp:770-804).
  
  int numTemporalCellsForMySpaceTimeCells = activeTemporalCellIDs.size();
  FieldContainer<double> temporalCoefficientContributions(numTemporalCellsForMySpaceTimeCells,temporalBasisCardinality); // (C,N)
  BasisReconciliation basisReconciliation;
  
  // more convenient to do this on host; if we want to do this in a high-intensity context, we would do better to rewrite to do this on device...  (Same is probably true generally for BasisReconciliation computations, throughout the code.)
  
  int temporalCellOrdinal = 0;
  std::map<GlobalIndexType,int> activeTemporalCellOrdinalLookup;
  for (const auto &temporalCellID : activeTemporalCellIDs)
  {
    activeTemporalCellOrdinalLookup[temporalCellID] = temporalCellOrdinal;
    
    temporalCellOrdinal++;
  }
  
  spaceTimeCellOrdinal = 0;
  for (const auto & cellID_spaceTime : cellIDs_spaceTime)
  {
    const auto &cellID_temporal = spaceTimeExtrudedMeshTopology->extrusionCellIDForPhaseSpaceCellID(cellID_spaceTime, 0);
    
    auto temporalCell = temporalMeshTopology->getCell(cellID_temporal);
    
    if (!temporalCell->isParent(temporalMeshTopology))
    {
      auto activeTemporalCellOrdinal = activeTemporalCellOrdinalLookup[cellID_temporal];
      for (int basisOrdinal=0; basisOrdinal<numSpatialSolutions; basisOrdinal++)
      {
        temporalCoefficientContributions(activeTemporalCellOrdinal,basisOrdinal) += solutionIntegralsHost(spaceTimeCellOrdinal,basisOrdinal);
      }
    }
    else // temporalCell is a parent: need to use BasisReconcilation
    {
      auto descendants = temporalCell->getDescendants(temporalMeshTopology, true);
      for (const auto & descendantID : descendants)
      {
        auto activeCell = temporalMeshTopology->getCell(descendantID);
        RefinementBranch refBranch = activeCell->refinementBranch(cellID_temporal);
        
        auto activeTemporalCellOrdinal = activeTemporalCellOrdinalLookup[descendantID];
        
        auto basisWeights = basisReconciliation.constrainedWeights(temporalBasis, refBranch, temporalBasis, 0);
        
        if (!basisWeights.isIdentity)
        {
          // cout << "basisWeights.weights:\n" << trueSpatialBasisWeights.weights;
          
          const auto & fineOrdinals   = basisWeights.fineOrdinals;
          const auto & coarseOrdinals = basisWeights.coarseOrdinals;
          auto fineIt = fineOrdinals.begin();
          const int numFineOrdinals   = fineOrdinals.size();
          const int numCoarseOrdinals = coarseOrdinals.size();
          for (int fineOrdinalOrdinal = 0; fineOrdinalOrdinal < numFineOrdinals; fineOrdinalOrdinal++)
          {
            auto coarseIt = coarseOrdinals.begin();
            for (int coarseOrdinalOrdinal = 0; coarseOrdinalOrdinal < numCoarseOrdinals; coarseOrdinalOrdinal++)
            {
              const double & weight = basisWeights.weights(fineOrdinalOrdinal,coarseOrdinalOrdinal);
              temporalCoefficientContributions(activeTemporalCellOrdinal,*fineIt) += weight * solutionIntegralsHost(spaceTimeCellOrdinal,*coarseIt);
              coarseIt++;
            }
            fineIt++;
          }
        }
        else
        {
          // basisWeights is identity
          // this could happen, I think, if the basis was just a constant
          auto descendants = temporalCell->getDescendants(temporalMeshTopology, true);
          
          for (const auto & descendantID : descendants)
          {
            auto activeTemporalCellOrdinal = activeTemporalCellOrdinalLookup[descendantID];
            for (int basisOrdinal=0; basisOrdinal<numSpatialSolutions; basisOrdinal++)
            {
              temporalCoefficientContributions(activeTemporalCellOrdinal,basisOrdinal) += solutionIntegralsHost(spaceTimeCellOrdinal,basisOrdinal);
            }
          }
        }
        
      }
    }
    
    spaceTimeCellOrdinal++;
  }
  
  
  auto &myTemporalCellIDs = mesh_temporal->cellIDsInPartition();

  // send the temporal cell contributions to their owners, and store in _temporalSolution
  {
    // we use a convenience method in MPIWrapper for this; note that this sends a bunch of extra data that a manual data-packing implementation would not need to.  (The convenience method requires that we have a vector of fixed-length entries; that's where the inefficiency comes from.)
    using DataType = std::tuple<IndexType,int,double>; // (temporalCellID,temporalBasisOrdinal,integralContribution)
    std::map<int,std::vector<DataType>> dataToSend;
    
    temporalCellOrdinal = 0;
    for (const auto & activeTemporalCellID : activeTemporalCellIDs)
    {
      auto activeTemporalCellOrdinal = activeTemporalCellOrdinalLookup[activeTemporalCellID];
      
      const PartitionIndexType temporalCellOwner = _temporalMesh->partitionForCellID(activeTemporalCellID);
      TEUCHOS_TEST_FOR_EXCEPTION(temporalCellOwner == -1, std::logic_error, "partition not assigned for temporalCellID -- inactive?");
      
      std::vector<DataType> & ownerDataVector = dataToSend[temporalCellOwner];
      
      for (int basisOrdinal=0; basisOrdinal<temporalBasisCardinality; basisOrdinal++)
      {
        const auto &value = temporalCoefficientContributions(activeTemporalCellOrdinal,basisOrdinal);
        
        DataType dataEntry = make_tuple(activeTemporalCellID,basisOrdinal,value);
        ownerDataVector.push_back(dataEntry);
      }
      
      temporalCellOrdinal++;
    }
    
    std::vector<DataType> dataReceived;
    MPIWrapper::sendDataVectors(MPIWrapper::EpetraComm(_temporalMesh->Comm()), dataToSend, dataReceived);
    
      // note that the following map logic is not maximally efficient, but it's unlikely to be a bottleneck due to the relatively small size of the data
    using std::map;
    using std::pair;
    map<IndexType, FieldContainer<double> > contributionsToMyTemporalCells;
    for (const DataType &dataEntry : dataReceived)
    {
      using std::get;
      const IndexType & temporalCellID = get<0>(dataEntry);
      const int &spatialBasisOrdinal   = get<1>(dataEntry);
      const double &value              = get<2>(dataEntry);
      if (contributionsToMyTemporalCells.find(temporalCellID) == contributionsToMyTemporalCells.end())
      {
        contributionsToMyTemporalCells[temporalCellID] = FieldContainer<double>(temporalBasisCardinality);
      }
      FieldContainer<double> &temporalCellContributions = contributionsToMyTemporalCells[temporalCellID];
      
      temporalCellContributions(spatialBasisOrdinal) += value;
    }
    // confirm that we have the expected number of entries
    if (contributionsToMyTemporalCells.size() != myTemporalCellIDs.size())
    {
      Camellia::print("locally owned physical cells", myTemporalCellIDs);
      std::cout << "contributionsToMyTemporalCells.size() = " << contributionsToMyTemporalCells.size() << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(contributionsToMyTemporalCells.size() != myTemporalCellIDs.size(), std::invalid_argument, "Received cell count differs from number of locally owned temporal cells");
    
    // copy into temporal solution.
    for (const auto &entry : contributionsToMyTemporalCells)
    {
      auto &temporalCellID    = entry.first;
      auto &basisCoefficients = entry.second;
      
      // because of the way that the temporal mesh was constructed, we know that the basis coefficients are the whole trial-space DofOrdering, so we can just set it like this:
      _temporalSolution->setSolnCoeffsForCellID(basisCoefficients, temporalCellID);
    }
  }
  
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(spatialCoefficients);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(solutionIntegrals);
}

void SpatialIntegralOfSpaceTimeSolution::exportViz(std::string parentDir, std::string name)
{
  HDF5Exporter viz(_temporalMesh, name, parentDir);
  viz.exportSolution(_temporalSolution,0);
}

FunctionPtr SpatialIntegralOfSpaceTimeSolution::temporalFunction()
{
  return _temporalFunction;
}

MeshPtr SpatialIntegralOfSpaceTimeSolution::temporalMesh()
{
  return _temporalMesh;
}

SolutionPtr SpatialIntegralOfSpaceTimeSolution::temporalSolution()
{
  return _temporalSolution;
}
