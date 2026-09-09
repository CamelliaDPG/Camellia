//
//  SolutionMoments.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 9/28/21.
//

#include "SolutionMoments.h"

#include "Camellia_TeuchosSerializers.hpp"
#include "doubleBasisConstruction.h"
#include "GlobalDofAssignment.h"
#include "SolutionMomentFunction.h"
#include "TensorBasis.h"

#include "CamelliaDebugUtility.h"

#include <Intrepid2_SerendipityBasis.hpp>
#include <Intrepid2_TensorBasis.hpp>

using namespace Camellia;

namespace
{
  //! Compute (n choose k)
  KOKKOS_INLINE_FUNCTION
  int choose(const int &n, const int &k)
  {
    int result = 1;
    if (n - k < k)
    {
      return choose(n, n-k);
    }
    else
    {
      for (int i=1; i<=k; i++)
      {
        result *= (n - k + i);
      }
      for (int i=2; i<=k; i++)
      {
        result /= i;
      }
    }
    return result;
  }

  //! Compute (n choose k) a^(n-k) b^k
  KOKKOS_INLINE_FUNCTION
  double binomialCoefficient(const double &a, const double &b, const int &n, const int &k)
  {
    return pow(a,n-k) * pow(b,k) * choose(n,k);
  }
}

SolutionMoments::SolutionMoments(const int &maxMomentNumber, SolutionPtr phaseSpaceSoln, VarPtr fieldVar, MeshPtr physicalMesh, int survivingExtrusionDimension)
:
_maxMomentNumber(maxMomentNumber),
_survivingExtrusionDimension(survivingExtrusionDimension),
_basisReconciliation(true) // true: cache results
{
  _soln = phaseSpaceSoln;
  _fieldVar = fieldVar;
  
  // check that physical mesh has the right number of dimensions:
  const int numDimensionsPhaseSpace   = phaseSpaceSoln->mesh()->getDimension();
  
  int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension == -1) ? 0 : 1;
  ExtrudedMeshTopology* phaseSpaceExtrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(phaseSpaceSoln->mesh()->getTopology().get());
  _numExtrusionDimensions = phaseSpaceExtrudedMeshTopology->numExtrusionDimensions();
  _numDimensionsPhysicalMesh = numDimensionsPhaseSpace - _numExtrusionDimensions + numSurvivingExtrusionDimensions;
  
  setMesh(physicalMesh);
  
  auto globalDofAssignment = _soln->mesh()->globalDofAssignment();
  int pInitial             = globalDofAssignment->getInitialH1Order()[0];
  int pMax                 = globalDofAssignment->getMaxH1Order();
  int pMin                 = globalDofAssignment->getMinH1Order();
  
  auto cellTopo = globalDofAssignment->getSampleElementType()->cellTopoPtr;
  
  std::map<int, BasisPtr> physicalDpToBasisMap;
  
  for (int p=pMin; p<=pMax; p++)
  {
    int dp = p - pInitial;
    auto solnMoments_dp = Teuchos::rcp( new SolutionMomentsForPolyOrder(dp, maxMomentNumber, phaseSpaceSoln, fieldVar, physicalMesh, survivingExtrusionDimension) );
    
    _momentsForDeltaP[dp] = solnMoments_dp;
  }
}

const Camellia::DeviceView<double> SolutionMoments::basisCoefficients() const
{
  return _spatialBasisCoefficients;
}

BasisPtr SolutionMoments::extrudedSpatialBasis()
{
  return _highestOrderExtrudedSpatialBasis;
}

void SolutionMoments::importCellData(std::vector<GlobalIndexType> configurationSpaceCellIDs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_configurationSpaceMesh == Teuchos::null, std::invalid_argument, "SolutionMoments::importCellData() requires a non-null configuration-space mesh");
  
  const int rank = _configurationSpaceMesh->Comm()->getRank();
  
  std::sort(configurationSpaceCellIDs.begin(), configurationSpaceCellIDs.end());
  configurationSpaceCellIDs.erase(std::unique(configurationSpaceCellIDs.begin(), configurationSpaceCellIDs.end()), configurationSpaceCellIDs.end());
  
  using RequestType = std::pair<IndexType,int>;          // (configurationSpaceCellID, requestingRank)
  using DataType    = std::tuple<IndexType,int,int,double>; // (configurationSpaceCellID,momentEntryOrdinal,spatialBasisOrdinal,value)
  
  // Collect requests only for cells that are not already available locally.  Use
  // a collective all-gather for the request phase so every rank participates and
  // every owning rank can see requests for its cells.  Coefficient data are still
  // sent only from owning ranks to ranks that actually need them below.
  std::vector<RequestType> myRequests;
  for (const auto &cellID : configurationSpaceCellIDs)
  {
    if (localCellOrdinal(cellID) == -1)
    {
      const int ownerRank = _configurationSpaceMesh->partitionForCellID(cellID);
      TEUCHOS_TEST_FOR_EXCEPTION((ownerRank < 0) || (ownerRank >= _configurationSpaceMesh->Comm()->getSize()), std::invalid_argument, "SolutionMoments::importCellData() could not determine a valid owning rank for a requested cell");
      myRequests.push_back(std::make_pair(cellID,rank));
    }
  }
  
  std::vector<RequestType> requestsReceived;
  std::vector<int> requestOffsets;
  MPIWrapper::allGatherVariable(_configurationSpaceMesh->Comm(), requestsReceived, myRequests, requestOffsets);
  
  auto spatialBasisCoefficientsHost = Kokkos::create_mirror_view(_spatialBasisCoefficients);
  Kokkos::deep_copy(spatialBasisCoefficientsHost,_spatialBasisCoefficients);
  
  const int numMomentEntries = _spatialBasisCoefficients.extent_int(0);
  const int basisCardinality = _spatialBasisCoefficients.extent_int(2);
  
  std::map<int,std::vector<DataType>> dataToSend;
  for (const auto &request : requestsReceived)
  {
    const IndexType &cellID       = request.first;
    const int &requestingRank     = request.second;
    const int ownerRank           = _configurationSpaceMesh->partitionForCellID(cellID);
    if (ownerRank != rank)
    {
      continue;
    }
    
    const int cellOrdinal         = localCellOrdinal(cellID);
    
    TEUCHOS_TEST_FOR_EXCEPTION(cellOrdinal == -1, std::invalid_argument, "SolutionMoments::importCellData() received a request for a cell that is not available on the owning rank");
    
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int momentOrdinal=0; momentOrdinal<numMomentEntries; momentOrdinal++)
      {
        const double &value = spatialBasisCoefficientsHost(momentOrdinal,cellOrdinal,basisOrdinal);
        dataToSend[requestingRank].push_back(std::make_tuple(cellID,momentOrdinal,basisOrdinal,value));
      }
    }
  }
  
  std::vector<DataType> dataReceived;
  MPIWrapper::sendDataVectors(_configurationSpaceMesh->Comm(), dataToSend, dataReceived);
  
  if (dataReceived.size() == 0)
  {
    return;
  }
  
  std::vector<IndexType> newPhysicalCellIDs = _physicalCellIDs;
  for (const DataType &dataEntry : dataReceived)
  {
    using std::get;
    const IndexType &cellID = get<0>(dataEntry);
    auto matchingEntry = std::lower_bound(newPhysicalCellIDs.begin(), newPhysicalCellIDs.end(), cellID);
    if ((matchingEntry == newPhysicalCellIDs.end()) || (*matchingEntry != cellID))
    {
      newPhysicalCellIDs.insert(matchingEntry, cellID);
    }
  }
  
  if (newPhysicalCellIDs.size() == _physicalCellIDs.size())
  {
    return;
  }
  
  std::vector<int> spatialBasisCoefficientsDims {numMomentEntries, static_cast<int>(newPhysicalCellIDs.size()), basisCardinality};
  auto newSpatialBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(spatialBasisCoefficientsDims, "SolutionMomentFunction imported spatial basis coefficients");
  auto newSpatialBasisCoefficientsHost = Kokkos::create_mirror_view(newSpatialBasisCoefficients);
  Kokkos::deep_copy(newSpatialBasisCoefficientsHost,0.0);
  
  for (int oldCellOrdinal=0; oldCellOrdinal<_physicalCellIDs.size(); oldCellOrdinal++)
  {
    const IndexType &cellID = _physicalCellIDs[oldCellOrdinal];
    auto matchingEntry = std::lower_bound(newPhysicalCellIDs.begin(), newPhysicalCellIDs.end(), cellID);
    TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == newPhysicalCellIDs.end()) || (*matchingEntry != cellID), std::invalid_argument, "existing SolutionMoments cell disappeared during import");
    const int newCellOrdinal = matchingEntry - newPhysicalCellIDs.begin();
    
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int momentOrdinal=0; momentOrdinal<numMomentEntries; momentOrdinal++)
      {
        newSpatialBasisCoefficientsHost(momentOrdinal,newCellOrdinal,basisOrdinal) = spatialBasisCoefficientsHost(momentOrdinal,oldCellOrdinal,basisOrdinal);
      }
    }
  }
  
  for (const DataType &dataEntry : dataReceived)
  {
    using std::get;
    const IndexType & cellID              = get<0>(dataEntry);
    const int &momentEntryOrdinal        = get<1>(dataEntry);
    const int &spatialBasisOrdinal       = get<2>(dataEntry);
    const double &value                  = get<3>(dataEntry);
    
    auto matchingEntry = std::lower_bound(newPhysicalCellIDs.begin(), newPhysicalCellIDs.end(), cellID);
    TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == newPhysicalCellIDs.end()) || (*matchingEntry != cellID), std::invalid_argument, "imported SolutionMoments cell was not added to the local cell list");
    const int newCellOrdinal = matchingEntry - newPhysicalCellIDs.begin();
    
    newSpatialBasisCoefficientsHost(momentEntryOrdinal,newCellOrdinal,spatialBasisOrdinal) = value;
  }
  
  Kokkos::deep_copy(newSpatialBasisCoefficients,newSpatialBasisCoefficientsHost);
  
  if (_spatialBasisCoefficientsHasBeenCheckedOut)
  {
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
  }
  _spatialBasisCoefficients = newSpatialBasisCoefficients;
  _spatialBasisCoefficientsHasBeenCheckedOut = true;
  _physicalCellIDs = newPhysicalCellIDs;
}

void SolutionMoments::gatherOnRankZero()
{
  const int rank = _configurationSpaceMesh->Comm()->getRank();
  
  // send the physical cell contributions to their owners, and store in _spatialBasisCoefficients container
  // we use a convenience method in MPIWrapper for this; note that this sends a bunch of extra data that a manual data-packing implementation would not need to.  (The convenience method requires that we have a vector of fixed-length entries; that's where the inefficiency comes from.)
  using DataType = std::tuple<IndexType,int,int,double>; // (physicalCellID,momentEntryOrdinal,spatialBasisOrdinal,value)
  const auto & myPhysicalCells = _configurationSpaceMesh->cellIDsInPartition();
  
  auto spatialBasisCoefficientsHost = Kokkos::create_mirror_view(_spatialBasisCoefficients);
  Kokkos::deep_copy(spatialBasisCoefficientsHost,_spatialBasisCoefficients);
  
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, _numExtrusionDimensions - numSurvivingExtrusionDimensions);
  const int basisCardinality = _spatialBasisCoefficients.extent_int(2);
 
  std::vector<DataType> dataVector;
  
  for (const auto & physicalCellID : myPhysicalCells)
  {
    auto cellOrdinal = this->localCellOrdinal(physicalCellID);
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int momentOrdinal=0; momentOrdinal<numMomentEntries; momentOrdinal++)
      {
        const auto & value = spatialBasisCoefficientsHost(momentOrdinal,cellOrdinal,basisOrdinal);
        DataType dataEntry = make_tuple(physicalCellID,momentOrdinal,basisOrdinal,value);
        dataVector.push_back(dataEntry);
      }
    }
  }
  
  std::map<int,std::vector<DataType>> dataToSend;
  PartitionIndexType destinationRank = 0;
  dataToSend[destinationRank] = dataVector;
  std::vector<DataType> dataReceived;
  MPIWrapper::sendDataVectors(_configurationSpaceMesh->Comm(), dataToSend, dataReceived);
  
  auto globalActiveCellIDs = _configurationSpaceMesh->getActiveCellIDsGlobal();
  const GlobalIndexType numGlobalActiveCells = globalActiveCellIDs.size();
  
  if (rank == 0)
  {
    // check out a fresh _spatialBasisCoefficients, if size has changed
    if (_spatialBasisCoefficients.extent_int(1) != numGlobalActiveCells)
    {
      // check in previous, if any
      if (_spatialBasisCoefficientsHasBeenCheckedOut)
      {
        SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
      }
      std::vector<int> spatialBasisCoefficientsDims {numMomentEntries, numGlobalActiveCells, basisCardinality};
      // check out a container matching latest dimensions
      _spatialBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(spatialBasisCoefficientsDims, "SolutionMomentFunction spatial basis coefficients");
      _spatialBasisCoefficientsHasBeenCheckedOut = true;
    }
    
    _physicalCellIDs = std::vector<IndexType>(globalActiveCellIDs.begin(), globalActiveCellIDs.end());
    
    auto spatialBasisCoefficientsHost = Kokkos::create_mirror_view(_spatialBasisCoefficients);
    Kokkos::deep_copy(spatialBasisCoefficientsHost,0.0);
    
    for (const DataType &dataEntry : dataReceived)
    {
      using std::get;
      const IndexType & physicalCellID = get<0>(dataEntry);
      const int &momentEntryOrdinal    = get<1>(dataEntry);
      const int &spatialBasisOrdinal   = get<2>(dataEntry);
      const double &value              = get<3>(dataEntry);
      
      // since we updated _physicalCellIDs above, localCellOrdinal() will return the right offset
      auto cellOrdinal = this->localCellOrdinal(physicalCellID);
      {
        // DEBUGGING
        bool outOfBounds = false;
        if (spatialBasisCoefficientsHost.extent_int(0) <= momentEntryOrdinal)
        {
          std::cout << "momentEntryOrdinal " << momentEntryOrdinal << " is out of bounds.\n";
          outOfBounds = true;
        }
        if (spatialBasisCoefficientsHost.extent_int(1) <= cellOrdinal)
        {
          std::cout << "cellOrdinal " << cellOrdinal << " is out of bounds.\n";
          outOfBounds = true;
        }
        if (spatialBasisCoefficientsHost.extent_int(2) <= spatialBasisOrdinal)
        {
          std::cout << "spatialBasisOrdinal " << spatialBasisOrdinal << " is out of bounds.\n";
          outOfBounds = true;
        }
        if (outOfBounds)
        {
          std::cout << "spatialBasisCoefficientsHost dimensions: (";
          std::cout << spatialBasisCoefficientsHost.extent_int(0) << ",";
          std::cout << spatialBasisCoefficientsHost.extent_int(1) << ",";
          std::cout << spatialBasisCoefficientsHost.extent_int(2) << ")\n";
        }
      }
      spatialBasisCoefficientsHost(momentEntryOrdinal,cellOrdinal,spatialBasisOrdinal) = value;
    }
    Kokkos::deep_copy(_spatialBasisCoefficients,spatialBasisCoefficientsHost);
  }
}

Camellia::DeviceView<double> SolutionMoments::getSpatialBasisCoefficients()
{
  return _spatialBasisCoefficients;
}

int SolutionMoments::localCellOrdinal(const IndexType &cellID) const
{
  auto matchingEntry = std::lower_bound(_physicalCellIDs.begin(), _physicalCellIDs.end(), cellID);
  if ((matchingEntry == _physicalCellIDs.end()) || (*matchingEntry != cellID))
  {
    // cellID not found
    return -1;
  }
  else
  {
    const int cellOrdinal = matchingEntry - _physicalCellIDs.begin();
    return cellOrdinal;
  }
}

void SolutionMoments::registerUpdateListener(SolutionMomentsUpdateListener* listener)
{
  TEUCHOS_TEST_FOR_EXCEPTION(listener == nullptr, std::invalid_argument, "listener may not be null");
  auto existing = std::find(_updateListeners.begin(), _updateListeners.end(), listener);
  if (existing == _updateListeners.end())
  {
    _updateListeners.push_back(listener);
  }
}

void SolutionMoments::unregisterUpdateListener(SolutionMomentsUpdateListener* listener)
{
  auto newEnd = std::remove(_updateListeners.begin(), _updateListeners.end(), listener);
  _updateListeners.erase(newEnd, _updateListeners.end());
}

int SolutionMoments::maxMomentNumber() const
{
  return _maxMomentNumber;
}

MeshPtr SolutionMoments::mesh()
{
  return _configurationSpaceMesh;
}

MeshPtr SolutionMoments::phaseSpaceMesh()
{
  return _soln->mesh();
}

int SolutionMoments::numMomentEntries() const
{
  const int numDims = numVelocityDims();
  return MomentIndices::getNumMomentEntries(_maxMomentNumber+1,numDims);
}

int SolutionMoments::numVelocityDims() const
{
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numDims = _numExtrusionDimensions - numSurvivingExtrusionDimensions;
  
  return numDims;
}

const std::vector<IndexType> & SolutionMoments::physicalCellIDs() const
{
  return _physicalCellIDs;
}
                               
void SolutionMoments::setMesh(MeshPtr configurationSpaceMesh)
{
  _configurationSpaceMesh = configurationSpaceMesh;
  if (_configurationSpaceMesh != Teuchos::null)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_numDimensionsPhysicalMesh != configurationSpaceMesh->getDimension(), std::invalid_argument, "dimension does not match expected");
  }
}

FunctionPtr SolutionMoments::solutionMomentFunction(const int &xMoment)
{
  TEUCHOS_TEST_FOR_EXCEPTION(xMoment > _maxMomentNumber, std::invalid_argument, "moment number exceeds _maxMomentNumber");
  
  // check that this is in fact a 1V situation:
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numDims = _numExtrusionDimensions - numSurvivingExtrusionDimensions;
  
  TEUCHOS_TEST_FOR_EXCEPTION(numDims != 1, std::invalid_argument, "single-argument moment index is only allowed for 1V");
  
//  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, numDims);
  const int momentEntryIndex = MomentIndices::getMomentIndex(xMoment, _maxMomentNumber+1);
  
  return Teuchos::rcp(new SolutionMomentFunction(*this, momentEntryIndex) );
}

FunctionPtr SolutionMoments::solutionMomentFunction(const int &xMoment, const int &yMoment)
{
  TEUCHOS_TEST_FOR_EXCEPTION(xMoment + yMoment > _maxMomentNumber, std::invalid_argument, "moment number exceeds _maxMomentNumber");
  
  // check that this is in fact a 2V situation:
  
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numDims = _numExtrusionDimensions - numSurvivingExtrusionDimensions;
  
  TEUCHOS_TEST_FOR_EXCEPTION(numDims != 2, std::invalid_argument, "two-argument moment index is only allowed for 2V");
  
//  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, numDims);
  const int momentEntryIndex = MomentIndices::getMomentIndex(xMoment, yMoment, _maxMomentNumber+1);
  
  return Teuchos::rcp(new SolutionMomentFunction(*this, momentEntryIndex) );
}

FunctionPtr SolutionMoments::solutionMomentFunction(const int &xMoment, const int &yMoment, const int &zMoment)
{
  TEUCHOS_TEST_FOR_EXCEPTION(xMoment + yMoment + zMoment > _maxMomentNumber, std::invalid_argument, "moment number exceeds _maxMomentNumber");
  
  // check that this is in fact a 3V situation:
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numDims = _numExtrusionDimensions - numSurvivingExtrusionDimensions;
  
  TEUCHOS_TEST_FOR_EXCEPTION(numDims != 3, std::invalid_argument, "three-argument moment index is only allowed for 3V");
  
//  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, numDims);
  const int momentEntryIndex = MomentIndices::getMomentIndex(xMoment, yMoment, zMoment, _maxMomentNumber+1);
  
  return Teuchos::rcp(new SolutionMomentFunction(*this, momentEntryIndex) );
}

//! recompute all moments.
void SolutionMoments::update()
{
  auto gda = _soln->mesh()->globalDofAssignment();
  int pMax = gda->getMaxH1Order();
  int pMin = gda->getMinH1Order();
  int p0   = gda->getInitialH1Order()[0];
  for (int p=pMin; p<=pMax; p++)
  {
    const int dp = p - p0;
    if (_momentsForDeltaP.find(dp) == _momentsForDeltaP.end())
    {
      _momentsForDeltaP[dp] = Teuchos::rcp( new SolutionMomentsForPolyOrder(dp, _maxMomentNumber, _soln, _fieldVar,
                                                                            _configurationSpaceMesh, _survivingExtrusionDimension) );
    }
    _momentsForDeltaP[dp]->update();
  }
  
  _highestOrderExtrudedSpatialBasis = _momentsForDeltaP[pMax-p0]->extrudedSpatialBasis();
    
  MeshPtr phaseSpaceMesh = _soln->mesh();
  const auto & myPhaseSpaceCells = phaseSpaceMesh->cellIDsInPartition();
  const auto & myPhysicalCells = _configurationSpaceMesh->cellIDsInPartition();
    
  using Intrepid2Basis   = Intrepid2::Basis<DeviceType,double,double>;
  
  _physicalCellIDs = std::vector<IndexType>(myPhysicalCells.begin(),myPhysicalCells.end());
  const int numCells = myPhysicalCells.size();
  int extrudedSpatialBasisCardinality = _highestOrderExtrudedSpatialBasis->getCardinality();
  
  const int numCellsPrevious                 = _spatialBasisCoefficients.extent_int(1);
  const int physicalBasisCardinalityPrevious = _spatialBasisCoefficients.extent_int(2);
  
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, _numExtrusionDimensions - numSurvivingExtrusionDimensions);
    
  // make sure _spatialBasisCoefficients is properly allocated
  if ((numCells != numCellsPrevious) || (extrudedSpatialBasisCardinality != physicalBasisCardinalityPrevious))
  {
    // reallocate _spatialBasisCoefficients containers
    std::vector<int> spatialBasisCoefficientsDims {numMomentEntries, numCells, extrudedSpatialBasisCardinality};
    
    // check in previous, if any
    if (_spatialBasisCoefficientsHasBeenCheckedOut)
    {
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
    }
    // check out current
    _spatialBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(spatialBasisCoefficientsDims, "SolutionMomentFunction spatial basis coefficients");
    _spatialBasisCoefficientsHasBeenCheckedOut = true;
  }
  
  auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(_soln->mesh()->getTopology()->baseMeshTopology());
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopology == NULL, std::invalid_argument, "phase space MeshTopology must be an ExtrudedMeshTopology");
  
  if (numCells > 0)
  {
    // special case for constant p
    // this allows us to use non-hierarchical bases if there have not been p refinements
    // (currently, prolongation only supported for hierarchical bases)
    if (pMin == pMax)
    {
      // just one moments-for-fixed-p object; copy from there:
      auto momentsForDeltaP = _momentsForDeltaP[pMin-p0];
      Kokkos::deep_copy(_spatialBasisCoefficients, momentsForDeltaP->getSpatialBasisCoefficients());
    }
    else
    {
      // Prolongate from various dp to _highestOrderExtrudedSpatialBasis
      const bool sumInto = true;
      // clear values before accumulating
      Kokkos::deep_copy(_spatialBasisCoefficients, 0.0);
      
      for (int p=pMin; p<=pMax; p++)
      {
        std::pair<int,int> prolongatorKey {p, pMax};
        
        auto momentsForDeltaP = _momentsForDeltaP[p-p0];
        
        if (_physicalMeshProlongators.find(prolongatorKey) == _physicalMeshProlongators.end())
        {
          auto coarseBasis = momentsForDeltaP->extrudedSpatialBasis()->intrepid2Basis();
          _physicalMeshProlongators[prolongatorKey] = Teuchos::rcp( new Prolongator(coarseBasis, _highestOrderExtrudedSpatialBasis->intrepid2Basis()) );
        }
        // TODO: something to enable (M,C,F) shape: either reshape, or add support for "(C1,C2,F)" use case to prolongator
        _physicalMeshProlongators[prolongatorKey]->apply(_spatialBasisCoefficients, momentsForDeltaP->getSpatialBasisCoefficients(), sumInto);
      }
    }
  }
  
  for (auto listener : _updateListeners)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(listener == nullptr, std::invalid_argument, "registered SolutionMoments update listener is null");
    listener->update();
  }
}

void SolutionMoments::valuesForAllMoments(DeviceView<double> values, BasisCachePtr configurationSpaceBasisCache)
{
  /*
   TODO: During update(), ensure that the physicalCellIDs corresponding to owned phase-space cellIDs are present in _spatialBasisCoefficients.
   (update() is expected/allowed to do MPI communication.  valuesForAllMoments() is not.)
   */
  
  TEUCHOS_FUNC_TIME_MONITOR("SolutionMoments::valuesForAllMoments()");
 
  const int numMomentEntries = _spatialBasisCoefficients.extent_int(0);
  TEUCHOS_TEST_FOR_EXCEPTION(numMomentEntries != values.extent_int(2), std::invalid_argument, "values must have shape (C,P,M)");
  
  // this implementation adapted from BasisSumFunction
  
  BasisPtr extrudedSpatialBasis = this->extrudedSpatialBasis();
  
  const int numDofs = extrudedSpatialBasis->getCardinality();
  const int spaceDim = configurationSpaceBasisCache->cellTopology()->getDimension();

  bool basisIsVolumeBasis = extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
  bool useCubPointsSideRefCell = basisIsVolumeBasis && configurationSpaceBasisCache->isSideCache();

  constFCPtr transformedValues = configurationSpaceBasisCache->getTransformedValues(extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
  
  // transformedValues has dimensions (C,F,P)
  // therefore, the rank of the sum is transformedValues->rank() - 3
  TEUCHOS_TEST_FOR_EXCEPTION(3 != values.rank(), std::invalid_argument, "values rank must be 3 (C,P,M).");
  TEUCHOS_TEST_FOR_EXCEPTION(3 != transformedValues->rank(), std::invalid_argument, "transformedValues rank must be 3 (C,F,P).");

  auto dv_transformedValues = transformedValues->checkoutDeviceView();
  
  auto spatialBasisCoefficients  = this->getSpatialBasisCoefficients();
  
  int numCellIDs = configurationSpaceBasisCache->cellIDs().size();
  std::vector<int> physicalCellOrdinalLookupDims {numCellIDs};
  auto physicalCellOrdinalLookup = SharedScratchStorage<IndexType>::sharedScratchStorage().checkoutScratchStorage(physicalCellOrdinalLookupDims, "SolutionMomentFunction physical cell ordinal lookup");
  
  const auto & physicalCellIDs   = this->physicalCellIDs();
  
  auto cellOrdinalLookupHost = Kokkos::create_mirror_view(physicalCellOrdinalLookup);
  int basisCacheCellOrdinal = 0;
  for (const auto &cellID : configurationSpaceBasisCache->cellIDs())
  {
    auto matchingEntry = std::lower_bound(physicalCellIDs.begin(), physicalCellIDs.end(), cellID);
    TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == physicalCellIDs.end()) || *matchingEntry != cellID,std::invalid_argument,"cellID not found");
    
    int cellOrdinal = matchingEntry - physicalCellIDs.begin();
    
    cellOrdinalLookupHost(basisCacheCellOrdinal) = cellOrdinal;
    basisCacheCellOrdinal++;
  }
  Kokkos::deep_copy(physicalCellOrdinalLookup, cellOrdinalLookupHost);
  
  Kokkos::deep_copy(values, 0.0); // initialize
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
  Kokkos::parallel_for("SolutionMomentFunction_values", FieldContainer<double>::getMDRangePolicy3(values),
  KOKKOS_LAMBDA (const array_index_type cellOrdinal, const array_index_type pointOrdinal, const array_index_type momentEntryOrdinal)
  {
    const int &cellOrdinalInSpatialBasisCoefficients = physicalCellOrdinalLookup(cellOrdinal);
    for (int dof = 0; dof < numDofs; dof++)
    {
      double weight = spatialBasisCoefficients(momentEntryOrdinal,cellOrdinalInSpatialBasisCoefficients,dof);
      values(cellOrdinal,pointOrdinal,momentEntryOrdinal) += dv_transformedValues.access(cellOrdinal,dof,pointOrdinal) * weight;
    }
  });
  transformedValues->checkinDeviceView(false); // false: don't copy back to host (has not changed)
  SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
}

SolutionMoments::~SolutionMoments()
{
  for (auto *listener : _updateListeners)
  {
    if (listener == nullptr) continue;
    auto relativisticListener = dynamic_cast<RelativisticFirstMomentFunction*>(listener);
    if (relativisticListener != nullptr)
    {
      relativisticListener->detachFromDrivingMoments();
    }
  }
  _updateListeners.clear();

  if (_spatialBasisCoefficientsHasBeenCheckedOut)
  {
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
  }
}


// below, we set up unit tests against the private classes and methods above
#include "Teuchos_UnitTestHarness.hpp"

TEUCHOS_UNIT_TEST( SolutionMoments_cpp, BinomialSum )
{
  // check that the binomialCoefficient() method works to compute appropriate weights.
  const double a = 0.75;
  const double b = 0.25;

  /*
   Binomial theorem says
   
   (a + b)^n = sum_{k=0..n} ( (n choose k) a^{n-k} b^k )
   
   Since a + b = 1, (a+b)^n = 1 as well.
   
   */
  
  const double tol = 1e-16; // we should loosen this…
  const int maxPower = 10;
  for (int n=0; n<=maxPower; n++)
  {
    double binomialSum = 0.0;
    for (int k=0; k<=n; k++)
    {
      const double binomialTerm = binomialCoefficient(a, b, n, k);
      binomialSum += binomialTerm;
    }
    out << "\nTesting n = " << n << std::endl;
    TEST_FLOATING_EQUALITY(binomialSum, 1.0, tol);
  }
}

TEUCHOS_UNIT_TEST( SolutionMoments_cpp, Choose )
{
  // manually compute factorials defining (n choose k)
  // compare to what we compute in the (more efficient) choose() method
  const int maxPower = 10; // small enough that we can use integer arithmetic for everything
  
  int n_factorial = 1;
  for (int n=0; n<=maxPower; n++)
  {
    if (n != 0)  n_factorial *= n;
    
    int k_factorial = 1;
    int n_minus_k_factorial = n_factorial;
    for (int k=0; k<=n; k++)
    {
      if (k != 0)
      {
        k_factorial *= k;

        if (k != n)
        {
          n_minus_k_factorial /= (n - (k - 1));
        }
      }
      int expected = n_factorial / (k_factorial * n_minus_k_factorial);
      int actual = choose(n, k);
      out << "Testing (" << n << " choose " << k << ")\n";
      TEST_EQUALITY(expected, actual);
    }
  }
}
