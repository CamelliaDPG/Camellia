//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  ErrorIndicator.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#include "ErrorIndicator.h"

#include "EnergyErrorIndicator.h"
#include "FunctionErrorIndicator.h"
#include "GlobalDofAssignment.h"
#include "GradientErrorIndicator.h"
#include "HessianErrorIndicator.h"
#include "MPIGlobalMinMax.h"
#include "UnrefinementProjectionErrorIndicator.h"

using namespace Camellia;

ErrorIndicator::ErrorIndicator(MeshPtr mesh)
{
  _mesh = mesh;
}

void ErrorIndicator::localCellsAboveErrorThreshold(double threshold, vector<GlobalIndexType> &cellsAboveThreshold)
{
  for (auto measureEntry : _localErrorMeasures) {
    GlobalIndexType cellID = measureEntry.first;
    double cellError = measureEntry.second;
    if (cellError > threshold)
    {
      cellsAboveThreshold.push_back(cellID);
    }
  }
}

void ErrorIndicator::localParentCellsBelowErrorThreshold(double threshold, vector<GlobalIndexType> &cellsBelowThreshold)
{
  for (auto measureEntry : _parentErrorMeasures) {
    GlobalIndexType cellID = measureEntry.first;
    double cellError = measureEntry.second;
    if (cellError < threshold)
    {
      cellsBelowThreshold.push_back(cellID);
    }
  }
}

const std::map<GlobalIndexType,double> & ErrorIndicator::localErrorMeasures() const
{
  return _localErrorMeasures;
}

double ErrorIndicator::maxError(const std::set<GlobalIndexType> localCellsToExclude) const
{
  const double localMax = maxLocalError(localCellsToExclude);
  const double globalMax = MPIWrapper::max(_mesh->Comm(), localMax);
  return globalMax;
}

double ErrorIndicator::maxLocalError(const std::set<GlobalIndexType> localCellsToExclude) const
{
  double maxErr = 0;
  for (auto measureEntry : _localErrorMeasures) {
    auto & cellID = measureEntry.first;
    if (localCellsToExclude.find(cellID) == localCellsToExclude.end())
    {
      double & cellError = measureEntry.second;
      maxErr = max(maxErr,cellError);
    }
  }
  return maxErr;
}

MeshPtr ErrorIndicator::mesh() const
{
  return _mesh;
}

double ErrorIndicator::minError() const
{
  double localMinError  = std::numeric_limits<double>::max();
  for (const auto & errEntry : _localErrorMeasures)
  {
    localMinError = std::fmin(localMinError, errEntry.second);
  }
  
  double globalMinError = MPIWrapper::min(_mesh->Comm(), localMinError);
  return globalMinError;
}

double ErrorIndicator::minParentError() const
{
  double localMinError  = std::numeric_limits<double>::max();
  for (const auto & errEntry : _parentErrorMeasures)
  {
    localMinError = std::fmin(localMinError, errEntry.second);
  }
  
  double globalMinError = MPIWrapper::min(_mesh->Comm(), localMinError);
  return globalMinError;
}

void ErrorIndicator::gatherParentError(const std::set<GlobalIndexType> &excludedChildren)
{
  // this method fills _parentErrorMeasures; clear it first
  _parentErrorMeasures.clear();
  
  // "owner" of the parent is the owner of the first child
  
  // We will communicate the off-rank error values for each child to the parent's owner.
  // MPIWrapper::sendDataMaps has signature
  /* sendDataMaps(Epetra_CommPtr Comm,
                  const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                  std::map<KeyType,ValueType> &receivedMap);*/
  // where ValueType and KeyType can be of arbitrary fixed-size type
  // and the keys to recipientDataMaps are the MPI ranks (PIDs) of the recipients
  // for us, KeyType can be cellID; the ValueType can be double
  map<int, map<GlobalIndexType,double > > recipientDataMaps;
  map<GlobalIndexType, double > receivedMap;
  
  const int myPID = _mesh->Comm()->getRank();
  
  auto meshTopo = _mesh->getTopology();
  
  // want to send errors I know to parent's owner
  const auto &myCellIDs = _mesh->cellIDsInPartition();
  std::set<GlobalIndexType> myParentCellIDs;
  for (const auto & cellID : myCellIDs)
  {
    CellPtr cell   = meshTopo->getCell(cellID);
    CellPtr parent = cell->getParent();
    
    if (parent != Teuchos::null)
    {
      // check that all siblings are active (if not, parent is not eligible for unrefinement, and should be ignored)
      bool parentHasGrandchildren = false;
      bool parentHasExcludedChild = false;
      for (const auto & childCell : parent->children())
      {
        if (childCell->numChildren() > 0)
        {
          // parent has grandchildren. ==> cellID's error can/should be ignored; its parent is not eligible for refinement.
          parentHasGrandchildren = true;
          break;
        }
        if (excludedChildren.find(childCell->cellIndex()) != excludedChildren.end())
        {
          parentHasExcludedChild = true;
          break;
        }
      }
      if (parentHasGrandchildren || parentHasExcludedChild)
      {
        // go to next cell in myCellIDs
        continue;
      }
      
      const GlobalIndexType firstChildID = parent->children()[0]->cellIndex();
      const int ownerPID = _mesh->globalDofAssignment()->partitionForCellID( firstChildID );
      if (ownerPID != myPID)
      {
        auto localErrorEntry = _localErrorMeasures.find(cellID);
        TEUCHOS_TEST_FOR_EXCEPTION(localErrorEntry == _localErrorMeasures.end(), std::invalid_argument, "cellID not found.  Did you remember to call measureError()?");
        double childError = localErrorEntry->second;
        recipientDataMaps[ownerPID][cellID] = childError;
      }
      else
      {
        myParentCellIDs.insert(parent->cellIndex());
      }
    }
  }
  
  std::map<GlobalIndexType,double> receivedErrorValues;
  MPIWrapper::sendDataMaps(_mesh->Comm(), recipientDataMaps, receivedErrorValues);
    
  for (auto parentCellID : myParentCellIDs)
  {
    CellPtr parent = meshTopo->getCell(parentCellID);
    double parentError = 0.0;
    auto childIDs = parent->getChildIndices(meshTopo);
    for (auto childID : childIDs)
    {
      double childError = 0.0;
      if (myCellIDs.find(childID) != myCellIDs.end())
      {
        auto localErrorEntry = _localErrorMeasures.find(childID);
        TEUCHOS_TEST_FOR_EXCEPTION(localErrorEntry == _localErrorMeasures.end(), std::invalid_argument, "local error for childID " + std::to_string(childID) + " not found.  Did you remember to call measureError()?");
        childError = localErrorEntry->second;
      }
      else
      {
        auto remoteErrorEntry = receivedErrorValues.find(childID);
        TEUCHOS_TEST_FOR_EXCEPTION(remoteErrorEntry == _localErrorMeasures.end(), std::invalid_argument, "remote error for childID " + std::to_string(childID) + " not found.  Did you remember to call measureError()?");
        childError = remoteErrorEntry->second;
      }
      parentError += childError;
    }
    _parentErrorMeasures[parentCellID] = parentError;
  }
}

std::vector< std::pair<GlobalIndexType,double> > ErrorIndicator::globalMaximumErrorCells(const int cellCountTarget, const std::set<GlobalIndexType> &excludedCells)
{
  if (cellCountTarget == 0)
  {
    // can skip computing errors, etc.
    return std::vector< std::pair<GlobalIndexType,double> >();
  }
  
  auto comm = _mesh->Comm();
  
  std::map<GlobalIndexType,double> localErrorMeasures = _localErrorMeasures; // copy, then erase any entries from excludedCells
  for (const auto & excludedCellID : excludedCells)
  {
    localErrorMeasures.erase(excludedCellID);
  }
  
  const int maxGatherCount = std::max(10000,cellCountTarget*4);
  auto globalEntries = MPIGlobalMinMax::globalMaximums(comm, localErrorMeasures, cellCountTarget, maxGatherCount);
  
  const int myPID = comm->getRank();
  
  auto meshTopo = _mesh->getTopology();
  
  const auto &myCellIDs = _mesh->cellIDsInPartition();
  
  if (globalEntries.size() > cellCountTarget)
  {
    // globalEntries is sorted, with max values at end.  If there is a tie, we want to include the entries that tied.
    int firstEntryToInclude = globalEntries.size() - cellCountTarget;
    double firstEntryError = globalEntries[firstEntryToInclude].second;
    while ((globalEntries[firstEntryToInclude].second == firstEntryError) && (firstEntryToInclude >= 0))
    {
      firstEntryError--;
    }
    // delete the entries that precede firstEntryToInclude
    globalEntries.erase(globalEntries.begin(),globalEntries.begin()+firstEntryToInclude);
  }
  return globalEntries;
}

std::vector< std::pair<GlobalIndexType,double> > ErrorIndicator::globalMinimumErrorParentCells(const int cellReductionTarget, const std::set<GlobalIndexType> &excludedChildren)
{
  if (cellReductionTarget == 0)
  {
    // can skip computing errors, etc.
    return std::vector< std::pair<GlobalIndexType,double> >();
  }
  
  gatherParentError(excludedChildren);
  
  auto meshTopo = _mesh->getTopology();
  std::map<GlobalIndexType,double> parentErrorMeasures;
  for (const auto &entry : _parentErrorMeasures)
  {
    // check children:
    auto parentCell = meshTopo->getCell(entry.first);
    auto childIDs = parentCell->getChildIndices(meshTopo);
    bool hasExcludedChild = false;
    for (const auto & childID : childIDs)
    {
      if (excludedChildren.find(childID) != excludedChildren.end())
      {
        hasExcludedChild = true;
        break;
      }
    }
    if (!hasExcludedChild)
    {
      parentErrorMeasures[entry.first] = entry.second;
    }
  }
  
  auto comm = _mesh->Comm();
  
  const int maxGatherCount = std::max(100000,cellReductionTarget*4);
  auto globalEntries = MPIGlobalMinMax::globalMinimums(comm, parentErrorMeasures, cellReductionTarget, maxGatherCount);
  
  const int myPID = comm->getRank();
  
  const auto &myCellIDs = _mesh->cellIDsInPartition();
  std::set<GlobalIndexType> myParentCellIDs;
  for (const auto & cellID : myCellIDs)
  {
    CellPtr cell   = meshTopo->getCell(cellID);
    CellPtr parent = cell->getParent();
    
    if (parent != Teuchos::null)
    {
      const GlobalIndexType firstChildID = parent->children()[0]->cellIndex();
      const int ownerPID = _mesh->globalDofAssignment()->partitionForCellID( firstChildID );
      // NOTE: if first child has children, ownerPID will be -1.  But that also means that this parent is not eligible for refinement, so we can safely omit it from the list (effectively, this parent is then not owned by any PID).
      if (ownerPID == myPID)
      {
        myParentCellIDs.insert(parent->cellIndex());
      }
    }
  }
  
  // do a global sum to get the child counts for each entry
  std::vector<int> globalChildCount(globalEntries.size());
  int i=0;
  for (const auto &entry : globalEntries)
  {
    const auto & parentCellID = entry.first;
    if (myParentCellIDs.find(parentCellID) != myParentCellIDs.end())
    {
      globalChildCount[i] = meshTopo->getCell(parentCellID)->numChildren();
    }
    i++;
  }
  
  MPIWrapper::entryWiseSum(comm, globalChildCount);
  
  // now, go through global entries again, counting children until we pass the target
  int cellReduction = 0;
  int j=0;
  for (j=0; j<globalChildCount.size(); j++)
  {
    cellReduction += globalChildCount[j] - 1; // -1 because the parent cell will remain
    if (cellReduction >= cellReductionTarget)
    {
      // want to include the jth entry among globalEntries; so increment j…
      j++;
      break;
    }
  }
  // we want the first j entries; delete the rest
  globalEntries.erase(globalEntries.begin()+j, globalEntries.end());
  return globalEntries;
}

std::vector< std::pair<GlobalIndexType,double> > ErrorIndicator::globalParentCellsBelowThreshold(const double &maxError, const std::set<GlobalIndexType> &excludedChildren)
{
  gatherParentError(excludedChildren);
  
  auto meshTopo = _mesh->getTopology();
  using ParentErrorEntry = std::pair<GlobalIndexType,double>;
  std::vector<ParentErrorEntry> parentErrorMeasuresVector;
  for (const auto &entry : _parentErrorMeasures)
  {
    // check children:
    auto parentCell = meshTopo->getCell(entry.first);
    auto childIDs = parentCell->getChildIndices(meshTopo);
    bool hasExcludedChild = false;
    for (const auto & childID : childIDs)
    {
      if (excludedChildren.find(childID) != excludedChildren.end())
      {
        hasExcludedChild = true;
        break;
      }
    }
    if (!hasExcludedChild)
    {
      const double &error = entry.second;
      if (error <= maxError)
      {
        parentErrorMeasuresVector.push_back(entry);
      }
    }
  }
  
  auto comm = _mesh->Comm();
    
  std::vector<ParentErrorEntry> gatheredValues;
  std::vector<int> offsets;
  MPIWrapper::allGatherVariable<ParentErrorEntry>(comm, gatheredValues, parentErrorMeasuresVector, offsets);
  
  return gatheredValues;
}

double ErrorIndicator::totalError() const
{
  double localTotalSquared = 0;
  for (auto measureEntry : _localErrorMeasures) {
    double cellError = measureEntry.second;
    localTotalSquared += cellError * cellError;
  }
  double globalTotalSquared = MPIWrapper::sum(_mesh->Comm(), localTotalSquared);
  return sqrt(globalTotalSquared);
}

// instantiate these for Scalar = double
template ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator<double>(TSolutionPtr<double> solution);
template ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator<double>(MeshPtr mesh,TLinearTermPtr<double> residual, TIPPtr<double> ip, int quadratureEnrichmentDegree);
template ErrorIndicatorPtr ErrorIndicator::functionErrorIndicator<double>(MeshPtr mesh, FunctionPtr f, int quadratureEnrichment);
template ErrorIndicatorPtr ErrorIndicator::gradientErrorIndicator<double>(TSolutionPtr<double> solution, VarPtr scalarVar);
template ErrorIndicatorPtr ErrorIndicator::hessianErrorIndicator<double>(TSolutionPtr<double> solution, VarPtr scalarVar);
template ErrorIndicatorPtr ErrorIndicator::unrefinementProjectionErrorIndicator<double>(TSolutionPtr<double> solution, VarPtr scalarVar);

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator(TSolutionPtr<Scalar> solution)
{
  return Teuchos::rcp( new EnergyErrorIndicator<Scalar>(solution) );
}

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                       int quadratureEnrichmentDegree)
{
  return Teuchos::rcp( new EnergyErrorIndicator<Scalar>(mesh, residual, ip, quadratureEnrichmentDegree) );
}

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::functionErrorIndicator(MeshPtr mesh, FunctionPtr f, int quadratureEnrichment)
{
  return Teuchos::rcp( new FunctionErrorIndicator<Scalar>(mesh,f,quadratureEnrichment));
}

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::gradientErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar)
{
  return Teuchos::rcp( new GradientErrorIndicator<Scalar>(solution, scalarVar) );
}

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::hessianErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar)
{
  return Teuchos::rcp( new HessianErrorIndicator<Scalar>(solution, scalarVar) );
}

template <typename Scalar>
ErrorIndicatorPtr ErrorIndicator::unrefinementProjectionErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar)
{
  return Teuchos::rcp( new UnrefinementProjectionErrorIndicator(solution, scalarVar) );
}
