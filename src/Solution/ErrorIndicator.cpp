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

double ErrorIndicator::maxError() const
{
  double localMax = maxLocalError();
  double globalMax;
  _mesh->Comm()->MaxAll(&localMax, &globalMax, 1);
  return globalMax;
}

double ErrorIndicator::maxLocalError() const
{
  double maxErr = 0;
  for (auto measureEntry : _localErrorMeasures) {
    double cellError = measureEntry.second;
    maxErr = max(maxErr,cellError);
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
  
  double globalMinError = localMinError;
  _mesh->Comm()->MinAll(&localMinError, &globalMinError, 1);
  return globalMinError;
}

double ErrorIndicator::minParentError() const
{
  double localMinError  = std::numeric_limits<double>::max();
  for (const auto & errEntry : _parentErrorMeasures)
  {
    localMinError = std::fmin(localMinError, errEntry.second);
  }
  
  double globalMinError = localMinError;
  _mesh->Comm()->MinAll(&localMinError, &globalMinError, 1);
  return globalMinError;
}

void ErrorIndicator::gatherParentError()
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
  
  const int myPID = _mesh->Comm()->MyPID();
  
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
      for (const auto & childCell : parent->children())
      {
        if (childCell->numChildren() > 0)
        {
          // parent has grandchildren. ==> cellID's error can/should be ignored; its parent is not eligible for refinement.
          parentHasGrandchildren = true;
          break;
        }
      }
      if (parentHasGrandchildren)
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

double ErrorIndicator::totalError() const
{
  double localTotalSquared = 0;
  for (auto measureEntry : _localErrorMeasures) {
    double cellError = measureEntry.second;
    localTotalSquared += cellError * cellError;
  }
  double globalTotalSquared;
  _mesh->Comm()->SumAll(&localTotalSquared, &globalTotalSquared, 1);
  return sqrt(globalTotalSquared);
}

// instantiate these for Scalar = double
template ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator<double>(TSolutionPtr<double> solution);
template ErrorIndicatorPtr ErrorIndicator::energyErrorIndicator<double>(MeshPtr mesh,TLinearTermPtr<double> residual, TIPPtr<double> ip, int quadratureEnrichmentDegree);
template ErrorIndicatorPtr ErrorIndicator::functionErrorIndicator<double>(MeshPtr mesh, FunctionPtr f, int quadratureEnrichment);
template ErrorIndicatorPtr ErrorIndicator::gradientErrorIndicator<double>(TSolutionPtr<double> solution, VarPtr scalarVar);
template ErrorIndicatorPtr ErrorIndicator::hessianErrorIndicator<double>(TSolutionPtr<double> solution, VarPtr scalarVar);

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
