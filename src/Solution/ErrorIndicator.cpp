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
