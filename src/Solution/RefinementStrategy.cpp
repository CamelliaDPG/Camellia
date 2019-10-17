//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "RefinementStrategy.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold,
                                                 double min_h, int max_p , bool preferPRefinements)
{
  _errorIndicator = errorIndicator;
  _relativeErrorThreshold = relativeErrorThreshold;
  _min_h = min_h;
  _max_p = max_p;
  _preferPRefinements = preferPRefinements;
  _enforceOneIrregularity = true;
}

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( TSolutionPtr<Scalar> solution,
                                                 double relativeEnergyThreshold, double min_h,
                                                 int max_p, bool preferPRefinements) :
TRefinementStrategy<Scalar>::TRefinementStrategy(ErrorIndicator::energyErrorIndicator(solution),
                                                 relativeEnergyThreshold, min_h, max_p, preferPRefinements)
{}

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                 double relativeEnergyThreshold, double min_h,
                                                 int max_p, bool preferPRefinements, int cubatureEnrichment) :
TRefinementStrategy<Scalar>::TRefinementStrategy(ErrorIndicator::energyErrorIndicator(mesh, residual, ip, cubatureEnrichment),
                                                 relativeEnergyThreshold, min_h, max_p, preferPRefinements)

{}

template <typename Scalar>
double TRefinementStrategy<Scalar>::computeTotalEnergyError()
{
  _errorIndicator->measureError();
  return _errorIndicator->totalError();
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::energyErrorRefinementStrategy(SolutionPtr soln, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(soln);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::energyErrorRefinementStrategy(MeshPtr mesh, TLinearTermPtr<Scalar> residual,
                                                                                 TIPPtr<Scalar> ip,
                                                                                 double relativeEnergyThreshold,
                                                                                 int cubatureEnrichmentDegree)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(mesh, residual, ip, cubatureEnrichmentDegree);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::gradientRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::gradientErrorIndicator(soln, scalarVar);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::hessianRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::hessianErrorIndicator(soln, scalarVar);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMinH(double value)
{
  _min_h = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setEnforceOneIrregularity(bool value)
{
  _enforceOneIrregularity = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setReportPerCellErrors(bool value)
{
  _reportPerCellErrors = value;
}

template <typename Scalar>
MeshPtr TRefinementStrategy<Scalar>::mesh()
{
  return _errorIndicator->mesh();
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();

  vector<GlobalIndexType> localCellsToRefine;
  getCellsAboveErrorThreshhold(localCellsToRefine); // calls _errorIndicator->measureError()
  double totalError = _errorIndicator->totalError();
  
  map<GlobalIndexType, double> cellMeasuresLocal;
  const set<GlobalIndexType>* myCellIDs = &mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : *myCellIDs)
  {
    cellMeasuresLocal[cellID] = mesh->getCellMeasure(cellID);
  }

//  if ( printToConsole && _reportPerCellErrors )
//  {
//    cout << "per-cell Energy Error Squared for cells with > 0.1% of squared energy error\n";
//    for (GlobalIndexType cellID : cellIDs)
//    {
//      double cellEnergyError = energyError.find(cellID)->second;
////      cout << "cellID " << cellID << " has energy error (not squared) " << cellEnergyError << endl;
//      double percent = (cellEnergyError*cellEnergyError) / (totalEnergyError*totalEnergyError) * 100;
//      if (percent > 0.1)
//      {
//        cout << cellID << ": " << cellEnergyError*cellEnergyError << " ( " << percent << " %)\n";
//      }
//    }
//  }

  // record results prior to refinement
  RefinementResults results = setResults(mesh->numActiveElements(), mesh->numGlobalDofs(), totalError);
  _results.push_back(results);
  
  vector<GlobalIndexType> myCellsToHRefine;
  vector<GlobalIndexType> myCellsToPRefine;
  
  double meshDim = mesh->getDimension();
  for (GlobalIndexType cellID : localCellsToRefine)
  {
    double h = pow(cellMeasuresLocal[cellID], 1.0 / meshDim); // dth root of volume measure = h
    int p = mesh->cellPolyOrder(cellID);

    if (!_preferPRefinements)
    {
      if (h > _min_h)
      {
        myCellsToHRefine.push_back(cellID);
      }
      else
      {
        myCellsToPRefine.push_back(cellID);
      }
    }
    else
    {
      if (p < _max_p)
      {
        myCellsToPRefine.push_back(cellID);
      }
      else
      {
        myCellsToHRefine.push_back(cellID);
      }
    }
  }

  GlobalIndexTypeToCast numCellsToHRefine = 0;
  GlobalIndexTypeToCast myNumCellsToHRefine = myCellsToHRefine.size();
  mesh->Comm()->SumAll(&myNumCellsToHRefine, &numCellsToHRefine, 1);
  GlobalIndexTypeToCast myCellOrdinalOffset = 0;
  mesh->Comm()->ScanSum(&myNumCellsToHRefine, &myCellOrdinalOffset, 1);
  myCellOrdinalOffset -= myNumCellsToHRefine;
  
  vector<GlobalIndexTypeToCast> globalCellsToRefineVector(numCellsToHRefine, 0);
  for (int i=0; i<myNumCellsToHRefine; i++)
  {
    globalCellsToRefineVector[myCellOrdinalOffset + i] = myCellsToHRefine[i];
  }
  vector<GlobalIndexTypeToCast> gatheredCellsToRefineVector(numCellsToHRefine, 0);
  mesh->Comm()->SumAll(&globalCellsToRefineVector[0], &gatheredCellsToRefineVector[0], numCellsToHRefine);

  vector<GlobalIndexType> cellsToRefine(gatheredCellsToRefineVector.begin(),gatheredCellsToRefineVector.end());;
  
  GlobalIndexTypeToCast numCellsToPRefine = 0;
  GlobalIndexTypeToCast myNumCellsToPRefine = myCellsToPRefine.size();
  mesh->Comm()->SumAll(&myNumCellsToPRefine, &numCellsToPRefine, 1);
  
  vector<GlobalIndexType> cellsToPRefine;
  if (numCellsToPRefine > 0)
  {
    myCellOrdinalOffset = 0;
    mesh->Comm()->ScanSum(&myNumCellsToPRefine, &myCellOrdinalOffset, 1);
    myCellOrdinalOffset -= myNumCellsToPRefine;

    vector<GlobalIndexTypeToCast> globalCellsToPRefineVector(numCellsToPRefine, 0);
    for (int i=0; i<myNumCellsToPRefine; i++)
    {
      globalCellsToPRefineVector[myCellOrdinalOffset + i] = myCellsToPRefine[i];
    }
    vector<GlobalIndexTypeToCast> gatheredCellsToPRefineVector(numCellsToPRefine, 0);
    mesh->Comm()->SumAll(&globalCellsToPRefineVector[0], &gatheredCellsToPRefineVector[0], numCellsToPRefine);

    cellsToPRefine = vector<GlobalIndexType>(gatheredCellsToPRefineVector.begin(),gatheredCellsToPRefineVector.end());
  }
  
  std::sort(cellsToRefine.begin(), cellsToRefine.end());
  std::sort(cellsToPRefine.begin(), cellsToPRefine.end());
  
  if (printToConsole)
  {
    if (cellsToRefine.size() > 0) Camellia::print("cells for h-refinement", cellsToRefine);
    if (cellsToPRefine.size() > 0) Camellia::print("cells for p-refinement", cellsToPRefine);
  }
  refineCells(cellsToRefine);
  pRefineCells(mesh, cellsToPRefine);

  bool repartitionAndRebuild = false;
  if (_enforceOneIrregularity)
  {
    mesh->enforceOneIrregularity(repartitionAndRebuild);
//    cout << "Enforced one irregularity.\n";
  }

  // now, repartition and rebuild:
  mesh->repartitionAndRebuild();
  
  GlobalIndexType activeElementCount = mesh->numActiveElements();
  GlobalIndexType globalDofCount = mesh->numGlobalDofs();
  
  if (printToConsole && (mesh->Comm()->MyPID() == 0))
  {
    cout << "Prior to refinement, total error: " << totalError << endl;
    cout << "After refinement, mesh has " << activeElementCount << " elements and " << globalDofCount << " global dofs" << endl;
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::getCellsAboveErrorThreshhold(vector<GlobalIndexType> &cellsToRefine)
{
  // measure error
  _errorIndicator->measureError();
  
  double maxError = _errorIndicator->maxError();
  
  double errorThreshold = maxError * _relativeErrorThreshold;
  _errorIndicator->localCellsAboveErrorThreshold(errorThreshold, cellsToRefine);

  
//  // greedy refinement algorithm - mark cells for refinement
//  MeshPtr mesh = this->mesh();
//  const map<GlobalIndexType,double>* rankLocalEnergy = &_solution->rankLocalEnergyError();
//  const set<GlobalIndexType>* myCellIDs = &mesh->cellIDsInPartition();
//
//  double localMaxError = 0.0;
//
//  for (GlobalIndexType cellID : *myCellIDs)
//  {
//    double cellEnergyError = rankLocalEnergy->find(cellID)->second;
//    localMaxError = max(cellEnergyError,localMaxError);
//  }
//  
//  double globalMaxError = 0.0;
//  mesh->Comm()->MaxAll(&localMaxError, &globalMaxError, 1);
//  
//  vector<GlobalIndexType> myCellsToRefine;
//  for (GlobalIndexType cellID : *myCellIDs)
//  {
//    double cellEnergyError = rankLocalEnergy->find(cellID)->second;
//    if ( cellEnergyError >= globalMaxError * _relativeEnergyThreshold )
//    {
//      myCellsToRefine.push_back(cellID);
//    }
//  }
//  
//  GlobalIndexTypeToCast numCellsToRefine = 0;
//  GlobalIndexTypeToCast myNumCellsToRefine = myCellsToRefine.size();
//  mesh->Comm()->SumAll(&myNumCellsToRefine, &numCellsToRefine, 1);
//  GlobalIndexTypeToCast myCellOrdinalOffset = 0;
//  mesh->Comm()->ScanSum(&myNumCellsToRefine, &myCellOrdinalOffset, 1);
//  myCellOrdinalOffset -= myNumCellsToRefine;
//  
//  vector<GlobalIndexTypeToCast> globalCellsToRefineVector(numCellsToRefine, 0);
//  for (int i=0; i<myNumCellsToRefine; i++)
//  {
//    globalCellsToRefineVector[myCellOrdinalOffset + i] = myCellsToRefine[i];
//  }
//  vector<GlobalIndexTypeToCast> gatheredCellsToRefineVector(numCellsToRefine, 0);
//  mesh->Comm()->SumAll(&globalCellsToRefineVector[0], &gatheredCellsToRefineVector[0], numCellsToRefine);
//  
//  cellsToRefine = vector<GlobalIndexType>(gatheredCellsToRefineVector.begin(), gatheredCellsToRefineVector.end());
}

// defaults to h-refinement
template <typename Scalar>
void TRefinementStrategy<Scalar>::refineCells(vector<GlobalIndexType> &cellIDs)
{
  MeshPtr mesh = this->mesh();
  hRefineCells(mesh, cellIDs);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::pRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  bool repartitionAndRebuild = false;
  int pToAdd = 1;
  set<GlobalIndexType> cellIDSet(cellIDs.begin(),cellIDs.end());
  mesh->pRefine(cellIDSet, pToAdd, repartitionAndRebuild);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  bool repartitionAndRebuild = false;
  mesh->hRefine(cellIDs, repartitionAndRebuild);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineUniformly()
{
  // record results prior to refinement
  _errorIndicator->measureError();
  double totalError = _errorIndicator->totalError();

  MeshPtr mesh = this->mesh();
  RefinementResults results = setResults(mesh->numElements(), mesh->numGlobalDofs(), totalError);
  _results.push_back(results);
  TRefinementStrategy<Scalar>::hRefineUniformly(mesh);
}

// ! static method
template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineUniformly(MeshPtr mesh)
{
  set<GlobalIndexType> cellsToRefine = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cellsToRefineVector(cellsToRefine.begin(),cellsToRefine.end());
  hRefineCells(mesh, cellsToRefineVector);
  mesh->repartitionAndRebuild();
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setRelativeErrorThreshold(double value)
{
  _relativeErrorThreshold = value;
}

template <typename Scalar>
RefinementResults TRefinementStrategy<Scalar>::setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError)
{
  RefinementResults solnResults;
  solnResults.numElements = numElements;
  solnResults.numDofs = numDofs;
  solnResults.totalEnergyError = totalEnergyError;
  return solnResults;
}

template <typename Scalar>
double TRefinementStrategy<Scalar>::getEnergyError(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].totalEnergyError;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumElements(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numElements;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumDofs(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numDofs;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

namespace Camellia
{
template class TRefinementStrategy<double>;
}
