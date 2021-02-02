//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "QuadraturePointFunction.h"

using namespace Camellia;

QuadraturePointFunction::QuadraturePointFunction(MeshPtr mesh, int quadratureEnrichment, int rank, bool boundaryValueOnly)
:
TFunction<double>("QuadraturePointFunction",rank)
{
  _mesh = mesh;
  _quadratureEnrichment = quadratureEnrichment;
  _boundaryValueOnly = boundaryValueOnly;
}

bool QuadraturePointFunction::boundaryValueOnly()
{
  return _boundaryValueOnly;
}

void QuadraturePointFunction::updateValues()
{
  // willUpdateValues() may do MPI communication, so it must be called unconditionally
  willUpdateValues();
  
  // for now, we are writing this with the assumption that all elements have the same ElementType
  // let's check that that's true on the elements that we see locally:
  auto & cellIDs = _mesh->cellIDsInPartition();
  ElementTypePtr elemType = Teuchos::null;
  for (auto cellID : cellIDs)
  {
    if (elemType == Teuchos::null)
    {
      // first cell
      elemType = _mesh->getElementType(cellID);
    }
    auto thisElemType = _mesh->getElementType(cellID);
    TEUCHOS_TEST_FOR_EXCEPTION(!thisElemType->equals(*elemType), std::invalid_argument, "Right now, QuadraturePointFunction only supports meshes with the same ElementType across all cells.");
  }
  
  std::vector<GlobalIndexType> cellIDsVector(cellIDs.begin(),cellIDs.end());
  _cellIDs = cellIDsVector;
  
  if (elemType != Teuchos::null)
  {
    // if we get here, then elemType is filled with the element type common to all the elements we see
    bool testVsTest = false; // NOTE that if we want to support using this for test inner product weights, we should add some API that sets this to true
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(_mesh, elemType, testVsTest, _quadratureEnrichment);
    
    
    basisCache->setCellIDs(cellIDsVector);
    
    const int numCells = cellIDs.size();
    const int fxnRank  = this->rank();
    const int spaceDim = basisCache->cellTopology()->getDimension();
    
    if (!_boundaryValueOnly)
    {
      auto numPoints = basisCache->getRefCellPointsFC().dimension(0);
      int numEntriesPerCell = numPoints;
      for (int d=0; d<fxnRank; d++)
      {
        numEntriesPerCell *= spaceDim;
      }
      
      if (fxnRank == 0)
      {
        _valuesForActiveCells = FieldContainer<double>(numCells,numPoints);
      }
      else if (fxnRank == 1)
      {
        _valuesForActiveCells = FieldContainer<double>(numCells,numPoints,spaceDim);
      }
      else if (fxnRank == 2)
      {
        _valuesForActiveCells = FieldContainer<double>(numCells,numPoints,spaceDim,spaceDim);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function rank");
      }
      
      this->updateValues(_valuesForActiveCells, basisCache);
    }
    else // _boundaryValueOnly = true
    {
      const int sideCount = elemType->cellTopoPtr->getSideCount();
      _sideValuesForActiveCells = std::vector< FieldContainer<double> >(sideCount);
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
        const int numPoints = sideBasisCache->getRefCellPointsFC().dimension(0);
        int numEntriesPerCell = numPoints;
        for (int d=0; d<fxnRank; d++)
        {
          numEntriesPerCell *= spaceDim;
        }
        
        if (fxnRank == 0)
        {
          _sideValuesForActiveCells[sideOrdinal] = FieldContainer<double>(numCells,numPoints);
        }
        else if (fxnRank == 1)
        {
          _sideValuesForActiveCells[sideOrdinal] = FieldContainer<double>(numCells,numPoints,spaceDim);
        }
        else if (fxnRank == 2)
        {
          _sideValuesForActiveCells[sideOrdinal] = FieldContainer<double>(numCells,numPoints,spaceDim,spaceDim);
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function rank");
        }
        this->updateValues(_sideValuesForActiveCells[sideOrdinal], sideBasisCache);
      }
    }
  }
}

void QuadraturePointFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  // 1. Check whether the _cellIDs are exactly what's in basisCache.  If so, just do a memcpy (deep_copy once we have Kokkos)
  // 2. If some cells in basisCache are not in _cellIDs, a choice: throw exception, or call updateValues().
  // 3. If basisCache cellIDs are all present in _cellIDs but not all _cellIDs are in basisCache, then do per-cell copying of values.  Could be smart about contiguous cells, but this cost seems unlikely to be a bottleneck…
  
  const int sideOrdinal = basisCache->getSideIndex();
  FieldContainer<double> & storedValuesContainer = (sideOrdinal == -1) ? _valuesForActiveCells : _sideValuesForActiveCells[sideOrdinal];
  
  if ((storedValuesContainer.size() == 0) && (values.size() > 0))
  {
    std::cout << "ERROR in QuadraturePointFunction: stored values are empty; maybe you forgot to call updateValues()?\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "stored values not initialized");
  }
  
  const int numPoints = values.dimension(1);
  TEUCHOS_TEST_FOR_EXCEPTION(numPoints != storedValuesContainer.dimension(1), std::invalid_argument, "Inconsistent point count.");
  
  const int fxnRank  = this->rank();
  auto spaceDim = basisCache->getRefCellPointsFC().dimension(1);
  int numEntriesPerCell = numPoints;
  for (int d=0; d<fxnRank; d++)
  {
    numEntriesPerCell *= spaceDim;
  }
  
  auto & cellIDsToCompute = basisCache->cellIDs();
  if (cellIDsToCompute == _cellIDs)
  {
    memcpy(&values[0],&storedValuesContainer[0],sizeof(double)*storedValuesContainer.size());
  }
  else
  {
    std::vector<int> cellOrdinals(cellIDsToCompute.size(),-1);
    int searchStartIndex = 0;
    int cellToComputeOrdinal = 0;
    for (auto cellID : cellIDsToCompute)
    {
      // looping for loop starts at last match (both vectors are likely to be in numeric order...)
      for (int i_loop=searchStartIndex; i_loop<_cellIDs.size()+searchStartIndex; i_loop++)
      {
        const int i = i_loop % _cellIDs.size();
        if (_cellIDs[i] == cellID)
        {
          cellOrdinals[cellToComputeOrdinal] = i;
          searchStartIndex = i;
          break;
        }
      }
      if (cellOrdinals[cellToComputeOrdinal] == -1)
      {
        std::cout << "Cell " << cellID << " not found.  Maybe a call to updateValues() is missing?\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
      }
      cellToComputeOrdinal++;
    }
    for (int i=0; i<cellOrdinals.size();)
    {
      // copy all contiguous entries starting with i
      int endIndex = i+1;
      while ((endIndex < cellOrdinals.size()) && (cellOrdinals[endIndex-1]+1 == cellOrdinals[endIndex]))
      {
        endIndex++;
      }
      const int valuesOffset               = i*numEntriesPerCell;
      const int valuesForActiveCellsOffset = cellOrdinals[i]*numEntriesPerCell;
      const int numContiguousCells         = endIndex - i;
      memcpy(&values[valuesOffset],&storedValuesContainer[valuesForActiveCellsOffset],sizeof(double)*numEntriesPerCell*numContiguousCells);
      i = endIndex;
    }
  }
}

void QuadraturePointFunction::willUpdateValues()
{}

QuadraturePointFunction::~QuadraturePointFunction()
{}

namespace Camellia
{
  class PrecomputedFunction : public QuadraturePointFunction
  {
    FunctionPtr _f;
  public:
    PrecomputedFunction(MeshPtr mesh, int quadratureEnrichment, FunctionPtr f, bool setValuesNow)
    :
    QuadraturePointFunction(mesh, quadratureEnrichment, f->rank(), f->boundaryValueOnly()),
    _f(f)
    {
      if (setValuesNow)
      {
        this->QuadraturePointFunction::updateValues();
      }
    }
    
    void updateValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      _f->values(values, basisCache);
    }
  };
}

Teuchos::RCP<QuadraturePointFunction> QuadraturePointFunction::precomputedFunction(MeshPtr mesh, int quadratureEnrichment, FunctionPtr f, bool setValuesNow)
{
  return Teuchos::rcp(new PrecomputedFunction(mesh, quadratureEnrichment, f, setValuesNow));
}
