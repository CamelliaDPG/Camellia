//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  GradientErrorIndicator.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#include "GradientErrorIndicator.h"

#include "CamelliaCellTools.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;

template <typename Scalar>
GradientErrorIndicator<Scalar>::GradientErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForGradient) : ErrorIndicator(soln->mesh())
{
  _solution = soln;
  _var = varForGradient;
  
  int spaceDim = _solution->mesh()->getDimension();
  _hPower = 1.0 + spaceDim / 2.0;
}

template <typename Scalar>
GradientErrorIndicator<Scalar>::GradientErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForGradient, double hPower) : ErrorIndicator(soln->mesh())
{
  _solution = soln;
  _var = varForGradient;
  _hPower = hPower;
}

//! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
template <typename Scalar>
void GradientErrorIndicator<Scalar>::measureError()
{
  // clear previous measures
  this->_localErrorMeasures.clear();
  
  // imitates https://dealii.org/developer/doxygen/deal.II/namespaceDerivativeApproximation.html
  
  // we require that u be a scalar field variable
  TEUCHOS_TEST_FOR_EXCEPTION(_var->rank() != 0, std::invalid_argument, "varForGradient must be a scalar variable");
  TEUCHOS_TEST_FOR_EXCEPTION(_var->varType() != FIELD, std::invalid_argument, "varForGradient must be a field variable");

  auto myCells = &_mesh->cellIDsInPartition();
  
  int onePoint = 1;
  MeshTopologyViewPtr meshTopo = _solution->mesh()->getTopology();
  int spaceDim = meshTopo->getDimension();
  
  set<GlobalIndexType> cellsAndNeighborsSet;
  for (GlobalIndexType myCellID : *myCells)
  {
    cellsAndNeighborsSet.insert(myCellID);
    CellPtr cell = meshTopo->getCell(myCellID);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    cellsAndNeighborsSet.insert(neighborIDs.begin(),neighborIDs.end());
  }
  vector<GlobalIndexType> cellsAndNeighbors(cellsAndNeighborsSet.begin(),cellsAndNeighborsSet.end());

  // get any off-rank solution data we may need:
  _solution->importSolutionForOffRankCells(cellsAndNeighborsSet);
  
  int cellsAndNeighborsCount = cellsAndNeighbors.size();
  
  Camellia::FieldContainer<double> cellValues(cellsAndNeighborsCount,onePoint); // values at cell centers
  Camellia::FieldContainer<double> cellCenters(cellsAndNeighborsCount,spaceDim);
  Camellia::FieldContainer<double> cellDiameter(cellsAndNeighborsCount,onePoint); // h-values
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForTopology;
  
  FunctionPtr hFunction = Function::h();
  FunctionPtr solnFunction = Function::solution(_var, _solution);
  Teuchos::Array<int> cellValueDim;
  cellValueDim.push_back(1);
  cellValueDim.push_back(1);
  
  map<GlobalIndexType,int> cellIDToOrdinal; // lookup table for value access
  
  // setup: compute cell centers, and solution values at those points
  for (int cellOrdinal=0; cellOrdinal<cellsAndNeighborsCount; cellOrdinal++)
  {
    GlobalIndexType cellID = cellsAndNeighbors[cellOrdinal];
    cellIDToOrdinal[cellID] = cellOrdinal;
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    if (basisCacheForTopology.find(cellTopo->getKey()) == basisCacheForTopology.end())
    {
      Camellia::FieldContainer<double> centroid(onePoint,spaceDim);
      int nodeCount = cellTopo->getNodeCount();
      Camellia::FieldContainer<double> cellNodes(nodeCount,spaceDim);
      CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
      for (int node=0; node<nodeCount; node++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          centroid(0,d) += cellNodes(node,d);
        }
      }
      for (int d=0; d<spaceDim; d++)
      {
        centroid(0,d) /= nodeCount;
      }
      basisCacheForTopology[cellTopo->getKey()] = BasisCache::basisCacheForReferenceCell(cellTopo, 0); // 0 cubature degree
      basisCacheForTopology[cellTopo->getKey()]->setRefCellPointsFC(centroid);
      basisCacheForTopology[cellTopo->getKey()]->setMesh(_solution->mesh());
    }
    BasisCachePtr basisCache = basisCacheForTopology[cellTopo->getKey()];
    basisCache->setPhysicalCellNodes(_solution->mesh()->physicalCellNodesForCell(cellID), {cellID}, false);
    
    Camellia::FieldContainer<double> cellValue(cellValueDim,&cellValues(cellOrdinal,0));
    solnFunction->values(cellValue, basisCache);
    for (int d=0; d<spaceDim; d++)
    {
      cellCenters(cellOrdinal,d) = basisCache->getPhysicalCubaturePointsFC()(0,0,d);
    }
    if (_hPower != 0)
    {
      cellDiameter(cellOrdinal,0) = _solution->mesh()->getCellMeasure(cellID);
    }
  }
  
  // now compute the gradients requested
  Camellia::FieldContainer<double> Y(spaceDim,spaceDim); // the matrix we'll invert to compute the gradient
  Camellia::FieldContainer<double> b(spaceDim); // RHS for matrix problem
  Camellia::FieldContainer<double> grad(spaceDim); // LHS for matrix problem
  vector<double> distanceVector(spaceDim);
  for (GlobalIndexType myCellID : *myCells)
  {
    Y.initialize(0.0);
    b.initialize(0.0);
    CellPtr cell = meshTopo->getCell(myCellID);
    int myOrdinalInCellAndNeighbors = cellIDToOrdinal[myCellID];
    double myValue = cellValues(myOrdinalInCellAndNeighbors,0);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      int neighborOrdinalInCellAndNeighbors = cellIDToOrdinal[neighborID];
      double neighborValue = cellValues(neighborOrdinalInCellAndNeighbors,0);
      
      double dist_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        distanceVector[d] = cellCenters(neighborOrdinalInCellAndNeighbors,d) - cellCenters(myOrdinalInCellAndNeighbors,d);
        dist_squared += distanceVector[d] * distanceVector[d];
      }
      
      for (int d1=0; d1<spaceDim; d1++)
      {
        b(d1) += distanceVector[d1] * (neighborValue - myValue) / dist_squared;
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += distanceVector[d1] * distanceVector[d2] / dist_squared;
        }
      }
    }
    SerialDenseWrapper::solveSystem(grad, Y, b);
    double l2_value_squared = 0;
    for (int d=0; d<spaceDim; d++)
    {
      l2_value_squared += grad(d) * grad(d);
    }
    if (_hPower == 0)
    {
      _localErrorMeasures[myCellID] = sqrt(l2_value_squared);
    }
    else
    {
      _localErrorMeasures[myCellID] = sqrt(l2_value_squared) * pow(cellDiameter(myOrdinalInCellAndNeighbors,0), _hPower);
    }
  }
}

// ETI for double 
template class Camellia::GradientErrorIndicator<double>;
