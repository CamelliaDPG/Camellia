//
//  MeshTransformationFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "TypeDefs.h"

#include <iostream>

#include "MeshTransformationFunction.h"
#include "ParametricCurve.h"
#include "Mesh.h"
#include "Element.h"
#include "BasisFactory.h"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "SerialDenseMatrixUtility.h"
#include "GnuPlotUtil.h"
#include "ParametricSurface.h"
#include "Projector.h"
#include "SpaceTimeBasisCache.h"
#include "TensorBasis.h"

#include "CamelliaCellTools.h"

#include "BasisCache.h"

using namespace Intrepid;
using namespace Camellia;

// TODO: move all the stuff to do with transfinite interpolation into ParametricSurface.cpp

VectorBasisPtr basisForTransformation(ElementTypePtr cellType)
{
  int polyOrder = std::max(cellType->trialOrderPtr->maxBasisDegree(), cellType->testOrderPtr->maxBasisDegree());

  CellTopoPtr cellTopo = cellType->cellTopoPtr;
  if (cellTopo->getTensorialDegree() > 0)
  {
    // for now, we assume that this means space-time.  (At some point, we may support tensor-product spatial cell topologies for
    // things like fast quadrature support, and this would need revisiting then.)
    // we also assume that the curvilinearity is purely spatial (and in fact, just 2D for now).
    cellTopo = CellTopology::cellTopology(cellTopo->getShardsTopology(), cellTopo->getTensorialDegree() - 1);
  }
  
  BasisPtr basis = BasisFactory::basisFactory()->getBasis(polyOrder, cellTopo, Camellia::FUNCTION_SPACE_VECTOR_HGRAD);
  VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<> *)basis.get(),false); // dynamic cast would be better
  return vectorBasis;
}

vector< ParametricCurvePtr > edgeLines(MeshPtr mesh, int cellID)
{
  vector< ParametricCurvePtr > lines;
  ElementPtr cell = mesh->getElement(cellID);
  vector<unsigned> vertexIndices = mesh->vertexIndicesForCell(cellID);
  for (int i=0; i<vertexIndices.size(); i++)
  {
    FieldContainer<double> v0 = mesh->vertexCoordinates(vertexIndices[i]);
    FieldContainer<double> v1 = mesh->vertexCoordinates(vertexIndices[(i+1)%vertexIndices.size()]);
    // 2D only for now
    TEUCHOS_TEST_FOR_EXCEPTION(v0.dimension(0) != 2, std::invalid_argument, "only 2D supported right now");
    lines.push_back(ParametricCurve::line(v0(0), v0(1), v1(0), v1(1)));
  }
  return lines;
}

void roundToOneOrZero(double &value, double tol)
{
  // if value is within tol of 1 or 0, replace value by 1 or 0
  if (abs(value-1)<tol)
  {
    value = 1;
  }
  else if (abs(value)<tol)
  {
    value = 0;
  }
}

class CellTransformationFunction : public TFunction<double>
{
  FieldContainer<double> _basisCoefficients;
  VectorBasisPtr _basis;
  Camellia::EOperator _op;
  int _cellIndex; // index into BasisCache's list of cellIDs; must be set prior to each call to values() (there's a reason why this is a private class!)

  FieldContainer<double> pointLatticeQuad(int numPointsTotal, const vector< ParametricCurvePtr > &edgeFunctions)
  {
    int spaceDim = 2;
    FieldContainer<double> pointLattice(numPointsTotal,spaceDim);

    ParametricSurfacePtr interpolant = ParametricSurface::transfiniteInterpolant(edgeFunctions);

    // arg to numPoints corresponds to "t1" ("x" direction t), the value to t2 ("y" direction t)
    vector< int > numPoints;
    int approxPoints1D = (int) sqrt( numPointsTotal );
    int remainder = numPointsTotal - approxPoints1D * approxPoints1D;
    for (int i=0; i<approxPoints1D; i++)
    {
      int extraPoint = (i < remainder) ? 1 : 0;
      numPoints.push_back(approxPoints1D + extraPoint);
    }

    if (edgeFunctions.size() != 4)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "wrong number of edgeFunctions");
    }

    int numPoints_t1 = numPoints.size();

    int pointIndex = 0;
    for (int t1_index = 0; t1_index < numPoints_t1; t1_index++)
    {
      int numPoints_t2 = numPoints[t1_index];
      double t1 = ((double)t1_index) / (double) (numPoints_t1 - 1);

      for (int t2_index=0; t2_index < numPoints_t2; t2_index++)
      {
        double t2 = ((double)t2_index) / (double) (numPoints_t2 - 1);

//        cout << "(t1,t2) = (" << t1 << ", " << t2 << ")" << endl;

        double x, y;
        interpolant->value(t1, t2, x, y);
        pointLattice(pointIndex,0) = x;
        pointLattice(pointIndex,1) = y;
        pointIndex++;
      }
    }
    return pointLattice;
  }

protected:
  CellTransformationFunction(VectorBasisPtr basis, FieldContainer<double> &basisCoefficients, Camellia::EOperator op) : TFunction<double>(1)
  {
    _basis = basis;
    _basisCoefficients = basisCoefficients;
    _op = op;
    _cellIndex = -1;
  }
public:
  CellTransformationFunction(MeshPtr mesh, int cellID, const vector< ParametricCurvePtr > &edgeFunctions) : TFunction<double>(1)
  {
    _cellIndex = -1;
    _op = OP_VALUE;
    ElementTypePtr elementType = mesh->getElementType(cellID);
    _basis = basisForTransformation(elementType);
    ParametricSurface::basisWeightsForProjectedInterpolant(_basisCoefficients, _basis, mesh, cellID);
  }

  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    // sets values(_cellIndex,P,D)
    TEUCHOS_TEST_FOR_EXCEPTION(_cellIndex == -1, std::invalid_argument, "must call setCellIndex before calling values!");

//    cout << "_basisCoefficients:\n" << _basisCoefficients;

    BasisCachePtr spaceTimeBasisCache;
    if (basisCache->cellTopologyIsSpaceTime())
    {
      // then we require that the basisCache provided be a space-time basis cache
      SpaceTimeBasisCache* spaceTimeCache = dynamic_cast<SpaceTimeBasisCache*>(basisCache.get());
      TEUCHOS_TEST_FOR_EXCEPTION(!spaceTimeCache, std::invalid_argument, "space-time requires a SpaceTimeBasisCache");
      spaceTimeBasisCache = basisCache;
      basisCache = spaceTimeCache->getSpatialBasisCache();
    }
    
    int numDofs = _basis->getCardinality();
    int spaceDim = basisCache->getSpaceDim();

    bool basisIsVolumeBasis = (spaceDim == _basis->domainTopology()->getDimension());
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
    
    int numPoints = values.dimension(1);

    // check if we're taking a temporal derivative
    int component;
    Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(_op, _basis->functionSpace(), component);
    if ((relatedOp == Intrepid::OPERATOR_GRAD) && (component==spaceDim)) {
      // then we are taking the temporal part of the Jacobian of the reference to curvilinear-reference space
      // based on our assumptions that curvilinearity is just in the spatial direction (and is orthogonally extruded in the
      // temporal direction), this is always the identity.
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        for (int d=0; d<values.dimension(2); d++)
        {
          if (d < spaceDim)
            values(_cellIndex,ptIndex,d) = 0.0;
          else
            values(_cellIndex,ptIndex,d) = 1.0;
        }
      }
      return;
    }
    constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);

    // transformedValues has dimensions (C,F,P,[D,D])
    // therefore, the rank of the sum is transformedValues->rank() - 3
    int rank = transformedValues->rank() - 3;
    TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");

    int spaceTimeSideOrdinal = (spaceTimeBasisCache != Teuchos::null) ? spaceTimeBasisCache->getSideIndex() : -1;
    
    // initialize the values we're responsible for setting
    if (_op == OP_VALUE)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        for (int d=0; d<values.dimension(2); d++)
        {
          if (d < spaceDim)
            values(_cellIndex,ptIndex,d) = 0.0;
          else if ((spaceTimeBasisCache != Teuchos::null) && (spaceTimeSideOrdinal == -1))
            values(_cellIndex,ptIndex,spaceDim) = spaceTimeBasisCache->getRefCellPoints()(ptIndex,spaceDim);
          else if ((spaceTimeBasisCache != Teuchos::null) && (spaceTimeSideOrdinal != -1))
          {
            if (spaceTimeBasisCache->cellTopology()->sideIsSpatial(spaceTimeSideOrdinal))
            {
              // TODO: check this -- pretty sure it's right, but need to check.
              values(_cellIndex,ptIndex,spaceDim) = spaceTimeBasisCache->getRefCellPoints()(ptIndex,spaceDim-1);
            }
            else
            {
              double temporalPoint;
              unsigned temporalNode = spaceTimeBasisCache->cellTopology()->getTemporalComponentSideOrdinal(spaceTimeSideOrdinal);
              if (temporalNode==0)
                temporalPoint = -1.0;
              else
                temporalPoint = 1.0;
              values(_cellIndex,ptIndex,spaceDim) = temporalPoint;
            }
          }
        }
      }
    }
    else if ((_op == OP_DX) || (_op == OP_DY) || (_op == OP_DZ))
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        for (int d=0; d<values.dimension(2); d++)
        {
          if (d < spaceDim)
            values(_cellIndex,ptIndex,d) = 0.0;
          else
            if (_op == OP_DZ)
              values(_cellIndex,ptIndex,d) = 1.0;
            else
              values(_cellIndex,ptIndex,d) = 0.0;
        }
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled _op");
    }

    int numSpatialPoints = transformedValues->dimension(2);
    int numTemporalPoints = numPoints / numSpatialPoints;
    TEUCHOS_TEST_FOR_EXCEPTION(numTemporalPoints * numSpatialPoints != numPoints, std::invalid_argument, "numPoints is not evenly divisible by numSpatialPoints");
    
    for (int i=0; i<numDofs; i++)
    {
      double weight = _basisCoefficients(i);
      for (int timePointOrdinal=0; timePointOrdinal<numTemporalPoints; timePointOrdinal++)
      {
        for (int spacePointOrdinal=0; spacePointOrdinal<numSpatialPoints; spacePointOrdinal++)
        {
          int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpatialPoints);
          for (int d=0; d<spaceDim; d++)
          {
            values(_cellIndex,spaceTimePointOrdinal,d) += weight * (*transformedValues)(_cellIndex,i,spacePointOrdinal,d);
          }
        }
      }
    }
  }

  int basisDegree()
  {
    return _basis->getDegree();
  }

  void setCellIndex(int cellIndex)
  {
    _cellIndex = cellIndex;
  }

  TFunctionPtr<double> dx()
  {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DX) );
  }

  TFunctionPtr<double> dy()
  {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DY) );
  }

  TFunctionPtr<double> dz()
  {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DZ) );
  }

  static Teuchos::RCP<CellTransformationFunction> cellTransformation(MeshPtr mesh, GlobalIndexType cellID, const vector< ParametricCurvePtr > edgeFunctions)
  {
    return Teuchos::rcp( new CellTransformationFunction(mesh,cellID,edgeFunctions));
  }
};

// protected method; used for dx(), dy(), dz():
MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, map< GlobalIndexType, TFunctionPtr<double>> cellTransforms, Camellia::EOperator op) : TFunction<double>(1)
{
  _mesh = mesh;
  _cellTransforms = cellTransforms;
  _op = op;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)

}

MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, set<GlobalIndexType> cellIDsToTransform) : TFunction<double>(1)   // vector-valued Function
{
  _op = OP_VALUE;
  _mesh = mesh;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)
  this->updateCells(cellIDsToTransform);
}

int MeshTransformationFunction::maxDegree()
{
  return _maxPolynomialDegree;
}

bool MeshTransformationFunction::mapRefCellPointsUsingExactGeometry(FieldContainer<double> &cellPoints, const FieldContainer<double> &refCellPoints, GlobalIndexType cellID)
{
  // returns true if the MeshTransformationFunction handles this cell, false otherwise
  // if true, then populates cellPoints
  if (_cellTransforms.find(cellID) == _cellTransforms.end())
  {
    return false;
  }
//  cout << "refCellPoints in mapRefCellPointsUsingExactGeometry():\n" << refCellPoints;
//
//  cout << "cellPoints prior to mapRefCellPointsUsingExactGeometry():\n" << cellPoints;

  int numPoints = refCellPoints.dimension(0);
  int spaceDim = refCellPoints.dimension(1);

  CellTopoPtr cellTopo = _mesh->getElementType(cellID)->cellTopoPtr;
  bool spaceTime = false;
  CellTopoPtr spaceTimeTopo;
  FieldContainer<double> refCellPointsSpaceTime, refCellPointsSpace;
  if (cellTopo->getTensorialDegree() > 0)
  {
    spaceDim = spaceDim - 1;
    spaceTime = true;
    spaceTimeTopo = cellTopo;
    cellTopo = CellTopology::cellTopology(cellTopo->getShardsTopology(), cellTopo->getTensorialDegree() - 1);
    refCellPointsSpaceTime = refCellPoints;
    // copy out the spatial points:
    refCellPointsSpace.resize(numPoints, spaceDim);
    for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        refCellPointsSpace(ptOrdinal,d) = refCellPointsSpaceTime(ptOrdinal,d);
      }
    }
  }
  else
  {
    refCellPointsSpace = refCellPoints;  // would be possible to avoid this copy by e.g. using a pointer in the call to mapToPhysicalFrame() below
  }
  if (cellTopo->getKey().first == shards::Quadrilateral<4>::key)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "points must be in 2D for the quad!");
    FieldContainer<double> parametricPoints(numPoints,spaceDim); // map to (t1,t2) space
    int whichCell = 0;
    CamelliaCellTools::mapToPhysicalFrame(parametricPoints,refCellPointsSpace,
                                          ParametricSurface::parametricQuadNodes(),
                                          cellTopo,whichCell);

//    cout << "parametricPoints in mapRefCellPointsUsingExactGeometry():\n" << parametricPoints;

    vector< ParametricCurvePtr > edgeFunctions = _mesh->parametricEdgesForCell(cellID);

    ParametricSurfacePtr interpolant = ParametricSurface::transfiniteInterpolant(edgeFunctions);

    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      double t1 = parametricPoints(ptIndex,0);
      double t2 = parametricPoints(ptIndex,1);

      double x,y;
      // transfinite interpolation:
      interpolant->value(t1, t2, x, y);

      cellPoints(ptIndex,0) = x;
      cellPoints(ptIndex,1) = y;
      if (spaceTime)
      {
        // per our assumptions on mesh transformations, we don't change the temporal components here:
        cellPoints(ptIndex,2) = refCellPoints(ptIndex,2);
      }
    }
  }
  else
  {
    // TODO: work out what to do for triangles (or perhaps even a general polygon)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled cell type");
  }
  
//  cout << "cellPoints after to mapRefCellPointsUsingExactGeometry():\n" << cellPoints;

  return true;
}

void MeshTransformationFunction::updateCells(const set<GlobalIndexType> &cellIDs)
{
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    Teuchos::RCP<CellTransformationFunction> cellTransform =  CellTransformationFunction::cellTransformation(_mesh, cellID, _mesh->parametricEdgesForCell(cellID));
    _cellTransforms[cellID] = cellTransform;
    _maxPolynomialDegree = std::max(_maxPolynomialDegree,cellTransform->basisDegree());
  }
}

void MeshTransformationFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  CHECK_VALUES_RANK(values);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  // identity map is the right thing most of the time
  // we'll do something different only where necessary
  int spaceDim = values.dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getDimension() != spaceDim, std::invalid_argument, "cellTopology dimension does not match the shape of the values container");
  if (_op == OP_VALUE)
  {
    values = basisCache->getPhysicalCubaturePoints(); // identity
  }
  else if (_op == OP_DX)
  {
    // identity map is 1 in all the x slots, 0 in all others
    int mod_value = 0; // the x slots are the mod spaceDim = 0 slots;
    for (int i=0; i<values.size(); i++)
    {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  }
  else if (_op == OP_DY)
  {
    // identity map is 1 in all the y slots, 0 in all others
    int mod_value = 1; // the y slots are the mod spaceDim = 1 slots;
    for (int i=0; i<values.size(); i++)
    {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  }
  else if (_op == OP_DZ)
  {
    // identity map is 1 in all the z slots, 0 in all others
    int mod_value = 2; // the z slots are the mod spaceDim = 2 slots;
    for (int i=0; i<values.size(); i++)
    {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  }
//  if (_op == OP_DX) {
//    cout << "values before cellTransformation:\n" << values;
//  }
  for (int cellIndex=0; cellIndex < cellIDs.size(); cellIndex++)
  {
    GlobalIndexType cellID = cellIDs[cellIndex];
    if (_cellTransforms.find(cellID) == _cellTransforms.end()) continue;
    TFunctionPtr<double> cellTransformation = _cellTransforms[cellID];
    ((CellTransformationFunction*)cellTransformation.get())->setCellIndex(cellIndex);
    cellTransformation->values(values, basisCache);
  }
//  if (_op == OP_DX) {
//    cout << "values after cellTransformation:\n" << values;
//  }
}

map< GlobalIndexType, TFunctionPtr<double> > applyOperatorToCellTransforms(const map< GlobalIndexType, TFunctionPtr<double> > &cellTransforms, Camellia::EOperator op)
{
  map<GlobalIndexType, TFunctionPtr<double> > newTransforms;
  for (map< GlobalIndexType, TFunctionPtr<double> >::const_iterator cellTransformIt = cellTransforms.begin();
       cellTransformIt != cellTransforms.end(); cellTransformIt++)
  {
    GlobalIndexType cellID = cellTransformIt->first;
    newTransforms[cellID] = TFunction<double>::op(cellTransformIt->second, op);
  }
  return newTransforms;
}

TFunctionPtr<double> MeshTransformationFunction::dx()
{
  Camellia::EOperator op = OP_DX;
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}

TFunctionPtr<double> MeshTransformationFunction::dy()
{
  if (_mesh->getDimension() < 2)
  {
    return TFunction<double>::null();
  }
  Camellia::EOperator op = OP_DY;
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}

TFunctionPtr<double> MeshTransformationFunction::dz()
{
  if (_mesh->getDimension() < 3)
  {
    return TFunction<double>::null();
  }
  Camellia::EOperator op = OP_DZ;
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}

void MeshTransformationFunction::didHRefine(const set<GlobalIndexType> &cellIDs)
{
  set<GlobalIndexType> childrenWithCurvedEdges;

  MeshTopology* topology = dynamic_cast<MeshTopology*>(_mesh->getTopology().get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!topology, std::invalid_argument, "Mesh::hRefine() called when _meshTopology is not an instance of MeshTopology--likely Mesh initialized with a pure MeshTopologyView, which cannot be h-refined.");

  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType parentCellID = *cellIDIt;
    vector<IndexType> childCells = topology->getCell(parentCellID)->getChildIndices();
    for (vector<IndexType>::iterator childCellIt = childCells.begin(); childCellIt != childCells.end(); childCellIt++)
    {
      unsigned childCellID = *childCellIt;
      if (topology->cellHasCurvedEdges(childCellID))
      {
        childrenWithCurvedEdges.insert(childCellID);
      }
    }
  }
  updateCells(childrenWithCurvedEdges);
}

void MeshTransformationFunction::didPRefine(const set<GlobalIndexType> &cellIDs)
{
  updateCells(cellIDs);
}

MeshTransformationFunction::~MeshTransformationFunction() {}
