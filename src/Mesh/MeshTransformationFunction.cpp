//
//  MeshTransformationFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "MeshTransformationFunction.h"
#include "ParametricFunction.h"

#include "Mesh.h"
#include "Element.h"
#include "BasisFactory.h"

#include "SerialDenseSolveWrapper.h"

BasisPtr basisForTransformation(ElementTypePtr cellType) {
  unsigned int cellTopoKey = cellType->cellTopoPtr->getKey();
  
  int polyOrder = max(cellType->trialOrderPtr->maxBasisDegree(), cellType->testOrderPtr->maxBasisDegree());
  
  return BasisFactory::getBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
}

vector< ParametricFunctionPtr > edgeLines(MeshPtr mesh, int cellID) {
  vector< ParametricFunctionPtr > lines;
  ElementPtr cell = mesh->getElement(cellID);
  vector<int> vertexIndices = mesh->vertexIndicesForCell(cellID);
  for (int i=0; i<vertexIndices.size(); i++) {
    FieldContainer<double> v0 = mesh->vertexCoordinates(vertexIndices[i]);
    FieldContainer<double> v1 = mesh->vertexCoordinates(vertexIndices[(i+1)%vertexIndices.size()]);
    // 2D only for now
    TEUCHOS_TEST_FOR_EXCEPTION(v0.dimension(0) != 2, std::invalid_argument, "only 2D supported right now");
    lines.push_back(ParametricFunction::line(v0(0), v0(1), v1(0), v1(1)));
  }
  return lines;
}

class CellTransformationFunction : public Function {
  FieldContainer<double> _basisCoefficients;
  BasisPtr _basis;
  EOperatorExtended _op;
  int _cellIndex; // index into BasisCache's list of cellIDs; must be set prior to each call to values() (there's a reason why this is a private class!)
protected:
  CellTransformationFunction(BasisPtr basis, FieldContainer<double> &basisCoefficients, EOperatorExtended op) {
    _basis = basis;
    _basisCoefficients = basisCoefficients;
    _op = op;
    _cellIndex = -1;
  }
public:
  CellTransformationFunction(MeshPtr mesh, int cellID, const vector< ParametricFunctionPtr > edgeFunctions) : Function(1) {
    _cellIndex = -1;
    _op = OP_VALUE;
    ElementPtr cell = mesh->getElement(cellID);
    int numEdges = edgeFunctions.size();
    _basis = basisForTransformation(cell->elementType());
    int cardinality = _basis->getCardinality();
    _basisCoefficients.resize(cardinality);
    int spaceDim = cell->elementType()->cellTopoPtr->getDimension();
    int numPoints = cardinality / spaceDim;
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "only 2D supported right now");
    FieldContainer<double> points(numPoints,spaceDim);
    FieldContainer<double> transformedPoints(numPoints,spaceDim);
    vector< ParametricFunctionPtr > straightEdges = edgeLines(mesh, cellID);
    int pointsForEdge[edgeFunctions.size()]; // unique points for the edge (we'll actually have one more than this, shared with its neighbor)
    int numPointsSet = 0;
    for (int i=0; i<numEdges; i++) {
      pointsForEdge[i] = numPoints / numEdges;
      numPointsSet += pointsForEdge[i];
    }
    int remainder = numPoints - numPointsSet;
    for (int i=0; i<remainder; i++) {
      pointsForEdge[i] += 1;
    }
    int pointIndex = 0;
    for (int i=0; i<numEdges; i++) {
      for (int j=0; j<pointsForEdge[i]; j++) {
        double t = ((double)j) / pointsForEdge[i];
        double x, y;
        straightEdges[i]->value(t, x, y);
        points(pointIndex,0) = x;
        points(pointIndex,1) = y;
        edgeFunctions[i]->value(t, x, y);
        transformedPoints(pointIndex,0) = x;
        transformedPoints(pointIndex,1) = y;
        pointIndex++;
      }
    }
    // now, since _basis is defined on refCell, compute preimage of points
    FieldContainer<double> refCellPoints(numPoints,spaceDim);
    CellTools<double>::mapToReferenceFrame(refCellPoints,points,mesh->physicalCellNodesForCell(cellID),
                                           *(cell->elementType()->cellTopoPtr),0);
    FieldContainer<double> basisValues(cardinality,numPoints,spaceDim);  // (F,P,D)
    _basis->getValues(basisValues, refCellPoints, Intrepid::OPERATOR_VALUE);
    // reshape to make a square matrix:
    basisValues.resize(cardinality, numPoints*spaceDim);
    // reshape transformedPoints to be a vector:
    transformedPoints.resize(numPoints*spaceDim);
    // rows should be the points, so we want to use the transpose of this
    SerialDenseSolveWrapper::solveSystem(_basisCoefficients, basisValues, transformedPoints);
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // sets values(_cellIndex,P,D)
    TEUCHOS_TEST_FOR_EXCEPTION(_cellIndex == -1, std::invalid_argument, "must call setCellIndex before calling values!");
    Teuchos::RCP<const FieldContainer<double> > transformedValues = basisCache->getTransformedValues(_basis, _op);
    // (C,F,P,D)
    
    // NOTE that it would be possible to refactor the below using pointer arithmetic to support _op values that don't
    // result in vector values (e.g. OP_X, OP_DIV).  But since there isn't any clear need for these as yet, we leave it for
    // later...
    
    int cardinality = _basisCoefficients.size();
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    // initialize the values we're responsible for setting
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      for (int d=0; d<spaceDim; d++) {
        values(_cellIndex,ptIndex,d) = 0.0;
      }
    }
    
    for (int i=0; i<cardinality; i++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d=0; d<spaceDim; d++) {
          values(_cellIndex,ptIndex,d) += _basisCoefficients(i) * (*transformedValues)(_cellIndex,i,ptIndex,d);
        }
      }
    }
  }
  
  void setCellIndex(int cellIndex) {
    _cellIndex = cellIndex;
  }

  FunctionPtr dx() {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DX) );
  }
  
  FunctionPtr dy() {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DY) );
  }
  
  FunctionPtr dz() {
    return Teuchos::rcp( new CellTransformationFunction(_basis,_basisCoefficients,OP_DZ) );
  }
  
  static FunctionPtr cellTransformation(MeshPtr mesh, int cellID, const vector< ParametricFunctionPtr > edgeFunctions) {
    return Teuchos::rcp( new CellTransformationFunction(mesh,cellID,edgeFunctions));
  }
};

typedef Teuchos::RCP<CellTransformationFunction> CellTransformationFunctionPtr;

MeshTransformationFunction::MeshTransformationFunction(map< int, FunctionPtr> cellTransforms, EOperatorExtended op) {
  _cellTransforms = cellTransforms;
  _op = op;
}

MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, set<int> cellIDsToTransform) : Function(1) { // vector-valued Function
  _op = OP_VALUE;
  for (set<int>::iterator cellIDIt = cellIDsToTransform.begin(); cellIDIt != cellIDsToTransform.end(); cellIDIt++) {
    int cellID = *cellIDIt;
    _cellTransforms[cellID] = CellTransformationFunction::cellTransformation(mesh, cellID, mesh->parametricEdgesForCell(cellID));
  }  
}

void MeshTransformationFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  vector<int> cellIDs = basisCache->cellIDs();
  // identity map is the right thing most of the time
  // we'll do something different only where necessary
  int spaceDim = basisCache->getSpaceDim();
  if (_op == OP_VALUE) {
    values = basisCache->getPhysicalCubaturePoints(); // identity
  } else if (_op == OP_DX) {
    // identity map is 1 in all the x slots, 0 in all others
    int mod_value = 0; // the x slots are the mod spaceDim = 0 slots;
    for (int i=0; i<values.size(); i++) {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  } else if (_op == OP_DY) {
    // identity map is 1 in all the y slots, 0 in all others
    int mod_value = 1; // the y slots are the mod spaceDim = 1 slots;
    for (int i=0; i<values.size(); i++) {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  } else if (_op == OP_DZ) {
    // identity map is 1 in all the z slots, 0 in all others
    int mod_value = 2; // the z slots are the mod spaceDim = 2 slots;
    for (int i=0; i<values.size(); i++) {
      values[i] = (i%spaceDim == mod_value) ? 1.0 : 0.0;
    }
  }
  for (int cellIndex=0; cellIndex < cellIDs.size(); cellIndex++) {
    int cellID = cellIDs[cellIndex];
    if (_cellTransforms.find(cellID) == _cellTransforms.end()) continue;
    FunctionPtr cellTransformation = _cellTransforms[cellID];
    ((CellTransformationFunction*)cellTransformation.get())->setCellIndex(cellIndex);
    cellTransformation->values(values, basisCache);
  }
}

map< int, FunctionPtr > applyOperatorToCellTransforms(const map< int, FunctionPtr > &cellTransforms, EOperatorExtended op) {
  map<int, FunctionPtr > newTransforms;
  for (map< int, FunctionPtr >::const_iterator cellTransformIt = cellTransforms.begin();
       cellTransformIt != cellTransforms.end(); cellTransformIt++) {
    int cellID = cellTransformIt->first;
    newTransforms[cellID] = Function::op(cellTransformIt->second, op);
  }
  return newTransforms;
}

FunctionPtr MeshTransformationFunction::dx() {
  EOperatorExtended op = OP_DX;
  return Teuchos::rcp( new MeshTransformationFunction(applyOperatorToCellTransforms(_cellTransforms, op),op));
}

FunctionPtr MeshTransformationFunction::dy() {
  EOperatorExtended op = OP_DY;
  return Teuchos::rcp( new MeshTransformationFunction(applyOperatorToCellTransforms(_cellTransforms, op),op));
}

FunctionPtr MeshTransformationFunction::dz() {
  EOperatorExtended op = OP_DZ;
  return Teuchos::rcp( new MeshTransformationFunction(applyOperatorToCellTransforms(_cellTransforms, op),op));
}