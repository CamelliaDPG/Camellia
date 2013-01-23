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

#include "Intrepid_FunctionSpaceTools.hpp"

#include "SerialDenseSolveWrapper.h"

#include "GnuPlotUtil.h"

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
  
  FieldContainer<double> pointLatticeQuad(int numPointsTotal, const vector< ParametricFunctionPtr > &edgeFunctions) {
    int spaceDim = 2;
    FieldContainer<double> pointLattice(numPointsTotal,spaceDim);
    
    // arg to numPoints corresponds to "t1" ("x" direction t), the value to t2 ("y" direction t)
    vector< int > numPoints;
    int approxPoints1D = (int) sqrt( numPointsTotal );
    int remainder = numPointsTotal - approxPoints1D * approxPoints1D;
    for (int i=0; i<approxPoints1D; i++) {
      int extraPoint = (i < remainder) ? 1 : 0;
      numPoints.push_back(approxPoints1D + extraPoint);
    }
    
    {
      // quick and dirty sanity check
      int actualPointCount = 0;
      for (int i=0; i<approxPoints1D; i++) {
        actualPointCount += numPoints[i];
      }
      if (actualPointCount == numPointsTotal) {
//        cout << "Actual point count matches.\n";
      } else {
        cout << "pointLatticeQuad: Actual point count does not match!\n";
      }
    }
    
    if (edgeFunctions.size() != 4) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "wrong number of edgeFunctions");
    }
    // adjust edgeFunctions so that opposite edges run parallel (in the mesh, they run counterclockwise)
    vector< ParametricFunctionPtr > parallelEdgeFunctions = edgeFunctions;
    parallelEdgeFunctions[2] = ParametricFunction::remapParameter(parallelEdgeFunctions[2], 1, 0);
    parallelEdgeFunctions[3] = ParametricFunction::remapParameter(parallelEdgeFunctions[3], 1, 0);
    
    int numPoints_t1 = numPoints.size();
    
    double x0, x1, x2, x3, y0, y1, y2, y3; // values for the various functions
    
    int pointIndex = 0;
    for (int t1_index = 0; t1_index < numPoints_t1; t1_index++) {
      int numPoints_t2 = numPoints[t1_index];
      double t1 = ((double)t1_index) / (double) (numPoints_t1 - 1);
      parallelEdgeFunctions[0]->value(t1, x0, y0);
      parallelEdgeFunctions[2]->value(t1, x2, y2);
      
      for (int t2_index=0; t2_index < numPoints_t2; t2_index++) {
        double t2 = ((double)t2_index) / (double) (numPoints_t2 - 1);
        
//        cout << "(t1,t2) = (" << t1 << ", " << t2 << ")" << endl;
        
        parallelEdgeFunctions[1]->value(t2, x1, y1);
        parallelEdgeFunctions[3]->value(t2, x3, y3);
        
        double x, y;
        
        if (t1 == 0) { // side 3 wins
          x = x3;
          y = y3;
        } else if (t1 == 1) { // side 1 wins
          x = x1;
          y = y1;
        } else if (t2==0) { // side 0
          x = x0;
          y = y0;
        } else if (t2==1) { // side 2
          x = x2;
          y = y2;
        } else { // take a convex combination (and hope it's in the interior!
          x = 0.5 * ( (1-t2) * x0 + t2 * x2 + (1-t1) * x3 + t1 * x1 );
          y = 0.5 * ( (1-t2) * y0 + t2 * y2 + (1-t1) * y3 + t1 * y1 );
        }

        pointLattice(pointIndex,0) = x;
        pointLattice(pointIndex,1) = y;
        pointIndex++;
      }
    }
    return pointLattice;
  }
  
protected:
  CellTransformationFunction(BasisPtr basis, FieldContainer<double> &basisCoefficients, EOperatorExtended op) : Function(1) {
    _basis = basis;
    _basisCoefficients = basisCoefficients;
    _op = op;
    _cellIndex = -1;
  }
public:
  CellTransformationFunction(MeshPtr mesh, int cellID, const vector< ParametricFunctionPtr > &edgeFunctions) : Function(1) {
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
    vector< ParametricFunctionPtr > straightEdges = edgeLines(mesh, cellID);

    FieldContainer<double> points(numPoints,spaceDim);
    FieldContainer<double> transformedPoints(numPoints,spaceDim);
    
    if ( numEdges != 4) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only quads supported right now");
//      points = pointLatticeTriangle(numPoints, straightEdges);
//      transformedPoints = pointLatticeTriangle(numPoints, edgeFunctions);
    } else {
      points = pointLatticeQuad(numPoints, straightEdges);
      transformedPoints = pointLatticeQuad(numPoints, edgeFunctions);
      
//      cout << "points:\n" << points;
//      cout << "transformedPoints:\n" << transformedPoints;
    }
    
    // now, since _basis is defined on refCell, compute preimage of points
    // (would be cleaner to do the parametric function thing on the ref cell)
    FieldContainer<double> refCellPoints(numPoints,spaceDim);
    CellTools<double>::mapToReferenceFrame(refCellPoints,points,mesh->physicalCellNodesForCell(cellID),
                                           *(cell->elementType()->cellTopoPtr),0);
    FieldContainer<double> basisValues(cardinality,numPoints,spaceDim);  // (F,P,D)
    _basis->getValues(basisValues, refCellPoints, Intrepid::OPERATOR_VALUE);
    
    // reshape to make a square matrix:
    basisValues.resize(cardinality, numPoints*spaceDim);
    
    GnuPlotUtil::writeXYPoints("/tmp/refCellPoints.dat", refCellPoints);
    GnuPlotUtil::writeXYPoints("/tmp/transformedPoints.dat", transformedPoints);
    
    // reshape transformedPoints to be a vector:
    transformedPoints.resize(numPoints*spaceDim);
    // rows should be the points, so we want to use the transpose of this
    SerialDenseSolveWrapper::solveSystem(_basisCoefficients, basisValues, transformedPoints, true);
    
//    if (_basis->getDegree()>1) {
//      cout << "---------- WORKING DATA -----------\n";
//      cout << "basisValues:\n" << basisValues;
//      cout << "refCellPoints:\n" << refCellPoints;
//    }
    
    bool checkCoefficients = true; // checks to make sure _basisCoefficients do the job
    if (checkCoefficients) {
      double tol = 1e-14;
      basisValues.resize(cardinality,numPoints,spaceDim);
      transformedPoints.resize(numPoints,spaceDim);
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
        double xValue = 0, yValue = 0;
        for (int basisOrdinal=0; basisOrdinal<cardinality; basisOrdinal++) {
          double weight = _basisCoefficients(basisOrdinal);
          xValue += weight * basisValues(basisOrdinal,ptIndex,0);
          yValue += weight * basisValues(basisOrdinal,ptIndex,1);
        }
        double xValueExpected = transformedPoints(ptIndex,0);
        double yValueExpected = transformedPoints(ptIndex,1);
        double xDiff = abs(xValue-xValueExpected);
        double yDiff = abs(yValue-yValueExpected);
        if ((xDiff > tol) || (yDiff > tol) ) {
          cout << "WARNING: MeshTransformationFunction fails to transform to desired point lattice.\n";
          cout << "Expected: (" <<  xValueExpected << "," << yValueExpected << "), but got (";
          cout << xValue << "," << yValue << ")\n";
        }
      }
      
      // one more check: use our values() method to compute values with these refCellPoints
      FieldContainer<double> transformedPointsToCheck(1,numPoints,spaceDim);
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      basisCache->setRefCellPoints(refCellPoints);
      this->setCellIndex(0);
      this->values(transformedPointsToCheck,basisCache);
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
        double xValue = transformedPointsToCheck(0,ptIndex,0);
        double yValue = transformedPointsToCheck(0,ptIndex,1);
        double xValueExpected = transformedPoints(ptIndex,0);
        double yValueExpected = transformedPoints(ptIndex,1);
        double xDiff = abs(xValue-xValueExpected);
        double yDiff = abs(yValue-yValueExpected);
        if ((xDiff > tol) || (yDiff > tol) ) {
          cout << "WARNING: MeshTransformationFunction fails to transform to desired point lattice (using the values() method).\n";
          cout << "Expected: (" <<  xValueExpected << "," << yValueExpected << "), but got (";
          cout << xValue << "," << yValue << ")\n";
        } else {
//          cout << "Point " << ptIndex << ": Success\n";
        }
      }
    }
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // sets values(_cellIndex,P,D)
    TEUCHOS_TEST_FOR_EXCEPTION(_cellIndex == -1, std::invalid_argument, "must call setCellIndex before calling values!");

//    cout << "_basisCoefficients:\n" << _basisCoefficients;
    
    int transformedCellIndex = _cellIndex;
    
    Teuchos::RCP<const FieldContainer<double> > transformedValues;
    if (_op == OP_VALUE) {
      // here, we depend on the fact that our basis (HGRAD_transform_VALUE) doesn't actually change under transformation
      int cardinality = _basis->getCardinality();
      int numPoints = basisCache->getRefCellPoints().dimension(0);
      int spaceDim = basisCache->getRefCellPoints().dimension(1);
      FieldContainer<double> basisValues(cardinality,numPoints,spaceDim);  // (F,P,D)
      _basis->getValues(basisValues, basisCache->getRefCellPoints(), Intrepid::OPERATOR_VALUE);
      
//      if (_basis->getDegree()>1) {
//        cout << "---------- values() DATA -----------\n";
//        cout << "basisValues:\n" << basisValues;
//        cout << "refCellPoints:\n" << basisCache->getRefCellPoints();
//      }
      
//      Teuchos::RCP<const FieldContainer<double> > refValues = basisCache->getValues(_basis, OP_VALUE);
//      int numCells = basisCache->cellIDs().size();
//      Teuchos::Array<int> dimensions;
//      refValues->dimensions(dimensions);
//      dimensions.insert(dimensions.begin(), numCells);
//      Teuchos::RCP<FieldContainer<double> > transformedValuesMutable = Teuchos::rcp(new FieldContainer<double>(dimensions));
//      FunctionSpaceTools::HGRADtransformVALUE<double>(*transformedValuesMutable,*refValues);
//      transformedValues = transformedValuesMutable;
      basisValues.resize(1,cardinality,numPoints,spaceDim);
      transformedValues = Teuchos::rcp(new FieldContainer<double>(basisValues));
      transformedCellIndex = 0; // we're in our own transformed container, so locally 0 is our cellIndex.
    } else {
      transformedValues = basisCache->getTransformedValues(_basis, _op);
//      cout << "transformedValues:\n" << *transformedValues;
    }
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
          values(_cellIndex,ptIndex,d) += _basisCoefficients(i) * (*transformedValues)(transformedCellIndex,i,ptIndex,d);
        }
      }
    }
  }
  
  int basisDegree() {
    return _basis->getDegree();
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
  
  static Teuchos::RCP<CellTransformationFunction> cellTransformation(MeshPtr mesh, int cellID, const vector< ParametricFunctionPtr > edgeFunctions) {
    return Teuchos::rcp( new CellTransformationFunction(mesh,cellID,edgeFunctions));
  }
};

typedef Teuchos::RCP<CellTransformationFunction> CellTransformationFunctionPtr;

MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, map< int, FunctionPtr> cellTransforms, EOperatorExtended op) : Function(1) {
  _mesh = mesh;
  _cellTransforms = cellTransforms;
  _op = op;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)
}

MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, set<int> cellIDsToTransform) : Function(1) { // vector-valued Function
  _op = OP_VALUE;
  _mesh = mesh;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)
  this->updateCells(cellIDsToTransform);
}

int MeshTransformationFunction::maxDegree() {
  return _maxPolynomialDegree;
}

void MeshTransformationFunction::updateCells(const set<int> &cellIDs) {
  for (set<int>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    int cellID = *cellIDIt;
    CellTransformationFunctionPtr cellTransform =  CellTransformationFunction::cellTransformation(_mesh, cellID, _mesh->parametricEdgesForCell(cellID));
    _cellTransforms[cellID] = cellTransform;
    _maxPolynomialDegree = max(_maxPolynomialDegree,cellTransform->basisDegree());
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
//  if (_op == OP_DX) {
//    cout << "values before cellTransformation:\n" << values;
//  }
  for (int cellIndex=0; cellIndex < cellIDs.size(); cellIndex++) {
    int cellID = cellIDs[cellIndex];
    if (_cellTransforms.find(cellID) == _cellTransforms.end()) continue;
    FunctionPtr cellTransformation = _cellTransforms[cellID];
    ((CellTransformationFunction*)cellTransformation.get())->setCellIndex(cellIndex);
    cellTransformation->values(values, basisCache);
  }
//  if (_op == OP_DX) {
//    cout << "values after cellTransformation:\n" << values;
//  }
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
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}

FunctionPtr MeshTransformationFunction::dy() {
  if (_mesh->getDimension() < 2) {
    return Function::null();
  }
  EOperatorExtended op = OP_DY;
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}

FunctionPtr MeshTransformationFunction::dz() {
  if (_mesh->getDimension() < 3) {
    return Function::null();
  }
  EOperatorExtended op = OP_DZ;
  return Teuchos::rcp( new MeshTransformationFunction(_mesh, applyOperatorToCellTransforms(_cellTransforms, op),op));
}