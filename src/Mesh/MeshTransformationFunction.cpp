//
//  MeshTransformationFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

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

#include "BasisCache.h"

typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;


// TODO: move all the stuff to do with transfinite interpolation into ParametricSurface.cpp

VectorBasisPtr basisForTransformation(ElementTypePtr cellType) {
  unsigned int cellTopoKey = cellType->cellTopoPtr->getKey();
  
  int polyOrder = max(cellType->trialOrderPtr->maxBasisDegree(), cellType->testOrderPtr->maxBasisDegree());
  
  BasisPtr basis = BasisFactory::basisFactory()->getBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
  VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<> *)basis.get(),false); // dynamic cast would be better
  return vectorBasis;
}

vector< ParametricCurvePtr > edgeLines(MeshPtr mesh, int cellID) {
  vector< ParametricCurvePtr > lines;
  ElementPtr cell = mesh->getElement(cellID);
  vector<unsigned> vertexIndices = mesh->vertexIndicesForCell(cellID);
  for (int i=0; i<vertexIndices.size(); i++) {
    FieldContainer<double> v0 = mesh->vertexCoordinates(vertexIndices[i]);
    FieldContainer<double> v1 = mesh->vertexCoordinates(vertexIndices[(i+1)%vertexIndices.size()]);
    // 2D only for now
    TEUCHOS_TEST_FOR_EXCEPTION(v0.dimension(0) != 2, std::invalid_argument, "only 2D supported right now");
    lines.push_back(ParametricCurve::line(v0(0), v0(1), v1(0), v1(1)));
  }
  return lines;
}

void roundToOneOrZero(double &value, double tol) {
  // if value is within tol of 1 or 0, replace value by 1 or 0
  if (abs(value-1)<tol) {
    value = 1;
  } else if (abs(value)<tol) {
    value = 0;
  }
}

class CellTransformationFunction : public Function {
  FieldContainer<double> _basisCoefficients;
  VectorBasisPtr _basis;
  EOperatorExtended _op;
  int _cellIndex; // index into BasisCache's list of cellIDs; must be set prior to each call to values() (there's a reason why this is a private class!)
  
  FieldContainer<double> pointLatticeQuad(int numPointsTotal, const vector< ParametricCurvePtr > &edgeFunctions) {
    int spaceDim = 2;
    FieldContainer<double> pointLattice(numPointsTotal,spaceDim);
    
    ParametricSurfacePtr interpolant = ParametricSurface::transfiniteInterpolant(edgeFunctions);
    
    // arg to numPoints corresponds to "t1" ("x" direction t), the value to t2 ("y" direction t)
    vector< int > numPoints;
    int approxPoints1D = (int) sqrt( numPointsTotal );
    int remainder = numPointsTotal - approxPoints1D * approxPoints1D;
    for (int i=0; i<approxPoints1D; i++) {
      int extraPoint = (i < remainder) ? 1 : 0;
      numPoints.push_back(approxPoints1D + extraPoint);
    }

    if (edgeFunctions.size() != 4) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "wrong number of edgeFunctions");
    }
    
    int numPoints_t1 = numPoints.size();
    
    int pointIndex = 0;
    for (int t1_index = 0; t1_index < numPoints_t1; t1_index++) {
      int numPoints_t2 = numPoints[t1_index];
      double t1 = ((double)t1_index) / (double) (numPoints_t1 - 1);
      
      for (int t2_index=0; t2_index < numPoints_t2; t2_index++) {
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
  CellTransformationFunction(VectorBasisPtr basis, FieldContainer<double> &basisCoefficients, EOperatorExtended op) : Function(1) {
    _basis = basis;
    _basisCoefficients = basisCoefficients;
    _op = op;
    _cellIndex = -1;
  }
public:
  CellTransformationFunction(MeshPtr mesh, int cellID, const vector< ParametricCurvePtr > &edgeFunctions) : Function(1) {
    _cellIndex = -1;
    _op = OP_VALUE;
    ElementPtr cell = mesh->getElement(cellID);
    _basis = basisForTransformation(cell->elementType());
    ParametricSurface::basisWeightsForProjectedInterpolant(_basisCoefficients, _basis, mesh, cellID);
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // sets values(_cellIndex,P,D)
    TEUCHOS_TEST_FOR_EXCEPTION(_cellIndex == -1, std::invalid_argument, "must call setCellIndex before calling values!");

//    cout << "_basisCoefficients:\n" << _basisCoefficients;
        
    int numDofs = _basis->getCardinality();
    
    int spaceDim = basisCache->getSpaceDim();
    
    bool basisIsVolumeBasis = true;
    if (spaceDim==2) {
      basisIsVolumeBasis = (_basis->domainTopology().getBaseKey() != shards::Line<2>::key);
    } else if (spaceDim==3) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim==3 not yet supported in basisIsVolumeBasis determination.");
    }
    
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
    
    constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);
    
    // transformedValues has dimensions (C,F,P,[D,D])
    // therefore, the rank of the sum is transformedValues->rank() - 3
    int rank = transformedValues->rank() - 3;
    TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
    
    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    // initialize the values we're responsible for setting
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      for (int d=0; d<spaceDim; d++) {
        values(_cellIndex,ptIndex,d) = 0.0;
      }
    }
    
    int entriesPerPoint = values.size() / (numCells * numPoints);
    for (int i=0;i<numDofs;i++){
      double weight = _basisCoefficients(i);
      for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
        int valueIndex = (_cellIndex*numPoints + ptIndex)*entriesPerPoint;
        int basisValueIndex = (_cellIndex*numPoints*numDofs + i*numPoints + ptIndex) * entriesPerPoint;
        double *value = &values[valueIndex];
        const double *basisValue = &((*transformedValues)[basisValueIndex]);
        for (int j=0; j<entriesPerPoint; j++) {
          *value++ += *basisValue++ * weight;
        }
      }
    }    
    
    // original implementation follows
    // (the above adapted from NewBasisSumFunction)
//    if (_op == OP_VALUE) {
//      // here, we depend on the fact that our basis (HGRAD_transform_VALUE) doesn't actually change under transformation
//      int cardinality = _basis->getCardinality();
//      const FieldContainer<double>* refCellPoints;
//      if (basisCache->isSideCache()) {
//        refCellPoints = &basisCache->getSideRefCellPointsInVolumeCoordinates();
//      } else {
//        refCellPoints = &basisCache->getRefCellPoints();
//      }
//      int numPoints = refCellPoints->dimension(0);
//      int spaceDim = basisCache->getSpaceDim();
//      FieldContainer<double> basisValues(cardinality,numPoints,spaceDim);  // (F,P,D)
//      _basis->getValues(basisValues, *refCellPoints, Intrepid::OPERATOR_VALUE);
//      basisValues.resize(1,cardinality,numPoints,spaceDim);
//      transformedValues = Teuchos::rcp(new FieldContainer<double>(basisValues));
//      transformedCellIndex = 0; // we're in our own transformed container, so locally 0 is our cellIndex.
//    } else {
//      bool useSideRefCellPoints = basisCache->isSideCache();
//      transformedValues = basisCache->getTransformedValues(_basis, _op, useSideRefCellPoints);
////      cout << "transformedValues:\n" << *transformedValues;
//    }
//    // (C,F,P,D)
//    
//    // NOTE that it would be possible to refactor the below using pointer arithmetic to support _op values that don't
//    // result in vector values (e.g. OP_X, OP_DIV).  But since there isn't any clear need for these as yet, we leave it for
//    // later...
//    
//    int cardinality = _basisCoefficients.size();
//    int numPoints = values.dimension(1);
//    int spaceDim = values.dimension(2);
//    
//    // initialize the values we're responsible for setting
//    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//      for (int d=0; d<spaceDim; d++) {
//        values(_cellIndex,ptIndex,d) = 0.0;
//      }
//    }
//    
//    for (int i=0; i<cardinality; i++) {
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        for (int d=0; d<spaceDim; d++) {
//          values(_cellIndex,ptIndex,d) += _basisCoefficients(i) * (*transformedValues)(transformedCellIndex,i,ptIndex,d);
//        }
//      }
//    }
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
  
  static Teuchos::RCP<CellTransformationFunction> cellTransformation(MeshPtr mesh, GlobalIndexType cellID, const vector< ParametricCurvePtr > edgeFunctions) {
    return Teuchos::rcp( new CellTransformationFunction(mesh,cellID,edgeFunctions));
  }
};

typedef Teuchos::RCP<CellTransformationFunction> CellTransformationFunctionPtr;

// protected method; used for dx(), dy(), dz():
MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, map< GlobalIndexType, FunctionPtr> cellTransforms, EOperatorExtended op) : Function(1) {
  _mesh = mesh;
  _cellTransforms = cellTransforms;
  _op = op;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)

}

MeshTransformationFunction::MeshTransformationFunction(MeshPtr mesh, set<GlobalIndexType> cellIDsToTransform) : Function(1) { // vector-valued Function
  _op = OP_VALUE;
  _mesh = mesh;
  _maxPolynomialDegree = 1; // 1 is the degree of the identity transform (x,y) -> (x,y)
  this->updateCells(cellIDsToTransform);
}

int MeshTransformationFunction::maxDegree() {
  return _maxPolynomialDegree;
}

bool MeshTransformationFunction::mapRefCellPointsUsingExactGeometry(FieldContainer<double> &cellPoints, const FieldContainer<double> &refCellPoints, GlobalIndexType cellID) {
  // returns true if the MeshTransformationFunction handles this cell, false otherwise
  // if true, then populates cellPoints
  if (_cellTransforms.find(cellID) == _cellTransforms.end()) {
    return false;
  }
//  cout << "refCellPoints in mapRefCellPointsUsingExactGeometry():\n" << refCellPoints;
//  
//  cout << "cellPoints prior to mapRefCellPointsUsingExactGeometry():\n" << cellPoints;
  
  Teuchos::RCP< shards::CellTopology > cellTopo = _mesh->getElement(cellID)->elementType()->cellTopoPtr;
  if (cellTopo->getKey() == shards::Quadrilateral<4>::key) {
    int numPoints = refCellPoints.dimension(0);
    int spaceDim = refCellPoints.dimension(1);
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "points must be in 2D for the quad!");
    FieldContainer<double> parametricPoints(numPoints,spaceDim); // map to (t1,t2) space
    int whichCell = 0;
    CellTools<double>::mapToPhysicalFrame(parametricPoints,refCellPoints,
                                          ParametricSurface::parametricQuadNodes(),
                                          *cellTopo,whichCell);
    
//    cout << "parametricPoints in mapRefCellPointsUsingExactGeometry():\n" << parametricPoints;

    vector< ParametricCurvePtr > edgeFunctions = _mesh->parametricEdgesForCell(cellID);
    
    ParametricSurfacePtr interpolant = ParametricSurface::transfiniteInterpolant(edgeFunctions);
    
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double t1 = parametricPoints(ptIndex,0);
      double t2 = parametricPoints(ptIndex,1);

      double x,y;
      // transfinite interpolation:
      interpolant->value(t1, t2, x, y);
      
      cellPoints(ptIndex,0) = x;
      cellPoints(ptIndex,1) = y;
    }
    
  } else {
    // TODO: work out what to do for triangles (or perhaps even a general polygon)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled cell type");
  }
  
//  cout << "cellPoints after to mapRefCellPointsUsingExactGeometry():\n" << cellPoints;
  
  return true;
}

void MeshTransformationFunction::updateCells(const set<GlobalIndexType> &cellIDs) {
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellTransformationFunctionPtr cellTransform =  CellTransformationFunction::cellTransformation(_mesh, cellID, _mesh->parametricEdgesForCell(cellID));
    _cellTransforms[cellID] = cellTransform;
    _maxPolynomialDegree = max(_maxPolynomialDegree,cellTransform->basisDegree());
  }
}

void MeshTransformationFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
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
    GlobalIndexType cellID = cellIDs[cellIndex];
    if (_cellTransforms.find(cellID) == _cellTransforms.end()) continue;
    FunctionPtr cellTransformation = _cellTransforms[cellID];
    ((CellTransformationFunction*)cellTransformation.get())->setCellIndex(cellIndex);
    cellTransformation->values(values, basisCache);
  }
//  if (_op == OP_DX) {
//    cout << "values after cellTransformation:\n" << values;
//  }
}

map< GlobalIndexType, FunctionPtr > applyOperatorToCellTransforms(const map< GlobalIndexType, FunctionPtr > &cellTransforms, EOperatorExtended op) {
  map<GlobalIndexType, FunctionPtr > newTransforms;
  for (map< GlobalIndexType, FunctionPtr >::const_iterator cellTransformIt = cellTransforms.begin();
       cellTransformIt != cellTransforms.end(); cellTransformIt++) {
    GlobalIndexType cellID = cellTransformIt->first;
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

void MeshTransformationFunction::didHRefine(const set<GlobalIndexType> &cellIDs) {
  set<GlobalIndexType> childrenWithCurvedEdges;
  
  MeshTopologyPtr topology = _mesh->getTopology();
  
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType parentCellID = *cellIDIt;
    vector<IndexType> childCells = topology->getCell(parentCellID)->getChildIndices();
    for (vector<IndexType>::iterator childCellIt = childCells.begin(); childCellIt != childCells.end(); childCellIt++) {
      unsigned childCellID = *childCellIt;
      if (topology->cellHasCurvedEdges(childCellID)) {
        childrenWithCurvedEdges.insert(childCellID);
      }
    }
  }
  updateCells(childrenWithCurvedEdges);
}

void MeshTransformationFunction::didPRefine(const set<GlobalIndexType> &cellIDs) {
  updateCells(cellIDs);
}

MeshTransformationFunction::~MeshTransformationFunction() {}