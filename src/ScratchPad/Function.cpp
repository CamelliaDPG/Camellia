//
//  Function.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Function.h"
#include "BasisCache.h"
#include "ExactSolution.h"
#include "Mesh.h"

// private class SimpleSolutionFunction:
class SimpleSolutionFunction : public Function {
  SolutionPtr _soln;
  VarPtr _var;
public:
  SimpleSolutionFunction(VarPtr var, SolutionPtr soln);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  // for reasons of efficiency, may want to implement div() and grad() as well
  
  string displayString();
  bool boundaryValueOnly();
};

Function::Function() {
  _rank = 0;
}
Function::Function(int rank) { 
  _rank = rank; 
}

string Function::displayString() {
  return "f";
}

int Function::rank() { 
  return _rank; 
}

void Function::values(FieldContainer<double> &values, EOperatorExtended op, BasisCachePtr basisCache) {
  switch (op) {
    case IntrepidExtendedTypes::OP_VALUE:
      this->values(values, basisCache);
      break;
    case IntrepidExtendedTypes::OP_DX:
      this->dx()->values(values, basisCache);
      break;
    case IntrepidExtendedTypes::OP_DY:
      this->dy()->values(values, basisCache);
      break;
    case IntrepidExtendedTypes::OP_DZ:
      this->dz()->values(values, basisCache);
      break;
    case IntrepidExtendedTypes::OP_GRAD:
      this->grad()->values(values, basisCache);
      break;
    case IntrepidExtendedTypes::OP_DIV:
      this->div()->values(values, basisCache);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
      break;
  }
  if (op==IntrepidExtendedTypes::OP_VALUE) {
    
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
  }
}

FunctionPtr Function::op(FunctionPtr f, IntrepidExtendedTypes::EOperatorExtended op) {
  if (f.get() == NULL) {
    return Teuchos::rcp( (Function*) NULL);
  }
  switch (op) {
    case IntrepidExtendedTypes::OP_VALUE:
      return f;  
    case IntrepidExtendedTypes::OP_DX:
      return f->dx();
    case IntrepidExtendedTypes::OP_DY:
      return f->dy();
    case IntrepidExtendedTypes::OP_DZ:
      return f->dz();
    case IntrepidExtendedTypes::OP_X:
      return f->x();
    case IntrepidExtendedTypes::OP_Y:
      return f->y();
    case IntrepidExtendedTypes::OP_Z:
      return f->z();
    case IntrepidExtendedTypes::OP_GRAD:
      return f->grad();
    case IntrepidExtendedTypes::OP_DIV:
      return f->div();
    case IntrepidExtendedTypes::OP_DOT_NORMAL:
      return f * Function::normal();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
      break;
  }
}

double Function::evaluate(FunctionPtr f, double x, double y) { // for testing; this isn't super-efficient
  static FieldContainer<double> value(1,1);
  static FieldContainer<double> physPoint(1,1,2);
  static Teuchos::RCP<DummyBasisCacheWithOnlyPhysicalCubaturePoints> dummyCache = Teuchos::rcp( new DummyBasisCacheWithOnlyPhysicalCubaturePoints(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
  if (f->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  f->values(value,dummyCache);
  return value[0];
}

FunctionPtr Function::x() {
  return Teuchos::rcp((Function *)NULL);
}
FunctionPtr Function::y() {
  return Teuchos::rcp((Function *)NULL);
}
FunctionPtr Function::z() {
  return Teuchos::rcp((Function *)NULL);
}

FunctionPtr Function::dx() {
  return Teuchos::rcp((Function *)NULL);
}
FunctionPtr Function::dy() {
  return Teuchos::rcp((Function *)NULL);
}
FunctionPtr Function::dz() {
  return Teuchos::rcp((Function *)NULL);
}
FunctionPtr Function::grad() {
  FunctionPtr dxFxn = dx();
  FunctionPtr dyFxn = dy();
  FunctionPtr dzFxn = dz();
  if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL)) {
    return Teuchos::rcp((Function *)NULL);
  } else if (dzFxn.get() == NULL) {
    return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn) );
  } else {
    return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn,dzFxn) );
  }
}

FunctionPtr Function::div() {
  if ( (x().get() == NULL) || (y().get() == NULL) ) {
    return null();
  }
  FunctionPtr dxFxn = x()->dx();
  FunctionPtr dyFxn = y()->dy();
  FunctionPtr zFxn = z();
  if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL)) {
    return null();
  } else if ((zFxn.get() == NULL) || (zFxn->dz().get() == NULL)) {
    return dxFxn + dyFxn;
  } else {
    return dxFxn + dyFxn + zFxn->dz();
  }
}

void Function::CHECK_VALUES_RANK(FieldContainer<double> &values) { // throws exception on bad values rank
  // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
  TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
}

void Function::addToValues(FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(valuesToAddTo);
  Teuchos::Array<int> dim;
  valuesToAddTo.dimensions(dim);
  FieldContainer<double> myValues(dim);
  this->values(myValues,basisCache);
  for (int i=0; i<myValues.size(); i++) {
    valuesToAddTo[i] += myValues[i];
  }
}

void Function::integrate(FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache,
                         bool sumInto) {
  TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only integrate scalar functions.");
  int numCells = cellIntegrals.dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  FieldContainer<double> values(numCells,numPoints);
  this->values(values,basisCache);
  if ( !sumInto ) {
    cellIntegrals.initialize(0);
  }
  FieldContainer<double> *weightedMeasures = &basisCache->getWeightedMeasures();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      cellIntegrals(cellIndex) += values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex);
    }
  }
}

double Function::integrate(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment) {
  double integral = 0;
  
  // TODO: rewrite this to compute in distributed fashion
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  
  for (vector< ElementTypePtr >::iterator typeIt = elementTypes.begin(); typeIt != elementTypes.end(); typeIt++) {
    ElementTypePtr elemType = *typeIt;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( elemType, mesh, false, cubatureDegreeEnrichment) ); // all elements of same type
    typedef Teuchos::RCP< Element > ElementPtr;
    vector< ElementPtr > cells = mesh->elementsOfTypeGlobal(elemType); // TODO: replace with local variant

    int numCells = cells.size();
    vector<int> cellIDs;
    for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
      cellIDs.push_back( cells[cellIndex]->cellID() );
    }
    // TODO: replace with non-global variant...
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodesGlobal(elemType), cellIDs, this->boundaryValueOnly());
    FieldContainer<double> cellIntegrals(numCells);
    if ( this->boundaryValueOnly() ) {
      // sum the integral over the sides...
      int numSides = elemType->cellTopoPtr->getSideCount();
      for (int i=0; i<numSides; i++) {
        this->integrate(cellIntegrals, basisCache->getSideBasisCache(i), true);
      }
    } else {
      this->integrate(cellIntegrals, basisCache);
    }
    for (int cellID = 0; cellID < numCells; cellID++) {
      integral += cellIntegrals(cellID);
    }
    
  }
  return integral;
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
void Function::scalarMultiplyFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
}

// divide values by this function (supported only when this is a scalar)
void Function::scalarDivideFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
void Function::scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // basisValues has dimensions (C,F,P,...)
//  cout << "scalarMultiplyBasisValues: basisValues:\n" << basisValues;
  scalarModifyBasisValues(basisValues,basisCache,MULTIPLY);
//  cout << "scalarMultiplyBasisValues: modified basisValues:\n" << basisValues;
}

// divide values by this function (supported only when this is a scalar)
void Function::scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // basisValues has dimensions (C,F,P,...)
  scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
}

// note that valuesDottedWithTensor isn't called by anything right now
// (it's totally untried!! -- trying for first time with NewBurgersDriver, in RHS)
void Function::valuesDottedWithTensor(FieldContainer<double> &values, 
                                      FunctionPtr tensorFunctionOfLikeRank, 
                                      BasisCachePtr basisCache) {
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != tensorFunctionOfLikeRank->rank(),std::invalid_argument,
                     "Can't dot functions of unlike rank");
  TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != 2, std::invalid_argument,
                     "values container should have size (numCells, numPoints" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  values.initialize(0.0);
  
  Teuchos::Array<int> tensorValueIndex(_rank+2); // +2 for numCells, numPoints indices
  tensorValueIndex[0] = numCells;
  tensorValueIndex[1] = numPoints;
  for (int d=0; d<_rank; d++) {
    tensorValueIndex[d+2] = spaceDim;
  }
  
  FieldContainer<double> myTensorValues(tensorValueIndex);
  this->values(myTensorValues,basisCache);
  FieldContainer<double> otherTensorValues(tensorValueIndex);
  tensorFunctionOfLikeRank->values(otherTensorValues,basisCache);
  
  // clear out the spatial indices of tensorValueIndex so we can use it as index
  for (int d=0; d<_rank; d++) {
    tensorValueIndex[d+2] = 0;
  }
  
  int entriesPerPoint = 1;
  for (int d=0; d<_rank; d++) {
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    tensorValueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      tensorValueIndex[1] = ptIndex;
      double *myValue = &myTensorValues[ myTensorValues.getEnumeration(tensorValueIndex) ];
      double *otherValue = &otherTensorValues[ otherTensorValues.getEnumeration(tensorValueIndex) ];
      double *value = &values(cellIndex,ptIndex);
      
      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
        *value += *myValue * *otherValue;
        myValue++; 
        otherValue++;
      }
    }
  }
}

void Function::scalarModifyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                          FunctionModificationType modType) {
  TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  FieldContainer<double> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);
  
  Teuchos::Array<int> valueIndex(values.rank());
  
  int entriesPerPoint = 1;
  for (int d=0; d < values.rank()-2; d++) {  // -2 for numCells, numPoints indices
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    valueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      valueIndex[1] = ptIndex;
      double *value = &values[ values.getEnumeration(valueIndex) ];
      double scalarValue = scalarValues(cellIndex,ptIndex);
      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
        if (modType == MULTIPLY) {
          *value++ *= scalarValue;
        } else if (modType == DIVIDE) {
          *value++ /= scalarValue;
        }
      }
    }
  }
}

void Function::scalarModifyBasisValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                       FunctionModificationType modType) {
  TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyBasisValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numFields = values.dimension(1);
  int numPoints = values.dimension(2);
  
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  FieldContainer<double> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);
  
//  cout << "scalarModifyBasisValues: scalarValues:\n" << scalarValues;
  
  Teuchos::Array<int> valueIndex(values.rank());
  
  int entriesPerPoint = 1;
  for (int d=0; d<values.rank()-3; d++) {  // -3 for numCells, numFields, numPoints indices
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    valueIndex[0] = cellIndex;
    for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
      valueIndex[1] = fieldIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        valueIndex[2] = ptIndex;
        double scalarValue = scalarValues(cellIndex,ptIndex);
        double *value = &values[ values.getEnumeration(valueIndex) ];
        for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
          if (modType == MULTIPLY) {
            *value++ *= scalarValue;
          } else if (modType == DIVIDE) {
            *value++ /= scalarValue;
          }
        }
      }
    }
  }
//  cout << "scalarModifyBasisValues: values:\n" << values;
}

void Function::writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath) {
  typedef CellTools<double>  CellTools;
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  
  BasisCachePtr basisCache;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh) );
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    int numSides = cellTopo.getSideCount();
    
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    for (int sideIndex=0; sideIndex < numSides; sideIndex++){
      int numCubPoints = basisCache->getSideBasisCache(sideIndex)->getPhysicalCubaturePoints().dimension(1);
      FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
      this->values(computedValues,basisCache->getSideBasisCache(sideIndex));
      
      // NOW loop over all cells to write solution to file
      for (int cellIndex=0;cellIndex < numCells;cellIndex++){
        FieldContainer<double> cellParities = mesh->cellSideParitiesForCell( cellIDs[cellIndex] );
        for (int pointIndex = 0; pointIndex < numCubPoints; pointIndex++){
          for (int dimInd=0;dimInd<spaceDim;dimInd++){
            fout << (basisCache->getSideBasisCache(sideIndex)->getPhysicalCubaturePoints())(cellIndex,pointIndex,dimInd) << " ";
          }
          fout << computedValues(cellIndex,pointIndex) << endl;
        }
        // insert NaN for matlab to plot discontinuities - WILL NOT WORK IN 3D
        for (int dimInd=0;dimInd<spaceDim;dimInd++){
          fout << "NaN" << " ";
        }
        fout << "NaN" << endl;
      }
    }
  }
  fout.close();
}

void Function::writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath) {
  // MATLAB format, supports scalar functions defined inside 2D volume right now...
  typedef CellTools<double>  CellTools;
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = 5;
  
  int numPoints = num1DPts * num1DPts;
  FieldContainer<double> refPoints(numPoints,spaceDim);
  for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
    for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
      int ptIndex = xPointIndex * num1DPts + yPointIndex;
      double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
      double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
      refPoints(ptIndex,0) = x;
      refPoints(ptIndex,1) = y;
    }
  }
  
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  
  fout << "numCells = " << mesh->activeElements().size() << endl;
  fout << "x=cell(numCells,1);y=cell(numCells,1);z=cell(numCells,1);" << endl;
  
  // initialize storage
  fout << "for i = 1:numCells" << endl;
  fout << "x{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "y{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "z{i} = zeros(" << num1DPts << ");"<<endl;
  fout << "end" << endl;
  int globalCellInd = 1; //matlab indexes from 1
  BasisCachePtr basisCache;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) { //thru quads/triangles/etc
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh) );
    basisCache->setRefCellPoints(refPoints);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, false); // false: don't create side cache

    FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();
    
    FieldContainer<double> computedValues(numCells,numPoints);
    this->values(computedValues, basisCache);	
    
    // NOW loop over all cells to write solution to file
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
        int ptIndex = xPointIndex*num1DPts + yPointIndex;
        for (int cellIndex=0;cellIndex < numCells;cellIndex++){	  
          fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << physCubPoints(cellIndex,ptIndex,0) << ";" << endl;
          fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << physCubPoints(cellIndex,ptIndex,1) << ";" << endl;
          fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,ptIndex) << ";" << endl;	  
        }
      }
    }
    globalCellInd+=numCells;
    
  } //end of element type loop 
  fout.close();
}

FunctionPtr Function::constant(double value) {
  return Teuchos::rcp( new ConstantScalarFunction(value) );
}

FunctionPtr Function::normal() { // unit outward-facing normal on each element boundary
  static FunctionPtr _normal = Teuchos::rcp( new UnitNormalFunction );
  return _normal;
}

FunctionPtr Function::sideParity() { // canonical direction on boundary (used for defining fluxes)
  static FunctionPtr _sideParity = Teuchos::rcp( new SideParityFunction );
  return _sideParity;
}


FunctionPtr Function::polarize(FunctionPtr f) {
  return Teuchos::rcp( new PolarizedFunction(f) );
}

FunctionPtr Function::solution(VarPtr var, SolutionPtr soln) {
  return Teuchos::rcp( new SimpleSolutionFunction(var, soln) );
}

FunctionPtr Function::vectorize(FunctionPtr f1, FunctionPtr f2) {
  return Teuchos::rcp( new VectorizedFunction(f1,f2) );
}

FunctionPtr Function::null() {
  static FunctionPtr _null = Teuchos::rcp( (Function*) NULL );
  return _null;
}

FunctionPtr Function::zero() {
  static FunctionPtr _zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  return _zero;
}

ConstantScalarFunction::ConstantScalarFunction(double value) { 
  _value = value;
  ostringstream valueStream;
  valueStream << value;
  _stringDisplay = valueStream.str();
}

ConstantScalarFunction::ConstantScalarFunction(double value, string stringDisplay) { 
  _value = value; 
  _stringDisplay = stringDisplay;
}

string ConstantScalarFunction::displayString() {
  return _stringDisplay;
}

bool ConstantScalarFunction::isZero() {
  return 0.0 == _value;
}

void ConstantScalarFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  for (int i=0; i < values.size(); i++) {
    values[i] = _value;
  }
}
void ConstantScalarFunction::scalarMultiplyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] *= _value;
    }
  }
}
void ConstantScalarFunction::scalarDivideFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] /= _value;
    }
  }
}
void ConstantScalarFunction::scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
  scalarMultiplyFunctionValues(basisValues,basisCache);
}
void ConstantScalarFunction::scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  scalarDivideFunctionValues(basisValues,basisCache);
}

double ConstantScalarFunction::value(double x, double y) {
  return value();
}

double ConstantScalarFunction::value() {
  return _value;
}

FunctionPtr ConstantScalarFunction::dx() {
  return Function::zero();
}

FunctionPtr ConstantScalarFunction::dy() {
  return Function::zero();
}

ConstantVectorFunction::ConstantVectorFunction(vector<double> value) : Function(1) { 
  _value = value; 
}

FunctionPtr ConstantVectorFunction::x() {
  return Teuchos::rcp( new ConstantScalarFunction( _value[0] ) );
}

FunctionPtr ConstantVectorFunction::y() {
  return Teuchos::rcp( new ConstantScalarFunction( _value[1] ) );
}

vector<double> ConstantVectorFunction::value() {
  return _value;
}

bool ConstantVectorFunction::isZero() {
  for (int d=0; d < _value.size(); d++) {
    if (0.0 != _value[d]) {
      return false;
    }
  }
  return true;
}

void ConstantVectorFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  // values are stored in (C,P,D) order, the important thing here being that we can do this:
  for (int i=0; i < values.size(); ) {
    for (int d=0; d < _value.size(); d++) {
      values[i++] = _value[d];
    }
  }
}

ExactSolutionFunction::ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID) : Function(0) {
  _exactSolution = exactSolution;
  _trialID = trialID;
}
void ExactSolutionFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  _exactSolution->solutionValues(values,_trialID,basisCache);
}

string ProductFunction::displayString() {
  ostringstream ss;
  ss << _f1->displayString() << " \\cdot " << _f2->displayString();
  return ss.str();
}

FunctionPtr ProductFunction::dx() {
  if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dx() + _f2 * _f1->dx();
}

FunctionPtr ProductFunction::dy() {
  if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dy() + _f2 * _f1->dy();
}

FunctionPtr ProductFunction::dz() {
  if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dz() + _f2 * _f1->dz();
}

int ProductFunction::productRank(FunctionPtr f1, FunctionPtr f2) {
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
}

ProductFunction::ProductFunction(FunctionPtr f1, FunctionPtr f2) : Function( productRank(f1,f2) ) {
  // for simplicity of values() code, ensure that rank of f1 ≤ rank of f2:
  if ( f1->rank() <= f2->rank() ) {
    _f1 = f1;
    _f2 = f2;
  } else {
    _f1 = f2;
    _f2 = f1;
  }
  // the following should be false for all the automatic products.  Added the test for debugging...
  if ((_f1->isZero()) || (_f2->isZero())) {
    cout << "Warning: creating a ProductFunction where one of the multiplicands is zero." << endl;
  }
}

bool ProductFunction::boundaryValueOnly() {
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

void ProductFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  if (( _f2->rank() > 0) && (this->rank() == 0)) { // tensor product resulting in scalar value
    _f2->valuesDottedWithTensor(values, _f1, basisCache);
  } else { // scalar multiplication by f1, then
    _f2->values(values,basisCache);
    _f1->scalarMultiplyFunctionValues(values, basisCache);
  }
}

QuotientFunction::QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor) : Function( f->rank() ) {
  if ( scalarDivisor->rank() != 0 ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
  if (scalarDivisor->isZero()) {
    cout << "WARNING: division by zero in QuotientFunction.\n";
  }
}

bool QuotientFunction::boundaryValueOnly() {
  return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
}

string QuotientFunction::displayString() {
  ostringstream ss;
  ss << _f->displayString() << " / " << _scalarDivisor->displayString();
  return ss.str();
}

void QuotientFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
}

FunctionPtr QuotientFunction::dx() {
  if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
}

FunctionPtr QuotientFunction::dy() {
  if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
}

FunctionPtr QuotientFunction::dz() {
  if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
}

SumFunction::SumFunction(FunctionPtr f1, FunctionPtr f2) : Function(f1->rank()) {
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
  TEUCHOS_TEST_FOR_EXCEPTION( f1->boundaryValueOnly() != f2->boundaryValueOnly(), std::invalid_argument,
                              "f1 and f2 must agree on their boundary-valuedness");
  _f1 = f1;
  _f2 = f2;
}

bool SumFunction::boundaryValueOnly() {
  // if either summand is BVO, then so is the sum...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

string SumFunction::displayString() {
  ostringstream ss;
  ss << "(" << _f1->displayString() << " + " << _f2->displayString() << ")";
  return ss.str();
}

void SumFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f1->values(values,basisCache);
  _f2->addToValues(values,basisCache);
}

FunctionPtr SumFunction::x() {
  if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
    return null();
  }
  return _f1->x() + _f2->x();
}

FunctionPtr SumFunction::y() {
  if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
    return null();
  }
  return _f1->y() + _f2->y();  
}
FunctionPtr SumFunction::z() {
  if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
    return null();
  }
  return _f1->z() + _f2->z();
}

FunctionPtr SumFunction::dx() {
  if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) ) {
    return null();
  }
  return _f1->dx() + _f2->dx();
}

FunctionPtr SumFunction::dy() {
  if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) ) {
    return null();
  }
  return _f1->dy() + _f2->dy();
}

FunctionPtr SumFunction::dz() {
  if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) ) {
    return null();
  }
  return _f1->dz() + _f2->dz();
}

double hFunction::value(double x, double y, double h) {
    return h;
}
void hFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    double h = sqrt(cellMeasures(cellIndex));
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y,h);
    }
  }
}

void SimpleFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y);
    }
  }
}

PolarizedFunction::PolarizedFunction( FunctionPtr f_of_xAsR_yAsTheta ) : Function(f_of_xAsR_yAsTheta->rank()) {
  _f = f_of_xAsR_yAsTheta;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::r() {
  static Teuchos::RCP<PolarizedFunction> _r = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Xn(1) ) ) );
  return _r;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::sin_theta() {
  static Teuchos::RCP<PolarizedFunction> _sin_theta = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Sin_y ) ) );
  return _sin_theta;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::cos_theta() {
  static Teuchos::RCP<PolarizedFunction> _cos_theta = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Cos_y ) ) );
  return _cos_theta;
}

void findAndReplace(string &str, const string &findStr, const string &replaceStr) {
  size_t found = str.find( findStr );
  while (found!=string::npos) {
    str.replace( found, findStr.length(), replaceStr );
    found = str.find( findStr );
  }
}

string PolarizedFunction::displayString() {
  string displayString = _f->displayString();
  findAndReplace(displayString, "x", "r");
  findAndReplace(displayString, "y", "\\theta");
  return displayString;
//  ostringstream ss( _f->displayString());
//  ss << "(r,\\theta)";
//  return ss.str();
}

FunctionPtr PolarizedFunction::dx() {
  // cast everything to FunctionPtrs:
  FunctionPtr sin_theta_fxn = sin_theta();
  FunctionPtr dtheta_fxn = dtheta();
  FunctionPtr dr_fxn = dr();
  FunctionPtr r_fxn = r();
  FunctionPtr cos_theta_fxn = cos_theta();
  return dr_fxn * cos_theta_fxn - dtheta_fxn * sin_theta_fxn / r_fxn;
}
FunctionPtr PolarizedFunction::dy() {
  FunctionPtr sin_theta_fxn = sin_theta();
  FunctionPtr dtheta_fxn = dtheta();
  FunctionPtr dr_fxn = dr();
  FunctionPtr r_fxn = r();
  FunctionPtr cos_theta_fxn = cos_theta();
  return dr_fxn * sin_theta_fxn + dtheta_fxn * cos_theta_fxn / r_fxn;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::dtheta() {
  return Teuchos::rcp( new PolarizedFunction( _f->dy() ) );
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::dr() {
  return Teuchos::rcp( new PolarizedFunction( _f->dx() ) );
}

bool PolarizedFunction::isZero() {
  return _f->isZero();
}

void PolarizedFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  FieldContainer<double> polarPoints = *points;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      double r = sqrt(x * x + y * y);
      double theta = acos(x/r);
      // now x = r cos theta, but need to guarantee that y = r sin theta (might differ in sign)
      // according to the acos docs, theta will be in [0, pi], so the rule is: (y < 0) --> theta *= -1;
      if (y < 0) theta *= -1.0;
      
      polarPoints(cellIndex, ptIndex, 0) = r;
      polarPoints(cellIndex, ptIndex, 1) = theta;
//      if (r == 0) {
//        cout << "r == 0!" << endl;
//      }
    }
  }
  BasisCachePtr dummyBasisCache = Teuchos::rcp( new DummyBasisCacheWithOnlyPhysicalCubaturePoints( polarPoints ) );
  _f->values(values,dummyBasisCache);
  if (_f->isZero()) {
    cout << "Warning: in PolarizedFunction, we are being asked for values when _f is zero.  This shouldn't happen.\n";
  }
  //cout << "polarPoints: \n" << polarPoints;
  //cout << "PolarizedFunction, values: \n" << values;
}

bool ScalarFunctionOfNormal::boundaryValueOnly() {
  return true;
}

void ScalarFunctionOfNormal::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  const FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      double n1 = (*sideNormals)(cellIndex,ptIndex,0);
      double n2 = (*sideNormals)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y,n1,n2);
    }
  }
}

SideParityFunction::SideParityFunction() : Function(0) {
//  cout << "SideParityFunction constructor.\n";
}

bool SideParityFunction::boundaryValueOnly() {
  return true;
}

string SideParityFunction::displayString() {
  return "sgn(n)";
}

void SideParityFunction::values(FieldContainer<double> &values, BasisCachePtr sideBasisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int sideIndex = sideBasisCache->getSideIndex();
  if (sideIndex == -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-sideBasisCache passed into SideParityFunction");
  }
  vector<int> cellIDs = sideBasisCache->cellIDs();
  if (cellIDs.size() != numCells) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIDs.size() != numCells");
  }
  Teuchos::RCP<Mesh> mesh = sideBasisCache->mesh();
  if (! mesh.get()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mesh unset in BasisCache.");
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int parity = mesh->cellSideParitiesForCell(cellIDs[cellIndex])(0,sideIndex);
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      values(cellIndex,ptIndex) = parity;
    }
  }
}

UnitNormalFunction::UnitNormalFunction(int comp) : Function( (comp<0)? 1 : 0) {
  _comp = comp;
}

FunctionPtr UnitNormalFunction::x() {
  return Teuchos::rcp( new UnitNormalFunction(0) );
}

FunctionPtr UnitNormalFunction::y() {
  return Teuchos::rcp( new UnitNormalFunction(1) );
}

bool UnitNormalFunction::boundaryValueOnly() {
  return true;
}

string UnitNormalFunction::displayString() {
  if (_comp == -1) {
    return " \\boldsymbol{n} ";
  } else {
    if (_comp == 0) {
      return " n_x ";
    }
    if (_comp == 1) {
      return " n_y ";
    }
    if (_comp == 2) {
      return " n_z ";
    }
    return "UnitNormalFunction with unexpected component";
  }
}

void UnitNormalFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  const FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (_comp == -1) {
        double n1 = (*sideNormals)(cellIndex,ptIndex,0);
        double n2 = (*sideNormals)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex,0) = n1;
        values(cellIndex,ptIndex,1) = n2;
      } else {
        double ni = (*sideNormals)(cellIndex,ptIndex,_comp);
        values(cellIndex,ptIndex) = ni;
      }
    }
  }
}

VectorizedFunction::VectorizedFunction(FunctionPtr f1, FunctionPtr f2) : Function(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
}
VectorizedFunction::VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3) : Function(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
  _fxns.push_back(f3);
}
void VectorizedFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  // this is not going to be particularly efficient, because values from the components need to be interleaved...
  Teuchos::Array<int> dims;
  values.dimensions(dims);
  int numComponents = dims[dims.size()-1];
  TEUCHOS_TEST_FOR_EXCEPTION( numComponents != _fxns.size(), std::invalid_argument, "number of components incorrect" );
  dims.pop_back(); // remove the last, dimensions argument
  FieldContainer<double> compValues(dims);
  int valuesPerComponent = compValues.size();
  
  int numComps = _fxns.size();
  for (int comp=0; comp < numComps; comp++) {
    FunctionPtr fxn = _fxns[comp];
    fxn->values(compValues, basisCache);
    for (int i=0; i < valuesPerComponent; i++) {
      values[ numComps * i + comp ] = compValues[ i ];
    }
  }
}

FunctionPtr VectorizedFunction::x() {
  return _fxns[0];
}

FunctionPtr VectorizedFunction::y() {
  return _fxns[1];
}

FunctionPtr operator*(FunctionPtr f1, FunctionPtr f2) {
  if ( f1->rank() == f2->rank() ) {
    // TODO: work out how to do this for other ranks?
    if (f1->isZero() || f2->isZero()) {
      return Function::zero();
    }
  }
  return Teuchos::rcp( new ProductFunction(f1,f2) );
}

FunctionPtr operator/(FunctionPtr f1, FunctionPtr scalarDivisor) {
  if ( (f1->rank() == 0) ) {
    // TODO: work out how to do this for other ranks?
    if ( f1->isZero() ) {
      return Function::zero();
    }
  }
  return Teuchos::rcp( new QuotientFunction(f1,scalarDivisor) );
}

FunctionPtr operator/(FunctionPtr f1, double divisor) {
  return f1 / Teuchos::rcp( new ConstantScalarFunction(divisor) );
}

//ConstantScalarFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  return Teuchos::rcp( new ConstantScalarFunction(f1->value() * f2->value()) );
//}
//
//ConstantScalarFunctionPtr operator/(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  return Teuchos::rcp( new ConstantScalarFunction(f1->value() / f2->value()) );  
//}

//ConstantVectorFunctionPtr operator*(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  vector<double> value = f1->value();
//  for (int d=0; d<value.size(); d++) {
//    value[d] *= f2->value();
//  }
//  return Teuchos::rcp( new ConstantVectorFunction(value) );  
//}
//
//ConstantVectorFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantVectorFunctionPtr f2) {
//  return f2 * f1;
//}
//
//ConstantVectorFunctionPtr operator/(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  vector<double> value = f1->value();
//  for (int d=0; d<value.size(); d++) {
//    value[d] /= f2->value();
//  }
//  return Teuchos::rcp( new ConstantVectorFunction(value) );  
//}

FunctionPtr operator*(double weight, FunctionPtr f) {
  return Teuchos::rcp( new ConstantScalarFunction(weight) ) * f;
}

FunctionPtr operator*(FunctionPtr f, double weight) {
  return weight * f;
}

FunctionPtr operator*(vector<double> weight, FunctionPtr f) {
  return Teuchos::rcp( new ConstantVectorFunction(weight) ) * f;
}

FunctionPtr operator*(FunctionPtr f, vector<double> weight) {
  return weight * f;
}

FunctionPtr operator+(FunctionPtr f1, FunctionPtr f2) {
  if ( f1->isZero() ) {
    return f2;
  }
  if ( f2->isZero() ) {
    return f1;
  }
  return Teuchos::rcp( new SumFunction(f1, f2) );
}

FunctionPtr operator-(FunctionPtr f1, FunctionPtr f2) {
  return f1 + -f2;
}

FunctionPtr operator-(FunctionPtr f) {
  return -1.0 * f;
}

string Sin_y::displayString() {
  return "\\sin y";
}

double Sin_y::value(double x, double y) {
  return sin(y);
}
FunctionPtr Sin_y::dx() {
  return Function::zero();
}
FunctionPtr Sin_y::dy() {
  return Teuchos::rcp( new Cos_y );
}

string Cos_y::displayString() {
  return "\\cos y";
}
double Cos_y::value(double x, double y) {
  return cos(y);
}
FunctionPtr Cos_y::dx() {
  return Function::zero();
}
FunctionPtr Cos_y::dy() {
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  return - sin_y;
}


string Exp_x::displayString() {
  return "e^x";
}
double Exp_x::value(double x, double y) {
  return exp(x);
}
FunctionPtr Exp_x::dx() {
  return Teuchos::rcp( new Exp_x );
}
FunctionPtr Exp_x::dy() {
  return Function::zero();
}


string Exp_y::displayString() {
  return "e^y";
}
double Exp_y::value(double x, double y) {
  return exp(y);
}
FunctionPtr Exp_y::dx() {
  return Function::zero();
}
FunctionPtr Exp_y::dy() {
  return Teuchos::rcp( new Exp_y );
}


string Xn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "x^" << _n ;
  } else if (_n == 1) {
    ss << "x";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Xn::Xn(int n) {
  _n = n;
}
double Xn::value(double x, double y) {
  return pow(x,_n);
}
FunctionPtr Xn::dx() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr x_n_minus = Teuchos::rcp( new Xn(_n-1) );
  return _n * x_n_minus;
}
FunctionPtr Xn::dy() {
  return Function::zero();
}

string Yn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "y^" << _n ;
  } else if (_n == 1) {
    ss << "y";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Yn::Yn(int n) {
  _n = n;
}
double Yn::value(double x, double y) {
  return pow(y,_n);
}

FunctionPtr Yn::dx() {
  return Function::zero();
}
FunctionPtr Yn::dy() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr y_n_minus = Teuchos::rcp( new Yn(_n-1) );
  return _n * y_n_minus;
}

SimpleSolutionFunction::SimpleSolutionFunction(VarPtr var, SolutionPtr soln) : Function(var->rank()) {
  _var = var;
  _soln = soln;
}

bool SimpleSolutionFunction::boundaryValueOnly() {
  return (_var->varType() == FLUX) || (_var->varType() == TRACE);
}

string SimpleSolutionFunction::displayString() {
  ostringstream str;
  str << "\\overline{" << _var->displayString() << "} ";
  return str.str();
}

void SimpleSolutionFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  bool dontWeightForCubature = false;
  _soln->solutionValues(values, _var->ID(), basisCache, dontWeightForCubature, _var->op());
}

FunctionPtr SimpleSolutionFunction::dx() {
  if (_var->op() != IntrepidExtendedTypes::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dx(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::dy() {
  if (_var->op() != IntrepidExtendedTypes::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dy(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::dz() {
  if (_var->op() != IntrepidExtendedTypes::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dz(), _soln);
  }
}

Cos_ay::Cos_ay(double a) {
  _a = a;
}
double Cos_ay::value(double x, double y) {
  return cos( _a * y );
}
FunctionPtr Cos_ay::dx() {
  return Function::zero();
}
FunctionPtr Cos_ay::dy() {
  return -_a * (FunctionPtr) Teuchos::rcp(new Sin_ay(_a));
}

string Cos_ay::displayString() {
  ostringstream ss;
  ss << "\\cos( " << _a << " y )";
  return ss.str();
}