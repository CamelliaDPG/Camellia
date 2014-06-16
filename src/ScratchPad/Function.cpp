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
#include "Teuchos_GlobalMPISession.hpp"
#include "MPIWrapper.h"
#include "CellCharacteristicFunction.h"

#include "Var.h"
#include "Solution.h"

#include "PhysicalPointCache.h"

#include "CamelliaCellTools.h"

// for adaptive quadrature
struct CacheInfo {
  ElementTypePtr elemType;
  GlobalIndexType cellID;
  FieldContainer<double> subCellNodes;
};

// private class ComponentFunction
class ComponentFunction : public Function {
  FunctionPtr _vectorFxn;
  int _component;
public:
  ComponentFunction(FunctionPtr vectorFunction, int componentIndex) {
    _vectorFxn = vectorFunction;
    _component = componentIndex;
    if (_vectorFxn->rank() < 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vector function must have rank 1 or greater");
    }
  }
  bool boundaryValueOnly() {
    return _vectorFxn->boundaryValueOnly();
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // note this allocation.  There might be ways of reusing memory here, if we had a slightly richer API.
    int spaceDim = basisCache->getSpaceDim();
    Teuchos::Array<int> dim;
    values.dimensions(dim);
    dim.push_back(spaceDim);

    FieldContainer<double> vectorValues(dim);
    _vectorFxn->values(vectorValues, basisCache);

    int numValues = values.size();
    for (int i=0; i<numValues; i++) {
      values[i] = vectorValues[spaceDim*i + _component];
    }
  }
};

// private class CellBoundaryRestrictedFunction
class CellBoundaryRestrictedFunction : public Function {
  FunctionPtr _fxn;
public:
  CellBoundaryRestrictedFunction(FunctionPtr fxn) : Function(fxn->rank()) {
    _fxn = fxn;
  }

  bool boundaryValueOnly() { return true; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    _fxn->values(values, basisCache);
  }
};

class MeshBoundaryCharacteristicFunction : public Function {

public:
  MeshBoundaryCharacteristicFunction() : Function(0) {

  }
  bool boundaryValueOnly() { return true; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    // scalar: values shape is (C,P)
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int sideIndex = basisCache->getSideIndex();
    MeshPtr mesh = basisCache->mesh();
    TEUCHOS_TEST_FOR_EXCEPTION(mesh.get() == NULL, std::invalid_argument, "MeshBoundaryCharacteristicFunction requires a mesh!");
    TEUCHOS_TEST_FOR_EXCEPTION(sideIndex == -1, std::invalid_argument, "MeshBoundaryCharacteristicFunction is only defined on cell boundaries");
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = basisCache->cellIDs()[cellIndex];
      bool onBoundary = mesh->boundary().boundaryElement(cellID, sideIndex);
      double value = onBoundary ? 1 : 0;
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        values(cellIndex,pointIndex) = value;
      }
    }
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return Function::zero();
  }
//  FunctionPtr dz() {
//    return Function::zero();
//  }
};

class MeshSkeletonCharacteristicFunction : public ConstantScalarFunction {

public:
  MeshSkeletonCharacteristicFunction() : ConstantScalarFunction(1, "|_{\\Gamma_h}") {

  }
  bool boundaryValueOnly() { return true; }
};

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

// private class JumpFunction:
//class JumpFunction : public Function {
//  FunctionPtr _fxn; // function defined cell-wise
//public:
//  JumpFunction(FunctionPtr fxn);
//  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
//  string displayString();
//  bool boundaryValueOnly();
//};

Function::Function() {
  _rank = 0;
  _displayString = this->displayString();
  _time = 0;
}
Function::Function(int rank) {
  _rank = rank;
  _displayString = this->displayString();
  _time = 0;
}

string Function::displayString() {
  return "f";
}

int Function::rank() {
  return _rank;
}

void Function::setTime(double time)
{
  _time = time;
}

double Function::getTime()
{
  return _time;
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
  if ( isNull(f) ) {
    return Function::null();
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
  return Teuchos::rcp((Function*)NULL);
}

bool Function::equals(FunctionPtr f, BasisCachePtr basisCacheForCellsToCompare, double tol) {
  if (f->rank() != this->rank()) {
    return false;
  }
  FunctionPtr thisPtr = Teuchos::rcp(this,false);
  FunctionPtr diff = thisPtr-f;

  int numCells = basisCacheForCellsToCompare->getPhysicalCubaturePoints().dimension(0);
  // compute L^2 norm of difference on the cells
  FieldContainer<double> diffs_squared(numCells);
  (diff*diff)->integrate(diffs_squared, basisCacheForCellsToCompare);
  double sum = 0;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    sum += diffs_squared[cellIndex];
  }
  return sqrt(sum) < tol;
}

double Function::evaluate(FunctionPtr f, double x) {
  static FieldContainer<double> value(1,1); // (C,P)
  static FieldContainer<double> physPoint(1,1,1);

  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  if (f->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  f->values(value,dummyCache);
  return value[0];
}

double Function::evaluate(FunctionPtr f, double x, double y) { // for testing; this isn't super-efficient
  static FieldContainer<double> value(1,1);
  static FieldContainer<double> physPoint(1,1,2);
  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
  if (f->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  f->values(value,dummyCache);
  return value[0];
}

double Function::evaluate(FunctionPtr f, double x, double y, double z) { // for testing; this isn't super-efficient
  static FieldContainer<double> value(1,1);
  static FieldContainer<double> physPoint(1,1,3);
  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
  dummyCache->writablePhysicalCubaturePoints()(0,0,2) = z;
  if (f->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  f->values(value,dummyCache);
  return value[0];
}

FunctionPtr Function::x() {
  return Function::null();
}
FunctionPtr Function::y() {
  return Function::null();
}
FunctionPtr Function::z() {
  return Function::null();
}

FunctionPtr Function::dx() {
  return Function::null();
}
FunctionPtr Function::dy() {
  return Function::null();
}
FunctionPtr Function::dz() {
  return Function::null();
}
FunctionPtr Function::curl() {
  FunctionPtr dxFxn = dx();
  FunctionPtr dyFxn = dy();
  FunctionPtr dzFxn = dz();
  
  if (dxFxn.get()==NULL) {
    return Function::null();
  } else if (dyFxn.get()==NULL) {
    // special case: in 1D, curl() returns a scalar
    return dxFxn;
  } else if (dzFxn.get() == NULL) {
    // in 2D, the rank of the curl operator depends on the rank of the Function
    if (_rank == 0) {
      return Teuchos::rcp( new VectorizedFunction(dyFxn,-dxFxn) );
    } else if (_rank == 1) {
      return dyFxn->x() - dxFxn->y();
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "curl() undefined for Functions of rank > 1");
    }
  } else {
    return Teuchos::rcp( new VectorizedFunction(dyFxn->z() - dzFxn->y(),
                                                dzFxn->x() - dxFxn->z(),
                                                dxFxn->y() - dyFxn->x()) );
  }
}

FunctionPtr Function::grad(int numComponents) {
  FunctionPtr dxFxn = dx();
  FunctionPtr dyFxn = dy();
  FunctionPtr dzFxn = dz();
  if (numComponents==-1) { // default: just use as many non-null components as available
    if (dxFxn.get()==NULL) {
      return Function::null();
    } else if (dyFxn.get()==NULL) {
      // special case: in 1D, grad() returns a scalar
      return dxFxn;
    } else if (dzFxn.get() == NULL) {
      return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn) );
    } else {
      return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn,dzFxn) );
    }
  } else if (numComponents==1) {
    // special case: we don't "vectorize" in 1D
    return dxFxn;
  } else if (numComponents==2) {
    if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL)) {
      return Function::null();
    } else {
      return Function::vectorize(dxFxn, dyFxn);
    }
  } else if (numComponents==3) {
    if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL) || (dzFxn.get()==NULL)) {
      return Function::null();
    } else {
      return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn,dzFxn) );
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported numComponents");
  return Teuchos::rcp((Function*) NULL);
}
//FunctionPtr Function::inverse() {
//  return Function::null();
//}

bool Function::isNull(FunctionPtr f) {
  return f.get() == NULL;
}

FunctionPtr Function::div() {
  if ( isNull(x()) || isNull(y()) ) {
    return null();
  }
  FunctionPtr dxFxn = x()->dx();
  FunctionPtr dyFxn = y()->dy();
  FunctionPtr zFxn = z();
  if ( isNull(dxFxn) || isNull(dyFxn) ) {
    return null();
  } else if ( isNull(zFxn) || isNull(zFxn->dz()) ) {
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
    //cout << "otherValue = " << valuesToAddTo[i] << "; myValue = " << myValues[i] << endl;
    valuesToAddTo[i] += myValues[i];
  }
}

double Function::integrate(BasisCachePtr basisCache) {
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  FieldContainer<double> cellIntegrals(numCells);
  this->integrate(cellIntegrals, basisCache);
  double sum = 0;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    sum += cellIntegrals[cellIndex];
  }
  return sum;
}

// added by Jesse to check positivity of a function
bool Function::isPositive(BasisCachePtr basisCache){
  bool isPositive = true;
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  FieldContainer<double> fxnValues(numCells,numPoints);
  this->values(fxnValues, basisCache);

  for (int i = 0;i<fxnValues.size();i++){
    if (fxnValues[i] <= 0.0){
      isPositive=false;
      break;
    }
  }
  return isPositive;
}

bool Function::isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich, bool testVsTest){
  bool isPositive = true;
  bool isPositiveOnPartition = true;
  int myPartition = Teuchos::GlobalMPISession::getRank();
  vector<ElementPtr> elems = mesh->elementsInPartition(myPartition);
  for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubEnrich);

    // if we want to check positivity on uniformly spaced points
    if ((*elemIt)->numSides()==4){ // tensor product structure only works with quads
      FieldContainer<double> origPts = basisCache->getRefCellPoints();
      int numPts1D = ceil(sqrt(origPts.dimension(0)));
      int numPts = numPts1D*numPts1D;
      FieldContainer<double> uniformSpacedPts(numPts,origPts.dimension(1));
      double h = 1.0/(numPts1D-1);
      int iter = 0;
      for (int i = 0;i<numPts1D;i++){
	for (int j = 0;j<numPts1D;j++){
	  uniformSpacedPts(iter,0) = 2*h*i-1.0;
	  uniformSpacedPts(iter,1) = 2*h*j-1.0;
	  iter++;
	}
      }
      basisCache->setRefCellPoints(uniformSpacedPts);
    }

    bool isPositiveOnCell = this->isPositive(basisCache);
    if (!isPositiveOnCell){
      isPositiveOnPartition = false;
      break;
    }
  }
  int numPositivePartitions = 1;
  if (!isPositiveOnPartition){
    numPositivePartitions = 0;
  }
  int totalPositivePartitions = MPIWrapper::sum(numPositivePartitions);
  if (totalPositivePartitions<Teuchos::GlobalMPISession::getNProc())
    isPositive=false;

  return isPositive;
}


// added by Jesse - integrate over only one cell
double Function::integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID,testVsTest,cubatureDegreeEnrichment);
  FieldContainer<double> cellIntegral(1);
  this->integrate(cellIntegral,basisCache);
  return cellIntegral(0);
}

FunctionPtr Function::cellCharacteristic(GlobalIndexType cellID) {
  return Teuchos::rcp( new CellCharacteristicFunction(cellID) );
}

FunctionPtr Function::cellCharacteristic(set<GlobalIndexType> cellIDs) {
  return Teuchos::rcp( new CellCharacteristicFunction(cellIDs) );
}

map<int, double> Function::cellIntegrals(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDs(activeCellIDs.begin(),activeCellIDs.end());
  return cellIntegrals(cellIDs,mesh,cubatureDegreeEnrichment,testVsTest);
}

map<int, double> Function::cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
  int myPartition = Teuchos::GlobalMPISession::getRank();

  int numCells = cellIDs.size();
  FieldContainer<double> integrals(numCells);
  for (int i = 0;i<numCells;i++){
    int cellID = cellIDs[i];
    if (mesh->partitionForCellID(cellID) == myPartition){
      integrals(i) = integrate(cellID,mesh,cubatureDegreeEnrichment,testVsTest);
    }
  }
  MPIWrapper::entryWiseSum(integrals);
  map<int,double> integralMap;
  for (int i = 0;i<numCells;i++){
    integralMap[cellIDs[i]] = integrals(i);
  }
  return integralMap;
}


// added by Jesse - adaptive quadrature rules
double Function::integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest) {
  double integral = 0.0;
  int myPartition = Teuchos::GlobalMPISession::getRank();

  vector<ElementPtr> elems = mesh->elementsInPartition(myPartition);

  // build initial list of subcells = all elements
  vector<CacheInfo> subCellCacheInfo;
  for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    GlobalIndexType cellID = (*elemIt)->cellID();
    ElementTypePtr elemType = (*elemIt)->elementType();
    CacheInfo cacheInfo = {elemType,cellID,mesh->physicalCellNodesForCell(cellID)};
    subCellCacheInfo.push_back(cacheInfo);
  }

  // adaptively refine
  bool allConverged = false;
  vector<CacheInfo> subCellsToCheck = subCellCacheInfo;
  int iter = 0;
  int maxIter = 1000; // arbitrary
  while (!allConverged && iter < maxIter){
    allConverged = true;
    ++iter;
    // check relative error, tag subcells to refine
    double tempIntegral = 0.0;
    set<GlobalIndexType> subCellsToRefine;
    for (int i = 0;i<subCellsToCheck.size();i++){
      ElementTypePtr elemType = subCellsToCheck[i].elemType;
      GlobalIndexType cellID = subCellsToCheck[i].cellID;
      FieldContainer<double> nodes = subCellsToCheck[i].subCellNodes;
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,mesh));
      int cubEnrich = 2; // arbitrary
      BasisCachePtr enrichedCache =  Teuchos::rcp(new BasisCache(elemType,mesh,testVsTest,cubEnrich));
      vector<GlobalIndexType> cellIDs;
      cellIDs.push_back(cellID);
      basisCache->setPhysicalCellNodes(nodes,cellIDs,true);
      enrichedCache->setPhysicalCellNodes(nodes,cellIDs,true);

      // calculate relative error for this subcell
      FieldContainer<double> cellIntegral(1),enrichedCellIntegral(1);
      this->integrate(cellIntegral,basisCache);
      this->integrate(enrichedCellIntegral,enrichedCache);
      double error = abs(enrichedCellIntegral(0)-cellIntegral(0))/abs(enrichedCellIntegral(0)); // relative error
      if (error > tol){
        allConverged = false;
        subCellsToRefine.insert(i);
        tempIntegral += enrichedCellIntegral(0);
      }else{
        integral += enrichedCellIntegral(0);
      }
    }
    if (iter == maxIter){
      integral += tempIntegral;
      cout << "maxIter reached for adaptive quadrature, returning integral estimate." << endl;
    }
    //    cout << "on iter " << iter << " with tempIntegral = " << tempIntegral << " and currrent integral = " << integral << " and " << subCellsToRefine.size() << " subcells to go. Allconverged =  " << allConverged << endl;

    // reconstruct subcell list
    vector<CacheInfo> newSubCells;
    for (set<GlobalIndexType>::iterator setIt = subCellsToRefine.begin();setIt!=subCellsToRefine.end();setIt++){
      CacheInfo newCacheInfo = subCellsToCheck[*setIt];
      unsigned cellTopoKey = newCacheInfo.elemType->cellTopoPtr->getKey();
      switch (cellTopoKey)
      {
        case shards::Quadrilateral<4>::key:
          {
            // break into 4 subcells
            int spaceDim = 2; int numCells = 1; // cell-by-cell

            FieldContainer<double> oldNodes = newCacheInfo.subCellNodes;
            oldNodes.resize(4,spaceDim);
            FieldContainer<double> newCellNodes(numCells,4,spaceDim);
            double ax,ay,bx,by,cx,cy,dx,dy,ex,ey;
            ax = .5*(oldNodes(1,0)+oldNodes(0,0)); ay = .5*(oldNodes(1,1)+oldNodes(0,1));
            bx = .5*(oldNodes(2,0)+oldNodes(1,0)); by = .5*(oldNodes(2,1)+oldNodes(1,1));
            cx = .5*(oldNodes(3,0)+oldNodes(2,0)); cy = .5*(oldNodes(3,1)+oldNodes(2,1));
            dx = .5*(oldNodes(3,0)+oldNodes(0,0)); dy = .5*(oldNodes(3,1)+oldNodes(0,1));
            ex = .5*(dx+bx); ey = .5*(cy+ay);

            // first cell
            newCellNodes(0,0,0) = oldNodes(0,0);
            newCellNodes(0,0,1) = oldNodes(0,1);
            newCellNodes(0,1,0) = ax;
            newCellNodes(0,1,1) = ay;
            newCellNodes(0,2,0) = ex;
            newCellNodes(0,2,1) = ey;
            newCellNodes(0,3,0) = dx;
            newCellNodes(0,3,1) = dy;
            newCacheInfo.subCellNodes = newCellNodes;
            newSubCells.push_back(newCacheInfo);

            // second cell
            newCellNodes(0,0,0) = ax;
            newCellNodes(0,0,1) = ay;
            newCellNodes(0,1,0) = oldNodes(1,0);
            newCellNodes(0,1,1) = oldNodes(1,1);
            newCellNodes(0,2,0) = bx;
            newCellNodes(0,2,1) = by;
            newCellNodes(0,3,0) = ex;
            newCellNodes(0,3,1) = ey;
            newCacheInfo.subCellNodes = newCellNodes;
            newSubCells.push_back(newCacheInfo);

            // third cell
            newCellNodes(0,0,0) = ex;
            newCellNodes(0,0,1) = ey;
            newCellNodes(0,1,0) = bx;
            newCellNodes(0,1,1) = by;
            newCellNodes(0,2,0) = oldNodes(2,0);
            newCellNodes(0,2,1) = oldNodes(2,1);
            newCellNodes(0,3,0) = cx;
            newCellNodes(0,3,1) = cy;
            newCacheInfo.subCellNodes = newCellNodes;
            newSubCells.push_back(newCacheInfo);
            // fourth cell
            newCellNodes(0,0,0) = dx;
            newCellNodes(0,0,1) = dy;
            newCellNodes(0,1,0) = ex;
            newCellNodes(0,1,1) = ey;
            newCellNodes(0,2,0) = cx;
            newCellNodes(0,2,1) = cy;
            newCellNodes(0,3,0) = oldNodes(3,0);
            newCellNodes(0,3,1) = oldNodes(3,1);
            newCacheInfo.subCellNodes = newCellNodes;
            newSubCells.push_back(newCacheInfo);
            break;
          }
        default: // case shards::Triangle<3>::key:{} // covers triangles for now
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized in adaptive quadrature routine; topology not implemented");
      }
    }
    // reset subCell list
    subCellsToCheck.clear();
    subCellsToCheck = newSubCells; // new list
  }

  return MPIWrapper::sum(integral);
}

void Function::integrate(FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache,
                         bool sumInto) {
  TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only integrate scalar functions.");
  int numCells = cellIntegrals.dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
//  cout << "integrate: basisCache->getPhysicalCubaturePoints():\n" << basisCache->getPhysicalCubaturePoints();
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
//    if ( (basisCache->cellIDs()[cellIndex]==0) && basisCache->isSideCache() && !sumInto)  {
//      cout << "sideIndex: " << basisCache->getSideIndex() << endl;
//      cout << "Function::integrate() values:\n";
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        cout << ptIndex << ": " << values(cellIndex,ptIndex) << endl;
//      }
//
//      cout << "weightedMeasures:\n";
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        cout << ptIndex << ": " << (*weightedMeasures)(cellIndex,ptIndex) << endl;
//      }
//
//      cout << "weighted values:\n";
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//        cout << ptIndex << ": " << values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex) << endl;
//      }
//    }
//    if (basisCache->getSideIndex() == 0) {
//      cout << "basisCache for side 0, physical cubature points:\n" << basisCache->getPhysicalCubaturePoints();
//      cout << "basisCache for side 0, integrate() values:\n" << values;
//      cout << "basisCache for side 0, weightedMeasures:\n" << *
//      weightedMeasures;
//    }
//    if (cellIndex==6) {
////      cout << "Function::integrate() values:\n" << values;
//      cout << "weightedMeasures:\n" << *weightedMeasures;
//    }
  }
}

// takes integral of jump over entire INTERIOR skeleton
double Function::integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment) {
  double integral = 0.0;
  vector<ElementPtr> elems = mesh->activeElements();
  for (vector<ElementPtr>::iterator elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int numSides = elem->numSides();
    for (int sideIndex = 0; sideIndex < numSides; sideIndex++){
      integral+= this->integralOfJump(mesh,elem->cellID(),sideIndex,cubatureDegreeEnrichment);
    }
  }
  return integral;
}

double Function::integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment) {
  // for boundaries, the jump is 0
  if (mesh->boundary().boundaryElement(cellID,sideIndex)) {
    return 0;
  }
  int neighborCellID = mesh->getElement(cellID)->getNeighborCellID(sideIndex);
  int neighborSideIndex = mesh->getElement(cellID)->getSideIndexInNeighbor(sideIndex);

  ElementTypePtr myType = mesh->getElement(cellID)->elementType();
  ElementTypePtr neighborType = mesh->getElement(neighborCellID)->elementType();

  // TODO: rewrite this to compute in distributed fashion
  vector<GlobalIndexType> myCellIDVector;
  myCellIDVector.push_back(cellID);
  vector<GlobalIndexType> neighborCellIDVector;
  neighborCellIDVector.push_back(neighborCellID);

  BasisCachePtr myCache = Teuchos::rcp(new BasisCache( myType, mesh, true, cubatureDegreeEnrichment));
  myCache->setPhysicalCellNodes(mesh->physicalCellNodesForCell(cellID), myCellIDVector, true);

  BasisCachePtr neighborCache = Teuchos::rcp(new BasisCache( neighborType, mesh, true, cubatureDegreeEnrichment));
  neighborCache->setPhysicalCellNodes(mesh->physicalCellNodesForCell(neighborCellID), neighborCellIDVector, true);

  double sideParity = mesh->cellSideParitiesForCell(cellID)[sideIndex];
  // cellIntegral will store the difference between my value and neighbor's
  FieldContainer<double> cellIntegral(1);
  this->integrate(cellIntegral, neighborCache->getSideBasisCache(neighborSideIndex), true);
//  cout << "Neighbor integral: " << cellIntegral[0] << endl;
  cellIntegral[0] *= -1;
  this->integrate(cellIntegral, myCache->getSideBasisCache(sideIndex), true);
//  cout << "integral difference: " << cellIntegral[0] << endl;

  // multiply by sideParity to make jump uniquely valued.
  return sideParity * cellIntegral(0);
}

double Function::integrate(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest, bool requireSideCache) {
  double integral = 0;

  int myPartition = Teuchos::GlobalMPISession::getRank();
  vector< ElementTypePtr > elementTypes = mesh->elementTypes(myPartition);

  for (vector< ElementTypePtr >::iterator typeIt = elementTypes.begin(); typeIt != elementTypes.end(); typeIt++) {
    ElementTypePtr elemType = *typeIt;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( elemType, mesh, testVsTest, cubatureDegreeEnrichment) ); // all elements of same type
    vector< ElementPtr > cells = mesh->elementsOfType(myPartition, elemType);

    int numCells = cells.size();
    vector<GlobalIndexType> cellIDs;
    for (IndexType cellIndex = 0; cellIndex < numCells; cellIndex++) {
      cellIDs.push_back( cells[cellIndex]->cellID() );
    }
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, this->boundaryValueOnly() || requireSideCache);
    FieldContainer<double> cellIntegrals(numCells);
    if ( this->boundaryValueOnly() ) {
      // sum the integral over the sides...
      int numSides = CamelliaCellTools::getSideCount(*elemType->cellTopoPtr);

      for (int i=0; i<numSides; i++) {
        this->integrate(cellIntegrals, basisCache->getSideBasisCache(i), true);
      }
    } else {
      this->integrate(cellIntegrals, basisCache);
    }
//    cout << "cellIntegrals:\n" << cellIntegrals;
    for (IndexType cellIndex = 0; cellIndex < numCells; cellIndex++) {
      integral += cellIntegrals(cellIndex);
    }
  }

  return MPIWrapper::sum(integral);
}

double Function::l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment) {
  FunctionPtr thisPtr = Teuchos::rcp( this, false );
  return sqrt( (thisPtr * thisPtr)->integrate(mesh, cubatureDegreeEnrichment) );
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

//  cout << "myTensorValues:\n" << myTensorValues;
//  cout << "otherTensorValues:\n" << otherTensorValues;

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
//        cout << "myValue: " << *myValue << "; otherValue: " << *otherValue << endl;
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
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh, true) );
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    int numSides = CamelliaCellTools::getSideCount(cellTopo);

    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      GlobalIndexType cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    int num1DPts = 15;
    FieldContainer<double> refPoints(num1DPts,1);
    for (int i=0; i < num1DPts; i++){
      double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
      refPoints(i,0) = x;
    }

    for (int sideIndex=0; sideIndex < numSides; sideIndex++){
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
      sideBasisCache->setRefCellPoints(refPoints);
      int numCubPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);


      FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
      this->values(computedValues,sideBasisCache);

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
  int num1DPts = 15;

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

  fout << "numCells = " << mesh->numActiveElements() << endl;
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
    vector<GlobalIndexType> cellIDs;
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

FunctionPtr Function::constant(vector<double> &value) {
  return Teuchos::rcp( new ConstantVectorFunction(value) );
}

FunctionPtr Function::meshBoundaryCharacteristic() {
  // 1 on mesh boundary, 0 elsewhere
  return Teuchos::rcp( new MeshBoundaryCharacteristicFunction );
}

FunctionPtr Function::h() {
  return Teuchos::rcp( new hFunction );
}

FunctionPtr Function::meshSkeletonCharacteristic() {
   // 1 on mesh skeleton, 0 elsewhere
  return Teuchos::rcp( new MeshSkeletonCharacteristicFunction );
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

FunctionPtr Function::restrictToCellBoundary(FunctionPtr f) {
  return Teuchos::rcp( new CellBoundaryRestrictedFunction(f) );
}

FunctionPtr Function::solution(VarPtr var, SolutionPtr soln) {
  return Teuchos::rcp( new SimpleSolutionFunction(var, soln) );
}

FunctionPtr Function::vectorize(FunctionPtr f1, FunctionPtr f2) {
  return Teuchos::rcp( new VectorizedFunction(f1,f2) );
}

FunctionPtr Function::vectorize(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3) {
  return Teuchos::rcp( new VectorizedFunction(f1,f2,f3) );
}

FunctionPtr Function::null() {
  static FunctionPtr _null = Teuchos::rcp( (Function*) NULL );
  return _null;
}

FunctionPtr Function::xn(int n) {
  return Teuchos::rcp( new Xn(n) );
}

FunctionPtr Function::yn(int n) {
  return Teuchos::rcp( new Yn(n) );
}

FunctionPtr Function::zn(int n) {
  return Teuchos::rcp( new Zn(n) );
}

FunctionPtr Function::xPart(FunctionPtr vectorFxn) {
  return Teuchos::rcp( new ComponentFunction(vectorFxn, 0) );
}

FunctionPtr Function::yPart(FunctionPtr vectorFxn) {
  return Teuchos::rcp( new ComponentFunction(vectorFxn, 1) );
}

FunctionPtr Function::zPart(FunctionPtr vectorFxn) {
  return Teuchos::rcp( new ComponentFunction(vectorFxn, 2) );
}

FunctionPtr Function::zero(int rank) {
  static FunctionPtr _zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  if (rank==0) {
    return _zero;
  } else {
    FunctionPtr zeroTensor = _zero;
    for (int i=0; i<rank; i++) {
      // THIS ASSUMES 2D--3D would be Function::vectorize(zeroTensor, zeroTensor, zeroTensor)...
      zeroTensor = Function::vectorize(zeroTensor, zeroTensor);
    }
    return zeroTensor;
  }
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

double ConstantScalarFunction::value(double x) {
  return value();
}

double ConstantScalarFunction::value(double x, double y) {
  return value();
}

double ConstantScalarFunction::value(double x, double y, double z) {
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

FunctionPtr ConstantScalarFunction::dz() {
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

ExactSolutionFunction::ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID)
: Function(exactSolution->exactFunctions().find(trialID)->second->rank()) {
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

FunctionPtr ProductFunction::x() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take x component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->x())) {
    return null();
  }
  return _f1 * _f2->x();
}

FunctionPtr ProductFunction::y() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take y component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->y())) {
    return null();
  }
  return _f1 * _f2->y();
}

FunctionPtr ProductFunction::z() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take z component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->z())) {
    return null();
  }
  return _f1 * _f2->z();
}

int ProductFunction::productRank(FunctionPtr f1, FunctionPtr f2) {
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
  return -1;
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

FunctionPtr SumFunction::grad(int numComponents) {
  if ( isNull(_f1->grad(numComponents)) || isNull(_f2->grad(numComponents)) ) {
    return null();
  } else {
    return _f1->grad(numComponents) + _f2->grad(numComponents);
  }
}
FunctionPtr SumFunction::div() {
  if ( isNull(_f1->div()) || isNull(_f2->div()) ) {
    return null();
  } else {
    return _f1->div() + _f2->div();
  }
}

string hFunction::displayString() {
  return "h";
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

// this is liable to be a bit slow!!
class ComposedFunction : public Function {
  FunctionPtr _f, _arg_g;
public:
  ComposedFunction(FunctionPtr f, FunctionPtr arg_g) : Function(f->rank()) {
    _f = f;
    _arg_g = arg_g;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
    FieldContainer<double> fArgPoints(numCells,numPoints,spaceDim);
    if (spaceDim==1) { // special case: arg_g is then reasonably scalar-valued
      fArgPoints.resize(numCells,numPoints);
    }
    _arg_g->values(fArgPoints,basisCache);
    if (spaceDim==1) {
      fArgPoints.resize(numCells,numPoints,spaceDim);
    }
    BasisCachePtr fArgCache = Teuchos::rcp( new PhysicalPointCache(fArgPoints) );
    _f->values(values, fArgCache);
  }
  FunctionPtr dx() {
    if (isNull(_f->dx()) || isNull(_arg_g->dx())) {
      return Function::null();
    }
    // chain rule:
    return _arg_g->dx() * Function::composedFunction(_f->dx(),_arg_g);
  }
  FunctionPtr dy() {
    if (isNull(_f->dy()) || isNull(_arg_g->dy())) {
      return Function::null();
    }
    // chain rule:
    return _arg_g->dy() * Function::composedFunction(_f->dy(),_arg_g);
  }
  FunctionPtr dz() {
    if (isNull(_f->dz()) || isNull(_arg_g->dz())) {
      return Function::null();
    }
    // chain rule:
    return _arg_g->dz() * Function::composedFunction(_f->dz(),_arg_g);
  }
};

FunctionPtr Function::composedFunction( FunctionPtr f, FunctionPtr arg_g) {
  cout << "WARNING: Function::composedFunction() called, but its implementation is not yet complete.\n";
  return Teuchos::rcp( new ComposedFunction(f,arg_g) );
}

double SimpleFunction::value(double x) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleFunction must implement value() for some number of arguments < spaceDim");
  return 0;
}

double SimpleFunction::value(double x, double y) {
  return value(x);
}

double SimpleFunction::value(double x, double y, double z) {
  return value(x,y);
}

void SimpleFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  int spaceDim = points->dimension(2);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (spaceDim == 1) {
        double x = (*points)(cellIndex,ptIndex,0);
        values(cellIndex,ptIndex) = value(x);
      } else if (spaceDim == 2) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = value(x,y);
      } else if (spaceDim == 3) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double z = (*points)(cellIndex,ptIndex,2);
        values(cellIndex,ptIndex) = value(x,y,z);
      }
    }
  }
}

SimpleVectorFunction::SimpleVectorFunction() : Function(1) {}

vector<double> SimpleVectorFunction::value(double x) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleVectorFunction must implement value() for some number of arguments < spaceDim");
  return vector<double>();
}

vector<double> SimpleVectorFunction::value(double x, double y) {
  return value(x);
}

vector<double> SimpleVectorFunction::value(double x, double y, double z) {
  return value(x,y);
}

void SimpleVectorFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  int spaceDim = points->dimension(2);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (spaceDim == 1) {
        double x = (*points)(cellIndex,ptIndex,0);
        values(cellIndex,ptIndex,0) = value(x)[0];
      } else if (spaceDim == 2) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex,0) = value(x,y)[0];
        values(cellIndex,ptIndex,1) = value(x,y)[1];
      } else if (spaceDim == 3) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double z = (*points)(cellIndex,ptIndex,2);
        values(cellIndex,ptIndex,0) = value(x,y,z)[0];
        values(cellIndex,ptIndex,1) = value(x,y,z)[1];
        values(cellIndex,ptIndex,2) = value(x,y,z)[2];
      }
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
  static const double PI  = 3.141592653589793238462;

  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  FieldContainer<double> polarPoints = *points;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      double r = sqrt(x * x + y * y);
      double theta = (r != 0) ? acos(x/r) : 0;
      // now x = r cos theta, but need to guarantee that y = r sin theta (might differ in sign)
      // according to the acos docs, theta will be in [0, pi], so the rule is: (y < 0) ==> theta := 2 pi - theta;
      if (y < 0) theta = 2*PI-theta;

      polarPoints(cellIndex, ptIndex, 0) = r;
      polarPoints(cellIndex, ptIndex, 1) = theta;
//      if (r == 0) {
//        cout << "r == 0!" << endl;
//      }
    }
  }
  BasisCachePtr dummyBasisCache = Teuchos::rcp( new PhysicalPointCache( polarPoints ) );
  _f->values(values,dummyBasisCache);
  if (_f->isZero()) {
    cout << "Warning: in PolarizedFunction, we are being asked for values when _f is zero.  This shouldn't happen.\n";
  }
//  cout << "polarPoints: \n" << polarPoints;
//  cout << "PolarizedFunction, values: \n" << values;
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
  vector<GlobalIndexType> cellIDs = sideBasisCache->cellIDs();
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

FunctionPtr UnitNormalFunction::z() {
  return Teuchos::rcp( new UnitNormalFunction(2) );
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
  int spaceDim = basisCache->getSpaceDim();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (_comp == -1) {
        for (int d=0; d<spaceDim; d++) {
          double nd = (*sideNormals)(cellIndex,ptIndex,d);
          values(cellIndex,ptIndex,d) = nd;
        }
      } else {
        double ni = (*sideNormals)(cellIndex,ptIndex,_comp);
        values(cellIndex,ptIndex) = ni;
      }
    }
  }
}

VectorizedFunction::VectorizedFunction(const vector< FunctionPtr > &fxns) : Function(fxns[0]->rank() + 1) {
  _fxns = fxns;
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
string VectorizedFunction::displayString() {
  ostringstream str;
  str << "(";
  for (int i=0; i<_fxns.size(); i++) {
    if (i > 0) str << ",";
    str << _fxns[i]->displayString();
  }
  str << ")";
  return str.str();
}
int VectorizedFunction::dim() {
  return _fxns.size();
}

void VectorizedFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  // this is not going to be particularly efficient, because values from the components need to be interleaved...
  Teuchos::Array<int> dims;
  values.dimensions(dims);
  int numComponents = dims[dims.size()-1];
  TEUCHOS_TEST_FOR_EXCEPTION( numComponents > _fxns.size(), std::invalid_argument, "too many components requested" );
  if (numComponents != _fxns.size()) {
    // we're asking for fewer components than we have functions.  We're going to say that's OK so long as the
    // unused functions are 0.
    for (int i=numComponents; i<_fxns.size(); i++) {
      if (!_fxns[i]->isZero()) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_fxns outnumber components and some of those extra Functions aren't zero!");
      }
    }
  }
  dims.pop_back(); // remove the last, dimensions argument
  FieldContainer<double> compValues(dims);
  int valuesPerComponent = compValues.size();

  for (int comp=0; comp < numComponents; comp++) {
    FunctionPtr fxn = _fxns[comp];
    fxn->values(compValues, basisCache);
    for (int i=0; i < valuesPerComponent; i++) {
      values[ numComponents * i + comp ] = compValues[ i ];
    }
  }
}

FunctionPtr VectorizedFunction::x() {
  return _fxns[0];
}

FunctionPtr VectorizedFunction::y() {
  return _fxns[1];
}

FunctionPtr VectorizedFunction::z() {
  if (dim() >= 3) {
    return _fxns[2];
  } else {
    return Function::null();
  }
}

FunctionPtr VectorizedFunction::di(int i) {
  // derivative in the ith coordinate direction
  EOperatorExtended op;
  switch (i) {
    case 0:
      op = IntrepidExtendedTypes::OP_DX;
      break;
    case 1:
      op = IntrepidExtendedTypes::OP_DY;
      break;
    case 2:
      op = IntrepidExtendedTypes::OP_DZ;
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid coordinate direction");
      break;
  }
  vector< FunctionPtr > fxns;
  for (int j = 0; j< dim(); j++) {
    FunctionPtr fj_di = Function::op(_fxns[j], op);
    if (isNull(fj_di)) {
      return Function::null();
    }
    fxns.push_back(fj_di);
  }
  // if we made it this far, then all components aren't null:
  return Teuchos::rcp( new VectorizedFunction(fxns) );
}

FunctionPtr VectorizedFunction::dx() {
  return di(0);
}
FunctionPtr VectorizedFunction::dy() {
  return di(1);
}
FunctionPtr VectorizedFunction::dz() {
  return di(2);
}

bool VectorizedFunction::isZero() {
  // vector function is zero if each of its components is zero.
  for (vector< FunctionPtr >::iterator fxnIt = _fxns.begin(); fxnIt != _fxns.end(); fxnIt++) {
    if (! (*fxnIt)->isZero() ) {
      return false;
    }
  }
  return true;
}

FunctionPtr operator*(FunctionPtr f1, FunctionPtr f2) {
  if (f1->isZero() || f2->isZero()) {
    if ( f1->rank() == f2->rank() ) {
      return Function::zero();
    } else if ((f1->rank() == 0) || (f2->rank() == 0)) {
      int result_rank = f1->rank() + f2->rank();
      return Function::zero(result_rank);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"functions have incompatible rank for product.");
    }
  }
  return Teuchos::rcp( new ProductFunction(f1,f2) );
}

FunctionPtr operator/(FunctionPtr f1, FunctionPtr scalarDivisor) {
  if ( f1->isZero() ) {
    return Function::zero(f1->rank());
  }
  return Teuchos::rcp( new QuotientFunction(f1,scalarDivisor) );
}

FunctionPtr operator/(FunctionPtr f1, double divisor) {
  return f1 / Teuchos::rcp( new ConstantScalarFunction(divisor) );
}

FunctionPtr operator/(double value, FunctionPtr scalarDivisor) {
  return Function::constant(value) / scalarDivisor;
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
  return Function::constant(weight) * f;
}

FunctionPtr operator*(FunctionPtr f, double weight) {
  return weight * f;
}

FunctionPtr operator*(vector<double> weight, FunctionPtr f) {
  return Function::constant(weight) * f;
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

FunctionPtr operator+(FunctionPtr f1, double value) {
  return f1 + Function::constant(value);
}

FunctionPtr operator+(double value, FunctionPtr f1) {
  return f1 + Function::constant(value);
}

FunctionPtr operator-(FunctionPtr f1, FunctionPtr f2) {
  return f1 + -f2;
}

FunctionPtr operator-(FunctionPtr f1, double value) {
  return f1 - Function::constant(value);
}

FunctionPtr operator-(double value, FunctionPtr f1) {
  return Function::constant(value) - f1;
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
FunctionPtr Sin_y::dz() {
  return Function::zero();
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
FunctionPtr Cos_y::dz() {
  return Function::zero();
}

string Sin_x::displayString() {
  return "\\sin x";
}

double Sin_x::value(double x, double y) {
  return sin(x);
}
FunctionPtr Sin_x::dx() {
  return Teuchos::rcp( new Cos_x );
}
FunctionPtr Sin_x::dy() {
  return Function::zero();
}
FunctionPtr Sin_x::dz() {
  return Function::zero();
}

string Cos_x::displayString() {
  return "\\cos x";
}
double Cos_x::value(double x, double y) {
  return cos(x);
}
FunctionPtr Cos_x::dx() {
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  return - sin_x;
}
FunctionPtr Cos_x::dy() {
  return Function::zero();
}
FunctionPtr Cos_x::dz() {
  return Function::zero();
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
double Xn::value(double x) {
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
FunctionPtr Xn::dz() {
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
FunctionPtr Yn::dz() {
  return Function::zero();
}

string Zn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "z^" << _n ;
  } else if (_n == 1) {
    ss << "z";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Zn::Zn(int n) {
  _n = n;
}
double Zn::value(double x, double y, double z) {
  return pow(z,_n);
}

FunctionPtr Zn::dx() {
  return Function::zero();
}
FunctionPtr Zn::dy() {
  return Function::zero();
}
FunctionPtr Zn::dz() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr z_n_minus = Teuchos::rcp( new Zn(_n-1) );
  return _n * z_n_minus;
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
  if (_var->varType()==FLUX) { // weight by sideParity
    sideParity()->scalarMultiplyFunctionValues(values, basisCache);
  }
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

//JumpFunction::JumpFunction(FunctionPtr fxn) : Function(fxn->rank()) {
//  _fxn = fxn;
//}
//void JumpFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
//  // TODO: implement this method
//  cout << "WARNING: JumpFunction::values() unimplemented." << endl;
//}
//string JumpFunction::displayString() {
//  ostringstream ss;
//  ss << "[" << _fxn->displayString() << "]";
//  return ss.str();
//}
//bool JumpFunction::boundaryValueOnly() {
//  return true;
//}

Cos_ax::Cos_ax(double a, double b) {
  _a = a;
  _b = b;
}
double Cos_ax::value(double x) {
  return cos( _a * x + _b);
}
FunctionPtr Cos_ax::dx() {
  return -_a * (FunctionPtr) Teuchos::rcp(new Sin_ax(_a,_b));
}
FunctionPtr Cos_ax::dy() {
  return Function::zero();
}

string Cos_ax::displayString() {
  ostringstream ss;
  ss << "\\cos( " << _a << " x )";
  return ss.str();
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


Sin_ax::Sin_ax(double a, double b) {
  _a = a;
  _b = b;
}
double Sin_ax::value(double x) {
  return sin( _a * x + _b);
}
FunctionPtr Sin_ax::dx() {
  return _a * (FunctionPtr) Teuchos::rcp(new Cos_ax(_a,_b));
}
FunctionPtr Sin_ax::dy() {
  return Function::zero();
}
string Sin_ax::displayString() {
  ostringstream ss;
  ss << "\\sin( " << _a << " x )";
  return ss.str();
}

Sin_ay::Sin_ay(double a) {
  _a = a;
}
double Sin_ay::value(double x, double y) {
  return sin( _a * y);
}
FunctionPtr Sin_ay::dx() {
  return Function::zero();
}
FunctionPtr Sin_ay::dy() {
  return _a * (FunctionPtr) Teuchos::rcp(new Cos_ay(_a));
}
string Sin_ay::displayString() {
  ostringstream ss;
  ss << "\\sin( " << _a << " y )";
  return ss.str();
}

Exp_ax::Exp_ax(double a) {
  _a = a;
}
double Exp_ax::value(double x, double y) {
  return exp( _a * x);
}
FunctionPtr Exp_ax::dx() {
  return _a * (FunctionPtr) Teuchos::rcp(new Exp_ax(_a));
}
FunctionPtr Exp_ax::dy() {
  return Function::zero();
}
string Exp_ax::displayString() {
  ostringstream ss;
  ss << "\\exp( " << _a << " x )";
  return ss.str();
}

Exp_ay::Exp_ay(double a) {
  _a = a;
}
double Exp_ay::value(double x, double y) {
  return exp( _a * y);
}
FunctionPtr Exp_ay::dx() {
  return Function::zero();
}
FunctionPtr Exp_ay::dy() {
  return _a * (FunctionPtr) Teuchos::rcp(new Exp_ay(_a));
}
string Exp_ay::displayString() {
  ostringstream ss;
  ss << "\\exp( " << _a << " y )";
  return ss.str();
}