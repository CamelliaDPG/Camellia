//
//  Function.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Function.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "CellCharacteristicFunction.h"
#include "ConstantScalarFunction.h"
#include "ConstantVectorFunction.h"
#include "ExactSolution.h"
#include "GlobalDofAssignment.h"
#include "hFunction.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "MinMaxFunctions.h"
#include "MonomialFunctions.h"
#include "PhysicalPointCache.h"
#include "PolarizedFunction.h"
#include "ProductFunction.h"
#include "QuotientFunction.h"
#include "SimpleFunction.h"
#include "SimpleSolutionFunction.h"
#include "SimpleVectorFunction.h"
#include "SideParityFunction.h"
#include "Solution.h"
#include "SumFunction.h"
#include "TrigFunctions.h"
#include "UnitNormalFunction.h"
#include "Var.h"
#include "VectorizedFunction.h"

#include "Intrepid_CellTools.hpp"
#include "Teuchos_GlobalMPISession.hpp"

namespace Camellia {
  // for adaptive quadrature
  struct CacheInfo {
    ElementTypePtr elemType;
    GlobalIndexType cellID;
    Intrepid::FieldContainer<double> subCellNodes;
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
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      // note this allocation.  There might be ways of reusing memory here, if we had a slightly richer API.
      int spaceDim = basisCache->getSpaceDim();
      Teuchos::Array<int> dim;
      values.dimensions(dim);
      dim.push_back(spaceDim);

      Intrepid::FieldContainer<double> vectorValues(dim);
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
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      _fxn->values(values, basisCache);
    }
  };

  class HeavisideFunction : public SimpleFunction {
    double _xShift;
  public:
    HeavisideFunction(double xShift=0.0) {
      _xShift = xShift;
    }
    double value(double x) {
      return (x < _xShift) ? 0.0 : 1.0;
    }
  };

  class MeshBoundaryCharacteristicFunction : public Function {

  public:
    MeshBoundaryCharacteristicFunction() : Function(0) {

    }
    bool boundaryValueOnly() { return true; }
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
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
        bool onBoundary = mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex);
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

  void Function::values(Intrepid::FieldContainer<double> &values, Camellia::EOperator op, BasisCachePtr basisCache) {
    switch (op) {
      case Camellia::OP_VALUE:
        this->values(values, basisCache);
        break;
      case Camellia::OP_DX:
        this->dx()->values(values, basisCache);
        break;
      case Camellia::OP_DY:
        this->dy()->values(values, basisCache);
        break;
      case Camellia::OP_DZ:
        this->dz()->values(values, basisCache);
        break;
      case Camellia::OP_GRAD:
        this->grad()->values(values, basisCache);
        break;
      case Camellia::OP_DIV:
        this->div()->values(values, basisCache);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
        break;
    }
    if (op==Camellia::OP_VALUE) {

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
    }
  }

  FunctionPtr Function::op(FunctionPtr f, Camellia::EOperator op) {
    if ( isNull(f) ) {
      return Function::null();
    }
    switch (op) {
      case Camellia::OP_VALUE:
        return f;
      case Camellia::OP_DX:
        return f->dx();
      case Camellia::OP_DY:
        return f->dy();
      case Camellia::OP_DZ:
        return f->dz();
      case Camellia::OP_X:
        return f->x();
      case Camellia::OP_Y:
        return f->y();
      case Camellia::OP_Z:
        return f->z();
      case Camellia::OP_GRAD:
        return f->grad();
      case Camellia::OP_DIV:
        return f->div();
      case Camellia::OP_DOT_NORMAL:
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
    Intrepid::FieldContainer<double> diffs_squared(numCells);
    (diff*diff)->integrate(diffs_squared, basisCacheForCellsToCompare);
    double sum = 0;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      sum += diffs_squared[cellIndex];
    }
    return sqrt(sum) < tol;
  }

  double Function::evaluate(MeshPtr mesh, double x) {
    int spaceDim = 1;
    Intrepid::FieldContainer<double> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires mesh to be 1D if only x is provided.");
    }

    physPoint(0,0) = x;

    vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
    if (cellIDs.size() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
    }
    GlobalIndexType cellID = cellIDs[0];
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    Intrepid::FieldContainer<double> refPoint(1,1,spaceDim);
    physPoint.resize(1,1,spaceDim);
    CamelliaCellTools::mapToReferenceFrame(refPoint, physPoint, mesh->getTopology(), cellID, basisCache->cubatureDegree());
    refPoint.resize(1,spaceDim);

    basisCache->setRefCellPoints(refPoint);

    this->values(value,basisCache);
    return value[0];
  }

  double Function::evaluate(MeshPtr mesh, double x, double y) {
    int spaceDim = 2;
    Intrepid::FieldContainer<double> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != spaceDim) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires mesh to be 2D if (x,y) is provided.");
    }

    physPoint(0,0) = x;
    physPoint(0,1) = y;

    vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
    if (cellIDs.size() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
    }
    GlobalIndexType cellID = cellIDs[0];
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    Intrepid::FieldContainer<double> refPoint(1,1,spaceDim);
    physPoint.resize(1,1,spaceDim);
    CamelliaCellTools::mapToReferenceFrame(refPoint, physPoint, mesh->getTopology(), cellID, basisCache->cubatureDegree());
    refPoint.resize(1,spaceDim);

    basisCache->setRefCellPoints(refPoint);

    this->values(value,basisCache);
    return value[0];
  }

  double Function::evaluate(MeshPtr mesh, double x, double y, double z) {
    int spaceDim = 3;
    Intrepid::FieldContainer<double> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != spaceDim) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires mesh to be 3D if (x,y,z) is provided.");
    }

    physPoint(0,0) = x;
    physPoint(0,1) = y;
    physPoint(0,2) = z;

    vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
    if (cellIDs.size() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
    }
    GlobalIndexType cellID = cellIDs[0];
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    Intrepid::FieldContainer<double> refPoint(1,1,spaceDim);

    physPoint.resize(1,1,spaceDim);
    CamelliaCellTools::mapToReferenceFrame(refPoint, physPoint, mesh->getTopology(), cellID, basisCache->cubatureDegree());
    refPoint.resize(1,spaceDim);
    basisCache->setRefCellPoints(refPoint);

    this->values(value,basisCache);
    return value[0];
  }

  double Function::evaluate(double x) {
    static Intrepid::FieldContainer<double> value(1,1); // (C,P)
    static Intrepid::FieldContainer<double> physPoint(1,1,1);

    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 0 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  double Function::evaluate(FunctionPtr f, double x) {
    return f->evaluate(x);
  }

  double Function::evaluate(double x, double y) {
    static Intrepid::FieldContainer<double> value(1,1);
    static Intrepid::FieldContainer<double> physPoint(1,1,2);
    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 0 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  double Function::evaluate(FunctionPtr f, double x, double y) { // for testing; this isn't super-efficient
    return f->evaluate(x, y);
  }

  double Function::evaluate(double x, double y, double z) { // for testing; this isn't super-efficient
    static Intrepid::FieldContainer<double> value(1,1);
    static Intrepid::FieldContainer<double> physPoint(1,1,3);
    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
    dummyCache->writablePhysicalCubaturePoints()(0,0,2) = z;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  double Function::evaluate(FunctionPtr f, double x, double y, double z) { // for testing; this isn't super-efficient
    return f->evaluate(x,y,z);
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
  FunctionPtr Function::t() {
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
  FunctionPtr Function::dt() {
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
    } else if (numComponents==4) {
      FunctionPtr dtFxn = dt();
      if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL) || (dzFxn.get()==NULL) || (dtFxn.get()==NULL)) {
        return Function::null();
      } else {
        return Teuchos::rcp( new VectorizedFunction(dxFxn,dyFxn,dzFxn,dtFxn) );
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported numComponents");
    return Teuchos::rcp((Function*) NULL);
  }
  //FunctionPtr Function::inverse() {
  //  return Function::null();
  //}

  FunctionPtr Function::heaviside(double xShift) {
    return Teuchos::rcp( new HeavisideFunction(xShift) );
  }

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

  void Function::CHECK_VALUES_RANK(Intrepid::FieldContainer<double> &values) { // throws exception on bad values rank
    // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
    if (values.rank() != _rank + 2) {
      cout << "values has incorrect rank.\n";
      TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
    }
  }

  void Function::addToValues(Intrepid::FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(valuesToAddTo);
    Teuchos::Array<int> dim;
    valuesToAddTo.dimensions(dim);
    Intrepid::FieldContainer<double> myValues(dim);
    this->values(myValues,basisCache);
    for (int i=0; i<myValues.size(); i++) {
      //cout << "otherValue = " << valuesToAddTo[i] << "; myValue = " << myValues[i] << endl;
      valuesToAddTo[i] += myValues[i];
    }
  }

  double Function::integrate(BasisCachePtr basisCache) {
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    Intrepid::FieldContainer<double> cellIntegrals(numCells);
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
    Intrepid::FieldContainer<double> fxnValues(numCells,numPoints);
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
        Intrepid::FieldContainer<double> origPts = basisCache->getRefCellPoints();
        int numPts1D = ceil(sqrt(origPts.dimension(0)));
        int numPts = numPts1D*numPts1D;
        Intrepid::FieldContainer<double> uniformSpacedPts(numPts,origPts.dimension(1));
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
    Intrepid::FieldContainer<double> cellIntegral(1);
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
    Intrepid::FieldContainer<double> integrals(numCells);
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
        Intrepid::FieldContainer<double> nodes = subCellsToCheck[i].subCellNodes;
        BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,mesh));
        int cubEnrich = 2; // arbitrary
        BasisCachePtr enrichedCache =  Teuchos::rcp(new BasisCache(elemType,mesh,testVsTest,cubEnrich));
        vector<GlobalIndexType> cellIDs;
        cellIDs.push_back(cellID);
        basisCache->setPhysicalCellNodes(nodes,cellIDs,true);
        enrichedCache->setPhysicalCellNodes(nodes,cellIDs,true);

        // calculate relative error for this subcell
        Intrepid::FieldContainer<double> cellIntegral(1),enrichedCellIntegral(1);
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
        if (newCacheInfo.elemType->cellTopoPtr->getTensorialDegree() > 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorial degree > 0 not supported here.");
        }
        unsigned cellTopoKey = newCacheInfo.elemType->cellTopoPtr->getKey().first;
        switch (cellTopoKey)
        {
          case shards::Quadrilateral<4>::key:
            {
              // break into 4 subcells
              int spaceDim = 2; int numCells = 1; // cell-by-cell

              Intrepid::FieldContainer<double> oldNodes = newCacheInfo.subCellNodes;
              oldNodes.resize(4,spaceDim);
              Intrepid::FieldContainer<double> newCellNodes(numCells,4,spaceDim);
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

  void Function::integrate(Intrepid::FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache,
                           bool sumInto) {
    TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only integrate scalar functions.");
    int numCells = cellIntegrals.dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  //  cout << "integrate: basisCache->getPhysicalCubaturePoints():\n" << basisCache->getPhysicalCubaturePoints();
    Intrepid::FieldContainer<double> values(numCells,numPoints);
    this->values(values,basisCache);
    if ( !sumInto ) {
      cellIntegrals.initialize(0);
    }

    Intrepid::FieldContainer<double> *weightedMeasures = &basisCache->getWeightedMeasures();
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        cellIntegrals(cellIndex) += values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex);
      }
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
    if (mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex)) {
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
    Intrepid::FieldContainer<double> cellIntegral(1);
    this->integrate(cellIntegral, neighborCache->getSideBasisCache(neighborSideIndex), true);
  //  cout << "Neighbor integral: " << cellIntegral[0] << endl;
    cellIntegral[0] *= -1;
    this->integrate(cellIntegral, myCache->getSideBasisCache(sideIndex), true);
  //  cout << "integral difference: " << cellIntegral[0] << endl;

    // multiply by sideParity to make jump uniquely valued.
    return sideParity * cellIntegral(0);
  }

  double Function::integrate(MeshPtr mesh, int cubatureDegreeEnrichment, bool testVsTest, bool requireSideCache,
                             bool spatialSidesOnly) {
    double integral = 0;

    set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, *cellIDIt, testVsTest, cubatureDegreeEnrichment);

      if ( this->boundaryValueOnly() ) {
        ElementTypePtr elemType = mesh->getElementType(*cellIDIt);
        int numSides = elemType->cellTopoPtr->getSideCount();

        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
          if (spatialSidesOnly && !elemType->cellTopoPtr->sideIsSpatial(sideOrdinal)) continue; // skip non-spatial sides if spatialSidesOnly is true
          double sideIntegral = this->integrate(basisCache->getSideBasisCache(sideOrdinal));
          integral += sideIntegral;
        }
      } else {
        integral += this->integrate(basisCache);
      }
    }
    return MPIWrapper::sum(integral);
  }

  double Function::l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool spatialSidesOnly) {
    FunctionPtr thisPtr = Teuchos::rcp( this, false );
    bool testVsTest = false, requireSideCaches = false;
    return sqrt( (thisPtr * thisPtr)->integrate(mesh, cubatureDegreeEnrichment, testVsTest, requireSideCaches, spatialSidesOnly) );
  }

  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  void Function::scalarMultiplyFunctionValues(Intrepid::FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
    // functionValues has dimensions (C,P,...)
    scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
  }

  // divide values by this function (supported only when this is a scalar)
  void Function::scalarDivideFunctionValues(Intrepid::FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
    // functionValues has dimensions (C,P,...)
    scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
  }

  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  void Function::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    // basisValues has dimensions (C,F,P,...)
  //  cout << "scalarMultiplyBasisValues: basisValues:\n" << basisValues;
    scalarModifyBasisValues(basisValues,basisCache,MULTIPLY);
  //  cout << "scalarMultiplyBasisValues: modified basisValues:\n" << basisValues;
  }

  // divide values by this function (supported only when this is a scalar)
  void Function::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    // basisValues has dimensions (C,F,P,...)
    scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
  }

  void Function::valuesDottedWithTensor(Intrepid::FieldContainer<double> &values,
                                        FunctionPtr tensorFunctionOfLikeRank,
                                        BasisCachePtr basisCache) {
    TEUCHOS_TEST_FOR_EXCEPTION( _rank != tensorFunctionOfLikeRank->rank(),std::invalid_argument,
                       "Can't dot functions of unlike rank");
    TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != 2, std::invalid_argument,
                       "values container should have size (numCells, numPoints" );
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = basisCache->getSpaceDim();

    values.initialize(0.0);

    Teuchos::Array<int> tensorValueIndex(_rank+2); // +2 for numCells, numPoints indices
    tensorValueIndex[0] = numCells;
    tensorValueIndex[1] = numPoints;
    for (int d=0; d<_rank; d++) {
      tensorValueIndex[d+2] = spaceDim;
    }

    Intrepid::FieldContainer<double> myTensorValues(tensorValueIndex);
    this->values(myTensorValues,basisCache);
    Intrepid::FieldContainer<double> otherTensorValues(tensorValueIndex);
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

  void Function::scalarModifyFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                                            FunctionModificationType modType) {
    TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = basisCache->getSpaceDim();

    Intrepid::FieldContainer<double> scalarValues(numCells,numPoints);
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

  void Function::scalarModifyBasisValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                                         FunctionModificationType modType) {
    TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyBasisValues only supported for scalar functions" );
    int numCells = values.dimension(0);
    int numFields = values.dimension(1);
    int numPoints = values.dimension(2);

    int spaceDim = basisCache->getSpaceDim();

    Intrepid::FieldContainer<double> scalarValues(numCells,numPoints);
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
    ofstream fout(filePath.c_str());
    fout << setprecision(15);
    vector< ElementTypePtr > elementTypes = mesh->elementTypes();
    vector< ElementTypePtr >::iterator elemTypeIt;
    int spaceDim = 2; // TODO: generalize to 3D...

    BasisCachePtr basisCache;
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
      ElementTypePtr elemTypePtr = *(elemTypeIt);
      basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh, true) );
      CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
      int numSides = cellTopo->getSideCount();

      Intrepid::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
      int numCells = physicalCellNodes.dimension(0);
      // determine cellIDs
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        GlobalIndexType cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
        cellIDs.push_back(cellID);
      }
      basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

      int num1DPts = 15;
      Intrepid::FieldContainer<double> refPoints(num1DPts,1);
      for (int i=0; i < num1DPts; i++){
        double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
        refPoints(i,0) = x;
      }

      for (int sideIndex=0; sideIndex < numSides; sideIndex++){
        BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
        sideBasisCache->setRefCellPoints(refPoints);
        int numCubPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);


        Intrepid::FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
        this->values(computedValues,sideBasisCache);

        // NOW loop over all cells to write solution to file
        for (int cellIndex=0;cellIndex < numCells;cellIndex++){
          Intrepid::FieldContainer<double> cellParities = mesh->cellSideParitiesForCell( cellIDs[cellIndex] );
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
    ofstream fout(filePath.c_str());
    fout << setprecision(15);
    int spaceDim = 2; // TODO: generalize to 3D...
    int num1DPts = 15;

    int numPoints = num1DPts * num1DPts;
    Intrepid::FieldContainer<double> refPoints(numPoints,spaceDim);
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

      Intrepid::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
      int numCells = physicalCellNodes.dimension(0);
      // determine cellIDs
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
        cellIDs.push_back(cellID);
      }
      basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, false); // false: don't create side cache

      Intrepid::FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();

      Intrepid::FieldContainer<double> computedValues(numCells,numPoints);
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

  FunctionPtr Function::normal_1D() { // unit outward-facing normal on each element boundary
    static FunctionPtr _normal_1D = Teuchos::rcp( new UnitNormalFunction(0) );
    return _normal_1D;
  }

  FunctionPtr Function::normalSpaceTime() { // unit outward-facing normal on each element boundary
    static FunctionPtr _normalSpaceTime = Teuchos::rcp( new UnitNormalFunction(-1,true) );
    return _normalSpaceTime;
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

  FunctionPtr Function::tn(int n) {
    return Teuchos::rcp( new Tn(n) );
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

  // this is liable to be a bit slow!!
  class ComposedFunction : public Function {
    FunctionPtr _f, _arg_g;
  public:
    ComposedFunction(FunctionPtr f, FunctionPtr arg_g) : Function(f->rank()) {
      _f = f;
      _arg_g = arg_g;
    }
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      CHECK_VALUES_RANK(values);
      int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      int spaceDim = basisCache->getSpaceDim();
      Intrepid::FieldContainer<double> fArgPoints(numCells,numPoints,spaceDim);
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
  //  cout << "WARNING: Function::composedFunction() called, but its implementation is not yet complete.\n";
    return Teuchos::rcp( new ComposedFunction(f,arg_g) );
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

  FunctionPtr Function::min(FunctionPtr f1, FunctionPtr f2) {
    return Teuchos::rcp( new MinFunction(f1, f2) );
  }

  FunctionPtr Function::min(FunctionPtr f1, double value) {
    return Teuchos::rcp( new MinFunction(f1, Function::constant(value)) );
  }

  FunctionPtr Function::min(double value, FunctionPtr f2) {
    return Teuchos::rcp( new MinFunction(f2, Function::constant(value)) );
  }

  FunctionPtr Function::max(FunctionPtr f1, FunctionPtr f2) {
    return Teuchos::rcp( new MaxFunction(f1, f2) );
  }

  FunctionPtr Function::max(FunctionPtr f1, double value) {
    return Teuchos::rcp( new MaxFunction(f1, Function::constant(value)) );
  }

  FunctionPtr Function::max(double value, FunctionPtr f2) {
    return Teuchos::rcp( new MaxFunction(f2, Function::constant(value)) );
  }
}
