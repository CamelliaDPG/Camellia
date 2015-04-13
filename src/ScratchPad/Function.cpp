//
//  Function.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "TypeDefs.h"

#include "Function.h"
#include "BasisCache.h"
#include "ExactSolution.h"
#include "Mesh.h"
#include "Teuchos_GlobalMPISession.hpp"
#include "MPIWrapper.h"
#include "CellCharacteristicFunction.h"

#include "Var.h"
#include "Solution.h"

#include "GlobalDofAssignment.h"

#include "PhysicalPointCache.h"

#include "CamelliaCellTools.h"

#include "Intrepid_CellTools.hpp"

// using namespace Intrepid;
// using namespace Camellia;

namespace Camellia {
  // for adaptive quadrature
  struct CacheInfo {
    ElementTypePtr elemType;
    GlobalIndexType cellID;
    Intrepid::FieldContainer<double> subCellNodes;
  };

  // private class ComponentFunction
  template <typename Scalar>
  class ComponentFunction : public Function<Scalar> {
    Teuchos::RCP<Function<Scalar> > _vectorFxn;
    int _component;
    public:
    ComponentFunction(Teuchos::RCP<Function<Scalar> > vectorFunction, int componentIndex) {
      _vectorFxn = vectorFunction;
      _component = componentIndex;
      if (_vectorFxn->rank() < 1) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vector function must have rank 1 or greater");
      }
    }
    bool boundaryValueOnly() {
      return _vectorFxn->boundaryValueOnly();
    }
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
      // note this allocation.  There might be ways of reusing memory here, if we had a slightly richer API.
      int spaceDim = basisCache->getSpaceDim();
      Teuchos::Array<int> dim;
      values.dimensions(dim);
      dim.push_back(spaceDim);

      Intrepid::FieldContainer<Scalar> vectorValues(dim);
      _vectorFxn->values(vectorValues, basisCache);

      int numValues = values.size();
      for (int i=0; i<numValues; i++) {
        values[i] = vectorValues[spaceDim*i + _component];
      }
    }
  };

  // private class CellBoundaryRestrictedFunction
  template <typename Scalar>
  class CellBoundaryRestrictedFunction : public Function<Scalar> {
    Teuchos::RCP<Function<Scalar> > _fxn;
    public:
    CellBoundaryRestrictedFunction(Teuchos::RCP<Function<Scalar> > fxn) : Function<Scalar>(fxn->rank()) {
      _fxn = fxn;
    }

    bool boundaryValueOnly() { return true; }
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
      _fxn->values(values, basisCache);
    }
  };

  class HeavisideFunction : public SimpleFunction<double> {
    double _xShift;
    public:
    HeavisideFunction(double xShift=0.0) {
      _xShift = xShift;
    }
    double value(double x) {
      return (x < _xShift) ? 0.0 : 1.0;
    }
  };

  class MeshBoundaryCharacteristicFunction : public Function<double> {
    public:
      MeshBoundaryCharacteristicFunction() : Function<double>(0) {}
      bool boundaryValueOnly() { return true; }
      void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
        this->CHECK_VALUES_RANK(values);
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
      Teuchos::RCP<Function<double> > dx() {
        return Function<double>::zero();
      }
      Teuchos::RCP<Function<double> > dy() {
        return Function<double>::zero();
      }
      //  Teuchos::RCP<Function<double> > dz() {
      //    return Function<double>::zero();
      //  }
  };

  class MeshSkeletonCharacteristicFunction : public ConstantScalarFunction<double> {
    public:
      MeshSkeletonCharacteristicFunction() : ConstantScalarFunction(1, "|_{\\Gamma_h}") {
      }
      bool boundaryValueOnly() { return true; }
  };

  // private class SimpleSolutionFunction:
  template <typename Scalar>
  class SimpleSolutionFunction : public Function<Scalar> {
    Teuchos::RCP<Solution<Scalar> > _soln;
    VarPtr _var;
    public:
    SimpleSolutionFunction(VarPtr var, Teuchos::RCP<Solution<Scalar> > soln);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    Teuchos::RCP<Function<Scalar> > x();
    Teuchos::RCP<Function<Scalar> > y();
    Teuchos::RCP<Function<Scalar> > z();

    Teuchos::RCP<Function<Scalar> > dx();
    Teuchos::RCP<Function<Scalar> > dy();
    Teuchos::RCP<Function<Scalar> > dz();
    // for reasons of efficiency, may want to implement div() and grad() as well

    void importCellData(std::vector<GlobalIndexType> cellIDs);

    string displayString();
    bool boundaryValueOnly();
  };

  template <typename Scalar>
  Function<Scalar>::Function() {
    _rank = 0;
    _displayString = this->displayString();
    _time = 0;
  }
  template <typename Scalar>
  Function<Scalar>::Function(int rank) {
    _rank = rank;
    _displayString = this->displayString();
    _time = 0;
  }

  template <typename Scalar>
  string Function<Scalar>::displayString() {
    return "f";
  }

  template <typename Scalar>
  int Function<Scalar>::rank() {
    return _rank;
  }

  template <typename Scalar>
  void Function<Scalar>::setTime(double time)
  {
    _time = time;
  }

  template <typename Scalar>
  double Function<Scalar>::getTime()
  {
    return _time;
  }

  template <typename Scalar>
  void Function<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache) {
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

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::op(Teuchos::RCP<Function<Scalar> > f, Camellia::EOperator op) {
    if ( Function<Scalar>::isNull(f) ) {
      return Function<Scalar>::null();
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
        return f * Function<Scalar>::normal();
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
        break;
    }
    return Teuchos::rcp((Function<Scalar>*)NULL);
  }

  template <typename Scalar>
  bool Function<Scalar>::equals(Teuchos::RCP<Function<Scalar> > f, BasisCachePtr basisCacheForCellsToCompare, double tol) {
    if (f->rank() != this->rank()) {
      return false;
    }
    Teuchos::RCP<Function<Scalar> > thisPtr = Teuchos::rcp(this,false);
    Teuchos::RCP<Function<Scalar> > diff = thisPtr-f;

    int numCells = basisCacheForCellsToCompare->getPhysicalCubaturePoints().dimension(0);
    // compute L^2 norm of difference on the cells
    Intrepid::FieldContainer<Scalar> diffs_squared(numCells);
    (diff*diff)->integrate(diffs_squared, basisCacheForCellsToCompare);
    Scalar sum = 0;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      sum += diffs_squared[cellIndex];
    }
    return sqrt(abs(sum)) < tol;
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(MeshPtr mesh, double x) {
    int spaceDim = 1;
    Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires mesh to be 1D if only x is provided.");
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

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(MeshPtr mesh, double x, double y) {
    int spaceDim = 2;
    Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != spaceDim) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires mesh to be 2D if (x,y) is provided.");
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

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(MeshPtr mesh, double x, double y, double z) {
    int spaceDim = 3;
    Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
    Intrepid::FieldContainer<double> physPoint(1,spaceDim);

    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 0 Function.");
    }
    if (mesh->getTopology()->getSpaceDim() != spaceDim) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires mesh to be 3D if (x,y,z) is provided.");
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

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(double x) {
    static Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
    static Intrepid::FieldContainer<double> physPoint(1,1,1);

    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 0 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(Teuchos::RCP<Function<Scalar> > f, double x) {
    return f->evaluate(x);
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(double x, double y) {
    static Intrepid::FieldContainer<Scalar> value(1,1);
    static Intrepid::FieldContainer<double> physPoint(1,1,2);
    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 0 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(Teuchos::RCP<Function<Scalar> > f, double x, double y) { // for testing; this isn't super-efficient
    return f->evaluate(x, y);
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(double x, double y, double z) { // for testing; this isn't super-efficient
    static Intrepid::FieldContainer<Scalar> value(1,1);
    static Intrepid::FieldContainer<double> physPoint(1,1,3);
    static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
    dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
    dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
    dummyCache->writablePhysicalCubaturePoints()(0,0,2) = z;
    if (this->rank() != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function<Scalar>::evaluate requires a rank 1 Function.");
    }
    this->values(value,dummyCache);
    return value[0];
  }

  template <typename Scalar>
  Scalar Function<Scalar>::evaluate(Teuchos::RCP<Function<Scalar> > f, double x, double y, double z) { // for testing; this isn't super-efficient
    return f->evaluate(x,y,z);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::x() {
    return Function<Scalar>::null();
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::y() {
    return Function<Scalar>::null();
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::z() {
    return Function<Scalar>::null();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::dx() {
    return Function<Scalar>::null();
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::dy() {
    return Function<Scalar>::null();
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::dz() {
    return Function<Scalar>::null();
  }
  // TODO: rework ParametricCurve (Function subclass) so that we can define dt() thus.
  //Teuchos::RCP<Function<Scalar> > Function<Scalar>::dt() {
  //  return Function<Scalar>::null();
  //}
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::curl() {
    Teuchos::RCP<Function<Scalar> > dxFxn = dx();
    Teuchos::RCP<Function<Scalar> > dyFxn = dy();
    Teuchos::RCP<Function<Scalar> > dzFxn = dz();

    if (dxFxn.get()==NULL) {
      return Function<Scalar>::null();
    } else if (dyFxn.get()==NULL) {
      // special case: in 1D, curl() returns a scalar
      return dxFxn;
    } else if (dzFxn.get() == NULL) {
      // in 2D, the rank of the curl operator depends on the rank of the Function
      if (_rank == 0) {
        return Teuchos::rcp( new VectorizedFunction<Scalar>(dyFxn,-dxFxn) );
      } else if (_rank == 1) {
        return dyFxn->x() - dxFxn->y();
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "curl() undefined for Functions of rank > 1");
      }
    } else {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dyFxn->z() - dzFxn->y(),
            dzFxn->x() - dxFxn->z(),
            dxFxn->y() - dyFxn->x()) );
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::grad(int numComponents) {
    Teuchos::RCP<Function<Scalar> > dxFxn = dx();
    Teuchos::RCP<Function<Scalar> > dyFxn = dy();
    Teuchos::RCP<Function<Scalar> > dzFxn = dz();
    if (numComponents==-1) { // default: just use as many non-null components as available
      if (dxFxn.get()==NULL) {
        return Function<Scalar>::null();
      } else if (dyFxn.get()==NULL) {
        // special case: in 1D, grad() returns a scalar
        return dxFxn;
      } else if (dzFxn.get() == NULL) {
        return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn) );
      } else {
        return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn,dzFxn) );
      }
    } else if (numComponents==1) {
      // special case: we don't "vectorize" in 1D
      return dxFxn;
    } else if (numComponents==2) {
      if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL)) {
        return Function<Scalar>::null();
      } else {
        return Function<Scalar>::vectorize(dxFxn, dyFxn);
      }
    } else if (numComponents==3) {
      if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL) || (dzFxn.get()==NULL)) {
        return Function<Scalar>::null();
      } else {
        return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn,dzFxn) );
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported numComponents");
    return Teuchos::rcp((Function<Scalar>*) NULL);
  }
  //Teuchos::RCP<Function<Scalar> > Function<Scalar>::inverse() {
  //  return Function<Scalar>::null();
  //}

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::heaviside(double xShift) {
    return Teuchos::rcp( new HeavisideFunction(xShift) );
  }

  template <typename Scalar>
  bool Function<Scalar>::isNull(Teuchos::RCP<Function<Scalar> > f) {
    return f.get() == NULL;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::div() {
    if ( Function<Scalar>::isNull(x()) || Function<Scalar>::isNull(y()) ) {
      return Function<double>::null();
    }
    Teuchos::RCP<Function<Scalar> > dxFxn = x()->dx();
    Teuchos::RCP<Function<Scalar> > dyFxn = y()->dy();
    Teuchos::RCP<Function<Scalar> > zFxn = z();
    if ( Function<Scalar>::isNull(dxFxn) || Function<Scalar>::isNull(dyFxn) ) {
      return Function<double>::null();
    } else if ( Function<Scalar>::isNull(zFxn) || Function<Scalar>::isNull(zFxn->dz()) ) {
      return dxFxn + dyFxn;
    } else {
      return dxFxn + dyFxn + zFxn->dz();
    }
  }

  template <typename Scalar>
  void Function<Scalar>::CHECK_VALUES_RANK(Intrepid::FieldContainer<Scalar> &values) { // throws exception on bad values rank
    // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
    if (values.rank() != _rank + 2) {
      cout << "values has incorrect rank.\n";
      TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
    }
  }

  template <typename Scalar>
  void Function<Scalar>::addToValues(Intrepid::FieldContainer<Scalar> &valuesToAddTo, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(valuesToAddTo);
    Teuchos::Array<int> dim;
    valuesToAddTo.dimensions(dim);
    Intrepid::FieldContainer<Scalar> myValues(dim);
    this->values(myValues,basisCache);
    for (int i=0; i<myValues.size(); i++) {
      //cout << "otherValue = " << valuesToAddTo[i] << "; myValue = " << myValues[i] << endl;
      valuesToAddTo[i] += myValues[i];
    }
  }

  template <typename Scalar>
  Scalar Function<Scalar>::integrate(BasisCachePtr basisCache) {
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    Intrepid::FieldContainer<Scalar> cellIntegrals(numCells);
    this->integrate(cellIntegrals, basisCache);
    Scalar sum = 0;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      sum += cellIntegrals[cellIndex];
    }
    return sum;
  }

  // added by Jesse to check positivity of a function
  template <typename Scalar>
  bool Function<Scalar>::isPositive(BasisCachePtr basisCache){
    // TEUCHOS_TEST_FOR_EXCEPTION(typeof(Scalar) != double, std::invalid_argument, "values has incorrect rank." );
    bool isPositive = true;
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    Intrepid::FieldContainer<Scalar> fxnValues(numCells,numPoints);
    this->values(fxnValues, basisCache);

    for (int i = 0;i<fxnValues.size();i++){
      if (fxnValues[i] <= 0.0){
        isPositive=false;
        break;
      }
    }
    return isPositive;
  }

  template <typename Scalar>
  bool Function<Scalar>::isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich, bool testVsTest){
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
  template <typename Scalar>
  Scalar Function<Scalar>::integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID,testVsTest,cubatureDegreeEnrichment);
    Intrepid::FieldContainer<Scalar> cellIntegral(1);
    this->integrate(cellIntegral,basisCache);
    return cellIntegral(0);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::cellCharacteristic(GlobalIndexType cellID) {
    return Teuchos::rcp( new CellCharacteristicFunction(cellID) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::cellCharacteristic(set<GlobalIndexType> cellIDs) {
    return Teuchos::rcp( new CellCharacteristicFunction(cellIDs) );
  }

  template <typename Scalar>
  map<int, Scalar> Function<Scalar>::cellIntegrals(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    vector<GlobalIndexType> cellIDs(activeCellIDs.begin(),activeCellIDs.end());
    return cellIntegrals(cellIDs,mesh,cubatureDegreeEnrichment,testVsTest);
  }

  template <typename Scalar>
  map<int, Scalar> Function<Scalar>::cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest){
    int myPartition = Teuchos::GlobalMPISession::getRank();

    int numCells = cellIDs.size();
    Intrepid::FieldContainer<Scalar> integrals(numCells);
    for (int i = 0;i<numCells;i++){
      int cellID = cellIDs[i];
      if (mesh->partitionForCellID(cellID) == myPartition){
        integrals(i) = integrate(cellID,mesh,cubatureDegreeEnrichment,testVsTest);
      }
    }
    MPIWrapper::entryWiseSum(integrals);
    map<int,Scalar> integralMap;
    for (int i = 0;i<numCells;i++){
      integralMap[cellIDs[i]] = integrals(i);
    }
    return integralMap;
  }


  // added by Jesse - adaptive quadrature rules
  template <typename Scalar>
  Scalar Function<Scalar>::integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest) {
    Scalar integral = 0.0;
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
        Intrepid::FieldContainer<Scalar> cellIntegral(1),enrichedCellIntegral(1);
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

  template <typename Scalar>
  void Function<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &cellIntegrals, BasisCachePtr basisCache, bool sumInto) {
    TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only integrate scalar functions.");
    int numCells = cellIntegrals.dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    //  cout << "integrate: basisCache->getPhysicalCubaturePoints():\n" << basisCache->getPhysicalCubaturePoints();
    Intrepid::FieldContainer<Scalar> values(numCells,numPoints);
    this->values(values,basisCache);
    if ( !sumInto ) {
      cellIntegrals.initialize(0);
    }

    Intrepid::FieldContainer<double> *weightedMeasures = &basisCache->getWeightedMeasures();
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        cellIntegrals(cellIndex) += values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex);
      }
      //    if ( (basisCache->cellIDs()[cellIndex]==0) && basisCache->isSideCache() && !sumInto)  {
      //      cout << "sideIndex: " << basisCache->getSideIndex() << endl;
      //      cout << "Function<Scalar>::integrate() values:\n";
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
      ////      cout << "Function<Scalar>::integrate() values:\n" << values;
      //      cout << "weightedMeasures:\n" << *weightedMeasures;
      //    }
    }
  }

  // takes integral of jump over entire INTERIOR skeleton
  template <typename Scalar>
  Scalar Function<Scalar>::integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment) {
    Scalar integral = 0.0;
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

  template <typename Scalar>
  Scalar Function<Scalar>::integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment) {
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
    Intrepid::FieldContainer<Scalar> cellIntegral(1);
    this->integrate(cellIntegral, neighborCache->getSideBasisCache(neighborSideIndex), true);
    //  cout << "Neighbor integral: " << cellIntegral[0] << endl;
    cellIntegral[0] *= -1;
    this->integrate(cellIntegral, myCache->getSideBasisCache(sideIndex), true);
    //  cout << "integral difference: " << cellIntegral[0] << endl;

    // multiply by sideParity to make jump uniquely valued.
    return sideParity * cellIntegral(0);
  }

  template <typename Scalar>
  Scalar Function<Scalar>::integrate(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest, bool requireSideCache,
      bool spatialSidesOnly) {
    Scalar integral = 0;

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
      //    cout << "Function<Scalar>::integrate: basisCache has " << basisCache->getPhysicalCubaturePoints().dimension(1) << " cubature points per cell.\n";
      Intrepid::FieldContainer<Scalar> cellIntegrals(numCells);
      if ( this->boundaryValueOnly() ) {
        int numSides = elemType->cellTopoPtr->getSideCount();

        for (int i=0; i<numSides; i++) {
          if (spatialSidesOnly && !elemType->cellTopoPtr->sideIsSpatial(i)) continue; // skip non-spatial sides if spatialSidesOnly is true
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

  // This needs work for complex
  template <typename Scalar>
  double Function<Scalar>::l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool spatialSidesOnly) {
    Teuchos::RCP<Function<Scalar> > thisPtr = Teuchos::rcp( this, false );
    bool testVsTest = false, requireSideCaches = false;
    return sqrt( (thisPtr * thisPtr)->integrate(mesh, cubatureDegreeEnrichment, testVsTest, requireSideCaches, spatialSidesOnly) );
  }

  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  template <typename Scalar>
  void Function<Scalar>::scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache) {
    // functionValues has dimensions (C,P,...)
    scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
  }

  // divide values by this function (supported only when this is a scalar)
  template <typename Scalar>
  void Function<Scalar>::scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache) {
    // functionValues has dimensions (C,P,...)
    scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
  }

  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  template <typename Scalar>
  void Function<Scalar>::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    // basisValues has dimensions (C,F,P,...)
    //  cout << "scalarMultiplyBasisValues: basisValues:\n" << basisValues;
    scalarModifyBasisValues(basisValues,basisCache,MULTIPLY);
    //  cout << "scalarMultiplyBasisValues: modified basisValues:\n" << basisValues;
  }

  // divide values by this function (supported only when this is a scalar)
  template <typename Scalar>
  void Function<Scalar>::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    // basisValues has dimensions (C,F,P,...)
    scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
  }

  template <typename Scalar>
  void Function<Scalar>::valuesDottedWithTensor(Intrepid::FieldContainer<Scalar> &values,
      Teuchos::RCP<Function<Scalar> > tensorFunctionOfLikeRank,
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

    Intrepid::FieldContainer<Scalar> myTensorValues(tensorValueIndex);
    this->values(myTensorValues,basisCache);
    Intrepid::FieldContainer<Scalar> otherTensorValues(tensorValueIndex);
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
        Scalar *myValue = &myTensorValues[ myTensorValues.getEnumeration(tensorValueIndex) ];
        Scalar *otherValue = &otherTensorValues[ otherTensorValues.getEnumeration(tensorValueIndex) ];
        Scalar *value = &values(cellIndex,ptIndex);

        for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
          *value += *myValue * *otherValue;
          //        cout << "myValue: " << *myValue << "; otherValue: " << *otherValue << endl;
          myValue++;
          otherValue++;
        }
      }
    }
  }

  template <typename Scalar>
  void Function<Scalar>::scalarModifyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
      FunctionModificationType modType) {
    TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);

    Intrepid::FieldContainer<Scalar> scalarValues(numCells,numPoints);
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
        Scalar *value = &values[ values.getEnumeration(valueIndex) ];
        Scalar scalarValue = scalarValues(cellIndex,ptIndex);
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

  template <typename Scalar>
  void Function<Scalar>::scalarModifyBasisValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
      FunctionModificationType modType) {
    TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyBasisValues only supported for scalar functions" );
    int numCells = values.dimension(0);
    int numFields = values.dimension(1);
    int numPoints = values.dimension(2);

    int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);

    Intrepid::FieldContainer<Scalar> scalarValues(numCells,numPoints);
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
          Scalar scalarValue = scalarValues(cellIndex,ptIndex);
          Scalar *value = &values[ values.getEnumeration(valueIndex) ];
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

  template <typename Scalar>
  void Function<Scalar>::writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath) {
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


        Intrepid::FieldContainer<Scalar> computedValues(numCells,numCubPoints); // first arg = 1 cell only
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

  template <typename Scalar>
  void Function<Scalar>::writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath) {
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

      Intrepid::FieldContainer<Scalar> computedValues(numCells,numPoints);
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

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::constant(Scalar value) {
    return Teuchos::rcp( new ConstantScalarFunction<Scalar>(value) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::constant(vector<Scalar> &value) {
    return Teuchos::rcp( new ConstantVectorFunction<Scalar>(value) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::meshBoundaryCharacteristic() {
    // 1 on mesh boundary, 0 elsewhere
    return Teuchos::rcp( new MeshBoundaryCharacteristicFunction );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::h() {
    return Teuchos::rcp( new hFunction );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::meshSkeletonCharacteristic() {
    // 1 on mesh skeleton, 0 elsewhere
    return Teuchos::rcp( new MeshSkeletonCharacteristicFunction );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::normal() { // unit outward-facing normal on each element boundary
    static Teuchos::RCP<Function<double> > _normal = Teuchos::rcp( new UnitNormalFunction );
    return _normal;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::normal_1D() { // unit outward-facing normal on each element boundary
    static Teuchos::RCP<Function<double> > _normal_1D = Teuchos::rcp( new UnitNormalFunction(0) );
    return _normal_1D;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::normalSpaceTime() { // unit outward-facing normal on each element boundary
    static Teuchos::RCP<Function<double> > _normalSpaceTime = Teuchos::rcp( new UnitNormalFunction(-1,true) );
    return _normalSpaceTime;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::sideParity() { // canonical direction on boundary (used for defining fluxes)
    static Teuchos::RCP<Function<double> > _sideParity = Teuchos::rcp( new SideParityFunction );
    return _sideParity;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::polarize(Teuchos::RCP<Function<Scalar> > f) {
    return Teuchos::rcp( new PolarizedFunction<Scalar>(f) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::restrictToCellBoundary(Teuchos::RCP<Function<Scalar> > f) {
    return Teuchos::rcp( new CellBoundaryRestrictedFunction<Scalar>(f) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::solution(VarPtr var, SolutionPtr soln) {
    return Teuchos::rcp( new SimpleSolutionFunction<Scalar>(var, soln) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::vectorize(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) {
    return Teuchos::rcp( new VectorizedFunction<Scalar>(f1,f2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::vectorize(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2, Teuchos::RCP<Function<Scalar> > f3) {
    return Teuchos::rcp( new VectorizedFunction<Scalar>(f1,f2,f3) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::null() {
    static Teuchos::RCP<Function<Scalar> > _null = Teuchos::rcp( (Function<Scalar>*) NULL );
    return _null;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::xn(int n) {
    return Teuchos::rcp( new Xn(n) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::yn(int n) {
    return Teuchos::rcp( new Yn(n) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::zn(int n) {
    return Teuchos::rcp( new Zn(n) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::tn(int n) {
    return Teuchos::rcp( new Tn(n) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::xPart(Teuchos::RCP<Function<Scalar> > vectorFxn) {
    return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 0) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::yPart(Teuchos::RCP<Function<Scalar> > vectorFxn) {
    return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 1) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > Function<Scalar>::zPart(Teuchos::RCP<Function<Scalar> > vectorFxn) {
    return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::zero(int rank) {
    static Teuchos::RCP<Function<double> > _zero = Teuchos::rcp( new ConstantScalarFunction<double>(0.0) );
    if (rank==0) {
      return _zero;
    } else {
      Teuchos::RCP<Function<double> > zeroTensor = _zero;
      for (int i=0; i<rank; i++) {
        // THIS ASSUMES 2D--3D would be Function<Scalar>::vectorize(zeroTensor, zeroTensor, zeroTensor)...
        zeroTensor = Function<double>::vectorize(zeroTensor, zeroTensor);
      }
      return zeroTensor;
    }
  }

  template <typename Scalar>
  ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value) {
    _value = value;
    ostringstream valueStream;
    valueStream << value;
    _stringDisplay = valueStream.str();
  }

  template <typename Scalar>
  ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value, string stringDisplay) {
    _value = value;
    _stringDisplay = stringDisplay;
  }

  template <typename Scalar>
  string ConstantScalarFunction<Scalar>::displayString() {
    return _stringDisplay;
  }

  template <typename Scalar>
  bool ConstantScalarFunction<Scalar>::isZero() {
    return 0.0 == _value;
  }

  template <typename Scalar>
  void ConstantScalarFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    for (int i=0; i < values.size(); i++) {
      values[i] = _value;
    }
  }
  template <typename Scalar>
  void ConstantScalarFunction<Scalar>::scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    if (_value != 1.0) {
      for (int i=0; i < values.size(); i++) {
        values[i] *= _value;
      }
    }
  }
  template <typename Scalar>
  void ConstantScalarFunction<Scalar>::scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    if (_value != 1.0) {
      for (int i=0; i < values.size(); i++) {
        values[i] /= _value;
      }
    }
  }
  template <typename Scalar>
  void ConstantScalarFunction<Scalar>::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
    scalarMultiplyFunctionValues(basisValues,basisCache);
  }
  template <typename Scalar>
  void ConstantScalarFunction<Scalar>::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
    scalarDivideFunctionValues(basisValues,basisCache);
  }

  template <typename Scalar>
  Scalar ConstantScalarFunction<Scalar>::value(double x) {
    return value();
  }

  template <typename Scalar>
  Scalar ConstantScalarFunction<Scalar>::value(double x, double y) {
    return value();
  }

  template <typename Scalar>
  Scalar ConstantScalarFunction<Scalar>::value(double x, double y, double z) {
    return value();
  }

  template <typename Scalar>
  Scalar ConstantScalarFunction<Scalar>::value() {
    return _value;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > ConstantScalarFunction<Scalar>::dx() {
    return Function<double>::zero();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > ConstantScalarFunction<Scalar>::dy() {
    return Function<double>::zero();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > ConstantScalarFunction<Scalar>::dz() {
    return Function<double>::zero();
  }

  template <typename Scalar>
  ConstantVectorFunction<Scalar>::ConstantVectorFunction(vector<Scalar> value) : Function<Scalar>(1) {
    _value = value;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ConstantVectorFunction<Scalar>::x() {
    return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[0] ) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ConstantVectorFunction<Scalar>::y() {
    return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[1] ) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ConstantVectorFunction<Scalar>::z() {
    if (_value.size() > 2) {
      return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[2] ) );
    } else {
      return Teuchos::null;
    }
  }

  template <typename Scalar>
  vector<Scalar> ConstantVectorFunction<Scalar>::value() {
    return _value;
  }

  template <typename Scalar>
  bool ConstantVectorFunction<Scalar>::isZero() {
    for (int d=0; d < _value.size(); d++) {
      if (0.0 != _value[d]) {
        return false;
      }
    }
    return true;
  }

  template <typename Scalar>
  void ConstantVectorFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    int spaceDim = values.dimension(2);
    if (spaceDim > _value.size()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim is greater than length of vector...");
    }
    // values are stored in (C,P,D) order, the important thing here being that we can do this:
    for (int i=0; i < values.size(); ) {
      for (int d=0; d < spaceDim; d++) {
        values[i++] = _value[d];
      }
    }
  }

  ExactSolutionFunction::ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID)
    : Function<double>(exactSolution->exactFunctions().find(trialID)->second->rank()) {
      _exactSolution = exactSolution;
      _trialID = trialID;
    }
  void ExactSolutionFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    _exactSolution->solutionValues(values,_trialID,basisCache);
  }

  template <typename Scalar>
  string ProductFunction<Scalar>::displayString() {
    ostringstream ss;
    ss << _f1->displayString() << " \\cdot " << _f2->displayString();
    return ss.str();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::dx() {
    if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply product rule:
    return _f1 * _f2->dx() + _f2 * _f1->dx();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::dy() {
    if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply product rule:
    return _f1 * _f2->dy() + _f2 * _f1->dy();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::dz() {
    if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply product rule:
    return _f1 * _f2->dz() + _f2 * _f1->dz();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::x() {
    if (this->rank() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take x component of scalar function.");
    }
    // otherwise, _f2 is the rank > 0 function
    if (Function<Scalar>::isNull(_f2->x())) {
      return Function<double>::null();
    }
    return _f1 * _f2->x();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::y() {
    if (this->rank() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take y component of scalar function.");
    }
    // otherwise, _f2 is the rank > 0 function
    if (Function<Scalar>::isNull(_f2->y())) {
      return Function<double>::null();
    }
    return _f1 * _f2->y();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > ProductFunction<Scalar>::z() {
    if (this->rank() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take z component of scalar function.");
    }
    // otherwise, _f2 is the rank > 0 function
    if (Function<Scalar>::isNull(_f2->z())) {
      return Function<double>::null();
    }
    return _f1 * _f2->z();
  }

  template <typename Scalar>
  int ProductFunction<Scalar>::productRank(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) {
    if (f1->rank() == f2->rank()) return 0;
    if (f1->rank() == 0) return f2->rank();
    if (f2->rank() == 0) return f1->rank();
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
    return -1;
  }

  template <typename Scalar>
  ProductFunction<Scalar>::ProductFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) : Function<Scalar>( productRank(f1,f2) ) {
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

  template <typename Scalar>
  bool ProductFunction<Scalar>::boundaryValueOnly() {
    return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
  }

  template <typename Scalar>
  void ProductFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    if (( _f2->rank() > 0) && (this->rank() == 0)) { // tensor product resulting in scalar value
      _f2->valuesDottedWithTensor(values, _f1, basisCache);
    } else { // scalar multiplication by f1, then
      _f2->values(values,basisCache);
      _f1->scalarMultiplyFunctionValues(values, basisCache);
    }
  }

  template <typename Scalar>
  QuotientFunction<Scalar>::QuotientFunction(Teuchos::RCP<Function<Scalar> > f, Teuchos::RCP<Function<Scalar> > scalarDivisor) : Function<Scalar>( f->rank() ) {
    if ( scalarDivisor->rank() != 0 ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
    }
    _f = f;
    _scalarDivisor = scalarDivisor;
    if (scalarDivisor->isZero()) {
      cout << "WARNING: division by zero in QuotientFunction.\n";
    }
  }

  template <typename Scalar>
  bool QuotientFunction<Scalar>::boundaryValueOnly() {
    return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
  }

  template <typename Scalar>
  string QuotientFunction<Scalar>::displayString() {
    ostringstream ss;
    ss << _f->displayString() << " / " << _scalarDivisor->displayString();
    return ss.str();
  }

  template <typename Scalar>
  void QuotientFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);
    _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > QuotientFunction<Scalar>::dx() {
    if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply quotient rule:
    return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > QuotientFunction<Scalar>::dy() {
    if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply quotient rule:
    return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > QuotientFunction<Scalar>::dz() {
    if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) ) {
      return Function<double>::null();
    }
    // otherwise, apply quotient rule:
    return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
  }

  template <typename Scalar>
  SumFunction<Scalar>::SumFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) : Function<Scalar>(f1->rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
    _f1 = f1;
    _f2 = f2;
  }

  template <typename Scalar>
  bool SumFunction<Scalar>::boundaryValueOnly() {
    // if either summand is BVO, then so is the sum...
    return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
  }

  template <typename Scalar>
  string SumFunction<Scalar>::displayString() {
    ostringstream ss;
    ss << "(" << _f1->displayString() << " + " << _f2->displayString() << ")";
    return ss.str();
  }

  template <typename Scalar>
  void SumFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    _f1->values(values,basisCache);
    _f2->addToValues(values,basisCache);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::x() {
    if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->x() + _f2->x();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::y() {
    if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->y() + _f2->y();
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::z() {
    if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->z() + _f2->z();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::dx() {
    if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->dx() + _f2->dx();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::dy() {
    if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->dy() + _f2->dy();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::dz() {
    if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) ) {
      return Function<double>::null();
    }
    return _f1->dz() + _f2->dz();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::grad(int numComponents) {
    if ( Function<Scalar>::isNull(_f1->grad(numComponents)) || Function<Scalar>::isNull(_f2->grad(numComponents)) ) {
      return Function<double>::null();
    } else {
      return _f1->grad(numComponents) + _f2->grad(numComponents);
    }
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SumFunction<Scalar>::div() {
    if ( Function<Scalar>::isNull(_f1->div()) || Function<Scalar>::isNull(_f2->div()) ) {
      return Function<double>::null();
    } else {
      return _f1->div() + _f2->div();
    }
  }

  MinFunction::MinFunction(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2) : Function<double>(f1->rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
    _f1 = f1;
    _f2 = f2;
  }

  bool MinFunction::boundaryValueOnly() {
    // if either summand is BVO, then so is the min...
    return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
  }

  string MinFunction::displayString() {
    ostringstream ss;
    ss << "min( " << _f1->displayString() << " , " << _f2->displayString() << " )";
    return ss.str();
  }

  void MinFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    Intrepid::FieldContainer<double> values2(values);
    _f1->values(values,basisCache);
    _f2->values(values2,basisCache);
    for(int i = 0; i < values.size(); i++) {
      values[i] = std::min(values[i],values2[i]);
    }
  }

  Teuchos::RCP<Function<double> > MinFunction::x() {
    if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::min(_f1->x(),_f2->x());
  }

  Teuchos::RCP<Function<double> > MinFunction::y() {
    if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::min(_f1->y(),_f2->y());
  }

  Teuchos::RCP<Function<double> > MinFunction::z() {
    if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::min(_f1->z(),_f2->z());
  }

  MaxFunction::MaxFunction(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2) : Function<double>(f1->rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
    _f1 = f1;
    _f2 = f2;
  }

  bool MaxFunction::boundaryValueOnly() {
    // if either summand is BVO, then so is the max...
    return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
  }

  string MaxFunction::displayString() {
    ostringstream ss;
    ss << "max( " << _f1->displayString() << " , " << _f2->displayString() << " )";
    return ss.str();
  }

  void MaxFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    Intrepid::FieldContainer<double> values2(values);
    _f1->values(values,basisCache);
    _f2->values(values2,basisCache);
    for(int i = 0; i < values.size(); i++) {
      values[i] = std::max(values[i],values2[i]);
    }
  }

  Teuchos::RCP<Function<double> > MaxFunction::x() {
    if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::max(_f1->x(),_f2->x());
  }

  Teuchos::RCP<Function<double> > MaxFunction::y() {
    if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::max(_f1->y(),_f2->y());
  }

  Teuchos::RCP<Function<double> > MaxFunction::z() {
    if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
      return Function<double>::null();
    }
    return Function<double>::max(_f1->z(),_f2->z());
  }

  string hFunction::displayString() {
    return "h";
  }

  double hFunction::value(double x, double y, double h) {
    return h;
  }
  void hFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    Intrepid::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
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
  class ComposedFunction : public Function<double> {
    Teuchos::RCP<Function<double> > _f, _arg_g;
    public:
    ComposedFunction(Teuchos::RCP<Function<double> > f, Teuchos::RCP<Function<double> > arg_g) : Function<double>(f->rank()) {
      _f = f;
      _arg_g = arg_g;
    }
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      this->CHECK_VALUES_RANK(values);
      int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
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
    Teuchos::RCP<Function<double> > dx() {
      if (Function<double>::isNull(_f->dx()) || Function<double>::isNull(_arg_g->dx())) {
        return Function<double>::null();
      }
      // chain rule:
      return _arg_g->dx() * Function<double>::composedFunction(_f->dx(),_arg_g);
    }
    Teuchos::RCP<Function<double> > dy() {
      if (Function<double>::isNull(_f->dy()) || Function<double>::isNull(_arg_g->dy())) {
        return Function<double>::null();
      }
      // chain rule:
      return _arg_g->dy() * Function<double>::composedFunction(_f->dy(),_arg_g);
    }
    Teuchos::RCP<Function<double> > dz() {
      if (Function<double>::isNull(_f->dz()) || Function<double>::isNull(_arg_g->dz())) {
        return Function<double>::null();
      }
      // chain rule:
      return _arg_g->dz() * Function<double>::composedFunction(_f->dz(),_arg_g);
    }
  };

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::composedFunction( Teuchos::RCP<Function<double> > f, Teuchos::RCP<Function<double> > arg_g) {
    return Teuchos::rcp( new ComposedFunction(f,arg_g) );
  }

  template <typename Scalar>
  Scalar SimpleFunction<Scalar>::value(double x) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleFunction must implement value() for some number of arguments < spaceDim");
    return 0;
  }

  template <typename Scalar>
  Scalar SimpleFunction<Scalar>::value(double x, double y) {
    return value(x);
  }

  template <typename Scalar>
  Scalar SimpleFunction<Scalar>::value(double x, double y, double z) {
    return value(x,y);
  }

  template <typename Scalar>
  Scalar SimpleFunction<Scalar>::value(double x, double y, double z, double t) {
    return value(x,y,z);
  }

  template <typename Scalar>
  void SimpleFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());

    if (points->dimension(1) != numPoints) {
      cout << "numPoints in values container does not match that in BasisCache's physical points.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numPoints in values container does not match that in BasisCache's physical points.");
    }

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
        } else if (spaceDim == 4) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          double z = (*points)(cellIndex,ptIndex,2);
          double t = (*points)(cellIndex,ptIndex,3);
          values(cellIndex,ptIndex) = value(x,y,z,t);
        }
      }
    }
  }

  template <typename Scalar>
  SimpleVectorFunction<Scalar>::SimpleVectorFunction() : Function<Scalar>(1) {}

  template <typename Scalar>
  vector<Scalar> SimpleVectorFunction<Scalar>::value(double x) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleVectorFunction must implement value() for some number of arguments < spaceDim");
    return vector<Scalar>();
  }

  template <typename Scalar>
  vector<Scalar> SimpleVectorFunction<Scalar>::value(double x, double y) {
    return value(x);
  }

  template <typename Scalar>
  vector<Scalar> SimpleVectorFunction<Scalar>::value(double x, double y, double z) {
    return value(x,y);
  }

  template <typename Scalar>
  void SimpleVectorFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
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

  template <typename Scalar>
  PolarizedFunction<Scalar>::PolarizedFunction( Teuchos::RCP<Function<Scalar> > f_of_xAsR_yAsTheta ) : Function<Scalar>(f_of_xAsR_yAsTheta->rank()) {
    _f = f_of_xAsR_yAsTheta;
  }

  template <typename Scalar>
  Teuchos::RCP<PolarizedFunction<double> > PolarizedFunction<Scalar>::r() {
    static Teuchos::RCP<PolarizedFunction<double> > _r = Teuchos::rcp( new PolarizedFunction<double>( Teuchos::rcp( new Xn(1) ) ) );
    return _r;
  }

  template <typename Scalar>
  Teuchos::RCP<PolarizedFunction<double> > PolarizedFunction<Scalar>::sin_theta() {
    static Teuchos::RCP<PolarizedFunction<double> > _sin_theta = Teuchos::rcp( new PolarizedFunction<double> ( Teuchos::rcp( new Sin_y ) ) );
    return _sin_theta;
  }

  template <typename Scalar>
  Teuchos::RCP<PolarizedFunction<double> > PolarizedFunction<Scalar>::cos_theta() {
    static Teuchos::RCP<PolarizedFunction<double> > _cos_theta = Teuchos::rcp( new PolarizedFunction<double> ( Teuchos::rcp( new Cos_y ) ) );
    return _cos_theta;
  }

  void findAndReplace(string &str, const string &findStr, const string &replaceStr) {
    size_t found = str.find( findStr );
    while (found!=string::npos) {
      str.replace( found, findStr.length(), replaceStr );
      found = str.find( findStr );
    }
  }

  template <typename Scalar>
  string PolarizedFunction<Scalar>::displayString() {
    string displayString = _f->displayString();
    findAndReplace(displayString, "x", "r");
    findAndReplace(displayString, "y", "\\theta");
    return displayString;
    //  ostringstream ss( _f->displayString());
    //  ss << "(r,\\theta)";
    //  return ss.str();
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > PolarizedFunction<Scalar>::dx() {
    // cast everything to Teuchos::RCP<Function<Scalar> >s:
    Teuchos::RCP<Function<double> > sin_theta_fxn = sin_theta();
    Teuchos::RCP<Function<Scalar> > dtheta_fxn = dtheta();
    Teuchos::RCP<Function<Scalar> > dr_fxn = dr();
    Teuchos::RCP<Function<double> > r_fxn = r();
    Teuchos::RCP<Function<double> > cos_theta_fxn = cos_theta();
    return dr_fxn * cos_theta_fxn - dtheta_fxn * sin_theta_fxn / r_fxn;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > PolarizedFunction<Scalar>::dy() {
    Teuchos::RCP<Function<double> > sin_theta_fxn = sin_theta();
    Teuchos::RCP<Function<Scalar> > dtheta_fxn = dtheta();
    Teuchos::RCP<Function<Scalar> > dr_fxn = dr();
    Teuchos::RCP<Function<double> > r_fxn = r();
    Teuchos::RCP<Function<double> > cos_theta_fxn = cos_theta();
    return dr_fxn * sin_theta_fxn + dtheta_fxn * cos_theta_fxn / r_fxn;
  }

  template <typename Scalar>
  Teuchos::RCP<PolarizedFunction<Scalar> > PolarizedFunction<Scalar>::dtheta() {
    return Teuchos::rcp( new PolarizedFunction<Scalar>( _f->dy() ) );
  }

  template <typename Scalar>
  Teuchos::RCP<PolarizedFunction<Scalar> > PolarizedFunction<Scalar>::dr() {
    return Teuchos::rcp( new PolarizedFunction<Scalar>( _f->dx() ) );
  }

  template <typename Scalar>
  bool PolarizedFunction<Scalar>::isZero() {
    return _f->isZero();
  }

  template <typename Scalar>
  void PolarizedFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    static const double PI  = 3.141592653589793238462;

    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    Intrepid::FieldContainer<double> polarPoints = *points;
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

  template <typename Scalar>
  bool ScalarFunctionOfNormal<Scalar>::boundaryValueOnly() {
    return true;
  }

  template <typename Scalar>
  void ScalarFunctionOfNormal<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    const Intrepid::FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
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

  SideParityFunction::SideParityFunction() : Function<double>(0) {
    //  cout << "SideParityFunction constructor.\n";
  }

  bool SideParityFunction::boundaryValueOnly() {
    return true;
  }

  string SideParityFunction::displayString() {
    return "sgn(n)";
  }

  void SideParityFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr sideBasisCache) {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int sideIndex = sideBasisCache->getSideIndex();
    if (sideIndex == -1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-sideBasisCache passed into SideParityFunction");
    }
    if (sideBasisCache->getCellSideParities().size() > 0) {
      // then we'll use this, and won't require that mesh and cellIDs are set
      if (sideBasisCache->getCellSideParities().dimension(0) != numCells) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideBasisCache->getCellSideParities() is non-empty, but the cell dimension doesn't match that of the values FieldContainer.");
      }

      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
        int parity = sideBasisCache->getCellSideParities()(cellOrdinal,sideIndex);
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++) {
          values(cellOrdinal,ptOrdinal) = parity;
        }
      }
    } else {
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
  }

  UnitNormalFunction::UnitNormalFunction(int comp, bool spaceTime) : Function<double>( (comp<0)? 1 : 0) {
    _comp = comp;
    _spaceTime = spaceTime;
  }

  Teuchos::RCP<Function<double> > UnitNormalFunction::x() {
    return Teuchos::rcp( new UnitNormalFunction(0,_spaceTime) );
  }

  Teuchos::RCP<Function<double> > UnitNormalFunction::y() {
    return Teuchos::rcp( new UnitNormalFunction(1,_spaceTime) );
  }

  Teuchos::RCP<Function<double> > UnitNormalFunction::z() {
    return Teuchos::rcp( new UnitNormalFunction(2,_spaceTime) );
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

  void UnitNormalFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
    const Intrepid::FieldContainer<double> *sideNormals = _spaceTime ? &(basisCache->getSideNormalsSpaceTime()) : &(basisCache->getSideNormals());
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

  template <typename Scalar>
  VectorizedFunction<Scalar>::VectorizedFunction(const vector< Teuchos::RCP<Function<Scalar> > > &fxns) : Function<Scalar>(fxns[0]->rank() + 1) {
    _fxns = fxns;
  }

  template <typename Scalar>
  VectorizedFunction<Scalar>::VectorizedFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) : Function<Scalar>(f1->rank() + 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
    _fxns.push_back(f1);
    _fxns.push_back(f2);
  }

  template <typename Scalar>
  VectorizedFunction<Scalar>::VectorizedFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2, Teuchos::RCP<Function<Scalar> > f3) : Function<Scalar>(f1->rank() + 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
    TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
    _fxns.push_back(f1);
    _fxns.push_back(f2);
    _fxns.push_back(f3);
  }

  template <typename Scalar>
  string VectorizedFunction<Scalar>::displayString() {
    ostringstream str;
    str << "(";
    for (int i=0; i<_fxns.size(); i++) {
      if (i > 0) str << ",";
      str << _fxns[i]->displayString();
    }
    str << ")";
    return str.str();
  }

  template <typename Scalar>
  int VectorizedFunction<Scalar>::dim() {
    return _fxns.size();
  }

  template <typename Scalar>
  void VectorizedFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    this->CHECK_VALUES_RANK(values);
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
    Intrepid::FieldContainer<Scalar> compValues(dims);
    int valuesPerComponent = compValues.size();

    for (int comp=0; comp < numComponents; comp++) {
      Teuchos::RCP<Function<Scalar> > fxn = _fxns[comp];
      fxn->values(compValues, basisCache);
      for (int i=0; i < valuesPerComponent; i++) {
        values[ numComponents * i + comp ] = compValues[ i ];
      }
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::x() {
    return _fxns[0];
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::y() {
    return _fxns[1];
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::z() {
    if (dim() >= 3) {
      return _fxns[2];
    } else {
      return Function<Scalar>::null();
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::di(int i) {
    // derivative in the ith coordinate direction
    Camellia::EOperator op;
    switch (i) {
      case 0:
        op = Camellia::OP_DX;
        break;
      case 1:
        op = Camellia::OP_DY;
        break;
      case 2:
        op = Camellia::OP_DZ;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid coordinate direction");
        break;
    }
    vector< Teuchos::RCP<Function<Scalar> > > fxns;
    for (int j = 0; j< dim(); j++) {
      Teuchos::RCP<Function<Scalar> > fj_di = Function<Scalar>::op(_fxns[j], op);
      if (Function<Scalar>::isNull(fj_di)) {
        return Function<Scalar>::null();
      }
      fxns.push_back(fj_di);
    }
    // if we made it this far, then all components aren't null:
    return Teuchos::rcp( new VectorizedFunction<Scalar>(fxns) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::dx() {
    return di(0);
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::dy() {
    return di(1);
  }
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > VectorizedFunction<Scalar>::dz() {
    return di(2);
  }

  template <typename Scalar>
  bool VectorizedFunction<Scalar>::isZero() {
    // vector function is zero if each of its components is zero.
    for (typename vector< Teuchos::RCP<Function<Scalar> > >::iterator fxnIt = _fxns.begin(); fxnIt != _fxns.end(); fxnIt++) {
      if (! (*fxnIt)->isZero() ) {
        return false;
      }
    }
    return true;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) {
    if (f1->isZero() || f2->isZero()) {
      if ( f1->rank() == f2->rank() ) {
        return Function<Scalar>::zero();
      } else if ((f1->rank() == 0) || (f2->rank() == 0)) {
        int result_rank = f1->rank() + f2->rank();
        return Function<Scalar>::zero(result_rank);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"functions have incompatible rank for product.");
      }
    }
    return Teuchos::rcp( new ProductFunction<Scalar>(f1,f2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > scalarDivisor) {
    if ( f1->isZero() ) {
      return Function<Scalar>::zero(f1->rank());
    }
    return Teuchos::rcp( new QuotientFunction<Scalar>(f1,scalarDivisor) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Teuchos::RCP<Function<Scalar> > f1, Scalar divisor) {
    return f1 / Teuchos::rcp( new ConstantScalarFunction<Scalar>(divisor) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Scalar value, Teuchos::RCP<Function<Scalar> > scalarDivisor) {
    return Function<Scalar>::constant(value) / scalarDivisor;
  }

  //ConstantScalarTeuchos::RCP<Function<Scalar> > operator*(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2) {
  //  return Teuchos::rcp( new ConstantScalarFunction(f1->value() * f2->value()) );
  //}
  //
  //ConstantScalarTeuchos::RCP<Function<Scalar> > operator/(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2) {
  //  return Teuchos::rcp( new ConstantScalarFunction(f1->value() / f2->value()) );
  //}

  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator*(ConstantVectorTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2) {
  //  vector<double> value = f1->value();
  //  for (int d=0; d<value.size(); d++) {
  //    value[d] *= f2->value();
  //  }
  //  return Teuchos::rcp( new ConstantVectorFunction(value) );
  //}
  //
  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator*(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantVectorTeuchos::RCP<Function<Scalar> > f2) {
  //  return f2 * f1;
  //}
  //
  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator/(ConstantVectorTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2) {
  //  vector<double> value = f1->value();
  //  for (int d=0; d<value.size(); d++) {
  //    value[d] /= f2->value();
  //  }
  //  return Teuchos::rcp( new ConstantVectorFunction(value) );
  //}

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Scalar weight, Teuchos::RCP<Function<Scalar> > f) {
    return Function<Scalar>::constant(weight) * f;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f, Scalar weight) {
    return weight * f;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(vector<Scalar> weight, Teuchos::RCP<Function<Scalar> > f) {
    return Function<Scalar>::constant(weight) * f;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f, vector<Scalar> weight) {
    return weight * f;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) {
    if ( f1->isZero() ) {
      return f2;
    }
    if ( f2->isZero() ) {
      return f1;
    }
    return Teuchos::rcp( new SumFunction<Scalar>(f1, f2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Teuchos::RCP<Function<Scalar> > f1, Scalar value) {
    return f1 + Function<Scalar>::constant(value);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Scalar value, Teuchos::RCP<Function<Scalar> > f1) {
    return f1 + Function<Scalar>::constant(value);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2) {
    return f1 + -f2;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f1, Scalar value) {
    return f1 - Function<Scalar>::constant(value);
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Scalar value, Teuchos::RCP<Function<Scalar> > f1) {
    return Function<Scalar>::constant(value) - f1;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f) {
    return -1.0 * f;
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::min(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2) {
    return Teuchos::rcp( new MinFunction(f1, f2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::min(Teuchos::RCP<Function<double> > f1, double value) {
    return Teuchos::rcp( new MinFunction(f1, Function<double>::constant(value)) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::min(double value, Teuchos::RCP<Function<double> > f2) {
    return Teuchos::rcp( new MinFunction(f2, Function<double>::constant(value)) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::max(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2) {
    return Teuchos::rcp( new MaxFunction(f1, f2) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::max(Teuchos::RCP<Function<double> > f1, double value) {
    return Teuchos::rcp( new MaxFunction(f1, Function<double>::constant(value)) );
  }

  template <typename Scalar>
  Teuchos::RCP<Function<double> > Function<Scalar>::max(double value, Teuchos::RCP<Function<double> > f2) {
    return Teuchos::rcp( new MaxFunction(f2, Function<double>::constant(value)) );
  }

  string Sin_y::displayString() {
    return "\\sin y";
  }

  double Sin_y::value(double x, double y) {
    return sin(y);
  }
  Teuchos::RCP<Function<double> > Sin_y::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Sin_y::dy() {
    return Teuchos::rcp( new Cos_y );
  }
  Teuchos::RCP<Function<double> > Sin_y::dz() {
    return Function<double>::zero();
  }

  string Cos_y::displayString() {
    return "\\cos y";
  }
  double Cos_y::value(double x, double y) {
    return cos(y);
  }
  Teuchos::RCP<Function<double> > Cos_y::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Cos_y::dy() {
    Teuchos::RCP<Function<double> > sin_y = Teuchos::rcp( new Sin_y );
    return - sin_y;
  }
  Teuchos::RCP<Function<double> > Cos_y::dz() {
    return Function<double>::zero();
  }

  string Sin_x::displayString() {
    return "\\sin x";
  }

  double Sin_x::value(double x, double y) {
    return sin(x);
  }
  Teuchos::RCP<Function<double> > Sin_x::dx() {
    return Teuchos::rcp( new Cos_x );
  }
  Teuchos::RCP<Function<double> > Sin_x::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Sin_x::dz() {
    return Function<double>::zero();
  }

  string Cos_x::displayString() {
    return "\\cos x";
  }
  double Cos_x::value(double x, double y) {
    return cos(x);
  }
  Teuchos::RCP<Function<double> > Cos_x::dx() {
    Teuchos::RCP<Function<double> > sin_x = Teuchos::rcp( new Sin_x );
    return - sin_x;
  }
  Teuchos::RCP<Function<double> > Cos_x::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Cos_x::dz() {
    return Function<double>::zero();
  }

  string Exp_x::displayString() {
    return "e^x";
  }
  double Exp_x::value(double x, double y) {
    return exp(x);
  }
  Teuchos::RCP<Function<double> > Exp_x::dx() {
    return Teuchos::rcp( new Exp_x );
  }
  Teuchos::RCP<Function<double> > Exp_x::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Exp_x::dz() {
    return Function<double>::zero();
  }

  string Exp_y::displayString() {
    return "e^y";
  }
  double Exp_y::value(double x, double y) {
    return exp(y);
  }
  Teuchos::RCP<Function<double> > Exp_y::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Exp_y::dy() {
    return Teuchos::rcp( new Exp_y );
  }
  Teuchos::RCP<Function<double> > Exp_y::dz() {
    return Function<double>::zero();
  }

  string Exp_z::displayString() {
    return "e^z";
  }
  double Exp_z::value(double x, double y, double z) {
    return exp(z);
  }
  Teuchos::RCP<Function<double> > Exp_z::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Exp_z::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Exp_z::dz() {
    return Teuchos::rcp( new Exp_z );
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
  Teuchos::RCP<Function<double> > Xn::dx() {
    if (_n == 0) {
      return Function<double>::zero();
    }
    Teuchos::RCP<Function<double> > x_n_minus = Teuchos::rcp( new Xn(_n-1) );
    return (double)_n * x_n_minus;
  }
  Teuchos::RCP<Function<double> > Xn::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Xn::dz() {
    return Function<double>::zero();
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

  Teuchos::RCP<Function<double> > Yn::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Yn::dy() {
    if (_n == 0) {
      return Function<double>::zero();
    }
    Teuchos::RCP<Function<double> > y_n_minus = Teuchos::rcp( new Yn(_n-1) );
    return (double)_n * y_n_minus;
  }
  Teuchos::RCP<Function<double> > Yn::dz() {
    return Function<double>::zero();
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

  Teuchos::RCP<Function<double> > Zn::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Zn::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Zn::dz() {
    if (_n == 0) {
      return Function<double>::zero();
    }
    Teuchos::RCP<Function<double> > z_n_minus = Teuchos::rcp( new Zn(_n-1) );
    return (double)_n * z_n_minus;
  }

  string Tn::displayString() {
    ostringstream ss;
    if ((_n != 1) && (_n != 0)) {
      ss << "t^" << _n ;
    } else if (_n == 1) {
      ss << "t";
    } else {
      ss << "(1)";
    }
    return ss.str();
  }
  Tn::Tn(int n) {
    _n = n;
  }
  double Tn::value(double x, double t) {
    return pow(t,_n);
  }
  double Tn::value(double x, double y, double t) {
    return pow(t,_n);
  }
  double Tn::value(double x, double y, double z, double t) {
    return pow(t,_n);
  }

  Teuchos::RCP<Function<double> > Tn::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Tn::dy() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Tn::dz() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Tn::dt() {
    if (_n == 0) {
      return Function<double>::zero();
    }
    Teuchos::RCP<Function<double> > z_n_minus = Teuchos::rcp( new Tn(_n-1) );
    return (double)_n * z_n_minus;
  }

  template <typename Scalar>
  SimpleSolutionFunction<Scalar>::SimpleSolutionFunction(VarPtr var, Teuchos::RCP<Solution<Scalar> > soln) : Function<Scalar>(var->rank()) {
    _var = var;
    _soln = soln;
  }

  template <typename Scalar>
  bool SimpleSolutionFunction<Scalar>::boundaryValueOnly() {
    return (_var->varType() == FLUX) || (_var->varType() == TRACE);
  }

  template <typename Scalar>
  string SimpleSolutionFunction<Scalar>::displayString() {
    ostringstream str;
    str << "\\overline{" << _var->displayString() << "} ";
    return str.str();
  }

  template <typename Scalar>
  void SimpleSolutionFunction<Scalar>::importCellData(std::vector<GlobalIndexType> cells) {
    int rank = Teuchos::GlobalMPISession::getRank();
    set<GlobalIndexType> offRankCells;
    const set<GlobalIndexType>* rankLocalCells = &_soln->mesh()->globalDofAssignment()->cellsInPartition(rank);
    for (int cellOrdinal=0; cellOrdinal < cells.size(); cellOrdinal++) {
      if (rankLocalCells->find(cells[cellOrdinal]) == rankLocalCells->end()) {
        offRankCells.insert(cells[cellOrdinal]);
      }
    }
    _soln->importSolutionForOffRankCells(offRankCells);
  }

  template <typename Scalar>
  void SimpleSolutionFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
    bool dontWeightForCubature = false;
    if (basisCache->mesh().get() != NULL) { // then we assume that the BasisCache is appropriate for solution's mesh...
      _soln->solutionValues(values, _var->ID(), basisCache, dontWeightForCubature, _var->op());
    } else {
      // the following adapted from PreviousSolutionFunction.  Probably would do well to consolidate
      // that class with this one at some point...
      LinearTermPtr solnExpression = 1.0 * _var;
      // get the physicalPoints, and make a basisCache for each...
      Intrepid::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
      Intrepid::FieldContainer<Scalar> value(1,1); // assumes scalar-valued solution function.
      int numCells = physicalPoints.dimension(0);
      int numPoints = physicalPoints.dimension(1);
      int spaceDim = physicalPoints.dimension(2);
      physicalPoints.resize(numCells*numPoints,spaceDim);
      vector< ElementPtr > elements = _soln->mesh()->elementsForPoints(physicalPoints, false); // false: don't make elements null just because they're off-rank.
      Intrepid::FieldContainer<double> point(1,1,spaceDim);
      Intrepid::FieldContainer<double> refPoint(1,spaceDim);
      int combinedIndex = 0;
      vector<GlobalIndexType> cellID;
      cellID.push_back(-1);
      BasisCachePtr basisCacheOnePoint;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++, combinedIndex++) {
          if (elements[combinedIndex].get()==NULL) continue; // no element found for point; skip it…
          ElementTypePtr elemType = elements[combinedIndex]->elementType();
          for (int d=0; d<spaceDim; d++) {
            point(0,0,d) = physicalPoints(combinedIndex,d);
          }
          if (elements[combinedIndex]->cellID() != cellID[0]) {
            cellID[0] = elements[combinedIndex]->cellID();
            basisCacheOnePoint = Teuchos::rcp( new BasisCache(elemType, _soln->mesh()) );
            basisCacheOnePoint->setPhysicalCellNodes(_soln->mesh()->physicalCellNodesForCell(cellID[0]),cellID,false); // false: don't createSideCacheToo
          }
          refPoint.resize(1,1,spaceDim); // CamelliaCellTools::mapToReferenceFrame wants a numCells dimension...  (perhaps it shouldn't, though!)
          // compute the refPoint:
          CamelliaCellTools::mapToReferenceFrame(refPoint,point,_soln->mesh()->getTopology(), cellID[0],
              _soln->mesh()->globalDofAssignment()->getCubatureDegree(cellID[0]));
          refPoint.resize(1,spaceDim);
          basisCacheOnePoint->setRefCellPoints(refPoint);
          //          cout << "refCellPoints:\n " << refPoint;
          //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePoints();
          solnExpression->evaluate(value, _soln, basisCacheOnePoint);
          //          cout << "value at point (" << point(0,0) << ", " << point(0,1) << ") = " << value(0,0) << endl;
          values(cellIndex,ptIndex) = value(0,0);
        }
      }
    }
    if (_var->varType()==FLUX) { // weight by sideParity
      Function<double>::sideParity()->scalarMultiplyFunctionValues(values, basisCache);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::dx() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->dx(), _soln);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::dy() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->dy(), _soln);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::dz() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->dz(), _soln);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::x() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->x(), _soln);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::y() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->y(), _soln);
    }
  }

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > SimpleSolutionFunction<Scalar>::z() {
    if (_var->op() != Camellia::OP_VALUE) {
      return Function<Scalar>::null();
    } else {
      return Function<Scalar>::solution(_var->z(), _soln);
    }
  }

  Cos_ax::Cos_ax(double a, double b) {
    _a = a;
    _b = b;
  }
  double Cos_ax::value(double x) {
    return cos( _a * x + _b);
  }
  Teuchos::RCP<Function<double> > Cos_ax::dx() {
    return -_a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Sin_ax(_a,_b));
  }
  Teuchos::RCP<Function<double> > Cos_ax::dy() {
    return Function<double>::zero();
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
  Teuchos::RCP<Function<double> > Cos_ay::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Cos_ay::dy() {
    return -_a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Sin_ay(_a));
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
  Teuchos::RCP<Function<double> > Sin_ax::dx() {
    return _a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Cos_ax(_a,_b));
  }
  Teuchos::RCP<Function<double> > Sin_ax::dy() {
    return Function<double>::zero();
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
  Teuchos::RCP<Function<double> > Sin_ay::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Sin_ay::dy() {
    return _a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Cos_ay(_a));
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
  Teuchos::RCP<Function<double> > Exp_ax::dx() {
    return _a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Exp_ax(_a));
  }
  Teuchos::RCP<Function<double> > Exp_ax::dy() {
    return Function<double>::zero();
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
  Teuchos::RCP<Function<double> > Exp_ay::dx() {
    return Function<double>::zero();
  }
  Teuchos::RCP<Function<double> > Exp_ay::dy() {
    return _a * (Teuchos::RCP<Function<double> >) Teuchos::rcp(new Exp_ay(_a));
  }
  string Exp_ay::displayString() {
    ostringstream ss;
    ss << "\\exp( " << _a << " y )";
    return ss.str();
  }
  template class Function<double>;
  template class ConstantScalarFunction<double>;
  template class ConstantVectorFunction<double>;
  template class SimpleFunction<double>;
  template class SimpleVectorFunction<double>;
  template Teuchos::RCP<Function<double> > operator+(double value, Teuchos::RCP<Function<double> > f1);
  template Teuchos::RCP<Function<double> > operator+(Teuchos::RCP<Function<double> > f1, double value);
  template Teuchos::RCP<Function<double> > operator-(double value, Teuchos::RCP<Function<double> > f1);
  template Teuchos::RCP<Function<double> > operator-(Teuchos::RCP<Function<double> > f1, double value);
  template Teuchos::RCP<Function<double> > operator*(double value, Teuchos::RCP<Function<double> > f1);
  template Teuchos::RCP<Function<double> > operator*(Teuchos::RCP<Function<double> > f1, double value);
  // template Teuchos::RCP<Function<double> > operator/(double value, Teuchos::RCP<Function<double> > f1);
  // template Teuchos::RCP<Function<double> > operator/(Teuchos::RCP<Function<double> > f1, double value);
}

