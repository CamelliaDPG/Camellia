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

namespace Camellia
{
// for adaptive quadrature
struct CacheInfo
{
  ElementTypePtr elemType;
  GlobalIndexType cellID;
  Intrepid::FieldContainer<double> subCellNodes;
};

// private class ComponentFunction
template <typename Scalar>
class ComponentFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _vectorFxn;
  int _component;
public:
  ComponentFunction(TFunctionPtr<Scalar> vectorFunction, int componentIndex)
  {
    _vectorFxn = vectorFunction;
    _component = componentIndex;
    if (_vectorFxn->rank() < 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vector function must have rank 1 or greater");
    }
  }
  bool boundaryValueOnly()
  {
    return _vectorFxn->boundaryValueOnly();
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    // note this allocation.  There might be ways of reusing memory here, if we had a slightly richer API.
    int spaceDim = basisCache->getSpaceDim();
    Teuchos::Array<int> dim;
    values.dimensions(dim);
    dim.push_back(spaceDim);

    Intrepid::FieldContainer<double> vectorValues(dim);
    _vectorFxn->values(vectorValues, basisCache);

    int numValues = values.size();
    for (int i=0; i<numValues; i++)
    {
      values[i] = vectorValues[spaceDim*i + _component];
    }
  }
};

// private class CellBoundaryRestrictedFunction
template <typename Scalar>
class CellBoundaryRestrictedFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _fxn;
public:
  CellBoundaryRestrictedFunction(TFunctionPtr<Scalar> fxn) : TFunction<Scalar>(fxn->rank())
  {
    _fxn = fxn;
  }

  bool boundaryValueOnly()
  {
    return true;
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    _fxn->values(values, basisCache);
  }
};

class HeavisideFunction : public SimpleFunction<double>
{
  double _xShift;
public:
  HeavisideFunction(double xShift=0.0)
  {
    _xShift = xShift;
  }
  double value(double x)
  {
    return (x < _xShift) ? 0.0 : 1.0;
  }
};

class MeshBoundaryCharacteristicFunction : public TFunction<double>
{
public:
  MeshBoundaryCharacteristicFunction() : TFunction<double>(0)
  {
  }
  bool boundaryValueOnly()
  {
    return true;
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    CHECK_VALUES_RANK(values);
    // scalar: values shape is (C,P)
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int sideIndex = basisCache->getSideIndex();
    MeshPtr mesh = basisCache->mesh();
    TEUCHOS_TEST_FOR_EXCEPTION(mesh.get() == NULL, std::invalid_argument, "MeshBoundaryCharacteristicFunction requires a mesh!");
    TEUCHOS_TEST_FOR_EXCEPTION(sideIndex == -1, std::invalid_argument, "MeshBoundaryCharacteristicFunction is only defined on cell boundaries");
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      int cellID = basisCache->cellIDs()[cellIndex];
      bool onBoundary = mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex);
      double value = onBoundary ? 1 : 0;
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
      {
        values(cellIndex,pointIndex) = value;
      }
    }
  }
  TFunctionPtr<double> dx()
  {
    return TFunction<double>::zero();
  }
  TFunctionPtr<double> dy()
  {
    return TFunction<double>::zero();
  }
  //  TFunctionPtr<double> dz() {
  //    return TFunction<double>::zero();
  //  }
};

class MeshSkeletonCharacteristicFunction : public ConstantScalarFunction<double>
{
public:
  MeshSkeletonCharacteristicFunction() : ConstantScalarFunction<double>(1, "|_{\\Gamma_h}")
  {
  }
  bool boundaryValueOnly()
  {
    return true;
  }
};

template <typename Scalar>
TFunction<Scalar>::TFunction()
{
  _rank = 0;
  _displayString = this->displayString();
  _time = 0;
}
template <typename Scalar>
TFunction<Scalar>::TFunction(int rank)
{
  _rank = rank;
  _displayString = this->displayString();
  _time = 0;
}

template <typename Scalar>
string TFunction<Scalar>::displayString()
{
  return "f";
}

template <typename Scalar>
int TFunction<Scalar>::rank()
{
  return _rank;
}

template <typename Scalar>
void TFunction<Scalar>::setTime(double time)
{
  _time = time;
}

template <typename Scalar>
double TFunction<Scalar>::getTime()
{
  return _time;
}

template <typename Scalar>
void TFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache)
{
  switch (op)
  {
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
  if (op==Camellia::OP_VALUE)
  {

  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::op(TFunctionPtr<Scalar> f, Camellia::EOperator op)
{
  if ( isNull(f) )
  {
    return TFunction<Scalar>::null();
  }
  switch (op)
  {
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
    return f * TFunction<Scalar>::normal();
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported operator");
    break;
  }
  return Teuchos::rcp((TFunction<Scalar>*)NULL);
}

template <typename Scalar>
bool TFunction<Scalar>::equals(TFunctionPtr<Scalar> f, BasisCachePtr basisCacheForCellsToCompare, double tol)
{
  if (f->rank() != this->rank())
  {
    return false;
  }
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp(this,false);
  TFunctionPtr<Scalar> diff = thisPtr-f;

  int numCells = basisCacheForCellsToCompare->getPhysicalCubaturePoints().dimension(0);
  // compute L^2 norm of difference on the cells
  Intrepid::FieldContainer<Scalar> diffs_squared(numCells);
  (diff*diff)->integrate(diffs_squared, basisCacheForCellsToCompare);
  Scalar sum = 0;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    sum += diffs_squared[cellIndex];
  }
  return sqrt(abs(sum)) < tol;
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(MeshPtr mesh, double x)
{
  int spaceDim = 1;
  Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
  Intrepid::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getSpaceDim() != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 1D if only x is provided.");
  }

  physPoint(0,0) = x;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
  if (cellIDs.size() == 0)
  {
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
Scalar TFunction<Scalar>::evaluate(MeshPtr mesh, double x, double y)
{
  int spaceDim = 2;
  Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
  Intrepid::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getSpaceDim() != spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 2D if (x,y) is provided.");
  }

  physPoint(0,0) = x;
  physPoint(0,1) = y;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
  if (cellIDs.size() == 0)
  {
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
Scalar TFunction<Scalar>::evaluate(MeshPtr mesh, double x, double y, double z)
{
  int spaceDim = 3;
  Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
  Intrepid::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getSpaceDim() != spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 3D if (x,y,z) is provided.");
  }

  physPoint(0,0) = x;
  physPoint(0,1) = y;
  physPoint(0,2) = z;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint);
  if (cellIDs.size() == 0)
  {
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
Scalar TFunction<Scalar>::evaluate(double x)
{
  static Intrepid::FieldContainer<Scalar> value(1,1); // (C,P)
  static Intrepid::FieldContainer<double> physPoint(1,1,1);

  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  this->values(value,dummyCache);
  return value[0];
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(TFunctionPtr<Scalar> f, double x)
{
  return f->evaluate(x);
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(double x, double y)
{
  static Intrepid::FieldContainer<Scalar> value(1,1);
  static Intrepid::FieldContainer<double> physPoint(1,1,2);
  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  this->values(value,dummyCache);
  return value[0];
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(TFunctionPtr<Scalar> f, double x, double y)   // for testing; this isn't super-efficient
{
  return f->evaluate(x, y);
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(double x, double y, double z)   // for testing; this isn't super-efficient
{
  static Intrepid::FieldContainer<Scalar> value(1,1);
  static Intrepid::FieldContainer<double> physPoint(1,1,3);
  static Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  dummyCache->writablePhysicalCubaturePoints()(0,0,0) = x;
  dummyCache->writablePhysicalCubaturePoints()(0,0,1) = y;
  dummyCache->writablePhysicalCubaturePoints()(0,0,2) = z;
  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 1 Function.");
  }
  this->values(value,dummyCache);
  return value[0];
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(TFunctionPtr<Scalar> f, double x, double y, double z)   // for testing; this isn't super-efficient
{
  return f->evaluate(x,y,z);
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::x()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::y()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::z()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::t()
{
  return TFunction<Scalar>::null();
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::dx()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::dy()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::dz()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::dt()
{
  return TFunction<Scalar>::null();
}
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::curl()
{
  TFunctionPtr<Scalar> dxFxn = dx();
  TFunctionPtr<Scalar> dyFxn = dy();
  TFunctionPtr<Scalar> dzFxn = dz();

  if (dxFxn.get()==NULL)
  {
    return TFunction<Scalar>::null();
  }
  else if (dyFxn.get()==NULL)
  {
    // special case: in 1D, curl() returns a scalar
    return dxFxn;
  }
  else if (dzFxn.get() == NULL)
  {
    // in 2D, the rank of the curl operator depends on the rank of the Function
    if (_rank == 0)
    {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dyFxn,-dxFxn) );
    }
    else if (_rank == 1)
    {
      return dyFxn->x() - dxFxn->y();
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "curl() undefined for Functions of rank > 1");
    }
  }
  else
  {
    return Teuchos::rcp( new VectorizedFunction<Scalar>(dyFxn->z() - dzFxn->y(),
                         dzFxn->x() - dxFxn->z(),
                         dxFxn->y() - dyFxn->x()) );
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::grad(int numComponents)
{
  TFunctionPtr<Scalar> dxFxn = dx();
  TFunctionPtr<Scalar> dyFxn = dy();
  TFunctionPtr<Scalar> dzFxn = dz();
  if (numComponents==-1)   // default: just use as many non-null components as available
  {
    if (dxFxn.get()==NULL)
    {
      return TFunction<Scalar>::null();
    }
    else if (dyFxn.get()==NULL)
    {
      // special case: in 1D, grad() returns a scalar
      return dxFxn;
    }
    else if (dzFxn.get() == NULL)
    {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn) );
    }
    else
    {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn,dzFxn) );
    }
  }
  else if (numComponents==1)
  {
    // special case: we don't "vectorize" in 1D
    return dxFxn;
  }
  else if (numComponents==2)
  {
    if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL))
    {
      return TFunction<Scalar>::null();
    }
    else
    {
      return TFunction<Scalar>::vectorize(dxFxn, dyFxn);
    }
  }
  else if (numComponents==3)
  {
    if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL) || (dzFxn.get()==NULL))
    {
      return TFunction<Scalar>::null();
    }
    else
    {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn,dzFxn) );
    }
  }
  else if (numComponents==4)
  {
    TFunctionPtr<Scalar> dtFxn = dt();
    if ((dxFxn.get() == NULL) || (dyFxn.get()==NULL) || (dzFxn.get()==NULL) || (dtFxn.get()==NULL))
    {
      return TFunction<Scalar>::null();
    }
    else
    {
      return Teuchos::rcp( new VectorizedFunction<Scalar>(dxFxn,dyFxn,dzFxn,dtFxn) );
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported numComponents");
  return Teuchos::rcp((TFunction<Scalar>*) NULL);
}
//template <typename Scalar>
//TFunctionPtr<Scalar> TFunction<Scalar>::inverse() {
//  return TFunction<Scalar>::null();
//}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::heaviside(double xShift)
{
  return Teuchos::rcp( new HeavisideFunction(xShift) );
}

template <typename Scalar>
bool TFunction<Scalar>::isNull(TFunctionPtr<Scalar> f)
{
  return f.get() == NULL;
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::div()
{
  if ( isNull(x()) || isNull(y()) )
  {
    return null();
  }
  TFunctionPtr<Scalar> dxFxn = x()->dx();
  TFunctionPtr<Scalar> dyFxn = y()->dy();
  TFunctionPtr<Scalar> zFxn = z();
  if ( isNull(dxFxn) || isNull(dyFxn) )
  {
    return null();
  }
  else if ( isNull(zFxn) || isNull(zFxn->dz()) )
  {
    return dxFxn + dyFxn;
  }
  else
  {
    return dxFxn + dyFxn + zFxn->dz();
  }
}

template <typename Scalar>
void TFunction<Scalar>::CHECK_VALUES_RANK(Intrepid::FieldContainer<Scalar> &values)   // throws exception on bad values rank
{
  // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
  if (values.rank() != _rank + 2)
  {
    cout << "values has incorrect rank.\n";
    TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
  }
}

template <typename Scalar>
void TFunction<Scalar>::addToValues(Intrepid::FieldContainer<Scalar> &valuesToAddTo, BasisCachePtr basisCache)
{
  CHECK_VALUES_RANK(valuesToAddTo);
  Teuchos::Array<int> dim;
  valuesToAddTo.dimensions(dim);
  Intrepid::FieldContainer<Scalar> myValues(dim);
  this->values(myValues,basisCache);
  for (int i=0; i<myValues.size(); i++)
  {
    //cout << "otherValue = " << valuesToAddTo[i] << "; myValue = " << myValues[i] << endl;
    valuesToAddTo[i] += myValues[i];
  }
}

template <typename Scalar>
Scalar  TFunction<Scalar>::integrate(BasisCachePtr basisCache)
{
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  Intrepid::FieldContainer<Scalar> cellIntegrals(numCells);
  this->integrate(cellIntegrals, basisCache);
  Scalar sum = 0;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    sum += cellIntegrals[cellIndex];
  }
  return sum;
}

// added by Jesse to check positivity of a function
// this should only be defined for doubles, but leaving it be for the moment
// TODO: Fix for complex
template <typename Scalar>
bool TFunction<Scalar>::isPositive(BasisCachePtr basisCache)
{
  bool isPositive = true;
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  Intrepid::FieldContainer<double> fxnValues(numCells,numPoints);
  this->values(fxnValues, basisCache);

  for (int i = 0; i<fxnValues.size(); i++)
  {
    if (fxnValues[i] <= 0.0)
    {
      isPositive=false;
      break;
    }
  }
  return isPositive;
}

// this should only be defined for doubles, but leaving it be for the moment
// TODO: Fix for complex
template <typename Scalar>
bool TFunction<Scalar>::isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich, bool testVsTest)
{
  bool isPositive = true;
  bool isPositiveOnPartition = true;
  int myPartition = Teuchos::GlobalMPISession::getRank();
  vector<ElementPtr> elems = mesh->elementsInPartition(myPartition);
  for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
  {
    int cellID = (*elemIt)->cellID();
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubEnrich);

    // if we want to check positivity on uniformly spaced points
    if ((*elemIt)->numSides()==4)  // tensor product structure only works with quads
    {
      Intrepid::FieldContainer<double> origPts = basisCache->getRefCellPoints();
      int numPts1D = ceil(sqrt(origPts.dimension(0)));
      int numPts = numPts1D*numPts1D;
      Intrepid::FieldContainer<double> uniformSpacedPts(numPts,origPts.dimension(1));
      double h = 1.0/(numPts1D-1);
      int iter = 0;
      for (int i = 0; i<numPts1D; i++)
      {
        for (int j = 0; j<numPts1D; j++)
        {
          uniformSpacedPts(iter,0) = 2*h*i-1.0;
          uniformSpacedPts(iter,1) = 2*h*j-1.0;
          iter++;
        }
      }
      basisCache->setRefCellPoints(uniformSpacedPts);
    }

    bool isPositiveOnCell = this->isPositive(basisCache);
    if (!isPositiveOnCell)
    {
      isPositiveOnPartition = false;
      break;
    }
  }
  int numPositivePartitions = 1;
  if (!isPositiveOnPartition)
  {
    numPositivePartitions = 0;
  }
  int totalPositivePartitions = MPIWrapper::sum(numPositivePartitions);
  if (totalPositivePartitions<Teuchos::GlobalMPISession::getNProc())
    isPositive=false;

  return isPositive;
}


// added by Jesse - integrate over only one cell
template <typename Scalar>
Scalar TFunction<Scalar>::integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest)
{
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,cellID,testVsTest,cubatureDegreeEnrichment);
  Intrepid::FieldContainer<Scalar> cellIntegral(1);
  this->integrate(cellIntegral,basisCache);
  return cellIntegral(0);
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::cellCharacteristic(GlobalIndexType cellID)
{
  return Teuchos::rcp( new CellCharacteristicFunction(cellID) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::cellCharacteristic(set<GlobalIndexType> cellIDs)
{
  return Teuchos::rcp( new CellCharacteristicFunction(cellIDs) );
}

template <typename Scalar>
map<int, Scalar> TFunction<Scalar>::cellIntegrals(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest)
{
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDs(activeCellIDs.begin(),activeCellIDs.end());
  return cellIntegrals(cellIDs,mesh,cubatureDegreeEnrichment,testVsTest);
}

template <typename Scalar>
map<int, Scalar> TFunction<Scalar>::cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool testVsTest)
{
  int myPartition = Teuchos::GlobalMPISession::getRank();

  int numCells = cellIDs.size();
  Intrepid::FieldContainer<Scalar> integrals(numCells);
  for (int i = 0; i<numCells; i++)
  {
    int cellID = cellIDs[i];
    if (mesh->partitionForCellID(cellID) == myPartition)
    {
      integrals(i) = integrate(cellID,mesh,cubatureDegreeEnrichment,testVsTest);
    }
  }
  MPIWrapper::entryWiseSum(integrals);
  map<int,Scalar> integralMap;
  for (int i = 0; i<numCells; i++)
  {
    integralMap[cellIDs[i]] = integrals(i);
  }
  return integralMap;
}


// added by Jesse - adaptive quadrature rules
// this only works for doubles at the moment
// TODO: Fix for complex
template <typename Scalar>
Scalar TFunction<Scalar>::integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest)
{
  double integral = 0.0;
  int myPartition = Teuchos::GlobalMPISession::getRank();

  vector<ElementPtr> elems = mesh->elementsInPartition(myPartition);

  // build initial list of subcells = all elements
  vector<CacheInfo> subCellCacheInfo;
  for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
  {
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
  while (!allConverged && iter < maxIter)
  {
    allConverged = true;
    ++iter;
    // check relative error, tag subcells to refine
    double tempIntegral = 0.0;
    set<GlobalIndexType> subCellsToRefine;
    for (int i = 0; i<subCellsToCheck.size(); i++)
    {
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
      if (error > tol)
      {
        allConverged = false;
        subCellsToRefine.insert(i);
        tempIntegral += enrichedCellIntegral(0);
      }
      else
      {
        integral += enrichedCellIntegral(0);
      }
    }
    if (iter == maxIter)
    {
      integral += tempIntegral;
      cout << "maxIter reached for adaptive quadrature, returning integral estimate." << endl;
    }
    //    cout << "on iter " << iter << " with tempIntegral = " << tempIntegral << " and currrent integral = " << integral << " and " << subCellsToRefine.size() << " subcells to go. Allconverged =  " << allConverged << endl;

    // reconstruct subcell list
    vector<CacheInfo> newSubCells;
    for (set<GlobalIndexType>::iterator setIt = subCellsToRefine.begin(); setIt!=subCellsToRefine.end(); setIt++)
    {
      CacheInfo newCacheInfo = subCellsToCheck[*setIt];
      if (newCacheInfo.elemType->cellTopoPtr->getTensorialDegree() > 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorial degree > 0 not supported here.");
      }
      unsigned cellTopoKey = newCacheInfo.elemType->cellTopoPtr->getKey().first;
      switch (cellTopoKey)
      {
      case shards::Quadrilateral<4>::key:
      {
        // break into 4 subcells
        int spaceDim = 2;
        int numCells = 1; // cell-by-cell

        Intrepid::FieldContainer<double> oldNodes = newCacheInfo.subCellNodes;
        oldNodes.resize(4,spaceDim);
        Intrepid::FieldContainer<double> newCellNodes(numCells,4,spaceDim);
        double ax,ay,bx,by,cx,cy,dx,dy,ex,ey;
        ax = .5*(oldNodes(1,0)+oldNodes(0,0));
        ay = .5*(oldNodes(1,1)+oldNodes(0,1));
        bx = .5*(oldNodes(2,0)+oldNodes(1,0));
        by = .5*(oldNodes(2,1)+oldNodes(1,1));
        cx = .5*(oldNodes(3,0)+oldNodes(2,0));
        cy = .5*(oldNodes(3,1)+oldNodes(2,1));
        dx = .5*(oldNodes(3,0)+oldNodes(0,0));
        dy = .5*(oldNodes(3,1)+oldNodes(0,1));
        ex = .5*(dx+bx);
        ey = .5*(cy+ay);

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
void TFunction<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &cellIntegrals, BasisCachePtr basisCache,
                                  bool sumInto)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only integrate scalar functions.");
  int numCells = cellIntegrals.dimension(0);
  if ( !sumInto )
  {
    cellIntegrals.initialize(0);
  }

  if (this->boundaryValueOnly() && ! basisCache->isSideCache() )
  {
    int sideCount = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
      int numPoints = sideCache->getPhysicalCubaturePoints().dimension(1);
      Intrepid::FieldContainer<Scalar> values(numCells,numPoints);
      this->values(values,sideCache);

      Intrepid::FieldContainer<double> *weightedMeasures = &sideCache->getWeightedMeasures();
      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
        {
          cellIntegrals(cellIndex) += values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex);
        }
      }
    }
  }
  else
  {
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    //  cout << "integrate: basisCache->getPhysicalCubaturePoints():\n" << basisCache->getPhysicalCubaturePoints();
    Intrepid::FieldContainer<Scalar> values(numCells,numPoints);
    this->values(values,basisCache);

    Intrepid::FieldContainer<double> *weightedMeasures = &basisCache->getWeightedMeasures();
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        cellIntegrals(cellIndex) += values(cellIndex,ptIndex) * (*weightedMeasures)(cellIndex,ptIndex);
      }
    }
  }
}

// takes integral of jump over entire INTERIOR skeleton
template <typename Scalar>
Scalar TFunction<Scalar>::integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment)
{
  Scalar integral = 0.0;
  vector<ElementPtr> elems = mesh->activeElements();
  for (vector<ElementPtr>::iterator elemIt=elems.begin(); elemIt!=elems.end(); elemIt++)
  {
    ElementPtr elem = *elemIt;
    int numSides = elem->numSides();
    for (int sideIndex = 0; sideIndex < numSides; sideIndex++)
    {
      integral+= this->integralOfJump(mesh,elem->cellID(),sideIndex,cubatureDegreeEnrichment);
    }
  }
  return integral;
}

template <typename Scalar>
Scalar TFunction<Scalar>::integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment)
{
  // for boundaries, the jump is 0
  if (mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex))
  {
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
Scalar TFunction<Scalar>::integrate(MeshPtr mesh, int cubatureDegreeEnrichment, bool testVsTest, bool requireSideCache,
                                    bool spatialSidesOnly)
{
  Scalar integral = 0;

  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, *cellIDIt, testVsTest, cubatureDegreeEnrichment);
    if ( this->boundaryValueOnly() )
    {
      ElementTypePtr elemType = mesh->getElementType(*cellIDIt);
      int numSides = elemType->cellTopoPtr->getSideCount();

      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        if (spatialSidesOnly && !elemType->cellTopoPtr->sideIsSpatial(sideOrdinal)) continue; // skip non-spatial sides if spatialSidesOnly is true
        Scalar sideIntegral = this->integrate(basisCache->getSideBasisCache(sideOrdinal));
        integral += sideIntegral;
      }
    }
    else
    {
      integral += this->integrate(basisCache);
    }
  }
  return MPIWrapper::sum(integral);
}

template <typename Scalar>
double TFunction<Scalar>::l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool spatialSidesOnly)
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( this, false );
  bool testVsTest = false, requireSideCaches = false;
  return sqrt( abs((thisPtr * thisPtr)->integrate(mesh, cubatureDegreeEnrichment, testVsTest, requireSideCaches, spatialSidesOnly)) );
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
template <typename Scalar>
void TFunction<Scalar>::scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache)
{
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
}

// divide values by this function (supported only when this is a scalar)
template <typename Scalar>
void TFunction<Scalar>::scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache)
{
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
// should only happen with double valued functions
// TODO: throw error for complex
template <typename Scalar>
void TFunction<Scalar>::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  // basisValues has dimensions (C,F,P,...)
  //  cout << "scalarMultiplyBasisValues: basisValues:\n" << basisValues;
  scalarModifyBasisValues(basisValues,basisCache,MULTIPLY);
  //  cout << "scalarMultiplyBasisValues: modified basisValues:\n" << basisValues;
}

// divide values by this function (supported only when this is a scalar)
// should only happen with double valued functions
// TODO: throw error for complex
template <typename Scalar>
void TFunction<Scalar>::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  // basisValues has dimensions (C,F,P,...)
  scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
}

template <typename Scalar>
void TFunction<Scalar>::valuesDottedWithTensor(Intrepid::FieldContainer<Scalar> &values,
    TFunctionPtr<Scalar> tensorFunctionOfLikeRank,
    BasisCachePtr basisCache)
{
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
  for (int d=0; d<_rank; d++)
  {
    tensorValueIndex[d+2] = spaceDim;
  }

  Intrepid::FieldContainer<Scalar> myTensorValues(tensorValueIndex);
  this->values(myTensorValues,basisCache);
  Intrepid::FieldContainer<Scalar> otherTensorValues(tensorValueIndex);
  tensorFunctionOfLikeRank->values(otherTensorValues,basisCache);

  //  cout << "myTensorValues:\n" << myTensorValues;
  //  cout << "otherTensorValues:\n" << otherTensorValues;

  // clear out the spatial indices of tensorValueIndex so we can use it as index
  for (int d=0; d<_rank; d++)
  {
    tensorValueIndex[d+2] = 0;
  }

  int entriesPerPoint = 1;
  for (int d=0; d<_rank; d++)
  {
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    tensorValueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      tensorValueIndex[1] = ptIndex;
      Scalar *myValue = &myTensorValues[ myTensorValues.getEnumeration(tensorValueIndex) ];
      Scalar *otherValue = &otherTensorValues[ otherTensorValues.getEnumeration(tensorValueIndex) ];
      Scalar *value = &values(cellIndex,ptIndex);

      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++)
      {
        *value += *myValue * *otherValue;
        //        cout << "myValue: " << *myValue << "; otherValue: " << *otherValue << endl;
        myValue++;
        otherValue++;
      }
    }
  }
}

template <typename Scalar>
void TFunction<Scalar>::scalarModifyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
    FunctionModificationType modType)
{
  TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getSpaceDim();

  Intrepid::FieldContainer<Scalar> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);

  Teuchos::Array<int> valueIndex(values.rank());

  int entriesPerPoint = 1;
  for (int d=0; d < values.rank()-2; d++)    // -2 for numCells, numPoints indices
  {
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    valueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      valueIndex[1] = ptIndex;
      Scalar *value = &values[ values.getEnumeration(valueIndex) ];
      Scalar scalarValue = scalarValues(cellIndex,ptIndex);
      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++)
      {
        if (modType == MULTIPLY)
        {
          *value++ *= scalarValue;
        }
        else if (modType == DIVIDE)
        {
          *value++ /= scalarValue;
        }
      }
    }
  }
}

// Should only work for doubles
// TODO: Throw exception for complex
template <typename Scalar>
void TFunction<Scalar>::scalarModifyBasisValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
    FunctionModificationType modType)
{
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
  for (int d=0; d<values.rank()-3; d++)    // -3 for numCells, numFields, numPoints indices
  {
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    valueIndex[0] = cellIndex;
    for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
    {
      valueIndex[1] = fieldIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        valueIndex[2] = ptIndex;
        double scalarValue = scalarValues(cellIndex,ptIndex);
        double *value = &values[ values.getEnumeration(valueIndex) ];
        for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++)
        {
          if (modType == MULTIPLY)
          {
            *value++ *= scalarValue;
          }
          else if (modType == DIVIDE)
          {
            *value++ /= scalarValue;
          }
        }
      }
    }
  }
  //  cout << "scalarModifyBasisValues: values:\n" << values;
}

// Not sure if this will work for complex
// TODO: Throw exception for complex
template <typename Scalar>
void TFunction<Scalar>::writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath)
{
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...

  BasisCachePtr basisCache;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh, true) );
    CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
    int numSides = cellTopo->getSideCount();

    Intrepid::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      GlobalIndexType cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    int num1DPts = 15;
    Intrepid::FieldContainer<double> refPoints(num1DPts,1);
    for (int i=0; i < num1DPts; i++)
    {
      double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
      refPoints(i,0) = x;
    }

    for (int sideIndex=0; sideIndex < numSides; sideIndex++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
      sideBasisCache->setRefCellPoints(refPoints);
      int numCubPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);


      Intrepid::FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
      this->values(computedValues,sideBasisCache);

      // NOW loop over all cells to write solution to file
      for (int cellIndex=0; cellIndex < numCells; cellIndex++)
      {
        Intrepid::FieldContainer<double> cellParities = mesh->cellSideParitiesForCell( cellIDs[cellIndex] );
        for (int pointIndex = 0; pointIndex < numCubPoints; pointIndex++)
        {
          for (int dimInd=0; dimInd<spaceDim; dimInd++)
          {
            fout << (basisCache->getSideBasisCache(sideIndex)->getPhysicalCubaturePoints())(cellIndex,pointIndex,dimInd) << " ";
          }
          fout << computedValues(cellIndex,pointIndex) << endl;
        }
        // insert NaN for matlab to plot discontinuities - WILL NOT WORK IN 3D
        for (int dimInd=0; dimInd<spaceDim; dimInd++)
        {
          fout << "NaN" << " ";
        }
        fout << "NaN" << endl;
      }
    }
  }
  fout.close();
}

// Not sure if this will work for complex
// TODO: Throw exception for complex
template <typename Scalar>
void TFunction<Scalar>::writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath)
{
  // MATLAB format, supports scalar functions defined inside 2D volume right now...
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = 15;

  int numPoints = num1DPts * num1DPts;
  Intrepid::FieldContainer<double> refPoints(numPoints,spaceDim);
  for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
  {
    for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
    {
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
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++)   //thru quads/triangles/etc
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh) );
    basisCache->setRefCellPoints(refPoints);

    Intrepid::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      int cellID = mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, false); // false: don't create side cache

    Intrepid::FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();

    Intrepid::FieldContainer<double> computedValues(numCells,numPoints);
    this->values(computedValues, basisCache);

    // NOW loop over all cells to write solution to file
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
    {
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
      {
        int ptIndex = xPointIndex*num1DPts + yPointIndex;
        for (int cellIndex=0; cellIndex < numCells; cellIndex++)
        {
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
TFunctionPtr<Scalar> TFunction<Scalar>::constant(Scalar value)
{
  return Teuchos::rcp( new ConstantScalarFunction<Scalar>(value) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::constant(vector<Scalar> &value)
{
  return Teuchos::rcp( new ConstantVectorFunction<Scalar>(value) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::meshBoundaryCharacteristic()
{
  // 1 on mesh boundary, 0 elsewhere
  return Teuchos::rcp( new MeshBoundaryCharacteristicFunction );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::h()
{
  return Teuchos::rcp( new hFunction );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::meshSkeletonCharacteristic()
{
  // 1 on mesh skeleton, 0 elsewhere
  return Teuchos::rcp( new MeshSkeletonCharacteristicFunction );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::normal()   // unit outward-facing normal on each element boundary
{
  static TFunctionPtr<double> _normal = Teuchos::rcp( new UnitNormalFunction );
  return _normal;
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::normal_1D()   // unit outward-facing normal on each element boundary
{
  static TFunctionPtr<double> _normal_1D = Teuchos::rcp( new UnitNormalFunction(0) );
  return _normal_1D;
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::normalSpaceTime()   // unit outward-facing normal on each element boundary
{
  static TFunctionPtr<double> _normalSpaceTime = Teuchos::rcp( new UnitNormalFunction(-1,true) );
  return _normalSpaceTime;
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::sideParity()   // canonical direction on boundary (used for defining fluxes)
{
  static TFunctionPtr<double> _sideParity = Teuchos::rcp( new SideParityFunction );
  return _sideParity;
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::polarize(TFunctionPtr<Scalar> f)
{
  return Teuchos::rcp( new PolarizedFunction<Scalar>(f) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::restrictToCellBoundary(TFunctionPtr<Scalar> f)
{
  return Teuchos::rcp( new CellBoundaryRestrictedFunction<Scalar>(f) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::solution(VarPtr var, TSolutionPtr<Scalar> soln)
{
  return Teuchos::rcp( new SimpleSolutionFunction<Scalar>(var, soln) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  return Teuchos::rcp( new VectorizedFunction<Scalar>(f1,f2) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3)
{
  return Teuchos::rcp( new VectorizedFunction<Scalar>(f1,f2,f3) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::null()
{
  static TFunctionPtr<Scalar> _null = Teuchos::rcp( (TFunction<Scalar>*) NULL );
  return _null;
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::xn(int n)
{
  return Teuchos::rcp( new Xn(n) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::yn(int n)
{
  return Teuchos::rcp( new Yn(n) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::zn(int n)
{
  return Teuchos::rcp( new Zn(n) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::tn(int n)
{
  return Teuchos::rcp( new Tn(n) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::xPart(TFunctionPtr<Scalar> vectorFxn)
{
  return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 0) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::yPart(TFunctionPtr<Scalar> vectorFxn)
{
  return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 1) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::zPart(TFunctionPtr<Scalar> vectorFxn)
{
  return Teuchos::rcp( new ComponentFunction<Scalar>(vectorFxn, 2) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::zero(int rank)
{
  static TFunctionPtr<double> _zero = Teuchos::rcp( new ConstantScalarFunction<Scalar>(0.0) );
  if (rank==0)
  {
    return _zero;
  }
  else
  {
    TFunctionPtr<double> zeroTensor = _zero;
    for (int i=0; i<rank; i++)
    {
      // THIS ASSUMES 2D--3D would be TFunction<Scalar>::vectorize(zeroTensor, zeroTensor, zeroTensor)...
      zeroTensor = TFunction<double>::vectorize(zeroTensor, zeroTensor);
    }
    return zeroTensor;
  }
}

// this is liable to be a bit slow!!
class ComposedFunction : public TFunction<double>
{
  TFunctionPtr<double> _f, _arg_g;
public:
  ComposedFunction(TFunctionPtr<double> f, TFunctionPtr<double> arg_g) : TFunction<double>(f->rank())
  {
    _f = f;
    _arg_g = arg_g;
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getSpaceDim();
    Intrepid::FieldContainer<double> fArgPoints(numCells,numPoints,spaceDim);
    if (spaceDim==1)   // special case: arg_g is then reasonably scalar-valued
    {
      fArgPoints.resize(numCells,numPoints);
    }
    _arg_g->values(fArgPoints,basisCache);
    if (spaceDim==1)
    {
      fArgPoints.resize(numCells,numPoints,spaceDim);
    }
    BasisCachePtr fArgCache = Teuchos::rcp( new PhysicalPointCache(fArgPoints) );
    _f->values(values, fArgCache);
  }
  TFunctionPtr<double> dx()
  {
    if (isNull(_f->dx()) || isNull(_arg_g->dx()))
    {
      return TFunction<double>::null();
    }
    // chain rule:
    return _arg_g->dx() * TFunction<double>::composedFunction(_f->dx(),_arg_g);
  }
  TFunctionPtr<double> dy()
  {
    if (isNull(_f->dy()) || isNull(_arg_g->dy()))
    {
      return TFunction<double>::null();
    }
    // chain rule:
    return _arg_g->dy() * TFunction<double>::composedFunction(_f->dy(),_arg_g);
  }
  TFunctionPtr<double> dz()
  {
    if (isNull(_f->dz()) || isNull(_arg_g->dz()))
    {
      return TFunction<double>::null();
    }
    // chain rule:
    return _arg_g->dz() * TFunction<double>::composedFunction(_f->dz(),_arg_g);
  }
};

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::composedFunction( TFunctionPtr<double> f, TFunctionPtr<double> arg_g)
{
  return Teuchos::rcp( new ComposedFunction(f,arg_g) );
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  if (f1->isZero() || f2->isZero())
  {
    if ( f1->rank() == f2->rank() )
    {
      return TFunction<Scalar>::zero();
    }
    else if ((f1->rank() == 0) || (f2->rank() == 0))
    {
      int result_rank = f1->rank() + f2->rank();
      return TFunction<Scalar>::zero(result_rank);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"functions have incompatible rank for product.");
    }
  }
  return Teuchos::rcp( new ProductFunction<Scalar>(f1,f2) );
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> scalarDivisor)
{
  if ( f1->isZero() )
  {
    return TFunction<Scalar>::zero(f1->rank());
  }
  return Teuchos::rcp( new QuotientFunction<Scalar>(f1,scalarDivisor) );
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, Scalar divisor)
{
  return f1 / TFunction<Scalar>::constant(divisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, int divisor)
{
  return f1 / Scalar(divisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(Scalar value, TFunctionPtr<Scalar> scalarDivisor)
{
  return TFunction<Scalar>::constant(value) / scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(int value, TFunctionPtr<Scalar> scalarDivisor)
{
  return Scalar(value) / scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(Scalar weight, TFunctionPtr<Scalar> f)
{
  return TFunction<Scalar>::constant(weight) * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, Scalar weight)
{
  return weight * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(int weight, TFunctionPtr<Scalar> f)
{
  return Scalar(weight) * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, int weight)
{
  return Scalar(weight) * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(vector<Scalar> weight, TFunctionPtr<Scalar> f)
{
  return TFunction<Scalar>::constant(weight) * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, vector<Scalar> weight)
{
  return weight * f;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  if ( f1->isZero() )
  {
    return f2;
  }
  if ( f2->isZero() )
  {
    return f1;
  }
  return Teuchos::rcp( new SumFunction<Scalar>(f1, f2) );
}

template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, Scalar value)
{
  return f1 + TFunction<Scalar>::constant(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator+(Scalar value, TFunctionPtr<Scalar> f1)
{
  return f1 + TFunction<Scalar>::constant(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, int value)
{
  return f1 + Scalar(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator+(int value, TFunctionPtr<Scalar> f1)
{
  return f1 + Scalar(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  return f1 + -f2;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, Scalar value)
{
  return f1 - TFunction<Scalar>::constant(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(Scalar value, TFunctionPtr<Scalar> f1)
{
  return TFunction<Scalar>::constant(value) - f1;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, int value)
{
  return f1 - Scalar(value);
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(int value, TFunctionPtr<Scalar> f1)
{
  return Scalar(value) - f1;
}

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f)
{
  return -1.0 * f;
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::min(TFunctionPtr<double> f1, TFunctionPtr<double> f2)
{
  return Teuchos::rcp( new MinFunction(f1, f2) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::min(TFunctionPtr<double> f1, double value)
{
  return Teuchos::rcp( new MinFunction(f1, TFunction<double>::constant(value)) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::min(double value, TFunctionPtr<double> f2)
{
  return Teuchos::rcp( new MinFunction(f2, TFunction<double>::constant(value)) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::max(TFunctionPtr<double> f1, TFunctionPtr<double> f2)
{
  return Teuchos::rcp( new MaxFunction(f1, f2) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::max(TFunctionPtr<double> f1, double value)
{
  return Teuchos::rcp( new MaxFunction(f1, TFunction<double>::constant(value)) );
}

template <typename Scalar>
TFunctionPtr<double> TFunction<Scalar>::max(double value, TFunctionPtr<double> f2)
{
  return Teuchos::rcp( new MaxFunction(f2, TFunction<double>::constant(value)) );
}

template class TFunction<double>;

template TFunctionPtr<double> operator*(TFunctionPtr<double> f1, TFunctionPtr<double> f2);
template TFunctionPtr<double> operator/(TFunctionPtr<double> f1, TFunctionPtr<double> scalarDivisor);
template TFunctionPtr<double> operator/(TFunctionPtr<double> f1, double divisor);
template TFunctionPtr<double> operator/(double value, TFunctionPtr<double> scalarDivisor);
template TFunctionPtr<double> operator/(TFunctionPtr<double> f1, int divisor);
template TFunctionPtr<double> operator/(int value, TFunctionPtr<double> scalarDivisor);

template TFunctionPtr<double> operator*(double weight, TFunctionPtr<double> f);
template TFunctionPtr<double> operator*(TFunctionPtr<double> f, double weight);
template TFunctionPtr<double> operator*(int weight, TFunctionPtr<double> f);
template TFunctionPtr<double> operator*(TFunctionPtr<double> f, int weight);
template TFunctionPtr<double> operator*(vector<double> weight, TFunctionPtr<double> f);
template TFunctionPtr<double> operator*(TFunctionPtr<double> f, vector<double> weight);

template TFunctionPtr<double> operator+(TFunctionPtr<double> f1, TFunctionPtr<double> f2);
template TFunctionPtr<double> operator+(TFunctionPtr<double> f1, double value);
template TFunctionPtr<double> operator+(double value, TFunctionPtr<double> f1);
template TFunctionPtr<double> operator+(TFunctionPtr<double> f1, int value);
template TFunctionPtr<double> operator+(int value, TFunctionPtr<double> f1);

template TFunctionPtr<double> operator-(TFunctionPtr<double> f1, TFunctionPtr<double> f2);
template TFunctionPtr<double> operator-(TFunctionPtr<double> f1, double value);
template TFunctionPtr<double> operator-(double value, TFunctionPtr<double> f1);
template TFunctionPtr<double> operator-(TFunctionPtr<double> f1, int value);
template TFunctionPtr<double> operator-(int value, TFunctionPtr<double> f1);

template TFunctionPtr<double> operator-(TFunctionPtr<double> f);
}
