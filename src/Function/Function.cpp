//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

//
//  Function.cpp
//  Camellia
//

#include "Function.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CellCharacteristicFunction.h"
#include "ConstantScalarFunction.h"
#include "ConstantVectorFunction.h"
#include "CubatureFactory.h"
#include "GlobalDofAssignment.h"
#include "hFunction.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "MinMaxFunctions.h"
#include "MonomialFunctions.h"
#include "NeighborFunction.h"
#include "PhysicalPointCache.h"
#include "PolarizedFunction.h"
#include "ProductFunction.h"
#include "QuotientFunction.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "SimpleSolutionFunction.h"
#include "SimpleVectorFunction.h"
#include "SideParityFunction.h"
#include "Solution.h"
#include "SqrtFunction.h"
#include "SumFunction.h"
#include "TrigFunctions.h"
#include "UnitNormalFunction.h"
#include "Var.h"
#include "VectorizedFunction.h"

#include "Intrepid_CellTools.hpp"
#include "Teuchos_GlobalMPISession.hpp"

namespace Camellia
{
  std::map<std::string,int> callsToFunctionsWithoutDevicePathImplementations;
  
  std::string functionsWithoutDevicePathImplementationsReport()
  {
    std::multimap<int,std::string> sortedByCount;
    for (auto entry : callsToFunctionsWithoutDevicePathImplementations)
    {
      sortedByCount.insert(std::make_pair(entry.second, entry.first));
    }
    
    ostringstream report;
    report << "These are the functions that were called that do not have their own device-path values() implementation:\n";
    for (auto entry : sortedByCount)
    {
      report << entry.second << " (" << entry.first << ")\n";
    }
    return report.str();
  }
  
  
// for adaptive quadrature
struct CacheInfo
{
  ElementTypePtr elemType;
  GlobalIndexType cellID;
  Camellia::FieldContainer<double> subCellNodes;
};

// private class ComponentFunction
template <typename Scalar>
class ComponentFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _vectorFxn;
  int _component;
public:
  ComponentFunction(TFunctionPtr<Scalar> vectorFunction, int componentIndex)
  :
  TFunction<Scalar>("ComponentFunction")
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
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    // note this allocation.  There might be ways of reusing memory here, if we had a slightly richer API.
    int spaceDim = basisCache->getSpaceDim();
    Teuchos::Array<int> dim;
    values.dimensions(dim);
    dim.push_back(spaceDim);

    Camellia::FieldContainer<double> vectorValues(dim);
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
  CellBoundaryRestrictedFunction(TFunctionPtr<Scalar> fxn) : TFunction<Scalar>("CellBoundaryRestrictedFunction",fxn->rank())
  {
    _fxn = fxn;
  }

  bool boundaryValueOnly()
  {
    return true;
  }
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    _fxn->values(values, basisCache);
  }
};

class HeavisideFunction : public SimpleFunction<double>
{
  double _xShift;
public:
  HeavisideFunction(double xShift=0.0)
  :
  SimpleFunction<double>("HeavisideFunction")
  {
    _xShift = xShift;
  }
  double value(double x)
  {
    return (x < _xShift) ? 0.0 : 1.0;
  }
  std::string displayString()
  {
    ostringstream name;
    name << "H_x(" << _xShift << ")";
    return name.str();
  }
};

class HeavisideFunctionY : public SimpleFunction<double>
{
  double _yShift;
public:
  HeavisideFunctionY(double yShift=0.0)
  :
  SimpleFunction<double>("HeavisideFunctionY")
  {
    _yShift = yShift;
  }
  double value(double x, double y)
  {
    return (y < _yShift) ? 0.0 : 1.0;
  }
  std::string displayString()
  {
    ostringstream name;
    name << "H_y(" << _yShift << ")";
    return name.str();
  }
};

class MeshBoundaryCharacteristicFunction : public TFunction<double>
{
public:
  MeshBoundaryCharacteristicFunction() : TFunction<double>("MeshBoundaryCharacteristicFunction",0)
  {
  }
  bool boundaryValueOnly()
  {
    return true;
  }
  string displayString()
  {
    return "|_{\\Gamma}";
  }
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
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

class MeshSkeletonCharacteristicFunction : public SimpleFunction<double>
{
public:
  MeshSkeletonCharacteristicFunction()
  :
  SimpleFunction<double>("MeshSkeletonCharacteristicFunction")
  {}
  bool boundaryValueOnly()
  {
    return true;
  }
  string displayString()
  {
    return "|_{\\Gamma_h}";
  }
  double value(double x)
  {
    return 1.0;
  }
};

template <typename Scalar>
TFunction<Scalar>::TFunction() : TFunction("(missing class name)",0) {}
  
template <typename Scalar>
TFunction<Scalar>::TFunction(const std::string &className)
{
  _rank = 0;
  _displayString = this->displayString();
  _time = 0;
  _className = className;
}

template <typename Scalar>
TFunction<Scalar>::TFunction(int rank) : TFunction("(missing class name)",rank) {}
  
template <typename Scalar>
TFunction<Scalar>::TFunction(const std::string &className, int rank)
{
  _rank = rank;
  _displayString = this->displayString();
  _time = 0;
  _className = className;
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
void TFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache)
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
    case Camellia::OP_DXDX:
      return f->dx()->dx();
    case Camellia::OP_DXDY:
      return f->dx()->dy();
    case Camellia::OP_DXDZ:
      return f->dx()->dz();
    case Camellia::OP_DYDY:
      return f->dy()->dy();
    case Camellia::OP_DYDZ:
      return f->dy()->dz();
    case Camellia::OP_DZDZ:
      return f->dz()->dz();
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
  Camellia::FieldContainer<Scalar> diffs_squared(numCells);
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
  Camellia::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getDimension() != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 1D if only x is provided.");
  }

  physPoint(0,0) = x;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint, true);
  if (cellIDs.size() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
  }
  Scalar value = evaluateAtMeshPoint(mesh,cellIDs[0],physPoint);
  return MPIWrapper::sum(value);
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(MeshPtr mesh, double x, double y)
{
  int spaceDim = 2;
  Camellia::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getDimension() != spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 2D if (x,y) is provided.");
  }

  physPoint(0,0) = x;
  physPoint(0,1) = y;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint, true);
  if (cellIDs.size() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
  }
  Scalar value = evaluateAtMeshPoint(mesh,cellIDs[0],physPoint);
  return MPIWrapper::sum(value);
}

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(MeshPtr mesh, double x, double y, double z)
{
  int spaceDim = 3;
  Camellia::FieldContainer<double> physPoint(1,spaceDim);

  if (this->rank() != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires a rank 0 Function.");
  }
  if (mesh->getTopology()->getDimension() != spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TFunction<Scalar>::evaluate requires mesh to be 3D if (x,y,z) is provided.");
  }

  physPoint(0,0) = x;
  physPoint(0,1) = y;
  physPoint(0,2) = z;

  vector<GlobalIndexType> cellIDs = mesh->cellIDsForPoints(physPoint, true);
  if (cellIDs.size() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point not found in mesh");
  }
  Scalar value = evaluateAtMeshPoint(mesh,cellIDs[0],physPoint);
  return MPIWrapper::sum(value);
}
  
  template <typename Scalar>
  Scalar TFunction<Scalar>::evaluateAtMeshPoint(MeshPtr mesh, GlobalIndexType cellID, Camellia::FieldContainer<double> &physicalPoint)
  {
    Scalar value;
    int spaceDim = mesh->getDimension();
    if (cellID != -1)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      Camellia::FieldContainer<double> refPoint(1,1,spaceDim);
      
      physicalPoint.resize(1,1,spaceDim);
      CamelliaCellTools::mapToReferenceFrame(refPoint, physicalPoint, mesh->getTopology(), cellID, basisCache->cubatureDegree());
      refPoint.resize(1,spaceDim);
      basisCache->setRefCellPoints(refPoint);
      
      Camellia::FieldContainer<Scalar> valueFC(1,1); // (C,P)
      this->values(valueFC,basisCache);
      value = valueFC[0];
    }
    else
    {
      value = 0;
    }
    return value;
  }

template <typename Scalar>
Scalar TFunction<Scalar>::evaluate(double x)
{
  Camellia::FieldContainer<Scalar> value(1,1); // (C,P)
  Camellia::FieldContainer<double> physPoint(1,1,1);

  Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
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
  Camellia::FieldContainer<Scalar> value(1,1);
  Camellia::FieldContainer<double> physPoint(1,1,2);
  Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
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
  Camellia::FieldContainer<Scalar> value(1,1);
  Camellia::FieldContainer<double> physPoint(1,1,3);
  Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
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
  TFunctionPtr<Scalar> TFunction<Scalar>::evaluateFunctionAt(TFunctionPtr<Scalar> f,
                                                             const map<int, TFunctionPtr<Scalar> > &valueMap)
  {
    if (f->isAbstract())
    {
      return f->evaluateAt(valueMap);
    }
    else
    {
      return f;
    }
  }
  
  template <typename Scalar>
  TFunctionPtr<Scalar> TFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
  {
    // this implementation should never be called, but if it is, there are two distinct errors that might have been made.
    if (this->isAbstract())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Abstract Function subclasses must implement evaluateAt()!");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "evaluateAt() member function should not be called on non-abstract functions.  Use the static evaluateAt(f,soln) version instead!");
    }
  }
  
template <typename Scalar>
size_t TFunction<Scalar>::getCellDataSize(GlobalIndexType cellID)
{
  // size in bytes
  auto members = this->memberFunctions();
  for (auto &f : members)
  {
    if (f == Teuchos::null)
    {
      std::cout << "ERROR: Function " << this->displayString() << " return a null FunctionPtr among its members in memberFunctions()...\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: Function " << this->displayString() << " return a null FunctionPtr among its members in memberFunctions()...");
    }
  }
  return getCellDataSize(members, cellID);
}

template <typename Scalar>
std::vector<GlobalIndexType> TFunction<Scalar>::localCellsWithNegativeValues(MeshPtr mesh, int cubatureDegreeEnrichment)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only compute the minimum value of scalar functions.");
  
  bool testVsTest = false;
  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
  
  vector<GlobalIndexType> myCellsWithNegativeValues;
  for (GlobalIndexType cellID : cellIDs)
  {
    double cellMin = numeric_limits<double>::max();
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    Camellia::FieldContainer<Scalar> fxnValues(numCells,numPoints);
    this->values(fxnValues,basisCache);
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        cellMin = std::min(cellMin,fxnValues(cellIndex,ptIndex));
      }
    }
    
    if (!basisCache->isSideCache())
    {
      int numSides = basisCache->cellTopology()->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
        int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
        int numSidePoints = sideCache->getPhysicalCubaturePoints().dimension(1);
        Camellia::FieldContainer<Scalar> fxnValues(numCells,numSidePoints);
        this->values(fxnValues,sideCache);
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          for (int ptIndex=0; ptIndex<numSidePoints; ptIndex++)
          {
            cellMin = std::min(cellMin,fxnValues(cellIndex,ptIndex));
          }
        }
      }
    }
    
    if (cellMin < 0.0)
    {
      myCellsWithNegativeValues.push_back(cellID);
    }
  }
  
  return myCellsWithNegativeValues;
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::neighborFunction(TFunctionPtr<Scalar> f, MeshPtr mesh)
{
  return Teuchos::rcp(new NeighborFunction<Scalar>(f,mesh));
}
  
template <typename Scalar>
void TFunction<Scalar>::packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  auto members = this->memberFunctions();
  packCellData(members, cellID, cellData, bufferLength);
}
  
template <typename Scalar>
size_t TFunction<Scalar>::unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
  auto members = this->memberFunctions();
  return unpackCellData(members, cellID, cellData, bufferLength);
}

template <typename Scalar>
size_t TFunction<Scalar>::getCellDataSize(const std::vector<FunctionPtr> &functions, GlobalIndexType cellID)
{
  size_t total = 0;
  for (auto &f : functions)
  {
    if (f == Teuchos::null)
    {
      std::cout << "getCellDataSize(functions, cellID) called with a null function in functions.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getCellDataSize(functions, cellID) called with a null function in functions.");
    }
    total += f->getCellDataSize(cellID);
  }
  return total;
}
  
template <typename Scalar>
void TFunction<Scalar>::packCellData(const std::vector<FunctionPtr> &functions, GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  char *dataPtr = cellData;
  for (auto &f : functions)
  {
    size_t cellDataSize = f->getCellDataSize(cellID);
    if (cellDataSize > bufferLength)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "bufferLength too small");
    }
    f->packCellData(cellID, dataPtr, cellDataSize);
    dataPtr += cellDataSize;
    bufferLength -= cellDataSize;
  }
}

template <typename Scalar>
size_t TFunction<Scalar>::unpackCellData(const std::vector<FunctionPtr> &functions, GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
  const char *dataPtr = cellData;
  size_t totalBytesConsumed = 0;
  for (auto &f : functions)
  {
    size_t bytesConsumed = f->unpackCellData(cellID, dataPtr, bufferLength);
    dataPtr            += bytesConsumed;
    bufferLength       -= bytesConsumed;
    totalBytesConsumed += bytesConsumed;
  }
  return totalBytesConsumed;
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
  TFunctionPtr<Scalar> TFunction<Scalar>::spatialComponent(int d)
  {
    switch (d) {
      case 1:
        return x();
        break;
      case 2:
        return y();
        break;
      case 3:
        return z();
        break;
        
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported spatial component number.");
        break;
    }
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
TFunctionPtr<Scalar> TFunction<Scalar>::di(int i) // 1-based: 1 for dx, 2 for dy(), 3 for dz()
{
  switch (i) {
    case 1: return dx();
    case 2: return dy();
    case 3: return dz();
    default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported component number");
  }
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
TFunctionPtr<double> TFunction<Scalar>::heavisideY(double yShift)
{
  return Teuchos::rcp( new HeavisideFunctionY(yShift) );
}

  template <typename Scalar>
  TFunctionPtr<Scalar> TFunction<Scalar>::hessian(int numComponents)
  {
    return this->grad(numComponents)->grad(numComponents);
  }

  template <typename Scalar>
  TLinearTermPtr<Scalar> TFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
  {
    if (this->isAbstract())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Abstract functions must implement Jacobian");
    }
    return Teuchos::rcp( new TLinearTerm<Scalar> );
  }

  template <typename Scalar>
  bool TFunction<Scalar>::isAbstract()
  {
    // this is abstract if any of its members is abstract; otherwise, concrete
    auto members = this->memberFunctions();
    for (auto member : members)
    {
      if (member->isAbstract())
      {
        return true;
      }
    }
    return false;
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
void TFunction<Scalar>::CHECK_VALUES_RANK(Camellia::FieldContainer<Scalar> &values)   // throws exception on bad values rank
{
  // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
  if (values.rank() != _rank + 2)
  {
    cout << "values has incorrect rank.\n";
    TEUCHOS_TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
  }
}

template <typename Scalar>
void TFunction<Scalar>::addToValues(Camellia::FieldContainer<Scalar> &valuesToAddTo, BasisCachePtr basisCache)
{
  CHECK_VALUES_RANK(valuesToAddTo);
  Teuchos::Array<int> dim;
  valuesToAddTo.dimensions(dim);
  Camellia::FieldContainer<Scalar> myValues(dim);
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
  Camellia::FieldContainer<Scalar> cellIntegrals(numCells);
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
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  Camellia::FieldContainer<double> fxnValues(numCells,numPoints);
  this->values(fxnValues, basisCache);

//  print("Cells on which we're checking positivity", basisCache->cellIDs());
//  std::cout << "Function values on those cells: \n";
//  std::cout << fxnValues;
  
  for (int i = 0; i<fxnValues.size(); i++)
  {
    if (fxnValues[i] <= 0.0)
    {
      return false;
    }
  }
  
  // since we're using quadrature points (which do not touch the sides of the cell),
  // good to check the sides as well.  Better still would be to explicitly include
  // quadrature points for each subcell topology in the points that we check, all
  // the way down to vertices.
  if (!basisCache->isSideCache())
  {
    int numSides = basisCache->cellTopology()->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
    {
      auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
      if (! this->isPositive(sideCache) )
      {
        return false;
      }
    }
  }
  
  // if we get here, no point has bee negative or zero.
  return true;
}

// this should only be defined for doubles, but leaving it be for the moment
// TODO: Fix for complex
template <typename Scalar>
bool TFunction<Scalar>::isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich, bool testVsTest)
{
  bool isPositive = true;
  bool isPositiveOnPartition = true;
  auto cellIDs = mesh->cellIDsInPartition();
  for (auto cellID : cellIDs)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubEnrich);

//    // if we want to check positivity on uniformly spaced points
//    if (basisCache->cellTopology()->getSideCount()==4)  // tensor product structure only works with quads
//    {
//      Camellia::FieldContainer<double> origPts = basisCache->getRefCellPoints();
//      int numPts1D = ceil(sqrt(origPts.dimension(0)));
//      int numPts = numPts1D*numPts1D;
//      Camellia::FieldContainer<double> uniformSpacedPts(numPts,origPts.dimension(1));
//      double h = 1.0/(numPts1D-1);
//      int iter = 0;
//      for (int i = 0; i<numPts1D; i++)
//      {
//        for (int j = 0; j<numPts1D; j++)
//        {
//          uniformSpacedPts(iter,0) = 2*h*i-1.0;
//          uniformSpacedPts(iter,1) = 2*h*j-1.0;
//          iter++;
//        }
//      }
//      basisCache->setRefCellPoints(uniformSpacedPts);
//    }

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
  Camellia::FieldContainer<Scalar> cellIntegral(1);
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
void TFunction<Scalar>::importDataForOffRankCells(MeshPtr mesh, const std::set<GlobalIndexType> &offRankCells)
{
  // code below is adapted from Solution::importSolutionForOffRankCells
  Epetra_CommPtr Comm = mesh->Comm();
  int rank = Comm->MyPID();
  
  // we require that all the cellIDs be locally known in terms of the geometry
  // (for distributed MeshTopology, this basically means that we only allow importing
  // Solution coefficients in the halo of the cells owned by this rank.)
  const set<IndexType>* locallyKnownActiveCells = &mesh->getTopology()->getLocallyKnownActiveCellIndices();
  for (GlobalIndexType cellID : offRankCells)
  {
    if (locallyKnownActiveCells->find(cellID) == locallyKnownActiveCells->end())
    {
      cout << "Requested cell " << cellID << " is not locally known on rank " << rank << endl;
      print("locally known cells", *locallyKnownActiveCells);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "importDataForOffRankCells requires cells to have locally available geometry.");
    }
  }
  
  // it appears to be important that the requests be sorted by MPI rank number
  // the requestMap below accomplishes that.
  
  map<int, vector<GlobalIndexTypeToCast>> requestMap;
  
  for (GlobalIndexType cellID : offRankCells)
  {
    int partitionForCell = mesh->globalDofAssignment()->partitionForCellID(cellID);
    if (partitionForCell != rank)
    {
      requestMap[partitionForCell].push_back(cellID);
    }
  }
  
  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;
  
  for (auto entry : requestMap)
  {
    int partition = entry.first;
    for (auto cellIDInPartition : entry.second)
    {
      myRequest.push_back(cellIDInPartition);
      myRequestOwners.push_back(partition);
    }
  }
  
  int myRequestCount = myRequest.size();
  Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*mesh->Comm());
  
  GlobalIndexTypeToCast* myRequestPtr = NULL;
  int *myRequestOwnersPtr = NULL;
  if (myRequest.size() > 0)
  {
    myRequestPtr = &myRequest[0];
    myRequestOwnersPtr = &myRequestOwners[0];
  }
  int numCellsToExport = 0;
  GlobalIndexTypeToCast* cellIDsToExport = NULL;  // we are responsible for deleting the allocated arrays
  int* exportRecipients = NULL;
  
  //    std::cout << "On rank " << rank << ", about to call CreateFromRecvs\n";
  distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numCellsToExport, cellIDsToExport, exportRecipients);
  
  const std::set<GlobalIndexType>* myCells = &mesh->globalDofAssignment()->cellsInPartition(-1);
  
  vector<int> sizes(numCellsToExport,0);
  vector<char> dataToExport; // bytes
  
  //    std::cout << "On rank " << rank << ", numCellsToExport = " << numCellsToExport << std::endl;
  for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
    if (myCells->find(cellID) == myCells->end())
    {
      cout << "cellID " << cellID << " does not belong to rank " << rank << endl;
      ostringstream myRankDescriptor;
      myRankDescriptor << "rank " << rank << ", cellID ownership";
      Camellia::print(myRankDescriptor.str().c_str(), *myCells);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested cellID does not belong to this rank!");
    }
    size_t dataSize = this->getCellDataSize(cellID);
    if (dataSize == 0) continue;
    vector<char> cellData(dataSize);
    this->packCellData(cellID, &cellData[0], dataSize);
    sizes[cellOrdinal] = dataSize;
    for (int byteOrdinal=0; byteOrdinal < dataSize; byteOrdinal++)
    {
      dataToExport.push_back(cellData[byteOrdinal]); // we could make this more efficient by resizing dataToExport and then doing a memcpy, or even resizing dataToExport and then calling packCellData(cellID, &dataToExport[offset], dataSize).  But this is a little less safe against programmer error.
    }
  }
  
  //    std::cout << "On rank " << rank << ", finished processing dataToExport.\n";
  int objSize = sizeof(char) / sizeof(char); // i.e., 1
  
  int importLength = 0;
  char* importedData = NULL;
  int* sizePtr = NULL;
  char* dataToExportPtr = NULL;
  if (numCellsToExport > 0)
  {
    sizePtr = &sizes[0];
    dataToExportPtr = (char *) &dataToExport[0];
  }
  //    std::cout << "On rank " << rank << ", about to call distributor->Do().\n";
  distributor->Do(dataToExportPtr, objSize, sizePtr, importLength, importedData);
  //    std::cout << "On rank " << rank << ", returned from distributor->Do().\n";
  const char* copyFromLocation = importedData;
  for (GlobalIndexType cellID : myRequest)
  {
    size_t bytesConsumed = this->unpackCellData(cellID, copyFromLocation, importLength);
    importLength -= bytesConsumed;
    copyFromLocation += bytesConsumed;
  }
  
  //    std::cout << "On rank " << rank << ", about to delete cellIDsToExport, etc.\n";
  if( cellIDsToExport != 0 ) delete [] cellIDsToExport;
  if( exportRecipients != 0 ) delete [] exportRecipients;
  if (importedData != 0 ) delete [] importedData;
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
      Camellia::FieldContainer<double> nodes = subCellsToCheck[i].subCellNodes;
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,mesh));
      int cubEnrich = 2; // arbitrary
      BasisCachePtr enrichedCache =  Teuchos::rcp(new BasisCache(elemType,mesh,testVsTest,cubEnrich));
      vector<GlobalIndexType> cellIDs;
      cellIDs.push_back(cellID);
      basisCache->setPhysicalCellNodes(nodes,cellIDs,true);
      enrichedCache->setPhysicalCellNodes(nodes,cellIDs,true);

      // calculate relative error for this subcell
      Camellia::FieldContainer<double> cellIntegral(1),enrichedCellIntegral(1);
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

        Camellia::FieldContainer<double> oldNodes = newCacheInfo.subCellNodes;
        oldNodes.resize(4,spaceDim);
        Camellia::FieldContainer<double> newCellNodes(numCells,4,spaceDim);
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
void TFunction<Scalar>::integrate(Camellia::FieldContainer<Scalar> &cellIntegrals, BasisCachePtr basisCache,
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
      Camellia::FieldContainer<Scalar> values(numCells,numPoints);
      this->values(values,sideCache);

      Camellia::FieldContainer<double> *weightedMeasures = &sideCache->getWeightedMeasures();
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
    Camellia::FieldContainer<Scalar> values(numCells,numPoints);
    this->values(values,basisCache);

    Camellia::FieldContainer<double> *weightedMeasures = &basisCache->getWeightedMeasures();
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
  const set<GlobalIndexType>* myCellIDs = &mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : *myCellIDs)
  {
    int numSides = mesh->getTopology()->getCell(cellID)->getSideCount();
    for (int sideIndex = 0; sideIndex < numSides; sideIndex++)
    {
      integral+= this->integralOfJump(mesh,cellID,sideIndex,cubatureDegreeEnrichment);
    }
  }
  return MPIWrapper::sum(*mesh->Comm(), integral);
}

template <typename Scalar>
Scalar TFunction<Scalar>::integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment)
{
  // for boundaries, the jump is 0
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  if (meshTopo->getCell(cellID)->isBoundary(sideIndex))
  {
    return 0;
  }
  pair<GlobalIndexType, unsigned> neighborInfo = meshTopo->getCell(cellID)->getNeighborInfo(sideIndex, meshTopo);
  int neighborCellID = neighborInfo.first;
  int neighborSideIndex = neighborInfo.second;

  ElementTypePtr myType = mesh->getElementType(cellID);
  ElementTypePtr neighborType = mesh->getElementType(neighborCellID);

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
  Camellia::FieldContainer<Scalar> cellIntegral(1);
  this->integrate(cellIntegral, neighborCache->getSideBasisCache(neighborSideIndex), true);
  //  cout << "Neighbor integral: " << cellIntegral[0] << endl;
  cellIntegral[0] *= -1;
  this->integrate(cellIntegral, myCache->getSideBasisCache(sideIndex), true);
  //  cout << "integral difference: " << cellIntegral[0] << endl;

  // multiply by sideParity to make jump uniquely valued.
  return sideParity * cellIntegral(0);
}
 
template <typename Scalar>
std::map<GlobalIndexType, double> TFunction<Scalar>::squaredL2NormOfJumps(MeshPtr mesh, bool weightBySideMeasure, int cubatureDegreeEnrichment, JumpCombinationType jumpCombination)
{
  // Computes the L^2 norm of the jumps of this function along the interior skeleton of the mesh
  
  /*
   We do the integration elementwise; on each face of each element, we decide whether the
   element "owns" the face, so that the term is only integrated once, and only on the side
   with finer quadrature, in the case of a locally refined mesh.
   */
  
    
  Epetra_CommPtr Comm = mesh->Comm();
  
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  const set<GlobalIndexType> & activeCellIDs = meshTopo->getLocallyKnownActiveCellIndices();
  const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
  
  // lambda for determining ownership
  auto isLocallyOwned = [&](GlobalIndexType cellID, int sideOrdinal) {
    CellPtr cell = meshTopo->getCell(cellID);
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
    GlobalIndexType neighborCellID = neighborInfo.first;
    if (neighborCellID == -1)
    {
      // boundary: we own the side because we own the cell
      return true;
    }
    unsigned mySideOrdinalInNeighbor = neighborInfo.second;
    if (activeCellIDs.find(neighborCellID) == activeCellIDs.end())
    {
      // no active neighbor on this side: either this is not an interior face (neighborCellID == -1),
      // or the neighbor is refined and therefore inactive.  If the latter, then the neighbor's
      // descendants will collectively "own" this side.
      return false;
    }
    
    // Finally, we need to check whether the neighbor is a "peer" in terms of h-refinements.
    // If so, we use the cellID to break the tie of ownership; lower cellID owns the face.
    CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
    pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
    bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
    if (neighborIsPeer && (cellID > neighborCellID))
    {
      // neighbor wins the tie-breaker
      return false;
    }
    
    // if we get here we own it
    return true;
  };
  
  // lambda for combining values
  auto combineValues = [&](Scalar v1, Scalar v2) {
    switch (jumpCombination)
    {
      case DIFFERENCE:
        return v1-v2;
      case SUM:
        return v1+v2;
    }
  };
  
  map<GlobalIndexType, vector<double> > sidel2norms; // key is cellID; values are the (squared) side contributions for that cell
  
  set<GlobalIndexType> offRankNeighbors;
  for (auto myCellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(myCellID);
    int sideCount = cell->getSideCount();
    
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      if (isLocallyOwned(myCellID,sideOrdinal))
      {
        pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
        GlobalIndexType neighborCellID = neighborInfo.first;
        if ((neighborCellID != -1) && myCellIDs.find(neighborCellID) == myCellIDs.end())
        {
          // off-rank, active neighbor for which we're responsible: ask for import
          offRankNeighbors.insert(neighborCellID);
        }
      }
    }
  }
  
  importDataForOffRankCells(mesh, offRankNeighbors);
  
  int sideDim = meshTopo->getDimension() - 1;
  
  Camellia::FieldContainer<double> emptyRefPointsVolume(0,meshTopo->getDimension()); // (P,D)
  Camellia::FieldContainer<double> emptyRefPointsSide(0,sideDim); // (P,D)
  Camellia::FieldContainer<double> emptyCubWeights(1,0); // (C,P)
  
  map<pair<CellTopologyKey,int>, Camellia::FieldContainer<double>> cubPointsForSideTopo;
  map<pair<CellTopologyKey,int>, Camellia::FieldContainer<double>> cubWeightsForSideTopo;
  
  CubatureFactory cubFactory;
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForVolumeTopo;         // used for "my" cells
  map<CellTopologyKey,BasisCachePtr> basisCacheForNeighborVolumeTopo; // used for neighbor cells
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnVolumeTopo;
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnNeighborVolumeTopo; // these can have permuted cubature points (i.e. we need to set them every time, so we can't share with basisCacheForSideOnVolumeTopo, which tries to avoid this)
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForReferenceCellTopo;
  
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    ElementTypePtr elemType = mesh->getElementType(cellID);
    
    Camellia::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
    BasisCachePtr cellBasisCacheVolume;
    if (basisCacheForVolumeTopo.find(cellTopo->getKey()) == basisCacheForVolumeTopo.end())
    {
      basisCacheForVolumeTopo[cellTopo->getKey()] = Teuchos::rcp( new BasisCache(physicalCellNodes, cellTopo,
                                                                                 emptyRefPointsVolume, emptyCubWeights) );
    }
    
    cellBasisCacheVolume = basisCacheForVolumeTopo[cellTopo->getKey()];
    cellBasisCacheVolume->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
    
    cellBasisCacheVolume->setCellIDs({cellID});
    cellBasisCacheVolume->setMesh(mesh);
    
    int sideCount = cell->getSideCount();
    if (sidel2norms.find(cellID) == sidel2norms.end())
    {
      sidel2norms[cellID] = vector<double>(sideCount, 0.0);
    }
    
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      if (! isLocallyOwned(cellID,sideOrdinal) ) continue;
      
      // if we get here, we own the face and should compute its contribution.
      int myTrialP = mesh->globalDofAssignment()->getH1Order(cellID)[0]; // for now, we assume isotropic in p
      int testSpaceEnrichment = mesh->testSpaceEnrichment();
      int myCubatureDegree = myTrialP + (myTrialP + testSpaceEnrichment + cubatureDegreeEnrichment);
      // assert that we are isotropic in p
      const auto myOrder = mesh->globalDofAssignment()->getH1Order(cellID);
      for (int d=1; d<myOrder.size(); d++)
      {
        if (myOrder[d] != myOrder[0])
        {
          std::cout << "squaredL2NormOfJumps does not support p-anisotropy right now; anisotropy detected in cell " << cellID << ".\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "squaredL2NormOfJumps does not support p-anisotropy");
        }
      }
      
      int cubaturePolyOrder = myCubatureDegree;
      pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
      GlobalIndexType neighborCellID = neighborInfo.first;
      if (neighborCellID != -1)
      {
        // figure out what the cubature degree should be
        DofOrderingPtr neighborTrialOrder = mesh->getElementType(neighborCellID)->trialOrderPtr;
        DofOrderingPtr neighborTestOrder = mesh->getElementType(neighborCellID)->testOrderPtr;
        
        int neighborTrialP = mesh->globalDofAssignment()->getH1Order(neighborCellID)[0];
        {
          const auto neighborOrder = mesh->globalDofAssignment()->getH1Order(neighborCellID);
          for (int d=1; d<neighborOrder.size(); d++)
          {
            if (neighborOrder[d] != neighborOrder[0])
            {
              std::cout << "squaredL2NormOfJumps does not support p-anisotropy right now; anisotropy detected in cell " << cellID << ".\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "squaredL2NormOfJumps does not support p-anisotropy");
            }
          }
        }
        int neighborCubatureDegree = neighborTrialP + (neighborTrialP + testSpaceEnrichment + cubatureDegreeEnrichment);
        
        cubaturePolyOrder = std::max(myCubatureDegree, neighborCubatureDegree);
      }
      
      // set up side basis cache
      CellTopoPtr mySideTopo = cellTopo->getSide(sideOrdinal); // for non-peers, this is the descendant cell topo
      
      pair<int,CellTopologyKey> sideCacheKey{sideOrdinal,cellTopo->getKey()};
      if (basisCacheForSideOnVolumeTopo.find(sideCacheKey) == basisCacheForSideOnVolumeTopo.end())
      {
        basisCacheForSideOnVolumeTopo[sideCacheKey] = Teuchos::rcp( new BasisCache(sideOrdinal, cellBasisCacheVolume,
                                                                                   emptyRefPointsSide, emptyCubWeights, -1));
      }
      BasisCachePtr cellBasisCacheSide = basisCacheForSideOnVolumeTopo[sideCacheKey];
      cellBasisCacheSide->setMesh(mesh);
      pair<CellTopologyKey,int> cubKey{mySideTopo->getKey(),cubaturePolyOrder};
      if (cubWeightsForSideTopo.find(cubKey) == cubWeightsForSideTopo.end())
      {
        int cubDegree = cubKey.second;
        if (sideDim > 0)
        {
          Teuchos::RCP<Intrepid::Cubature<double> > sideCub;
          if (cubDegree >= 0)
            sideCub = cubFactory.create(mySideTopo, cubDegree);
          
          int numCubPointsSide;
          
          if (sideCub != Teuchos::null)
            numCubPointsSide = sideCub->getNumPoints();
          else
            numCubPointsSide = 0;
          
          Camellia::FieldContainer<double> cubPoints(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
          Camellia::FieldContainer<double> cubWeights(numCubPointsSide);
          if (numCubPointsSide > 0)
            sideCub->getCubature(cubPoints, cubWeights);
          cubPointsForSideTopo[cubKey] = cubPoints;
          cubWeightsForSideTopo[cubKey] = cubWeights;
        }
        else
        {
          int numCubPointsSide = 1;
          Camellia::FieldContainer<double> cubPoints(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
          Camellia::FieldContainer<double> cubWeights(numCubPointsSide);
          
          cubPoints.initialize(0.0);
          cubWeights.initialize(1.0);
          cubPointsForSideTopo[cubKey] = cubPoints;
          cubWeightsForSideTopo[cubKey] = cubWeights;
        }
      }
      if (cellBasisCacheSide->cubatureDegree() != cubaturePolyOrder)
      {
        cellBasisCacheSide->setRefCellPoints(cubPointsForSideTopo[cubKey], cubWeightsForSideTopo[cubKey], cubaturePolyOrder, false);
      }
      cellBasisCacheSide->setPhysicalCellNodes(cellBasisCacheVolume->getPhysicalCellNodes(), {cellID}, false);
      
      int numCells = 1;
      int numPoints = cellBasisCacheSide->getRefCellPoints().dimension(0);
      Camellia::FieldContainer<Scalar> myValues;
      int spaceDim = sideDim + 1;
      if      (this->rank() == 0) myValues.resize(numCells, numPoints);
      else if (this->rank() == 1) myValues.resize(numCells, numPoints, spaceDim);
      else if (this->rank() == 2) myValues.resize(numCells, numPoints, spaceDim, spaceDim);
      else                   TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported Function rank");
        
      values(myValues, cellBasisCacheSide);
      Camellia::FieldContainer<Scalar> neighborValues(myValues); // size according to myValues
      neighborValues.initialize(0.0); // fill with zeros for the case that we're on the boundary
      if (neighborCellID != -1)
      {
        // Now the geometrically challenging bit: we need to line up the physical points in
        // the cellBasisCacheSide with those in a BasisCache for the neighbor cell
        auto neighbor = meshTopo->getCell(neighborCellID);
        auto mySideOrdinalInNeighbor = neighborInfo.second;
        pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
        bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
        
        CellTopoPtr neighborTopo = neighbor->topology();
        CellTopoPtr sideTopo = neighborTopo->getSide(mySideOrdinalInNeighbor); // for non-peers, this is my ancestor's cell topo
        int nodeCount = sideTopo->getNodeCount();
        
        unsigned permutationFromMeToNeighbor;
        Camellia::FieldContainer<double> myRefPoints = cellBasisCacheSide->getRefCellPoints();
        
        if (!neighborIsPeer) // then we have some refinements relative to neighbor
        {
          /*******   Map my ref points to my ancestor ******/
          pair<GlobalIndexType,unsigned> ancestorInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
          GlobalIndexType ancestorCellIndex = ancestorInfo.first;
          unsigned ancestorSideOrdinal = ancestorInfo.second;
          
          RefinementBranch refinementBranch = cell->refinementBranchForSide(sideOrdinal, meshTopo);
          RefinementBranch sideRefinementBranch = RefinementPattern::sideRefinementBranch(refinementBranch, ancestorSideOrdinal);
          Camellia::FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(sideRefinementBranch);
          
          cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
          BasisCachePtr ancestralBasisCache = Teuchos::rcp(new BasisCache(cellNodes,sideTopo,cubaturePolyOrder,false)); // false: don't create side cache too
          
          ancestralBasisCache->setRefCellPoints(myRefPoints, emptyCubWeights, cubaturePolyOrder, true);
          
          // now, the "physical" points in ancestral cache are the ones we want
          myRefPoints = ancestralBasisCache->getPhysicalCubaturePoints();
          myRefPoints.resize(myRefPoints.dimension(1),myRefPoints.dimension(2)); // strip cell dimension
          
          /*******  Determine ancestor's permutation of the side relative to neighbor ******/
          CellPtr ancestor = meshTopo->getCell(ancestorCellIndex);
          vector<IndexType> ancestorSideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
          
          CellTopoPtr sideTopo = ancestor->topology()->getSide(ancestorSideOrdinal);
          
          // the following permutations map from canonical ordering to that seen by the cell
          // we want to construct a mapping that takes us from ancestor's ordering to neighbor's
          unsigned ancestorSidePermutation = ancestor->subcellPermutation(sideDim, sideOrdinal);
          unsigned neighborSidePermutation = neighbor->subcellPermutation(sideDim, mySideOrdinalInNeighbor);
          
          // invert ancestor permutation; this takes us from my ordering to canonical
          unsigned ancestorInverse = CamelliaCellTools::permutationInverse(sideTopo, ancestorSidePermutation);
          // compose this with neighbor permutation; this takes us from canonical to neighbor
          permutationFromMeToNeighbor = CamelliaCellTools::permutationComposition(sideTopo, ancestorInverse, neighborSidePermutation);
        }
        else
        {
          // neighbor is peer, but might not be active (i.e., might have descendants)
          
          // the following permutations map from canonical ordering to that seen by the cell
          // we want to construct a mapping that takes us from my ordering to neighbor's
          unsigned mySidePermutation       = cell->subcellPermutation(sideDim, sideOrdinal);
          unsigned neighborSidePermutation = neighbor->subcellPermutation(sideDim, mySideOrdinalInNeighbor);
          
          // invert my permutation; this takes us from my ordering to canonical
          unsigned myInverse = CamelliaCellTools::permutationInverse(sideTopo, mySidePermutation);
          // compose this with neighbor permutation; this takes us from canonical to neighbor
          permutationFromMeToNeighbor = CamelliaCellTools::permutationComposition(sideTopo, myInverse, neighborSidePermutation);
        }
        
        Camellia::FieldContainer<double> permutedRefNodes(nodeCount,sideDim);
        CamelliaCellTools::refCellNodesForTopology(permutedRefNodes, sideTopo, permutationFromMeToNeighbor);
        permutedRefNodes.resize(1,nodeCount,sideDim); // add cell dimension to make this a "physical" node container
        if (basisCacheForReferenceCellTopo.find(sideTopo->getKey()) == basisCacheForReferenceCellTopo.end())
        {
          basisCacheForReferenceCellTopo[sideTopo->getKey()] = BasisCache::basisCacheForReferenceCell(sideTopo, -1);
        }
        BasisCachePtr referenceBasisCache = basisCacheForReferenceCellTopo[sideTopo->getKey()];
        referenceBasisCache->setRefCellPoints(myRefPoints,emptyCubWeights,cubaturePolyOrder,false);
        std::vector<GlobalIndexType> cellIDs = {0}; // unused
        referenceBasisCache->setPhysicalCellNodes(permutedRefNodes, cellIDs, false);
        // now, the "physical" points are the ones we should use as ref points for the neighbor
        Camellia::FieldContainer<double> neighborRefCellPoints = referenceBasisCache->getPhysicalCubaturePoints();
        neighborRefCellPoints.resize(numPoints,sideDim); // strip cell dimension to convert to a "reference" point container
        
        Camellia::FieldContainer<double> neighborCellNodes = mesh->physicalCellNodesForCell(neighborCellID);
        if (basisCacheForNeighborVolumeTopo.find(neighborTopo->getKey()) == basisCacheForNeighborVolumeTopo.end())
        {
          basisCacheForNeighborVolumeTopo[neighborTopo->getKey()] = Teuchos::rcp( new BasisCache(neighborCellNodes, neighborTopo,
                                                                                                 emptyRefPointsVolume, emptyCubWeights) );
        }
        BasisCachePtr neighborVolumeCache = basisCacheForNeighborVolumeTopo[neighborTopo->getKey()];
        neighborVolumeCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
        
        pair<int,CellTopologyKey> neighborSideCacheKey{mySideOrdinalInNeighbor,neighborTopo->getKey()};
        if (basisCacheForSideOnNeighborVolumeTopo.find(neighborSideCacheKey) == basisCacheForSideOnNeighborVolumeTopo.end())
        {
          basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey]
          = Teuchos::rcp( new BasisCache(mySideOrdinalInNeighbor, neighborVolumeCache, emptyRefPointsSide, emptyCubWeights, -1));
        }
        BasisCachePtr neighborSideCache = basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey];
        neighborSideCache->setMesh(mesh);
        neighborSideCache->setRefCellPoints(neighborRefCellPoints, emptyCubWeights, cubaturePolyOrder, false);
        neighborSideCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
        
        // if there aren't periodic BCs, perform the sanity check.  If there are periodic BCs, we expect to fail the check...
        if (meshTopo->getPeriodicBCs().size() == 0)
        {
          // Sanity check that the physical points agree:
          double tol = 1e-8;
          Camellia::FieldContainer<double> myPhysicalPoints = cellBasisCacheSide->getPhysicalCubaturePoints();
          Camellia::FieldContainer<double> neighborPhysicalPoints = neighborSideCache->getPhysicalCubaturePoints();
          
          bool pointsMatch = (myPhysicalPoints.size() == neighborPhysicalPoints.size()); // true unless we find a point that doesn't match
          double maxDiff = 0.0;
          if (pointsMatch)
          {
            for (int i=0; i<myPhysicalPoints.size(); i++)
            {
              double diff = abs(myPhysicalPoints[i]-neighborPhysicalPoints[i]);
              if (diff > tol)
              {
                pointsMatch = false;
                maxDiff = std::max(diff,maxDiff);
              }
            }
          }
          
          if (!pointsMatch)
          {
            cout << "WARNING: pointsMatch is false; maxDiff = " << maxDiff << "\n";
//            cout << "myPhysicalPoints:\n" << myPhysicalPoints;
//            cout << "neighborPhysicalPoints:\n" << neighborPhysicalPoints;
          }
        }
        this->values(neighborValues, neighborSideCache);
      }
      
      double sideL2Jump = 0.0;
      auto & physCubWeights = cellBasisCacheSide->getWeightedMeasures(); // (C,P) container
      double sideMeasure = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const int cellOrdinal = 0; // 0 because we're in a single-cell BasisCache
        double weight = physCubWeights(cellOrdinal,pointOrdinal);
        sideMeasure += weight;
        if (this->rank() == 0)
        {
          Scalar diff = combineValues(neighborValues(cellOrdinal,pointOrdinal), myValues(cellOrdinal,pointOrdinal));
          sideL2Jump += diff * diff * weight;
//          cout << "on cell " << cellID << endl;
//          cout << "neighbor value = " << neighborValues(cellOrdinal,pointOrdinal) << endl;
//          cout << "my value = " << myValues(cellOrdinal,pointOrdinal) << endl;
//          cout << "diff = " << diff << endl;
        }
        else if (this->rank() == 1)
        {
          for (int d1=0; d1<spaceDim; d1++)
          {
            Scalar diff = combineValues(neighborValues(cellOrdinal,pointOrdinal,d1), myValues(cellOrdinal,pointOrdinal,d1));
            sideL2Jump += diff * diff * weight;
          }
        }
        else if (this->rank() == 2)
        {
          for (int d1=0; d1<spaceDim; d1++)
          {
            for (int d2=0; d2<spaceDim; d2++)
            {
              Scalar diff = combineValues(neighborValues(cellOrdinal,pointOrdinal,d1,d2), myValues(cellOrdinal,pointOrdinal,d1,d2));
              sideL2Jump += diff * diff * weight;
            }
          }
        }
      }
      if (!weightBySideMeasure) sidel2norms[cellID][sideOrdinal] = sideL2Jump;
      else                      sidel2norms[cellID][sideOrdinal] = sideL2Jump * sideMeasure;
    }
  }
  
  // Each side L^2 norm is now stored exactly once; we need to set neighbor values, communicating them via MPI if necessary
  // values that belong to MPI-local cells get stored in existing sideL2Norms container
  // values that need to be communicated get stored in offRankSideL2Norms
  map<GlobalIndexType,vector<double> > offRankSideL2Norms;
  for (GlobalIndexType cellID : myCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    
    int sideCount = cell->getSideCount();
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      if (! isLocallyOwned(cellID,sideOrdinal)) continue;
      
      pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
      GlobalIndexType neighborCellID = neighborInfo.first;
      if (neighborCellID == -1) continue; // boundary: no neighbor
      unsigned mySideOrdinalInNeighbor = neighborInfo.second;
      auto neighbor = meshTopo->getCell(neighborCellID);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
      bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
      
      bool neighborIsMPILocal = (myCellIDs.find(neighborCellID) != myCellIDs.end());
      map<GlobalIndexType,vector<double> >* mapForStorage; // points either to sideL2Norms or to offRankSideL2Norms
      
      if (neighborIsMPILocal)
      {
        // then space should have been allocated above; confirm this
        if (sidel2norms.find(neighborCellID) == sidel2norms.end())
        {
          cout << "Internal Error: sideL2Norms does not have space allocated for local neighbor with cell ID " << neighborCellID << endl;
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error: space not allocated for neighborCellID");
        }
        mapForStorage = &sidel2norms;
      }
      else
      {
        // make sure space is allocated
        if (offRankSideL2Norms.find(neighborCellID) == offRankSideL2Norms.end())
        {
          auto neighborSideCount = neighbor->getSideCount();
          offRankSideL2Norms[neighborCellID] = vector<double>(neighborSideCount,0.0);
        }
        mapForStorage = &offRankSideL2Norms;
      }
      
      if (!neighborIsPeer)
      {
        // then we are the descendant of a refined cell that neighbors an unrefined one
        // this means that the neighbor may have multiple contributors on that side; we should sum into
        (*mapForStorage)[neighborCellID][mySideOrdinalInNeighbor] += sidel2norms[cellID][sideOrdinal];
      }
      else
      {
        // we should only store one thing; as a sanity check, we make sure the pre-existing value is 0.0
        double oldValue = (*mapForStorage)[neighborCellID][mySideOrdinalInNeighbor];
        if (oldValue != 0.0)
        {
          cout << "Error for neighbor ID " << neighborCellID << " on side " << mySideOrdinalInNeighbor << ": ";
          cout << "has nonzero value " << oldValue << " prior to being set.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "oldValue != 0.0");
        }
        (*mapForStorage)[neighborCellID][mySideOrdinalInNeighbor] = sidel2norms[cellID][sideOrdinal];
      }
    }
  }
  
  // Now, communicate the off-rank neighbor values to their owners.
  // MPIWrapper::sendDataMaps has signature
  /* sendDataMaps(Epetra_CommPtr Comm,
                  const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                  std::map<KeyType,ValueType> &receivedMap);*/
  // where ValueType and KeyType can be of arbitrary fixed-size type
  // and the keys to recipientDataMaps are the MPI ranks (PIDs) of the recipients
  // for us, KeyType can be pair<cellID,sideOrdinal> the ValueType can be normValue
  map<int, map<pair<GlobalIndexType,int>,double > > recipientDataMaps;
  map<pair<GlobalIndexType,int>, double > receivedMap;
  
  // fill in recipientDataMaps
  for (auto entry : offRankSideL2Norms)
  {
    GlobalIndexType cellID = entry.first;
    vector<double> &sideNorms = entry.second;
    int ownerPID = mesh->globalDofAssignment()->partitionForCellID( cellID );
    for (int sideOrdinal = 0; sideOrdinal < sideNorms.size(); sideOrdinal++)
    {
      // multiple MPI ranks may have something to say about this cell; only one should
      // have anything for a given sideOrdinal.  If we don't have anything to say about
      // a side, our sideNorms entry for that will be 0.0.  (If what we have to say is 0.0,
      // then it's fine not to say that.)
      if (sideNorms[sideOrdinal] != 0.0)
      {
        recipientDataMaps[ownerPID][{cellID,sideOrdinal}] = sideNorms[sideOrdinal];
//        {
//          // DEBUGGING
//          int rank = Comm->MyPID();
//          cout << "On rank " << rank << ", sending key {" << cellID << "," << sideOrdinal << "} with value " << sideNorms[sideOrdinal];
//          cout << " to rank " << ownerPID << endl;
//        }
      }
    }
  }
  MPIWrapper::sendDataMaps(mesh->Comm(), recipientDataMaps, receivedMap);
  // incorporate the received data into our sideL2Norms
  for (auto receivedEntry : receivedMap)
  {
    GlobalIndexType cellID = receivedEntry.first.first;
    int sideOrdinal = receivedEntry.first.second;
    double normContribution = receivedEntry.second; // there may be several, if neighbor was refined
    sidel2norms[cellID][sideOrdinal] += normContribution;
//    {
//      // DEBUGGING
//      int rank = Comm->MyPID();
//      cout << "On rank " << rank << ", received key {" << cellID << "," << sideOrdinal << "} with value " << normContribution << endl;
//    }
  }
  // Finally, sum the side contributions for each MPI-local cell, take square root (because L^2 norm), and return the result
  map<GlobalIndexType,double> cellNorms;
  for (auto & entry : sidel2norms)
  {
    GlobalIndexType cellID = entry.first;
    vector<double> & cellSideNorms = entry.second;
    double cellTotal = 0.0;
    for (auto sideContribution : cellSideNorms)
    {
      cellTotal += sideContribution;
    }
    // BK: For most cases we need, it is actually better not to perform the square root
    cellNorms[cellID] = cellTotal;
    // cellNorms[cellID] = sqrt(cellTotal);
  }
  return cellNorms;
}

template <typename Scalar>
Scalar TFunction<Scalar>::integrate(MeshPtr mesh, int cubatureDegreeEnrichment, bool testVsTest, bool requireSideCache,
                                    bool spatialSidesOnly)
{
  Scalar integral = 0;
  MeshTopologyViewPtr meshTopoView = mesh->getTopology();

  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : cellIDs)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
    if ( this->boundaryValueOnly() )
    {
      ElementTypePtr elemType = mesh->getElementType(cellID);
      int numSides = elemType->cellTopoPtr->getSideCount();

      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
      {
        if (spatialSidesOnly && !elemType->cellTopoPtr->sideIsSpatial(sideOrdinal)) continue; // skip non-spatial sides if spatialSidesOnly is true
        Scalar sideIntegral = this->integrate(basisCache->getSideBasisCache(sideOrdinal));
        // this will hit each interior side twice; therefore we divide by two if the side is interior
        auto neighborCell = meshTopoView->getCell(cellID)->getNeighbor(sideOrdinal, meshTopoView);
        if (neighborCell != Teuchos::null) // interior side, then
        {
          sideIntegral /= 2.0;
        }
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
  bool TFunction<Scalar>::isZero(BasisCachePtr basisCache)
  {
    // default implementation: we only attest to being zero on basisCache if we are zero everywhere
    return isZero();
  }

template <typename Scalar>
double TFunction<Scalar>::l1norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool spatialSidesOnly)
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( this, false );
  bool testVsTest = false, requireSideCaches = false;
  TFunctionPtr<double> magnitudeFxn = TFunction<Scalar>::sqrtFunction(thisPtr * thisPtr);
  return abs(magnitudeFxn->integrate(mesh, cubatureDegreeEnrichment, testVsTest, requireSideCaches, spatialSidesOnly));
}
  
template <typename Scalar>
double TFunction<Scalar>::l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment, bool spatialSidesOnly)
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( this, false );
  bool testVsTest = false, requireSideCaches = false;
  return sqrt( abs((thisPtr * thisPtr)->integrate(mesh, cubatureDegreeEnrichment, testVsTest, requireSideCaches, spatialSidesOnly)) );
}

// BK: Method to compute the global maximum of a function
template <typename Scalar>
double TFunction<Scalar>::linfinitynorm(MeshPtr mesh, int cubatureDegreeEnrichment)
{
  // Kind of crude, but works!
  TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only compute the L^infty norm of scalar functions, at least for now.");
  double localMax = 0.0;
  bool testVsTest = false;
  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : cellIDs)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    Camellia::FieldContainer<Scalar> fxnValues(numCells,numPoints);
    this->values(fxnValues,basisCache);

    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        localMax = std::max(localMax,abs(fxnValues(cellIndex,ptIndex)));
      }
    }
  }
  double globalMax = 0.0;
  mesh->Comm()->MaxAll(&localMax, &globalMax, 1);
  return globalMax;
}
  
  template <typename Scalar>
  double TFunction<Scalar>::maximumValue(MeshPtr mesh, int cubatureDegreeEnrichment)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only compute the maximum value of scalar functions.");
    double localMax = -numeric_limits<double>::max();
    bool testVsTest = false;
    set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
    for (GlobalIndexType cellID : cellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
      int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      Camellia::FieldContainer<Scalar> fxnValues(numCells,numPoints);
      this->values(fxnValues,basisCache);
      
      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
        {
          localMax = std::max(localMax,fxnValues(cellIndex,ptIndex));
        }
      }
      if (!basisCache->isSideCache())
      {
        int numSides = basisCache->cellTopology()->getSideCount();
        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
        {
          auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
          int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
          int numSidePoints = sideCache->getPhysicalCubaturePoints().dimension(1);
          Camellia::FieldContainer<Scalar> fxnValues(numCells,numSidePoints);
          this->values(fxnValues,sideCache);
          for (int cellIndex=0; cellIndex<numCells; cellIndex++)
          {
            for (int ptIndex=0; ptIndex<numSidePoints; ptIndex++)
            {
              localMax = std::max(localMax,fxnValues(cellIndex,ptIndex));
            }
          }
        }
      }
    }
    double globalMax = 0.0;
    mesh->Comm()->MaxAll(&localMax, &globalMax, 1);
    return globalMax;
  }
  
  template <typename Scalar>
  double TFunction<Scalar>::minimumValue(MeshPtr mesh, int cubatureDegreeEnrichment)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_rank != 0, std::invalid_argument, "can only compute the minimum value of scalar functions.");
    double localMin = numeric_limits<double>::max();
    bool testVsTest = false;
    set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
    for (GlobalIndexType cellID : cellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
      int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      Camellia::FieldContainer<Scalar> fxnValues(numCells,numPoints);
      this->values(fxnValues,basisCache);
      
      for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
        {
          localMin = std::min(localMin,fxnValues(cellIndex,ptIndex));
        }
      }

      if (!basisCache->isSideCache())
      {
        int numSides = basisCache->cellTopology()->getSideCount();
        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
        {
          auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
          int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
          int numSidePoints = sideCache->getPhysicalCubaturePoints().dimension(1);
          Camellia::FieldContainer<Scalar> fxnValues(numCells,numSidePoints);
          this->values(fxnValues,sideCache);
          for (int cellIndex=0; cellIndex<numCells; cellIndex++)
          {
            for (int ptIndex=0; ptIndex<numSidePoints; ptIndex++)
            {
              localMin = std::min(localMin,fxnValues(cellIndex,ptIndex));
            }
          }
        }
      }
    }
    double globalMin = 0.0;
    mesh->Comm()->MinAll(&localMin, &globalMin, 1);
    return globalMin;
  }

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> TFunction<Scalar>::memberFunctions()
{
  return std::vector<TFunctionPtr<Scalar>>();
}
  
// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
template <typename Scalar>
void TFunction<Scalar>::scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache)
{
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
}

// divide values by this function (supported only when this is a scalar)
template <typename Scalar>
void TFunction<Scalar>::scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache)
{
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
// should only happen with double valued functions
// TODO: throw error for complex
template <typename Scalar>
void TFunction<Scalar>::scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
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
void TFunction<Scalar>::scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  // basisValues has dimensions (C,F,P,...)
  scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
}

template <typename Scalar>
void TFunction<Scalar>::valuesDottedWithTensor(Camellia::FieldContainer<Scalar> &values,
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

  std::vector<int> tensorValueIndex(_rank+2); // +2 for numCells, numPoints indices
  tensorValueIndex[0] = numCells;
  tensorValueIndex[1] = numPoints;
  for (int d=0; d<_rank; d++)
  {
    tensorValueIndex[d+2] = spaceDim;
  }
  
  Camellia::FieldContainer<Scalar> myTensorValues(tensorValueIndex);
  this->values(myTensorValues,basisCache);
  Camellia::FieldContainer<Scalar> otherTensorValues(tensorValueIndex);
  tensorFunctionOfLikeRank->values(otherTensorValues,basisCache);

  //  cout << "myTensorValues:\n" << myTensorValues;
  //  cout << "otherTensorValues:\n" << otherTensorValues;

  // clear out the spatial indices of tensorValueIndex so we can use it as index
  for (int d=0; d<_rank; d++)
  {
    tensorValueIndex[d+2] = 0;
  }
  
  auto getMyTensorValuesEnumeration = SerialDenseWrapper::getEnumerator(tensorValueIndex, myTensorValues);
  auto getOtherTensorValuesEnumeration = SerialDenseWrapper::getEnumerator(tensorValueIndex, otherTensorValues);
  
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
      Scalar *myValue = &myTensorValues[ getMyTensorValuesEnumeration() ];
      Scalar *otherValue = &otherTensorValues[ getOtherTensorValuesEnumeration() ];
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
void TFunction<Scalar>::scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
    FunctionModificationType modType)
{
  TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getSpaceDim();

  Camellia::FieldContainer<Scalar> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);

  std::vector<int> valueIndex(values.rank());
  
  auto getValuesEnumeration = SerialDenseWrapper::getEnumerator(valueIndex, values);
  
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
      Scalar *value = &values[ getValuesEnumeration() ];
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
void TFunction<Scalar>::scalarModifyBasisValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache,
    FunctionModificationType modType)
{
  TEUCHOS_TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyBasisValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numFields = values.dimension(1);
  int numPoints = values.dimension(2);

  int spaceDim = basisCache->getSpaceDim();

  Camellia::FieldContainer<double> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);

  //  cout << "scalarModifyBasisValues: scalarValues:\n" << scalarValues;

  std::vector<int> valueIndex(values.rank());

  auto getValuesEnumeration = SerialDenseWrapper::getEnumerator(valueIndex, values);
  
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
        double *value = &values[ getValuesEnumeration() ];
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

  PartitionIndexType rank = mesh->Comm()->MyPID();
  
  BasisCachePtr basisCache;
  for (ElementTypePtr elemTypePtr : elementTypes)
  {
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh, true) );
    CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
    int numSides = cellTopo->getSideCount();

    Camellia::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<GlobalIndexType> cellIDs = mesh->globalDofAssignment()->cellIDsOfElementType(rank, elemTypePtr);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    int num1DPts = 15;
    Camellia::FieldContainer<double> refPoints(num1DPts,1);
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


      Camellia::FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
      this->values(computedValues,sideBasisCache);

      // NOW loop over all cells to write solution to file
      for (int cellIndex=0; cellIndex < numCells; cellIndex++)
      {
        Camellia::FieldContainer<double> cellParities = mesh->cellSideParitiesForCell( cellIDs[cellIndex] );
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
  Camellia::FieldContainer<double> refPoints(numPoints,spaceDim);
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
  
  int rank = mesh->Comm()->MyPID();
  
  for (ElementTypePtr elemTypePtr : elementTypes)   //thru quads/triangles/etc
  {
    basisCache = Teuchos::rcp( new BasisCache(elemTypePtr, mesh) );
    basisCache->setRefCellPoints(refPoints);

    Camellia::FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<GlobalIndexType> cellIDs = mesh->globalDofAssignment()->cellIDsOfElementType(rank, elemTypePtr);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, false); // false: don't create side cache

    Camellia::FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();

    Camellia::FieldContainer<double> computedValues(numCells,numPoints);
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
TFunctionPtr<Scalar> TFunction<Scalar>::constant(vector<Scalar> value)
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
TFunctionPtr<Scalar> TFunction<Scalar>::solution(VarPtr var, TSolutionPtr<Scalar> soln, const std::string &solutionIdentifierExponent)
{
  TEUCHOS_TEST_FOR_EXCEPTION(var->varType() == FLUX, std::invalid_argument, "For flux variables, must provide a weightFluxesBySideParity argument");
  bool weightFluxesBySideParity = false; // inconsequential for non-fluxes
  int solutionOrdinal = 0;
  return Teuchos::rcp( new SimpleSolutionFunction<Scalar>(var, soln, weightFluxesBySideParity, solutionOrdinal, solutionIdentifierExponent) );
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::solution(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, const std::string &solutionIdentifierExponent)
{
  int solutionOrdinal = 0;
  return Teuchos::rcp( new SimpleSolutionFunction<Scalar>(var, soln, weightFluxesBySideParity, solutionOrdinal, solutionIdentifierExponent) );
}

  template <typename Scalar>
  TFunctionPtr<Scalar> TFunction<Scalar>::solution(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, int solutionOrdinal, const std::string &solutionIdentifierExponent)
  {
    return Teuchos::rcp( new SimpleSolutionFunction<Scalar>(var, soln, weightFluxesBySideParity, solutionOrdinal, solutionIdentifierExponent) );
  }
  
  template <typename Scalar>
  TFunctionPtr<Scalar> TFunction<Scalar>::sqrtFunction(TFunctionPtr<Scalar> f)
  {
    ConstantScalarFunction<Scalar>* f_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f.get());
    if (f_constant != NULL)
    {
      Scalar value = f_constant->value();
      return TFunction<Scalar>::constant(sqrt(value));
    }
    
    return Teuchos::rcp( new SqrtFunction<Scalar>(f) );
  }
  
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  return vectorize({f1,f2});
}

template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3)
{
  return vectorize({f1,f2,f3});// Teuchos::rcp( new VectorizedFunction<Scalar>(f1,f2,f3) );
}
  
template <typename Scalar>
TFunctionPtr<Scalar> TFunction<Scalar>::vectorize(vector<TFunctionPtr<Scalar>> components)
{
  return Teuchos::rcp( new VectorizedFunction<Scalar>(components) );
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
      // assume 3D; no real harm in having the extra zeros in 2D...
      zeroTensor = TFunction<double>::vectorize(zeroTensor, zeroTensor, zeroTensor);
    }
    return zeroTensor;
  }
}

// this is liable to be a bit slow!!
class ComposedFunction : public TFunction<double>
{
  TFunctionPtr<double> _f, _arg_g;
public:
  ComposedFunction(TFunctionPtr<double> f, TFunctionPtr<double> arg_g) : TFunction<double>("ComposedFunction", f->rank())
  {
    _f = f;
    _arg_g = arg_g;
  }
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getSpaceDim();
    Camellia::FieldContainer<double> fArgPoints(numCells,numPoints,spaceDim);
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
  
  TFunctionPtr<double> evaluateAt(const map<int, TFunctionPtr<double> > &valueMap)
  {
    auto f = TFunction<double>::evaluateFunctionAt(_f, valueMap);
    auto g = TFunction<double>::evaluateFunctionAt(_arg_g, valueMap);
    return TFunction<double>::composedFunction(f,g);
  }
  
  TLinearTermPtr<double> jacobian(const map<int, TFunctionPtr<double> > &valueMap)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Jacobian evaluation is not yet supported for composed functions");
  }
  
  std::vector<TFunctionPtr<double>> memberFunctions()
  {
    return {_f, _arg_g};
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
  if (f1 == Teuchos::null)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "f1 is null!");
  }
  else if (f2 == Teuchos::null)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "f2 is null!");
  }
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
  
  ProductFunction<Scalar>* f1_product = dynamic_cast<ProductFunction<Scalar>*> (f1.get());
  ProductFunction<Scalar>* f2_product = dynamic_cast<ProductFunction<Scalar>*> (f2.get());;
  
  ConstantScalarFunction<Scalar>* f1_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f1.get());
  if (f1_constant)
  {
    // check if it's 1.0; then the product will be just f2
    if (f1_constant->value() == 1)
    {
      return f2;
    }
  }
  
  ConstantScalarFunction<Scalar>* f2_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f2.get());
  if (f2_constant)
  {
    if (f2_constant->value() == 1)
    {
      return f1;
    }
  }
  
  if (f1_constant && f2_constant)
  {
    return Function::constant(f1_constant->value() * f2_constant->value());
  }
  
  // if one multiplicand is a constant scalar, put that one in front:
  if (f2_constant)
  {
    if (f1_product)
    {
      TFunctionPtr<Scalar> f11 = f1_product->f1();
      TFunctionPtr<Scalar> f12 = f1_product->f2();
      // since we put constants in front, if one of f11 or f12 is constant, f11 will be
      // multiplying like so gives a chance to combine the constants:
      return (f2 * f11) * f12;
    }
    else
      return Teuchos::rcp( new ProductFunction<Scalar>(f2,f1) );
  }
  else if (f1_constant)
  {
    if (f2_product)
    {
      TFunctionPtr<Scalar> f21 = f2_product->f1();
      TFunctionPtr<Scalar> f22 = f2_product->f2();
      // since we put constants in front, if one of f21 or f22 is constant, f21 will be
      // multiplying like so gives a chance to combine the constants:
      return (f1 * f21) * f22;
    }
    else
      return Teuchos::rcp( new ProductFunction<Scalar>(f1,f2) );
  }
  else
  {
    if (f1_product && f2_product)
    {
      // let's check whether their first multiplicands are constants:
      TFunctionPtr<Scalar> f11 = f1_product->f1();
      TFunctionPtr<Scalar> f12 = f1_product->f2();
      TFunctionPtr<Scalar> f21 = f2_product->f1();
      TFunctionPtr<Scalar> f22 = f2_product->f2();
      
      ConstantScalarFunction<Scalar>* f11_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f11.get());
      ConstantScalarFunction<Scalar>* f21_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f21.get());
      if (f11_constant && f21_constant)
      {
        return (f11_constant->value() * f21_constant->value()) * (f12 * f22);
      }
    }
    return Teuchos::rcp( new ProductFunction<Scalar>(f1,f2) );
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> scalarDivisor)
{
  if ( f1->isZero() )
  {
    return TFunction<Scalar>::zero(f1->rank());
  }
  ConstantScalarFunction<Scalar>* f1_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f1.get());
  if (f1_constant)
  {
    // then check if scalarDivisor is constant, too:
    ConstantScalarFunction<Scalar>* scalarDivisor_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (scalarDivisor.get());
    if (scalarDivisor_constant)
    {
      return Function::constant(f1_constant->value() / scalarDivisor_constant->value());
    }
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
  ConstantScalarFunction<Scalar>* f1_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f1.get());
  if (f1_constant)
  {
    // then check if f2 is constant, too:
    ConstantScalarFunction<Scalar>* f2_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (f2.get());
    if (f2_constant)
    {
      return Function::constant(f1_constant->value() + f2_constant->value());
    }
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
