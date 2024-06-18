//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BC.h"

#include "BasisFactory.h"
#include "BCFunction.h"
#include "Function.h"
#include "PhysicalPointCache.h"
#include "Projector.h"
#include "SpatiallyFilteredFunction.h"
#include "TypeDefs.h"
#include "Var.h"

using namespace Camellia;

template <typename Scalar>
class BCLogicalOrFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f;
public:
  BCLogicalOrFunction(TFunctionPtr<Scalar> f1, SpatialFilterPtr sf1, TFunctionPtr<Scalar> f2, SpatialFilterPtr sf2) : TFunction<Scalar>("BCLogicalOrFunction", f1->rank())
  {
    // The logic here is: If sf1 matches, use f1 to compute the values.
    //                    If sf1 does not match, and sf2 does, use f2 to compute the values.
    //                    The filter sf2 & !sf1 implements the clause "sf1 does not match, and sf2 does"
    FunctionPtr f1_filtered = SpatiallyFilteredFunction<Scalar>::spatiallyFilteredFunction(f1, sf1);
    SpatialFilterPtr sf2_minus_sf1 = sf2 & !sf1;
    FunctionPtr f2_filtered = SpatiallyFilteredFunction<Scalar>::spatiallyFilteredFunction(f2, sf2_minus_sf1);
    FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
    if (! f1_filtered->boundaryValueOnly())
    {
      // then force it to be:
      f1_filtered = f1_filtered * boundaryRestriction;
    }
    if (! f2_filtered->boundaryValueOnly())
    {
      // then force it to be:
      f2_filtered = f2_filtered * boundaryRestriction;
    }
    _f = f1_filtered + f2_filtered;
  }
  bool boundaryValueOnly() override
  {
    return true;
  }
  void setTime(double time)
  {
    this->_time = time;
    _f->setTime(time);
  }
  using TFunction<Scalar>::values;
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    _f->values(values, basisCache);
  }
  virtual void values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    _f->values(values, modType, basisCache);
  }
};

template <typename Scalar>
void TBC<Scalar>::addDirichlet(VarPtr var, int tagID, TFunctionPtr<Scalar> valueFunction)
{
  _dirichletTagBCs[tagID].push_back({var,valueFunction});
}

template class BCLogicalOrFunction<double>;

template <typename Scalar>
void TBC<Scalar>::addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, TFunctionPtr<Scalar> valueFunction )
{
//  if ((traceOrFlux->varType() != TRACE) && (traceOrFlux->varType() != FLUX))
//  {
//    cout << "WARNING: adding Dirichlet BC for variable that is neither a trace nor a flux.\n";
//  }

  if (_dirichletBCs.find( traceOrFlux->ID() ) != _dirichletBCs.end() )
  {
    // "or" the existing condition with the new one:
    SpatialFilterPtr existingFilter = _dirichletBCs[ traceOrFlux->ID() ].first;
    TFunctionPtr<Scalar> existingFunction = _dirichletBCs[ traceOrFlux->ID() ].second;
    valueFunction = Teuchos::rcp( new BCLogicalOrFunction<Scalar>(existingFunction, existingFilter,
                                  valueFunction, spatialPoints) );
    spatialPoints = Teuchos::rcp( new SpatialFilterLogicalOr( existingFilter, spatialPoints ) );
    //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only one Dirichlet condition is allowed per variable.");
  }
  _dirichletBCs[ traceOrFlux->ID() ] = make_pair( spatialPoints, valueFunction );
}

template <typename Scalar>
void TBC<Scalar>::addZeroMeanConstraint( VarPtr field )
{
  if ( field->varType() != FIELD )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraints only supported for field vars");
  }
  _zeroMeanConstraints.insert( field->ID() );
}

template <typename Scalar>
void TBC<Scalar>::removeZeroMeanConstraint( int fieldID )
{
  if (_zeroMeanConstraints.find(fieldID) != _zeroMeanConstraints.end())
  {
    _zeroMeanConstraints.erase( _zeroMeanConstraints.find(fieldID) );
  }
}

template <typename Scalar>
void TBC<Scalar>::addSinglePointBC( int fieldID, Scalar value, MeshPtr mesh, GlobalIndexType vertexNumber )
{
  TEUCHOS_TEST_FOR_EXCEPTION(mesh->getTopology()->isDistributed(), std::invalid_argument, "addSinglePointBC() not supported for distributed MeshTopology.  Use addSpatialPointBC instead");
  if (vertexNumber == -1) vertexNumber = 0;
  Camellia::FieldContainer<double> vertexCoords = mesh->vertexCoordinates(vertexNumber);
  vector<double> vertex(vertexCoords.size());
  for (int i=0; i<vertex.size(); i++)
    vertex[i] = vertexCoords[i];
  this->addSpatialPointBC(fieldID,value,vertex);
}

// ! Adds point constraint at the specified spatial point (which must correspond to a vertex in the mesh).  For space-time meshes, indicates that the value should be imposed at every temporal degree of freedom corresponding to the spatial point.
template <typename Scalar>
void TBC<Scalar>::addSpatialPointBC(int fieldID, Scalar value, vector<double> spatialPoint)
{
  _singlePointBCs[ fieldID ] = make_pair(spatialPoint, value);
}

// ! Remove the specified point constraint.
template <typename Scalar>
void TBC<Scalar>::removeSpatialPointBC(int fieldID)
{
  if (_singlePointBCs.find(fieldID) != _singlePointBCs.end())
  {
    _singlePointBCs.erase(fieldID);
  }
}

template <typename Scalar>
map< int, TDirichletBC<Scalar> > & TBC<Scalar>::dirichletBCs()
{
  return _dirichletBCs;
}

template <typename Scalar>
TBCPtr<Scalar> TBC<Scalar>::copyImposingZero()
{
  //returns a copy of this BC object, except with all zero Functions
  TBCPtr<Scalar> zeroBC = Teuchos::rcp( new TBC<Scalar>(*this) );
  map< int, TDirichletBC<Scalar> >* dirichletBCs = &(zeroBC->dirichletBCs());
  for (typename map< int, TDirichletBC<Scalar> >::iterator bcIt = dirichletBCs->begin();
       bcIt != dirichletBCs->end(); ++bcIt)
  {
    bcIt->second.second = TFunction<Scalar>::zero();
  }

  for (auto spatialPointBC : _singlePointBCs)
  {
    int trialID = spatialPointBC.first;
    vector<double> vertex = spatialPointBC.second.first;
    Scalar zero = 0.0;
    zeroBC->addSpatialPointBC(trialID, zero, vertex);
  }

  // TODO: copy geometric BC entries...
//  for (auto geometricBCEntry : _geometricBCs)
//  {
//    int varID = geometricBCEntry.first;
//    auto geometricPrescriptions = geometricBCEntry.second;
//    for (auto geometricPrescription : geometricPrescriptions)
//    {
//      zeroBC->addDirichlet();
//    }
//  }
  
  return zeroBC;
}

template <typename Scalar>
pair< SpatialFilterPtr, TFunctionPtr<Scalar> > TBC<Scalar>::getDirichletBC(int varID)
{
  if (_dirichletBCs.find(varID) == _dirichletBCs.end())
  {
    cout << "No Dirichlet BC for the indicated variable...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No Dirichlet BC for the indicated variable...");
  }
  return _dirichletBCs[varID];
}

template <typename Scalar>
map< int, vector<pair<VarPtr, TFunctionPtr<Scalar>>>> TBC<Scalar>::getDirichletTagBCs() const
{
  return _dirichletTagBCs;
}

template <typename Scalar>
TFunctionPtr<Scalar> TBC<Scalar>::getSpatiallyFilteredFunctionForDirichletBC(int varID)
{
  pair< SpatialFilterPtr, TFunctionPtr<Scalar> > dirichletBC = getDirichletBC(varID);

  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(dirichletBC.second, dirichletBC.first) );
}

template <typename Scalar>
bool TBC<Scalar>::isLegacySubclass()
{
  return _legacyBCSubclass;
}

template <typename Scalar>
void TBC<Scalar>::setTime(double time)
{
  _time = time;
  for (typename map< int, TDirichletBC<Scalar> >::iterator bcIt = dirichletBCs().begin();
       bcIt != dirichletBCs().end(); ++bcIt)
  {
    bcIt->second.second->setTime(time);
  }
}

template <typename Scalar>
bool TBC<Scalar>::bcsImposed(int varID)
{
  // returns true if there are any BCs anywhere imposed on varID
  if (_legacyBCSubclass)
  {
    cout << "legacy BC subclasses must override bcsImposed().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "legacy BC subclasses must override bcsImposed().");
  }
  else
  {
    return _dirichletBCs.find(varID) != _dirichletBCs.end();
  }
}

template <typename Scalar>
void TBC<Scalar>::imposeBC(Camellia::FieldContainer<Scalar> &dirichletValues, Camellia::FieldContainer<bool> &imposeHere,
                           int varID, Camellia::FieldContainer<double> &unitNormals, BasisCachePtr basisCache)
{
  if (_legacyBCSubclass)
  {
    // by default, call legacy version:
    // (basisCache->getPhysicalCubaturePointsFC() doesn't really return *cubature* points, but the boundary points
    //  that we're interested in)
    Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePointsFC();
    imposeBC(varID,physicalPoints,unitNormals,dirichletValues,imposeHere);
  }
  else
  {
    const Camellia::FieldContainer<double> & physicalPoints = basisCache->getPhysicalCubaturePointsFC();

    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);

    TEUCHOS_TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells )
                                || ( dirichletValues.dimension(1) != numPoints )
                                || ( dirichletValues.rank() != 2  ),
                                std::invalid_argument,
                                "dirichletValues dimensions should be (numCells,numPoints).");
    TEUCHOS_TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells )
                                || ( imposeHere.dimension(1) != numPoints )
                                || ( imposeHere.rank() != 2  ),
                                std::invalid_argument,
                                "imposeHere dimensions should be (numCells,numPoints).");

    imposeHere.initialize(false);
    if ( _dirichletBCs.find(varID) == _dirichletBCs.end() )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Attempt to impose BC on varID without BCs.");
    }

    TDirichletBC<Scalar> bc = _dirichletBCs[varID];
    SpatialFilterPtr filter = bc.first;
    TFunctionPtr<Scalar> f = bc.second;

    if (basisCache->isSideCache())
    {
      unsigned sideOrdinal = basisCache->getSideIndex();
      // for non-space-time BasisCaches, every side is spatial
      bool isSpatialSide = !basisCache->cellTopologyIsSpaceTime() || basisCache->cellTopology()->sideIsSpatial(sideOrdinal);
      if (isSpatialSide && filter->matchesSpatialSides())
      {
        filter->matchesPoints(imposeHere,basisCache);
        f->values(dirichletValues,basisCache);
      }
      else if (!isSpatialSide && filter->matchesTemporalSides())
      {
        filter->matchesPoints(imposeHere,basisCache);
        f->values(dirichletValues,basisCache);
      }
    }
    else
    {
      filter->matchesPoints(imposeHere,basisCache);
      f->values(dirichletValues,basisCache);
    }
  }
}

template <typename Scalar>
void TBC<Scalar>::imposeBC(DeviceView<Scalar> dirichletValues, DeviceView<bool> imposeHere,
                           int varID, DeviceViewConst<Scalar> unitNormals, BasisCachePtr basisCache)
{
  Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePointsFC();
  
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( dirichletValues.extent_int(0) != numCells )
                             || ( dirichletValues.extent_int(1) != numPoints )
                             || ( dirichletValues.rank() != 2  ),
                             std::invalid_argument,
                             "dirichletValues dimensions should be (numCells,numPoints).");
  TEUCHOS_TEST_FOR_EXCEPTION( ( imposeHere.extent_int(0) != numCells )
                             || ( imposeHere.extent_int(1) != numPoints )
                             || ( imposeHere.rank() != 2  ),
                             std::invalid_argument,
                             "imposeHere dimensions should be (numCells,numPoints).");
  
  // Not sure why this exception is thrown -- where do we assume spaceDim ≤ 3 -- commenting out for now.
//  int spaceDim = physicalPoints.dimension(2);
//  TEUCHOS_TEST_FOR_EXCEPTION( spaceDim > 3, std::invalid_argument,
//                             "spaceDim > 3 not yet supported by imposeBC." );
  
  Kokkos::deep_copy(imposeHere,false);
  
  if ( _dirichletBCs.find(varID) == _dirichletBCs.end() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Attempt to impose BC on varID without BCs.");
  }
  
  TDirichletBC<Scalar> bc = _dirichletBCs[varID];
  SpatialFilterPtr filter = bc.first;
  TFunctionPtr<Scalar> f = bc.second;
  
  auto & points   = basisCache->getPhysicalCubaturePointsFC();
  auto pointsView = points.checkoutDeviceView();
  
  if (basisCache->isSideCache())
  {
    unsigned sideOrdinal = basisCache->getSideIndex();
    // for non-space-time BasisCaches, every side is spatial
    bool isSpatialSide = !basisCache->cellTopologyIsSpaceTime() || basisCache->cellTopology()->sideIsSpatial(sideOrdinal);
    
    if (isSpatialSide && filter->matchesSpatialSides())
    {
      filter->matchesPoints(imposeHere,pointsView);
      points.checkinDeviceView(false); // false: don't deep copy to host (no changes to points)
      f->values(dirichletValues,ASSIGN,basisCache);
    }
    else if (!isSpatialSide && filter->matchesTemporalSides())
    {
      filter->matchesPoints(imposeHere,pointsView);
      points.checkinDeviceView(false); // false: don't deep copy to host (no changes to points)
      f->values(dirichletValues,ASSIGN,basisCache);
    }
  }
  else
  {
    filter->matchesPoints(imposeHere,pointsView);
    points.checkinDeviceView(false); // false: don't deep copy to host (no changes to points)
    f->values(dirichletValues,ASSIGN,basisCache);
  }
}

template <typename Scalar>
void TBC<Scalar>::imposeBC(int varID, Camellia::FieldContainer<double> &physicalPoints,
                           Camellia::FieldContainer<double> &unitNormals,
                           Camellia::FieldContainer<Scalar> &dirichletValues,
                           Camellia::FieldContainer<bool> &imposeHere)
{
  if (_legacyBCSubclass)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TBC<Scalar>::imposeBC unimplemented.");
  }
  else
  {
    cout << "ERROR: this version of imposeBC (the singleton version) is only supported by legacy subclasses.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "this version of imposeBC (the singleton version) is only supported by legacy subclasses.");
//    if (_singlePointBCs.find(varID) == _singlePointBCs.end()) {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "this version of TBC<Scalar>::imposeBC only supports singleton points.");
//    }
//    DirichletBC bc = _singlePointBCs[varID];
//    SpatialFilterPtr filter = bc.first;
//    TFunctionPtr<double> f = bc.second;
//
//
//
//    BasisCachePtr basisCache = Teuchos::rcp( new PhysicalPointCache(physicalPoints) );
//
//    filter->matchesPoints(imposeHere,basisCache);
//    f->values(dirichletValues,basisCache);
//
////    cout << "TBC<Scalar>::imposeBC (singleton BC implementation) called for varID " << varID << endl;
//
//    bool pointMatched = false; // make sure we just impose this once
//    for (int i=0; i<imposeHere.size(); i++) {
//      if (imposeHere[i]) {
//        if (pointMatched) {
//          // then don't impose here
//          imposeHere[i] = false;
//        } else {
//          pointMatched = true;
//        }
//      }
//    }
  }
}

template <typename Scalar>
bool TBC<Scalar>::singlePointBC(int varID)   // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
{
  if (_legacyBCSubclass)
  {
    return false;
  }
  else
  {
    return _singlePointBCs.find(varID) != _singlePointBCs.end();
  }
}

template <typename Scalar>
bool TBC<Scalar>::shouldImposeZeroMeanConstraint(int varID)
{
  if (_legacyBCSubclass)
  {
    return false;
  }
  else
  {
    return _zeroMeanConstraints.find(varID) != _zeroMeanConstraints.end();
  }
}

// basisCoefficients has dimensions (C,F)
template <typename Scalar>
void TBC<Scalar>::coefficientsForBC(Camellia::FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction<Scalar>> bcFxn,
                                    BasisPtr basis, BasisCachePtr sideBasisCache)
{
  int numFieldsExpected;
  int sideDim = sideBasisCache->cellTopology()->getDimension() - 1;
  if (sideDim == basis->domainTopology()->getDimension())
  {
    numFieldsExpected = basis->getCardinality();
  }
  else
  {
    
    unsigned sideOrdinal = sideBasisCache->getSideIndex();
    numFieldsExpected = basis->dofOrdinalsForSide(sideOrdinal).size();
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION( basisCoefficients.dimension(1) != numFieldsExpected, std::invalid_argument, "inconsistent basisCoefficients dimensions");

  Projector<double>::projectFunctionOntoBasisInterpolating(basisCoefficients, bcFxn, basis, sideBasisCache);
}

template <typename Scalar>
void TBC<Scalar>::removeSinglePointBC(int fieldID)
{
  if (_singlePointBCs.find(fieldID) != _singlePointBCs.end())
  {
    _singlePointBCs.erase(fieldID);
  }
}

template <typename Scalar>
Scalar TBC<Scalar>::valueForSinglePointBC(int varID)
{
  if (_singlePointBCs.find(varID) != _singlePointBCs.end())
    return _singlePointBCs[varID].second;
  else
    return -1;
}

template <typename Scalar>
vector<double> TBC<Scalar>::pointForSpatialPointBC(int varID)
{
  if (_singlePointBCs.find(varID) != _singlePointBCs.end())
    return _singlePointBCs[varID].first;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No spatial point BC imposed for variable");
}

template <typename Scalar>
set<int> TBC<Scalar>::getZeroMeanConstraints()
{
  return _zeroMeanConstraints;
}

template <typename Scalar>
TBCPtr<Scalar> TBC<Scalar>::bc()
{
  return Teuchos::rcp(new TBC<Scalar>(false)); // false: not legacy subclass
}

namespace Camellia
{
template class TBC<double>;
}
