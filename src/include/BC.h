#ifndef DPG_BC
#define DPG_BC

/*
 *  BC.h
 *
 */

// abstract class

#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "BCFunction.h"
#include "SpatialFilter.h"

namespace Camellia
{
template <typename Scalar>
class TBC
{
  bool _legacyBCSubclass;
  set< int > _zeroMeanConstraints; // variables on which ZMCs imposed
  map< int, pair< vector<double>, Scalar> > _singlePointBCs; // variables on which single-point conditions imposed
  map< int, TDirichletBC<Scalar> > _dirichletBCs; // key: trialID
protected:
  map< int, TDirichletBC<Scalar> > &dirichletBCs();
  double _time;

public:
  TBC(bool legacySubclass) : _legacyBCSubclass(legacySubclass) {}
  virtual bool bcsImposed(int varID); // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(Intrepid::FieldContainer<Scalar> &dirichletValues, Intrepid::FieldContainer<bool> &imposeHere,
                        int varID, Intrepid::FieldContainer<double> &unitNormals, BasisCachePtr basisCache);

  virtual void imposeBC(int varID, Intrepid::FieldContainer<double> &physicalPoints,
                        Intrepid::FieldContainer<double> &unitNormals,
                        Intrepid::FieldContainer<Scalar> &dirichletValues,
                        Intrepid::FieldContainer<bool> &imposeHere);

  virtual bool singlePointBC(int varID); // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
  virtual Scalar valueForSinglePointBC(int varID);
  virtual vector<double> pointForSpatialPointBC(int varID);

  virtual bool shouldImposeZeroMeanConstraint(int varID);

  bool isLegacySubclass();

  // basisCoefficients has dimensions (C,F)
  virtual void coefficientsForBC(Intrepid::FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction<Scalar>> bcFxn, BasisPtr basis, BasisCachePtr sideBasisCache);

  virtual ~TBC() {}

  void addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, TFunctionPtr<Scalar> valueFunction );
  
  // ! Adds point constraint at the specified vertex number in the specified spatial mesh.  Deprecated; use addSpatialPointBC instead (even for pure-spatial meshes).
  void addSinglePointBC( int fieldID, Scalar value, MeshPtr spatialMesh, GlobalIndexType meshVertexNumber = -1 );
  
  // ! Adds point constraint at the specified spatial point (which must correspond to a vertex in the mesh).  Deprecated; use addSpatialPointBC instead (even for pure-spatial meshes).  For space-time meshes, indicates that the value should be imposed at every temporal degree of freedom corresponding to the spatial point.
  void addSpatialPointBC(int fieldID, Scalar value, vector<double> spatialPoint);
  // ! Remove the specified point constraint.
  void removeSpatialPointBC(int fieldID);

  // ! Remove the specified point constraint.  Deprecated version; use removeSpatialPointBC() instead.
  void removeSinglePointBC( int fieldID );
  
  void addZeroMeanConstraint( VarPtr field );
  void removeZeroMeanConstraint( int fieldID );

  TBCPtr<Scalar> copyImposingZero();//returns a copy of this BC object, except with all zero Functions

  void setTime(double time);
  double getTime()
  {
    return _time;
  }

  pair< SpatialFilterPtr, TFunctionPtr<Scalar> > getDirichletBC(int varID);

  TFunctionPtr<Scalar> getSpatiallyFilteredFunctionForDirichletBC(int varID);

  set<int> getZeroMeanConstraints();

  static TBCPtr<Scalar> bc();
};

extern template class TBC<double>;
}


#endif
