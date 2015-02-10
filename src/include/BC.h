#ifndef DPG_BC
#define DPG_BC

/*
 *  BC.h
 *
 */

// abstract class

#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "Projector.h"
#include "BCFunction.h"
#include "SpatialFilter.h"

using namespace Intrepid;

class BC {
  bool _legacyBCSubclass;
  
  typedef pair< SpatialFilterPtr, FunctionPtr > DirichletBC;
  set< int > _zeroMeanConstraints; // variables on which ZMCs imposed
  map< int, pair<GlobalIndexType,double> > _singlePointBCs; // variables on which single-point conditions imposed
  map< int, DirichletBC > _dirichletBCs; // key: trialID
protected:
  map< int, DirichletBC > &dirichletBCs();
  double _time;
  
public:
  BC(bool legacySubclass) : _legacyBCSubclass(legacySubclass) {}
  virtual bool bcsImposed(int varID); // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                        int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache);
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere);
  
  virtual bool singlePointBC(int varID); // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
  virtual double valueForSinglePointBC(int varID); 
  virtual GlobalIndexType vertexForSinglePointBC(int varID); // returns -1 when no vertex was specified for the imposition
  
  virtual bool imposeZeroMeanConstraint(int varID);
  
  bool isLegacySubclass();
  
  // basisCoefficients has dimensions (C,F)
  virtual void coefficientsForBC(FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction> bcFxn, BasisPtr basis, BasisCachePtr sideBasisCache);
  
  virtual ~BC() {}
  
  void addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, FunctionPtr valueFunction );
  void addSinglePointBC( int fieldID, double value, GlobalIndexType meshVertexNumber = -1 );
  void removeSinglePointBC( int fieldID );
  void addZeroMeanConstraint( VarPtr field );
  void removeZeroMeanConstraint( int fieldID );
  
  Teuchos::RCP<BC> copyImposingZero();//returns a copy of this BC object, except with all zero Functions
  
  void setTime(double time);
  double getTime() { return _time; }
  
  pair< SpatialFilterPtr, FunctionPtr > getDirichletBC(int varID);
  
  FunctionPtr getSpatiallyFilteredFunctionForDirichletBC(int varID);
  
  static Teuchos::RCP<BC> bc();
};

typedef Teuchos::RCP<BC> BCPtr;


#endif