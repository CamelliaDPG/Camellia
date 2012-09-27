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

using namespace Intrepid;

class BC {
public:
  virtual bool bcsImposed(int varID) = 0; // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                        int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache);
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere);
  
  virtual bool singlePointBC(int varID);
  
  virtual bool imposeZeroMeanConstraint(int varID);
  // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
  
  // basisCoefficients has dimensions (C,F)
  virtual void coefficientsForBC(FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction> bcFxn, BasisPtr basis, BasisCachePtr sideBasisCache);
};

typedef Teuchos::RCP<BC> BCPtr;


#endif