//
//  PenaltyConstraints.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_PenaltyConstraints_h
#define Camellia_PenaltyConstraints_h

#include "LocalStiffnessMatrixFilter.h"
#include "SpatialFilter.h"
#include "Constraint.h"

class PenaltyConstraints : public LocalStiffnessMatrixFilter {
private:
  vector< Constraint > _constraints;
public: 
  void addConstraint(const Constraint &c, SpatialFilterPtr sf) {
    Constraint sfc = Constraint::spatiallyFilteredConstraint(c,sf);
    _constraints.push_back(sfc);
  }
  void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector, 
              BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) {
    // localStiffnessMatrix has dimensions (C,F,F) where F are the fields for the trial space basis
    int numCells  = localStiffnessMatrix.dimension(0);
    int numFields = localStiffnessMatrix.dimension(1);
    
    double penaltyWeight = 1e7;
    
    // Assumes that all elements are of like type--but they'd have to be, to have a single localStiffness FC
    ElementTypePtr elemType = mesh->getElement(basisCache->cellIDs()[0])->elementType();
    
    FieldContainer<double> constraintMatrix(numCells,numFields,numFields);
    FieldContainer<double> constraintLoad(numCells,numFields);
    for (vector<Constraint>::iterator constIt = _constraints.begin(); constIt != _constraints.end(); constIt++) {
      LinearTermPtr lt = constIt->linearTerm();
      FunctionPtr f = constIt->f();
      
      lt->integrate(constraintMatrix,elemType->trialOrderPtr,lt,elemType->trialOrderPtr,basisCache);
      lt->integrate(constraintLoad,elemType->trialOrderPtr,f,basisCache);
    }
//    cout << "constraintMatrix:\n" << constraintMatrix;
//    cout << "constraintLoad:\n" << constraintLoad;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int i=0; i<numFields; i++) {
        localRHSVector(cellIndex,i) += penaltyWeight * constraintLoad(cellIndex,i);
        for (int j=0; j<numFields; j++) {
          localStiffnessMatrix(cellIndex,i,j) += penaltyWeight * constraintMatrix(cellIndex,i,j);
        }
      }
    }
  }
};

#endif
