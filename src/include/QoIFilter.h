//
//  QOIFilter.h
//  Camellia-debug
//
//  Created by Nate Roberts on 10/22/13.
//
//

#ifndef Camellia_debug_QOIFilter_h
#define Camellia_debug_QOIFilter_h

#include "TypeDefs.h"

#include "LocalStiffnessMatrixFilter.h"
#include "LinearTerm.h"

class QoIFilter : public LocalStiffnessMatrixFilter {
private:
  LinearTermPtr _qoi;
public:
  QoIFilter(LinearTermPtr qoi) {
    _qoi = qoi;
  }
  
  void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector,
              BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) {
    // localStiffnessMatrix has dimensions (C,F,F) where F are the fields for the trial space basis
    int numCells  = localStiffnessMatrix.dimension(0);
    int numFields = localStiffnessMatrix.dimension(1);
    
    TEUCHOS_TEST_FOR_EXCEPTION(numFields != localStiffnessMatrix.dimension(2), std::invalid_argument,
                               "localStiffnessMatrix.dim(1) != localStiffnessMatrix.dim(2)");
    TEUCHOS_TEST_FOR_EXCEPTION(numFields != localRHSVector.dimension(1), std::invalid_argument,
                               "localRHSVector.dim(1) != localStiffnessMatrix.dim(1)");
    
    // Assumes that all elements are of like type--but they'd have to be, to have a single localStiffness FC
    ElementTypePtr elemType = mesh->getElement(basisCache->cellIDs()[0])->elementType();
    
    FieldContainer<double> qoiLoad(numCells,numFields);
    _qoi->integrate(qoiLoad,elemType->trialOrderPtr,basisCache);
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int i=0; i<numFields; i++) {
        localRHSVector(cellIndex,i) += qoiLoad(cellIndex,i);
      }
    }
  }
  
  static Teuchos::RCP<LocalStiffnessMatrixFilter> qoiFilter(LinearTermPtr qoi) {
    return Teuchos::rcp( new QoIFilter(qoi) );
  }
};

#endif
