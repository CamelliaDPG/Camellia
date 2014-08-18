//
//  TikhonovRegularizationFilter.h
//  Camellia-debug
//
//  Created by Truman Ellis on 8/15/14.
//
//

#ifndef Camellia_debug_TikhonovRegularizationFilter_h
#define Camellia_debug_TikhonovRegularizationFilter_h

#include "LocalStiffnessMatrixFilter.h"
#include "LinearTerm.h"

// #include "Intrepid_FunctionSpaceTools.hpp"

class TikhonovRegularizationFilter : public LocalStiffnessMatrixFilter {
private:
  LinearTermPtr _lt;
public:
  TikhonovRegularizationFilter(LinearTermPtr lt) {
    _lt = lt;
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
    DofOrderingPtr dofOrdering = elemType->trialOrderPtr;

    FieldContainer<double> regularizationMatrix(numCells,numFields,numFields);
    _lt->integrate(regularizationMatrix,dofOrdering,_lt,dofOrdering,basisCache,basisCache->isSideCache());

    // IntrepidExtendedTypes::EOperatorExtended trialOperator =  IntrepidExtendedTypes::OP_VALUE;
    // int trialID = 0;
    // BasisPtr trialBasis1 = trialOrderPtr->getBasis(trialID);
    // FieldContainer<double> trialValuesTransformed = *(basisCache->getTransformedValues(trialBasis1,trialOperator));
    
    // FunctionSpaceTools::integrate<double>(regularizationMatrix,trialValuesTransformed,trialValuesTransformed,COMP_CPP);
    // cout << regularizationMatrix << endl;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
      for (int i=0; i<numFields; i++)
        for (int j=0; j<numFields; j++)
          localStiffnessMatrix(cellIndex,i,j) += regularizationMatrix(cellIndex,i,j);
  }
  
  // static Teuchos::RCP<LocalStiffnessMatrixFilter> qoiFilter(LinearTermPtr qoi) {
  //   return Teuchos::rcp( new TikhonovRegularizationFilter(qoi) );
  // }
};

#endif
