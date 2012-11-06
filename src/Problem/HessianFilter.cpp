#include "HessianFilter.h"

#include "BasisCache.h"
#include "DofOrdering.h"

void HessianFilter::filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector, BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc){

  //  cout << "numCells = " << localStiffnessMatrix.dimension(0) << ", numTestDofs = " << localStiffnessMatrix.dimension(1) << ", numTrialDofs = " << localStiffnessMatrix.dimension(2) << endl;

  vector<int> cellIDs = basisCache->cellIDs();
  int numCells = cellIDs.size();
  if (numCells>0){
    ElementTypePtr elemTypePtr = mesh->elements()[cellIDs[0]]->elementType(); // assumes all elements in basisCache are of the same type.

    int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
    int numTestDofs = elemTypePtr->testOrderPtr->totalDofs();
    //    cout << "num Cells = " << numCells << ",num Trial dofs = " << numTrialDofs << ", numTest = " << numTestDofs << endl;

    FieldContainer<double> cellSideParities = mesh->cellSideParities(elemTypePtr);

    FieldContainer<double> hessianStiffness(numCells,numTrialDofs,numTrialDofs );
    _hessianBF->bubnovStiffness(hessianStiffness, elemTypePtr, cellSideParities, basisCache);
    //    FieldContainer<double> hessianStiffness(numCells,numTestDofs,numTrialDofs);
    //    _hessianBF->stiffnessMatrix(hessianStiffness, elemTypePtr, cellSideParities, basisCache, false);

    for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
      for (int i = 0;i<numTrialDofs;i++){
	for (int j = 0;j<numTrialDofs;j++){
	  localStiffnessMatrix(cellIndex,i,j) += hessianStiffness(cellIndex,i,j);
	}    
      }
    }
  }
}
