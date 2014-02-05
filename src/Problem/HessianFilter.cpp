#include "HessianFilter.h"

#include "BasisCache.h"
#include "DofOrdering.h"

void HessianFilter::filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector, BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc){

  //  cout << "numCells = " << localStiffnessMatrix.dimension(0) << ", numTestDofs = " << localStiffnessMatrix.dimension(1) << ", numTrialDofs = " << localStiffnessMatrix.dimension(2) << endl;

  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  GlobalIndexType numCells = cellIDs.size();
  if (numCells>0){
    ElementTypePtr elemTypePtr = mesh->getElement(cellIDs[0])->elementType(); // assumes all elements in basisCache are of the same type.

    int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
    //    cout << "num Cells = " << numCells << ",num Trial dofs = " << numTrialDofs << ", numTest = " << numTestDofs << endl;

    FieldContainer<double> cellSideParities = mesh->cellSideParities(elemTypePtr);

    FieldContainer<double> hessianStiffness(numCells,numTrialDofs,numTrialDofs );
    _hessianBF->bubnovStiffness(hessianStiffness, elemTypePtr, cellSideParities, basisCache);
    //    FieldContainer<double> hessianStiffness(numCells,numTestDofs,numTrialDofs);
    //    _hessianBF->stiffnessMatrix(hessianStiffness, elemTypePtr, cellSideParities, basisCache, false);

    /*
    double maxDiff = 0.0;
    for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
      for (int i = 0;i<numTrialDofs;i++){
        for (int j = 0;j<numTrialDofs;j++){
          maxDiff = max(maxDiff,abs(hessianStiffness(cellIndex,i,j)-hessianStiffness(cellIndex,j,i)));
          cout << hessianStiffness(cellIndex,i,j) << " ";
        }
        cout << endl;
      }
    }
    cout << "symmetry max diff in Hessian = " << maxDiff << endl;

    for (int rank = 0; rank < hessianStiffness.rank(); rank++){
      cout << "hessian stiffness dimension " << rank << " = " << hessianStiffness.dimension(rank) << endl;
    }
    for (int rank = 0; rank < localStiffnessMatrix.rank(); rank++){
      cout << "local stiffness dimension " << rank << " = " << localStiffnessMatrix.dimension(rank) << endl;
    }
    */

    for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
      for (int i = 0;i<numTrialDofs;i++){
        for (int j = 0;j<numTrialDofs;j++){
          localStiffnessMatrix(cellIndex,i,j) += hessianStiffness(cellIndex,i,j);
        }    
      }
    }
  }
}
