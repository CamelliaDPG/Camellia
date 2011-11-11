#include "Projector.h"
#include <stdlib.h> //for vectors

#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;

// currently restrict ourselves to only projecting L2 inner products
Projector::Projector(Teuchos::RCP<L2InnerProduct>ip){
  _ip = ip; 
}

void Projector::projectFunction(Teuchos::RCP<Solution> solution, Teuchos::RCP<Mesh> mesh, int trialID, Teuchos::RCP<AbstractFunction>fxn){    
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
    FieldContainer<double> allPhysicalCellNodesForType = mesh->physicalCellNodes(elemTypePtr);
    int totalCellsForType = allPhysicalCellNodesForType.dimension(0);
    Teuchos::Array<int> nodeDimensions, parityDimensions;
    for (int cellIndex=0;cellIndex < totalCellsForType;cellIndex++) {
      int numCells = 1; // otherwise the batch size
      nodeDimensions[0] = numCells;
      FieldContainer<double> physicalCellNodes(nodeDimensions,&allPhysicalCellNodesForType(cellIndex,0,0));
      
      CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
      FieldContainer<double> ipMatrix;  
      _ip->computeInnerProductMatrix(ipMatrix,trialOrderingPtr, *(cellTopoPtr.get()),physicalCellNodes);
      
      FieldContainer<double> ipVector;
      _ip->computeInnerProductVector(ipVector,trialOrderingPtr, *(cellTopoPtr.get()),physicalCellNodes,fxn); // TODO - write this routine as part 

      /*
      Epetra_SerialDenseMatrix solnCoeffs;
      Epetra_SerialDenseSolver solver;     
      Epetra_SerialDenseMatrix ipMatrixCopy;
      Epetra_SerialDenseMatrix ipVectorCopy;
      solver.setMatrix(ipMatrixCopy);
      solver.setVectors(solnCoeffs,ipVectorCopy)
      */
     
      FieldContainer<double> solnCoeffsToSet; // need to copy solution to these

      vector<int> trialIDs = mesh->bilinearForm().trialIDs();
      vector<int>::iterator trialIterator;
      for (trialIterator=trialIDs.begin();trialIterator!=trialIDs.end();trialIterator++){
	int trialID = *trialIterator;
	int sideIndex = 0; //corresponding to only setting field variables in the "in"side at the moment.
	solution->setSolnCoeffsForCellID(solnCoeffsToSet, cellIndex, trialID, sideIndex);
      }
      
    }//over cells
  }//
}
