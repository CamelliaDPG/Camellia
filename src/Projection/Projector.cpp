#include "Projector.h"
#include <stdlib.h> 

#include "Shards_CellTopology.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Intrepid_FieldContainer.hpp"
// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "BasisValueCache.h"
#include "BasisFactory.h"

#include <Epetra_SerialDenseVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_LAPACK.h>
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"


typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn, Teuchos::RCP< Basis<double,FieldContainer<double> > > basis, const FieldContainer<double> &physicalCellNodes) {

  shards::CellTopology cellTopo = basis->getBaseCellTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());

  int basisRank = BasisFactory::getBasisRank(basis);
  int ID = 0; // only one entry for this fake dofOrderPtr
  dofOrderPtr->addEntry(ID,basis,basisRank);
  int maxTrialDegree = dofOrderPtr->maxBasisDegree();

  // do not build side caches - no projections for sides supported at the moment
  BasisValueCache basisCache(physicalCellNodes, cellTopo, *(dofOrderPtr), maxTrialDegree, false);
  // assume only L2 projections
  EOperatorExtended op = IntrepidExtendedTypes::OPERATOR_VALUE;
  
  // have information, build inner product matrix
  int numDofs = basis->getCardinality();
  FieldContainer<double> cubPoints = basisCache.getPhysicalCubaturePoints();    
  FieldContainer<double> basisValues = *(basisCache.getTransformedValues(basis, op));
  FieldContainer<double> testBasisValues = *basisCache.getTransformedWeightedValues(basis, op);
  
  FieldContainer<double> functionValues;
  fxn->getValues(functionValues, cubPoints);
  
  int numCells = physicalCellNodes.dimension(0);

  FieldContainer<double> gramMatrix(numCells,numDofs,numDofs);
  FieldContainer<double> ipVector(numCells,numDofs);
  FunctionSpaceTools::integrate<double>(gramMatrix,basisValues,testBasisValues,COMP_CPP);
  FunctionSpaceTools::integrate<double>(ipVector,functionValues,testBasisValues,COMP_CPP); 

  basisCoefficients.resize(numCells,numDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++){

    Epetra_SerialDenseSolver solver;

    Epetra_SerialDenseMatrix A(Copy,
			       &gramMatrix(cellIndex,0,0),
			       gramMatrix.dimension(2), 
			       gramMatrix.dimension(2),  
			       gramMatrix.dimension(1)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    Epetra_SerialDenseVector b(Copy,
			       &functionValues(cellIndex,0),
			       functionValues.dimension(1));
    
    Epetra_SerialDenseVector x(functionValues.dimension(1));
    
    solver.SetMatrix(A);
    int info = solver.SetVectors(x,b);
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to SetVectors with error " << info << endl;
    }
    info = solver.Solve();
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to solve with error " << info << endl;
    }
    for (int i=0;i<numDofs;i++){
      basisCoefficients(cellIndex,i) = x(i);
    }   
    
  } 
}

