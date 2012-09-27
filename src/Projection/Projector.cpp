#include "Projector.h"
#include <stdlib.h> 

#include "Shards_CellTopology.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Intrepid_FieldContainer.hpp"
// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"

#include <Epetra_SerialDenseVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_LAPACK.h>
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;

typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, FunctionPtr fxn, 
                                         BasisPtr basis, BasisCachePtr basisCache, IPPtr ip, VarPtr v) {
  shards::CellTopology cellTopo = basis->getBaseCellTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());
  
  // have information, build inner product matrix
  int numDofs = basis->getCardinality();
  const FieldContainer<double> *cubPoints = &(basisCache->getPhysicalCubaturePoints());
  
  int numCells = cubPoints->dimension(0);
  int numPts = cubPoints->dimension(1);
  FieldContainer<double> functionValues(numCells,numPts);
  fxn->values(functionValues, basisCache);  
  
  FieldContainer<double> gramMatrix(numCells,numDofs,numDofs);
  FieldContainer<double> ipVector(numCells,numDofs);
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering );
  dofOrdering->addEntry(v->ID(), basis, v->rank());
  ip->computeInnerProductMatrix(gramMatrix, dofOrdering, basisCache);
  ip->computeInnerProductVector(ipVector, v, fxn, dofOrdering, basisCache);
  
  basisCoefficients.resize(numCells,numDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
    
    Epetra_SerialDenseSolver solver;
    
    Epetra_SerialDenseMatrix A(Copy,
                               &gramMatrix(cellIndex,0,0),
                               gramMatrix.dimension(2), 
                               gramMatrix.dimension(2),  
                               gramMatrix.dimension(1)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    Epetra_SerialDenseVector b(Copy,
                               &ipVector(cellIndex,0),
                               ipVector.dimension(1));
    
    Epetra_SerialDenseVector x(gramMatrix.dimension(1));
    
    solver.SetMatrix(A);
    int info = solver.SetVectors(x,b);
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to SetVectors with error " << info << endl;
    }
    
    bool equilibrated = false;
    if (solver.ShouldEquilibrate()){
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();      
      equilibrated = true;
    }   
    
    info = solver.Solve();
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to solve with error " << info << endl;
    }
    
    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "projection: unequilibration FAILED with error: " << successLocal << endl;
      }
    }
    
    for (int i=0;i<numDofs;i++){
      basisCoefficients(cellIndex,i) = x(i);
    }
    
  }
}

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, FunctionPtr fxn, 
                                         BasisPtr basis, BasisCachePtr basisCache) {
  // TODO: rewrite this to use the version above (define L2 inner product for a dummy variable, then pass in this ip and varPtr)
  shards::CellTopology cellTopo = basis->getBaseCellTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());

  // assume only L2 projections
  IntrepidExtendedTypes::EOperatorExtended op =  IntrepidExtendedTypes::OP_VALUE;
  
  // have information, build inner product matrix
  int numDofs = basis->getCardinality();
  const FieldContainer<double> *cubPoints = &(basisCache->getPhysicalCubaturePoints());
  
  constFCPtr basisValues = basisCache->getTransformedValues(basis, op);
  constFCPtr testBasisValues = basisCache->getTransformedWeightedValues(basis, op);
  
  int numCells = cubPoints->dimension(0);
  int numPts = cubPoints->dimension(1);
  FieldContainer<double> functionValues(numCells,numPts);
  fxn->values(functionValues, basisCache);  
  
  FieldContainer<double> gramMatrix(numCells,numDofs,numDofs);
  FieldContainer<double> ipVector(numCells,numDofs);
  FunctionSpaceTools::integrate<double>(gramMatrix,*basisValues,*testBasisValues,COMP_CPP);
  FunctionSpaceTools::integrate<double>(ipVector,functionValues,*testBasisValues,COMP_CPP); 
  
  basisCoefficients.resize(numCells,numDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
    
    Epetra_SerialDenseSolver solver;
    
    Epetra_SerialDenseMatrix A(Copy,
                               &gramMatrix(cellIndex,0,0),
                               gramMatrix.dimension(2), 
                               gramMatrix.dimension(2),  
                               gramMatrix.dimension(1)); // stride -- fc stores in row-major order (a.o.t. SDM)
    
    Epetra_SerialDenseVector b(Copy,
                               &ipVector(cellIndex,0),
                               ipVector.dimension(1));
    
    Epetra_SerialDenseVector x(gramMatrix.dimension(1));
    
    /*
     cout << "matrix A = " << endl;
     for (int i=0;i<gramMatrix.dimension(2);i++){
     for (int j=0;j<gramMatrix.dimension(1);j++){
     cout << A(i,j) << " ";
     }
     cout << endl;
     }
     cout << endl;
     
     cout << "vector B = " << endl;
     for (int i=0;i<functionValues.dimension(1);i++){
     cout << b(i) << endl;
     }
     */
    
    solver.SetMatrix(A);
    int info = solver.SetVectors(x,b);
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to SetVectors with error " << info << endl;
    }
    
    bool equilibrated = false;
    if (solver.ShouldEquilibrate()){
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();      
      equilibrated = true;
    }   
    
    info = solver.Solve();
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to solve with error " << info << endl;
    }
    
    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "projection: unequilibration FAILED with error: " << successLocal << endl;
      }
    }
    
    for (int i=0;i<numDofs;i++){
      basisCoefficients(cellIndex,i) = x(i);
    }   
    
  } 
}

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn, Teuchos::RCP< Basis<double,FieldContainer<double> > > basis, const FieldContainer<double> &physicalCellNodes) {

  shards::CellTopology cellTopo = basis->getBaseCellTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());

  int basisRank = BasisFactory::getBasisRank(basis);
  int ID = 0; // only one entry for this fake dofOrderPtr
  dofOrderPtr->addEntry(ID,basis,basisRank);
  int maxTrialDegree = dofOrderPtr->maxBasisDegree();

  // do not build side caches - no projections for sides supported at the moment
  BasisCache basisCache(physicalCellNodes, cellTopo, *(dofOrderPtr), maxTrialDegree, false);
  // assume only L2 projections
  IntrepidExtendedTypes::EOperatorExtended op =  IntrepidExtendedTypes::OP_VALUE;

  // have information, build inner product matrix
  int numDofs = basis->getCardinality();
  FieldContainer<double> cubPoints = basisCache.getPhysicalCubaturePoints();    

  FieldContainer<double> basisValues = *(basisCache.getTransformedValues(basis, op));
  FieldContainer<double> testBasisValues = *(basisCache.getTransformedWeightedValues(basis, op));

  int numCells = physicalCellNodes.dimension(0);
  int numPts = cubPoints.dimension(1);
  FieldContainer<double> functionValues;
  fxn->getValues(functionValues, cubPoints);  

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
			       &ipVector(cellIndex,0),
			       ipVector.dimension(1));
    
    Epetra_SerialDenseVector x(gramMatrix.dimension(1));
    
    /*
    cout << "matrix A = " << endl;
    for (int i=0;i<gramMatrix.dimension(2);i++){
      for (int j=0;j<gramMatrix.dimension(1);j++){
	cout << A(i,j) << " ";
      }
      cout << endl;
    }
    cout << endl;

    cout << "vector B = " << endl;
    for (int i=0;i<functionValues.dimension(1);i++){
      cout << b(i) << endl;
    }
    */

    solver.SetMatrix(A);
    int info = solver.SetVectors(x,b);
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to SetVectors with error " << info << endl;
    }

    bool equilibrated = false;
    if (solver.ShouldEquilibrate()){
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();      
      equilibrated = true;
    }   

    info = solver.Solve();
    if (info!=0){
      cout << "projectFunctionOntoBasis: failed to solve with error " << info << endl;
    }

    if (equilibrated) {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "projection: unequilibration FAILED with error: " << successLocal << endl;
      }
    }

    for (int i=0;i<numDofs;i++){
      basisCoefficients(cellIndex,i) = x(i);
    }   
    
  } 
}

