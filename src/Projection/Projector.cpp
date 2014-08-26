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

#include "Function.h"
#include "VarFactory.h"

#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"

typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, FunctionPtr fxn, 
                                         BasisPtr basis, BasisCachePtr basisCache, IPPtr ip, VarPtr v,
                                         set<int> fieldIndicesToSkip) {
  shards::CellTopology cellTopo = basis->domainTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());
  
  if (! fxn.get()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }
  
  int cardinality = basis->getCardinality();
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numDofs = cardinality - fieldIndicesToSkip.size();
  if (numDofs==0) {
    // we're skipping all the fields, so just initialize basisCoefficients to 0 and return
    basisCoefficients.resize(numCells,cardinality);
    basisCoefficients.initialize(0);
    return;
  }
  
  FieldContainer<double> gramMatrix(numCells,cardinality,cardinality);
  FieldContainer<double> ipVector(numCells,cardinality);

  // fake a DofOrdering
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering );
  if (! basisCache->isSideCache()) {
    dofOrdering->addEntry(v->ID(), basis, v->rank());
  } else {
    dofOrdering->addEntry(v->ID(), basis, v->rank(), basisCache->getSideIndex());
  }
  
  ip->computeInnerProductMatrix(gramMatrix, dofOrdering, basisCache);
  ip->computeInnerProductVector(ipVector, v, fxn, dofOrdering, basisCache);
  
//  cout << "physical points for projection:\n" << basisCache->getPhysicalCubaturePoints();
//  cout << "gramMatrix:\n" << gramMatrix;
//  cout << "ipVector:\n" << ipVector;
  
  map<int,int> oldToNewIndices;
  if (fieldIndicesToSkip.size() > 0) {
    // the code to do with fieldIndicesToSkip might not be terribly efficient...
    // (but it's not likely to be called too frequently)
    int i_indices_skipped = 0;
    for (int i=0; i<cardinality; i++) {
      int new_index;
      if (fieldIndicesToSkip.find(i) != fieldIndicesToSkip.end()) {
        i_indices_skipped++;
        new_index = -1;
      } else {
        new_index = i - i_indices_skipped;
      }
      oldToNewIndices[i] = new_index;
    }
    
    FieldContainer<double> gramMatrixFiltered(numCells,numDofs,numDofs);
    FieldContainer<double> ipVectorFiltered(numCells,numDofs);
    // now filter out the values that we're to skip
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int i=0; i<cardinality; i++) {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered == -1) {
          continue;
        }
        ipVectorFiltered(cellIndex,i_filtered) = ipVector(cellIndex,i);
        
        for (int j=0; j<cardinality; j++) {
          int j_filtered = oldToNewIndices[j];
          if (j_filtered == -1) {
            continue;
          }
          gramMatrixFiltered(cellIndex,i_filtered,j_filtered) = gramMatrix(cellIndex,i,j);
        }
      }
    }
//    cout << "gramMatrixFiltered:\n" << gramMatrixFiltered;
//    cout << "ipVectorFiltered:\n" << ipVectorFiltered;
    gramMatrix = gramMatrixFiltered;
    ipVector = ipVectorFiltered;
  }
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
    
    // TODO: rewrite to take advantage of SerialDenseWrapper...
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
    
    basisCoefficients.resize(numCells,cardinality);
    for (int i=0;i<cardinality;i++) {
      if (fieldIndicesToSkip.size()==0) {
        basisCoefficients(cellIndex,i) = x(i);
      } else {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered==-1) {
          basisCoefficients(cellIndex,i) = 0.0;
        } else {
          basisCoefficients(cellIndex,i) = x(i_filtered);
        }
      }
    }
    
  }
}

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, FunctionPtr fxn, 
                                         BasisPtr basis, BasisCachePtr basisCache) {
  VarFactory varFactory;
  VarPtr var;
  if (! basisCache->isSideCache()) {
    if (fxn->rank()==0) {
      var = varFactory.fieldVar("dummyField");
    } else if (fxn->rank()==1) {
      var = varFactory.fieldVar("dummyField",VECTOR_L2);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "projectFunctionOntoBasis does not yet support functions of rank > 1.");
    }
  } else {
    // for present purposes, distinction between trace and flux doesn't really matter,
    // except that parities come into the IP computation for fluxes (even though they'll cancel),
    // and since basisCache doesn't necessarily have parities defined (especially in tests),
    // it's simpler all around to use traces.
    var = varFactory.traceVar("dummyTrace");
  }
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(var); // simple L^2 IP
  projectFunctionOntoBasis(basisCoefficients, fxn, basis, basisCache, ip, var);
  
  /*
  // TODO: rewrite this to use the version above (define L2 inner product for a dummy variable, then pass in this ip and varPtr)
  shards::CellTopology cellTopo = basis->domainTopology();
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
  FunctionSpaceTools::integrate<double>(gramMatrix,*basisValues,*testBasisValues,COMP_BLAS);
  FunctionSpaceTools::integrate<double>(ipVector,functionValues,*testBasisValues,COMP_BLAS); 
  
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
        
    if (cellIndex==0) {
      cout << "legacy projectFunctionOntoBasis: matrix A for cellIndex 0 = " << endl;
      for (int i=0;i<gramMatrix.dimension(2);i++){
        for (int j=0;j<gramMatrix.dimension(1);j++){
          cout << A(i,j) << " ";
        }
        cout << endl;
      }
      cout << endl;
      
      cout << "legacy projectFunctionOntoBasis: vector B for cellIndex 0 = " << endl;
      for (int i=0;i<ipVector.dimension(1);i++){
        cout << b(i) << endl;
      }
    }
    
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
  }*/
}

void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn, BasisPtr basis,
                                         const FieldContainer<double> &physicalCellNodes) {

  shards::CellTopology cellTopo = basis->domainTopology();
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
  FunctionSpaceTools::integrate<double>(gramMatrix,basisValues,testBasisValues,COMP_BLAS);
  FunctionSpaceTools::integrate<double>(ipVector,functionValues,testBasisValues,COMP_BLAS); 
  
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

void Projector::projectFunctionOntoBasisInterpolating(FieldContainer<double> &basisCoefficients, FunctionPtr fxn,
                                                      BasisPtr basis, BasisCachePtr domainBasisCache) {
  basisCoefficients.initialize(0);
  shards::CellTopology domainTopo = basis->domainTopology();
  unsigned domainDim = domainTopo.getDimension();
  
  IPPtr ip;
  
  bool traceVar = domainBasisCache->isSideCache();
  
  pair<IPPtr, VarPtr> ipVarPair = IP::standardInnerProductForFunctionSpace(basis->functionSpace(), traceVar);
  ip = ipVarPair.first;
  VarPtr v = ipVarPair.second;
  
  IPPtr ip_l2 = Teuchos::rcp( new IP );
  ip_l2->addTerm(v);
  
  // for now, make all projections use L^2... (having some issues with gradients and cell Jacobians--I think we need the restriction of the cell Jacobian to the subcell, e.g., and it's not clear how to do that...)
  ip = ip_l2;
  
  FieldContainer<double> referenceDomainNodes(domainTopo.getVertexCount(),domainDim);
  CamelliaCellTools::refCellNodesForTopology(referenceDomainNodes, domainTopo);
  
  int basisCardinality = basis->getCardinality();
  
  set<int> allDofs;
  for (int i=0; i<basisCardinality; i++) {
    allDofs.insert(i);
  }
  
  for (int d=0; d<=domainDim; d++) {
    FunctionPtr projectionThusFar = NewBasisSumFunction::basisSumFunction(basis, basisCoefficients);
    FunctionPtr fxnToApproximate = fxn - projectionThusFar;
    int subcellCount = domainTopo.getSubcellCount(d);
    for (int subcord=0; subcord<subcellCount; subcord++) {
      set<int> subcellDofOrdinals = basis->dofOrdinalsForSubcell(d, subcord);
      if (subcellDofOrdinals.size() > 0) {
        FieldContainer<double> refCellPoints;
        FieldContainer<double> cubatureWeightsSubcell; // allows us to integrate over the fine subcell even when domain is higher-dimensioned
        if (d == 0) {
          refCellPoints.resize(1,domainDim);
          for (int d1=0; d1<domainDim; d1++) {
            refCellPoints(0,d1) = referenceDomainNodes(subcord,d1);
          }
          cubatureWeightsSubcell.resize(1);
          cubatureWeightsSubcell(0) = 1.0;
        } else {
          shards::CellTopology subcellTopo = domainTopo.getBaseCellTopologyData(d, subcord);
//          Teuchos::RCP<Cubature<double> > subcellCubature = cubFactory.create(subcellTopo, domainBasisCache->cubatureDegree());
          BasisCachePtr subcellCache = Teuchos::rcp( new BasisCache(subcellTopo, domainBasisCache->cubatureDegree(), false) );
          int numPoints = subcellCache->getRefCellPoints().dimension(0);
          refCellPoints.resize(numPoints,domainDim);
          cubatureWeightsSubcell = subcellCache->getCubatureWeights();
          
          if (d == domainDim) {
            refCellPoints = subcellCache->getRefCellPoints();
          } else {
            CamelliaCellTools::mapToReferenceSubcell(refCellPoints, subcellCache->getRefCellPoints(), d,
                                                     subcord, domainTopo);
          }
        }
        domainBasisCache->setRefCellPoints(refCellPoints, cubatureWeightsSubcell);
        IPPtr ipForProjection = (d==0) ? ip_l2 : ip; // just use values at vertices (ignore derivatives)
        set<int> dofsToSkip = allDofs;
        for (set<int>::iterator dofOrdinalIt=subcellDofOrdinals.begin(); dofOrdinalIt != subcellDofOrdinals.end(); dofOrdinalIt++) {
          dofsToSkip.erase(*dofOrdinalIt);
        }
        FieldContainer<double> newBasisCoefficients;
        projectFunctionOntoBasis(newBasisCoefficients, fxnToApproximate, basis, domainBasisCache, ipForProjection, v, dofsToSkip);
        for (int cellOrdinal=0; cellOrdinal<newBasisCoefficients.dimension(0); cellOrdinal++) {
          for (set<int>::iterator dofOrdinalIt=subcellDofOrdinals.begin(); dofOrdinalIt != subcellDofOrdinals.end(); dofOrdinalIt++) {
            basisCoefficients(cellOrdinal,*dofOrdinalIt) = newBasisCoefficients(cellOrdinal,*dofOrdinalIt);
          }
        }
      }
    }
  }
}

