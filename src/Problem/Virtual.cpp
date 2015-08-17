//
//  Virtual.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/28/14.
//
//

#include "Virtual.h"
#include "BF.h"
#include "Epetra_Time.h"

#include "DofOrderingFactory.h"
#include "SerialDenseWrapper.h"

#include "BasisCache.h"
#include "RHS.h"

using namespace Camellia;

using namespace Intrepid;

class VBF : public BF {
  Virtual _virtualTerms;
public:
  VBF(Virtual virtualTerms, VarFactoryPtr vf) : BF(vf) {
    _virtualTerms = virtualTerms;
  }
  
  virtual void localStiffnessMatrixAndRHS(FieldContainer<double> &localStiffness, FieldContainer<double> &rhsVector,
                                          IPPtr ip, BasisCachePtr ipBasisCache,
                                          RHSPtr rhs, BasisCachePtr basisCache) {
    double testMatrixAssemblyTime = 0, testMatrixInversionTime = 0, localStiffnessDeterminationFromTestsTime = 0;
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    
    Epetra_Time timer(Comm);
    
    // localStiffness should have dim. (numCells, numTrialFields, numTrialFields)
    MeshPtr mesh = basisCache->mesh();
    if (mesh.get() == NULL) {
      cout << "localStiffnessMatrix requires BasisCache to have mesh set.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires BasisCache to have mesh set.");
    }
    const vector<GlobalIndexType>* cellIDs = &basisCache->cellIDs();
    int numCells = cellIDs->size();
    if (numCells != localStiffness.dimension(0)) {
      cout << "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness");
    }
    
    ElementTypePtr elemType = mesh->getElementType((*cellIDs)[0]); // we assume all cells provided are of the same type
    DofOrderingPtr trialOrder = elemType->trialOrderPtr;
    DofOrderingPtr fieldOrder = mesh->getDofOrderingFactory().getFieldOrdering(trialOrder);
    DofOrderingPtr traceOrder = mesh->getDofOrderingFactory().getTraceOrdering(trialOrder);
    
    map<int, int> stiffnessIndexForTraceIndex;
    map<int, int> stiffnessIndexForFieldIndex;
    set<int> varIDs = trialOrder->getVarIDs();
    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(varID);
      bool isTrace = (sidesForVar->size() > 1);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++) {
        int sideOrdinal = *sideIt;
        vector<int> dofIndices = trialOrder->getDofIndices(varID,sideOrdinal);
        if (isTrace) {
          vector<int> traceDofIndices = traceOrder->getDofIndices(varID,sideOrdinal);
          for (int i=0; i<traceDofIndices.size(); i++) {
            stiffnessIndexForTraceIndex[traceDofIndices[i]] = dofIndices[i];
          }
        } else {
          vector<int> fieldDofIndices = fieldOrder->getDofIndices(varID);
          for (int i=0; i<fieldDofIndices.size(); i++) {
            stiffnessIndexForFieldIndex[fieldDofIndices[i]] = dofIndices[i];
          }
        }
      }
    }
    
    int numTrialDofs = trialOrder->totalDofs();
    if ((numTrialDofs != localStiffness.dimension(1)) || (numTrialDofs != localStiffness.dimension(2))) {
      cout << "localStiffness should have dimensions (C,numTrialFields,numTrialFields).\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffness should have dimensions (C,numTrialFields,numTrialFields).");
    }
    
    map<int,int> traceTestMap, fieldTestMap;
    int numEquations = _virtualTerms.getFieldTestVars().size();
    for (int eqn=0; eqn<numEquations; eqn++) {
      VarPtr testVar = _virtualTerms.getFieldTestVars()[eqn];
      VarPtr traceVar = _virtualTerms.getAssociatedTrace(testVar);
      VarPtr fieldVar = _virtualTerms.getAssociatedField(testVar);
      traceTestMap[traceVar->ID()] = testVar->ID();
      fieldTestMap[fieldVar->ID()] = testVar->ID();
    }
    
    int maxDegreeField = fieldOrder->maxBasisDegree();
    int testDegreeInterior = maxDegreeField + _virtualTerms.getTestEnrichment();
    int testDegreeTrace = testDegreeInterior + 2;
    
    DofOrderingPtr testOrderInterior = mesh->getDofOrderingFactory().getRelabeledDofOrdering(fieldOrder, fieldTestMap);
    testOrderInterior = mesh->getDofOrderingFactory().setBasisDegree(testOrderInterior, testDegreeInterior, false);
    DofOrderingPtr testOrderTrace = mesh->getDofOrderingFactory().setBasisDegree(testOrderInterior, testDegreeTrace, true); // this has a bunch of extra dofs (interior guys)
    
    map<int, int> remappedTraceIndices; // go from the index that includes the interior dofs to one that doesn't
    set<int> testIDs = testOrderTrace->getVarIDs();
    int testTraceIndex = 0;
    for (set<int>::iterator testIDIt = testIDs.begin(); testIDIt != testIDs.end(); testIDIt++) {
      int testID = *testIDIt;
      BasisPtr basis = testOrderTrace->getBasis(testID);
      set<int> interiorDofs = basis->dofOrdinalsForInterior();
      for (int basisOrdinal=0; basisOrdinal<basis->getCardinality(); basisOrdinal++) {
        if (interiorDofs.find(basisOrdinal) == interiorDofs.end()) {
          int dofIndex = testOrderTrace->getDofIndex(testID, basisOrdinal);
          remappedTraceIndices[dofIndex] = testTraceIndex;
          testTraceIndex++;
        }
      }
    }
    
//    DofOrderingPtr testOrderTrace = mesh->getDofOrderingFactory().getRelabeledDofOrdering(traceOrder, traceTestMap);
//    testOrderTrace = mesh->getDofOrderingFactory().setBasisDegree(testOrderTrace, testDegreeTrace);
    
    int numTestInteriorDofs = testOrderInterior->totalDofs();
    int numTestTraceDofsIncludingInterior = testOrderTrace->totalDofs();
    int numTestTraceDofs = testTraceIndex;
    int numTestDofs = numTestTraceDofs + numTestInteriorDofs;
    
    timer.ResetStartTime();
    
    bool printTimings = true;
    
    if (printTimings) {
      cout << "numCells: " << numCells << endl;
      cout << "numTestDofs: " << numTestDofs << endl;
    }
    
    FieldContainer<double> rhsVectorTest(numCells,testOrderInterior->totalDofs()); // rhsVector is zero for the "trace" test dofs
    {
      // project the load f onto the space of interior test dofs.
      LinearTermPtr f = rhs->linearTerm();
      set<int> testIDs = f->varIDs();
      for (int eqn=0; eqn<numEquations; eqn++) {
        VarPtr v = _virtualTerms.getFieldTestVars()[eqn];
        
        if (testIDs.find(v->ID()) != testIDs.end()) {
          BasisPtr testInteriorBasis = testOrderInterior->getBasis(v->ID());
          FieldContainer<double> fValues(numCells,testInteriorBasis->getCardinality());
//          DofOrderingPtr oneVarOrderingTest = Teuchos::rcp(new DofOrdering(testInteriorBasis->domainTopology()));
          DofOrderingPtr oneVarOrderingTest = Teuchos::rcp(new DofOrdering);
          oneVarOrderingTest->addEntry(v->ID(), testInteriorBasis, testInteriorBasis->rangeRank());

          LinearTermPtr f_v = Teuchos::rcp( new LinearTerm );
          typedef pair< FunctionPtr, VarPtr > LinearSummand;
          vector<LinearSummand> summands = f->summands();
          for (int i=0; i<summands.size(); i++) {
            FunctionPtr f = summands[i].first;
            if (v->ID() == summands[i].second->ID()) {
              f_v->addTerm(f * v);
              f_v->integrate(fValues, oneVarOrderingTest, basisCache);
            }
          }
          
          LinearTermPtr v_lt = 1.0 * v;
          FieldContainer<double> l2(numCells,testInteriorBasis->getCardinality(),testInteriorBasis->getCardinality());
          v_lt->integrate(l2,oneVarOrderingTest,v_lt,oneVarOrderingTest,basisCache,basisCache->isSideCache());

          Teuchos::Array<int> testTestDim(2), testOneDim(2);
          testTestDim[0] = testInteriorBasis->getCardinality();
          testTestDim[1] = testInteriorBasis->getCardinality();
          testOneDim[0] = testInteriorBasis->getCardinality();
          testOneDim[1] = 1;
          FieldContainer<double> projection(testOneDim);
          for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
            FieldContainer<double> l2cell(testTestDim,&l2(cellOrdinal,0,0));
            FieldContainer<double> f_cell(testOneDim,&fValues(cellOrdinal,0));
            
            SerialDenseWrapper::solveSystemUsingQR(projection, l2cell, f_cell);
            
            // rows in projection correspond to Ae_i, columns to the e_j.  I.e. projection coefficients for e_i are found in the ith row
            for (int basisOrdinal_j=0; basisOrdinal_j<projection.dimension(0); basisOrdinal_j++) {
              int testIndex = testOrderInterior->getDofIndex(v->ID(), basisOrdinal_j);
              rhsVectorTest(cellOrdinal,testIndex) = projection(basisOrdinal_j,0);
            }
          }
        }
      }
    }
    
    // project strong operator applied to field terms, and use this to populate the top left portion of stiffness matrix:
    {
      FieldContainer<double> trialFieldTestInterior(numCells, fieldOrder->totalDofs(), testOrderInterior->totalDofs());
      for (int eqn=0; eqn<numEquations; eqn++) {
        LinearTermPtr Au = _virtualTerms.getFieldOperators()[eqn];
        VarPtr v = _virtualTerms.getFieldTestVars()[eqn];
        set<int> fieldIDs = Au->varIDs();
        for (set<int>::iterator fieldIt = fieldIDs.begin(); fieldIt != fieldIDs.end(); fieldIt++) {
          int fieldID = *fieldIt;
//          int testID = fieldTestMap[fieldID];
//          
//          LinearTermPtr testInteriorVar = 1.0 * this->varFactory().test(testID);
          
          BasisPtr vBasis = testOrderInterior->getBasis(v->ID());
          BasisPtr fieldTrialBasis = fieldOrder->getBasis(fieldID);
//          DofOrderingPtr oneVarOrderingTest = Teuchos::rcp(new DofOrdering(vBasis->domainTopology()));
          DofOrderingPtr oneVarOrderingTest = Teuchos::rcp(new DofOrdering());
          oneVarOrderingTest->addEntry(v->ID(), vBasis, vBasis->rangeRank());
          FieldContainer<double> Au_values(numCells,vBasis->getCardinality(),fieldTrialBasis->getCardinality());
          FieldContainer<double> l2(numCells,vBasis->getCardinality(),vBasis->getCardinality());

          DofOrderingPtr oneVarOrderingTrial = Teuchos::rcp(new DofOrdering());
//          DofOrderingPtr oneVarOrderingTrial = Teuchos::rcp(new DofOrdering(fieldTrialBasis->domainTopology()));
          oneVarOrderingTrial->addEntry(fieldID, fieldTrialBasis, fieldTrialBasis->rangeRank());
          
          LinearTermPtr Au_restricted_to_field = Teuchos::rcp( new LinearTerm );
          typedef pair< FunctionPtr, VarPtr > LinearSummand;
          vector<LinearSummand> summands = Au->summands();
          for (int i=0; i<summands.size(); i++) {
            FunctionPtr f = summands[i].first;
            VarPtr v = summands[i].second;
            if (v->ID() == fieldID) {
              Au_restricted_to_field->addTerm(f * v);
            }
          }
          
          LinearTermPtr v_lt = 1.0 * v;
          
          Au_restricted_to_field->integrate(Au_values,oneVarOrderingTrial,v_lt,oneVarOrderingTest,basisCache,basisCache->isSideCache());
          v_lt->integrate(l2,oneVarOrderingTest,v_lt,oneVarOrderingTest,basisCache,basisCache->isSideCache());
          double maxValue = 0;
          for (int i=0; i<l2.size(); i++) {
            maxValue = max(abs(l2[i]),maxValue);
          }
          cout << "maxValue in l2 is " << maxValue << endl;
          Teuchos::Array<int> testTestDim(2), trialTestDim(2);
          testTestDim[0] = vBasis->getCardinality();
          testTestDim[1] = vBasis->getCardinality();
          trialTestDim[0] = vBasis->getCardinality();
          trialTestDim[1] = fieldTrialBasis->getCardinality();
          FieldContainer<double> projection(trialTestDim);
          for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
            FieldContainer<double> l2cell(testTestDim,&l2(cellOrdinal,0,0));
            FieldContainer<double> AuCell(trialTestDim,&Au_values(cellOrdinal,0,0));
            // TODO: confirm that I'm doing the right projection here.  I could be missing a key point, but it seems to me that we must
            //       project onto an *orthonormal* basis here, to achieve the required identity structure of the (field,field) part of the
            //       Gram matrix.  OTOH, it looks to me like the computation here achieves exactly that, even though I didn't initially
            //       have that in mind...
//            SerialDenseWrapper::solveSystemUsingQR(projection, l2cell, AuCell);
            SerialDenseWrapper::solveSystemMultipleRHS(projection, l2cell, AuCell);
            
            // rows in projection correspond to Ae_i, columns to the e_j.  I.e. projection coefficients for e_i are found in the ith row
            for (int basisOrdinal_i=0; basisOrdinal_i<projection.dimension(0); basisOrdinal_i++) {
              int testIndex = testOrderInterior->getDofIndex(v->ID(), basisOrdinal_i);
              for (int basisOrdinal_j=0; basisOrdinal_j<projection.dimension(1); basisOrdinal_j++) {
                int trialIndex = fieldOrder->getDofIndex(fieldID, basisOrdinal_j); // in the *trial* space
                trialFieldTestInterior(cellOrdinal,trialIndex,testIndex) = projection(basisOrdinal_i,basisOrdinal_j);
              }
            }
          }
        }
      }
      Teuchos::Array<int> trialTestDim(2);
      trialTestDim[0] = fieldOrder->totalDofs();
      trialTestDim[1] = testOrderInterior->totalDofs();
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
        FieldContainer<double> trialFieldTrialField(fieldOrder->totalDofs(), fieldOrder->totalDofs());
        FieldContainer<double> trialTestCell(trialTestDim, &trialFieldTestInterior(cellOrdinal,0,0));
        SerialDenseWrapper::multiply(trialFieldTrialField, trialTestCell, trialTestCell, 'N', 'T'); // transpose the second one
        // now, accumulate into localStiffness
        for (int i=0; i<trialFieldTrialField.dimension(0); i++) {
          int stiff_i = stiffnessIndexForFieldIndex[i];
          for (int j=0; j<trialFieldTrialField.dimension(1); j++) {
            int stiff_j = stiffnessIndexForFieldIndex[j];
            localStiffness(cellOrdinal,stiff_i,stiff_j) = trialFieldTrialField(i,j);
          }
        }
      }
      // multiply RHS integrated against the interior test space by the trialFieldTestInterior
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
        Teuchos::Array<int> trialTestDim(2), oneTestDim(2);
        trialTestDim[0] = fieldOrder->totalDofs();
        trialTestDim[1] = testOrderInterior->totalDofs();
        oneTestDim[0] = 1;
        oneTestDim[1] = testOrderInterior->totalDofs();
        FieldContainer<double> trialTestCell(trialTestDim, &trialFieldTestInterior(cellOrdinal,0,0));
        FieldContainer<double> rhsTestCell(oneTestDim, &rhsVectorTest(cellOrdinal,0));
        FieldContainer<double> rhsTrialCell(1, fieldOrder->totalDofs());
        
        SerialDenseWrapper::multiply(rhsTrialCell, rhsTestCell, trialTestCell, 'N', 'T');
        
        for (int fieldIndex=0; fieldIndex<fieldOrder->totalDofs(); fieldIndex++) {
          int stiffIndex = stiffnessIndexForFieldIndex[fieldIndex];
          rhsVector(cellOrdinal,stiffIndex) = rhsTrialCell(0, fieldIndex);
        }
      }
    }
    
    FieldContainer<double> ipMatrixTraceIncludingInterior(numCells,numTestTraceDofsIncludingInterior,numTestTraceDofsIncludingInterior);
    int numTestTerms = _virtualTerms.getTestNormOperators().size();
    for (int i=0; i<numTestTerms; i++) {
      LinearTermPtr testTerm = _virtualTerms.getTestNormOperators()[i];
      LinearTermPtr boundaryTerm = _virtualTerms.getTestNormBoundaryOperators()[i];
      testTerm->integrate(ipMatrixTraceIncludingInterior,testOrderTrace,boundaryTerm,testOrderTrace,ipBasisCache,ipBasisCache->isSideCache());
    }
    FieldContainer<double> ipMatrixTrace(numCells,numTestTraceDofs,numTestTraceDofs);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      for (int i_dofIndex=0; i_dofIndex<numTestTraceDofsIncludingInterior; i_dofIndex++) {
        if (remappedTraceIndices.find(i_dofIndex) == remappedTraceIndices.end()) {
          continue;
        }
        int i_remapped = remappedTraceIndices[i_dofIndex];
        for (int j_dofIndex=0; j_dofIndex<numTestTraceDofsIncludingInterior; j_dofIndex++) {
          if (remappedTraceIndices.find(j_dofIndex) == remappedTraceIndices.end()) {
            continue;
          }
          int j_remapped = remappedTraceIndices[j_dofIndex];
          ipMatrixTrace(cellOrdinal,i_remapped,j_remapped) = ipMatrixTraceIncludingInterior(cellOrdinal,i_dofIndex,j_dofIndex);
        }
      }
    }
    
    testMatrixAssemblyTime += timer.ElapsedTime();
    //      cout << "ipMatrix:\n" << ipMatrix;
    
    timer.ResetStartTime();
    
    cout << "NOTE: we do not yet enforce continuity on the trace test space.\n"; // I *think* this is fine, but I'm not dead certain -- we do of course in the end enforce continuity in GDAMinimumRule
    
    // now, determine the trace part of the bilinear form matrix
    FieldContainer<double> bfMatrixTraceTraceIncludingTestInterior(numCells,testOrderTrace->totalDofs(),traceOrder->totalDofs());
    FieldContainer<double> bfMatrixFieldTraceIncludingTestInterior(numCells,testOrderTrace->totalDofs(),fieldOrder->totalDofs());
    for (int eqn=0; eqn<numEquations; eqn++) {
      VarPtr traceVar = _virtualTerms.getTraceVars()[eqn];
      LinearTermPtr termTraced = traceVar->termTraced();
      LinearTermPtr strongOperator = _virtualTerms.getFieldOperators()[eqn];
      VarPtr testVar = _virtualTerms.getFieldTestVars()[eqn];
      
      // want to determine \hat{C}(\hat{e}_i, \phi_j) for \phi_j with support on the boundary
      // the \phi_j's with support on the boundary are the ones associated with the trace
      
      LinearTermPtr trialTerm = 1.0 * traceVar;
      LinearTermPtr testTerm;
      
      if (traceVar->varType() == TRACE) {
        testTerm = Function::normal() * testVar;
      } else {
        testTerm = 1.0 * testVar;
      }
      
//      trialTerm->integrate(bfMatrixTrace,traceOrder,testTerm,testOrderTrace,basisCache,basisCache->isSideCache());
      trialTerm->integrate(bfMatrixTraceTraceIncludingTestInterior,traceOrder,testTerm,testOrderTrace,basisCache,basisCache->isSideCache());
      termTraced->integrate(bfMatrixFieldTraceIncludingTestInterior,fieldOrder,-testTerm,testOrderTrace,basisCache,basisCache->isSideCache());
    }
    
    FieldContainer<double> bfMatrixFieldTrace(numCells,numTestTraceDofs,bfMatrixFieldTraceIncludingTestInterior.dimension(2));
    FieldContainer<double> bfMatrixTraceTrace(numCells,numTestTraceDofs,bfMatrixTraceTraceIncludingTestInterior.dimension(2));
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      for (int i_dofIndex=0; i_dofIndex<numTestTraceDofsIncludingInterior; i_dofIndex++) {
        if (remappedTraceIndices.find(i_dofIndex) == remappedTraceIndices.end()) {
          continue;
        }
        int i_remapped = remappedTraceIndices[i_dofIndex];
        for (int j_dofIndex=0; j_dofIndex<bfMatrixFieldTrace.dimension(2); j_dofIndex++) {
          bfMatrixFieldTrace(cellOrdinal,i_remapped,j_dofIndex) = bfMatrixFieldTraceIncludingTestInterior(cellOrdinal,i_dofIndex,j_dofIndex);
        }
        for (int j_dofIndex=0; j_dofIndex<bfMatrixTraceTrace.dimension(2); j_dofIndex++) {
          bfMatrixTraceTrace(cellOrdinal,i_remapped,j_dofIndex) = bfMatrixTraceTraceIncludingTestInterior(cellOrdinal,i_dofIndex,j_dofIndex);
        }
      }
    }
    
    Teuchos::Array<int> ipMatrixDim(2), bfMatrixTraceTraceDim(2), bfMatrixFieldTraceDim(2);
    Teuchos::Array<int> traceTraceStiffDim(2), fieldTraceStiffDim(2), fieldFieldStiffDim(2);
    ipMatrixDim[0] = ipMatrixTrace.dimension(1);
    ipMatrixDim[1] = ipMatrixTrace.dimension(2);
    
    bfMatrixTraceTraceDim[0] = bfMatrixTraceTrace.dimension(1);
    bfMatrixTraceTraceDim[1] = bfMatrixTraceTrace.dimension(2);
    
    bfMatrixFieldTraceDim[0] = bfMatrixFieldTrace.dimension(1);
    bfMatrixFieldTraceDim[1] = bfMatrixFieldTrace.dimension(2);
    
    traceTraceStiffDim[0] = traceOrder->totalDofs();
    traceTraceStiffDim[1] = traceTraceStiffDim[0];
    
    fieldTraceStiffDim[0] = fieldOrder->totalDofs();
    fieldTraceStiffDim[1] = traceOrder->totalDofs(); // rectangular
    
    fieldFieldStiffDim[0] = fieldOrder->totalDofs();
    fieldFieldStiffDim[1] = fieldOrder->totalDofs();
    
    FieldContainer<double> traceTraceStiffCell(traceTraceStiffDim);
    FieldContainer<double> fieldTraceStiffCell(fieldTraceStiffDim);
    FieldContainer<double> fieldFieldStiffCell(fieldFieldStiffDim);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      FieldContainer<double> ipMatrixCell(ipMatrixDim,&ipMatrixTrace(cellOrdinal,0,0));
      
      FieldContainer<double> optTestCoeffsTraceTrace(numTestTraceDofs,traceOrder->totalDofs());
      FieldContainer<double> bfMatrixTraceTraceCell(bfMatrixTraceTraceDim,&bfMatrixTraceTrace(cellOrdinal,0,0));
      int result = SerialDenseWrapper::solveSystemUsingQR(optTestCoeffsTraceTrace, ipMatrixCell, bfMatrixTraceTraceCell);
      SerialDenseWrapper::multiply(traceTraceStiffCell, bfMatrixTraceTraceCell, optTestCoeffsTraceTrace, 'T', 'N');
      
      // copy into the appropriate spot in localStiffness:
      for (int i=0; i<traceTraceStiffDim[0]; i++) {
        int i_stiff = stiffnessIndexForTraceIndex[i];
        for (int j=0; j<traceTraceStiffDim[1]; j++) {
          int j_stiff = stiffnessIndexForTraceIndex[j];
          localStiffness(cellOrdinal,i_stiff,j_stiff) = traceTraceStiffCell(i,j);
        }
      }

      // because of the way the matrix blocks line up, we actually don't have to do a second inversion of ipMatrixCell for this part
      FieldContainer<double> bfMatrixFieldTraceCell(bfMatrixFieldTraceDim,&bfMatrixFieldTrace(cellOrdinal,0,0));
      SerialDenseWrapper::multiply(fieldTraceStiffCell, bfMatrixFieldTraceCell, optTestCoeffsTraceTrace, 'T', 'N');

      // copy into the appropriate spots in localStiffness (taking advantage of symmetry):
      for (int i=0; i<fieldTraceStiffDim[0]; i++) {
        int i_stiff = stiffnessIndexForFieldIndex[i];
        for (int j=0; j<fieldTraceStiffDim[1]; j++) {
          int j_stiff = stiffnessIndexForTraceIndex[j];
          localStiffness(cellOrdinal,i_stiff,j_stiff) = fieldTraceStiffCell(i,j);
          localStiffness(cellOrdinal,j_stiff,i_stiff) = fieldTraceStiffCell(i,j);
        }
      }
      
      // because of the way the matrix blocks line up, we do have some trace contributions in the (field, field) portion of the matrix
      // these get added to the field contributions (hence the +=)
      FieldContainer<double> optTestCoeffsFieldTrace(numTestTraceDofs,fieldOrder->totalDofs());
      result = SerialDenseWrapper::solveSystemUsingQR(optTestCoeffsFieldTrace, ipMatrixCell, bfMatrixFieldTraceCell);
      SerialDenseWrapper::multiply(fieldFieldStiffCell, bfMatrixFieldTraceCell, optTestCoeffsFieldTrace, 'T', 'N');
      for (int i=0; i<fieldFieldStiffDim[0]; i++) {
        int i_stiff = stiffnessIndexForFieldIndex[i];
        for (int j=0; j<fieldFieldStiffDim[1]; j++) {
          int j_stiff = stiffnessIndexForFieldIndex[j];
          localStiffness(cellOrdinal,i_stiff,j_stiff) += fieldFieldStiffCell(i,j);
        }
      }
    }
    
    testMatrixInversionTime += timer.ElapsedTime();
    //      cout << "optTestCoeffs:\n" << optTestCoeffs;
    
    if (printTimings) {
      cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
      cout << "testMatrixInversionTime: " << testMatrixInversionTime << " seconds.\n";
      cout << "localStiffnessDeterminationFromTestsTime: " << localStiffnessDeterminationFromTestsTime << " seconds.\n";
    }
  }
};

Virtual::Virtual(int testEnrichment) {
  _testEnrichment = testEnrichment;
}

void Virtual::addAssociation(VarPtr testVar, VarPtr fieldVar, VarPtr traceVar) {
  _testAssociations[testVar->ID()] = make_pair(fieldVar, traceVar);
}

void Virtual::addEquation(LinearTermPtr Au, VarPtr u_hat, VarPtr v) { // call this once for every equation
  /* assumptions:
   - the u_hat has an appropriate termTraced defined
   - the equation in the BF is (u, A^* v) + < u_hat, v > = ...
   - A is the formal adjoint of A^*; i.e. when we integrate by parts we should get:
               (Au, v) - < tr (u), v > + < u_hat, v >
     where tr (u) is the termTraced provided in u_hat
   */
  _fieldOperators.push_back(Au);
  _traceVars.push_back(u_hat);
  _fieldTests.push_back(v);
}

void Virtual::addTestNormTerm(LinearTermPtr testNormOperator, LinearTermPtr testBoundaryOperator) { // call this once for each entry in the test norm (i.e. the squared terms in the sum of squares)
  _testNormOperators.push_back(testNormOperator);
  _testBoundaryOperators.push_back(testBoundaryOperator);
}

VarPtr Virtual::getAssociatedField(VarPtr testVar) {
  return _testAssociations[testVar->ID()].first;
}

VarPtr Virtual::getAssociatedTrace(VarPtr testVar) {
  return _testAssociations[testVar->ID()].second;
}

const vector<VarPtr> & Virtual::getTraceVars() {
  return _traceVars;
}

const vector<LinearTermPtr> & Virtual::getFieldOperators() {
  return _fieldOperators;
}

const vector<VarPtr> & Virtual::getFieldTestVars() {
  return _fieldTests;
}

const vector<LinearTermPtr> & Virtual::getTestNormBoundaryOperators() {
  return _testBoundaryOperators;
}

const vector<LinearTermPtr> & Virtual::getTestNormOperators() {
  return _testNormOperators;
}

int Virtual::getTestEnrichment() {
  return _testEnrichment;
}

BFPtr Virtual::virtualBF(Virtual &virtualTerms, VarFactoryPtr vf) {
  return Teuchos::rcp( new VBF(virtualTerms, vf) );
}