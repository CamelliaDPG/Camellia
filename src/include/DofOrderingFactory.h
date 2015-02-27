#ifndef DOF_ORDERING_FACTORY
#define DOF_ORDERING_FACTORY

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

/*
 *  DofOrderingFactory.h
 *
 */


// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Camellia includes
#include "CamelliaIntrepidExtendedTypes.h"
#include "DofOrdering.h"
#include "Var.h"

using namespace std;

class DofOrderingFactory {
private:
  struct Comparator {
    bool operator() (const DofOrderingPtr &lhs, const DofOrderingPtr &rhs) {
      // return true if lhs < rhs
      set<int> lhsVarIDs = lhs->getVarIDs();
      set<int> rhsVarIDs = rhs->getVarIDs();
      if ( lhsVarIDs.size() != rhsVarIDs.size() ) {
        return lhsVarIDs.size() < rhsVarIDs.size();
      }
      CellTopoPtr lhsCellTopo = lhs->cellTopology();
      CellTopoPtr rhsCellTopo = rhs->cellTopology();
      
      if (lhsCellTopo->getKey() != rhsCellTopo->getKey()) {
        return lhsCellTopo->getKey() < rhsCellTopo->getKey();
      }
      
      set<int>::iterator lhsVarIterator;
      set<int>::iterator rhsVarIterator = rhsVarIDs.begin();
      for (lhsVarIterator = lhsVarIDs.begin(); lhsVarIterator != lhsVarIDs.end(); lhsVarIterator++) {
        int lhsVarID = *lhsVarIterator;
        int rhsVarID = *rhsVarIterator;
        if (lhsVarID != rhsVarID) {
          return lhsVarID < rhsVarID;
        }
        const vector<int>* lhsSidesForVar = &lhs->getSidesForVarID(lhsVarID);
        const vector<int>* rhsSidesForVar = &rhs->getSidesForVarID(rhsVarID);
        if (lhsSidesForVar->size() != rhsSidesForVar->size()) {
          return lhsSidesForVar->size() < rhsSidesForVar->size();
        }
        for (int i=0; i<lhsSidesForVar->size(); i++) {
          int lhsSideIndex = (*lhsSidesForVar)[i];
          int rhsSideIndex = (*rhsSidesForVar)[i];
          if (lhsSideIndex != rhsSideIndex) {
            return lhsSideIndex < rhsSideIndex;
          }
          BasisPtr lhsBasis = lhs->getBasis(lhsVarID,lhsSideIndex);
          BasisPtr rhsBasis = rhs->getBasis(rhsVarID,rhsSideIndex);
          if ( lhsBasis.get() != rhsBasis.get() ) { // different pointers ==> different bases
            return lhsBasis.get() < rhsBasis.get();
          }
          // the following loop is necessary for distinguishing between DofOrderings
          // that have conforming traces from those that do not...
          for (int basisOrdinal=0; basisOrdinal < lhsBasis->getCardinality(); basisOrdinal++) {
            int lhsDofIndex = lhs->getDofIndex(lhsVarID,basisOrdinal,lhsSideIndex);
            int rhsDofIndex = rhs->getDofIndex(lhsVarID,basisOrdinal,rhsSideIndex);
            if (lhsDofIndex != rhsDofIndex) {
              return lhsDofIndex < rhsDofIndex;
            }
          }
        }
        rhsVarIterator++;
      }
      return false;
    }
  };
  set<DofOrderingPtr, Comparator > _testOrderings;
  set<DofOrderingPtr, Comparator > _trialOrderings;

  map<DofOrdering*, DofOrderingPtr > _fieldOrderingForTrial;
  map<DofOrdering*, DofOrderingPtr > _traceOrderingForTrial;
  
  BFPtr _bilinearForm;
  map<DofOrdering*,bool> _isConforming;
  map<int, int> _testOrderEnhancements;
  map<int, int> _trialOrderEnhancements;
  void addConformingVertexPairings(int varID, DofOrderingPtr dofOrdering, CellTopoPtr cellTopo);
  int polyOrder(DofOrderingPtr dofOrdering, bool isTestOrdering);
  DofOrderingPtr pRefine(DofOrderingPtr dofOrdering,
                         CellTopoPtr, int pToAdd, bool isTestOrdering);
public:
  DofOrderingFactory(BFPtr bilinearForm);
  DofOrderingFactory(BFPtr bilinearForm,
                     map<int,int> trialOrderEnhancements,
                     map<int,int> testOrderEnhancements);
  DofOrderingPtr testOrdering(int polyOrder, const shards::CellTopology &cellTopo);
  DofOrderingPtr trialOrdering(int polyOrder, const shards::CellTopology &cellTopo,
                               bool conformingVertices = true);
  
  DofOrderingPtr testOrdering(int polyOrder, CellTopoPtr cellTopo);
  DofOrderingPtr trialOrdering(int polyOrder, CellTopoPtr cellTopo, bool conformingVertices = true);
  
  int testPolyOrder(DofOrderingPtr testOrdering);
  int trialPolyOrder(DofOrderingPtr trialOrdering);
  DofOrderingPtr pRefineTest(DofOrderingPtr testOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);
  DofOrderingPtr pRefineTrial(DofOrderingPtr trialOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);
  
  DofOrderingPtr pRefineTest(DofOrderingPtr testOrdering, CellTopoPtr cellTopo, int pToAdd = 1);
  DofOrderingPtr pRefineTrial(DofOrderingPtr trialOrdering, CellTopoPtr cellTopo, int pToAdd = 1);
  
  DofOrderingPtr setSidePolyOrder(DofOrderingPtr dofOrdering, int sideIndexToSet, int newPolyOrder, bool replacePatchBasis);
  
  DofOrderingPtr getRelabeledDofOrdering(DofOrderingPtr dofOrdering, map<int,int> &oldKeysNewValues);
  
  DofOrderingPtr setBasisDegree(DofOrderingPtr dofOrdering, int basisDegree, bool replaceDiscontinuousFSWithContinuous); // sets all basis functions to have the same poly. degree, without regard for the function space they belong to.  ("polyOrder" in DofOrderingFactory usually is relative to the H^1 order, so that L^2 bases have degree 1 less.)
  
  DofOrderingPtr getTrialOrdering(DofOrdering &ordering);
  DofOrderingPtr getTestOrdering(DofOrdering &ordering);

  DofOrderingPtr getFieldOrdering(DofOrderingPtr trialOrdering); // the sub-ordering that contains only the fields
  DofOrderingPtr getTraceOrdering(DofOrderingPtr trialOrdering); // the sub-ordering that contains only the traces
    
  map<int, int> getTestOrderEnhancements();
  map<int, int> getTrialOrderEnhancements();
  
  int matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex, 
                 CellTopoPtr firstCellTopo,
                 DofOrderingPtr &secondOrdering, int secondSideIndex,
                 CellTopoPtr secondCellTopo);
  void childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                        CellTopoPtr childTopo, int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
                        DofOrderingPtr &parentTrialOrdering, int sideIndex,
                        CellTopoPtr parentTopo);
  void assignMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex, 
                        CellTopoPtr cellTopo,
                        vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide );
  void assignPatchBasis(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                        const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
                        int childIndexInParentSide, CellTopoPtr childCellTopo);
  DofOrderingPtr upgradeSide(DofOrderingPtr dofOrdering,
                             CellTopoPtr cellTopo,
                             map<int,BasisPtr> varIDsToUpgrade,
                             int sideToUpgrade);
  map<int, BasisPtr> getMultiBasisUpgradeMap(vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide);
  map<int, BasisPtr> getPatchBasisUpgradeMap(const DofOrderingPtr childTrialOrdering, int childSideIndex,
                                             const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
                                             int childIndexInParentSide);
  bool sideHasMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex);

  
//  DofOrderingPtr trialOrdering(int polyOrder, int* sidePolyOrder, const shards::CellTopology &cellTopo,
//                                          bool conformingVertices = true);
};

typedef Teuchos::RCP<DofOrderingFactory> DofOrderingFactoryPtr;

#endif
