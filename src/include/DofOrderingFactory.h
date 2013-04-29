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

// DPG includes
#include "CamelliaIntrepidExtendedTypes.h"
#include "DofOrdering.h"

using namespace std;

class BilinearForm;

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
      set<int>::iterator lhsVarIterator;
      set<int>::iterator rhsVarIterator = rhsVarIDs.begin();
      for (lhsVarIterator = lhsVarIDs.begin(); lhsVarIterator != lhsVarIDs.end(); lhsVarIterator++) {
        int lhsVarID = *lhsVarIterator;
        int rhsVarID = *rhsVarIterator;
        if (lhsVarID != rhsVarID) {
          return lhsVarID < rhsVarID;
        }
        int lhsSidesForVar = lhs->getNumSidesForVarID(lhsVarID);
        int rhsSidesForVar = rhs->getNumSidesForVarID(rhsVarID);
        if (lhsSidesForVar != rhsSidesForVar) {
          return lhsSidesForVar < rhsSidesForVar;
        }
        for (int i=0; i<lhsSidesForVar; i++) {
          BasisPtr lhsBasis = lhs->getBasis(lhsVarID,i);
          BasisPtr rhsBasis = rhs->getBasis(rhsVarID,i);
          if ( lhsBasis.get() != rhsBasis.get() ) { // different pointers ==> different bases
            return lhsBasis.get() < rhsBasis.get();
          }
          // the following loop is necessary for distinguishing between DofOrderings
          // that have conforming traces from those that do not...
          for (int basisOrdinal=0; basisOrdinal < lhsBasis->getCardinality(); basisOrdinal++) {
            int lhsDofIndex = lhs->getDofIndex(lhsVarID,basisOrdinal,i);
            int rhsDofIndex = rhs->getDofIndex(lhsVarID,basisOrdinal,i);
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
  Teuchos::RCP<BilinearForm> _bilinearForm;
  map<DofOrdering*,bool> _isConforming;
  map<int, int> _testOrderEnhancements;
  map<int, int> _trialOrderEnhancements;
  void addConformingVertexPairings(int varID, DofOrderingPtr dofOrdering,
                                   const shards::CellTopology &cellTopo);
  int polyOrder(DofOrderingPtr dofOrdering, bool isTestOrdering);
  DofOrderingPtr pRefine(DofOrderingPtr dofOrdering,
                         const shards::CellTopology &cellTopo, int pToAdd, bool isTestOrdering);
public:
  DofOrderingFactory(Teuchos::RCP<BilinearForm> bilinearForm);
  DofOrderingFactory(Teuchos::RCP<BilinearForm> bilinearForm,
                     map<int,int> trialOrderEnhancements,
                     map<int,int> testOrderEnhancements);
  DofOrderingPtr testOrdering(int polyOrder, const shards::CellTopology &cellTopo); // NOTE: for now only handles 2D/quads (lines in 1D for sides, too)
  DofOrderingPtr trialOrdering(int polyOrder, const shards::CellTopology &cellTopo,
                               bool conformingVertices = true);
  int testPolyOrder(DofOrderingPtr testOrdering);
  int trialPolyOrder(DofOrderingPtr trialOrdering);
  DofOrderingPtr pRefineTest(DofOrderingPtr testOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);
  DofOrderingPtr pRefineTrial(DofOrderingPtr trialOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);
  DofOrderingPtr setSidePolyOrder(DofOrderingPtr dofOrdering, int sideIndexToSet, int newPolyOrder, bool replacePatchBasis);
  DofOrderingPtr getTrialOrdering(DofOrdering &ordering);
  DofOrderingPtr getTestOrdering(DofOrdering &ordering);
  int matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex, 
                 const shards::CellTopology &firstCellTopo,
                 DofOrderingPtr &secondOrdering, int secondSideIndex,
                 const shards::CellTopology &secondCellTopo);
  void childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                        const shards::CellTopology &childTopo, int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
                        DofOrderingPtr &parentTrialOrdering, int sideIndex,
                        const shards::CellTopology &parentTopo);
  void assignMultiBasis(DofOrderingPtr &trialOrdering, int sideIndex, 
                        const shards::CellTopology &cellTopo,
                        vector< pair< DofOrderingPtr,int > > &childTrialOrdersForSide );
  void assignPatchBasis(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                        const DofOrderingPtr parentTrialOrdering, int parentSideIndex,
                        int childIndexInParentSide,const shards::CellTopology &childCellTopo);
  DofOrderingPtr upgradeSide(DofOrderingPtr dofOrdering,
                             const shards::CellTopology &cellTopo, 
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

#endif
