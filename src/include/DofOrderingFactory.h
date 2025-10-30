// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef DOF_ORDERING_FACTORY
#define DOF_ORDERING_FACTORY

/*
 *  DofOrderingFactory.h
 *
 */

#include "TypeDefs.h"


// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Camellia_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Camellia includes
#include "CamelliaIntrepidExtendedTypes.h"
#include "DofOrdering.h"
#include "Var.h"
#include "VarFactory.h"

using namespace std;

namespace Camellia
{
class DofOrderingFactory
{
public:
  using Tag = std::pair<std::string,int>;
private:
  struct Comparator
  {
    bool operator() (const DofOrderingPtr &lhs, const DofOrderingPtr &rhs) const
    {
      // return true if lhs < rhs
      set<int> lhsVarIDs = lhs->getVarIDs();
      set<int> rhsVarIDs = rhs->getVarIDs();
      if ( lhsVarIDs.size() != rhsVarIDs.size() )
      {
        return lhsVarIDs.size() < rhsVarIDs.size();
      }
      CellTopoPtr lhsCellTopo = lhs->cellTopology();
      CellTopoPtr rhsCellTopo = rhs->cellTopology();

      if (lhsCellTopo->getKey() != rhsCellTopo->getKey())
      {
        return lhsCellTopo->getKey() < rhsCellTopo->getKey();
      }

      set<int>::iterator lhsVarIterator;
      set<int>::iterator rhsVarIterator = rhsVarIDs.begin();
      for (lhsVarIterator = lhsVarIDs.begin(); lhsVarIterator != lhsVarIDs.end(); lhsVarIterator++)
      {
        int lhsVarID = *lhsVarIterator;
        int rhsVarID = *rhsVarIterator;
        if (lhsVarID != rhsVarID)
        {
          return lhsVarID < rhsVarID;
        }
        const vector<int>* lhsSidesForVar = &lhs->getSidesForVarID(lhsVarID);
        const vector<int>* rhsSidesForVar = &rhs->getSidesForVarID(rhsVarID);
        if (lhsSidesForVar->size() != rhsSidesForVar->size())
        {
          return lhsSidesForVar->size() < rhsSidesForVar->size();
        }
        for (unsigned i=0; i<lhsSidesForVar->size(); i++)
        {
          int lhsSideIndex = (*lhsSidesForVar)[i];
          int rhsSideIndex = (*rhsSidesForVar)[i];
          if (lhsSideIndex != rhsSideIndex)
          {
            return lhsSideIndex < rhsSideIndex;
          }
          BasisPtr lhsBasis = lhs->getBasis(lhsVarID,lhsSideIndex);
          BasisPtr rhsBasis = rhs->getBasis(rhsVarID,rhsSideIndex);
          if ( lhsBasis.get() != rhsBasis.get() )   // different pointers ==> different bases
          {
            return lhsBasis.get() < rhsBasis.get();
          }
          // the following loop is necessary for distinguishing between DofOrderings
          // that have conforming traces from those that do not...
          for (int basisOrdinal=0; basisOrdinal < lhsBasis->getCardinality(); basisOrdinal++)
          {
            int lhsDofIndex = lhs->getDofIndex(lhsVarID,basisOrdinal,lhsSideIndex);
            int rhsDofIndex = rhs->getDofIndex(lhsVarID,basisOrdinal,rhsSideIndex);
            if (lhsDofIndex != rhsDofIndex)
            {
              return lhsDofIndex < rhsDofIndex;
            }
          }
        }
        rhsVarIterator++;
      }
      return false;
    }
  };
  set<DofOrderingPtr, Comparator > _testOrderingsSet;
  set<DofOrderingPtr, Comparator > _trialOrderingsSet;

  map<DofOrdering*, DofOrderingPtr > _fieldOrderingForTrial;
  map<DofOrdering*, DofOrderingPtr > _traceOrderingForTrial;
  
  // trial_key arguments: (polyOrder, CellTopo, conforming (legacy, for max. rule), volume tags, side tags.  Volume tags, side tags only included if there are tag-filtered variables matching them.
  // test_key is the same, but skip the "conforming" bool argument.
  using trial_key = std::tuple<vector<int>,CellTopologyKey,bool,set<Tag>,map<int,set<Tag> > >;
  using test_key  = std::tuple<vector<int>,CellTopologyKey,     set<Tag>,map<int,set<Tag> > >;
  
  map<trial_key, DofOrderingPtr> _trialOrderings; // bool: "conforming" (used by GDAMaximumRule2D)
  map< test_key, DofOrderingPtr> _testOrderings;
  
  VarFactoryPtr _varFactory;
  map<DofOrdering*,bool> _isConforming;
  map<int, int> _testOrderEnhancements;
  map<int, int> _trialOrderEnhancements;
  void addConformingVertexPairings(int varID, DofOrderingPtr dofOrdering, CellTopoPtr cellTopo);
  int polyOrder(DofOrderingPtr dofOrdering, bool isTestOrdering);
  DofOrderingPtr pRefine(DofOrderingPtr dofOrdering,
                         CellTopoPtr, int pToAdd, bool isTestOrdering);
public:
  DofOrderingFactory(VarFactoryPtr varFactory);
  DofOrderingFactory(VarFactoryPtr varFactory,
                     map<int,int> trialOrderEnhancements,
                     map<int,int> testOrderEnhancements);
  // Deprecated constructors, use VarFactoryPtr version
  DofOrderingFactory(TBFPtr<double> bilinearForm);
  DofOrderingFactory(TBFPtr<double> bilinearForm,
                     map<int,int> trialOrderEnhancements,
                     map<int,int> testOrderEnhancements);
  DofOrderingPtr testOrdering(vector<int> &polyOrder, const shards::CellTopology &cellTopo);
  DofOrderingPtr trialOrdering(vector<int> &polyOrder, const shards::CellTopology &cellTopo,
                               bool conformingVertices = true);

  DofOrderingPtr testOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo);
  DofOrderingPtr testOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo,
                              const std::set< Tag > &tagsForVolume, const std::map<int, std::set< Tag > > &tagsForSide);
  
  DofOrderingPtr trialOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo, bool conformingVertices = true);
  DofOrderingPtr trialOrdering(vector<int> &polyOrder, CellTopoPtr cellTopo, bool conformingVertices,
                               const std::set< Tag > &tagsForVolume, const std::map<int, std::set< Tag > > &tagsForSide);

  int testPolyOrder(DofOrderingPtr testOrdering);
  int trialPolyOrder(DofOrderingPtr trialOrdering);
  DofOrderingPtr pRefineTest(DofOrderingPtr testOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);
  DofOrderingPtr pRefineTrial(DofOrderingPtr trialOrdering, const shards::CellTopology &cellTopo, int pToAdd = 1);

  DofOrderingPtr pRefineTest(DofOrderingPtr testOrdering, CellTopoPtr cellTopo, int pToAdd = 1);
  DofOrderingPtr pRefineTrial(DofOrderingPtr trialOrdering, CellTopoPtr cellTopo, int pToAdd = 1);

  DofOrderingPtr setSidePolyOrder(DofOrderingPtr dofOrdering, int sideIndexToSet, int newPolyOrder);

//  DofOrderingPtr getRelabeledDofOrdering(DofOrderingPtr dofOrdering, map<int,int> &oldKeysNewValues);

  DofOrderingPtr setBasisDegree(DofOrderingPtr dofOrdering, int basisDegree, bool replaceDiscontinuousFSWithContinuous); // sets all basis functions to have the same poly. degree, without regard for the function space they belong to.  ("polyOrder" in DofOrderingFactory usually is relative to the H^1 order, so that L^2 bases have degree 1 less.)

  DofOrderingPtr getFieldOrdering(DofOrderingPtr trialOrdering); // the sub-ordering that contains only the fields
  DofOrderingPtr getTraceOrdering(DofOrderingPtr trialOrdering); // the sub-ordering that contains only the traces

  map<int, int> getTestOrderEnhancements();
  map<int, int> getTrialOrderEnhancements();

  int getTestOrderEnhancement(int varID) const;
  int getTrialOrderEnhancement(int varID) const;
  
  int matchSides(DofOrderingPtr &firstOrdering, int firstSideIndex,
                 CellTopoPtr firstCellTopo,
                 DofOrderingPtr &secondOrdering, int secondSideIndex,
                 CellTopoPtr secondCellTopo);
  void childMatchParent(DofOrderingPtr &childTrialOrdering, int childSideIndex,
                        CellTopoPtr childTopo, int childIndexInParentSide, // == where in the multi-basis are we, if there is a multi-basis?
                        DofOrderingPtr &parentTrialOrdering, int sideIndex,
                        CellTopoPtr parentTopo);
  DofOrderingPtr upgradeSide(DofOrderingPtr dofOrdering,
                             CellTopoPtr cellTopo,
                             map<int,BasisPtr> varIDsToUpgrade,
                             int sideToUpgrade);

  //  DofOrderingPtr trialOrdering(int polyOrder, int* sidePolyOrder, const shards::CellTopology &cellTopo,
  //                                          bool conformingVertices = true);
};
}

#endif
