//
//  GDAMinimumRuleConstraints.h
//  Camellia
//
//  Created by Nate Roberts on 6/2/15.
//
//

#ifndef Camellia_GDAMinimumRuleConstraintTree_h
#define Camellia_GDAMinimumRuleConstraintTree_h

#include "RefinementPattern.h"
#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"

/***
 
 
 
 ***/

namespace Camellia
{
  // some forward declarations:
  struct AnnotatedEntity;
  struct CellConstraints;
  class ConstraintTreeNode;
  class GDAMinimumRule;
  
  typedef Teuchos::RCP<ConstraintTreeNode> ConstraintTreeNodePtr;
  typedef std::pair<ConstraintTreeNodePtr,std::pair<unsigned,unsigned>> ConstrainedSubcellID;
  
  class ConstraintTreeNode
  {
    GlobalIndexType _cellID;
    unsigned _sideOrdinal;
    unsigned _subcellOrdinalInSide;
    unsigned _subcellDimension;
    std::map<std::pair<unsigned,unsigned>,ConstraintTreeNodePtr> _constraints; // keys: (dim,subcordInSubcell)
    unsigned _pathLength;
  public:
    GlobalIndexType cellID();
    unsigned pathLength();
    unsigned sideOrdinal();
    unsigned subcellOrdinalInSide();
    unsigned subcellDimension();
  public:
    ConstraintTreeNode(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcellDimension, unsigned subcellOrdinalInSide, unsigned pathLength);
    std::vector<ConstraintTreeNodePtr> children();
    ConstraintTreeNodePtr constraintForSubsubcell(unsigned subsubcellDimension, unsigned subsubcellOrdinalInSubcell);
    void processSubcells(GDAMinimumRule* gdaMinRule, const CellConstraints &cellConstraints,
                         std::map<std::pair<GlobalIndexType,unsigned>,std::vector<ConstrainedSubcellID>> &newVisitedDomains,
                         std::set<std::pair<GlobalIndexType,unsigned>> &previouslyVisitedDomains);
    void pruneConstraint(unsigned subcellDimension, unsigned subcellOrdinal);
  };
  
  class GDAMinimumRuleConstraintTree
  {
    ConstraintTreeNodePtr _root;
  public:
    GDAMinimumRuleConstraintTree(GDAMinimumRule* gdaMinimumRule, GlobalIndexType cellID, unsigned sideOrdinal);
    ConstraintTreeNodePtr getRoot();
  };
}

#endif
