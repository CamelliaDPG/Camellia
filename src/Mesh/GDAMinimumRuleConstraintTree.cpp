//
//  GDAMinimumRuleConstraintTree.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/2/15.
//
//

#include "GDAMinimumRuleConstraintTree.h"

#include "CamelliaCellTools.h"
#include "GDAMinimumRule.h"

#include <iostream>
using namespace std;

static const bool PRINT_DEBUG_OUTPUT = false;

namespace Camellia
{
  ConstraintTreeNode::ConstraintTreeNode(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcellDimension,
                                         unsigned subcellOrdinalInSide, unsigned pathLength)
  {
    _cellID = cellID;
    _sideOrdinal = sideOrdinal;
    _subcellDimension = subcellDimension;
    _subcellOrdinalInSide = subcellOrdinalInSide;
    _pathLength = pathLength;
  }
  
  GlobalIndexType ConstraintTreeNode::cellID()
  {
    return _cellID;
  }
  
  std::vector<ConstraintTreeNodePtr> ConstraintTreeNode::children()
  {
    std::vector<ConstraintTreeNodePtr> children;
    for (auto entry : _constraints)
    {
      children.push_back(entry.second);
    }
    return children;
  }
  
  ConstraintTreeNodePtr ConstraintTreeNode::constraintForSubsubcell(unsigned subsubcellDimension, unsigned subsubcellOrdinalInSubcell)
  {
    auto entry = _constraints.find({subsubcellDimension,subsubcellOrdinalInSubcell});
    if ( entry == _constraints.end())
    {
      return Teuchos::null;
    }
    else
    {
      return entry->second;
    }
  }
  
  unsigned ConstraintTreeNode::pathLength()
  {
    return _pathLength;
  }
  
  unsigned ConstraintTreeNode::sideOrdinal()
  {
    return _sideOrdinal;
  }
  
  unsigned ConstraintTreeNode::subcellOrdinalInSide()
  {
    return _subcellOrdinalInSide;
  }
  
  unsigned ConstraintTreeNode::subcellDimension()
  {
    return _subcellDimension;
  }
  
  void ConstraintTreeNode::processSubcells(GDAMinimumRule* gdaMinRule, const CellConstraints &cellConstraints,
                                           map<pair<GlobalIndexType,unsigned>,vector<ConstrainedSubcellID>> &newVisitedDomains,
                                           set<pair<GlobalIndexType,unsigned>> &previouslyVisitedDomains)
  {
    map<pair<GlobalIndexType,unsigned>,unsigned> justVisitedDomains; // keys: (cellID, sideOrdinal); value: the dimension of the constrained subcell
    
    CellTopoPtr cellTopo = gdaMinRule->elementType(_cellID)->cellTopoPtr;
    unsigned sideDim = cellTopo->getDimension() - 1;
    CellTopoPtr sideTopo = cellTopo->getSubcell(sideDim, _sideOrdinal);
    CellTopoPtr subcellTopo = sideTopo->getSubcell(_subcellDimension, _subcellOrdinalInSide);
    pair<GlobalIndexType,unsigned> thisDomainID = {_cellID,_sideOrdinal};
    if (PRINT_DEBUG_OUTPUT)
    {
      cout << "Processing subcells for cell " << _cellID << ", side " << _sideOrdinal << ", subcell " << _subcellOrdinalInSide << " of dimension ";
      cout << _subcellDimension << endl;
    }
    for (int dim=_subcellDimension; dim >= 0; dim--)
    {
      int subsubcellCount = subcellTopo->getSubcellCount(dim);
      for (int subsubcellOrdinal=0; subsubcellOrdinal<subsubcellCount; subsubcellOrdinal++)
      {
        unsigned subsubcellOrdinalInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, _subcellDimension, _subcellOrdinalInSide,
                                                                                dim, subsubcellOrdinal);
        unsigned subsubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, _sideOrdinal, dim, subsubcellOrdinalInSide);
        const AnnotatedEntity* constrainingEntity = &cellConstraints.subcellConstraints[dim][subsubcellOrdinalInCell];
        GlobalIndexType constrainingCellID = constrainingEntity->cellID;
        unsigned constrainingSide = constrainingEntity->sideOrdinal;
        pair<GlobalIndexType,unsigned> domainID = {constrainingCellID,constrainingSide};
        if (thisDomainID == domainID)
        {
          // unconstrained/self-constrained: don't do anything
        }
        else if (previouslyVisitedDomains.find(domainID) == previouslyVisitedDomains.end())
        {
          unsigned constrainingSubcellOrdinalInSide = constrainingEntity->subcellOrdinal; // AnnotatedEntity.subcellordinal is defined to be the subcell ordinal in the *domain*
          ConstraintTreeNodePtr constraintNode = Teuchos::rcp( new ConstraintTreeNode(constrainingCellID, constrainingSide, constrainingEntity->dimension,
                                                                                      constrainingSubcellOrdinalInSide, _pathLength+1));
          _constraints[{dim,subsubcellOrdinal}] = constraintNode;
          ConstrainedSubcellID constrainedSubcellID;
          constrainedSubcellID.first = Teuchos::rcp(this,false); // false: weak reference
          constrainedSubcellID.second = {dim,subsubcellOrdinal};
          newVisitedDomains[domainID].push_back(constrainedSubcellID);
          justVisitedDomains[domainID] = dim;
          if (PRINT_DEBUG_OUTPUT)
          {
            cout << "subsubcell " << subsubcellOrdinal << " of dimension " << dim << " constrained by ";
            cout << "cell " << constrainingCellID << ", side " << constrainingSide << ", subcell ";
            cout << constrainingSubcellOrdinalInSide << " of dimension " << constrainingEntity->dimension << endl;
          }
        }
      }
    }
  }
  
  // ! subcellOrdinal is relative to the subcell represented by this node
  void ConstraintTreeNode::pruneConstraint(unsigned subcellDimension, unsigned subcellOrdinal)
  {
    _constraints.erase({subcellDimension,subcellOrdinal});
  }
  
  GDAMinimumRuleConstraintTree::GDAMinimumRuleConstraintTree(GDAMinimumRule* gdaMinimumRule, GlobalIndexType cellID, unsigned sideOrdinal)
  {
    bool haveWarnedAboutMultipleWinners = false;
    if (PRINT_DEBUG_OUTPUT) cout << "****** Creating constraint tree for cell " << cellID << ", side " << sideOrdinal << " ******\n";
    int sideDim = gdaMinimumRule->elementType(cellID)->cellTopoPtr->getDimension() - 1;
    int sideOrdinalInSide = 0;
    int pathLength = 0;
    _root = Teuchos::rcp( new ConstraintTreeNode(cellID, sideOrdinal, sideDim, sideOrdinalInSide, pathLength));
    
    MeshTopologyPtr meshTopo = gdaMinimumRule->getMeshTopology();
    set<pair<GlobalIndexType,unsigned>> previouslyVisitedDomains;
    vector<ConstraintTreeNodePtr> newLeafNodes = {_root};
    while (newLeafNodes.size() > 0)
    {
      vector<ConstraintTreeNodePtr> leafNodes = newLeafNodes;
      map<pair<GlobalIndexType,unsigned>,vector<ConstrainedSubcellID>> newVisitedDomains;
      newLeafNodes = {};
      for (auto leafNode : leafNodes)
      {
        CellConstraints cellConstraints = gdaMinimumRule->getCellConstraints(leafNode->cellID());
        leafNode->processSubcells(gdaMinimumRule, cellConstraints, newVisitedDomains, previouslyVisitedDomains);
      }
      for (auto newDomainEntry : newVisitedDomains)
      {
        auto constrainingDomainID = newDomainEntry.first;
        previouslyVisitedDomains.insert(constrainingDomainID);
        if (PRINT_DEBUG_OUTPUT)
        {
          cout << "constrainingDomain (cellID, sideOrdinal) = (" << constrainingDomainID.first;
          cout << "," << constrainingDomainID.second << ")\n";
        }
        unsigned d_high = 0;
        for (auto constrainedSubcellEntry : newDomainEntry.second)
        {
          auto subcellID = constrainedSubcellEntry.second;
          d_high = max(subcellID.first, d_high);
        }
        IndexType winningSubcellEntityIndex = -1; // the entity index in MeshTopology for the "winning" constrained entity
        for (auto constrainedSubcellEntry : newDomainEntry.second)
        {
          ConstraintTreeNodePtr constrainedNode = constrainedSubcellEntry.first;
          CellPtr constrainedCell = meshTopo->getCell(constrainedNode->cellID());
          CellTopoPtr cellTopo = constrainedCell->topology();
          CellTopoPtr sideTopo = cellTopo->getSide(constrainedNode->sideOrdinal());
          CellTopoPtr subcellTopo = sideTopo->getSubcell(constrainedNode->subcellDimension(), constrainedNode->subcellOrdinalInSide());
          
          auto subcellID = constrainedSubcellEntry.second;
          unsigned subcellDimension = subcellID.first;
          unsigned subcellOrdinal = subcellID.second; // ordinal is relative to the subcell identified by the constrained node
          
          unsigned subcellOrdinalInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo,
                                                                               constrainedNode->subcellDimension(), constrainedNode->subcellOrdinalInSide(),
                                                                               subcellDimension, subcellOrdinal);
          unsigned subcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo,
                                                                               sideTopo->getDimension(), constrainedNode->sideOrdinal(),
                                                                               subcellDimension, subcellOrdinalInSide);
          
          IndexType subcellEntityIndex = constrainedCell->entityIndex(subcellDimension, subcellOrdinalInCell);
          TEUCHOS_TEST_FOR_EXCEPTION(subcellEntityIndex == -1, std::invalid_argument,
                                     "GDAMinimumRuleConstraintTree internal error: entity not found in constrained cell");
          
          if ((subcellDimension < d_high) || (subcellEntityIndex == winningSubcellEntityIndex))
          {
            constrainedNode->pruneConstraint(subcellDimension,subcellOrdinal);
          }
          else
          {
            if (PRINT_DEBUG_OUTPUT)
            {
              cout << "(subcellDimension, subcellOrdinal) = (";
              cout << subcellDimension << "," << subcellOrdinal << ")\n";
            }
            if (winningSubcellEntityIndex == -1)
            {
              winningSubcellEntityIndex = subcellEntityIndex;
            }
            else
            {
              if (!haveWarnedAboutMultipleWinners)
              {
                cout << "WARNING: GDAMinimumRuleConstraintTree found multiple 'winning' entities; will allow for now (i.e., this is no longer a tree).  ";
                cout << "This is known to be possible for 2-irregular 3D meshes.  We recommend using 1-irregular meshes.  ";
                cout << "Note that this is only a warning; GDAMinimumRule will check for incompatible entries and throw an exception if these are found.  ";
                cout << "Encountered the multiple winners while " << "creating constraint tree for cell " << cellID << ", side " << sideOrdinal << ".\n";
              }
//              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GDAMinimumRuleConstraintTree internal error: multiple 'winning' entities");
            }
          }
        }
      }
      
      // set the newLeafNodes:
      for (auto leafNode : leafNodes)
      {
        vector<ConstraintTreeNodePtr> children = leafNode->children();
        newLeafNodes.insert(newLeafNodes.end(),children.begin(),children.end());
      }
    }
  }
  
  ConstraintTreeNodePtr GDAMinimumRuleConstraintTree::getRoot()
  {
    return _root;
  }
}
