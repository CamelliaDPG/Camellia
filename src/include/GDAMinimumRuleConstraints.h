//
//  GDAMinimumRuleConstraints.h
//  Camellia
//
//  Created by Nate Roberts on 8/18/14.
//
//

#ifndef Camellia_GDAMinimumRuleConstraints_h
#define Camellia_GDAMinimumRuleConstraints_h

#include "Teuchos_RCP.hpp"
#include "IndexType.h"
#include "RefinementPattern.h"
#include "BasisReconciliation.h"

class GDAMinimumRule;

// GDAMinimumRule helper classes for keeping track of the directed, acyclic graph that allows determination of global dofs in terms of local dofs on an element:
class GDAMinimumRuleConstraintEntry;

typedef Teuchos::RCP<GDAMinimumRuleConstraintEntry> ConstraintEntryPtr;

struct AnnotatedEntity {
  GlobalIndexType cellID;
  unsigned sideOrdinal;    // -1 for volume-based constraint determination (i.e. for cases when the basis domain is the whole cell)
  unsigned subcellOrdinal; // subcell ordinal in the domain (cell for volume-based, side for side-based)
  unsigned dimension; // subcells can be constrained by subcells of higher dimension (i.e. this is not redundant!)
  
  bool operator < (const AnnotatedEntity & other) const {
    return (cellID < other.cellID) || (sideOrdinal < other.sideOrdinal) || (subcellOrdinal < other.subcellOrdinal) || (dimension < other.dimension);
  }
  
  bool operator == (const AnnotatedEntity & other) const {
    return !(*this < other) && !(other < *this);
  }
  
  bool operator != (const AnnotatedEntity & other) const {
    return !(*this == other);
  }
};

class GDAMinimumRuleConstraints {
  map< AnnotatedEntity, ConstraintEntryPtr> _constraintEntries; // flat lookup to ensure uniqueness
  
  // recursive, ensures all prior entries processed before applying constraint to thisEntry:
  static void computeConstraintWeights(map<AnnotatedEntity, SubBasisReconciliationWeights> & constraintWeights,
                                       GDAMinimumRule* minRule, ConstraintEntryPtr previousEntry, ConstraintEntryPtr thisEntry, VarPtr var );
  
public:
  bool constraintEntryExists( AnnotatedEntity &subcellInfo );
  ConstraintEntryPtr getConstraintEntry( AnnotatedEntity &subcellInfo );
  void setConstraintEntry(AnnotatedEntity &subcellInfo, ConstraintEntryPtr entry);
  
  static void computeConstraintWeights( map<AnnotatedEntity, SubBasisReconciliationWeights> & constraintWeights,
                                        GDAMinimumRule* minRule, ConstraintEntryPtr rootEntry, VarPtr var );
  
  static BasisPtr getBasis(GDAMinimumRule* minRule, ConstraintEntryPtr entry, VarPtr var);

};

class GDAMinimumRuleConstraintEntry {
  int _entityDim;
  GlobalIndexType _entityIndex;
  
  // "out edges": we *either* have subcell entries *or* a single constrainingEntity (not both; vertices may have neither)
  vector< Teuchos::RCP<GDAMinimumRuleConstraintEntry> > _subcellEntries;
  Teuchos::RCP<GDAMinimumRuleConstraintEntry> _constrainingEntity;
  
  // "in" edges: (map ensures no duplication)
  map< AnnotatedEntity, ConstraintEntryPtr > _priorEntries;
  
  CellPtr _cell;
  int _sideOrdinal;
  int _subcellOrdinalInSide;
  int _subcellOrdinalInCell;
  GDAMinimumRule* _minRule;
  
  // the following members will be populated for constrained entries (i.e. those for which isUnconstrained() returns false)
  RefinementBranch _volumeRefinements;
  CellPtr _ancestralCell;
  unsigned _ancestralSideOrdinal;
  unsigned _ancestralSubcellOrdinal;
  unsigned _ancestralSubcellDimension;
  unsigned _composedPermutation;
  
  void determineAncestralSideOrdinal(AnnotatedEntity &constrainingSubcellInfo); // assumes _ancestralCell, _ancestralSubcellOrdinal, and _ancestralSubcellDimension have been set
public:
  GDAMinimumRuleConstraintEntry(GDAMinimumRuleConstraints &definedConstraints, GDAMinimumRule* minRule,
                                CellPtr cell, int sideOrdinal, int subcellDim, int subcellOrdinalInSide);
  
  void addPriorEntry(ConstraintEntryPtr priorEntry);
  
  unsigned ancestralSideOrdinal();
  unsigned ancestralSubcellOrdinal();
  unsigned ancestralSubcellDimension();
  
  CellPtr cell();
  unsigned composedPermutation();
  int entityDim();
  GlobalIndexType entityIndex();
  
  vector< ConstraintEntryPtr > getPriorEntries();
  
  bool isUnconstrained();
  vector< ConstraintEntryPtr > nextEntries();
  int subcellOrdinalInCell();
  int subcellOrdinalInSide();
  
  AnnotatedEntity subcellInfo();
  
  RefinementBranch volumeRefinements();
};

#endif
