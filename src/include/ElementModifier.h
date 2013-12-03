//
//  ElementModifier.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 11/26/13.
//
//

#ifndef Camellia_debug_ElementModifier_h
#define Camellia_debug_ElementModifier_h

class ElementModifier {
public:
  // "owned" means that the buck stops here--it's our discretization that others reconcile to.
  //  But it's worth noting that even for owned dofs, there may not be a one-to-one correspondence to
  //  the global degrees of freedom.  E.g. traces of H^1 in 2D will have their vertex dofs identified with each
  //  other, eliminating a global dof.
  
  // "unowned" means the opposite--so between these two methods, all local dofs are accounted for
  set<int> &ownedLocalDofs();    // local dof indices, as seen by DofOrdering
  set<int> &unownedLocalDofs(); // local dof indices which are owned by other elements
  
  map<int, int> &localDofOwner(); // key: local dof index.  Value: neighbor cellID
  
  set<int> &modifiedGlobalDofIndices();
  void getModifiedValues(FieldContainer<double> &modifiedValues, const FieldContainer<double> &localValues);

// steps:
/*
 1. On each element side, check neighbor to determine ownership.  (Rule: the coarser neighbor wins.  For ties, the neighbor with smaller cellID wins.)
 2. Run BasisReconciliation on element sides to determine internal weights and owned global dof count.
 3. Run BasisReconciliation to determine weights between neighbors.
 
 Once 1-3 have been done for every element:
 4. For each element, determine a "net" modification matrix that goes from local dof indices to globally modified values.  To do this, we must walk the ownership graph.  This will be more efficient if we special-case dof identifications.
 
 
 */

};

#endif
