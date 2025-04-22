//
//  UnrefinementProjectionErrorIndicator.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 8/13/20.
//

#include "UnrefinementProjectionErrorIndicator.h"

#include "BasisSumFunction.h"
#include "GlobalDofAssignment.h"

using namespace Camellia;

UnrefinementProjectionErrorIndicator::UnrefinementProjectionErrorIndicator(SolutionPtr soln, VarPtr var)
:
ErrorIndicator(soln->mesh()),
_soln(soln),
_var(var)
{}

void UnrefinementProjectionErrorIndicator::measureError()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "measureError() should not be called on UnrefinementProjectionErrorIndicator");
}

void UnrefinementProjectionErrorIndicator::gatherParentError(const std::set<GlobalIndexType> &excludedChildren)
{
  /*
  For all parents that we we own:
    a. gather child cell coefficients in Solution
    b. project from children onto parent (just as we would do during an unrefinement)
    c. measure the difference on each child (can use the TwoStreamInstability overkill comparison as a starting point for this)
    d. determine parent error sum the error from the children (square root of squared sum)
  */
  
  // this method fills _parentErrorMeasures; clear it first
  _parentErrorMeasures.clear();
  
  const int myPID = _mesh->Comm()->getRank();
  
  auto meshTopo = _mesh->getTopology();
  
  // want to gather cell coefficients for children
  const auto &myCellIDs = _mesh->cellIDsInPartition();
  std::set<GlobalIndexType> myParentCellIDs;
  std::set<GlobalIndexType> childCellIDsToRequest;
  for (const auto & cellID : myCellIDs)
  {
    CellPtr cell   = meshTopo->getCell(cellID);
    CellPtr parent = cell->getParent();
    
    if (parent != Teuchos::null)
    {
      // check that all siblings are active (if not, parent is not eligible for unrefinement, and should be ignored)
      bool parentHasGrandchildren = false;
      bool parentHasExcludedChild = false;
      for (const auto & childCell : parent->children())
      {
        if (childCell->numChildren() > 0)
        {
          // parent has grandchildren. ==> cellID's error can/should be ignored; its parent is not eligible for refinement.
          parentHasGrandchildren = true;
          break;
        }
        if (excludedChildren.find(childCell->cellIndex()) != excludedChildren.end())
        {
          parentHasExcludedChild = true;
          break;
        }
      }
      if (parentHasGrandchildren || parentHasExcludedChild)
      {
        // go to next cell in myCellIDs
        continue;
      }
      
      const auto &children = parent->children();
      
      const GlobalIndexType firstChildID = children[0]->cellIndex();
      const int ownerPID = _mesh->globalDofAssignment()->partitionForCellID( firstChildID );
      if (ownerPID == myPID)
      {
        myParentCellIDs.insert(parent->cellIndex());
        for (int childOrdinal=1; childOrdinal<children.size(); childOrdinal++)
        {
          auto childCellID = children[childOrdinal]->cellIndex();
          if (myPID != _mesh->globalDofAssignment()->partitionForCellID(childCellID))
          {
            childCellIDsToRequest.insert(childCellID);
          }
        }
      }
    }
  }
  
  _soln->importSolutionForOffRankCells(childCellIDsToRequest);
  
  // **** Project from children onto parent; project back, and measure error **** //
  const int numLHSes = _soln->numSolutions();
  for (const GlobalIndexType & parentCellID : myParentCellIDs)
  {
    auto childCellIDs = meshTopo->getCell(parentCellID)->getChildIndices(meshTopo);
    auto parentElementType = _soln->mesh()->getElementType(parentCellID);
    for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
    {
      _soln->projectChildrenOntoParent(parentCellID, lhsOrdinal);
      
      std::vector< FieldContainer<double> > savedChildCoefficents(childCellIDs.size());
      
      // save coefficients before we clobber them by projecting back from parent
      for (int childOrdinal=0; childOrdinal<childCellIDs.size(); childOrdinal++)
      {
        const bool warnAboutOffRankCells = false;
        savedChildCoefficents[childOrdinal] = _soln->allCoefficientsForCellID(childCellIDs[childOrdinal], warnAboutOffRankCells);
      }
      
      _soln->projectOldCellOntoNewCells(parentCellID, parentElementType, childCellIDs);
      
      double parentErrorSquared = 0.0;
      for (int childOrdinal=0; childOrdinal<childCellIDs.size(); childOrdinal++)
      {
        const auto & childCellID = childCellIDs[childOrdinal];
        
        const bool warnAboutOffRankCells = false;
        const auto & childCoefficients = _soln->allCoefficientsForCellID(childCellIDs[childOrdinal], warnAboutOffRankCells);
        
        auto childDofOrdering = _soln->mesh()->getElementType(childCellID)->trialOrderPtr;
        BasisPtr basis = childDofOrdering->getBasis(_var->ID());
        
        FieldContainer<double> differenceBasisCoefficients(1, basis->getCardinality());
        
        for (int basisOrdinal=0; basisOrdinal<basis->getCardinality(); basisOrdinal++)
        {
          int dofIndex = childDofOrdering->getDofIndex(_var->ID(), basisOrdinal);
          differenceBasisCoefficients(0, basisOrdinal) = savedChildCoefficents[childOrdinal](dofIndex) - childCoefficients(dofIndex);
        }
        
        FunctionPtr difference   = Teuchos::rcp( new BasisSumFunction(basis, differenceBasisCoefficients) );
        FunctionPtr errorSquared = difference * difference;
        
        BasisCachePtr basisCacheForCell = BasisCache::basisCacheForCell(_mesh, childCellID);
        parentErrorSquared += errorSquared->integrate(basisCacheForCell);
        
        // restore saved coefficients
        _soln->setLocalCoefficientsForCell(childCellID, savedChildCoefficents[childOrdinal]);
      }
      _parentErrorMeasures[parentCellID] = std::sqrt(parentErrorSquared);
//      std::cout << "Error for parent cell " << parentCellID << ": " << _parentErrorMeasures[parentCellID] << std::endl;
    }
  }
}
