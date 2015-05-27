//
//  SubBasisDofMatrixMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "SubBasisDofMapper.h"
#include "SubBasisDofPermutationMapper.h"
#include "SubBasisDofMatrixMapper.h"

using namespace Intrepid;
using namespace Camellia;

SubBasisDofMapperPtr SubBasisDofMapper::subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals)
{
  if (dofOrdinalFilter.size() != globalDofOrdinals.size())
  {
    cout << "ERROR: cannot make a permutation mapper for dof lists of different size.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: cannot make a permutation mapper for dof lists of different size.\n");
  }

  // since the permutation mapper isn't quite ready for prime time (constraint transposition not yet supported), we make a simple matrix-based mapper instead
  return Teuchos::rcp(new SubBasisDofPermutationMapper(dofOrdinalFilter, globalDofOrdinals));
//  int n = dofOrdinalFilter.size();
//  FieldContainer<double> identityMatrix(n,n);
//  for (int i=0; i<n; i++) {
//    identityMatrix(i,i) = 1.0;
//  }
//  return subBasisDofMapper(dofOrdinalFilter, globalDofOrdinals, identityMatrix);
}

SubBasisDofMapperPtr SubBasisDofMapper::subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, const FieldContainer<double> &constraintMatrix)
{
  bool checkForPermutation = true;

  if (checkForPermutation)
  {
    // check the constraint matrix to see if it's a permutation
    if (constraintMatrix.dimension(0) == constraintMatrix.dimension(1))
    {
      double tol = 1e-8;
      int n = constraintMatrix.dimension(0);
      bool foundInvalidEntry = false; // entry that isn't 1 or 0, or a second 1 in a row/column, or a row/column without a 1
      // check columns
      for (int i=0; i<n; i++)
      {
        bool foundUnitEntryInColumn = false;
        for (int j=0; j<n; j++)
        {
          double val = constraintMatrix(i,j);
          if (abs(val-1) < tol)   // 1 entry
          {
            if (foundUnitEntryInColumn)
            {
              foundInvalidEntry = true;
              break;
            }
            foundUnitEntryInColumn = true;
          }
          else if (abs(val) < tol)     // 0 entry
          {

          }
          else
          {
            foundInvalidEntry = true;
            break;
          }
        }
        if (! foundUnitEntryInColumn)
        {
          foundInvalidEntry = true;
        }
        if (foundInvalidEntry)
        {
          break;
        }
      }
      if (! foundInvalidEntry)
      {
        int permutation[n]; // the inverse permutation is what we'll want to apply to the globalDofOrdinals so that they're in the right order for SubBasisDofPermutationMapper
        // check rows
        for (int j=0; j<n; j++)
        {
          bool foundUnitEntryInRow = false;
          for (int i=0; i<n; i++)
          {
            double val = constraintMatrix(i,j);
            if (abs(val-1) < tol)   // 1 entry
            {
              if (foundUnitEntryInRow)   // this is a second unit entry, so invalid
              {
                foundInvalidEntry = true;
                break;
              }
              foundUnitEntryInRow = true;
              permutation[i] = j;
            }
            else if (abs(val) < tol)     // 0 entry
            {

            }
            else
            {
              foundInvalidEntry = true;
              break;
            }
          }
          if (! foundUnitEntryInRow)
          {
            foundInvalidEntry = true;
          }
          if (foundInvalidEntry)
          {
            break;
          }
        }
        if (! foundInvalidEntry)   // then we do have a permutation matrix
        {
          // permute the globalDofOrdinals according to the inverse permutation
          vector<GlobalIndexType> permutedGlobalDofOrdinals = globalDofOrdinals;
          for (int i=0; i<n; i++)
          {
            permutedGlobalDofOrdinals[permutation[i]] = globalDofOrdinals[i];
          }
          return SubBasisDofMapper::subBasisDofMapper(dofOrdinalFilter, permutedGlobalDofOrdinals);
        }
      }
    }
  }
  return Teuchos::rcp(new SubBasisDofMatrixMapper(dofOrdinalFilter, globalDofOrdinals, constraintMatrix));
}

SubBasisDofMapper::~SubBasisDofMapper()
{

}