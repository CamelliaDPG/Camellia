//
//  SubBasisDofMatrixMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "SubBasisDofMatrixMapper.h"

#include "CamelliaDebugUtility.h"
#include "SerialDenseWrapper.h"

using namespace std;

using namespace Intrepid;
using namespace Camellia;

SubBasisDofMatrixMapper::SubBasisDofMatrixMapper(const set<int> &basisDofOrdinalFilter, const vector<GlobalIndexType> &mappedGlobalDofOrdinals, const FieldContainer<double> &constraintMatrix)
{
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _mappedGlobalDofOrdinals = mappedGlobalDofOrdinals;
  _constraintMatrix = constraintMatrix;

  // The constraint matrix should have size (fine,coarse) -- which is to say (local, global)
  if (_constraintMatrix.dimension(0) != basisDofOrdinalFilter.size())
  {
    cout << "ERROR: constraint matrix row dimension must match the local sub-basis size.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraint matrix row dimension must match the local sub-basis size");
  }
  if (_constraintMatrix.dimension(1) != mappedGlobalDofOrdinals.size())
  {
    cout << "ERROR: constraint matrix column dimension must match the number of mapped global dof ordinals.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraint matrix column dimension must match the number of mapped global dof ordinals");
  }

//  { // CHECKING THE STRUCTURE OF THE CONSTRAINT MATRIX:
//    static bool havePrintedOne = false;
//    if (!havePrintedOne)
//    {
//      havePrintedOne = true;
//      
//      set<int> visitedRows;
//      set<int> visitedColumns;
//      
//      vector<pair<set<int>,set<int>>> blocks;
//
//      auto newColumnsForRow = [this, &visitedColumns] (int row, set<int> &previousColumns) -> set<int>
//      {
//        set<int> newColumns;
//        for (int j=0; j<_constraintMatrix.dimension(1); j++)
//        {
//          if (visitedColumns.find(j) != visitedColumns.end()) continue;
//          if (previousColumns.find(j) != previousColumns.end()) continue;
//          if (abs(_constraintMatrix(row,j)) > 0)
//          {
//            newColumns.insert(j);
//          }
//        }
//        return newColumns;
//      };
//      
//      auto newRowsForColumn = [this, &visitedRows] (int col, set<int> &previousRows) -> set<int>
//      {
//        set<int> newRows;
//        for (int i=0; i<_constraintMatrix.dimension(0); i++)
//        {
//          if (visitedRows.find(i) != visitedRows.end()) continue;
//          if (previousRows.find(i) != previousRows.end()) continue;
//          if (abs(_constraintMatrix(i,col)) > 0)
//          {
//            newRows.insert(i);
//          }
//        }
//        return newRows;
//      };
//      
//      for (int i=0; i<_constraintMatrix.dimension(0); i++)
//      {
//        if (visitedRows.find(i) != visitedRows.end()) continue;
//       
//        pair<set<int>,set<int>> block; // the block that includes i
//        
//        set<int> newRows = {i};
//        while (newRows.size() > 0)
//        {
//          block.first.insert(newRows.begin(),newRows.end());
//        
//          set<int> previousNewRows = newRows;
//          newRows.clear();
//          for (int row : previousNewRows)
//          {
//            set<int> newColumns = newColumnsForRow(row,block.second);
//            if (newColumns.size() > 0)
//            {
//              block.second.insert(newColumns.begin(),newColumns.end());
//            }
//            
//            for (int col : newColumns)
//            {
//              set<int> newRowsForCol = newRowsForColumn(col,block.first);
//              newRows.insert(newRowsForCol.begin(), newRowsForCol.end());
//            }
//          }
//        }
//        blocks.push_back(block);
//        visitedRows.insert(block.first.begin(),block.first.end());
//        visitedColumns.insert(block.second.begin(),block.second.end());
//      }
//      
//      cout << "first _constraintMatrix has " << blocks.size() << " blocks.\n";
//      int i = 0;
//      for (auto block : blocks)
//      {
//        cout << "BLOCK " << i << ":\n";
//        print("rows", block.first);
//        print("columns", block.second);
//        i++;
//      }
////      cout << "first _constraintMatrix (nonzeros only):\n";
////      if (abs(_constraintMatrix(i,j)) > 0)
////        cout << setw(10) << i << setw(10) << j << setw(10) << _constraintMatrix(i,j) << endl;
//
//    }
//  }
}

const set<int> & SubBasisDofMatrixMapper::basisDofOrdinalFilter()
{
  return _basisDofOrdinalFilter;
}

const FieldContainer<double> &SubBasisDofMatrixMapper::constraintMatrix()
{
  return _constraintMatrix;
}

FieldContainer<double> SubBasisDofMatrixMapper::getConstraintMatrix()
{
  return _constraintMatrix;
}

bool SubBasisDofMatrixMapper::isNegatedPermutation()
{
  return false;
}

bool SubBasisDofMatrixMapper::isPermutation()
{
  return false;
}

FieldContainer<double> SubBasisDofMatrixMapper::mapData(bool transposeConstraint, FieldContainer<double> &localData, bool applyOnLeftOnly)
{
  // localData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  bool didReshape = false;
  if (localData.rank() == 1)
  {
    // reshape as a rank 2 container (column vector as a matrix):
    localData.resize(localData.dimension(0),1);
    didReshape = true;
  }
  if (localData.rank() != 2)
  {
    cout << "localData must have rank 1 or 2.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must have rank 1 or 2");
  }
  int constraintRows = transposeConstraint ? _constraintMatrix.dimension(1) : _constraintMatrix.dimension(0);
  int constraintCols = transposeConstraint ? _constraintMatrix.dimension(0) : _constraintMatrix.dimension(1);
  int dataCols = localData.dimension(1);
  int dataRows = localData.dimension(0);

  if ((dataCols==0) || (dataRows==0) || (constraintRows==0) || (constraintCols==0))
  {
    cout << "degenerate matrix encountered.\n";
  }

  // given the multiplication we'll do, we need constraint columns = data rows
  if (constraintCols != dataRows)
  {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData() for left-multiplication by constraint matrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }
  // (could also test that the dimensions match what we expect in terms of the size of the mapped global dof ordinals or basisDofOrdinal filter)

  FieldContainer<double> result1(constraintRows,dataCols);

  char constraintTransposeFlag = transposeConstraint ? 'T' : 'N';
  char dataTransposeFlag = 'N';

  SerialDenseWrapper::multiply(result1,_constraintMatrix,localData,constraintTransposeFlag,dataTransposeFlag);

  if (didReshape)   // change the shape of localData back, and return result
  {
    localData.resize(localData.dimension(0));
    result1.resize(result1.size());
    return result1;
  }

  if (applyOnLeftOnly) return result1;

  if (constraintCols != dataCols)
  {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData() for right-multiplication by constraint matrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }

  constraintTransposeFlag = (!transposeConstraint) ? 'T' : 'N'; // opposite of the above choice, since now we multiply on the right
  char resultTransposeFlag = 'N';

  FieldContainer<double> result(constraintRows,constraintRows);
  SerialDenseWrapper::multiply(result,result1,_constraintMatrix,resultTransposeFlag,constraintTransposeFlag);

  return result;
}

void SubBasisDofMatrixMapper::mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &allLocalData, const vector<int> &basisOrdinalsInLocalData,
                                                         const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                                         bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData)
{
  const set<int>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<int> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());
  FieldContainer<double> subBasisData(basisOrdinalFilter->size());
  int dofCount = basisOrdinalFilter->size();
  if (allLocalData.rank()==1)
  {
    for (int i=0; i<dofCount; i++)
    {
      subBasisData[i] = allLocalData[basisOrdinalsInLocalData[dofIndices[i]]];
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mapDataIntoGlobalContainer only supports rank 1 basis data");
  }
  
  // subBasisData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  // reshape as a rank 2 container (column vector as a matrix):
  subBasisData.resize(subBasisData.dimension(0),1);
  
  this->mapSubBasisDataIntoGlobalContainer(subBasisData, globalIndexToOrdinal, fittableDofsOnly, fittableDofIndices, globalData);
}

void SubBasisDofMatrixMapper::mapDataIntoGlobalContainer(const FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
    bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData)
{
  // like calling mapData, above, with transposeConstraint = true

  const set<int>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<int> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());
  FieldContainer<double> subBasisData(basisOrdinalFilter->size());
  int dofCount = basisOrdinalFilter->size();
  if (wholeBasisData.rank()==1)
  {
    for (int i=0; i<dofCount; i++)
    {
      subBasisData[i] = wholeBasisData[dofIndices[i]];
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mapDataIntoGlobalContainer only supports rank 1 basis data");
  }

  // subBasisData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  // reshape as a rank 2 container (column vector as a matrix):
  subBasisData.resize(subBasisData.dimension(0),1);

  this->mapSubBasisDataIntoGlobalContainer(subBasisData, globalIndexToOrdinal, fittableDofsOnly, fittableDofIndices, globalData);
}

void SubBasisDofMatrixMapper::mapSubBasisDataIntoGlobalContainer(const FieldContainer<double> &subBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                                                 bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData)
{
  // like calling mapData, above, with transposeConstraint = true
  
  int constraintRows = _constraintMatrix.dimension(1);
  int constraintCols = _constraintMatrix.dimension(0);
  int dataCols = subBasisData.dimension(1);
  int dataRows = subBasisData.dimension(0);
  
  if ((dataCols==0) || (dataRows==0) || (constraintRows==0) || (constraintCols==0))
  {
    cout << "degenerate matrix encountered.\n";
  }
  
  // given the multiplication we'll do, we need constraint columns = data rows
  if (constraintCols != dataRows)
  {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData() for left-multiplication by constraint matrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }
  // (could also test that the dimensions match what we expect in terms of the size of the mapped global dof ordinals or basisDofOrdinal filter)
  
  FieldContainer<double> result1(constraintRows,dataCols);
  
  char constraintTransposeFlag = 'T';
  char dataTransposeFlag = 'N';
  
  SerialDenseWrapper::multiply(result1,_constraintMatrix,subBasisData,constraintTransposeFlag,dataTransposeFlag);
  
  for (int i=0; i<result1.size(); i++)
  {
    GlobalIndexType globalIndex_i = _mappedGlobalDofOrdinals[i];
    if (fittableDofsOnly && (fittableDofIndices.find(globalIndex_i) == fittableDofIndices.end())) continue; // skip this one
    unsigned globalOrdinal_i = globalIndexToOrdinal.find(globalIndex_i)->second;
    globalData[globalOrdinal_i] += result1[i];
  }
}

const vector<GlobalIndexType> & SubBasisDofMatrixMapper::mappedGlobalDofOrdinals()
{
  return _mappedGlobalDofOrdinals;
}

set<GlobalIndexType> SubBasisDofMatrixMapper::mappedGlobalDofOrdinalsForBasisOrdinals(set<int> &basisDofOrdinals)
{
  int i=0;
  set<GlobalIndexType> globalIndices;
  for (int myBasisDofOrdinal : _basisDofOrdinalFilter)
  {
    if (basisDofOrdinals.find(myBasisDofOrdinal) != basisDofOrdinals.end())
    {
      // then examine the constraint matrix entries corresponding to the dof ordinal; nonzero entries should be recorded
      double tol = 1e-15;
      for (int j=0; j<_constraintMatrix.dimension(1); j++)
      {
        if (abs(_constraintMatrix(i,j)) > tol)
        {
          globalIndices.insert(_mappedGlobalDofOrdinals[j]);
        }
      }
    }
    i++;
  }
  return globalIndices;
}

SubBasisDofMapperPtr SubBasisDofMatrixMapper::negatedDofMapper()
{
  FieldContainer<double> negatedConstraintMatrix = _constraintMatrix;
  SerialDenseWrapper::multiplyFCByWeight(negatedConstraintMatrix, -1);
  return Teuchos::rcp( new SubBasisDofMatrixMapper(_basisDofOrdinalFilter, _mappedGlobalDofOrdinals, negatedConstraintMatrix) );
}

SubBasisDofMapperPtr SubBasisDofMatrixMapper::restrictDofOrdinalFilter(const set<int> &newDofOrdinalFilter)
{
  set<int> restrictedDofOrdinalFilter;
  vector<int> rowsToKeep;
  set<int> colsToKeep;
  int i = 0;
  for (int basisDofOrdinal : _basisDofOrdinalFilter)
  {
    if (newDofOrdinalFilter.find(basisDofOrdinal) != newDofOrdinalFilter.end())
    {
      rowsToKeep.push_back(i);
      restrictedDofOrdinalFilter.insert(basisDofOrdinal);
      double tol = 1e-15;
      for (int j=0; j<_constraintMatrix.dimension(1); j++)
      {
        if (abs(_constraintMatrix(i,j)) > tol)
        {
          colsToKeep.insert(j);
        }
      }
    }
    i++;
  }
  FieldContainer<double> restrictedConstraintMatrix(rowsToKeep.size(),colsToKeep.size());
  vector<GlobalIndexType> restrictedMappedGlobalDofOrdinals(colsToKeep.size());
  
  int restricted_j = 0;
  for (int j : colsToKeep)
  {
    restrictedMappedGlobalDofOrdinals[restricted_j++] = _mappedGlobalDofOrdinals[j];
  }
  
  int restricted_i = 0;
  for (int i : rowsToKeep)
  {
    int restricted_j = 0;
    for (int j : colsToKeep)
    {
      restrictedConstraintMatrix(restricted_i, restricted_j) = _constraintMatrix(i,j);
      restricted_j++;
    }
    restricted_i++;
  }
  return SubBasisDofMapper::subBasisDofMapper(restrictedDofOrdinalFilter, restrictedMappedGlobalDofOrdinals, restrictedConstraintMatrix);
//  return Teuchos::rcp(new SubBasisDofMatrixMapper(restrictedDofOrdinalFilter, restrictedMappedGlobalDofOrdinals, restrictedConstraintMatrix));
}

SubBasisDofMapperPtr SubBasisDofMatrixMapper::restrictGlobalDofOrdinals(const set<GlobalIndexType> &newGlobalDofOrdinals) // this dof mapper, restricted to the specified global dof ordinals
{
  // leave unimplemented for now -- first let's debug the other restriction method
  
  cout << "SubBasisDofMatrixMapper::restrictGlobalDofOrdinals() has not yet been implemented.  Return null.\n";
  return Teuchos::null;
}