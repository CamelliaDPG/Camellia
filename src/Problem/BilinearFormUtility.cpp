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

#include "BasisFactory.h"
#include "IP.h"

// Intrepid includes
#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Teuchos_LAPACK.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

#include "BilinearFormUtility.h"
#include "BasisCache.h"

#include "Solution.h"

#include "CamelliaCellTools.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
bool BilinearFormUtility<Scalar>::_warnAboutZeroRowsAndColumns = true;

template <typename Scalar>
void BilinearFormUtility<Scalar>::setWarnAboutZeroRowsAndColumns( bool value )
{
  _warnAboutZeroRowsAndColumns = value;
}

template <typename Scalar>
bool BilinearFormUtility<Scalar>::warnAboutZeroRowsAndColumns()
{
  return _warnAboutZeroRowsAndColumns;
}

template <typename Scalar>
bool BilinearFormUtility<Scalar>::checkForZeroRowsAndColumns(string name, FieldContainer<Scalar> &array, bool checkRows, bool checkCols)
{
  // for now, only support rank 3 FCs
  double tol = 1e-15;
  static int warningsIssued = 0; // max of 20
  if ( array.rank() != 3)
  {
    TEUCHOS_TEST_FOR_EXCEPTION( array.rank() != 3, std::invalid_argument, "checkForZeroRowsAndColumns only supports rank-3 FieldContainers.");
  }
  int numCells = array.dimension(0);
  int numRows = array.dimension(1);
  int numCols = array.dimension(2);
  bool zeroRowOrColFound = false;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    if (checkRows)
    {
      for (int i=0; i<numRows; i++)
      {
        bool nonZeroFound = false;
        int j=0;
        while ((!nonZeroFound) && (j<numCols))
        {
          if (abs(array(cellIndex,i,j)) > tol) nonZeroFound = true;
          j++;
        }
        if ( ! nonZeroFound )
        {
          if (_warnAboutZeroRowsAndColumns)
          {
            warningsIssued++;
            cout << "warning: in matrix " << name << " for cell " << cellIndex << ", row " << i << " is all zeros." << endl;

            if ( (warningsIssued == 20) && _warnAboutZeroRowsAndColumns )
            {
              cout << "20 warnings issued.  Suppressing future warnings about zero rows and columns\n";
              _warnAboutZeroRowsAndColumns = false;
            }
          }
          zeroRowOrColFound = true;
        }
      }
    }
    if (checkCols)
    {
      for (int j=0; j<numCols; j++)
      {
        bool nonZeroFound = false;
        int i=0;
        while ((!nonZeroFound) && (i<numRows))
        {
          if (abs(array(cellIndex,i,j)) > tol) nonZeroFound = true;
          i++;
        }
        if ( ! nonZeroFound )
        {
          if (_warnAboutZeroRowsAndColumns)
          {
            warningsIssued++;
            cout << "warning: in matrix " << name << " for cell " << cellIndex << ", column " << j << " is all zeros." << endl;

            if ( (warningsIssued == 20) && _warnAboutZeroRowsAndColumns )
            {
              cout << "20 warnings issued.  Suppressing future warnings about zero rows and columns\n";
              _warnAboutZeroRowsAndColumns = false;
            }
          }
          zeroRowOrColFound = true;
        }
      }
    }
  }
  return !zeroRowOrColFound; // return TRUE if no zero row or col found
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::transposeFCMatrices(FieldContainer<Scalar> &fcTranspose,
    const FieldContainer<Scalar> &fc)
{
  // check dimensions
  TEUCHOS_TEST_FOR_EXCEPTION( ( fc.dimension(0) != fcTranspose.dimension(0) ),
                              std::invalid_argument,
                              "fc.dimension(0) and fcTranspose.dimension(0) (numCells) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( fc.dimension(1) != fcTranspose.dimension(2) ),
                              std::invalid_argument,
                              "fc.dimension(1) and fcTranspose.dimension(2) (numRows) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( fc.dimension(2) != fcTranspose.dimension(1) ),
                              std::invalid_argument,
                              "fc.dimension(2) and fcTranspose.dimension(1) (numCols) do not match.");
  // transposes (C,i,j) --> (C,j,i)
  int numCells = fc.dimension(0);
  int numRows = fc.dimension(1);
  int numCols = fc.dimension(2);
  for (int cellIndex=0; cellIndex < numCells; cellIndex++)
  {
    for (int i=0; i < numRows; i++)
    {
      for (int j=0; j < numCols; j++)
      {
        fcTranspose(cellIndex,j,i) = fc(cellIndex,i,j);
      }
    }
  }
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::computeStiffnessMatrix(FieldContainer<Scalar> &stiffness,
    FieldContainer<Scalar> &innerProductMatrix,
    FieldContainer<Scalar> &optimalTestWeights)
{
  // stiffness has dimensions (numCells, numTrialDofs, numTrialDofs)
  // innerProductMatrix has dim. (numCells, numTestDofs, numTestDofs)
  // optimalTestWeights has dim. (numCells, numTrialDofs, numTestDofs)
  // all this does is computes stiffness = weights^T * innerProductMatrix * weights
  int numCells = stiffness.dimension(0);
  int numTrialDofs = stiffness.dimension(1);
  int numTestDofs = innerProductMatrix.dimension(1);

  // check that all the dimensions are compatible:
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(0) != numCells ),
                              std::invalid_argument,
                              "stiffness.dimension(0) and optimalTestWeights.dimension(0) (numCells) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != numTrialDofs ),
                              std::invalid_argument,
                              "numTrialDofs and optimalTestWeights.dimension(1) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(2) != numTestDofs ),
                              std::invalid_argument,
                              "numTestDofs and optimalTestWeights.dimension(2) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( innerProductMatrix.dimension(2) != innerProductMatrix.dimension(1) ),
                              std::invalid_argument,
                              "innerProductMatrix.dimension(1) and innerProductMatrix.dimension(2) do not match.");

  TEUCHOS_TEST_FOR_EXCEPTION( ( stiffness.dimension(1) != stiffness.dimension(2) ),
                              std::invalid_argument,
                              "stiffness.dimension(1) and stiffness.dimension(2) do not match.");

  stiffness.initialize(0);

  for (int cellIndex=0; cellIndex < numCells; cellIndex++)
  {
    Epetra_SerialDenseMatrix weightsT(::Copy,
                                      &optimalTestWeights(cellIndex,0,0),
                                      optimalTestWeights.dimension(2), // stride
                                      optimalTestWeights.dimension(2),optimalTestWeights.dimension(1));

    Epetra_SerialDenseMatrix ipMatrixT(::Copy,
                                       &innerProductMatrix(cellIndex,0,0),
                                       innerProductMatrix.dimension(2), // stride
                                       innerProductMatrix.dimension(2),innerProductMatrix.dimension(1));

    Epetra_SerialDenseMatrix   stiffT (::View,
                                       &stiffness(cellIndex,0,0),
                                       stiffness.dimension(2), // stride
                                       stiffness.dimension(2),stiffness.dimension(1));

    Epetra_SerialDenseMatrix intermediate( numTrialDofs, numTestDofs );

    // account for the fact that SDM is column-major and FC is row-major:
    //   (weightsT) * (ipMatrixT)^T * (weightsT)^T
    int success = intermediate.Multiply('T','T',1.0,weightsT,ipMatrixT,0.0);

    if (success != 0)
    {
      cout << "computeStiffnessMatrix: intermediate.Multiply() failed with error code " << success << endl;
    }

    success = stiffT.Multiply('N','N',1.0,intermediate,weightsT,0.0);
    // stiffT is technically the transpose of stiffness, but the construction A^T * B * A is symmetric even in general...

    if (success != 0)
    {
      cout << "computeStiffnessMatrix: stiffT.Multiply() failed with error code " << success << endl;
    }
  }

  if ( ! checkForZeroRowsAndColumns("stiffness",stiffness) )
  {
    //cout << "stiffness: " << stiffness;
  }

  bool enforceNumericalSymmetry = false;
  if (enforceNumericalSymmetry)
  {
    for (unsigned int c=0; c < numCells; c++)
      for (unsigned int i=0; i < numTrialDofs; i++)
        for (unsigned int j=i+1; j < numTrialDofs; j++)
        {
          stiffness(c,i,j) = (stiffness(c,i,j) + stiffness(c,j,i)) / 2.0;
          stiffness(c,j,i) = stiffness(c,i,j);
        }
  }
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::computeStiffnessMatrixForCell(FieldContainer<Scalar> &stiffness, Teuchos::RCP<Mesh> mesh, int cellID)
{
  DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
  DofOrderingPtr testOrder  = mesh->getElementType(cellID)->testOrderPtr;
  CellTopoPtr     cellTopo  = mesh->getElementType(cellID)->cellTopoPtr;
  FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
  FieldContainer<double> cellSideParities  = mesh->cellSideParitiesForCell(cellID);
  int numCells = 1;
  stiffness.resize(numCells,testOrder->totalDofs(),trialOrder->totalDofs());
  computeStiffnessMatrix(stiffness,mesh->bilinearForm(),trialOrder,testOrder,cellTopo,physicalCellNodes,cellSideParities);
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::computeStiffnessMatrix(FieldContainer<Scalar> &stiffness, TBFPtr<Scalar> bilinearForm,
    Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
    CellTopoPtr cellTopo, FieldContainer<double> &physicalCellNodes,
    FieldContainer<double> &cellSideParities)
{
  // this method is deprecated--here basically until we can revise tests, etc. to use the BasisCache version

  // physicalCellNodes: the nodal points for the element(s) with topology cellTopo
  //                 The dimensions are (numCells, numNodesPerElement, spaceDimension)
  DefaultCubatureFactory<double>  cubFactory;

  int maxTestDegree = testOrdering->maxBasisDegree();

  bool createSideCachesToo = true;
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(physicalCellNodes, cellTopo, *trialOrdering, maxTestDegree, createSideCachesToo));

  bilinearForm->stiffnessMatrix(stiffness,trialOrdering,testOrdering,cellSideParities,basisCache);
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::computeRHS(FieldContainer<Scalar> &rhsVector,
    TBFPtr<Scalar> bilinearForm, RHS &rhs,
    FieldContainer<Scalar> &optimalTestWeights,
    Teuchos::RCP<DofOrdering> testOrdering,
    shards::CellTopology &cellTopo,
    FieldContainer<double> &physicalCellNodes)
{
  // this method is deprecated--here basically until we can revise tests, etc. to use the BasisCache version
  // Get numerical integration points and weights

  // physicalCellNodes: the nodal points for the element(s) with topology cellTopo
  //                 The dimensions are (numCells, numNodesPerElement, spaceDimension)
  DefaultCubatureFactory<double>  cubFactory;

  int cubDegreeTest = testOrdering->maxBasisDegree();
  int cubDegree = 2*cubDegreeTest;

  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(physicalCellNodes, cellTopo, cubDegree)); // DON'T create side caches, too

//  computeRHS(rhsVector,bilinearForm,rhs,optimalTestWeights,testOrdering,basisCache);
  rhs.integrateAgainstOptimalTests(rhsVector,optimalTestWeights,testOrdering,basisCache);
}

template <typename Scalar>
void BilinearFormUtility<Scalar>::weightCellBasisValues(FieldContainer<double> &basisValues, const FieldContainer<double> &weights, int offset)
{
  // weights are (numCells, offset+numFields)
  // basisValues are (numCells, numFields, ...)
  int numCells = basisValues.dimension(0);
  int numFields = basisValues.dimension(1);

  Teuchos::Array<int> dimensions;
  basisValues.dimensions(dimensions);

  int numAffectedValues = 1;
  for (int dimIndex=2; dimIndex<dimensions.size(); dimIndex++)
  {
    numAffectedValues *= dimensions[dimIndex];
  }

  Teuchos::Array<int> index(dimensions.size(),0);

  for (int cellIndex=0; cellIndex < numCells; cellIndex++)
  {
    index[0] = cellIndex;
    for (int fieldIndex=0; fieldIndex < numFields; fieldIndex++)
    {
      index[1] = fieldIndex;
      int enumIndex = basisValues.getEnumeration(index);
      for (int valIndex=enumIndex; valIndex < numAffectedValues + enumIndex; valIndex++)
      {
        basisValues[valIndex] *= weights(cellIndex,fieldIndex+offset);
      }
    }
  }
}
namespace Camellia
{
template class BilinearFormUtility<double>;
}
