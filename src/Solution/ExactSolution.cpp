
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
 *  ExactSolution.cpp
 *
 */

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "ExactSolution.h"

#include "Function.h"

double ExactSolution::L2NormOfError(Solution &solution, int trialID, int cubDegree) {
  Teuchos::RCP<Mesh> mesh = solution.mesh();
  double totalErrorSquared = 0.0;
  // check if the trialID is one for which a single-point BC was imposed:
  // (in that case, we need to subtract off the average value of the solution when computing norm
  double solutionLift = 0.0;
//  if (solution.bc()->singlePointBC(trialID)) {
//    solutionLift = -solution.meanValue(trialID);
//    cout << "solutionLift for trialID " << trialID << ": " << solutionLift << endl;
//  }
  
  vector<ElementTypePtr> elemTypes = mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCells = mesh->numElementsOfType(elemTypePtr);
    FieldContainer<double> errorSquaredPerCell(numCells);
    
    int numSides;
    
    if (! solution.mesh()->bilinearForm()->isFluxOrTrace(trialID)) {
      numSides = 1;
    } else {
      numSides = elemTypePtr->cellTopoPtr->getSideCount();
    }

    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      L2NormOfError(errorSquaredPerCell, solution, elemTypePtr,trialID,sideIndex, cubDegree, solutionLift);
      
      for (int i=0; i<numCells; i++) {
        totalErrorSquared += errorSquaredPerCell(i);
      }
    }
  }
  return sqrt(totalErrorSquared);
}

void ExactSolution::L2NormOfError(FieldContainer<double> &errorSquaredPerCell, Solution &solution, ElementTypePtr elemTypePtr, int trialID, int sideIndex, int cubDegree, double solutionLift) {
//  BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh = Teuchos::rcp( (Mesh*) NULL ), bool testVsTest=false, int cubatureDegreeEnrichment = 0)

  DofOrdering dofOrdering = *(elemTypePtr->trialOrderPtr.get());
  BasisPtr basis = dofOrdering.getBasis(trialID,sideIndex);
  
  bool boundaryIntegral = solution.mesh()->bilinearForm()->isFluxOrTrace(trialID);
  
  BasisCachePtr basisCache;
  if (cubDegree <= 0) { // then take the default cub. degree
    basisCache = Teuchos::rcp( new BasisCache( elemTypePtr, solution.mesh() ) );
  } else {
    // we could eliminate the logic below if we just added BasisCache::setCubatureDegree()...
    // (the logic below is just to make the enriched cubature match the requested cubature degree...)
    int maxTrialDegree;
    if (!boundaryIntegral) {
      maxTrialDegree = elemTypePtr->trialOrderPtr->maxBasisDegreeForVolume();
    } else {
      maxTrialDegree = elemTypePtr->trialOrderPtr->maxBasisDegree(); // generally, this will be the trace degree
    }
    int maxTestDegree = elemTypePtr->testOrderPtr->maxBasisDegree();
    int cubDegreeEnrichment = max(cubDegree - (maxTrialDegree + maxTestDegree), 0);
    basisCache = Teuchos::rcp( new BasisCache( elemTypePtr, solution.mesh(), false, cubDegreeEnrichment) );
  }
  
  // much of this code is the same as what's in the volume integration in computeStiffness...
  FieldContainer<double> physicalCellNodes = solution.mesh()->physicalCellNodesGlobal(elemTypePtr);
  vector<GlobalIndexType> cellIDs = solution.mesh()->cellIDsOfTypeGlobal(elemTypePtr);
  basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true);
  
  if (boundaryIntegral) {
    basisCache = basisCache->getSideBasisCache(sideIndex);
  }
  
  FieldContainer<double> weightedMeasure = basisCache->getWeightedMeasures();
  FieldContainer<double> weightedErrorSquared;
  
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numCubPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  Teuchos::Array<int> dimensions;
  dimensions.push_back(numCells);
  dimensions.push_back(numCubPoints);
  
  int basisRank = BasisFactory::getBasisRank(basis);
  if (basisRank==1) {
    dimensions.push_back(spaceDim);
  }
  
  FieldContainer<double> computedValues(dimensions);
  FieldContainer<double> exactValues(dimensions);
  
  if (solutionLift != 0.0) {
    int size = computedValues.size();
    for (int i=0; i<size; i++) {
      computedValues[i] += solutionLift;
    }
  }
  
  solution.solutionValues(computedValues, trialID, basisCache);
  this->solutionValues(exactValues, trialID, basisCache);
  
//  cout << "ExactSolution: exact values:\n" << exactValues;
//  cout << "ExactSolution: computed values:\n" << computedValues;
  
  FieldContainer<double> errorSquared(numCells,numCubPoints);
  
  squaredDifference(errorSquared,computedValues,exactValues);
  
  weightedErrorSquared.resize(numCells,numCubPoints);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++) {
      // following two lines for viewing in the debugger:
      double weight = weightedMeasure(cellIndex,ptIndex);
      double errorSquaredVal = errorSquared(cellIndex,ptIndex);
      weightedErrorSquared(cellIndex,ptIndex) = errorSquared(cellIndex,ptIndex) * weightedMeasure(cellIndex,ptIndex);
    }
  }
  
  // compute the integral
  errorSquaredPerCell.initialize(0.0);
  int numPoints = weightedErrorSquared.dimension(1);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      errorSquaredPerCell(cellIndex) += weightedErrorSquared(cellIndex,ptIndex);
    }
  }
}

bool ExactSolution::functionDefined(int trialID) {
  // not supported by legacy subclasses
  return _exactFunctions.find(trialID) != _exactFunctions.end();
}

void ExactSolution::squaredDifference(FieldContainer<double> &diffSquared, FieldContainer<double> &values1, FieldContainer<double> &values2) {
  // two possibilities for values:
  // (C,P) or (C,P,D)
  // output is (C,P) regardless
  int numCells = diffSquared.dimension(0);
  int numPoints = diffSquared.dimension(1);
  bool vectorValued = (values1.rank() == 3);
  for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
    for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
      double value = 0.0;
      if (vectorValued) {
        int spaceDim = values1.dimension(2);
        for (int i=0; i<spaceDim; i++) {
          value += (values1(cellIndex,ptIndex,i) - values2(cellIndex,ptIndex,i)) * (values1(cellIndex,ptIndex,i) - values2(cellIndex,ptIndex,i));
        }
      } else {
        value = (values1(cellIndex,ptIndex) - values2(cellIndex,ptIndex)) * (values1(cellIndex,ptIndex) - values2(cellIndex,ptIndex));
      }
      diffSquared(cellIndex,ptIndex) = value;
    }
  }
}

void ExactSolution::solutionValues(FieldContainer<double> &values, int trialID,
                                   FieldContainer<double> &physicalPoints) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(spaceDim);
  
//  cout << "ExactSolution: physicalPoints:\n" << physicalPoints;
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      FieldContainer<double> point(pointDimensions,&physicalPoints(cellIndex,ptIndex,0));
      double value = solutionValue(trialID, point);
      values(cellIndex,ptIndex) = value;
    }
  }
}

void ExactSolution::solutionValues(FieldContainer<double> &values, int trialID, BasisCachePtr basisCache) {
  if (_exactFunctions.find(trialID) != _exactFunctions.end() ) {
    _exactFunctions[trialID]->values(values,basisCache);
    return;
  }
  
  // TODO: change ExactSolution::solutionValues (below) to take a *const* points FieldContainer, to avoid this copy:
  FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
  if (basisCache->getSideIndex() >= 0) {
    FieldContainer<double> unitNormals = basisCache->getSideNormals();
    this->solutionValues(values,trialID,points,unitNormals);
  } else {
    this->solutionValues(values,trialID,points);
  }
}

void ExactSolution::solutionValues(FieldContainer<double> &values, 
                                   int trialID,
                                   FieldContainer<double> &physicalPoints,
                                   FieldContainer<double> &unitNormals) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(spaceDim);
  
  //  cout << "ExactSolution: physicalPoints:\n" << physicalPoints;
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      FieldContainer<double> point(pointDimensions,&physicalPoints(cellIndex,ptIndex,0));
      FieldContainer<double> unitNormal(pointDimensions,&unitNormals(cellIndex,ptIndex,0));
      double value = solutionValue(trialID, point, unitNormal);
      values(cellIndex,ptIndex) = value;
    }
  }
}

Teuchos::RCP<BilinearForm> ExactSolution::bilinearForm() {
  return _bilinearForm;
}

Teuchos::RCP<BC> ExactSolution::bc() {
  return _bc;
}
Teuchos::RCP<RHS> ExactSolution::rhs() {
  return _rhs;
}

ExactSolution::ExactSolution() {
  
}

ExactSolution::ExactSolution(Teuchos::RCP<BilinearForm> bf, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, int H1Order) {
  _bilinearForm = bf;
  _bc = bc;
  _rhs = rhs;
  _H1Order = H1Order;
}

double ExactSolution::solutionValue(int trialID, FieldContainer<double> &physicalPoint) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
  return 0;
//  int spaceDim = physicalPoint.size();
//  double x = physicalPoint(0);
//  double y = physicalPoint(1);
//  if (spaceDim == 2) {
//    if (_exactFunctions.find(trialID) != _exactFunctions.end() ) {
//      SimpleFunctionPtr fxn = _exactFunctions.find(trialID)->second;
//      return fxn->value(x,y);
//    } else {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No function set for trialID.");
//    }
//  } else {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ExactSolution / SimpleFunction doesn't yet support spaceDim != 2");
//  }
}

double ExactSolution::solutionValue(int trialID, FieldContainer<double> &physicalPoint,
                                    FieldContainer<double> &unitNormal) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
  return 0;
//  int spaceDim = physicalPoint.size();
//  double x = physicalPoint(0);
//  double y = physicalPoint(1);
//  double n1 = unitNormal(0);
//  double n2 = unitNormal(1);
//  if (spaceDim == 2) {
//    if (_exactNormalFunctions.find(trialID) != _exactNormalFunctions.end() ) {
//      ScalarFunctionOfNormalPtr fxn = _exactNormalFunctions.find(trialID)->second;
//      return fxn->value(x,y,n1,n2);
//    } else if (_exactFunctions.find(trialID) != _exactFunctions.end() ) {
//      SimpleFunctionPtr fxn = _exactFunctions.find(trialID)->second;
//      return fxn->value(x,y);
//    } else {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No function set for trialID.");
//    }
//  } else {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ExactSolution / SimpleFunction doesn't yet support spaceDim != 2");
//  }
}

int ExactSolution::H1Order() { // return -1 for non-polynomial solutions
  return _H1Order;
}

const map< int, FunctionPtr > ExactSolution::exactFunctions() {
  return _exactFunctions;
}

void ExactSolution::setSolutionFunction( VarPtr var, FunctionPtr varFunction ) {
  _exactFunctions[var->ID()] = varFunction;
}