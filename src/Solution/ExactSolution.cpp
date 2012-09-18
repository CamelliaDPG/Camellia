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


#include "ExactSolution.h"

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
  // much of this code is the same as what's in the volume integration in computeStiffness...
  FieldContainer<double> physicalCellNodes = solution.mesh()->physicalCellNodesGlobal(elemTypePtr);
  
  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
  
  // Check that cellTopo and physicalCellNodes agree
  TEUCHOS_TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
                     std::invalid_argument,
                     "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
                     std::invalid_argument,
                     "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");
  
  typedef CellTools<double>  CellTools;
  typedef FunctionSpaceTools fst;
  
  DofOrdering dofOrdering = *(elemTypePtr->trialOrderPtr.get());
  
  // Get numerical integration points and weights
  DefaultCubatureFactory<double>  cubFactory;     
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = dofOrdering.getBasis(trialID,sideIndex);
  int basisRank = dofOrdering.getBasisRank(trialID);
  if (cubDegree <= 0) {
    cubDegree = 2*basis->getDegree();
  }
  
  FieldContainer<double> weightedMeasure;
  FieldContainer<double> weightedErrorSquared;
  FieldContainer<double> physCubPoints;
  FieldContainer<double> cubPointsSide;
  FieldContainer<double> sideNormals;
  int numCubPoints;
  
  bool boundaryIntegral = solution.mesh()->bilinearForm()->isFluxOrTrace(trialID);
  if ( !boundaryIntegral ) {
    if (sideIndex != 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(sideIndex != 0,std::invalid_argument,
                         "For field variables, sideIndex argument should always be 0.")
    }
    // volume integral
    Teuchos::RCP<Cubature<double> > cellTopoCub = cubFactory.create(cellTopo, cubDegree); 
    
    int cubDim       = cellTopoCub->getDimension();
    numCubPoints = cellTopoCub->getNumPoints();
    
    FieldContainer<double> cubPoints(numCubPoints, cubDim);
    FieldContainer<double> cubWeights(numCubPoints);
    
    cellTopoCub->getCubature(cubPoints, cubWeights);
    
    // 1. Determine Jacobians
    // Compute cell Jacobians and their determinants (for measure)
    // Containers for Jacobian
    FieldContainer<double> cellJacobian(numCells, numCubPoints, spaceDim, spaceDim);
    FieldContainer<double> cellJacobDet(numCells, numCubPoints);
    
    CellTools::setJacobian(cellJacobian, cubPoints, physicalCellNodes, cellTopo);
    CellTools::setJacobianDet(cellJacobDet, cellJacobian );
    
    // compute weighted measure
    weightedMeasure.resize(numCells, numCubPoints);
    fst::computeCellMeasure<double>(weightedMeasure, cellJacobDet, cubWeights);
    
    // compute physicalCubaturePoints, the transformed cubature points on each cell:
    physCubPoints.resize(numCells, numCubPoints, spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints,cubPoints,physicalCellNodes,cellTopo);
    
    //fst::multiplyMeasure<double>(weightedErrorSquared, weightedMeasure, errorSquared);
  } else {
    // boundary integral
    shards::CellTopology side(cellTopo.getCellTopologyData(spaceDim-1,sideIndex)); // create relevant subcell (side) topology
    int sideDim = side.getDimension();                              
    Teuchos::RCP<Cubature<double> > sideCub = cubFactory.create(side, cubDegree);
    numCubPoints = sideCub->getNumPoints();
    cubPointsSide.resize(numCubPoints, sideDim); // cubature points from the pov of the side (i.e. a 1D set)
    FieldContainer<double> cubWeightsSide(numCubPoints);
    FieldContainer<double> cubPointsSideRefCell(numCubPoints, spaceDim); // cubPointsSide from the pov of the ref cell
    FieldContainer<double> jacobianSideRefCell(numCells, numCubPoints, spaceDim, spaceDim);
    
    sideCub->getCubature(cubPointsSide, cubWeightsSide);
    
    // compute geometric cell information
    //cout << "computing geometric cell info for boundary integral." << endl;
    CellTools::mapToReferenceSubcell(cubPointsSideRefCell, cubPointsSide, sideDim, (int)sideIndex, cellTopo);
    CellTools::setJacobian(jacobianSideRefCell, cubPointsSideRefCell, physicalCellNodes, cellTopo);
    
    // map side cubature points in reference parent cell domain to physical space
    physCubPoints.resize(numCells, numCubPoints, spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints, cubPointsSideRefCell, physicalCellNodes, cellTopo);
    
    // compute weighted edge measure
    weightedMeasure.resize(numCells, numCubPoints);
    FunctionSpaceTools::computeEdgeMeasure<double>(weightedMeasure, jacobianSideRefCell,
                                                   cubWeightsSide, sideIndex, cellTopo);
    
    sideNormals.resize(numCells, numCubPoints, spaceDim);
    FieldContainer<double> normalLengths(numCells, numCubPoints);
    CellTools::getPhysicalSideNormals(sideNormals, jacobianSideRefCell, sideIndex, cellTopo);
    
    // make unit length
    RealSpaceTools<double>::vectorNorm(normalLengths, sideNormals, NORM_TWO);
    FunctionSpaceTools::scalarMultiplyDataData<double>(sideNormals, normalLengths, sideNormals, true);
  }
  Teuchos::Array<int> dimensions;
  dimensions.push_back(numCells);
  dimensions.push_back(numCubPoints);
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
  
  if ( ! boundaryIntegral) {
    solution.solutionValues(computedValues, elemTypePtr, trialID, physCubPoints);
    this->solutionValues(exactValues,trialID, physCubPoints);
  } else {
    solution.solutionValues(computedValues, elemTypePtr, trialID, physCubPoints, cubPointsSide, sideIndex);
    this->solutionValues(exactValues,trialID, physCubPoints, sideNormals);
  }
  
//  cout << "ExactSolution: exact values:\n" << exactValues;
//  cout << "ExactSolution: computed values:\n" << computedValues;
  
  FieldContainer<double> errorSquared(dimensions);
  
  squaredDifference(errorSquared,computedValues,exactValues);
  
  weightedErrorSquared.resize(dimensions);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++) {
      if (basisRank==0) {
        // following two lines for viewing in the debugger:
        double weight = weightedMeasure(cellIndex,ptIndex);
        double errorSquaredVal = errorSquared(cellIndex,ptIndex);
        weightedErrorSquared(cellIndex,ptIndex) = errorSquared(cellIndex,ptIndex) * weightedMeasure(cellIndex,ptIndex);
      } else {
        for (int i=0; i<spaceDim; i++){
          weightedErrorSquared(cellIndex,ptIndex,i) = errorSquared(cellIndex,ptIndex,i) * weightedMeasure(cellIndex,ptIndex);            
        }
      }
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