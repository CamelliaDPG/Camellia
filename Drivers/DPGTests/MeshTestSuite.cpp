/*
 *  MeshTestSuite.cpp
 *
 */

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

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "PoissonExactSolution.h"
#include "PoissonBilinearForm.h"
#include "PoissonExactSolutionLinear.h"
#include "PoissonExactSolutionQuadratic.h"
#include "PoissonExactSolutionCubic.h"
#include "PoissonExactSolutionQuartic.h"
#include "ElementTypeFactory.h"

#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"

#include "MeshTestSuite.h"
#include "MeshTestUtility.h"

#include "Mesh.h"
#include "PoissonRHSLinear.h"
#include "PoissonBCLinear.h"
#include "MathInnerProduct.h"
#include "Solution.h"
#include "Element.h"
#include "ElementType.h"
#include "BasisFactory.h"

#include <sstream>

using namespace Intrepid;

typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;

void MeshTestSuite::runTests(int &numTestsRun, int &numTestsPassed) {
  numTestsRun++;
  if (testMeshSolvePointwise() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testPointContainment() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testHRefinementForConfusion() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testRefinementPattern() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testHRefinement() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testFluxIntegration() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testFluxNorm() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if ( testSinglePointBC() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if ( testSolutionForSingleElementUpgradedSide() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if ( testSolutionForMultipleElementTypes() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if ( testPRefinement() ) {
    numTestsPassed++;
  }  
  numTestsRun++;
  if (testDofOrderingFactory() ) {
    numTestsPassed++;
  }  
  numTestsRun++;
  if (testBasisRefinement() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testPoissonConvergence() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testSacadoExactSolution() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testExactSolution(true) ) { // check L2 error computation
    numTestsPassed++;
  }
  numTestsRun++;
  if (testExactSolution(false) ) { // check actual solution
    numTestsPassed++;
  }
  numTestsRun++;
  if (testBuildMesh() ) {
    numTestsPassed++;
  }

}

bool MeshTestSuite::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, const FieldContainer<double> &testPointsRefCoords,
                                              bool reportErrors) {
  // iterates through all active elements, and checks that each edge agrees with its neighbor basis.
  // first pass is just for PatchBasis, but intent is to extend to apply to MultiBasis, too.
  
  bool success = true;
  
  double tol = 2e-15; // small tolerance
  double maxDiff = 0.0;
    
  int numPoints = testPointsRefCoords.dimension(0);
  
  // in 2D, the neighbor's view of the test points will be simply flipped relative to its peer
  // (which might be the ancestor of the element that we test below)
  FieldContainer<double> neighborTestPointsRefCoords(testPointsRefCoords); // alloc the right size container
  FieldContainer<double> ancestorTestPointsRefCoords(testPointsRefCoords); // alloc the right size container
  
  int numElements = mesh->activeElements().size();
  vector<int> fluxIDs = mesh->bilinearForm()->trialBoundaryIDs();
  for (int cellIndex=0; cellIndex<numElements; cellIndex++) {
    Teuchos::RCP<Element> elem = mesh->activeElements()[cellIndex];
    DofOrderingPtr trialOrder = elem->elementType()->trialOrderPtr;
    int cellID = elem->cellID();
    int numSides = elem->numSides();
    
    for (int sideIndex=0; sideIndex < numSides; sideIndex++) {
      int neighborSideIndex;
      Teuchos::RCP<Element> neighbor = mesh->ancestralNeighborForSide(elem,sideIndex,neighborSideIndex);
      if ( (neighbor->cellID() == -1) || neighbor->isParent() ) { // boundary or broken neighbor
        // if broken neighbor, we'll handle this element when we handle neighbor's active descendants.
        continue;
      }
      DofOrderingPtr neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
      int ancestorCellID = neighbor->getNeighborCellID(neighborSideIndex);
      int subSideIndexInNeighbor = -1;
      // because of the above, we can assume, below, that elem is the small guy if the two aren't peers.
      ancestorTestPointsRefCoords = testPointsRefCoords;
      int ancestorSideIndex = sideIndex; // if necessary, we'll walk up the family tree to the real ancestor, getting the right sideIndices along the way...
      if ( ancestorCellID != cellID ) { // i.e. the two are *not* peers: elem small, neighbor big 
        // then we need to map the coords into ancestor's ref space
        // simplest (though not most efficient!) to do this by iteratively mapping into parent's space
        Teuchos::RCP<Element> currentElement = mesh->getElement(cellID);
        while (currentElement->cellID() != ancestorCellID) {
          FieldContainer<double> oldAncestorTestPointsRefCoords = ancestorTestPointsRefCoords;
          currentElement->getSidePointsInParentRefCoords(ancestorTestPointsRefCoords, ancestorSideIndex, 
                                                         oldAncestorTestPointsRefCoords);
          ancestorSideIndex = currentElement->parentSideForSideIndex(ancestorSideIndex);
          currentElement = mesh->getElement(currentElement->getParent()->cellID());
        }
        vector< pair<int,int> > descendantsForSide = mesh->getElement(ancestorCellID)->getDescendantsForSide(ancestorSideIndex);
        int subSideIndexInAncestor = 0;
        for (vector< pair<int,int> >::iterator descIt = descendantsForSide.begin(); descIt != descendantsForSide.end(); 
             descIt++, subSideIndexInAncestor++) {
          if (descIt->first == cellID) {
            subSideIndexInNeighbor = mesh->neighborChildPermutation(subSideIndexInAncestor, descendantsForSide.size());
          }
        }
      }
      
      mesh->getElement(ancestorCellID)->getSidePointsInNeighborRefCoords(neighborTestPointsRefCoords, ancestorSideIndex,
                                                                         ancestorTestPointsRefCoords);
      
      for (vector<int>::iterator fluxIt = fluxIDs.begin(); fluxIt != fluxIDs.end(); fluxIt++) {
        int fluxID = *fluxIt;
        BasisPtr basis = trialOrder->getBasis(fluxID,sideIndex);
        BasisPtr neighborBasis = neighborTrialOrder->getBasis(fluxID,neighborSideIndex);
        FieldContainer<double> values(basis->getCardinality(),numPoints);
        FieldContainer<double> neighborValues(neighborBasis->getCardinality(),numPoints);
        basis->getValues(values,testPointsRefCoords,Intrepid::OPERATOR_VALUE);
        neighborBasis->getValues(neighborValues,neighborTestPointsRefCoords,Intrepid::OPERATOR_VALUE);
        bool failedHere = false;
        for (int dofOrdinal=0; dofOrdinal < basis->getCardinality(); dofOrdinal++) {
          int neighborDofOrdinal = mesh->neighborDofPermutation(dofOrdinal,basis->getCardinality());
          if (BasisFactory::isMultiBasis(neighborBasis)) {
            neighborDofOrdinal = ((MultiBasis*) neighborBasis.get())->relativeToAbsoluteDofOrdinal(neighborDofOrdinal,subSideIndexInNeighbor);
          }
          for (int pointIndex = 0; pointIndex < numPoints; pointIndex++) {
            double diff = abs(neighborValues(neighborDofOrdinal,pointIndex) - values(dofOrdinal,pointIndex));
            if (diff > tol) {
              success = false;
              failedHere = true;
              if (reportErrors) {
                cout << "values for point " << pointIndex << " and my dofOrdinal " << dofOrdinal;
                cout << " differ from those for neighbor dofOrdinal " << neighborDofOrdinal;
                cout << ": " << values(dofOrdinal,pointIndex) << " != " << neighborValues(neighborDofOrdinal,pointIndex) << endl;
              }
            }
            maxDiff = max(diff,maxDiff);
          }
        }
        if (failedHere && reportErrors) {
          cout << "cellID " << cellID << "'s testPoints:\n" << testPointsRefCoords;
          
          cout << "neighbor cellID " << neighbor->cellID() << "'s testPoints:\n" << neighborTestPointsRefCoords;
          // for debugging, some console output:
          cout << "values for cellID " << cellID << ", fluxID " << fluxID << ", side " << sideIndex << ":\n";
          cout << values;
          
          cout << "values for neighbor cellID " << neighbor->cellID() << ", fluxID " << fluxID << ", side " << neighborSideIndex << ":\n";
          cout << neighborValues;
          cout << "**** neighborBasesAgreeOnSides suppressing further output re. disagreement ****\n\n";
          reportErrors = false;
        }
      }
    }
  }
  
  if ( ! success ) {
    cout << "neighboring bases disagree on point values; maxDiff: " << maxDiff << endl;
  }
  return success;
}

bool MeshTestSuite::testBasisRefinement() {
  int basisRank;
  int initialPolyOrder = 3;
  
  bool success = true;
  
  EFunctionSpaceExtended hgrad = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  Teuchos::RCP<Basis<double,FieldContainer<double> > > basis = BasisFactory::getBasis(basisRank, initialPolyOrder, quad_4.getKey(), hgrad);
  if (basis->getDegree() != initialPolyOrder) {  // since it's hgrad, that's a problem (hvol would be initialPolyOrder-1)
    success = false;
    cout << "testBasisRefinement: initial BasisFactory call returned a different-degree basis than expected..." << endl;
    cout << "testBasisRefinement: expected: " << initialPolyOrder << "; actual: " << basis->getDegree() << endl;
  }
  int additionalP = 4;
  basis = BasisFactory::addToPolyOrder(basis, additionalP);
  if (basis->getDegree() != initialPolyOrder+additionalP) {  // since it's hgrad, that's a problem (hvol would be initialPolyOrder-1)
    success = false;
    cout << "testBasisRefinement: addToPolyOrder call returned a different-degree basis than expected..." << endl;
    cout << "testBasisRefinement: expected: " << initialPolyOrder+additionalP << "; actual: " << basis->getDegree() << endl;    
  }
  return success;
}

bool MeshTestSuite::testFluxIntegration() {
  double tol = 2e-12; // had to increase for triangles
  bool success = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int pToAdd = 1;
  int horizontalCells=1,verticalCells=1;
  bool triangulate = false;
  vector<int> polyOrders;
  vector<double> expectedValues;
  //polyOrders.push_back(0);
  // for p=0, phi=1, so the integral around the unit square is its perimeter (4)
  //expectedValues.push_back(4.0);
  polyOrders.push_back(1);
  // for p=1, phi=x+2y-1.5, and the integral around the perimeter is 0
  expectedValues.push_back(0.0);
  
  for (int i=0; i<expectedValues.size(); i++) {
    PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, polyOrders[i]);
    exactSolution.setUseSinglePointBCForPHI(true);
    int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                      exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
    // Poisson is set up such that the solution should be (x + 2y)^p
    
    solution.solve();
    
    FieldContainer<double> integral(myMesh->numElements());
    
    solution.integrateFlux(integral,PoissonBilinearForm::PHI_HAT);
    
    double diff = abs(expectedValues[i] - integral(0));
    
    if (diff > tol) {
      success = false;
      cout << "Failure: Integral of phi_hat solution of Poisson was " << integral(0) << "; expected " << expectedValues[i] << endl;
    }
  }
  return success;
}

bool MeshTestSuite::testFluxNorm() {
  double tol = 2e-12; // had to increase for triangles
  bool success = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int pToAdd = 1;
  int horizontalCells=1,verticalCells=1;
  bool triangulate = false;
  vector<int> polyOrders;
  vector<double> expectedValues;
  //polyOrders.push_back(0);
  // for p=0, phi=1, so the integral around the unit square is its perimeter (4)
  //expectedValues.push_back(4.0);
  polyOrders.push_back(1);
  int p = 1;
  // for p=1, phi=x+2y-1.5, and the L2 norm around the perimeter is sqrt(10/3)
  expectedValues.push_back(sqrt(10.0/3.0));
  
  for (int i=0; i<expectedValues.size(); i++) {
    PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, polyOrders[i]);
    exactSolution.setUseSinglePointBCForPHI(true);
    int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                      exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
      
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
    // Poisson is set up such that the solution should be x + 2y
    
    double L2normFlux = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI_HAT);
    //cout << "L2 norm of phi_hat " << L2normFlux << endl;
    
    double normDiff = abs(expectedValues[i] - L2normFlux);
    
    if (normDiff > tol) {
      success = false;
      cout << "Failure: Norm of phi_hat solution of Poisson was " << L2normFlux << "; expected " << expectedValues[i] << endl;
    }
    solution.solve();

    double fluxDiff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI_HAT);
    fluxDiff = fluxDiff / L2normFlux;
    
    //cout << "Relative L2 Error in phi_hat solution of Poisson: " << fluxDiff << endl;
    if (fluxDiff > tol) {
      success = false;
      cout << "Failure: Error in phi_hat solution of Poisson was " << fluxDiff << "; tolerance set to " << tol << endl;
    }
  }
  return success;
}

bool MeshTestSuite::testSacadoExactSolution() {
  double tol = 1.6e-9; // had to increase for triangles, and again for single-point imposition.
  bool success = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  for (int cellTopoIndex=0; cellTopoIndex<2; cellTopoIndex++) {
    bool triangulate = (cellTopoIndex == 1);
    int pToAdd = 1;
    for (int i=1; i<6; i++) {
      PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, i);
      exactSolution.setUseSinglePointBCForPHI(true); // otherwise, we'd be using the zero-mean condition, and the solution doesn't have zero mean on this domain...
      int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
      Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, 2, 2, exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
      
      Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
      
      Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
      // Poisson is set up such that the solution should be x + 2y
      
      double diff;

      double L2norm = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI);
      double L2normFlux = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI_HAT);
      
      solution.solve();
      diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI);
      //cout << "L2 Error in solution of phi Poisson (Sacado version): " << diff << endl;
      diff = diff / L2norm;
      
      //cout << "Relative L2 Error in phi solution of Poisson (Sacado version): " << diff << endl;
      if (diff > tol) {
        success = false;
        cout << "Failure: Error in phi solution of Poisson (Sacado version) was " << diff << "; tolerance set to " << tol << endl;
      }
      //cout << "L2 norm of phi_hat " << L2normFlux << endl;
      double fluxDiff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI_HAT);
      fluxDiff = fluxDiff / L2normFlux;
      
      //cout << "Relative L2 Error in phi_hat solution of Poisson (Sacado version): " << fluxDiff << endl;
      if (fluxDiff > tol) {
        success = false;
        cout << "Failure: Error in phi_hat solution of Poisson (Sacado version) was " << fluxDiff << "; tolerance set to " << tol << endl;
      }
    }
  }
  return success;
  
}

bool MeshTestSuite::testPoissonConvergence() {
  bool success = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
/*  int sqrtElements = 1;
  for (int i=1; i<3; i++) {
    int order = 3;
    
    PoissonExactSolution exactSolution(PoissonExactSolution::EXPONENTIAL);
    Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), order, order+1);
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  
    double L2norm = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
    double diff;
    
    solution.solve();
    diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15);
    cout << "POISSON EXPONENTIAL (p=" << order-1 << "): L2 Error in solution for " << sqrtElements << "x" << sqrtElements << " mesh: " << diff << endl;
    diff = diff / L2norm;
    
    cout << "Relative L2 Error: " << diff << endl;
    //if (diff > tol) {
    //  success = false;
    //  cout << "Failure: Error in solution of Poisson (Sacado version) was " << diff << "; tolerance set to " << tol << endl;
    //}
    ostringstream fileName;
    fileName << "PoissonPhiSolution.p=" << order-1 << "." << sqrtElements << "x" << sqrtElements << ".dat";
    solution.writeToFile(PoissonBilinearForm::PHI, fileName.str());
    sqrtElements *= 2;
  }*/
 
 /*
  // p-convergence
  int sqrtElements = 8;
  for (int i=1; i<6; i++) {
    int order = i+1; // so that the field variables are of order i
    
    PoissonExactSolution exactSolution(PoissonExactSolution::EXPONENTIAL);
    Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), order, order+1);
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
    
    double L2norm = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
    double diff;
    
    solution.solve();
    diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15);
    cout << "POISSON EXPONENTIAL (p=" << order-1 << "): L2 Error in solution for " << sqrtElements << "x" << sqrtElements << " mesh: " << diff << endl;
    diff = diff / L2norm;
    
    cout << "Relative L2 Error: " << diff << endl;
    // if (diff > tol) {
    //  success = false;
    //  cout << "Failure: Error in solution of Poisson (Sacado version) was " << diff << "; tolerance set to " << tol << endl;
    // }
    ostringstream fileName;
    fileName << "PoissonPhiSolution.p=" << order-1 << "." << sqrtElements << "x" << sqrtElements << ".dat";
    solution.writeToFile(PoissonBilinearForm::PHI, fileName.str());
  }*/
  return success;
}

bool MeshTestSuite::testExactSolution(bool checkL2Norm) {
  // a test of the L2 error computation
  // we build a solution that's just 0, and check that the error
  // is the L2 norm of the exact solution itself.
  bool success = true;
  int numTests = 1;
  
  double tol = 5e-11;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  Teuchos::RCP<PoissonExactSolutionLinear> exactLinear = Teuchos::rcp(new PoissonExactSolutionLinear());
  Teuchos::RCP<PoissonExactSolutionQuadratic> exactQuadratic = Teuchos::rcp(new PoissonExactSolutionQuadratic());
  Teuchos::RCP<PoissonExactSolutionCubic> exactCubic = Teuchos::rcp(new PoissonExactSolutionCubic());
  Teuchos::RCP<PoissonExactSolutionQuartic> exactQuartic = Teuchos::rcp(new PoissonExactSolutionQuartic());
  
  //cout << "************************************************\n";
  //exactCubic->bilinearForm()->printTrialTestInteractions();
  //cout << "************************************************\n";
  
  vector<Teuchos::RCP<ExactSolution> > exactSolutions;
  vector<double> expectedPhiNorms;
  exactSolutions.push_back(exactLinear);
  expectedPhiNorms.push_back(sqrt(7.0/6.0)); // sqrt(integral of (x+y)^2 over (0,1)^2) -- thanks, Mathematica!
  exactSolutions.push_back(exactQuadratic);
  expectedPhiNorms.push_back(sqrt(28.0/45.0)); // sqrt(integral of (x^2+y^2)^2 over (0,1)^2) -- thanks, Mathematica!
  exactSolutions.push_back(exactCubic);
  expectedPhiNorms.push_back(sqrt(27.0/28.0));  // sqrt(integral of (x^3+2y^3)^2 over (0,1)^2) -- thanks, Mathematica!
  exactSolutions.push_back(exactQuartic);
  expectedPhiNorms.push_back(sqrt(143.0/504.0)); // sqrt(integral of (x^4+x^3 y)^2 over (0,1)^2) -- thanks, Mathematica!
  
  // TODO: figure out why the quartic and cubic fail on the unit cell but not on the ref quad
  
  for (int i=0; i<exactSolutions.size(); i++) {
    Teuchos::RCP<ExactSolution> exactSolution = exactSolutions[i];
    int order = exactSolution->H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, 3, 3, exactSolution->bilinearForm(), order, order+1);
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution->bilinearForm()));
    
    Solution solution(myMesh, exactSolution->bc(), exactSolution->rhs(), ip);
    // Poisson is set up such that the solution should be x + y
    
    double diff;
    
    if (checkL2Norm) {
      // don't solve; just compute the error compared to a 0 solution
      double phiError = exactSolution->L2NormOfError(solution, PoissonBilinearForm::PHI);
      double expected = expectedPhiNorms[i];
      diff = abs(phiError - expected);
      //cout << "for 1x1 mesh, L2 norm of phi for PoissonExactSolution: " << phiError << endl;
      
      if (diff > tol) {
        success = false;
        cout << "Expected norm of exact solution to be " << expected << " but PoissonExactSolution gave " << phiError << endl;
      }
    } else {
      solution.solve();
      diff = exactSolution->L2NormOfError(solution, PoissonBilinearForm::PHI);
      //cout << "Error in solution of Poisson exactly recoverable solution " << i << ": " << diff << endl;
      if (diff > tol) {
        success = false;
        cout << "Failure: Error in solution of Poisson was " << diff << "; tolerance set to " << tol << endl;
      }
    }
  }
  
  return success;
}

bool MeshTestSuite::testBuildMesh() {
  bool success = true;
  
  int order = 2; // linear on interior
  
  FieldContainer<double> quadPoints(4,2);
  
   quadPoints(0,0) = -1.0; // x1
   quadPoints(0,1) = -1.0; // y1
   quadPoints(1,0) = 1.0;
   quadPoints(1,1) = -1.0;
   quadPoints(2,0) = 1.0;
   quadPoints(2,1) = 1.0;
   quadPoints(3,0) = -1.0;
   quadPoints(3,1) = 1.0;
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new PoissonBilinearForm() );
  
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, 1, 1, bilinearForm, order, order);
  // some basic sanity checks:
  int numElementsExpected = 1;
  if (myMesh->numElements() != numElementsExpected) {
    cout << "mymesh->numElements() != numElementsExpected; numElements()=" << myMesh->numElements() << endl;
    success = false;
  }
  bool localSuccess = MeshTestUtility::checkMeshDofConnectivities(myMesh);

  if (!localSuccess) {
    cout << "MeshTestUtility::checkMeshDofConnectivities failed for 1x1 mesh." << endl;
    success = false;
  }
  
  Teuchos::RCP<Mesh> myMesh2x1 = Mesh::buildQuadMesh(quadPoints, 2, 1, bilinearForm, order, order);
  // some basic sanity checks:
  numElementsExpected = 2;
  if (myMesh2x1->numElements() != numElementsExpected) {
    cout << "myMesh2x1.numElements() != numElementsExpected; numElements()=" << myMesh2x1->numElements() << endl;
    success = false;
  }
  localSuccess = MeshTestUtility::checkMeshDofConnectivities(myMesh2x1);
  
  if (!localSuccess) {
    cout << "MeshTestUtility::checkMeshDofConnectivities failed for 2x1 mesh." << endl;
    success = false;
  }
  return success;
}

bool MeshTestSuite::testMeshSolvePointwise() {
  bool success = true;
  int numTests = 1;
  
  double tol = 2.5e-14;
  
  int order = 2; // linear on interior
    
  FieldContainer<double> quadPoints(4,2);

  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new PoissonBilinearForm() );
  
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, 1, 1, bilinearForm, order, order+1);
  // some basic sanity checks:
  int numElementsExpected = 1;
  
  Teuchos::RCP<PoissonBCLinear> bc = Teuchos::rcp( new PoissonBCLinear() );
  Teuchos::RCP<PoissonRHSLinear> rhs = Teuchos::rcp( new PoissonRHSLinear() );
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  
  Solution solution(myMesh, bc, rhs, ip);
  // Poisson is set up such that the solution should be x + y
  
  int PHI = 2;
  int numPoints = 3;
  FieldContainer<double> solnValues(1,numPoints);
  FieldContainer<double> expectedSolnValues(1,numPoints);
  FieldContainer<double> testPoints(1,numPoints,2);
  
  testPoints(0,0,0) = 0.0;
  testPoints(0,0,1) = 1.0;
  testPoints(0,1,0) = 1.0;
  testPoints(0,1,1) = 0.0;
  testPoints(0,2,0) = 0.5;
  testPoints(0,2,1) = 0.0;
  
  expectedSolnValues(0,0) = 1.0;
  expectedSolnValues(0,1) = 1.0;
  expectedSolnValues(0,2) = 0.5;
  
  solution.solve();
  
  solution.solutionValues(solnValues,myMesh->elements()[0]->elementType(),PHI,
                          testPoints);
  
  for (int i=0; i<numPoints; i++) {
    double diff = abs(expectedSolnValues(0,i) - solnValues(0,i));
    if ( diff > tol ) {
      cout << "Solve 1-element Poisson: expected " << expectedSolnValues(0,i) << ", but soln was " << solnValues(0,i) << " -- diff=" << diff << endl;
      success = false;
    }
  }
  
  // now same thing, but larger mesh
  // in this test, we do use knowledge of the way the mesh elements get laid out
  // (they go top to bottom first, then left to right--i.e. columnwise)
  // the whole mesh in this test is the (-1,1) square.
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  int horizontalElements = 2, verticalElements = 2;
  Teuchos::RCP<Mesh> myMesh2x2 = Mesh::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bilinearForm, order, order+1);
  
  Solution solution2x2(myMesh2x2, bc, rhs, ip);

  int numPointsPerElement = 1;
  int numElements = horizontalElements * verticalElements;
  int spaceDim = 2;
  testPoints.resize(numElements,numPointsPerElement,spaceDim);
  // could replace the following by something that figures out a point (or several) in the
  // middle of each element (using the vertices of each element).
  // That would eliminate the dependence on the layout ordering of the mesh.
  /*testPoints(0,0,0) = 0.0;
  testPoints(0,0,1) = 1.0;
  testPoints(1,0,0) = -0.5;
  testPoints(1,0,1) = -0.5;
  testPoints(2,0,0) = 0.5;
  testPoints(2,0,1) = 0.5;
  testPoints(3,0,0) = 1.0;
  testPoints(3,0,1) = -1.0;*/
  
  // diagonosing failure in Solution: do we succeed if all the refPoints are the same?
  // (picking point at top right of each element)
  testPoints(0,0,0) = 0.0;
  testPoints(0,0,1) = 0.0;
  testPoints(1,0,0) = 0.0;
  testPoints(1,0,1) = 1.0;
  testPoints(2,0,0) = 1.0;
  testPoints(2,0,1) = 0.0;
  testPoints(3,0,0) = 1.0;
  testPoints(3,0,1) = 1.0;

  expectedSolnValues.resize(numElements,numPointsPerElement);
  /*for (int elemIndex=0; elemIndex<numElements; elemIndex++) {
    for (int ptIndex=0; ptIndex<numPointsPerElement; ptIndex++) {
      double x = testPoints(elemIndex,ptIndex,0);
      double y = testPoints(elemIndex,ptIndex,1);
      expectedSolnValues(elemIndex,ptIndex) = x + y;
    }
  }*/
  PoissonExactSolutionLinear exactSolution;
  exactSolution.solutionValues(expectedSolnValues, PoissonBilinearForm::PHI,
                               testPoints);
  
  solution2x2.solve();
  solnValues.resize(numElements,numPointsPerElement); // four elements, one test point each
  solution2x2.solutionValues(solnValues,myMesh2x2->elements()[0]->elementType(),PoissonBilinearForm::PHI,
                          testPoints);
  
  for (int elemIndex=0; elemIndex<numElements; elemIndex++) {
    for (int ptIndex=0; ptIndex<numPointsPerElement; ptIndex++) {
      double diff = abs(expectedSolnValues(elemIndex,ptIndex) - solnValues(elemIndex,ptIndex));
      if ( diff > tol ) {
        cout << "Solve 4-element Poisson: expected " << expectedSolnValues(elemIndex,ptIndex) << ", but soln was " << solnValues(elemIndex,ptIndex) << endl;
        success = false;
      }
    }
  }
  
  // now try using the elementsForPoints variant of solutionValues
  solnValues.resize(numElements*numPointsPerElement); // four elements, one test point each
  testPoints.resize(numElements*numPointsPerElement,spaceDim);
  solution2x2.solutionValues(solnValues,PoissonBilinearForm::PHI,testPoints);
  
  for (int elemIndex=0; elemIndex<numElements; elemIndex++) {
    for (int ptIndex=0; ptIndex<numPointsPerElement; ptIndex++) {
      int solnIndex = elemIndex*numPointsPerElement + ptIndex;
      double diff = abs(expectedSolnValues(elemIndex,ptIndex) - solnValues(solnIndex));
      if ( diff > tol ) {
        cout << "Solve 4-element Poisson: expected " << expectedSolnValues(elemIndex,ptIndex) << ", but soln was " << solnValues(solnIndex) << " (using elementsForPoints solutionValues)" << endl;
        success = false;
      }
    }
  }
  
  // would be better to actually do the meshing, etc. with reference to the BC & RHS given by PoissonExactSolution,
  // but we do know that these are the same, so we'll just use the solutions we already have....
  double phiError = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI);
  //cout << "for 1x1 mesh, L2 error in phi for PoissonExactSolutionLinear: " << phiError << endl;
  phiError = exactSolution.L2NormOfError(solution2x2, PoissonBilinearForm::PHI);
  //cout << "for 2x2 mesh, L2 error in phi for PoissonExactSolutionLinear: " << phiError << endl;
  
  return success;
}

bool MeshTestSuite::testDofOrderingFactory() {
  bool success = true;
  int polyOrder = 3; 
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new PoissonBilinearForm());
  
  Teuchos::RCP<DofOrdering> conformingOrdering,nonConformingOrdering;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  
  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  nonConformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, false);
  
  Teuchos::RCP<DofOrdering> conformingOrderingCopy;
  conformingOrderingCopy = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  
  if (conformingOrderingCopy.get() != conformingOrdering.get() ) {
    cout << "testDofOrderingFactory: created a second copy of conforming ordering (uniqueness violated)." << endl;
    success = false;
  }
  
  // Several of these tests assert that indices are laid out in the same order in trialOrdering(), pRefine(), matchSides()
  conformingOrderingCopy = dofOrderingFactory.pRefine(conformingOrdering, quad_4, 0); // don't really refine
  
  if (conformingOrderingCopy.get() != conformingOrdering.get() ) {
    cout << "testDofOrderingFactory: conformingOrdering with pRefine==0 differs from original." << endl;
    success = false;
  }
  
  int pToAdd = 3;  
  
  conformingOrderingCopy = dofOrderingFactory.pRefine(conformingOrdering, quad_4, pToAdd);
  
  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, true);

  if (conformingOrderingCopy.get() != conformingOrdering.get() ) {
    cout << "testDofOrderingFactory: conformingOrdering with pRefine==3 differs from fresh one with polyOrder+3." << endl;
    success = false;
  }
  
  // TODO: add test of matchSides...
  // first, create two orderings of different polyOrder.  Then matchSides.  Then check that the higher-degree guy won, and that they do have the same Basis (pointer comparison).
  Teuchos::RCP<DofOrdering> nonConformingOrderingLowerOrder = dofOrderingFactory.trialOrdering(polyOrder, quad_4, false);
  Teuchos::RCP<DofOrdering> nonConformingOrderingHigherOrder = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, false);
  Teuchos::RCP<DofOrdering> conformingOrderingLowerOrder = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  Teuchos::RCP<DofOrdering> conformingOrderingHigherOrder = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, true);
  
  int higherSideInLowerElementConforming = 0; int lowerElementOtherSideNonConforming = 1;
  int lowerSideInHigherElementConforming = 2;
  int higherSideInLowerElementNonConforming = 0; int lowerElementOtherSideConforming = 1;
  int lowerSideInHigherElementNonConforming = 2;
  dofOrderingFactory.matchSides(nonConformingOrderingLowerOrder, higherSideInLowerElementNonConforming, quad_4,
                                nonConformingOrderingHigherOrder, lowerSideInHigherElementNonConforming, quad_4);
  dofOrderingFactory.matchSides(conformingOrderingLowerOrder, higherSideInLowerElementConforming,
                                quad_4, conformingOrderingHigherOrder, lowerSideInHigherElementConforming, quad_4);
  
  // check that the lower-order guys have basis that agrees with the higher-order at that edge
  BasisPtr lowerBasis = nonConformingOrderingLowerOrder->getBasis(PoissonBilinearForm::PHI_HAT,higherSideInLowerElementNonConforming); 
  BasisPtr lowerBasisOtherSide = nonConformingOrderingLowerOrder->getBasis(PoissonBilinearForm::PHI_HAT,lowerElementOtherSideNonConforming);
  if ( lowerBasis->getDegree() == lowerBasisOtherSide->getDegree() ) {
    success = false;
    cout << "FAILURE: After matchSides (non-conforming), the lower-order side doesn't appear to have been refined." << endl;
  }
  BasisPtr higherBasis = nonConformingOrderingHigherOrder->getBasis(PoissonBilinearForm::PHI_HAT,lowerSideInHigherElementNonConforming);
  if (lowerBasis.get() != higherBasis.get()) {
    success = false;
    cout << "FAILURE: After matchSides (non-conforming), sides have differing bases." << endl;
  }
  
  lowerBasis = conformingOrderingLowerOrder->getBasis(PoissonBilinearForm::PHI_HAT,higherSideInLowerElementConforming);
  lowerBasisOtherSide = conformingOrderingLowerOrder->getBasis(PoissonBilinearForm::PHI_HAT,lowerElementOtherSideConforming);
  if ( lowerBasis->getDegree() == lowerBasisOtherSide->getDegree() ) {
    success = false;
    cout << "FAILURE: After matchSides (conforming), the lower-order side doesn't appear to have been refined." << endl;
  }
  higherBasis = conformingOrderingHigherOrder->getBasis(PoissonBilinearForm::PHI_HAT,lowerSideInHigherElementConforming);
  if (lowerBasis.get() != higherBasis.get()) {
    success = false;
    cout << "FAILURE: After matchSides (conforming), sides have differing bases." << endl;
  }
  
  // next test: matchSides again, but now with the two guys that agree on a given side.  Nothing should change.
  Teuchos::RCP<DofOrdering> nonConformingOrderingLowerOrderCopy = nonConformingOrderingLowerOrder;
  Teuchos::RCP<DofOrdering> nonConformingOrderingHigherOrderCopy = nonConformingOrderingHigherOrder;
  Teuchos::RCP<DofOrdering> conformingOrderingLowerOrderCopy = conformingOrderingLowerOrder;
  Teuchos::RCP<DofOrdering> conformingOrderingHigherOrderCopy = conformingOrderingHigherOrder;
  
  dofOrderingFactory.matchSides(nonConformingOrderingLowerOrder, higherSideInLowerElementNonConforming, quad_4,
                                nonConformingOrderingHigherOrder, lowerSideInHigherElementNonConforming, quad_4);
  dofOrderingFactory.matchSides(conformingOrderingLowerOrder, higherSideInLowerElementConforming,
                                quad_4, conformingOrderingHigherOrder, lowerSideInHigherElementConforming, quad_4);
  
  if ( nonConformingOrderingLowerOrderCopy.get() != nonConformingOrderingLowerOrder.get() ) {
    success = false;
    cout << "FAILURE: After second call to matchSides (non-conforming), LowerOrder changed (matchSides not idempotent)." << endl;
  }
  if ( nonConformingOrderingHigherOrderCopy.get() != nonConformingOrderingHigherOrder.get() ) {
    success = false;
    cout << "FAILURE: After second call to matchSides (non-conforming), HigherOrder changed (matchSides not idempotent)." << endl;
  }  

  if ( conformingOrderingLowerOrderCopy.get() != conformingOrderingLowerOrder.get() ) {
    success = false;
    cout << "FAILURE: After second call to matchSides (conforming), LowerOrder changed (matchSides not idempotent)." << endl;
  }
  if ( conformingOrderingHigherOrderCopy.get() != conformingOrderingHigherOrder.get() ) {
    success = false;
    cout << "FAILURE: After second call to matchSides (conforming), HigherOrder changed (matchSides not idempotent)." << endl;
  }  
  
  // final test: take the upgraded ordering, and increase its polynomial order so that it matches that of the higher-degree guy.  Check that this is the same Ordering as a fresh one with that polynomial order.
  nonConformingOrderingLowerOrder = dofOrderingFactory.pRefine(nonConformingOrderingLowerOrder, quad_4, pToAdd);
  nonConformingOrderingHigherOrder = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, false);
  if ( nonConformingOrderingLowerOrder.get() != nonConformingOrderingHigherOrder.get() ) {
    success = false;
    cout << "FAILURE: After p-refinement of upgraded Ordering (non-conforming), DofOrdering doesn't match a fresh one with that p-order." << endl;    
  }
  
  conformingOrderingLowerOrder = dofOrderingFactory.pRefine(conformingOrderingLowerOrder, quad_4, pToAdd);
  conformingOrderingHigherOrder = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, true);
  if ( conformingOrderingLowerOrder.get() != conformingOrderingHigherOrder.get() ) {
    success = false;
    cout << "FAILURE: After p-refinement of upgraded Ordering (conforming), DofOrdering doesn't match a fresh one with that p-order." << endl;    
  }
  
  return success;
}

bool MeshTestSuite::testHRefinement() {
  bool success = true;
  
  // first, build a simple mesh
  
  double tol = 2e-11;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
  int sqrtElements = 2;
  
  int polyOrder = 1;
  PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, polyOrder); // 0 doesn't mean constant, but a particular solution...
  exactSolution.setUseSinglePointBCForPHI(true); // because these don't have zero mean on the domain...
  
  int H1Order = exactSolution.H1Order();
  int horizontalCells = 1; int verticalCells = 1;
  
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
  vector<int> cellsToRefine;
  for (int i=0; i<horizontalCells*verticalCells; i++) {
    cellsToRefine.push_back(i);
  }
  
  quadPoints.resize(1,4,2);
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(&quad_4,false);

  //RefinementPattern noRefinementPattern(quad_4_ptr,quadPoints);
  
  int numElementsStart = mesh->numElements();
  int numGlobalDofs = mesh->numGlobalDofs();
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  Solution origSolution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  origSolution.solve();
  
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  double y[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  FieldContainer<double> testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
    }
  }
  int trialID = PoissonBilinearForm::PHI;
  FieldContainer<double> valuesOriginal(testPoints.dimension(0));
  origSolution.solutionValues(valuesOriginal,trialID,testPoints);
  
  mesh->hRefine(cellsToRefine,RefinementPattern::noRefinementPatternQuad());
  
  int numElementsEnd = mesh->numElements(); // should be twice as many
  if ( numElementsEnd != 2*numElementsStart ) {
    success = false;
    cout << "FAILURE: Expected noRefinementPattern to produce 1 child for each parent\n";
  }
  
  if ( mesh->numGlobalDofs() != numGlobalDofs ) {
    success = false;
    cout << "FAILURE: Expected noRefinementPattern to produce no change in # dofs\n";
  }
  
  for (int i=0; i<cellsToRefine.size(); i++) {
    Teuchos::RCP<Element> parent = mesh->elements()[cellsToRefine[i]];
    Teuchos::RCP<Element> child = parent->getChild(0);
    if ( parent->elementType() != child->elementType() ) {
      success = false;
      cout << "FAILURE: Expected noRefinementPattern to produce no change in element type. \n";
    }
  }
  
  // try solving --> make sure that we get the same solution before and after "refinement"
  Solution solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  solution.solve();

  FieldContainer<double> valuesNew(valuesOriginal.dimension(0));
  solution.solutionValues(valuesNew,trialID,testPoints);
  
  for (int valueIndex=0; valueIndex<valuesOriginal.dimension(0); valueIndex++) {
    double diff = abs(valuesOriginal[valueIndex] - valuesNew[valueIndex]);
    if (diff > tol) {
      success = false;
      cout << "FAILURE: Expected noRefinementPattern to produce no change in solution. \n";
    }
  }
  
  // TODO: try a regular refinement pattern.  Check that this is a 4x4 mesh, and try solving.
  //    --> Make sure solution is the same as when we just start with 4x4 mesh.
  quadPoints.resize(4,2);
  Teuchos::RCP<Mesh> fineMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells*2, verticalCells*2, exactSolution.bilinearForm(), H1Order, H1Order+1);
  origSolution = Solution(fineMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  origSolution.solve();
  
  cellsToRefine.clear();
  for (int i=0; i<mesh->activeElements().size(); i++) {
    int cellID = mesh->activeElements()[i]->cellID();
    cellsToRefine.push_back(cellID);
  }
  mesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  numElementsEnd = mesh->numElements(); // should be another 4x
  if ( numElementsEnd != 2*numElementsStart + numElementsStart*4 ) {
    success = false;
    cout << "FAILURE: Expected regularRefinementPattern to produce 4 children for each parent\n";
  }
  
  if ( mesh->numGlobalDofs() != fineMesh->numGlobalDofs() ) {
    success = false;
    cout << "FAILURE: Expected uniform regular refinement to produce the same # dofs as  \n";
  }
  
  // TODO: check that the element types of fineMesh and refined mesh match...
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: after uniform regular refinement, mesh fails consistency check.\n";
  }
  
  solution.solve();
  
//  solution.writeToFile(PoissonBilinearForm::PHI, "phi_refined.dat");
//  origSolution.writeToFile(PoissonBilinearForm::PHI, "phi_fine.dat");
  
  double refinedError = exactSolution.L2NormOfError(solution,PoissonBilinearForm::PHI);
  double fineError = exactSolution.L2NormOfError(origSolution,PoissonBilinearForm::PHI);
  
//  cout << "refinedError:" << refinedError << endl;
//  cout << "fineError:" << fineError << endl;
  
  double diff = abs(refinedError - fineError);
  
  if (diff > tol) {
    cout << "FAILURE: after uniform regular refinement, L2 error different from originally fine mesh.\n";
    cout << "Difference of L2 error in refined vs. originally fine mesh: " << diff << endl;
  }
  
  // TODO: work out how to fix solution.equals to work with meshes whose cells may be in different orders...
  /*if ( ! solution.equals(origSolution, 1e-8) ) { // start with very relaxed tol... TODO: tighten this once we're passing...
    success = false;
    cout << "FAILURE: Expected solution of fine mesh and refined mesh to be equal. \n";
  }*/
  
  cellsToRefine.clear();
  
  // start with a fresh 2x1 mesh:
  horizontalCells = 2; verticalCells = 1;
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 8; // 1 to start!
  for (int i=0; i<numRefinements; i++) {
    vector< pair<int,int> > descendants = mesh->elements()[0]->getDescendantsForSide(1);
    int cellID = descendants[0].first;
    cellsToRefine.clear();
    cellsToRefine.push_back(cellID);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: after 'deep' refinement, mesh fails consistency check.\n";
  }
  
  // the following line should not be necessary, but if Solution's data structures aren't rebuilt properly, it might be...
  solution = Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  solution.solve();
  
  refinedError = exactSolution.L2NormOfError(solution,PoissonBilinearForm::PHI);
  
  if (refinedError > tol) {
    success = false;
    cout << "FAILURE: after 'deep' refinement for exactly recoverable solution, L2 error greater than tolerance.\n";
    cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
  }
  
//  solution.writeToFile(PoissonBilinearForm::PHI, "phi_refined_again.dat");
  
  // try to reproduce a parity error discovered by Jesse when enforcing 1-irregularity
  horizontalCells = 1; verticalCells = 2;
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
  cellsToRefine.clear();
  cellsToRefine.push_back(1); // top cell -- will refine to create 2,3,4,5 (2,3 are the bottom)
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  cellsToRefine.clear();
  cellsToRefine.push_back(2);
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  cellsToRefine.clear();
  cellsToRefine.push_back(3);
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  cellsToRefine.clear();
  cellsToRefine.push_back(0);
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());  
  if ( ! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "testHRefinement failed mesh consistency test in imitating 1-irregularity resolution" << endl;
  }
  return success;
}

void MeshTestSuite::printParities(Teuchos::RCP<Mesh> mesh) {
  int numElements = mesh->activeElements().size();
  for (int cellIndex=0; cellIndex<numElements; cellIndex++) {
    Teuchos::RCP<Element> elem = mesh->activeElements()[cellIndex];
    
    cout << "parities for cellID " << elem->cellID() << ": ";
    for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
      int parity = mesh->parityForSide(elem->cellID(),sideIndex);
      cout << parity;
      if (sideIndex != elem->numSides()-1) cout << ", ";
    }
    cout << endl;
  }
}

bool MeshTestSuite::testHRefinementForConfusion() {
  bool success = true;
  
  // first, build a simple mesh
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
  int sqrtElements = 2;
  
  double epsilon = 1e-1;
  double beta_x = 1.0, beta_y = 1.0;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y);
  
  int H1Order = 3;
  int horizontalCells = 4; int verticalCells = 4;
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));

  vector<int> cellsToRefine;
  cellsToRefine.clear();
  
  // start with a fresh 2x1 mesh:
  horizontalCells = 4; verticalCells = 4;
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+2);
  
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 2;
  for (int i=0; i<numRefinements; i++) {
    vector< pair<int,int> > descendants = mesh->elements()[0]->getDescendantsForSide(1);
    int numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.push_back(cellID);
    }
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());

    // same thing for north side
    descendants = mesh->elements()[0]->getDescendantsForSide(2);
    numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.push_back(cellID);
    }
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: after 'deep' refinement, mesh fails consistency check.\n";
  }
  
  // the following line should not be necessary, but if Solution's data structures aren't rebuilt properly, it might be...
  Solution solution = Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  solution.solve();
  
  double refinedError = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U);
  
  // relaxed tolerance
  double tol = 1e-1;
  if ((refinedError > tol) || (refinedError != refinedError)) { // second compare: is refinedError NaN?
    success = false;
    cout << "FAILURE: after 'deep' refinement for smooth solution, L2 error greater than tolerance.\n";
    cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
  }
  
  solution.writeFieldsToFile(ConfusionBilinearForm::U, "confusion_demo.m");
  
  return success;
}

bool MeshTestSuite::testPRefinement() {
  bool success = true;
  
  double tol = 2.5e-11;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
   int sqrtElements = 2;
  vector< PoissonExactSolution > exactSolutions;
  
  PoissonExactSolution exactExponential(PoissonExactSolution::EXPONENTIAL);
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 2); // 0 doesn't mean constant, but a particular solution...
  exactPolynomial.setUseSinglePointBCForPHI(true); // because these don't have zero mean on the domain...
  exactExponential.setUseSinglePointBCForPHI(true);
  
  exactSolutions.push_back( exactExponential );
  exactSolutions.push_back( exactPolynomial );
  
  int H1Order = exactPolynomial.H1Order();
  // 1st test is a bit complex:
  // 1. create two meshes, one quadratic (linear in phi), one cubic (quadratic in phi)
  //    (both of these meshes should recover the solution exactly)
  // 2. create a third mesh, also quadratic, but p-refine one element, and then solve.
  // 3. in the refined element, its phi dofs should be identical to those in the cubic mesh; 
  //    in the others, its phi dofs should be identical to those in the quadratic mesh...
  int horizontalCells = 2; int verticalCells = 2;
  int refinedCellID = 2;
  
  Teuchos::RCP<Mesh> mesh1 = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order, H1Order+1);
  Teuchos::RCP<Mesh> mesh2 = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order+1, H1Order+2);
  Teuchos::RCP<DPGInnerProduct> ip0 = Teuchos::rcp(new MathInnerProduct(exactExponential.bilinearForm()));
  vector<int> cellsToRefine;
  Solution solution1(mesh1, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip0);  
  solution1.solve();
  Solution solution2(mesh2, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip0);  
  solution2.solve();
  
  double error1 = exactPolynomial.L2NormOfError(solution1, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  double error2 = exactPolynomial.L2NormOfError(solution2, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error1 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of sufficiently high degree..." << endl;
  }
  if (error2 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of more than sufficiently high degree... (tol: " << tol << "; error2: " << error2 << ")" << endl;
  }
  Teuchos::RCP<Mesh> mesh3 = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order, H1Order+1);
  cellsToRefine.push_back(refinedCellID);
  mesh3->pRefine(cellsToRefine);
  
  if ( ! MeshTestUtility::checkMeshConsistency(mesh3) ) {
    cout << "After p-refinement, mesh consistency test FAILED." << endl;
    success = false;
  }
  
  Solution solution3(mesh3, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(),ip0);
  solution3.solve();
  double error3 = exactPolynomial.L2NormOfError(solution3, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  //cout << "refined mesh error: " << error3 << endl;
  
  for (int cellID=0; cellID< horizontalCells * verticalCells; cellID++) {
    FieldContainer<double> expectedSolnDofs; // for phi
    if (cellID != refinedCellID) {
      solution1.solnCoeffsForCellID(expectedSolnDofs,cellID,PoissonBilinearForm::PHI);
    } else {
      solution2.solnCoeffsForCellID(expectedSolnDofs,cellID,PoissonBilinearForm::PHI);
    }
    FieldContainer<double> actualSolnDofs;
    solution3.solnCoeffsForCellID(actualSolnDofs,cellID,PoissonBilinearForm::PHI);
    if ( actualSolnDofs.size() != expectedSolnDofs.size() ) {
      cout << "FAILURE: for cellID " << cellID << ", actualSolnDofs.size() != expectedSolnDofs.size() (";
      cout << actualSolnDofs.size() << " vs. " << expectedSolnDofs.size() << ")" << endl;
    } else {
      for (int i=0; i<actualSolnDofs.size(); i++) {
        double diff = abs(actualSolnDofs(i)-expectedSolnDofs(i));
        if (diff > tol * 10 ) { // * 10 because we can be a little more tolerant of the Dof values than, say, the overall L2 error.
          cout << "FAILURE: In cellID " << cellID << ", p-refined mesh differs in phi solution from expected ";
          cout << "in basis ordinal " << i << "(diff=" << diff << ")" << endl;
        }
      }
    }
  }
//  ostringstream fileName;
//  fileName << "PoissonPhiSolution_Manu_LinearOnLinear.p=" << H1Order-1 << ".4x1.dat";
//  solution1.writeToFile(PoissonBilinearForm::PHI, fileName.str());
//  fileName.str(""); // clear out the filename
//  fileName << "PoissonPhiSolution_Manu_LinearOnQuadratic.p=" << H1Order << ".4x1.dat";
//  solution2.writeToFile(PoissonBilinearForm::PHI, fileName.str());
//  fileName.str("");
//  fileName << "PoissonPhiSolution_Manu_LinearOnQuadratic.p=1or2.4x1.dat";
//  solution3.writeToFile(PoissonBilinearForm::PHI, fileName.str());
//  fileName.str("");
  
  // Do a test that just refines everywhere, albeit in several steps...  Compare that with solution starting with a higher-order mesh (should be identical).
  mesh1 = Mesh::buildQuadMesh(quadPoints, 4, 3, exactExponential.bilinearForm(), H1Order, H1Order+1);
  mesh2 = Mesh::buildQuadMesh(quadPoints, 4, 3, exactExponential.bilinearForm(), H1Order+1, H1Order+2);
  ip0 = Teuchos::rcp(new MathInnerProduct(exactExponential.bilinearForm()));
  cellsToRefine.clear();
  for (int i=5; i<8; i++) {
    cellsToRefine.push_back(i);
  }
  mesh1->pRefine(cellsToRefine);
  cellsToRefine.clear();
  for (int i=8; i<mesh1->elements().size(); i++) {
    cellsToRefine.push_back(i);
  }
  mesh1->pRefine(cellsToRefine);
  cellsToRefine.clear();
  for (int i=0; i<5; i++) {
    cellsToRefine.push_back(i);
  }
  mesh1->pRefine(cellsToRefine);
  // now the mesh should be uniform again...
  if (mesh1->elementTypes().size() != 1) {
    cout << "FAILURE: refined-everywhere uniform mesh should have only one element type." << endl;
    vector<Teuchos::RCP<ElementType> > elementTypes = mesh1->elementTypes();
    vector<Teuchos::RCP<ElementType> >::iterator typeIt;
    int typeIndex = 0;
    for (typeIt=elementTypes.begin(); typeIt != elementTypes.end(); typeIt++) {
      cout << "trialOrdering for type " << typeIndex << ":" << endl;
      cout << *((*typeIt)->trialOrderPtr.get());
      cout << "testOrdering for type " << typeIndex << ":" << endl;
      cout << *((*typeIt)->testOrderPtr.get());
      typeIndex++;
    }
    cout << "trialOrdering for uniform mesh:" << endl;
    cout << *(mesh2->elementTypes()[0]->trialOrderPtr.get());
    cout << "testOrdering for  uniform mesh:" << endl;
    cout << *(mesh2->elementTypes()[0]->testOrderPtr.get());
    
    success = false;
  }
  
  solution1 = Solution(mesh1, exactExponential.bc(), exactExponential.ExactSolution::rhs(), ip0);  
  solution1.solve();
  solution2 = Solution(mesh2, exactExponential.bc(), exactExponential.ExactSolution::rhs(), ip0);  
  solution2.solve();
  
  error1 = exactExponential.L2NormOfError(solution1, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  error2 = exactExponential.L2NormOfError(solution2, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  double diff = abs(error1-error2);
  if (diff > tol) {
    success = false;
    cout << "FAILURE: Refined everywhere mesh gives different solution than fresh mesh with same orders." << endl;
  }
   for (int i=0; i<exactSolutions.size(); i++) {
     PoissonExactSolution exactSolution = exactSolutions[i];
     
     Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), H1Order, H1Order+1);
     
     cellsToRefine.clear();
     cellsToRefine.push_back(2);
     
     Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
     Solution solution(myMesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
     
     double L2norm = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
     double diff;
     double prev_error = 1.0; // relative error starts at 1
     
     myMesh->pRefine(cellsToRefine);
     // now that we've refined 1 element, its 2 neighbors are also changed, but the diagonal elemnt has not.
     // depending on how the mesh oriented the elements, we might have either 3 or 4 element types.
     // (as presently implemented, it should be 4 types).
     int numElementTypes = myMesh->elementTypes().size();
     if ( (numElementTypes != 3) && (numElementTypes != 4) ) {
       success = false;
       cout << "FAILURE: After 1st p-refinement, expected 3 or 4 elementTypes, but had " << numElementTypes << endl;
       // abort:
       return success;
     } else {
       //cout << "numElementTypes after 1st p-refinement: " << numElementTypes << endl;
     }
     if ( ! MeshTestUtility::checkMeshConsistency(myMesh) ) {
       success = false;
       cout << "FAILURE: After 1st p-refinement, MeshTestUtility::checkMeshConsistency failed." << endl;
     }

     solution.solve();
     diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15);
     //cout << "1st p-refinement test: POISSON Manuf. #" << i << " (p=" << H1Order-1 << "): L2 Error in solution for " << sqrtElements << "x" << sqrtElements << " mesh: " << diff << endl;
     diff = diff / L2norm;
     
     //cout << "Relative L2 Error: " << diff << endl;
     prev_error = diff;
     
     myMesh->pRefine(cellsToRefine);
     if ( ! MeshTestUtility::checkMeshConsistency(myMesh) ) {
       success = false;
       cout << "FAILURE: After 2nd p-refinement, MeshTestUtility::checkMeshConsistency failed." << endl;
     }
     solution.solve();
     diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15);
     //cout << "2nd p-refinement test: POISSON Manuf. #" << i << " (p=" << H1Order-1 << "): L2 Error in solution for " << sqrtElements << "x" << sqrtElements << " mesh: " << diff << endl;
     diff = diff / L2norm;
     //cout << "Relative L2 Error: " << diff << endl;
     
     if ( (diff / prev_error > 1.02) && (prev_error > tol) ) {
       success = false;
       cout << "FAILURE: relative error increased by more than 2% after 2nd p-refinement" << endl;
     }
     
     cellsToRefine.push_back(0);
     cellsToRefine.push_back(1);
     cellsToRefine.push_back(3);
     myMesh->pRefine(cellsToRefine);
     if ( ! MeshTestUtility::checkMeshConsistency(myMesh) ) {
       success = false;
       cout << "FAILURE: After 3rd p-refinement, MeshTestUtility::checkMeshConsistency failed." << endl;
     }
     solution.solve();
     diff = exactSolution.L2NormOfError(solution, PoissonBilinearForm::PHI,15);
     //cout << "3rd p-refinement test: POISSON Manuf. #" << i << " (p=" << H1Order-1 << "): L2 Error in solution for " << sqrtElements << "x" << sqrtElements << " mesh: " << diff << endl;
     diff = diff / L2norm;
     //cout << "Relative L2 Error: " << diff << endl;
     
     if ( (diff / prev_error > 1.02) && (prev_error > tol) ) {
       success = false;
       cout << "FAILURE: relative error increased by more than 2% after 3rd p-refinement" << endl;
     }
     
     //if (diff > tol) {
     //  success = false;
     //  cout << "Failure: Error in solution of Poisson (Sacado version) was " << diff << "; tolerance set to " << tol << endl;
     //}
//     ostringstream fileName;
//     fileName << "PoissonPhiSolution_Manu_" << i << ".p=" << H1Order-1 << "." << sqrtElements << "x" << sqrtElements << ".dat";
//     solution.writeToFile(PoissonBilinearForm::PHI, fileName.str());
   } 
  
  return success;
}

bool MeshTestSuite::testSinglePointBC() {
  bool success = true;
  double tol = 5e-12;
  
  int horizontalCells = 4, verticalCells = 4;
  int pToAdd = 1;
  
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 1);
  
  exactPolynomial.setUseSinglePointBCForPHI(true);
  
  int order = 2;
  // instead of ref quad, use unit cell (force a transformation)
  FieldContainer<double> quadPoints(4,2);
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), order, order+pToAdd);
  if ( ! MeshTestUtility::checkMeshConsistency(mesh) ) {
    cout << "In singlePointBC test, mesh consistency test FAILED for non-conforming mesh." << endl;
    success = false;
  }
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution(mesh, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip);
  double L2norm = exactPolynomial.L2NormOfError(solution, PoissonBilinearForm::PHI);
  solution.solve();
  
  double error = exactPolynomial.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using single-point BC, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "(L2 norm of solution: " << L2norm << ")\n";
  } else {
    //cout << "Success! Single-point BC Poisson error: " << error << endl;
  }
  
  PoissonExactSolution exactPolynomialConforming = PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 1, true); // use conforming traces
  
  exactPolynomialConforming.setUseSinglePointBCForPHI(true);
  
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, 
                               exactPolynomialConforming.bilinearForm(), order, order+pToAdd);
  if ( ! MeshTestUtility::checkMeshConsistency(mesh) ) {
    cout << "In singlePointBC test, mesh consistency test FAILED for conforming mesh." << endl;
    success = false;
  }
  
  Teuchos::RCP<ElementType> elemTypePtr = mesh->elements()[0]->elementType();
  Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
  if ( checkDofOrderingHasNoOverlap(trialOrdering) ) {
    cout << "FAILURE: expected trialOrdering (which is conforming) not to be a 1-1 map, but it is." << endl;
    success = false;
  }
  ip = Teuchos::rcp(new MathInnerProduct(exactPolynomialConforming.bilinearForm()));
  Solution solutionConforming = Solution(mesh, exactPolynomialConforming.bc(), 
                                         exactPolynomialConforming.ExactSolution::rhs(), ip);
  solutionConforming.solve();
  
  error = exactPolynomialConforming.L2NormOfError(solutionConforming, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using single-point BC with conforming traces, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "(L2 norm of true solution: " << L2norm << ")\n";
  } else {
    //cout << "Success! Single-point BC Poisson error (conforming traces): " << error << endl;
  }
  
  return success;
}

bool MeshTestSuite::testSolutionForMultipleElementTypes() {
  // tests whether we can recover an exact solution in the space
  // on a uniform mesh which has artificially created multiple ElementTypes.
  // (physically, it's the same mesh--it's just that we break the uniquing
  //  provided by ElementTypeFactory...)
  bool success = true;
  
  double tol = 1e-12;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 0); // 0 doesn't mean constant, but a particular solution...
  exactPolynomial.setUseSinglePointBCForPHI(true);
  
  int order = 2; // H1 order ==> L2 order of order-1.
  int horizontalCells = 2; int verticalCells = 2;
  
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), order, order+1);
  Teuchos::RCP<DPGInnerProduct> ip0 = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution1(mesh, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip0);  
  solution1.solve();
  double error1 = exactPolynomial.L2NormOfError(solution1, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error1 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of sufficiently high degree..." << endl;
  } 
  // give each element its own type
  Teuchos::RCP<ElementType> elemTypePtr = mesh->elements()[0]->elementType();
  for (int i=0; i<mesh->elements().size(); i++) {
    Teuchos::RCP<ElementType> newElemTypePtr = Teuchos::rcp(new ElementType(elemTypePtr->trialOrderPtr,
                                                                            elemTypePtr->testOrderPtr,
                                                                            elemTypePtr->cellTopoPtr));
    mesh->elements()[i]->setElementType(newElemTypePtr);
  }
  // and rebuild the mesh data structures:
  mesh->rebuildLookups();
  Solution solution2(mesh, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip0);
  solution2.solve();
  double error2 = exactPolynomial.L2NormOfError(solution2, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error2 > tol) {
    success = false;
    cout << "FAILURE: Solution failed to solve on uniform mesh with two element types..." << endl;
    ostringstream fileName;
    fileName << "PoissonPhiSolution_FAILURE.p=" << order-1 << "." << horizontalCells << "x" << verticalCells << ".dat";
    solution2.writeToFile(PoissonBilinearForm::PHI, fileName.str());
    cout << "Wrote solution out to disk at: " << fileName.str() << endl;
  }
  return success;
}

bool MeshTestSuite::testSolutionForSingleElementUpgradedSide() {
  bool success = true;
  double tol = 5e-13;
  
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 0);
  
  exactPolynomial.setUseSinglePointBCForPHI(true);
  
  int order = 2;
  // use ref quad
  FieldContainer<double> quadPoints(4,2);
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;  
  
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, 1, 1, exactPolynomial.bilinearForm(), order, order);
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution(mesh, exactPolynomial.bc(), exactPolynomial.ExactSolution::rhs(), ip);
  solution.solve();
  
  double error = exactPolynomial.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using single-point BC, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "Single-point BC Poisson error (standard element): " << error << endl;
  }
  
  Teuchos::RCP<ElementType> elemTypePtr = mesh->elements()[0]->elementType();
  Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
  // create a second mesh so we can have another DofOrdering of higher order to match...
  Teuchos::RCP<Mesh> mesh2 = Mesh::buildQuadMesh(quadPoints, 1, 1, exactPolynomial.bilinearForm(), order+1, order);
  Teuchos::RCP<DofOrdering> trialOrderingToMatch = mesh2->elements()[0]->elementType()->trialOrderPtr;
  DofOrderingFactory dofOrderingFactory(exactPolynomial.bilinearForm());
  int sideIndex1 = 0, sideIndex2 = 2; // match EAST of our true element with WEST of the fake one from the other mesh
  int upgradedOrdering = dofOrderingFactory.matchSides(trialOrdering,sideIndex1,cellTopo,
                                                       trialOrderingToMatch,sideIndex2,cellTopo);
  if (upgradedOrdering != 1) {
    cout << "FAILURE: expected trialOrdering to be upgraded, but it was not." << endl;
    success = false;
    return success;
  }
  if ( ! checkDofOrderingHasNoOverlap(trialOrdering) ) {
    cout << "FAILURE: expected trialOrdering (which is non-conforming) to be a 1-1 map, but it isn't." << endl;
    success = false;
  }
  
  
  elemTypePtr = Teuchos::rcp(new ElementType(trialOrdering,elemTypePtr->testOrderPtr,
                                                  elemTypePtr->cellTopoPtr) );
  mesh->elements()[0]->setElementType(elemTypePtr);
  mesh->rebuildLookups();
  solution.solve();
  
  error = exactPolynomial.L2NormOfError(solution, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using upgraded side, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "Single-point BC Poisson error (upgraded side): " << error << endl;
    ostringstream fileName;
    fileName << "testSolutionForSingleElementUpgradedSide_FAILURE.p=" << order-1 << ".dat";
    solution.writeToFile(PoissonBilinearForm::PHI, fileName.str());
    cout << "Wrote solution out to disk at: " << fileName.str() << endl;
  }
  
  return success;
}

bool MeshTestSuite::checkDofOrderingHasNoOverlap(Teuchos::RCP<DofOrdering> dofOrdering) {
  // checkDofOrderingHasNoOverlap returns true if no two (varID,basisOrdinal,sideIndex) tuples map to same dofIndex
  // (won't be true for orderings that have H1 dofs that are conforming)
  bool noOverlap = true;
  set<int> dofIndices;
  vector<int> varIDs = dofOrdering->getVarIDs();
  vector<int>::iterator varIDIt;
  for (varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++) {
    int varID = *varIDIt;
    int numSides = dofOrdering->getNumSidesForVarID(varID);
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      BasisPtr basis = dofOrdering->getBasis(varID,sideIndex);
      int basisCardinality = basis->getCardinality();
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
        int dofIndex = dofOrdering->getDofIndex(varID, basisOrdinal, sideIndex);
        if (dofIndices.find(dofIndex) != dofIndices.end() ) {
          noOverlap = false;
        }
        dofIndices.insert(dofIndex);
      }
    }
  }
  return noOverlap;
}

bool MeshTestSuite::testRefinementPattern() {
  
  bool success = true;
  
  // first, build a simple mesh
  
  double tol = 2e-11;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
  int sqrtElements = 2;
  
  int polyOrder = 1;
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new PoissonBilinearForm() );
  
  int H1Order = 2;
  int horizontalCells = 2; int verticalCells = 1;
  
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bilinearForm, H1Order, H1Order+1);
  
  vector<int> cellsToRefine;
  cellsToRefine.push_back(0);
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  
  Teuchos::RCP<Element> elem = mesh->elements()[0];
  int numChildrenExpected = 4;
  if (elem->numChildren() != numChildrenExpected) {
    success = false;
    cout << "After refinement, wrong number of children." << endl;
  }
  
  int sideIndex = 0;
  vector< vector< pair<int,int> > > expectedChildrenForSide(4);
  expectedChildrenForSide[sideIndex].push_back( make_pair(0, 0) );
  expectedChildrenForSide[sideIndex].push_back( make_pair(1, 0) );
  
  sideIndex = 1;
  expectedChildrenForSide[sideIndex].push_back( make_pair(1, 1) );
  expectedChildrenForSide[sideIndex].push_back( make_pair(2, 1) );
  
  sideIndex = 2;
  expectedChildrenForSide[sideIndex].push_back( make_pair(2, 2) );
  expectedChildrenForSide[sideIndex].push_back( make_pair(3, 2) );
  
  sideIndex = 3;
  expectedChildrenForSide[sideIndex].push_back( make_pair(3, 3) );
  expectedChildrenForSide[sideIndex].push_back( make_pair(0, 3) );
  
  int numSides = 4;
  for (sideIndex = 0; sideIndex< numSides; sideIndex++) {
    vector< pair< int, int> > childrenForSide = elem->childIndicesForSide(sideIndex);
    if (! vectorPairsEqual(childrenForSide, expectedChildrenForSide[sideIndex])) {
      success = false;
      cout << "FAILURE: testRefinementPattern childrenForSide not the expected for side " << sideIndex << endl;
    }
  }
  
  cellsToRefine.clear();
  
  
  return success;
}

bool MeshTestSuite::vectorPairsEqual( vector< pair<int,int> > &first, vector< pair<int,int> > &second) {
  if (first.size() != second.size() ) {
    return false;
  }
  int size = first.size();
  for (int i = 0; i<size; i++) {
    pair<int,int> firstEntry  = first[i];
    pair<int,int> secondEntry = second[i];
    if (firstEntry.first != secondEntry.first) {
      return false;
    }
    if (firstEntry.second != secondEntry.second) {
      return false;
    }
  }
  return true;
}

bool MeshTestSuite::testPointContainment() {
  double tol = 2e-12; // had to increase for triangles
  bool success = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  // instead of ref quad, use unit cell (force a transformation)
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int pToAdd = 1;
  int horizontalCells=1,verticalCells=1;
  bool triangulate = false;
  int polyOrder = 2;
  vector<double> expectedValues;
  
  PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, polyOrder);
  exactSolution.setUseSinglePointBCForPHI(true);
  int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                  exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
  
  vector<double> x, y; vector<int> inside;
  x.push_back(0.5); y.push_back(0.5); inside.push_back(0);
  x.push_back(-0.5); y.push_back(0.5); inside.push_back(-1);
  x.push_back(1.0); y.push_back(0.5); inside.push_back(0);
  x.push_back(1.0); y.push_back(1.0); inside.push_back(0);
  x.push_back(0.0); y.push_back(1.0); inside.push_back(0);
  x.push_back(-0.000001); y.push_back(0.0); inside.push_back(-1);
  int numPoints = inside.size();
  int spaceDim = 2;
  FieldContainer<double> points(numPoints,spaceDim);
  for (int pointIndex = 0; pointIndex < numPoints; pointIndex++) {
    int cellID = inside[pointIndex];
    points(pointIndex,0) = x[pointIndex];
    points(pointIndex,1) = y[pointIndex];
  }
  
  typedef Teuchos::RCP< Element > ElementPtr;
  vector<ElementPtr> elements = myMesh->elementsForPoints(points);
  int testIndex = 0;
  for (vector<ElementPtr>::iterator elemIt=elements.begin(); elemIt != elements.end(); elemIt++) {
    if (elemIt->get()) {
      if ((*elemIt)->cellID() != inside[testIndex]) success = false;
    } else {
      if (inside[testIndex] != -1) success = false;
    }
    testIndex++;
  }
  
  return success;
}
