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
#include "ElementTypeFactory.h"

#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"

#include "MeshTestSuite.h"
#include "MeshTestUtility.h"

#include "Mesh.h"
#include "MathInnerProduct.h"
#include "Solution.h"
#include "Element.h"
#include "ElementType.h"
#include "BasisFactory.h"
#include <sstream>

#include "MeshUtilities.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "GnuPlotUtil.h"
#include "RefinementStrategy.h"

#include "MeshFactory.h"

// just for the test to close a mesh anisotropically
#include "Solution.h"
#include "VarFactory.h"
#include "RHS.h"
#include "BC.h"
#include "BF.h"

#include "HDF5Exporter.h"

#include "CamelliaCellTools.h"

using namespace Intrepid;

void MeshTestSuite::runTests(int &numTestsRun, int &numTestsPassed) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (rank==0)
    cout << "WARNING: skipping unrefinement test.\n";
  /*
   numTestsRun++;
   if (testHUnrefinementForConfusion() ) {
   numTestsPassed++;
   }
   */
  numTestsRun++;
  if (testHRefinementForConfusion() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testPointContainment() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if ( testPRefinement() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testMeshSolvePointwise() ) {
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
  
  // next three added by Jesse
  numTestsRun++;
  if (testPRefinementAdjacentCells() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testJesseMultiBasisRefinement() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  if (testJesseAnisotropicRefinement() ) {
    numTestsPassed++;
  }
}

bool MeshTestSuite::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, const FieldContainer<double> &testPointsRefCoords,
                                              bool reportErrors) {
  // iterates through all active elements, and checks that each edge agrees with its neighbor basis.
  
  // NOTE that this will fail to check certain sides in a general 3+D anisotropic mesh: specifically, where neighbors are anisotropically refined in different directions.
  
  // NOTE also that this only checks trace bases right now.  (Both this and the above should not be too hard to remedy.)
  
  bool success = true;
  
  double tol = 1e-12;
  double maxDiff = 0.0;
  
  unsigned spaceDim = mesh->getTopology()->getSpaceDim();
  unsigned sideDim = spaceDim - 1;
  
  int numPoints = testPointsRefCoords.dimension(0);
  
  // in 2D, the neighbor's view of the test points will be simply flipped relative to its peer
  // (which might be the ancestor of the element that we test below)
  FieldContainer<double> neighborTestPointsRefCoords(testPointsRefCoords); // alloc the right size container
  FieldContainer<double> ancestorTestPointsRefCoords(testPointsRefCoords); // alloc the right size container

  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();
  
  vector<int> traceIDs = mesh->bilinearForm()->trialBoundaryIDs();
  vector< ElementPtr > activeElements = mesh->activeElements();
  int numElements = activeElements.size();
  for (int cellIndex=0; cellIndex<numElements; cellIndex++) {
    Teuchos::RCP<Element> elem = activeElements[cellIndex];
    DofOrderingPtr trialOrder = elem->elementType()->trialOrderPtr;
    int cellID = elem->cellID();
    
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    int numSides = elem->numSides();
    
    BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    for (int sideOrdinal=0; sideOrdinal < numSides; sideOrdinal++) {
      int neighborSideOrdinal;
      Teuchos::RCP<Element> neighbor = mesh->ancestralNeighborForSide(elem,sideOrdinal,neighborSideOrdinal);
      if ( (neighbor.get() == NULL) || neighbor->isParent() ) { // boundary or broken neighbor
        // if broken neighbor, we'll handle this element when we handle neighbor's active descendants.
        continue;
      }
      DofOrderingPtr neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
      
      CellPtr neighborCell = mesh->getTopology()->getCell(neighbor->cellID());
      
      // NEW STRATEGY (dimensionally independent, and independent of max/min rule choice):
      /*
       - we know here that neighbor is at least as large as the cell.
       - therefore, we ask the cell for the RefinementBranch which will reconcile it to its neighbor on this side
       - then we restrict said RefinementBranch to the appropriate (ancestral) side
       - we map the test points from the leaf of the side RefinementBranch to the reference space for the ancestor
       - we then set the "physical" cell nodes for a BasisCache on the ancestor according to their relative permutation.
       - the "physical cubature points" on this BasisCache will then be reference points for the neighbor's side
       - for each trace variable, evaluate both bases at the appropriate points to determine the element-local discretization along that side.
       - use the mesh's GDA to map each set of local values to global values.
       */
      
      RefinementBranch cellRefBranch = cell->refinementBranchForSide(sideOrdinal);
      unsigned ancestralSideOrdinal = cell->ancestralSubcellOrdinalAndDimension(sideDim, sideOrdinal).first;
      
      CellPtr ancestralCell = cell->ancestralCellForSubcell(sideDim, sideOrdinal);
      
      RefinementBranch sideRefBranch = RefinementPattern::subcellRefinementBranch(cellRefBranch, sideDim, ancestralSideOrdinal);

      CellTopoPtr sideTopo = cell->topology()->getSubcell(sideDim, sideOrdinal);
      BasisCachePtr sideTopoCache = Teuchos::rcp( new BasisCache(sideTopo,1,false) );
      sideTopoCache->setRefCellPoints(testPointsRefCoords);
      
      FieldContainer<double> fineSideRefNodes;
      if (sideRefBranch.size() > 0) {
        fineSideRefNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(sideRefBranch);
      } else {
        fineSideRefNodes.resize(sideTopo->getVertexCount(),sideTopo->getDimension());
        CamelliaCellTools::refCellNodesForTopology(fineSideRefNodes, sideTopo);
      }
      fineSideRefNodes.resize(1,fineSideRefNodes.dimension(0),fineSideRefNodes.dimension(1));
      sideTopoCache->setPhysicalCellNodes(fineSideRefNodes, vector<GlobalIndexType>(), false);
      
      FieldContainer<double> testPointsInAncestralSide = sideTopoCache->getPhysicalCubaturePoints();
      // strip off cell dimension:
      testPointsInAncestralSide.resize(testPointsInAncestralSide.dimension(1),testPointsInAncestralSide.dimension(2));
      
      CellTopoPtr ancestralSideTopo = ancestralCell->topology()->getSubcell(sideDim, ancestralSideOrdinal);
      BasisCachePtr ancestralSideTopoCache = Teuchos::rcp(new BasisCache(ancestralSideTopo,1,false));
      ancestralSideTopoCache->setRefCellPoints(testPointsInAncestralSide);
      
      // determine permutation of ancestral reference space to get from ancestor's view to neighbor's:
      unsigned ancestralPermutation = ancestralCell->subcellPermutation(sideDim, ancestralSideOrdinal);
      unsigned neighborPermutation = neighborCell->subcellPermutation(sideDim, neighborSideOrdinal);
      unsigned neighborPermutationInverse = CamelliaCellTools::permutationInverse(ancestralSideTopo, neighborPermutation);
      unsigned composedPermutation = CamelliaCellTools::permutationComposition(ancestralSideTopo, neighborPermutationInverse, ancestralPermutation);
      
      // when you set physical cell nodes according to the coarse-to-fine permutation, then the reference-to-physical map
      // is fine-to-coarse (which is what we want).  Because the composedPermutation is fine-to-coarse, we want its inverse:
      unsigned composedPermutationInverse = CamelliaCellTools::permutationInverse(ancestralSideTopo, composedPermutation);
      
      FieldContainer<double> permutedAncestralSideReferenceNodes(ancestralSideTopo->getVertexCount(),sideDim);
      CamelliaCellTools::refCellNodesForTopology(permutedAncestralSideReferenceNodes, ancestralSideTopo, composedPermutationInverse);
      // add cell dimension:
      permutedAncestralSideReferenceNodes.resize(1,permutedAncestralSideReferenceNodes.dimension(0),permutedAncestralSideReferenceNodes.dimension(1));
      ancestralSideTopoCache->setPhysicalCellNodes(permutedAncestralSideReferenceNodes, vector<GlobalIndexType>(), false);
      
      FieldContainer<double> neighborRefPoints = ancestralSideTopoCache->getPhysicalCubaturePoints();
      // strip cell dimension:
      neighborRefPoints.resize(neighborRefPoints.dimension(1),neighborRefPoints.dimension(2));
      
      BasisCachePtr sideCache = cellBasisCache->getSideBasisCache(sideOrdinal);
      sideCache->setRefCellPoints(testPointsRefCoords);
      
      BasisCachePtr neighborCache = BasisCache::basisCacheForCell(mesh, neighbor->cellID());
      BasisCachePtr neighborSideCache = neighborCache->getSideBasisCache(neighborSideOrdinal);
      neighborSideCache->setRefCellPoints(neighborRefPoints);
      
//      cout << "cell physicalCubature points:\n" << sideCache->getPhysicalCubaturePoints();
//      cout << "neighbor physicalCubaturePoints:\n" << neighborSideCache->getPhysicalCubaturePoints();
      
      // sanity check: physicalCubaturePoints agree:
      double maxCubatureDiff;
      if (! fcsAgree(sideCache->getPhysicalCubaturePoints(), neighborSideCache->getPhysicalCubaturePoints(), 1e-15, maxCubatureDiff)) {
        cout << "TEST ERROR: physical cubature points differ.\n";
        success = false;
        return success;
      }
      
      for (vector<int>::iterator traceIt = traceIDs.begin(); traceIt != traceIDs.end(); traceIt++) {
        int traceID = *traceIt;
        BasisPtr basis = trialOrder->getBasis(traceID,sideOrdinal);
        BasisPtr neighborBasis = neighborTrialOrder->getBasis(traceID,neighborSideOrdinal);
        
        FieldContainer<double> cellLocalValues = *sideCache->getTransformedValues(basis, OP_VALUE);
        FieldContainer<double> neighborLocalValues = *neighborSideCache->getTransformedValues(neighborBasis, OP_VALUE);
        
//        cout << "cellLocalValues:\n" << cellLocalValues;
//        cout << "neighborLocalValues:\n" << neighborLocalValues;
        
        FieldContainer<double> cellGlobalValues, neighborGlobalValues;
        FieldContainer<GlobalIndexType> cellGlobalDofIndices, neighborGlobalDofIndices;
        
        Teuchos::Array<int> dim;
        cellLocalValues.dimensions(dim); // CFP[D,D,...]
        dim.remove(2); // CF[D,D,...]
        dim.remove(0); // F[D,D,...]
        int numValuesPerPoint = 1;
        for (int i=1; i<dim.size(); i++) {
          numValuesPerPoint *= dim[i];
        }
        Teuchos::Array<int> valueEnumeration;
        cellLocalValues.dimensions(valueEnumeration);
        for (int i=0; i<valueEnumeration.size(); i++) {
          valueEnumeration[i] = 0;
        }

        FieldContainer<double> cellLocalValuesForPoint(trialOrder->totalDofs());
        FieldContainer<double> neighborLocalValuesForPoint(neighborTrialOrder->totalDofs());
        
        for (int ptOrdinal=0; ptOrdinal < numPoints; ptOrdinal++) {
          valueEnumeration[2] = ptOrdinal;
          for (int i=0; i<numValuesPerPoint; i++) {
            for (int dofOrdinal=0; dofOrdinal < basis->getCardinality(); dofOrdinal++) {
              valueEnumeration[1] = dofOrdinal;
              unsigned valueOffset = cellLocalValues.getEnumeration(valueEnumeration);
              unsigned localDofIndex = trialOrder->getDofIndex(traceID, dofOrdinal, sideOrdinal);
              cellLocalValuesForPoint[localDofIndex] = cellLocalValues[valueOffset+i];
            }
            for (int dofOrdinal=0; dofOrdinal < neighborBasis->getCardinality(); dofOrdinal++) {
              valueEnumeration[1] = dofOrdinal;
              unsigned valueOffset = neighborLocalValues.getEnumeration(valueEnumeration);
              unsigned localDofIndex = neighborTrialOrder->getDofIndex(traceID, dofOrdinal, neighborSideOrdinal);
              neighborLocalValuesForPoint[localDofIndex] = neighborLocalValues[valueOffset+i];
            }

//          cout << "cellLocalValuesForPoint:\n" << cellLocalValuesForPoint;
//          cout << "neighborLocalValuesForPoint:\n" << neighborLocalValuesForPoint;
          
            gda->interpretLocalData(cellID, cellLocalValuesForPoint, cellGlobalValues, cellGlobalDofIndices);
            gda->interpretLocalData(neighbor->cellID(), neighborLocalValuesForPoint, neighborGlobalValues, neighborGlobalDofIndices);
            
            // it's a bit hackish, but we do know that the global values we're interested in are the non-zero ones
            // so we eliminate the zeros:
            
            double zeroTol = 1e-14; // anything less than this is considered to be zero
            vector<int> nonzeroOrdinals;
            for (int i=0; i<cellGlobalValues.size(); i++) {
              if (abs(cellGlobalValues[i]) > zeroTol) {
                nonzeroOrdinals.push_back(i);
              }
            }
            FieldContainer<double> cellGlobalValuesSideRestriction(nonzeroOrdinals.size());
            FieldContainer<GlobalIndexType> cellGlobalDofIndicesSideRestriction(nonzeroOrdinals.size());
            for (int i=0; i<nonzeroOrdinals.size(); i++) {
              cellGlobalValuesSideRestriction(i) = cellGlobalValues(nonzeroOrdinals[i]);
              cellGlobalDofIndicesSideRestriction(i) = cellGlobalDofIndices(nonzeroOrdinals[i]);
            }
            
            nonzeroOrdinals.clear();
            for (int i=0; i<neighborGlobalValues.size(); i++) {
              if (abs(neighborGlobalValues[i]) > zeroTol) {
                nonzeroOrdinals.push_back(i);
              }
            }
            FieldContainer<double> neighborGlobalValuesSideRestriction(nonzeroOrdinals.size());
            FieldContainer<GlobalIndexType> neighborGlobalDofIndicesSideRestriction(nonzeroOrdinals.size());
            for (int i=0; i<nonzeroOrdinals.size(); i++) {
              neighborGlobalValuesSideRestriction(i) = neighborGlobalValues(nonzeroOrdinals[i]);
              neighborGlobalDofIndicesSideRestriction(i) = neighborGlobalDofIndices(nonzeroOrdinals[i]);
            }

            // first, check that we agree on the number of global dof indices.
            if (neighborGlobalDofIndicesSideRestriction.size() != cellGlobalDofIndicesSideRestriction.size()) {
              success = false;
              cout << "neighborBasesAgreeOnSides() failure: # of global dof indices for cell and neighbor do not match.\n";
              cout << "cellGlobalDofIndicesSideRestriction:\n" << cellGlobalDofIndicesSideRestriction;
              cout << "neighborGlobalDofIndicesSideRestriction:\n" << neighborGlobalDofIndicesSideRestriction;
            
              cout << "cellGlobalDofIndices:\n" << cellGlobalDofIndices;
              cout << "neighborGlobalDofIndices:\n" << neighborGlobalDofIndices;
              
              cout << "cellGlobalValues:\n" << cellGlobalValues;
              cout << "neighborGlobalValues:\n" << neighborGlobalValues;
              continue;
            }
            
            // replace the containers containing dofs not of interest with the side containers that have
            cellGlobalDofIndices = cellGlobalDofIndicesSideRestriction;
            cellGlobalValues = cellGlobalValuesSideRestriction;
            
            neighborGlobalDofIndices = neighborGlobalDofIndicesSideRestriction;
            neighborGlobalValues = neighborGlobalValuesSideRestriction;
            
            // we do allow that the dofIndices lists are in differing order.  So we build a little lookup table here:
            map<GlobalIndexType, double > cellValues;
            for (int dofOrdinal=0; dofOrdinal<cellGlobalDofIndices.size(); dofOrdinal++) {
              GlobalIndexType globalDofIndex = cellGlobalDofIndices(dofOrdinal);
              cellValues[globalDofIndex] = cellGlobalValues(dofOrdinal);
            }
            
            bool failedHere = false;
            for (int dofOrdinal=0; dofOrdinal<neighborGlobalDofIndices.size(); dofOrdinal++) {
              GlobalIndexType globalDofIndex = neighborGlobalDofIndices(dofOrdinal);
              if (cellValues.find(globalDofIndex) == cellValues.end()) {
                cout << "neighborBasesAgreeOnSides() failure: cell and neighbor do not agree on global dof indices.\n";
                success = false;
                continue;
              }
              double value = neighborGlobalValues(dofOrdinal);
              double cellValue = cellValues[globalDofIndex];
              double diff = abs( value - cellValue );
              if (diff > tol) {
                success = false;
                failedHere = true;
                maxDiff = max(diff,maxDiff);
                if (reportErrors) {
                  cout << "values for point " << ptOrdinal << " and globalDofIndex " << globalDofIndex;
                  cout << " differ: " << value << " != " << cellValue << endl;
                }
              }
            }
            if (failedHere && reportErrors) {
              cout << "cellID " << cellID << "'s testPoints:\n" << testPointsRefCoords;
              
              cout << "neighbor cellID " << neighbor->cellID() << "'s testPoints:\n" << neighborTestPointsRefCoords;
              // for debugging, some console output:
              cout << "values for cellID " << cellID << ", traceID " << traceID << ", side " << sideOrdinal << ":\n";
              cout << cellGlobalValues;
              
              cout << "values for neighbor cellID " << neighbor->cellID() << ", traceID " << traceID << ", side " << neighborSideOrdinal << ":\n";
              cout << neighborGlobalValues;
              
              cout << "cellGlobalDofIndices:\n" << cellGlobalDofIndices;
              cout << "neighborGlobalDofIndices:\n" << neighborGlobalDofIndices;
              cout << "**** neighborBasesAgreeOnSides suppressing further output re. disagreement ****\n\n";
              
              reportErrors = false;
            }
          }
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
  int initialPolyOrder = 3;
  
  bool success = true;
  
  IntrepidExtendedTypes::EFunctionSpace hgrad = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  BasisPtr basis = BasisFactory::basisFactory()->getBasis(initialPolyOrder, quad_4.getKey(), hgrad);
  if (basis->getDegree() != initialPolyOrder) {  // since it's hgrad, that's a problem (hvol would be initialPolyOrder-1)
    success = false;
    cout << "testBasisRefinement: initial BasisFactory call returned a different-degree basis than expected..." << endl;
    cout << "testBasisRefinement: expected: " << initialPolyOrder << "; actual: " << basis->getDegree() << endl;
  }
  int additionalP = 4;
  basis = BasisFactory::basisFactory()->addToPolyOrder(basis, additionalP);
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
    int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    vector<double> zeroPoint = exactSolution.getPointForBCImposition();
    Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                           exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
    IndexType vertexIndex;
    if (! myMesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
    }
    exactSolution.setUseSinglePointBCForPHI(true, vertexIndex);
    
    IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
    // Poisson is set up such that the solution should be (x + 2y)^p
    
    solution.solve();
    
    FieldContainer<double> integral(myMesh->numElements());
    
    VarPtr phi_hat = PoissonBilinearForm::poissonBilinearForm()->varFactory().traceVar(PoissonBilinearForm::S_PHI_HAT);
    
    solution.integrateFlux(integral,phi_hat->ID());
    
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
    vector<double> zeroPoint = exactSolution.getPointForBCImposition();
    int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    
    Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                           exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
    IndexType vertexIndex;
    if (! myMesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
    }
    exactSolution.setUseSinglePointBCForPHI(true, vertexIndex);
    
    IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    
    Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
    // Poisson is set up such that the solution should be x + 2y
    
    VarPtr phi_hat = PoissonBilinearForm::poissonBilinearForm()->varFactory().traceVar(PoissonBilinearForm::S_PHI_HAT);
    
    double L2normFlux = exactSolution.L2NormOfError(solution, phi_hat->ID());
    //cout << "L2 norm of phi_hat " << L2normFlux << endl;
    
    double normDiff = abs(expectedValues[i] - L2normFlux);
    
    if (normDiff > tol) {
      success = false;
      cout << "Failure: Norm of phi_hat solution of Poisson was " << L2normFlux << "; expected " << expectedValues[i] << endl;
    }
    solution.solve();
    
    double fluxDiff = exactSolution.L2NormOfError(solution, phi_hat->ID());
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
  double tol = 1e-8; // had to increase for triangles, and again for single-point imposition, and again after the switch to Cholesky solve, and again for vesta.
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
      int horizontalCells = 2, verticalCells = 2;
      int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
      vector<double> zeroPoint = exactSolution.getPointForBCImposition();
      Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                             exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
      IndexType vertexIndex;
      if (! myMesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
      }
      exactSolution.setUseSinglePointBCForPHI(true, vertexIndex);
      
      IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
      
      Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
      // Poisson is set up such that the solution should be x + 2y
      
      double diff;
      
      VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);
      
      VarPtr phi_hat = PoissonBilinearForm::poissonBilinearForm()->varFactory().traceVar(PoissonBilinearForm::S_PHI_HAT);
      
      double L2norm = exactSolution.L2NormOfError(solution, phi->ID());
      double L2normFlux = exactSolution.L2NormOfError(solution, phi_hat->ID());
      
      solution.solve();
      diff = exactSolution.L2NormOfError(solution, phi->ID());
      //cout << "L2 Error in solution of phi Poisson (Sacado version): " << diff << endl;
      diff = diff / L2norm;
      
      //cout << "Relative L2 Error in phi solution of Poisson (Sacado version): " << diff << endl;
      if (diff > tol) {
        success = false;
        cout << "Failure: Error in phi solution of Poisson (Sacado version) was " << diff << "; tolerance set to " << tol << endl;
      }
      //cout << "L2 norm of phi_hat " << L2normFlux << endl;
      double fluxDiff = exactSolution.L2NormOfError(solution, phi_hat->ID());
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
   Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), order, order+1);
   
   IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
   
   Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
   
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
   Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), order, order+1);
   
   IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
   
   Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
   
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
  
  Teuchos::RCP<ExactSolution> exactLinear = PoissonExactSolution::poissonExactPolynomialSolution(1);
  Teuchos::RCP<ExactSolution> exactQuadratic = PoissonExactSolution::poissonExactPolynomialSolution(2);
  Teuchos::RCP<ExactSolution> exactCubic = PoissonExactSolution::poissonExactPolynomialSolution(3);
  Teuchos::RCP<ExactSolution> exactQuartic = PoissonExactSolution::poissonExactPolynomialSolution(4);
  
  //cout << "************************************************\n";
  //exactCubic->bilinearForm()->printTrialTestInteractions();
  //cout << "************************************************\n";
  
  vector<Teuchos::RCP<ExactSolution> > exactSolutions;
  exactSolutions.push_back(exactLinear);
  exactSolutions.push_back(exactQuadratic);
  exactSolutions.push_back(exactCubic);
  exactSolutions.push_back(exactQuartic);
  
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);
  
  for (int i=0; i<exactSolutions.size(); i++) {
    Teuchos::RCP<ExactSolution> exactSolution = exactSolutions[i];
    int order = exactSolution->H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
    Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, 3, 3, exactSolution->bilinearForm(), order, order+1);
    
    IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution->bilinearForm()));
    
    Solution solution(myMesh, exactSolution->ExactSolution::bc(), exactSolution->rhs(), ip);
    // Poisson is set up such that the solution should be x + y
    
    double diff;
    
    if (checkL2Norm) {
      // don't solve; just compute the error compared to a 0 solution
      double phiError = exactSolution->L2NormOfError(solution, phi->ID());
      FunctionPtr phiExact = exactSolution->exactFunctions().find(phi->ID())->second;
      double expected = phiExact->l2norm(myMesh);
      diff = abs(phiError - expected);
      //cout << "for 1x1 mesh, L2 norm of phi for PoissonExactSolution: " << phiError << endl;
      
      if (diff > tol) {
        success = false;
        cout << "Expected norm of exact solution to be " << expected << " but PoissonExactSolution gave " << phiError << endl;
      }
    } else {
      solution.solve();
      diff = exactSolution->L2NormOfError(solution, phi->ID());
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
  
  Teuchos::RCP<BilinearForm> bilinearForm = PoissonBilinearForm::poissonBilinearForm();
  
  Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, 1, 1, bilinearForm, order, order);
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
  
  Teuchos::RCP<Mesh> myMesh2x1 = MeshFactory::buildQuadMesh(quadPoints, 2, 1, bilinearForm, order, order);
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
  
  double tol = 2e-14;
  
  double maxDiff = 0;
  
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
  
  Teuchos::RCP<BilinearForm> bilinearForm = PoissonBilinearForm::poissonBilinearForm();
  
  Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, 1, 1, bilinearForm, order, order+1);
  
  Teuchos::RCP<ExactSolution> exactLinear = PoissonExactSolution::poissonExactPolynomialSolution(1);
  
  BCPtr bc = exactLinear->bc();
  RHSPtr rhs = exactLinear->rhs();
  IPPtr ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);
  
  Solution solution(myMesh, bc, rhs, ip);
  
  int PHI = phi->ID();
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
  
  // Poisson is set up such that the solution should be x + 2 * y - 1.5
  expectedSolnValues(0,0) = testPoints(0,0,0) + 2 * testPoints(0,0,1) - 1.5;
  expectedSolnValues(0,1) = testPoints(0,1,0) + 2 * testPoints(0,1,1) - 1.5;
  expectedSolnValues(0,2) = testPoints(0,2,0) + 2 * testPoints(0,2,1) - 1.5;
  
  solution.solve();
  
  solution.importGlobalSolution(); // so that the solnValues for cell 0 are available on all ranks
  solution.solutionValues(solnValues,PHI,testPoints);
  
  for (int i=0; i<numPoints; i++) {
    double diff = abs(expectedSolnValues(0,i) - solnValues(0,i));
    maxDiff = max(maxDiff, diff);
    if ( diff > tol ) {
      cout << "Solve 1-element Poisson: expected " << expectedSolnValues(0,i) << ", but soln was " << solnValues(0,i) << " -- diff=" << diff << endl;
      success = false;
    }
  }
  
  // now same thing, but larger mesh
  // in this test, we do use knowledge of the way the mesh elements get laid out
  // (they go top to bottom first, then left to right--i.e. columnwise)
  // the whole mesh in this test is the (0,1) square.
  int horizontalElements = 2, verticalElements = 2;
  Teuchos::RCP<Mesh> myMesh2x2 = MeshFactory::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bilinearForm, order, order+1);
  
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
  
  // diagnosing failure in Solution: do we succeed if all the refPoints are the same?
  // (picking point in middle of each element)
  testPoints(0,0,0) = 0.25;
  testPoints(0,0,1) = 0.25;
  testPoints(1,0,0) = 0.25;
  testPoints(1,0,1) = 0.75;
  testPoints(2,0,0) = 0.75;
  testPoints(2,0,1) = 0.25;
  testPoints(3,0,0) = 0.75;
  testPoints(3,0,1) = 0.75;
  
  expectedSolnValues.resize(numElements,numPointsPerElement);
  
  FunctionPtr phi_exact = exactLinear->exactFunctions().find(phi->ID())->second;
  
  for (int cellOrdinal=0; cellOrdinal<numElements; cellOrdinal++) {
    for (int ptOrdinal=0; ptOrdinal<numPointsPerElement; ptOrdinal++) {
      double x = testPoints(cellOrdinal,ptOrdinal,0);
      double y = testPoints(cellOrdinal,ptOrdinal,1);
      expectedSolnValues(cellOrdinal,ptOrdinal) = Function::evaluate(phi_exact, x, y);
    }
  }
  
  solution2x2.solve();
  solution2x2.importGlobalSolution();
  solnValues.resize(numElements,numPointsPerElement); // four elements, one test point each
  solution2x2.solutionValues(solnValues,phi->ID(),testPoints);
  
  for (int elemIndex=0; elemIndex<numElements; elemIndex++) {
    for (int ptIndex=0; ptIndex<numPointsPerElement; ptIndex++) {
      double diff = abs(expectedSolnValues(elemIndex,ptIndex) - solnValues(elemIndex,ptIndex));
      maxDiff = max(maxDiff, diff);
      if ( diff > tol ) {
        cout << "Solve 4-element Poisson: expected " << expectedSolnValues(elemIndex,ptIndex) << " at point (";
        cout << testPoints(elemIndex,0,0) << "," << testPoints(elemIndex,0,1) << "), but soln was " << solnValues(elemIndex,ptIndex) << " (diff of " << diff << ")" << endl;
        success = false;
      }
    }
  }
  
  // now try using the elementsForPoints variant of solutionValues
  solnValues.resize(numElements*numPointsPerElement); // four elements, one test point each
  testPoints.resize(numElements*numPointsPerElement,spaceDim);
  solution2x2.solutionValues(solnValues,phi->ID(),testPoints);
  
  for (int elemIndex=0; elemIndex<numElements; elemIndex++) {
    for (int ptIndex=0; ptIndex<numPointsPerElement; ptIndex++) {
      int solnIndex = elemIndex*numPointsPerElement + ptIndex;
      double diff = abs(expectedSolnValues(elemIndex,ptIndex) - solnValues(solnIndex));
      maxDiff = max(maxDiff, diff);
      if ( diff > tol ) {
        cout << "Solve 4-element Poisson: expected " << expectedSolnValues(elemIndex,ptIndex) << ", but soln was " << solnValues(solnIndex) << " (using elementsForPoints solutionValues)" << endl;
        success = false;
      }
    }
  }
  
  // would be better to actually do the meshing, etc. with reference to the BC & RHS given by PoissonExactSolution,
  // but we do know that these are the same, so we'll just use the solutions we already have....
  double phiError = exactLinear->L2NormOfError(solution, phi->ID());
  //cout << "for 1x1 mesh, L2 error in phi for PoissonExactSolutionLinear: " << phiError << endl;
  phiError = exactLinear->L2NormOfError(solution2x2, phi->ID());
  //cout << "for 2x2 mesh, L2 error in phi for PoissonExactSolutionLinear: " << phiError << endl;
  
//  cout << "maxDiff: " << maxDiff << endl;
  
  return success;
}

bool MeshTestSuite::testDofOrderingFactory() {
  bool success = true;
  int polyOrder = 3;
  
  BFPtr bilinearForm = PoissonBilinearForm::poissonBilinearForm();
  
  VarPtr phi_hat = bilinearForm->varFactory().traceVar(PoissonBilinearForm::S_PHI_HAT);
  
  Teuchos::RCP<DofOrdering> conformingOrdering,nonConformingOrdering;
  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  
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
  conformingOrderingCopy = dofOrderingFactory.pRefineTrial(conformingOrdering, quad_4, 0); // don't really refine
  
  if (conformingOrderingCopy.get() != conformingOrdering.get() ) {
    cout << "testDofOrderingFactory: conformingOrdering with pRefine==0 differs from original." << endl;
    success = false;
  }
  
  int pToAdd = 3;
  
  conformingOrderingCopy = dofOrderingFactory.pRefineTrial(conformingOrdering, quad_4, pToAdd);
  
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
  BasisPtr lowerBasis = nonConformingOrderingLowerOrder->getBasis(phi_hat->ID(),higherSideInLowerElementNonConforming);
  BasisPtr lowerBasisOtherSide = nonConformingOrderingLowerOrder->getBasis(phi_hat->ID(),lowerElementOtherSideNonConforming);
  if ( lowerBasis->getDegree() == lowerBasisOtherSide->getDegree() ) {
    success = false;
    cout << "FAILURE: After matchSides (non-conforming), the lower-order side doesn't appear to have been refined." << endl;
  }
  BasisPtr higherBasis = nonConformingOrderingHigherOrder->getBasis(phi_hat->ID(),lowerSideInHigherElementNonConforming);
  if (lowerBasis.get() != higherBasis.get()) {
    success = false;
    cout << "FAILURE: After matchSides (non-conforming), sides have differing bases." << endl;
  }
  
  lowerBasis = conformingOrderingLowerOrder->getBasis(phi_hat->ID(),higherSideInLowerElementConforming);
  lowerBasisOtherSide = conformingOrderingLowerOrder->getBasis(phi_hat->ID(),lowerElementOtherSideConforming);
  if ( lowerBasis->getDegree() == lowerBasisOtherSide->getDegree() ) {
    success = false;
    cout << "FAILURE: After matchSides (conforming), the lower-order side doesn't appear to have been refined." << endl;
  }
  higherBasis = conformingOrderingHigherOrder->getBasis(phi_hat->ID(),lowerSideInHigherElementConforming);
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
  nonConformingOrderingLowerOrder = dofOrderingFactory.pRefineTrial(nonConformingOrderingLowerOrder, quad_4, pToAdd);
  nonConformingOrderingHigherOrder = dofOrderingFactory.trialOrdering(polyOrder+pToAdd, quad_4, false);
  if ( nonConformingOrderingLowerOrder.get() != nonConformingOrderingHigherOrder.get() ) {
    success = false;
    cout << "FAILURE: After p-refinement of upgraded Ordering (non-conforming), DofOrdering doesn't match a fresh one with that p-order." << endl;
  }
  
  conformingOrderingLowerOrder = dofOrderingFactory.pRefineTrial(conformingOrderingLowerOrder, quad_4, pToAdd);
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
  int H1Order = exactSolution.H1Order();
  int horizontalCells = 2; int verticalCells = 2;
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
  vector<double> zeroPoint = exactSolution.getPointForBCImposition();
  IndexType vertexIndex;
  if (! mesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertexIndex not found!");
  }
  exactSolution.setUseSinglePointBCForPHI(true, vertexIndex);
  
  vector<GlobalIndexType> cellsToRefine;
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
  IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  Solution origSolution(mesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
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
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);

  int trialID = phi->ID();
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
    Teuchos::RCP<Element> parent = mesh->getElement(cellsToRefine[i]);
    Teuchos::RCP<Element> child = parent->getChild(0);
    if ( parent->elementType() != child->elementType() ) {
      success = false;
      cout << "FAILURE: Expected noRefinementPattern to produce no change in element type. \n";
    }
  }
  
  // try solving --> make sure that we get the same solution before and after "refinement"
  Solution solution(mesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
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
  Teuchos::RCP<Mesh> fineMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells*2, verticalCells*2, exactSolution.bilinearForm(), H1Order, H1Order+1);
  BCPtr fineMeshBC = Teuchos::rcp( new BC(*exactSolution.ExactSolution::bc()) ); // copy
  
  IndexType fineMeshVertexIndex;
  if (! fineMesh->getTopology()->getVertexIndex(zeroPoint, fineMeshVertexIndex)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertexIndex not found!");
  }
  FunctionPtr phiExact = exactSolution.exactFunctions().find(phi->ID())->second;
  double value = Function::evaluate(phiExact, zeroPoint[0], zeroPoint[1]);
  fineMeshBC->addSinglePointBC(phi->ID(), value, fineMeshVertexIndex); // replaces existing

  origSolution = Solution(fineMesh, fineMeshBC, exactSolution.ExactSolution::rhs(), ip);
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
  
  double refinedError = exactSolution.L2NormOfError(solution,phi->ID());
  double fineError = exactSolution.L2NormOfError(origSolution,phi->ID());
  
  //  cout << "refinedError:" << refinedError << endl;
  //  cout << "fineError:" << fineError << endl;
  
  double diff = abs(refinedError - fineError);
  
  if (diff > tol) {
    cout << "FAILURE: after uniform regular refinement, L2 error different from originally fine mesh.\n";
    cout << "Difference of L2 error in refined vs. originally fine mesh: " << diff << endl;
    success = false;
    
    SolutionPtr origSolnPtr = Teuchos::rcp(&origSolution, false);
    SolutionPtr refinedSolnPtr = Teuchos::rcp(&solution, false);
    
    HDF5Exporter::exportSolution("/tmp/", "originalFine", origSolnPtr);
    HDF5Exporter::exportSolution("/tmp/", "refinedSoln", refinedSolnPtr);
  }
  
  // TODO: work out how to fix solution.equals to work with meshes whose cells may be in different orders...
  /*if ( ! solution.equals(origSolution, 1e-8) ) { // start with very relaxed tol... TODO: tighten this once we're passing...
   success = false;
   cout << "FAILURE: Expected solution of fine mesh and refined mesh to be equal. \n";
   }*/
  
  cellsToRefine.clear();
  
  // start with a fresh 2x2 mesh:
  horizontalCells = 2; verticalCells = 2;
  mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 8; // 1 to start!
  for (int i=0; i<numRefinements; i++) {
    vector< pair<int,int> > descendants = mesh->getElement(0)->getDescendantsForSide(1);
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
  solution = Solution(mesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
  solution.solve();
  
  refinedError = exactSolution.L2NormOfError(solution,phi->ID());
  
  if (refinedError > tol) {
    success = false;
    cout << "FAILURE: after 'deep' refinement for exactly recoverable solution, L2 error greater than tolerance.\n";
    cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
  }
  
  //  solution.writeToFile(PoissonBilinearForm::PHI, "phi_refined_again.dat");
  
  // try to reproduce a parity error discovered by Jesse when enforcing 1-irregularity
  horizontalCells = 1; verticalCells = 2;
  mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);
  
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
  IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  
  vector<GlobalIndexType> cellsToRefine;
  cellsToRefine.clear();
  
  // start with a fresh 2x2 mesh:
  horizontalCells = 2; verticalCells = 2;
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+2);
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: initial mesh fails consistency check.\n";
  }

//  cout << "About to refine mesh.\n";
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 2;
  for (int i=0; i<numRefinements; i++) {
    vector< pair<int,int> > descendants = mesh->getElement(0)->getDescendantsForSide(1);
    int numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.push_back(cellID);
    }
//    cout << "h-refining east side: refIndex " << i << endl;
//    Camellia::print("h-refining", cellsToRefine);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    // same thing for north side
    descendants = mesh->getElement(0)->getDescendantsForSide(2);
    numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.push_back(cellID);
    }
//    cout << "h-refining north side: refIndex " << i << endl;
//    Camellia::print("h-refining", cellsToRefine);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
//  cout << "finished mesh refinements.\n";
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: after 'deep' refinement, mesh fails consistency check.\n";
  }
  
//  cout << "About to create solution object on refined mesh.\n";
  Solution solution(mesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
//  cout << "About to solve.\n";
  solution.solve();
  
  double refinedError = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U_ID);
  
  // relaxed tolerance
  double tol = 1e-1;
  if ((refinedError > tol) || (refinedError != refinedError)) { // second compare: is refinedError NaN?
    success = false;
    cout << "FAILURE: after 'deep' refinement for smooth solution, L2 error greater than tolerance.\n";
    cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
  }
  
  // solution.writeFieldsToFile(ConfusionBilinearForm::U, "confusion_demo.m");
  
  return success;
}

bool MeshTestSuite::testHUnrefinementForConfusion() {
  bool success = true;
  
  double tol = 1e-14;
  
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
  
  // before we hRefine, compute a solution for comparison after refinement
  IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  
  set<GlobalIndexType> cellsToRefine;
  cellsToRefine.clear();
  
  int horizontalCells = 1; int verticalCells = 1;
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+2);
  
  Solution solution = Solution(mesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
  
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 2;
  for (int i=0; i<numRefinements; i++) {
    solution.solve();
    double errorBeforeRefinement = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U_ID);
    vector< pair<int,int> > descendants = mesh->getElement(0)->getDescendantsForSide(1);
    int numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.insert(cellID);
    }
    
    cout << "b4 refining num edge cID entries = " << mesh->numEdgeToCellIDEntries() << endl;
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    cout << "b4 unref num edge cID entries = " << mesh->numEdgeToCellIDEntries() << endl;
    mesh->hUnrefine(cellsToRefine);
    cout << "num edge cID entries = " << mesh->numEdgeToCellIDEntries() << endl;
    if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
      success = false;
      cout << "FAILURE: after unrefinement, mesh fails consistency check.\n";
    }
    solution.solve();
    double errorAfterUnrefinement = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U_ID);
    if ( abs(errorAfterUnrefinement - errorBeforeRefinement) > tol) {
      success = false;
      cout << "errorAfterUnrefinement != errorBeforeRefinement: " << errorAfterUnrefinement << " != " << errorBeforeRefinement << endl;
    }
    // redo the refinement:
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    // same thing for north side
    solution.solve();
    errorBeforeRefinement = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U_ID);
    descendants = mesh->getElement(0)->getDescendantsForSide(2);
    numDescendants = descendants.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendants; j++ ) {
      int cellID = descendants[j].first;
      cellsToRefine.insert(cellID);
    }
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    mesh->hUnrefine(cellsToRefine);
    if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
      success = false;
      cout << "FAILURE: after unrefinement, mesh fails consistency check.\n";
    }
    solution.solve();
    errorAfterUnrefinement = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U_ID);
    if ( abs(errorAfterUnrefinement - errorBeforeRefinement) > tol) {
      success = false;
      cout << "errorAfterUnrefinement != errorBeforeRefinement: " << errorAfterUnrefinement << " != " << errorBeforeRefinement << endl;
    }
    // redo the refinement:
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
  
  if (! MeshTestUtility::checkMeshConsistency(mesh) ) {
    success = false;
    cout << "FAILURE: after unrefinement, mesh fails consistency check.\n";
  }
  
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
  
  Teuchos::RCP<Mesh> mesh1 = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order, H1Order+1);
  Teuchos::RCP<Mesh> mesh2 = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order+1, H1Order+2);
  
  vector<double> zeroPointPolynomial = exactPolynomial.getPointForBCImposition();
  vector<double> zeroPointExponential = exactPolynomial.getPointForBCImposition();
  IndexType vertexIndexPolynomial, vertexIndexExponential;
  if (! mesh1->getTopology()->getVertexIndex(zeroPointPolynomial, vertexIndexPolynomial) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  if (! mesh1->getTopology()->getVertexIndex(zeroPointPolynomial, vertexIndexExponential) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  exactPolynomial.setUseSinglePointBCForPHI(true, vertexIndexPolynomial);
  exactExponential.setUseSinglePointBCForPHI(true, vertexIndexExponential);

  IPPtr ip0 = Teuchos::rcp(new MathInnerProduct(exactExponential.bilinearForm()));
  vector<GlobalIndexType> cellsToRefine;
  Solution solution1(mesh1, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip0);
  solution1.solve();
  Solution solution2(mesh2, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip0);
  solution2.solve();
  
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);
  
  double error1 = exactPolynomial.L2NormOfError(solution1, phi->ID(),15); // high fidelity L2norm
  double error2 = exactPolynomial.L2NormOfError(solution2, phi->ID(),15); // high fidelity L2norm
  if (error1 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of sufficiently high degree..." << endl;
  }
  if (error2 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of more than sufficiently high degree... (tol: " << tol << "; error2: " << error2 << ")" << endl;
  }

  Teuchos::RCP<Mesh> mesh3 = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), H1Order, H1Order+1);
  DofOrderingPtr trialOrdering = mesh3->getElementType(refinedCellID)->trialOrderPtr;
  mesh3->getDofOrderingFactory().trialPolyOrder(trialOrdering);
  int polyOrderBeforeRefinement = mesh3->getDofOrderingFactory().trialPolyOrder(trialOrdering);
  cellsToRefine.push_back(refinedCellID);
  mesh3->pRefine(cellsToRefine);
  trialOrdering = mesh3->getElementType(refinedCellID)->trialOrderPtr;
  int polyOrderAfterRefinement = mesh3->getDofOrderingFactory().trialPolyOrder(trialOrdering);
  
  if (polyOrderAfterRefinement != polyOrderBeforeRefinement + 1) {
    cout << "poly order after refinement is not 1 greater than before refinement.\n";
    cout << "poly order before refinement: " << polyOrderBeforeRefinement << endl;
    cout << "poly order after refinement:  " << polyOrderAfterRefinement << endl;
    success = false;
  }
  
//  MeshPtr meshes[3];
//  meshes[0] = mesh1;
//  meshes[1] = mesh2;
//  meshes[2] = mesh3;
//  for (int cellID=0; cellID<horizontalCells * verticalCells; cellID++) {
//    cout << "cellID " << cellID << ", poly order:\n";
//    for (int meshNumber = 1; meshNumber<=3; meshNumber++) {
//      MeshPtr mesh = meshes[meshNumber-1];
//      DofOrderingPtr trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
//      int polyOrder = mesh->getDofOrderingFactory().trialPolyOrder(trialOrdering);
//      cout << "mesh " << meshNumber << ": " << polyOrder << endl;
//    }
//  }

  if ( ! MeshTestUtility::checkMeshConsistency(mesh3) ) {
    cout << "After p-refinement, mesh consistency test FAILED." << endl;
    success = false;
  }
  
  Solution solution3(mesh3, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(),ip0);
  solution3.solve();
//  double error3 = exactPolynomial.L2NormOfError(solution3, PoissonBilinearForm::PHI,15); // high fidelity L2norm
  //cout << "refined mesh error: " << error3 << endl;
  
  for (int cellID=0; cellID< horizontalCells * verticalCells; cellID++) {
    FieldContainer<double> expectedSolnDofs; // for phi
    if (cellID != refinedCellID) {
      solution1.solnCoeffsForCellID(expectedSolnDofs,cellID,phi->ID());
    } else {
      solution2.solnCoeffsForCellID(expectedSolnDofs,cellID,phi->ID());
    }
    FieldContainer<double> actualSolnDofs;
    solution3.solnCoeffsForCellID(actualSolnDofs,cellID,phi->ID());
    if ( actualSolnDofs.size() != expectedSolnDofs.size() ) {
      cout << "FAILURE: for cellID " << cellID << ", actualSolnDofs.size() != expectedSolnDofs.size() (";
      cout << actualSolnDofs.size() << " vs. " << expectedSolnDofs.size() << ")" << endl;
    } else {
      for (int i=0; i<actualSolnDofs.size(); i++) {
        double diff = abs(actualSolnDofs(i)-expectedSolnDofs(i));
        if (diff > tol * 10 ) { // * 10 because we can be a little more tolerant of the Dof values than, say, the overall L2 error.
          cout << "FAILURE: In cellID " << cellID << ", p-refined mesh differs in phi solution from expected ";
          cout << "in basis ordinal " << i << " (diff=" << diff << ")" << endl;
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
  mesh1 = MeshFactory::buildQuadMesh(quadPoints, 4, 3, exactExponential.bilinearForm(), H1Order, H1Order+1);
  mesh2 = MeshFactory::buildQuadMesh(quadPoints, 4, 3, exactExponential.bilinearForm(), H1Order+1, H1Order+2);
  ip0 = Teuchos::rcp(new MathInnerProduct(exactExponential.bilinearForm()));
  cellsToRefine.clear();
  for (int i=5; i<8; i++) {
    cellsToRefine.push_back(i);
  }
  mesh1->pRefine(cellsToRefine);
  cellsToRefine.clear();
  for (int i=8; i<mesh1->numElements(); i++) {
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
  
  solution1 = Solution(mesh1, exactExponential.ExactSolution::bc(), exactExponential.ExactSolution::rhs(), ip0);
  solution1.solve();
  solution2 = Solution(mesh2, exactExponential.ExactSolution::bc(), exactExponential.ExactSolution::rhs(), ip0);
  solution2.solve();
  
  error1 = exactExponential.L2NormOfError(solution1, phi->ID(),15); // high fidelity L2norm
  error2 = exactExponential.L2NormOfError(solution2, phi->ID(),15); // high fidelity L2norm
  double diff = abs(error1-error2);
  if (diff > tol) {
    success = false;
    cout << "FAILURE: Refined everywhere mesh gives different solution than fresh mesh with same orders." << endl;
  }
  for (int i=0; i<exactSolutions.size(); i++) {
    PoissonExactSolution exactSolution = exactSolutions[i];
    
    Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, sqrtElements, sqrtElements, exactSolution.bilinearForm(), H1Order, H1Order+1);
    
    cellsToRefine.clear();
    cellsToRefine.push_back(2);
    
    IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
    Solution solution(myMesh, exactSolution.ExactSolution::bc(), exactSolution.ExactSolution::rhs(), ip);
    
    double L2norm = exactSolution.L2NormOfError(solution, phi->ID(),15); // high fidelity L2norm
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
    diff = exactSolution.L2NormOfError(solution, phi->ID(),15);
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
    diff = exactSolution.L2NormOfError(solution, phi->ID(),15);
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
    diff = exactSolution.L2NormOfError(solution, phi->ID(),15);
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
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), order, order+pToAdd);
  
  vector<double> zeroPoint = exactPolynomial.getPointForBCImposition();
  IndexType vertexIndex;
  if (! mesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  exactPolynomial.setUseSinglePointBCForPHI(true, vertexIndex);
  
  if ( ! MeshTestUtility::checkMeshConsistency(mesh) ) {
    cout << "In singlePointBC test, mesh consistency test FAILED for non-conforming mesh." << endl;
    success = false;
  }
  IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution(mesh, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip);
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);

  double L2norm = exactPolynomial.L2NormOfError(solution, phi->ID());
  solution.solve();
  
  double error = exactPolynomial.L2NormOfError(solution, phi->ID(),15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using single-point BC, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "(L2 norm of solution: " << L2norm << ")\n";
  } else {
    //cout << "Success! Single-point BC Poisson error: " << error << endl;
  }
  
  PoissonExactSolution exactPolynomialConforming = PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 1, true); // use conforming traces
  
  exactPolynomialConforming.setUseSinglePointBCForPHI(true, vertexIndex); // can reuse vertexIndex because we have the same mesh layout
  
  mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                    exactPolynomialConforming.bilinearForm(), order, order+pToAdd);
  if ( ! MeshTestUtility::checkMeshConsistency(mesh) ) {
    cout << "In singlePointBC test, mesh consistency test FAILED for conforming mesh." << endl;
    success = false;
  }
  
  Teuchos::RCP<ElementType> elemTypePtr = mesh->getElement(0)->elementType();
  Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
  if ( checkDofOrderingHasNoOverlap(trialOrdering) ) {
    cout << "FAILURE: expected trialOrdering (which is conforming) not to be a 1-1 map, but it is." << endl;
    success = false;
  }
  ip = Teuchos::rcp(new MathInnerProduct(exactPolynomialConforming.bilinearForm()));
  Solution solutionConforming = Solution(mesh, exactPolynomialConforming.ExactSolution::bc(),
                                         exactPolynomialConforming.ExactSolution::rhs(), ip);
  solutionConforming.solve();
  
  error = exactPolynomialConforming.L2NormOfError(solutionConforming, phi->ID(), 15); // high fidelity L2norm
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
  
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 0); // 0 means a zero solution
  int order = 2; // H1 order ==> L2 order of order-1.
  int horizontalCells = 2; int verticalCells = 2;
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactPolynomial.bilinearForm(), order, order+1);

  vector<double> zeroPoint = exactPolynomial.getPointForBCImposition();
  IndexType vertexIndex;
  if (! mesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  exactPolynomial.setUseSinglePointBCForPHI(true, vertexIndex);
  
  IPPtr ip0 = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution1(mesh, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip0);
  solution1.solve();
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);

  double error1 = exactPolynomial.L2NormOfError(solution1, phi->ID(),15); // high fidelity L2norm
  if (error1 > tol) {
    success = false;
    cout << "FAILURE: Failed to resolve exact polynomial on mesh of sufficiently high degree..." << endl;
  }
  // give each element its own type
  Teuchos::RCP<ElementType> elemTypePtr = mesh->getElement(0)->elementType();
  for (int i=0; i<mesh->numElements(); i++) {
    Teuchos::RCP<ElementType> newElemTypePtr = Teuchos::rcp(new ElementType(elemTypePtr->trialOrderPtr,
                                                                            elemTypePtr->testOrderPtr,
                                                                            elemTypePtr->cellTopoPtr));
    mesh->getElement(i)->setElementType(newElemTypePtr);
  }
  // and rebuild the mesh data structures:
  mesh->rebuildLookups();
  Solution solution2(mesh, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip0);
  solution2.solve();
  double error2 = exactPolynomial.L2NormOfError(solution2, phi->ID(),15); // high fidelity L2norm
  if (error2 > tol) {
    success = false;
    cout << "FAILURE: Solution failed to solve on uniform mesh with two element types..." << endl;
    ostringstream fileName;
    fileName << "PoissonPhiSolution_FAILURE.p=" << order-1 << "." << horizontalCells << "x" << verticalCells << ".dat";
    solution2.writeToFile(phi->ID(), fileName.str());
    cout << "Wrote solution out to disk at: " << fileName.str() << endl;
  }
  return success;
}

bool MeshTestSuite::testSolutionForSingleElementUpgradedSide() {
  bool success = true;
  double tol = 5e-13;
  
  PoissonExactSolution exactPolynomial(PoissonExactSolution::POLYNOMIAL, 0);
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
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, 1, 1, exactPolynomial.bilinearForm(), order, order);
  
  vector<double> zeroPoint = exactPolynomial.getPointForBCImposition();
  IndexType vertexIndex;
  if (! mesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  exactPolynomial.setUseSinglePointBCForPHI(true,vertexIndex);

  IPPtr ip = Teuchos::rcp(new MathInnerProduct(exactPolynomial.bilinearForm()));
  Solution solution(mesh, exactPolynomial.ExactSolution::bc(), exactPolynomial.ExactSolution::rhs(), ip);
  solution.solve();
  
  VarPtr phi = PoissonBilinearForm::poissonBilinearForm()->varFactory().fieldVar(PoissonBilinearForm::S_PHI);

  double error = exactPolynomial.L2NormOfError(solution, phi->ID(),15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using single-point BC, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "Single-point BC Poisson error (standard element): " << error << endl;
  }
  
  Teuchos::RCP<ElementType> elemTypePtr = mesh->getElement(0)->elementType();
  Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
  CellTopoPtr cellTopo = elemTypePtr->cellTopoPtr;
  // create a second mesh so we can have another DofOrdering of higher order to match...
  Teuchos::RCP<Mesh> mesh2 = MeshFactory::buildQuadMesh(quadPoints, 1, 1, exactPolynomial.bilinearForm(), order+1, order);
  Teuchos::RCP<DofOrdering> trialOrderingToMatch = mesh2->getElement(0)->elementType()->trialOrderPtr;
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
  mesh->getElement(0)->setElementType(elemTypePtr);
  mesh->rebuildLookups();
  solution.solve();
  
  error = exactPolynomial.L2NormOfError(solution, phi->ID(),15); // high fidelity L2norm
  if (error > tol) {
    success = false;
    cout << "FAILURE: When using upgraded side, failed to resolve linear polynomial... (error: " << error << ")" << endl;
    cout << "Single-point BC Poisson error (upgraded side): " << error << endl;
    ostringstream fileName;
    fileName << "testSolutionForSingleElementUpgradedSide_FAILURE.p=" << order-1 << ".dat";
    solution.writeToFile(phi->ID(), fileName.str());
    cout << "Wrote solution out to disk at: " << fileName.str() << endl;
  }
  
  return success;
}

bool MeshTestSuite::checkDofOrderingHasNoOverlap(Teuchos::RCP<DofOrdering> dofOrdering) {
  // checkDofOrderingHasNoOverlap returns true if no two (varID,basisOrdinal,sideIndex) tuples map to same dofIndex
  // (won't be true for orderings that have H1 dofs that are conforming)
  bool noOverlap = true;
  set<int> dofIndices;
  set<int> varIDs = dofOrdering->getVarIDs();
  set<int>::iterator varIDIt;
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
  
  Teuchos::RCP<BilinearForm> bilinearForm = PoissonBilinearForm::poissonBilinearForm();
  
  int H1Order = 2;
  int horizontalCells = 2; int verticalCells = 1;
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bilinearForm, H1Order, H1Order+1);
  
  vector<GlobalIndexType> cellsToRefine;
  cellsToRefine.push_back(0);
  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  
  Teuchos::RCP<Element> elem = mesh->getElement(0);
  int numChildrenExpected = 4;
  if (elem->numChildren() != numChildrenExpected) {
    success = false;
    cout << "After refinement, wrong number of children." << endl;
  }
  
  int sideIndex = 0;
  vector< vector< pair<unsigned,unsigned> > > expectedChildrenForSide(4);
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
    vector< pair< unsigned, unsigned> > childrenForSide = elem->childIndicesForSide(sideIndex);
    if (! vectorPairsEqual(childrenForSide, expectedChildrenForSide[sideIndex])) {
      success = false;
      cout << "FAILURE: testRefinementPattern childrenForSide not the expected for side " << sideIndex << endl;
    }
  }
  
  cellsToRefine.clear();
  
  
  return success;
}

bool MeshTestSuite::vectorPairsEqual( vector< pair<unsigned,unsigned> > &first, vector< pair<unsigned,unsigned> > &second) {
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
  vector<double> zeroPoint = exactSolution.getPointForBCImposition();
  int order = exactSolution.H1Order(); // matters for getting enough cubature points, and of course recovering the exact solution
  Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                         exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
  IndexType vertexIndex;
  if (! myMesh->getTopology()->getVertexIndex(zeroPoint, vertexIndex) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found!");
  }
  exactSolution.setUseSinglePointBCForPHI(true, vertexIndex);
  
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
  
  // now try it for a case that has failed in the past:
  quadPoints(0,0) = 4.000000e+00;
  quadPoints(0,1) = 5.288500e-01;
  quadPoints(1,0) = 4.18750e+00;
  quadPoints(1,1) = 5.288500e-01;
  quadPoints(2,0) = 4.18750e+00;
  quadPoints(2,1) = 0.764425;
  quadPoints(3,0) = 4.000000e+00;
  quadPoints(3,1) = 0.764425;
  
  horizontalCells = 1;
  verticalCells = 1;
  myMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                      exactSolution.bilinearForm(), order, order+pToAdd, triangulate);
  // two uniform refinements:
  myMesh->hRefine(myMesh->getActiveCellIDs(), RefinementPattern::regularRefinementPatternQuad());
  myMesh->hRefine(myMesh->getActiveCellIDs(), RefinementPattern::regularRefinementPatternQuad());
  
  points.resize(1,spaceDim);
  points[0] = 4.04226;
  points[1] = 0.646637;
  
  elements = myMesh->elementsForPoints(points);
  
  // we don't really have anything to check at this point: if it crashes, that's the problem.
  // but even prior to fixing the instigating issue, it doesn't crash, so there must be something
  // else going on...
  //  cout << "In new mesh test, matched element " << elements[0]->cellID() << endl;
  
  return success;
}
bool MeshTestSuite::testJesseMultiBasisRefinement(){
  bool success = true;
  int polyOrder = 2;
  PoissonExactSolution exactSolution(PoissonExactSolution::POLYNOMIAL, polyOrder);
  int order = exactSolution.H1Order();
  int pToAdd = 1;
  MeshPtr mesh = MeshUtilities::buildUnitQuadMesh(4,2, exactSolution.bilinearForm(), order, order+pToAdd);
  
  ////////////////////////////////////////////////////////////////////
  // REFINE MESH TO TRIGGER EXCEPTION
  ////////////////////////////////////////////////////////////////////
  /*
   for (int i = 0;i<3;i++){
   vector<int> xC;
   vector<int> yC;
   vector<int> rC;
   switch (i){
   case 0:
   rC.push_back(0);
   rC.push_back(2);
   yC.push_back(1);
   yC.push_back(3);
   break;
   case 1:
   yC.push_back(4);
   yC.push_back(6);
   yC.push_back(10);
   rC.push_back(15);
   break;
   case 2:
   break;
   case 3:
   
   break;
   }
   mesh->hRefine(xC, RefinementPattern::xAnisotropicRefinementPatternQuad());
   mesh->hRefine(yC, RefinementPattern::yAnisotropicRefinementPatternQuad());
   mesh->hRefine(rC, RefinementPattern::regularRefinementPatternQuad());
   mesh->enforceOneIrregularity();
   for (int i = 0;i<mesh->numActiveElements();i++){
   int cellID = mesh->getActiveElement(i)->cellID();
   vector<double> c = mesh->getCellCentroid(cellID);
   cout << "centroid of cell " << cellID << " = " << c[0] << ", " << c[1] << endl;
   }
   cout << endl;
   }
   */
  for (int i = 0;i<5;i++){
    vector<GlobalIndexType> xC;
    vector<GlobalIndexType> yC;
    vector<GlobalIndexType> rC;
    switch (i){
      case 0:
        rC.push_back(2);
        rC.push_back(4);
        rC.push_back(6);
        break;
      case 1:
        rC.push_back(9);
        rC.push_back(12);
        rC.push_back(13);
        rC.push_back(16);
        rC.push_back(17);
        break;
      case 2:
        rC.push_back(20);
        rC.push_back(21);
        rC.push_back(24);
        rC.push_back(25);
        rC.push_back(28);
        rC.push_back(29);
        rC.push_back(32);
        yC.push_back(30);
        break;
      case 3:
        rC.push_back(33);
        rC.push_back(36);
        rC.push_back(37);
        rC.push_back(43);
        rC.push_back(46);
        rC.push_back(47);
        rC.push_back(51);
        rC.push_back(54);
        rC.push_back(55);
        rC.push_back(58);
        rC.push_back(60);
        rC.push_back(61);
        
        yC.push_back(18);
        yC.push_back(19);
        yC.push_back(26);
        yC.push_back(31);
        yC.push_back(34);
        yC.push_back(35);
        yC.push_back(38);
        yC.push_back(39);
        yC.push_back(40);
        yC.push_back(41);
        yC.push_back(50);
        yC.push_back(52);
        yC.push_back(56);
        yC.push_back(57);
        break;
      case 4:
        rC.push_back(93);  // refinement breaks on this one
        break;
    }
    //    if (i==4) {
    //      FieldContainer<double> vertices(4,2);
    //      mesh->verticesForCell(vertices, 78);
    //      cout << "vertices for cell 78:\n" << vertices;
    //      mesh->verticesForCell(vertices, 91);
    //      cout << "vertices for cell 91:\n" << vertices;
    //      mesh->verticesForCell(vertices, 93);
    //      cout << "vertices for cell 93:\n" << vertices;
    //      GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/multiBasisRefinementTest.dat", mesh);
    //    }
    mesh->hRefine(xC, RefinementPattern::xAnisotropicRefinementPatternQuad());
    mesh->hRefine(yC, RefinementPattern::yAnisotropicRefinementPatternQuad());
    mesh->hRefine(rC, RefinementPattern::regularRefinementPatternQuad());
    mesh->enforceOneIrregularity();
  }
  return success;
}

// test a second crash that is observed in anisotropic NavierStokes refinement
bool MeshTestSuite::testJesseAnisotropicRefinement(){
  int order = 1;
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  
  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  
  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  
  RHSPtr rhs = RHS::rhs();
  BCPtr bc = BC::bc();
  
  ////////////////////   CREATE BCs   ///////////////////////
  
  int pToAdd = 1;
  MeshPtr mesh = MeshUtilities::buildUnitQuadMesh(2, convectionBF, order, order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  
  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  ////////////////////////////////////////////////////////////////////
  // REFINE MESH TO TRIGGER EXCEPTION
  ////////////////////////////////////////////////////////////////////
  vector<ElementPtr> elems = mesh->activeElements();
  
  // create "swastika" mesh
  vector<GlobalIndexType> xC,yC;
  yC.push_back(1);yC.push_back(2);
  xC.push_back(0);xC.push_back(3);
  
  mesh->hRefine(xC, RefinementPattern::xAnisotropicRefinementPatternQuad());
  mesh->hRefine(yC, RefinementPattern::yAnisotropicRefinementPatternQuad());
  elems = mesh->activeElements();
  
  // trigger naive algorithm infinite loop (deadlock?)
  xC.clear();yC.clear();
  xC.push_back(6);
  mesh->hRefine(xC,RefinementPattern::xAnisotropicRefinementPatternQuad());
  //  mesh->hRefine(xC,RefinementPattern::regularRefinementPatternQuad());
  //  mesh->enforceOneIrregularity();
  
  RefinementStrategy refinementStrategy(solution,.2);
  bool success = refinementStrategy.enforceAnisotropicOneIrregularity(xC,yC);
  
  
  return success;
}

// test a second crash that is observed in anisotropic NavierStokes refinement
bool MeshTestSuite::testPRefinementAdjacentCells(){
  bool success = true;
  int order = 1;
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  
  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  
  RHSPtr rhs = RHS::rhs();
  BCPtr bc = BC::bc();
  
  ////////////////////   CREATE BCs   ///////////////////////
  
  int pToAdd = 1;
  MeshPtr mesh = MeshUtilities::buildUnitQuadMesh(2, convectionBF, order, order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  
  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  ////////////////////////////////////////////////////////////////////
  // REFINE MESH TO TRIGGER EXCEPTION
  ////////////////////////////////////////////////////////////////////
  
  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  mesh->registerSolution(backgroundFlow); // to trigger issue with p-refinements
  map<int, Teuchos::RCP<Function> > functionMap; functionMap[u->ID()] = Function::constant(3.14);
  backgroundFlow->projectOntoMesh(functionMap);
  
  vector<GlobalIndexType> ids;
  ids.push_back(1);
  ids.push_back(3);
  mesh->pRefine(ids); 
  
  return success;
}
