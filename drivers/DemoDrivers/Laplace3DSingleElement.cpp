//
//  Laplace3DSingleElement.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 10/18/13.
//
//

/*
   The basic idea with this driver is just to exercise the part of Camellia's DPG apparatus
   that doesn't depend on either Solution or Mesh.  We do the bits that would depend on Solution
   and Mesh "manually" -- we use a DofOrdering to manage the dof indices, and invert the stiffness matrix
   with some serial matrix solver.
 */

#include "InnerProductScratchPad.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "ExactSolution.h"

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"

#include "BilinearFormUtility.h"

#include "BCFunction.h"

#include "SerialDenseWrapper.h"

#include "BasisSumFunction.h"

FieldContainer<double> referenceCubeNodes() {
  FieldContainer<double> cubePoints(8,3);
  cubePoints(0,0) = -1;
  cubePoints(0,1) = -1;
  cubePoints(0,2) = -1;
  
  cubePoints(1,0) = 1;
  cubePoints(1,1) = -1;
  cubePoints(1,2) = -1;
  
  cubePoints(2,0) = 1;
  cubePoints(2,1) = 1;
  cubePoints(2,2) = -1;
  
  cubePoints(3,0) = -1;
  cubePoints(3,1) = 1;
  cubePoints(3,2) = -1;
  
  cubePoints(4,0) = -1;
  cubePoints(4,1) = -1;
  cubePoints(4,2) = 1;
  
  cubePoints(5,0) = 1;
  cubePoints(5,1) = -1;
  cubePoints(5,2) = 1;
  
  cubePoints(6,0) = 1;
  cubePoints(6,1) = 1;
  cubePoints(6,2) = 1;
  
  cubePoints(7,0) = -1;
  cubePoints(7,1) = 1;
  cubePoints(7,2) = 1;
  return cubePoints;
}

void printDofIndicesForVariable(DofOrderingPtr dofOrdering, VarPtr var, int sideIndex) {
  vector<int> dofIndices = dofOrdering->getDofIndices(var->ID(), sideIndex);
  cout << "dofIndices for " << var->name() << ", side " << sideIndex << ":" << endl;
  for (int i=0; i<dofIndices.size(); i++) {
    cout << dofIndices[i] << " ";
  }
  cout << endl;
}

double evaluateSoln(FunctionPtr fxn, double x, double y, double z) {
  // uses the fact that the reference element and physical are identical
  
  FieldContainer<double> value(1,1);
  FieldContainer<double> physPoint(1,3);
  physPoint(0,0) = x;
  physPoint(0,1) = y;
  physPoint(0,2) = z;
  
  shards::CellTopology cellTopo(shards::getCellTopologyData<shards::Hexahedron<8> >() );
  
  FieldContainer<double> refCubeNodes = referenceCubeNodes();
  refCubeNodes.resize(1,8,3);
  
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(refCubeNodes, cellTopo, 0) );
  if (fxn->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  basisCache->setRefCellPoints(physPoint); // this is where we use the fact that the ref element matches physical
  fxn->values(value,basisCache);
  return value[0];
}

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank = mpiSession.getRank();
  
  int minPolyOrder = 3;
  int maxPolyOrder = 3;
  int pToAdd = 2;

  if (rank==0) {
    cout << "min H^1 Order: " << minPolyOrder << "\n";
    cout << "max H^1 Order: " << maxPolyOrder << "\n";
    cout << "pToAdd: " << pToAdd << "\n";
  }
  
  VarFactory vf;
  // trial variables:
  VarPtr phi = vf.fieldVar("\\phi");
  VarPtr psi1 = vf.fieldVar("\\psi_{1}");
  VarPtr psi2 = vf.fieldVar("\\psi_{2}");
  VarPtr psi3 = vf.fieldVar("\\psi_{3}");
  VarPtr phi_hat = vf.traceVar("\\widehat{\\phi}");
  VarPtr psi_hat_n = vf.fluxVar("\\widehat{\\psi}_n");
  // test variables
  VarPtr q = vf.testVar("q", HDIV);
  VarPtr v = vf.testVar("v", HGRAD);
  // bilinear form
  BFPtr bf = Teuchos::rcp( new BF(vf) );
  bf->addTerm(phi, q->div());
  bf->addTerm(psi1, q->x());
  bf->addTerm(psi2, q->y());
  bf->addTerm(psi3, q->z());
  bf->addTerm(-phi_hat, q->dot_normal());
  
  bf->addTerm(-psi1, v->dx());
  bf->addTerm(-psi2, v->dy());
  bf->addTerm(-psi3, v->dz());
  bf->addTerm(psi_hat_n, v);
  
  // define exact solution functions
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
//  FunctionPtr phi_exact = Function::constant(2);
//  FunctionPtr phi_exact = x;
  FunctionPtr phi_exact = y;
//  FunctionPtr phi_exact = z;

  FunctionPtr psi1_exact = phi_exact->dx();
  FunctionPtr psi2_exact = phi_exact->dy();
  FunctionPtr psi3_exact = phi_exact->dz();
  
  // set up BCs
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);
  
  // RHS
  Teuchos::RCP< RHSEasy > rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
  rhs->addTerm(f * v);
  
  // exact solution object
  Teuchos::RCP<ExactSolution> exactSolution = Teuchos::rcp( new ExactSolution(bf,bc,rhs) );
  exactSolution->setSolutionFunction(phi, phi_exact);
  exactSolution->setSolutionFunction(psi1, psi1_exact );
  exactSolution->setSolutionFunction(psi2, psi2_exact );
  exactSolution->setSolutionFunction(psi3, psi3_exact );
  
  // inner product
  IPPtr ip = bf->graphNorm();
  
  if (rank==0) {
    cout << "Laplace bilinear form:\n";
    bf->printTrialTestInteractions();
  }
  
  // for now, let's use the reference cell.  (Jacobian should be the identity.)
  FieldContainer<double> cubePoints = referenceCubeNodes();
  
  int numCells = 1;
  cubePoints.resize(numCells,8,3); // first argument is cellIndex; we'll just have 1
  
  Teuchos::RCP<shards::CellTopology> hexTopoPtr;
  hexTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));

  FieldContainer<double> cellSideParities(numCells, hexTopoPtr->getSideCount());
  cellSideParities.initialize(1); // since we have only a single element, all parities are arbitary: set to 1
  
  int sideDim = 2;
  int delta_k = 2;
  
  for (int polyOrder=minPolyOrder; polyOrder<=maxPolyOrder; polyOrder++) {
    BasisPtr hGradBasisQuad = BasisFactory::getBasis(polyOrder, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    BasisPtr l2BasisQuad = BasisFactory::getBasis(polyOrder, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
    
    BasisPtr hGradBasisHex = BasisFactory::getBasis(polyOrder, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    BasisPtr hDivBasisHex = BasisFactory::getBasis(polyOrder, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
    BasisPtr l2BasisHex = BasisFactory::getBasis(polyOrder, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);

    BasisPtr hGradBasisHexTest = BasisFactory::getBasis(polyOrder + delta_k, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    BasisPtr hDivBasisHexTest = BasisFactory::getBasis(polyOrder + delta_k, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
    
    // Define trial discretization
    DofOrderingPtr trialOrderPtr = Teuchos::rcp( new DofOrdering );
    trialOrderPtr->addEntry(phi->ID(), l2BasisHex, l2BasisQuad->rangeRank());
    trialOrderPtr->addEntry(psi1->ID(), l2BasisHex, l2BasisQuad->rangeRank());
    trialOrderPtr->addEntry(psi2->ID(), l2BasisHex, l2BasisQuad->rangeRank());
    trialOrderPtr->addEntry(psi3->ID(), l2BasisHex, l2BasisQuad->rangeRank());
    
    // traces:
    for (int sideOrdinal = 0; sideOrdinal < hexTopoPtr->getSideCount(); sideOrdinal++) {
      trialOrderPtr->addEntry(phi_hat->ID(), hGradBasisQuad, l2BasisQuad->rangeRank(), sideOrdinal);
    }
    // fluxes:
    for (int sideOrdinal = 0; sideOrdinal < hexTopoPtr->getSideCount(); sideOrdinal++) {
      trialOrderPtr->addEntry(psi_hat_n->ID(), l2BasisQuad, l2BasisQuad->rangeRank(), sideOrdinal);
    }
    
    // for the trace of H^1 (phi_hat), need to add identifications of vertex and edge dofs.
    map< unsigned, vector< pair<unsigned, unsigned> > > vertexNodeToSideVertex;  // key: hex's node #.  Values: sideOrdinal, quad's vertex #.
    map< pair<unsigned, unsigned>, vector< pair< pair<unsigned, unsigned>, bool > > > edgeNodeToSideEdge;  // key: hex's vertex node numbers, in ascending order.  Values: ( (sideOrdinal, quad's edge #), orientation (true if we didn't need to flip the hex node ordering) ).

    int vertexDim = 0;
    int edgeDim = 1;
    int faceDim = 2;
    
    for (int sideOrdinal = 0; sideOrdinal < hexTopoPtr->getSideCount(); sideOrdinal++) {
      shards::CellTopology sideTopo(hexTopoPtr->getCellTopologyData(sideDim, sideOrdinal));
//      cout << "Side " << sideOrdinal << endl;
      for (int edgeOrdinal=0; edgeOrdinal < sideTopo.getEdgeCount(); edgeOrdinal++) {
        // how does the hex identify this edge?
        // that's not quite the right question to ask, I think, particularly since our Hexahedron<8> only has vertex nodes,
        // but clearly Intrepid's basis classes have a way of ordering edges in each topology.
        // TODO: fix this call (broken!)
        unsigned hexNodeForVertex0 = hexTopoPtr->getNodeMap(faceDim, sideOrdinal, edgeOrdinal);
        unsigned hexNodeForVertex1 = hexTopoPtr->getNodeMap(faceDim, sideOrdinal, (edgeOrdinal+1)%(sideTopo.getVertexCount()));
        
        unsigned hexNodeMin = min(hexNodeForVertex0,hexNodeForVertex1);
        unsigned hexNodeMax = max(hexNodeForVertex0,hexNodeForVertex1);
        bool orientationAgreesWithNodeOrdering = (hexNodeForVertex0==hexNodeMin);
        
        pair< unsigned, unsigned > edge = make_pair(hexNodeMin, hexNodeMax);
        
        if (edgeNodeToSideEdge.find(edge) == edgeNodeToSideEdge.end()) {
          edgeNodeToSideEdge[edge] = vector< pair< pair<unsigned, unsigned>, bool > >();
        }
        edgeNodeToSideEdge[edge].push_back(make_pair(make_pair(sideOrdinal, edgeOrdinal), orientationAgreesWithNodeOrdering));
      }
      for (int vertexOrdinal=0; vertexOrdinal < sideTopo.getVertexCount(); vertexOrdinal++) {
        // how does the hex identify this vertex?
        unsigned hexNodeForVertex = hexTopoPtr->getNodeMap(faceDim, sideOrdinal, vertexOrdinal);
        if (vertexNodeToSideVertex.find(hexNodeForVertex) == vertexNodeToSideVertex.end()) {
          vertexNodeToSideVertex[hexNodeForVertex] = vector< pair<unsigned, unsigned> >();
        }
        vertexNodeToSideVertex[hexNodeForVertex].push_back(make_pair(sideOrdinal, vertexOrdinal));
      }
    }
    // DEBUG loop over vertexNodeToSideVertex:
    for (map< unsigned, vector< pair<unsigned, unsigned> > >::iterator vertexNodeIter = vertexNodeToSideVertex.begin();
         vertexNodeIter != vertexNodeToSideVertex.end(); vertexNodeIter++) {
      int hexNodeForVertex = vertexNodeIter->first;
      cout << "vertex " << hexNodeForVertex << ", (side, vertex) pairs: ";
      vector< pair<unsigned, unsigned> > vertices = vertexNodeIter->second;
      for (vector< pair<unsigned, unsigned> >::iterator verticesIter = vertices.begin();
           verticesIter != vertices.end(); verticesIter++) {
        int sideOrdinal = verticesIter->first;
        int vertexOrdinalInSide = verticesIter->second;
        cout << "(" << sideOrdinal << ", " << vertexOrdinalInSide << ") ";
      }
      cout << endl;
    }
    // DEBUG loop over edgeNodeToSideEdge:
    for (map< pair<unsigned, unsigned>, vector< pair< pair<unsigned, unsigned>, bool > > > ::iterator edgeNodeIter = edgeNodeToSideEdge.begin();
         edgeNodeIter != edgeNodeToSideEdge.end(); edgeNodeIter++) {
      unsigned v0 = edgeNodeIter->first.first;
      unsigned v1 = edgeNodeIter->first.second;
      cout << "edge (" << v0 << "," << v1 << ") (side, edge) pairs: ";
      vector< pair < pair<unsigned, unsigned>, bool > > edges = edgeNodeIter->second;
      for (vector< pair < pair<unsigned, unsigned>, bool > >::iterator edgesIter = edges.begin();
           edgesIter != edges.end(); edgesIter++) {
        bool orientationsAgree = edgesIter->second;
        int sideOrdinal = edgesIter->first.first;
        int edgeOrdinalInSide = edgesIter->first.second;
        cout << "(" << sideOrdinal << ", " << edgeOrdinalInSide << ") ";
      }
      cout << endl;
    }
    // now, we want to identify all those dofs:
    for (map< unsigned, vector< pair<unsigned, unsigned> > >::iterator vertexNodeIter = vertexNodeToSideVertex.begin();
         vertexNodeIter != vertexNodeToSideVertex.end(); vertexNodeIter++) {
      vector< pair<unsigned, unsigned> > vertices = vertexNodeIter->second;
      vector< pair<unsigned, unsigned> >::iterator verticesIter = vertices.begin();
      unsigned firstSideOrdinal = verticesIter->first;
      unsigned firstVertexOrdinal = verticesIter->second;
      unsigned firstVertexDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)
                                                    ->getDofOrdinal(vertexDim, firstVertexOrdinal, 0); // 0: only one vertex dof, so its index is 0
      
      while (++verticesIter != vertices.end()) {
        unsigned sideOrdinal = verticesIter->first;
        unsigned vertexOrdinal = verticesIter->second;
        unsigned vertexDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), sideOrdinal)
                                                 ->getDofOrdinal(vertexDim, vertexOrdinal, 0); // 0: only one vertex dof, so its index is 0
        trialOrderPtr->addIdentification(phi_hat->ID(), firstSideOrdinal, firstVertexDofOrdinal,
                                         sideOrdinal, vertexDofOrdinal);
        cout << "(" << firstSideOrdinal << ", " << firstVertexOrdinal;
        cout << ") <-> (" << sideOrdinal << ", " << vertexOrdinal << ")" << endl;
      }
    }

    for (map< pair<unsigned, unsigned>, vector< pair< pair<unsigned, unsigned>, bool > > > ::iterator edgeNodeIter = edgeNodeToSideEdge.begin();
         edgeNodeIter != edgeNodeToSideEdge.end(); edgeNodeIter++) {
      vector< pair < pair<unsigned, unsigned>, bool > > edges = edgeNodeIter->second;
      vector< pair < pair<unsigned, unsigned>, bool > >::iterator edgeIter = edges.begin();
      if (edgeIter == edges.end()) continue;
      // there could be a bunch of these (or 0, for linear elements).  How do we determine the correct permutation?
      // for now, we make the assumption that the layout is CCW or CW consistently for all sides of the element, which
      // would mean that the orderings are reversed
      
      bool firstOrientationAgrees = edgeIter->second;
      int firstSideOrdinal = edgeIter->first.first;
      int firstEdgeOrdinalInSide = edgeIter->first.second;
      
      BasisPtr firstBasis = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal);
      
      int numDofs = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)->dofOrdinalsForSubcell(edgeDim, firstEdgeOrdinalInSide).size();

      while (++edgeIter != edges.end()) {
        unsigned sideOrdinal = edgeIter->first.first;
        unsigned edgeOrdinal = edgeIter->first.second;
        unsigned orientationAgrees = edgeIter->second;
        for (int dofNumber=0; dofNumber<numDofs; dofNumber++) {
          // this is the point where we consider the relative permutation of the identified bases...
          // when we do this in a mesh, we'll need to do something more sophisticated.
          if (orientationAgrees != firstOrientationAgrees) {
            unsigned firstDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)
                                                 ->getDofOrdinal(edgeDim, edgeOrdinal, numDofs - dofNumber - 1);
            unsigned edgeDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), sideOrdinal)
                                                   ->getDofOrdinal(edgeDim, edgeOrdinal, dofNumber);
            trialOrderPtr->addIdentification(phi_hat->ID(), firstSideOrdinal, firstDofOrdinal,
                                             sideOrdinal, edgeDofOrdinal);
          } else {
            cout << "Unexpectedly, two sides agree on edge orientation.\n";
            unsigned firstDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)
                                                    ->getDofOrdinal(edgeDim, edgeOrdinal, dofNumber);
            unsigned edgeDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), sideOrdinal)
                                                    ->getDofOrdinal(edgeDim, edgeOrdinal, dofNumber);
            trialOrderPtr->addIdentification(phi_hat->ID(), firstSideOrdinal, firstDofOrdinal,
                                             sideOrdinal, edgeDofOrdinal);
          }
          
        }
      }
    }
    // after adding all those identifications, need to rebuild the DofOrdering's index:
    trialOrderPtr->rebuildIndex();
    
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 0);
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 1);
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 2);
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 3);
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 4);
    printDofIndicesForVariable(trialOrderPtr, phi_hat, 5);
    
    // Define test discretization
    DofOrderingPtr testOrderPtr = Teuchos::rcp( new DofOrdering );
    testOrderPtr->addEntry(q->ID(), hDivBasisHexTest, hDivBasisHex->rangeRank());
    testOrderPtr->addEntry(v->ID(), hGradBasisHexTest, hGradBasisHex->rangeRank());
    
    int numTrialDofs = trialOrderPtr->totalDofs();
    int numTestDofs = testOrderPtr->totalDofs();
    
    if (rank==0)
      cout << "For k = " << polyOrder - 1 << ", " << numTrialDofs << " trial dofs and " << numTestDofs << " test dofs.\n";
    
//    cout << "*** trial ordering *** \n" << *trialOrderPtr;
//    cout << "*** test ordering *** \n" << *testOrderPtr;
    
    // Create ElementType for cube
    Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(trialOrderPtr, testOrderPtr, hexTopoPtr) );
    
    // Create BasisCache for ElementType and cubePoints
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemTypePtr) );
    vector<int> cellIDs;
    cellIDs.push_back(0);
    basisCache->setPhysicalCellNodes(cubePoints, cellIDs, true);
    
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,Teuchos::rcp((Mesh*) NULL), true));
    ipBasisCache->setPhysicalCellNodes(cubePoints,cellIDs,false); // false: no side cache for IP
    
    CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    // Mimic/copy Solution's treatment of stiffness and optimal test functions.
    FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
    ip->computeInnerProductMatrix(ipMatrix, testOrderPtr, ipBasisCache);
    
    FieldContainer<double> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
    
    int optSuccess = bf->optimalTestWeights(optTestCoeffs, ipMatrix, elemTypePtr, cellSideParities, basisCache);
    
    if (optSuccess != 0) {
      cout << "Error while solving for optimal test weights.\n";
    }
    
    FieldContainer<double> finalStiffness(numCells,numTrialDofs,numTrialDofs);
    
    BilinearFormUtility::computeStiffnessMatrix(finalStiffness,ipMatrix,optTestCoeffs);
    
    FieldContainer<double> localRHSVector(numCells, numTrialDofs);
    rhs->integrateAgainstOptimalTests(localRHSVector, optTestCoeffs, testOrderPtr, basisCache);
    
    // Apply BCs
    // here, we know that phi_hat is what we're interested in, so we skip the usual loop over varIDs
    // we also know that we're applying the BC on every side, so we skip any check that the BCs are applicable
    int varID = phi_hat->ID();
    bool isTrace = true;
    Teuchos::RCP<BCFunction> bcFunction = Teuchos::rcp(new BCFunction(bc, varID, isTrace));
    FieldContainer<double> bcVector(trialOrderPtr->totalDofs());
    set<int> bcDofIndices;
    for (int sideIndex=0; sideIndex < cellTopoPtr->getSideCount(); sideIndex++) {
      BasisPtr basis = trialOrderPtr->getBasis(varID,sideIndex);
      FieldContainer<double> dirichletValues(numCells, basis->getCardinality());
      bc->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideIndex));
//      cout << "dirichletValues for side " << sideIndex << ":" << endl << dirichletValues;
      int cellIndex = 0;
      for (int basisOrdinal=0; basisOrdinal < basis->getCardinality(); basisOrdinal++) {
        int localDofIndex = trialOrderPtr->getDofIndex(varID, basisOrdinal, sideIndex);
        // we can also skip any global dof index lookup, since we're dealing with just one element
        bcVector(localDofIndex) = dirichletValues(cellIndex,basisOrdinal);
        bcDofIndices.insert(localDofIndex);
      }
    }
//    cout << "bcVector:\n" << bcVector;
    // now, multiply the "finalStiffness" by the bcVector to determine what we have to subtract:
    finalStiffness.resize(numTrialDofs,numTrialDofs);
    bcVector.resize(numTrialDofs,1); // vector as a 2D array
    FieldContainer<double> rhsAdjustment(numTrialDofs,1);
    SerialDenseWrapper::multiply(rhsAdjustment, finalStiffness, bcVector);
    rhsAdjustment.resize(numTrialDofs);
    localRHSVector.resize(numTrialDofs);
    bcVector.resize(numTrialDofs);
    
    // adjust RHS:
    for (int dofIndex=0; dofIndex<numTrialDofs; dofIndex++) {
      localRHSVector(dofIndex) -= rhsAdjustment(dofIndex);
    }
    
    // zero out the stiffness matrix rows and columns for dirichlet values
    for (set<int>::iterator dofIndexIt = bcDofIndices.begin(); dofIndexIt != bcDofIndices.end(); dofIndexIt++) {
      int dofIndex = *dofIndexIt;
      for (int i=0; i<numTrialDofs; i++) {
        finalStiffness(i,dofIndex) = 0;
        finalStiffness(dofIndex,i) = 0;
      }
      // set the dirichlet value:
      finalStiffness(dofIndex, dofIndex) = 1;
      localRHSVector(dofIndex) = bcVector(dofIndex);
    }
    
    // solve:
    FieldContainer<double>solution(numTrialDofs);
    SerialDenseWrapper::solveSystem(solution, finalStiffness, localRHSVector);
    
    bool bcsCorrect = true;
    for (set<int>::iterator dofIndexIt = bcDofIndices.begin(); dofIndexIt != bcDofIndices.end(); dofIndexIt++) {
      int dofIndex = *dofIndexIt;
      
      if (solution(dofIndex) != bcVector(dofIndex)) {
        cout << solution(dofIndex) << " = solution(" << dofIndex;
        cout << ") != bcVector(" << dofIndex << ") = " << bcVector(dofIndex) << "\n";
        bcsCorrect = false;
      }
    }
    if (bcsCorrect)
      cout << "BCs were correctly imposed.\n";
    else
      cout << "BCs were NOT correctly imposed.\n";
    
//    cout << "solution coefficients:\n" << solution;
    
    // check solution
    map<int, VarPtr > trialVars = vf.trialVars();
    for (map<int, VarPtr >::iterator trialVarIt = trialVars.begin(); trialVarIt != trialVars.end(); trialVarIt++) {
      VarPtr var = trialVarIt->second;
      if (var->varType() == FIELD) {
        BasisPtr basis = trialOrderPtr->getBasis(var->ID());
        FieldContainer<double> solnCoefficients(basis->getCardinality());
        for (int basisOrdinal=0; basisOrdinal<basis->getCardinality(); basisOrdinal++) {
          int localDofIndex = trialOrderPtr->getDofIndex(var->ID(), basisOrdinal);
          solnCoefficients(basisOrdinal) = solution(localDofIndex);
        }
        FunctionPtr solnFxn = NewBasisSumFunction::basisSumFunction(basis, solnCoefficients);
        FunctionPtr exactFxn = exactSolution->exactFunctions().find(var->ID())->second;
        double integral = solnFxn->integrate(basisCache);
        cout << "integral of solnFxn for " << var->name() << ": " << integral <<  "\n";
        integral = exactFxn->integrate(basisCache);
        cout << "integral of exactFxn for " << var->name() << ": " << integral << "\n";
        double l2Error = ((exactFxn - solnFxn) * (exactFxn - solnFxn))->integrate(basisCache);
        cout << "L^2 error for trial variable " << var->name() << ": " << l2Error << endl;
        cout << "exact at (0,0,0): "  << Function::evaluate(exactFxn, 0,0,0) << endl;
        cout << "sol'n at (0,0,0): "  << evaluateSoln(solnFxn, 0,0,0) << endl;
        cout << "exact at (0,0,1): "  << Function::evaluate(exactFxn, 0,0,1) << endl;
        cout << "sol'n at (0,0,1): "  << evaluateSoln(solnFxn, 0,0,1) << endl;
        cout << "exact at (0,1,0): "  << Function::evaluate(exactFxn, 0,1,0) << endl;
        cout << "sol'n at (0,1,0): "  << evaluateSoln(solnFxn, 0,1,0) << endl;
        
        cout << "exact at (1,0,0): "  << Function::evaluate(exactFxn, 1,0,0) << endl;
        cout << "sol'n at (1,0,0): "  << evaluateSoln(solnFxn, 1,0,0) << endl;
        
        cout << "exact at (1,1,0): "  << Function::evaluate(exactFxn, 1,1,0) << endl;
        cout << "sol'n at (1,1,0): "  << evaluateSoln(solnFxn, 1,1,0) << endl;
        
        cout << "exact at (1,1,1): "  << Function::evaluate(exactFxn, 1,1,1) << endl;
        cout << "sol'n at (1,1,1): "  << evaluateSoln(solnFxn, 1,1,1) << endl;
      }
    }
  }

  // Steps to solve:
  // 1. Define trial discretization
  // 2. Define test discretization
  // 3. Create ElementType for cube
  // 4. Create BasisCache for ElementType and cubePoints
  // 5. Mimic/copy Solution's treatment of stiffness and optimal test functions.
  // 6. Apply BCs
  // 7. Invert stiffness (using SerialDenseMatrixUtility, say)
  // 8. Create BasisSumFunctions to represent solution
  // 9. Check solution.

}