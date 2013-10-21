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

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank = mpiSession.getRank();
  
  int minPolyOrder = 2;
  int maxPolyOrder = 5;
  int pToAdd = 2;

  if (rank==0) {
    cout << "minPolyOrder: " << minPolyOrder << "\n";
    cout << "maxPolyOrder: " << maxPolyOrder << "\n";
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
  bf->addTerm(-phi, q->div());
  bf->addTerm(-psi1, q->x());
  bf->addTerm(-psi2, q->y());
  bf->addTerm(-psi3, q->z());
  bf->addTerm(phi_hat, q->dot_normal());
  
  bf->addTerm(-psi1, v->dx());
  bf->addTerm(-psi2, v->dy());
  bf->addTerm(-psi3, v->dz());
  bf->addTerm(psi_hat_n, v);
  
  // define exact solution functions
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact = x * y * z;
  FunctionPtr psi1_exact = phi_exact->dx();
  FunctionPtr psi2_exact = phi_exact->dy();
  FunctionPtr psi3_exact = phi_exact->dz();
  
  // set up BCs
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);
  
  // RHS
  Teuchos::RCP< RHSEasy > rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
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
  
  cubePoints(0,0) = -1;
  cubePoints(0,1) = -1;
  cubePoints(0,2) = -1;
  
  cubePoints(0,0) = -1;
  cubePoints(0,1) = -1;
  cubePoints(0,2) = -1;
  
  cubePoints.resize(1,8,3); // first argument is cellIndex; we'll just have 1
  
  Teuchos::RCP<shards::CellTopology> hexTopoPtr;
  hexTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));

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
      trialOrderPtr->addEntry(psi_hat_n->ID(), l2BasisQuad, l2BasisQuad->rangeRank(), sideOrdinal);
    }
    
    // for the trace of H^1 (phi_hat), need to add identifications of vertex and edge dofs.
    map< unsigned, vector< pair<unsigned, unsigned> > > vertexNodeToSideVertex;  // key: hex's node #.  Values: sideOrdinal, quad's vertex #.
    map< unsigned, vector< pair<unsigned, unsigned> > > edgeNodeToSideEdge;  // key: hex's node #.  Values: sideOrdinal, quad's edge #.

    int vertexDim = 0;
    int edgeDim = 1;
    
    for (int sideOrdinal = 0; sideOrdinal < hexTopoPtr->getSideCount(); sideOrdinal++) {
      shards::CellTopology sideTopo(hexTopoPtr->getCellTopologyData(sideDim, sideOrdinal));
      for (int edgeOrdinal=0; edgeOrdinal < sideTopo.getEdgeCount(); edgeOrdinal++) {
        // how does the hex identify this edge?
        unsigned hexNodeForEdge = hexTopoPtr->getNodeMap(edgeDim, sideOrdinal, edgeOrdinal);
        if (edgeNodeToSideEdge.find(hexNodeForEdge) == edgeNodeToSideEdge.end()) {
          edgeNodeToSideEdge[hexNodeForEdge] = vector< pair<unsigned, unsigned> >();
        }
        edgeNodeToSideEdge[hexNodeForEdge].push_back(make_pair(sideOrdinal, edgeOrdinal));
      }
      for (int vertexOrdinal=0; vertexOrdinal < sideTopo.getVertexCount(); vertexOrdinal++) {
        // how does the hex identify this vertex?
        unsigned hexNodeForVertex = hexTopoPtr->getNodeMap(vertexDim, sideOrdinal, vertexOrdinal);
        if (vertexNodeToSideVertex.find(hexNodeForVertex) == vertexNodeToSideVertex.end()) {
          vertexNodeToSideVertex[hexNodeForVertex] = vector< pair<unsigned, unsigned> >();
        }
        vertexNodeToSideVertex[hexNodeForVertex].push_back(make_pair(sideOrdinal, vertexOrdinal));
      }
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
      }
    }

    for (map< unsigned, vector< pair<unsigned, unsigned> > >::iterator edgeNodeIter = edgeNodeToSideEdge.begin();
         edgeNodeIter != edgeNodeToSideEdge.end(); edgeNodeIter++) {
      // there could be a bunch of these (or 0, for linear elements).  How do we determine the correct permutation?
      // for now, we make the assumption that the layout is CCW or CW consistently for all sides of the element, which
      // would mean that the orderings are reversed
      vector< pair<unsigned, unsigned> > edges = edgeNodeIter->second;
      vector< pair<unsigned, unsigned> >::iterator edgeIter = edges.begin();
      unsigned firstSideOrdinal = edgeIter->first;
      unsigned firstEdgeOrdinal = edgeIter->second;
      BasisPtr firstBasis = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal);
      
      int numDofs = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)->dofOrdinalsForSubcell(edgeDim, firstEdgeOrdinal).size();

      while (++edgeIter != edges.end()) {
        unsigned sideOrdinal = edgeIter->first;
        unsigned edgeOrdinal = edgeIter->second;
        for (int dofNumber=0; dofNumber<numDofs; dofNumber++) {
          // this is the point where we make an assumption about the relative permutation of the identified bases...
          // when we do this in a mesh, we'll need to do something more sophisticated.
          unsigned firstDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), firstSideOrdinal)
                                               ->getDofOrdinal(edgeDim, edgeOrdinal, numDofs - dofNumber - 1);
          unsigned edgeDofOrdinal = trialOrderPtr->getBasis(phi_hat->ID(), sideOrdinal)
                                                 ->getDofOrdinal(edgeDim, edgeOrdinal, dofNumber);
          trialOrderPtr->addIdentification(phi_hat->ID(), firstSideOrdinal, firstDofOrdinal,
                                           sideOrdinal, edgeDofOrdinal);
          
        }
      }
    }
    
    // Define test discretization
    DofOrderingPtr testOrderPtr = Teuchos::rcp( new DofOrdering );
    testOrderPtr->addEntry(q->ID(), hDivBasisHexTest, hDivBasisHex->rangeRank());
    testOrderPtr->addEntry(v->ID(), hGradBasisHexTest, hGradBasisHex->rangeRank());
    
    cout << "For k = " << polyOrder - 1 << ", " << trialOrderPtr->totalDofs() << " trial dofs and " << testOrderPtr->totalDofs() << " test dofs.\n";
    
    // Create ElementType for cube
    Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(trialOrderPtr, testOrderPtr, hexTopoPtr) );
    
    // Create BasisCache for ElementType and cubePoints
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemTypePtr) );
    basisCache->setPhysicalCellNodes(cubePoints, vector<int>(1,1), true);
    
    
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