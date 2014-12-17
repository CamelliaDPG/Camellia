//
//  BasisFactoryTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/17/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaCellTools.h"
#include "CellTopology.h"
#include "BasisFactory.h"

using namespace Camellia;

namespace {
  vector< CellTopoPtr > getShardsTopologies() {
    vector< CellTopoPtr > shardsTopologies;
    
    shardsTopologies.push_back(CellTopology::point());
    shardsTopologies.push_back(CellTopology::line());
    shardsTopologies.push_back(CellTopology::quad());
    shardsTopologies.push_back(CellTopology::triangle());
    shardsTopologies.push_back(CellTopology::hexahedron());
    //  shardsTopologies.push_back(CellTopology::tetrahedron()); // tetrahedron not yet supported by permutation
    return shardsTopologies;
  }
  
  TEUCHOS_UNIT_TEST( BasisFactory, GetNodalBasisForCellTopology_Shards )
  {
    // For each node, there should be exactly one basis function with value 1, and the rest should be 0.
    
    std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();
    
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr topo = shardsTopologies[topoOrdinal];
      
      FieldContainer<double> refNodes(topo->getNodeCount(),topo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(refNodes, topo);

      BasisPtr nodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(topo);
      
      TEST_EQUALITY(nodalBasis->getCardinality(), topo->getNodeCount());
      
      FieldContainer<double> basisValues(nodalBasis->getCardinality(),topo->getNodeCount());
      
      nodalBasis->getValues(basisValues, refNodes, Intrepid::OPERATOR_VALUE);
      
      double tol = 1e-15;
      
      std::map<int, int> nodeToBasisOrdinal;
      for (int basisOrdinal=0; basisOrdinal<nodalBasis->getCardinality(); basisOrdinal++) {
        int nonZeroLocation = -1;
        for (int node=0; node < topo->getNodeCount(); node++) {
          double value = basisValues(basisOrdinal,node);
          if (abs(value) > tol) {
            TEST_EQUALITY(nonZeroLocation, -1); // shouldn't have more than one for any basisOrdinal
            nonZeroLocation = node;
            TEST_FLOATING_EQUALITY(value, 1.0, tol);
            TEST_ASSERT(nodeToBasisOrdinal.find(node) == nodeToBasisOrdinal.end()); // ensure uniqueness of the node --> basisOrdinal mapping
            nodeToBasisOrdinal[node] = basisOrdinal;
          }
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisFactory, GetNodalBasisForCellTopology_SpaceTime )
  {
    // For each node, there should be exactly one basis function with value 1, and the rest should be 0.
    
    std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();
    
    int tensorialDegree = 1;
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];
      CellTopoPtr topo = CellTopology::cellTopology(shardsTopo->getShardsTopology(), tensorialDegree);
      
      FieldContainer<double> refNodes(topo->getNodeCount(),topo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(refNodes, topo);
      
      BasisPtr nodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(topo);
      
      TEST_EQUALITY(nodalBasis->getCardinality(), topo->getNodeCount());
      
      FieldContainer<double> basisValues(nodalBasis->getCardinality(),topo->getNodeCount());
      
//      cout << "refNodes:\n" << refNodes;
      
      nodalBasis->getValues(basisValues, refNodes, Intrepid::OPERATOR_VALUE);
      
//      cout << "basisValues:\n" << basisValues;
      
      double tol = 1e-15;
      
      std::map<int, int> nodeToBasisOrdinal;
      for (int basisOrdinal=0; basisOrdinal<nodalBasis->getCardinality(); basisOrdinal++) {
        int nonZeroLocation = -1;
        for (int node=0; node < topo->getNodeCount(); node++) {
          double value = basisValues(basisOrdinal,node);
          if (abs(value) > tol) {
            TEST_EQUALITY(nonZeroLocation, -1); // shouldn't have more than one for any basisOrdinal
            nonZeroLocation = node;
            TEST_FLOATING_EQUALITY(value, 1.0, tol);
            TEST_ASSERT(nodeToBasisOrdinal.find(node) == nodeToBasisOrdinal.end()); // ensure uniqueness of the node --> basisOrdinal mapping
            nodeToBasisOrdinal[node] = basisOrdinal;
          }
        }
      }
    }
  }
} // namespace