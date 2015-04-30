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
using namespace Intrepid;

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
      
      TEST_EQUALITY(nodalBasis->rangeDimension(), topo->getDimension());
      
      TEST_EQUALITY(nodalBasis->getCardinality(), topo->getNodeCount());
      
      FieldContainer<double> basisValues(nodalBasis->getCardinality(),topo->getNodeCount());
      
      nodalBasis->getValues(basisValues, refNodes, Intrepid::OPERATOR_VALUE);
      
      // require that the nodal basis be numbered the same way as the topology i.e.
      // basisValues(i,j) = (i==j) ? 1 : 0;
      
      for (int basisOrdinal=0; basisOrdinal<nodalBasis->getCardinality(); basisOrdinal++) {
        for (int node=0; node < topo->getNodeCount(); node++) {
          double expectedValue = (node==basisOrdinal) ? 1 : 0;
          double value = basisValues(basisOrdinal,node);
          TEST_EQUALITY(expectedValue, value);
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
      TEST_EQUALITY(nodalBasis->rangeDimension(), topo->getDimension());
      
      TEST_EQUALITY(nodalBasis->getCardinality(), topo->getNodeCount());
      
      FieldContainer<double> basisValues(nodalBasis->getCardinality(),topo->getNodeCount());
      
//      cout << "refNodes:\n" << refNodes;
      
      nodalBasis->getValues(basisValues, refNodes, Intrepid::OPERATOR_VALUE);
      
//      cout << "basisValues:\n" << basisValues;
      
      // require that the nodal basis be numbered the same way as the topology i.e.
      // basisValues(i,j) = (i==j) ? 1 : 0;
    
      for (int basisOrdinal=0; basisOrdinal<nodalBasis->getCardinality(); basisOrdinal++) {
        for (int node=0; node < topo->getNodeCount(); node++) {
          double expectedValue = (node==basisOrdinal) ? 1 : 0;
          double value = basisValues(basisOrdinal,node);
          TEST_EQUALITY(expectedValue, value);
        }
      }
//      if (!success) {
//        cout << "basisValues:\n" << basisValues;
//      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisFactory, GetConformingBasisDomainEqualsCellTopology_Shards )
  {
    std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();
    
    int polyOrder = 1;
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD; // HGRAD convenient because defined even on line topology
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr topo = shardsTopologies[topoOrdinal];
      
      FieldContainer<double> refNodes(topo->getNodeCount(),topo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(refNodes, topo);
      
      BasisPtr conformingBasis = BasisFactory::basisFactory()->getConformingBasis(polyOrder, topo, fs);
      CellTopoPtr domainTopo = conformingBasis->domainTopology();
      TEST_EQUALITY(domainTopo->getKey(), topo->getKey());
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisFactory, GetConformingBasisDomainEqualsCellTopology_SpaceTime )
  {
    std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();
    
    int tensorialDegree = 1;
    int polyOrder = 1;
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD; // HGRAD convenient because defined even on line topology
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];
      CellTopoPtr topo = CellTopology::cellTopology(shardsTopo->getShardsTopology(), tensorialDegree);
      
      FieldContainer<double> refNodes(topo->getNodeCount(),topo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(refNodes, topo);
      
      BasisPtr conformingBasis = BasisFactory::basisFactory()->getConformingBasis(polyOrder, topo, fs);
      CellTopoPtr domainTopo = conformingBasis->domainTopology();
      TEST_EQUALITY(domainTopo->getKey(), topo->getKey());
    }
  }
  
} // namespace
