//
//  BasisTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Basis.h"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"

#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

#include "doubleBasisConstruction.h"
#include "CamelliaCellTools.h"

#include "Intrepid_FieldContainer.hpp"

#include "Basis.h"

#include "TensorBasis.h"

#include "BasisFactory.h"

namespace {
  TEUCHOS_UNIT_TEST( Basis, LineC1_Unisolvence )
  {
    int polyOrder = 1;
    BasisPtr linearBasis = Camellia::intrepidLineHGRAD(polyOrder);
    
    CellTopoPtr cellTopo = linearBasis->domainTopology();
    
    FieldContainer<double> refCellNodes(cellTopo->getNodeCount(), cellTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
    
    set<int> knownNodes;
    
    FieldContainer<double> valuesAtNodes(linearBasis->getCardinality(), cellTopo->getNodeCount()); // (F, P)
    
    linearBasis->getValues(valuesAtNodes, refCellNodes, OPERATOR_VALUE);
    
    for (int basisOrdinal=0; basisOrdinal < linearBasis->getCardinality(); basisOrdinal++) {
      for (int nodeOrdinal=0; nodeOrdinal < cellTopo->getNodeCount(); nodeOrdinal++) {
        if (valuesAtNodes(basisOrdinal, nodeOrdinal) != 0.0) {
          // if it's not 0, then it should be 1
          TEST_ASSERT(valuesAtNodes(basisOrdinal, nodeOrdinal) == 1.0);
          // if it is 1, then this should be a node for which we haven't had a 1.0 value
          TEST_ASSERT(knownNodes.find(nodeOrdinal) == knownNodes.end());
          knownNodes.insert(nodeOrdinal);
        }
      }
    }
    TEST_ASSERT(knownNodes.size() == linearBasis->getCardinality());
  }

  // Define the templated unit test.
  TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Basis, ScalarPolynomialBasisUnisolvence, ScalarPolynomialBasisType ) {
    int polyOrder = 1;
    ScalarPolynomialBasisType basis(polyOrder, POINTTYPE_SPECTRAL);
    
    shards::CellTopology cellTopo = basis.getBaseCellTopology();
    
    FieldContainer<double> dofCoords(basis.getCardinality(), cellTopo.getDimension());
    basis.getDofCoords(dofCoords);
    
    FieldContainer<double> valuesAtNodes(basis.getCardinality(), dofCoords.dimension(0)); // (F, P)
    
    basis.getValues(valuesAtNodes, dofCoords, OPERATOR_VALUE);
    
    for (int basisOrdinal=0; basisOrdinal < basis.getCardinality(); basisOrdinal++) {
      for (int nodeOrdinal=0; nodeOrdinal < dofCoords.dimension(0); nodeOrdinal++) {
        if (basisOrdinal==nodeOrdinal) {
          TEST_ASSERT(valuesAtNodes(basisOrdinal,nodeOrdinal) == 1.0);
        } else {
          TEST_ASSERT(valuesAtNodes(basisOrdinal,nodeOrdinal) == 0.0);
        }
      }
    }
  }
  
  //
  // Instantiate the unit test for various values of RealType.
  //
  // Typedefs to work around Bug 5757 (TYPE values cannot have spaces).
  typedef ::Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, ::Intrepid::FieldContainer<double> > HGRAD_LINE_TYPE;
  
  typedef ::Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, ::Intrepid::FieldContainer<double> > HGRAD_QUAD_TYPE;
  typedef ::Intrepid::Basis_HDIV_QUAD_In_FEM<double, ::Intrepid::FieldContainer<double> > HDIV_QUAD_TYPE;
  
  typedef ::Intrepid::Basis_HGRAD_HEX_Cn_FEM<double, ::Intrepid::FieldContainer<double> > HGRAD_HEX_TYPE;
  typedef ::Intrepid::Basis_HDIV_HEX_In_FEM<double, ::Intrepid::FieldContainer<double> > HDIV_HEX_TYPE;
  
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_LINE_TYPE )
  
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_QUAD_TYPE )
  
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_HEX_TYPE )

} // namespace