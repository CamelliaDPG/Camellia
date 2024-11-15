//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
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

#include "Intrepid_HDIV_TET_In_FEM.hpp"

#include "Intrepid_HGRAD_TRI_Cn_FEM_ORTH.hpp"
#include "Intrepid_HDIV_TRI_In_FEM.hpp"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"

#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

#include "doubleBasisConstruction.h"
#include "CamelliaCellTools.h"

#include "Camellia_FieldContainer.hpp"

#include "Basis.h"
#include "BasisFactory.h"
#include "MeshFactory.h"
#include "PointBasis.h"
#include "SerialDenseWrapper.h"
#include "TensorBasis.h"

using namespace Camellia;

using ArrayScalar = Camellia::FieldContainer<double>;

namespace
{
void testBasisAllDofOrdinalsAssignedToSubcells(BasisPtr basis, Teuchos::FancyOStream &out, bool &success)
{
  int domainDim = basis->domainTopology()->getDimension();
  int allSubcellDofOrdinalsCount = basis->dofOrdinalsForSubcells(domainDim, true).size();
  TEST_EQUALITY(basis->getCardinality(), allSubcellDofOrdinalsCount);
}
  
  void testBasisL2Eigenvalues(VarPtr var, BasisPtr basis, int H1Order, Teuchos::FancyOStream &out, bool &success)
  {
    CellTopoPtr cellTopo = basis->domainTopology();
    BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(cellTopo, H1Order*2);
    
    DofOrderingPtr testOrder = Teuchos::rcp( new DofOrdering(cellTopo) );
    
    int basisRank = basis->rangeRank();
    testOrder->addEntry(var->ID(),basis,basisRank);
    
    int numTestDofs = testOrder->totalDofs();
    
    int numCells = 1;
    Teuchos::LAPACK<int, double> lapack;
    
    IPPtr ipTauL2 = IP::ip();
    ipTauL2->addTerm(var);
    
    Camellia::FieldContainer<double> gramMatrix(numCells,numTestDofs,numTestDofs);
    ipTauL2->computeInnerProductMatrix(gramMatrix, testOrder, basisCache);
//    {
//      gramMatrix.resize(gramMatrix.dimension(1),gramMatrix.dimension(2));
//      SerialDenseWrapper::writeMatrixToMatlabFile("gramMatrix_tauL2.dat", gramMatrix);
//      gramMatrix.resize(1,gramMatrix.dimension(0),gramMatrix.dimension(1));
//    }
    
    gramMatrix.resize(numTestDofs,numTestDofs);
    Camellia::FieldContainer<double> lambda_real(numTestDofs), lambda_imag(numTestDofs);
    Camellia::FieldContainer<double> eigvectors(numTestDofs, numTestDofs);
    bool computeRightEigvectors = true;
    int result = SerialDenseWrapper::eigenvalues(gramMatrix, lambda_real, lambda_imag, eigvectors, computeRightEigvectors);
    
    if (result != 0)
    {
      success = false;
      out << "FAILURE: eigenvalues() call returned " << result << " error code.\n";
    }
    
    // all eigenvalues should be real, and all positive:
    for (int i=0; i< numTestDofs; i++)
    {
      TEST_COMPARE(lambda_real(i), >, 1e-15);
      //        out << lambda_real(i) << endl;
      
      if (lambda_real(i) < 1e-15)
      {
        // eigvector will be in the ith row of the Camellia::FieldContainer (ith column per LAPACK, but that's Fortran ordering)
        out << "eigenvector for negative/near zero eigvalue (" << lambda_real(i) << "):\n";
        out << "v = [";
        out << setprecision(15);
        for (int j=0; j<numTestDofs; j++)
        {
          out << eigvectors(i,j) << " ";
        }
        out << "];\n";
      }
    }
    
    double tol = 1e-12;
    for (int i=0; i< numTestDofs; i++)
    {
      if (abs(lambda_imag(i)) > tol)
      {
        out << "For i=" << i << ", ith eigvalue has nonzero imaginary part: " << lambda_imag(i) << endl;
        success = false;
      }
    }
  }

  void testBasisL2Eigenvalues(CellTopoPtr cellTopo, int H1Order, Space space, Teuchos::FancyOStream &out, bool &success)
  {
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr tau = vf->testVar("tau", space);
  
    Camellia::EFunctionSpace fs = efsForSpace(space);
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, cellTopo, fs);
    
    testBasisL2Eigenvalues(tau, basis, H1Order, out, success);
  }
  
  void testTriangleBasisL2Eigenvalues(int H1Order, Space space, Teuchos::FancyOStream &out, bool &success)
  {
    CellTopoPtr cellTopo = CellTopology::triangle();
    testBasisL2Eigenvalues(cellTopo, H1Order, space, out, success);
  }
  
  void testQuadBasisL2Eigenvalues(int H1Order, Space space, Teuchos::FancyOStream &out, bool &success)
  {
    CellTopoPtr cellTopo = CellTopology::quad();
    testBasisL2Eigenvalues(cellTopo, H1Order, space, out, success);
  }

TEUCHOS_UNIT_TEST( Basis, LineC1_Unisolvence )
{
  int polyOrder = 1;
  BasisPtr linearBasis = Camellia::intrepid2LineHGRAD(polyOrder);

  CellTopoPtr cellTopo = linearBasis->domainTopology();

  Camellia::FieldContainer<double> refCellNodes(cellTopo->getNodeCount(), cellTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);

  set<int> knownNodes;

  Camellia::FieldContainer<double> valuesAtNodes(linearBasis->getCardinality(), cellTopo->getNodeCount()); // (F, P)

  linearBasis->getValues(valuesAtNodes, refCellNodes, OP_VALUE);

  for (int basisOrdinal=0; basisOrdinal < linearBasis->getCardinality(); basisOrdinal++)
  {
    for (int nodeOrdinal=0; nodeOrdinal < cellTopo->getNodeCount(); nodeOrdinal++)
    {
      if (valuesAtNodes(basisOrdinal, nodeOrdinal) != 0.0)
      {
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

TEUCHOS_UNIT_TEST( Basis, DofOrdinalsAssigned_PointBasis )
{
  BasisPtr pointBasis = Teuchos::rcp( new PointBasis<> );
  testBasisAllDofOrdinalsAssignedToSubcells(pointBasis, out, success);
}

// Define the templated unit test.
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Basis, ScalarPolynomialBasisUnisolvence, ScalarPolynomialBasisType )
{
  int polyOrder = 1;
  ScalarPolynomialBasisType basis(polyOrder, Intrepid::POINTTYPE_SPECTRAL);

  shards::CellTopology cellTopo = basis.getBaseCellTopology();

  Camellia::FieldContainer<double> dofCoords(basis.getCardinality(), cellTopo.getDimension());
  basis.getDofCoords(dofCoords);

  Camellia::FieldContainer<double> valuesAtNodes(basis.getCardinality(), dofCoords.dimension(0)); // (F, P)

  basis.getValues(valuesAtNodes, dofCoords, Intrepid::OPERATOR_VALUE);

  for (int basisOrdinal=0; basisOrdinal < basis.getCardinality(); basisOrdinal++)
  {
    for (int nodeOrdinal=0; nodeOrdinal < dofCoords.dimension(0); nodeOrdinal++)
    {
      if (basisOrdinal==nodeOrdinal)
      {
        TEST_ASSERT(valuesAtNodes(basisOrdinal,nodeOrdinal) == 1.0);
      }
      else
      {
        TEST_ASSERT(valuesAtNodes(basisOrdinal,nodeOrdinal) == 0.0);
      }
    }
  }
}

//
// Instantiate the unit test for various values of RealType.
//
// Typedefs to work around Bug 5757 (TYPE values cannot have spaces).
  using ArrayScalar = ::Intrepid::FieldContainer<double>;
  using HGRAD_LINE_TYPE = ::Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, ArrayScalar >;
  
  using HGRAD_QUAD_TYPE = ::Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, ArrayScalar >;
  using HDIV_QUAD_TYPE  = ::Intrepid::Basis_HDIV_QUAD_In_FEM <double, ArrayScalar >;

  using HGRAD_HEX_TYPE  = ::Intrepid::Basis_HGRAD_HEX_Cn_FEM<double, ArrayScalar >;
  using HDIV_HEX_TYPE   = ::Intrepid::Basis_HDIV_HEX_In_FEM <double, ArrayScalar >;

TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_LINE_TYPE )

TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_QUAD_TYPE )

TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Basis, ScalarPolynomialBasisUnisolvence, HGRAD_HEX_TYPE )

  TEUCHOS_UNIT_TEST( Basis, HDIV_Triangle_L2_Eigenvalues )
  {
    Space space = HDIV;
    success = true;
    // Intrepid2's nodal basis fails this test for k > 11, presumably due to ill conditioning.  Would be interesting to try the hierarchical basis.
    const int max_k = 11; // Intrepid2::Parameters::MaxOrder; // we're now using Intrepid2 basis functions for HDIV…
    for (int k=1; k<=max_k; k++) // poly order k
    {
      bool newSuccess = true;
      testTriangleBasisL2Eigenvalues(k, space, out, newSuccess);
      if (!newSuccess)
      {
        out << "******** Failed test for k = " << k << " ***********\n";
      }
      success = newSuccess && success;
    }
  }
  
  TEUCHOS_UNIT_TEST( Basis, HGRAD_Triangle_L2_Eigenvalues )
  {
    Space space = HGRAD;
    for (int k=1; k<9; k++) // poly order k
    {
      testTriangleBasisL2Eigenvalues(k, space, out, success);
    }
  }

  TEUCHOS_UNIT_TEST( Basis, HGRAD_ORTH_Triangle_L2_Eigenvalues )
  {
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
    int polyOrder = 5;
    int scalarRank = 0;
    int spaceDim = 2;
    BasisPtr basis = Teuchos::rcp( new IntrepidBasisWrapper< double, ArrayScalar >( Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM_ORTH<double, ArrayScalar >(polyOrder)), spaceDim, scalarRank, fs) );
    
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr H1Var = vf->testVar("v", HGRAD);
    
    testBasisL2Eigenvalues(H1Var, basis, polyOrder, out, success);
  }
        
  TEUCHOS_UNIT_TEST( Basis, HDIV_Triangle_L2_Eigenvalues_Equispaced )
  {
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
    int polyOrder = 5;
    int vectorRank = 1;
    int spaceDim = 2;
    BasisPtr basis = Teuchos::rcp( new IntrepidBasisWrapper< double, ArrayScalar >( Teuchos::rcp( new Intrepid::Basis_HDIV_TRI_In_FEM<double, ArrayScalar >(polyOrder, Intrepid::POINTTYPE_EQUISPACED)), spaceDim, vectorRank, fs) );
    
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr H1Var = vf->testVar("v", HDIV);
    
    testBasisL2Eigenvalues(H1Var, basis, polyOrder, out, success);
  }
   
// the fact that this test fails is the reason why we have, for now, disabled POINTTYPE_WARPBLEND for the HDIV triangle basis in BasisFactory.
// since we don't use Intrepid::Basis_HDIV_TRI_In_FEM in this mode, we disable this test for now.
// (If/when we come up with a patch, we can revisit.)
  TEUCHOS_UNIT_TEST( Basis, HDIV_Triangle_L2_Eigenvalues_Warpblend )
  {
    out << "\n\n******* Testing linear independence of Intrepid Basis_HDIV_TRI_In_FEM *******";
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
    int polyOrder = 5;
    int vectorRank = 1;
    int spaceDim = 2;
    BasisPtr basis = Teuchos::rcp( new IntrepidBasisWrapper< double, ArrayScalar >( Teuchos::rcp( new Intrepid::Basis_HDIV_TRI_In_FEM<double, ArrayScalar >(polyOrder, Intrepid::POINTTYPE_WARPBLEND)), spaceDim, vectorRank, fs) );
    
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr H1Var = vf->testVar("v", HDIV);
    
    testBasisL2Eigenvalues(H1Var, basis, polyOrder, out, success);
    out << "\n\n*************************************************************************************\n";
    out <<     "**   Failure of this test likely indicates that you have not patched Intrepid      **\n";
    out <<     "** Basis_HDIV_TRI_In_FEM with the patch supplied with the Camellia distribution.   **\n";
    out <<     "*************************************************************************************\n";
  }
        
  TEUCHOS_UNIT_TEST( Basis, HDIV_Tetrahedron_L2_Eigenvalues_Warpblend )
  {
    Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
    int vectorRank = 1;
    int spaceDim = 3;

    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr H1Var = vf->testVar("v", HDIV);

    for (int polyOrder=1; polyOrder<7; polyOrder++)
    {
      BasisPtr basis = Teuchos::rcp( new IntrepidBasisWrapper< double, ArrayScalar >( Teuchos::rcp( new Intrepid::Basis_HDIV_TET_In_FEM<double, ArrayScalar >(polyOrder, Intrepid::POINTTYPE_WARPBLEND)), spaceDim, vectorRank, fs) );
      
      testBasisL2Eigenvalues(H1Var, basis, polyOrder, out, success);
    }
  }

  TEUCHOS_UNIT_TEST( Basis, HDIV_Quad_L2_Eigenvalues )
  {
    Space space = HDIV;
    success = true;
    for (int k=1; k<9; k++) // poly order k
    {
      bool newSuccess = true;
      testQuadBasisL2Eigenvalues(k, space, out, newSuccess);
      if (!newSuccess)
      {
        out << "******** Failed test for k = " << k << " ***********\n";
      }
      success = newSuccess && success;
    }
  }
  
  TEUCHOS_UNIT_TEST( Basis, HGRAD_Quad_L2_Eigenvalues )
  {
    Space space = HGRAD;
    for (int k=1; k<9; k++) // poly order k
    {
      testQuadBasisL2Eigenvalues(k, space, out, success);
    }
  }
} // namespace
