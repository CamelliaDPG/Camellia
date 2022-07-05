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
    const int max_k = Intrepid2::Parameters::MaxOrder; // we're now using Intrepid2 basis functions for HDIV…
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
    out << "\n\n*************************************************************************************\n";
    out <<     "**   Failure of this test likely indicates that you have not patched Intrepid      **\n";
    out <<     "** Basis_HDIV_TRI_In_FEM with the patch supplied with the Camellia distribution.   **\n";
    out <<     "*************************************************************************************\n";
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
  
  // the following could be used to demonstrate the issue with Intrepid::Basis_HDIV_TRI_In_FEM when POINTTYPE_WARPBLEND is used
  TEUCHOS_UNIT_TEST( Basis, HDIV_Triangle_Degree5_isLI )
  {
    out << "\n\n******* Test against Intrepid Basis_HDIV_TRI_In_FEM *******\n\n";
    
    // not really a full test that it's LI, but a single challenge test arising from the eigenvalues test
    int polyOrder = 5;
    
    // for unpatched basis, the following weights give a function that's numerically zero.
    vector<double> basisWeights = {-0.0271876426528315, 0.0105320315940818, 0.077688497765228, -0.0058632022585512, -0.0353433079244504, -0.00394932333379972, 0.0412750090022222, -0.0097628894272694, 0.0941355362428019, -0.0524904770297655, 0.0800647592849774, -0.00551811017891473, 0.0409184885921116, -0.0527994693991828, -0.00693980719376711, 1.32455787568392e-12, -0.120038174049383, -0.0252799905921949, 0.0070455857152597, 0.0132660908523773, -0.013892148780441, -0.00238159372478065, -0.0243407882936013, -0.0358175923821551, -0.0197612818151589, -0.017852843314715, 0.154252662222855, -0.0263304478825742, -0.0899376366067355, -0.00181412794606629, -0.06766134863319, -0.620424047393435, 0.270645394534335, 0.627892119236821, -0.244562105387417};
    
    int spaceDim = 2;
    vector<double> dimensions = {1.0,1.0};
    int meshWidth = 1;
    vector<int> elementCounts = {meshWidth,meshWidth};
    
    Intrepid::Basis_HDIV_TRI_In_FEM<double, ArrayScalar > basis(polyOrder, Intrepid::POINTTYPE_WARPBLEND);

    Intrepid::DefaultCubatureFactory<double> cubFactory;
    shards::CellTopology triangleTopo(shards::getCellTopologyData<shards::Triangle<3> >() );
    
    Teuchos::RCP<Intrepid::Cubature<double> > cellTopoCub = cubFactory.create(triangleTopo, polyOrder);
    int numPoints;
  
    numPoints = cellTopoCub->getNumPoints();
    
    Camellia::FieldContainer<double> points(numPoints, spaceDim);
    Camellia::FieldContainer<double> weights(numPoints);
    
    cellTopoCub->getCubature(points, weights);
    
    int basisCardinality = basis.getCardinality();
    Camellia::FieldContainer<double> basisValues(basisCardinality,numPoints,spaceDim);
    basis.getValues(basisValues, points, Intrepid::OPERATOR_VALUE);
    
    Camellia::FieldContainer<double> weightedSum(numPoints,spaceDim);
    for (int i=0; i<basisCardinality; i++)
    {
      double weight = basisWeights[i];
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          weightedSum(pointOrdinal,d) += weight * basisValues(i,pointOrdinal,d);
        }
      }
    }
    
    double maxValueFound = 0;
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        maxValueFound = max(maxValueFound, abs(weightedSum(pointOrdinal,d)));
      }
    }
    
    double tol = 1e-10;
    TEST_COMPARE(maxValueFound, >, tol);
    out << "\n\n*************************************************************************************\n";
    out <<     "**   Failure of this test likely indicates that you have not patched Intrepid      **\n";
    out <<     "** Basis_HDIV_TRI_In_FEM with the patch supplied with the Camellia distribution.   **\n";
    out <<     "*************************************************************************************\n";
  }
} // namespace
