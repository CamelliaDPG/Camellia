#include "MeshTests.h"
#include "MeshFactory.h"

void MeshTests::SetUp()
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr fhat = varFactory.fluxVar("\\widehat{f}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  bilinearForm = Teuchos::rcp( new BF(varFactory) );
  bilinearForm->addTerm(fhat, v);
  bilinearForm->addTerm(sigma1, -v->dx());
  bilinearForm->addTerm(sigma2, -v->dy());
  bilinearForm->addTerm(sigma1, tau->x());
  bilinearForm->addTerm(sigma2, tau->y());
  bilinearForm->addTerm(uhat, tau->dot_normal());
  bilinearForm->addTerm(u, tau->div());
}

//TODO: MeshTestSuite::neighborBasesAgreeOnSides never gets run. Should we implement here?
//TODO: Scrapping old style Poisson tests for the time being will reimplement from scratch later.

TEST_F(MeshTests, TestBasisRefinement)
{
  int basisRank;
  int initialPolyOrder = 3;
  
  EFunctionSpaceExtended hgrad = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  Teuchos::RCP<Basis<double,FieldContainer<double> > > basis = BasisFactory::basisFactory()->getBasis(basisRank, initialPolyOrder, quad_4.getKey(), hgrad);
  // since it's hgrad, that's a problem (hvol would be initialPolyOrder-1)
  EXPECT_EQ(initialPolyOrder, basis->getDegree())
    << "testBasisRefinement: initial BasisFactory call returned a different-degree basis than expected..." << endl
    << "testBasisRefinement: expected: " << initialPolyOrder << "; actual: " << basis->getDegree() << endl;

  int additionalP = 4;
  basis = BasisFactory::basisFactory()->addToPolyOrder(basis, additionalP);
  // since it's hgrad, that's a problem (hvol would be initialPolyOrder-1)
  EXPECT_EQ(initialPolyOrder+additionalP, basis->getDegree())
    << "testBasisRefinement: addToPolyOrder call returned a different-degree basis than expected..." << endl
    << "testBasisRefinement: expected: " << initialPolyOrder+additionalP << "; actual: " << basis->getDegree() << endl;    
}

TEST_F(MeshTests, TestBuildMesh)
{
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
  
  Teuchos::RCP<Mesh> myMesh = MeshFactory::buildQuadMesh(quadPoints, 1, 1, bilinearForm, order, order);
  // some basic sanity checks:
  int numElementsExpected = 1;
  EXPECT_EQ(numElementsExpected, myMesh->numElements())
    << "mymesh->numElements() != numElementsExpected; numElements()=" << myMesh->numElements() << endl;
  EXPECT_TRUE(MeshTestUtility::checkMeshDofConnectivities(myMesh))
    << "MeshTestUtility::checkMeshDofConnectivities failed for 1x1 mesh." << endl;
  
  Teuchos::RCP<Mesh> myMesh2x1 = MeshFactory::buildQuadMesh(quadPoints, 2, 1, bilinearForm, order, order);
  // some basic sanity checks:
  numElementsExpected = 2;
  EXPECT_EQ(numElementsExpected, myMesh2x1->numElements())
    << "myMesh2x1.numElements() != numElementsExpected; numElements()=" << myMesh2x1->numElements() << endl;
  EXPECT_TRUE(MeshTestUtility::checkMeshDofConnectivities(myMesh2x1))
    << "MeshTestUtility::checkMeshDofConnectivities failed for 2x1 mesh." << endl;
}

TEST_F(MeshTests, OtherMeshTests)
{
  EXPECT_TRUE(false) << "Still need to implement the rest of the Mesh Tests." << endl;
}
