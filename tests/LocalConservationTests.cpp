#include "LocalConservationTests.h"

#include "InnerProductScratchPad.h"
#include "Mesh.h"

typedef Teuchos::RCP<Element> ElementPtr;

// void SteadyConservationTests::SetUp()
// {
//   cp.epsilon = 1.0;
//   cp.checkLocalConservation = true;
//   cp.printLocalConservation = false;
//   cp.enforceLocalConservation = true;
//   vector<double> beta;
//   beta.push_back(2.0);
//   beta.push_back(1.0);
//   cp.defineVariables();
//   cp.beta = beta;
//   cp.defineBilinearForm(cp.beta);
//   cp.setRobustZeroMeanIP(cp.beta);
//   cp.defineRightHandSide();
//   cp.defineBoundaryConditions();
//   cp.defineMesh();
// }
// 
// TEST_F(SteadyConservationTests, TestZeroMeanTerm)
// {
//   cp.H1Order = 1;
//   cp.pToAdd = 0;
//   cp.ip = Teuchos::rcp(new IP());
//   cp.ip->addZeroMeanTerm( cp.v );
//   cp.defineMesh();
//   ElementTypePtr elemType = cp.mesh->elementTypes()[0];
//   BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, cp.mesh, true) );
//   FieldContainer<double> ipMat(1, elemType->testOrderPtr->totalDofs(), elemType->testOrderPtr->totalDofs());
//   vector<int> vIndices = elemType->testOrderPtr->getDofIndices(cp.v->ID());
//   vector<int> cellIDs;
//   cellIDs.push_back(0); 
//   bool createSideCacheToo = false;
// 
//   basisCache->setPhysicalCellNodes(cp.mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);
//   cp.ip->computeInnerProductMatrix(ipMat, elemType->testOrderPtr, basisCache);
//   for (int i=0; i < vIndices.size(); i++)
//     for (int j=0; j < vIndices.size(); j++)
//       EXPECT_NEAR(1./16., ipMat(0, vIndices[i], vIndices[j]), 1e-15);
// }
// 
// TEST_F(SteadyConservationTests, TestLocalConservation)
// {
//   char *argv[11] = {"./RunTests"};
//   cp.solveSteady(1, argv);
//   double tol = 1e-14;
//   EXPECT_LE(cp.fluxImbalances[0], tol);
//   EXPECT_LE(cp.fluxImbalances[1], tol);
//   EXPECT_LE(cp.fluxImbalances[2], tol);
// }
// 
// void TransientConservationTests::SetUp()
// {
//   cp.epsilon = 1.0;
//   cp.checkLocalConservation = true;
//   cp.printLocalConservation = false;
//   cp.enforceLocalConservation = true;
//   vector<double> beta;
//   beta.push_back(2.0);
//   beta.push_back(1.0);
//   cp.defineVariables();
//   cp.beta = beta;
//   cp.defineBilinearForm(cp.beta);
//   cp.setRobustZeroMeanIP(cp.beta);
//   cp.defineRightHandSide();
//   cp.defineBoundaryConditions();
//   cp.defineMesh();
// }
