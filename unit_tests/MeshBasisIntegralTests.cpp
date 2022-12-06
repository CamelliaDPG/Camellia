//
// © 2016-2022 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MeshBasisIntegralsTests
//  Camellia
//
//  Created by Nate Roberts on 11/28/22.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "MeshBasisIntegrals.h"

using namespace Camellia;

namespace
{
  //! test turning on exactly one member of the basis at a time (coefficient of 1) in a Solution object; compare the integral of this (using the Function class) with the integral given by MeshBasisIntegrals.
  void basis1DFunctionTest(BasisPtr basis_1D, FunctionPtr weight, int quadratureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    int xElements = 1; // just use a single element
    
    // create a sample 1D Mesh, with just a single trial field variable, f.
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVar("f");
    auto v = vf->testVar("v", L2);
    
    auto bf = BF::bf(vf);
    
    double xLeft  = 0;
    double xRight = 1;
    MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(xLeft, xRight, xElements);
    
    int H1Order  = basis_1D->getDegree() + 1;
    int pToAdd = 0; // Bubnov-Galerkin
    bool allowPruning      = false;
    auto mesh = MeshFactory::minRuleMesh(spatialMeshTopo, bf, H1Order, pToAdd, Teuchos::null, allowPruning);
    
    auto activeCellIDs = mesh->getActiveCellIDsGlobal();
    MeshBasisIntegrals meshBasisIntegrals(mesh, basis_1D, weight, quadratureEnrichment, activeCellIDs);
    
    auto soln = Solution::solution(bf, mesh);
    
    int basisCardinality = basis_1D->getCardinality();
    
    auto solnFxn = Function::solution(f, soln);
    
    int numCells = activeCellIDs.size();
    
    FieldContainer<double> singleSolnCoefficients(basisCardinality);
    FieldContainer<double> solnCoefficients(numCells,basisCardinality,basisCardinality);
    FieldContainer<double> expectedSolnIntegrals(numCells,basisCardinality);
    
    vector<IndexType> cellIDs(activeCellIDs.begin(),activeCellIDs.end());
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      singleSolnCoefficients(basisOrdinal) = 1.0;
      
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        auto cellID = cellIDs[cellOrdinal];
        solnCoefficients(cellOrdinal,basisOrdinal,basisOrdinal) = 1.0;
        soln->setLocalCoefficientsForCell(cellID, singleSolnCoefficients); // local: does not generate coefficients for the global vector object
        
        const double weightedSolnIntegral = (solnFxn * weight)->integrate(mesh, quadratureEnrichment); // TODO: make this more precise; won't be the same quadrature degree as the MeshBasisIntegrals integration -- we are over-integrating by the degree of the test space, more or less.
        expectedSolnIntegrals(cellOrdinal,basisOrdinal) = weightedSolnIntegral;
      }
      
      // set this coefficient back to 0 for the next pass through the loop
      singleSolnCoefficients(basisOrdinal) = 0.0;
    }
    
    FieldContainer<double> actualSolnIntegrals(numCells,basisCardinality);
    auto actualSolnIntegralsView = actualSolnIntegrals.checkoutDeviceView();
    auto solnCoefficientsView = solnCoefficients.checkoutDeviceView();
    meshBasisIntegrals.solutionIntegral(actualSolnIntegralsView, solnCoefficientsView, cellIDs);
    solnCoefficients.checkinDeviceView(false);
    actualSolnIntegrals.checkinDeviceView(true); // true: do copy back to host
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        TEST_FLOATING_EQUALITY(expectedSolnIntegrals(cellOrdinal,basisOrdinal), actualSolnIntegrals(cellOrdinal,basisOrdinal), tol);
      }
    }
  }

TEUCHOS_UNIT_TEST( MeshBasisIntegrals, LineBasis_ConstantWeight_p1 )
{
  FunctionPtr weight = Function::constant(M_PI);
  int quadratureEnrichment = 0;
  double tol = 1e-15;
  
  int p = 1; // H^1 order: basis degree will be 0
  EFunctionSpace fs = FUNCTION_SPACE_HVOL;
  
  auto line = CellTopology::line();
  auto basis = BasisFactory::basisFactory()->getBasis(p, line, fs);
  
  basis1DFunctionTest(basis, weight, quadratureEnrichment, tol, out, success);
}

TEUCHOS_UNIT_TEST( MeshBasisIntegrals, LineBasis_ConstantWeight_p2 )
{
  FunctionPtr weight = Function::constant(M_PI);
  int quadratureEnrichment = 0;
  double tol = 1e-15;
  
  int p = 2; // H^1 order: basis degree will be one less
  EFunctionSpace fs = FUNCTION_SPACE_HVOL;
  
  auto line = CellTopology::line();
  auto basis = BasisFactory::basisFactory()->getBasis(p, line, fs);
  
  basis1DFunctionTest(basis, weight, quadratureEnrichment, tol, out, success);
}

TEUCHOS_UNIT_TEST( MeshBasisIntegrals, LineBasis_ConstantWeight_p3 )
{
  FunctionPtr weight = Function::constant(M_PI);
  int quadratureEnrichment = 0;
  double tol = 1e-15;
  
  int p = 3; // H^1 order: basis degree will be one less
  EFunctionSpace fs = FUNCTION_SPACE_HVOL;
  
  auto line = CellTopology::line();
  auto basis = BasisFactory::basisFactory()->getBasis(p, line, fs);
  
  basis1DFunctionTest(basis, weight, quadratureEnrichment, tol, out, success);
}

TEUCHOS_UNIT_TEST( MeshBasisIntegrals, LineBasis_ConstantWeight_p4 )
{
  FunctionPtr weight = Function::constant(M_PI);
  int quadratureEnrichment = 0;
  double tol = 1e-15;
  
  int p = 4; // H^1 order: basis degree will be one less
  EFunctionSpace fs = FUNCTION_SPACE_HVOL;
  
  auto line = CellTopology::line();
  auto basis = BasisFactory::basisFactory()->getBasis(p, line, fs);
  
  basis1DFunctionTest(basis, weight, quadratureEnrichment, tol, out, success);
}

} // namespace
