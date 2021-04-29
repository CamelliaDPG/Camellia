//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"

#include "Function.h"
#include "MonomialFunctions.h"

using namespace Camellia;

namespace
{
  MeshPtr makeSingleElementExtrudedTestMesh( int baseSpaceDim, int numTensorExtrusions, std::vector<int> H1Orders )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(H1Orders.size() != numTensorExtrusions+1, std::invalid_argument, "number of H1Orders must be 1 for base dim, plus 1 for each extrusion dimension");
    
    MeshPtr mesh;

    std::vector<double> x0        (baseSpaceDim, 0.0);
    std::vector<double> xWidths   (baseSpaceDim, 1.0);
    std::vector<int>    meshWidths(baseSpaceDim, 1  );
    auto baseMeshTopology = MeshFactory::rectilinearMeshTopology(xWidths, meshWidths, x0);
    
    std::vector<double> v0         (numTensorExtrusions, 0.0);
    std::vector<double> vWidths    (numTensorExtrusions, 2.0);
    std::vector<int>    vMeshWidths(numTensorExtrusions, 1);
    MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(baseMeshTopology, v0, vWidths, vMeshWidths);
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr u    = varFactory->fieldVar("u");
    VarPtr uhat = varFactory->fluxVar("uhat");

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = BF::bf(varFactory);

    ////////////////////   BUILD MESH   ///////////////////////
    int pToAdd = 1;
    mesh = Teuchos::rcp( new Mesh (extrudedMeshTopology, bf, H1Orders, pToAdd) );

    return mesh;
  }

  TEUCHOS_UNIT_TEST( MonomialFunction, DerivativeOnExtrudedMesh1D1V )
  {
    const int baseDim = 1;
    const int extrusionDim = 1;
    const double tol = 1e-15;
    std::vector<int> H1Orders {2,2};
    
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y  = Function::yn(1);
    FunctionPtr xy = x * y;
    
    auto xy_dx = xy->dx(); // d/dx ( xy ) = y
    double xy_dx_err = (xy_dx - y)->l2norm(mesh);
    TEST_COMPARE(xy_dx_err, <, tol);
    
    auto xy_dy = xy->dy(); // d/dy ( xy ) = x
    double xy_dy_err = (xy_dy - x)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
    
    // same thing, but now using MonomialFunction class directly
    x = MonomialFunction::xi_n(0,1);
    y = MonomialFunction::xi_n(1,1);
    xy = x * y;
    
    xy_dy = xy->dy(); // d/dy ( xy ) = x
    xy_dx_err = (xy_dx - y)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
    
    xy_dy = xy->dy(); // d/dy ( xy ) = x
    xy_dy_err = (xy_dy - x)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
  }
} // namespace
