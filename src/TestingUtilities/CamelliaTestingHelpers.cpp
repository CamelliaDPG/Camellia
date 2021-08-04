//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CamelliaTestingHelpers.h"

using namespace Camellia;

namespace Camellia
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
} // namespace Camellia
