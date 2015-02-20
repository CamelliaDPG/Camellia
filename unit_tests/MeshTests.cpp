//
//  MeshTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/19/15.
//
//

#include "PoissonFormulation.h"
#include "MeshFactory.h"

#include <cstdio>

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST( Mesh, SaveAndLoad )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    int H1Order = 2;
    vector<int> elemCounts(2);
    elemCounts[0] = 2;
    elemCounts[1] = 3;
    
    vector<double> dims(2);
    dims[0] = 1.2;
    dims[1] = 1.4;
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dims, elemCounts, H1Order);
    
    string meshFile = "SavedMesh.HDF5";
    mesh->saveToHDF5(meshFile);
    
    MeshPtr loadedMesh = MeshFactory::loadFromHDF5(form.bf(), meshFile);
    TEST_EQUALITY(loadedMesh->globalDofCount(), mesh->globalDofCount());
    
    // delete the file we created
    remove(meshFile.c_str());
  }
} // namespace
