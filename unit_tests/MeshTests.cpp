//
//  MeshTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/19/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "GlobalDofAssignment.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "StokesVGPFormulation.h"

#include <cstdio>

using namespace Camellia;
using namespace Intrepid;

namespace
{
MeshPtr makeTestMesh( int spaceDim, bool spaceTime )
{
  MeshPtr mesh;
  if ((spaceDim == 1) && spaceTime)
  {
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(CellTopology::line(), tensorialDegree);

    vector<double> v00 = {-1,-1};
    vector<double> v10 = { 1,-1};
    vector<double> v20 = { 2,-1};
    vector<double> v01 = {-1, 1};
    vector<double> v11 = { 1, 1};
    vector<double> v21 = { 2, 1};

    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00); // 0
    spaceTimeVertices.push_back(v10); // 1
    spaceTimeVertices.push_back(v20); // 2
    spaceTimeVertices.push_back(v01); // 3
    spaceTimeVertices.push_back(v11); // 4
    spaceTimeVertices.push_back(v21); // 5

    vector<unsigned> spaceTimeLine1VertexList;
    vector<unsigned> spaceTimeLine2VertexList;
    spaceTimeLine1VertexList.push_back(0);
    spaceTimeLine1VertexList.push_back(1);
    spaceTimeLine1VertexList.push_back(3);
    spaceTimeLine1VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(1);
    spaceTimeLine2VertexList.push_back(2);
    spaceTimeLine2VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(5);

    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    spaceTimeElementVertices.push_back(spaceTimeLine2VertexList);

    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    spaceTimeCellTopos.push_back(line_x_time);

    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory->fluxVar("uhat");

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = BF::bf(varFactory);

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 3, pToAdd = 1;
    mesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
  }
  else
  {
    // TODO: handle other mesh options
  }
  return mesh;
}

TEUCHOS_UNIT_TEST( Mesh, ParitySpaceTime1D )
{
  int spaceDim = 1;
  bool spaceTime = true;
  MeshPtr spaceTimeMesh = makeTestMesh(spaceDim, spaceTime);

  set<GlobalIndexType> cellIDs = spaceTimeMesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    FieldContainer<double> parities = spaceTimeMesh->globalDofAssignment()->cellSideParitiesForCell(cellID);
    CellPtr cell = spaceTimeMesh->getTopology()->getCell(cellID);
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      double parity = parities[sideOrdinal];
      if (cell->getNeighbor(sideOrdinal,spaceTimeMesh->getTopology()) == Teuchos::null)
      {
        // where there is no neighbor, the parity should be 1.0
        TEST_EQUALITY(parity, 1.0);
      }
      else
      {
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,spaceTimeMesh->getTopology());
        GlobalIndexType neighborCellID = neighborInfo.first;
        unsigned neighborSide = neighborInfo.second;
        FieldContainer<double> neighborParities = spaceTimeMesh->globalDofAssignment()->cellSideParitiesForCell(neighborCellID);
        double neighborParity = neighborParities[neighborSide];
        TEST_EQUALITY(parity, -neighborParity);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Mesh, NormalSpaceTime1D )
{
  int spaceDim = 1;
  bool spaceTime = true;
  MeshPtr spaceTimeMesh = makeTestMesh(spaceDim, spaceTime);

  double tol = 1e-15;
  set<GlobalIndexType> cellIDs = spaceTimeMesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh,cellID);

    CellPtr cell = spaceTimeMesh->getTopology()->getCell(cellID);
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      FieldContainer<double> sideNormalsSpaceTime = basisCache->getSideBasisCache(sideOrdinal)->getSideNormalsSpaceTime();
      int numPoints = sideNormalsSpaceTime.dimension(1);

      // check that the normals are unit length:
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        double lengthSquared = 0;
        for (int d=0; d<spaceTimeMesh->getDimension(); d++)
        {
          lengthSquared += sideNormalsSpaceTime(0,ptOrdinal,d) * sideNormalsSpaceTime(0,ptOrdinal,d);
        }
        double length = sqrt(lengthSquared);
        TEST_FLOATING_EQUALITY(length,1.0,tol);
      }

      if (cell->getNeighbor(sideOrdinal,spaceTimeMesh->getTopology()) != Teuchos::null)
      {
        // then we also want to check that pointwise the normals are opposite each other
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,spaceTimeMesh->getTopology());
        GlobalIndexType neighborCellID = neighborInfo.first;
        unsigned neighborSide = neighborInfo.second;
        BasisCachePtr neighborBasisCache = BasisCache::basisCacheForCell(spaceTimeMesh,neighborCellID);
        FieldContainer<double> neighborSideNormals = neighborBasisCache->getSideBasisCache(neighborSide)->getSideNormalsSpaceTime();

        // NOTE: here we implicitly assume that the normals at each point will be the same, because we don't
        //       do anything to make neighbors' physical points come in the same order.  For now, this is true
        //       of our test meshes.
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          for (int d=0; d<spaceTimeMesh->getDimension(); d++)
          {
            double cell_d = sideNormalsSpaceTime(0,ptOrdinal,d);
            double neighbor_d = neighborSideNormals(0,ptOrdinal,d);
            TEST_FLOATING_EQUALITY(cell_d, -neighbor_d, tol);
          }
        }
      }
    }
  }
}

void testSaveAndLoad2D(BFPtr bf, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 2;
  vector<int> elemCounts = {3,2};
  vector<double> dims = {1.0,2.0};

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dims, elemCounts, H1Order);

  string meshFile = "SavedMesh.HDF5";
  mesh->saveToHDF5(meshFile);

  MeshPtr loadedMesh = MeshFactory::loadFromHDF5(bf, meshFile);
  TEST_EQUALITY(loadedMesh->globalDofCount(), mesh->globalDofCount());

  // delete the file we created
  remove(meshFile.c_str());

  // just to confirm that we can manipulate the loaded mesh:
  set<GlobalIndexType> cellsToRefine;
  cellsToRefine.insert(0);
  loadedMesh->pRefine(cellsToRefine);
}

TEUCHOS_UNIT_TEST( Mesh, SaveAndLoadPoissonConforming )
{
  int spaceDim = 2;
  bool conformingTraces = true;
  PoissonFormulation form(spaceDim,conformingTraces);
  testSaveAndLoad2D(form.bf(), out, success);
}

TEUCHOS_UNIT_TEST( Mesh, SaveAndLoadStokesConforming )
{
  int spaceDim = 2;
  bool conformingTraces = true;
  StokesVGPFormulation form(spaceDim,conformingTraces);
  testSaveAndLoad2D(form.bf(), out, success);
}
} // namespace
