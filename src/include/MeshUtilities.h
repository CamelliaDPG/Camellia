// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef MESH_UTIL
#define MESH_UTIL

#include "TypeDefs.h"

#include "Mesh.h"
#include "SpatialFilter.h"

#include "IP.h"

namespace Camellia
{
class MeshUtilities
{
public:

  static SpatialFilterPtr rampBoundary(double rampHeight);
  static MeshPtr buildRampMesh(double rampHeight, BFPtr bilinearForm, int H1Order, int pTest);

  static SpatialFilterPtr longRampBoundary(double rampHeight);
  static MeshPtr buildLongRampMesh(double rampHeight, BFPtr bilinearForm, int H1Order, int pTest);

  static MeshPtr buildFrontFacingStep(BFPtr bilinearForm, int H1Order, int pTest);

  static MeshPtr buildUnitQuadMesh(int horizontalCells, int verticalCells, BFPtr bilinearForm, int H1Order, int pTest);

  static MeshPtr buildUnitQuadMesh(int nCells, BFPtr bilinearForm, int H1Order, int pTest);

  static double computeMaxLocalConditionNumber(IPPtr ip, MeshPtr mesh, bool jacobiScaling=true, string sparseFileToWriteTo="");

};
}

#endif
