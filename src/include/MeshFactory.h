// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MeshFactory.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#ifndef Camellia_debug_MeshFactory_h
#define Camellia_debug_MeshFactory_h

#include "TypeDefs.h"

#include "Mesh.h"

#include <Teuchos_ParameterList.hpp>
#include "EpetraExt_ConfigDefs.h"

// static class for creating meshes

namespace Camellia
{
class MeshFactory
{
  static map<int,int> _emptyIntIntMap; // just defined here to implement a default argument to constructor (there's likely a better way)
public:
  // These versions are all deprecated, new versions should take in a VarFactoryPtr instead of BFPtr
#ifdef HAVE_EPETRAEXT_HDF5
  static MeshPtr loadFromHDF5(TBFPtr<double> bf, string filename, Epetra_CommPtr Comm = Teuchos::null);
#endif
  static MeshPtr hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                            TBFPtr<double> bilinearForm, int H1Order, int pToAddTest);

  static MeshPtr shiftedHemkerMesh(double xLeft, double xRight,
                                   double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                   TBFPtr<double> bilinearForm, int H1Order, int pToAddTest);

  static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius);

  static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight,
      double yBottom, double yTop, double cylinderRadius);

  static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight,
      double yBottom, double yTop,
      double cylinderRadius, double embeddedSquareSideLength);

  static MeshGeometryPtr halfHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius);

  static MeshGeometryPtr confinedCylinderGeometry(double cylinderRadius);
  static MeshGeometryPtr halfConfinedCylinderGeometry(double cylinderRadius);

  static MeshGeometryPtr shiftedSquareCylinderGeometry(double xLeft, double xRight, double meshHeight, double squareDiameter);


  // legacy method that originally belonged to Mesh:
  static MeshPtr buildQuadMesh(const Camellia::FieldContainer<double> &quadBoundaryPoints,
                               int horizontalElements, int verticalElements,
                               TBFPtr<double> bilinearForm,
                               int H1Order, int pTest, bool triangulate=false, bool useConformingTraces=true,
                               map<int,int> trialOrderEnhancements=_emptyIntIntMap,
                               map<int,int> testOrderEnhancements=_emptyIntIntMap);

  // legacy method that originally belonged to Mesh:
  static MeshPtr buildQuadMeshHybrid(const Camellia::FieldContainer<double> &quadBoundaryPoints,
                                     int horizontalElements, int verticalElements,
                                     TBFPtr<double> bilinearForm,
                                     int H1Order, int pTest, bool useConformingTraces);
  
//  static MeshTopologyPtr extrudedMeshTopology(MeshTopologyPtr baseMeshTopology, double v0, double vWidth, int numSubdivisions);
  
  static MeshTopologyPtr extrudedMeshTopology(MeshTopologyPtr baseMeshTopology, std::vector<double> extrusionStartCoords, std::vector<double> extrusionExtents, std::vector<int> extrusionSubdivisions);
  
  static MeshTopologyPtr extrudedMeshTopology(MeshTopologyPtr baseMeshTopology, std::vector<std::vector<double>> extrusionVertices);
  
  // produces a tensor-product mesh topology using intervalMeshTopology() and extrudedMeshTopology()
  static MeshTopologyPtr hypercubeMeshTopology(vector<double> dimensions, vector<int> elementCounts,
                                               vector<double> x0 = vector<double>());
  
  static MeshTopologyPtr importMOABMesh(string filePath, bool readInParallel=false);
  
  static MeshPtr intervalMesh(TBFPtr<double> bf, double xLeft, double xRight, int numElements, int H1Order, int delta_k); // 1D equispaced

  static MeshTopologyPtr intervalMeshTopology(double xLeft, double xRight, int numElements, bool periodicBCs=false); // 1D equispaced
  static MeshTopologyPtr intervalMeshTopology(std::vector<double> vertices, bool periodicBCs=false); // arbitrarily spaced 1D
  static MeshTopologyPtr intervalMeshTopology(double xLeft, double xRight, int numElements, vector<PeriodicBCPtr> periodicBCs);

  static MeshPtr minRuleMesh(MeshTopologyViewPtr meshTopo, TBFPtr<double> bf, int H1Order, int delta_k, Teuchos_CommPtr Comm = Teuchos::null, bool allowPruning = true);
  
  static MeshPtr minRuleMesh(MeshTopologyViewPtr meshTopo, TBFPtr<double> bf, vector<int> H1Order, int delta_k,
                             Teuchos_CommPtr Comm = Teuchos::null, bool allowPruning = true);
  
  static MeshPtr quadMesh(Teuchos::ParameterList &parameters, Teuchos_CommPtr Comm = Teuchos::null);

  static MeshPtr quadMesh(TBFPtr<double> bf, int H1Order, int pToAddTest=2,
                          double width=1.0, double height=1.0,
                          int horizontalElements=1, int verticalElements=1,
                          bool divideIntoTriangles=false,
                          double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>(),
                          Teuchos_CommPtr Comm = Teuchos::null);

  static MeshPtr quadMeshMinRule(TBFPtr<double> bf, int H1Order, int pToAddTest=2,
                                 double width=1.0, double height=1.0,
                                 int horizontalElements=1, int verticalElements=1,
                                 bool divideIntoTriangles=false,
                                 double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>(),
                                 Teuchos_CommPtr Comm = Teuchos::null);

  static MeshPtr quadMesh(TBFPtr<double> bf, int H1Order, Camellia::FieldContainer<double> &quadNodes, int pToAddTest=2,
                          Teuchos_CommPtr Comm = Teuchos::null);

  static void quadMeshCellIDs(Camellia::FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles);

  static MeshTopologyPtr quadMeshTopology(double width=1.0, double height=1.0,
                                          int horizontalElements=1, int verticalElements=1,
                                          bool divideIntoTriangles=false,
                                          double x0=0.0, double y0=0.0,
                                          vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

  //! define a rectilinear mesh which disallows MeshTopology "pruning": the deletion of geometric entities that are not required on the local MPI rank.
  static MeshPtr rectilinearMeshNoPruning(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts,
                                          int H1Order, int pToAddTest=-1, vector<double> x0 = vector<double>(),
                                          map<int,int> trialOrderEnhancements = map<int,int>(),
                                          map<int,int> testOrderEnhancements = map<int,int>(),
                                          Teuchos_CommPtr Comm = Teuchos::null);
  
  static MeshPtr rectilinearMesh(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts,
                                 int H1Order, int pToAddTest=-1, vector<double> x0 = vector<double>(),
                                 map<int,int> trialOrderEnhancements = map<int,int>(),
                                 map<int,int> testOrderEnhancements = map<int,int>(),
                                 Teuchos_CommPtr Comm = Teuchos::null,
                                 bool allowMeshTopologyPruning = true);

  static MeshTopologyPtr rectilinearMeshTopology(vector<double> dimensions, vector<int> elementCounts,
                                                 vector<double> x0 = vector<double>(),
                                                 vector<PeriodicBCPtr> periodicBCs = vector<PeriodicBCPtr>());

  static MeshPtr readMesh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  static MeshPtr readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  // returns a newly-constructed MeshTopology that matches the passed-in MeshTopology in its root cells (including cell numbering, but not necessarily vertex or other entity numbering).
  static MeshTopologyPtr rootMeshTopologyCopy(MeshTopologyPtr meshTopology);
  
  static MeshPtr spaceTimeMesh(MeshTopologyPtr spatialMeshTopology, double t0, double t1,
                               TBFPtr<double> bf, int spatialH1Order, int temporalH1Order, int pToAdd,
                               Teuchos_CommPtr Comm = Teuchos::null);

  static MeshTopologyPtr spaceTimeMeshTopology(MeshTopologyPtr spatialMeshTopology, double t0, double t1, int temporalDivisions=1);
  
  static Intrepid2::Data<double, Kokkos::HostSpace> vertexDivisions1D(const double &startCoord, const double &domainExtent, const int &numElements);
};
}

#endif
