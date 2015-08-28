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
  static MeshPtr loadFromHDF5(TBFPtr<double> bf, string filename);
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

  static MeshGeometryPtr shiftedSquareCylinderGeometry(double xLeft, double xRight, double meshHeight, double squareDiameter);


  // legacy method that originally belonged to Mesh:
  static MeshPtr buildQuadMesh(const Intrepid::FieldContainer<double> &quadBoundaryPoints,
                               int horizontalElements, int verticalElements,
                               TBFPtr<double> bilinearForm,
                               int H1Order, int pTest, bool triangulate=false, bool useConformingTraces=true,
                               map<int,int> trialOrderEnhancements=_emptyIntIntMap,
                               map<int,int> testOrderEnhancements=_emptyIntIntMap);

  // legacy method that originally belonged to Mesh:
  static MeshPtr buildQuadMeshHybrid(const Intrepid::FieldContainer<double> &quadBoundaryPoints,
                                     int horizontalElements, int verticalElements,
                                     TBFPtr<double> bilinearForm,
                                     int H1Order, int pTest, bool useConformingTraces);

  static MeshTopologyPtr importMOABMesh(string filePath);
  
  static MeshPtr intervalMesh(TBFPtr<double> bf, double xLeft, double xRight, int numElements, int H1Order, int delta_k); // 1D equispaced

  static MeshTopologyPtr intervalMeshTopology(double xLeft, double xRight, int numElements); // 1D equispaced

  static MeshPtr quadMesh(Teuchos::ParameterList &parameters);

  static MeshPtr quadMesh(TBFPtr<double> bf, int H1Order, int pToAddTest=2,
                          double width=1.0, double height=1.0,
                          int horizontalElements=1, int verticalElements=1,
                          bool divideIntoTriangles=false,
                          double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

  static MeshPtr quadMeshMinRule(TBFPtr<double> bf, int H1Order, int pToAddTest=2,
                                 double width=1.0, double height=1.0,
                                 int horizontalElements=1, int verticalElements=1,
                                 bool divideIntoTriangles=false,
                                 double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

  static MeshPtr quadMesh(TBFPtr<double> bf, int H1Order, Intrepid::FieldContainer<double> &quadNodes, int pToAddTest=2);

  static void quadMeshCellIDs(Intrepid::FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles);

  static MeshTopologyPtr quadMeshTopology(double width=1.0, double height=1.0,
                                          int horizontalElements=1, int verticalElements=1,
                                          bool divideIntoTriangles=false,
                                          double x0=0.0, double y0=0.0,
                                          vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

  static MeshPtr rectilinearMesh(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts,
                                 int H1Order, int pToAddTest=-1, vector<double> x0 = vector<double>());

  static MeshTopologyPtr rectilinearMeshTopology(vector<double> dimensions, vector<int> elementCounts,
      vector<double> x0 = vector<double>());

  static MeshPtr readMesh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  static MeshPtr readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  static MeshPtr spaceTimeMesh(MeshTopologyPtr spatialMeshTopology, double t0, double t1,
                               TBFPtr<double> bf, int spatialH1Order, int temporalH1Order, int pToAdd);

  static MeshTopologyPtr spaceTimeMeshTopology(MeshTopologyPtr spatialMeshTopology, double t0, double t1, int temporalDivisions=1);
};
}

#endif
