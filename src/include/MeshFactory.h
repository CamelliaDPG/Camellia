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

namespace Camellia {
  template <typename Scalar>
  class TMeshFactory {
    static map<int,int> _emptyIntIntMap; // just defined here to implement a default argument to constructor (there's likely a better way)
  public:
#ifdef HAVE_EPETRAEXT_HDF5
    static TMeshPtr<Scalar> loadFromHDF5(TBFPtr<Scalar> bf, string filename);
#endif
    static TMeshPtr<Scalar> hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                              TBFPtr<Scalar> bilinearForm, int H1Order, int pToAddTest);

    static TMeshPtr<Scalar> shiftedHemkerMesh(double xLeft, double xRight,
                                     double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                     TBFPtr<Scalar> bilinearForm, int H1Order, int pToAddTest);

    static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius);

    static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight,
                                                 double yBottom, double yTop, double cylinderRadius);

    static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight,
                                                 double yBottom, double yTop,
                                                 double cylinderRadius, double embeddedSquareSideLength);


    // legacy method that originally belonged to Mesh:
    static TMeshPtr<Scalar> buildQuadMesh(const Intrepid::FieldContainer<double> &quadBoundaryPoints,
                                 int horizontalElements, int verticalElements,
                                 TBFPtr<Scalar> bilinearForm,
                                 int H1Order, int pTest, bool triangulate=false, bool useConformingTraces=true,
                                 map<int,int> trialOrderEnhancements=_emptyIntIntMap,
                                 map<int,int> testOrderEnhancements=_emptyIntIntMap);

    // legacy method that originally belonged to Mesh:
    static TMeshPtr<Scalar> buildQuadMeshHybrid(const Intrepid::FieldContainer<double> &quadBoundaryPoints,
                                       int horizontalElements, int verticalElements,
                                       TBFPtr<Scalar> bilinearForm,
                                       int H1Order, int pTest, bool useConformingTraces);

    static TMeshPtr<Scalar> intervalMesh(TBFPtr<Scalar> bf, double xLeft, double xRight, int numElements, int H1Order, int delta_k); // 1D equispaced

    static MeshTopologyPtr intervalMeshTopology(double xLeft, double xRight, int numElements); // 1D equispaced

    static TMeshPtr<Scalar> quadMesh(Teuchos::ParameterList &parameters);

    static TMeshPtr<Scalar> quadMesh(TBFPtr<Scalar> bf, int H1Order, int pToAddTest=2,
                            double width=1.0, double height=1.0,
                            int horizontalElements=1, int verticalElements=1,
                            bool divideIntoTriangles=false,
                            double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

    static TMeshPtr<Scalar> quadMeshMinRule(TBFPtr<Scalar> bf, int H1Order, int pToAddTest=2,
                                   double width=1.0, double height=1.0,
                                   int horizontalElements=1, int verticalElements=1,
                                   bool divideIntoTriangles=false,
                                   double x0=0.0, double y0=0.0, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

    static TMeshPtr<Scalar> quadMesh(TBFPtr<Scalar> bf, int H1Order, Intrepid::FieldContainer<double> &quadNodes, int pToAddTest=2);

    static void quadMeshCellIDs(Intrepid::FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles);

    static MeshTopologyPtr quadMeshTopology(double width=1.0, double height=1.0,
                                            int horizontalElements=1, int verticalElements=1,
                                            bool divideIntoTriangles=false,
                                            double x0=0.0, double y0=0.0,
                                            vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());

    static TMeshPtr<Scalar> rectilinearMesh(TBFPtr<Scalar> bf, vector<double> dimensions, vector<int> elementCounts,
                                   int H1Order, int pToAddTest=-1, vector<double> x0 = vector<double>());

    static MeshTopologyPtr rectilinearMeshTopology(vector<double> dimensions, vector<int> elementCounts,
                                                   vector<double> x0 = vector<double>());

    static TMeshPtr<Scalar> readMesh(string filePath, TBFPtr<Scalar> bilinearForm, int H1Order, int pToAdd);

    static TMeshPtr<Scalar> readTriangle(string filePath, TBFPtr<Scalar> bilinearForm, int H1Order, int pToAdd);

    static TMeshPtr<Scalar> spaceTimeMesh(MeshTopologyPtr spatialMeshTopology, double t0, double t1,
                                 TBFPtr<Scalar> bf, int spatialH1Order, int temporalH1Order, int pToAdd);

    static MeshTopologyPtr spaceTimeMeshTopology(MeshTopologyPtr spatialMeshTopology, double t0, double t1);
  };

  extern template class TMeshFactory<double>;
}

#endif
