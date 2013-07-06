//
//  MeshFactory.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#ifndef Camellia_debug_MeshFactory_h
#define Camellia_debug_MeshFactory_h

#include "Mesh.h"

// static class for creating meshes

class MeshFactory {
public:
  static MeshPtr quadMesh(BilinearFormPtr bf, int H1Order, int pToAddTest=2,
                          double width=1.0, double height=1.0, int horizontalElements=1, int verticalElements=1);
  
  static MeshPtr quadMesh(BilinearFormPtr bf, int H1Order, FieldContainer<double> &quadNodes, int pToAddTest=2);
  
  static MeshPtr hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                            BilinearFormPtr bilinearForm, int H1Order, int pToAddTest)
  {
    return shiftedHemkerMesh(-meshWidth/2, meshWidth/2, meshHeight, cylinderRadius, bilinearForm, H1Order, pToAddTest);
  }

  static MeshPtr shiftedHemkerMesh(double xLeft, double xRight, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                            BilinearFormPtr bilinearForm, int H1Order, int pToAddTest);

  static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius);
  
  static MeshGeometryPtr shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius);



};

#endif
