#include "StandardConfusionProblem.h"

#include "Mesh.h"

class InflowBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x) < tol) ;
      bool yMatch = (abs(y) < tol) ;
      return xMatch || yMatch;
    }
};

class OutflowBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x-1.0) < tol);
      bool yMatch = (abs(y-1.0) < tol);
      return xMatch || yMatch;
    }
};

// boundary value for u
class ZeroBC : public Function {
  public:
    ZeroBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      double tol=1e-14;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

// boundary value for sigma_n
class InletBC : public Function {
  public:
    InletBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      double tol=1e-14;
      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(y) < tol)
            values(cellIndex, ptIndex) = 1 - x;
          else if (abs(x) < tol)
            values(cellIndex, ptIndex) = 1 - y;
          else
            values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

////////////////////   DEFINE INNER PRODUCT   ///////////////////////
void StandardConfusionProblem::defineInnerProduct(vector<double> beta)
{
  setRobustIP(beta);
}

////////////////////   CREATE BCs   ///////////////////////
void StandardConfusionProblem::defineBoundaryConditions()
{
  bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  FunctionPtr u0 = Teuchos::rcp( new ZeroBC );
  bc->addDirichlet(uhat, outflowBoundary, u0);

  FunctionPtr u_inlet = Teuchos::rcp( new InletBC );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_inlet);
}

////////////////////   BUILD MESH   ///////////////////////
void StandardConfusionProblem::defineMesh()
{
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) = 0.0; // x1
  meshBoundary(0,1) = 0.0; // y1
  meshBoundary(1,0) = 1.0;
  meshBoundary(1,1) = 0.0;
  meshBoundary(2,0) = 1.0;
  meshBoundary(2,1) = 1.0;
  meshBoundary(3,0) = 0.0;
  meshBoundary(3,1) = 1.0;

  int horizontalCells = 1, verticalCells = 1;

  // create a pointer to a new mesh:
  mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      confusionBF, H1Order, H1Order+pToAdd, false);
}

void StandardConfusionProblem::runProblem(int argc, char *argv[])
{
  defineVariables();
  beta.push_back(2);
  beta.push_back(1);
  defineBilinearForm(beta);
  defineInnerProduct(beta);
  defineRightHandSide();
  defineBoundaryConditions();
  defineMesh();
  solveSteady(argc, argv, "Confusion");
}
