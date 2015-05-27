#ifndef CAMELLIA_NEW_MESH_TESTS
#define CAMELLIA_NEW_MESH_TESTS

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "MeshTopology.h"

class MeshTopologyTests : public TestSuite
{
private:
  void setup();
  void teardown();

  vector<double> makeVertex(double v0);
  vector<double> makeVertex(double v0, double v1);
  vector<double> makeVertex(double v0, double v1, double v2);

  vector< vector<double> > quadPoints(double x0, double y0, double width, double height);
  vector< vector<double> > hexPoints(double x0, double y0, double z0, double width, double height, double depth) ;
  MeshTopologyPtr makeRectMesh(double x0, double y0, double width, double height,
                               unsigned horizontalCells, unsigned verticalCells);
  MeshTopologyPtr makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
                              unsigned horizontalCells, unsigned verticalCells, unsigned depthCells);
public:
  MeshTopologyTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName()
  {
    return "MeshTopologyTests";
  }

  bool test1DMesh();
  bool test2DMesh();
  bool test3DMesh();

  bool testEntityConstraints();
  bool testCellsForEntity();
  bool testConstraintRelaxation();

  bool testNeighborRelationships();
};

#endif
