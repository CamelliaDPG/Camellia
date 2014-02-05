#include "ElementTests.h"
#include "MeshFactory.h"

void ElementTests::SetUp()
{
  /**********************************
   
   _________________________________
   |               |               |
   |               |               |
   |               |               |
   |       1       |       3       |
   |               |               |
   |               |               |
   |               |               |
   ---------------------------------
   |       |       |               |
   |   7   |   6   |               |
   |       |       |               |
   |-------0-------|       2       |
   |       | 11| 10|               |
   |   4   |---5---|               |
   |       | 8 | 9 |               |
   ---------------------------------
   
   in the code:
   _sw: 0
   _nw: 1
   _se: 2
   _ne: 3
   _sw_se: 5
   _sw_ne: 6
   _sw_se_se:  9
   _sw_se_ne: 10

   *********************************/
  
  // first, build a simple mesh
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = 2;
  int testOrder = H1Order;
  int horizontalCells = 2; int verticalCells = 2;
  
  double eps = 1.0; // not really testing for sharp gradients right now--just want to see if things basically work
  vector<double> beta_const;
  beta_const.push_back(1.0);
  beta_const.push_back(1.0);

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  _confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  _confusionBF->addTerm(sigma1 / eps, tau->x());
  _confusionBF->addTerm(sigma2 / eps, tau->y());
  _confusionBF->addTerm(u, tau->div());
  _confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  _confusionBF->addTerm( sigma1, v->dx() );
  _confusionBF->addTerm( sigma2, v->dy() );
  _confusionBF->addTerm( beta_const * u, - v->grad() );
  _confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  _mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _confusionBF, H1Order, testOrder);
      
  // the right way to determine the southwest element, etc. is as follows:
  FieldContainer<double> points(4,2);
  // southwest center:
  points(0,0) = 0.25; points(0,1) = 0.25;
  // southeast center:
  points(1,0) = 0.75; points(1,1) = 0.25;
  // northwest center:
  points(2,0) = 0.25; points(2,1) = 0.75;
  // northeast center:
  points(3,0) = 0.75; points(3,1) = 0.75;
  vector<ElementPtr> elements = _mesh->elementsForPoints(points);
  
  _sw = elements[0]; // as presently implemented, cellID = 0
  _se = elements[1]; // as presently implemented, cellID = 2
  _nw = elements[2]; // as presently implemented, cellID = 1
  _ne = elements[3]; // as presently implemented, cellID = 3

  vector<GlobalIndexType> cellIDsToRefine;
  cellIDsToRefine.push_back(_sw->cellID());
  
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  // get the new elements; these are all within the original SW quadrant
  // southwest center:
  points(0,0) = 0.125; points(0,1) = 0.125;
  // southeast center:
  points(1,0) = 0.375; points(1,1) = 0.125;
  // northwest center:
  points(2,0) = 0.125; points(2,1) = 0.375;
  // northeast center:
  points(3,0) = 0.375; points(3,1) = 0.375;
  
  elements = _mesh->elementsForPoints(points);
  _sw_se = elements[1]; // as presently implemented, cellID = 5
  _sw_ne = elements[3]; // as presently implemented, cellID = 6
  
  cellIDsToRefine.clear();
  cellIDsToRefine.push_back(_sw_se->cellID());
  
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());

  // get the new elements; these are all within the SE of the original SW quadrant
  // southwest center:
  points(0,0) = 0.31125; points(0,1) = 0.31125;
  // southeast center:
  points(1,0) = 0.4375; points(1,1) = 0.31125;
  // northwest center:
  points(2,0) = 0.31125; points(2,1) = 0.4375;
  // northeast center:
  points(3,0) = 0.4375; points(3,1) = 0.4375;  
  
  elements = _mesh->elementsForPoints(points);
  _sw_se_se = elements[1]; // as presently implemented, cellID =  9
  _sw_se_ne = elements[3]; // as presently implemented, cellID = 10
  
  // setup test points:
  static const int NUM_POINTS_1D = 3;
  double x[NUM_POINTS_1D] = {-0.5,0.25,0.75};
  
  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    _testPoints1D(i, 0) = x[i];
  }
}

TEST_F(ElementTests, TestNeighborPointMapping)
{
  double tol = 1e-15;
  double maxDiff;
  
  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  // determine expected values
  int numPoints = _testPoints1D.dimension(0);

  FieldContainer<double> neighborPointsForSW_east_side = FieldContainer<double>(numPoints,1);
  // universal rule is just flip along (-1,1): x --> -x
  for (int i=0; i<numPoints; i++) {
    neighborPointsForSW_east_side(i, 0) = - _testPoints1D(i,0);
  }
  
  // determine actual values, and compare
  FieldContainer<double> neighborRefPoints = FieldContainer<double>(numPoints,1);
  _sw->getSidePointsInNeighborRefCoords(neighborRefPoints, EAST, _testPoints1D);
  
  EXPECT_TRUE(fcsAgree(neighborRefPoints,neighborPointsForSW_east_side,tol,maxDiff))
    << "Failure in mapping to neighbor ref coords; maxDiff = " << maxDiff << endl;
}

TEST_F(ElementTests, TestParentPointMapping)
{
  double tol = 1e-15;
  double maxDiff;
  
  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  // determine expected values
  int numPoints = _testPoints1D.dimension(0);
  FieldContainer<double> sw_east_points_for_ne_child = FieldContainer<double>(numPoints,1);
  // along the east side, ne child is the second: so rule is add 1 and divide by 2
  for (int i=0; i<numPoints; i++) {
    sw_east_points_for_ne_child(i, 0) = (_testPoints1D(i,0)+1.0)/2.0;
  }
  FieldContainer<double> sw_east_points_for_se_child = FieldContainer<double>(numPoints,1);
  // along the east side, se child is the first: so rule is subtract 1 and divide by 2
  for (int i=0; i<numPoints; i++) {
    sw_east_points_for_se_child(i, 0) = (_testPoints1D(i,0)-1.0)/2.0;
  }
  
  // determine actual values, and compare
  FieldContainer<double> parentRefPoints = FieldContainer<double>(numPoints,1);
  _sw_ne->getSidePointsInParentRefCoords(parentRefPoints, EAST, _testPoints1D);
  EXPECT_TRUE(fcsAgree(parentRefPoints,sw_east_points_for_ne_child,tol,maxDiff))
    << "Failure in mapping to parent ref coords for NE child; maxDiff = " << maxDiff << endl;

  _sw_se->getSidePointsInParentRefCoords(parentRefPoints, EAST, _testPoints1D);
  EXPECT_TRUE(fcsAgree(parentRefPoints,sw_east_points_for_se_child,tol,maxDiff))
    << "Failure in mapping to parent ref coords for SE child; maxDiff = " << maxDiff << endl;
}
