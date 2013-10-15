//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "HConvergenceStudy.h"
#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "BasisFactory.h"
#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "NavierStokesFormulation.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
//#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "RefinementHistory.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

#include "ParameterFunction.h"

#include "MeshFactory.h"

#include "SolutionExporter.h"

#include "MassFluxFunction.h"

#include "choice.hpp"
#include "mpi_choice.hpp"

using namespace std;

static double Re = 40;

double radius = 0.5; // cylinder radius; 0.5 because Re is relative to *diameter*
double meshHeight;
double xLeft, xRight;
FunctionPtr inflowSpeed;

bool velocityConditionsTopAndBottom;
bool velocityConditionsRight;
bool streamwiseGradientConditionsRight;

Teuchos::RCP<BCEasy> bc;
Teuchos::RCP<PenaltyConstraints> pc;

Teuchos::RCP<BCEasy> streamBC;

VarFactory varFactory;
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;

// stream Vars required by BCs:
VarPtr phi_hat;

class LeftBoundary : public SpatialFilter {
  double _left;
public:
  LeftBoundary(double xLeft) {
    double tol = 1e-14;
    _left = xLeft + tol;
  }
  bool matchesPoint(double x, double y) {
    bool matches = x < _left;
//    cout << "Left boundary ";
//    if (matches) {
//      cout << "matches";
//    } else {
//      cout << "does not match";
//    }
//    cout << " point (" << x << ", " << y << ")\n";
    return matches;
  }
};

class RightBoundary : public SpatialFilter {
  double _right;
public:
  RightBoundary(double xRight) {
    double tol = 1e-14;
    _right = xRight - tol;
  }
  bool matchesPoint(double x, double y) {
    return x > _right;
  }
};


class TopBoundary : public SpatialFilter {
  double _top;
public:
  TopBoundary(double yTop) {
    double tol = 1e-14;
    _top = yTop - tol;
  }
  bool matchesPoint(double x, double y) {
    return y > _top;
  }
};

class BottomBoundary : public SpatialFilter {
  double _bottom;
public:
  BottomBoundary(double yBottom) {
    double tol = 1e-14;
    _bottom = yBottom + tol;
  }
  bool matchesPoint(double x, double y) {
    return y < _bottom;
  }
};

class NearCylinder : public SpatialFilter {
  double _enlarged_radius;
public:
  NearCylinder(double radius) {
    double enlargement_factor = 1.2;
    _enlarged_radius = radius * enlargement_factor;
  }
  bool matchesPoint(double x, double y) {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius) {
      return true;
    } else {
      return false;
    }
  }
};

class BoundaryVelocity : public SimpleFunction {
  double _left, _right, _top, _bottom, _radius;
  int _comp;
public:
  BoundaryVelocity(double width, double height, double radius, int component) {
    double tol = 1e-14;
    _left = - width / 2.0 + tol;
    _right = width / 2.0 - tol;
    _top = height / 2.0 - tol;
    _bottom = - height / 2.0 + tol;
    _radius = radius;
    _comp = component; // 0 for x, 1 for y
  }
  
  double value(double x, double y) {
    // widen the radius to allow for some geometry error
    double enlarged_radius = _radius * 1.2;
    if (x*x + y*y < enlarged_radius * enlarged_radius) {
      // then we're on the cylinder
      return 0.0;
    }
    if (_comp == 0) { // u0 == 1 everywhere that we set u0
      return 1.0;
    } else {
      return 0.0;
    }
  }
};

FieldContainer<double> pointGrid(double xMin, double xMax, double yMin, double yMax, int numPoints) {
  vector<double> points1D_x, points1D_y;
  for (int i=0; i<numPoints; i++) {
    points1D_x.push_back( xMin + (xMax - xMin) * ((double) i) / (numPoints-1) );
    points1D_y.push_back( yMin + (yMax - yMin) * ((double) i) / (numPoints-1) );
  }
  int spaceDim = 2;
  FieldContainer<double> points(numPoints*numPoints,spaceDim);
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      points(pointIndex,0) = points1D_x[i];
      points(pointIndex,1) = points1D_y[j];
    }
  }
  return points;
}

FieldContainer<double> solutionDataFromRefPoints(FieldContainer<double> &refPoints, SolutionPtr solution, VarPtr u) {
  int numPointsPerCell = refPoints.dimension(0);

  MeshPtr mesh = solution->mesh();
  int numCells = mesh->numActiveElements();
  int numPoints = numCells * numPointsPerCell;
  FieldContainer<double> xyzData(numPoints, 3);
  
  vector< ElementTypePtr > elementTypes = mesh->elementTypes(); // global element types list
  vector< ElementTypePtr >::iterator elemTypeIt;

  int globalPtIndex = 0;
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, mesh));
    basisCache->setRefCellPoints(refPoints);
    
    vector<int> globalCellIDs = mesh->cellIDsOfTypeGlobal(elemTypePtr);
    
    FieldContainer<double> solutionValues(globalCellIDs.size(),numPointsPerCell);
    FieldContainer<double> physicalCellNodesForType = mesh->physicalCellNodesGlobal(elemTypePtr);

    basisCache->setPhysicalCellNodes(physicalCellNodesForType, globalCellIDs, false); // false: don't create side cache
    
    solution->solutionValues(solutionValues, u->ID(), basisCache);
    
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    
    for (int cellIndex = 0; cellIndex < globalCellIDs.size(); cellIndex++) {
      for (int localPtIndex=0; localPtIndex<numPointsPerCell; localPtIndex++) {
        xyzData(globalPtIndex,0) = physicalPoints(cellIndex,localPtIndex,0);
        xyzData(globalPtIndex,1) = physicalPoints(cellIndex,localPtIndex,1);
        xyzData(globalPtIndex,2) = solutionValues(cellIndex,localPtIndex);
        globalPtIndex++;
      }
    }
  }
  
  return xyzData;
}

set<double> logContourLevels(double height, int numPointsTop=50) {
  set<double> levels;
  double level = height;
  for (int i=0; i<numPointsTop; i++) {
    levels.insert(level);
    levels.insert(-level);
    level /= 2.0;
  }
  return levels;
}

vector< int > cellIDsForVertices(MeshPtr mesh, const FieldContainer<double> &vertices) {
  // this method not meant to be efficient: searches vertices in a brute force way
  int numVertices = vertices.dimension(0);
  int spaceDim = vertices.dimension(1);
  vector< int > cellIDs(numVertices);
  
  double tol = 1e-14;
  
  int numSides = 4; // only quads supported right now
  FieldContainer<double> cellVertices(numSides, spaceDim);
  
  vector< ElementPtr > activeElements = mesh->activeElements();
  for ( vector< ElementPtr >::iterator elemIt = activeElements.begin();
       elemIt != activeElements.end(); elemIt++) {
    int cellID = (*elemIt)->cellID();
    mesh->verticesForCell(cellVertices, cellID);
    for (int i=0; i<numVertices; i++) {
      for (int vertexIndex=0; vertexIndex<numSides; vertexIndex++) {
        int matches = true;
        for (int d=0; d<spaceDim; d++) {
          if (abs(cellVertices(vertexIndex,d) - vertices(i,d)) > tol ) {
            matches = false;
          }
        }
        if (matches) {
          cellIDs[i] = cellID;
        }
      }
    }
  }
  return cellIDs;
}

FunctionPtr friction(SolutionPtr soln) {
  // friction is given by (sigma n) x n (that's a cross product)
  FunctionPtr n = Function::normal();
  LinearTermPtr f_lt = n->y() * (sigma11->times_normal_x() + sigma12->times_normal_y())
                     - n->x() * (sigma21->times_normal_x() + sigma22->times_normal_y());
  
  FunctionPtr f = Teuchos::rcp( new PreviousSolutionFunction(soln, f_lt) );
  return f;
}

double dragCoefficient(SolutionPtr soln, double radius, bool neglectPressure = false) {
  // a more efficient way of doing this would be to actually identify the cells on the boundary
  // or the ones near the cylinder, and only do the integral over those.  The present approach
  // just ensures that the integrand will be zero except in the region of interest...
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  
  FunctionPtr frictionFxn = friction(soln);
  
  FunctionPtr pressure = Teuchos::rcp( new PreviousSolutionFunction(soln, p) );
  
  if (neglectPressure) {
    pressure = Function::zero();
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
  
  // taken from Schäfer and Turek.  We negate everything because our normals are relative to
  // elements, whereas the normal in the formula is going out from the cylinder...
  FunctionPtr dF_D = Teuchos::rcp( new SpatiallyFilteredFunction( (- frictionFxn * n->y() + pressure * n->x()) * boundaryRestriction,
                                                                 nearCylinder));
  
  double F_D = dF_D->integrate(soln->mesh());
  
  return 2 * F_D;
}

double liftCoefficient(SolutionPtr soln, double radius, bool neglectPressure = false) {
  // a more efficient way of doing this would be to actually identify the cells on the boundary
  // or the ones near the cylinder, and only do the integral over those.  The present approach
  // just ensures that the integrand will be zero except in the region of interest...
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  
  FunctionPtr frictionFxn = friction(soln);
  
  FunctionPtr pressure = Teuchos::rcp( new PreviousSolutionFunction(soln, p) );
  
  if (neglectPressure) {
    pressure = Function::zero();
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr boundaryRestriction = Function::meshBoundaryCharacteristic();
  
  // taken from Schäfer and Turek.  We negate everything because our normals are relative to
  // elements, whereas the normal in the formula is going out from the cylinder...
  FunctionPtr dF_L = Teuchos::rcp( new SpatiallyFilteredFunction( (frictionFxn * n->x() + pressure * n->y()) * boundaryRestriction,
                                                                 nearCylinder));
  
  double F_L = dF_L->integrate(soln->mesh());
  
  return 2 * F_L;
}

double pressureDifference(FunctionPtr pressure, double radius, MeshPtr mesh) {
  // first thing: find elements for vertices (-r, 0) and (r, 0)
  // (we're using here the fact that these start out as element vertices, and therefore remain such,
  //  as well as the fact that our geometry transformation leaves vertices unmoved.)
  int numPoints = 2; // front and rear
  int spaceDim = 2;
  int numSides = 4; // only quads supported right now
  FieldContainer<double> points(numPoints,spaceDim);
  points(0,0) = -radius;
  points(0,1) = 0;
  points(1,0) = radius;
  points(1,1) = 0;
  vector< int > cellIDs = cellIDsForVertices(mesh, points);
  // find the vertex indices of the points in their respective elements
  int leftPointVertexIndex = -1, rightPointVertexIndex = -1;
  FieldContainer<double> leftElementVertices(numSides,spaceDim);
  FieldContainer<double> rightElementVertices(numSides,spaceDim);
  
  mesh->verticesForCell( leftElementVertices, cellIDs[0]);
  mesh->verticesForCell(rightElementVertices, cellIDs[1]);

  double tol = 1e-14;
  for (int vertexIndex=0; vertexIndex < numSides; vertexIndex++) {
    if (   (abs(leftElementVertices(vertexIndex,0) - points(0,0)) < tol)
        && (abs(leftElementVertices(vertexIndex,1) - points(0,1)) < tol) )
    {
      leftPointVertexIndex = vertexIndex;
    }
    if (   (abs(rightElementVertices(vertexIndex,0) - points(1,0)) < tol)
        && (abs(rightElementVertices(vertexIndex,1) - points(1,1)) < tol) )
    {
      rightPointVertexIndex = vertexIndex;
    }
  }
  if (leftPointVertexIndex == -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Couldn't find leftPointVertexIndex");
  }
  if (rightPointVertexIndex == -1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Couldn't find rightPointVertexIndex");
  }
  FieldContainer<double> referenceVertices = RefinementPattern::noRefinementPatternQuad()->verticesOnReferenceCell();
  BasisCachePtr leftBasisCache = BasisCache::basisCacheForCell(mesh, cellIDs[0]);
  referenceVertices.resize(numSides,spaceDim); // reshape to get rid of cellIndex dimension
  leftBasisCache->setRefCellPoints(referenceVertices);
  
  BasisCachePtr rightBasisCache = BasisCache::basisCacheForCell(mesh, cellIDs[1]);
  referenceVertices.resize(numSides,spaceDim); // reshape to get rid of cellIndex dimension
  rightBasisCache->setRefCellPoints(referenceVertices);
  
  FieldContainer<double> leftValues(1,numSides);
  FieldContainer<double> rightValues(1,numSides);
  
  pressure->values(leftValues, leftBasisCache);
  pressure->values(rightValues, rightBasisCache);
  
  double leftValue = leftValues(0,leftPointVertexIndex);
  double rightValue = rightValues(0,rightPointVertexIndex);
  
//  cout << "left physical point for pressure computation: (" << leftBasisCache->getPhysicalCubaturePoints()(0,leftPointVertexIndex,0);
//  cout << ", " << leftBasisCache->getPhysicalCubaturePoints()(0,leftPointVertexIndex,1) << ")\n";
//  
//  cout << "right physical point for pressure computation: (" << rightBasisCache->getPhysicalCubaturePoints()(0,rightPointVertexIndex,0);
//  cout << ", " << rightBasisCache->getPhysicalCubaturePoints()(0,rightPointVertexIndex,1) << ")\n";
  
  return leftValue - rightValue;
}


void makeRoughlyIsotropic(MeshPtr hemkerMeshNoCurves, double radius, bool enforceOneIrregularity) {
  // start by identifying the various elements: there are 10 of interest to us
  // to find the thin banded elements, note that radius * 3 will be outside the bounding square
  // and that radius / 2 will be inside the band
  FieldContainer<double> elementPoints(10,2);
  // ESE band
  elementPoints(0,0) =   radius * 3;
  elementPoints(0,1) = - radius / 2;
  // ENE band
  elementPoints(1,0) = radius * 3;
  elementPoints(1,1) = radius / 2;
  // WSW band
  elementPoints(2,0) = - radius * 3;
  elementPoints(2,1) = - radius / 2;
  // WNW band
  elementPoints(3,0) = - radius * 3;
  elementPoints(3,1) =   radius / 2;
  // NNE band
  elementPoints(4,0) =   radius / 2;
  elementPoints(4,1) =   radius * 3;
  // NNW band
  elementPoints(5,0) = - radius / 2;
  elementPoints(5,1) =   radius * 3;
  // SSE band
  elementPoints(6,0) =   radius / 2;
  elementPoints(6,1) = - radius * 3;
  // SSE band
  elementPoints(7,0) = - radius / 2;
  elementPoints(7,1) = - radius * 3;
  // NE big element
  elementPoints(8,0) = radius * 3;
  elementPoints(8,1) = radius * 3;
  // SE big element
  elementPoints(9,0) =   radius * 3;
  elementPoints(9,1) = - radius * 3;

  vector< ElementPtr > elements = hemkerMeshNoCurves->elementsForPoints(elementPoints);
  
  vector<int> horizontalBandCellIDs;
  horizontalBandCellIDs.push_back(elements[0]->cellID());
  horizontalBandCellIDs.push_back(elements[1]->cellID());
  horizontalBandCellIDs.push_back(elements[2]->cellID());
  horizontalBandCellIDs.push_back(elements[3]->cellID());
  
  vector<int> verticalBandCellIDs;
  verticalBandCellIDs.push_back(elements[4]->cellID());
  verticalBandCellIDs.push_back(elements[5]->cellID());
  verticalBandCellIDs.push_back(elements[6]->cellID());
  verticalBandCellIDs.push_back(elements[7]->cellID());
  
  // the bigger, fatter guys in the corners count as horizontal bands (because that's the direction of their anisotropy)
  horizontalBandCellIDs.push_back(elements[8]->cellID());
  horizontalBandCellIDs.push_back(elements[9]->cellID());
  
  Teuchos::RCP<RefinementPattern> verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
  Teuchos::RCP<RefinementPattern> horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuad();
  
  FieldContainer<double> vertices(4,2);
  
  // horizontal bands want vertical cuts, and vice versa
  for (vector<int>::iterator cellIDIt = horizontalBandCellIDs.begin();
       cellIDIt != horizontalBandCellIDs.end(); cellIDIt++) {
    int cellID = *cellIDIt;
//    cout << "Identified cell " << cellID << " as a horizontal band.\n";
    // work out what the current aspect ratio is
    hemkerMeshNoCurves->verticesForCell(vertices, cellID);
//    cout << "vertices for cell " << cellID << ":\n" << vertices;
    // here, we use knowledge of the implementation of the hemker mesh generation:
    // we know that the first edges are always horizontal...
    double xDiff = abs(vertices(1,0)-vertices(0,0));
    double yDiff = abs(vertices(2,1)-vertices(1,1));
    
    set<int> cellIDsToRefine;
    cellIDsToRefine.insert(cellID);
    double aspect = xDiff / yDiff;
    while (aspect > 2.0) {
      hemkerMeshNoCurves->hRefine(cellIDsToRefine, verticalCut);
      
      // the next set of cellIDsToRefine are the children of the ones just refined
      set<int> childCellIDs;
      for (set<int>::iterator refinedCellIDIt = cellIDsToRefine.begin();
           refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++) {
        int refinedCellID = *refinedCellIDIt;
        set<int> refinedCellChildren = hemkerMeshNoCurves->getElement(refinedCellID)->getDescendants();
        childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
      }
      
      cellIDsToRefine = childCellIDs;
      aspect /= 2;
    }
  }
  
  // horizontal bands want vertical cuts, and vice versa
  for (vector<int>::iterator cellIDIt = verticalBandCellIDs.begin();
       cellIDIt != verticalBandCellIDs.end(); cellIDIt++) {
    int cellID = *cellIDIt;
//    cout << "Identified cell " << cellID << " as a vertical band.\n";
    // work out what the current aspect ratio is
    hemkerMeshNoCurves->verticesForCell(vertices, cellID);
    // here, we use knowledge of the implementation of the hemker mesh generation:
    // we know that the first edges are always horizontal...
    double xDiff = abs(vertices(1,0)-vertices(0,0));
    double yDiff = abs(vertices(2,1)-vertices(1,1));
    
    set<int> cellIDsToRefine;
    cellIDsToRefine.insert(cellID);
    double aspect = yDiff / xDiff;
    while (aspect > 2.0) {
      hemkerMeshNoCurves->hRefine(cellIDsToRefine, horizontalCut);
      
      // the next set of cellIDsToRefine are the children of the ones just refined
      set<int> childCellIDs;
      for (set<int>::iterator refinedCellIDIt = cellIDsToRefine.begin();
           refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++) {
        int refinedCellID = *refinedCellIDIt;
        set<int> refinedCellChildren = hemkerMeshNoCurves->getElement(refinedCellID)->getDescendants();
        childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
      }
      
      cellIDsToRefine = childCellIDs;
      aspect /= 2;
    }
  }
  if (enforceOneIrregularity)
    hemkerMeshNoCurves->enforceOneIrregularity();
}

int cornerDescendantID(ElementPtr cell, int side1, int side2) {
  vector< pair<int,int> > cell_descendants_side1_neighbors = cell->getDescendantsForSide(side1); // (cellID, sideIndex) pairs
  set<int> cell_side1_neighbors_set;
  for (int i=0; i<cell_descendants_side1_neighbors.size(); i++) {
    cell_side1_neighbors_set.insert(cell_descendants_side1_neighbors[i].first);
  }
  vector< pair<int,int> > cell0_descendants_cylinder = cell->getDescendantsForSide(side2); // (cellID, sideIndex) pairs
  
  int cornerCellID = -1;
  for (int i=0; i<cell0_descendants_cylinder.size(); i++) {
    int cellID = cell0_descendants_cylinder[i].first;
    if (cell_side1_neighbors_set.find(cellID) != cell_side1_neighbors_set.end()) {
      cornerCellID = cellID;
    }
  }

  return cornerCellID;
}

void printNeighbors(ElementPtr cell) {
  cout << "cell " << cell->cellID() << " neighbors: ";
  for (int i=0; i<4; i++) {
    cout << cell->getNeighborCellID(i) << " ";
  }
  cout << endl;
}

void recreateBCs() { // recreates both bc and pc, as necessary
  int rank     = Teuchos::GlobalMPISession::getRank();
  
  FunctionPtr zero = Function::zero();
  bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  SpatialFilterPtr top          = Teuchos::rcp( new TopBoundary(meshHeight/2.0) );
  SpatialFilterPtr bottom       = Teuchos::rcp( new BottomBoundary(-meshHeight/2.0) );
  SpatialFilterPtr left         = Teuchos::rcp( new LeftBoundary(xLeft) );
  SpatialFilterPtr right        = Teuchos::rcp( new RightBoundary(xRight) );
  
  bc->addDirichlet(u1hat,nearCylinder,zero);
  bc->addDirichlet(u2hat,nearCylinder,zero);
  bc->addDirichlet(u1hat,left,inflowSpeed);
  bc->addDirichlet(u2hat,left,zero);
  
  SpatialFilterPtr topAndBottom = SpatialFilter::unionFilter(top, bottom);
  
  pc = Teuchos::rcp(new PenaltyConstraints);
  
  // define traction components in terms of field variables
  FunctionPtr n = Function::normal();
  LinearTermPtr t1 = n->x() * (2 * sigma11 - p) + n->y() * (sigma12 + sigma21);
  LinearTermPtr t2 = n->x() * (sigma12 + sigma21) + n->y() * (2 * sigma22 - p);
  
  if (velocityConditionsTopAndBottom) {
    bc->addDirichlet(u1hat,topAndBottom,inflowSpeed);
    bc->addDirichlet(u2hat,topAndBottom,zero);
    
    if (velocityConditionsRight) {
      bc->addDirichlet(u1hat,right,inflowSpeed);
      bc->addDirichlet(u2hat,right,zero);
      
      if (rank==0) {
        cout << "velocity conditions everywhere: imposing zero mean on pressure.\n";
      }
      bc->addZeroMeanConstraint(p);
    } else if (streamwiseGradientConditionsRight) {
      if (rank==0)
        cout << "Imposing streamwise gradient == 0 at outflow with penalty constraints.\n";
      pc->addConstraint(sigma11==zero, right);
      pc->addConstraint(sigma21==zero, right);
    } else {
      if (rank==0)
        cout << "Imposing zero traction at outflow with penalty constraints.\n";
      // outflow: both traction components are 0
      pc->addConstraint(t1==zero, right);
      pc->addConstraint(t2==zero, right);
      
    }
  } else { // else, no-traction conditions
    // t1n, t2n are *pseudo*-tractions
    // we use penalty conditions for the true traction
    
    //      if (rank==0)
    //        cout << "EXPERIMENTALLY, imposing zero-mean constraint on pressure.\n";
    //      bc->addZeroMeanConstraint(p);
    
    bool imposeZeroSecondTraction = true;
    
    if (imposeZeroSecondTraction) {
      pc->addConstraint(t1==zero,topAndBottom);
      pc->addConstraint(t2==zero,topAndBottom);
      if (rank==0) {
        cout << "imposing zero second traction (t2) at top and bottom\n";
      }
    } else {
      // at top, we impose u2 = 0 and t1 = 0
      bc->addDirichlet(u2hat, topAndBottom, zero);
      pc->addConstraint(t1==zero,topAndBottom);
      if (rank==0) {
        cout << "imposing zero second velocity (u2) at top and bottom\n";
      }
    }
    
    if (velocityConditionsRight) {
      bc->addDirichlet(u1hat,right,inflowSpeed);
      bc->addDirichlet(u2hat,right,zero);
    } else if (streamwiseGradientConditionsRight) {
      if (rank==0)
        cout << "Imposing streamwise gradient == 0 at outflow with penalty constraints.\n";
      pc->addConstraint(sigma11==zero, right);
      pc->addConstraint(sigma21==zero, right);
    } else {
      // outflow: both traction components are 0
      pc->addConstraint(t1==zero, right);
      pc->addConstraint(t2==zero, right);
    }
    
    if (rank==0)
      cout << "Imposing zero-traction conditions using penalty constraints.\n";
  }
  
  if ( (velocityConditionsRight && velocityConditionsTopAndBottom) ) { // i.e. there are NO penalty constraints -- set to be NULL
    pc = Teuchos::rcp( (PenaltyConstraints *) NULL);
  }
}

void recreateStreamBCs() {
  FunctionPtr zero = Function::zero();
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  SpatialFilterPtr top          = Teuchos::rcp( new TopBoundary(meshHeight/2.0) );
  SpatialFilterPtr bottom       = Teuchos::rcp( new BottomBoundary(-meshHeight/2.0) );
  SpatialFilterPtr left         = Teuchos::rcp( new LeftBoundary(xLeft) );
  SpatialFilterPtr right        = Teuchos::rcp( new RightBoundary(xRight) );
  streamBC = Teuchos::rcp( new BCEasy );
  // wherever we enforce velocity BCs, enforce BCs on phi, too
  // phi, the streamfunction, can be used to measure mass flux between two points
  // reverse engineering that fact, we can use y as the BC for phi
  FunctionPtr y = Function::yn();
  streamBC->addDirichlet(phi_hat, nearCylinder, zero); // had had this commented out; zero makes sense by analogy to the cavity flow problem.
  streamBC->addDirichlet(phi_hat, left, y);
  streamBC->addDirichlet(phi_hat, top, y);
  streamBC->addDirichlet(phi_hat, bottom, y);
}

int main(int argc, char *argv[]) {
  int rank = 0;
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();

#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  
  try {
    // read args:
    int polyOrder = args.Input<int>("--polyOrder", "L^2 (field) polynomial order");
    int numRefs = args.Input<int>("--numRefs", "Number of refinements", 6);
    Re = args.Input<double>("--Re", "Reynolds number", 40);
    
    string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
    string solnFile = args.Input<string>("--solnFile", "file with solution data", "");
    string solnSaveFile = args.Input<string>("--solnSaveFile", "file to which to save solution data", "nsHemker.solution");
    string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "nsHemkerRefinements.replay");

    int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
    double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 1e-8);
    
    meshHeight = args.Input<double>("--meshHeight", "mesh height", 30);
    
    bool useStraightEdgedGeometry = args.Input<bool>("--noCurves", "use straight-edge geometric approximation", false);
    
    double dt = args.Input<double>("--timeStep", "time step (0 for none)", 0); // 0.5 used to be the standard value
    
    bool parabolicInflow = args.Input<bool>("--parabolicInflow", "use parabolic inflow (false for uniform)", false);
    
    bool makeMeshRoughlyIsotropic = args.Input<bool>("--isotropicMesh", "make starting mesh roughly isotropic", true);
    bool enforceOneIrregularity = args.Input<bool>("--oneIrregular", "enforce 1-irregularity", true);
    
    xLeft = args.Input<double>("--xLeft", "x coordinate of the leftmost boundary", -7.5);
    xRight = args.Input<double>("--xRight", "x coordinate of the rightmost boundary", 22.5);
    
    bool refineInPFirst = args.Input<bool>("--pFirst", "prefer p-refinements", false);
    
    velocityConditionsTopAndBottom = args.Input<bool>("--velocityConditionsTopAndBottom", "impose velocity BCs on top and bottom boundaries", false);

    velocityConditionsRight = args.Input<bool>("--velocityConditionsRight", "impose velocity BCs on right boundaries", false);
    
    streamwiseGradientConditionsRight = args.Input<bool>("--streamwiseGradientConditionsRight", "impose streamwise gradient BCs on right boundaries", false);

    bool skipPostProcessing = args.Input<bool>("--skipPostProcessing", "skip computations of mass flux, pressure, and stream solution", false);
    
    bool useMumps = args.Input<bool>("--useMumps", "use MUMPS as global linear solver", true);
    
    bool useZeroInitialGuess = args.Input<bool>("--useZeroInitialGuess", "use zero initial guess (incompatible with BCs, but they're weakly enforced so it's OK)", true);
    
    args.Process();
    
    bool artificialTimeStepping = (dt > 0);
    
    bool useLineSearch = false;
    
    int pToAdd = 2; // for optimal test function approximation
    bool reportPerCellErrors  = true;

    bool startWithZeroSolutionAfterRefinement = false;
    
    bool useCondensedSolve = true;
    
    bool useScaleCompliantGraphNorm = false;
    bool enrichVelocity = useScaleCompliantGraphNorm;
    
    if (useScaleCompliantGraphNorm) {
      cout << "WARNING: useScaleCompliantGraphNorm = true, but support for this is not yet implemented in Hemker driver.\n";
    }
    
    Teuchos::RCP<Solver> solver;
    if (useMumps) {
#ifdef USE_MUMPS
      solver = Teuchos::rcp(new MumpsSolver());
#else
      if (rank==0)
        cout << "useMumps = true, but USE_MUMPS is unset.  Exiting...\n";
      exit(1);
#endif
    } else {
      solver = Teuchos::rcp(new KluSolver());
    }
    
//    // usage: polyOrder [numRefinements]
//    // parse args:
//    if ((argc != 4) && (argc != 3) && (argc != 2) && (argc != 5)) {
//      cout << "Usage: NavierStokesHemkerDriver fieldPolyOrder [numRefinements=10 [Reyn=5]]\n";
//      return -1;
//    }
//    int polyOrder = atoi(argv[1]);
//    int numRefs = 10;
//    if ( argc == 3) {
//      numRefs = atoi(argv[2]);
//    }
//    if ( argc == 4) {
//      numRefs = atoi(argv[2]);
//      Re = atof(argv[3]);
//    }
    if (rank == 0) {
      cout << "NEW as of 8-3-13: using L^2 tolerance relative to L^2 norm of background flow.\n";
      cout << "numRefinements = " << numRefs << endl;
      cout << "Re = " << Re << endl;
      if (artificialTimeStepping) cout << "dt = " << dt << endl;
      if (!startWithZeroSolutionAfterRefinement) {
        cout << "NOTE: experimentally, NOT starting with 0 solution after refinement...\n";
      }
      if (useCondensedSolve) {
        cout << "using condensed solve.\n";
      } else {
        cout << "not using condensed solve.\n";
      }
      if (useMumps) {
        cout << "using MUMPS for global linear solves.\n";
      } else {
        cout << "using KLU for global linear solves.\n";
      }
      if (velocityConditionsTopAndBottom) {
        cout << "imposing velocity BCs on top and bottom boundaries.\n";
      } else {
        cout << "imposing zero-traction BCs on top and bottom boundaries.\n";
      }
      if (velocityConditionsRight) {
        cout << "imposing velocity BCs on outflow boundary.\n";
      } else if (streamwiseGradientConditionsRight) {
        cout << "imposing streamwise gradient BCs on outflow boundary.\n";
      } else {
        cout << "imposing zero-traction BCs on outflow boundary.\n";
      }
    }

    // define meshes:
    int H1Order = polyOrder + 1;

    // get variable definitions:
    VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
    u1 = varFactory.fieldVar(VGP_U1_S);
    u2 = varFactory.fieldVar(VGP_U2_S);
    sigma11 = varFactory.fieldVar(VGP_SIGMA11_S);
    sigma12 = varFactory.fieldVar(VGP_SIGMA12_S);
    sigma21 = varFactory.fieldVar(VGP_SIGMA21_S);
    sigma22 = varFactory.fieldVar(VGP_SIGMA22_S);
    p = varFactory.fieldVar(VGP_P_S);
    
    u1hat = varFactory.traceVar(VGP_U1HAT_S);
    u2hat = varFactory.traceVar(VGP_U2HAT_S);
    t1n = varFactory.fluxVar(VGP_T1HAT_S);
    t2n = varFactory.fluxVar(VGP_T2HAT_S);
    
    v1 = varFactory.testVar(VGP_V1_S, HGRAD);
    v2 = varFactory.testVar(VGP_V2_S, HGRAD);
    tau1 = varFactory.testVar(VGP_TAU1_S, HDIV);
    tau2 = varFactory.testVar(VGP_TAU2_S, HDIV);
    q = varFactory.testVar(VGP_Q_S, HGRAD);
    
  //  double width = 60, height = 20;
    FunctionPtr zero = Function::zero();
    
//    double xLeft = -7.5;
//    double xRight = 42.5;
    double yTop = meshHeight / 2.0;
    double yBottom = - meshHeight / 2.0;
    
    if (! parabolicInflow) {
      inflowSpeed = Function::constant(1.0);
    } else {
      // following Schäfer and Turek -- though we multiply by 10 to get a unit diameter
      xLeft = -2.0;
      xRight = 20.5;
      yTop = 2.1;
      yBottom = -2.0;
      meshHeight = yTop - yBottom;
      
      FunctionPtr y = Function::yn();
      double nu_ref = .02;
      double D_ref = .1;
      double Uref = Re * nu_ref / D_ref;
      double Um = 0.3 / Uref;
      inflowSpeed = (4 * Um / (meshHeight*meshHeight)) * (y - yBottom) * (yTop - y);
      
      if (rank==0) cout << "WARNING: parabolicInflow known not to be fully consistent with Schäfer and Turek's results!\n";
    }
    
    MeshGeometryPtr geometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, yBottom, yTop, radius); //MeshFactory::hemkerGeometry(width,height,radius);
    


    {
      // print out some geometry info (checking something)
//      vector< FieldContainer<double> > vertices = geometry->vertices();
//      if (rank ==0) {
//        for (int i=0; i<vertices.size(); i++) {
//          cout << "vertex " << i << ": (" << vertices[i][0] << "," << vertices[i][1] << ")\n";
//        }
//      }
    }
    
    if (useStraightEdgedGeometry) {
      // right now, this is going to end up meaning that we have an octagon instead of a circle
      MeshGeometryPtr straightEdgeGeometry = Teuchos::rcp( new MeshGeometry(geometry->vertices(), geometry->elementVertices()));
      // (and that will remain true even as we refine)
      geometry = straightEdgeGeometry;
    }
    
    VGPNavierStokesProblem problem = VGPNavierStokesProblem(Function::constant(Re),geometry,
                                                            H1Order, pToAdd,
                                                            Function::zero(), Function::zero(), // zero forcing function
                                                            useScaleCompliantGraphNorm); // enrich velocity if using compliant graph norm
    problem.setSolver(solver);
    SolutionPtr solution = problem.backgroundFlow();
    SolutionPtr solnIncrement = problem.solutionIncrement();
    solution->setReportTimingResults(false);
    solnIncrement->setReportTimingResults(false);
    
    recreateBCs();
    
    // set pc and bc -- pc in particular may be null
    problem.backgroundFlow()->setFilter(pc);
    problem.solutionIncrement()->setFilter(pc);

    problem.backgroundFlow()->setBC(bc);
    problem.solutionIncrement()->setBC(bc);
    
    Teuchos::RCP<Mesh> mesh = problem.mesh();
    mesh->registerSolution(solution);
    mesh->registerSolution(solnIncrement);
    
//    {
//      if (rank==0) {
//        ElementPtr cell0 = mesh->getElement(0);
//        ElementPtr cell7 = mesh->getElement(7);
//        printNeighbors(cell0);
//        printNeighbors(cell7);
//      }
//      vector<int> cellIDs;
//      cellIDs.push_back(0);
//      cellIDs.push_back(7);
//      mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
//      
//      if (rank==0) {
//        ElementPtr cell0 = mesh->getElement(0);
//        int cornerCellID = cornerDescendantID(cell0, 0, 3); // side 0: cell7, side 3: cylinder
//        cout << "cell0 cornerCellID = " << cornerCellID << endl;
//        printNeighbors(mesh->getElement(cornerCellID));
//        
//        ElementPtr cell7 = mesh->getElement(7);
//        cornerCellID = cornerDescendantID(cell7, 2, 3); // side 2: cell0, side 3: cylinder
//        cout << "cell7 cornerCellID = " << cornerCellID << endl;
//        printNeighbors(mesh->getElement(cornerCellID));
//      }
//    }
    
    Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
    mesh->registerObserver(refHistory);
    
    ParameterFunctionPtr dt_inv = ParameterFunction::parameterFunction(1.0 / dt); //Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
    if (artificialTimeStepping) {
      //    // LHS gets u_inc / dt:
      BFPtr bf = problem.bf();
      FunctionPtr dt_inv_fxn = Teuchos::rcp(dynamic_cast< Function* >(dt_inv.get()), false);
      bf->addTerm(-dt_inv_fxn * u1, v1);
      bf->addTerm(-dt_inv_fxn * u2, v2);
      problem.setIP( bf->graphNorm() ); // graph norm has changed...
    }
    
    if (useScaleCompliantGraphNorm) {
      problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm());
    }
    
    // define bilinear form for stream function:
    VarFactory streamVarFactory;
    phi_hat = streamVarFactory.traceVar("\\widehat{\\phi}");
    VarPtr psin_hat = streamVarFactory.fluxVar("\\widehat{\\psi}_n");
    VarPtr psi_1 = streamVarFactory.fieldVar("\\psi_1");
    VarPtr psi_2 = streamVarFactory.fieldVar("\\psi_2");
    VarPtr phi = streamVarFactory.fieldVar("\\phi");
    VarPtr q_s = streamVarFactory.testVar("q_s", HGRAD);
    VarPtr v_s = streamVarFactory.testVar("v_s", HDIV);
    BFPtr streamBF = Teuchos::rcp( new BF(streamVarFactory) );
    streamBF->addTerm(psi_1, q_s->dx());
    streamBF->addTerm(psi_2, q_s->dy());
    streamBF->addTerm(-psin_hat, q_s);
    
    streamBF->addTerm(psi_1, v_s->x());
    streamBF->addTerm(psi_2, v_s->y());
    streamBF->addTerm(phi, v_s->div());
    streamBF->addTerm(-phi_hat, v_s->dot_normal());
    
    Teuchos::RCP<Mesh> streamMesh;
    
    bool useConformingTraces = true;
    map<int, int> trialOrderEnhancements;
    if (enrichVelocity) {
      trialOrderEnhancements[u1->ID()] = 1;
      trialOrderEnhancements[u2->ID()] = 1;
    }
    streamMesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                   streamBF, H1Order, pToAdd,
                                   useConformingTraces, trialOrderEnhancements) );
    streamMesh->setEdgeToCurveMap(geometry->edgeToCurveMap());
    
    mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
    
    if (rank==0) {
      GnuPlotUtil::writeComputationalMeshSkeleton("preliminaryHemkerMesh", mesh, true);
    }
    
    // now, let's get the elements to be roughly isotropic
    // to do spatial lookups, we need a mesh without curves (this is a bit ugly)
    // we don't care which BF we use, so we use streamBF because it's cheaper
    MeshPtr proxyMesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                               streamBF, H1Order, pToAdd,
                                               useConformingTraces, trialOrderEnhancements) );
    // now, register the real mesh with the proxy
    proxyMesh->registerObserver(mesh);
    // and make the proxy roughly isotropic:
    if (makeMeshRoughlyIsotropic) {
      makeRoughlyIsotropic(proxyMesh, radius, enforceOneIrregularity);
    }

    if (rank==0) {
      GnuPlotUtil::writeComputationalMeshSkeleton("initialHemkerMesh", mesh, true);
    }
    
    if (rank == 0) {
      cout << "Starting mesh has " << problem.mesh()->numActiveElements() << " elements and ";
      cout << mesh->numGlobalDofs() << " total dofs.\n";
      cout << "polyOrder = " << polyOrder << endl; 
      cout << "pToAdd = " << pToAdd << endl;
      
      if (enforceOneIrregularity) {
        cout << "Enforcing 1-irregularity.\n";
      } else {
        cout << "NOT enforcing 1-irregularity.\n";
      }
    }
    
    if (replayFile.length() > 0) {
      RefinementHistory refHistory;
      refHistory.loadFromFile(replayFile);
      refHistory.playback(mesh);
    }
    if (solnFile.length() > 0) {
      solution->readFromFile(solnFile);
    }
    
    map< int, FunctionPtr > initialGuess;
    initialGuess[u1->ID()] = Function::constant(1.0);
    initialGuess[u1hat->ID()] = Function::constant(1.0);
    // all other variables: use zero initial guess (the implicit one)
    
    ////////////////////   CREATE BCs   ///////////////////////
    FunctionPtr u1_prev = Function::solution(u1,solution);
    FunctionPtr u2_prev = Function::solution(u2,solution);
    
    FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
    FunctionPtr u2hat_prev = Function::solution(u2hat,solution);

    ////////////////////   SOLVE & REFINE   ///////////////////////
    
  //  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
    if (rank==0) cout << "using sigma-based vorticity definition.\n";
    FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - Re * sigma12 + Re * sigma21 ) ); // Re because sigma = 1/Re grad u
    FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, p) );
    
    double delta = pressureDifference(p_prev, radius, mesh);
    if (rank==0) cout << "computed pressure delta on initial solution as " << delta << endl;
    
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    
    double energyThreshold = 0.20; // for mesh refinements
    Teuchos::RCP<RefinementStrategy> refinementStrategy;
  //  if (rank==0) cout << "NOTE: using solution, not solnIncrement, for refinement strategy.\n";
  //  refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold ));
    double min_h = 0;
    int maxP = 11;
    refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold, min_h, maxP, refineInPFirst ));
    
    refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
    refinementStrategy->setReportPerCellErrors(reportPerCellErrors);
    
    if (true) { // do regular refinement strategy...
      bool printToConsole = rank==0;
      FunctionPtr u1_incr = Function::solution(u1, solnIncrement);
      FunctionPtr u2_incr = Function::solution(u2, solnIncrement);
      FunctionPtr sigma11_incr = Function::solution(sigma11, solnIncrement);
      FunctionPtr sigma12_incr = Function::solution(sigma12, solnIncrement);
      FunctionPtr sigma21_incr = Function::solution(sigma21, solnIncrement);
      FunctionPtr sigma22_incr = Function::solution(sigma22, solnIncrement);
      FunctionPtr p_incr = Function::solution(p, solnIncrement);
      
      FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
      + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
      + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;

      FunctionPtr u1_prev = Function::solution(u1, solution);
      FunctionPtr u2_prev = Function::solution(u2, solution);
      FunctionPtr sigma11_prev = Function::solution(sigma11, solution);
      FunctionPtr sigma12_prev = Function::solution(sigma12, solution);
      FunctionPtr sigma21_prev = Function::solution(sigma21, solution);
      FunctionPtr sigma22_prev = Function::solution(sigma22, solution);
      FunctionPtr p_prev = Function::solution(p, solution);
      
      FunctionPtr l2_prev = u1_prev * u1_prev + u2_prev * u2_prev + p_prev * p_prev
      + sigma11_prev * sigma11_prev + sigma12_prev * sigma12_prev
      + sigma21_prev * sigma21_prev + sigma22_prev * sigma22_prev;
      
      for (int refIndex=0; refIndex<numRefs; refIndex++){
        if (startWithZeroSolutionAfterRefinement) {
          // start with a fresh initial guess for each adaptive mesh:
          solution->clear();
          if (useZeroInitialGuess) {
            cout << "using zero initial guess for now...\n";
    //        solution->projectOntoMesh(initialGuess);
            problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement) -- necessary to impose BCs
          } else {
            solution->projectOntoMesh(initialGuess); // for this to work initialGuess must match all the non-zero BCs
          }
        }
        
        double incr_norm, prev_norm;
        do {
          problem.iterate(useLineSearch,useCondensedSolve);
          incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
          prev_norm = sqrt(l2_prev->integrate(problem.mesh()));
          if (prev_norm > 0) {
            incr_norm /= prev_norm;
          }
          if (rank==0) {
            cout << "\x1B[2K"; // Erase the entire current line.
            cout << "\x1B[0E"; // Move to the beginning of the current line.
            cout << "Refinement # " << refIndex << ", iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
            flush(cout);
          }
        } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));

        if (rank==0) {
          cout << "\nFor refinement " << refIndex << ", num iterations: " << problem.iterationCount() << endl;
        }
        
        // compute pressure difference between front and back of cylinder
        double delta_pressure = pressureDifference(p_prev, radius, mesh);
        if (rank==0) {
          cout << "pressure difference (front to back of cylinder): " << delta_pressure << endl;
        }
        
        // compute lift coefficient:
        double c_L = liftCoefficient(solution, radius);
//        double c_L_neglectingPressure = liftCoefficient(solution, radius, true);
        if (rank==0) {
          cout << "lift coefficient: " << c_L << endl;
//          cout << "lift coefficient neglecting pressure contribution: " << c_L_neglectingPressure << endl;
        }
        
        // compute drag coefficient:
        double c_D = dragCoefficient(solution, radius);
//        double c_D_neglectingPressure = dragCoefficient(solution, radius, true);
        if (rank==0) {
          cout << "drag coefficient: " << c_D << endl;
//          cout << "drag coefficient neglecting pressure contribution: " << c_D_neglectingPressure << endl;
        }
        
        // reset iteration count to 1 (for the background flow):
        problem.setIterationCount(1);
        
        if (rank==0) {
          if (solnSaveFile.length() > 0) {
            solution->writeToFile(solnSaveFile);
          }
        }
        
        refinementStrategy->refine(false); // don't print to console // (rank==0); // print to console on rank 0
        if (rank==0) {
          cout << "After refinement, mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " global dofs" << endl;
        }
        
//        if (rank==0) { // DEBUGGING code
//          ElementPtr cell0 = mesh->getElement(0);
//          int cornerCellID = cornerDescendantID(cell0, 0, 3); // side 0: cell7, side 3: cylinder
//          cout << "cell0 cornerCellID = " << cornerCellID << endl;
//          printNeighbors(mesh->getElement(cornerCellID));
//
//          ElementPtr cell7 = mesh->getElement(7);
//          cornerCellID = cornerDescendantID(cell7, 2, 3); // side 2: cell0, side 3: cylinder
//          cout << "cell7 cornerCellID = " << cornerCellID << endl;
//          printNeighbors(mesh->getElement(cornerCellID));
//        }
//        
        if (saveFile.length() > 0) {
          if (rank == 0) {
            refHistory->saveToFile(saveFile);
          }
        }
        
      }
      // skip final solve if we haven't changed the solution that was loaded from disk:
      if ((solnFile.length() == 0) || (numRefs > 0)) {
        // one more solve on the final refined mesh:
        if (rank==0) cout << "Final solve:\n";
        if (startWithZeroSolutionAfterRefinement) {
          // start with a fresh (zero) initial guess for each adaptive mesh:
          solution->clear();
          problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
        }
        double incr_norm, prev_norm;
        do {
          problem.iterate(useLineSearch,useCondensedSolve);
          incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
          prev_norm = sqrt(l2_prev->integrate(problem.mesh()));
          if (prev_norm > 0) {
            incr_norm /= prev_norm;
          }
          if (rank==0) {
            cout << "\x1B[2K"; // Erase the entire current line.
            cout << "\x1B[0E"; // Move to the beginning of the current line.
            cout << "Iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
            flush(cout);
          }
        } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
        if (rank==0) cout << endl;
      }
    }

    if (rank==0) {
      if (solnSaveFile.length() > 0) {
        solution->writeToFile(solnSaveFile);
      }
    }

    // compute pressure difference between front and back of cylinder
    double delta_pressure = pressureDifference(p_prev, radius, mesh);
    if (rank==0) {
      cout << "pressure difference (front to back of cylinder): " << delta_pressure << endl;
    }
    
    // compute lift coefficient:
    double c_L = liftCoefficient(solution, radius);
//    double c_L_neglectingPressure = liftCoefficient(solution, radius, true);
    if (rank==0) {
      cout << "lift coefficient: " << c_L << endl;
//      cout << "lift coefficient neglecting pressure contribution: " << c_L_neglectingPressure << endl;
    }
    
    // compute drag coefficient:
    double c_D = dragCoefficient(solution, radius);
    double c_D_neglectingPressure = dragCoefficient(solution, radius, true);
    if (rank==0) {
      cout << "drag coefficient: " << c_D << endl;
//      cout << "drag coefficient neglecting pressure contribution: " << c_D_neglectingPressure << endl;
    }
    
    double energyErrorTotal = solution->energyErrorTotal();
    double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
    if (rank == 0) {
      cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
      cout << "Final incremental energy error: " << incrementalEnergyErrorTotal << ".)\n";
    }
    
    if (!skipPostProcessing) {
      FunctionPtr u1_sq = u1_prev * u1_prev;
      FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
      FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
      FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat->times_normal_x() + u2hat->times_normal_y()) );
      
      // check that the zero mean pressure is being correctly imposed:
      double p_avg = p_prev->integrate(mesh);
      if (rank==0)
        cout << "Integral of pressure: " << p_avg << endl;
      
      // integrate massFlux over each element (a test):
      // fake a new bilinear form so we can integrate against 1 
      VarPtr testOne = varFactory.testVar("1",CONSTANT_SCALAR);
      BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
      LinearTermPtr massFluxTerm = massFlux * testOne;
      
      CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
      DofOrderingFactory dofOrderingFactory(fakeBF);
      int fakeTestOrder = H1Order;
      DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);
      
      int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
      vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
      map<int, double> massFluxIntegral; // cellID -> integral
      double maxMassFluxIntegral = 0.0;
      double totalMassFlux = 0.0;
      double totalAbsMassFlux = 0.0;
      double maxCellMeasure = 0;
      double minCellMeasure = 1;
      for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
        ElementTypePtr elemType = *elemTypeIt;
        vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
        vector<int> cellIDs;
        for (int i=0; i<elems.size(); i++) {
          cellIDs.push_back(elems[i]->cellID());
          if (elems[i]->cellID()==0) {
            cout << "cellID 0\n"; // this line for setting a breakpoint.
          }
        }
        FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
        BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh,true,15) ); // enrich by a bunch
        basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
        FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
        FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
        massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          // pick out the ones for testOne:
          massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
        }
        // find the largest:
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
        }
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
          minCellMeasure = min(minCellMeasure,cellMeasures(i));
          maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
          totalMassFlux += massFluxIntegral[cellID];
          totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    //      if (rank==0) {
    //        cout << "driver: massFluxIntegral[" << cellID << "] = " << massFluxIntegral[cellID] << endl;
    //      }
        }
      }
      if (rank==0) {
        cout << "largest mass flux: " << maxMassFluxIntegral << endl;
        cout << "total mass flux: " << totalMassFlux << endl;
        cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
        cout << "largest h: " << sqrt(maxCellMeasure) << endl;
        cout << "smallest h: " << sqrt(minCellMeasure) << endl;
        cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
      }
      
      FunctionPtr newMassFlux = Teuchos::rcp( new MassFluxFunction(massFlux) );
      FunctionPtr absMassFlux = Teuchos::rcp( new MassFluxFunction(massFlux,true) );
      
      totalAbsMassFlux = absMassFlux->integrate(mesh,11,false,true); // 11: enrich cubature a bunch
      totalMassFlux = massFlux->integrate(mesh,11,false,true); // 11: enrich cubature a bunch

      if (rank==0) {
        cout << "new total mass flux: " << totalMassFlux << endl;
        cout << "new sum of mass flux absolute value: " << totalAbsMassFlux << endl;
      }
    }
    
    if (rank==0){
      GnuPlotUtil::writeComputationalMeshSkeleton("finalHemkerMesh", mesh);
      
#ifdef USE_VTK
        VTKExporter exporter(solution, mesh, varFactory);
        exporter.exportSolution("nsHemkerSoln", H1Order*2);
        
        exporter.exportFunction(vorticity, "HemkerVorticity");
#endif
      
  //    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
  //    u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
  //    solution->writeFieldsToFile(u1->ID(), "u1.m");
  //    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
  //    solution->writeFieldsToFile(u2->ID(), "u2.m");
  //    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
  //    solution->writeFieldsToFile(p->ID(), "p.m");
      
  //    polyOrderFunction->writeValuesToMATLABFile(mesh, "hemkerPolyOrders.m");
    }
    
    if ( !skipPostProcessing ) {
      Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
      streamRHS->addTerm(vorticity * q_s);
      ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
      ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
      ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
      
      recreateStreamBCs();
      
      IPPtr streamIP = Teuchos::rcp( new IP );
      streamIP->addTerm(q_s);
      streamIP->addTerm(q_s->grad());
      streamIP->addTerm(v_s);
      streamIP->addTerm(v_s->div());
      SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
      
      if (rank==0) {
        cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
        cout << "solving for approximate stream function...\n";
      }

      if (useCondensedSolve) {
        streamSolution->condensedSolve(solver);
      } else {
        streamSolution->solve(solver);
      }
      energyErrorTotal = streamSolution->energyErrorTotal();
      if (rank == 0) {
        cout << "...solved.\n";
        cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
      }
      
      if (rank==0){
#ifdef USE_VTK
        VTKExporter streamExporter(streamSolution, streamMesh, streamVarFactory);
        streamExporter.exportSolution("hemkerStreamSoln", H1Order*2);
#endif

        // the commented-out code below doesn't really work because gnuplot requires a "point grid" in physical space...
    //    FieldContainer<double> refPoints = pointGrid(-1, 1, -1, 1, H1Order);
    //    FieldContainer<double> pointData = solutionDataFromRefPoints(refPoints, streamSolution, phi);
    //    string patchDataPath = "phi_navierStokes_hemker.dat";
    //    GnuPlotUtil::writeXYPoints(patchDataPath, pointData);
    //    set<double> patchContourLevels = logContourLevels(height);
    //    vector<string> patchDataPaths;
    //    patchDataPaths.push_back(patchDataPath);
    //    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPaths, "hemkerNavierStokes.p");
      }
    }
    
    if (saveFile.length() > 0) {
      if (rank == 0) {
        refHistory->saveToFile(saveFile);
        cout << "Saved refinement history to " << saveFile << endl;
      }
    }
    if (solnSaveFile.length() > 0) {
      if (rank==0) {
        solution->writeToFile(solnSaveFile);
        cout << "Saved solution to " << solnSaveFile << endl;
      }
    }

    
  } catch ( std::exception& e )
  {
    if (rank==0) {
      cout << e.what() << endl;
      throw e;
    }
  }
  
  return 0;
}
