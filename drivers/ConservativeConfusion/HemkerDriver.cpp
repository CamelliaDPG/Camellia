//  BendDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

bool enforceLocalConservation = true;
double epsilon = 1e-4;
int numRefs = 0;
int nseg = 32;
bool ReadMesh = false;
bool CircleMesh = false;
bool TriangulateMesh = false;

double pi = 2.0*acos(0.0);

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class LeftBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(x+3) < tol);
  }
};

class RightBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool rightMatch = (abs(x-9) < tol);
    return rightMatch;
  }
};

class TopBottomBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool topMatch = (abs(y-3) < tol);
    bool bottomMatch;
    // if (ReadMesh)
    //   bottomMatch = (abs(y) < tol);
    // else
      bottomMatch = (abs(y+3) < tol);
    return topMatch || bottomMatch;
  }
};

class CircleBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-3;
    return (abs(x*x+y*y) < 1+tol);
  }
};

class ZeroBC : public Function {
  public:
    ZeroBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

// boundary value for sigma_n
class OneBC : public Function {
  public:
    OneBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = 1;
        }
      }
    }
};

class Beta : public Function {
public:
  Beta() : Function(1) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d = 0; d < spaceDim; d++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex,0) = 1;
          values(cellIndex,ptIndex,1) = 0;
        }
      }
    }
  }
};

class IPWeight : public Function {
  public:
    IPWeight() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      double a = 2;

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          // if (x > 0 && abs(y) < 1+1e-3 && x < a)
          //   values(cellIndex, ptIndex) = epsilon + (x-sqrt(1-y*y))/(a-sqrt(1-y*y));
          if (x > 0 && sqrt(x*x+y*y) < a)
          {
            double dr = sqrt(x*x+y*y) - 1;
            values(cellIndex, ptIndex) = epsilon + dr/(a-1);
          }
          else
            values(cellIndex, ptIndex) = 1;
        }
      }
    }
};

int main(int argc, char *argv[]) {
  // Process command line arguments
  if (argc > 1)
    numRefs = atof(argv[1]);
  if (argc > 2)
    nseg = atof(argv[2]);
  if (argc > 3)
    epsilon = atof(argv[3]);
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
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
  
  FunctionPtr beta = Teuchos::rcp(new Beta());
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / epsilon, tau->x());
  confusionBF->addTerm(sigma2 / epsilon, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // mathematician's norm
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(tau);
  mathIP->addTerm(tau->div());

  mathIP->addTerm(v);
  mathIP->addTerm(v->grad());

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  qoptIP->addTerm( v );
  qoptIP->addTerm( tau / epsilon + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  if (!enforceLocalConservation)
    robIP->addTerm( ip_scaling * v );
  robIP->addTerm( sqrt(epsilon) * v->grad() );
  // Weight these two terms for inflow
  FunctionPtr ip_weight = Teuchos::rcp( new IPWeight() );
  robIP->addTerm( ip_weight * beta * v->grad() );
  robIP->addTerm( ip_weight * tau->div() );
  robIP->addTerm( ip_scaling/sqrt(epsilon) * tau );
  if (enforceLocalConservation)
    robIP->addZeroMeanTerm( v );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  // Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp( new PenaltyConstraints );
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  SpatialFilterPtr tbBoundary = Teuchos::rcp( new TopBottomBoundary );
  // SpatialFilterPtr rBoundary = Teuchos::rcp( new RightBoundary );
  SpatialFilterPtr circleBoundary = Teuchos::rcp( new CircleBoundary );
  FunctionPtr u0 = Teuchos::rcp( new ZeroBC );
  FunctionPtr u1 = Teuchos::rcp( new OneBC );
  // FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  bc->addDirichlet(beta_n_u_minus_sigma_n, lBoundary, u0);
  bc->addDirichlet(beta_n_u_minus_sigma_n, tbBoundary, u0);
  bc->addDirichlet(uhat, circleBoundary, u1);
  // pc->addConstraint(beta_n_u_minus_sigma_n - uhat == u0, rBoundary);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 3, pToAdd = 2;
  Teuchos::RCP<Mesh> mesh;
  if (ReadMesh)
    mesh = Mesh::readTriangle(Camellia_MeshDir+"Hemker/Hemker.1", confusionBF, H1Order, pToAdd);
  else
#if 0
  {
    vector< FieldContainer<double> > vertices;
    FieldContainer<double> pt(2);
    vector< vector<int> > elementIndices;
    vector<int> el(3);

    pt(0) = -3; pt(1) = -3;
    vertices.push_back(pt);
    pt(0) = 9; pt(1) = -3;
    vertices.push_back(pt);
    pt(0) = 9; pt(1) = 3;
    vertices.push_back(pt);
    pt(0) = -3; pt(1) = 3;
    vertices.push_back(pt);
    pt(0) = 1; pt(1) = 0;
    vertices.push_back(pt);
    pt(0) = 0; pt(1) = 1;
    vertices.push_back(pt);
    pt(0) = -1; pt(1) = 0;
    vertices.push_back(pt);
    pt(0) = 0; pt(1) = -1;
    vertices.push_back(pt);
    pt(0) = 3; pt(1) = 3;
    vertices.push_back(pt);
    pt(0) = 3; pt(1) = -3;
    vertices.push_back(pt);

    el[0] = 0; el[1] = 7; el[2] = 6;
    elementIndices.push_back(el);
    el[0] = 1; el[1] = 8; el[2] = 9;
    elementIndices.push_back(el);
    el[0] = 5; el[1] = 3; el[2] = 6;
    elementIndices.push_back(el);
    el[0] = 6; el[1] = 3; el[2] = 0;
    elementIndices.push_back(el);
    el[0] = 7; el[1] = 0; el[2] = 9;
    elementIndices.push_back(el);
    el[0] = 1; el[1] = 2; el[2] = 8;
    elementIndices.push_back(el);
    el[0] = 4; el[1] = 7; el[2] = 9;
    elementIndices.push_back(el);
    el[0] = 5; el[1] = 8; el[2] = 3;
    elementIndices.push_back(el);
    el[0] = 9; el[1] = 8; el[2] = 4;
    elementIndices.push_back(el);
    el[0] = 4; el[1] = 8; el[2] = 5;
    elementIndices.push_back(el);

    mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, confusionBF, H1Order, pToAdd) );  
  }
#else
  {
    // Generate Mesh
    vector< FieldContainer<double> > vertices;
    FieldContainer<double> pt(2);
    vector< vector<int> > elementIndices;
    vector<int> el(4);
    vector<int> el1(3);
    vector<int> el2(3);
    // Inner Square
    double S;
    if (CircleMesh)
      S = 1.5;
    else
      S = 1 + 1./nseg;
    double angle;
    // Bottom edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = -3.*pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = -S + double(i)/nseg*2*S;
        pt(1) = -S;
      }
      vertices.push_back(pt);
      el[0] = i; 
      el[1] = i + 1;
      el[2] = 4*nseg + i + 1;
      el[3] = 4*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Right edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = -pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = S;
        pt(1) = -S + double(i)/nseg*2*S;
      }
      vertices.push_back(pt);
      el[0] = nseg + i; 
      el[1] = nseg + i + 1;
      el[2] = 5*nseg + i + 1;
      el[3] = 5*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Top edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = S - double(i)/nseg*2*S;
        pt(1) = S;
      }
      vertices.push_back(pt);
      el[0] = 2*nseg + i; 
      el[1] = 2*nseg + i + 1;
      el[2] = 6*nseg + i + 1;
      el[3] = 6*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Left edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = 3.*pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = -S;
        pt(1) = S - double(i)/nseg*2*S;
      }
      vertices.push_back(pt);
      el[0] = 3*nseg + i; 
      el[1] = 3*nseg + i + 1;
      el[2] = 7*nseg + i + 1;
      el[3] = 7*nseg + i;
      if (i == nseg-1)
      {
        el[1] = 0;
        el[2] = 4*nseg;
      }
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // if (TriangulateMesh)
    // {
    //   elementIndices[elementIndices.size()-2][1] = 0;
    //   elementIndices[elementIndices.size()-2][2] = 4*nseg-1;
    //   cout << "Left Element 1: " 
    //     << elementIndices[elementIndices.size()-2][0]
    //     << elementIndices[elementIndices.size()-2][1]
    //     << elementIndices[elementIndices.size()-2][2] << endl;
    //   cout << "Left Element 2: " 
    //     << elementIndices[elementIndices.size()-1][0]
    //     << elementIndices[elementIndices.size()-1][1]
    //     << elementIndices[elementIndices.size()-1][2] << endl;
    // }
    // else
    // {
    //   elementIndices.back()[1] = 0;
    //   elementIndices.back()[2] = 4*nseg;
    // }
    // elementIndices[4*nseg-1][1] = 0;
    // elementIndices[4*nseg-1][2] = 4*nseg;
    // Circle
    for (int i=0; i < 4*nseg; i++)
    {
      angle = 5./4.*pi + 2.*pi*i/(4*nseg);
      pt(0) = cos(angle);
      pt(1) = sin(angle);
      vertices.push_back(pt);
    }
    // Outer Rectangle
    int N = vertices.size();
    // Below square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = -S + double(i)/nseg*2*S;
      pt(1) = -3.0;
      vertices.push_back(pt);
      el[0] = N + i;
      el[1] = N + i + 1;
      el[2] = i + 1;
      el[3] = i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = S;
    pt(1) = -3.0;
    vertices.push_back(pt);
    // Right of square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = 9.0;
      pt(1) = -S + double(i)/nseg*2*S;
      vertices.push_back(pt);
      el[0] = N + nseg+1 + i;
      el[1] = N + nseg+1 + i + 1;
      el[2] = nseg + i + 1;
      el[3] = nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = 9.0;
    pt(1) = S;
    vertices.push_back(pt);
    // Above square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = S - double(i)/nseg*2*S;
      pt(1) = 3.0;
      vertices.push_back(pt);
      el[0] = N + 2*(nseg+1) + i;
      el[1] = N + 2*(nseg+1) + i + 1;
      el[2] = 2*nseg + i + 1;
      el[3] = 2*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = -S;
    pt(1) = 3.0;
    vertices.push_back(pt);
    // Left of square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = -3.0;
      pt(1) = S - double(i)/nseg*2*S;
      vertices.push_back(pt);
      el[0] = N + 3*(nseg+1) + i;
      el[1] = N + 3*(nseg+1) + i + 1;
      el[2] = 3*nseg + i + 1;
      el[3] = 3*nseg + i;
      if (i == nseg-1)
      {
        el[2] = 0;
      }
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // if (TriangulateMesh)
    // {
    //   elementIndices[elementIndices.size()-2][2] = 0;
    //   elementIndices[elementIndices.size()-1][1] = 0;
    // }
    // else
    //   elementIndices.back()[2] = 0;
    pt(0) = -3.0;
    pt(1) = -S;
    vertices.push_back(pt);
    // Bottom left corner
    pt(0) = -3.0;
    pt(1) = -3.0;
    vertices.push_back(pt);
    el[0] = N + 4*(nseg+1);
    el[1] = N;
    el[2] = 0;
    el[3] = N + 4*(nseg+1) - 1;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Bottom right corner
    pt(0) = 9.0;
    pt(1) = -3.0;
    vertices.push_back(pt);
    el[0] = N + nseg;
    el[1] = N + 4*(nseg+1) + 1;
    el[2] = N + nseg+1;
    el[3] = nseg;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Top right corner
    pt(0) = 9.0;
    pt(1) = 3.0;
    vertices.push_back(pt);
    el[0] = 2*nseg;
    el[1] = N + 2*(nseg+1)-1;
    el[2] = N + 4*(nseg+1) + 2;
    el[3] = N + 2*(nseg+1);
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Top Left corner
    pt(0) = -3.0;
    pt(1) = 3.0;
    vertices.push_back(pt);
    el[0] = N + 3*(nseg+1);
    el[1] = 3*nseg;
    el[2] = N + 3*(nseg+1)-1;
    el[3] = N + 4*(nseg+1) + 3;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);

    // cout << "Vertices:" << endl;
    // for (int i=0; i < vertices.size(); i++)
    // {
    //   cout << vertices[i](0) << " " << vertices[i](1) << endl;
    // }
    // cout << "Connectivity:" << endl;
    // for (int i=0; i < elementIndices.size(); i++)
    // {
    //   if (TriangulateMesh)
    //     cout << elementIndices[i][0]<<" "<< elementIndices[i][1]<<" "<< elementIndices[i][2]<< endl;
    //   else
    //     cout << elementIndices[i][0]<<" "<< elementIndices[i][1]<<" "<< elementIndices[i][2]<<" "<<elementIndices[i][3] << endl;
    // }
    mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, confusionBF, H1Order, pToAdd) );  
  }
#endif
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, mathIP) );
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  // solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.25; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    stringstream outfile;
    outfile << "hemker_" << refIndex;
    solution->writeFieldsToVTK(outfile.str(), 2);
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  // one more solve on the final refined mesh:
  solution->solve(false);

  // Check conservation by testing against one
  VarPtr testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  // Create a fake bilinear form for the testing
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  // Define our mass flux
  FunctionPtr massFlux= Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
  LinearTermPtr massFluxTerm = massFlux * testOne;

  Teuchos::RCP<shards::CellTopology> quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(fakeBF);
  int fakeTestOrder = H1Order;
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);
  
  int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
  vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
  map<int, double> massFluxIntegral; // cellID -> integral
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
    vector<int> cellIDs;
    for (int i=0; i<elems.size(); i++) {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh) );
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
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }
  
  
  // Print results from processor with rank 0
  if (rank==0){
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;

    stringstream outfile;
    outfile << "hemker_" << numRefs;
    solution->writeFieldsToVTK(outfile.str(), 2);
  }
  
  return 0;
}
