//
//  ScratchPadDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/26/12.
//  Copyright (c) 2012. All rights reserved.
//

#include <iostream>

#include "InnerProductScratchPad.h"
#include "MathInnerProduct.h"
#include "StokesBilinearForm.h"
#include "StokesMathBilinearForm.h"
#include "ElementType.h"
#include "TestSuite.h"
#include "BasisFactory.h"
#include "DofOrderingFactory.h"
#include "StokesManufacturedSolution.h"
#include "HConvergenceStudy.h"

typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<DPGInnerProduct> DPGInnerProductPtr;
typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;
typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP<ElementType> ElementTypePtr;
typedef Teuchos::RCP<BF> BFPtr;

class SquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x+1.0) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y+1.0) < tol) || (abs(y-1.0) < tol);
    return xMatch && yMatch;
  }
};

// BC for the "exponential" manufactured solution for Stokes
class StokesManufacturedSolutionBC_u1 : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex,0) = -exp(x) * ( y * cos(y) + sin(y) );
      }
    }
  }
};

class StokesManufacturedSolutionBC_u2 : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex,0) = exp(x) * y * sin(y);
      }
    }
  }
};



class beta : public Function {
public:
  beta() : Function(1) {
    
  }
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
          values(cellIndex,ptIndex,0) = y;
          values(cellIndex,ptIndex,0) = -x;
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
  FieldContainer<double> expectedValues;
  FieldContainer<double> actualValues;
  
  // define nodes for test
  FieldContainer<double> quadPoints(1,4,2);
  
  quadPoints(0,0,0) = 0.0; // x1
  quadPoints(0,0,1) = 0.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = 0.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = 0.0;
  quadPoints(0,3,1) = 1.0;
  
  // 1. Implement the math norm for Stokes
  VarFactory varFactory; 
  VarPtr q1 = varFactory.testVar("q_1", HDIV, StokesMathBilinearForm::Q_1);
  VarPtr q2 = varFactory.testVar("q_2", HDIV, StokesMathBilinearForm::Q_2);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD, StokesMathBilinearForm::V_1);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD, StokesMathBilinearForm::V_2);
  VarPtr v3 = varFactory.testVar("v_3", HGRAD, StokesMathBilinearForm::V_3);
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr sigma1n = varFactory.fluxVar("\\widehat{P - \\sigma_{1n}}");
  VarPtr sigma2n = varFactory.fluxVar("\\widehat{P - \\sigma_{2n}}");
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_11");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_12");
  VarPtr sigma21 = varFactory.fieldVar("\\sigma_21");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_22");
  VarPtr p = varFactory.fieldVar("p");
  
  double mu = 1.0;
  
  BFPtr stokesBFMath = Teuchos::rcp( new BF(varFactory) );  
  // q1 terms:
  stokesBFMath->addTerm(u1,q1->div());
  stokesBFMath->addTerm(mu * sigma11,q1->x());
  stokesBFMath->addTerm(mu * sigma12,q1->y());
  stokesBFMath->addTerm(-u1hat, q1->dot_normal());
  
  // q2 terms:
  stokesBFMath->addTerm(u2, q2->div());
  stokesBFMath->addTerm(mu * sigma21,q2->x());
  stokesBFMath->addTerm(mu * sigma22,q2->y());
  stokesBFMath->addTerm(-u2hat, q2->dot_normal());
  
  // v1:
  stokesBFMath->addTerm(sigma11,v1->dx());
  stokesBFMath->addTerm(sigma12,v1->dy());
  stokesBFMath->addTerm( - p, v1->dx() );
  stokesBFMath->addTerm( sigma1n, v1);

  // v2:
  stokesBFMath->addTerm(sigma21,v2->dx());
  stokesBFMath->addTerm(sigma22,v2->dy());
  stokesBFMath->addTerm( -p, v2->dy());
  stokesBFMath->addTerm( sigma2n, v2);
  
  // v3:
  stokesBFMath->addTerm(-u1,v3->dx());
  stokesBFMath->addTerm(-u2,v3->dy());
  stokesBFMath->addTerm(1.0 * u1hat->times_normal_x() + u2hat->times_normal_y(), v3);
  
  int trialOrder = 1;
  int pToAdd = 0;
  int testOrder = trialOrder + pToAdd;
  CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(stokesBFMath);
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(testOrder, *quadTopoPtr);
  DofOrderingPtr trialOrdering = dofOrderingFactory.trialOrdering(trialOrder, *quadTopoPtr);
  
  // just use testOrdering for both trial and test spaces (we only use to define BasisCache)
  ElementTypePtr elemType  = Teuchos::rcp( new ElementType(trialOrdering, testOrdering, quadTopoPtr) );
  BasisCachePtr ipBasisCache = Teuchos::rcp( new BasisCache(elemType, true) ); // true: test vs. test
  ipBasisCache->setPhysicalCellNodes(quadPoints,vector<int>(1),false); // false: don't create side cache
  
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(v1);
  mathIP->addTerm(v1->grad());
  mathIP->addTerm(v2);
  mathIP->addTerm(v2->grad());
  mathIP->addTerm(v3);
  mathIP->addTerm(v3->grad());
  mathIP->addTerm(q1);
  mathIP->addTerm(q1->div());
  mathIP->addTerm(q2);
  mathIP->addTerm(q2->div());
  
  Teuchos::RCP<BilinearForm> stokesBF = Teuchos::rcp(new StokesMathBilinearForm(mu));
  DPGInnerProductPtr autoMathIP = Teuchos::rcp( new MathInnerProduct(stokesBF) );
  
  int numCells = quadPoints.dimension(0);
  expectedValues.resize(numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  actualValues.resize  (numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  
  autoMathIP->computeInnerProductMatrix(expectedValues,testOrdering,ipBasisCache);
  mathIP->computeInnerProductMatrix(actualValues,testOrdering,ipBasisCache);
  
  double tol = 1e-14;
  double maxDiff = 0.0;
  if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Test failed: automatic mathematician's inner product differs from new IP; maxDiff " << maxDiff << ".\n";
    cout << "Automatic: \n" << expectedValues;
    cout << "New IP: \n" << actualValues;
  } else {
    cout << "Automatic mathematician's inner product and new IP agree!!\n";
  }
  
  cout << "*** Math IP: ***\n";
  mathIP->printInteractions();
  
  IPPtr qoptIP = Teuchos::rcp(new IP());
                                               
  double beta = 1e-1;
  qoptIP->addTerm( q1->x() / mu + v1->dx() );
  qoptIP->addTerm( q1->x() / (2.0 * mu) + q2->y() / (2.0 * mu) );
  qoptIP->addTerm( q1->y() / (2.0 * mu) + q2->x() / (2.0 * mu) + v1->dy() + v2->dx() );
  qoptIP->addTerm( q2->y() / (2.0 * mu) + v2->dy() );
  qoptIP->addTerm( q1->y() - q2->x() );
  qoptIP->addTerm( q1->div() - v3->dx() );
  
  qoptIP->addTerm( sqrt(beta) * q1 );
  qoptIP->addTerm( sqrt(beta) * q2 );
  qoptIP->addTerm( sqrt(beta) * v1 );
  qoptIP->addTerm( sqrt(beta) * v2 );
  qoptIP->addTerm( sqrt(beta) * v3 );
  
  cout << "*** Quasi-Optimal IP: ***\n";
  qoptIP->printInteractions();
  
  // just run the quasi-optimal (we don't have a good way of testing it right now)
  qoptIP->computeInnerProductMatrix(actualValues,testOrdering,ipBasisCache);

  // test stiffness matrix computation:
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
  basisCache->setPhysicalCellNodes(quadPoints,vector<int>(1),true); // true: do create side cache
  FieldContainer<double> cellSideParities(numCells,quadTopoPtr->getSideCount());
  cellSideParities.initialize(1.0); // not worried here about neighbors actually having opposite parity -- just want the two BF implementations to agree...
  expectedValues.resize(numCells, testOrdering->totalDofs(), trialOrdering->totalDofs() );
  actualValues.resize(numCells, testOrdering->totalDofs(), trialOrdering->totalDofs() );
  stokesBF->stiffnessMatrix(expectedValues, elemType, cellSideParities, basisCache);
  stokesBFMath->stiffnessMatrix(actualValues, elemType, cellSideParities, basisCache);
  
  if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Test failed: old Stokes stiffness differs from new; maxDiff " << maxDiff << ".\n";
    cout << "Old: \n" << expectedValues;
    cout << "New: \n" << actualValues;
    cout << "TrialDofOrdering: \n" << *trialOrdering;
    cout << "TestDofOrdering:\n" << *testOrdering;
  } else {
    cout << "Old and new Stokes stiffness agree!!\n";
  }
  
  // create BCs:
  Teuchos::RCP<BCEasy> stokesBC = Teuchos::rcp(new BCEasy());
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new SquareBoundary() );
  FunctionPtr u1fn = Teuchos::rcp( new StokesManufacturedSolutionBC_u1() );
  FunctionPtr u2fn = Teuchos::rcp( new StokesManufacturedSolutionBC_u2() );
  stokesBC->addDirichlet(u1hat,entireBoundary,u1fn);
  stokesBC->addDirichlet(u2hat,entireBoundary,u2fn);
  stokesBC->addZeroMeanConstraint(p);
  
  Teuchos::RCP<ExactSolution> exactSolution = Teuchos::rcp( new StokesManufacturedSolution(StokesManufacturedSolution::EXPONENTIAL) );
  // for the above solution choice, RHS is actually zero
  Teuchos::RCP<RHS> zeroRHS = Teuchos::rcp( new RHSEasy() );
  
  int minLogElements = 0, maxLogElements = 4;
  int H1Order = 2;
  pToAdd = 2;
  HConvergenceStudy study = HConvergenceStudy(exactSolution, stokesBFMath, zeroRHS,
                                              stokesBC, qoptIP, minLogElements, 
                                              maxLogElements, H1Order, pToAdd);
  quadPoints.resize(4,2);
  
  study.solve(quadPoints);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  int polyOrder = H1Order-1;
  ostringstream filePathPrefix;
  filePathPrefix << "stokesScratchPad/u1_p" << polyOrder;
  
  study.writeToFiles(filePathPrefix.str(),u1->ID(),u1hat->ID());
  filePathPrefix.str("");
  filePathPrefix << "stokesScratchPad/u2_p" << polyOrder;
  study.writeToFiles(filePathPrefix.str(),u2->ID(),u2hat->ID());
  
  filePathPrefix.str("");
  filePathPrefix << "stokesScratchPad/pressure_p" << polyOrder;
  study.writeToFiles(filePathPrefix.str(),p->ID());
  
  return 0;
}