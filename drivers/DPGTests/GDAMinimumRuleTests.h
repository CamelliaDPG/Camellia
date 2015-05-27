//
//  GDAMinimumRuleTests.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/26/14.
//
//

#ifndef __Camellia_debug__GDAMinimumRuleTests__
#define __Camellia_debug__GDAMinimumRuleTests__

#include <iostream>

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "MeshTopology.h"

class GDAMinimumRuleTests : public TestSuite {
private:
  void setup();
  void teardown();
  
  SolutionPtr confusionExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, bool divideIntoTriangles);
  SolutionPtr poissonExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, FunctionPtr phi_exact, bool divideIntoTriangles);
  SolutionPtr stokesExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order,
                                  FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool divideIntoTriangles);
  SolutionPtr stokesCavityFlowSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, bool divideIntoTriangles);
  
  bool subTestCompatibleSolutionsAgree(int horizontalCells, int verticalCells, int H1Order, int numUniformRefinements, bool divideIntoTriangles);
  
  bool testHangingNodePoisson(bool useQuads);
  bool testHangingNodeStokes(bool useQuads);
  
  SolutionPtr poissonExactSolution1D(int horizontalCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces);
  
  SolutionPtr poissonExactSolution3D(int horizontalCells, int verticalCells, int depthCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces);
  
  SolutionPtr poissonExactSolution3DHangingNodes(int irregularity, FunctionPtr phi_exact, int H1Order);
  
  bool checkLocalGlobalConsistency(MeshPtr mesh);
  
  FieldContainer<double> _testPoints1D, _testPoints2D; // points on ref cell
  
public:
  GDAMinimumRuleTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "GDAMinimumRuleTests"; }
  
  bool testLocalInterpretationConsistency();
  bool testGlobalToLocalToGlobalConsistency(); // should be able to map global to local and back, and get the same results.

  bool testMultiCellMesh();
  bool testSingleCellMesh();
  
  bool test1IrregularityEnforcement();
  bool testHRefinements();

  bool testHangingNodePoissonTriangle();
  bool testHangingNodeStokesTriangle();
  
  bool testHangingNodePoissonQuad();
  bool testHangingNodeStokesQuad();
  
  bool testPoissonCompatibleMeshWithHeterogeneousOrientations2D();
  
};


#endif /* defined(__Camellia_debug__GDAMinimumRuleTests__) */
