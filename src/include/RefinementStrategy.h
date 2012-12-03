//
//  AdaptiveSolveStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_RefinementStrategy_h
#define Camellia_RefinementStrategy_h

class Solution;
class Mesh;

#include "Teuchos_RCP.hpp"

using namespace std;

class RefinementStrategy {
protected:
  typedef Teuchos::RCP<Solution> SolutionPtr;
  
  struct RefinementResults {
    int numElements;
    int numDofs;
    double totalEnergyError;
  };
  
  static void setResults(RefinementResults &solnResults, int numElements, int numDofs, double totalEnergyError);
  SolutionPtr _solution;
  double _relativeEnergyThreshold;
  bool _enforceOneIrregularity;
  bool _reportPerCellErrors;  
  vector< RefinementResults > _results;
public:
  RefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold);
  void setEnforceOneIrregularity(bool value);
  virtual void refine(bool printToConsole=false);
  virtual void refineCells(vector<int> &cellIDs);
  static void pRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs);
  static void hRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs);
  static void hRefineUniformly(Teuchos::RCP<Mesh> mesh);
  void getCellsAboveErrorThreshhold(vector<int> &cellsToRefine);
  void setReportPerCellErrors(bool value);
};

#endif
