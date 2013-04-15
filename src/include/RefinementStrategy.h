//
//  AdaptiveSolveStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_RefinementStrategy_h
#define Camellia_RefinementStrategy_h

#include "Teuchos_RCP.hpp"

class Solution;
class Mesh;
typedef Teuchos::RCP<Solution> SolutionPtr;

using namespace std;

class RefinementStrategy {
protected:
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
  double _anisotropicThreshhold;
  double _maxAspectRatio;
  vector< RefinementResults > _results;
  double _min_h;
public:
  RefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold, double min_h = 0);
  void setEnforceOneIrregularity(bool value);
  void setAnisotropicThreshhold(double value);
  void setMaxAspectRatio(double value);

  virtual void refine(bool printToConsole=false);
  virtual void refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr);
  void refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr, map<int,double> &threshMap);
  void refine(bool printToConsole, map<int,double> &xErr, map<int,double> &yErr, map<int,double> &threshMap, map<int, bool> useHRefMap);

  void getAnisotropicCellsToRefine(map<int,double> &xErr, map<int,double> &yErr, vector<int> &xCells, vector<int> &yCells, vector<int> &regCells);
  void getAnisotropicCellsToRefine(map<int,double> &xErr, map<int,double> &yErr, vector<int> &xCells, vector<int> &yCells, vector<int> &regCells, map<int,double> &threshMap);
  bool enforceAnisotropicOneIrregularity(vector<int> &xCells, vector<int> &yCells);

  virtual void refineCells(vector<int> &cellIDs);
  static void pRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs);
  static void hRefineCells(Teuchos::RCP<Mesh> mesh, const vector<int> &cellIDs);
  static void hRefineUniformly(Teuchos::RCP<Mesh> mesh);
  void getCellsAboveErrorThreshhold(vector<int> &cellsToRefine);
  void setMinH(double value);
  void setReportPerCellErrors(bool value);
};

#endif
