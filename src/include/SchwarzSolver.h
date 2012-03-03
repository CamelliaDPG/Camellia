//
//  SchwarzSolver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SchwarzSolver_h
#define Camellia_SchwarzSolver_h

#include "Solver.h"

class SchwarzSolver : public Solver {
  int _overlapLevel;
  int _maxIters;
  bool _printToConsole;
  double _tol;
public:
  SchwarzSolver(int overlapLevel, int maxIters, double tol);
  void setPrintToConsole(bool printToConsole);
  int solve();
};

#endif
