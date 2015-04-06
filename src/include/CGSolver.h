//
//  CGSolver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_CGSolver_h
#define Camellia_CGSolver_h

#include "Solver.h"

namespace Camellia {
	class CGSolver : public Solver {
	  int _maxIters;
	  bool _printToConsole;
	  double _tol;
	public:
	  CGSolver(int maxIters, double tol);
	  void setPrintToConsole(bool printToConsole);
	  int solve();
	  void setTolerance(double tol);
	};
}

#endif
