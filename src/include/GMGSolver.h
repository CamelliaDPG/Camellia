//
//  GMGSolver.h
//  Camellia
//
//  Created by Nate Roberts on 7/7/14.
//
//

#ifndef __Camellia_debug__GMGSolver__
#define __Camellia_debug__GMGSolver__

#include "Solver.h"
#include "Mesh.h"
#include "GMGOperator.h"

class GMGSolver : public Solver {
  int _maxIters;
  bool _printToConsole;
  double _tol;
  
  Epetra_Map _finePartitionMap;
  
  GMGOperator _gmgOperator;
public:
  GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver);
  void setPrintToConsole(bool printToConsole);
  int solve();
  void setTolerance(double tol);
};

#endif /* defined(__Camellia_debug__GMGSolver__) */
