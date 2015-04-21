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

namespace Camellia {
  class GMGSolver : public Solver {
    int _maxIters;
    bool _printToConsole;
    double _tol;

    Epetra_Map _finePartitionMap;

    GMGOperator _gmgOperator;

    bool _applySmoothing;    // whether we should add the smoothing term (almost always want this--basically we turn off for some tests)
    bool _diagonalScaling;   // whether we should scale the entire system by the diagonal

    bool _computeCondest;

    int _azOutput;

    bool _useCG; // otherwise, will use GMRES

    // info about the last call to solve()
    double _condest; // -1 if none exists
    int _iterationCount;

    int _azConvergenceOption; // defaults to AZ_rhs

    bool _printIterationCountIfNoAzOutput;

    std::vector< int > _iterationCountLog; // each time solve() is called, we push_back the number of iterations we run

    int solve(bool rebuildCoarseStiffness);
  public:
    GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter,
              Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);
    GMGSolver(TSolutionPtr<double> fineSolution, MeshPtr coarseMesh, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);

    double condest();

    int iterationCount();

    void setPrintToConsole(bool printToConsole);

    int resolve();
    int solve();

    void setApplySmoothingOperator(bool applySmoothingOp);

    void setComputeConditionNumberEstimate(bool value);

    void setUseDiagonalScaling(bool value);

    void setTolerance(double tol);

    GMGOperator & gmgOperator() {
      return _gmgOperator;
    }

    void setAztecConvergenceOption(int value);
    void setAztecOutput(int value);

    void setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap);

    void setUseConjugateGradient(bool value); // otherwise will use GMRES

    void setPrintIterationCount(bool value);

    vector<int> getIterationCountLog();
  };
}

#endif /* defined(__Camellia_debug__GMGSolver__) */
