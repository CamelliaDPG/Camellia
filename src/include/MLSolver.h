//
//  MLSolver.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 6/28/13.
//
//

#ifndef Camellia_debug_MLSolver_h
#define Camellia_debug_MLSolver_h

// Trilinos/ML includes
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"

#include "AztecOO.h"

using namespace std;

namespace Camellia
{
class MLSolver : public Solver
{
  double _resTol;
  int _maxIters;
public:
  MLSolver(double residualTolerance = 1e-12, int maxIters = 50000)
  {
    _resTol = residualTolerance;
    _maxIters = maxIters;
  }

  int solve()
  {
#ifdef ML_SCALING
    const int ntimers=4;
    enum {total, probBuild, precBuild, solve};
    ml_DblLoc timeVec[ntimers], maxTime[ntimers], minTime[ntimers];

    for (int i=0; i<ntimers; i++) timeVec[i].rank = Comm.MyPID();
    timeVec[total].value = MPI_Wtime();
#endif
#ifdef ML_SCALING
    timeVec[probBuild].value = MPI_Wtime();
#endif
    // As we wish to use AztecOO, we need to construct a solver object
    // for this problem
    AztecOO solver( problem() );
#ifdef ML_SCALING
    timeVec[probBuild].value = MPI_Wtime() - timeVec[probBuild].value;
#endif
    // ML Preconditioner boilerplate from example code--something to try...

    // create a parameter list for ML options
    ParameterList MLList;

    // Sets default parameters for classic smoothed aggregation. After this
    // call, MLList contains the default values for the ML parameters,
    // as required by typical smoothed aggregation for symmetric systems.
    // Other sets of parameters are available for non-symmetric systems
    // ("DD" and "DD-ML"), and for the Maxwell equations ("maxwell").

    int maxLevels = 8;
    char parameter[80];

    ML_Epetra::MultiLevelPreconditioner* MLPrec;
    bool useAztecAsSolver = false;
    bool useNSSADefaults = false;
    bool useSADefaults = true;
    if (useAztecAsSolver)
    {
      ML_Epetra::SetDefaults("SA",MLList);
      int iters = 5;
      for (int ilevel = 0 ; ilevel < maxLevels ; ++ilevel)
      {
        sprintf(parameter,"smoother: type (level %d)", ilevel);
        MLList.set(parameter, "Aztec");
        sprintf(parameter,"smoother: Aztec as solver (level %d)", ilevel);
        MLList.set(parameter, true);
        sprintf(parameter,"smoother: sweeps (level %d)", ilevel);
        MLList.set(parameter, iters);
      }
    }
    else if (useNSSADefaults)
    {
      ML_Epetra::SetDefaults("NSSA",MLList);
    }
    else if (useSADefaults)
    {
      ML_Epetra::SetDefaults("SA",MLList);
    }
    else
    {
      ML_Epetra::SetDefaults("SA",MLList);

      // overwrite some parameters. Please refer to the user's guide
      // for more information
      // some of the parameters do not differ from their default value,
      // and they are here reported for the sake of clarity

      int fillin = 10;
      int sweeps = 10;

      // output level, 0 being silent and 10 verbose
      MLList.set("ML output", 10);
      // maximum number of levels
      MLList.set("max levels",maxLevels);
      // set finest level to 0
      MLList.set("increasing or decreasing","increasing");

      // use Uncoupled scheme to create the aggregate
      MLList.set("aggregation: type", "Uncoupled");

      Teuchos::RCP<std::vector<int> > aztecOptions = Teuchos::rcp(new std::vector<int>(AZ_OPTIONS_SIZE));
      Teuchos::RCP<std::vector<double> > aztecParams  = Teuchos::rcp(new std::vector<double>(AZ_PARAMS_SIZE));

      int* options=  &(*aztecOptions)[0];
      double* params =  &(*aztecParams)[0];

      AZ_defaults(options,params);
      options[AZ_graph_fill] = fillin;
      options[AZ_precond] = AZ_Neumann; // AZ_Jacobi // AZ_ls; // AZ_sym_GS; // AZ_dom_decomp;
      options[AZ_subdomain_solve] = AZ_ilu;

      int k = 20; // 20th order Neumann
      options[AZ_poly_ord] = k;

      for (int ilevel = 0 ; ilevel < maxLevels ; ++ilevel)
      {
        sprintf(parameter,"smoother: type (level %d)", ilevel);
        MLList.set(parameter, "Aztec");
        sprintf(parameter,"smoother: sweeps (level %d)", ilevel);
        MLList.set(parameter, sweeps);
        sprintf(parameter,"smoother: Aztec options (level %d)", ilevel);
        MLList.set(parameter, aztecOptions);
        sprintf(parameter,"smoother: Aztec params (level %d)", ilevel);
        MLList.set(parameter, aztecParams);
      }

      //    MLList.set("smoother: type","Chebyshev");
      //    MLList.set("smoother: sweeps",10); // seems that more sweeps mean better preconditioning (fewer iterations), but greater cost per iterations

      // use both pre and post smoothing
      MLList.set("smoother: pre or post", "both");

#ifdef HAVE_ML_AMESOS
      // solve with serial direct solver KLU
      MLList.set("coarse: type","Amesos-KLU");
#else
      // this is for testing purposes only, you should have
      // a direct solver for the coarse problem (either Amesos, or the SuperLU/
      // SuperLU_DIST interface of ML)
      MLList.set("coarse: type","Jacobi");
      cout << "WARNING: using Jacobi for coarse problem (not recommended!) \n";
#endif
    }

    // Creates the preconditioning object. We suggest to use `new' and
    // `delete' because the destructor contains some calls to MPI (as
    // required by ML and possibly Amesos). This is an issue only if the
    // destructor is called **after** MPI_Finalize().
    Epetra_RowMatrix *A = problem().GetMatrix();

    MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);

    // verify unused parameters on process 0 (put -1 to print on all
    // processes)
    //    MLPrec->PrintUnused(0);
    //    MLPrec->PrintList();
#ifdef ML_SCALING
    timeVec[precBuild].value = MPI_Wtime() - timeVec[precBuild].value;
#endif

    //    MLPrec->TestSmoothers(MLList);

    // ML allows the user to cheaply recompute the preconditioner. You can
    // simply uncomment the following line:
    //
    // MLPrec->ReComputePreconditioner();
    //
    // It is supposed that the linear system matrix has different values, but
    // **exactly** the same structure and layout. The code re-built the
    // hierarchy and re-setup the smoothers and the coarse solver using
    // already available information on the hierarchy. A particular
    // care is required to use ReComputePreconditioner() with nonzero
    // threshold.

    // =========================== end of ML part =============================

    // tell AztecOO to use the ML preconditioner, specify the solver
    // and the output, then solve with 500 maximum iterations and 1e-12
    // of tolerance (see AztecOO's user guide for more details)

#ifdef ML_SCALING
    timeVec[solve].value = MPI_Wtime();
#endif
    solver.SetPrecOperator(MLPrec);
    //    solver.SetAztecOption(AZ_solver, AZ_GMRESR); // could do AZ_cg -- we are SPD (but so far it seems cg takes a bit longer...)
    solver.SetAztecOption(AZ_output, 512);
    int result = solver.Iterate(_maxIters, _resTol);
#ifdef ML_SCALING
    timeVec[solve].value = MPI_Wtime() - timeVec[solve].value;
#endif

    // destroy the preconditioner
    delete MLPrec;

    return result;
  }
};
}


#endif
