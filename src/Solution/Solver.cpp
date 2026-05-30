//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Solver.h"

#include "GMGSolver.h"
#include "Mesh.h"
#include "Solution.h"
#include "SuperLUDistSolver.h"
#include "PardisoSolver.h"

using namespace Camellia;

template <typename Scalar>
void TSolver<Scalar>::printAvailableSolversReport()
{
  cout << "Available solvers:\n";
  cout << solverChoiceString(KLU) << endl;
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  cout << solverChoiceString(SuperLUDist) << endl;
#endif
#ifdef HAVE_AMESOS_MUMPS
  cout << solverChoiceString(MUMPS) << endl;
#endif
#ifdef HAVE_AMESOS_PARDISO_MKL
  cout << solverChoiceString(Pardiso) << endl;
#endif
//  cout << solverChoiceString(GMG) << endl;
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getSolver(SolverChoice choice, bool saveFactorization,
    double residualTolerance, int maxIterations,
    TSolutionPtr<double> fineSolution, TSolverPtr<Scalar> coarseSolver)
{
  switch (choice)
  {
  case KLU:
    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "klu") );
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  case SuperLUDist:
    return Teuchos::rcp( new SuperLUDistSolver(saveFactorization) );
//    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "superlu_dist") );
#endif
#ifdef HAVE_AMESOS_MUMPS
  case MUMPS:
    return Teuchos::rcp( new MumpsSolver(saveFactorization) );
#endif
#ifdef HAVE_AMESOS_PARDISO_MKL
  case Pardiso:
    return Teuchos::rcp( new PardisoSolver(saveFactorization) );
    // return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "pardisomkl") );
#endif
      
  case GMG:
  {
    Teuchos::ParameterList pl;
    pl.set("kCoarse", 0);
    int delta_k = fineSolution->mesh()->testSpaceEnrichment(); // might matter in cases where GMGOperator actually computes some local stiffness matrices on its own (currently, this can happen on coarse meshes when static condensation is employed).
    pl.set("delta_k", delta_k);
    pl.set("jumpToCoarsePolyOrder", false);
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(fineSolution->mesh(), pl);
    
    vector<MeshPtr> prunedMeshes;
    int minDofCount = 2000; // skip any coarse meshes that have fewer dofs than this
    for (int i=0; i<meshesCoarseToFine.size()-2; i++) // leave the last two meshes, so we can guarantee there are at least two
    {
      MeshPtr mesh = meshesCoarseToFine[i];
      GlobalIndexType numGlobalDofs;
      if (fineSolution->usesCondensedSolve())
        numGlobalDofs = mesh->numFluxDofs(); // this might under-count, in case e.g. of pressure constraints.  But it's meant as a heuristic anyway.
      else
        numGlobalDofs = mesh->numGlobalDofs();
      
      if (numGlobalDofs > minDofCount)
      {
        prunedMeshes.push_back(mesh);
      }
    }
    prunedMeshes.push_back(meshesCoarseToFine[meshesCoarseToFine.size()-2]);
    prunedMeshes.push_back(meshesCoarseToFine[meshesCoarseToFine.size()-1]);
    
    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(fineSolution, prunedMeshes, maxIterations, residualTolerance, GMGOperator::V_CYCLE, GMGOperator::DEFAULT_ADDITIVE_SCHWARZ,
                                                                   coarseSolver, fineSolution->usesCondensedSolve(), false));
    
    gmgSolver->setAztecOutput(0);
    gmgSolver->setComputeConditionNumberEstimate(false);

    return gmgSolver;
  }
  default:
    cout << "Solver choice " << solverChoiceString(choice) << " not recognized.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Solver choice not recognized!");
  }
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getDirectSolver(bool saveFactorization) {
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::SuperLUDist, saveFactorization);
#elif defined(HAVE_AMESOS_MUMPS)
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::MUMPS, saveFactorization);
#elif defined(HAVE_AMESOS_PARDISO_MKL)
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::Pardiso, saveFactorization);
#else
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::KLU, saveFactorization);
#endif
}

namespace Camellia
{
template class TSolver<double>;
template class TAmesos2Solver<double>;
}
