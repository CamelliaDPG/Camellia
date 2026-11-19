//
//  MatrixFreeDPGMultigrid.cpp
//  Camellia
//

#include "MatrixFreeDPGMultigrid.h"

#include "BasisReconciliation.h"
#include "BC.h"
#include "GMGOperator.h"
#include "MatrixFreeGMGPreconditioner.h"
#include "Mesh.h"
#include "Solution.h"
#include "Solver.h"

#include "Amesos2.hpp"

#include <chrono>
#include <cmath>

using namespace Camellia;

ChebyshevSmoother::ChebyshevSmoother(Teuchos::RCP<const TOperator<double>> A, Teuchos::RCP<const TOperator<double>> Dinv, int degree,
                                     double lambdaMax, double eigenRatio)
: _A(A), _Dinv(Dinv), _degree(degree), _lambdaMax(lambdaMax), _lambdaMin(lambdaMax / eigenRatio)
{
  TEUCHOS_TEST_FOR_EXCEPTION(degree < 1, std::invalid_argument, "ChebyshevSmoother: degree must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(!(lambdaMax > 0) || !(eigenRatio > 1), std::invalid_argument, "ChebyshevSmoother: need lambdaMax > 0, eigenRatio > 1");
}

void ChebyshevSmoother::smooth(const MV &b, MV &x) const
{
  // Saad, Iterative Methods, Alg. 12.1 (Chebyshev acceleration), preconditioned by D^{-1}
  const double theta = 0.5 * (_lambdaMax + _lambdaMin), delta = 0.5 * (_lambdaMax - _lambdaMin);
  const double sigma = theta / delta;
  double rho = 1.0 / sigma;
  const size_t vectorCount = b.getNumVectors();
  auto map = b.getMap();
  MV r(map, vectorCount), z(map, vectorCount), d(map, vectorCount);
  _A->apply(x, r);
  r.update(1.0, b, -1.0); // r = b - A x
  _Dinv->apply(r, z);
  d.update(1.0 / theta, z, 0.0);
  x.update(1.0, d, 1.0);
  for (int k=1; k<_degree; k++)
  {
    _A->apply(d, z);
    r.update(-1.0, z, 1.0); // r = b - A x
    _Dinv->apply(r, z);
    const double rhoNew = 1.0 / (2.0 * sigma - rho);
    d.update(2.0 * rhoNew / delta, z, rhoNew * rho);
    x.update(1.0, d, 1.0);
    rho = rhoNew;
  }
}

double ChebyshevSmoother::estimateLambdaMax(const TOperator<double> &A, const TOperator<double> &Dinv, int iterations, double boost)
{
  auto map = A.getDomainMap();
  MV x(map, 1), Ax(map, 1), y(map, 1);
  x.randomize();
  x.scale(1.0 / x.getVector(0)->norm2());
  double lambda = 0;
  for (int i=0; i<iterations; i++)
  {
    A.apply(x, Ax);
    Dinv.apply(Ax, y);
    lambda = y.getVector(0)->norm2(); // |D^{-1} A x| with |x| = 1
    x.update(1.0 / lambda, y, 0.0);
  }
  return boost * lambda;
}

double ChebyshevSmoother::estimateLambdaMaxLanczos(const TOperator<double> &A, const TOperator<double> &Dinv, int iterations, double boost)
{
  auto map = A.getDomainMap();
  MV b(map, 1), x(map, 1);
  b.randomize();
  const ConjugateGradientResult result = conjugateGradient(A, &Dinv, b, x, 0.0, iterations);
  TEUCHOS_TEST_FOR_EXCEPTION(!(result.lambdaMax > 0), std::logic_error, "estimateLambdaMaxLanczos: no Ritz value");
  return boost * result.lambdaMax;
}

namespace
{
  //! an Amesos2 factorization of a matrix, applied as its inverse
  class DirectInverse : public TOperator<double>
  {
  public:
    using MV = TVector<double>;
    explicit DirectInverse(TMatrixPtr<double> A) : _A(A)
    {
      _solver = Amesos2::create<TMatrix<double>, MV>("KLU2", A);
      _solver->symbolicFactorization();
      _solver->numericFactorization();
    }
    Teuchos::RCP<const TMap> getDomainMap() const override { return _A->getDomainMap(); }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _A->getRangeMap(); }
    void apply(const MV &X, MV &Y, Teuchos::ETransp = Teuchos::NO_TRANS, double alpha = 1.0, double beta = 0.0) const override
    {
      MV solution(_A->getDomainMap(), X.getNumVectors());
      MV rhs(X, Teuchos::Copy);
      _solver->setX(Teuchos::rcpFromRef(solution));
      _solver->setB(Teuchos::rcpFromRef(rhs));
      _solver->solve();
      Y.update(alpha, solution, beta);
    }
  private:
    TMatrixPtr<double> _A;
    Teuchos::RCP<Amesos2::Solver<TMatrix<double>, MV>> _solver;
  };
}

MatrixFreeDPGTwoLevel::MatrixFreeDPGTwoLevel(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution, MeshPtr coarseMesh)
: MatrixFreeDPGTwoLevel(fineOperator, fineSolution, coarseMesh, Options())
{
}

MatrixFreeDPGTwoLevel::MatrixFreeDPGTwoLevel(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution, MeshPtr coarseMesh,
                                             const Options &options)
{
  construct(fineOperator, fineSolution, std::vector<MeshPtr>{coarseMesh}, options);
}

MatrixFreeDPGTwoLevel::MatrixFreeDPGTwoLevel(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution,
                                             const std::vector<MeshPtr> &coarseMeshesCoarseToFine, const Options &options)
{
  construct(fineOperator, fineSolution, coarseMeshesCoarseToFine, options);
}

void MatrixFreeDPGTwoLevel::construct(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution,
                                      const std::vector<MeshPtr> &coarseMeshesCoarseToFine, const Options &options)
{
  TEUCHOS_TEST_FOR_EXCEPTION(coarseMeshesCoarseToFine.empty(), std::invalid_argument, "MatrixFreeDPGTwoLevel: no coarse mesh");
  MeshPtr coarseMesh = coarseMeshesCoarseToFine.back();
  _coarseLevelCount = int(coarseMeshesCoarseToFine.size());
  using MV = TVector<double>;
  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); };

  // the prolongation, as the conventional GMG constructs it
  auto t0 = Clock::now();
  TMapPtr finePartitionMap = fineSolution->getPartitionMap();
  BCPtr zeroBCs = fineSolution->bc()->copyImposingZero();
  auto basisReconciliation = Teuchos::rcp(new BasisReconciliation(true));
  GMGOperator gmgOperator(zeroBCs, coarseMesh, fineSolution->ip(), fineSolution->mesh(), fineSolution->getDofInterpreter(),
                          finePartitionMap, Solver::getDirectSolver(), false, basisReconciliation);
  TMatrixPtr<double> P = gmgOperator.constructProlongationOperator();
  _setupTimes.push_back({"prolongation", seconds(t0)});

  // the Galerkin coarse operator, and its factorization
  t0 = Clock::now();
  TMatrixPtr<double> coarseOperator = fineOperator->galerkinCoarseOperator(P);
  _coarseDofCount = coarseOperator->getGlobalNumRows();
  _setupTimes.push_back({"coarse operator", seconds(t0)});
  t0 = Clock::now();
  Teuchos::RCP<const TOperator<double>> coarseInverse;
  if (coarseMeshesCoarseToFine.size() == 1)
  {
    coarseInverse = Teuchos::rcp(new DirectInverse(coarseOperator));
    _setupTimes.push_back({"coarse factorization", seconds(t0)});
  }
  else
  {
    // the conventional GMG on the coarse level's Solution (whose map is P's domain), given the Galerkin operator
    SolutionPtr coarseSolution = gmgOperator.getCoarseSolution();
    _coarseGMG = Teuchos::rcp(new GMGSolver(coarseSolution, coarseMeshesCoarseToFine, 1, 0.0, GMGOperator::V_CYCLE,
                                            GMGOperator::DEFAULT_ADDITIVE_SCHWARZ, Solver::getDirectSolver(true), false));
    _coarseGMG->setSmootherType(GMGOperator::DEFAULT_ADDITIVE_SCHWARZ);
    _coarseGMG->gmgOperator()->setFineStiffnessMatrix(coarseOperator);
    coarseInverse = _coarseGMG->gmgOperator();
    _setupTimes.push_back({"coarse GMG hierarchy", seconds(t0)});
  }

  // the smoother
  t0 = Clock::now();
  Teuchos::RCP<const TOperator<double>> Dinv;
  if (options.transportSweep)
  {
    std::vector<int> fluxOwner;
    const std::vector<int> order = fineOperator->upwindCellOrder(nullptr, nullptr, &fluxOwner, nullptr, 0.0, nullptr, options.weightedCycleBreaking);
    Dinv = fineOperator->transportSweep(order, fluxOwner);
  }
  else if (options.elementSchwarz)
  {
    Dinv = fineOperator->elementAdditiveSchwarz();
  }
  else if (options.traceBlockDiagonal)
  {
    Dinv = fineOperator->traceBlockJacobi();
  }
  else
  {
    auto diagonal = Teuchos::rcp(new MV(fineOperator->getRangeMap(), 1));
    fineOperator->diagonal(*diagonal);
    Dinv = Teuchos::rcp(new DiagonalScalingOperator(diagonal));
  }
  _setupTimes.push_back({"smoother diagonal", seconds(t0)});
  t0 = Clock::now();
  _lambdaMax = options.lanczosLambdaMax ? ChebyshevSmoother::estimateLambdaMaxLanczos(*fineOperator, *Dinv, options.lanczosIterations, options.lambdaMaxBoost)
                                        : ChebyshevSmoother::estimateLambdaMax(*fineOperator, *Dinv, options.powerIterations, options.lambdaMaxBoost);
  _setupTimes.push_back({"lambda_max", seconds(t0)});
  auto smoother = std::make_shared<ChebyshevSmoother>(fineOperator, Dinv, options.chebyshevDegree, _lambdaMax, options.eigenRatio);

  // the V-cycle
  using GMG = MatrixFreeGMGPreconditioner;
  GMG::Level fine;
  fine.map = fineOperator->getRangeMap();
  fine.applyOperator = GMG::operatorApply(fineOperator);
  fine.preSmooth = [smoother](const MV &b, MV &x) { smoother->smooth(b, x); };
  fine.postSmooth = fine.preSmooth;
  fine.restrictResidual = [P](const MV &r, MV &coarse) { P->apply(r, coarse, Teuchos::TRANS); };
  fine.prolongateCorrection = [P](const MV &coarse, MV &correction) { P->apply(coarse, correction, Teuchos::NO_TRANS); };
  GMG::Level coarse;
  coarse.map = coarseOperator->getRangeMap();
  coarse.coarseSolve = [coarseInverse](const MV &b, MV &x) { coarseInverse->apply(b, x, Teuchos::NO_TRANS, 1.0, 0.0); };
  _preconditioner = Teuchos::rcp(new GMG(std::vector<GMG::Level>{fine, coarse}, "MatrixFreeDPGTwoLevel"));
}
