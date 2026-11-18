//
//  SeparableGramInverse.cpp
//  Camellia
//

#include "SeparableGramInverse.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

extern "C" void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha,
                       const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c,
                       const int *ldc);

using namespace Camellia;

namespace
{
  //! cyclic Jacobi eigensolver for a small symmetric matrix A (row-major n x n, overwritten): A = Q diag(w) Q^T
  void symmetricEigen(int n, std::vector<double> &A, std::vector<double> &w, std::vector<double> &Q)
  {
    Q.assign(n * n, 0.0);
    for (int i=0; i<n; i++) Q[i*n+i] = 1.0;
    for (int sweep=0; sweep<100; sweep++)
    {
      double offDiagonal = 0, diagonal = 0;
      for (int i=0; i<n; i++)
      {
        diagonal += A[i*n+i] * A[i*n+i];
        for (int j=i+1; j<n; j++) offDiagonal += A[i*n+j] * A[i*n+j];
      }
      if (offDiagonal <= 1e-32 * diagonal) break;
      for (int p=0; p<n; p++)
        for (int q=p+1; q<n; q++)
        {
          const double apq = A[p*n+q];
          if (apq == 0.0) continue;
          const double theta = (A[q*n+q] - A[p*n+p]) / (2 * apq);
          const double t = ((theta >= 0) ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
          const double c = 1 / std::sqrt(t * t + 1), s = t * c;
          for (int k=0; k<n; k++) // A <- A J (columns p, q)
          {
            const double akp = A[k*n+p], akq = A[k*n+q];
            A[k*n+p] = c * akp - s * akq;
            A[k*n+q] = s * akp + c * akq;
          }
          for (int k=0; k<n; k++) // A <- J^T A (rows p, q)
          {
            const double apk = A[p*n+k], aqk = A[q*n+k];
            A[p*n+k] = c * apk - s * aqk;
            A[q*n+k] = s * apk + c * aqk;
          }
          for (int k=0; k<n; k++) // Q <- Q J
          {
            const double qkp = Q[k*n+p], qkq = Q[k*n+q];
            Q[k*n+p] = c * qkp - s * qkq;
            Q[k*n+q] = s * qkp + c * qkq;
          }
        }
    }
    w.resize(n);
    for (int i=0; i<n; i++) w[i] = A[i*n+i];
  }

  //! generalized eigenproblem K V = M V diag(lambda), V^T M V = I, eigenvalues increasing; M SPD, K symmetric
  void generalizedEigen(int n, const std::vector<double> &M, const std::vector<double> &K, std::vector<double> &lambda, std::vector<double> &V)
  {
    // M = L L^T
    std::vector<double> L(n * n, 0.0);
    for (int j=0; j<n; j++)
    {
      double d = M[j*n+j];
      for (int k=0; k<j; k++) d -= L[j*n+k] * L[j*n+k];
      if (!(d > 0)) throw std::invalid_argument("SeparableGramInverse: a one-dimensional mass matrix is not positive definite");
      L[j*n+j] = std::sqrt(d);
      for (int i=j+1; i<n; i++)
      {
        double s = M[i*n+j];
        for (int k=0; k<j; k++) s -= L[i*n+k] * L[j*n+k];
        L[i*n+j] = s / L[j*n+j];
      }
    }
    // Linv = L^{-1} (lower triangular)
    std::vector<double> Linv(n * n, 0.0);
    for (int j=0; j<n; j++)
    {
      Linv[j*n+j] = 1 / L[j*n+j];
      for (int i=j+1; i<n; i++)
      {
        double s = 0;
        for (int k=j; k<i; k++) s -= L[i*n+k] * Linv[k*n+j];
        Linv[i*n+j] = s / L[i*n+i];
      }
    }
    // C = Linv K Linv^T, symmetrized
    std::vector<double> T(n * n, 0.0), C(n * n, 0.0);
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
      {
        double s = 0;
        for (int k=0; k<=i; k++) s += Linv[i*n+k] * K[k*n+j];
        T[i*n+j] = s;
      }
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
      {
        double s = 0;
        for (int k=0; k<=j; k++) s += T[i*n+k] * Linv[j*n+k];
        C[i*n+j] = s;
      }
    for (int i=0; i<n; i++)
      for (int j=i+1; j<n; j++) C[i*n+j] = C[j*n+i] = 0.5 * (C[i*n+j] + C[j*n+i]);
    std::vector<double> w, Q;
    symmetricEigen(n, C, w, Q);
    // V = Linv^T Q, columns sorted by eigenvalue
    std::vector<int> order(n);
    for (int i=0; i<n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] < w[b]; });
    lambda.resize(n);
    V.assign(n * n, 0.0);
    for (int jNew=0; jNew<n; jNew++)
    {
      const int j = order[jNew];
      lambda[jNew] = w[j];
      for (int i=0; i<n; i++)
      {
        double s = 0;
        for (int k=i; k<n; k++) s += Linv[k*n+i] * Q[k*n+j];
        V[i*n+jNew] = s;
      }
    }
  }
}

SeparableGramInverse::SeparableGramInverse(const std::vector<std::vector<double>> &mass, const std::vector<std::vector<double>> &stiffness)
: _M(mass), _K(stiffness)
{
  if (mass.empty() || (mass.size() != stiffness.size()))
    throw std::invalid_argument("SeparableGramInverse: need one mass and one stiffness matrix per direction");
  const int d = int(mass.size());
  _n.resize(d);
  _stride.resize(d);
  _V.resize(d);
  _lambda.resize(d);
  for (int k=0; k<d; k++)
  {
    const int n = int(std::lround(std::sqrt(double(mass[k].size()))));
    if ((n < 1) || (int(mass[k].size()) != n * n) || (int(stiffness[k].size()) != n * n))
      throw std::invalid_argument("SeparableGramInverse: one-dimensional matrices must be square and of equal size in each direction");
    _n[k] = n;
    _stride[k] = _size;
    _size *= n;
    generalizedEigen(n, mass[k], stiffness[k], _lambda[k], _V[k]);
  }
}

void SeparableGramInverse::checkScales(const Scales &scales) const
{
  if (int(scales.derivativeWeights.size()) != directionCount())
    throw std::invalid_argument("SeparableGramInverse: need one derivative weight per direction");
  if (!(scales.jacobian > 0) || !(scales.valueWeight > 0))
    throw std::invalid_argument("SeparableGramInverse: the jacobian and the value weight must be positive");
  for (double b : scales.derivativeWeights)
    if (!(b >= 0)) throw std::invalid_argument("SeparableGramInverse: derivative weights must be nonnegative");
}

void SeparableGramInverse::modeProduct(int direction, const std::vector<double> &A, bool transpose, double *x, double *work,
                                       int vectorCount) const
{
  // x holds vectorCount tensors one after another, so as an array it is (outer, n, inner) with inner fastest, outer
  // counting the directions after this one and the vectors.  In column-major terms each outer slab is X_o (inner x n),
  // and the mode product is Y_o = X_o op(A)^T, where A is row-major (so column-major A^T).
  const int n = _n[direction], inner = _stride[direction];
  const int outer = (_size / (inner * n)) * vectorCount;
  const double one = 1.0, zero = 0.0;
  // column-major view of row-major A is A^T; we need op(A)^T: A^T when !transpose, A when transpose
  const char transA = transpose ? 'T' : 'N';
  if (inner == 1)
  {
    // all slabs at once: X (n x outer) column-major, Y = op(A) X
    const char t = transpose ? 'N' : 'T';
    dgemm_(&t, "N", &n, &outer, &n, &one, A.data(), &n, x, &n, &zero, work, &n);
  }
  else
  {
    for (int o=0; o<outer; o++)
    {
      const double *xo = x + size_t(o) * n * inner;
      double *yo = work + size_t(o) * n * inner;
      dgemm_("N", &transA, &inner, &n, &n, &one, xo, &inner, A.data(), &n, &zero, yo, &inner);
    }
  }
  std::copy(work, work + size_t(_size) * vectorCount, x);
}

std::vector<double> SeparableGramInverse::eigenvalueDiagonal(const Scales &scales) const
{
  checkScales(scales);
  std::vector<double> D(_size, scales.valueWeight);
  for (int k=0; k<directionCount(); k++)
  {
    const double b = scales.derivativeWeights[k];
    if (b == 0.0) continue;
    const int n = _n[k], s = _stride[k];
    for (int i=0; i<_size; i++) D[i] += b * std::max(0.0, _lambda[k][(i / s) % n]);
  }
  for (double &value : D) value *= scales.jacobian;
  return D;
}

void SeparableGramInverse::applyInverse(const Scales &scales, double *x, int vectorCount) const
{
  const std::vector<double> D = eigenvalueDiagonal(scales);
  std::vector<double> work(size_t(_size) * vectorCount);
  for (int k=0; k<directionCount(); k++) modeProduct(k, _V[k], true, x, work.data(), vectorCount);  // V^T x
  for (int v=0; v<vectorCount; v++)
  {
    double *xv = x + size_t(v) * _size;
    for (int i=0; i<_size; i++) xv[i] /= D[i];
  }
  for (int k=0; k<directionCount(); k++) modeProduct(k, _V[k], false, x, work.data(), vectorCount); // V z
}

void SeparableGramInverse::applyEigenbasisTranspose(double *x, int vectorCount) const
{
  std::vector<double> work(size_t(_size) * vectorCount);
  for (int k=0; k<directionCount(); k++) modeProduct(k, _V[k], true, x, work.data(), vectorCount);
}

void SeparableGramInverse::apply(const Scales &scales, double *x, int vectorCount) const
{
  checkScales(scales);
  std::vector<double> work(_size), massApplied(_size), term(_size), result(_size);
  const int d = directionCount();
  for (int v=0; v<vectorCount; v++)
  {
    double *xv = x + size_t(v) * _size;
    std::copy(xv, xv + _size, massApplied.data());
    for (int k=0; k<d; k++) modeProduct(k, _M[k], false, massApplied.data(), work.data(), 1);
    for (int i=0; i<_size; i++) result[i] = scales.valueWeight * massApplied[i];
    for (int k=0; k<d; k++)
    {
      const double b = scales.derivativeWeights[k];
      if (b == 0.0) continue;
      std::copy(xv, xv + _size, term.data());
      for (int e=0; e<d; e++) modeProduct(e, (e == k) ? _K[e] : _M[e], false, term.data(), work.data(), 1);
      for (int i=0; i<_size; i++) result[i] += b * term[i];
    }
    for (int i=0; i<_size; i++) xv[i] = scales.jacobian * result[i];
  }
}

double SeparableGramInverse::factorResidual() const
{
  double residual = 0;
  for (int k=0; k<directionCount(); k++)
  {
    const int n = _n[k];
    const std::vector<double> &V = _V[k];
    const double lambdaScale = std::max(1.0, _lambda[k].back());
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
      {
        double vmv = 0, vkv = 0;
        for (int a=0; a<n; a++)
          for (int b=0; b<n; b++)
          {
            vmv += V[a*n+i] * _M[k][a*n+b] * V[b*n+j];
            vkv += V[a*n+i] * _K[k][a*n+b] * V[b*n+j];
          }
        residual = std::max(residual, std::abs(vmv - ((i == j) ? 1.0 : 0.0)));
        residual = std::max(residual, std::abs(vkv - ((i == j) ? _lambda[k][i] : 0.0)) / lambdaScale);
      }
  }
  return residual;
}
