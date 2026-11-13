#pragma once

#include <sstream>
#include "ModeSplitting.hpp"
#include "SchurChebyshev.hpp"
#include "ExplicitSchur.hpp"
#include <Eigen/Sparse>
#include <cstdlib>
#include "TensorProductAssembly.hpp"
#include "../../PlanarMITL/ComparisonProblem.hpp"

namespace PlanarMITLBoba
{
/// Integrate s=beta.grad(product of bubbles), including the continuous internal flux.
/// This is a continuum manufactured RHS, independently of the trial basis.
/// Assemble a four-mode TT matrix as a sparse matrix without forming it densely.  Each core C_d(r_{d-1}, n, m, r_d) is
/// sparse in practice (block structure in the cell index), so the product is built core by core as sparse Kronecker
/// products carrying the rank index: after core d the partial matrix has rows (r_d, rows_1..d) and cols (cols_1..d).
/// Linear index convention matches the TT (first mode fastest).
inline Eigen::SparseMatrix<double> sparseMatrix(const TTMatrix &matrix, double drop = 0.0);

/// Lower-memory variant: TT-round a copy of the matrix first (rank paths are what multiply the partial products), then
/// assemble.  Used for the large Gram block.
inline Eigen::SparseMatrix<double> sparseMatrixRounded(const TTMatrix &matrix, double tolerance, double drop = 0.0)
{
  auto copy = matrix;
  DpgBoba::round(copy, tolerance);
  return sparseMatrix(copy, drop);
}

inline Eigen::SparseMatrix<double> sparseMatrix(const TTMatrix &matrix, double drop)
{
  using Sp = Eigen::SparseMatrix<double>;
  // partial[r] : sparse matrix (rows_1..d x cols_1..d), one per current right rank index r
  std::vector<Sp> partial(1, Sp(1, 1));
  partial[0].insert(0, 0) = 1.0;
  size_t rowsSoFar = 1, colsSoFar = 1;
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &core = matrix.cores[d];
    const size_t rl = core.sizes(0), n = core.sizes(1), m = core.sizes(2), rr = core.sizes(3);
    auto v = core.const_view();
    std::vector<Sp> next(rr, Sp(rowsSoFar * n, colsSoFar * m));
    std::vector<Eigen::Triplet<double>> trip;
    for (size_t b = 0; b < rr; ++b)
    {
      trip.clear();
      for (size_t a = 0; a < rl; ++a)
      {
        const Sp &P = partial[a];
        if (P.nonZeros() == 0) continue;
        for (size_t j = 0; j < m; ++j)
          for (size_t i = 0; i < n; ++i)
          {
            const double c = v({a, i, j, b});
            if (c == 0.0) continue;
            // new linear indices: old index fastest, then this mode's index
            for (int k = 0; k < P.outerSize(); ++k)
              for (Sp::InnerIterator it(P, k); it; ++it)
                trip.emplace_back(int(it.row() + rowsSoFar * i), int(it.col() + colsSoFar * j), it.value() * c);
          }
      }
      next[b].setFromTriplets(trip.begin(), trip.end());
      if (drop > 0) next[b].prune(drop, 1.0);
      next[b].makeCompressed();
    }
    partial = std::move(next);
    rowsSoFar *= n; colsSoFar *= m;
  }
  return partial[0];
}

/// Element blocks of a TT matrix whose cores are block diagonal in each axis's cell index (the Gram matrix): for each
/// element (cell multi-index), contract the cores restricted to that cell's rows/cols.  Writes all blocks, in element
/// order (x cell fastest), to <prefix>_Gelem.bin, plus each block's global row indices to <prefix>_Gelem_idx.bin.
inline void writeElementBlocks(const TTMatrix &matrix, const std::vector<Axis> &axes, const std::string &prefix)
{
  std::array<size_t, 4> cells, nb;
  for (size_t d = 0; d < 4; ++d)
  {
    cells[d] = axes[d].vertices.size() - 1;
    nb[d] = matrix.core_rows(d) / cells[d];
    if (nb[d] * cells[d] != matrix.core_rows(d) || matrix.core_cols(d) != matrix.core_rows(d))
      throw std::invalid_argument("writeElementBlocks: matrix is not square and cell-blocked");
  }
  std::ofstream out(prefix + "_Gelem.bin", std::ios::binary), idxOut(prefix + "_Gelem_idx.bin", std::ios::binary);
  std::ofstream meta(prefix + "_Gelem.txt");
  const size_t blockSize = nb[0] * nb[1] * nb[2] * nb[3];
  meta << cells[0] << ' ' << cells[1] << ' ' << cells[2] << ' ' << cells[3] << ' '
       << nb[0] << ' ' << nb[1] << ' ' << nb[2] << ' ' << nb[3] << ' ' << blockSize << '\n';
  const size_t rowStride1 = matrix.core_rows(0), rowStride2 = rowStride1 * matrix.core_rows(1),
               rowStride3 = rowStride2 * matrix.core_rows(2);
  std::vector<double> block(blockSize * blockSize);
  std::vector<int> idx(blockSize);
  for (size_t c3 = 0; c3 < cells[3]; ++c3)
    for (size_t c2 = 0; c2 < cells[2]; ++c2)
      for (size_t c1 = 0; c1 < cells[1]; ++c1)
        for (size_t c0 = 0; c0 < cells[0]; ++c0)
        {
          const std::array<size_t, 4> c{c0, c1, c2, c3};
          // partial[a] is the (rows x cols) block over the first d axes, for right rank index a
          std::vector<Eigen::MatrixXd> partial(1, Eigen::MatrixXd::Ones(1, 1));
          for (size_t d = 0; d < 4; ++d)
          {
            const auto &core = matrix.cores[d];
            auto v = core.const_view();
            const size_t rl = core.sizes(0), rr = core.sizes(3), n = nb[d];
            const size_t base = c[d] * n;
            std::vector<Eigen::MatrixXd> next(rr, Eigen::MatrixXd::Zero(partial[0].rows() * n, partial[0].cols() * n));
            for (size_t b = 0; b < rr; ++b)
              for (size_t a = 0; a < rl; ++a)
              {
                Eigen::MatrixXd local(n, n);
                for (size_t j = 0; j < n; ++j)
                  for (size_t i = 0; i < n; ++i) local(i, j) = v({a, base + i, base + j, b});
                if (local.isZero(0.0)) continue;
                // Kronecker product kron(local, partial[a]): earlier axes fastest
                const auto &P = partial[a];
                for (size_t j = 0; j < n; ++j)
                  for (size_t i = 0; i < n; ++i)
                    if (local(i, j) != 0.0)
                      next[b].block(i * P.rows(), j * P.cols(), P.rows(), P.cols()) += local(i, j) * P;
              }
            partial = std::move(next);
          }
          const Eigen::MatrixXd &K = partial[0];
          out.write(reinterpret_cast<const char*>(K.data()), std::streamsize(blockSize * blockSize * sizeof(double)));
          size_t k = 0;
          for (size_t i3 = 0; i3 < nb[3]; ++i3)
            for (size_t i2 = 0; i2 < nb[2]; ++i2)
              for (size_t i1 = 0; i1 < nb[1]; ++i1)
                for (size_t i0 = 0; i0 < nb[0]; ++i0)
                  idx[k++] = int((c0 * nb[0] + i0) + rowStride1 * (c1 * nb[1] + i1) + rowStride2 * (c2 * nb[2] + i2) +
                                 rowStride3 * (c3 * nb[3] + i3));
          idxOut.write(reinterpret_cast<const char*>(idx.data()), std::streamsize(blockSize * sizeof(int)));
        }
}

inline void writeSparse(const Eigen::SparseMatrix<double> &A, const std::string &name)
{
  std::ofstream out(name, std::ios::binary);
  const long long rows = A.rows(), cols = A.cols(), nnz = A.nonZeros();
  out.write(reinterpret_cast<const char*>(&rows), sizeof rows);
  out.write(reinterpret_cast<const char*>(&cols), sizeof cols);
  out.write(reinterpret_cast<const char*>(&nnz), sizeof nnz);
  std::vector<int> ri, ci; std::vector<double> vals; ri.reserve(nnz); ci.reserve(nnz); vals.reserve(nnz);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
    { ri.push_back(int(it.row())); ci.push_back(int(it.col())); vals.push_back(it.value()); }
  out.write(reinterpret_cast<const char*>(ri.data()), std::streamsize(nnz * sizeof(int)));
  out.write(reinterpret_cast<const char*>(ci.data()), std::streamsize(nnz * sizeof(int)));
  out.write(reinterpret_cast<const char*>(vals.data()), std::streamsize(nnz * sizeof(double)));
}

inline BlockVector comparisonRHS(const System &system, const std::vector<Axis> &axes,
                                const LorentzFactors &h, const PlanarMITL::ComparisonProblem &problem)
{
  std::array<Eigen::VectorXd,4> value, derivative;
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &axis = axes[d];
    value[d].resize(axis.points.size()); derivative[d].resize(axis.points.size());
    for (Eigen::Index i = 0; i < axis.points.size(); ++i)
    {
      value[d](i) = problem.bubble(axis.points(i),axis.vertices.front(),axis.vertices.back());
      derivative[d](i) = problem.bubbleDerivative(axis.points(i),axis.vertices.front(),axis.vertices.back());
    }
  }
  const auto &profile = fieldProfile();
  auto term = [&](size_t direction, const Eigen::VectorXd &ux,
                  const Eigen::VectorXd &uz, double coefficient, int field = -1) {
    boba::Array<host_vector,4> factors;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd weighted = axes[d].weights.array()*(d == direction ? derivative[d] : value[d]).array();
      if (d == 2) weighted.array() *= ux.array();
      if (d == 3) weighted.array() *= uz.array();
      if (field >= 0 && profile.varying() && d < 2) weighted.array() *= profile.values(axes[d].points, field, int(d)).array();
      Eigen::VectorXd integral = axes[d].test.transpose()*weighted;
      factors[d] = host_vector({size_t(integral.size())});
      auto view = factors[d].view();
      for (Eigen::Index i = 0; i < integral.size(); ++i) view(i) = integral(i);
    }
    return coefficient*boba::make_tt_from_vectors<4,host_space,double>(factors);
  };
  const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(axes[2].points.size());
  const Eigen::VectorXd onesZ = Eigen::VectorXd::Ones(axes[3].points.size());
  auto source = term(1,onesX,onesZ,1);
  if (problem.Ex != 0) source += term(2,onesX,onesZ,problem.qOverM*problem.Ex,0);
  for (Eigen::Index k = 0; k < h.left.cols(); ++k)
  {
    Eigen::VectorXd vx = h.left.col(k).array()*axes[2].points.array();
    Eigen::VectorXd vz = h.right.col(k).array()*axes[3].points.array();
    source += term(0,vx,h.right.col(k),1);
    if (problem.By != 0)
    {
      source += term(2,h.left.col(k),vz,-problem.qOverM*problem.By,1);
      source += term(3,vx,h.right.col(k),problem.qOverM*problem.By,1);
    }
    DpgBoba::round(source,problem.svdTolerance);
  }
  BlockVector rhs(system.sizes.size());
  for (size_t b = 0; b < rhs.block_size; ++b) rhs(b) = zero_tt_vector<4>(system.sizes[b]);
  rhs(0) = source;
  return rhs;
}

inline int runTensorComparison(int argc, char **argv)
{
  using namespace PlanarMITL;
  ComparisonProblem problem;
  const int parsed = problem.parse(argc,argv,"tensor");
  if (parsed) return parsed == 2 ? 0 : parsed;
  DpgBoba::AmenOptions options;
  options.tolerance = problem.tolerance; options.roundingTolerance = problem.svdTolerance;
  options.maxSweeps = problem.maxSweeps; options.kickrank = problem.kickrank;
  options.maxRank = problem.maxRank; options.verbose = problem.verbose;
  options.localSolver = problem.localSolver;
  options.localTolerance = problem.localTolerance;
  options.maxLocalIterations = problem.maxLocalIterations;
  options.infSupEnrichment = problem.infSupEnrichment;
  options.rieszTolerance = problem.rieszSvdTolerance;
  options.denseFillThreshold = problem.denseFillThreshold;
  const auto start = ComparisonClock::now();
  basisChoice() = BasisChoice{problem.testBasis, problem.fieldBasis, problem.traceBasis};
  std::cout << "bases test=" << problem.testBasis << " field=" << problem.fieldBasis << " trace=" << problem.traceBasis << '\n';
  const auto vertices = problem.vertices();
  std::vector<Axis> axes;
  for (const auto &mesh : vertices) axes.emplace_back(mesh,problem.p,problem.dp,problem.quadratureDegree);
  LorentzFactors h(axes[2],axes[3],problem.c,problem.coefficientTolerance);
  TestNormWeight norm;
  norm.alpha = problem.alpha; norm.meshScale = problem.alphaMeshScale; norm.transportScale = problem.alphaTransportScale;
  norm.c = problem.c; norm.accelerationBounds = problem.accelerationBoundValues(); norm.separable = problem.separableTestNorm;
  {
    auto &profile = fieldProfile();
    profile.variation = problem.fieldVariation;
    profile.x0 = vertices[0].front(); profile.x1 = vertices[0].back(); profile.t0 = vertices[1].front(); profile.t1 = vertices[1].back();
    std::cout << "fields: Ex=" << problem.Ex << " By=" << problem.By << " variation=" << profile.variation
              << " test_norm=" << (norm.separable ? "separable (N5)" : "graph") << " acceleration_bounds=" << norm.accelerationBounds[0]
              << "," << norm.accelerationBounds[1] << '\n';
  }
  auto system = assemble(axes,h,problem.qOverM,problem.Ex,problem.By,norm,problem.svdTolerance);
  auto rhs = comparisonRHS(system,axes,h,problem);
  const double assemblySeconds = elapsed(start);
  if (std::getenv("PLANAR_MITL_G8_RANKS"))
  {
    reportCellsFirstGramRanks(axes, h, problem.qOverM, problem.Ex, problem.By, norm, system.names);
    return 0;
  }
  if (const char* gTest = std::getenv("PLANAR_MITL_G_SOLVE_TEST"))
  {
    // Cost of the inner TT solve a Schur-complement method needs every iteration: single-block AMEn on G e = l
    // (G^{-1} l has the rank profile of G^{-1} B x for an iterate).  Value: comma-separated local solvers to try.
    BlockMatrix gram(1, 1); gram({0, 0}) = system.matrix({0, 0});
    BlockVector l(1); l(0) = rhs(0);
    std::stringstream solvers(gTest); std::string item;
    while (std::getline(solvers, item, ','))
    {
      auto gOptions = options; gOptions.localSolver = std::stoi(item); gOptions.infSupEnrichment = false;
      const auto t0 = ComparisonClock::now();
      auto e = DpgBoba::solveAmen(gram, l, gOptions);
      const double seconds = elapsed(t0);
      auto residual = l(0); residual -= gram({0, 0}) * e(0); DpgBoba::round(residual, 1e-14);
      std::cout << "g_solve_test local_solver=" << item << " seconds=" << seconds << " ranks=" << e(0).ranks_string()
                << " relative_residual=" << boba::norm_frobenius(residual) / boba::norm_frobenius(l(0)) << std::endl;
    }
    return 0;
  }
  if (const char* sparsePrefix = std::getenv("PLANAR_MITL_DUMP_SPARSE"))
  {
    // Debug: row-0 blocks (Gram G and test x trial blocks B_j) as sparse COO files, RHS blocks, TT mode sizes.
    std::ofstream meta(std::string(sparsePrefix) + "_blocks.txt");
    writeElementBlocks(system.matrix({0,0}), axes, sparsePrefix);
    for (size_t j = 1; j < system.names.size(); ++j)
    {
      auto A = sparseMatrix(system.matrix({0,j}), 1e-14);
      writeSparse(A, std::string(sparsePrefix) + "_0_" + std::to_string(j) + ".coo");
      meta << 0 << ' ' << j << ' ' << A.rows() << ' ' << A.cols() << ' ' << system.names[0] << ' ' << system.names[j]
           << ' ' << A.nonZeros() << '\n';
    }
    std::ofstream modes(std::string(sparsePrefix) + "_modes.txt");
    for (size_t b = 0; b < system.names.size(); ++b)
    {
      auto full = rhs(b).decompress();
      std::ofstream out(std::string(sparsePrefix) + "_rhs_" + std::to_string(b) + ".bin", std::ios::binary);
      out.write(reinterpret_cast<const char*>(full.data()), std::streamsize(rhs(b).get_full_size()*sizeof(double)));
      modes << b << ' ' << system.names[b];
      for (size_t d = 0; d < 4; ++d) modes << ' ' << system.sizes[b][d];
      modes << '\n';
    }
  }
  if (const char* dumpPrefix = std::getenv("PLANAR_MITL_DUMP_GLOBAL"))
  {
    // Debug: write the decompressed global blocks (row-major per block (i,j), raw doubles) for conditioning studies.
    std::ofstream meta(std::string(dumpPrefix) + "_blocks.txt");
    for (size_t i = 0; i < system.names.size(); ++i)
      for (size_t j = 0; j < system.names.size(); ++j)
      {
        const auto &block = system.matrix({i,j});
        Eigen::MatrixXd dense = denseMatrix(block);
        const std::string name = std::string(dumpPrefix) + "_" + std::to_string(i) + "_" + std::to_string(j) + ".bin";
        std::ofstream out(name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(dense.data()), std::streamsize(dense.size()*sizeof(double)));
        meta << i << ' ' << j << ' ' << dense.rows() << ' ' << dense.cols() << ' ' << system.names[i] << ' ' << system.names[j] << '\n';
      }
    // RHS blocks (full vectors) and per-block TT mode sizes (x, t, ux, uz; x fastest in the linear index).
    std::ofstream modes(std::string(dumpPrefix) + "_modes.txt");
    for (size_t b = 0; b < system.names.size(); ++b)
    {
      auto full = rhs(b).decompress();
      const size_t count = rhs(b).get_full_size();
      std::ofstream out(std::string(dumpPrefix) + "_rhs_" + std::to_string(b) + ".bin", std::ios::binary);
      out.write(reinterpret_cast<const char*>(full.data()), std::streamsize(count*sizeof(double)));
      modes << b << ' ' << system.names[b];
      for (size_t d = 0; d < 4; ++d) modes << ' ' << system.sizes[b][d];
      modes << '\n';
    }
  }
  const auto solveStart = ComparisonClock::now();
  BlockVector solution;
  std::array<bool,4> splitMask{false,false,false,false};
  if (problem.splitX) splitMask[0] = true;
  if (problem.splitAxes == "auto")
    for (size_t d = 0; d < 4; ++d) splitMask[d] = axes[d].vertices.size() > 2;
  else if (!problem.splitAxes.empty())
  {
    if (problem.splitAxes.size() != 4) throw std::invalid_argument("--splitAxes takes a 4-digit mask or auto");
    for (size_t d = 0; d < 4; ++d) splitMask[d] = problem.splitAxes[d] == '1';
  }
  if (problem.explicitSchur)
  {
    ExplicitSchurOptions ex;
    ex.sTolerance = problem.explicitSchurTolerance; ex.tolerance = problem.chebTolerance; ex.rtol = problem.chebRtol;
    ex.iterations = problem.chebIterations; ex.every = problem.chebEvery; ex.boundsSteps = problem.chebBoundsSteps;
    ex.boundsTolerance = problem.chebBoundsTolerance; ex.lminSafety = problem.chebLminSafety;
    ex.lambdaMaxMode = problem.explicitSchurLambdaMax; ex.powerIterations = problem.explicitSchurPowerIterations;
    ex.powerTolerance = problem.explicitSchurPowerTolerance; ex.boostFactor = problem.explicitSchurBoost;
    ex.powerStart = problem.explicitSchurPowerStart;
    ex.lambdaMinDiagnosticIterations = problem.explicitSchurLambdaMinIterations;
    ex.solver = problem.explicitSchurSolver;
    ex.amen = options; ex.amen.roundingTolerance = problem.chebTolerance; ex.amen.infSupEnrichment = false;
    ex.amen.scaleSites = std::getenv("PLANAR_MITL_AMEN_SCALE_SITES") != nullptr;   // S~ is Jacobi-scaled already
    ex.split = splitMask;
    solution = solveExplicitSchur(system,rhs,axes,h,problem.qOverM,problem.Ex,problem.By,ex,norm);
  }
  else if (problem.schurChebyshev)
  {
    ChebyshevOptions cheb;
    cheb.lmin = problem.chebLmin; cheb.lmax = problem.chebLmax; cheb.tolerance = problem.chebTolerance;
    cheb.operatorTolerance = problem.svdTolerance; cheb.iterations = problem.chebIterations; cheb.every = problem.chebEvery;
    cheb.kroneckerFile = problem.chebKronecker; cheb.rtol = problem.chebRtol; cheb.factoredField = problem.chebFactoredField;
    cheb.boundsTolerance = problem.chebBoundsTolerance; cheb.boundsSteps = problem.chebBoundsSteps; cheb.lminSafety = problem.chebLminSafety; cheb.jacobiTolerance = problem.chebJacobiTolerance; cheb.scaled = problem.chebScaled; cheb.zipUp = problem.chebZipUp; cheb.structuredField = problem.chebStructuredField;
    solution = solveSchurChebyshev(system,rhs,axes,h,problem.qOverM,problem.Ex,problem.By,cheb,norm);
  }
  else if (splitMask[0] || splitMask[1] || splitMask[2] || splitMask[3])
  {
    std::cout << "split_axes=" << splitMask[0] << splitMask[1] << splitMask[2] << splitMask[3] << '\n';
    solution = solveSplitAxes(system,rhs,axes,splitMask,problem.svdTolerance,options,
                              std::getenv("PLANAR_MITL_CHECK_SPLIT") != nullptr);
  }
  else if (problem.splitModes)
  {
    SplitBlockMatrix splitMatrix; SplitBlockVector splitRHS; ModeSplit split;
    splitSystem(system,rhs,axes,problem.svdTolerance,splitMatrix,splitRHS,split);
    for (size_t b = 0; b < system.names.size(); ++b)
      std::cout << "split block=" << system.names[b] << " sizes=" << splitRHS(b).sizes()
                << " test_row_ranks=" << splitMatrix({0,b}).ranks_string() << '\n';
    if (std::getenv("PLANAR_MITL_CHECK_SPLIT"))
    {
      // Consistency check: (split A)(split x) merged must equal A x for a random four-mode x.
      auto x = DpgBoba::initialGuess(rhs,3);
      SplitBlockVector sx(x.block_size);
      for (size_t b = 0; b < x.block_size; ++b) sx(b) = splitVector(x(b),split.basisSizes[b],1e-14);
      auto merged = mergeSolution(sx);
      for (size_t i = 0; i < x.block_size; ++i)
      {
        auto d = merged(i); d -= x(i);
        auto Ax = system.matrix({i,0})*x(0); auto SAx = splitMatrix({i,0})*sx(0);
        for (size_t j = 1; j < x.block_size; ++j) { Ax += system.matrix({i,j})*x(j); SAx += splitMatrix({i,j})*sx(j); }
        BlockVector one(1); SplitBlockVector sone(1); sone(0) = SAx; one = mergeSolution(sone);
        auto e = one(0); e -= Ax;
        std::cout << "split_check block=" << system.names[i] << " vector_roundtrip=" << boba::norm_frobenius(d)/boba::norm_frobenius(x(i))
                  << " apply_difference=" << boba::norm_frobenius(e)/(boba::norm_frobenius(Ax)+1e-300) << '\n';
      }
    }
    solution = mergeSolution(DpgBoba::solveAmen(splitMatrix,splitRHS,options));
  }
  else solution = DpgBoba::solveAmen(system.matrix,rhs,options);
  const double solveSeconds = elapsed(solveStart);
  const double totalSeconds = elapsed(start);

  // Residual evaluation, storage accounting, and output are outside the timed solve.
  auto metrics = problem.metadata();
  metrics["assembly_seconds"] = assemblySeconds;
  metrics["linear_solve_seconds"] = solveSeconds;
  metrics["total_seconds"] = totalSeconds;
  metrics["inverse_gamma_rank"] = h.left.cols();
  metrics["inverse_gamma_svd_error"] = h.relativeError;
  metrics["mixed_residual"] = DpgBoba::reportResidual(system.matrix,rhs,solution,
                                                    std::min(1e-14,problem.svdTolerance*0.01));
  // Diagnostic: true TT ranks of each returned block, by rounding a copy at several relative tolerances.
  for (size_t b = 0; b < solution.block_size; ++b)
  {
    std::cout << "rounded_ranks block=" << system.names[b];
    for (double tol : {1e-8, 1e-5, 1e-3, 1e-2})
    {
      auto copy = solution(b);
      DpgBoba::round(copy, tol);
      std::cout << " tol" << tol << "=" << copy.ranks_string();
    }
    std::cout << '\n';
  }
  double storedMatrix = 0, storedSolution = 0, storedRHS = 0, fullTrial = 0, fullMixed = 0;
  for (size_t b = 0; b < solution.block_size; ++b)
  {
    const double full = solution(b).get_full_size();
    storedSolution += solution(b).get_number_elements();
    storedRHS += rhs(b).get_number_elements();
    fullMixed += full;
    if (b) fullTrial += full;
    std::cout << "comparison block=" << system.names[b] << " solution_ranks=" << solution(b).ranks_string()
              << " scalars=" << solution(b).get_number_elements() << " full=" << full << '\n';
    for (size_t j = 0; j < solution.block_size; ++j)
      storedMatrix += system.matrix({b,j}).get_number_elements();
  }
  metrics["tt_matrix_scalars"] = storedMatrix;
  metrics["tt_solution_scalars"] = storedSolution;
  metrics["tt_rhs_scalars"] = storedRHS;
  metrics["full_trial_dofs"] = fullTrial;
  metrics["full_mixed_dofs"] = fullMixed;
  metrics["field_dofs"] = solution(1).get_full_size();
  metrics["tt_field_scalars"] = solution(1).get_number_elements();
  metrics["field_compression"] = solution(1).get_full_size()/solution(1).get_number_elements();
  metrics["mixed_vector_compression"] = fullMixed/storedSolution;
  const double nw = solution(0).get_full_size();
  metrics["dense_active_operator_scalars"] = nw*nw+2*nw*fullTrial;
  metrics["dense_operator_compression"] = metrics["dense_active_operator_scalars"]/storedMatrix;

  std::vector<Axis> samples;
  boba::Array<host_matrix,4> evaluation;
  Sizes sampleSizes;
  for (size_t d = 0; d < 4; ++d)
  {
    samples.emplace_back(vertices[d],problem.p,problem.dp,problem.sampleDegree);
    evaluation[d] = hostMatrix(samples[d].field);
    sampleSizes[d] = samples[d].points.size();
  }
  auto sampled = boba::make_ttm_from_matrices<4,host_space,double>(evaluation)*solution(1);
  auto full = sampled.decompress();
  auto values = full.const_view();
  std::ofstream out(problem.outputPrefix+"_samples.csv");
  if (!out) throw std::runtime_error("Cannot write comparison samples");
  out << std::setprecision(17) << "x,t,ux,uz,weight,f,exact\n";
  size_t count = 1;
  for (size_t d = 0; d < 4; ++d) count *= sampleSizes[d];
  double norm2 = 0, error2 = 0;
  for (size_t k = 0; k < count; ++k)
  {
    size_t ordinal = k;
    double weight = 1, exact = 1;
    for (size_t d = 0; d < 4; ++d)
    {
      const size_t i = ordinal%sampleSizes[d]; ordinal /= sampleSizes[d];
      const double point = samples[d].points(i);
      out << point << ',';
      weight *= samples[d].weights(i);
      exact *= problem.bubble(point,vertices[d].front(),vertices[d].back());
    }
    const double value = values(k);
    out << weight << ',' << value << ',' << exact << '\n';
    norm2 += weight*exact*exact;
    error2 += weight*(value-exact)*(value-exact);
  }
  metrics["relative_exact_l2_error"] = std::sqrt(error2/norm2);
  const bool passed = std::isfinite(metrics["mixed_residual"]) && metrics["mixed_residual"] <= problem.tolerance;
  metrics["solve_passed"] = passed ? 1 : 0;
  writeMetrics(problem.outputPrefix+"_metrics.json",metrics);
  std::cout << std::setprecision(12) << "tensor total_seconds=" << totalSeconds
            << " relative_exact_l2_error=" << metrics["relative_exact_l2_error"]
            << " field_compression=" << metrics["field_compression"] << '\n';
  return passed ? 0 : 2; // preserve failed-run diagnostics for the comparison table
}
} // namespace PlanarMITLBoba
