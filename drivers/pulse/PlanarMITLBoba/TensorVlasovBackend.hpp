#pragma once
// The TT Vlasov backend for the production planar MITL driver (VlasovSpaceTimeFormulation::setVlasovBackend): each
// fixed-point substep samples the current Maxwell fields on the slab's (x, t) quadrature grid, solves the slab by
// TensorVlasovSlab (explicit S + block AMEn, warm-started), and hands the moments at the configuration cubature
// points to the Maxwell formulation as its current sources.  One MPI rank; a tensor-product phase-space mesh.
#include "TensorVlasovSlab.hpp"

#include "BasisCache.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "Function.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "TensorVlasovMomentSampling1D2V.h"
#include "VlasovSpaceTimeFormulation.h"

#include <functional>
#include <memory>
#include <set>

namespace PlanarMITLBoba
{
struct TensorVlasovBackendState
{
  PlanarMITL::ComparisonProblem problem;   // TT settings; c, qOverM, inflow shape set by the driver
  std::function<double()> inflowDensity;   // the cathode Maxwellian's density for the current substep
  double fieldTolerance = 1e-10;           // relative SVD tolerance of the sampled (x, t) fields
  std::unique_ptr<TensorVlasovSlab> slab;  // the current slab
  double slabT0 = std::numeric_limits<double>::quiet_NaN();
  int slabOrdinal = -1;
  double seconds = 0;                      // total backend time
};

namespace BackendDetail
{
/// sorted distinct values of coordinate `component` over the vertices of the active cells of a mesh
inline std::vector<double> meshCoordinates(Camellia::MeshPtr mesh, int component)
{
  std::vector<double> all;
  for (auto cellID : mesh->getActiveCellIDsGlobal())
  {
    const auto nodes = mesh->physicalCellNodesForCell(cellID);   // (1, nodes, dim)
    for (int n = 0; n < nodes.dimension(1); ++n) all.push_back(nodes(0, n, component));
  }
  std::sort(all.begin(), all.end());
  std::vector<double> out;
  const double scale = all.empty() ? 1.0 : std::max(std::abs(all.front()), std::abs(all.back())) + 1e-300;
  for (double v : all) if (out.empty() || v - out.back() > 1e-12 * scale) out.push_back(v);
  return out;
}
} // namespace BackendDetail

/// the moments of the slab's f at the configuration cubature points (cells in cellIDsInPartition order), in the layout
/// of computeVlasovMomentsByPhaseSpaceCell1D2V
inline Camellia::TensorVlasovMoments1D2V tensorMomentsAtConfigurationCubature(const TensorVlasovSlab &slab,
                                                                                Camellia::MeshPtr configurationMesh,
                                                                                bool relativistic)
{
  const auto &cellSet = configurationMesh->cellIDsInPartition();
  std::vector<double> x, t;
  int pointsPerCell = -1;
  for (auto cellID : cellSet)
  {
    auto basisCache = Camellia::BasisCache::basisCacheForCell(configurationMesh, cellID);
    const auto &points = basisCache->getPhysicalCubaturePointsFC();   // (1, P, 2)
    if (pointsPerCell < 0) pointsPerCell = points.dimension(1);
    if (points.dimension(1) != pointsPerCell) throw std::runtime_error("TT backend: configuration cells differ in cubature");
    for (int p = 0; p < pointsPerCell; ++p) { x.push_back(points(0, p, 0)); t.push_back(points(0, p, 1)); }
  }
  const auto m = slab.moments(Eigen::Map<Eigen::VectorXd>(x.data(), Eigen::Index(x.size())),
                              Eigen::Map<Eigen::VectorXd>(t.data(), Eigen::Index(t.size())), relativistic);
  Camellia::TensorVlasovMoments1D2V out;
  out.numConfigurationCells = int(cellSet.size());
  out.numPointsPerConfigurationCell = pointsPerCell;
  out.numVelocityPoints = 0;
  auto toStd = [](const Eigen::VectorXd &v) { return std::vector<double>(v.data(), v.data() + v.size()); };
  out.moments = Camellia::TensorPlanar1D2VSpeciesMoments(toStd(m.density), toStd(m.x), toStd(m.z));
  return out;
}

/// the backend function: (formulation, substep) -> relative L2 change in f
inline std::function<double(Camellia::VlasovSpaceTimeFormulation &, int)>
makeTensorVlasovBackend(std::shared_ptr<TensorVlasovBackendState> state)
{
  return [state](Camellia::VlasovSpaceTimeFormulation &form, int substep) -> double {
    using namespace Camellia;
    const auto start = std::chrono::steady_clock::now();
    if (form.numSpecies() != 1) throw std::invalid_argument("TT Vlasov backend: one species");
    auto configurationMesh = form.configurationSpaceMesh();
    auto maxwell = Teuchos::rcp_dynamic_cast<ConfigurationSpaceFormulationMaxwell1D2V>(form.configurationSpaceFormulation());
    if (maxwell == Teuchos::null) throw std::invalid_argument("TT Vlasov backend: 1D2V Maxwell configuration formulation");
    // a new slab: the axes from the phase-space mesh (coordinates x, t, u_x, u_z); t = t_n data from the last slab
    if (!state->slab || form.t0() != state->slabT0)
    {
      std::vector<std::vector<double>> vertices;
      for (int d = 0; d < 4; ++d) vertices.push_back(BackendDetail::meshCoordinates(form.vlasovMesh(), d));
      auto close = [](const std::vector<double> &a, const std::vector<double> &b) {
        if (a.size() != b.size()) return false;
        const double scale = std::max(std::abs(a.front()), std::abs(a.back())) + 1e-300;
        for (size_t i = 0; i < a.size(); ++i) if (std::abs(a[i] - b[i]) > 1e-10 * scale) return false;
        return true;
      };
      if (!close(vertices[0], BackendDetail::meshCoordinates(configurationMesh, 0)) || !close(vertices[1], BackendDetail::meshCoordinates(configurationMesh, 1)))
      {
        auto show = [](const std::vector<double> &v) {
          std::ostringstream o; o << v.size() << " [" << v.front() << " .. " << v.back() << "]:";
          for (size_t i = 0; i < std::min<size_t>(v.size(), 6); ++i) o << " " << v[i];
          return o.str();
        };
        std::ostringstream o;
        for (int d = 0; d < 4; ++d) o << "\n  phase " << d << ": " << show(vertices[size_t(d)]);
        for (int d = 0; d < 2; ++d) o << "\n  config " << d << ": " << show(BackendDetail::meshCoordinates(configurationMesh, d));
        throw std::runtime_error("TT Vlasov backend: phase-space and configuration (x, t) vertices differ" + o.str());
      }
      size_t cells = 1;
      for (const auto &v : vertices) cells *= v.size() - 1;
      if (form.vlasovMesh()->numActiveElements() != cells)
        throw std::runtime_error("TT Vlasov backend: the phase-space mesh is not the tensor product of its axes ("
                                 + std::to_string(form.vlasovMesh()->numActiveElements()) + " cells, product " + std::to_string(cells) + ")");
      auto next = std::make_unique<TensorVlasovSlab>(state->problem, vertices);
      if (state->slab && state->slab->solved()) next->setInitialTrace(*state->slab);
      state->slab = std::move(next);
      state->slabT0 = form.t0();
      ++state->slabOrdinal;
      std::cout << "TT Vlasov backend: slab " << state->slabOrdinal << " t in [" << form.t0() << ", " << form.t1() << "], "
                << vertices[0].size() - 1 << " x " << vertices[1].size() - 1 << " x " << vertices[2].size() - 1 << " x "
                << vertices[3].size() - 1 << " cells" << std::endl;
    }
    TensorVlasovSlab &slab = *state->slab;
    // fields at the slab's (x, t) quadrature grid
    auto solution = form.configurationSpaceSolution();
    FunctionPtr E = Function::solution(maxwell->E_fieldVar(0), solution);
    FunctionPtr B = Function::solution(maxwell->B_fieldVar(1), solution);
    const auto &xp = slab.xPoints(), &tp = slab.tPoints();
    Eigen::MatrixXd Ex(xp.size(), tp.size()), By(xp.size(), tp.size());
    for (Eigen::Index j = 0; j < tp.size(); ++j)
      for (Eigen::Index i = 0; i < xp.size(); ++i)
      {
        Ex(i, j) = E->evaluate(configurationMesh, xp(i), tp(j));
        By(i, j) = B->evaluate(configurationMesh, xp(i), tp(j));
      }
    slab.setFields(Ex, By, state->fieldTolerance);
    slab.setInflowDensity(state->inflowDensity());
    const double change = slab.solve();
    // the current sources for the Maxwell solve that follows
    const auto moments = tensorMomentsAtConfigurationCubature(slab, configurationMesh, form.useRelativisticCurrentMoments());
    const int entries = moments.numConfigurationCells * moments.numPointsPerConfigurationCell;
    std::vector<double> n0Values(size_t(entries), 0.0);
    auto data = TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(moments.numConfigurationCells, moments.numPointsPerConfigurationCell,
                                                                  {form.q(0)}, maxwell->eps0(), {moments.moments}, n0Values);
    maxwell->setCurrentSources(data);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    state->seconds += seconds;
    std::cout << "TT Vlasov backend: slab " << state->slabOrdinal << " substep " << substep + 1 << " field ranks "
              << slab.fieldRanks()[0] << "," << slab.fieldRanks()[1] << " |E| max " << Ex.cwiseAbs().maxCoeff() << " |B| max "
              << By.cwiseAbs().maxCoeff() << " n_emit " << state->inflowDensity() << " rel_change " << change << " seconds "
              << seconds << std::endl;
    return change;
  };
}
} // namespace PlanarMITLBoba
