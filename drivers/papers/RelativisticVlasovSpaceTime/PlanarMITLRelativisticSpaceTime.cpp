//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaMemoryTools.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "CubatureFactory.h"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "DoerflerRefinementStrategy.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MonomialFunctions.h"
#include "ParameterFunction.h"
#include "PhaseSpaceVisualization.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TensorBilinearForm.h"
#include "TensorLocalSchurComplement.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "TensorMaxwellRHSMomentData.h"
#include "TensorMesh.h"
#include "TensorOperatorData.h"
#include "TensorOperatorFactory.h"
#include "TensorPlanarMITLMomentComparisonAdapter.h"
#include "TensorVar.h"
#include "TensorVarDiscretization.h"
#include "TensorVlasovMomentSampling1D2V.h"
#include "TrigFunctions.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

namespace
{
  bool gEnforcePositivity = false;
  double gProximalGalerkinAlpha0 = 1.0;
  double gProximalGalerkinAlphaMultiplier = 1.0;

  using Phase0Clock = std::chrono::steady_clock;
  using Phase0TimePoint = Phase0Clock::time_point;
  using Phase0Fields = std::vector<std::pair<std::string,std::string>>;

  std::string phase0DoubleString(double value)
  {
    std::ostringstream os;
    os << std::setprecision(6) << std::scientific << value;
    return os.str();
  }

  class Phase0Instrumentation
  {
    bool _enabled;
    int _rank;
    std::ofstream _file;
  public:
    Phase0Instrumentation(bool enabled, int rank, const std::string &timingFilePath)
    : _enabled(enabled), _rank(rank)
    {
      if (_enabled && (_rank == 0) && (timingFilePath != ""))
      {
        _file.open(timingFilePath.c_str(), std::ios::out | std::ios::app);
        if (!_file.is_open())
        {
          std::cout << "WARNING: unable to open Phase 0 timing file '" << timingFilePath << "'; timings will only be written to stdout.\n";
        }
      }
    }

    bool enabled() const { return _enabled; }
    Phase0TimePoint now() const { return Phase0Clock::now(); }

    double elapsedSince(const Phase0TimePoint &start) const
    {
      return std::chrono::duration<double>(Phase0Clock::now() - start).count();
    }

    void logDuration(const std::string &event, double seconds, const Phase0Fields &fields = Phase0Fields())
    {
      if (!_enabled || (_rank != 0)) return;

      std::ostringstream line;
      line << "PHASE0_TIMING event=" << event;
      line << " seconds=" << phase0DoubleString(seconds);
      for (const auto &field : fields)
      {
        line << " " << field.first << "=" << field.second;
      }
      line << "\n";

      std::cout << line.str();
      if (_file.is_open())
      {
        _file << line.str();
        _file.flush();
      }
    }
  };

  Phase0Fields phase0MeshStats(const Teuchos::RCP<VlasovSpaceTimeFormulation> &form)
  {
    Phase0Fields fields;
    if (form != Teuchos::null)
    {
      auto vlasovMesh = form->vlasovMesh();
      auto configMesh = form->configurationSpaceMesh();
      if (vlasovMesh != Teuchos::null)
      {
        fields.push_back({"vlasov_dofs", std::to_string(vlasovMesh->numGlobalDofs())});
        fields.push_back({"vlasov_elements", std::to_string(vlasovMesh->numActiveElements())});
      }
      if (configMesh != Teuchos::null)
      {
        fields.push_back({"config_dofs", std::to_string(configMesh->numGlobalDofs())});
        fields.push_back({"config_elements", std::to_string(configMesh->numActiveElements())});
      }
    }
    return fields;
  }

  std::string phase0TensorGraphSchurBackendName(const std::string &backend)
  {
    if (backend == "off") return "off";
    if (backend == "assembled_crs") return "assembled_crs_graph_schur";
    if (backend == "matrix_free") return "matrix_free_graph_schur";
    return "invalid";
  }

  bool phase0TensorGraphSchurBackendIsValid(const std::string &backend)
  {
    return (backend == "off") || (backend == "assembled_crs") || (backend == "matrix_free");
  }

  bool phase8MomentBackendIsValid(const std::string &backend)
  {
    return (backend == "legacy") ||
           (backend == "tensor_native") ||
           (backend == "dry_run_compare") ||
           (backend == "compare");
  }

  bool phase8MomentBackendRequestsDryRunComparison(const std::string &backend)
  {
    return (backend == "tensor_native") ||
           (backend == "dry_run_compare") ||
           (backend == "compare");
  }

  //! Writes the electron density n_e = int f du (and Jx and E_x) on a dense, uniform (pointsPerDim x pointsPerDim) grid of
  //! points on each configuration (x,t) cell, as CSV gathered to rank 0.  The density is the exact velocity moment of the
  //! polynomial f at each point (not an interpolant), so plotting it at fine resolution shows the actual discrete solution.
  //! Collective.  Columns: cell, x, t, n_e, Jx, E_x; rows are ordered cell, then t-major, x-minor reference point.  The
  //! header records pointsPerDim and the polynomial degree of f.
  void writeDensityDiagnostic(const std::string &path, Teuchos::RCP<VlasovSpaceTimeFormulation> form,
                              int speciesOrdinal, int pointsPerDim, int velocityQuadratureDegree, double lightSpeed, double q)
  {
    MeshPtr configurationMesh = form->configurationSpaceMesh();
    const auto &cellIDSet = configurationMesh->cellIDsInPartition();
    std::vector<GlobalIndexType> cellIDs(cellIDSet.begin(), cellIDSet.end());

    const int pointCount = pointsPerDim * pointsPerDim;
    FieldContainer<double> refPoints(pointCount, 2);
    for (int i=0; i<pointsPerDim; i++) // t
    {
      for (int j=0; j<pointsPerDim; j++) // x
      {
        refPoints(i * pointsPerDim + j, 0) = -1.0 + 2.0 * j / (pointsPerDim - 1);
        refPoints(i * pointsPerDim + j, 1) = -1.0 + 2.0 * i / (pointsPerDim - 1);
      }
    }
    auto moments = computeVlasovMomentsByPhaseSpaceCell1D2V(cellIDs, configurationMesh, form->vlasovSolution(), form->f(speciesOrdinal),
                                                            velocityQuadratureDegree, true, lightSpeed, refPoints);

    // Polynomial degree of f (max over cells and directions): on each configuration cell, n_e is a polynomial of at most this
    // degree in each of x and t (configuration cells are everywhere at least as fine as the phase-space cells over them), so
    // plot_density.py can recover it exactly from the samples when pointsPerDim > degree.
    MeshPtr phaseSpaceMesh = form->vlasovMesh();
    const int fID = form->f(speciesOrdinal)->ID();
    int fDegree = 0;
    for (GlobalIndexType phaseSpaceCellID : phaseSpaceMesh->cellIDsInPartition())
    {
      fDegree = std::max(fDegree, phaseSpaceMesh->getElementType(phaseSpaceCellID)->trialOrderPtr->getBasis(fID)->getDegree());
    }
    fDegree = MPIWrapper::max(phaseSpaceMesh->Comm(), fDegree);

    FunctionPtr E = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
    const int columnCount = 6;
    std::vector<double> myRows;
    myRows.reserve(cellIDs.size() * pointCount * columnCount);
    for (int cellOrdinal=0; cellOrdinal<static_cast<int>(cellIDs.size()); cellOrdinal++)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(configurationMesh, cellIDs[cellOrdinal]);
      basisCache->setRefCellPointsFC(refPoints);
      const auto &physicalPoints = basisCache->getPhysicalCubaturePointsFC();
      FieldContainer<double> EValues(1, pointCount);
      E->values(EValues, basisCache);
      for (int p=0; p<pointCount; p++)
      {
        const int entry = cellOrdinal * pointCount + p;
        myRows.insert(myRows.end(), {double(cellIDs[cellOrdinal]), physicalPoints(0,p,0), physicalPoints(0,p,1),
                                     moments.moments._density[entry], q * moments.moments._xProperVelocityMoment[entry], EValues(0,p)});
      }
    }
    std::vector<double> allRows;
    std::vector<int> offsets;
    MPIWrapper::allGatherVariable(configurationMesh->Comm(), allRows, myRows, offsets);

    if (configurationMesh->Comm()->getRank() != 0) return;
    std::ofstream out(path);
    TEUCHOS_TEST_FOR_EXCEPTION(!out, std::runtime_error, "could not open density diagnostic file " + path);
    out << "# pointsPerDim=" << pointsPerDim << " degree=" << fDegree << "\n";
    out << "cell,x,t,n_e,Jx,E_x\n";
    out << std::setprecision(12);
    for (size_t row=0; row<allRows.size() / columnCount; row++)
    {
      const double *values = &allRows[row * columnCount];
      out << static_cast<long>(values[0]);
      for (int column=1; column<columnCount; column++) out << "," << values[column];
      out << "\n";
    }
  }

  TensorMaxwellRHSMomentData phase8BuildLegacyMomentReferenceData(const std::vector<GlobalIndexType> &configurationCellIDs,
                                                                  MeshPtr configurationMesh,
                                                                  FunctionPtr densityFunction,
                                                                  FunctionPtr JxFunction,
                                                                  FunctionPtr JzFunction,
                                                                  double q,
                                                                  double eps0,
                                                                  double n0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(configurationMesh == Teuchos::null, std::invalid_argument,
                               "Phase 8 legacy moment reference packing requires a non-null configuration mesh.");
    TEUCHOS_TEST_FOR_EXCEPTION(densityFunction == Teuchos::null, std::invalid_argument,
                               "Phase 8 legacy moment reference packing requires a non-null density function.");
    TEUCHOS_TEST_FOR_EXCEPTION(JxFunction == Teuchos::null, std::invalid_argument,
                               "Phase 8 legacy moment reference packing requires a non-null Jx function.");
    TEUCHOS_TEST_FOR_EXCEPTION(JzFunction == Teuchos::null, std::invalid_argument,
                               "Phase 8 legacy moment reference packing requires a non-null Jz function.");
    TEUCHOS_TEST_FOR_EXCEPTION(eps0 == 0.0, std::invalid_argument,
                               "Phase 8 legacy moment reference packing requires nonzero eps0.");

    TensorMaxwellRHSMomentData referenceData;
    int expectedPointCount = -1;
    for (int cellOrdinal=0; cellOrdinal<static_cast<int>(configurationCellIDs.size()); cellOrdinal++)
    {
      const auto cellID = configurationCellIDs[cellOrdinal];
      auto basisCache = BasisCache::basisCacheForCell(configurationMesh, cellID);
      const auto &points = basisCache->getPhysicalCubaturePointsFC();
      const int pointCount = points.dimension(1);
      if (expectedPointCount < 0)
      {
        expectedPointCount = pointCount;
        referenceData = TensorMaxwellRHSMomentData(static_cast<int>(configurationCellIDs.size()), expectedPointCount);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(pointCount != expectedPointCount, std::invalid_argument,
                                 "Phase 8 legacy moment reference packing requires a uniform configuration point count per cell.");

      FieldContainer<double> densityValues(1, pointCount);
      FieldContainer<double> JxValues(1, pointCount);
      FieldContainer<double> JzValues(1, pointCount);
      densityFunction->values(densityValues, basisCache);
      JxFunction->values(JxValues, basisCache);
      JzFunction->values(JzValues, basisCache);

      for (int pointOrdinal=0; pointOrdinal<pointCount; pointOrdinal++)
      {
        referenceData.rhoRHS(cellOrdinal, pointOrdinal) = q / eps0 * (densityValues(0, pointOrdinal) - n0);
        referenceData.Jx(cellOrdinal, pointOrdinal)     = JxValues(0, pointOrdinal);
        referenceData.Jz(cellOrdinal, pointOrdinal)     = JzValues(0, pointOrdinal);
      }
    }
    if (expectedPointCount < 0)
    {
      referenceData = TensorMaxwellRHSMomentData(0, 0);
    }
    return referenceData;
  }

  std::vector<double> phase0UniformVertices(double left, double right, int elementCount)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(elementCount <= 0, std::invalid_argument,
                               "Phase 0 tensor graph-norm Schur microbenchmark requires positive element counts.");
    std::vector<double> vertices(elementCount + 1, left);
    for (int vertexOrdinal=0; vertexOrdinal<=elementCount; vertexOrdinal++)
    {
      vertices[vertexOrdinal] = left + (right - left) * static_cast<double>(vertexOrdinal) / static_cast<double>(elementCount);
    }
    return vertices;
  }

  std::vector<ActiveLeafSet1D> phase0RootLeafSets(int elementCount)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(elementCount <= 0, std::invalid_argument,
                               "Phase 0 tensor graph-norm Schur microbenchmark requires positive element counts.");
    return std::vector<ActiveLeafSet1D>(elementCount, ActiveLeafSet1D({{0,0}}));
  }

  TensorVarPtr phase0TensorFieldVar1D1T2V(const std::string &name)
  {
    std::vector<VarPtr> componentVars;
    componentVars.push_back(Teuchos::rcp(new Var(2000, 0, name + "_x",  Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
    componentVars.push_back(Teuchos::rcp(new Var(2001, 0, name + "_t",  Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
    componentVars.push_back(Teuchos::rcp(new Var(2002, 0, name + "_ux", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
    componentVars.push_back(Teuchos::rcp(new Var(2003, 0, name + "_uy", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
    return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, componentVars));
  }

  TensorMeshPtr makePhase0TensorGraphSchurMesh1D1T2V(double x0,
                                                     double xWidth,
                                                     double t0,
                                                     double tWidth,
                                                     double uxMin,
                                                     double uxMax,
                                                     double uyMin,
                                                     double uyMax,
                                                     int xElementCount,
                                                     int tElementCount,
                                                     int uxElementCount,
                                                     int uyElementCount,
                                                     int componentPolyOrder,
                                                     int testSpaceEnrichment)
  {
    auto component0 = Teuchos::rcp(new ComponentMeshTopology1D(phase0UniformVertices(x0, x0 + xWidth, xElementCount), phase0RootLeafSets(xElementCount), false, componentPolyOrder));
    auto component1 = Teuchos::rcp(new ComponentMeshTopology1D(phase0UniformVertices(t0, t0 + tWidth, tElementCount), phase0RootLeafSets(tElementCount), false, componentPolyOrder));
    auto component2 = Teuchos::rcp(new ComponentMeshTopology1D(phase0UniformVertices(uxMin, uxMax, uxElementCount), phase0RootLeafSets(uxElementCount), false, componentPolyOrder));
    auto component3 = Teuchos::rcp(new ComponentMeshTopology1D(phase0UniformVertices(uyMin, uyMax, uyElementCount), phase0RootLeafSets(uyElementCount), false, componentPolyOrder));
    auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2, component3}));
    auto factory = TensorVarFactory::factory();
    auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), componentPolyOrder, testSpaceEnrichment));
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(phase0TensorFieldVar1D1T2V("f"), Teuchos::null, std::vector<int>{0,0,0,0})));
    tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(phase0TensorFieldVar1D1T2V("w"), Teuchos::null, std::vector<int>{0,0,0,0})));
    return tensorMesh;
  }

  TensorBilinearFormPtr makePhase0TensorGraphSchurForm1D1T2V(double cScaled)
  {
    auto form = Teuchos::rcp(new TensorBilinearForm("phase0-tensor-relativistic-graph-schur"));
    auto f = phase0TensorFieldVar1D1T2V("f");
    auto w = phase0TensorFieldVar1D1T2V("w");
    auto ux = TensorFunction::componentCoordinate(2);
    auto uy = TensorFunction::componentCoordinate(3);
    auto gamma = Camellia::sqrt(TensorFunction::constant(1.0) + TensorFunction::constant(1.0 / (cScaled * cScaled)) * (ux * ux + uy * uy));
    auto vx = TensorFunction::constant(-1.0) * ux * reciprocal(gamma);
    auto vy = TensorFunction::constant(0.5) * uy * reciprocal(gamma);
    form->addTerm(TensorBilinearTerm(f, w, TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 0), vx));
    form->addTerm(TensorBilinearTerm(f, w, TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 1), TensorFunction::constant(-1.0)));
    form->addTerm(TensorBilinearTerm(f, w, TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 2), TensorFunction::constant(-0.5) * (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1))));
    form->addTerm(TensorBilinearTerm(f, w, TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 3), vy + TensorFunction::constant(-0.75) * TensorFunction::componentCoordinate(1)));
    return form;
  }

  double phase0VectorL2Norm(const std::vector<double> &values)
  {
    double normSquared = 0.0;
    for (double value : values) normSquared += value * value;
    return std::sqrt(normSquared);
  }

  void runPhase0TensorGraphSchurMicroApply(const std::string &backend,
                                           double cScaled,
                                           double x0,
                                           double xWidth,
                                           double t0,
                                           double tWidth,
                                           double uxMin,
                                           double uxMax,
                                           double uyMin,
                                           double uyMax,
                                           double l2Weight,
                                           int microXElements,
                                           int microTElements,
                                           int microUxElements,
                                           int microUyElements,
                                           int microComponentPolyOrder,
                                           int microTestSpaceEnrichment,
                                           Phase0Instrumentation &phase0)
  {
    if (backend == "off") return;
    TEUCHOS_TEST_FOR_EXCEPTION(!phase0TensorGraphSchurBackendIsValid(backend), std::invalid_argument,
                               "Unsupported Phase 0 tensor graph-norm Schur backend option: " + backend);

    auto tensorMesh = makePhase0TensorGraphSchurMesh1D1T2V(x0, xWidth, t0, tWidth, uxMin, uxMax, uyMin, uyMax,
                                                           microXElements,
                                                           microTElements,
                                                           microUxElements,
                                                           microUyElements,
                                                           microComponentPolyOrder,
                                                           microTestSpaceEnrichment);
    auto dofLayout = tensorMesh->createDofLayout();
    auto tensorForm = makePhase0TensorGraphSchurForm1D1T2V(cScaled);
    TensorOperatorData operatorData(tensorMesh, dofLayout);
    const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "f");
    std::vector<double> input(trialDofCount, 0.0);
    for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = 0.1 + 0.025 * static_cast<double>((gid % 17) + 1);

    const auto applyStart = phase0.now();
    std::vector<double> output;
    if (backend == "matrix_free")
    {
      MatrixFreeGraphNormSchurTensorOperatorFactory graphFactory(l2Weight);
      auto graphOp = graphFactory.build(tensorMesh,
                                        dofLayout,
                                        tensorForm,
                                        TENSOR_TRIAL_SPACE,
                                        "f",
                                        TENSOR_TRIAL_SPACE,
                                        "f");
      graphOp->apply(input, output);
    }
    else
    {
      auto graphMatrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                                   dofLayout,
                                                                                                   tensorForm,
                                                                                                   "f",
                                                                                                   "w",
                                                                                                   l2Weight);
      TVector<double> x(graphMatrix->getDomainMap(), 1);
      TVector<double> y(graphMatrix->getRangeMap(), 1);
      auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
      for (size_t localOrdinal=0; localOrdinal<graphMatrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[graphMatrix->getDomainMap()->getGlobalElement(localOrdinal)];
      graphMatrix->apply(x, y);
      output.assign(trialDofCount, 0.0);
      auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t localOrdinal=0; localOrdinal<graphMatrix->getRangeMap()->getLocalNumElements(); localOrdinal++) output[graphMatrix->getRangeMap()->getGlobalElement(localOrdinal)] = yHost(localOrdinal,0);
    }

    phase0.logDuration("tensor_graph_schur_micro_apply",
                       phase0.elapsedSince(applyStart),
                       {{"backend", phase0TensorGraphSchurBackendName(backend)},
                        {"micro_x_elements", std::to_string(microXElements)},
                        {"micro_t_elements", std::to_string(microTElements)},
                        {"micro_ux_elements", std::to_string(microUxElements)},
                        {"micro_uy_elements", std::to_string(microUyElements)},
                        {"micro_component_p", std::to_string(microComponentPolyOrder)},
                        {"micro_test_enrichment", std::to_string(microTestSpaceEnrichment)},
                        {"trial_dofs", std::to_string(trialDofCount)},
                        {"output_norm", phase0DoubleString(phase0VectorL2Norm(output))}});
  }
}

class Maxwellian1T2V : public TFunction<double>
{
  Teuchos::RCP<ParameterFunction> _ux, _uy, _sigma, _n;
public:
  Maxwellian1T2V(double ux, double uy, double sigma, double n)
  :
  _ux(ParameterFunction::parameterFunction(ux,"ux")),
  _uy(ParameterFunction::parameterFunction(uy,"uy")),
  _sigma(ParameterFunction::parameterFunction(sigma,"sigma")),
  _n(ParameterFunction::parameterFunction(n,"n"))
  {}
  
  Maxwellian1T2V(Teuchos::RCP<ParameterFunction> ux,
                 Teuchos::RCP<ParameterFunction> uy,
                 Teuchos::RCP<ParameterFunction> sigma,
                 Teuchos::RCP<ParameterFunction> n)
  :
  _ux(ux),
  _uy(uy),
  _sigma(sigma),
  _n(n)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double n     =     _n->getConstantValue();
    const double sigma = _sigma->getConstantValue();
    const double ux    =    _ux->getConstantValue();
    const double uy    =    _uy->getConstantValue();
    
    const double weight = n / (2. * M_PI * sigma * sigma); // 2V normalization: the integral over (ux, uy) is n
//    std::cout << "weight: " << weight << std::endl;
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,2); // vx is the third dimension for 1D1T2V
        const double vy = phaseSpacePoints(cellOrdinal,pointOrdinal,3); // vy is the fourth dimension for 1D1T2V
        const double exponent = - 0.5 * ((vx - ux) * (vx - ux) + (vy - uy) * (vy - uy)) / (sigma * sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dvx()
  {
    auto vx = MonomialFunction::xi_n(2, 1);
    FunctionPtr f_M = Teuchos::rcp( new Maxwellian1T2V(_ux, _uy,_sigma,_n));
    FunctionPtr ux    = _ux;
    FunctionPtr sigma = _sigma;
    return (ux - vx) / (sigma * sigma) * f_M;
  }
  
  TFunctionPtr<double> dvy()
  {
    auto vy = MonomialFunction::xi_n(3, 1);
    FunctionPtr f_M = Teuchos::rcp( new Maxwellian1T2V(_ux, _uy,_sigma,_n));
    FunctionPtr uy    = _uy;
    FunctionPtr sigma = _sigma;
    return (uy - vy) / (sigma * sigma) * f_M;
  }
  
  TFunctionPtr<double> dz() override
  {
    return dvx();
  }
  
  TFunctionPtr<double> dt() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 3) // 1-based
    {
      return dvx();
    }
    else if (i == 4) // 1-based
    {
      return dvy();
    }
    else
    {
      return Function::zero();
    }
  }
  
  void set_n(double nValue)
  {
    _n->setValue(nValue);
  }
  
  double get_n() const
  {
    return _n->getConstantValue();
  }
  
  void set_sigma(double sigmaValue)
  {
    _sigma->setValue(sigmaValue);
  }
  
  void set_ux(double uxValue)
  {
    _ux->setValue(uxValue);
  }
  
  void set_uy(double uyValue)
  {
    _uy->setValue(uyValue);
  }
};

//! Time profile of the emitted density for space-charge-limited emission:
//!   n(t) = sum_j a_j B_j(s),  s = (t - t0)/(t1 - t0),
//! with B_j the degree-m Bernstein polynomials on the current time slab [t0, t1] (set per slab with setSlab()).  The
//! Bernstein basis is nonnegative and sums to one, so a_j >= 0 gives n >= 0; a_0 and a_m are the slab-start and slab-end
//! values.  Multiplies the inflow Maxwellian, whose amplitude is then 1.  Time is the second coordinate in 1D1T2V.
class EmissionTimeProfile : public TFunction<double>
{
  std::vector<double> _coefficients;
  double _t0 = 0, _t1 = 1;
public:
  explicit EmissionTimeProfile(int degree, double n) : _coefficients(degree + 1, n) {}

  static double bernstein(int m, int j, double s)
  {
    double binomial = 1;
    for (int i=1; i<=j; i++) binomial = binomial * (m - j + i) / i;
    return binomial * std::pow(s, j) * std::pow(1.0 - s, m - j);
  }

  void setSlab(double t0, double t1) { _t0 = t0; _t1 = t1; }
  int degree() const { return int(_coefficients.size()) - 1; }
  const std::vector<double> &coefficients() const { return _coefficients; }
  void setCoefficients(const std::vector<double> &a) { _coefficients = a; }
  double mean() const
  {
    double sum = 0;
    for (double a : _coefficients) sum += a;
    return sum / _coefficients.size();
  }

  double valueAtS(double s) const
  {
    const int m = degree();
    double value = 0;
    for (int j=0; j<=m; j++) value += _coefficients[j] * bernstein(m, j, s);
    return value;
  }
  double value(double t) const { return valueAtS((t - _t0) / (_t1 - _t0)); }

  //! Re-expands the profile for the next slab, of the same width, as the constant equal to its value at the current
  //! slab's end (the emission at the start of the next slab continues from the end of this one).
  void advanceToNextSlab(double nextT0, double nextT1)
  {
    const double endValue = _coefficients.back();
    std::fill(_coefficients.begin(), _coefficients.end(), endValue);
    setSlab(nextT0, nextT1);
  }

  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & points = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = points.dimension(0);
    const int numPoints = points.dimension(1);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = value(points(cellOrdinal,pointOrdinal,1));
      }
    }
  }
  TFunctionPtr<double> dx() override { return Function::zero(); }
  TFunctionPtr<double> dy() override { return Function::zero(); }
  TFunctionPtr<double> dz() override { return Function::zero(); }
};

class TemporalRamp1D1T : public TFunction<double>
{
  double _rampWidth; // amount of time before the value is 1.0
public:
  TemporalRamp1D1T(double rampWidth)
  :
  _rampWidth(rampWidth)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spaceTimePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = spaceTimePoints.dimension(0);
    const int numPoints = spaceTimePoints.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double t = spaceTimePoints(cellOrdinal,pointOrdinal,1); // t is the second dimension for 1D1T
        double value;
        if (t < _rampWidth)
        {
          value = t / _rampWidth;
        }
        else
        {
          value = 1.0;
        }
        values(cellOrdinal,pointOrdinal) = value;
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    auto heavisideT = Function::heavisideY(_rampWidth); // 1 for y < _rampWidth, 0 otherwise
    return (1.0 / _rampWidth) * heavisideT;
  }
  
  TFunctionPtr<double> dz() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dt() override
  {
    return dy();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 2) // 1-based
    {
      return dt();
    }
    else
    {
      return Function::zero();
    }
  }
};

class ErrorIndicatorFunction : public Function
{
  ErrorIndicatorPtr _errorIndicator;
public:
  ErrorIndicatorFunction(ErrorIndicatorPtr errorIndicator)
  :
  Function("ErrorIndicatorFunction", 0, Function::VARIES_IN_NO_DIMENSION), // constant on each cell
  _errorIndicator(errorIndicator)
  {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    // we assume that measureError() has been called before values() is called
    // we also assume that values() will only be called for MPI-local cells
    
    const auto & errorMeasures = _errorIndicator->localErrorMeasures();
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      const auto &cellID = cellIDs[cellOrdinal];
      auto entry = errorMeasures.find(cellID);
      if (entry == errorMeasures.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found in local errorMeasures");
      }
      const double error = entry->second;
      
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = error;
      }
    }
  }
  
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
  {
    auto underlyingView1 = values.template getUnderlyingView<1>();
    
    const int numCells = underlyingView1.extent_int(0);
    
    const auto & errorMeasures = _errorIndicator->localErrorMeasures();
    
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "underlying view doesn't have the expected length");
    
    auto underlyingViewHost = Kokkos::create_mirror_view(underlyingView1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      const auto &cellID = cellIDs[cellOrdinal];
      auto entry = errorMeasures.find(cellID);
      if (entry == errorMeasures.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found in local errorMeasures");
      }
      const double error = entry->second;
      
      underlyingViewHost(cellOrdinal) = error;
    }
    
    Kokkos::deep_copy(underlyingView1, underlyingViewHost);
  }
};

class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function(0) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
      }
    }
  }
};

void checkMesh( MeshPtr mesh)
{
  const auto & myCellIDs = mesh->cellIDsInPartition();
  
  auto globalDofAssignment = mesh->globalDofAssignment();
  
  const int sideDim = mesh->getDimension() - 1;
  
  for (auto & cellID : myCellIDs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      // CHECK PARITY CONSISTENCY
      auto neighborInfo = cell->getNeighborInfo(sideOrdinal, mesh->getTopology());
      const IndexType & neighborCellID = neighborInfo.first;
      const auto & mySideOrdinalInNeighbor = neighborInfo.second;
      if (neighborCellID != -1)
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        int neighborParity = globalDofAssignment->sideParity(neighborCellID, mySideOrdinalInNeighbor);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != -neighborParity, std::invalid_argument, "inconsistent parities");
      }
      else
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != 1, std::invalid_argument, "incorrect parity for boundary side");
      }
      
      // CHECK SIDE VERTEX CONSISTENCY
      auto vertices = cell->vertices();
      auto cellTopo = cell->topology();
      
      auto sideTopo = cellTopo->getSide(sideOrdinal);
      
      int sideVertexCount = sideTopo->getVertexCount();
      
      int sideIndex = cell->entityIndex(sideDim, sideOrdinal);
      auto sideVertices = mesh->getTopology()->getEntityVertexIndices(sideDim, sideIndex);
      
      for (int v_ord = 0; v_ord < sideVertexCount; v_ord++)
      {
        int v_ord_in_cell = cellTopo->getNodeMap(sideDim, sideOrdinal, v_ord);
        
        int vertexIndexCell = vertices[v_ord_in_cell];
        int vertexIndexSide = sideVertices[v_ord];
        
        TEUCHOS_TEST_FOR_EXCEPTION(vertexIndexCell != vertexIndexSide, std::invalid_argument, "cell and side have inconsistent vertex indices");
      }
    }
  }
}

//! Vertices on [left, right] that resolve a feature at center: a uniform core of elements of width coreElementWidth
//! covering [center - coreHalfWidth, center + coreHalfWidth] (clipped to the domain), and element widths growing
//! geometrically by growthRatio from the core out to each boundary.  A final element narrower than half its
//! predecessor is merged into it.
std::vector<double> gradedVertices(const double left, const double right, const double center,
                                   const double coreElementWidth, const double coreHalfWidth, const double growthRatio)
{
  TEUCHOS_TEST_FOR_EXCEPTION((coreElementWidth <= 0) || (growthRatio < 1) || (right <= left), std::invalid_argument,
                             "gradedVertices(): invalid arguments");
  const double coreLeft  = std::max(left,  center - coreHalfWidth);
  const double coreRight = std::min(right, center + coreHalfWidth);
  // geometrically graded vertices from start (exclusive) to end (inclusive), first width growthRatio * coreElementWidth
  auto gradedSide = [&](const double start, const double end) {
    std::vector<double> sideVertices;
    const double length = std::abs(end - start);
    const double direction = (end > start) ? 1.0 : -1.0;
    double position = 0, width = coreElementWidth * growthRatio;
    while (length - position > 0)
    {
      const double remaining = length - position;
      if (remaining <= width * 1.5)
      {
        sideVertices.push_back(end); // last element absorbs the remainder (at most 1.5x its nominal width)
        break;
      }
      position += width;
      sideVertices.push_back(start + direction * position);
      width *= growthRatio;
    }
    return sideVertices;
  };
  std::vector<double> vertices;
  auto leftSide = gradedSide(coreLeft, left);
  for (auto it = leftSide.rbegin(); it != leftSide.rend(); it++) vertices.push_back(*it);
  const int numCoreElements = std::max(1, int(std::ceil((coreRight - coreLeft) / coreElementWidth - 1e-10)));
  for (int i=0; i<=numCoreElements; i++)
  {
    vertices.push_back(coreLeft + (coreRight - coreLeft) * double(i) / double(numCoreElements));
  }
  for (double vertex : gradedSide(coreRight, right)) vertices.push_back(vertex);
  return vertices;
}

//! Ensures that value is a vertex, if it lies strictly inside the vertex range: moves the nearer vertex of the interval
//! containing it if that vertex is within a quarter of the interval width, and otherwise inserts value.  (The inflow
//! boundary condition is imposed on whole mesh sides, so a side straddling u_x = 0 on the cathode or anode would receive
//! it on its outflow part too.)
void ensureVertexAt(std::vector<double> &vertices, const double value)
{
  for (size_t i=0; i+1<vertices.size(); i++)
  {
    const double left = vertices[i], right = vertices[i+1];
    if ((value <= left) || (value >= right)) continue;
    const double width = right - left;
    if      ((value - left) < 0.25 * width) vertices[i]   = value;
    else if ((right - value) < 0.25 * width) vertices[i+1] = value;
    else    vertices.insert(vertices.begin() + i + 1, value);
    return;
  }
}

std::vector<double> uniformVertices(const double left, const double width, const int numElements)
{
  std::vector<double> vertices(numElements+1);
  for (int i=0; i<=numElements; i++) vertices[i] = left + double(i) / double(numElements) * width;
  return vertices;
}

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T2V(const double c,
                                                              const double  x0, const double  xWidth,
                                                              const double  t0, const double  tWidth,
                                                              const std::vector<double> &velocitySpaceVertices_x,
                                                              const std::vector<double> &velocitySpaceVertices_y,
                                                              const int xElements,  const int tElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const RelativisticVlasovSpaceTimeFormulation::TestSpaceInnerProduct testIP,
                                                              const double L2WeightGraph,
                                                              const bool usePeriodicBCs, const bool useSerendipity,
                                                              const bool useGradedSpatialMesh,
                                                              const int numUniformRefinements,
                                                              const double L2WeightGraphMeshScale,
                                                              const std::vector<double> &L2WeightGraphDirectionScales,
                                                              const double L2WeightGraphTransportScale,
                                                              const std::vector<double> &L2WeightGraphAccelerationBounds)
{
  Teuchos::RCP<ConfigurationSpaceFormulationMaxwell1D2V> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, eps0) );
  
  MeshTopologyPtr spatialMeshTopo;
  if (useGradedSpatialMesh)
  {
    spatialMeshTopo = MeshFactory::geometricGradedIntervalMeshTopology(x0, x0+xWidth, xElements);
  }
  else
  {
    spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
  }
  
  std::vector<double> temporalVertices(tElements + 1);
  for (int i=0; i<=tElements; i++)
  {
    const double t = double(i) / double(tElements) * tWidth + t0;
    temporalVertices[i] = t;
  }
  
  std::vector<double> velocitySpaceVertices_z; // empty -- 1D2V
  
  Teuchos::RCP<VlasovSpaceTimeFormulation> form;
  
  EBasisFamily basisFamily = useSerendipity ? BASIS_FAMILY_SERENDIPITY : BASIS_FAMILY_HIERARCHICAL;
    
  form = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m, configForm, spatialMeshTopo,
                                                                 temporalVertices,
                                                                 velocitySpaceVertices_x,
                                                                 velocitySpaceVertices_y,
                                                                 velocitySpaceVertices_z,
                                                                 H1Order, delta_k,
                                                                 numUniformRefinements,
                                                                 testIP,
                                                                 L2WeightGraph,
                                                                 basisFamily,
                                                                 0,
                                                                 gEnforcePositivity,
                                                                 L2WeightGraphMeshScale,
                                                                 L2WeightGraphDirectionScales,
                                                                 L2WeightGraphTransportScale,
                                                                 L2WeightGraphAccelerationBounds));
  if (gEnforcePositivity)
  {
    form->setProximalGalerkinParameters(gProximalGalerkinAlpha0, gProximalGalerkinAlphaMultiplier);
  }
  
  return form;
}

std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > bcFunctions(Teuchos::RCP<VlasovSpaceTimeFormulation> form, FunctionPtr f_leftInflow)
{
  std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > functions;
  
  auto inflowTag = form->tagVlasovSpatialInflow();
  
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->spatialComponent(1); // 1-based
  FunctionPtr vx = form->v_i(0);
  
  FunctionPtr tn_leftInflowExact = f_leftInflow * vx; //  * sgn * n_x are also in there, but those are trivially resolved.
            
  const bool weightFluxesBySideParity = false;
  FunctionPtr tn_soln = Function::solution(form->t_n_hat(), form->vlasovSolutionPreviousSubstep(), weightFluxesBySideParity);
  functions.push_back({{tn_leftInflowExact, tn_soln},inflowTag});
  return functions;
}

void refineForBC(SolutionPtr soln, VarPtr bcVar, std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > &bcFunctionsToResolve,
                 int maxElements, double bcResolutionTolerance)
{
  MeshPtr mesh = soln->mesh();
  int rank = mesh->Comm()->getRank();
  int H1Order = mesh->globalDofAssignment()->getInitialH1Order()[0];
  
  for (auto bcPair : bcFunctionsToResolve)
  {
    FunctionPtr bcFxnExact = bcPair.first.first;
    FunctionPtr bcFxnSoln  = bcPair.first.second;
    int bcTag = bcPair.second;
    
    const std::string & tagName = DIRICHLET_SET_TAG_NAME;
    FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
    
    // project onto soln:
    std::map<int,FunctionPtr> bcMap;
    bcMap[bcVar->ID()] = bcFxnExact;
    soln->projectOntoMesh(bcMap);
    
    FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
    int quadratureEnrichment = 4;
    
    auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(mesh, bc_err, quadratureEnrichment);
    errorIndicator->measureError();
    
    double refinementThresholdForInitialData = 0.80;
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
    refStrategy->setMaxP(H1Order);
    
    double bc_norm = (bcFxnExact * bcRestriction)->l2norm(mesh, quadratureEnrichment);
    double relErr = bc_err->l2norm(mesh, quadratureEnrichment) / bc_norm;
    
    if (rank == 0)
    {
      std::cout << "L^2 norm of BC: " << bc_norm << std::endl;
    }
    
    FunctionPtr bcErrIndicator = Teuchos::rcp( new ErrorIndicatorFunction(errorIndicator) );
    
    int numBCRefinements = 0;
    int numElements = mesh->numActiveElements();
    
    while ((relErr > bcResolutionTolerance) && (numElements < maxElements))
    {
      errorIndicator->measureError();
//      bcViz->exportSolution(soln, {cellIDFunction, bcFxnExact, bcFxnSoln * bcRestriction, bcFxnExact * bcRestriction, bc_err * bc_err, bcErrIndicator * bcRestriction}, {"cellID", "bcFxnExact (volume)", "bcFxnSoln", "bcFxnExact", "bcErr^2", "bcErrIndicator"}, numBCRefinements);
      
      if (rank==0)
      {
        std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << numElements << " elements and BC's relErr is " << relErr << ".\n";
      }
      numBCRefinements++;
      
      refStrategy->refine(rank==0);
      
      // project the BC onto the mesh again
      soln->projectOntoMesh(bcMap);
      
      bc_norm = (bcFxnExact * bcRestriction)->l2norm(mesh, quadratureEnrichment);
      relErr = bc_err->l2norm(mesh, quadratureEnrichment) / bc_norm;
      
      numElements = mesh->numActiveElements();
      
      if (rank==0) std::cout << "\n\n *************** BC refinement " << numBCRefinements << " complete *************** \n\n";
      
//        form->vlasovMesh()->getTopology()->baseMeshTopology()->printAllEntities();
    }
    errorIndicator->measureError();
//    bcViz->exportSolution(soln, {cellIDFunction, bcFxnExact, bcFxnSoln * bcRestriction, bcFxnExact * bcRestriction, bc_err * bc_err, bcErrIndicator * bcRestriction}, {"cellID", "bcFxnExact (volume)", "bcFxnSoln", "bcFxnExact", "bcErr^2", "bcErrIndicator"}, numBCRefinements);
    if (rank==0)
    {
      std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << mesh->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
//      std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
    }
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
  {
    mpiSession.barrier();

    const double e                = 1.602176634e-19;                                          // electron's electric charge in Coulombs
    const double m_e              =  9.10938356e-31;                                          // m_e, mass of electron in kg
    
    const double permittivity     = 8.85418782e-12;                                           // permittivity of free space (epsilon_0)
    const double permeability     = 4 * M_PI * 1.e-7;                                         // permeability of free space (mu_0)
    const double D                = 0.01;
    const double c                = 2.997924580e8;
    double V                      = 1.e6; // gap voltage
    double J_initial = 1e10;
    
    double W                      = 1e4;                                                      // beam intensity in [eV] -- copied from cold diode problem; may want a different value, though Lau et al. seem to suggest the precise number does not matter too much.
    cmdp.setOption("W", &W, "beam intensity [eV]");
    
    cmdp.setOption("V", &V, "gap voltage [V]");
    
    cmdp.setOption("J", &J_initial, "current density [A/m^2] at inflow");
    
    bool useL2TestSpace = false; // otherwise, use graph norm with L2WeightGraph on the L^2 term.  In initial testing, this appears to lead to worse conditioning than the graph norm with a large weight on its L^2 term.
    cmdp.setOption("useL2TestNorm", "useGraphTestNorm", &useL2TestSpace);
    
    double L2WeightGraph    = 1e6; // this is much larger than our usual 1.0 factor, but it yields better conditioning for this problem, better fixed-point iteration convergence, and reduced non-physical f negativity.  It appears 1e6 is preferable in each of these regards to 1e4, the value we used for relativistic cold diode, though more careful testing might be warranted.
    cmdp.setOption("L2WeightGraph", &L2WeightGraph, "Weight for the L^2 part of the graph norm");
    double L2WeightGraphMeshScale = 0; // the transport part of the graph norm scales like (|beta|/h)^2, so on the graded spatial mesh a fixed weight that controls the large elements under-weights the small ones near the cathode (production-scl-graph4e2-x16 blew up there).
    cmdp.setOption("L2WeightGraphMeshScale", &L2WeightGraphMeshScale,
                   "If positive, the weight for the L^2 part of the graph norm on element K is max(L2WeightGraph, L2WeightGraphMeshScale / h_x(K)^2)");
    std::string L2WeightGraphDirectionScalesString = ""; // tensor-product alternative to L2WeightGraphMeshScale: a product of one-dimensional factors adds no tensor rank (matters for low-rank tensor solvers)
    cmdp.setOption("L2WeightGraphDirectionScales", &L2WeightGraphDirectionScalesString,
                   "Comma-separated scales s_d for the coordinates (x, t, u_x, u_y); if given, the weight for the L^2 part of the graph norm on element K is "
                   "L2WeightGraph * prod_d (1 + s_d / (L2WeightGraph h_d(K)^2)).  Omitted trailing scales are zero.  Exclusive with L2WeightGraphMeshScale.");
    double L2WeightGraphTransportScale = 0; // per-element transport scale, accounting for the element's velocity (slow velocity cells near the cathode need far less weight than 1/h_x^2) and for h_t
    cmdp.setOption("L2WeightGraphTransportScale", &L2WeightGraphTransportScale,
                   "If positive, the weight for the L^2 part of the graph norm on element K is "
                   "L2WeightGraph + C [ (v_x,max(K) / h_x)^2 + (1 / h_t)^2 + sum_u (a_u,max / h_u)^2 ], with v = u / gamma bounded over the element's "
                   "velocity box and a_u,max from L2WeightGraphAccelerationBounds.  Exclusive with the other L2WeightGraph scalings.");
    std::string L2WeightGraphAccelerationBoundsString = "";
    cmdp.setOption("L2WeightGraphAccelerationBounds", &L2WeightGraphAccelerationBoundsString,
                   "Comma-separated bounds on |a| in the velocity directions (u_x, u_y), for L2WeightGraphTransportScale; omitted ones are zero.");
    
    // proper velocity domain, specified as fraction of speed of light:
    double beta0                = -0.995; // negative: electrons turned around by the magnetic field can return to the cathode
    double betaMax              = 0.995;
    double cScaleFactor         = 0.5; // velocity will be scaled by cScaleFactor * c.
    
    double minMagneticFluxRatio = 1.07; // A_min is the minimum magnetic flux; we set A_a = A_min * minMagneticFluxRatio.  Ratios above 1 are insulated.  1.07 is what Adam Darr told me to do to get something that flows halfway across the gap before reversing.
    bool usePeriodicBCs         = false;
    bool enforceQuasiNeutrality = false;
    bool useL2Trace             = true;
    bool useSerendipityBasis    = false;

    cmdp.setOption("ratio", &minMagneticFluxRatio, "magnetic flux ratio; values above 1 are insulated");

    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);

    int delta_k = 2;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1

    cmdp.setOption("p", &H1Order);
    cmdp.setOption("dp", &delta_k);

    int xElements = 4;
    int vElements = 40;
    int tElements = 5;
    int numUniformRefinements = 0;

    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements, "number of uniform refinements applied to the base mesh");

    bool gradedVelocityMesh = false;
    double velocityCoreWidthFactor = 1.0, velocityCoreHalfWidthFactor = 3.0, velocityGrowthRatio = 2.0;
    cmdp.setOption("gradedVelocityMesh", "uniformVelocityMesh", &gradedVelocityMesh,
                   "grade the velocity mesh around the inflow beam (ignores vElements); see velocityCore* and velocityGrowthRatio.  "
                   "The graded mesh is the base mesh: combine a coarse grading with numUniformRefinements, since GMG's h-coarsening "
                   "uses only the uniform refinements (otherwise GMG coarsens in p alone).");
    cmdp.setOption("velocityCoreWidthFactor", &velocityCoreWidthFactor, "graded velocity mesh: core element width, in units of sigma");
    cmdp.setOption("velocityCoreHalfWidthFactor", &velocityCoreHalfWidthFactor, "graded velocity mesh: half-width of the uniform core around the beam, in units of sigma");
    cmdp.setOption("velocityGrowthRatio", &velocityGrowthRatio, "graded velocity mesh: geometric growth ratio of element widths outside the core");

    double x0        = 0.0;
    double t0        = 0.0;
    double tWidth    = 1.0;
    
    cmdp.setOption("beta0", &beta0, "minimum speed-of-light fraction in velocity domain");
    cmdp.setOption("betaMax", &betaMax, "maximum speed-of-light fraction in velocity domain");
    cmdp.setOption("cScaleFactor", &cScaleFactor, "velocity scaling, specified as a fraction of the speed of light (i.e., 1.0 value corresponds to velocity scaling of c).");
    
    bool useTimeSlabs = true;
    int numTimeSlabs = 1;

    cmdp.setOption("numTimeSlabs", &numTimeSlabs);
    cmdp.setOption("tWidth", &tWidth);
    
    double temporalRampDuration = 0.0; // default value (<= 0 indicates no ramp)
    cmdp.setOption("temporalRampDuration", &temporalRampDuration, "Duration of temporal ramp for injection BC");

    double sigma        = -1.0;  // if negative (i.e. left unset by user), set using sigmaFactor
    cmdp.setOption("sigma", &sigma, "(proper) thermal velocity at inflow, in mesh proper velocity units");
    double sigmaFactor  = 0.25; // (proper) thermal velocity = sigmaFactor * u0
    cmdp.setOption("sigmaFactor", &sigmaFactor, "(proper) thermal velocity at inflow as a fraction of u0");
    
    bool useAdaptivity = false;
    int maxRefinements = 3;
    int maxRefinementsFinalSlab = -1; // defaults to same as maxRefinements
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 100000; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    double doerflerTarget = 0; // if set to positive value by user, use Doerfler instead of greedy strategy
    bool useEnergyForUnrefinements = false;
    double relErrorTarget = 0.01; // can stop adaptivity one refinement after this is met
    double relErrorTargetFinalSlab = -1.0; // defaults to same as relErrorTarget

    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("maxRefinements", &maxRefinements);
    cmdp.setOption("maxRefinementsFinalSlab", &maxRefinementsFinalSlab);
    cmdp.setOption("relErrorTarget", &relErrorTarget);
    cmdp.setOption("relErrorTargetFinalSlab", &relErrorTargetFinalSlab);
    cmdp.setOption("refinementThreshold", &refinementThreshold);
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
    cmdp.setOption("doerflerTarget", &doerflerTarget);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);

    std::string outputDir = "/tmp";
    bool saveViz = true; // Changed from false to true
    std::string resultsFilePrefix = "";

    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);

    int solutionOutputFrequency = 1;                    // only used if runOverkillVersusAdaptiveComparison is true
    int solutionComparisonFrequency = 1;                // only used if runOverkillVersusAdaptiveComparison is true
    
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("solutionComparisonFrequency", &solutionComparisonFrequency);
    
    int maxDofsForDirectSolve = 20000;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    int iterativeSolveMaxIters  = 10000;

    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);

    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 50; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    
    // in the context of adaptivity, we can use looser tolerances for fixed point and iterative solves, prior to final solution
    double coarseIterativeSolveTolerance = 1e-4;
    int coarseIterativeSolveMaxIters  = 2000;
    cmdp.setOption("coarseIterativeSolveTolerance", &coarseIterativeSolveTolerance);
    cmdp.setOption("coarseIterativeSolveMaxIters",  &coarseIterativeSolveMaxIters);
    
    int coarseMinSubsteps = 2;
    int coarseMaxSubsteps = 10;
    double coarseFixedPointTol = 1e-2;
    
    cmdp.setOption("coarseMinSubsteps",   &coarseMinSubsteps);
    cmdp.setOption("coarseMaxSubsteps",   &coarseMaxSubsteps);
    cmdp.setOption("coarseFixedPointTol", &coarseFixedPointTol);

    cmdp.setOption("enforcePositivity", "noEnforcePositivity", &gEnforcePositivity,
                   "Use proximal Galerkin positivity enforcement for the Vlasov unknown.");
    cmdp.setOption("proximalGalerkinAlpha0", &gProximalGalerkinAlpha0,
                   "Initial alpha parameter for proximal Galerkin positivity enforcement.");
    cmdp.setOption("proximalGalerkinAlphaMultiplier", &gProximalGalerkinAlphaMultiplier,
                   "Alpha multiplier for proximal Galerkin positivity enforcement between slabs.");

    double bcResolutionTolerance = 1e-1;  // only applies if doing an adaptive solve
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    
    double initialTimeDivision = 0;
    cmdp.setOption("initialTimeDivision", &initialTimeDivision);

    bool phase0Instrumentation = false;
    std::string phase0TimingFile = "";
    std::string phase0TensorGraphSchurBackend = "off";
    std::string phase8MomentBackend = "tensor_native";
    double phase8MomentCompareAbsTol = 1e-10;
    double phase8MomentCompareRelTol = 1e-8;
    double phase8MomentCompareRelFloor = 1e-14;
    bool phase8MomentCompareStrict = false;
    int phase8MomentVelocityQuadratureDegree = -1;
    int phase0TensorGraphSchurXElements = 2;
    int phase0TensorGraphSchurTElements = 1;
    int phase0TensorGraphSchurUxElements = 1;
    int phase0TensorGraphSchurUyElements = 2;
    int phase0TensorGraphSchurComponentP = 2;
    int phase0TensorGraphSchurTestEnrichment = 1;
    cmdp.setOption("phase0Instrumentation", "noPhase0Instrumentation", &phase0Instrumentation,
                   "Emit grep-friendly Phase 0 timing lines for Planar MITL baseline/performance runs.");
    std::string emissionModel = "spaceChargeLimited";
    cmdp.setOption("emissionModel", &emissionModel,
                   "cathode emission: spaceChargeLimited (adjust the emitted density within each slab's fixed-point iteration so "
                   "that E(0) = 0 at the slab's final time), taper (fixed per-slab injection with the E(0) taper), or constant");
    int sclEmissionDegree = -1;
    cmdp.setOption("sclEmissionDegree", &sclEmissionDegree,
                   "spaceChargeLimited: -1 (default) holds the emitted density constant within each slab and drives E(0) to zero "
                   "at the slab's final time; m >= 0 lets the emitted density be a nonnegative degree-m polynomial in time within "
                   "each slab (Bernstein coefficients a_j >= 0) and imposes the complementarity condition a_j >= 0, g_j <= 0, "
                   "a_j g_j = 0 on the Bernstein-weighted means g_j of E(0,t)/E0 (with every a_j > 0, the first m+1 Legendre "
                   "moments of E(0,t) vanish; m = 0: the slab-mean of E(0)).");
    int sclEmissionQuadraturePoints = 6;
    cmdp.setOption("sclEmissionQuadraturePoints", &sclEmissionQuadraturePoints,
                   "spaceChargeLimited with sclEmissionDegree >= 0: Gauss-Legendre points in time for the weighted means of E(0,t)");
    double sclEmissionTolerance = 1e-3;
    cmdp.setOption("sclEmissionTolerance", &sclEmissionTolerance,
                   "spaceChargeLimited: converged when |E(0)| at the slab's final time (or, for sclEmissionDegree >= 0, each "
                   "weighted mean of E(0,t)/E0 off its bound) is below this fraction of |E0|");
    bool taperInjection = true;
    double taperUpperThreshold = 0.50, taperLowerThreshold = 0.10;
    cmdp.setOption("taperInjection", "noTaperInjection", &taperInjection,
                   "reduce the inflow current as E(0) falls (to mitigate virtual cathode formation); see taperUpperThreshold and taperLowerThreshold");
    cmdp.setOption("taperUpperThreshold", &taperUpperThreshold, "injection taper starts when E(0)/E0 falls below this fraction");
    cmdp.setOption("taperLowerThreshold", &taperLowerThreshold, "injection is off when E(0)/E0 falls below this fraction");
    bool accuracyDiagnostics = false;
    cmdp.setOption("accuracyDiagnostics", "noAccuracyDiagnostics", &accuracyDiagnostics,
                   "At the end of each time slab, print integral quantities (f, E^2, Jx over the slab) for comparing discretizations.");
    std::string coarseSolver = "default";
    cmdp.setOption("coarseSolver", &coarseSolver,
                   "Direct solver for the coarsest GMG level: default (KLU2, or MUMPS if available), KLU, SuperLUDist, or MUMPS.  SuperLUDist distributes the coarse factorization and uses nested-dissection ordering (requires Trilinos built with SuperLU_DIST).");
    bool densityDiagnostic = false;
    int densityDiagnosticPointsPerDim = 16;
    cmdp.setOption("densityDiagnostic", "noDensityDiagnostic", &densityDiagnostic,
                   "At the end of each time slab, write the electron density (exact velocity moment of f), Jx, and E_x on a dense grid of points per (x,t) cell to <outputDir>/density_slab<N>.csv; plot with plot_density.py.");
    cmdp.setOption("densityDiagnosticPointsPerDim", &densityDiagnosticPointsPerDim,
                   "Points per dimension (x and t, including cell boundaries) on each configuration cell for densityDiagnostic.");
    cmdp.setOption("phase0TimingFile", &phase0TimingFile,
                   "Optional rank-0 file for Phase 0 timing lines; stdout is always used when instrumentation is enabled.");
    cmdp.setOption("phase0TensorGraphSchurBackend", &phase0TensorGraphSchurBackend,
                   "Optional Phase 0 tensor-native graph-norm Schur micro-apply backend: off, assembled_crs, or matrix_free.");
    cmdp.setOption("phase8MomentBackend", &phase8MomentBackend,
                   "Compatibility-only Phase 8 Planar MITL moment backend label.  Planar MITL now unconditionally uses tensor-native moment reductions for production Maxwell current sources.");
    cmdp.setOption("phase8MomentCompareAbsTol", &phase8MomentCompareAbsTol,
                   "Absolute tolerance for Phase 8 tensor-vs-legacy rhoRHS/Jx/Jz moment comparisons.");
    cmdp.setOption("phase8MomentCompareRelTol", &phase8MomentCompareRelTol,
                   "Relative tolerance for Phase 8 tensor-vs-legacy rhoRHS/Jx/Jz moment comparisons.");
    cmdp.setOption("phase8MomentCompareRelFloor", &phase8MomentCompareRelFloor,
                   "Positive denominator floor for Phase 8 tensor-vs-legacy relative moment comparisons.");
    cmdp.setOption("phase8MomentCompareStrict", "noPhase8MomentCompareStrict", &phase8MomentCompareStrict,
                   "When enabled, abort if Phase 8 tensor-vs-legacy moment differences exceed tolerances.");
    cmdp.setOption("phase8MomentVelocityQuadratureDegree", &phase8MomentVelocityQuadratureDegree,
                   "Velocity quadrature degree for Phase 8 sampled Vlasov HostMatrix bridge; negative uses the max local phase-space cell cubature degree.");
    cmdp.setOption("phase0TensorGraphSchurXElements",  &phase0TensorGraphSchurXElements,  "Number of x cells in the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    cmdp.setOption("phase0TensorGraphSchurTElements",  &phase0TensorGraphSchurTElements,  "Number of t cells in the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    cmdp.setOption("phase0TensorGraphSchurUxElements", &phase0TensorGraphSchurUxElements, "Number of ux cells in the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    cmdp.setOption("phase0TensorGraphSchurUyElements", &phase0TensorGraphSchurUyElements, "Number of uy cells in the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    cmdp.setOption("phase0TensorGraphSchurComponentP", &phase0TensorGraphSchurComponentP, "Per-component polynomial order for the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    cmdp.setOption("phase0TensorGraphSchurTestEnrichment", &phase0TensorGraphSchurTestEnrichment, "Test-space enrichment for the optional Phase 0 tensor graph-norm Schur microbenchmark.");
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }
    auto parseCommaSeparatedDoubles = [](const std::string &listString)
    {
      std::vector<double> list;
      std::istringstream listStream(listString);
      std::string entry;
      while (std::getline(listStream, entry, ','))
      {
        list.push_back(std::stod(entry));
      }
      return list;
    };
    const std::vector<double> L2WeightGraphDirectionScales    = parseCommaSeparatedDoubles(L2WeightGraphDirectionScalesString);
    const std::vector<double> L2WeightGraphAccelerationBounds = parseCommaSeparatedDoubles(L2WeightGraphAccelerationBoundsString);
    if (!phase0TensorGraphSchurBackendIsValid(phase0TensorGraphSchurBackend))
    {
      if (rank == 0)
      {
        std::cout << "Unsupported phase0TensorGraphSchurBackend='" << phase0TensorGraphSchurBackend
                  << "'; expected off, assembled_crs, or matrix_free.\n";
      }
      return -1;
    }
    if (!phase8MomentBackendIsValid(phase8MomentBackend))
    {
      if (rank == 0)
      {
        std::cout << "Unsupported phase8MomentBackend='" << phase8MomentBackend
                  << "'; expected legacy, tensor_native, dry_run_compare, or compare.\n";
      }
      return -1;
    }
    if ((phase8MomentCompareAbsTol < 0.0) || (phase8MomentCompareRelTol < 0.0) ||
        (phase8MomentCompareRelFloor <= 0.0) || (phase8MomentVelocityQuadratureDegree == 0))
    {
      if (rank == 0)
      {
        std::cout << "Phase 8 moment comparison requires nonnegative tolerances, a positive relative floor, and a nonzero velocity quadrature degree (negative selects p).\n";
      }
      return -1;
    }
    if ((phase0TensorGraphSchurXElements <= 0) || (phase0TensorGraphSchurTElements <= 0) ||
        (phase0TensorGraphSchurUxElements <= 0) || (phase0TensorGraphSchurUyElements <= 0) ||
        (phase0TensorGraphSchurComponentP < 0) || (phase0TensorGraphSchurTestEnrichment < 0))
    {
      if (rank == 0)
      {
        std::cout << "Phase 0 tensor graph-norm Schur microbenchmark sizes require positive element counts and nonnegative p/enrichment.\n";
      }
      return -1;
    }

    Phase0Instrumentation phase0(phase0Instrumentation, rank, phase0TimingFile);
    const auto phase0TotalStart = phase0.now();
    phase0.logDuration("start", 0.0,
                       {{"mpi_ranks", std::to_string(mpiSession.getNProc())},
                        {"x_elements", std::to_string(xElements)},
                        {"v_elements", gradedVelocityMesh ? std::string("graded") : std::to_string(vElements)},
                        {"t_elements", std::to_string(tElements)},
                        {"uniform_refinements", std::to_string(numUniformRefinements)},
                        {"p", std::to_string(H1Order)},
                        {"dp", std::to_string(delta_k)},
                        {"num_time_slabs", std::to_string(numTimeSlabs)},
                        {"serendipity", useSerendipityBasis ? "1" : "0"},
                         {"moment_backend", phase8MomentBackend}});
    
    using TestIP = RelativisticVlasovSpaceTimeFormulation::TestSpaceInnerProduct;
    const TestIP testIP = useL2TestSpace ? TestIP::L2TestSpaceInnerProduct : TestIP::GraphTestSpaceInnerProduct;
    
    if (maxRefinementsFinalSlab < 0)
    {
      maxRefinementsFinalSlab = maxRefinements;
    }
    if (relErrorTargetFinalSlab < 0)
    {
      relErrorTargetFinalSlab = relErrorTarget;
    }
    
    if (rank == 0)
    {
      std::cout << "Proximal Galerkin positivity enforcement: " << (gEnforcePositivity ? "enabled" : "disabled") << "\n";
      std::cout << "Phase 0 tensor graph-norm Schur backend: " << phase0TensorGraphSchurBackendName(phase0TensorGraphSchurBackend)
                << " (micro-apply selection only; legacy production solve remains active)\n";
      if (gEnforcePositivity)
      {
        std::cout << "  proximalGalerkinAlpha0 = " << gProximalGalerkinAlpha0 << "\n";
        std::cout << "  proximalGalerkinAlphaMultiplier = " << gProximalGalerkinAlphaMultiplier << "\n";
      }
    }
    
    // scalings found in Lau et al. eqns (1.1) - (1.10)
    const double velocityScaling = c; // 3.0 x 10^8 [m/s]
    const double voltageScaling  = m_e * c * c / e; // 511 [kV]
    const double vectorPotentialScaling = m_e * c / e; // 1.706 x 10^-3 [T-m]
    const double currentScaling = (m_e * c) / (e * permeability); // 1.358 [kA]
    const double spatialScaling = D; // .01 [m]
    const double timeScaling    = spatialScaling / velocityScaling;
    const double EScaling       = voltageScaling / spatialScaling;
    const double JScaling       = currentScaling / (spatialScaling*spatialScaling);
    const double nScaling       = permittivity * m_e * c * c / (e * e * spatialScaling * spatialScaling); // Eq. (1.10)
    
    const double cScaled = c / velocityScaling;
    
    const double gamma_beam       = 1 + W * e / (m_e * c * c);
    const double beta_beam        = sqrt(1. - 1./(gamma_beam * gamma_beam));
    const double v_beam           = c * beta_beam;

    const double q                = -1.0;
    const double m                =  1.0;
    
    const double J0 = J_initial / JScaling;
    
    double v0     = v_beam / velocityScaling;
    double ux0    = v0 * gamma_beam; // drift velocity in x direction
    double uy0    = 0.0;             // drift velocity in y direction
    double n0     = J0 / ux0;
    
    if (sigma < 0)
    {
      sigma = sigmaFactor * ux0;
    }
    else
    {
      sigmaFactor = sigma / ux0;
    }
    
    const double V_scaled = V / voltageScaling;
    
    const double A_min    = sqrt(V_scaled * V_scaled + 2. * V_scaled);
    
    const double A_a      = A_min * minMagneticFluxRatio;
    const double omega_p  = sqrt( e * e * nScaling / (m_e * permittivity));
    const double eps0     =  1.0 / (omega_p * omega_p * timeScaling * timeScaling);
        
    double xWidth = D / spatialScaling;
    
    const double uMin = beta0   * cScaled / sqrt(1-beta0*beta0);
    const double uMax = betaMax * cScaled / sqrt(1-betaMax*betaMax);
    
    const double uxMin = uMin;
    const double uxMax = uMax;
    
    // the beam is centered at uy0 = 0, so the uy domain is symmetric about 0
    const double uyMin = -uMax;
    const double uyMax =  uMax;
    
    std::vector<double> velocitySpaceVertices_x, velocitySpaceVertices_y;
    if (gradedVelocityMesh)
    {
      velocitySpaceVertices_x = gradedVertices(uxMin, uxMax, ux0, velocityCoreWidthFactor * sigma, velocityCoreHalfWidthFactor * sigma, velocityGrowthRatio);
      velocitySpaceVertices_y = gradedVertices(uyMin, uyMax, uy0, velocityCoreWidthFactor * sigma, velocityCoreHalfWidthFactor * sigma, velocityGrowthRatio);
    }
    else
    {
      velocitySpaceVertices_x = uniformVertices(uxMin, uxMax - uxMin, vElements);
      velocitySpaceVertices_y = uniformVertices(uyMin, uyMax - uyMin, vElements);
    }
    ensureVertexAt(velocitySpaceVertices_x, 0.0);
    ensureVertexAt(velocitySpaceVertices_y, 0.0);
    const int vxElements = int(velocitySpaceVertices_x.size()) - 1;
    const int vyElements = int(velocitySpaceVertices_y.size()) - 1;
    
    const double E0 = -V_scaled / xWidth;
    
    if (rank == 0)
    {
      std::cout << "Running relativistic planar MITL problem.\n";
      std::cout << "A_min:              " << A_min << " [units]\n";
      std::cout << "A_a:                " << A_a << " [units]\n";
      std::cout << "J:                  " << J_initial << " [A/m^2]\n";
      std::cout << "gap voltage:        " << V << " [V]\n";
      std::cout << "E scaling :         " << EScaling << std::endl;
      std::cout << "potential scaling : " << voltageScaling << std::endl;
      std::cout << "spatial scaling :   " << spatialScaling << std::endl;
      std::cout << "n scaling :         " << nScaling << std::endl;
      std::cout << "temporal scaling :  " << timeScaling << std::endl;
      std::cout << "velocity scaling :  " << velocityScaling << std::endl;
      std::cout << "c scaled:           " << cScaled  << std::endl;
      std::cout << "computational domain width: " << xWidth << std::endl;
      std::cout << "physical domain width: " << xWidth * spatialScaling << std::endl;
      std::cout << "initial time division: " << initialTimeDivision << std::endl;
      
      std::cout << "v_beam:   " << v_beam << " [m/s] (" << v_beam / c  << " c)" << std::endl;
      std::cout << "uMin:     " << uMin << " (" << beta0 << " c)\n";
      std::cout << "uMax:     " << uMax << " (" << betaMax << " c)\n";
      std::cout << "uy range: [" << uyMin << ", " << uyMax << "]\n";
      auto printVertices = [](const std::string &name, const std::vector<double> &vertices) {
        std::cout << name << " vertices (" << vertices.size() - 1 << " elements):";
        for (double v : vertices) std::cout << " " << v;
        std::cout << std::endl;
      };
      printVertices("ux", velocitySpaceVertices_x);
      printVertices("uy", velocitySpaceVertices_y);
      std::cout << "sigma:    " << sigma << std::endl;
      std::cout << "u0:       " << ux0 << std::endl;
      std::cout << "v0:       " << v0 << std::endl;
      std::cout << "gamma0:   " << gamma_beam << std::endl;
      std::cout << "n0:       " << n0 << std::endl;
      std::cout << "omega_p:  " << omega_p << std::endl;
    }
    
    const bool useGeometricGradedSpatialMesh = true;
    
    auto phase0FormulationSetupStart = phase0.now();
    auto form = getFormulation1D1T2V(cScaled, x0, xWidth, t0, tWidth, velocitySpaceVertices_x, velocitySpaceVertices_y, xElements, tElements,
                                     q, eps0, m, H1Order, delta_k, testIP, L2WeightGraph, usePeriodicBCs,
                                     useSerendipityBasis, useGeometricGradedSpatialMesh, numUniformRefinements,
                                     L2WeightGraphMeshScale, L2WeightGraphDirectionScales,
                                     L2WeightGraphTransportScale, L2WeightGraphAccelerationBounds);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    phase0.logDuration("formulation_setup", phase0.elapsedSince(phase0FormulationSetupStart), phase0MeshStats(form));

    // Per-substep Vlasov solve breakdown.  G/B/K come from the BF's optimal-test timing callback (rank-local; summed over
    // ranks below to match the Solution timers): G = Gram matrix assembly, B = enriched stiffness assembly,
    // K = per-element Cholesky solve of G and formation of B^T G^{-1} B.
    double phase0TimeG = 0, phase0TimeB = 0, phase0TimeK = 0;
    int phase0OptimalTestElementCount = 0;
    std::function<void(int, double, double, double, double, ElementTypePtr)> phase0OptimalTestTimingCallback =
      [&](int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType) {
        phase0OptimalTestElementCount += numElements;
        phase0TimeG += timeG;
        phase0TimeB += timeB;
        phase0TimeK += timeT + timeK;
      };
    if (phase0.enabled())
    {
      form->vlasovBF()->setOptimalTestTimingCallback(phase0OptimalTestTimingCallback);
      form->setSubstepSolveProfileCallback([&](const VlasovSpaceTimeFormulation::SubstepSolveProfile &profile) {
        auto comm = MPIWrapper::TeuchosCommWorld();
        Phase0Fields fields;
        fields.push_back({"substep",               std::to_string(profile.substepOrdinal)});
        fields.push_back({"solver",                profile.usedGeometricMultigrid ? "gmg" : "direct"});
        fields.push_back({"iterations",            std::to_string(profile.iterationCount)});
        fields.push_back({"global_dofs",           std::to_string(profile.numGlobalDofs)});
        fields.push_back({"optimal_test_elements", std::to_string(MPIWrapper::sum(comm, phase0OptimalTestElementCount))});
        fields.push_back({"sum_G",                 phase0DoubleString(MPIWrapper::sum(comm, phase0TimeG))});
        fields.push_back({"sum_B",                 phase0DoubleString(MPIWrapper::sum(comm, phase0TimeB))});
        fields.push_back({"sum_K",                 phase0DoubleString(MPIWrapper::sum(comm, phase0TimeK))});
        fields.push_back({"sum_local_stiffness",   phase0DoubleString(profile.totalTimeLocalStiffness)});
        fields.push_back({"sum_global_assembly",   phase0DoubleString(profile.totalTimeGlobalAssembly)});
        fields.push_back({"sum_bc_imposition",     phase0DoubleString(profile.totalTimeBCImposition)});
        fields.push_back({"sum_linear_solve",      phase0DoubleString(profile.totalTimeSolve)});
        fields.push_back({"gmg_construction",      phase0DoubleString(profile.wallTimeGMGConstruction)});
        fields.push_back({"moment_update",         phase0DoubleString(profile.wallTimeMomentUpdate)});
        fields.push_back({"configuration_solve",   phase0DoubleString(profile.wallTimeConfigurationSolve)});
        fields.push_back({"configuration_update",  phase0DoubleString(profile.wallTimeConfigurationUpdate)});
        for (const auto &entry : profile.gmgTimings)
        {
          std::string key = "gmg_" + entry.first;
          std::replace(key.begin(), key.end(), ' ', '_');
          fields.push_back({key, phase0DoubleString(entry.second)});
        }
        phase0.logDuration("vlasov_substep", profile.wallTimeVlasovSolve, fields);
        // memory at phase boundaries of the Vlasov solve: max and sum of current resident size over ranks, and max high-water mark
        const int numCheckpoints = profile.memoryCheckpoints.size();
        if (MPIWrapper::min(comm, numCheckpoints) == MPIWrapper::max(comm, numCheckpoints))
        {
          for (int i=0; i<numCheckpoints; i++)
          {
            const auto &checkpoint = profile.memoryCheckpoints[i];
            Phase0Fields memoryFields;
            memoryFields.push_back({"substep",       std::to_string(profile.substepOrdinal)});
            memoryFields.push_back({"checkpoint",    checkpoint.name});
            memoryFields.push_back({"max_rss_gb",    phase0DoubleString(MPIWrapper::max(comm, checkpoint.currentGB))});
            memoryFields.push_back({"sum_rss_gb",    phase0DoubleString(MPIWrapper::sum(comm, checkpoint.currentGB))});
            memoryFields.push_back({"max_hwm_gb",    phase0DoubleString(MPIWrapper::max(comm, checkpoint.highWaterMarkGB))});
            phase0.logDuration("vlasov_memory", 0.0, memoryFields);
          }
        }
        phase0TimeG = phase0TimeB = phase0TimeK = 0;
        phase0OptimalTestElementCount = 0;
      });
    }
    runPhase0TensorGraphSchurMicroApply(phase0TensorGraphSchurBackend,
                                        cScaled,
                                        x0,
                                        xWidth,
                                        t0,
                                        tWidth,
                                        uxMin,
                                        uxMax,
                                        uyMin,
                                        uyMax,
                                        L2WeightGraph,
                                        phase0TensorGraphSchurXElements,
                                        phase0TensorGraphSchurTElements,
                                        phase0TensorGraphSchurUxElements,
                                        phase0TensorGraphSchurUyElements,
                                        phase0TensorGraphSchurComponentP,
                                        phase0TensorGraphSchurTestEnrichment,
                                        phase0);
    
//    form->setExportViz(true, "/tmp"); // DEBUGGING
    
    std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true
    
    // we'll use this function to allow labeling of visualization output
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
    //    form->setUseGeometricMultigrid(useGeometricMultigrid);
    form->setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    //    form->setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if ((maxSubsteps > 3) && !useAdaptivity) // for adaptive solves, after the 0 refinement, we'll already be pretty close to the right solution, and looser tolerances will tend to push us away.  Not worth it to special-case the 0 refinement, I think.  So we only use the looser CG tolerances below for non-adaptive solves.
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 10000.0;
      cgTols[1] = iterativeSolveTolerance * 100.0;
    }
    form->setIterativeSolveTolerances(cgTols);
    form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    
    std::vector<double> cgTolsCoarse(coarseMaxSubsteps,coarseIterativeSolveTolerance);
    if ((coarseMaxSubsteps > 3) && !useAdaptivity) // for adaptive solves, after the 0 refinement, we'll already be pretty close to the right solution, and looser tolerances will tend to push us away.  Not worth it to special-case the 0 refinement, I think.  So we only use the looser CG tolerances below for non-adaptive solves.
    {
      // use looser tolerances for the first couple steps
      cgTolsCoarse[0] = coarseIterativeSolveTolerance * 100.0;
      cgTolsCoarse[1] = coarseIterativeSolveTolerance * 10.0;
    }
    
    FunctionPtr f0 = Function::zero(); // start with vacuum
    FunctionPtr f_leftInflow;
    
    Teuchos::RCP<Maxwellian1T2V> fM_u0;
    Teuchos::RCP<EmissionTimeProfile> emissionProfile; // spaceChargeLimited with sclEmissionDegree >= 0
    
    {
      {
        // pin potential to zero at left, 1 megavolt at right:
        SpatialFilterPtr fullBoundaryFilter  = SpatialFilter::allSpace();
        bool restrictToBoundary = true; // cold diode: not periodic
        SpatialFilterPtr  leftBoundaryFilter = SpatialFilter::matchingX(x0);
        SpatialFilterPtr rightBoundaryFilter = SpatialFilter::matchingX(x0+xWidth);
        auto  leftBoundaryTag = form->createConfigurationMeshSideSetTag( leftBoundaryFilter, restrictToBoundary);
        auto rightBoundaryTag = form->createConfigurationMeshSideSetTag(rightBoundaryFilter, restrictToBoundary);
        auto VE_traceVar = form->VE_configurationSpaceTrace();
        form->addConfigurationSpaceBC( leftBoundaryTag, VE_traceVar, Function::zero());
        const double V_scaled = V / voltageScaling;
        form->addConfigurationSpaceBC(rightBoundaryTag, VE_traceVar, Function::constant(V_scaled));
        auto A_traceVar = form->configurationSpaceFormulation()->A_hat(1); // A_y is the one enabled for 1D2V Maxwell
        form->addConfigurationSpaceBC( leftBoundaryTag, A_traceVar, Function::zero());
        form->addConfigurationSpaceBC(rightBoundaryTag, A_traceVar, Function::constant(A_a)); // A_a is already scaled (dimensionless).
        
        // set a Maxwellian at the inflow:
        fM_u0 = Teuchos::rcp(new Maxwellian1T2V( ux0, uy0, sigma, n0) );
        if ((emissionModel == "spaceChargeLimited") && (sclEmissionDegree >= 0))
        {
          // the Maxwellian has unit amplitude; the (time-varying) emitted density is carried by the profile
          emissionProfile = Teuchos::rcp(new EmissionTimeProfile(sclEmissionDegree, n0));
          emissionProfile->setSlab(t0, t0 + tWidth);
          fM_u0->set_n(1.0);
        }
        FunctionPtr positiveRestriction = Function::heaviside(0,2); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
        
        if (temporalRampDuration > 0)
        {
          FunctionPtr temporalRamp = Teuchos::rcp(new TemporalRamp1D1T(temporalRampDuration));
          f_leftInflow = FunctionPtr(fM_u0) * positiveRestriction * temporalRamp;
        }
        else
        {
          f_leftInflow = FunctionPtr(fM_u0) * positiveRestriction;
        }
        if (emissionProfile != Teuchos::null) f_leftInflow = f_leftInflow * FunctionPtr(emissionProfile);
        
        form->setInitialAndInflowConditions(f0, f_leftInflow, initialTimeDivision); // no right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        if (useAdaptivity)
        {
          bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
        }
      }
    }
    
    auto phase0InitialBCRefinementStart = phase0.now();
    refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form->configurationSpaceMesh()->repartitionAndRebuild();
    if (rank == 0)
    {
      std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
    }
    phase0.logDuration("initial_bc_refinement", phase0.elapsedSince(phase0InitialBCRefinementStart), phase0MeshStats(form));
    
    std::ostringstream suffix;
    if (!useL2Trace)
    {
      suffix << "_H1Trace";
    }
    {
      suffix << "_relSigma" << sigmaFactor;
    }
    {
      suffix << "_J" << J_initial;
    }
    if (useSerendipityBasis)
    {
      suffix << "_serendipity";
    }
    else
    {
      suffix << "_fullTensorBasis";
    }
    suffix << "_p" << H1Order;
    suffix << "_dp" << delta_k;
    suffix << "_" << xElements << "x" << tElements << "x" << vxElements << "x" << vyElements;
    suffix << "_ratio" << minMagneticFluxRatio;
    if (initialTimeDivision > 0)
    {
      suffix << "_timeRamp" << initialTimeDivision;
    }
    if (useL2TestSpace)
    {
      suffix << "_L2Test";
    }
    else
    {
      suffix << "_GraphL2" << L2WeightGraph;
      if (L2WeightGraphMeshScale > 0) suffix << "_hScale" << L2WeightGraphMeshScale;
      if (!L2WeightGraphDirectionScales.empty())
      {
        suffix << "_hScales";
        for (double scale : L2WeightGraphDirectionScales) suffix << "_" << scale;
      }
      if (L2WeightGraphTransportScale > 0)
      {
        suffix << "_transportScale" << L2WeightGraphTransportScale;
        for (double bound : L2WeightGraphAccelerationBounds) suffix << "_a" << bound;
      }
    }
    
    const std::string problemPrefix = "relativistic-planar-MITL";
    
    std::string AMR_suffix = "";
    if (useAdaptivity)
    {
      std::ostringstream AMR_suffix_ss;
      AMR_suffix_ss << "_AMR_";
      if (doerflerTarget > 0)
      {
        AMR_suffix_ss << "Doerfler" << doerflerTarget << "_";
      }
      else
      {
        AMR_suffix_ss << "theta" << refinementThreshold << "_";
      }
      AMR_suffix = AMR_suffix_ss.str();
    }
    
    auto getAMRSuffixForRefNumber = [&](int refNumber = -1) {
      std::ostringstream AMR_suffix_ss;
      AMR_suffix_ss << AMR_suffix;
      if (useAdaptivity)
      {
        AMR_suffix_ss << refNumber << "_refs";
      }
      return AMR_suffix_ss.str();
    };
    
    auto getResultsFilePath = [&](int refNumber = -1) {
      std::ostringstream resultsFileStream;
      resultsFileStream << outputDir << "/";
      if (resultsFilePrefix == "")
      {
        resultsFileStream << "results_" << problemPrefix << "_" << suffix.str();
      }
      else
      {
        resultsFileStream << resultsFilePrefix;
      }
      resultsFileStream << getAMRSuffixForRefNumber(refNumber);
      resultsFileStream << ".dat";
      
      return resultsFileStream.str();
    };
    
    auto getSpatialMeshFilePath = [&](int refNumber = -1) {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << problemPrefix << "_mesh1D" << suffix.str();
      
      spatialMeshInfoPathStringStream << getAMRSuffixForRefNumber(refNumber);
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getRefinementReportFilePath = [&]() {
      std::ostringstream refReportPathStringStream;
      refReportPathStringStream << outputDir << "/";
      refReportPathStringStream << problemPrefix << "_refReport" << suffix.str();
      
      refReportPathStringStream << getAMRSuffixForRefNumber(maxRefinements);
      refReportPathStringStream << ".dat";
      
      return refReportPathStringStream.str();
    };
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
    
    std::ostringstream maxwellSS;
    maxwellSS << problemPrefix << "Maxwell" << suffix.str();
    
    if (useAdaptivity)
    {
      vlasovSS  << getAMRSuffixForRefNumber(maxRefinements);
      maxwellSS << getAMRSuffixForRefNumber(maxRefinements);
    }
    
    const std::string directoryPath = outputDir;
    
    if (coarseSolver != "default") form->setCoarseSolverChoice(Solver::solverChoiceFromString(coarseSolver)); // carried to later slabs by nextTimeSlab()

    // Space-charge-limited emission: within each slab's fixed-point iteration, after each configuration-space solve, adjust
    // the emitted density so that the cathode field at the slab's final time vanishes.  Newton iteration on E(0)(n_emit),
    // with a slope s = dE(0)/dn_emit that is
    //   - estimated from the last two (n_emit, E(0)) pairs only when n_emit changed appreciably between them (a secant step
    //     between equal densities is undefined; this is what stalled the previous secant iteration),
    //   - kept, rather than replaced, when the estimate is unavailable or non-positive (E(0) increases with n_emit), and
    //   - carried from slab to slab, since the sheath's response changes slowly.
    // The density is held while |E(0)| is within tolerance (so that the slab's final f is consistent with the final
    // density), and updated whenever it is not -- including after an earlier substep converged, since E(0) can drift as f
    // relaxes -- limited per substep as described below.  The fixed-point iteration does not stop until |E(0)| < sclEmissionTolerance |E0|.  The emitted density
    // persists into the next slab as its starting value.
    const bool spaceChargeLimitedEmission = (emissionModel == "spaceChargeLimited");
    TEUCHOS_TEST_FOR_EXCEPTION(!spaceChargeLimitedEmission && (emissionModel != "taper") && (emissionModel != "constant"),
                               std::invalid_argument, "emissionModel must be spaceChargeLimited, taper, or constant");
    if (emissionModel == "constant") taperInjection = false;
    if (spaceChargeLimitedEmission)
    {
      struct SCLState
      {
        double slope = -1;                      // dE(0)/dn_emit; negative until first estimated
        bool havePrevious = false;              // previous substep's (n, E) available (within this slab)
        double nPrevious = 0, EPrevious = 0;
      };
      auto sclState = std::make_shared<SCLState>();
      if (emissionProfile != Teuchos::null)
      {
        // Time-varying emission: n_emit(t) = sum_j a_j B_j(s) on each slab (Bernstein basis, a_j >= 0).  Space-charge-
        // limited emission is a complementarity condition: n_emit >= 0, the cathode field not accelerating electrons off
        // the cathode (E(0)/E0 <= 0), and n_emit E(0) = 0 -- where the field has reversed, no emission.  Weakly, with
        //   g_j = (m+1) int_0^1 (E(0, t(s)) / E0) B_j(s) ds,
        // a weighted mean of E(0)/E0 near the time of coefficient j,
        //   a_j >= 0,  g_j <= 0,  a_j g_j = 0,  j = 0..m.
        // When every a_j > 0 this is the vanishing of the first m+1 Legendre moments of E(0, t).  Solved with a semismooth
        // (active-set) Newton step on min(sigma a_j, -g_j) = 0, with a Jacobian dg/da carried from substep to substep and
        // slab to slab, initialized as sigma W (W the normalized Bernstein Gram matrix: the field responding locally in time
        // to the emitted density, with gain sigma estimated from the mean) and refined by Broyden updates.
        std::vector<double> gaussPoints, gaussWeights;
        {
          CubatureFactory cubatureFactory;
          auto cubature = cubatureFactory.create(CellTopology::line(), std::vector<int>{2 * sclEmissionQuadraturePoints - 1});
          auto refPoints = cubature->allocateCubaturePoints();
          auto refWeights = cubature->allocateCubatureWeights();
          cubature->getCubature(refPoints, refWeights);
          auto refPointsFC  = FieldContainerConversion<double>::fromTensorPoints(refPoints);
          auto refWeightsFC = FieldContainerConversion<double>::fromTensorData(refWeights);
          for (int i=0; i<refPointsFC.dimension(0); i++) { gaussPoints.push_back(refPointsFC(i,0)); gaussWeights.push_back(refWeightsFC(i)); }
        }
        struct SCLProfileState
        {
          std::vector<double> J;                  // dg/d(a/n0), row major; empty until the gain is first estimated
          bool havePrevious = false;              // previous substep's (a, g) available (within this slab)
          std::vector<double> aPrevious, gPrevious;
        };
        auto profileState = std::make_shared<SCLProfileState>();
        auto profile = emissionProfile;
        form->setSubstepConfigurationCallback([=](VlasovSpaceTimeFormulation &slabForm, int substepOrdinal) -> bool {
          if (substepOrdinal == 0) profileState->havePrevious = false;
          const int m = profile->degree();
          const int N = m + 1;
          FunctionPtr E = Function::solution(slabForm.configurationSpaceFormulation()->E_fieldVar(0), slabForm.configurationSpaceSolution());
          const double t0Slab = slabForm.t0(), t1Slab = slabForm.t1();
          std::vector<double> g(N, 0.0);
          double maxAbsE = 0;
          for (size_t q=0; q<gaussPoints.size(); q++)
          {
            const double s = 0.5 * (gaussPoints[q] + 1.0);
            const double Eq = E->evaluate(slabForm.configurationSpaceMesh(), x0, t0Slab + s * (t1Slab - t0Slab)); // collective
            maxAbsE = std::max(maxAbsE, std::abs(Eq));
            for (int j=0; j<N; j++) g[j] += 0.5 * N * gaussWeights[q] * (Eq / E0) * EmissionTimeProfile::bernstein(m, j, s);
          }
          std::vector<double> a = profile->coefficients();
          for (double &aj : a) aj /= n0;
          const double tol = sclEmissionTolerance;
          bool converged = true;
          for (int j=0; j<N; j++) converged = converged && ((a[j] > 0) ? (std::abs(g[j]) < tol) : (g[j] < tol));

          // Jacobian: initialized from the gain of the mean, then Broyden updates (accepted only for appreciable changes in
          // a, since g also moves as f relaxes independently of a, and only if the mean-mode gain stays within a factor of
          // four of its previous value and negative).
          auto meanGain = [N](const std::vector<double> &J) { double sum = 0; for (double Jij : J) sum += Jij; return sum / N; };
          if (profileState->havePrevious)
          {
            std::vector<double> da(N), dg(N);
            double daNorm2 = 0, aMax = 1e-3;
            for (int j=0; j<N; j++)
            {
              da[j] = a[j] - profileState->aPrevious[j];
              dg[j] = g[j] - profileState->gPrevious[j];
              daNorm2 += da[j] * da[j];
              aMax = std::max(aMax, std::max(a[j], profileState->aPrevious[j]));
            }
            if (std::sqrt(daNorm2) > 0.02 * aMax)
            {
              std::vector<double> JNew = profileState->J;
              if (JNew.empty())
              {
                double daSum = 0, dgSum = 0;
                for (int j=0; j<N; j++) { daSum += da[j]; dgSum += dg[j]; }
                const double sigma = (std::abs(daSum) > 0) ? dgSum / daSum : 0.0;
                if (sigma < 0)
                {
                  JNew.assign(N * N, 0.0);
                  for (int i=0; i<N; i++)
                    for (int j=0; j<N; j++)
                    {
                      // (m+1) int_0^1 B_i B_j ds = (m+1) C(m,i) C(m,j) / ((2m+1) C(2m, i+j))
                      auto binomial = [](int n, int k) { double b = 1; for (int l=1; l<=k; l++) b = b * (n - k + l) / l; return b; };
                      JNew[i*N+j] = sigma * N * binomial(m,i) * binomial(m,j) / ((2*m + 1) * binomial(2*m, i+j));
                    }
                }
              }
              else
              {
                for (int i=0; i<N; i++)
                {
                  double predicted = 0;
                  for (int j=0; j<N; j++) predicted += JNew[i*N+j] * da[j];
                  for (int j=0; j<N; j++) JNew[i*N+j] += (dg[i] - predicted) * da[j] / daNorm2;
                }
                const double oldGain = meanGain(profileState->J), newGain = meanGain(JNew);
                if (!((newGain < 0) && (newGain > 4.0 * oldGain) && (newGain < 0.25 * oldGain))) JNew = profileState->J;
              }
              profileState->J = JNew;
            }
          }
          profileState->havePrevious = true;
          profileState->aPrevious = a;
          profileState->gPrevious = g;

          std::vector<double> aNext = a;
          std::vector<bool> active(N, false);
          if (!converged)
          {
            if (profileState->J.empty())
            {
              // no gain yet: a proportional step.  g_j is 1 in vacuum (emit more), 0 when space-charge limited, and
              // negative when the field has reversed (emit less).
              for (int j=0; j<N; j++) aNext[j] = ((a[j] > 0) ? a[j] : 1e-3) * (1.0 + g[j]);
            }
            else
            {
              // Newton step for the complementarity problem, with the linear model g' = g + J (a' - a): the linear
              // complementarity problem a' >= 0, g' <= 0, a'_j g'_j = 0, solved exactly by trying each active set
              // (2^N of them; N <= 3 or so).  Choosing the active set from the sign of the unconstrained step instead
              // can fix the wrong coefficient at zero: the field responds mainly to the slab's total emitted charge,
              // so J is nearly singular and the unconstrained step can make an early coefficient negative even when
              // the solution needs it positive (and a late one zero).
              auto solveFree = [&](const std::vector<bool> &isActive, std::vector<double> &aTrial) -> bool {
                // solve J_FF a'_F = J_FF a_F - g_F + J_FA a_A, with a'_A = 0
                std::vector<int> F;
                for (int j=0; j<N; j++) if (!isActive[j]) F.push_back(j);
                const int nF = int(F.size());
                aTrial.assign(N, 0.0);
                if (nF == 0) return true;
                std::vector<double> A(nF * nF), b(nF);
                for (int r=0; r<nF; r++)
                {
                  const int i = F[r];
                  b[r] = -g[i];
                  for (int j=0; j<N; j++) b[r] += profileState->J[i*N+j] * a[j];
                  for (int c=0; c<nF; c++) A[r*nF+c] = profileState->J[i*N+F[c]];
                }
                // Gaussian elimination with partial pivoting
                for (int k=0; k<nF; k++)
                {
                  int pivot = k;
                  for (int r=k+1; r<nF; r++) if (std::abs(A[r*nF+k]) > std::abs(A[pivot*nF+k])) pivot = r;
                  if (std::abs(A[pivot*nF+k]) < 1e-14) return false;
                  for (int c=0; c<nF; c++) std::swap(A[k*nF+c], A[pivot*nF+c]);
                  std::swap(b[k], b[pivot]);
                  for (int r=k+1; r<nF; r++)
                  {
                    const double factor = A[r*nF+k] / A[k*nF+k];
                    for (int c=k; c<nF; c++) A[r*nF+c] -= factor * A[k*nF+c];
                    b[r] -= factor * b[k];
                  }
                }
                std::vector<double> x(nF);
                for (int r=nF-1; r>=0; r--)
                {
                  double sum = b[r];
                  for (int c=r+1; c<nF; c++) sum -= A[r*nF+c] * x[c];
                  x[r] = sum / A[r*nF+r];
                }
                for (int r=0; r<nF; r++) aTrial[F[r]] = x[r];
                return true;
              };
              // among the active sets satisfying the linearized conditions (there is one when -J is a P-matrix), take
              // the one with the smallest step; if none does (J inaccurate), the one with the smallest violation
              std::vector<double> best;
              double bestViolation = std::numeric_limits<double>::max(), bestStep = std::numeric_limits<double>::max();
              const double scale = std::max(1e-3, *std::max_element(a.begin(), a.end()));
              for (int mask=0; mask<(1<<N); mask++)
              {
                std::vector<bool> isActive(N);
                for (int j=0; j<N; j++) isActive[j] = (mask >> j) & 1;
                std::vector<double> aTrial;
                if (!solveFree(isActive, aTrial)) continue;
                double violation = 0, step = 0;
                for (int j=0; j<N; j++)
                {
                  double gPredicted = g[j];
                  for (int k=0; k<N; k++) gPredicted += profileState->J[j*N+k] * (aTrial[k] - a[k]);
                  if (isActive[j]) violation = std::max(violation, gPredicted);        // need g'_j <= 0
                  else             violation = std::max(violation, -aTrial[j] / scale); // need a'_j >= 0
                  step += (aTrial[j] - a[j]) * (aTrial[j] - a[j]);
                }
                const bool better = (violation <= 1e-12) ? ((bestViolation > 1e-12) || (step < bestStep))
                                                         : ((bestViolation > 1e-12) && (violation < bestViolation));
                if (better)
                {
                  best = aTrial; bestViolation = violation; bestStep = step;
                  active = isActive;
                }
              }
              std::vector<double> da(N, 0.0);
              if (!best.empty()) for (int j=0; j<N; j++) da[j] = std::max(0.0, best[j]) - a[j];
              // Step limits as in the constant-in-slab controller: a coefficient can rise by up to half of n0 (or double)
              // per substep; never negative.
              for (int j=0; j<N; j++)
              {
                const double aBase = (a[j] > 0) ? a[j] : 1e-3;
                aNext[j] = std::max(0.0, std::min(a[j] + std::max(aBase, 0.5), a[j] + da[j]));
              }
            }
            std::vector<double> aNextScaled = aNext;
            for (double &aj : aNextScaled) aj *= n0;
            profile->setCoefficients(aNextScaled);
          }
          if (rank == 0)
          {
            std::cout << "space-charge-limited emission (degree " << m << "): substep " << substepOrdinal+1 << ", g =";
            for (int j=0; j<N; j++) std::cout << " " << g[j];
            std::cout << ", max |E(0,t)|/|E0| = " << maxAbsE / std::abs(E0) << ", n_emit coefficients/n0 =";
            for (int j=0; j<N; j++) std::cout << " " << a[j];
            if (!converged)
            {
              std::cout << " ->";
              for (int j=0; j<N; j++) std::cout << " " << aNext[j] << (active[j] ? "*" : "");
            }
            std::cout << ", dg/dn = ";
            if (profileState->J.empty()) std::cout << "(not yet estimated)";
            else for (int i=0; i<N*N; i++) std::cout << (i == 0 ? "[" : (i % N == 0 ? "; " : " ")) << profileState->J[i] << (i == N*N-1 ? "]" : "");
            std::cout << " per n0"
                      << (converged ? " (converged)\n" : "\n");
          }
          return converged;
        });
      }
      else
      form->setSubstepConfigurationCallback([=](VlasovSpaceTimeFormulation &slabForm, int substepOrdinal) -> bool {
        if (substepOrdinal == 0) sclState->havePrevious = false; // the slope is kept across slabs
        FunctionPtr E = Function::solution(slabForm.configurationSpaceFormulation()->E_fieldVar(0), slabForm.configurationSpaceSolution());
        const double tEval = std::nextafter(slabForm.t1(), slabForm.t0());
        const double E_cathode = E->evaluate(slabForm.configurationSpaceMesh(), x0, tEval); // collective
        const double nCurrent = fM_u0->get_n();
        // Space-charge-limited emission is a complementarity condition: n_emit >= 0, the cathode field not accelerating
        // electrons off the cathode, and n_emit E(0) = 0.  So the state with emission (effectively) off and the field
        // reversed (E(0)/E0 < 0: more electron charge near the cathode than emission would put there) is converged, as is
        // E(0) = 0 with emission on.
        const double nOff = 1e-3 * n0; // emission at or below this is "off"
        const bool emissionOffFieldReversed = (nCurrent <= nOff) && (E_cathode / E0 < 0);
        const bool converged = (std::abs(E_cathode) < sclEmissionTolerance * std::abs(E0)) || emissionOffFieldReversed;

        // slope estimate from the last two substeps, if the density changed appreciably between them
        if (sclState->havePrevious)
        {
          // Accept a slope only from an appreciable change in n_emit (E(0) also moves as f relaxes, independent of n_emit, so
          // small changes give noise-dominated estimates), and, once a slope is known, only within a factor of four of it.
          const double dn = nCurrent - sclState->nPrevious;
          if (std::abs(dn) > 0.02 * std::max(std::max(nCurrent, sclState->nPrevious), 1e-3 * n0))
          {
            const double measuredSlope = (E_cathode - sclState->EPrevious) / dn;
            const bool plausible = (measuredSlope > 0) &&
                                   ((sclState->slope <= 0) || ((measuredSlope > 0.25 * sclState->slope) && (measuredSlope < 4.0 * sclState->slope)));
            if (plausible) sclState->slope = measuredSlope;
          }
        }
        sclState->havePrevious = true;
        sclState->nPrevious = nCurrent;
        sclState->EPrevious = E_cathode;

        // E0 < 0; too little emitted charge leaves E(0) < 0 (same sign as E0), too much reverses it.
        double nNext;
        const double nBase = (nCurrent > 0) ? nCurrent : 1e-3 * n0;
        if (sclState->slope > 0)
        {
          nNext = nCurrent - E_cathode / sclState->slope;
        }
        else
        {
          // no slope yet: a proportional step.  E(0)/E0 is 1 in vacuum (emit more), 0 when space-charge limited, and negative
          // when the cathode field has reversed (emit less).
          nNext = nBase * (1.0 + E_cathode / E0);
        }
        // Step limits, relative to n0 rather than to the current density (which can be far below its target): the density
        // can fall by half, or to the "off" level nOff, per substep, or rise by up to half of n0; never negative.  (A
        // geometric limit alone, e.g. a factor of ten per substep, lets the density collapse to 1e-30 n0 and below over a
        // few slabs while the field stays reversed; from there it cannot restart.)
        const double nMax = nCurrent + std::max(nBase, 0.5 * n0);
        const double nMin = std::min(nCurrent, std::max(0.5 * nCurrent, nOff));
        nNext = std::max(nMin, std::min(nMax, nNext));
        if (converged) nNext = nCurrent;
        fM_u0->set_n(nNext);
        if (rank == 0)
        {
          std::cout << "space-charge-limited emission: substep " << substepOrdinal+1 << ", E(0) = " << E_cathode
                    << " (" << E_cathode / E0 << " E0), n_emit = " << nCurrent / n0 << " n0 -> " << nNext / n0 << " n0"
                    << ", dE(0)/dn = " << ((sclState->slope > 0) ? sclState->slope * n0 : 0.0) << " per n0"
                    << (converged ? " (converged)\n" : "\n");
        }
        return converged;
      });
    }
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    //  form->vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
    
    FunctionPtr potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
    FunctionPtr density = Teuchos::null;
    FunctionPtr velocity = Teuchos::null;
    FunctionPtr velocity_z = Teuchos::null;
    FunctionPtr Jx = Teuchos::null;
    FunctionPtr Jz = Teuchos::null;
    FunctionPtr vx_rel_mean = Teuchos::null;
    FunctionPtr vz_rel_mean = Teuchos::null;
    FunctionPtr vx_rel_mean_clipped = Teuchos::null;
    FunctionPtr vz_rel_mean_clipped = Teuchos::null;
    
    const int speciesOrdinal = 0;
    auto f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
        
    if (!useAdaptivity)
    {
      for (int timeSlabOrdinal=0; timeSlabOrdinal<numTimeSlabs; timeSlabOrdinal++)
      {
        if (rank == 0)
        {
          std::cout << "*******************************************************************\n";
          std::cout << "*                     Starting Time Slab " << timeSlabOrdinal;
          if      (timeSlabOrdinal < 10)   std::cout << "                        *\n";
          else if (timeSlabOrdinal < 100)  std::cout << "                       *\n";
          else if (timeSlabOrdinal < 1000) std::cout << "                      *\n";
          else                             std::cout << "                     *\n";
          std::cout << "*******************************************************************\n";
        }
        auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
        if (numVlasovDofs > maxDofsForDirectSolve)
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
          }
          form->setUseGeometricMultigrid(true);
        }
        else
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
          }
          form->setUseGeometricMultigrid(false);
        }
        
        auto phase0SlabSolveStart = phase0.now();
        form->solve();
        {
          auto fields = phase0MeshStats(form);
          fields.push_back({"slab", std::to_string(timeSlabOrdinal)});
          fields.push_back({"memory_hwm_gb", phase0DoubleString(memoryHighWaterMarkInGigabytes())});
          phase0.logDuration("slab_solve", phase0.elapsedSince(phase0SlabSolveStart), fields);
        }
        
        if (rank == 0)
        {
          double memoryHWM_GB = memoryHighWaterMarkInGigabytes();
          std::cout << "Time slab " << timeSlabOrdinal << ", memory HWM: " << memoryHWM_GB << " GB\n";
        }
        
        {
          int quadratureEnrichment = 5;
          const double f_min = f_soln->minimumValue(form->vlasovMesh(), quadratureEnrichment);
          const double f_max = f_soln->maximumValue(form->vlasovMesh(), quadratureEnrichment);
          
          if (rank == 0)
          {
            std::cout << "Time slab " << timeSlabOrdinal << ", f min: " << f_min << std::endl;
            std::cout << "Time slab " << timeSlabOrdinal << ", f max: " << f_max << std::endl;
          }
        }
        
        if (rank == 0)
        {
          std::cout << "Updating tensor-native moment diagnostics.\n";
        }
        
        auto phase0MomentUpdateStart = phase0.now();
        const auto &configurationCellIDSet = form->configurationSpaceMesh()->cellIDsInPartition();
        std::vector<GlobalIndexType> configurationCellIDs(configurationCellIDSet.begin(), configurationCellIDSet.end());
        const int phase8VelocityQuadratureDegree = (phase8MomentVelocityQuadratureDegree > 0) ? phase8MomentVelocityQuadratureDegree
                                                                                               : defaultVlasovMomentVelocityQuadratureDegree1D2V(form->vlasovSolution());
        auto vlasovMoments = computeVlasovMomentsByPhaseSpaceCell1D2V(configurationCellIDs,
                                                                      form->configurationSpaceMesh(),
                                                                      form->vlasovSolution(),
                                                                      form->f(speciesOrdinal),
                                                                      phase8VelocityQuadratureDegree,
                                                                      true,
                                                                      cScaled);
        const int numTensorEntries = vlasovMoments.numConfigurationCells * vlasovMoments.numPointsPerConfigurationCell;
        std::vector<double> n0Values(numTensorEntries, n0);
        auto tensorMomentData = TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(vlasovMoments.numConfigurationCells,
                                                                                  vlasovMoments.numPointsPerConfigurationCell,
                                                                                  {q},
                                                                                  eps0,
                                                                                  {vlasovMoments.moments},
                                                                                  n0Values);
        auto rhoRHS = TensorMaxwellRHSFunctionAdapter::rhoRHSFunction(tensorMomentData);
        density = (eps0 / q) * rhoRHS + Function::constant(n0);
        Jx = TensorMaxwellRHSFunctionAdapter::JxFunction(tensorMomentData);
        Jz = TensorMaxwellRHSFunctionAdapter::JzFunction(tensorMomentData);
        vx_rel_mean = Jx / (q * density);
        vz_rel_mean = Jz / (q * density);
        velocity = vx_rel_mean;
        velocity_z = vz_rel_mean;
        vx_rel_mean_clipped = vx_rel_mean;
        vz_rel_mean_clipped = vz_rel_mean;

        phase0.logDuration("moment_update", phase0.elapsedSince(phase0MomentUpdateStart),
                           {{"slab", std::to_string(timeSlabOrdinal)},
                            {"moment_backend", "tensor_native"},
                            {"local_config_cells", std::to_string(vlasovMoments.numConfigurationCells)},
                            {"points_per_cell", std::to_string(vlasovMoments.numPointsPerConfigurationCell)},
                            {"velocity_points", std::to_string(vlasovMoments.numVelocityPoints)},
                            {"velocity_quadrature_degree", std::to_string(phase8VelocityQuadratureDegree)}});
        
        if (rank == 0)
        {
          std::cout << "Tensor-native moment diagnostics updated.\n";
        }
        
        if (saveViz)
        {
          std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
          std::string maxwellSlabStr = maxwellSS.str() + "_slab" + to_string(timeSlabOrdinal);
          
          //            auto vlasovVizSlab  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
          auto maxwellVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), maxwellSlabStr, directoryPath) );
          
          if (rank == 0)
          {
            std::cout << "Exporting Maxwell solution.\n";
          }
          double vizTimeValue = 0.0;
          int num1DPoints = 20; // hardcoded, fine resolution for now.
          //            maxwellVizSlab->exportSolution(form->configurationSpaceSolution(),{density}, {"n_e"}, vizTimeValue, num1DPoints);
          //            maxwellVizSlab->exportSolution(form->configurationSpaceSolution());
          // The tensor-native moment functions (density, Jx, Jz, and the mean velocities derived from them) are defined
          // only at the configuration-space cubature points of the rank-local cells they were sampled on, so the exporter
          // cannot evaluate them at its own points; export the Maxwell solution fields only.
          maxwellVizSlab->exportSolution(form->configurationSpaceSolution(), {cellIDFunction}, {"cell ID (Maxwell)"}, vizTimeValue, num1DPoints);
          //            vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"});
          
          const bool exportVlasov = false; // PhaseSpaceVisualization is not working in 1D1T2V -- unclear where the issue lies.
          if (exportVlasov)
          {
            if (rank == 0)
            {
              std::cout << "Exporting Vlasov solution.\n";
            }
            PhaseSpaceVisualization vlasovViz( form->vlasovSolution(), form->f(0), vlasovSlabStr, directoryPath);
            vlasovViz.exportTimeStep(vizTimeValue, num1DPoints);
          }
        }
        
        if (densityDiagnostic)
        {
          const std::string densityPath = outputDir + "/density_slab" + to_string(timeSlabOrdinal) + ".csv";
          writeDensityDiagnostic(densityPath, form, speciesOrdinal, densityDiagnosticPointsPerDim, phase8VelocityQuadratureDegree, cScaled, q);
          if (rank == 0) std::cout << "Wrote density diagnostic to " << densityPath << ".\n";
        }

        auto configMesh = form->configurationSpaceMesh();
        FunctionPtr E_soln = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
        
        // Evaluate just inside the slab end, rather than exactly at t1(), to avoid
        // floating-point / point-location boundary ambiguity at the time-slab boundary.
        const double tEval = std::nextafter(form->t1(), form->t0());
        double E_left = E_soln->evaluate(configMesh, x0, tEval); // when called with mesh arg, evaluate() locates the owning rank, evaluates there, and then MPI-communicates the result to all (without mesh arg, it will silently get a wrong (0.0) answer on some ranks!)

        if (accuracyDiagnostics)
        {
          // integrals over the slab (phase space for f; configuration space-time for E^2, Jx); all collective
          const int cubatureEnrichment = 2;
          const double fIntegral   = f_soln->integrate(form->vlasovMesh(), cubatureEnrichment);
          const double E2Integral  = (E_soln * E_soln)->integrate(configMesh, cubatureEnrichment);
          // the tensor-native moment functions are only defined at the configuration cubature points they were sampled on
          const double JxIntegral  = Jx->integrate(configMesh, 0);
          const double nIntegral   = density->integrate(configMesh, 0);
          if (rank == 0)
          {
            std::cout << std::setprecision(10);
            std::cout << "ACCURACY slab=" << timeSlabOrdinal << " int_f=" << fIntegral << " int_E2=" << E2Integral
                      << " int_Jx=" << JxIntegral << " int_n=" << nIntegral << " E0_end=" << E_left << std::endl;
            std::cout << std::setprecision(6);
          }
        }
        
//        {
//          // DEBUGGING:
//          std::cout << "On rank " << rank << ", E_left: " << E_left << std::endl;
//        }
        
        if (spaceChargeLimitedEmission)
        {
          if (rank == 0)
          {
            if (emissionProfile == Teuchos::null)
            {
              std::cout << "E(0) at final time is " << E_left << " (" << E_left / E0 << " E0); emitted density "
                        << fM_u0->get_n() / n0 << " n0 (space-charge-limited emission).\n";
            }
            else
            {
              // slab-start, slab-mean, and slab-end emitted density, and the slab-end field
              std::cout << "E(0) at final time is " << E_left << " (" << E_left / E0 << " E0); emitted density "
                        << emissionProfile->valueAtS(0.0) / n0 << " -> " << emissionProfile->valueAtS(1.0) / n0
                        << " n0 (mean " << emissionProfile->mean() / n0 << " n0; space-charge-limited emission, degree " << emissionProfile->degree() << ").\n";
            }
          }
        }
        else
        {
        // reduce the inflow current when E_left / E0 gets small -- want to avoid/mitigate virtual cathode creation.
        const double lowerThreshold = taperLowerThreshold; // turn off once we get here
        const double upperThreshold = taperUpperThreshold; // start tapering here
        double ratio = E_left / E0; // E0 is negative -> if E_left is negative, ratio will be positive
        if (!taperInjection)
        {
          ratio = 1.0;
        }
        else if (ratio < lowerThreshold)
        {
          ratio = 0;
        }
        else if (ratio > upperThreshold)
        {
          ratio = 1;
        }
        else
        {
          // shift and scale so that ratio will be 1 at upperThreshold, and 0 at lower
          ratio = (ratio - lowerThreshold) / (upperThreshold - lowerThreshold);
        }
        double n = n0 * ratio;
        
        fM_u0->set_n(n);
        
        if (rank == 0)
        {
          double percent = int(ratio * 10000) / 100.0; // keep two digits after the decimal point
          double upperThresholdPercent = int(upperThreshold * 100);
          double lowerThresholdPercent = int(lowerThreshold * 100);
          std::cout << "E(0) at final time is " << E_left << "; setting injection current to " << percent << "%";
          std::cout << (taperInjection ? ".\n" : " (injection taper disabled).\n");
          double EPercent = int(E_left / E0 * 100);
          if (!taperInjection)
          {
            // no taper messages
          }
          else if (ratio >= 1)
          {
            std::cout << "Will begin reducing injection current once E(0) is at " << upperThresholdPercent << "% of initial E(0) value (now at ";
            std::cout << EPercent << "%)\n";
          }
          else if (ratio > 0)
          {
            std::cout << "Will turn off injection once E(0) is at " << lowerThresholdPercent << "% of initial E(0) value (now at ";
            std::cout << EPercent << "%)\n";
          }
          else
          {
            std::cout << "Injection is off: E(0) is below " << lowerThresholdPercent << "% of initial E(0) value (now at " << EPercent << "%)\n";
          }
        }
        } // !spaceChargeLimitedEmission
        
        // clear solution-backed diagnostic functions before advancing slabs to avoid stale listener / solution lifetimes
        Jx = Teuchos::null;
        Jz = Teuchos::null;
        vx_rel_mean = Teuchos::null;
        vz_rel_mean = Teuchos::null;
        vx_rel_mean_clipped = Teuchos::null;
        vz_rel_mean_clipped = Teuchos::null;
        density = Teuchos::null;
        velocity = Teuchos::null;
        velocity_z = Teuchos::null;
        potential = Teuchos::null;
        f_soln = Teuchos::null;

        // next time slab
        auto phase0NextTimeSlabStart = phase0.now();
        if (emissionProfile != Teuchos::null) emissionProfile->advanceToNextSlab(form->t1(), form->t1() + tWidth);
        form = form->nextTimeSlab(tWidth, tElements);
        phase0.logDuration("next_time_slab", phase0.elapsedSince(phase0NextTimeSlabStart),
                           {{"completed_slab", std::to_string(timeSlabOrdinal)}});
        if (phase0.enabled()) form->vlasovBF()->setOptimalTestTimingCallback(phase0OptimalTestTimingCallback); // the substep profile callback is carried by nextTimeSlab()
        
        {
          // solutions have been recreated for the new slab.  We need to update anything that depends on the solution.
          potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
          density = Teuchos::null;
          velocity = Teuchos::null;
          velocity_z = Teuchos::null;
          Jx = Teuchos::null;
          Jz = Teuchos::null;
          vx_rel_mean = Teuchos::null;
          vz_rel_mean = Teuchos::null;
          vx_rel_mean_clipped = Teuchos::null;
          vz_rel_mean_clipped = Teuchos::null;
          f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
        }
      }
    }
    else // useAdaptivity = true
    {
      for (int timeSlabOrdinal=0; timeSlabOrdinal<numTimeSlabs; timeSlabOrdinal++)
      {
        Teuchos::RCP<HDF5Exporter> maxwellVizSlab, maxwellVizSlabFinalRef;
        
        Teuchos::RCP<PhaseSpaceVisualization> vlasovVizSlab, vlasovVizSlabFinalRef;
        
        const bool exportVlasov = false; // PhaseSpaceVisualization is not working in 1D1T2V -- unclear where the issue lies.
        
        if (saveViz)
        {
          std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
          std::string maxwellSlabStr = maxwellSS.str() + "_slab" + to_string(timeSlabOrdinal);
          
          if (exportVlasov)
          {
            vlasovVizSlab = Teuchos::rcp( new PhaseSpaceVisualization( form->vlasovSolution(), form->f(0), vlasovSlabStr, directoryPath));
          }
          maxwellVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), maxwellSlabStr, directoryPath) );
        }
        
        if (rank == 0)
        {
          std::cout << "*******************************************************************\n";
          std::cout << "*                     Starting Time Slab " << timeSlabOrdinal;
          if      (timeSlabOrdinal < 10)   std::cout << "                        *\n";
          else if (timeSlabOrdinal < 100)  std::cout << "                       *\n";
          else if (timeSlabOrdinal < 1000) std::cout << "                      *\n";
          else                             std::cout << "                     *\n";
          std::cout << "*******************************************************************\n";
        }
        RefinementStrategyPtr refStrategy;
        
        if (doerflerTarget <= 0)
        {
          refStrategy = Teuchos::rcp(new RefinementStrategy(form->vlasovSolution(), refinementThreshold));
          refStrategy->setMaxElements(maxElements); // allow unrefinements
        }
        else
        {
          ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form->vlasovSolution());
          
          refStrategy = Teuchos::rcp(new DoerflerRefinementStrategy(errorIndicator, doerflerTarget) );
        }
        
        double energyError;
        auto energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form->vlasovSolution());
        
        if (timeSlabOrdinal == numTimeSlabs - 1)
        {
          // final time slab
          maxRefinements = maxRefinementsFinalSlab;
          relErrorTarget = relErrorTargetFinalSlab;
        }
        
        for (int refinementNumber=0; refinementNumber<=maxRefinements; refinementNumber++)
        {
          auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
          if (numVlasovDofs > maxDofsForDirectSolve)
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
            }
            form->setUseGeometricMultigrid(true);
          }
          else
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
            }
            form->setUseGeometricMultigrid(false);
          }
          if (refinementNumber < maxRefinements)
          {
            // at least one refinement left; use coarse tolerances
            form->setIterativeSolveTolerances(cgTolsCoarse);
            form->setIterativeSolveMaxIterations(coarseIterativeSolveMaxIters);
            form->setFixedPointTolerance(coarseFixedPointTol);
            form->setFixedPointMinSubsteps(coarseMinSubsteps);
            form->setFixedPointMaxSubsteps(coarseMaxSubsteps);
          }
          else
          {
            form->setIterativeSolveTolerances(cgTols);
            form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
            form->setFixedPointTolerance(fixedPointTol);
            form->setFixedPointMinSubsteps(minSubsteps);
            form->setFixedPointMaxSubsteps(maxSubsteps);
          }
          
          auto phase0AdaptiveSolveStart = phase0.now();
          form->solve();
          {
            auto fields = phase0MeshStats(form);
            fields.push_back({"slab", std::to_string(timeSlabOrdinal)});
            fields.push_back({"refinement", std::to_string(refinementNumber)});
            fields.push_back({"memory_hwm_gb", phase0DoubleString(memoryHighWaterMarkInGigabytes())});
            phase0.logDuration("adaptive_slab_solve", phase0.elapsedSince(phase0AdaptiveSolveStart), fields);
          }
          
          if (rank == 0)
          {
            double memoryHWM_GB = memoryHighWaterMarkInGigabytes();
            std::cout << "Time slab " << timeSlabOrdinal << ", refinement " << refinementNumber << ", memory HWM (rank 0): " << memoryHWM_GB << " GB\n";
          }
          
          auto phase0AdaptiveMomentUpdateStart = phase0.now();
          const auto &configurationCellIDSet = form->configurationSpaceMesh()->cellIDsInPartition();
          std::vector<GlobalIndexType> configurationCellIDs(configurationCellIDSet.begin(), configurationCellIDSet.end());
          const int phase8VelocityQuadratureDegree = (phase8MomentVelocityQuadratureDegree > 0) ? phase8MomentVelocityQuadratureDegree
                                                                                                 : defaultVlasovMomentVelocityQuadratureDegree1D2V(form->vlasovSolution());
          auto vlasovMoments = computeVlasovMomentsByPhaseSpaceCell1D2V(configurationCellIDs,
                                                                        form->configurationSpaceMesh(),
                                                                        form->vlasovSolution(),
                                                                        form->f(speciesOrdinal),
                                                                        phase8VelocityQuadratureDegree,
                                                                        true,
                                                                        cScaled);
          const int numTensorEntries = vlasovMoments.numConfigurationCells * vlasovMoments.numPointsPerConfigurationCell;
          std::vector<double> n0Values(numTensorEntries, n0);
          auto tensorMomentData = TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(vlasovMoments.numConfigurationCells,
                                                                                    vlasovMoments.numPointsPerConfigurationCell,
                                                                                    {q},
                                                                                    eps0,
                                                                                    {vlasovMoments.moments},
                                                                                    n0Values);
          auto rhoRHS = TensorMaxwellRHSFunctionAdapter::rhoRHSFunction(tensorMomentData);
          density = (eps0 / q) * rhoRHS + Function::constant(n0);
          Jx = TensorMaxwellRHSFunctionAdapter::JxFunction(tensorMomentData);
          Jz = TensorMaxwellRHSFunctionAdapter::JzFunction(tensorMomentData);
          vx_rel_mean = Jx / (q * density);
          vz_rel_mean = Jz / (q * density);
          velocity = vx_rel_mean;
          velocity_z = vz_rel_mean;
          vx_rel_mean_clipped = vx_rel_mean;
          vz_rel_mean_clipped = vz_rel_mean;
          phase0.logDuration("adaptive_moment_update", phase0.elapsedSince(phase0AdaptiveMomentUpdateStart),
                             {{"slab", std::to_string(timeSlabOrdinal)},
                              {"refinement", std::to_string(refinementNumber)},
                              {"moment_backend", "tensor_native"},
                              {"local_config_cells", std::to_string(vlasovMoments.numConfigurationCells)},
                              {"points_per_cell", std::to_string(vlasovMoments.numPointsPerConfigurationCell)},
                              {"velocity_points", std::to_string(vlasovMoments.numVelocityPoints)},
                              {"velocity_quadrature_degree", std::to_string(phase8VelocityQuadratureDegree)}});
          
          energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
          
          RieszRepPtr f_rieszRep;
          
          {
            std::map<int, FunctionPtr> f_map;
            auto f_var = form->f(0); // 0: species ordinal
            f_map[f_var->ID()] = Function::solution(f_var, form->vlasovSolution());
            auto f_functional = form->vlasovBF()->testFunctional(f_map); // bilinear form evaluated at the f solution
            f_rieszRep  = RieszRep::rieszRep(form->vlasovMesh(), form->vlasovSolution()->ip(),  f_functional);
          }
          
          RieszRepPtr tn_rieszRep;
          {
            std::map<int, FunctionPtr> tn_map;
            auto tn_var = form->t_n_hat(0); // 0: species ordinal
            const bool weightFluxesBySideParity = true;
            tn_map[tn_var->ID()] = Function::solution(tn_var, form->vlasovSolution(), weightFluxesBySideParity);
            auto tn_functional = form->vlasovBF()->testFunctional(tn_map); // bilinear form evaluated at the tn solution
            tn_rieszRep = RieszRep::rieszRep(form->vlasovMesh(), form->vlasovSolution()->ip(), tn_functional);
          }
          double  f_norm =  f_rieszRep->getNorm();
          double tn_norm = tn_rieszRep->getNorm();
          double total_norm = sqrt(f_norm * f_norm + tn_norm * tn_norm);
          double relEnergyError = energyError / total_norm;
          
          bool errorTargetMet = (relEnergyError < relErrorTarget);
          
          if (rank == 0)
          {
            std::cout << "\n********************************************************************************\n";
            std::cout << "Time slab " << timeSlabOrdinal << ", ";
            std::cout << "solution after refinement " << refinementNumber << ": relative error " << relEnergyError * 100.0 << "%" << std::endl;
            if (errorTargetMet && (refinementNumber < maxRefinements))
            {
              std::cout << "Energy error target met; will perform one more solve with tightened tolerances.\n";
            }
            std::cout << "********************************************************************************\n\n";
          }
          
          if (errorTargetMet && (refinementNumber < maxRefinements))
          {
            form->setIterativeSolveTolerances(cgTols);
            form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
            form->setFixedPointTolerance(fixedPointTol);
            form->setFixedPointMinSubsteps(minSubsteps);
            form->setFixedPointMaxSubsteps(maxSubsteps);
            auto phase0AdaptiveResolveStart = phase0.now();
            form->solve();
            phase0.logDuration("adaptive_slab_resolve", phase0.elapsedSince(phase0AdaptiveResolveStart),
                               {{"slab", std::to_string(timeSlabOrdinal)},
                                {"refinement", std::to_string(refinementNumber)}});
            
            energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
            f_rieszRep->computeRieszRep(); // recompute now that we've solved to tighter tolerance
            f_norm = f_rieszRep->getNorm();
            tn_rieszRep->computeRieszRep();
            tn_norm = tn_rieszRep->getNorm();
            total_norm = sqrt(f_norm * f_norm + tn_norm * tn_norm);
            relEnergyError = energyError / total_norm;
            errorTargetMet = (relEnergyError < relErrorTarget);
            
            if (rank == 0)
            {
              std::cout << "\n*******************************************************************\n";
              std::cout << "After re-solve, relative error " << relEnergyError * 100.0 << "%" << std::endl;
              if ( !errorTargetMet )
              {
                std::cout << "New error does not meet target; will not yet exit refinement loop early.\n";
              }
              std::cout << "*******************************************************************\n\n";
            }
          }
                    
          if (saveViz)
          {
            // (moment-derived functions cannot be evaluated by the exporter; see the uniform-mesh export above)
            maxwellVizSlab->exportSolution(form->configurationSpaceSolution(), {cellIDFunction}, {"cell ID (Maxwell)"}, refinementNumber);
            if (exportVlasov)
            {
              vlasovVizSlab->exportTimeStep(refinementNumber);
            }
          }
          
          if (!errorTargetMet && (refinementNumber < maxRefinements))
          {
            auto phase0AdaptiveRefinementStart = phase0.now();
            refStrategy->refine(rank==0);
            
            // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
            form->configurationSpaceMesh()->repartitionAndRebuild();
            phase0.logDuration("adaptive_refinement", phase0.elapsedSince(phase0AdaptiveRefinementStart),
                               {{"slab", std::to_string(timeSlabOrdinal)},
                                {"refinement", std::to_string(refinementNumber)}});
            auto maxwellElementCount = form->configurationSpaceMesh()->numActiveElements();
            auto maxwellDofCount     = form->configurationSpaceMesh()->numGlobalDofs();
            if (rank == 0)
            {
              std::cout << "Maxwell mesh has " << maxwellElementCount << " active elements and " << maxwellDofCount << " global degrees of freedom.\n";
            }
          }
          else
          {
            if (saveViz)
            {
              std::string  vlasovSlabStr = "finalRef" + vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
              std::string maxwellSlabStr = "finalRef" + maxwellSS.str() + "_slab" + to_string(timeSlabOrdinal);
              
              if (exportVlasov)
              {
                vlasovVizSlabFinalRef = Teuchos::rcp( new PhaseSpaceVisualization( form->vlasovSolution(), form->f(0), vlasovSlabStr, directoryPath));
              }
              maxwellVizSlabFinalRef = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), maxwellSlabStr, directoryPath) );
              
              const int refNumber = 0; // "time" for ParaView
              const int defaultNum1DPoints = 2 * H1Order;
              maxwellVizSlabFinalRef->exportSolution(form->configurationSpaceSolution(), {density,velocity,velocity_z,Jx,Jz,vx_rel_mean,vz_rel_mean,vx_rel_mean_clipped,vz_rel_mean_clipped,cellIDFunction}, {"n_e","ux_mean","uz_mean_raw","Jx","Jz","vx_rel_mean_signed","vz_rel_mean_signed","vx_rel_mean_clipped","vz_rel_mean_clipped","cell ID (Maxwell)"}, refNumber, defaultNum1DPoints);
              
              if (exportVlasov)
              {
                vlasovVizSlabFinalRef->exportTimeStep(refNumber, defaultNum1DPoints);
              }
            }
            break; // exits loop early if error target met
          }
        } // refinement loop
        
        if (timeSlabOrdinal < numTimeSlabs - 1)
        {
          // clear solution-backed diagnostic functions before advancing slabs to avoid stale listener / solution lifetimes
          Jx = Teuchos::null;
          Jz = Teuchos::null;
          vx_rel_mean = Teuchos::null;
          vz_rel_mean = Teuchos::null;
          vx_rel_mean_clipped = Teuchos::null;
          vz_rel_mean_clipped = Teuchos::null;
          density = Teuchos::null;
          velocity = Teuchos::null;
          velocity_z = Teuchos::null;
          potential = Teuchos::null;
          f_soln = Teuchos::null;

          // next time slab
          auto phase0NextTimeSlabStart = phase0.now();
          form = form->nextTimeSlab(tWidth, tElements);
          phase0.logDuration("next_time_slab", phase0.elapsedSince(phase0NextTimeSlabStart),
                             {{"completed_slab", std::to_string(timeSlabOrdinal)}});
          if (phase0.enabled()) form->vlasovBF()->setOptimalTestTimingCallback(phase0OptimalTestTimingCallback); // the substep profile callback is carried by nextTimeSlab()
          
          {
            // solutions have been recreated for the new slab.  We need to update anything that depends on the solution.
            potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
            density = Teuchos::null;
            velocity = Teuchos::null;
            velocity_z = Teuchos::null;
            Jx = Teuchos::null;
            Jz = Teuchos::null;
            vx_rel_mean = Teuchos::null;
            vz_rel_mean = Teuchos::null;
            vx_rel_mean_clipped = Teuchos::null;
            vz_rel_mean_clipped = Teuchos::null;
            f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
            
            bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
          }
          
          // refine the new slab to ensure resolution of the BC.
          auto phase0AdaptiveBCRefinementStart = phase0.now();
          refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form->configurationSpaceMesh()->repartitionAndRebuild();
          phase0.logDuration("adaptive_bc_refinement", phase0.elapsedSince(phase0AdaptiveBCRefinementStart),
                             {{"slab", std::to_string(timeSlabOrdinal + 1)}});
          if (rank == 0)
          {
            std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
          }
        }
      } // timeSlabOrdinal loop
      
    } // end time slab + useAdaptivity

    phase0.logDuration("total", phase0.elapsedSince(phase0TotalStart));
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
