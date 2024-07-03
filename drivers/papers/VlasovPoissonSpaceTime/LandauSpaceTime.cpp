//
// © 2016-2022 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MaxwellianFunction.h"
#include "MimeticVlasovSpaceTimeFormulation.h"
#include "MonomialFunctions.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "SpatialIntegralOfSpaceTimeSolution.h"
#include "TrigFunctions.h"
#include "VlasovSpaceTimeFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

using std::vector;
using std::pair;
using std::string;

class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function(0) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
      }
    }
  }
};

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const double x0, const double xWidth,
                                                              const double t0, const double tWidth,
                                                              const double v0, const double vWidth,
                                                              const int xElements, const int tElements, const int vElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const bool usePeriodicBCs, const bool useSerendipity,
                                                              const bool useMimeticDPG,
                                                              const int numUniformRefinements)
{
  Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME) );
  
  MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
  
  std::vector<double> temporalVertices(tElements + 1);
  for (int i=0; i<=tElements; i++)
  {
    const double t = double(i) / double(tElements) * tWidth + t0;
    temporalVertices[i] = t;
  }
  
  std::vector<double> velocitySpaceVertices_x(vElements+1);
  for (int i=0; i<=vElements; i++)
  {
    const double v = double(i) / double(vElements) * vWidth + v0;
    velocitySpaceVertices_x[i] = v;
  }
  
  std::vector<double> velocitySpaceVertices_y; // empty
  std::vector<double> velocitySpaceVertices_z; // empty
  
  Teuchos::RCP<VlasovSpaceTimeFormulation> form;
  
  if (!useMimeticDPG)
  {
    form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, useSerendipity, configForm, spatialMeshTopo,
                                                       temporalVertices,
                                                       velocitySpaceVertices_x,
                                                       velocitySpaceVertices_y,
                                                       velocitySpaceVertices_z,
                                                       H1Order, delta_k,
                                                       numUniformRefinements));
  }
  else
  {
    form = Teuchos::rcp(new MimeticVlasovSpaceTimeFormulation(q, m, useSerendipity, configForm, spatialMeshTopo,
                                                              temporalVertices,
                                                              velocitySpaceVertices_x,
                                                              velocitySpaceVertices_y,
                                                              velocitySpaceVertices_z,
                                                              H1Order, delta_k,
                                                              numUniformRefinements));
  }
  return form;
}

double roundToThreeSignificantDigits(double value) {
  if (value == 0.0) {
      return 0.0;
  }
  
  int precision = 3 - static_cast<int>(std::floor(std::log10(std::abs(value))) + 1);
  double roundedValue = std::round(value * std::pow(10, precision)) / std::pow(10, precision);
  return roundedValue;
}

vector<pair<double,double>> evaluateTemporalSolutionFunction(SolutionPtr temporalSolution, FunctionPtr temporalSolutionFunction,
                                                             double t0, double t1, double dt)
{
  auto mesh = temporalSolution->mesh();
  auto activeCells = mesh->getTopology()->getLocallyKnownActiveCellIndices(); // we assume mesh geometry is not distributed
  temporalSolution->importSolutionForOffRankCells(activeCells);
  
  // confirm that the mesh geometry is not distributed:
  const int numActiveCells = activeCells.size();
  TEUCHOS_TEST_FOR_EXCEPTION(mesh->numActiveElements() != numActiveCells, std::invalid_argument, "Error: not all active cell indices are locally known");
  
  const double tol = 1e-10;
  
  int rank = mesh->Comm()->MyPID();
  std::map<IndexType, vector<double> > pointsForCell;
  
  vector<double> allPoints;
  {
    double t = t0;
    int t_ordinal = 0;
    while (t <= t1 + tol)
    {
      allPoints.push_back(t);
      t_ordinal++;
      t = t0 + t_ordinal * dt;
    }
  }
  const int numPoints = allPoints.size();
  
  vector<pair<double,double>> fxnValues(numPoints);

  if (rank == 0)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(CellTopology::line(), 1);
    basisCache->setMesh(mesh);

    auto meshTopo = mesh->getTopology();
    
    FieldContainer<double> vertices(2,1); // (N,D)
    for (const auto &cellID : activeCells)
    {
      meshTopo->verticesForCell(vertices, cellID);
      double t_left  = vertices(0,0);
      double t_right = vertices(1,0);
      if (t_right != t1)
      {
        // rightmost cell owns its final point; intermediate cells do not
        t_right -= tol * 2;
      }

      // determine the first matching point that intersects the cell (if any)
      int t_firstOrdinal = (t_left - t0 + tol) / dt; // in allPoints
      if (allPoints[t_firstOrdinal] + tol < t_left)
      {
        // then go to the next point
        t_firstOrdinal = (t_left - t0 + dt + tol) / dt; // in allPoints
      }
      if (t_firstOrdinal >= numPoints) continue; // can't have any matches in this cell, then
      TEUCHOS_TEST_FOR_EXCEPTION(allPoints[t_firstOrdinal] + tol < t_left, std::logic_error, "logic error determining t_firstOrdinal");

      vector<double> points;

      double t = dt * t_firstOrdinal + t0;
      while (t < t_right + tol)
      {
        points.push_back(t);
        t += dt;
      }
      pointsForCell[cellID] = points;

      double refCellScaling = 2.0 / (t_right - t_left);
      int numPointsForCell = points.size();
      
      if (numPointsForCell == 0) continue;
      
      FieldContainer<double> refCellPoints(numPointsForCell,1);
      for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
      {
        const auto &point = points[pointOrdinal];
        double refPoint = (point - t_left) * refCellScaling - 1.0; // -1.0 is the image of t_left on the reference cell
        refCellPoints(pointOrdinal,0) = refPoint;
      }

      FieldContainer<double> cellFxnValues(1,numPointsForCell);
      auto physicalCellNodes = mesh->physicalCellNodesForCell(cellID);

      basisCache->setRefCellPointsFC(refCellPoints);
      basisCache->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
      temporalSolutionFunction->values(cellFxnValues, basisCache);

      for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
      {
        fxnValues[t_firstOrdinal + pointOrdinal] = {points[pointOrdinal],cellFxnValues(0,pointOrdinal)};
      }
    }
  }
  return fxnValues;
  
//  vector<pair<double,double>> fxnValues;
//  double t = t0;
//  while (t <= t1)
//  {
//    double value = temporalSolutionFunction->evaluate(mesh, t); // inefficient: does a mesh search for t, evaluates on the owner, and MPI-communicates it everywhere
//    fxnValues.push_back({t,value});
//    t += dt;
//  }
//  return fxnValues;
}

//! should only call on rank 0
void exportFourierValues(const string &EFieldMode1File, bool append, vector<pair<double,double>> &t_ln_E1_pairs)
{
  std::ofstream EFieldMode1FileStream;
  
  if (append)
  {
    EFieldMode1FileStream.open(EFieldMode1File, std::ios_base::app);
  }
  else
  {
    EFieldMode1FileStream.open(EFieldMode1File, std::ios::out);
    EFieldMode1FileStream << "t\tln(abs(b_1))\n";
  }
  EFieldMode1FileStream << setprecision(15);

  for (const auto & t_ln_b1 : t_ln_E1_pairs)
  {
    const auto &t     = t_ln_b1.first;
    const auto &ln_b1 = t_ln_b1.second;
    EFieldMode1FileStream << t     << "\t";
    EFieldMode1FileStream << ln_b1 << "\n";
  }
  EFieldMode1FileStream.close();
}

//! should only call on rank 0
void exportTimeSeriesValues(const string &timeSeriesFile, bool append, vector<pair<double,double>> &t_value_pairs, const std::string &valueHeader)
{
  std::ofstream timeSeriesFileStream;
  
  if (append)
  {
    timeSeriesFileStream.open(timeSeriesFile, std::ios_base::app);
  }
  else
  {
    timeSeriesFileStream.open(timeSeriesFile, std::ios::out);
    timeSeriesFileStream << "t\t" << valueHeader << "\n";
  }
  timeSeriesFileStream << setprecision(15);

  for (const auto & t_value : t_value_pairs)
  {
    const auto &t     = t_value.first;
    const auto &value = t_value.second;
    timeSeriesFileStream << t     << "\t";
    timeSeriesFileStream << value << "\n";
  }
  timeSeriesFileStream.close();
}

vector<pair<double,double>> getPeakValues(double &previousFourierValue, double &previousTime, bool &fourierValueIsIncreasing, const vector<pair<double,double>> &fourierValues)
{
  vector<pair<double,double>> peakValues;
  
  for (auto entry : fourierValues)
  {
    const double & t            = entry.first;
    const double & fourierValue = entry.second;
    
    if (fourierValueIsIncreasing && (fourierValue < previousFourierValue))
    {
      // reversal: previous is a local maximum:
      peakValues.push_back({previousTime,previousFourierValue});
    }
    fourierValueIsIncreasing = (fourierValue > previousFourierValue);
    
    previousTime         = t;
    previousFourierValue = fourierValue;
  }
  return peakValues;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  if (rank==0)
    std::cout << "Kokkos bounds checking is enabled.\n";
#else
  if (rank==0)
    std::cout << "Kokkos bounds checking is disabled.\n";
#endif
  
  {
    mpiSession.barrier();
    
    double k_x    = 0.5; // wavelength
    double u0     = 0.0; // initial drift velocity
    double gamma  = 1e-2; // initial perturbation -- A from Heath et al.
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      = -1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double t0     =  0.0; // just computing for first time slab right now; will do 40 time slabs in all
    double tWidth =  2.0;
    
    double v0     = -4.5;
    double vWidth =  9.0;
    
    bool useSerendipityBasis = false;
    bool useMimeticDPG = false; // otherwise, will use standard DPG (the default)
    
    int xElements = 5;
    int vElements = 4;
    int tElements = 1;
    int delta_k   = 2;
    int H1Order = 4; // this gives f, E (which are in L^2) a polynomial order of 3
    std::string problemChoiceString = "landau";
    bool saveViz = true;
    bool exportDebugViz = false; // if set to true, will output separate viz file for each substep solve (to /tmp)
    std::string outputDir = "/tmp";
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
        
    int maxDofsForDirectSolve = 20000;
    int numUniformRefinements = 2; // takes a coarse 5x4x1 mesh to a 20x16x4
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int worksetSize = -1;
    
    bool skipPostProcessing = false;
    
    int numTimeSlabs = 40; // 40 * 2 (tWidth) = 80 final time
    bool useStandardAssembly = false; // use sum-factorized assembly
    
    std::string EFieldMode1File = "", EFieldMode1DecayRateFile = "";
    std::string resultsFilePrefix = "";
    std::string vlasovTimingsOutputFile = "";
    
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("worksetSize", &worksetSize, "workset size for Vlasov assembly; -1 indicates to allow Solution to determine automatically");
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("vlasovTimingsOutput", &vlasovTimingsOutputFile);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("tWidth", &tWidth, "span of time domain for each time slab");
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("numTimeSlabs", &numTimeSlabs);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("exportDebugViz", "noDebugViz", &exportDebugViz);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("useMimeticDPG", "useStandardDPG", &useMimeticDPG);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    cmdp.setOption("useStandardAssembly", "useSumFactorizedAssembly", &useStandardAssembly);
    cmdp.setOption("skipPostProcessing", "doPostProcessing", &skipPostProcessing, "skip post-processing steps (intended use is performance timings)");
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    bool usePeriodicBCs = true;
    
    auto form = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                     useSerendipityBasis, useMimeticDPG, numUniformRefinements);
    
    if (worksetSize != -1)
    {
      form->vlasovSolution()->setCellBatchSize(worksetSize);
    }
    
    if (useStandardAssembly)
    {
      // override vlasov IP
      auto vlasovBF = form->vlasovBF();
      form->vlasovSolution()->setIP(vlasovBF->graphNorm());
    }
    
    auto sampleElementType = form->vlasovMesh()->globalDofAssignment()->getSampleElementType();
    if (rank == 0)
    {
      std::cout  << "trial ordering:\n";
      sampleElementType->trialOrderPtr->print(std::cout);
      std::cout  << "test ordering:\n";
      sampleElementType->testOrderPtr->print(std::cout);
    }
    
    auto traceVar     = form->t_n_hat();
    auto traceSolnFxn = Function::solution(traceVar, form->vlasovSolution(), true);
    
    std::vector<std::string> otherFunctionNames;
    std::vector<FunctionPtr> otherFunctions;
    
    if (useMimeticDPG)
    {
      std::map<int, FunctionPtr> gcEqualsOne;
      MimeticVlasovSpaceTimeFormulation* mimeticForm = dynamic_cast<MimeticVlasovSpaceTimeFormulation*>(form.get());
      gcEqualsOne[mimeticForm->g_c()->ID()] = Function::constant(1.0);
      LinearTermPtr bfAtPsiEqualsOne = mimeticForm->vlasovSolution()->bf()->testFunctional(gcEqualsOne);
      
      auto repOfBFAtGcEqualsOne = RieszRep::rieszRep(mimeticForm->vlasovMesh(), mimeticForm->vlasovSolution()->ip(), bfAtPsiEqualsOne);
      otherFunctionNames.push_back("1_w(w)");
      otherFunctions.push_back(RieszRep::repFunction(mimeticForm->w(), repOfBFAtGcEqualsOne));
      otherFunctionNames.push_back("1_w(w_v)");
      otherFunctions.push_back(RieszRep::repFunction(mimeticForm->w_v(0), repOfBFAtGcEqualsOne));
      otherFunctionNames.push_back("1_w(w_a)");
      otherFunctions.push_back(RieszRep::repFunction(mimeticForm->w_a(0), repOfBFAtGcEqualsOne));
      otherFunctionNames.push_back("1_w(w_c)");
      otherFunctions.push_back(RieszRep::repFunction(mimeticForm->w_c(), repOfBFAtGcEqualsOne));
    }
    
    form->setExportViz(exportDebugViz);
    
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
    otherFunctions.push_back(cellIDFunction);
    otherFunctionNames.push_back({"cell ID (Vlasov)"});
    
    form->setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
    }
    form->setIterativeSolveTolerances(cgTols);
    form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    
    form->setMaxMomentForConservationDiagnostics(2); // mass, momentum, and energy
    
    // compute mesh widths after uniform refinements:
    int xElementsTotal = xElements;
    int tElementsTotal = tElements;
    int vElementsTotal = vElements;
    if (numUniformRefinements > 0)
    {
      for (int i=0; i<numUniformRefinements; i++)
      {
        xElementsTotal *= 2;
        tElementsTotal *= 2;
        vElementsTotal *= 2;
      }
    }
    
    FunctionPtr f0;
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    
    {
      const int firstVelocityDimensionZeroBased = 2; // 1D1T1V
      const int numVelocityDims                 = 1;
      FunctionPtr fM_u0        = MaxwellianFunction::maxwellian(firstVelocityDimensionZeroBased, numVelocityDims, {u0}, sigma * sigma, n0);
      f0                       = fM_u0 * (1. + gamma * cos_kx);
    }
    form->setInitialConditions(f0);
    
    // need to pin the potential at 0 to eliminate the constant mode.
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(x0);
    bool restrictToBoundary = false; // periodic; x0 = 0 does *not* lie on a mesh boundary
    auto xLeftTag = form->createConfigurationMeshSideSetTag(xLeftFilter, restrictToBoundary);
    auto VE_traceVar = form->VE_configurationSpaceTrace();
    form->addConfigurationSpaceBC(xLeftTag, VE_traceVar, Function::zero());
    
    std::ostringstream suffix;
    if (useSerendipityBasis)
    {
      suffix << "_serendipity";
    }
    if (useMimeticDPG)
    {
      suffix << "_mimeticDPG";
    }
    suffix << "_p" << H1Order;
    suffix << "_dp" << delta_k;
    suffix << "_" << xElementsTotal << "x" << tElementsTotal << "x" << vElementsTotal;
    
    const std::string problemPrefix = problemChoiceString;
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
    
    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    std::ostringstream fourierSS;
    fourierSS << problemPrefix << "Fourier" << suffix.str();
    
    const std::string directoryPath = outputDir;
    
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz, fourierViz;
    if (saveViz)
    {
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    auto fullFileName = [&] (const std::string &simpleName) -> string
    {
      std::ostringstream fileNameStream;
      fileNameStream << outputDir << "/";
      fileNameStream << simpleName << "_" << problemPrefix << suffix.str();
      
      fileNameStream << "_" << numTimeSlabs << "_timeSlabs";
      fileNameStream << ".dat";
      return fileNameStream.str();
    };
    
    if (EFieldMode1File == "")
    {
      EFieldMode1File = fullFileName("EFieldMode1");
    }
    
    if (EFieldMode1DecayRateFile == "")
    {
      EFieldMode1DecayRateFile = fullFileName("EFieldMode1DecayRate");
    }
    
    string totalMassFile     = fullFileName("totalMass");
    string totalMomentumFile = fullFileName("totalMomentum");
    string totalEnergyFile   = fullFileName("totalEnergy");
    
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    double initialMass, initialMomentum, initialEnergy;
    
    double maxMassLoss = 0, maxMomentumLoss = 0, maxEnergyLoss = 0;
    double maxMassLoss_t = 0, maxMomentumLoss_t = 0, maxEnergyLoss_t = 0;
    
    // define the weight function we'll need for the first Fourier mode
    FunctionPtr sinFxnFirstFourier;
    {
      const double T = xWidth;
      const int modeNumber = 1;
      sinFxnFirstFourier = Teuchos::rcp(new Sin_ax(2. * modeNumber * M_PI / T, 0.));
    }
    int fourierEnrichment = 5; // quadrature enrichment for Fourier mode integral
    
    // set up timings printer
    double totalTimeG_vlasov = 0, totalTimeB_vlasov = 0, totalTimeT_vlasov = 0, totalTimeK_vlasov = 0;
    int totalElementCount_vlasov = 0;
    std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> vlasovOptimalTestTimingCallback;
    vlasovOptimalTestTimingCallback = [&totalTimeG_vlasov, &totalTimeB_vlasov, &totalTimeT_vlasov, &totalTimeK_vlasov, &totalElementCount_vlasov]
      (int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType) {
        totalElementCount_vlasov += numElements;
        totalTimeG_vlasov        += timeG;
        totalTimeB_vlasov        += timeB;
        totalTimeT_vlasov        += timeT;
        totalTimeK_vlasov        += timeK;
    };
    double totalTimeG_vlasov_core_integration = 0;
    int totalElementCount_vlasov_core_integration = 0; // should match totalElementCount_vlasov
    double totalTime_block1 = 0, totalTime_block2 = 0, totalTime_block3 = 0, totalTime_block4 = 0;
    std::function<void(int numElements, double timeG_integrate, double time_computeCellMeasure, double time_getJacobianData, double time_configSpaceFunction, double time_intrepid2JacobianAllocAndCompute)> vlasovGramCoreIntegrationTimingCallback;
    vlasovGramCoreIntegrationTimingCallback = [&totalTimeG_vlasov_core_integration, &totalElementCount_vlasov_core_integration,
     &totalTime_block1, &totalTime_block2, &totalTime_block3, &totalTime_block4]
      (int numElements, double timeCoreIntegration, double time_block1, double time_block2, double time_block3, double time_block4) {
        totalElementCount_vlasov_core_integration += numElements;
        totalTimeG_vlasov_core_integration        += timeCoreIntegration;
        totalTime_block1 += time_block1;
        totalTime_block2 += time_block2;
        totalTime_block3 += time_block3;
        totalTime_block4 += time_block4;
    };
    double totalTimeG_poisson = 0, totalTimeB_poisson = 0, totalTimeT_poisson = 0, totalTimeK_poisson = 0;
    int totalElementCount_poisson = 0;
    std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> poissonOptimalTestTimingCallback;
    poissonOptimalTestTimingCallback = [&totalTimeG_poisson, &totalTimeB_poisson, &totalTimeT_poisson, &totalTimeK_poisson, &totalElementCount_poisson]
      (int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType) {
        totalElementCount_poisson += numElements;
        totalTimeG_poisson        += timeG;
        totalTimeB_poisson        += timeB;
        totalTimeT_poisson        += timeT;
        totalTimeK_poisson        += timeK;
    };
    // register the timing callbacks
    form->vlasovBF()->setOptimalTestTimingCallback(vlasovOptimalTestTimingCallback);
    form->configurationSpaceSolution()->bf()->setOptimalTestTimingCallback(poissonOptimalTestTimingCallback);
    form->setVlasovGramAssemblyTimingCallback(vlasovGramCoreIntegrationTimingCallback);
    
    vector<pair<double,double>> fourierPeakValues; // (t, ln(|E1|))
    bool logEFieldSineModeIsIncreasing = false; // if it was increasing, and then decreases: local maximum
    double previousFourierValue = 0;
    double t_previous = 0;
    for (int slabNumber=0; slabNumber<numTimeSlabs; slabNumber++)
    {
      if (rank == 0)
      {
        std::cout << "\n\n******************************* Slab Number " << slabNumber << " *******************************\n";
      }
      double slabStartTime = slabNumber * tWidth;
      double slabEndTime   = slabStartTime + tWidth;
      
      const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
      auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
      
      FunctionPtr E_soln    = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
      FunctionPtr potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
      
      auto density  = solutionMoments->solutionMomentFunction(0);
      auto momentum = solutionMoments->solutionMomentFunction(1);
      
      auto velocity = momentum / density;
      
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
      
      form->solve();
      
      if (!skipPostProcessing)
      {
        SpatialIntegralOfSpaceTimeSolution fourierIntegral(form->configurationSpaceSolution(), form->configurationSpaceFormulation()->E_fieldVar(0), sinFxnFirstFourier, fourierEnrichment);
        
        if (fourierViz == Teuchos::null)
        {
          fourierViz = Teuchos::rcp(new HDF5Exporter (fourierIntegral.temporalMesh(), fourierSS.str(), directoryPath) );
        }
        else
        {
          fourierViz->setMesh(fourierIntegral.temporalMesh());
        }
        // first Fourier mode of the electric field:
        FunctionPtr E1 = (2.0 / xWidth) * fourierIntegral.temporalFunction();
        // what we are most interested in plotting is ln |E1|:
        FunctionPtr abs_E1 = AbsFunction::abs(E1);
        FunctionPtr ln_abs_E1 = Teuchos::rcp (new Ln<double>(abs_E1) );
        fourierViz->exportFunction({E1, ln_abs_E1}, {"E1","ln(|E1|)"}, slabNumber); // TODO: revise the Fourier output to time/slab independent… (GnuPlotUtil??)
        
        double dt_export = 0.01; // granularity for plots, etc.
        // output time values that are an even multiple of dt
        double t0 = int(slabStartTime / dt_export) * dt_export;
        if (t0 < slabStartTime)
        {
          t0 += dt_export;
        }
        double t1 = int(slabEndTime / dt_export) * dt_export;
        if (std::abs(t1 - slabEndTime) < 1e-10)
        {
          // if the final output time would match the end of the slab, cheat back by dt_fourier, unless it's the final slab
          // (each slab "owns" its start time; only the final slab "owns" its end time)
          if (slabNumber != numTimeSlabs - 1)
          {
            t1 -= dt_export;
          }
        }
        
        if (t0 <= t1)
        {
          auto fourierValues = evaluateTemporalSolutionFunction(fourierIntegral.temporalSolution(), ln_abs_E1, t0, t1, dt_export);
          int numTemporalValues = (t1 - t0 + 1e-10) / dt_export + 1;
          auto totalMassValues     = form->totalMassTimeSeries     (t0, dt_export, numTemporalValues);
          auto totalMomentumValues = form->totalXMomentumTimeSeries(t0, dt_export, numTemporalValues);
          auto totalEnergyValues   = form->totalEnergyTimeSeries   (t0, dt_export, numTemporalValues);
          
          if (slabNumber == 0)
          {
            initialMass     = totalMassValues[0].second;
            initialMomentum = totalMomentumValues[0].second;
            initialEnergy   = totalEnergyValues[0].second;
            
            if (rank == 0)
            {
              std::cout << "Total Mass at t=0:     " << initialMass     << std::endl;
              std::cout << "Total Momentum at t=0: " << initialMomentum << std::endl;
              std::cout << "Total Energy at t=0:   " << initialEnergy   << std::endl;
            }
          }
          for (auto entry : totalMassValues)
          {
            const double massLoss = initialMass - entry.second;
            if (std::abs(massLoss) > std::abs(maxMassLoss) )
            {
              maxMassLoss   = massLoss;
              maxMassLoss_t = entry.first;
            }
          }
          for (auto entry : totalMomentumValues)
          {
            const double momentumLoss = initialMomentum - entry.second;
            if (std::abs(momentumLoss) > std::abs(maxMomentumLoss) )
            {
              maxMomentumLoss   = momentumLoss;
              maxMomentumLoss_t = entry.first;
            }
          }
          for (auto entry : totalEnergyValues)
          {
            const double energyLoss = initialEnergy - entry.second;
            if (std::abs(energyLoss) > std::abs(maxEnergyLoss) )
            {
              maxEnergyLoss   = energyLoss;
              maxEnergyLoss_t = entry.first;
            }
          }
          
          auto totalNetFlux = traceSolnFxn->integrate(form->vlasovMesh());
          
          if (rank == 0)
          {
            const bool append = (slabNumber > 0);
            exportTimeSeriesValues(EFieldMode1File,   append, fourierValues,       "ln(|E1|)");
            exportTimeSeriesValues(totalMassFile,     append, totalMassValues,     "mass");
            exportTimeSeriesValues(totalMomentumFile, append, totalMomentumValues, "momentum");
            exportTimeSeriesValues(totalEnergyFile,   append, totalEnergyValues,   "energy");
            
            form->printLocalConservationReportForSlab(slabNumber);
            
            std::cout << "total net mass flux (integral of trace variable) is " << totalNetFlux << std::endl;
            std::cout << "max mass loss thus far is at time " << maxMassLoss_t << ": " << maxMassLoss << std::endl;
            std::cout << "max momentum loss thus far is at time " << maxMomentumLoss_t << ": " << maxMomentumLoss << std::endl;
            std::cout << "max energy loss thus far is at time " << maxEnergyLoss_t << ": " << maxEnergyLoss << std::endl;
          }
          
          auto slabFourierPeakValues = getPeakValues(previousFourierValue, t_previous, logEFieldSineModeIsIncreasing, fourierValues);
          const int previousPeakCount = fourierPeakValues.size();
          fourierPeakValues.insert(fourierPeakValues.end(), slabFourierPeakValues.begin(), slabFourierPeakValues.end());
          
          const int numNewPeaks = slabFourierPeakValues.size();
          if (rank == 0)
          {
            // this reporting is designed to match the console output of our time-marching Landau driver:
            for (int newPeakOrdinal = 0; newPeakOrdinal < numNewPeaks; newPeakOrdinal++)
            {
              const auto &t     = slabFourierPeakValues[newPeakOrdinal].first;
              const auto &value = slabFourierPeakValues[newPeakOrdinal].second;
              std::cout << "local maximum of logarithm of dominant E-field mode at t=" << t << ": " << value;
              
              const int peakOrdinal = newPeakOrdinal + previousPeakCount;
              if (peakOrdinal > 0)
              {
                // then report the local rate:
                const auto &t_prev     = fourierPeakValues[peakOrdinal-1].first;
                const auto &value_prev = fourierPeakValues[peakOrdinal-1].second;
                double rate = (value - value_prev) / (t - t_prev);
                std::cout << " (rate: " << rate << ")";
              }
              std::cout << std::endl;
            }
            
            // if the new peaks have taken us past the seventh, compute an overall rate by computing slope from third to seventh peak
            if ((previousPeakCount < 7) && (slabFourierPeakValues.size() >= 7))
            {
              const double thirdPeak   = fourierPeakValues[2].second;
              const double seventhPeak = fourierPeakValues[6].second;
              
              const double third_t   = fourierPeakValues[2].first;
              const double seventh_t = fourierPeakValues[6].first;
              
              const double mode1DecayRate = (seventhPeak - thirdPeak) / (seventh_t - third_t);
              std::cout << "Overall decay rate: " << mode1DecayRate << std::endl;
            }
          }
        }
        
        {
          /*
           // output to EFieldMode1File,
           //t_ln_b1_pairs;
           if (rank == 0)
           {
           std::ofstream EFieldMode1DecayRateFileStream(EFieldMode1DecayRateFile, std::ios::out);
           EFieldMode1DecayRateFileStream << setprecision(15);
           EFieldMode1DecayRateFileStream << "E-Field, Mode 1 Decay Rate: " << mode1DecayRate << std::endl;
           EFieldMode1DecayRateFileStream.close();
           }
           */
        }
        
        solutionMoments->update();
        
        if (saveViz)
        {
          poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, slabNumber);
          vlasovViz->exportSolution(form->vlasovSolution(), otherFunctions, otherFunctionNames, slabNumber);
        }
      } // end "post-processing" block
      
      if (slabNumber + 1 < numTimeSlabs)
      {
        form = form->nextTimeSlab(tWidth, tElements);
        
        if (saveViz)
        {
          // mesh has changed; update viz exporters (not a common use case; let's hope it works!):
          poissonViz->setMesh(form->configurationSpaceMesh());
          vlasovViz->setMesh(form->vlasovMesh());
        }
      }
      Epetra_CommPtr Comm = MPIWrapper::CommWorld();
      totalTimeG_vlasov = MPIWrapper::sum(*Comm, totalTimeG_vlasov);
      totalTimeB_vlasov = MPIWrapper::sum(*Comm, totalTimeB_vlasov);
      totalTimeT_vlasov = MPIWrapper::sum(*Comm, totalTimeT_vlasov);
      totalTimeK_vlasov = MPIWrapper::sum(*Comm, totalTimeK_vlasov);
      totalElementCount_vlasov = MPIWrapper::sum(*Comm, totalElementCount_vlasov);
      totalTimeG_poisson = MPIWrapper::sum(*Comm, totalTimeG_poisson);
      totalTimeB_poisson = MPIWrapper::sum(*Comm, totalTimeB_poisson);
      totalTimeT_poisson = MPIWrapper::sum(*Comm, totalTimeT_poisson);
      totalTimeK_poisson = MPIWrapper::sum(*Comm, totalTimeK_poisson);
      totalElementCount_poisson = MPIWrapper::sum(*Comm, totalElementCount_poisson);
      
      totalTimeG_vlasov_core_integration        = MPIWrapper::sum(*Comm, totalTimeG_vlasov_core_integration);
      totalElementCount_vlasov_core_integration = MPIWrapper::sum(*Comm, totalElementCount_vlasov_core_integration);
      
      if (rank == 0)
      {
//        cout << "Running average time/element for Vlasov optimal test computations:\n";
//        cout << "G:   " <<  totalTimeG / totalElementCount << " sec.\n";
//        cout << "B:   " <<  totalTimeB / totalElementCount << " sec.\n";
//        cout << "T:   " <<  totalTimeT / totalElementCount << " sec.\n";
//        cout << "K:   " <<  totalTimeK / totalElementCount << " sec.\n";
//        cout << "RHS: " << totalTimeRHS / totalRHSElements << " sec.\n";
        
        cout << "Running total time for Vlasov optimal test computations:\n";
        cout << "G (core): " <<  totalTimeG_vlasov_core_integration << " sec.\n";
        cout << "G:        " <<  totalTimeG_vlasov << " sec.\n";
        cout << "B:        " <<  totalTimeB_vlasov << " sec.\n";
        cout << "K:        " <<  totalTimeK_vlasov << " sec.\n";
        
        cout << "totalTime_block1: " << totalTime_block1 << endl;
        cout << "totalTime_block2: " << totalTime_block2 << endl;
        cout << "totalTime_block3: " << totalTime_block3 << endl;
        cout << "totalTime_block4: " << totalTime_block4 << endl;
        
        cout << "Running total time for Poisson optimal test computations:\n";
        cout << "G:   " <<  totalTimeG_poisson << " sec.\n";
        cout << "B:   " <<  totalTimeB_poisson << " sec.\n";
        cout << "K:   " <<  totalTimeK_poisson << " sec.\n";
        
        if (vlasovTimingsOutputFile != "")
        {
          bool fileExists = false;
          {
            std::ifstream file(vlasovTimingsOutputFile);
            fileExists = file.good();
          }
          if (!fileExists)
          {
            std::ofstream file(vlasovTimingsOutputFile);
            file << "p_test\tworksetSize\tH1Order\tdelta_k\tG_core_time\tG_time\tB_time\tGAlgorithm\n";
            file.close();
          }
          std::ofstream file(vlasovTimingsOutputFile, std::ios::app);
          file << H1Order + delta_k << "\t" << worksetSize << "\t" << H1Order << "\t" << delta_k << "\t" << roundToThreeSignificantDigits(totalTimeG_vlasov_core_integration);
          file << "\t" << roundToThreeSignificantDigits(totalTimeG_vlasov) << "\t" << roundToThreeSignificantDigits(totalTimeB_vlasov);
          if (useStandardAssembly)
          {
            file << "\t" << "standard";
          }
          else
          {
            file << "\t" << "sum-factorized";
          }
          file << "\n";
          file.close();
        }
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
