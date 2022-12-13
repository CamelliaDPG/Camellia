//
// © 2016-2022 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
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

class Maxwellian1T1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1T1V(double u, double sigma, double n)
  :
  _u(u),
  _sigma(sigma),
  _n(n)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
//    std::cout << "weight: " << weight << std::endl;
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,2); // vx is the third dimension for 1D1T1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
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
  
  TFunctionPtr<double> dz() override
  {
    auto vx = MonomialFunction::xi_n(2, 1);
    FunctionPtr f_M = Teuchos::rcp( new Maxwellian1T1V(_u,_sigma,_n));
    return (_u - vx) / (_sigma * _sigma) * f_M;
  }
  
  TFunctionPtr<double> dt() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 3) // 1-based
    {
      return dz();
    }
    else
    {
      return Function::zero();
    }
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
                                                              const int H1Order, const bool usePeriodicBCs, const bool useSerendipity,
                                                              const int numUniformRefinements)
{
  const int delta_k = 2;
  
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
  
  auto form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, useSerendipity, configForm, spatialMeshTopo,
                                                          temporalVertices,
                                                          velocitySpaceVertices_x,
                                                          velocitySpaceVertices_y,
                                                          velocitySpaceVertices_z,
                                                          H1Order, delta_k,
                                                          numUniformRefinements));
  return form;
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
    
    int numTimeSlabs = 40; // 40 * 2 (tWidth) = 80 final time
    
    std::string EFieldMode1File = "", EFieldMode1DecayRateFile = "";
    std::string resultsFilePrefix = "";
    
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
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
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    bool usePeriodicBCs = true;
    
    auto form = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, usePeriodicBCs, useSerendipityBasis, numUniformRefinements);
    
    form->setExportViz(exportDebugViz);
    
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
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
      FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n0) );
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
    
    if (EFieldMode1File == "")
    {
      std::ostringstream EFieldMode1FileStream;
      EFieldMode1FileStream << outputDir << "/";
      EFieldMode1FileStream << "EFieldMode1_" << problemPrefix << suffix.str();
      
      EFieldMode1FileStream << "_" << numTimeSlabs << "_timeSlabs";
      EFieldMode1FileStream << ".dat";
      EFieldMode1File = EFieldMode1FileStream.str();
    }
    
    if (EFieldMode1DecayRateFile == "")
    {
      std::ostringstream EFieldMode1DecayRateFileStream;
      EFieldMode1DecayRateFileStream << outputDir << "/";
      EFieldMode1DecayRateFileStream << "EFieldMode1DecayRate_" << problemPrefix << "_" << suffix.str();
      
      EFieldMode1DecayRateFileStream << "_" << numTimeSlabs << "_timeSlabs";
      EFieldMode1DecayRateFileStream << ".dat";
      EFieldMode1DecayRateFile = EFieldMode1DecayRateFileStream.str();
    }
    
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    // define the weight function we'll need for the first Fourier mode
    FunctionPtr sinFxnFirstFourier;
    {
      const double T = xWidth;
      const int modeNumber = 1;
      sinFxnFirstFourier = Teuchos::rcp(new Sin_ax(2. * modeNumber * M_PI / T, 0.));
    }
    int fourierEnrichment = 5; // quadrature enrichment for Fourier mode integral
    
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
      
      double dt_fourier = 0.01; // granularity for plots, etc.
      // output time values that are an even multiple of dt
      double t0 = int(slabStartTime / dt_fourier) * dt_fourier;
      if (t0 < slabStartTime)
      {
        t0 += dt_fourier;
      }
      double t1 = int(slabEndTime / dt_fourier) * dt_fourier;
      if (std::abs(t1 - slabEndTime) < 1e-10)
      {
        // if the final output time would match the end of the slab, cheat back by dt_fourier, unless it's the final slab
        // (each slab "owns" its start time; only the final slab "owns" its end time)
        if (slabNumber != numTimeSlabs - 1)
        {
          t1 -= dt_fourier;
        }
      }

      if (t0 <= t1)
      {
        auto fourierValues = evaluateTemporalSolutionFunction(fourierIntegral.temporalSolution(), ln_abs_E1, t0, t1, dt_fourier);
        
        if (rank == 0)
        {
          const bool append = (slabNumber > 0);
          exportFourierValues(EFieldMode1File, append, fourierValues);
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
        vlasovViz->exportSolution(form->vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"}, slabNumber);
      }
      
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
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
