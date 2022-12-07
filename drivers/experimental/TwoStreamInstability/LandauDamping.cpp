//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "Boltzmann1D1VFormulation.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

class Maxwellian1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1V(double u, double sigma, double n)
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
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
};


class Cos_kx_shifted : public TFunction<double>
{
  double _k;
  double _t = -1.0; // -1.0 indicates space-time context
  bool _spaceTime;
public:
  // space-time constructor:
  Cos_kx_shifted(double k)
  : TFunction<double>("Cos(kxv - t)",0,VARIES_IN_DIMENSION_0 | VARIES_IN_DIMENSION_1 | VARIES_IN_DIMENSION_2)
  {
    _k = k;
    _spaceTime = true;
  }
  
  // time-marching constructor
  Cos_kx_shifted(double k, double t)
  : TFunction<double>("Cos(kxv - t)",0,VARIES_IN_DIMENSION_0 | VARIES_IN_DIMENSION_1)
  {
    _k = k;
    _t = t;
    _spaceTime = false;
  }

  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
  {
    if (_spaceTime)
    {
      auto xPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(0); // xPoints might have more coordinates than just x, but it will certainly have x in the first entry
      auto tPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(1);
      auto vPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(2);
      
      auto underlyingData = values.template getUnderlyingView<4>(); // (C,P0,PT0,PV0) -- Again, it may be that data has some additional point dimensions, but the underlying view must support variation in C, P0 (x coordinate), and PV0 (v_x coordinate) dimensions, and thus have rank exactly 3
      
      using ExecutionSpace = Camellia::DeviceExecutionSpace;
      Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>> policy({0,0,0,0},{underlyingData.extent(0),underlyingData.extent(1),underlyingData.extent(2),underlyingData.extent(3)});
      
      
      // avoid implicit references to this
      const double k = _k;
      
      const int xCoordIndex = 0; // in xPoints
      const int tCoordIndex = 0; // in tPoints
      const int vCoordIndex = 0; // in vPoints
      
      Kokkos::parallel_for("Cos(kxv - t) Data values - space-time", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &xPointOrdinal, const int &tPointOrdinal, const int &vPointOrdinal) {
        auto & result = underlyingData(cellOrdinal, xPointOrdinal, tPointOrdinal, vPointOrdinal);
        
        const auto & x = xPoints(cellOrdinal,xPointOrdinal,xCoordIndex);
        const auto & t = tPoints(cellOrdinal,tPointOrdinal,tCoordIndex);
        const auto & v = vPoints(cellOrdinal,vPointOrdinal,vCoordIndex);
        result = std::cos(k * x * v - t);
      });
    }
    else
    {
      auto xPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(0); // xPoints might have more coordinates than just x, but it will certainly have x in the first entry
      auto vPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(1);
      
      auto underlyingData = values.template getUnderlyingView<3>(); // (C,P0,PV0) -- Again, it may be that data has some additional point dimensions, but the underlying view must support variation in C, P0 (x coordinate), and PV0 (v_x coordinate) dimensions, and thus have rank exactly 3
      
      using ExecutionSpace = Camellia::DeviceExecutionSpace;
      Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>> policy({0,0,0},{underlyingData.extent(0),underlyingData.extent(1),underlyingData.extent(2)});
      
      // avoid implicit references to this
      const double k = _k;
      const double t = _t;
      
      const int xCoordIndex = 0; // in xPoints
      const int vCoordIndex = 0; // in vPoints
      
      Kokkos::parallel_for("Cos(kxv - t) Data values", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &xPointOrdinal, const int &vPointOrdinal) {
        auto & result = underlyingData(cellOrdinal, xPointOrdinal, vPointOrdinal);
        
        const auto & x = xPoints(cellOrdinal,xPointOrdinal,xCoordIndex);
        const auto & v = vPoints(cellOrdinal,vPointOrdinal,vCoordIndex);
        result = std::cos(k * x * v - t);
      });
    }
  }

  void values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
    const Camellia::FieldContainer<double> &physicalPoints = basisCache->getPhysicalCubaturePointsFC();
    auto pointsView = physicalPoints.checkoutDeviceView(); // (C,P,D)
    auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values); // (C,P) dimension
    
    // avoid implicit references to this
    const double k = _k;
    const double t = _t;
    
    // on present implementation, we support time-marching (1D1V), and space-time (1D1T1V).
    // in the first case, vCoordIndex should be 1; in the second, 2.
    
    const bool spaceTime = basisCache->cellTopology()->getDimension() == 3;
    const int xCoordIndex = 0;                 // in pointsView
    const int vCoordIndex = spaceTime ? 2 : 1; // in pointsView
    
    Kokkos::parallel_for("Cos(kxv - t) values", rangePolicy,
    KOKKOS_LAMBDA (const int cellOrdinal, const int pointOrdinal) {
      const double &x = pointsView(cellOrdinal,pointOrdinal,xCoordIndex);
      const double &v = pointsView(cellOrdinal,pointOrdinal,vCoordIndex);
      assignValue(values(cellOrdinal,pointOrdinal), modType, std::cos(k * x * v - t) );
    });
    physicalPoints.checkinDeviceView(false);
  }

  TFunctionPtr<double> dx()
  {
    return Teuchos::null; // derivatives of this function are unsupported -- we just use it to measure L^2 errors…
  }
  TFunctionPtr<double> dy()
  {
    return Teuchos::null; // derivatives of this function are unsupported -- we just use it to measure L^2 errors…
  }
  TFunctionPtr<double> dz()
  {
    return Teuchos::null; // derivatives of this function are unsupported -- we just use it to measure L^2 errors…
  }

  std::string displayString()
  {
    std::ostringstream ss;
    if (_spaceTime)
    {
      ss << "\\cos( " << _k << " xv - t )";
    }
    else
    {
      ss << "\\cos( " << _k << " xv - " << _t << " )";
    }
    return ss.str();
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  // block off variables so that everything gets destructed before the finalize() call at the end.
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    mpiSession.barrier();
    
    bool implicit = true;
    double k_x    = 0.5; // wavelength
    double u0     = 0.0; // initial drift velocity
    double gamma  = -1.0; // weight on cosine term (sinusoidal perturbation size -- this is "A" in Heath et al.); defaults to 0.1 for advection, 0.01 for linear Landau
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      = -1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  4 * M_PI;
    
    const double DEFAULT_V0     = -4.5;
    const double DEFAULT_VWIDTH =  9.0;
    double v0     = DEFAULT_V0;
    double vWidth = DEFAULT_VWIDTH;
    
    bool enforceQuasiNeutrality = false;
    
    bool useAdaptivity = false;
    bool pureAdvection = false;

    int xElements = 8;
    int vElements = 8;
    int numUniformRefinements = 0;
    int delta_k   = 2;
    double dt = 0.001;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int minSubsteps = 3;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-6;
    int numTimeSteps = 500;
    int outputFrequency = 10; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    bool saveViz = true;
    std::string outputDir = "/tmp";
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 20000;
    int fixedCountRefinementTarget = -1; // set to positive number to used fixed-count ref strategy instead of greedy strategy
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    std::string EFieldMode1File = "";
    std::string EFieldMode1DecayRateFile = "";
    
    cmdp.setOption("pureAdvection", "landauDamping", &pureAdvection);
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt );
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    if (pureAdvection)
    {
      q = 0;
      if (gamma == -1)
      {
        // unset by user
        gamma = .1;
      }
    }
    else
    {
      // linear Landau
      gamma = 0.01;
    }
    
    bool usePeriodicBCs = true;
    
    using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
    auto formulationChoice = FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE;
    
    ERKTableauPtr erkTableau = Teuchos::null;
    
    const int vlasovTraceEnrichment = 0;
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements);
    
    form.setVerboseSolve(false); // set to TRUE for debugging
    
    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    form.setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
      form.setIterativeSolveTolerances(cgTols);
    }
    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    form.setFixedPointTolerance(fixedPointTol);
    form.setFixedPointMinSubstepsForImplicit(minSubsteps);
    form.setFixedPointMaxSubstepsForImplicit(maxSubsteps);
    
    FunctionPtr cos_kx = Teuchos::rcp(new Cos_ax(k_x, 0.));
    FunctionPtr fM     = Teuchos::rcp(new Maxwellian1V( u0, n0, sigma) );
    FunctionPtr f0     = (1. + gamma * cos_kx) * fM;
    FunctionPtr df0    = f0 - fM;
    
    double fM_at_0_0  = fM->evaluate(0,0);
    double f0_at_0_0  = f0->evaluate(0,0);
    double df0_at_0_0 = df0->evaluate(0,0);
  
    if (rank == 0)
    {
      std::cout << " fM(0,0) = " <<  fM_at_0_0 << std::endl;
      std::cout << " f0(0,0) = " <<  f0_at_0_0 << std::endl;
      std::cout << "df0(0,0) = " << df0_at_0_0 << std::endl;
    }
    
    form.setInitialConditions(f0);
    
    std::ostringstream suffix;
    suffix << "_dt" << dt;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;

    std::string problemPrefix;
    if (pureAdvection)
    {
      problemPrefix = "Advection";
    }
    else
    {
      problemPrefix = "LinearLandau";
    }

    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix <<"Vlasov" << suffix.str();

    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (resultsFile == "")
    {
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
      
      resultsFileStream << "_" << numTimeSteps << "_timeSteps";
      resultsFileStream << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    if (EFieldMode1File == "")
    {
      std::ostringstream EFieldMode1FileStream;
      EFieldMode1FileStream << outputDir << "/";
      EFieldMode1FileStream << "EFieldMode1_" << problemPrefix << "_" << suffix.str();
      
      EFieldMode1FileStream << "_" << numTimeSteps << "_timeSteps";
      EFieldMode1FileStream << ".dat";
      EFieldMode1File = EFieldMode1FileStream.str();
    }
    
    if (EFieldMode1DecayRateFile == "")
    {
      std::ostringstream EFieldMode1DecayRateFileStream;
      EFieldMode1DecayRateFileStream << outputDir << "/";
      EFieldMode1DecayRateFileStream << "EFieldMode1DecayRate_" << problemPrefix << "_" << suffix.str();
      
      EFieldMode1DecayRateFileStream << "_" << numTimeSteps << "_timeSteps";
      EFieldMode1DecayRateFileStream << ".dat";
      EFieldMode1DecayRateFile = EFieldMode1DecayRateFileStream.str();
    }
    
    auto getSpatialMeshFilePath = [&]() {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
      
      spatialMeshInfoPathStringStream << numTimeSteps << "_timeSteps";
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getTimeStepReportFilePath = [&]() {
      std::ostringstream timeStepReportPathStringStream;
      timeStepReportPathStringStream << outputDir << "/";
      timeStepReportPathStringStream << "timeStepReport_" << problemPrefix << suffix.str();
      
      timeStepReportPathStringStream << numTimeSteps << "_timeSteps";
      timeStepReportPathStringStream << ".dat";
      
      return timeStepReportPathStringStream.str();
    };
    
    auto refReportFilePath = getTimeStepReportFilePath();
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tBC err\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double bcErr, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << bcErr << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    const std::string directoryPath = outputDir;
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    Teuchos::RCP<HDF5Exporter> boltzmannViz, gaussViz;
    if (saveViz)
    {
      boltzmannViz = Teuchos::rcp(new HDF5Exporter(form.boltzmannMesh(), vlasovSS.str(), directoryPath));
      gaussViz = Teuchos::rcp(new HDF5Exporter (form.gaussMesh(), poissonSS.str(), directoryPath) );
    }
    
    // create a Solution object to store the *projection* of fM:
    SolutionPtr projection_fM = Solution::solution(form.boltzmannMesh()->bilinearForm(),form.boltzmannMesh());
    std::map<int,FunctionPtr> fM_projectionMap;
    fM_projectionMap[form.f()->ID()] = fM;
    projection_fM->projectOntoMesh(fM_projectionMap);
    FunctionPtr fM_soln = Function::solution(form.f(), projection_fM);
    
    FunctionPtr f_soln  = Function::solution(form.f(), form.boltzmannSolution());
    FunctionPtr df_soln = f_soln - fM_soln; // what Heath et al. refer to as delta f: the difference between solution and equilibrium.
    
    
    
    auto f0_err  = (f_soln - f0)->l2norm(form.boltzmannMesh());
    auto f0_norm = f0->l2norm(form.boltzmannMesh());
    if (rank == 0)
    {
      std::cout << "Relative projection error (L^2) of initial conditions: " << f0_err / f0_norm << std::endl;
    }
    
    auto df0_err  = (df0 - df_soln)->l2norm(form.boltzmannMesh());
    auto df0_norm = df0->l2norm(form.boltzmannMesh());
    if (rank == 0)
    {
      std::cout << "Projection error (L^2) of df0, relative to norm of f: " << df0_err / f0_norm << std::endl;
      std::cout << "Projection error (L^2) of df0, relative to norm of df: " << df0_err / df0_norm << std::endl;
    }
    
    double t = 0.0;
    if (saveViz)
    {
      gaussViz->exportSolution(form.gaussSolution(),t);
      std::vector<FunctionPtr> otherFunctions {df_soln, df0};
      std::vector<std::string> otherFunctionNames {"df", "df0"};
      boltzmannViz->exportSolution(form.boltzmannSolution(),otherFunctions,otherFunctionNames,t);
    }
    
    FunctionPtr E_soln = Function::solution(form.E(), form.gaussSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form.boltzmannSolution(), form.f(), form.gaussMesh()) );
        
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    auto velocity = momentum / density;
        
    form.setPerformRefinementsDuringTimeSteps(false);
    
    const double f0_L2 = f0->l2norm(form.boltzmannMesh());
    auto reportExactError = [&](double t) {
      if (pureAdvection)
      {
        FunctionPtr cos_kxv_minus_t = Teuchos::rcp(new Cos_kx_shifted(k_x, t) );
        FunctionPtr f_exact = (1. + gamma * cos_kxv_minus_t) * fM;
        
        FunctionPtr err = f_exact - f_soln;
        const double rel_err_L2 = err->l2norm(form.boltzmannMesh()) / f0_L2;
        if (rank == 0)
        {
          std::cout << "rel_err_L2 = " << rel_err_L2 << std::endl;
        }
      }
    };
    
    auto getLogMaxNetDensity = [&]() -> double
    {
      {
        // the di() methods are 1-based:
        static const int SPATIAL_INDEX  = 1;
        static const int VELOCITY_INDEX = 1;
        
        // SpatialFilter's components are zero-based
        static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
        static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
        
        auto allActivePhysicalCells = form.gaussMesh()->getActiveCellIDsGlobal();
        
        form.gaussSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
        
        // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
        solutionMoments->update();
        solutionMoments->gatherOnRankZero();
        
        const GlobalIndexType firstPhysicalCellID = *allActivePhysicalCells.begin();
        
        double logMaxDensity = 0;
        if (rank == 0)
        {
          // (We could revise this to compute max density locally on each rank; that would be more efficient than gathering the whole solution to rank 0.)
          const int cubatureEnrichmentDegree = 5; // over-integrate to get more sample points
          auto basisCache = BasisCache::basisCacheForCell(form.gaussMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
          std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
          
          const int numPoints = 101; // 101 sample points per cell -- includes the endpoints
          FieldContainer<double> refPoints(numPoints,1); // 1: spaceDim
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            // ref space goes from -1 to 1
            refPoints(pointOrdinal,0) = -1. + pointOrdinal * 2. / (numPoints-1);
          }
          basisCache->setRefCellPointsFC(refPoints);
                    
          auto physicalCellNodes = form.gaussMesh()->physicalCellNodes(physicalCellIDsVector);
          basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
          
          FieldContainer<double> density_values_FC(physicalCellIDsVector.size(), numPoints);
          density->values(density_values_FC, basisCache);
          
          const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
          
          const int numCells = physicalQuadraturePoints.dimension(0);
          
          double maxNetDensity = std::numeric_limits<double>::lowest();
          for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
          {
            const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
            for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
            {
              const auto &density = density_values_FC(cellOrdinal,pointOrdinal);
              maxNetDensity = std::max(std::abs(1-density), maxNetDensity);
            }
          }
          
//          std::cout << "maxNetDensity: " << maxNetDensity << std::endl;
          logMaxDensity = std::log10(maxNetDensity);
        }
        
        logMaxDensity = MPIWrapper::sum(logMaxDensity); // only rank 0 non-zero, so this is same as sending from rank 0.
        return logMaxDensity;
      }
    };
    
    // computes the sinusoidal part of the nth Fourier mode
    auto getElectricFieldSineMode = [&](const int &n) -> double
    {
      const double T = xWidth;
      FunctionPtr sinFxn = Teuchos::rcp(new Sin_ax(2. * n * M_PI / T, 0.));
      
      const double integral = (E_soln * sinFxn)->integrate(form.gaussMesh());
      return 2.0 / T * integral;
    };
    
    double previousLogEFieldSineMode = std::log(std::abs(getElectricFieldSineMode(1)));
    bool logEFieldSineModeIsIncreasing = false; // if it was increasing, and then decreases: local maximum
    
    std::vector<std::pair<double,double> > t_ln_b1_pairs;      // record (t, ln(abs(b_1))) at each time step
    std::vector<std::pair<double,double> > t_ln_b1_peak_pairs; // record (t, ln(abs(b_1))) at each local maximum of b_1, where b_1 is getElectricFieldSineMode(1)
    
    double mode1DecayRate = 0.0;
    
    auto numPoissonDofs = form.gaussMesh()->numGlobalDofs();
    if (rank == 0)
    {
      std::cout << "numPoissonDofs = " << numPoissonDofs << std::endl;
    }
    {
      auto numVlasovDofs = form.boltzmannSolution()->mesh()->numGlobalDofs();
      if (numVlasovDofs > maxDofsForDirectSolve)
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
        }
        form.setUseGeometricMultigrid(true);
      }
      else
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
        }
        form.setUseGeometricMultigrid(false);
      }
    }
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      form.takeTimeStep();
      t += dt;
      
      const double currentLogEFieldSineMode = std::log(std::abs(getElectricFieldSineMode(1)));
      
      if (logEFieldSineModeIsIncreasing && (currentLogEFieldSineMode < previousLogEFieldSineMode))
      {
        // reversal: previous is a local maximum:
        t_ln_b1_peak_pairs.push_back({t-dt,previousLogEFieldSineMode});
        if ((rank == 0) && (!pureAdvection))
        {
          std::cout << "local maximum of logarithm of dominant E-field mode at t=" << t-dt << ": " << previousLogEFieldSineMode;
          if (t_ln_b1_peak_pairs.size() >= 2)
          {
            double t_prev  = t_ln_b1_peak_pairs[t_ln_b1_peak_pairs.size() - 2].first;
            double b1_prev = t_ln_b1_peak_pairs[t_ln_b1_peak_pairs.size() - 2].second;
            double rate = (previousLogEFieldSineMode - b1_prev) / (t - dt - t_prev);
            std::cout << " (rate: " << rate << ")";
          }
          std::cout << std::endl;
        }
        if (t_ln_b1_peak_pairs.size() == 7)
        {
          const double thirdPeak   = t_ln_b1_peak_pairs[2].second;
          const double seventhPeak = t_ln_b1_peak_pairs[6].second;
          
          const double third_t   = t_ln_b1_peak_pairs[2].first;
          const double seventh_t = t_ln_b1_peak_pairs[6].first;
          
          mode1DecayRate = (seventhPeak - thirdPeak) / (seventh_t - third_t);
          if (rank == 0)
          {
            std::cout << "Overall decay rate: " << mode1DecayRate << std::endl;
          }
        }
        
      }
      logEFieldSineModeIsIncreasing = (currentLogEFieldSineMode > previousLogEFieldSineMode);
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        t_ln_b1_pairs.push_back({t,currentLogEFieldSineMode});
        
        reportExactError(t); // only does something when pureAdvection is set
        
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          std::vector<FunctionPtr> otherFunctions {df_soln};
          std::vector<std::string> otherFunctionNames {"df"};
          boltzmannViz->exportSolution(form.boltzmannSolution(),otherFunctions,otherFunctionNames,t);
          gaussViz->exportSolution(form.gaussSolution(), {density,velocity}, {"n_e","v"},t);
        }
        
        if (pureAdvection)
        {
          const double currentLogMaxNetDensity = getLogMaxNetDensity();
          const double   exactLogMaxNetDensity = - std::log10(M_E) * t * t / 8. - 1.; // caption to Heath figure 1, but with the second term negated (my correction).
          if (rank == 0)
          {
            std::cout << "log of max net density (actual): " << currentLogMaxNetDensity << std::endl;
            std::cout << "log of max net density (exact):  " << exactLogMaxNetDensity << std::endl;
          }
        }
        else
        {
          if (rank == 0)
          {
            std::cout << "Completed time step " << timeStepNumber << std::endl;
          }
        }
        
//        double E_mag = E_squared->integrate(form.gaussMesh());
      }
      previousLogEFieldSineMode = currentLogEFieldSineMode;
    } // end time step loop
    
    FunctionPtr potential = Function::solution(form.VE(), form.gaussSolution());
    
    // output to EFieldMode1File,
    //t_ln_b1_pairs;
    if (rank == 0)
    {
      std::ofstream EFieldMode1FileStream(EFieldMode1File, std::ios::out);
        
      EFieldMode1FileStream << "t\tln(abs(b_1))\n";
      EFieldMode1FileStream << setprecision(15);
      
      for (const auto & t_ln_b1 : t_ln_b1_pairs)
      {
        const auto &t     = t_ln_b1.first;
        const auto &ln_b1 = t_ln_b1.second;
        
        EFieldMode1FileStream << t     << "\t";
        EFieldMode1FileStream << ln_b1 << "\n";
      }
      
      EFieldMode1FileStream.close();
      
      std::ofstream EFieldMode1DecayRateFileStream(EFieldMode1DecayRateFile, std::ios::out);
      
      EFieldMode1DecayRateFileStream << setprecision(15);
      EFieldMode1DecayRateFileStream << "E-Field, Mode 1 Decay Rate: " << mode1DecayRate << std::endl;
      
      EFieldMode1DecayRateFileStream.close();
    }
    
    // output to results file: // TODO: consider deleting results output; we don't actually need it for Landau
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      // the di() methods are 1-based:
      static const int SPATIAL_INDEX  = 1;
      static const int VELOCITY_INDEX = 1;
      
      // SpatialFilter's components are zero-based
      static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
      static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
      
      auto allActivePhysicalCells = form.gaussMesh()->getActiveCellIDsGlobal();
      
      form.gaussSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
      
      // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
      solutionMoments->update();
      solutionMoments->gatherOnRankZero();
      
      if (rank == 0)
      {
        std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> ne_values; // map from x coordinate to final density value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> velocity_values; // map from x coordinate to final velocity value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> potential_values; // map from x coordinate to final potential value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
        GlobalIndexType firstPhysicalCellID = *allActivePhysicalCells.begin();
        
        const int cubatureEnrichmentDegree = 5; // over-integrate to make up for undermeasurement on coarse initial meshes
        auto basisCache = BasisCache::basisCacheForCell(form.gaussMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
        std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
        
        const int numPoints = basisCache->getRefCellPoints().extent_int(0);
        auto & cubWeights = basisCache->getCubatureWeightsFC();
        
        auto physicalCellNodes = form.gaussMesh()->physicalCellNodes(physicalCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
        
        FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
        E_soln->values(E_values_FC, basisCache);
        
        FieldContainer<double> potential_values_FC(physicalCellIDsVector.size(), numPoints);
        potential->values(potential_values_FC, basisCache);
        
        FieldContainer<double> density_values_FC(physicalCellIDsVector.size(), numPoints);
        density->values(density_values_FC, basisCache);
        
        FieldContainer<double> velocity_values_FC(physicalCellIDsVector.size(), numPoints);
        velocity->values(velocity_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells         = physicalQuadraturePoints.dimension(0);
        
        double maxDensity = std::numeric_limits<double>::lowest();
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
        {
          const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
            const double pointWeight = cubWeights(pointOrdinal) * cellWidth / 2.0; // 2.0: width of the reference line.
            
            E_values[x]           = E_values_FC(cellOrdinal,pointOrdinal);
            potential_values[x]   = potential_values_FC(cellOrdinal,pointOrdinal);
            ne_values[x]          = density_values_FC(cellOrdinal,pointOrdinal);
            velocity_values[x]    = velocity_values_FC(cellOrdinal,pointOrdinal);
            pointWeight_values[x] = pointWeight;
            
            maxDensity = std::max(ne_values[x], maxDensity);
          }
        }
        
        std::ofstream resultsFileStream(resultsFile, std::ios::out);
        
        resultsFileStream << "x\tquadrature weight\tE\tphi\tne\tvx\n";
        resultsFileStream << setprecision(15);
        
        for (const auto & pointEntry : pointWeight_values)
        {
          const auto x           = pointEntry.first;
          const auto pointWeight = pointEntry.second;
          const auto E           = E_values[pointEntry.first]        ;
          const auto potential   = potential_values[pointEntry.first];
          const auto ne          = ne_values[pointEntry.first]       ;
          const auto velocity    = velocity_values[pointEntry.first] ;
          
          resultsFileStream << x           << "\t";
          resultsFileStream << pointWeight << "\t";
          resultsFileStream << E           << "\t";
          resultsFileStream << potential   << "\t";
          resultsFileStream << ne          << "\t";
          resultsFileStream << velocity    << "\n";
        }
        
        resultsFileStream.close();
        
        auto spatialMeshFile = getSpatialMeshFilePath();
        std::ofstream spatialMeshFileStream(spatialMeshFile, std::ios::out);
        // let's collect the spatial nodes and sort them
        std::set<double> spatialNodes;
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
        {
          spatialNodes.insert(physicalCellNodes(cellOrdinal,0,0)); // (min x, min t)
          spatialNodes.insert(physicalCellNodes(cellOrdinal,1,0)); // (max x, min t)
        }
        resultsFileStream << "x\n";
        for (const auto & node : spatialNodes)
        {
          spatialMeshFileStream << node << "\n";
        }
        spatialMeshFileStream.close();
      } // rank == 0
    } // results file output
  } // block off variables so dtors get called before finalize()
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
