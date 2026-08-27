//
// © 2016-2024 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "ConfigurationSpaceFormulation.h"
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
#include "PhysicalSolutionExtrusion.hpp"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovSpaceTimeFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <iomanip>

using namespace Camellia;

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

    for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
      }
    }
  }
};

static const std::string S_TWO_STREAM   = "two-stream";

enum ProblemChoice
{
  TWO_STREAM
};

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_TWO_STREAM,TWO_STREAM}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{TWO_STREAM,S_TWO_STREAM}};

ProblemChoice getProblemChoice(const std::string &problemChoiceString)
{
  const auto &entry = stringToProblemChoice.find(problemChoiceString);
  if (entry == stringToProblemChoice.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice string");
  }
  else
  {
    return entry->second;
  }
}

std::string getProblemChoiceString(ProblemChoice problem)
{
  const auto &entry = problemChoiceToString.find(problem);
  if (entry == problemChoiceToString.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice");
  }
  else
  {
    return entry->second;
  }
}

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const double x0, const double xWidth,
                                                              const double t0, const double tWidth,
                                                              const double v0, const double vWidth,
                                                              const int xElements, const int tElements, const int vElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const bool usePeriodicBCs, const EBasisFamily &basisFamily,
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
  
  form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                     temporalVertices,
                                                     velocitySpaceVertices_x,
                                                     velocitySpaceVertices_y,
                                                     velocitySpaceVertices_z,
                                                     H1Order, delta_k,
                                                     numUniformRefinements,
                                                     basisFamily));
  return form;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
  {
    mpiSession.barrier();
    
    bool implicit = true;
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // computational density
    
    double q      =   1.0;
    double eps0   =   1.0;
    double m      =   1.0;
    double x0     =   0.0;
    double xWidth =  2 * M_PI / k_x;
    double t0     =   0.0;
    double tWidth =  50.0;
    
    const double DEFAULT_V0     = -10.0;
    const double DEFAULT_VWIDTH =  20.0;
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = true;
    bool useSerendipityBasis = false;
    
    int numTimeSlabs = 25;
    
    int xElements = 32;
    int vElements = 20;
    int tElements = 2;
    int delta_k   = 4;
    int H1Order = 4; // this gives f, E (which are in L^2) a polynomial order of 3
    int outputFrequency = 10; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    std::string problemChoiceString = "two-stream";
    ProblemChoice problemChoice = TWO_STREAM;
    bool saveViz = true;
    std::string outputDir = ".";
    std::string inputDir  = ""; // set this to compute error between full representation and an approximation stored (explicitly) in inputDir; filenames should match those generated on output
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    
    const double initialTimeDivision = 0;
    
    int maxDofsForDirectSolve = 20000;
    int numUniformRefinements = 0;
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    
    bool useHierarchicalBasis = true;
    
    std::string resultsFilePrefix = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("p", &H1Order);
    cmdp.setOption("dp", &delta_k);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("inputDir", &inputDir, "directory with approximate (TT) basis coefficients for comparison");
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("numTimeSlabs", &numTimeSlabs);
    cmdp.setOption("problem", &problemChoiceString);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("useHierarchicalBasis", "useNodalBasis", &useHierarchicalBasis);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    problemChoice = getProblemChoice(problemChoiceString);
    
    vector<double> fL2NormForSlab(numTimeSlabs, 0.0); // norm of computed discrete solution on each time slab
    double fL2NormSquared = 0; // squared L^2 norm, cumulative across all time slabs.
    vector<double> fL2ErrorForSlab(numTimeSlabs, 0.0); // absolute error between input solution and computed discrete solution, on each time slab
    double fL2ErrorSquared = 0; // cumulative squared L^2 norm of the error across all time slabs
    
    // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
    double velocityScaling  = 1.;
    double timeScaling      = 1.;
    double spatialScaling   = 1.;
    double EScaling         = 1.;
    double nScaling         = 1.;
    double potentialScaling = 1.;
    
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = DEFAULT_VWIDTH;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = DEFAULT_V0;
    }
    
    bool usePeriodicBCs = true;
    
    if (problemChoice == TWO_STREAM)
    {
      if (rank == 0)
      {
        std::cout << "Running two-stream problem; time domain goes from 0 to " << tWidth << ".\n";
      }
    }
    else
    {
      std::cout << "Unsupported problem choice.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported problem choice.");
    }
    
    tWidth /= numTimeSlabs;
    
    EBasisFamily basisFamily = useHierarchicalBasis ? BASIS_FAMILY_HIERARCHICAL : BASIS_FAMILY_NODAL;
    
    auto form = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                     basisFamily, numUniformRefinements);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    auto globalCellOrdinal = [&](const int &cellX, const int &cellT, const int &cellV) -> GlobalIndexType
    {
      // cell index first: x fastest-moving
      GlobalIndexType cellIndex = cellX + cellT * xElements + cellV * xElements * tElements;
      return cellIndex;
    };
    
    auto fBasis = form->vlasovMesh()->getElement(0)->elementType()->trialOrderPtr->getBasis(form->f()->ID());
    int fCardinality = fBasis->getCardinality();
    int fCardinality1D = fBasis->intrepid2TensorBasisComponents()[0]->getCardinality(); // we're assuming isotropic here
    
    auto localDofOrdinal = [&](const int &basisX, const int &basisT, const int &basisV) -> GlobalIndexType
    {
      // local dof index: x fastest-moving
      GlobalIndexType dofIndex = basisX + basisT * fCardinality1D + basisV * fCardinality1D * fCardinality1D;
      
      return dofIndex;
    };
    
    using std::vector;
    using std::pair;
    vector<pair<pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true
    
    // we'll use this function to allow labeling of visualization output
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
    
    if (numUniformRefinements > 0)
    {
      for (int i=0; i<numUniformRefinements; i++)
      {
        const auto & activeCells  = form->vlasovMesh()->getActiveCellIDsGlobal();
        form->vlasovMesh()->hRefine(activeCells);
        
        xElements *= 2;
        tElements *= 2;
        vElements *= 2;
        
        // it should be the case that Poisson mesh is automatically refined when the Vlasov mesh is
      }
    }
    
    FunctionPtr f0;
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    
    if (problemChoice == TWO_STREAM)
    {
      double n = 0.5 * n0;
      FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n) );
      FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1T1V(-u0, sigma, n) );
      f0                       = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    }
    
    form->setInitialConditions(f0);
    
    // need to pin the potential at 0 to eliminate the constant mode.
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(x0);
    bool restrictToBoundary = false; // periodic; x0 = 0 does *not* lie on a mesh boundary
    auto xLeftTag = form->createConfigurationMeshSideSetTag(xLeftFilter, restrictToBoundary);
    auto VE_traceVar = form->VE_configurationSpaceTrace();
    form->addConfigurationSpaceBC(xLeftTag, VE_traceVar, Function::zero());
    
    std::ostringstream suffix;
    if (!useL2Trace)
    {
      suffix << "_H1Trace";
    }
    suffix << "_u0" << u0;
    suffix << "_p" << H1Order;
    suffix << "_dp" << delta_k;
    suffix << "_" << xElements << "x" << tElements << "x" << vElements;
    if (useHierarchicalBasis)
      suffix << "_hierarchical";
    else
      suffix << "_nodal";
    if (initialTimeDivision > 0)
    {
      suffix << "_timeRamp" << initialTimeDivision;
    }
    
    const std::string problemPrefix = problemChoiceString;
    
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
      resultsFileStream << ".dat";
      
      return resultsFileStream.str();
    };
    
    auto getSpatialMeshFilePath = [&](int refNumber = -1) {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << problemPrefix << "_mesh1D" << suffix.str();
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
    
    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
        
    const std::string directoryPath = outputDir;
    
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
    if (saveViz)
    {
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    FunctionPtr E_soln = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    //  form->vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
    
    FunctionPtr potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
    
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    auto velocity = momentum / density;
    
    for (int timeSlabOrdinal=0; timeSlabOrdinal<numTimeSlabs; timeSlabOrdinal++)
    {
      std::string basisChoice = useHierarchicalBasis ? "hierarchical" : "nodal";
      std::string fFilename = "f_" + basisChoice + to_string(numTimeSlabs) + "slabs_p" + to_string(H1Order) + "_dp" + to_string(delta_k) + "_" + to_string(xElements) + "x" + to_string(tElements) + "x" + to_string(vElements) + "_slab" + std::to_string(timeSlabOrdinal) + ".dat";
      
      if (rank == 0)
      {
        std::cout << "\n\n";
        std::cout << "*******************************************************************\n";
        std::cout << "*                     Starting Time Slab " << timeSlabOrdinal;
        if      (timeSlabOrdinal < 10)   std::cout << "                        *\n";
        else if (timeSlabOrdinal < 100)  std::cout << "                       *\n";
        else if (timeSlabOrdinal < 1000) std::cout << "                      *\n";
        else                             std::cout << "                     *\n";
        std::cout << "*******************************************************************\n\n\n";
      }
      
      const auto & myCells = form->vlasovMesh()->cellIDsInPartition();
      
      // if an inputDir was specified, set up approxSoln to store the f coefficients specified there
      SolutionPtr approxSoln;
      Camellia::FieldContainer<double> fInputCoefficients(xElements*tElements*vElements, fCardinality); // initialized to 0
      if (inputDir.length() > 0)
      {
        approxSoln = Solution::solution(form->vlasovMesh());
        
        // we read in redundantly on every rank, just to avoid writing the MPI communication code
        std::string inputFilename = inputDir + "/" + fFilename;
        if (rank == 0) std::cout << "Reading from file: " << inputFilename << std::endl;
        std::ifstream fFileStream(inputFilename, std::ios::in);
        
        if (!fFileStream) {
          if (rank == 0)
          {
            std::cerr << "ERROR: Could not open file " << inputFilename << std::endl;
          }
          
#ifdef HAVE_MPI
          MPI_Finalize();
#endif
          return -1;
        }
        
        // this loop must match the output loop (if that changes, this should, too)
        for (int cv=0; cv<vElements; cv++)
        {
          for (int fv=0; fv<fCardinality1D; fv++)
          {
            for (int ct=0; ct<tElements; ct++)
            {
              for (int ft=0; ft<fCardinality1D; ft++)
              {
                for (int cx=0; cx<xElements; cx++)
                {
                  for (int fx=0; fx<fCardinality1D; fx++)
                  {
                    GlobalIndexType cellIndex = globalCellOrdinal(cx, ct, cv);
                    GlobalIndexType  dofIndex = localDofOrdinal(fx, ft, fv);
                    
                    int globalDofIndex;
                    double dofValue;
                    
                    fFileStream >> globalDofIndex;
                    fFileStream >> dofValue;
                    
                    fInputCoefficients(cellIndex,dofIndex) = dofValue;
                  }
                }
              }
            }
          }
        }
        
        for (auto myCellID : myCells)
        {
          Camellia::FieldContainer<double> fDofs(fCardinality);
          for (int dofOrdinal=0; dofOrdinal<fCardinality; dofOrdinal++)
          {
            fDofs(dofOrdinal) = fInputCoefficients(myCellID,dofOrdinal);
          }
          
          approxSoln->setSolnCoeffsForCellID(fDofs, myCellID, form->f()->ID(), VOLUME_INTERIOR_SIDE_INDEX);
        }
        
        auto f_approx = Function::solution(form->f(), approxSoln);
        double f_approx_l2 = f_approx->l2norm(form->vlasovMesh());
        
        if (rank == 0)
        {
          std::cout << "f_approx_l2: " << setprecision(3) << f_approx_l2 << std::endl;
        }
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
      
      form->solve();
      
      solutionMoments->update();
      
      if (saveViz)
      {
        std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
        std::string poissonSlabStr = poissonSS.str() + "_slab" + to_string(timeSlabOrdinal);
        
        auto vlasovVizSlab  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
        auto poissonVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSlabStr, directoryPath) );
        
        poissonVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
        vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"});
      }
      
      // gather all the f coefficients to rank 0
      // a simple way to do this is just to write owned coefficients into an array of 0s, and then
      // sum across all MPI ranks.
      Camellia::FieldContainer<double> fCoefficients(xElements*tElements*vElements, fCardinality); // initialized to 0
      auto vlasovSoln = form->vlasovSolution();
      for (auto myCellID : myCells)
      {
        Camellia::FieldContainer<double> fDofs(fCardinality);
        vlasovSoln->getBasisCoefficientsForCell(fDofs, form->f()->ID(), myCellID, STANDARD_SOLUTION, 0);
        for (auto dofOrdinal=0; dofOrdinal<fCardinality; dofOrdinal++)
        {
          fCoefficients(myCellID, dofOrdinal) = fDofs(dofOrdinal);
        }
      }
      MPIWrapper::entryWiseSum(form->vlasovMesh()->Comm(), fCoefficients);
      
      if (inputDir.length() > 0)
      {
        auto f_soln   = Function::solution(form->f(), vlasovSoln);
        auto f_approx = Function::solution(form->f(), approxSoln);
        
        FunctionPtr f_err = f_soln - f_approx;
        double l2_err = f_err->l2norm(form->vlasovMesh());
        
        // compute L^2 norm of the solution so we can give a relative error value.
        double f_l2 = f_soln->l2norm(form->vlasovMesh());
        
        double f_approx_l2 = f_approx->l2norm(form->vlasovMesh());
        
        fL2NormForSlab[timeSlabOrdinal]  = f_l2;
        fL2ErrorForSlab[timeSlabOrdinal] = l2_err;
        
        fL2NormSquared  += f_l2 * f_l2;
        fL2ErrorSquared += l2_err * l2_err;
        
        if (rank == 0)
        {
          std::cout << "\n\n";
          
          std::cout << "Comparing input solution with computed solution:\n";
          std::cout << "f_l2:        " << setprecision(3) << f_l2 << std::endl;
          std::cout << "f_approx_l2: " << setprecision(3) << f_approx_l2 << std::endl;
          std::cout << "Relative L^2 error between input and full representation: " << setprecision(3) << l2_err / f_l2 << std::endl;
          
          std::cout << "\n\n";
        }
        
      } // inputDir.length() > 0
      
      if (rank == 0)
      {
        // write f to file on rank 0
        // f_25slabs_p4_dp4_32x2x20
        using std::to_string;
        std::string outputFilename = outputDir + "/" + fFilename;
        std::ofstream fFileStream(outputFilename, std::ios::out);
        
        fFileStream << setprecision(15);
        
        // this loop determines the output order
        int globalDofIndex = 1; // 1-based for MATLAB-style output
        for (int cv=0; cv<vElements; cv++)
        {
          for (int fv=0; fv<fCardinality1D; fv++)
          {
            for (int ct=0; ct<tElements; ct++)
            {
              for (int ft=0; ft<fCardinality1D; ft++)
              {
                for (int cx=0; cx<xElements; cx++)
                {
                  for (int fx=0; fx<fCardinality1D; fx++)
                  {
                    GlobalIndexType cellIndex = globalCellOrdinal(cx, ct, cv);
                    GlobalIndexType  dofIndex = localDofOrdinal(fx, ft, fv);
                    
                    fFileStream << globalDofIndex << "\t" << fCoefficients(cellIndex,dofIndex) << std::endl;
                    globalDofIndex++;
                  }
                }
              }
            }
          }
        }
        fFileStream.close();
        std::cout << "Wrote f coefficient data to file: " << outputFilename << std::endl;
        std::cout << "Ordering is layout-left (fx,cx,ft,ct,fv,cv): ";
        std::cout << fCardinality1D << " x ";
        std::cout << xElements      << " x ";
        std::cout << fCardinality1D << " x ";
        std::cout << tElements      << " x ";
        std::cout << fCardinality1D << " x ";
        std::cout << vElements      << std::endl;
      }
            
      // next time slab
      form = form->nextTimeSlab(tWidth, tElements);
      
      {
        // solutions have been recreated for the new slab.  We need to update anything that depends on the solution
        const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
        solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
        
        potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
        
        density  = solutionMoments->solutionMomentFunction(0);
        momentum = solutionMoments->solutionMomentFunction(1);
        
        velocity = momentum / density;
      }
    } // timeSlabOrdinal loop
    
    if ((rank == 0) && (inputDir != ""))
    {
      // Header
      std::cout << std::left << std::setw(15) << "Time Slab"
                << std::setw(30) << "Norm of f"
                << std::setw(30) << "Absolute Error"
                << std::setw(30) << "Relative Error" << std::endl;

      // Separator
      std::cout << std::string(105, '-') << std::endl;
      
      // Print the data
      for (size_t timeSlabOrdinal = 0; timeSlabOrdinal < numTimeSlabs; timeSlabOrdinal++) {
          std::cout << std::left << std::setw(15) << timeSlabOrdinal
                    << std::scientific << std::setprecision(3) << std::setw(30) << fL2NormForSlab[timeSlabOrdinal]
                    << std::setw(30) << fL2ErrorForSlab[timeSlabOrdinal]
                    << std::setw(30) << fL2ErrorForSlab[timeSlabOrdinal] / fL2NormForSlab[timeSlabOrdinal] << std::endl;
      }
      
      // Separator
      std::cout << std::string(105, '-') << std::endl << std::endl;
      
      // Totals
      std::cout << std::left << std::setw(15) << "All Slabs"
                << std::scientific << std::setprecision(3) << std::setw(30) << sqrt(fL2NormSquared)
                << std::setw(30) << sqrt(fL2ErrorSquared)
                << std::setw(30) << sqrt(fL2ErrorSquared) / sqrt(fL2NormSquared) << std::endl;
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
