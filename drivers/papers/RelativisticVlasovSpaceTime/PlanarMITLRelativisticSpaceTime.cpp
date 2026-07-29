//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaMemoryTools.h"
#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
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
#include "TrigFunctions.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

namespace
{
  bool gEnforcePositivity = false;
  double gProximalGalerkinAlpha0 = 1.0;
  double gProximalGalerkinAlphaMultiplier = 1.0;
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
    
    const double weight = n / (sqrt(2. * M_PI) * sigma);
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

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T2V(const double c,
                                                              const double  x0, const double  xWidth,
                                                              const double  t0, const double  tWidth,
                                                              const double vx0, const double vxWidth,
                                                              const double vy0, const double vyWidth,
                                                              const int xElements,  const int tElements,
                                                              const int vxElements, const int vyElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const RelativisticVlasovSpaceTimeFormulation::TestSpaceInnerProduct testIP,
                                                              const double L2WeightGraph,
                                                              const bool usePeriodicBCs, const bool useSerendipity,
                                                              const bool useGradedSpatialMesh,
                                                              const int numUniformRefinements)
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
  
  std::vector<double> velocitySpaceVertices_x(vxElements+1);
  for (int i=0; i<=vxElements; i++)
  {
    const double vx = double(i) / double(vxElements) * vxWidth + vx0;
    velocitySpaceVertices_x[i] = vx;
  }
  
  std::vector<double> velocitySpaceVertices_y(vyElements+1);
  for (int i=0; i<=vyElements; i++)
  {
    const double vy = double(i) / double(vyElements) * vyWidth + vy0;
    velocitySpaceVertices_y[i] = vy;
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
                                                                 gEnforcePositivity));
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
    
    // proper velocity domain, specified as fraction of speed of light:
    double beta0                = 0.00;
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
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }
    
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
    
    const double uyMin = uMin;
    const double uyMax = uMax;
    
    const auto vxElements = vElements;
    const auto vyElements = vElements;
    
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
      std::cout << "sigma:    " << sigma << std::endl;
      std::cout << "u0:       " << ux0 << std::endl;
      std::cout << "v0:       " << v0 << std::endl;
      std::cout << "gamma0:   " << gamma_beam << std::endl;
      std::cout << "n0:       " << n0 << std::endl;
      std::cout << "omega_p:  " << omega_p << std::endl;
    }
    
    const bool useGeometricGradedSpatialMesh = true;
    
    auto form = getFormulation1D1T2V(cScaled, x0, xWidth, t0, tWidth, uxMin, uxMax-uxMin, uyMin, uyMax-uyMin, xElements, tElements, vxElements, vyElements,
                                     q, eps0, m, H1Order, delta_k, testIP, L2WeightGraph, usePeriodicBCs,
                                     useSerendipityBasis, useGeometricGradedSpatialMesh, numUniformRefinements);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
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
        
        form->setInitialAndInflowConditions(f0, f_leftInflow, initialTimeDivision); // no right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        if (useAdaptivity)
        {
          bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
        }
      }
    }
    
    refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form->configurationSpaceMesh()->repartitionAndRebuild();
    if (rank == 0)
    {
      std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
    }
    
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
    suffix << "_" << xElements << "x" << tElements << "x" << vElements;
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
    
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    //  form->vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
    
    FunctionPtr potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
    
    auto density    = solutionMoments->solutionMomentFunction(0,0);
    auto momentum   = solutionMoments->solutionMomentFunction(1,0); // x proper velocity moment
    auto momentum_z = solutionMoments->solutionMomentFunction(0,1); // second retained component moment; in 1D2V this is z
    
    auto velocity   = momentum   / density;
    auto velocity_z = momentum_z / density;
    
    auto relativisticForm = Teuchos::rcp_dynamic_cast<RelativisticVlasovSpaceTimeFormulation>(form);
    TEUCHOS_TEST_FOR_EXCEPTION(relativisticForm == Teuchos::null, std::invalid_argument, "Expected RelativisticVlasovSpaceTimeFormulation in PlanarMITLRelativisticSpaceTime");
    auto Jx = relativisticForm->currentX();
    auto Jz = relativisticForm->currentZ();
    auto vx_rel_mean = Jx / (q * density);
    auto vz_rel_mean = Jz / (q * density);
    
    const int speciesOrdinal = 0;
    auto f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
    auto f_positive = Function::max(f_soln, Function::constant(0.0));
    auto f_positive_soln = Solution::solution(form->vlasovMesh());
    {
      std::map<int,FunctionPtr> projectionMap;
      projectionMap[form->f(speciesOrdinal)->ID()] = f_positive;
      f_positive_soln->projectOntoMesh(projectionMap);
    }
    auto positiveMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, f_positive_soln, form->f(speciesOrdinal), form->configurationSpaceMesh()));
    positiveMoments->update();
    auto density_positive = positiveMoments->solutionMomentFunction(0,0);
    auto vx_rel_positive_moment = relativisticForm->relativisticFirstMoment(0, speciesOrdinal, positiveMoments.get(), f_positive_soln);
    auto vz_rel_positive_moment = relativisticForm->relativisticFirstMoment(1, speciesOrdinal, positiveMoments.get(), f_positive_soln);
    auto vx_rel_mean_clipped = vx_rel_positive_moment / density_positive;
    auto vz_rel_mean_clipped = vz_rel_positive_moment / density_positive;
        
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
        
        form->solve();
        
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
          std::cout << "Updating solution moments.\n";
        }
        
        solutionMoments->update();
        positiveMoments->update();
        
        if (rank == 0)
        {
          std::cout << "Solution moments updated.\n";
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
          maxwellVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,velocity_z,Jx,Jz,vx_rel_mean,vz_rel_mean,vx_rel_mean_clipped,vz_rel_mean_clipped,cellIDFunction}, {"n_e","ux_mean","uz_mean_raw","Jx","Jz","vx_rel_mean_signed","vz_rel_mean_signed","vx_rel_mean_clipped","vz_rel_mean_clipped","cell ID (Maxwell)"}, vizTimeValue, num1DPoints);
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
        
        auto configMesh = form->configurationSpaceMesh();
        FunctionPtr E_soln = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
        
        // Evaluate just inside the slab end, rather than exactly at t1(), to avoid
        // floating-point / point-location boundary ambiguity at the time-slab boundary.
        const double tEval = std::nextafter(form->t1(), form->t0());
        double E_left = E_soln->evaluate(configMesh, x0, tEval); // when called with mesh arg, evaluate() locates the owning rank, evaluates there, and then MPI-communicates the result to all (without mesh arg, it will silently get a wrong (0.0) answer on some ranks!)
        
//        {
//          // DEBUGGING:
//          std::cout << "On rank " << rank << ", E_left: " << E_left << std::endl;
//        }
        
        // reduce the inflow current when E_left / E0 gets small -- want to avoid/mitigate virtual cathode creation.
        double lowerThreshold = 0.10; // turn off once we get here
        double upperThreshold = 0.50; // start tapering here
        double ratio = E_left / E0; // E0 is negative -> if E_left is negative, ratio will be positive
        if (ratio < lowerThreshold)
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
        // DEBUGGING: for now, we turn off tapering by setting ratio to 1.0
        ratio = 1.0;
        double n = n0 * ratio;
        
        fM_u0->set_n(n);
        
        if (rank == 0)
        {
          double percent = int(ratio * 10000) / 100.0; // keep two digits after the decimal point
          double upperThresholdPercent = int(upperThreshold * 100);
          double lowerThresholdPercent = int(lowerThreshold * 100);
          std::cout << "E(0) at final time is " << E_left << "; setting injection current to " << percent << "%.\n";
          double EPercent = int(E_left / E0 * 100);
          if (ratio >= 1)
          {
            std::cout << "Will begin reducing injection current once E(0) is at " << upperThresholdPercent << "% of initial E(0) value (now at ";
            std::cout << EPercent << "%)\n";
          }
          else if (ratio > 0)
          {
            std::cout << "Will turn off injection once E(0) is at " << lowerThresholdPercent << "% of initial E(0) value (now at ";
            std::cout << EPercent << "%)\n";
          }
        }
        
        // clear solution-backed diagnostic functions before advancing slabs to avoid stale listener / solution lifetimes
        Jx = Teuchos::null;
        Jz = Teuchos::null;
        vx_rel_mean = Teuchos::null;
        vz_rel_mean = Teuchos::null;
        vx_rel_mean_clipped = Teuchos::null;
        vz_rel_mean_clipped = Teuchos::null;
        vx_rel_positive_moment = Teuchos::null;
        vz_rel_positive_moment = Teuchos::null;
        density_positive = Teuchos::null;
        positiveMoments = Teuchos::null;
        f_positive_soln = Teuchos::null;
        f_positive = Teuchos::null;
        density = Teuchos::null;
        momentum = Teuchos::null;
        momentum_z = Teuchos::null;
        velocity = Teuchos::null;
        velocity_z = Teuchos::null;
        potential = Teuchos::null;
        f_soln = Teuchos::null;

        // next time slab
        form = form->nextTimeSlab(tWidth, tElements);
        
        {
          // solutions have been recreated for the new slab.  We need to update anything that depends on the solution
          const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
          solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
          
          potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
          
          density    = solutionMoments->solutionMomentFunction(0,0);
          momentum   = solutionMoments->solutionMomentFunction(1,0); // x proper-velocity moment
          momentum_z = solutionMoments->solutionMomentFunction(0,1); // second retained component; z in 1D2V
          
          velocity   = momentum   / density; // x raw moment ratio
          velocity_z = momentum_z / density;
          
          relativisticForm = Teuchos::rcp_dynamic_cast<RelativisticVlasovSpaceTimeFormulation>(form);
          TEUCHOS_TEST_FOR_EXCEPTION(relativisticForm == Teuchos::null, std::invalid_argument, "Expected RelativisticVlasovSpaceTimeFormulation in PlanarMITLRelativisticSpaceTime");
          Jx = relativisticForm->currentX();
          Jz = relativisticForm->currentZ();
          vx_rel_mean = Jx / (q * density);
          vz_rel_mean = Jz / (q * density);
          
          f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
          f_positive = Function::max(f_soln, Function::constant(0.0));
          f_positive_soln = Solution::solution(form->vlasovMesh());
          {
            std::map<int,FunctionPtr> projectionMap;
            projectionMap[form->f(speciesOrdinal)->ID()] = f_positive;
            f_positive_soln->projectOntoMesh(projectionMap);
          }
          positiveMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, f_positive_soln, form->f(speciesOrdinal), form->configurationSpaceMesh()));
          positiveMoments->update();
          density_positive = positiveMoments->solutionMomentFunction(0,0);
          vx_rel_positive_moment = relativisticForm->relativisticFirstMoment(0, speciesOrdinal, positiveMoments.get(), f_positive_soln);
          vz_rel_positive_moment = relativisticForm->relativisticFirstMoment(1, speciesOrdinal, positiveMoments.get(), f_positive_soln);
          vx_rel_mean_clipped = vx_rel_positive_moment / density_positive;
          vz_rel_mean_clipped = vz_rel_positive_moment / density_positive;
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
          
          form->solve();
          
          if (rank == 0)
          {
            double memoryHWM_GB = memoryHighWaterMarkInGigabytes();
            std::cout << "Time slab " << timeSlabOrdinal << ", refinement " << refinementNumber << ", memory HWM (rank 0): " << memoryHWM_GB << " GB\n";
          }
          
          solutionMoments->update();
          positiveMoments->update();
          
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
            form->solve();
            
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
            maxwellVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,velocity_z,Jx,Jz,vx_rel_mean,vz_rel_mean,vx_rel_mean_clipped,vz_rel_mean_clipped,cellIDFunction}, {"n_e","ux_mean","uz_mean_raw","Jx","Jz","vx_rel_mean_signed","vz_rel_mean_signed","vx_rel_mean_clipped","vz_rel_mean_clipped","cell ID (Maxwell)"}, refinementNumber);
            if (exportVlasov)
            {
              vlasovVizSlab->exportTimeStep(refinementNumber);
            }
          }
          
          if (!errorTargetMet && (refinementNumber < maxRefinements))
          {
            refStrategy->refine(rank==0);
            
            // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
            form->configurationSpaceMesh()->repartitionAndRebuild();
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
          vx_rel_positive_moment = Teuchos::null;
          vz_rel_positive_moment = Teuchos::null;
          density_positive = Teuchos::null;
          positiveMoments = Teuchos::null;
          f_positive_soln = Teuchos::null;
          f_positive = Teuchos::null;
          density = Teuchos::null;
          momentum = Teuchos::null;
          momentum_z = Teuchos::null;
          velocity = Teuchos::null;
          velocity_z = Teuchos::null;
          potential = Teuchos::null;
          f_soln = Teuchos::null;

          // next time slab
          form = form->nextTimeSlab(tWidth, tElements);
          
          {
            // solutions have been recreated for the new slab.  We need to update anything that depends on the solution
            const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
            solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
            
            potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
            
            density    = solutionMoments->solutionMomentFunction(0,0);
            momentum   = solutionMoments->solutionMomentFunction(1,0); // x proper-velocity moment
            momentum_z = solutionMoments->solutionMomentFunction(0,1); // second retained component; z in 1D2V
            
            velocity   = momentum   / density; // x raw moment ratio
            velocity_z = momentum_z / density;
            
            relativisticForm = Teuchos::rcp_dynamic_cast<RelativisticVlasovSpaceTimeFormulation>(form);
            TEUCHOS_TEST_FOR_EXCEPTION(relativisticForm == Teuchos::null, std::invalid_argument, "Expected RelativisticVlasovSpaceTimeFormulation in PlanarMITLRelativisticSpaceTime");
            Jx = relativisticForm->currentX();
            Jz = relativisticForm->currentZ();
            vx_rel_mean = Jx / (q * density);
            vz_rel_mean = Jz / (q * density);
            
            f_soln = Function::solution(form->f(speciesOrdinal), form->vlasovSolution());
            f_positive = Function::max(f_soln, Function::constant(0.0));
            f_positive_soln = Solution::solution(form->vlasovMesh());
            {
              std::map<int,FunctionPtr> projectionMap;
              projectionMap[form->f(speciesOrdinal)->ID()] = f_positive;
              f_positive_soln->projectOntoMesh(projectionMap);
            }
            positiveMoments = Teuchos::rcp(new SolutionMoments(maxMomentNumber, f_positive_soln, form->f(speciesOrdinal), form->configurationSpaceMesh()));
            positiveMoments->update();
            density_positive = positiveMoments->solutionMomentFunction(0,0);
            vx_rel_positive_moment = relativisticForm->relativisticFirstMoment(0, speciesOrdinal, positiveMoments.get(), f_positive_soln);
            vz_rel_positive_moment = relativisticForm->relativisticFirstMoment(1, speciesOrdinal, positiveMoments.get(), f_positive_soln);
            vx_rel_mean_clipped = vx_rel_positive_moment / density_positive;
            vz_rel_mean_clipped = vz_rel_positive_moment / density_positive;
            
            bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
          }
          
          // refine the new slab to ensure resolution of the BC.
          refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form->configurationSpaceMesh()->repartitionAndRebuild();
          if (rank == 0)
          {
            std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
          }
        }
      } // timeSlabOrdinal loop
      
    } // end time slab + useAdaptivity
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
