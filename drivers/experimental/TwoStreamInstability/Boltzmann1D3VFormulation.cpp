//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Boltzmann1D3VFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/29/2020.
//
//

#include "Boltzmann1D3VFormulation.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

using namespace Camellia;

const string Boltzmann1D3VFormulation::S_F = "f";
const string Boltzmann1D3VFormulation::S_E = "E";

const string Boltzmann1D3VFormulation::S_F_HAT = "\\widehat{f}";
const string Boltzmann1D3VFormulation::S_T_N_HAT = "\\widehat{t}_n";
const string Boltzmann1D3VFormulation::S_E_HAT = "\\widehat{E}";

const string Boltzmann1D3VFormulation::S_W = "w";
const string Boltzmann1D3VFormulation::S_TAU = "\\tau";

/*
 Worth noting that VelocitySpaceExtrusion1D3V codifies an inefficient use of memory:
 it is simply duplicating values in the velocity dimensions.  It would be better to
 avoid the duplication by adding some facilities to Function; I like the idea of
 adding a method
    bool variesInDimension(int dim);
 and then having some mechanism by which the arrays that Functions that are constant
 in some dimension can only specify values for the dimensions in which they vary.
 Doing this more or less requires reworking the way points are stored, though, to
 express tensor-product points.  We have some loose plans to do something like this
 in Intrepid2.
 */
class VelocitySpaceExtrusion1D3V : public TFunction<double>
{
  FunctionPtr _g_1D; // arbitrary function in physical space -- can depend on mesh/physical solution
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
public:
  VelocitySpaceExtrusion1D3V( FunctionPtr g_1D, Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap)
  :
  Function(g_1D->rank()),
  _g_1D(g_1D),
  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported right now");
  }
  
  void updateValues()
  {
    auto & phaseSpaceCellIDs = _phaseSpacePhysicalMeshMap->phaseMesh()->cellIDsInPartition();
    
    const set<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    
    const std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDs.begin(), physicalCellIDs.end());
    _g_1D->importCellData(physicalCellIDsVector);
  }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
    bool isSideCache = (basisCache->getSideIndex() != -1);
    int physicalSpaceSideOrdinal;
    if (!isSideCache)
    {
      physicalSpaceSideOrdinal = -1;
    }
    else // isSideCache == true
    {
      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
      // what follows assumes 1D3V with 4D hypercube elements; let's throw an exception if we're not on a 4D hypercube
      auto hypercube_4D = CellTopology::cellTopology(CellTopology::line(), 3);
      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != hypercube_4D->getKey(), std::invalid_argument, "Only 4D hypercubes supported right now");
      
      /*
       subcell ordering in tensor cell topology is as follows.
       
       We take the one-lower-dimensional component topology T and make two copies, T0 and T1.
       
       The subcells of dimension d in the tensor topology are ordered as follows:
       - all subcells of dimension d from T0
       - all subcells of dimension d from T1
       - all (subcells of dimension (d-1) from T) x Line.
       
       Since our physical 1D cell is the first tensorial component, the phase space sides that correspond to physical sides are exactly
       the last two phase space sides.  (For higher physical dimensions where the physical cell has N sides, the phase
       space sides of interest will be the last N sides.)
       */
      
      const int numPhysicalSides = 2;
      
      if (phaseSpaceSideOrdinal < hypercube_4D->getSideCount() - numPhysicalSides)
      {
        // this side does not map to a physical side; runs parallel to physical space
        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
        if (_g_1D->boundaryValueOnly())
        {
          // if _g_1D is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
          values.initialize(0.0);
          return;
        }
        else
        {
          // we can and should evaluate _g_1D on the spatial element interior
          physicalSpaceSideOrdinal = -1;
        }
      }
      else
      {
        physicalSpaceSideOrdinal = phaseSpaceSideOrdinal - (hypercube_4D->getSideCount() - numPhysicalSides);
      }
    }
    
    using std::map;
    using std::pair;
    using std::set;
    using std::vector;
    
    // determine physical cell IDs we need to compute on
    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
    
    const bool testVsTest = basisCache->testVsTest();
    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();

    int basisCache1DQuadratureEnrichment;
    if (testVsTest)
    {
      // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
    }
    else
    {
      // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
    }
    
    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
    for ( auto & physicalCellID : physicalCellIDs )
    {
      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
    }
    
    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
    for ( auto & physicalCellID : physicalCellIDs )
    {
      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
      auto physicalBasisCache = basisCachePair.first;
      auto offsets            = basisCachePair.second;
      
      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
      
      const int numCells  = 1;
      const int numPoints = basisCacheToUse->getRefCellPointsFC().dimension(0);
      Camellia::FieldContainer<double> values(numCells,numPoints);
      
      _g_1D->values(values, basisCacheToUse);
      physicalValuesMap[physicalCellID] = values;
      physicalPointsMap[physicalCellID] = basisCacheToUse->getPhysicalCubaturePoints();
    }

    if (phaseSpaceCellIDs.size() == 0)
    {
      return;
    }
 
    const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
    
    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
    {
      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
      
      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
      
      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
      
      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
      {
        auto physicalCellID             = physicalPointIterator.physicalCellID();
        auto & values1D                 = physicalValuesMap[physicalCellID];
        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
        double & value                  = values1D(0, pointOrdinalInPhysicalCell);
        
        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
        {
          const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
          values(phaseSpaceCellOrdinal,tensorPointOrdinal) = value;
        }
      }
    }
  }
  
  bool boundaryValueOnly() override
  {
    return _g_1D->boundaryValueOnly();
  }
};

Boltzmann1D3VFormulation::Boltzmann1D3VFormulation(Boltzmann1D3VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                                                   const double x0, const double xWidth, const int xElementsCoarse,
                                                   const Kokkos::Array<double,3> v0, const Kokkos::Array<double,3> vDomainWidths, const Kokkos::Array<int,3> vMeshWidths,
                                                   const int H1Order, const int delta_k, const bool usePeriodicBCs, ERKTableauPtr erkTableau)
:
_formulationChoice(formulationChoice)
{
  const int spaceDim = 1;
  const int velocityDim = 3;
  _delta_k = delta_k;
  
  SolutionStagePtr solnStage = Teuchos::null;
  _haveTableau = (erkTableau != Teuchos::null);
  if (_haveTableau)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(this->usesImplicitSolve(), std::invalid_argument, "ERK Tableau is only supported for explicit formulations...");
    solnStage = SolutionStage::solutionStage(erkTableau, dt);
    _dt = solnStage->dt();
  }
  else
  {
    _dt   = ParameterFunction::parameterFunction(dt);
  }
  
  bool useConformingTraces = true; // governs E_hat space (though in 1D this is just a point value regardless)
  
  _q    = ParameterFunction::parameterFunction(q);
  _eps0 = ParameterFunction::parameterFunction(eps0);
  _m    = ParameterFunction::parameterFunction(m);
  
  _n0   = ParameterFunction::parameterFunction(0.0); // will be set/updated during solve
  
  auto intervalMeshTopology      = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElementsCoarse, usePeriodicBCs);
  
  std::vector<double> extrusionStartCoords {v0[0], v0[1], v0[2]};
  std::vector<double> extrusionExtents {vDomainWidths[0], vDomainWidths[1], vDomainWidths[2]};
  std::vector<int> extrusionMeshWidths {vMeshWidths[0], vMeshWidths[1], vMeshWidths[2]};
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopology, extrusionStartCoords, extrusionExtents, extrusionMeshWidths);
  
  if (usePeriodicBCs)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopology->getPeriodicBCs().size() == 0, std::logic_error, "periodic BCs not found in extruded mesh");
  }
  
  const bool implicit  = usesImplicitSolve();
  const bool l2Trace   = hasL2Trace();
  const bool ultraweak = _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_H1_TRACE
                      || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE;
  
  if (ultraweak)
  {
    Space tauSpace    = (spaceDim > 1)      ? HDIV : HGRAD;
    Space E_hat_space = useConformingTraces ? HGRAD : L2;
    Space ESpace      = (spaceDim > 1)      ? VECTOR_L2 : L2;

    // fields
    VarPtr f; // Boltzmann
    VarPtr Ex; // Gauss

    // traces
    VarPtr Ex_hat, t_n_hat, f_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

    // tests
    VarPtr w;   // Boltzmann test function
    VarPtr tau; // Gauss

    VarFactoryPtr gaussVF = VarFactory::varFactory();
    
    Ex = gaussVF->fieldVar(S_E, ESpace);
    if (spaceDim == 1)
    {
      auto fluxWeight = Function::normal_1D() * Function::sideParity(); // for "termTraced" argument below
      Ex_hat = gaussVF->fluxVar(S_E_HAT, Ex * fluxWeight, E_hat_space);
    }
    else
    {
      Ex_hat = gaussVF->traceVar(S_E_HAT, Ex, E_hat_space);
    }
    tau   = gaussVF->testVar(S_TAU, tauSpace);
    
    _gaussBF = BF::bf(gaussVF);
    if (spaceDim == 1)
      _gaussBF->addTerm(-Ex, tau->dx());
    else
      _gaussBF->addTerm(-Ex, tau->grad());
    
    _gaussBF->addTerm(Ex_hat, tau);
    Epetra_CommPtr comm = Teuchos::null;
    const bool allowPruning = false;
    MeshPtr gaussMesh = MeshFactory::minRuleMesh(intervalMeshTopology, _gaussBF, H1Order, delta_k, comm, allowPruning);
    gaussMesh->globalDofAssignment()->setAllowMeshTopologyPruning(false);
    _gaussSoln = Solution::solution(_gaussBF, gaussMesh);
    auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
    
    FunctionPtr dt = FunctionPtr(_dt);
    FunctionPtr Ex_soln = Function::solution(Ex, gaussSolnWeakPtr);
    
    FunctionPtr v_x = Function::yn(1); // we are using the y dimension for velocity space
    FunctionPtr m = FunctionPtr(_m);
    
    VarFactoryPtr boltzmannVF = VarFactory::varFactory();
    
    f = boltzmannVF->fieldVar(S_F);
    w = boltzmannVF->testVar(S_W, HGRAD); // v lies in HGRAD of the whole 1D3V space
    
    if (l2Trace)
    {
      t_n_hat = boltzmannVF->fluxVar(S_T_N_HAT);
    }
    else
    {
      f_hat = boltzmannVF->traceVar(S_F_HAT);
    }
    
    _boltzmannBF = BF::bf(boltzmannVF);
    
    map<int,int> trialOrderEnhancements;
    map<int,int> testOrderEnhancements;
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1;
    }
    
    MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
    MeshPtr boltzmannMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _boltzmannBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
    
    _phaseSpacePhysicalMeshMap = Teuchos::rcp( new PhasePhysicalMeshMap( gaussMesh, boltzmannMesh ) );
    auto meshMapWeakPtr        = Teuchos::rcp( _phaseSpacePhysicalMeshMap.get(), false);
    
    _Ex_extruded = Teuchos::rcp( new VelocitySpaceExtrusion1D3V(Ex_soln, meshMapWeakPtr) );

    _boltzmannBF->addTerm(f / _dt, w);
    
    if (l2Trace)
    {
      _boltzmannBF->addTerm(t_n_hat, w); // for explicit, t_n should be understood as the trace of the solution at prior time…
    }
    else
    {
      FunctionPtr n = Function::normal();
      _boltzmannBF->addTerm((v_x * n->x() + (q/m) * _Ex_extruded * n->y()) * f_hat, w); // for explicit, f_hat should be understood as the trace of the solution at prior time…
    }

    if (implicit)
    {
      _boltzmannBF->addTerm(-v_x * f, w->di(1)); // di() is 1-based: this means the first dimension of the mesh (x).
      _boltzmannBF->addTerm(-(q/m)*_Ex_extruded*f, w->di(2)); // this is the part of the E dot grad_v w term that survives the electrostatic 1D3V Vlasov-Poisson assumptions; will need to add more terms for Maxwell
    }
    
    _boltzmannSoln     = Solution::solution(_boltzmannBF,boltzmannMesh);
    if (_haveTableau)
    {
      // then previous solution is in the same Solution object as the "current" solution
      _boltzmannSoln->setSolutionStage(solnStage);
      _boltzmannSolnPrev = _boltzmannSoln;
      // register for notifications
      auto thisPtr = Teuchos::rcp(this,false);
      solnStage->addObserver(thisPtr);
    }
    else
    {
      _boltzmannSolnPrev = Solution::solution(_boltzmannBF,boltzmannMesh);
    }
    
    _f_velocityIntegral = Teuchos::rcp( new SolutionMomentFunction(_boltzmannSoln, f, 0, _gaussSoln->mesh(), -1) );
    
    // set RHS for the two problems
    auto gaussRHS = RHS::rhs();
    // casts to allow FunctionPtr overloads to be used:
    auto qFunction = FunctionPtr(_q);
    auto mFunction = FunctionPtr(_m);
    auto epsFxn = FunctionPtr(_eps0);
    auto n0Fxn  = FunctionPtr(_n0); // _n0 will be initialized when initial values are specified
    auto f_velocityIntegralFxn = FunctionPtr(_f_velocityIntegral);
    gaussRHS->addTerm(qFunction / epsFxn * (f_velocityIntegralFxn - n0Fxn) * tau);
    _gaussSoln->setRHS(gaussRHS);
    
    auto boltzmannRHS = RHS::rhs();
    auto boltzmannPrevSolnWeakPtr = Teuchos::rcp(_boltzmannSolnPrev.get(), false);
    
    const bool weightFluxesBySideParity = true;
    const SolutionComponent solnComponent = STANDARD_SOLUTION;
    const int stageOrdinal = _haveTableau ? -1 : 0;
    auto f_previous = Function::solution(f, boltzmannPrevSolnWeakPtr, weightFluxesBySideParity,
                                         solnComponent, stageOrdinal, std::string("k"));
    
    boltzmannRHS->addTerm(f_previous / dt * w);
    
    if (!implicit)
    {
      boltzmannRHS->addTerm(v_x * f_previous * w->di(1)); // dw/dx
      boltzmannRHS->addTerm((q/m)*_Ex_extruded*f_previous * w->di(2)); // dw/(dvx)
    }
    
    _boltzmannSoln->setRHS(boltzmannRHS);
    
    // set the test-space inner products.
    _gaussSoln->setIP(_gaussBF->naiveNorm(spaceDim));
    
    gaussMesh->registerSolution(_gaussSoln);
    boltzmannMesh->registerSolution(_boltzmannSoln);
    
    IPPtr boltzmannTestNorm = IP::ip();
    const bool useSimpleL2Norm = false;
    if (useSimpleL2Norm)
    {
      boltzmannTestNorm->addTerm(w);
      std::cout << "NOTE: using simple L^2 norm for explicit Boltzmann...\n";
    }
    else
    {
      const bool useExperimentalNormForExplicit = false;
      if (!useExperimentalNormForExplicit || implicit)
      {
        // experiments so far suggest that the norm below is the most robust for explicit time-stepping
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v_x * w->di(1) + (q/m)*_Ex_extruded * w->di(2)); // di() is 1-based: dx, dvx
      }
      else
      {
        std::cout << "NOTE: using experimental test norm for explicit Boltzmann...\n";
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v_x * w->di(1) + 1e-3 * (q/m)*_Ex_extruded * w->di(2)); // di() is 1-based: dx, dvx
      }
    }
    _boltzmannSoln->setIP(boltzmannTestNorm);
    
    // set BCs
    // we are periodic in x; set homogeneous conditions on the v boundaries
    auto boltzmannBC = BC::bc();
    auto traceVar = l2Trace ? t_n_hat : f_hat;
    
    for (int d=0; d<velocityDim; d++)
    {
      double v0_d = v0[d];
      double v1_d = v0_d + vDomainWidths[d];
      boltzmannBC->addDirichlet(traceVar, SpatialFilter::matchingComponent(spaceDim + d, v0_d), Function::zero());
      boltzmannBC->addDirichlet(traceVar, SpatialFilter::matchingComponent(spaceDim + d, v1_d), Function::zero());
    }
    _boltzmannSoln->setBC(boltzmannBC);
    
    auto gaussBC = BC::bc();
    gaussBC->addDirichlet(Ex_hat, SpatialFilter::matchingX(x0), Function::zero());
    _gaussSoln->setBC(gaussBC);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported Boltzmann1D3VFormulationChoice");
  }
  PhysicalMeshUpdater::registerForUpdates(_boltzmannSoln->mesh(), _gaussSoln->mesh());
}

MeshPtr Boltzmann1D3VFormulation::boltzmannMesh()
{
  return _boltzmannSoln->mesh();
}

SolutionPtr Boltzmann1D3VFormulation::boltzmannPreviousSolution()
{
  return _boltzmannSolnPrev;
}

SolutionPtr Boltzmann1D3VFormulation::boltzmannSolution()
{
  return _boltzmannSoln;
}

FunctionPtr Boltzmann1D3VFormulation::E_extruded()
{
  return _Ex_extruded;
}

VarPtr Boltzmann1D3VFormulation::f()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr Boltzmann1D3VFormulation::f_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

MeshPtr Boltzmann1D3VFormulation::gaussMesh()
{
  return _gaussSoln->mesh();
}

SolutionPtr Boltzmann1D3VFormulation::gaussSolution()
{
  return _gaussSoln;
}

bool Boltzmann1D3VFormulation::hasL2Trace() const {
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE;
}

VarPtr Boltzmann1D3VFormulation::E()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->trialVar(S_E);
}

VarPtr Boltzmann1D3VFormulation::E_hat()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->trialVar(S_E_HAT);
}

Teuchos::RCP<PhasePhysicalMeshMap> Boltzmann1D3VFormulation::phaseSpacePhysicalMeshMap()
{
  return _phaseSpacePhysicalMeshMap;
}

void Boltzmann1D3VFormulation::setBoltzmannBCs(BCPtr bc)
{
  _boltzmannSoln->setBC(bc);
}

void Boltzmann1D3VFormulation::setGaussBCs(BCPtr bc)
{
  _gaussSoln->setBC(bc);
}

void Boltzmann1D3VFormulation::setInitialConditions(FunctionPtr f0)
{
  _boltzmannMeshMeasure = Function::constant(1.0)->integrate(_boltzmannSoln->mesh());
  
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[this->f()->ID()] = f0;
  if ( !hasL2Trace() )
  {
    projectionMap[this->f_hat()->ID()] = f0;
  }
  
  const int solutionOrdinal = 0;
  _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  if (!_haveTableau)
  {
    _boltzmannSolnPrev->setSolution(_boltzmannSoln);
  }
  
  _gaussMeshMeasure = Function::constant(1.0)->integrate(_gaussSoln->mesh());

  updateElectricField();
  
  // set initial tn value:
  auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
  
  if (hasL2Trace())
  {
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr q(_q);
    FunctionPtr m(_m);
    FunctionPtr v = Function::yn(1);
    FunctionPtr n = Function::normal();
    FunctionPtr tn_0 = f0 * (v * n->x() + (q/m) * _Ex_extruded * n->y()) * sgn;
    projectionMap[this->t_n_hat()->ID()] = tn_0;
    _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    if (!_haveTableau)
    {
      _boltzmannSolnPrev->setSolution(_boltzmannSoln);
    }
  }
  _t = 0;
}

void Boltzmann1D3VFormulation::setStage(const int stageNumber)
{
  updateElectricField();
}

void Boltzmann1D3VFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double Boltzmann1D3VFormulation::t() const
{
  return _t;
}

void Boltzmann1D3VFormulation::takeTimeStep()
{
  // for implicit solves, we cycle through the f solve and the E solve a few times;
  // for explicit, we do each solve just once per time step.
  const bool implicit = _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
  int numSubsteps = implicit ? 3 : 1;
  
  bool useGMG = true;
  if (useGMG && (_gmgSolver == Teuchos::null))
  {
    int kCoarse = 0;
    const double cgTol = 1e-6;
    const int maxIters = 200;

    bool useCondensedSolve = _boltzmannSoln->usesCondensedSolve();

    //    auto multigridStrategy = GMGOperator::V_CYCLE;
    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_boltzmannSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_boltzmannSoln->mesh()};
    }
    
    _gmgSolver = Teuchos::rcp( new GMGSolver(_boltzmannSoln, meshes, maxIters, cgTol, multigridStrategy,
                                             Solver::getDirectSolver(true), useCondensedSolve) );
    
    _gmgSolver->setSmootherType(smootherChoice);
    
    const int azOutputLevel = 0;
    _gmgSolver->setAztecOutput(azOutputLevel);
  }
  
  for (int substepOrdinal=0; substepOrdinal < numSubsteps; substepOrdinal++)
  {
    // solve for f:
    if (_gmgSolver == Teuchos::null)
      _boltzmannSoln->solve();
    else
      _boltzmannSoln->solve(_gmgSolver);
    
    if ( !_haveTableau )
    {
      if (substepOrdinal == numSubsteps-1)
      {
        // last substep
        _boltzmannSolnPrev->setSolution(_boltzmannSoln);
      }
      // if we have a tableau, then electric field will be updated when the solution stage is set.
      updateElectricField();
    }
  }
  
  _t += _dt->getConstantValue();
}

void Boltzmann1D3VFormulation::updateElectricField()
{
  // update f integral, n0
  _f_velocityIntegral->updateValues();
  double f_integral   = _f_velocityIntegral->integrate(_gaussSoln->mesh());
  double n0 = f_integral / _gaussMeshMeasure;
//  std::cout << "1D3V n0 = " << n0 << std::endl;
  _n0->setValue(n0);
  
  // solve for E, E_hat:
  _gaussSoln->solve();
  
  // post-process to make E have a zero mean:
  FunctionPtr E_soln = Function::solution(E(), _gaussSoln);
  double E_avg = E_soln->integrate(_gaussSoln->mesh()) / _gaussMeshMeasure;
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[E()->ID()] = E_soln - E_avg;
  projectionMap[E_hat()->ID()] = E_soln - E_avg;
  _gaussSoln->projectOntoMesh(projectionMap);
  
  static_cast<VelocitySpaceExtrusion1D3V*>(_Ex_extruded.get())->updateValues();
}

bool Boltzmann1D3VFormulation::usesImplicitSolve() const
{
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
}

// test variables:
VarPtr Boltzmann1D3VFormulation::w()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

VarPtr Boltzmann1D3VFormulation::tau()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->testVar(S_TAU);
}

VarPtr Boltzmann1D3VFormulation::t_n_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

void testVelocitySpace3DIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-14;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double vWidth = 1.0;
  
  const int xElements   = 4;
  const int vElements_x = 4;
  const int vElements_y = 1;
  const int vElements_z = 1;
  
  const Kokkos::Array<double,3> v0 {0, 0, 0};
  const Kokkos::Array<double,3> vDomainWidths {vWidth, vWidth, vWidth};
  const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
  
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  
  using FormulationChoice = Camellia::Boltzmann1D3VFormulation::Boltzmann1D3VFormulationChoice;
  
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };

  // worth running all the formulationChoices generally, but 99% of the time it's as good
  // to just test the main choice of interest to us.  Can comment out the above in favor of the
  // below when testing in a debugger, both to focus attention, and to speed up the test.
//  std::vector<FormulationChoice> formulationChoices {
//    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
//  };
  
  set<GlobalIndexType> boltzmannCellsToRefine;
  if (doRefinement)
  {
    boltzmannCellsToRefine = {0,5,8};
  }
  
  // TestCase: (function to be integrated, (integral over velocity space, integral over whole domain))
  using TestCase = std::pair<FunctionPtr, std::pair<FunctionPtr, double> >;
  std::vector<TestCase> testCases;
  using std::make_pair;
  {
    // test case 1: constant everywhere
    const double g_value = 0.5;
    // exact distribution function:
    FunctionPtr g = Function::constant(g_value);
    // the distribution function, integrated over all velocity dimensions (a function of x, defined on physical mesh)
    FunctionPtr velocitySpaceIntegral = g * vDomainWidths[0] * vDomainWidths[1] * vDomainWidths[2];
    // the integral of the distribution function over all of phase space:
    double integral = g_value * xWidth * vDomainWidths[0] * vDomainWidths[1] * vDomainWidths[2];
    
    testCases.push_back(make_pair(g, make_pair(velocitySpaceIntegral, integral)));
  }
  {
    // test case 2: g = x
    FunctionPtr g = Function::xn(1);
    FunctionPtr velocitySpaceIntegral = g * vDomainWidths[0] * vDomainWidths[1] * vDomainWidths[2];
    double x_integral = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0);
    double integral = x_integral * vDomainWidths[0] * vDomainWidths[1] * vDomainWidths[2];

    testCases.push_back(make_pair(g, make_pair(velocitySpaceIntegral, integral)));
  }
  {
    // test case 3: g = x * v_x
    FunctionPtr g = Function::xn(1) * Function::yn(1);
    double vx_integral = (0.5 * (v0[0] + vDomainWidths[0]) * (v0[0] + vDomainWidths[0]) - 0.5 * v0[0] * v0[0]);
    FunctionPtr velocitySpaceIntegral = Function::xn(1) * vx_integral * vDomainWidths[1] * vDomainWidths[2];
    double x_integral = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0);
    double integral = x_integral * vx_integral * vDomainWidths[1] * vDomainWidths[2];

    testCases.push_back(make_pair(g, make_pair(velocitySpaceIntegral, integral)));
  }
  
  for (auto formulationChoice : formulationChoices)
  {
    Boltzmann1D3VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vDomainWidths, vMeshWidths, H1Order, delta_k, usePeriodicBCs, Teuchos::null);
    
    if (boltzmannCellsToRefine.size() > 0)
    {
      auto boltzmannMesh = form.boltzmannMesh();
      
      auto sampleCell     = boltzmannMesh->getTopology()->getCell(0);
      auto sampleCellTopo = sampleCell->topology();
      auto refPattern = RefinementPattern::regularRefinementPattern(sampleCellTopo);
      
      boltzmannMesh->hRefine(boltzmannCellsToRefine, refPattern);
    }
    
    int testCaseNumber = 1;
    for (auto testCase : testCases)
    {
      out << "\n\n******* Test Case " << testCaseNumber++ << " *******\n";
      
      FunctionPtr g = testCase.first;
      FunctionPtr expectedVelocitySpaceIntegral = testCase.second.first;
      const double expectedIntegral = testCase.second.second;
      
      double actualExactIntegral = expectedVelocitySpaceIntegral->integrate(form.gaussMesh()); // integrate on the 1D mesh
      TEST_FLOATING_EQUALITY(expectedIntegral, actualExactIntegral, tol);
      
      auto f = form.f();
      std::map<int, FunctionPtr> projectionMap;
      projectionMap[f->ID()] = g;
      
      auto boltzmannSoln = form.boltzmannSolution();
      const int solutionOrdinal = 0;
      boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
      
      auto f_soln = Function::solution(f, boltzmannSoln);
      const double projectionErrL2 = (g - f_soln)->l2norm(boltzmannSoln->mesh());
      double f_magnitudeExpected = g->l2norm(boltzmannSoln->mesh());
      TEST_COMPARE(projectionErrL2, <, tol * f_magnitudeExpected);
      
      auto gaussMesh = form.gaussMesh();
      
      auto velocitySpaceIntegral = Teuchos::rcp( new SolutionMomentFunction(boltzmannSoln, f, 0, gaussMesh, -1) );
      FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedVelocitySpaceIntegral;
      
      velocitySpaceIntegral->updateValues();
      
//      {
//        // DEBUGGING
//        auto myCells = form.gaussMesh()->cellIDsInPartition();
//        for (auto cellID: myCells)
//        {
//          out << "cell " << cellID << " integral: " << velocitySpaceIntegral->integrate(BasisCache::basisCacheForCell(form.gaussMesh(), cellID)) << std::endl;
//        }
//      }
      
      double actualInterpolatedIntegral = velocitySpaceIntegral->integrate(form.gaussMesh());
//      out << "actualInterpolatedIntegral = " << actualInterpolatedIntegral << std::endl;
      
      TEST_FLOATING_EQUALITY(expectedIntegral, actualInterpolatedIntegral, tol);
      
      double magnitudeExpected = expectedVelocitySpaceIntegral->l2norm(gaussMesh);
      double err_l2 = err->l2norm(gaussMesh);
      
      TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
    }
  }
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionBoltzmann1D3VFormulation, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace3DIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionBoltzmann1D3VFormulation, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace3DIntegral(doRefinement, out, success);
}

class SideRestriction : public TFunction<double>
{
  std::vector<int> _sidesToMatch;
public:
  SideRestriction(std::vector<int> sidesToMatch)
  :
  Function(0),
  _sidesToMatch(sidesToMatch)
  {}
  
  bool boundaryValueOnly() override
  {
    return true;
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int sideOrdinal = basisCache->getSideIndex();
    bool sideMatches = false;
    for (auto sideToMatch : _sidesToMatch)
    {
      if (sideToMatch == sideOrdinal)
      {
        sideMatches = true;
        break;
      }
    }
    const double value = sideMatches ? 1.0 : 0.0;
    values.initialize(value);
  }
};

void testVelocityExtrusion1D3V(bool volumeVariable, bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  
  double vWidth = 20.0;
  
  const int xElements   = 2;
  const int vElements_x = 2;
  const int vElements_y = 1;
  const int vElements_z = 1;
  
  const Kokkos::Array<double,3> v0 {0, 0, 0};
  const Kokkos::Array<double,3> vDomainWidths {vWidth, vWidth, vWidth};
  const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
  
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 1;
  const bool usePeriodicBCs = true;
  
  using FormulationChoice = Camellia::Boltzmann1D3VFormulation::Boltzmann1D3VFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };
  
  for (auto formulationChoice : formulationChoices)
  {
    Boltzmann1D3VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vDomainWidths, vMeshWidths, H1Order, delta_k, usePeriodicBCs, Teuchos::null);
    
    FunctionPtr x  = Function::xn(1);
    
    FunctionPtr soln_1D;
    
    auto gaussSoln = form.gaussSolution();
    std::map<int, FunctionPtr> projectionMap;
    if (volumeVariable)
    {
      auto E = form.E();
      projectionMap[E->ID()] = x;
      soln_1D = Function::solution(E, gaussSoln);
    }
    else
    {
      auto E     = form.E();
      auto E_hat = form.E_hat();
      projectionMap[E->ID()] = x * (xWidth - x);
      projectionMap[E_hat->ID()] = x * (xWidth - x); // for skeleton values, due to periodic BCs, want something that agrees on left/right side: having zeros there makes this simpler
      soln_1D = Function::solution(E_hat, gaussSoln, false); // false: don't weight by parity (makes uniquely-valued)
    }
    
    TEST_ASSERT(soln_1D->boundaryValueOnly() == !volumeVariable);
    
    auto gaussMesh = form.gaussMesh();
    auto boltzmannMesh = form.boltzmannMesh();
    
    auto meshMap = Teuchos::rcp( new PhasePhysicalMeshMap( gaussMesh, boltzmannMesh ) );
    auto extrusion = Teuchos::rcp( new VelocitySpaceExtrusion1D3V(soln_1D, meshMap) );
    
    auto sampleCell     = boltzmannMesh->getTopology()->getCell(0);
    auto sampleCellTopo = sampleCell->topology();
    
    if (doRefinement)
    {
      const set<GlobalIndexType> cellIDs_4D {0,1};
      auto refPattern = RefinementPattern::regularRefinementPattern(sampleCellTopo);
      boltzmannMesh->hRefine(cellIDs_4D, refPattern);
    }
    
    const int solutionOrdinal = 0;
    gaussSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    extrusion->updateValues();
    
    FunctionPtr expected;
    if (volumeVariable)
    {
      expected = x;
    }
    else
    {
      // restrict x to the spatial sides of the mesh skeleton
      // these are the sides of the hypercube corresponding to left and right sides of the physical cell (sides that do not vary in x)
      // because of the way tensor cell topologies are numbered, the final sides are the ones of interest: here, the final two sides;
      const int sideCount = sampleCellTopo->getSideCount();
      std::vector<int> spatialSides { sideCount - 2, sideCount - 1 };
      FunctionPtr sideRestriction = Teuchos::rcp(new SideRestriction(spatialSides) );
      expected = x * (xWidth - x) * sideRestriction;
    }
    
    FunctionPtr err = FunctionPtr(extrusion) - expected;
    
  //  double magnitudeActual = extrusion->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
    
    double magnitudeExpected = expected->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
    double err_l2 = err->l2norm(boltzmannMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion1D3VBoltzmann1D3VFormulation, SkeletonExtrusion )
{
  const bool volumeVariable = false;
  const bool doRefinement = false;
  testVelocityExtrusion1D3V(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion1D3VBoltzmann1D3VFormulation, SkeletonExtrusionWithRefinement )
{
  const bool volumeVariable = false;
  const bool doRefinement = true;
  testVelocityExtrusion1D3V(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion1D3VBoltzmann1D3VFormulation, VolumeExtrusion )
{
  const bool volumeVariable = true;
  const bool doRefinement = false;
  testVelocityExtrusion1D3V(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion1D3VBoltzmann1D3VFormulation, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  testVelocityExtrusion1D3V(volumeVariable, doRefinement, out, success);
}
