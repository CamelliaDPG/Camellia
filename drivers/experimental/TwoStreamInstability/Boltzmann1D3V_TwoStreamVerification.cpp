//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "Boltzmann1D3VFormulation.h"
#include "Boltzmann1D1VFormulation.h" // for verification, we also solve 1D1V
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "HDF5Exporter.h"
#include "PhaseSpaceVisualization.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include "Intrepid2_Cubature.hpp"

/**
  This driver is for verification of the Boltzmann 1D3V formulation, by comparison to the 1D1V formulation applied to
 the two-stream instability problem.  The driver can be run on coarse or fine meshes, with arbitrary time step (formulations
 are implicit).  It is expected to pass for essentially any valid inputs.  For simplicity, this driver does not do any refinements
 of the meshes.
 */

using namespace Camellia;

//! Maxwellian that only varies in the first velocity dimension; in e.g. 1D3V this is simply extruded in the last two velocity dimensions
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
  _n(n),
  TFunction<double>("Maxwellian1V",0, VARIES_IN_DIMENSION_1) // varies in v_x
  {}
  
  using TFunction<double>::values;
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V,1D3V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
  
  void values(Intrepid2::Data<double, Camellia::DeviceType> data, BasisCachePtr basisCache) override
  {
    // when running in 1D3V, we require that data have shape (C,P0,P1,P2,P3), where P0 is the x dimension, and P1, P2, P3 are for v_x, v_z, v_y
    // it should be the case that the underlying view has shape (C,P1), whether we are in 1D1V or 1D3V.
    
    const auto & physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
    
    int vxTensorComponentOrdinal;
    int vxPointDimension;
    if (physicalCubaturePoints.numTensorComponents() >= 2)
    {
      vxTensorComponentOrdinal = 1;
      vxPointDimension = 0;
    }
    else
    {
      // trivial tensor product structure
      vxTensorComponentOrdinal = 0;
      const int numPhysicalDimensions = basisCache->cellTopology()->getBaseTopology()->getDimension();
      vxPointDimension = numPhysicalDimensions; // vx dimension
    }
    
    auto vxPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(vxTensorComponentOrdinal);
    auto underlyingData = data.getUnderlyingView<2>(); // (C,P1)
    
    using ExecutionSpace = Camellia::DeviceExecutionSpace;
    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{underlyingData.extent(0),underlyingData.extent(1)});
    
    // avoid implicit references to this
    double u = _u;
    double sigma = _sigma;
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
    
    Kokkos::parallel_for("compute Maxwellian1V", policy,
    KOKKOS_LAMBDA (const int &cellOrdinal, const int &vxPointOrdinal) {
      auto & result = underlyingData(cellOrdinal, vxPointOrdinal);
      
      const auto & vx = vxPoints(cellOrdinal,vxPointOrdinal,vxPointDimension);
      
      const double exponent = - 0.5 * (vx - u) * (vx - u) / (sigma * sigma);
      result = weight * std::exp(exponent);
    });
  }
  
  FunctionPtr di(int i) override
  {
    // this function only varies in the second dimension (the first velocity dimension)
    if (i != 2)
    {
      return Function::zero();
    }
    else
    {
      FunctionPtr f = Teuchos::rcp( new Maxwellian1V(_u,_sigma,_n));
      auto vx = Function::yn(1);
      FunctionPtr w = - (vx - _u) / (_sigma * _sigma); // chain rule: derivative of the exponent
      return w * f;
    }
  }
  
  std::string displayString() override
  {
    return "[Maxwellian]";
//    std::ostringstream stringStream;
//    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
//    stringStream << weight << " e^[" << "(vx - " << _u << ")^2 / " << 2 * _sigma * _sigma << "]";
//    
//    return stringStream.str();
  }
};

static const std::string S_TWO_STREAM   = "two-stream";
static const std::string S_BUMP_ON_TAIL = "bump-on-tail";

enum ProblemChoice
{
  TWO_STREAM,
  BUMP_ON_TAIL
};

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_BUMP_ON_TAIL,BUMP_ON_TAIL},{S_TWO_STREAM,TWO_STREAM}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{BUMP_ON_TAIL,S_BUMP_ON_TAIL},{TWO_STREAM,S_TWO_STREAM}};

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

double computeL2Difference(SolutionMomentFunction &f1D3V_xvx, SolutionPtr f1D1V_soln, VarPtr f_var)
{
  // the two should be defined on geometrically identical meshes, but the MPI distribution and numbering of the cells may differ
  // We compute the difference on the 1D1V mesh; we send coefficients from 1D3V to the 1D1V cell owners
  
  auto myCells1D3V = f1D3V_xvx.mesh()->cellIDsInPartition();
  
  auto mesh1D1V = f1D1V_soln->mesh();
  
  auto basisCoefficients = f1D3V_xvx.basisCoefficients();
  auto basisCoefficientsToSend = Kokkos::create_mirror_view(basisCoefficients); // host view
  Kokkos::deep_copy(basisCoefficientsToSend,basisCoefficients);
  
  int basisCardinality = basisCoefficientsToSend.extent_int(1);
  
  using DataType = std::tuple<IndexType,int,double>; // (receivingCellID,spatialBasisOrdinal,coefficient)
  std::map<int,std::vector<DataType>> dataToSend;
  
  for (const auto &sendingCellID : myCells1D3V)
  {
    int cellOrdinal = f1D3V_xvx.localCellOrdinal(sendingCellID);
    
    auto cellCentroid = f1D3V_xvx.mesh()->getTopology()->getCellCentroid(sendingCellID);
    std::vector<std::vector<double> > cellCentroids {cellCentroid};
    IndexType receivingCellID = mesh1D1V->getTopology()->cellIDsWithCentroids(cellCentroids)[0];
    
    const PartitionIndexType receivingCellOwner = mesh1D1V->partitionForCellID(receivingCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(receivingCellOwner == -1, std::logic_error, "partition not assigned for physicalCellID -- inactive?");
    
    std::vector<DataType> & ownerDataVector = dataToSend[receivingCellOwner];
    
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      const auto & value = basisCoefficientsToSend(cellOrdinal,basisOrdinal);
      DataType dataEntry = make_tuple(receivingCellID,basisOrdinal,value);
      
      ownerDataVector.push_back(dataEntry);
    }
  }
  
  std::vector<DataType> dataReceived;
  MPIWrapper::sendDataVectors(mesh1D1V->Comm(), dataToSend, dataReceived);
  
  const auto & myReceivingCells = mesh1D1V->cellIDsInPartition();
  
  int numCells = myReceivingCells.size();
  std::vector<int> receivedBasisCoefficientsDims {numCells, basisCardinality};
  std::map<IndexType,int> cellOrdinalFromCellID;
  
  int cellOrdinal = 0;
  for (const auto & myReceivingCellID : myReceivingCells)
  {
    cellOrdinalFromCellID[myReceivingCellID] = cellOrdinal++;
  }
  
  auto receivedBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(receivedBasisCoefficientsDims, "received spatial basis coefficients");
  auto solnBasisCoefficients     = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(receivedBasisCoefficientsDims, "solution basis coefficients");
  auto receivedBasisCoefficientsHost = Kokkos::create_mirror_view(receivedBasisCoefficients);
  auto solnBasisCoefficientsHost = Kokkos::create_mirror_view(solnBasisCoefficients);
  
  for (const DataType & dataEntry : dataReceived)
  {
    using std::get;
    const IndexType & receivingCellID = get<0>(dataEntry);
    const int & basisOrdinal          = get<1>(dataEntry);
    const double & basisCoefficient   = get<2>(dataEntry);
    
    auto cellOrdinalEntry = cellOrdinalFromCellID.find(receivingCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(cellOrdinalEntry == cellOrdinalFromCellID.end(), std::invalid_argument, "cellID not found/not owned by this rank");
    
    const int cellOrdinal = cellOrdinalEntry->second;
    
    receivedBasisCoefficientsHost(cellOrdinal,basisOrdinal) = basisCoefficient;
  }
  
  // now, populate the 1D1V solution basis coefficients container
  Camellia::FieldContainer<double> cellBasisCoefficients(basisCardinality);
  cellOrdinal = 0;
  for (const auto & myReceivingCellID : myReceivingCells)
  {
    f1D1V_soln->solnCoeffsForCellID(cellBasisCoefficients, myReceivingCellID, f_var->ID());
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      solnBasisCoefficientsHost(cellOrdinal,basisOrdinal) = cellBasisCoefficients(basisOrdinal);
    }
    cellOrdinal++;
  }
  
  // take the L^2 norm of each cell
  
  // do quadrature
  auto basis = mesh1D1V->getElementType(0)->trialOrderPtr->getBasis(f_var->ID());
  const int basisDegree = basis->getDegree();
  const int quadratureDegree = basisDegree * 2;
  shards::CellTopology line_topo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );
  
  // Once we are using Intrepid2 bases, the below will be what we want
//  Intrepid2::DefaultCubatureFactory cub_factory;
//  auto quadrature = cub_factory.create<Camellia::DeviceExecutionSpace, double, double>(line_topo.getKey(), quadratureDegree);
//
//  auto refPoints         = quadrature->allocateCubaturePoints();
//  auto quadratureWeights = quadrature->allocateCubatureWeights();
//  quadrature->getCubature(refPoints, quadratureWeights);
//
//  auto op = Intrepid2::OPERATOR_VALUE;
//  auto basisValues = basis->allocateBasisValues(refPoints, op);
//
//  basis->getValues(basisValues, refPoints, op);
  
  // Until we are using Intrepid2 bases, we work with FieldContainers:
  CubatureFactory cubFactory;
  auto cubature = cubFactory.create(CellTopology::quad(), quadratureDegree); // quad: once-extruded line
  const int numPoints   = cubature->getNumPoints();
  const int cubatureDim = cubature->getDimension();
  std::vector<int> cubatureWeightDims {numPoints};
  std::vector<int> cubaturePointDims {numPoints, cubatureDim};
  
  auto refPoints         = cubature->allocateCubaturePoints();
  auto quadratureWeights = cubature->allocateCubatureWeights();
  cubature->getCubature(refPoints, quadratureWeights);
  
  auto refPointsFC = FieldContainerConversion<double>::fromTensorPoints(refPoints);
  
  cubature->getCubature(refPoints, quadratureWeights);
  
  FieldContainer<double> basisValues(basisCardinality,numPoints);
  basis->getValues(basisValues, refPointsFC, Intrepid::OPERATOR_VALUE);
  
  using ExecutionSpace = Camellia::DeviceExecutionSpace;
  Kokkos::RangePolicy<ExecutionSpace> policy(0, numCells);
    
  std::vector<int> cellDims {numCells};
  auto cellMeasures = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(cellDims, "cellMeasures");
  auto cellMeasuresHost = Kokkos::create_mirror_view(cellMeasures);
  
  cellOrdinal = 0;
  for (const auto & myReceivingCellID : myReceivingCells)
  {
    cellMeasuresHost(cellOrdinal) = mesh1D1V->getCellMeasure(myReceivingCellID);
    cellOrdinal++;
  }
  Kokkos::deep_copy(cellMeasures, cellMeasuresHost);
  Kokkos::deep_copy(receivedBasisCoefficients, receivedBasisCoefficientsHost);
  Kokkos::deep_copy(solnBasisCoefficients, solnBasisCoefficientsHost);
  
  auto squaredCellDiffL2Norms = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(cellDims, "squaredCellDiffL2Norms");
  auto squaredCellSolnL2Norms = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(cellDims, "squaredCellSolnL2Norms");
  
  auto basisValuesView = basisValues.checkoutDeviceView();
  // the following assumes that the VALUE transformation of basis is the identity; this is true for H^1, but not for H(div), etc.
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int cellOrdinal)
  {
    double refErrIntegral  = 0.0; // integral of difference in reference space
    double refSolnIntegral = 0.0; // integral of solution in reference space
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      const double weight = quadratureWeights(pointOrdinal);
      
      double diffFxnValue = 0;
      double solnFxnValue = 0;
      for (int i=0; i<basisCardinality; i++)
      {
        const double & basisCoefficientSoln = solnBasisCoefficients(cellOrdinal,i);
        const double   basisCoefficientDiff = receivedBasisCoefficients(cellOrdinal,i) - basisCoefficientSoln;
        const double & basisPointValue = basisValuesView(i,pointOrdinal);
        diffFxnValue += basisCoefficientDiff * basisPointValue;
        solnFxnValue += basisCoefficientSoln * basisPointValue;
      }
      refErrIntegral  += diffFxnValue * diffFxnValue * weight;
      refSolnIntegral += solnFxnValue * solnFxnValue * weight;
    }
    squaredCellDiffL2Norms(cellOrdinal) = refErrIntegral * cellMeasures(cellOrdinal);
    squaredCellSolnL2Norms(cellOrdinal) = refSolnIntegral * cellMeasures(cellOrdinal);
  });
  
  basisValues.checkinDeviceView(false);
  Kokkos::fence();
  
  double rankLocalSumDiff;
  double rankLocalSumSoln;
  Kokkos::parallel_reduce("Sum of squares of local cell-wise L^2 norms of differences", numCells, KOKKOS_LAMBDA (const int& cellOrdinal, double& threadLocalSum ) {
    threadLocalSum += squaredCellDiffL2Norms(cellOrdinal);
  },rankLocalSumDiff);
  
  Kokkos::parallel_reduce("Sum of squares of local cell-wise L^2 norms of solution", numCells, KOKKOS_LAMBDA (const int& cellOrdinal, double& threadLocalSum ) {
    threadLocalSum += squaredCellSolnL2Norms(cellOrdinal);
  },rankLocalSumSoln);
  
  double globalSumDiff = MPIWrapper::sum(*mesh1D1V->Comm(), rankLocalSumDiff);
  double globalSumSoln = MPIWrapper::sum(*mesh1D1V->Comm(), rankLocalSumSoln);
  
  double relativeDiff_L2 = std::sqrt(globalSumDiff / globalSumSoln);
  
  // check in what we checked out:
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(receivedBasisCoefficients);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(solnBasisCoefficients);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(cellMeasures);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(squaredCellDiffL2Norms);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(squaredCellSolnL2Norms);
  
  return relativeDiff_L2;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  mpiSession.barrier();
  
  bool success = true;
  {
    bool implicit = true;
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    
    const Kokkos::Array<double,3> v0 {-10.0, -10.0, -10.0};
    const Kokkos::Array<double,3> vDomainWidths {20, 20, 20};
    
    const double f_scaling_3V = 1.0 / (vDomainWidths[1] * vDomainWidths[2]); // scale to ensure 1D,3V agree on the initial density
    
    //  const bool useAdaptivity = false; // adaptivity not supported; we'd need to do something to keep 1D3V and 1D1V meshes in sync.
    
    // relative tolerance: require 1D1V and 1D3V solutions to be within 1e-8 of each other, in terms of |E|^2 norm.
    // We also require the f_xvx solution to agree to this tolerance.  The E tolerance is the harder one to hit, since
    // |E|^2 can be pretty close to zero.
    // The agreement should hold for arbitrary meshes (coarse and fine ones) and time step sizes: mathematically, 1D3V
    // and 1D1V should be identical for this problem.  The only differences should be due to roundoff and solver stopping criteria.
    double tol = 1e-8;
    
    int xElements = 2;
    int vElements_x = 2;
    int delta_k   = 2;
    double dt_max = 1e-1;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int numTimeSteps = 10;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    std::string problemChoiceString = "two-stream"; // "bump-on-tail" the alternative
    ProblemChoice problemChoice = TWO_STREAM;
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements_x", &vElements_x);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("u0", &u0);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
  //  cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("problem", &problemChoiceString);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    const int vElements_y = 1, vElements_z = 1;
    
    const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
    
    const int rank = mpiSession.getRank();
    
    problemChoice = getProblemChoice(problemChoiceString);
    
    if (problemChoice == BUMP_ON_TAIL)
    {
      k_x = 0.3;
      xWidth =  2 * M_PI / k_x;
      u0 = 4.5;
      n0 = 0.1;
    }
    
    using FormulationChoice1D3V = Camellia::Boltzmann1D3VFormulation::Boltzmann1D3VFormulationChoice;
    FormulationChoice1D3V formulationChoice1D3V;
    using FormulationChoice1D1V = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
    FormulationChoice1D1V formulationChoice1D1V;
    if (implicit)
    {
      if (useL2Trace)
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_IMPLICIT_L2_TRACE;
        formulationChoice1D1V = FormulationChoice1D1V::ULTRAWEAK_IMPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_IMPLICIT_H1_TRACE;
        formulationChoice1D1V = FormulationChoice1D1V::ULTRAWEAK_IMPLICIT_H1_TRACE;
      }
    }
    else
    {
      if (useL2Trace)
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_EXPLICIT_L2_TRACE;
        formulationChoice1D1V = FormulationChoice1D1V::ULTRAWEAK_EXPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice1D3V = FormulationChoice1D3V::ULTRAWEAK_EXPLICIT_H1_TRACE;
        formulationChoice1D1V = FormulationChoice1D1V::ULTRAWEAK_EXPLICIT_H1_TRACE;
      }
    }
    
    ERKTableauPtr erkTableau;
    if (!implicit)
    {
      erkTableau = ERKTableau::ssprk3();
    }
    
    double dt = dt_max;
    
    const bool usePeriodicBCs = true;
    Boltzmann1D3VFormulation form1D3V(formulationChoice1D3V, q, eps0, m, dt, x0, xWidth, xElements, v0,    vDomainWidths,    vMeshWidths,    H1Order, delta_k, usePeriodicBCs, erkTableau);
    Boltzmann1D1VFormulation form1D1V(formulationChoice1D1V, q, eps0, m, dt, x0, xWidth, xElements, v0[0], vDomainWidths[0], vMeshWidths[0], H1Order, delta_k, usePeriodicBCs, erkTableau);
    
    FunctionPtr f0;
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    
    if (problemChoice == TWO_STREAM)
    {
      double n = 0.5 * n0;
      FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
      FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
      f0                       = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    }
    else if (problemChoice == BUMP_ON_TAIL)
    {
      FunctionPtr fM_0 = Teuchos::rcp(new Maxwellian1V( 0.0, 1.0, 1.0-n0) );
      FunctionPtr fM_b = Teuchos::rcp(new Maxwellian1V(  u0, 0.5,     n0) );
      f0               = (fM_0 + 0.5 * fM_b) * (1. + gamma * cos_kx);
      std::cout << "f0 for bump-on-tail: " << f0 << "\n";
    }
    
    form1D3V.setInitialConditions(f0 * f_scaling_3V);
    form1D1V.setInitialConditions(f0);
    
    std::ostringstream suffix;
    if (implicit)
    {
      suffix << "Implicit";
    }
    if (useL2Trace)
    {
      suffix << "_L2Trace";
    }
    else
    {
      suffix << "_H1Trace";
    }
    suffix << "_u0" << u0;
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_" << xElements << "x" << vElements_x << "x" << vElements_y << "x" << vElements_z;
    
    std::string problemPrefix;
    if (problemChoice == TWO_STREAM)
    {
      problemPrefix = "TwoStream";
    }
    else if (problemChoice == BUMP_ON_TAIL)
    {
      problemPrefix = "BumpOnTail";
    }
    
    std::ostringstream boltzmannSS1D1V;
    boltzmannSS1D1V << problemPrefix <<"Boltzmann1D1V" << suffix.str();
    
    std::ostringstream boltzmannSS1D3V;
    boltzmannSS1D3V << problemPrefix <<"Boltzmann1D3V" << suffix.str();
    
    std::ostringstream gaussSS1D1V;
    gaussSS1D1V << problemPrefix << "Gauss1D1V" << suffix.str();
    
    std::ostringstream gaussSS1D3V;
    gaussSS1D3V << problemPrefix << "Gauss1D3V" << suffix.str();
    
  //  std::cout << "boltzmann name: " << boltzmannSS.str() << std::endl;
  //  std::cout << "gauss name: " << gaussSS.str() << std::endl;
    
    const std::string directoryPath = ".";
    
    HDF5Exporter boltzmannViz1D1V(form1D1V.boltzmannMesh(), boltzmannSS1D1V.str(), directoryPath);
    
    PhaseSpaceVisualization phaseSpaceViz1D3V(form1D3V.boltzmannSolution(), form1D3V.f(), boltzmannSS1D3V.str(), ".");
    HDF5Exporter gaussViz1D3V(form1D3V.gaussMesh(), gaussSS1D3V.str(), directoryPath);
    HDF5Exporter gaussViz1D1V(form1D1V.gaussMesh(), gaussSS1D1V.str(), directoryPath);
    
    std::ostringstream E_magnitudeFilePath;
    E_magnitudeFilePath << directoryPath << "/" << "E_magnitude" << suffix.str() << ".dat";
    std::fstream E_magnitudeOut;
    if (rank == 0)
    {
      E_magnitudeOut.open(E_magnitudeFilePath.str(), std::ios::out);
      if (!E_magnitudeOut.is_open())
      {
        std::cout << "Failed to open file " << E_magnitudeFilePath.str() << " on rank 0.\n";
        return -1;
      }
      else
      {
        std::cout << "Will write integral of |E|^2 to " << E_magnitudeFilePath.str() << "\n";
      }
    }
    
    double t = 0.0;
    gaussViz1D3V.exportSolution(form1D3V.gaussSolution(),t);
    gaussViz1D1V.exportSolution(form1D1V.gaussSolution(),t);
    boltzmannViz1D1V.exportSolution(form1D1V.boltzmannSolution(),t);
    phaseSpaceViz1D3V.exportTimeStep(t);
    
    FunctionPtr E_soln1D3V     = Function::solution(form1D3V.E(), form1D3V.gaussSolution());
    FunctionPtr E_squared1D3V  = E_soln1D3V * E_soln1D3V;
    
    FunctionPtr E_soln1D1V     = Function::solution(form1D1V.E(), form1D1V.gaussSolution());
    FunctionPtr E_squared1D1V  = E_soln1D1V * E_soln1D1V;
    
    FunctionPtr f_soln1D3V = Function::solution(form1D3V.f(), form1D3V.boltzmannSolution());
    
    // check that f0, and the initial projected solution is constant in v_y, v_z dimensions
    double f0_dvy_l2 = f0->di(3)->l2norm(form1D3V.boltzmannMesh());
    double f0_dvz_l2 = f0->di(4)->l2norm(form1D3V.boltzmannMesh());
    
    if (((f0_dvy_l2 > tol) || (f0_dvz_l2 > tol)) && (rank == 0))
    {
      std::cout << "ERROR: f0 varies in vy and/or vz.\n";
      std::cout << "NOTE: f_dvy_l2 = " << f0_dvy_l2 << std::endl;
      std::cout << "NOTE: f_dvz_l2 = " << f0_dvz_l2 << std::endl;
    }
    
    FunctionPtr f_soln1D3V_dvy = f_soln1D3V->di(3); // di() is 1-based: this is dvy
    FunctionPtr f_soln1D3V_dvz = f_soln1D3V->di(4); // dvz
    double f_dvy_l2 = f_soln1D3V_dvy->l2norm(form1D3V.boltzmannMesh());
    double f_dvz_l2 = f_soln1D3V_dvz->l2norm(form1D3V.boltzmannMesh());
    
    if (((f_dvy_l2 > tol) || (f_dvz_l2 > tol)) && (rank == 0))
    {
      std::cout << "ERROR: initial solution varies in vy and/or vz.\n";
      std::cout << "NOTE: f_dvy_l2 = " << f_dvy_l2 << std::endl;
      std::cout << "NOTE: f_dvz_l2 = " << f_dvz_l2 << std::endl;
    }
    
    auto subMeshTopo1D3V_xvx = SubExtrudedMeshTopology::subExtrudedMeshTopology(form1D3V.boltzmannMesh()->getTopology(), 0);
    MeshPtr subMesh1D3V_xvx = MeshFactory::minRuleMesh(subMeshTopo1D3V_xvx, form1D3V.boltzmannMesh()->bilinearForm(), H1Order, delta_k, form1D3V.boltzmannMesh()->Comm(), false);
    SolutionMomentFunction f_xvx(form1D3V.boltzmannSolution(), form1D3V.f(), 0, subMesh1D3V_xvx, 0); // 0: first extruded dimension is not integrated
    
    double maxDiff = 0.0;
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      form1D3V.takeTimeStep();
      form1D1V.takeTimeStep();
      t += dt;
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        double E_mag1D3V = E_squared1D3V->integrate(form1D3V.gaussMesh());
        double E_mag1D1V = E_squared1D1V->integrate(form1D1V.gaussMesh());
        
        // check that the 1D3V solution is constant in v_y, v_z dimensions
        double f_dvy_l2 = f_soln1D3V_dvy->l2norm(form1D3V.boltzmannMesh());
        double f_dvz_l2 = f_soln1D3V_dvz->l2norm(form1D3V.boltzmannMesh());
        
        f_xvx.updateValues();
        double relativeDiff_L2 = computeL2Difference(f_xvx, form1D1V.boltzmannSolution(), form1D1V.f());
        
        const double relativeDiff = std::abs(E_mag1D3V - E_mag1D1V) / std::max(E_mag1D3V, E_mag1D1V);
        maxDiff = std::max(relativeDiff,maxDiff);
        if ((relativeDiff > tol) || (f_dvy_l2 > tol) || (f_dvz_l2 > tol))
        {
          success = false;
        }

        boltzmannViz1D1V.exportSolution(form1D1V.boltzmannSolution(),t);
        gaussViz1D3V.exportSolution(form1D3V.gaussSolution(),t);
        gaussViz1D1V.exportSolution(form1D1V.gaussSolution(),t);
        phaseSpaceViz1D3V.exportTimeStep(t);

        if (rank == 0)
        {
          std::cout << "Completed time step " << timeStepNumber << " (t = " << t << ")" << std::endl;
          std::cout << "|E|^2 integral = " << E_mag1D3V << std::endl;
          E_magnitudeOut << t << "\t" << E_mag1D3V << std::endl << std::flush;
          
          std::cout << "relative difference in |E|^2 integral between 1D3V and 1D1V: " << relativeDiff << std::endl;
          std::cout << "relative difference in f_xvx solution between 1D3V and 1D1V: " << relativeDiff_L2 << std::endl;
          if (relativeDiff > tol)
          {
            std::cout << "NOTE: |E|^2 integral, 1D1V = " << E_mag1D1V << std::endl;
          }
          if ((f_dvy_l2 > tol) || (f_dvz_l2 > tol))
          {
            std::cout << "NOTE: f_dvy_l2 = " << f_dvy_l2 << std::endl;
            std::cout << "NOTE: f_dvz_l2 = " << f_dvz_l2 << std::endl;
          }
        }
      }
    }
    if (rank == 0)
    {
      E_magnitudeOut.close();
      if (success)
      {
        std::cout << "SUCCESS: ";
      }
      else
      {
        std::cout << "FAILURE: ";
      }
      std::cout << "max relative difference in |E|^2 integral was " << maxDiff << std::endl;
    }
  }
  
  bool printAllocationReport = false; // (rank == 0);
  
  Camellia::finalize(printAllocationReport);
  Kokkos::finalize();
  
  if (success)
    return 0;
  else
    return -1;
}
