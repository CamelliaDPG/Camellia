//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AdditiveSchwarz.h"
#include "BasisFactory.h"
#include "CamelliaDebugUtility.h"
#include "ConfigurationSpaceFormulationTrivial.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GMGOperator.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "LinearElasticityFormulation.h"
#include "MeshFactory.h"
#include "MonomialFunctions.h"
#include "NavierStokesVGPFormulation.h"
#include "RefinementStrategy.h"
#include "Solver.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"
#include "VlasovSpaceTimeFormulation.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "Epetra_SerialComm.h"

#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_IC.h"
#include "Ifpack_ILU.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;

bool runGMGOperatorInDebugMode;
int maxDofsForKLU;
double coarseCGTol;
int coarseMaxIterations;
bool narrateSolution;
bool narrateCoarseSolution;
bool saveVisualizationOutput = false;
bool useNodalBasis = true;  // default: use built-in intrepid basis
FunctionPtr errToPlot;
string errFunctionName;

string getFactorizationTypeString(GMGOperator::FactorType factorizationType)
{
  switch (factorizationType)
  {
  case GMGOperator::Direct:
    return "Direct";
  case GMGOperator::IC:
    return "IC";
  case GMGOperator::ILU:
    return "ILU";
  default:
    return "Unknown";
  }
}

GMGOperator::FactorType getFactorizationType(string factorizationTypeString)
{
  if (factorizationTypeString == "Direct")
  {
    return GMGOperator::Direct;
  }
  if (factorizationTypeString == "IC") return GMGOperator::IC;
  if (factorizationTypeString == "ILU") return GMGOperator::ILU;

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "factorization type not recognized");
}

Teuchos::RCP<Epetra_Operator> CamelliaAdditiveSchwarzPreconditioner(::Epetra_RowMatrix* A, int overlapLevel, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
    GMGOperator::FactorType schwarzBlockFactorization,
    int levelOfFill, double fillRatio, bool hOnly)
{
  int rank = mesh->Comm()->getRank();
  int sideDim = mesh->getTopology()->getDimension() - 1;
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization)
  {
  case GMGOperator::Direct:
    {
      AdditiveSchwarz<Ifpack_Amesos>* schwarz = new AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim);
      preconditioner = Teuchos::rcp( schwarz );
    }
    break;
  case GMGOperator::IC:
    preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_IC>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim) );
    List.set("fact: ict level-of-fill", fillRatio);
    break;
  case GMGOperator::ILU:
    preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_ILU>(A, overlapLevel, mesh, dofInterpreter, hOnly, sideDim) );
    List.set("fact: level-of-fill", levelOfFill);
    break;
  default:
    break;
  }

  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }

  err = preconditioner->Initialize();
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }
  
  AdditiveSchwarz<Ifpack_Amesos>* schwarzDirect = dynamic_cast<AdditiveSchwarz<Ifpack_Amesos>*>(preconditioner.get());
  
  if (schwarzDirect)
  {
    int maxIncidenceCount = schwarzDirect->MaxGlobalIncidenceCount();
    int maxNeighborCount = schwarzDirect->MaxGlobalOverlapSideNeighborCount();
    if (rank == 0) cout << "Schwarz: max incidence count: " << maxIncidenceCount << endl;
    if (rank == 0) cout << "Schwarz: max neighbor count: " << maxNeighborCount << endl;
    if (rank == 0) cout << "Theoretical sigma: 1/" << (2 + maxIncidenceCount) << endl;
    if (rank == 0) cout << "Our sigma: 1/" << (1 + maxNeighborCount) << endl;
  }

  return preconditioner;
}

Teuchos::RCP<Epetra_Operator> IfPackAdditiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel,
    GMGOperator::FactorType schwarzBlockFactorization,
    int levelOfFill, double fillRatio)
{
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization)
  {
  case GMGOperator::Direct:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel) );
    break;
  case GMGOperator::IC:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_IC>(A, overlapLevel) );
    List.set("fact: ict level-of-fill", fillRatio);
    break;
  case GMGOperator::ILU:
    preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_ILU>(A, overlapLevel) );
    List.set("fact: level-of-fill", levelOfFill);
    break;
  default:
    break;
  }

  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }


  err = preconditioner->Initialize();
  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0)
  {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }

  return preconditioner;
}

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "GlobalDofAssignment.h"

#include "CondensedDofInterpreter.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PoissonFormulation.h"
#include "ConvectionDiffusionFormulation.h"
#include "StokesVGPFormulation.h"

#include "GMGSolver.h"

enum ProblemChoice
{
  Poisson,
  ConvectionDiffusion,
  ConvectionDiffusionExperimental,
  Stokes,
  NavierStokes,
  LinearElasticity,
  H1NormGalerkinFormulation, // mass + stiffness for H^1, no DPG
  VlasovSpaceTime
};

enum RunManyPreconditionerChoices
{
  GMGAlgebraicSchwarz, // GMG with algebraic Schwarz smoother
  GMGGeometricSchwarz, // GMG with geometric Schwarz smoother
  GMGChebyshev,        // GMG with Chebyshev smoother
  GMGPointSymmetricGaussSeidel, // GMG with point symmetric Gauss Seidel smoother
  GMGBlockSymmetricGaussSeidel, // GMG with block symmetric Gauss Seidel smoother
  GMGPointJacobi, // GMG with point symmetric Jacobi smoother
  GMGBlockJacobi, // GMG with block symmetric Jacobi smoother
  AllGMG,     // Schwarz smoother, multiple overlap values, both algebraic and geometric Schwarz
  All         // All of the above, including the DontPrecondition option
};

namespace H1NormGalerkin {
  FunctionPtr exactSolution()
  {
    FunctionPtr x = MonomialFunction::xn(1);
    FunctionPtr y = MonomialFunction::yn(1);
    FunctionPtr z = MonomialFunction::zn(1);
    
    FunctionPtr soln = x*x*(x-1)*(x-1) + y*y*(y-1)*(y-1) + z*z*(z-1)*(z-1);
    
    return soln;
  }

  FunctionPtr rhs()
  {
    auto f = exactSolution();
    auto f_laplacian = f->dx()->dx() + f->dy()->dy() + f->dz()->dz();
    return f - f_laplacian;
  }
}

Teuchos::RCP<VlasovSpaceTimeFormulation>
vlasovFormulation1D1T1V(const double x0, const double xWidth,
                        const double t0, const double tWidth,
                        const double v0, const double vWidth,
                        const int xElements, const int tElements, const int vElements,
                        double q, double eps0, double m,
                        const int H1Order, const int delta_k,
                        const bool usePeriodicBCs, const bool useNodalBasis,
                        const int numUniformRefinements)
{
  Teuchos::RCP<ConfigurationSpaceFormulationTrivial> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationTrivial(TimeDiscretizationType::SPACE_TIME) );
  
  std::vector<FunctionPtr> E_constant(1, Function::constant(1.0));
  configForm->setE(E_constant);
  
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
  
  EBasisFamily basisFamily = useNodalBasis ? BASIS_FAMILY_NODAL : BASIS_FAMILY_HIERARCHICAL;
  
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

Teuchos::RCP<GMGSolver> initializeGMGSolver(SolutionPtr solution, int AztecOutputLevel, GMGOperator::SmootherChoice smootherType,
                                            int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
                                            int chebyshevDegree, double chebyshevRatio,
                                            bool useStaticCondensation, bool hOnly, bool useHierarchicalNeighborsForSchwarz,
                                            int H1Order_coarse, bool jumpToCoarsePolyOrder,
                                            Solver::SolverChoice coarseSolverChoice, int cgMaxIterations, double cgTol,
                                            double smootherWeight, bool useWeightMatrixForSchwarz)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  Teuchos::ParameterList pl;
  pl.set("kCoarse", H1Order_coarse-1);
  int delta_k = solution->mesh()->testSpaceEnrichment(); // might matter in cases where GMGOperator actually computes some local stiffness matrices on its own (currently, this can happen on coarse meshes when static condensation is employed).
  pl.set("delta_k", delta_k);
  pl.set("jumpToCoarsePolyOrder", jumpToCoarsePolyOrder);
  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(solution->mesh(), pl);
  
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine,
                                                                 cgMaxIterations, cgTol,
                                                                 GMGOperator::V_CYCLE,
                                                                 smootherType,
                                                                 Solver::getDirectSolver(true),
                                                                 useStaticCondensation));
  
  gmgSolver->setAztecOutput(0);
  gmgSolver->setComputeConditionNumberEstimate(false);
  
  gmgSolver->setNarrateOnRankZero(narrateSolution, "fine GMGSolver");
  gmgSolver->gmgOperator()->setNarrateOnRankZero(narrateSolution, "fine GMGOperator");
  gmgSolver->gmgOperator()->getCoarseSolution()->setNarrateOnRankZero(narrateCoarseSolution, "coarse solution");
  
  Teuchos::RCP<Solver> coarseSolver = Teuchos::null;
  
  if ((smootherType == GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) && hOnly)
  {
    // then use hierarchical neighbor relationship
    gmgSolver->gmgOperator()->setUseHierarchicalNeighborsForSchwarz(useHierarchicalNeighborsForSchwarz);
    
    if (rank==0)
    {
      if (useHierarchicalNeighborsForSchwarz)
        cout << "using hierarchical Schwarz neighbors option\n";
      else
        cout << "NOT using hierarchical Schwarz neighbors option\n";
    }
  }
  
  gmgSolver->setAztecOutput(AztecOutputLevel);
  gmgSolver->setUseConjugateGradient(true);
  gmgSolver->setComputeConditionNumberEstimate(true);
  gmgSolver->gmgOperator()->setSchwarzFactorizationType(schwarzBlockFactorization);
  gmgSolver->gmgOperator()->setLevelOfFill(schwarzLevelOfFill);
  gmgSolver->gmgOperator()->setFillRatio(schwarzFillRatio);
  gmgSolver->gmgOperator()->setChebyshevDegree(chebyshevDegree);
  gmgSolver->gmgOperator()->setChebyshevRatio(chebyshevRatio);
  gmgSolver->gmgOperator()->setSmootherType(smootherType);
  gmgSolver->gmgOperator()->setSmootherOverlap(schwarzOverlap);
  gmgSolver->gmgOperator()->setMultigridStrategy(Camellia::GMGOperator::V_CYCLE);
  if (smootherWeight != -1)
  {
    gmgSolver->gmgOperator()->setSmootherWeight(smootherWeight);
  }
  else
    gmgSolver->gmgOperator()->setUseSchwarzScalingWeight(true);
  if (dimensionForSchwarzNeighbors != -1)
  {
    gmgSolver->gmgOperator()->setDimensionForSchwarzNeighborRelationship(dimensionForSchwarzNeighbors);
  }
  
  gmgSolver->gmgOperator()->setUseSchwarzDiagonalWeight(useWeightMatrixForSchwarz);
  
  gmgSolver->gmgOperator()->setDebugMode(runGMGOperatorInDebugMode);
  
  if (saveVisualizationOutput)
  {
    Teuchos::RCP<HDF5Exporter> exporter = Teuchos::rcp(new HDF5Exporter(solution->mesh(), "preconditionerError", "./") );
    gmgSolver->gmgOperator()->setFunctionExporter(exporter, errToPlot, errFunctionName);
  }
  
  return gmgSolver;
}

string smootherString(GMGOperator::SmootherChoice smoother)
{
  switch (smoother)
  {
  case GMGOperator::IFPACK_ADDITIVE_SCHWARZ:
    return "IfPack-Schwarz";
  case GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ:
    return "Camellia-Schwarz";
  case GMGOperator::CHEBYSHEV:
    return "Chebyshev";
  case GMGOperator::NONE:
    return "None";
  case GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL:
    return "Block-Gauss-Seidel";
  case GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL:
    return "Point-Gauss-Seidel";
  case GMGOperator::BLOCK_JACOBI:
    return "Block-Jacobi";
  case GMGOperator::POINT_JACOBI:
    return "Point-Jacobi";
  default:
    return "Unknown";
  }
}

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, MeshPtr &coarseMesh, IPPtr &graphNorm, double graphNormBeta, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces, bool useStaticCondensation,
                                     int numCells, int H1Order, int delta_k, int H1Order_coarse, bool useNodalBasis,
                                     int rootMeshNumCells, bool hOnly, bool useHierarchicalNeighborsForSchwarz, bool useZeroMeanConstraints,
                                     bool enhanceFieldsForH1TracesWhenConforming,
                                     Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulationFine,
                                     Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulationCoarse,
                                     Teuchos::RCP<VlasovSpaceTimeFormulation> &vlasovForm)
{
  BFPtr bf;
  BCPtr bc;
  RHSPtr rhs;
  MeshPtr mesh;
  
  int rank = Teuchos::GlobalMPISession::getRank();

  BasisFamilyPtr basisFamily;
  if (useNodalBasis)
  {
    basisFamily = BasisFamily::nodalBasisFamily();
  }
  else
  {
    basisFamily = BasisFamily::hierarchicalBasisFamily();
  }
  
  double width = 1.0; // in each dimension
  vector<double> x0(spaceDim,0); // origin is the default

  VarPtr p; // pressure

  map<int,int> trialOrderEnhancements;
  FunctionPtr exactSolution;
  VarPtr varExact; // the variable of which we have the exact solution in exactSolution

  switch (problemChoice)
  {
    case Poisson:
      {
        PoissonFormulation formulation(spaceDim, conformingTraces);
        
        bf = formulation.bf();
        
        rhs = RHS::rhs();
        FunctionPtr f = Function::constant(1.0);
        
        VarPtr q = formulation.v();
        rhs->addTerm( f * q );
        
        bc = BC::bc();
        SpatialFilterPtr boundary = SpatialFilter::allSpace();
        VarPtr phi_hat = formulation.u_hat();
        bc->addDirichlet(phi_hat, boundary, Function::zero());
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          trialOrderEnhancements[formulation.u()->ID()] = 1;
        }
        
        if (saveVisualizationOutput)
        {
          // we don't have an exact solution, so let's just plot phi instead...
          varExact = formulation.u();
          exactSolution = Function::zero();
        }
      }
      break;
    case ConvectionDiffusion:
      {
        double epsilon = 1e-2;
        x0 = vector<double>(spaceDim,-1.0);
        
        FunctionPtr beta;
        FunctionPtr beta_x = Function::constant(1);
        FunctionPtr beta_y = Function::constant(2);
        FunctionPtr beta_z = Function::constant(3);
        if (spaceDim == 1)
          beta = beta_x;
        else if (spaceDim == 2)
          beta = Function::vectorize(beta_x, beta_y);
        else if (spaceDim == 3)
          beta = Function::vectorize(beta_x, beta_y, beta_z);
        ConvectionDiffusionFormulation formulation(spaceDim, conformingTraces, beta, epsilon);
        
        bf = formulation.bf();
        
        rhs = RHS::rhs();
        FunctionPtr f = Function::constant(1.0);
        
        VarPtr v = formulation.v();
        rhs->addTerm( f * v );
        
        bc = BC::bc();
        VarPtr uhat = formulation.uhat();
        VarPtr tc = formulation.tc();
        SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
        SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
        SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
        SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
        SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
        SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
        FunctionPtr zero = Function::zero();
        FunctionPtr one = Function::constant(1);
        FunctionPtr x = Function::xn(1);
        FunctionPtr y = Function::yn(1);
        FunctionPtr z = Function::zn(1);
        if (spaceDim == 1)
        {
          bc->addDirichlet(tc, inflowX, -one);
          bc->addDirichlet(uhat, outflowX, zero);
        }
        if (spaceDim == 2)
        {
          bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
          bc->addDirichlet(uhat, outflowX, zero);
          bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
          bc->addDirichlet(uhat, outflowY, zero);
        }
        if (spaceDim == 3)
        {
          bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
          bc->addDirichlet(uhat, outflowX, zero);
          bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
          bc->addDirichlet(uhat, outflowY, zero);
          bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
          bc->addDirichlet(uhat, outflowZ, zero);
        }
        
        //    if (rank==0) cout << "NOTE: TRYING SOMETHING IN ConvectionDiffusion's IP.\n";
        //
        //    VarPtr tau = formulation.tau();
        //    map<int,double> trialWeights; // leave empty for unit weights (default)
        //    map<int,double> testL2Weights;
        //    testL2Weights[v->ID()] = 1.0;
        //    testL2Weights[tau->ID()] = 1.0 / epsilon;
        //    graphNorm = bf->graphNorm(trialWeights,testL2Weights);
        //    if (rank==0)
        //    {
        //      bf->printTrialTestInteractions();
        //      cout << "RHS: " << rhs->linearTerm()->displayString() << endl;
        //    }
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          trialOrderEnhancements[formulation.u()->ID()] = 1;
        }
      }
      break;
    case ConvectionDiffusionExperimental:
      {
        double epsilon = 1e-2;
        x0 = vector<double>(spaceDim,-1.0);
        
        FunctionPtr beta;
        FunctionPtr beta_x = Function::constant(1);
        FunctionPtr beta_y = Function::constant(2);
        FunctionPtr beta_z = Function::constant(3);
        if (spaceDim == 1)
          beta = beta_x;
        else if (spaceDim == 2)
          beta = Function::vectorize(beta_x, beta_y);
        else if (spaceDim == 3)
          beta = Function::vectorize(beta_x, beta_y, beta_z);
        
        FunctionPtr zero = Function::zero();
        FunctionPtr one = Function::constant(1);
        
        Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
        Space uhat_space = conformingTraces ? HGRAD : L2;
        Space vSpace = (spaceDim > 1) ? VECTOR_L2 : L2;
        
        // fields
        VarPtr u;
        VarPtr sigma;
        
        // traces
        VarPtr uhat, tc;
        
        // tests
        VarPtr v;
        VarPtr tau;
        
        VarFactoryPtr vf = VarFactory::varFactory();
        u = vf->fieldVar("u");
        sigma = vf->fieldVar("sigma", vSpace);
        
        TFunctionPtr<double> n = TFunction<double>::normal();
        TFunctionPtr<double> parity = TFunction<double>::sideParity();
        
        if (spaceDim > 1)
          uhat = vf->traceVar("uhat", u, uhat_space);
        else
          uhat = vf->fluxVar("uhat", u * (parity * Function::normal_1D()), uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
        
        
        if (spaceDim > 1)
          tc = vf->fluxVar("tc", (beta*u-sigma) * (n * parity));
        else
          tc = vf->fluxVar("tc", (beta*u-sigma) * (parity * Function::normal_1D()));
        
        v = vf->testVar("v", HGRAD);
        tau = vf->testVar("tau", tauSpace);
        
        bf = Teuchos::rcp( new BF(vf) );
        
        if (spaceDim==1)
        {
          // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
          bf->addTerm(sigma, tau);
          bf->addTerm(u * epsilon, tau->dx());
          bf->addTerm(-epsilon * uhat, tau);
          
          bf->addTerm(-beta*u + sigma, v->dx());
          bf->addTerm(tc, v);
        }
        else
        {
          // compared with the standard formulation, we multiply all tau terms by epsilon (tau doesn't enter RHS)
          bf->addTerm(sigma, tau);
          bf->addTerm(u * epsilon, tau->div());
          bf->addTerm(-epsilon * uhat, tau->dot_normal());
          
          bf->addTerm(-beta*u + sigma, v->grad());
          bf->addTerm(tc, v);
        }
        
        rhs = RHS::rhs();
        FunctionPtr f = Function::constant(1.0);
        
        rhs->addTerm( f * v );
        
        bc = BC::bc();
        SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
        SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
        SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
        SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
        SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
        SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
        FunctionPtr x = Function::xn(1);
        FunctionPtr y = Function::yn(1);
        FunctionPtr z = Function::zn(1);
        if (spaceDim == 1)
        {
          bc->addDirichlet(tc, inflowX, -one);
          bc->addDirichlet(uhat, outflowX, zero);
        }
        if (spaceDim == 2)
        {
          bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
          bc->addDirichlet(uhat, outflowX, zero);
          bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
          bc->addDirichlet(uhat, outflowY, zero);
        }
        if (spaceDim == 3)
        {
          bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
          bc->addDirichlet(uhat, outflowX, zero);
          bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
          bc->addDirichlet(uhat, outflowY, zero);
          bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
          bc->addDirichlet(uhat, outflowZ, zero);
        }
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          trialOrderEnhancements[u->ID()] = 1;
        }
      }
      break;
    case Stokes:
      {
        double mu = 1.0;
        StokesVGPFormulation formulation = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
        
        p = formulation.p();
        
        bf = formulation.bf();
        graphNorm = bf->graphNorm();
        
        rhs = RHS::rhs();
        
        FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
        FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
        FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
        FunctionPtr exp_z = Teuchos::rcp( new Exp_z );
        
        FunctionPtr x = Function::xn(1);
        FunctionPtr y = Function::yn(1);
        FunctionPtr z = Function::zn(1);
        
        FunctionPtr u1_exact, u2_exact, u3_exact, p_exact;
        
        if (spaceDim == 2)
        {
          // this one was in the Cockburn Kanschat LDG Stokes paper
          u1_exact = - exp_x * ( y * cos_y + sin_y );
          u2_exact = exp_x * y * sin_y;
          p_exact = 2.0 * exp_x * sin_y;
        }
        else
        {
          // this one is inspired by the 2D one
          u1_exact = - exp_x * ( y * cos_y + sin_y );
          u2_exact = exp_x * y * sin_y + exp_z * y * cos_y;
          u3_exact = - exp_z * (cos_y - y * sin_y);
          p_exact = 2.0 * exp_x * sin_y + 2.0 * exp_z * cos_y;
          // DEBUGGING:
          //      u1_exact = Function::zero();
          //      u2_exact = Function::zero();
          //      u3_exact = x;
          //      p_exact = Function::zero();
        }
        if (saveVisualizationOutput)
        {
          exactSolution = u1_exact;
          varExact = formulation.u(1);
          errFunctionName = "u1_error";
        }
        
        // to ensure zero mean for p, need the domain carefully defined:
        x0 = vector<double>(spaceDim,-1.0);
        
        width = 2.0;
        
        bc = BC::bc();
        // our usual way of adding in the zero mean constraint results in a negative eigenvalue
        // therefore, for now, we use a single-point BC
        //    bc->addZeroMeanConstraint(formulation.p());
        SpatialFilterPtr boundary = SpatialFilter::allSpace();
        bc->addDirichlet(formulation.u_hat(1), boundary, u1_exact);
        bc->addDirichlet(formulation.u_hat(2), boundary, u2_exact);
        if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, u3_exact);
        
        FunctionPtr f1, f2, f3;
        if (spaceDim==2)
        {
          f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
          f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
        }
        else
        {
          f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
          f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
          f3 = -p_exact->dz() + mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
        }
        
        VarPtr v1 = formulation.v(1);
        VarPtr v2 = formulation.v(2);
        
        VarPtr v3;
        if (spaceDim==3) v3 = formulation.v(3);
        
        RHSPtr rhs = RHS::rhs();
        if (spaceDim==2)
          rhs->addTerm(f1 * v1 + f2 * v2);
        else
          rhs->addTerm(f1 * v1 + f2 * v2 + f3 * v3);
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          for (int d=0; d<spaceDim; d++)
            trialOrderEnhancements[formulation.u(d+1)->ID()] = 1;
        }
      }
      break;
    case LinearElasticity:
      {
        //    cout << "LinearElasticity not yet supported!\n";
        //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "LinearElasticity not yet supported!");
        TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "only spaceDim = 2 is supported");
        
        double mu = 1.0, lambda = 1.0;
        LinearElasticityFormulation formulation = LinearElasticityFormulation::steadyFormulation(spaceDim, lambda, mu, conformingTraces);
        
        bf = formulation.bf();
        graphNorm = bf->graphNorm();
        
        const static double PI  = 3.141592653589793238462;
        
        FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI) );
        FunctionPtr sin_pi_y = Teuchos::rcp( new Sin_ay(PI) );
        
        FunctionPtr u1_exact, u2_exact, u3_exact;
        
        if (spaceDim == 2)
        {
          // u_x = sin (pi x) sin (pi y)
          // u_y = sin (pi x) sin (pi y)
          
          u1_exact = sin_pi_x * sin_pi_y;
          u2_exact = sin_pi_x * sin_pi_y;
        }
        else
        {
          // TODO: define an exact solution for 3D
        }
        
        x0 = vector<double>(spaceDim,0.0); // origin
        
        width = 1.0;
        
        bc = BC::bc();
        
        FunctionPtr zero = Function::zero();
        SpatialFilterPtr boundary = SpatialFilter::allSpace();
        bc->addDirichlet(formulation.u_hat(1), boundary, zero);
        bc->addDirichlet(formulation.u_hat(2), boundary, zero);
        if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, zero);
        
        //    vector<FunctionPtr> f_vector(spaceDim, zero);
        //    for (int i=1; i<= spaceDim; i++)
        //    {
        //      for (int j=1; j<= spaceDim; j++)
        //      {
        //        for (int k=1; k<= spaceDim; k++)
        //        {
        //          FunctionPtr u_k;
        //          switch (k) {
        //            case 1:
        //              u_k = u1_exact;
        //              break;
        //            case 2:
        //              u_k = u2_exact;
        //              break;
        //            case 3:
        //              u_k = u3_exact;
        //            default:
        //              break;
        //          }
        //          for (int l=1; l<= spaceDim; l++)
        //          {
        //            FunctionPtr u_k_lj = u_k->grad()->spatialComponent(l)->grad()->spatialComponent(j);
        //            double E_ijkl = formulation.E(i, j, k, l);
        ////            cout << i << ", " << j << ", " << k << ", " << l << ": ";
        ////            cout << -C_ijkl << " * " << u_k_lj->displayString() << endl;
        //            if (E_ijkl == 0) f_vector[i-1] = f_vector[i-1] + zero;
        //            else f_vector[i-1] = f_vector[i-1] -E_ijkl * u_k_lj;
        ////            cout << f_vector[i-1]->displayString() << endl;
        //          }
        //        }
        //      }
        //
        ////      cout << "f[" << i << "]: " << f_vector[i-1]->displayString() << endl;
        //    }
        //
        //    FunctionPtr f = Function::vectorize(f_vector);
        
        FunctionPtr u_exact = (spaceDim == 2) ? Function::vectorize(u1_exact, u2_exact) : Function::vectorize(u1_exact, u2_exact, u3_exact);
        FunctionPtr f = formulation.forcingFunction(u_exact);
        
        if (saveVisualizationOutput)
        {
          exactSolution = u1_exact;
          varExact = formulation.u(1);
          errFunctionName = "u1_error";
        }
        
        VarPtr v1 = formulation.v(1);
        VarPtr v2 = formulation.v(2);
        
        VarPtr v3;
        if (spaceDim==3) v3 = formulation.v(3);
        
        rhs = formulation.rhs(f);
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          for (int d=0; d<spaceDim; d++)
            trialOrderEnhancements[formulation.u(d+1)->ID()] = 1;
        }
      }
      break;
    case NavierStokes:
      {
        if (spaceDim != 2)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // we can add support for a 3D exact solution later...
        }
        
        MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({2.0,2.0}, {rootMeshNumCells, rootMeshNumCells}, {-0.5,0.0});
        MeshTopologyViewPtr coarseMeshTopo;
        if (!hOnly)
        {
          coarseMeshTopo = meshTopo;
        }
        int meshWidthCells = rootMeshNumCells;
        while (meshWidthCells < numCells)
        {
          vector<IndexType> activeCellIDs = meshTopo->getActiveCellIndicesGlobal(); // should match between coarseMesh and mesh
          IndexType nextCellIndexFine = meshTopo->nextCellID();
          for (IndexType cellIndex : activeCellIDs)
          {
            CellTopoPtr cellTopo = meshTopo->getCell(cellIndex)->topology();
            RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
            meshTopo->refineCell(cellIndex, refPattern, nextCellIndexFine);
            nextCellIndexFine += refPattern->numChildren();
          }
          meshWidthCells *= 2;
          
          if (hOnly && (meshWidthCells == numCells))
          {
            // final iteration for h-multigrid: coarseMesh should be the parents of the most recently refined guys
            set<IndexType> cellsForCoarseMesh;
            cellsForCoarseMesh.insert(activeCellIDs.begin(),activeCellIDs.end());
            coarseMeshTopo = meshTopo->getView(cellsForCoarseMesh);
          }
        }
        
        // TODO: respect basisFamily choice for Navier-Stokes
        
        // set up classical Kovasznay flow solution:
        double Re = 40.0;
        navierStokesFormulationFine = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, H1Order-1, delta_k)));
        navierStokesFormulationCoarse = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, coarseMeshTopo, H1Order_coarse-1, delta_k)));
        
        int k_low_order = 1;
        NavierStokesVGPFormulation lowestOrderForm(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k_low_order-1, delta_k));
        
        FunctionPtr u1, u2, p;
        NavierStokesVGPFormulation::getKovasznaySolution(Re, u1, u2, p);
        
        mesh = navierStokesFormulationFine->solutionIncrement()->mesh();
        coarseMesh = navierStokesFormulationCoarse->solutionIncrement()->mesh();
        
        if (useZeroMeanConstraints)
        {
          navierStokesFormulationFine->addZeroMeanPressureCondition();
          lowestOrderForm.addZeroMeanPressureCondition();
          double p_mean = p->integrate(mesh);
          p = p - p_mean;
        }
        else
        {
          navierStokesFormulationFine->addPointPressureCondition({0.5,1.0});
          lowestOrderForm.addPointPressureCondition({0.5,1.0});
          double p_center = p->evaluate(0.5, 1.0);
          p = p - p_center;
        }
        
        FunctionPtr u = Function::vectorize({u1, u2});
        FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunctionSteady(spaceDim, Re, u, p);
        
        int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.
        
        navierStokesFormulationFine->addInflowCondition(SpatialFilter::allSpace(), u);
        navierStokesFormulationFine->setForcingFunction(forcingFunction);
        
        lowestOrderForm.addInflowCondition(SpatialFilter::allSpace(), u);
        lowestOrderForm.setForcingFunction(forcingFunction);
        
        lowestOrderForm.solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
        
        if (rank == 0) cout << "Navier-Stokes: taking three Newton steps on first-order mesh; using this as background flow.\n";
        for (int i=0; i<3; i++)
        {
          lowestOrderForm.solveAndAccumulate();
        }
        lowestOrderForm.solution()->projectFieldVariablesOntoOtherSolution(navierStokesFormulationFine->solution());
        
        navierStokesFormulationFine->solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
        navierStokesFormulationFine->solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
        
        if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // if it seems important, we can add support for this later...
        }
        
        graphNorm = navierStokesFormulationFine->bf()->graphNorm();
        
        solution = navierStokesFormulationFine->solutionIncrement();
        solution->setUseCondensedSolve(useStaticCondensation);
        
        if (saveVisualizationOutput)
        {
          exactSolution = u1;
          varExact = navierStokesFormulationFine->u(1);
          errFunctionName = "u1_error";
          FunctionPtr u1_background = Function::solution(varExact, navierStokesFormulationFine->solution());
          FunctionPtr u1_increment = Function::solution(varExact, navierStokesFormulationFine->solutionIncrement());
          // exact increment is (u1 - u1_background)
          errToPlot = u1_increment - (u1 - u1_background);
        }
        return; // return early for Navier-Stokes; we've already set up the Solution objects, etc.
      }
      break;
    case H1NormGalerkinFormulation:
      {
        VarFactoryPtr vf = VarFactory::varFactory();
        VarPtr u     = vf->fieldVar("u", HGRAD);
        vf->setBasisFamily(BasisFamily::nodalBasisFamily());
        
        // test variables are discretized the same way as trial, so can be represented by the same VarPtrs; we relabel them for clarity:
        auto v            = u;
        auto bubnovVF = vf->getBubnovFactory(Camellia::VarFactory::BUBNOV_TRIAL);
        bf = BF::bf(bubnovVF);
        
        bf->addTerm(u, v);
        bf->addTerm(u->grad(), v->grad());
        
        rhs = RHS::rhs();
        FunctionPtr f_rhs = H1NormGalerkin::rhs();
        
        rhs->addTerm( f_rhs * v );
        
        bc = BC::bc();
      }
      break;
    case VlasovSpaceTime:
      {
        TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 3, std::invalid_argument, "Only 1D1T1V Vlasov-Poisson is supported right now; to specify this, spaceDim should be set to 3");
        double q = 1.0;
        double eps0 = 1.0;
        double m = 1.0;
        const bool usePeriodicBCs = false;
        vlasovForm = vlasovFormulation1D1T1V(x0[0], width, x0[1], width, x0[2], width,
                                             rootMeshNumCells, rootMeshNumCells, rootMeshNumCells,
                                             q, eps0, m,
                                             H1Order, delta_k, usePeriodicBCs,
                                             useNodalBasis,0);
        
        trialOrderEnhancements = vlasovForm->vlasovMesh()->getDofOrderingFactory().getTrialOrderEnhancements();
        
        BCPtr  bc  = BC::bc();
        RHSPtr rhs = RHS::rhs();
        
        FunctionPtr x = Function::xn(1); // x^1
        FunctionPtr t = Function::yn(1); // t^1
        FunctionPtr v = vlasovForm->v_i(0);
        
        FunctionPtr E = vlasovForm->configurationSpaceFormulation()->E_i(0);
        
        FunctionPtr sin_pi_x_over_2 = TrigFunctions<double>::sin(0.5 * M_PI * x);
        FunctionPtr cos_2_pi_t      = TrigFunctions<double>::cos(2.0 * M_PI * t);
        FunctionPtr cos_pi_v        = TrigFunctions<double>::cos(      M_PI * v);
        
        FunctionPtr f_exact = sin_pi_x_over_2 * cos_2_pi_t + cos_pi_v;
        
        VarPtr w      = vlasovForm->w();
        VarPtr tn_hat = vlasovForm->t_n_hat();
        
        FunctionPtr rhs_exact = v * f_exact->dx() + f_exact->dy() + E * f_exact->dz();
        rhs->addTerm(rhs_exact * w);
        
        // impose boundary conditions on tn at the inflow.  This is defined by:
        //   (v,1,E) \cdot n < 0.
        // n for us is always parallel to one of the axes.  So depending on which side we are on,
        // there will be a requirement on v * n_x, n_t, or E n_v.  Since E is 1, and our domain is [0,1]^3,
        // the inflow boundary is as follows:
        // - the plane x = 0,
        // - the plane t = 0,
        // - the plane v = 0
        SpatialFilterPtr inflow = SpatialFilter::lessThanOrEqualToX(0.0) | SpatialFilter::lessThanOrEqualToY(0.0) | SpatialFilter::lessThanOrEqualToZ(0.0);
        FunctionPtr n = Function::normal();
        FunctionPtr tn_exact = v * f_exact * n->x() + f_exact * n->y() + f_exact * E * n->z();
        bc->addDirichlet(tn_hat, inflow, tn_exact);
        
        mesh = vlasovForm->vlasovMesh();
        solution = vlasovForm->vlasovSolution();
        
        solution->setBC(bc);
        solution->setRHS(rhs);
        
        bf = vlasovForm->vlasovBF();
        graphNorm = solution->ip();
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported problem choice");
  }

  if (problemChoice != VlasovSpaceTime)
  {
    vector<double> dimensions;
    vector<int> elementCounts;
    
    for (int d=0; d<spaceDim; d++)
    {
      dimensions.push_back(width);
      elementCounts.push_back(rootMeshNumCells);
    }
    bf->varFactory()->setBasisFamily(basisFamily);
    mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0, trialOrderEnhancements);
  }
  
  // now that we have mesh, add pressure constraint for Stokes (imposing zero at origin--want to aim for center of mesh)
  if (problemChoice == Stokes)
  {
    if (!useZeroMeanConstraints)
    {
      vector<double> origin(spaceDim,0);
      IndexType vertexIndex;
      bool vertexFound = mesh->getTopology()->getVertexIndex(origin, vertexIndex);
      vertexFound = MPIWrapper::globalOr(mesh->Comm(), vertexFound);
      if (!vertexFound)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "origin vertex not found");
      }
      bc->addSpatialPointBC(p->ID(), 0.0, origin);
    }
    else
    {
      bc->addZeroMeanConstraint(p);
    }
  }

  int numUniformRefinements = 0;
  int meshWidth = rootMeshNumCells;
  while (meshWidth < numCells)
  {
    numUniformRefinements++;
    meshWidth *= 2;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(meshWidth != numCells, std::invalid_argument, "meshWidth must be a power of two times rootMeshNumCells");
  
  int meshWidthCells = rootMeshNumCells;
  while (meshWidthCells < numCells)
  {
    set<IndexType> activeCellIDs = mesh->getActiveCellIDsGlobal(); // should match between coarseMesh and mesh
    mesh->hRefine(activeCellIDs);
    meshWidthCells *= 2;
    
    if (hOnly && (meshWidthCells == numCells))
    {
      // final iteration for h-multigrid: coarseMesh should be the parents of the most recently refined guys
      set<IndexType> cellsForCoarseMesh;
      cellsForCoarseMesh.insert(activeCellIDs.begin(),activeCellIDs.end());
      MeshTopologyViewPtr coarseMeshTopo = mesh->getTopology()->getView(cellsForCoarseMesh);
      MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicyFromRefinedMesh(coarseMeshTopo, mesh);
      coarseMesh = Teuchos::rcp( new Mesh(coarseMeshTopo, bf, H1Order_coarse, delta_k, map<int,int>(), map<int,int>(), inducedPartitionPolicy) ) ;
    }
  }

  if (meshWidthCells != numCells)
  {
    int rank = Teuchos::GlobalMPISession::getRank();
    if (rank == 0)
    {
      cout << "Warning: may have overrefined mesh; mesh has width " << meshWidthCells << ", not " << numCells << endl;
    }
  }

  // coarse and fine mesh share a MeshTopology.  This means that they should not be further refined (they won't be, here)
  if (!hOnly)
  {
    // ensure that coarse and fine mesh share a common partitioning
    MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(mesh);
    coarseMesh = Teuchos::rcp(new Mesh(mesh->getTopology(), mesh->bilinearForm(), H1Order_coarse, delta_k, trialOrderEnhancements,
                                       map<int,int>(), inducedPartitionPolicy));
  }
  
  if ((problemChoice != H1NormGalerkinFormulation) && (graphNorm == Teuchos::null)) // if set previously, honor that...
    graphNorm = bf->graphNorm();

  if (problemChoice != VlasovSpaceTime)
  {
    solution = Solution::solution(mesh, bc, rhs, graphNorm);
  }
  solution->setUseCondensedSolve(useStaticCondensation);
  solution->setZMCsAsGlobalLagrange(false); // fine grid solution shouldn't impose ZMCs (should be handled in coarse grid solve)
  
  if ((saveVisualizationOutput) && (varExact != Teuchos::null))
  {
    FunctionPtr var_soln = Function::solution(varExact, solution);
    // exact increment is (u1 - u1_background)
    errToPlot = exactSolution - var_soln;
  }
}

void run(ProblemChoice problemChoice, int &iterationCount, int spaceDim, int numCells,
         int H1Order, int delta_k, int H1Order_coarse, bool jumpToCoarsePolyOrder, bool useNodalBasis,
         bool conformingTraces, bool useStaticCondensation,
         GMGOperator::SmootherChoice smootherType,
         int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
         int chebyshevDegree, double chebyshevRatio,
         Solver::SolverChoice coarseSolverChoice, double cgTol, int cgMaxIterations, int AztecOutputLevel,
         bool reportTimings, double &solveTime, double &error, bool reportEnergyError, int numCellsRootMesh,
         bool hOnly, bool useHierarchicalNeighborsForSchwarz, bool useZeroMeanConstraints,
         bool writeAndExit, double smootherWeight, bool useWeightMatrixForSchwarz,
         bool enhanceFieldsForH1TracesWhenConforming, double graphNormBeta)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (H1Order_coarse == -1)
  {
    if (hOnly)
      H1Order_coarse = H1Order;
    else
    {
      H1Order_coarse = (H1Order > 1) ? ceil(H1Order / 2.0) : 1;
    }
  }
  
  if (numCellsRootMesh == -1)
  {
    int evenDivisor = numCells;
    
    while ((evenDivisor/2) * 2 == evenDivisor)
    {
      evenDivisor /= 2;
    }
    
    numCellsRootMesh = numCells;
    if (problemChoice == Stokes)
    {
      // need origin to be a vertex in root mesh; this is true if the number of root cells is even
      numCellsRootMesh = (evenDivisor % 2 == 0) ? evenDivisor : evenDivisor * 2;
    }
    else
    {
      numCellsRootMesh = max(evenDivisor,1);
    }
  }
  if ((numCellsRootMesh == numCells) && hOnly)
  {
    cout << "Too few cells in root mesh.  Aborting.\n";
    exit(1);
  }
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time initializationTimer(Comm);

  SolutionPtr solution;
  MeshPtr k0Mesh;
  IPPtr graphNorm;
  Teuchos::RCP<NavierStokesVGPFormulation> navierStokesFormFine, navierStokesFormCoarse; // NULL if not doing Navier-Stokes
  Teuchos::RCP<VlasovSpaceTimeFormulation> vlasovForm;
  initializeSolutionAndCoarseMesh(solution, k0Mesh, graphNorm, graphNormBeta, problemChoice, spaceDim, conformingTraces, useStaticCondensation,
                                  numCells, H1Order, delta_k, H1Order_coarse, useNodalBasis, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz,
                                  useZeroMeanConstraints, enhanceFieldsForH1TracesWhenConforming,
                                  navierStokesFormFine, navierStokesFormCoarse, vlasovForm);
  
  MeshPtr mesh = solution->mesh();
  BCPtr bc = solution->bc();

  int coarseElements = k0Mesh->numActiveElements(), fineElements = mesh->numActiveElements();
  int fineDofs = mesh->numGlobalDofs(), coarseDofs = k0Mesh->numGlobalDofs();
  if (rank == 0)
  {
    cout << "fine mesh has " << fineElements << " active elements and " << fineDofs << " degrees of freedom.\n";
    cout << "coarse mesh has " << coarseElements << " active elements and " << coarseDofs << " degrees of freedom.\n";
  }
  
  
//  {
//    // DEBUGGING
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//    if (rank==0)
//      minRule->printGlobalDofInfo();
//  }
  
//  if (hOnly)
//  {
//    // then replace the k0Mesh with the h-coarsened mesh:
//    Teuchos::ParameterList pl;
//    pl.set("kCoarse", 0);
//    pl.set("delta_k", 1); // this should not really matter in this context
//    pl.set("jumpToCoarsePolyOrder", false);
//    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(mesh, pl); // not the most efficient way to do this, but it should work...
//
//    k0Mesh = meshesCoarseToFine[meshesCoarseToFine.size()-2];
////    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
////    MeshTopologyPtr coarseMeshTopo = meshTopo->getRootMeshTopology();
////    int H1OrderP0 = k + 1;
////    k0Mesh = Teuchos::rcp( new Mesh(coarseMeshTopo, k0Mesh->bilinearForm(), H1OrderP0, delta_k) );
//  }

  double initializationTime = initializationTimer.ElapsedTime();
  int numCoarseGlobalDofs = k0Mesh->numGlobalDofs();
  int numFineGlobalDofs = solution->mesh()->numGlobalDofs();
  if (narrateSolution && (rank==0))
  {
    cout << "Solution (" << numFineGlobalDofs << " fine global dofs) and k0 mesh (";
    cout << numCoarseGlobalDofs << " coarse global dofs) initialized in " << initializationTime << " seconds.\n";
  }
  
  // to be fairer to preconditioners whose construction is cheap, we include construction time in the solve time (this is a change, made 7-7-15)
  Epetra_Time solveTimer(Comm);
  
  Teuchos::RCP<Solver> solver = initializeGMGSolver(solution, AztecOutputLevel, smootherType,
                                                    schwarzOverlap, dimensionForSchwarzNeighbors,
                                                    schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio,
                                                    chebyshevDegree, chebyshevRatio,
                                                    useStaticCondensation, hOnly, useHierarchicalNeighborsForSchwarz,
                                                    H1Order_coarse, jumpToCoarsePolyOrder, coarseSolverChoice,
                                                    cgMaxIterations, cgTol,
                                                    smootherWeight, useWeightMatrixForSchwarz);

  if (writeAndExit)
  {
    solution->initializeStiffnessAndLoad();
    solution->setProblem(solver);
    solution->applyDGJumpTerms();
    solution->populateStiffnessAndLoad();
    Teuchos::RCP<GMGOperator> op = ((GMGSolver*)solver.get())->gmgOperator();
    Teuchos::RCP< Epetra_CrsMatrix > A = solution->getStiffnessMatrix();
    op->setFineStiffnessMatrix(A.get());
    
    Teuchos::RCP< Epetra_CrsMatrix > M;
    {
      Teuchos::RCP< Epetra_CrsMatrix > A_coarse = op->getCoarseStiffnessMatrix();
      if (rank==0) cout << "writing A_coarse to A_coarse.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_coarse.dat",*A_coarse, NULL, NULL, false);

      Teuchos::RCP< Epetra_CrsMatrix > P = op->getProlongationOperator();
      if (rank==0) cout << "writing P to P.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("P.dat",*P, NULL, NULL, false);

      if (rank==0) cout << "writing A to A_fine.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_fine.dat",*A, NULL, NULL, false);

      if (rank==0) cout << "converting smoother to matrix representation…\n";
      Teuchos::RCP< Epetra_CrsMatrix > S = op->getSmootherAsMatrix();
      if (rank==0) cout << "writing smoother to S.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("S.dat",*S, NULL, NULL, false);

      Teuchos::RCP<GMGOperator> op = ((GMGSolver*)solver.get())->gmgOperator();
      if (rank==0) cout << "converting op to matrix representation…\n";
      auto opAsMatrix = op->getMatrixRepresentation();
      if (rank==0) cout << "writing op to op.dat.\n";
        EpetraExt::RowMatrixToMatrixMarketFile("op.dat",*opAsMatrix, NULL, NULL, false);
      
      if (op->getSmootherWeightVector() != Teuchos::null)
      {
        if (rank==0) cout << "writing smoother weight vector to w_vector.dat\n";
        Teuchos::RCP<Epetra_MultiVector> w = op->getSmootherWeightVector();
        EpetraExt::MultiVectorToMatrixMarketFile("w_vector.dat", *w, NULL, NULL, false);
      }
      
      // for now, we just do this on rank 0.  For a big mesh, we might want to distribute this
      set<GlobalIndexType> myCellIndices = mesh->globalDofAssignment()->cellsInPartition(-1);
      vector<GlobalIndexTypeToCast> myCellIndicesVector(myCellIndices.begin(),myCellIndices.end());
      Epetra_MpiComm Comm(MPI_COMM_WORLD);
      
      Epetra_Map elemMap(-1, myCellIndicesVector.size(), &myCellIndicesVector[0], 0, Comm);
      Epetra_CrsMatrix E(::Copy, elemMap, 0);
      
      for (GlobalIndexType cellIndex : myCellIndices)
      {
        CellPtr cell = mesh->getTopology()->getCell(cellIndex);
        vector<CellPtr> neighbors = cell->getNeighbors(mesh->getTopology());
        neighbors.push_back(cell); // for connectivity, cell counts as its own neighbor
        
        vector<double> values(neighbors.size(),1.0);
        vector<GlobalIndexTypeToCast> neighborIndices;
        for (CellPtr neighbor : neighbors)
        {
          neighborIndices.push_back(neighbor->cellIndex());
        }
        E.InsertGlobalValues(cellIndex, values.size(), &values[0], &neighborIndices[0]);
      }
      E.FillComplete();
      EpetraExt::RowMatrixToMatrixMarketFile("E.dat",E, NULL, NULL, false);
      if (rank==0) cout << "wrote fine mesh element connectivity matrix to E.dat.\n";
      
      {
      // some debugging stuff:
//        Epetra_CrsMatrix identity(::Copy, A->RowMap(), 0);
        Epetra_CrsMatrix identity(::Copy, A->RowMap(), A->ColMap(), 0);
        int myRows = identity.RowMap().NumMyElements();
        int myCols = identity.ColMap().NumMyElements();
        Camellia::FieldContainer<GlobalIndexTypeToCast> colLIDs(myCols);
        for (int LID=0; LID<myCols; LID++)
        {
          colLIDs[LID] = LID;
        }
        for (int LID=0; LID<myRows; LID++)
        {
          Camellia::FieldContainer<double> myData(myCols);
          for (int j=0; j<myCols; j++)
          {
            if (j==LID) myData(j) = 1.0;
            else myData(j) = 0.0;
          }
          identity.InsertMyValues(LID, myCols, &myData[0], &colLIDs[0]);
//          cout << "on rank " << rank << ", inserting 1.0 at (" << myGIDs[LID] << "," << myGIDs[LID] << ")\n";
        }
        identity.FillComplete();
        
        if (rank==0) cout << "writing identity to I.dat.\n";
        EpetraExt::RowMatrixToMatrixMarketFile("I.dat",identity, NULL, NULL, false);
        op->setFineStiffnessMatrix(&identity);

        if (rank==0) cout << "writing smoother for identity to W.dat.\n";
        Teuchos::RCP< Epetra_CrsMatrix > W = op->getSmootherAsMatrix();
        EpetraExt::RowMatrixToMatrixMarketFile("W.dat",*W, NULL, NULL, false);
      }
      
      return;
    }
  }
  
  int result;
  
  if (problemChoice != NavierStokes)
  {
    result = solution->solve(solver);
  }
  else
  {
//    result = 0;
//    if (rank==0) cout << "Navier-Stokes: doing two Newton steps and recording the results for the last\n";
//    
//    navierStokesFormFine->setSolver(solver);
//    result = navierStokesFormFine->solveAndAccumulate();
//    
//    solver = initializeGMGSolver(solution, AztecOutputLevel, smootherType,
//                                 schwarzOverlap, schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio,
//                                 useStaticCondensation, hOnly, useHierarchicalNeighborsForSchwarz, k0Mesh, coarseSolverChoice,
//                                 cgMaxIterations, cgTol,
//                                 smootherWeight, useWeightMatrixForSchwarz);
    navierStokesFormFine->setSolver(solver);
    result = navierStokesFormFine->solveAndAccumulate(); // second step has non-zero background flow (implying variable coefficients)
  }

  solveTime = solveTimer.ElapsedTime();

  if (result == 0)
  {
    iterationCount = ((GMGSolver *) solver.get())->iterationCount();
  }
  else
  {
    iterationCount = -1;
  }

  if (reportTimings) solution->reportTimings();
  solution->clearComputedResiduals(); // force recomputation
  bool computeEnergyError = (problemChoice != H1NormGalerkinFormulation);
  double energyErrorTotal = computeEnergyError ? solution->energyErrorTotal() : -1;

  GMGSolver* fineSolver = dynamic_cast<GMGSolver*>(solver.get());
  if (fineSolver != NULL)
  {
    if (rank==0) cout << "************   Fine GMG Solver, timings   *************\n";
    fineSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);

//    fineSolver->gmgOperator()->reportTimings();

    GMGSolver* coarseSolver = dynamic_cast<GMGSolver*>(fineSolver->gmgOperator()->getCoarseSolver().get());
    if (coarseSolver != NULL)
    {
      if (rank==0) cout << "\n************   Coarse GMG Solver, timings   *************\n";
      coarseSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);
      vector<int> iterationCountLog = coarseSolver->getIterationCountLog();
      if (rank==0) Camellia::print("coarseSolver iteration counts:",iterationCountLog);
      double totalIterationCount = 0;
      for (int i=0; i<iterationCountLog.size(); i++)
      {
        totalIterationCount += iterationCountLog[i];
      }
      if (rank==0) cout << "Average coarse solver iteration count: " << totalIterationCount / iterationCountLog.size() << "\n\n";
    }
  }

  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if ((rank==0) && reportEnergyError)
  {
    cout << "Mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Energy error: " << energyErrorTotal << endl;
  }
  error = energyErrorTotal; // in future, might want to compute L^2 norm of error instead of energy error, in cases where we know exact solution

  if (saveVisualizationOutput)
  {
    static int exportOrdinal = 0;
    ostringstream exporterName;
    exporterName << "preconditionerSolution_" << exportOrdinal++;
    
    Teuchos::RCP<HDF5Exporter> exporter = Teuchos::rcp(new HDF5Exporter(solution->mesh(), exporterName.str(), "./") );
    exporter->exportSolution(solution);
    
  }
}

void runMany(ProblemChoice problemChoice, int spaceDim, int delta_k, int minCells,
             bool conformingTraces, bool useStaticCondensation,
             GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
             int chebyshevDegree, double chebyshevRatio,
             Solver::SolverChoice coarseSolverChoice,
             double cgTol, int cgMaxIterations, int aztecOutputLevel, RunManyPreconditionerChoices preconditionerChoices,
             int H1Order, int H1Order_coarse, bool jumpToCoarsePolyOrder, bool useNodalBasis,
             int overlapLevel, int dimensionForSchwarzNeighbors, int numCellsRootMesh, bool reportTimings,
             bool hOnly, bool useHierarchicalNeighborsForSchwarz,
             int maxCells, bool useZeroMeanConstraints, double smootherWeight,
             bool useWeightMatrixForSchwarz, bool enhanceFieldsForH1TracesWhenConforming, double graphNormBeta)
{
  int rank = Teuchos::GlobalMPISession::getRank();

  string problemChoiceString;
  switch (problemChoice)
  {
    case Poisson:
      problemChoiceString = "Poisson";
        break;
    case ConvectionDiffusion:
      problemChoiceString = "ConvectionDiffusion";
      break;
    case ConvectionDiffusionExperimental:
      problemChoiceString = "ConvectionDiffusionExperimental";
      break;
    case LinearElasticity:
      problemChoiceString = "LinearElasticity";
      break;
    case Stokes:
      problemChoiceString = "Stokes";
      break;
    case NavierStokes:
      problemChoiceString = "Navier-Stokes";
      break;
    case VlasovSpaceTime:
      problemChoiceString = "VlasovSpaceTime";
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled problem choice");
      break;
  }

  string preconditionerChoiceString;
  switch (preconditionerChoices)
  {
  case All:
    preconditionerChoiceString = "All";
    break;
  case AllGMG:
    preconditionerChoiceString = "AllGMG";
    break;
  case GMGAlgebraicSchwarz:
    preconditionerChoiceString = "GMGAlgebraicSchwarz";
    break;
  case GMGChebyshev:
    preconditionerChoiceString = "GMGChebyshev";
    break;
  case GMGGeometricSchwarz:
    preconditionerChoiceString = "GMGGeometricSchwarz";
    break;
  case GMGBlockJacobi:
    preconditionerChoiceString = "GMGBlockJacobi";
    break;
  case GMGPointJacobi:
    preconditionerChoiceString = "GMGPointJacobi";
    break;
  case GMGBlockSymmetricGaussSeidel:
    preconditionerChoiceString = "GMGBlockGaussSeidel";
    break;
  case GMGPointSymmetricGaussSeidel:
    preconditionerChoiceString = "GMGPointGaussSeidel";
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled preconditioner choice subset");
    break;
  }

  vector<GMGOperator::SmootherChoice> smootherChoices;

  switch (preconditionerChoices)
  {
  case GMGAlgebraicSchwarz:
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    break;
  case GMGGeometricSchwarz:
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    break;
  case GMGChebyshev:
    smootherChoices.push_back(GMGOperator::CHEBYSHEV);
    break;
  case GMGBlockJacobi:
    smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
    break;
  case GMGPointJacobi:
    smootherChoices.push_back(GMGOperator::POINT_JACOBI);
    break;
  case GMGBlockSymmetricGaussSeidel:
    smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
    break;
  case GMGPointSymmetricGaussSeidel:
    smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
    break;
  case All:
  case AllGMG:
    smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    smootherChoices.push_back(GMGOperator::CHEBYSHEV);
//      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
//    smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
//      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
//    smootherChoices.push_back(GMGOperator::POINT_JACOBI);
    break;
  }

  vector<int> H1OrderValues;
  if (H1Order == -1)
  {
    if (H1Order_coarse < 2)
      H1OrderValues.push_back(2);
    if (H1Order_coarse < 3)
      H1OrderValues.push_back(3);
    if (spaceDim < 3) H1OrderValues.push_back(4);
    if (spaceDim < 2) H1OrderValues.push_back(8);
    if (spaceDim < 2) H1OrderValues.push_back(16);
    if ((H1OrderValues.size() == 0) && !hOnly)
      H1OrderValues = {H1Order_coarse + 1};
    else if ((H1OrderValues.size() == 0) && hOnly)
      H1OrderValues = {H1Order_coarse};
  }
  else
  {
    H1OrderValues.push_back(H1Order);
  }

  vector<int> numCellsValues;
  int numCells = minCells;
  while (numCells <= maxCells)
  {
    // want to do as many as we can with just one cell per processor
    numCellsValues.push_back(numCells);
    numCells *= 2;
  }

  ostringstream results;
  results << "Preconditioner\tSmoother\tOverlap\tnum_cells\tmesh_width\tk\tIterations\tSolve_time\tError\n";

  vector<GMGOperator::SmootherChoice> smootherChoiceValues;
  smootherChoiceValues = smootherChoices;
  {
    for (vector<GMGOperator::SmootherChoice>::iterator smootherIt = smootherChoiceValues.begin(); smootherIt != smootherChoiceValues.end(); smootherIt++)
    {
      GMGOperator::SmootherChoice smoother = *smootherIt;

      vector<int> overlapValues;
      if ((smoother==GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) || (smoother==GMGOperator::IFPACK_ADDITIVE_SCHWARZ))
      {
        if (overlapLevel == -1)
        {
          overlapValues.push_back(0);
          overlapValues.push_back(1);
          if (spaceDim < 3) overlapValues.push_back(2);
        }
        else
        {
          overlapValues.push_back(overlapLevel);
        }
      }
      else
      {
        overlapValues.push_back(0);
      }

      // smoother choice description
      string S_str = smootherString(smoother);
      string M_str = "GMG"; // preconditioner descriptor for output
      
      for (vector<int>::iterator overlapValueIt = overlapValues.begin(); overlapValueIt != overlapValues.end(); overlapValueIt++)
      {
        int overlapValue = *overlapValueIt;
        for (vector<int>::iterator numCellsValueIt = numCellsValues.begin(); numCellsValueIt != numCellsValues.end(); numCellsValueIt++)
        {
          int numCells1D = *numCellsValueIt;
          for (auto H1Order : H1OrderValues)
          {
            int iterationCount;
            bool reportEnergyError = false;
            double solveTime, error;
            bool writeAndExit = false; // not supported for runMany (since it always writes to the same disk location)
            run(problemChoice, iterationCount, spaceDim, numCells1D, H1Order, delta_k,
                H1Order_coarse, jumpToCoarsePolyOrder, useNodalBasis, conformingTraces,
                useStaticCondensation, smoother,
                overlapValue, dimensionForSchwarzNeighbors,
                schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio,
                chebyshevDegree, chebyshevRatio,
                coarseSolverChoice,
                cgTol, cgMaxIterations, aztecOutputLevel, reportTimings, solveTime, error,
                reportEnergyError, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz, useZeroMeanConstraints, writeAndExit,
                smootherWeight, useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming, graphNormBeta);

            int numCells = pow((double)numCells1D, spaceDim);

            ostringstream thisResult;

            thisResult << M_str << "\t" << S_str << "\t" << overlapValue << "\t" << numCells << "\t";
            thisResult << numCells1D << "\t" << H1Order << "\t" << iterationCount << "\t" << solveTime;
            thisResult << "\t" << error << endl;

            if (rank==0) cout << thisResult.str();

            results << thisResult.str();
          }
        }
      }
    }
    if (rank==0) cout << results.str(); // output results so far
  }

  if (rank == 0)
  {
    ostringstream filename;
    filename << problemChoiceString << "Driver" << spaceDim << "D_";
    if ((preconditionerChoiceString == "GMGGeometricSchwarz") && !useWeightMatrixForSchwarz)
    {
      filename << "GMGGeometricSchwarzUnweighted";
    }
    else if ((preconditionerChoiceString == "GMGGeometricSchwarz") && useWeightMatrixForSchwarz)
    {
      filename << "GMGGeometricSchwarzWEIGHTED";
    }
    else
    {
      filename << preconditionerChoiceString;
    }
    if (schwarzBlockFactorization != GMGOperator::Direct)
      filename << "_schwarzFactorization_" << getFactorizationTypeString(schwarzBlockFactorization);
    if (overlapLevel != -1)
    {
      filename << "_overlap" << overlapLevel;
    }
    if (dimensionForSchwarzNeighbors != -1)
    {
      filename << "_schwarzNeighborDim" << dimensionForSchwarzNeighbors;
    }
    if (H1Order != -1)
    {
      filename << "_p" << H1Order;
    }
    filename << "_deltak" << delta_k;
    if (H1Order_coarse != 1)
      filename << "_pCoarse" << H1Order_coarse;

    if (hOnly)
    {
      filename << "_hOnly";
      if (useHierarchicalNeighborsForSchwarz)
        filename << "TruncatedNeighbors";
      else
        filename << "StandardNeighbors";
    }
    
    // if coarse solver is not direct, then include in the file name:
    if ((coarseSolverChoice != Solver::KLU) && (coarseSolverChoice != Solver::MUMPS) && (coarseSolverChoice != Solver::SuperLUDist))
      filename << "_coarseSolver_" << Solver::solverChoiceString(coarseSolverChoice);
    if (useStaticCondensation)
      filename << "_withStaticCondensation";
    if (conformingTraces)
      filename << "_conformingTraces";
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
      filename << "_enhancedFields";
    if (graphNormBeta != 1.0)
      filename << "_graphNormBeta" << graphNormBeta;
    if (cgTol != 1e-10)
      filename << "_cgTol" << cgTol;
    if (! useNodalBasis)
      filename << "_hierarchicalBasis";
    else
      filename << "_nodalBasis";
    filename << "_results.dat";
    ofstream fout(filename.str().c_str());
    fout << results.str();
    fout.close();
    cout << "Wrote results to " << filename.str() << ".\n";
  }
}

int main(int argc, char *argv[])
{
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  
  {
    int rank = Teuchos::GlobalMPISession::getRank();
    
    runGMGOperatorInDebugMode = false;
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
    
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    int H1Order = -1;        // poly order for H^1 variables (-1 for a range of values)
    int H1Order_coarse = -1; // poly order for H^1 variables on the coarse mesh
    int delta_k = -1;   // test space enrichment; -1 for default detection (defaults to spaceDim)
    bool jumpToCoarsePolyOrder = false;
    
    bool conformingTraces = false;
    
    int numCells = 2;
    int numCellsRootMesh = -1;
    
    maxDofsForKLU = 2000; // used when defining coarsest solve on 3-level solver -- will use SuperLUDist if not KLU
    coarseCGTol = 1e-10;
    coarseMaxIterations = 2000;
    
    int AztecOutputLevel = 1;
    int cgMaxIterations = 25000;
    int schwarzOverlap = -1;
    
    double smootherWeight = -1.0;
    
    double chebyshevRatio = 30.0;
    int chebyshevDegree = 15;
    
    int spaceDim = 1;
    
    bool useCondensedSolve = false;
    
    bool useWeightMatrixForSchwarz = false;
    
    string smootherChoiceStr = "Camellia-Schwarz";
        
    double cgTol = 1e-10;
    
    double fillRatio = 5;
    int levelOfFill = 2;
    
    bool runAutomatic = false;
    
    bool reportTimings = false;
    
    int dimensionForSchwarzNeighbors = -1;
    
    bool enhanceFieldsForH1TracesWhenConforming = false; // new 10-27-15
    
    bool hOnly = false;
    
    bool useHierarchicalNeighborsForSchwarz = false; // new 12-22-15
    
    narrateSolution = false;
    narrateCoarseSolution = false;
    
    bool useZeroMeanConstraints = false;
    
    bool writeAndExit = false;
    
    string schwarzFactorizationTypeString = "Direct";
    
    string problemChoiceString = "Poisson";
    
    string coarseSolverChoiceString = "GMG";
    
    string runManySubsetString = "All";
    
    double graphNormBeta = 1.0;
    
    int runManyMinCells = 2;
    int maxCells = -1;
        
    cmdp.setOption("problem",&problemChoiceString,"problem choice: Poisson, ConvectionDiffusion, LinearElasticity, Stokes, Navier-Stokes");
    
    cmdp.setOption("p",&H1Order,"polynomial order for field variable u");
    cmdp.setOption("dp", &delta_k, "test space polynomial order enrichment");
    cmdp.setOption("useNodalBasis", "useHierarchicalBasis", &useNodalBasis);
    
    cmdp.setOption("coarseSolver", &coarseSolverChoiceString, "coarse solver choice: KLU, MUMPS, SuperLUDist, SimpleML");
    cmdp.setOption("p_coarse", &H1Order_coarse, "polynomial order for H^1 variables on coarse grid");
    cmdp.setOption("jumpToCoarsePolyOrder", "coarsenPolyOrderByHalves", &jumpToCoarsePolyOrder, "if false, p-coarsen by halves until coarsePolyOrder is reached; if true, use at most two p levels");
    
    cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
    cmdp.setOption("dimensionForSchwarzNeighbors", &dimensionForSchwarzNeighbors, "dimension to use for Schwarz neighbor relationship (only supported for geometric Schwarz; default of -1 means to use spaceDim-1 (face neighbors).");
    cmdp.setOption("graphNormBeta", &graphNormBeta);
    
    cmdp.setOption("smoother", &smootherChoiceStr);
    
    cmdp.setOption("hOnly", "notHOnly", &hOnly);
    cmdp.setOption("useHierarchicalNeighborsForSchwarz", "useNaturalNeighborsForSchwarz", &useHierarchicalNeighborsForSchwarz);
    
    cmdp.setOption("enhanceFieldsForH1TracesWhenConforming", "equalOrderFieldsForH1TracesWhenConforming", &enhanceFieldsForH1TracesWhenConforming);
    
    cmdp.setOption("chebyshevRatio", &chebyshevRatio, "Chebyshev: ratio between max/min eigenvalues used for weight determination");
    cmdp.setOption("chebyshevDegree", &chebyshevDegree, "Chebyshev: polynomial degree of smoother");
    cmdp.setOption("schwarzLevelOfFill", &levelOfFill, "Schwarz block factorization: level of fill for ILU");
    
    cmdp.setOption("schwarzFactorization", &schwarzFactorizationTypeString, "Schwarz block factorization strategy: Direct, IC, ILU");
    cmdp.setOption("schwarzFillRatio", &fillRatio, "Schwarz block factorization: fill ratio for IC");
    cmdp.setOption("schwarzLevelOfFill", &levelOfFill, "Schwarz block factorization: level of fill for ILU");
    cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
    cmdp.setOption("maxDofsForKLU",&maxDofsForKLU, "for multi-level solves, maximum number of dofs to use with KLU solve at coarsest level");
    cmdp.setOption("narrateSolution", "dontNarrateSolution", &narrateSolution);
    cmdp.setOption("narrateCoarseSolution", "dontNarrateCoarseSolution", &narrateCoarseSolution);
    
    cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");
    
    cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");
    
    cmdp.setOption("smootherWeight", &smootherWeight, "smoother weight for GMGOperator.");
    
    cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
    cmdp.setOption("numCells", &numCells, "number of cells in the mesh");
    cmdp.setOption("numCellsRootMesh", &numCellsRootMesh, "number of cells in the root mesh");
    
    cmdp.setOption("maxIterations", &cgMaxIterations, "maximum number of CG iterations");
    cmdp.setOption("cgTol", &cgTol, "CG convergence tolerance");
    
    cmdp.setOption("coarseCGTol", &coarseCGTol, "coarse solve CG tolerance");
    cmdp.setOption("coarseMaxIterations", &coarseMaxIterations, "coarse solve max iterations");
    
    cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");
    
    cmdp.setOption("runMany", "runOne", &runAutomatic, "Run in automatic mode (ignores several input parameters)");
    cmdp.setOption("runManySubset", &runManySubsetString, "DontPrecondition, AllGMG, AllSchwarz, or All");
    cmdp.setOption("runManyMinCells", &runManyMinCells, "Minimum number of cells to use for mesh width");
    cmdp.setOption("runManyMaxCells", &maxCells, "Maximum number of cells to use for mesh width");
    
    cmdp.setOption("writeAndExit", "runNormally", &writeAndExit, "Write A, A_coarse, P, and S to disk, and exit without computing anything.");
    
    cmdp.setOption("useWeightedSchwarz","useUnweightedSchwarz",&useWeightMatrixForSchwarz, "Use weight matrix ('W' in Fischer and Lottes) to scale Schwarz smoother according to multiplicities.  Only applies to GMG geometric Schwarz.");
    
    cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");
    
    cmdp.setOption("gmgOperatorDebug", "gmgOperatorNormal", &runGMGOperatorInDebugMode, "Run GMGOperator in a debug mode");
    
    cmdp.setOption("saveVisualizationOutput", "dontSaveVisualizationOutput", &saveVisualizationOutput);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    ProblemChoice problemChoice;
    
    if (problemChoiceString == "Poisson")
    {
      problemChoice = Poisson;
    }
    else if (problemChoiceString == "ConvectionDiffusion")
    {
      problemChoice = ConvectionDiffusion;
    }
    else if (problemChoiceString == "ConvectionDiffusionExperimental")
    {
      problemChoice = ConvectionDiffusionExperimental;
    }
    else if (problemChoiceString == "LinearElasticity")
    {
      problemChoice = LinearElasticity;
    }
    else if (problemChoiceString == "Stokes")
    {
      problemChoice = Stokes;
    }
    else if (problemChoiceString == "Navier-Stokes")
    {
      problemChoice = NavierStokes;
    }
    else if (problemChoiceString == "H1NormGalerkin")
    {
      problemChoice = H1NormGalerkinFormulation;
    }
    else if (problemChoiceString == "VlasovSpaceTime")
    {
      problemChoice = VlasovSpaceTime;
    }
    else
    {
      if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    GMGOperator::SmootherChoice smootherChoice;
    
    if (smootherChoiceStr == "None")
    {
      smootherChoice = GMGOperator::NONE;
    }
    else if (smootherChoiceStr == "IfPack-Schwarz")
    {
      smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    }
    else if (smootherChoiceStr == "Camellia-Schwarz")
    {
      smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
    }
    else if (smootherChoiceStr == "Point-Jacobi")
    {
      smootherChoice = GMGOperator::POINT_JACOBI;
    }
    else if (smootherChoiceStr == "Block-Jacobi")
    {
      smootherChoice = GMGOperator::BLOCK_JACOBI;
    }
    else if (smootherChoiceStr == "Point-Gauss-Seidel")
    {
      smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    }
    else if (smootherChoiceStr == "Block-Gauss-Seidel")
    {
      smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL;
    }
    else if (smootherChoiceStr == "Chebyshev")
    {
      smootherChoice = GMGOperator::CHEBYSHEV;
    }
    else
    {
      if (rank==0) cout << "Smoother choice string not recognized.\n";
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    Solver::SolverChoice coarseSolverChoice = Solver::solverChoiceFromString(coarseSolverChoiceString);
    
    RunManyPreconditionerChoices runManySubsetChoice;
    
    if (runManySubsetString == "All")
    {
      runManySubsetChoice = All;
    }
    else if (runManySubsetString == "AllGMG")
    {
      runManySubsetChoice = AllGMG;
    }
    else if (runManySubsetString == "GMGAlgebraicSchwarz")
    {
      runManySubsetChoice = GMGAlgebraicSchwarz;
    }
    else if (runManySubsetString == "GMGGeometricSchwarz")
    {
      runManySubsetChoice = GMGGeometricSchwarz;
    }
    else if (runManySubsetString == "GMGPointGaussSeidel")
    {
      runManySubsetChoice = GMGPointSymmetricGaussSeidel;
    }
    else if (runManySubsetString == "GMGBlockGaussSeidel")
    {
      runManySubsetChoice = GMGBlockSymmetricGaussSeidel;
    }
    else if (runManySubsetString == "GMGPointJacobi")
    {
      runManySubsetChoice = GMGPointJacobi;
    }
    else if (runManySubsetString == "GMGBlockJacobi")
    {
      runManySubsetChoice = GMGBlockJacobi;
    }
    else if (runManySubsetString == "GMGChebyshev")
    {
      runManySubsetChoice = GMGChebyshev;
    }
    else
    {
      if (rank==0) cout << "Run many subset string not recognized.\n";
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    GMGOperator::FactorType schwarzFactorType = getFactorizationType(schwarzFactorizationTypeString);
    
    if (delta_k==-1)
    {
      if (problemChoice != H1NormGalerkinFormulation)
      {
        delta_k = spaceDim;
      }
      else
      {
        delta_k = 0;
      }
    }
    
    if (! runAutomatic)
    {
      int iterationCount;
      bool reportEnergyError = true;
      
      if (schwarzOverlap == -1) schwarzOverlap = 0;
      if (H1Order == -1) H1Order = 3;
      
      double solveTime, error;
      
      run(problemChoice, iterationCount, spaceDim, numCells, H1Order, delta_k, H1Order_coarse, jumpToCoarsePolyOrder,
          useNodalBasis, conformingTraces,
          useCondensedSolve, smootherChoice, schwarzOverlap, dimensionForSchwarzNeighbors,
          schwarzFactorType, levelOfFill, fillRatio,
          chebyshevDegree, chebyshevRatio,
          coarseSolverChoice,
          cgTol, cgMaxIterations, AztecOutputLevel, reportTimings, solveTime, error,
          reportEnergyError, numCellsRootMesh, hOnly, useHierarchicalNeighborsForSchwarz,
          useZeroMeanConstraints, writeAndExit, smootherWeight,
          useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming, graphNormBeta);
      
      if (rank==0) cout << "Iteration count: " << iterationCount << "; solve time " << solveTime << " seconds." << endl;
    }
    else
    {
      if (maxCells == -1)
      {
        // by default, ensure max of 1 cell per MPI node
        int nProc = Teuchos::GlobalMPISession::getNProc();
        maxCells = 1;
        while (maxCells*2 <= nProc)
        {
          maxCells *= 2;
        }
      }
      if (problemChoice == Stokes)
      {
        maxCells = max(maxCells,2);
        runManyMinCells = max(runManyMinCells,2);
      }
      
      if (rank==0)
      {
        cout << "Running " << problemChoiceString << " solver in automatic mode (subset: ";
        cout << runManySubsetString << "), with spaceDim " << spaceDim;
        cout << ", delta_k = " << delta_k << ", ";
        if (conformingTraces)
          cout << "conforming traces, ";
        else
          cout << "non-conforming traces, ";
        cout << "CG tolerance = " << cgTol << ", max iterations = " << cgMaxIterations << endl;
      }
      
      runMany(problemChoice, spaceDim, delta_k, runManyMinCells,
              conformingTraces, useCondensedSolve,
              schwarzFactorType, levelOfFill, fillRatio,
              chebyshevDegree, chebyshevRatio,
              coarseSolverChoice,
              cgTol, cgMaxIterations, AztecOutputLevel,
              runManySubsetChoice, H1Order, H1Order_coarse, jumpToCoarsePolyOrder, useNodalBasis,
              schwarzOverlap, dimensionForSchwarzNeighbors, numCellsRootMesh, reportTimings,
              hOnly, useHierarchicalNeighborsForSchwarz, maxCells,
              useZeroMeanConstraints, smootherWeight, useWeightMatrixForSchwarz, enhanceFieldsForH1TracesWhenConforming,
              graphNormBeta);
    }
  }
  Camellia::finalize(false);
  return 0;
}
