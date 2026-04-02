//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SqrtFunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/25.
//
//
#include "Teuchos_UnitTestHarness.hpp"

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

using namespace Camellia;

namespace
{
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
                                              int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization,
                                              int chebyshevDegree, double chebyshevRatio,
                                              bool useStaticCondensation, bool hOnly, bool useHierarchicalNeighborsForSchwarz,
                                              int H1Order_coarse, bool jumpToCoarsePolyOrder,
                                              Solver::SolverChoice coarseSolverChoice, int cgMaxIterations, double cgTol,
                                              double smootherWeight)
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
          
          FunctionPtr u_exact = (spaceDim == 2) ? Function::vectorize(u1_exact, u2_exact) : Function::vectorize(u1_exact, u2_exact, u3_exact);
          FunctionPtr f = formulation.forcingFunction(u_exact);
          
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
  }

  void run(Teuchos::FancyOStream &out,
           ProblemChoice problemChoice, int &iterationCount, int spaceDim, int numCells,
           int H1Order, int delta_k, int H1Order_coarse, bool jumpToCoarsePolyOrder, bool useNodalBasis,
           bool conformingTraces, bool useStaticCondensation,
           GMGOperator::SmootherChoice smootherType,
           int schwarzOverlap, int dimensionForSchwarzNeighbors, GMGOperator::FactorType schwarzBlockFactorization,
           int chebyshevDegree, double chebyshevRatio,
           Solver::SolverChoice coarseSolverChoice, double cgTol, int cgMaxIterations, int AztecOutputLevel,
           bool reportTimings, double &solveTime, double &error, bool reportEnergyError, int numCellsRootMesh,
           bool hOnly, bool useHierarchicalNeighborsForSchwarz, bool useZeroMeanConstraints,
           double smootherWeight,
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
      out << "Too few cells in root mesh.  Aborting.\n";
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
      out << "fine mesh has " << fineElements << " active elements and " << fineDofs << " degrees of freedom.\n";
      out << "coarse mesh has " << coarseElements << " active elements and " << coarseDofs << " degrees of freedom.\n";
    }

    double initializationTime = initializationTimer.ElapsedTime();
    int numCoarseGlobalDofs = k0Mesh->numGlobalDofs();
    int numFineGlobalDofs = solution->mesh()->numGlobalDofs();
    
    // to be fairer to preconditioners whose construction is cheap, we include construction time in the solve time (this is a change, made 7-7-15)
    Epetra_Time solveTimer(Comm);
    
    Teuchos::RCP<Solver> solver = initializeGMGSolver(solution, AztecOutputLevel, smootherType,
                                                      schwarzOverlap, dimensionForSchwarzNeighbors,
                                                      schwarzBlockFactorization,
                                                      chebyshevDegree, chebyshevRatio,
                                                      useStaticCondensation, hOnly, useHierarchicalNeighborsForSchwarz,
                                                      H1Order_coarse, jumpToCoarsePolyOrder, coarseSolverChoice,
                                                      cgMaxIterations, cgTol,
                                                      smootherWeight);
    
    int result;
    
    if (problemChoice != NavierStokes)
    {
      result = solution->solve(solver);
    }
    else
    {
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

    if (reportTimings)
    {
      GMGSolver* fineSolver = dynamic_cast<GMGSolver*>(solver.get());
      if (fineSolver != NULL)
      {
        if (rank==0) out << "************   Fine GMG Solver, timings   *************\n";
        fineSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);
        
        //    fineSolver->gmgOperator()->reportTimings();
        
        GMGSolver* coarseSolver = dynamic_cast<GMGSolver*>(fineSolver->gmgOperator()->getCoarseSolver().get());
        if (coarseSolver != NULL)
        {
          if (rank==0) out << "\n************   Coarse GMG Solver, timings   *************\n";
          coarseSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);
          vector<int> iterationCountLog = coarseSolver->getIterationCountLog();
          if (rank==0) Camellia::print("coarseSolver iteration counts:",iterationCountLog);
          double totalIterationCount = 0;
          for (int i=0; i<iterationCountLog.size(); i++)
          {
            totalIterationCount += iterationCountLog[i];
          }
          if (rank==0) out << "Average coarse solver iteration count: " << totalIterationCount / iterationCountLog.size() << "\n\n";
        }
      }
    }

    GlobalIndexType numFluxDofs = mesh->numFluxDofs();
    GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
    if ((rank==0) && reportEnergyError)
    {
      out << "Mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
      out << numGlobalDofs << " total dofs, including fields).\n";
      out << "Energy error: " << energyErrorTotal << endl;
    }
    error = energyErrorTotal; // in future, might want to compute L^2 norm of error instead of energy error, in cases where we know exact solution
  }

void runTestCases(const ProblemChoice problem, const GMGOperator::SmootherChoice smoother,
                  std::vector< std::tuple<int,int,int,int> > &testCases,
                  bool &success, Teuchos::FancyOStream &out) // test cases are 4-tuples: (p, overlap, chebyshevDegree, iterationCount)
{
  const int spaceDim = 3;
  const int numCells = 3;
  const int numCellsRootMesh = numCells; // p-multigrid
  
  const bool useNodalBasis = false; // use hierarchical instead (better preconditioning)
  
  const int delta_k = (problem != H1NormGalerkinFormulation) ? 1 :0;
  const int H1Order_coarse = 1;
  const bool jumpToCoarsePolyOrder = false;
  const bool useStaticCondensation = false;
  const bool conformingTraces = false;
  const int dimensionForSchwarzNeighbors = -1;
  const bool enhanceFieldsForH1TracesWhenConforming = false; // new 10-27-15
  const bool hOnly = false;
  const bool useHierarchicalNeighborsForSchwarz = false; // new 12-22-15
  const bool useZeroMeanConstraints = false;
  const GMGOperator::FactorType smootherFactorization = Camellia::GMGOperator::Direct;
  const Solver::SolverChoice coarseSolverChoice = Camellia::TSolver<double>::KLU;
  const bool reportEnergyError = false;
  const bool reportTimings = false;
  const double graphNormBeta = 1.0;
  
  const double cgTol = 1e-10;
  const int cgMaxIters = 400;
  const int outputLevel = 0;
  
  const double smootherWeight = -1.0;
  const double chebyshevRatio = 30.0;

  int iterationTol = 5;            // we tolerate a difference of up to 5 in iteration count
  double iterationFractionTol = 0.10; // we tolerate a difference of up to 10% in iteration count
  
  int iterationCount; // result
  double solveTime;
  double error;
  
  for (const auto testCase : testCases)
  {
    int H1Order = std::get<0>(testCase);
    int overlap = std::get<1>(testCase);
    int chebyshevDegree = std::get<2>(testCase);
    int expectedIterations = std::get<3>(testCase);
    
    out << "p=" << H1Order;
    if (overlap >= 0)
    {
      out << ", overlap " << overlap;
    }
    if (chebyshevDegree >= 0)
    {
      out << ", Chebyshev degree " << chebyshevDegree;
    }
    
    out << std::endl;
    
    run(out, problem, iterationCount, spaceDim, numCells,
        H1Order, delta_k, H1Order_coarse, jumpToCoarsePolyOrder, useNodalBasis,
        conformingTraces, useStaticCondensation,
        smoother,
        overlap, dimensionForSchwarzNeighbors, smootherFactorization,
        chebyshevDegree, chebyshevRatio,
        coarseSolverChoice, cgTol, cgMaxIters, outputLevel,
        reportTimings, solveTime, error, reportEnergyError, numCellsRootMesh,
        hOnly, useHierarchicalNeighborsForSchwarz, useZeroMeanConstraints,
        smootherWeight,
        enhanceFieldsForH1TracesWhenConforming, graphNormBeta);
    
    out << "expected iteration count: " << expectedIterations << std::endl;
    out << "actual iteration count:   " << iterationCount << std::endl;
    
    const int minIterations = std::min( expectedIterations - iterationTol, int(expectedIterations * (1. - iterationFractionTol)));
    const int maxIterations = std::max( expectedIterations + iterationTol, int(expectedIterations * (1. + iterationFractionTol)));
    
    TEST_COMPARE(iterationCount, >, minIterations);
    TEST_COMPARE(iterationCount, <, maxIterations);
  }
}

  TEUCHOS_UNIT_TEST( Preconditioning, H1NormGalerkinAdditiveSchwarz )
  {
    const ProblemChoice problem = H1NormGalerkinFormulation;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
    
    // test cases are 4-tuples: (p, overlap, chebyshevDegree, iterationCount)
    const int c = -1; // unused chebyshevDegree
    std::vector< std::tuple<int,int,int,int> > testCases {{2,0,c,33},{2,1,c,21},{3,0,c,38},{3,1,c,21}};
    runTestCases(problem, smoother, testCases, success, out);
  }

  TEUCHOS_UNIT_TEST( Preconditioning, H1NormGalerkinChebyshev )
  {
    const ProblemChoice problem = H1NormGalerkinFormulation;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CHEBYSHEV;
    
    // test cases are 4-tuples: (p, overlap, chebyshev degree, iterationCount)
    const int o = -1; // unused overlap
    std::vector< std::tuple<int,int,int,int> > testCases {{2,o,8,15},{2,o,16,10}}; // for p=3, seeing some CG non-positivity failures for e.g. 16 MPI ranks, so not testing those cases.  Belos's CG implementation is too intolerant!!
    runTestCases(problem, smoother, testCases, success, out);
  }

  TEUCHOS_UNIT_TEST( Preconditioning, PoissonAdditiveSchwarz )
  {
    const ProblemChoice problem = Poisson;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
    
    // test cases are 4-tuples: (p, overlap, chebyshevDegree, iterationCount)
    const int c = -1; // unused chebyshevDegree
    std::vector< std::tuple<int,int,int,int> > testCases {{2,0,c,16},{2,1,c,12},{3,0,c,20},{3,1,c,15}};
    runTestCases(problem, smoother, testCases, success, out);
  }

  TEUCHOS_UNIT_TEST( Preconditioning, PoissonChebyshev )
  {
    const ProblemChoice problem = Poisson;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CHEBYSHEV;
    
    // test cases are 4-tuples: (p, overlap, chebyshev degree, iterationCount)
    const int o = -1; // unused overlap
    std::vector< std::tuple<int,int,int,int> > testCases {{2,o,8,7},{2,o,16,5},{3,o,8,6},{3,o,16,5}};
    runTestCases(problem, smoother, testCases, success, out);
  }

  TEUCHOS_UNIT_TEST( Preconditioning, VlasovSpaceTimeAdditiveSchwarz )
  {
    const ProblemChoice problem = VlasovSpaceTime;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
    
    // test cases are 4-tuples: (p, overlap, chebyshevDegree, iterationCount)
    const int c = -1; // unused chebyshevDegree
    std::vector< std::tuple<int,int,int,int> > testCases {{2,0,c,178},{2,1,c,34},{3,0,c,253},{3,1,c,38}};
    runTestCases(problem, smoother, testCases, success, out);
  }

  TEUCHOS_UNIT_TEST( Preconditioning, VlasovSpaceTimeChebyshev )
  {
    const ProblemChoice problem = VlasovSpaceTime;
    const GMGOperator::SmootherChoice smoother = Camellia::GMGOperator::CHEBYSHEV;
    
    // test cases are 4-tuples: (p, overlap, chebyshev degree, iterationCount)
    const int o = -1; // unused overlap
    std::vector< std::tuple<int,int,int,int> > testCases {{2,o,8,117},{2,o,16,73},{3,o,8,150},{3,o,16,108}};
    runTestCases(problem, smoother, testCases, success, out);
  }
} // namespace
