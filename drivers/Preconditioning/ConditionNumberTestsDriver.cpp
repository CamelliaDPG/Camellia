//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CamelliaDebugUtility.h"
#include "CondensedDofInterpreter.h"
#include "ConvectionDiffusionFormulation.h"
#include "E_to_T.hpp"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "GMGOperator.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "LinearElasticityFormulation.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "SerialDenseWrapper.h"
#include "Solver.h"
#include "StokesVGPFormulation.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_IC.h"
#include "Ifpack_ILU.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;


enum ProblemChoice
{
  Poisson,
  ConvectionDiffusion,
  ConvectionDiffusionExperimental,
  ConvectionDiffusionWeightedVariables,
  Stokes,
  NavierStokes,
  LinearElasticity
};

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, IPPtr &graphNorm,
                                     double graphNormBeta, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces, bool useStaticCondensation,
                                     int meshWidth, int k, int delta_k, bool useZeroMeanConstraints,
                                     Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulation)
{
  BFPtr bf;
  BCPtr bc;
  RHSPtr rhs;
  MeshPtr mesh;
  
  double width = 1.0; // in each dimension
  vector<double> x0(spaceDim,0); // origin is the default

  VarPtr p; // pressure

  map<int,int> trialOrderEnhancements;
  FunctionPtr exactSolution;
  VarPtr varExact; // the variable of which we have the exact solution in exactSolution

  if (problemChoice == Poisson)
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
  }
  else if (problemChoice == ConvectionDiffusion)
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
  }
  else if (problemChoice == ConvectionDiffusionExperimental)
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
  }
  else if (problemChoice == Stokes)
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
  }
  else if (problemChoice == LinearElasticity)
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
    
    VarPtr v1 = formulation.v(1);
    VarPtr v2 = formulation.v(2);
    
    VarPtr v3;
    if (spaceDim==3) v3 = formulation.v(3);
    
    rhs = formulation.rhs(f);
  }
  else if (problemChoice == NavierStokes)
  {
    if (spaceDim != 2)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // we can add support for a 3D exact solution later...
    }
    
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({2.0,2.0}, {meshWidth, meshWidth}, {-0.5,0.0});
    
    // set up classical Kovasznay flow solution:
    double Re = 40.0;
    navierStokesFormulation = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k, delta_k)));

    int k_low_order = 1;
    NavierStokesVGPFormulation lowestOrderForm(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k_low_order, delta_k));
    
    FunctionPtr u1, u2, p;
    NavierStokesVGPFormulation::getKovasznaySolution(Re, u1, u2, p);
    
    mesh = navierStokesFormulation->solutionIncrement()->mesh();
    
    if (useZeroMeanConstraints)
    {
      navierStokesFormulation->addZeroMeanPressureCondition();
      lowestOrderForm.addZeroMeanPressureCondition();
      double p_mean = p->integrate(mesh);
      p = p - p_mean;
    }
    else
    {
      navierStokesFormulation->addPointPressureCondition({0.5,1.0});
      lowestOrderForm.addPointPressureCondition({0.5,1.0});
      double p_center = p->evaluate(0.5, 1.0);
      p = p - p_center;
    }
    
    FunctionPtr u = Function::vectorize({u1, u2});
    FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunctionSteady(spaceDim, Re, u, p);
    
    int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.

    navierStokesFormulation->addInflowCondition(SpatialFilter::allSpace(), u);
    navierStokesFormulation->setForcingFunction(forcingFunction);
    
    lowestOrderForm.addInflowCondition(SpatialFilter::allSpace(), u);
    lowestOrderForm.setForcingFunction(forcingFunction);

    lowestOrderForm.solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    
//    if (rank == 0) cout << "Navier-Stokes: taking three Newton steps on first-order mesh; using this as background flow.\n";
    for (int i=0; i<3; i++)
    {
      lowestOrderForm.solveAndAccumulate();
    }
    lowestOrderForm.solution()->projectFieldVariablesOntoOtherSolution(navierStokesFormulation->solution());
    
    navierStokesFormulation->solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    navierStokesFormulation->solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    
    graphNorm = navierStokesFormulation->bf()->graphNorm();
    
    solution = navierStokesFormulation->solutionIncrement();
    solution->setUseCondensedSolve(useStaticCondensation);
    
    return; // return early for Navier-Stokes; we've already set up the Solution objects, etc.
  }

  int H1Order = k + 1;

  vector<double> dimensions;
  vector<int> elementCounts;
  
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(meshWidth);
  }
  mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0, trialOrderEnhancements);
  
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
  
  if (graphNorm == Teuchos::null) // if set previously, honor that...
    graphNorm = bf->graphNorm();

  solution = Solution::solution(mesh, bc, rhs, graphNorm);
  solution->setUseCondensedSolve(useStaticCondensation);
}

double conditionNumberLAPACK(TMatrixPtr<double> stiffnessMatrix)
{
  Camellia::FieldContainer<double> A;
  SerialDenseWrapper::extractFCFromTMatrix(stiffnessMatrix, A);
  
  bool use2norm = false;
  
  if (use2norm)
  {
    Camellia::FieldContainer<double> lambda_real(A.dimension(0)), lambda_imag(A.dimension(0));
    SerialDenseWrapper::eigenvalues(A, lambda_real, lambda_imag);
    double max_eig = 0.0, min_eig = 1e300;
    for (int i=0; i<lambda_real.size(); i++)
    {
      double real_part = lambda_real(i), imag_part = lambda_imag(i);
      double eig_mag = sqrt(real_part * real_part + imag_part * imag_part);
      max_eig = max(eig_mag, max_eig);
      min_eig = min(eig_mag, min_eig);
    }
    //  cout << "A:\n" << A;
    //  cout << "max_eig: " << max_eig << endl;
    //  cout << "min_eig: " << min_eig << endl;
    return max_eig / min_eig;
  }
  else // 1-norm
  {
    return SerialDenseWrapper::condest(A);
  }
}

void run(ProblemChoice problemChoice, int spaceDim, int numCells, int k, int delta_k, bool conformingTraces,
         bool useStaticCondensation, bool useZeroMeanConstraints, double graphNormBeta, double &condNumber)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time initializationTimer(Comm);

  SolutionPtr solution;
  IPPtr graphNorm;
  Teuchos::RCP<NavierStokesVGPFormulation> navierStokesForm; // NULL if not doing Navier-Stokes
  
  initializeSolutionAndCoarseMesh(solution, graphNorm, graphNormBeta, problemChoice, spaceDim, conformingTraces,
                                  useStaticCondensation, numCells, k, delta_k, useZeroMeanConstraints, navierStokesForm);
  
  MeshPtr mesh = solution->mesh();
  BCPtr bc = solution->bc();

//  int numElements = mesh->numActiveElements();
//  int fineDofs = mesh->numGlobalDofs();
//  if (rank == 0)
//  {
//    cout << "mesh has " << numElements << " active elements and " << fineDofs << " degrees of freedom.\n";
//  }
//  
  solution->initializeLHSVector();
  solution->initializeStiffnessAndLoad();
  solution->populateStiffnessAndLoad();
  
  auto eStiffness = solution->getStiffnessMatrix();
  auto tStiffness = E_to_T::matrix(eStiffness);
  condNumber = conditionNumberLAPACK(tStiffness);
}

void runMany(ProblemChoice problemChoice, int spaceDim, int k,
             int delta_k_min, int delta_k_max,
             int minCells, int maxCells,
             bool conformingTraces, bool useStaticCondensation,
             bool useZeroMeanConstraints, double graphNormBeta)
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
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled problem choice");
    break;
  }
  
  vector<int> kValues;
  if (k == -1)
  {
    kValues.push_back(1);
    kValues.push_back(2);
    if (spaceDim < 3) kValues.push_back(4);
    if (spaceDim < 2) kValues.push_back(8);
    if (spaceDim < 2) kValues.push_back(16);
  }
  else
  {
    kValues.push_back(k);
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
  results << "mesh_width\tk\tdelta_k\tcond_num\n";

  for (vector<int>::iterator numCellsValueIt = numCellsValues.begin(); numCellsValueIt != numCellsValues.end(); numCellsValueIt++)
  {
    int numCells1D = *numCellsValueIt;
    for (int k : kValues)
    {
      for (int delta_k = delta_k_min; delta_k <= delta_k_max; delta_k++)
      {
        double condNum;

        run(problemChoice, spaceDim, numCells1D, k, delta_k, conformingTraces,
            useStaticCondensation, useZeroMeanConstraints, graphNormBeta, condNum);
        
        int numCells = pow((double)numCells1D, spaceDim);
        
        ostringstream thisResult;
        
        thisResult << numCells1D << "\t" << k << "\t" << delta_k << "\t" << condNum << endl;
        
        if (rank==0) cout << thisResult.str();
        results << thisResult.str();
      }
    }
  }
  
  if (rank == 0)
  {
    ostringstream filename;
    filename << problemChoiceString << "CondNums" << spaceDim << "D";
    if (k != -1)
    {
      filename << "_k" << k;
    }
    filename << "_deltakmin" << delta_k_min;
    filename << "_deltakmax" << delta_k_max;
    
    // if coarse solver is not direct, then include in the file name:
    if (useStaticCondensation)
      filename << "_withStaticCondensation";
    if (conformingTraces)
      filename << "_conformingTraces";
    if (graphNormBeta != 1.0)
      filename << "_graphNormBeta" << graphNormBeta;
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

  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0);
  int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  int spaceDim = 1;
  int k = -1; // poly order for field variables (-1 for a range of values)
  int delta_k_min = 1;
  int delta_k_max = 10;

  bool conformingTraces = true;
  bool useCondensedSolve = false;
  bool useZeroMeanConstraints = false;

  string problemChoiceString = "Poisson";

  double graphNormBeta = 1.0;

  int meshWidthMin = 2;
  int meshWidthMax = 2;
  
  cmdp.setOption("problem",&problemChoiceString,"problem choice: Poisson, ConvectionDiffusion, LinearElasticity, Stokes, Navier-Stokes");

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k_min", &delta_k_min, "test space polynomial order enrichment: min value to test");
  cmdp.setOption("delta_k_max", &delta_k_max, "test space polynomial order enrichment: max value to test");

  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("graphNormBeta", &graphNormBeta);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);

  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");
  
  cmdp.setOption("meshWidthMin", &meshWidthMin, "mesh width: min value to test");
  cmdp.setOption("meshWidthMax", &meshWidthMax, "mesh width: max value to test");

  cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");

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
  else
  {
    if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (rank==0)
  {
    cout << "Running " << problemChoiceString << ", with spaceDim " << spaceDim;
    cout << ", delta_k_min = " << delta_k_min << ", ";
    cout << ", delta_k_max = " << delta_k_max << ", ";
    if (conformingTraces)
      cout << "conforming traces";
    else
      cout << "non-conforming traces";
    cout << endl;
  }

  runMany(problemChoice, spaceDim, k,
          delta_k_min, delta_k_max,
          meshWidthMin, meshWidthMax,
          conformingTraces, useCondensedSolve,
          useZeroMeanConstraints, graphNormBeta);
  return 0;
}
