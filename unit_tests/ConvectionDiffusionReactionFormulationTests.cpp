//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConvectionDiffusionReactionFormulationTests
//  Camellia
//
//  Created by Nate Roberts on Mar 3, 2016
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "ConvectionDiffusionReactionFormulation.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"
#include "TypeDefs.h"

#include "Teuchos_LAPACK.hpp"

using namespace Camellia;

namespace
{
  void testSolve(int spaceDim, ConvectionDiffusionReactionFormulation::FormulationChoice formulationChoice,
                 Teuchos::FancyOStream &out, bool &success)
  {
    // solve for a polynomial manufactured solution:
    FunctionPtr x = Function::xn(1), y = Function::yn(1), z = Function::zn(1);
    FunctionPtr u_exact;
    int u_degree = 2;
    int beta_degree;
    // want beta to have zero divergence
    FunctionPtr beta;
    if (spaceDim == 1)
    {
      u_exact = x * x;
      beta = Function::constant(1.0);
      beta_degree = 0;
    }
    else if (spaceDim == 2)
    {
      u_exact = x * x + 2 * y * y;
      beta = Function::vectorize(y,x);
      beta_degree = 1;
    }
    else if (spaceDim == 3)
    {
      u_exact = x * x + 2 * y * y + 3 * z * z;
      beta = Function::constant({1,2,3});
      beta_degree = 0;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported spaceDim");
    }
    
    double alpha = 3.14159;
    double epsilon = 1e-3;
    ConvectionDiffusionReactionFormulation form(formulationChoice, spaceDim, beta, epsilon, alpha);
    
    FunctionPtr f = form.forcingFunction(u_exact);
    RHSPtr rhs = form.rhs(f);
    
    int H1Order = u_degree + beta_degree + 1; // discretization of flux involve beta * u
    int delta_k = 2;
    vector<double> meshDim(spaceDim,1.0);
    vector<int> elementWidths(spaceDim,1);
    BCPtr bc = BC::bc();
    IPPtr ip;
    // enforce a Dirichlet BC everywhere
    if (formulationChoice == ConvectionDiffusionReactionFormulation::ULTRAWEAK)
    {
      bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_exact);
      ip = form.bf()->graphNorm();;
    }
    else if (formulationChoice == ConvectionDiffusionReactionFormulation::PRIMAL)
    {
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_exact);
      ip = form.bf()->naiveNorm(spaceDim);
    }
    else if ((formulationChoice == ConvectionDiffusionReactionFormulation::SUPG) || (formulationChoice == ConvectionDiffusionReactionFormulation::LEAST_SQUARES) )
    {
      ip = Teuchos::null;
      delta_k = 0;
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_exact);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported formulation choice");
    }
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), meshDim, elementWidths, H1Order, delta_k);
    
    SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, ip);
    soln->setCubatureEnrichmentDegree(beta_degree); //set to match the degree of beta
    
    // because valgrind is complaining that SuperLU_Dist is doing some incorrect reads here, we
    // replace with KLU.  (Valgrind does not then complain.)
    SolverPtr solver = Solver::getSolver(Solver::KLU, false);
        
    soln->solve(solver);
    
    FunctionPtr u_soln = Function::solution(form.u(), soln);
    
    double tol = 5e-10; // the 3D solves do seem to want a looser tolerance (epsilon is fairly small, so maybe that's OK)
    double diff = (u_exact - u_soln)->l2norm(mesh);
    TEST_COMPARE(diff, <, tol);
  }
  
  void testUltraweakTraces(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    ConvectionDiffusionReactionFormulation::FormulationChoice formulationChoice = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    
    // set up an exactly resolvable polynomial manufactured solution:
    FunctionPtr x = Function::xn(1), y = Function::yn(1), z = Function::zn(1);
    FunctionPtr u_exact;
    int u_degree = 2;
    int beta_degree;
    // want beta to have zero divergence
    FunctionPtr beta;
    if (spaceDim == 1)
    {
      u_exact = x * x;
      beta = Function::constant(1.0);
      beta_degree = 0;
    }
    else if (spaceDim == 2)
    {
      u_exact = x * x + 2 * y * y;
//      beta = Function::vectorize(y,x);
      beta = Function::constant({1,2});
      beta_degree = 0;
    }
    else if (spaceDim == 3)
    {
      u_exact = x * x + 2 * y * y + 3 * z * z;
      beta = Function::constant({1,2,3});
      beta_degree = 0;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported spaceDim");
    }
    
    double alpha = 3.14159;
    double epsilon = 1e-2;
    ConvectionDiffusionReactionFormulation form(formulationChoice, spaceDim, beta, epsilon, alpha);
    
    BCPtr bc = BC::bc();
    IPPtr ip;
    // enforce a Dirichlet BC everywhere
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_exact);
    ip = form.bf()->graphNorm();;
    
    FunctionPtr f = form.forcingFunction(u_exact);
    RHSPtr rhs = form.rhs(f);
    
    int H1Order = u_degree + beta_degree + 1; // discretization of flux involve beta * u
    int delta_k = 2;
    vector<double> meshDim(spaceDim,1.0);
    vector<int> elementWidths(spaceDim,1); // single-element solution
    
    VarPtr u = form.u();
    VarPtr u_hat = form.u_hat();
    VarPtr sigma = form.sigma();
    VarPtr sigma_n = form.sigma_n();
    
    FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
    map<int, FunctionPtr> solnMap;
    solnMap[u->ID()] = u_exact;
    solnMap[sigma->ID()] = sqrt(epsilon) * u_exact->grad();
    solnMap[u_hat->ID()] = u_hat->termTraced()->evaluate(solnMap) * traceFxn;
    solnMap[sigma_n->ID()] = sigma_n->termTraced()->evaluate(solnMap);
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), meshDim, elementWidths, H1Order, delta_k);
    
    SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, form.bf()->graphNorm());
    soln->setCubatureEnrichmentDegree(beta_degree); //set to match the degree of beta
    mesh->registerSolution(soln);
    
    const int solutionOrdinal = 0;
    soln->projectOntoMesh(solnMap, solutionOrdinal);
    
    // now, we should have zero energy error
    double energyError = soln->energyErrorTotal();
    double tol = 1e-12;
    
    FunctionPtr u_soln = Function::solution(u, soln);
    FunctionPtr sigma_soln = Function::solution(sigma, soln);
    FunctionPtr sigma_exact = sqrt(epsilon) * u_exact->grad();
    FunctionPtr u_hat_exact = solnMap[u_hat->ID()];
    FunctionPtr u_hat_soln = Function::solution(u_hat, soln);
    FunctionPtr sigma_n_exact = solnMap[sigma_n->ID()];
    FunctionPtr sigma_n_soln = Function::solution(sigma_n, soln, false);
    
    double u_error = abs((u_exact - u_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(u_error, <, tol);

    double sigma_error = abs((sigma_exact - sigma_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(sigma_error, <, tol);
    
    double u_hat_error = abs((u_hat_exact - u_hat_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(u_hat_error, <, tol);
    
    double sigma_n_error = abs((sigma_n_exact - sigma_n_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(sigma_n_error, <, tol);
    
    TEST_COMPARE(energyError, <, tol);
  
    // now, the actual test: when we refine, do we still have zero energy error?
    mesh->hRefine(vector<GlobalIndexType>{0});
    
    energyError = soln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    u_error = abs((u_exact - u_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(u_error, <, tol);
    
    sigma_error = abs((sigma_exact - sigma_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(sigma_error, <, tol);
    
    u_hat_error = abs((u_hat_exact - u_hat_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(u_hat_error, <, tol);
    
    sigma_n_error = abs((sigma_n_exact - sigma_n_soln)->l2norm(mesh,beta_degree));
    TEST_COMPARE(sigma_n_error, <, tol);
    
//    HDF5Exporter exporter(mesh, "sigma_n", "/tmp");
//    exporter.exportFunction({sigma_n_soln, sigma_n_exact, sigma_n_exact - sigma_n_soln}, {"sigma_n_soln","sigma_n_exact","sigma_n_error"});
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, DofCount_ContinuousTriangles )
  {
    MPIWrapper::CommWorld()->Barrier();
    
    // a couple tests to ensure that the dof counts are correct in linear, triangular Bubnov-Galerkin meshes
    int spaceDim = 2;
    bool useTriangles = true;
    FunctionPtr x = Function::xn(1), y = Function::yn(1);
    FunctionPtr beta = Function::vectorize(y,x);
    double alpha = 0;
    double epsilon = 1e-2;

    ConvectionDiffusionReactionFormulation form(ConvectionDiffusionReactionFormulation::SUPG, spaceDim, beta, epsilon, alpha);
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    
    int H1Order = 1, delta_k = 0;
    MeshPtr mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);

    // create a serial MeshTopology for accurate counting of global vertices
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    int vertexDim = 0;
    int numVertices = meshTopo->getEntityCount(vertexDim);
    
    int globalDofCount = mesh->numGlobalDofs();
    TEST_EQUALITY(numVertices, globalDofCount);
    
    elementCounts = {16,16};
    mesh = MeshFactory::quadMeshMinRule(form.bf(), H1Order, delta_k, dimensions[0], dimensions[1],
                                        elementCounts[0], elementCounts[1], useTriangles);
    meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                             elementCounts[0], elementCounts[1], useTriangles);

    numVertices = meshTopo->getEntityCount(vertexDim);
    
    globalDofCount = mesh->numGlobalDofs();
    TEST_EQUALITY(numVertices, globalDofCount);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, ForcingFunction )
  {
    int spaceDim = 3;
    FunctionPtr x = Function::xn(1), y = Function::yn(1), z = Function::zn(1);
    FunctionPtr u_exact = x * x + 2 * y * y + 3 * z * z;
    FunctionPtr u_grad = Function::vectorize(2 * x, 4 * y, 6 * z);
    FunctionPtr u_laplacian = Function::constant(12.0);
    
    // want beta to have zero divergence
    FunctionPtr beta = Function::vectorize(y,z,x);
    double alpha = 3.14159;
    double epsilon = 1e-3;
    
    FunctionPtr f_expected = -epsilon * u_laplacian + beta * u_grad + alpha * u_exact;
    
    ConvectionDiffusionReactionFormulation form(ConvectionDiffusionReactionFormulation::ULTRAWEAK, spaceDim, beta, epsilon, alpha);
    
    FunctionPtr f_actual = form.forcingFunction(u_exact);
    
    // set up a single-element mesh for comparing f_actual with f_expected
    int H1Order = 2;
    int delta_k = 1;
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0,1.0}, {1,1,1}, H1Order, delta_k);
    double tol = 1e-15;
  
    double diff = (f_actual-f_expected)->l2norm(mesh);
    TEST_COMPARE(diff, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, GramMatrixCholeskyFactorization_UltraweakTriangles )
  {
    bool useTriangles = true;
    
    int polyOrder = 2;
    int delta_k = 2;
    int spaceDim = 2;
    
    double alpha = 0.0;
    double epsilon = 1.0;
    
    FunctionPtr beta = Function::constant({2.0,1.0});
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    ConvectionDiffusionReactionFormulation form(formulation, spaceDim, beta, epsilon, alpha);
    
    BFPtr bf = form.bf();
    int H1Order = polyOrder + 1;
    
    vector<double> dimensions = {1.0,1.0};
    int meshWidth = 1;
    vector<int> elementCounts = {meshWidth,meshWidth};

    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    IPPtr ip = bf->graphNorm();
//    ip->printInteractions();
    
    GlobalIndexType cellID = 1;
    int numCells = 1;
    if (mesh->myCellsInclude(cellID))
    {
      ElementTypePtr elementType = mesh->getElementType(cellID);
      DofOrderingPtr testOrder = elementType->testOrderPtr;
      int numTestDofs = testOrder->totalDofs();
      Camellia::FieldContainer<double> gramMatrix(numCells,numTestDofs,numTestDofs);
      bool testVsTest = true;
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);
      ip->computeInnerProductMatrix(gramMatrix, testOrder, basisCache);
//      {
//        gramMatrix.resize(numTestDofs,numTestDofs);
//        SerialDenseWrapper::writeMatrixToMatlabFile("gramMatrix.dat", gramMatrix);
//        gramMatrix.resize(1,numTestDofs,numTestDofs);
//      }
      int INFO = 0;
      int N = numTestDofs;
      char UPLO = 'L';
      Teuchos::LAPACK<int, double> lapack;
      lapack.POTRF(UPLO, N, &gramMatrix[0], N, &INFO);

      TEST_EQUALITY(INFO, 0);
      
      VarPtr tau = form.tau(), v = form.v();
      BasisPtr tauBasis = testOrder->getBasis(tau->ID());
      BasisPtr vBasis = testOrder->getBasis(v->ID());
      DofOrderingPtr tauOrdering = Teuchos::rcp( new DofOrdering(CellTopology::triangle()) );
      tauOrdering->addEntry(tau->ID(), tauBasis, tau->rank());
      DofOrderingPtr vOrdering = Teuchos::rcp( new DofOrdering(CellTopology::triangle()) );
      vOrdering->addEntry(v->ID(), vBasis, v->rank());
      
      IPPtr ipTau = IP::ip(), ip_v = IP::ip();
      ipTau->addTerm(tau);
      ipTau->addTerm(tau->div());
      ipTau->addTerm(tau);
      
      ip_v->addTerm(v->grad());
      ip_v->addTerm(-beta * v->grad());
      ip_v->addTerm(v);
      
      int vTestDofs = vBasis->getCardinality(), tauTestDofs = tauBasis->getCardinality();
      gramMatrix.resize(numCells,vTestDofs,vTestDofs);
      ip_v->computeInnerProductMatrix(gramMatrix, vOrdering, basisCache);
//      {
//        gramMatrix.resize(gramMatrix.dimension(1),gramMatrix.dimension(2));
//        SerialDenseWrapper::writeMatrixToMatlabFile("gramMatrix_v.dat", gramMatrix);
//        gramMatrix.resize(1,gramMatrix.dimension(0),gramMatrix.dimension(1));
//      }
      INFO = 0;
      N = vTestDofs;
      lapack.POTRF(UPLO, N, &gramMatrix[0], N, &INFO);

      TEST_EQUALITY(INFO, 0);

      gramMatrix.resize(numCells,tauTestDofs,tauTestDofs);
      ipTau->computeInnerProductMatrix(gramMatrix, tauOrdering, basisCache);
//      {
//        gramMatrix.resize(gramMatrix.dimension(1),gramMatrix.dimension(2));
//        SerialDenseWrapper::writeMatrixToMatlabFile("gramMatrix_tau.dat", gramMatrix);
//        gramMatrix.resize(1,gramMatrix.dimension(0),gramMatrix.dimension(1));
//      }
      INFO = 0;
      N = tauTestDofs;
      lapack.POTRF(UPLO, N, &gramMatrix[0], N, &INFO);
      
      TEST_EQUALITY(INFO, 0);
      
      IPPtr ipTauL2 = IP::ip();
      ipTauL2->addTerm(tau);
      
      gramMatrix.resize(numCells,tauTestDofs,tauTestDofs);
      ipTauL2->computeInnerProductMatrix(gramMatrix, tauOrdering, basisCache);
//      {
//        gramMatrix.resize(gramMatrix.dimension(1),gramMatrix.dimension(2));
//        SerialDenseWrapper::writeMatrixToMatlabFile("gramMatrix_tauL2.dat", gramMatrix);
//        gramMatrix.resize(1,gramMatrix.dimension(0),gramMatrix.dimension(1));
//      }
      INFO = 0;
      N = tauTestDofs;
      lapack.POTRF(UPLO, N, &gramMatrix[0], N, &INFO);
      
      TEST_EQUALITY(INFO, 0);
    }
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolvePrimal_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveSUPG_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::SUPG;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveSUPG_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::SUPG;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveSUPG_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::SUPG;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveUltraweak_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
    testSolve(spaceDim, formulation, out, success);
  }

  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveLeastSquares_1D )
  {
    int spaceDim = 1;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::LEAST_SQUARES;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveLeastSquares_2D )
  {
    int spaceDim = 2;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::LEAST_SQUARES;
    testSolve(spaceDim, formulation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, SolveLeastSquares_3D_Slow )
  {
    int spaceDim = 3;
    ConvectionDiffusionReactionFormulation::FormulationChoice formulation = ConvectionDiffusionReactionFormulation::LEAST_SQUARES;
    testSolve(spaceDim, formulation, out, success);
  }

//  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, TracesUltraweak_1D )
//  {
//    int spaceDim = 1;
//    testUltraweakTraces(spaceDim, out, success);
//  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, TracesUltraweak_2D )
  {
    int spaceDim = 2;
    testUltraweakTraces(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConvectionDiffusionReactionFormulation, TracesUltraweak_3D )
  {
    int spaceDim = 3;
    testUltraweakTraces(spaceDim, out, success);
  }
} // namespace
