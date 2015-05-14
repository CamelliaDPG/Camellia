#include "AdditiveSchwarz.h"
#include "CamelliaDebugUtility.h"
#include "ExpFunction.h"
#include "GMGOperator.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "Solver.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

#include "AztecOO.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_IC.h"
#include "Ifpack_ILU.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;
using namespace Intrepid;

bool runGMGOperatorInDebugMode;
int maxDofsForKLU;
double coarseCGTol;
int coarseMaxIterations;

string getFactorizationTypeString(GMGOperator::FactorType factorizationType) {
  switch (factorizationType) {
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

GMGOperator::FactorType getFactorizationType(string factorizationTypeString) {
  if (factorizationTypeString == "Direct") {
    return GMGOperator::Direct;
  }
  if (factorizationTypeString == "IC") return GMGOperator::IC;
  if (factorizationTypeString == "ILU") return GMGOperator::ILU;

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "factorization type not recognized");
}

Teuchos::RCP<Epetra_Operator> CamelliaAdditiveSchwarzPreconditioner(::Epetra_RowMatrix* A, int overlapLevel, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                                                                    GMGOperator::FactorType schwarzBlockFactorization,
                                                                    int levelOfFill, double fillRatio, bool hOnly) {

  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization) {
    case GMGOperator::Direct:
      preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel, mesh, dofInterpreter, hOnly) );
      break;
    case GMGOperator::IC:
      preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_IC>(A, overlapLevel, mesh, dofInterpreter, hOnly) );
      List.set("fact: ict level-of-fill", fillRatio);
      break;
    case GMGOperator::ILU:
      preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_ILU>(A, overlapLevel, mesh, dofInterpreter, hOnly) );
      List.set("fact: level-of-fill", levelOfFill);
      break;
    default:
      break;
  }

  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }

  err = preconditioner->Initialize();
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }

  return preconditioner;
}

Teuchos::RCP<Epetra_Operator> IfPackAdditiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel,
                                                                  GMGOperator::FactorType schwarzBlockFactorization,
                                                                  int levelOfFill, double fillRatio) {
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner;
  Teuchos::ParameterList List;
  switch (schwarzBlockFactorization) {
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
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }


  err = preconditioner->Initialize();
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();

  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }

  return preconditioner;
}

class AztecSolver : public Solver {
  int _maxIters;
  double _tol;
  int _schwarzOverlap;
  bool _useSchwarzPreconditioner;
  bool _hierarchical; // for Camellia Schwarz preconditioners

  int _iterationCount;

  int _azOutputLevel;

  int _levelOfFill;
  double _fillRatio;

  MeshPtr _mesh;
  Teuchos::RCP<DofInterpreter> _dofInterpreter;
  GMGOperator::FactorType _schwarzBlockFactorization;
public:
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              GMGOperator::FactorType schwarzBlockFactorization, int levelOfFill, double fillRatio) {
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _levelOfFill = levelOfFill;
    _fillRatio = fillRatio;
    _hierarchical = false;
  }

  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              GMGOperator::FactorType schwarzBlockFactorization, int levelOfFill, double fillRatio,
              MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter, bool hierarchical) {
    _mesh = mesh;
    _dofInterpreter = dofInterpreter;
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _azOutputLevel = 1;
    _levelOfFill = levelOfFill;
    _fillRatio = fillRatio;
    _hierarchical = hierarchical;
  }
  void setAztecOutputLevel(int AztecOutputLevel) {
    _azOutputLevel = AztecOutputLevel;
  }

  int solve() {
    AztecOO solver(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());

    solver.SetAztecOption(AZ_solver, AZ_cg);

    Teuchos::RCP<Epetra_CrsMatrix> A = this->_stiffnessMatrix;

    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null) {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _mesh, _dofInterpreter, _schwarzBlockFactorization,
                                                             _levelOfFill, _fillRatio, _hierarchical);

//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);

    } else {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _schwarzBlockFactorization, _levelOfFill, _fillRatio);
    }

    if (_useSchwarzPreconditioner) {
      solver.SetPrecOperator(preconditioner.get());
      solver.SetAztecOption(AZ_precond, AZ_user_precond);
    } else {
      solver.SetAztecOption(AZ_precond, AZ_none);
    }

    solver.SetAztecOption(AZ_output, _azOutputLevel);

    solver.SetAztecOption(AZ_conv, AZ_r0); // convergence is relative to the initial residual

    int solveResult = solver.Iterate(_maxIters,_tol);

    int remainingIters = _maxIters;

    const double* status = solver.GetAztecStatus();
    int whyTerminated = status[AZ_why];

    int rank = Teuchos::GlobalMPISession::getRank();

    int maxRestarts = 1;
    int numRestarts = 0;
    while ((whyTerminated==AZ_loss) && (numRestarts < maxRestarts)) {
      remainingIters -= status[AZ_its];
      if (rank==0) cout << "Aztec warned that the recursive residual indicates convergence even though the true residual is too large.  Restarting with the new solution as initial guess, with maxIters = " << remainingIters << endl;
      solveResult = solver.Iterate(remainingIters,_tol);
      whyTerminated = status[AZ_why];
      numRestarts++;
    }
    remainingIters -= status[AZ_its];
    _iterationCount = _maxIters - remainingIters;

    switch (whyTerminated) {
      case AZ_normal:
//        cout << "whyTerminated: AZ_normal " << endl;
        break;
      case AZ_param:
        cout << "whyTerminated: AZ_param " << endl;
        break;
      case AZ_breakdown:
        cout << "whyTerminated: AZ_breakdown " << endl;
        break;
      case AZ_loss:
        cout << "whyTerminated: AZ_loss " << endl;
        break;
      case AZ_ill_cond:
        cout << "whyTerminated: AZ_ill_cond " << endl;
        break;
      case AZ_maxits:
        cout << "whyTerminated: AZ_maxits " << endl;
        break;
      default:
        break;
    }

    _iterationCount = status[AZ_its];

    return solveResult;
  }
  Teuchos::RCP< Epetra_CrsMatrix > getPreconditionerMatrix(const Epetra_Map &map) {
    Teuchos::RCP<Epetra_CrsMatrix> A = this->_stiffnessMatrix;
    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null) {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _mesh, _dofInterpreter,
                                                             _schwarzBlockFactorization, _levelOfFill, _fillRatio,_hierarchical);

//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);

    } else {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A.get(), _schwarzOverlap, _schwarzBlockFactorization,
                                                           _levelOfFill, _fillRatio);
    }

    return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, map);
  }
  int iterationCount() {
    return _iterationCount;
  }
};

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

enum ProblemChoice {
  Poisson,
  ConvectionDiffusion,
  Stokes,
  NavierStokes
};

enum RunManyPreconditionerChoices {
  DontPrecondition,
  GMGAlgebraicSchwarz, // GMG with algebraic Schwarz smoother
  GMGGeometricSchwarz, // GMG with geometric Schwarz smoother
  GMGPointSymmetricGaussSeidel, // GMG with point symmetric Gauss Seidel smoother
  GMGBlockSymmetricGaussSeidel, // GMG with block symmetric Gauss Seidel smoother
  GMGPointJacobi, // GMG with point symmetric Jacobi smoother
  GMGBlockJacobi, // GMG with block symmetric Jacobi smoother
  AlgebraicSchwarz, // algebraic Schwarz preconditioner
  GeometricSchwarz, // geometric Schwarz preconditioner
  AllGMG,     // Schwarz smoother, multiple overlap values, both algebraic and geometric Schwarz
  AllSchwarz, // Schwarz as the only preconditioner; multiple overlap values, both algebraic and geometric Schwarz
  All         // All of the above, including the DontPrecondition option
};

string smootherString(GMGOperator::SmootherChoice smoother) {
  switch (smoother) {
    case GMGOperator::IFPACK_ADDITIVE_SCHWARZ:
      return "IfPack-Schwarz";
    case GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ:
      return "Camellia-Schwarz";
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

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, MeshPtr &coarseMesh, IPPtr &graphNorm, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces, bool useStaticCondensation, int numCells, int k, int delta_k,
                                     int rootMeshNumCells, bool useZeroMeanConstraints = false) {
  BFPtr bf;
  BCPtr bc;
  RHSPtr rhs;
  MeshPtr mesh;

  double width = 1.0; // in each dimension
  vector<double> x0(spaceDim,0); // origin is the default

  VarPtr p; // pressure

  if (problemChoice == Poisson) {
    PoissonFormulation formulation(spaceDim, conformingTraces);

    bf = formulation.bf();

    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);

    VarPtr q = formulation.q();
    rhs->addTerm( f * q );

    bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    VarPtr phi_hat = formulation.phi_hat();
    bc->addDirichlet(phi_hat, boundary, Function::zero());
  } else if (problemChoice == ConvectionDiffusion) {
    double epsilon = 1e-2;
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
    if (spaceDim == 1) {
      bc->addDirichlet(tc, inflowX, -one);
      bc->addDirichlet(uhat, outflowX, zero);
    }
    if (spaceDim == 2) {
      bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
      bc->addDirichlet(uhat, outflowY, zero);
    }
    if (spaceDim == 3) {
      bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
      bc->addDirichlet(uhat, outflowX, zero);
      bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
      bc->addDirichlet(uhat, outflowY, zero);
      bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
      bc->addDirichlet(uhat, outflowZ, zero);
    }
  } else if (problemChoice == Stokes) {

    StokesVGPFormulation formulation(spaceDim, conformingTraces);

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

    if (spaceDim == 2) {
      // this one was in the Cockburn Kanschat LDG Stokes paper
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y;
      p_exact = 2.0 * exp_x * sin_y;
    } else {
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

    double mu = 1.0;

    FunctionPtr f1, f2, f3;
    if (spaceDim==2) {
      f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
      f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    } else {
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

  int H1Order = k + 1;

  BFPtr bilinearForm = bf;

  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++) {
    dimensions.push_back(width);
    elementCounts.push_back(rootMeshNumCells);
  }
  mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0);

  // now that we have mesh, add pressure constraint for Stokes (imposing zero at origin--want to aim for center of mesh)
  if ((problemChoice == Stokes) || (problemChoice==NavierStokes)) {
    if (!useZeroMeanConstraints) {
      vector<double> origin(spaceDim,0);
      IndexType vertexIndex;

      if (!mesh->getTopology()->getVertexIndex(origin, vertexIndex)) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "origin vertex not found");
      }
      bc->addSinglePointBC(p->ID(), 0, vertexIndex);
    } else {
      bc->addZeroMeanConstraint(p);
    }
  }

  int H1Order_coarse = 0 + 1;
  coarseMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order_coarse, delta_k, x0);

  // get a sample cell topology:
  CellTopoPtr cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);

  int meshWidthCells = rootMeshNumCells;
  while (meshWidthCells < numCells) {
    set<IndexType> activeCellIDs = mesh->getActiveCellIDs(); // should match between coarseMesh and mesh
    mesh->hRefine(activeCellIDs, refPattern);
    coarseMesh->hRefine(activeCellIDs, refPattern);

    meshWidthCells *= 2;
  }

  if (meshWidthCells != numCells) {
    int rank = Teuchos::GlobalMPISession::getRank();
    if (rank == 0) {
      cout << "Warning: may have overrefined mesh; mesh has width " << meshWidthCells << ", not " << numCells << endl;
    }
  }

  graphNorm = bf->graphNorm();

  solution = Solution::solution(mesh, bc, rhs, graphNorm);
  solution->setUseCondensedSolve(useStaticCondensation);
  solution->setZMCsAsGlobalLagrange(false); // fine grid solution shouldn't impose ZMCs (should be handled in coarse grid solve)
}

void run(ProblemChoice problemChoice, int &iterationCount, int spaceDim, int numCells, int k, int delta_k, bool conformingTraces,
         bool useStaticCondensation, bool precondition, bool schwarzOnly, GMGOperator::SmootherChoice smootherType,
         int schwarzOverlap, GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
         Solver::SolverChoice coarseSolverChoice, double cgTol, int cgMaxIterations, int AztecOutputLevel,
         bool reportTimings, double &solveTime, bool reportEnergyError, int numCellsRootMesh, bool hOnly, bool useZeroMeanConstraints,
         bool writeAndExit) {
  int rank = Teuchos::GlobalMPISession::getRank();

  if (hOnly && (numCellsRootMesh == -1)) {
    // then use a single level of h-coarsening as the root mesh.
    numCellsRootMesh = numCells / 2;
    if (numCellsRootMesh == 0) {
      cout << "Too few cells in root mesh.  Aborting.\n";
      exit(1);
    }
  } else if (numCellsRootMesh == -1) {
    numCellsRootMesh = numCells;
  }

  SolutionPtr solution;
  MeshPtr k0Mesh;
  IPPtr graphNorm;
  initializeSolutionAndCoarseMesh(solution, k0Mesh, graphNorm, problemChoice, spaceDim, conformingTraces, useStaticCondensation,
                                  numCells, k, delta_k, numCellsRootMesh, useZeroMeanConstraints);

  MeshPtr mesh = solution->mesh();
  BCPtr bc = solution->bc();

  if (hOnly) {
    // then replace the k0Mesh with the h-coarsened mesh:
    MeshTopologyPtr coarseMeshTopo = mesh->getTopology()->getRootMeshTopology();
    int H1OrderP0 = k + 1;
    k0Mesh = Teuchos::rcp( new Mesh(coarseMeshTopo, mesh->bilinearForm(), H1OrderP0, delta_k) );
  }

  Teuchos::RCP<Solver> solver;
  if (!precondition) {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition, schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio) );
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  } else if (schwarzOnly) {
    if (smootherType==GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap, precondition, schwarzBlockFactorization,
                                             schwarzLevelOfFill, schwarzFillRatio, mesh, solution->getDofInterpreter(), hOnly) );
    } else {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition, schwarzBlockFactorization,
                                             schwarzLevelOfFill, schwarzFillRatio) );
    }
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  } else {
    BCPtr zeroBCs = bc->copyImposingZero();
    bool saveFactorization = true;

    Teuchos::RCP<Solver> coarseSolver = Teuchos::null;
    GMGSolver* gmgSolver = new GMGSolver(solution, k0Mesh, cgMaxIterations, cgTol, coarseSolver, useStaticCondensation);

    if (coarseSolverChoice != Solver::GMGSolver_1_Level_h) {
      coarseSolver = Solver::getSolver(coarseSolverChoice, saveFactorization,
                                       coarseCGTol, coarseMaxIterations);
    } else {
      MeshTopologyPtr coarsestMeshTopo = k0Mesh->getTopology()->getRootMeshTopology();
      int H1OrderP0 = 0 + 1;
      MeshPtr coarsestMesh = Teuchos::rcp( new Mesh(coarsestMeshTopo, k0Mesh->bilinearForm(), H1OrderP0, delta_k) );

      int numGlobalDofs = coarsestMesh->numGlobalDofs();

      // debugging:
      // if (rank==0) cout << "coarsest mesh, dof count: " << numGlobalDofs << endl;
      // put all coarsest mesh cells on rank 0, where KLU will solve anyway:
      // (turning off for now because this seems to slow things down significantly on BG/Q)
//      coarsestMesh->setPartitionPolicy(MeshPartitionPolicy::oneRankPartitionPolicy(0));

      SolverPtr coarsestSolver;

      if (numGlobalDofs <= maxDofsForKLU) {
        coarsestSolver = Solver::getSolver(Solver::KLU, saveFactorization);
      } else {
        coarsestSolver = Solver::getSolver(Solver::SuperLUDist, saveFactorization);
      }

      coarseSolver = Solver::getSolver(coarseSolverChoice, saveFactorization,
                                       coarseCGTol, coarseMaxIterations,
                                       gmgSolver->gmgOperator().getCoarseSolution(),
                                       coarsestMesh, coarsestSolver);
      GMGSolver* coarseSolverGMG = static_cast<GMGSolver*>( coarseSolver.get() );

      coarseSolverGMG->gmgOperator().setSmootherType(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    }

    gmgSolver->gmgOperator().setCoarseSolver(coarseSolver);

    if ((smootherType == GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) && hOnly)
    {
      // then use hierarchical neighbor relationship
      gmgSolver->gmgOperator().setUseHierarchicalNeighborsForSchwarz(true);
      if (rank==0) cout << "using new hierarchical Schwarz neighbors option";
    }

//    GMGSolver* gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh, solution->getDofInterpreter(),
//                                         solution->getPartitionMap(), cgMaxIterations, cgTol, coarseSolver,
//                                         useStaticCondensation);
    gmgSolver->setAztecOutput(AztecOutputLevel);
//    gmgSolver->setComputeConditionNumberEstimate(false);

    gmgSolver->setUseConjugateGradient(true);
    gmgSolver->setComputeConditionNumberEstimate(false);
    gmgSolver->gmgOperator().setSchwarzFactorizationType(schwarzBlockFactorization);
    gmgSolver->gmgOperator().setLevelOfFill(schwarzLevelOfFill);
    gmgSolver->gmgOperator().setFillRatio(schwarzFillRatio);
//    cout << "Set GMGOperator level of fill to " << schwarzLevelOfFill << endl;
//    cout << "Set GMGOperator fill ratio to " << schwarzFillRatio << endl;
    gmgSolver->gmgOperator().setSmootherType(smootherType);
    gmgSolver->gmgOperator().setSmootherOverlap(schwarzOverlap);
    gmgSolver->gmgOperator().setDebugMode(runGMGOperatorInDebugMode);
    solver = Teuchos::rcp( gmgSolver ); // we use "new" above, so we can let this RCP own the memory
  }

//  if (problemChoice==Stokes) {
//    if (rank==0) cout << "Writing fine Stokes matrix to /tmp/A_stokes.dat.\n";
//    solution->setWriteMatrixToFile(true, "/tmp/A_stokes.dat");
//  }

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time timer(Comm);

  int result = solution->solve(solver);

  solveTime = timer.ElapsedTime();

  if (result == 0) {
    if (!precondition) {
      iterationCount = ((AztecSolver *) solver.get())->iterationCount();
    } else if (schwarzOnly) {
      iterationCount = ((AztecSolver *) solver.get())->iterationCount();
    } else {
      iterationCount = ((GMGSolver *) solver.get())->iterationCount();
    }
  } else {
    iterationCount = -1;
  }

  if (reportTimings) solution->reportTimings();
  double energyErrorTotal = solution->energyErrorTotal();

  GMGSolver* fineSolver = dynamic_cast<GMGSolver*>(solver.get());
  if (fineSolver != NULL) {
    if (rank==0) cout << "************   Fine GMG Solver, timings   *************\n";
    fineSolver->gmgOperator().reportTimings();

    GMGSolver* coarseSolver = dynamic_cast<GMGSolver*>(fineSolver->gmgOperator().getCoarseSolver().get());
    if (coarseSolver != NULL) {
      if (rank==0) cout << "************   Coarse GMG Solver, timings   *************\n";
      coarseSolver->gmgOperator().reportTimings();
      vector<int> iterationCountLog = coarseSolver->getIterationCountLog();
      if (rank==0) Camellia::print("coarseSolver iteration counts:",iterationCountLog);
      double totalIterationCount = 0;
      for (int i=0; i<iterationCountLog.size(); i++) {
        totalIterationCount += iterationCountLog[i];
      }
      if (rank==0) cout << "Average coarse solver iteration count: " << totalIterationCount / iterationCountLog.size() << endl;
    }
  }

  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if ((rank==0) && reportEnergyError) {
    cout << "Mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Energy error: " << energyErrorTotal << endl;
  }

  if (writeAndExit) {
    Teuchos::RCP< Epetra_CrsMatrix > A = solution->getStiffnessMatrix();
    Teuchos::RCP< Epetra_CrsMatrix > M;
    if (schwarzOnly) {
      if (rank==0) cout << "writeAndExit not yet supported for schwarzOnly.\n";
    } else {
      GMGOperator* op = &((GMGSolver*)solver.get())->gmgOperator();

      Teuchos::RCP< Epetra_CrsMatrix > A_coarse = op->getCoarseStiffnessMatrix();
      if (rank==0) cout << "writing A_coarse to A_coarse.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_coarse.dat",*A_coarse, NULL, NULL, false);

      Teuchos::RCP< Epetra_CrsMatrix > P = op->getProlongationOperator();
      if (rank==0) cout << "writing P to P.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("P.dat",*P, NULL, NULL, false);

      if (rank==0) cout << "writing A to A_fine.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("A_fine.dat",*A, NULL, NULL, false);

      if (rank==0) cout << "writing smoother to S.dat.\n";
      Teuchos::RCP< Epetra_CrsMatrix > S = op->getSmootherAsMatrix();
      EpetraExt::RowMatrixToMatrixMarketFile("S.dat",*S, NULL, NULL, false);

      return;
    }
  }

//  if (rank==0) cout << "NOTE: Exported solution for debugging.\n";
//  HDF5Exporter::exportSolution("/tmp/testSolution", "testSolution", solution);
//
//  solution->solve();
//  if (rank==0) cout << "NOTE: Exported direct solution for debugging.\n";
//  HDF5Exporter::exportSolution("/tmp/testSolution", "testSolution_direct", solution);
//  energyErrorTotal = solution->energyErrorTotal();
//  if (rank==0) cout << "Direct solution has energy error " << energyErrorTotal << endl;
}

void runMany(ProblemChoice problemChoice, int spaceDim, int delta_k, int minCells,
             bool conformingTraces, bool useStaticCondensation,
             GMGOperator::FactorType schwarzBlockFactorization, int schwarzLevelOfFill, double schwarzFillRatio,
             Solver::SolverChoice coarseSolverChoice,
             double cgTol, int cgMaxIterations, int aztecOutputLevel, RunManyPreconditionerChoices preconditionerChoices,
             int k, int overlapLevel, int numCellsRootMesh, bool reportTimings, bool hOnly, int maxCells, bool useZeroMeanConstraints) {
  int rank = Teuchos::GlobalMPISession::getRank();

  string problemChoiceString;
  switch (problemChoice) {
    case Poisson:
      problemChoiceString = "Poisson";
      break;
    case ConvectionDiffusion:
      problemChoiceString = "ConvectionDiffusion";
      break;
    case Stokes:
      problemChoiceString = "Stokes";
      break;
    case NavierStokes:
      problemChoiceString = "NavierStokes";
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled problem choice");
      break;
  }

  string preconditionerChoiceString;
  switch (preconditionerChoices) {
    case All:
      preconditionerChoiceString = "All";
      break;
    case AllGMG:
      preconditionerChoiceString = "AllGMG";
      break;
    case AllSchwarz:
      preconditionerChoiceString = "AllSchwarz";
      break;
    case DontPrecondition:
      preconditionerChoiceString = "NoPrecondition";
      break;
    case GMGAlgebraicSchwarz:
      preconditionerChoiceString = "GMGAlgebraicSchwarz";
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
    case AlgebraicSchwarz:
      preconditionerChoiceString = "AlgebraicSchwarz";
      break;
    case GeometricSchwarz:
      preconditionerChoiceString = "GeometricSchwarz";
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled preconditioner choice subset");
      break;
  }

  vector<bool> preconditionValues;

  vector<bool> schwarzOnly_maxChoices;
  vector<GMGOperator::SmootherChoice> smootherChoices;

  switch (preconditionerChoices) {
    case DontPrecondition:
      preconditionValues.push_back(false);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::NONE);
      break;
    case GMGAlgebraicSchwarz:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      break;
    case GMGGeometricSchwarz:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
      break;
    case GMGBlockJacobi:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
      break;
    case GMGPointJacobi:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::POINT_JACOBI);
      break;
    case GMGBlockSymmetricGaussSeidel:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
      break;
    case GMGPointSymmetricGaussSeidel:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
      break;
    case AlgebraicSchwarz:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(true);
      smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      break;
    case GeometricSchwarz:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(true);
      smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
      break;
    case AllGMG:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
//      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
      smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
//      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
      smootherChoices.push_back(GMGOperator::POINT_JACOBI);
      break;
    case AllSchwarz:
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(true);
      smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
      break;
    case All:
      preconditionValues.push_back(false);
      preconditionValues.push_back(true);
      schwarzOnly_maxChoices.push_back(false);
      schwarzOnly_maxChoices.push_back(true);
      smootherChoices.push_back(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      smootherChoices.push_back(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
//      smootherChoices.push_back(GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL);
      smootherChoices.push_back(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
//      smootherChoices.push_back(GMGOperator::BLOCK_JACOBI);
      smootherChoices.push_back(GMGOperator::POINT_JACOBI);
      break;
  }

  vector<int> kValues;
  if (k == -1) {
    kValues.push_back(1);
    kValues.push_back(2);
    if (spaceDim < 3) kValues.push_back(4);
    if (spaceDim < 2) kValues.push_back(8);
    if (spaceDim < 2) kValues.push_back(16);
  } else {
    kValues.push_back(k);
  }

  vector<int> numCellsValues;
  int numCells = minCells;
  while (pow((double)numCells,spaceDim) <= maxCells) {
    // want to do as many as we can with just one cell per processor
    numCellsValues.push_back(numCells);
    numCells *= 2;
  }

  ostringstream results;
  results << "Preconditioner\tSmoother\tOverlap\tnum_cells\tmesh_width\tk\tIterations\tSolve_time\n";

  for (vector<bool>::iterator preconditionChoiceIt = preconditionValues.begin(); preconditionChoiceIt != preconditionValues.end(); preconditionChoiceIt++) {
    bool precondition = *preconditionChoiceIt;

    vector<bool> schwarzOnlyValues;
    vector<GMGOperator::SmootherChoice> smootherChoiceValues;
    if (precondition) {
      schwarzOnlyValues = schwarzOnly_maxChoices;
      smootherChoiceValues = smootherChoices;
    } else {
      // schwarzOnly and smootherChoice ignored; just use one of each
      schwarzOnlyValues.push_back(false);
      smootherChoiceValues.push_back(GMGOperator::NONE);
    }
    for (vector<bool>::iterator schwarzOnlyChoiceIt = schwarzOnlyValues.begin(); schwarzOnlyChoiceIt != schwarzOnlyValues.end(); schwarzOnlyChoiceIt++) {
      bool schwarzOnly = *schwarzOnlyChoiceIt;
      for (vector<GMGOperator::SmootherChoice>::iterator smootherIt = smootherChoiceValues.begin(); smootherIt != smootherChoiceValues.end(); smootherIt++) {
        GMGOperator::SmootherChoice smoother = *smootherIt;

        vector<int> overlapValues;
        if (precondition && ((smoother==GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ) || (smoother==GMGOperator::IFPACK_ADDITIVE_SCHWARZ))) {
          if (overlapLevel == -1) {
            overlapValues.push_back(0);
            overlapValues.push_back(1);
            if (spaceDim < 3) overlapValues.push_back(2);
          } else {
            overlapValues.push_back(overlapLevel);
          }
        } else {
          overlapValues.push_back(0);
        }

        // smoother choice description
        string S_str = smootherString(smoother);
        string M_str; // preconditioner descriptor for output
        if (!precondition) {
          M_str = "None";
        } else {
          if (schwarzOnly) {
            M_str = S_str;
            S_str = "-"; // no smoother
          } else {
            M_str = "GMG";
          }
        }

        for (vector<int>::iterator overlapValueIt = overlapValues.begin(); overlapValueIt != overlapValues.end(); overlapValueIt++) {
          int overlapValue = *overlapValueIt;
          for (vector<int>::iterator numCellsValueIt = numCellsValues.begin(); numCellsValueIt != numCellsValues.end(); numCellsValueIt++) {
            int numCells1D = *numCellsValueIt;
            for (vector<int>::iterator kValueIt = kValues.begin(); kValueIt != kValues.end(); kValueIt++) {
              int k = *kValueIt;

              int iterationCount;
              bool reportEnergyError = false;
              double solveTime;
              bool writeAndExit = false; // not supported for runMany (since it always writes to the same disk location)
              run(problemChoice, iterationCount, spaceDim, numCells1D, k, delta_k, conformingTraces,
                  useStaticCondensation, precondition, schwarzOnly, smoother, overlapValue,
                  schwarzBlockFactorization, schwarzLevelOfFill, schwarzFillRatio, coarseSolverChoice,
                  cgTol, cgMaxIterations, aztecOutputLevel, reportTimings, solveTime,
                  reportEnergyError, numCellsRootMesh, hOnly, useZeroMeanConstraints, writeAndExit);

              int numCells = pow((double)numCells1D, spaceDim);

              ostringstream thisResult;

              thisResult << M_str << "\t" << S_str << "\t" << overlapValue << "\t" << numCells << "\t";
              thisResult << numCells1D << "\t" << k << "\t" << iterationCount << "\t" << solveTime << endl;

              if (rank==0) cout << thisResult.str();

              results << thisResult.str();
            }
          }
        }
      }
      if (rank==0) cout << results.str(); // output results so far
    }
  }

  if (rank == 0) {
    ostringstream filename;
    filename << problemChoiceString << "Driver" << spaceDim << "D_";
    filename << preconditionerChoiceString;
    if (schwarzBlockFactorization != GMGOperator::Direct)
      filename << "_schwarzFactorization_" << getFactorizationTypeString(schwarzBlockFactorization);
    if (overlapLevel != -1) {
      filename << "_overlap" << overlapLevel;
    }
    if (k != -1) {
      filename << "_k" << k;
    }

    if (hOnly) {
      filename << "_hOnly";
    }

    // if coarse solver is not direct, then include in the file name:
    if ((coarseSolverChoice != Solver::KLU) && (coarseSolverChoice != Solver::MUMPS) && (coarseSolverChoice != Solver::SuperLUDist))
      filename << "_coarseSolver_" << Solver::solverChoiceString(coarseSolverChoice);
    filename << "_results.dat";
    ofstream fout(filename.str().c_str());
    fout << results.str();
    fout.close();
    cout << "Wrote results to " << filename.str() << ".\n";
  }
}

int main(int argc, char *argv[]) {
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
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

  int k = -1; // poly order for field variables
  int delta_k = -1;   // test space enrichment; -1 for default detection (defaults to spaceDim)

  bool conformingTraces = false;
  bool precondition = true;

  int numCells = 2;
  int numCellsRootMesh = -1;

  maxDofsForKLU = 2000; // used when defining coarsest solve on 3-level solver -- will use SuperLUDist if not KLU
  coarseCGTol = 1e-6;
  coarseMaxIterations = 2000;

  int AztecOutputLevel = 1;
  int cgMaxIterations = 25000;
  int schwarzOverlap = -1;

  int spaceDim = 1;

  bool useCondensedSolve = false;

  string smootherChoiceStr = "IfPack-Schwarz";

  bool schwarzOnly = false;

  double cgTol = 1e-10;

  double fillRatio = 5;
  int levelOfFill = 2;

  bool runAutomatic = false;

  bool reportTimings = false;

  bool hOnly = false;

  bool useZeroMeanConstraints = false;

  bool writeAndExit = false;

  string schwarzFactorizationTypeString = "Direct";

  string problemChoiceString = "Poisson";

  string coarseSolverChoiceString = "KLU";

  string runManySubsetString = "All";

  int runManyMinCells = 2;
  int maxCells = -1;

  cmdp.setOption("problem",&problemChoiceString,"problem choice: Poisson, ConvectionDiffusion, Stokes, Navier-Stokes");

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("coarseSolver", &coarseSolverChoiceString, "coarse solver choice: KLU, MUMPS, SuperLUDist, SimpleML");

  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);

  cmdp.setOption("useSchwarzPreconditioner", "useGMGPreconditioner", &schwarzOnly);
  cmdp.setOption("smoother", &smootherChoiceStr);

  cmdp.setOption("hOnly", "notHOnly", &hOnly);

  cmdp.setOption("schwarzFactorization", &schwarzFactorizationTypeString, "Schwarz block factorization strategy: Direct, IC, ILU");
  cmdp.setOption("schwarzFillRatio", &fillRatio, "Schwarz block factorization: fill ratio for IC");
  cmdp.setOption("schwarzLevelOfFill", &levelOfFill, "Schwarz block factorization: level of fill for ILU");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("precondition", "dontPrecondition", &precondition);

  cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");

  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");

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

  cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");

  cmdp.setOption("gmgOperatorDebug", "gmgOperatorNormal", &runGMGOperatorInDebugMode, "Run GMGOperator in a debug mode");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  ProblemChoice problemChoice;

  if (problemChoiceString == "Poisson") {
    problemChoice = Poisson;
  } else if (problemChoiceString == "ConvectionDiffusion") {
    problemChoice = ConvectionDiffusion;
  } else if (problemChoiceString == "Stokes") {
    problemChoice = Stokes;
  } else if (problemChoiceString == "Navier-Stokes") {
    if (rank==0) cout << "Navier-Stokes not yet supported by this driver!\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  } else {
    if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  GMGOperator::SmootherChoice smootherChoice;

  if (smootherChoiceStr == "None") {
    smootherChoice = GMGOperator::NONE;
  } else if (smootherChoiceStr == "IfPack-Schwarz") {
    smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
  } else if (smootherChoiceStr == "Camellia-Schwarz") {
    smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;
  } else if (smootherChoiceStr == "Point-Jacobi") {
    smootherChoice = GMGOperator::POINT_JACOBI;
  } else if (smootherChoiceStr == "Block-Jacobi") {
    smootherChoice = GMGOperator::BLOCK_JACOBI;
  } else if (smootherChoiceStr == "Point-Gauss-Seidel") {
    smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
  } else if (smootherChoiceStr == "Block-Gauss-Seidel") {
    smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL;
  } else {
    if (rank==0) cout << "Smoother choice string not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  Solver::SolverChoice coarseSolverChoice = Solver::solverChoiceFromString(coarseSolverChoiceString);

  RunManyPreconditionerChoices runManySubsetChoice;

  if (runManySubsetString == "All") {
    runManySubsetChoice = All;
  } else if (runManySubsetString == "DontPrecondition") {
    runManySubsetChoice = DontPrecondition;
  } else if (runManySubsetString == "AllGMG") {
    runManySubsetChoice = AllGMG;
  } else if (runManySubsetString == "AllSchwarz") {
    runManySubsetChoice = AllSchwarz;
  } else if (runManySubsetString == "AlgebraicSchwarz") {
    runManySubsetChoice = AlgebraicSchwarz;
  } else if (runManySubsetString == "GeometricSchwarz") {
    runManySubsetChoice = GeometricSchwarz;
  } else if (runManySubsetString == "GMGAlgebraicSchwarz") {
    runManySubsetChoice = GMGAlgebraicSchwarz;
  } else if (runManySubsetString == "GMGGeometricSchwarz") {
    runManySubsetChoice = GMGGeometricSchwarz;
  } else if (runManySubsetString == "GMGPointGaussSeidel") {
    runManySubsetChoice = GMGPointSymmetricGaussSeidel;
  } else if (runManySubsetString == "GMGPointJacobi") {
    runManySubsetChoice = GMGPointJacobi;
  } else {
    if (rank==0) cout << "Run many subset string not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  GMGOperator::FactorType schwarzFactorType = getFactorizationType(schwarzFactorizationTypeString);

  if (delta_k==-1) delta_k = spaceDim;

  if (! runAutomatic) {
    int iterationCount;
    bool reportEnergyError = true;

    if (schwarzOverlap == -1) schwarzOverlap = 0;
    if (k == -1) k = 2;

    double solveTime;

    run(problemChoice, iterationCount, spaceDim, numCells, k, delta_k, conformingTraces,
        useCondensedSolve, precondition, schwarzOnly, smootherChoice, schwarzOverlap,
        schwarzFactorType, levelOfFill, fillRatio, coarseSolverChoice,
        cgTol, cgMaxIterations, AztecOutputLevel, reportTimings, solveTime, reportEnergyError, numCellsRootMesh, hOnly, useZeroMeanConstraints, writeAndExit);

    if (rank==0) cout << "Iteration count: " << iterationCount << "; solve time " << solveTime << " seconds." << endl;
  } else {
    if (rank==0) {
      cout << "Running " << problemChoiceString << " solver in automatic mode (subset: ";
      cout << runManySubsetString << "), with spaceDim " << spaceDim;
      cout << ", delta_k = " << delta_k << ", ";
      if (conformingTraces)
        cout << "conforming traces, ";
      else
        cout << "non-conforming traces, ";
      cout << "CG tolerance = " << cgTol << ", max iterations = " << cgMaxIterations << endl;
    }

    if (maxCells == -1) {
      maxCells = Teuchos::GlobalMPISession::getNProc(); // by default, ensure max of 1 cell per MPI node
    }
    runMany(problemChoice, spaceDim, delta_k, runManyMinCells,
            conformingTraces, useCondensedSolve,
            schwarzFactorType, levelOfFill, fillRatio,
            coarseSolverChoice,
            cgTol, cgMaxIterations, AztecOutputLevel,
            runManySubsetChoice, k, schwarzOverlap, numCellsRootMesh, reportTimings, hOnly, maxCells, useZeroMeanConstraints);
  }
  return 0;
}
