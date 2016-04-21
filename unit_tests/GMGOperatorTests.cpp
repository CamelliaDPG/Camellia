//
//  GMGOperatorTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/4/15.
//
//

#include "EpetraExt_RowMatrixOut.h"
#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaDebugUtility.h"
#include "CamelliaTestingHelpers.h"
//#include "ConvectionDiffusionReactionFormulation.h"
#include "GDAMinimumRule.h"
#include "GMGOperator.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "NavierStokesVGPFormulation.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "SpaceTimeHeatDivFormulation.h"
#include "StokesVGPFormulation.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
  enum FormulationChoice {
    Poisson, Stokes
  };
  
  double getSchwarzWeight(MeshPtr mesh, int overlapLevel)
  {
    bool useStaticCondensation = false;
    
    BFPtr bf = mesh->bilinearForm();
    BCPtr bc = BC::bc();
    IPPtr ip = bf->graphNorm();
    RHSPtr rhs = RHS::rhs();
    
    SolutionPtr fineSoln = Solution::solution(mesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    
    GMGOperator gmgOperator(bc,mesh,ip,mesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.setSmootherOverlap(overlapLevel);
    
    return gmgOperator.computeSchwarzSmootherWeight();
  }
  
  void testEpetraMatrixIsIdentity(Teuchos::RCP<Epetra_CrsMatrix> P, Teuchos::FancyOStream &out, bool &success)
  {
    int myRowCount = P->Map().NumMyElements();
    
    GlobalIndexTypeToCast globalEntryCount = P->Map().NumGlobalElements();
    Intrepid::FieldContainer<GlobalIndexTypeToCast> colIndices(globalEntryCount);
    Intrepid::FieldContainer<double> colValues(globalEntryCount);
    
//    printMapSummary(P->Map(), "P->Map()");
    
    for (int localRowIndex=0; localRowIndex<myRowCount; localRowIndex++)
    {
      GlobalIndexTypeToCast globalRowIndex = P->Map().GID(localRowIndex);
      int numEntries;
      P->ExtractMyRowCopy(localRowIndex, globalEntryCount, numEntries, &colValues(0), &colIndices(0));
      bool diagEntryFound = false;
      
      double oneTol=1e-12;
      double zeroTol=1e-10; // how small the off-diagonal entries should be
      for (int colEntryOrdinal=0; colEntryOrdinal<numEntries; colEntryOrdinal++)
      {
        GlobalIndexTypeToCast localColIndex = colIndices(colEntryOrdinal);
        GlobalIndexTypeToCast globalColIndex = P->DomainMap().GID(localColIndex);
        double expectedValue;
        double actualValue = colValues(colEntryOrdinal);
        if (globalColIndex != globalRowIndex)
        {
          // expect 0 for off-diagonals
          if (abs(actualValue) > zeroTol)
          {
            out << "FAILURE: off-diagonal (" << globalRowIndex << "," << globalColIndex << ") = " << actualValue << " ≠ 0\n";
            success = false;
          }
//          TEST_COMPARE(abs(actualValue), <, tol);
        }
        else
        {
          // expect 1 on the diagonal
          expectedValue = 1.0;
          if (abs(expectedValue - actualValue) > oneTol)
          {
            out << "FAILURE: diagonal (" << globalRowIndex << "," << globalColIndex << ") = " << actualValue << " ≠ 1.0 (diff = ";
            out << abs(expectedValue-actualValue) << ")\n";
            success = false;
          }
          diagEntryFound = true;
        }
      }
      if (!diagEntryFound)
      {
        int rank = Teuchos::GlobalMPISession::getRank();
        out << "on rank " << rank << ", no diagonal entry found for global row " << globalRowIndex;
        out << " (num col entries: " << numEntries << ")\n";
        success = false;
      }
    }
    
//    { // DEBUGGING: export to disk
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";
//    }
  }

  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorHangingNode_2D)
  {
    // take a mesh with a hanging node, using the same mesh for coarse and fine in a GMGOperator.
    // test that the prolongation operator is the identity
    int spaceDim = 2;
    bool useConformingTraces = true;
    int H1Order = useConformingTraces ? 1 : 2; // make trace variables linear
    bool useStaticCondensation = false;
    vector<int> cellCounts = {1,2}; // two elements
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);
    
    // refine element 0:
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    BCPtr bc = BC::bc();
    IPPtr ip = bf->graphNorm();
    RHSPtr rhs = RHS::rhs();
    
    SolutionPtr fineSoln = Solution::solution(mesh);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setBC(bc);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    
    GMGOperator gmgOperator(bc,mesh,ip,mesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.constructProlongationOperator();
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.getProlongationOperator();
    testEpetraMatrixIsIdentity(P, out, success);
  }

  void testIdentityProlongationOperator(MeshPtr mesh, bool useStaticCondensation,
                                        Teuchos::FancyOStream &out, bool &success)
  {
    BFPtr bf = mesh->bilinearForm();
    BCPtr bc = BC::bc();
    IPPtr ip = bf->graphNorm();
    RHSPtr rhs = RHS::rhs();
    
    SolutionPtr fineSoln = Solution::solution(mesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    
    GMGOperator gmgOperator(bc,mesh,ip,mesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.constructProlongationOperator();
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.getProlongationOperator();
    testEpetraMatrixIsIdentity(P, out, success);
  }
  
  void testIdentityProlongationOperatorUniform(FormulationChoice formChoice, int spaceDim,
                                               bool useConformingTraces, bool useStaticCondensation,
                                               Teuchos::FancyOStream &out, bool &success)
  {
    // take a uniform mesh, using the same mesh for coarse and fine in a GMGOperator.
    // test that the prolongation operator is the identity
//    int H1Order = useConformingTraces ? 1 : 2; // make trace variables linear
    int H1Order = 1;

    vector<int> cellCounts(spaceDim,2);
    BFPtr bf;
    BCPtr bc = BC::bc();
    MeshPtr mesh;
    int delta_k = spaceDim;
    vector<double> dimensions(spaceDim,1.0);

    if (formChoice == Poisson)
    {
      PoissonFormulation form(spaceDim, useConformingTraces);
      bf = form.bf();
      mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);
    }
    else
    {
      double mu = 1.0;
      StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim,mu,useConformingTraces);
      bf = form.bf();
      mesh = MeshFactory::rectilinearMesh(bf, dimensions, cellCounts, H1Order, delta_k);
      bc->addSinglePointBC(form.p()->ID(), 0.0, mesh);
    }
    testIdentityProlongationOperator(mesh, useStaticCondensation, out, success);
  }
  
  void testIdentityProlongationOperatorComplexMeshSpaceTime(bool useConformingTraces, bool useStaticCondensation,
                                                            Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = 2;
    double pi = atan(1)*4;
    
    double t0 = 0;
    double t1 = pi;
    int temporalDivisions = 1;
    
    vector<double> x0 = {0.0, 0.0};;
    vector<double> dims = {2*pi, 2*pi};
    vector<int> numElements = {2,2};
    
    MeshTopologyPtr spatial2DMeshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
    MeshTopologyPtr meshTopo = MeshFactory::spaceTimeMeshTopology(spatial2DMeshTopo, t0, t1, temporalDivisions);
    
    // some refinements in an effort to replicate an issue...
    // 1. Uniform refinement
    IndexType nextElement = meshTopo->cellCount();
    set<IndexType> cellsToRefine = meshTopo->getActiveCellIndices();
    CellTopoPtr cellTopo = meshTopo->getCell(0)->topology();
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
    for (IndexType cellIndex : cellsToRefine)
    {
      meshTopo->refineCell(cellIndex, refPattern, nextElement);
      nextElement += refPattern->numChildren();
    }
    // 2. Selective refinement
    cellsToRefine = {4,15,21,30};
    for (IndexType cellIndex : cellsToRefine)
    {
      meshTopo->refineCell(cellIndex, refPattern, nextElement);
      nextElement += refPattern->numChildren();
    }
    
    int fieldPolyOrder = 1;
    double epsilon = 1.0;
    SpaceTimeHeatDivFormulation form(spaceDim, epsilon);
    form.initializeSolution(meshTopo, fieldPolyOrder);
    
    MeshPtr formMesh = form.solution()->mesh();
    testIdentityProlongationOperator(formMesh, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorUniform_2D )
  {
    int spaceDim = 2;
    bool useConformingTraces = true;
    bool useStaticCondensation = false;

    testIdentityProlongationOperatorUniform(Poisson, spaceDim, useConformingTraces, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorUniform_StokesCondensed2D )
  {
    int spaceDim = 2;
    bool useConformingTraces = false;
    bool useStaticCondensation = true;
    
    testIdentityProlongationOperatorUniform(Stokes, spaceDim, useConformingTraces, useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorHangingNodeMesh_Triangles )
  {
    int polyOrder = 1, delta_k = 1;
    int spaceDim = 2;
    bool useTriangles = true;
    
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    // bilinear form
    BFPtr bf = form.bf();
    
    // set up mesh
    int H1Order = polyOrder + 1;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,2};
    
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    set<GlobalIndexType> cellsToRefine = {2};
    mesh->hRefine(cellsToRefine);
    
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//    minRule->printGlobalDofInfo();
//    mesh->getTopology()->printAllEntitiesInBaseMeshTopology();
    
    bool useStaticCondensation = false;
    testIdentityProlongationOperator(mesh, useStaticCondensation, out, success);
  }

//  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorHangingNodeComplexMesh_Triangles )
//  {
//    /* 
//     This test follows a particular case that caused issues with convection-diffusion.
//     
//     It's sufficiently computationally intensive that probably it isn't worth keeping in the main test suite.
//     
//     */
//    
//    int polyOrder = 1, delta_k = 1;
//    int spaceDim = 2;
//    double epsilon = 1e-2;
//    double beta_1 = 2.0, beta_2 = 1.0;
//    bool useTriangles = true; // otherwise, quads
//    
//    double alpha = 0; // no reaction term
//    FunctionPtr beta = Function::constant({beta_1,beta_2});
//    
//    ConvectionDiffusionReactionFormulation::FormulationChoice formulation;
//    formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
//    
//    ConvectionDiffusionReactionFormulation form(formulation, spaceDim, beta, epsilon, alpha);
//    
//    // bilinear form
//    BFPtr bf = form.bf();
//    
//    // set up mesh
//    int H1Order = polyOrder + 1;
//    vector<double> dimensions = {1.0,1.0};
//    vector<int> elementCounts = {8,8};
//    
//    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
//                                                elementCounts[0], elementCounts[1], useTriangles);
//    
//    set<GlobalIndexType> cellsToRefine = {1, 3, 5, 7, 8, 9, 10, 11, 13, 47, 63, 78, 79, 94, 95, 110, 111, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127};
//    mesh->hRefine(cellsToRefine,false);
//    mesh->enforceOneIrregularity();
//    
//    cellsToRefine = {0, 15, 16, 31, 46, 62, 128, 132, 134, 136, 138, 140, 142, 148, 150, 156, 158, 160, 162, 164, 167, 168, 169, 171, 175, 176, 179, 183, 184, 185, 187, 191, 192, 193, 195, 199, 202, 203, 207, 209, 210, 211, 215, 217, 218, 219, 223, 225, 226, 227, 231, 233, 234, 235, 239, 241, 242, 243, 244, 245, 247};
//    mesh->hRefine(cellsToRefine,false);
//    mesh->enforceOneIrregularity();
//    
//    cellsToRefine = {32, 48, 64, 112, 130, 165, 177, 198, 201, 260, 263, 267, 271, 324, 327, 328, 331, 332, 340, 343, 348, 349, 350, 351, 352, 354, 355, 359, 360, 363, 367, 368, 369, 371, 375, 376, 377, 379, 383, 384, 385, 387, 390, 391, 394, 395, 398, 399, 403, 406, 409, 410, 411, 413, 414, 415, 419, 422, 425, 426, 427, 429, 430, 431, 435, 438, 441, 442, 443, 445, 446, 447, 451, 454, 457, 458, 459, 461, 462, 463, 467, 470, 473, 474, 475, 477, 478, 479, 480, 481, 483, 487, 488, 489, 491, 519};
//    mesh->hRefine(cellsToRefine,false);
//    mesh->enforceOneIrregularity();
//    
//    cellsToRefine = {80, 96, 252, 254, 255, 261, 325, 329, 333, 335, 339, 341, 345, 347, 353, 361, 362, 389, 393, 397, 511, 534, 535, 543, 545, 547, 549, 550, 551, 554, 556, 558, 559, 560, 563, 567, 571, 572, 575, 576, 579, 580, 583, 584, 586, 587, 588, 590, 591, 592, 594, 595, 596, 598, 599, 600, 602, 612, 613, 614, 615, 616, 617, 618, 619, 624, 625, 626, 627, 632, 633, 634, 635, 636, 638, 639, 644, 647, 652, 655, 659, 660, 663, 668, 669, 671, 675, 676, 677, 679, 683, 684, 685, 687, 690, 691, 692, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 715, 718, 722, 725, 726, 727, 729, 730, 731, 734, 737, 738, 739, 741, 742, 743, 747, 750, 754, 757, 758, 759, 761, 762, 763, 766, 769, 770, 771, 773, 774, 775, 779, 782, 786, 789, 790, 791, 793, 794, 795, 798, 801, 802, 803, 805, 806, 807, 811, 814, 818, 821, 822, 823, 825, 826, 827, 830, 833, 834, 835, 837, 838, 839, 843, 845, 846, 849, 850, 852, 853, 854, 855, 856, 857, 858, 859, 862, 864, 865, 866, 867, 869, 870, 871, 872, 873, 875, 879, 880, 881, 883, 887, 888, 889, 891, 895, 896, 897, 899, 903, 907, 910, 915};
//    mesh->hRefine(cellsToRefine,false);
//    mesh->enforceOneIrregularity();
//    
//    bool useStaticCondensation = true;
//    testIdentityProlongationOperator(mesh, useStaticCondensation, out, success);
//  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorComplexMesh_SpaceTime_Slow )
  {
    bool useConformingTraces = true;
    bool useStaticCondensation = false;
    
    testIdentityProlongationOperatorComplexMeshSpaceTime(useConformingTraces, useStaticCondensation, out, success);
  }
  
  // commented out this test because it's kind of expensive (and has always passed)
//  TEUCHOS_UNIT_TEST( GMGOperator, IdentityProlongationOperatorComplexMesh_SpaceTimeCondensed )
//  {
//    bool useConformingTraces = true;
//    bool useStaticCondensation = true;
//    
//    testIdentityProlongationOperatorComplexMeshSpaceTime(useConformingTraces, useStaticCondensation, out, success);
//  }
  
  void testProlongationOperatorLine(bool useStaticCondensation, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     
     Take a 1D, 1-element mesh with an exact solution.
     Refine.
     Compute the GMGOperator's prolongation of the coarse exact solution onto the fine mesh.
     Check that the solution on the fine mesh is exact.
     
     */
    
    int spaceDim = 1;
    bool useConformingTraces = false;
    int H1Order = 2, delta_k = spaceDim;
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
    FunctionPtr phi_exact;
    if (H1Order >= 3)
      phi_exact = Function::xn(2); // x^2 exact solution
    else
      phi_exact = Function::constant(2); // constant exact solution
    FunctionPtr psi_exact = phi_exact->dx();
    
    double xLeft = 0.0, xRight = 1.0;
    int coarseElementCount = 1;

    MeshPtr mesh = MeshFactory::intervalMesh(bf, xLeft, xRight, coarseElementCount, H1Order, delta_k);
    // for debugging, do a refinement first:
    //  mesh->hRefine(mesh->getActiveCellIDs());
    
    SolutionPtr coarseSoln = Solution::solution(mesh);
    
    BCPtr bc = BC::bc();
    bc->addDirichlet(phi_hat, SpatialFilter::allSpace(), phi_exact);
    VarPtr q = form.q();
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(phi_exact->dx()->dx() * q);
    coarseSoln->setRHS(rhs);
    coarseSoln->setBC(bc);
    coarseSoln->setUseCondensedSolve(useStaticCondensation);
    
    IPPtr ip = bf->graphNorm();
    coarseSoln->setIP(ip);
    
    FunctionPtr n = Function::normal_1D();
    FunctionPtr parity = Function::sideParity();
    
    map<int, FunctionPtr> exactSolnMap;
    exactSolnMap[phi->ID()] = phi_exact;
    exactSolnMap[psi->ID()] = psi_exact;
    exactSolnMap[phi_hat->ID()] = phi_exact * n * parity;
    exactSolnMap[psi_n_hat->ID()] = psi_exact * n * parity;
    
    coarseSoln->projectOntoMesh(exactSolnMap);
    
    { // DEBUGGING
      bool warnAboutOffRank = false;
      set<GlobalIndexType> myCellIDs = coarseSoln->mesh()->cellIDsInPartition();
      for (GlobalIndexType cellID : myCellIDs) {
        FieldContainer<double> coefficients = coarseSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
        out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
        DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
        printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
      }
    }
    
    double energyError = coarseSoln->energyErrorTotal();
    
    // sanity check: our exact solution should give us 0 energy error
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
    if (energyError > tol)
    {
      map<GlobalIndexType, double> errorForMyCells = coarseSoln->rankLocalEnergyError();
      for (auto entry : errorForMyCells)
      {
        out << "Error for cell " << entry.first << ": " << entry.second << endl;
      }
      out << *coarseSoln->getLHSVector();
    }
    
    MeshPtr fineMesh = mesh->deepCopy();
    
    fineMesh->hRefine(fineMesh->getActiveCellIDs());
    //  fineMesh->hRefine(fineMesh->getActiveCellIDs());
    
    SolutionPtr fineSoln = Solution::solution(fineMesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    // again, a sanity check, now on the fine solution:
    fineSoln->projectOntoMesh(exactSolnMap);
    bool warnAboutOffRank = false;
    set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
      DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
    }
    
    energyError = fineSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    GMGOperator gmgOperator(bc,mesh,ip,fineMesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
    Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();
    
    fineSoln->initializeLHSVector();
    fineSoln->getLHSVector()->PutScalar(0); // set a 0 global solution
    
    fineSoln->importSolution();      // imports the 0 solution onto each cell
    fineSoln->clearComputedResiduals();
    
    P->Multiply(false, *coarseSolutionVector, *fineSoln->getLHSVector());
    
//    cout << "coarseSolutionVector:\n"  << *coarseSolutionVector;
//    cout << "P * coarseSolutionVector:\n"  << *fineSoln->getLHSVector();
    
    fineSoln->importSolution();
    
    energyError = fineSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (fineSoln) ********************\n\n";
      DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
    }

//    { // DEBUGGING: export to disk
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P,NULL,NULL,false);
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "wrote prolongation operator matrix to /tmp/P.dat\n";
//      
//      GDAMinimumRule* fineGDA = dynamic_cast<GDAMinimumRule*>(fineMesh->globalDofAssignment().get());
//      fineGDA->printGlobalDofInfo();
//      
//      GDAMinimumRule* coarseGDA = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//      coarseGDA->printGlobalDofInfo();
//    }

  }

  void testProlongationOperatorQuad(bool simple, bool useStaticCondensation, bool useConformingTraces,
                                    bool testHMultigrid, int levels, bool uniform,
                                    Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = 2;
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    VarPtr phi = form.phi(), psi = form.psi(), phi_hat = form.phi_hat(), psi_n_hat = form.psi_n_hat();
    
    int H1Order, delta_k = 1;
    
    map<int,int> trialOrderEnhancements;
    
    FunctionPtr phi_exact, psi_exact;
    if (simple)
    {
      H1Order = 1;
      phi_exact = Function::constant(3.14159);
      FunctionPtr zero = Function::zero();
      psi_exact = Function::vectorize(zero, zero);
    }
    else
    {
      H1Order = 3;
      phi_exact = Function::xn(2) + Function::yn(1);
      psi_exact = phi_exact->grad();
    }
    
//    if (useConformingTraces)
//    {
//      trialOrderEnhancements[phi->ID()] = 1;
//    }
    
    int coarseElementCount = testHMultigrid ? 1 : 2;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,coarseElementCount);
    vector<double> x0(spaceDim,0.0);
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0, trialOrderEnhancements);
    
    SolutionPtr coarseSoln = Solution::solution(mesh);
    BCPtr bc = BC::bc();
    coarseSoln->setBC(bc);
    coarseSoln->setUseCondensedSolve(useStaticCondensation);
    
    VarPtr q = form.q();
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
    rhs->addTerm(f * q);
    coarseSoln->setRHS(rhs);
    IPPtr ip = bf->graphNorm();
    coarseSoln->setIP(ip);
    coarseSoln->setUseCondensedSolve(useStaticCondensation);
    
    map<int, FunctionPtr> exactSolnMap;
    exactSolnMap[phi->ID()] = phi_exact;
    exactSolnMap[psi->ID()] = psi_exact;
    
    FunctionPtr phi_hat_exact   =   phi_hat->termTraced()->evaluate(exactSolnMap);
    FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactSolnMap);
    
    exactSolnMap[phi_hat->ID()]   = phi_hat_exact;
    exactSolnMap[psi_n_hat->ID()] = psi_n_hat_exact;
    
    coarseSoln->projectOntoMesh(exactSolnMap);
    
    double energyError = coarseSoln->energyErrorTotal();
    
    // sanity check: our exact solution should give us 0 energy error
    double tol = 1e-12;
    TEST_COMPARE(energyError, <, tol);
    
    MeshPtr fineMesh = mesh->deepCopy();
    GlobalIndexType lastRefinedCellID = -1;
    for (int i=0; i<levels; i++)
    {
      set<GlobalIndexType> cellsToRefine;
      if (uniform)
      {
        cellsToRefine = fineMesh->getActiveCellIDs();
      }
      else
      {
        // refine the one we last refined (in p case), or its child (h case)
        if (lastRefinedCellID == -1)
        {
          // then refine the first active guy:
          lastRefinedCellID = *fineMesh->getActiveCellIDs().begin();
        }
        else if (testHMultigrid)
        {
          CellPtr lastRefinedCell = fineMesh->getTopology()->getCell(lastRefinedCellID);
          lastRefinedCellID = lastRefinedCell->getChildIndices(fineMesh->getTopology())[0];
        }
        cellsToRefine.insert(lastRefinedCellID);
      }
      
      if (testHMultigrid)
      {
        fineMesh->hRefine(cellsToRefine);
        fineMesh->enforceOneIrregularity();
      }
      else
        fineMesh->pRefine(cellsToRefine);
    }
    
    SolutionPtr exactSoln = Solution::solution(fineMesh);
    exactSoln->setIP(ip);
    exactSoln->setRHS(rhs);
    
    // again, a sanity check, now on the exact fine solution:
    exactSoln->projectOntoMesh(exactSolnMap);
    energyError = exactSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    SolutionPtr fineSoln = Solution::solution(fineMesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    GMGOperator gmgOperator(bc,mesh,ip,fineMesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
    Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();
    
    fineSoln->initializeLHSVector();
    fineSoln->getLHSVector()->PutScalar(0); // set a 0 global solution
    fineSoln->importSolution();      // imports the 0 solution onto each cell
    fineSoln->clearComputedResiduals();
    
    P->Multiply(false, *coarseSolutionVector, *fineSoln->getLHSVector());
    
    fineSoln->importSolution();
    
    set<GlobalIndexType> cellIDs = fineSoln->mesh()->getActiveCellIDs();
    
    energyError = fineSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    bool warnAboutOffRank = false;
    VarFactoryPtr vf = bf->varFactory();
    //    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    //      cout << "\n\n******************** Dofs for cell " << *cellIDIt << " (fineSoln before subtracting exact) ********************\n";
    //      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(*cellIDIt, warnAboutOffRank);
    //      DofOrderingPtr trialOrder = fineMesh->getElementType(*cellIDIt)->trialOrderPtr;
    //      printLabeledDofCoefficients(vf, trialOrder, coefficients);
    //    }
    
    set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
    
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = exactSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
      DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
    }
    
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (fineSoln) ********************\n\n";
      DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, form.bf()->varFactory(), trialOrder, coefficients);
    }
    
    fineSoln->addSolution(exactSoln, -1.0); // should recover zero solution this way
    
    // import global solution data onto each rank:
    fineSoln->importSolutionForOffRankCells(cellIDs);
    
    for (GlobalIndexType cellID : cellIDs) {
      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      FieldContainer<double> expectedCoefficients(coefficients.size()); // zero coefficients
      
      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(coefficients, expectedCoefficients, tol, tol);
      //    cout << "\n\n******************** Dofs for cell " << cellID << " (fineSoln after subtracting exact) ********************\n\n";
      //    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      //    printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, coefficients);
    }
  }

  void testQuadFineCoarseSwap(bool simple, bool useStaticCondensation, Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = 2;
    PoissonFormulation formConforming(spaceDim, true);
    PoissonFormulation formNonconforming(spaceDim, false);
    BFPtr bfConforming = formConforming.bf();
    BFPtr bfNonconforming = formNonconforming.bf();
    VarPtr phi = formConforming.phi(), psi = formConforming.psi(), phi_hat = formConforming.phi_hat(), psi_n_hat = formConforming.psi_n_hat();
    
    int H1Order, delta_k = 1;
    
    FunctionPtr phi_exact, psi_exact;
    if (simple)
    {
      H1Order = 1;
      phi_exact = Function::constant(3.14159);
      FunctionPtr zero = Function::zero();
      psi_exact = Function::vectorize(zero, zero);
    }
    else
    {
      H1Order = 3;
      phi_exact = Function::xn(2) + Function::yn(1);
      psi_exact = phi_exact->grad();
    }
    
    int coarseElementCount = 1;
    vector<double> dimensions(2,1.0);
    vector<int> elementCounts(2,coarseElementCount);
    MeshPtr meshConforming = MeshFactory::rectilinearMesh(bfConforming, dimensions, elementCounts, H1Order, delta_k);
    map<int,int> trialEnhancements;
    trialEnhancements[phi_hat->ID()] = 1;
    MeshPtr meshNonconforming = Teuchos::rcp( new Mesh(meshConforming->getTopology(), bfNonconforming, H1Order, delta_k, trialEnhancements) );
    
    SolutionPtr coarseSoln = Solution::solution(meshNonconforming);
    BCPtr bc = BC::bc();
    coarseSoln->setBC(bc);
    coarseSoln->setUseCondensedSolve(useStaticCondensation);
    
    VarPtr q = formConforming.q();
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
    rhs->addTerm(f * q);
    coarseSoln->setRHS(rhs);
    IPPtr ip = bfConforming->graphNorm();
    coarseSoln->setIP(ip);
    coarseSoln->setUseCondensedSolve(useStaticCondensation);
    
    map<int, FunctionPtr> exactSolnMap;
    exactSolnMap[phi->ID()] = phi_exact;
    exactSolnMap[psi->ID()] = psi_exact;
    
    FunctionPtr phi_hat_exact   =   phi_hat->termTraced()->evaluate(exactSolnMap);
    FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactSolnMap);
    
    exactSolnMap[phi_hat->ID()]   = phi_hat_exact;
    exactSolnMap[psi_n_hat->ID()] = psi_n_hat_exact;
    
    coarseSoln->projectOntoMesh(exactSolnMap);
    
    double energyError = coarseSoln->energyErrorTotal();
    
    // sanity check: our exact solution should give us 0 energy error
    double tol = 1e-12;
    TEST_COMPARE(energyError, <, tol);
    
    MeshPtr fineMesh = meshConforming;
    MeshPtr coarseMesh = meshNonconforming;
    
    SolutionPtr fineSoln = Solution::solution(fineMesh);
    fineSoln->setBC(bc);
    fineSoln->setIP(ip);
    fineSoln->setRHS(rhs);
    fineSoln->setUseCondensedSolve(useStaticCondensation);
    
    SolverPtr coarseSolver = Solver::getSolver(Solver::KLU, true);
    GMGOperator gmgOperator(bc,meshNonconforming,ip,fineMesh,fineSoln->getDofInterpreter(),
                            fineSoln->getPartitionMap(),coarseSolver, useStaticCondensation);
    gmgOperator.setFineCoarseRolesSwapped(true);
    
    Teuchos::RCP<Epetra_CrsMatrix> P = gmgOperator.constructProlongationOperator();
    Teuchos::RCP<Epetra_FEVector> coarseSolutionVector = coarseSoln->getLHSVector();
    
    cout << "Outputting P to file.\n";
    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*P, NULL, NULL, false); // false: don't write header
    
    fineSoln->projectOntoMesh(exactSolnMap);
    coarseSoln->initializeLHSVector();
    coarseSoln->getLHSVector()->PutScalar(0); // set a 0 global solution
    coarseSoln->importSolution();             // imports the 0 solution onto each cell
    coarseSoln->clearComputedResiduals();
    
    P->Multiply(true, *fineSoln->getLHSVector(), *coarseSolutionVector); // true: transpose
    
    coarseSoln->importSolution();
    
    set<GlobalIndexType> cellIDs = coarseSoln->mesh()->getActiveCellIDs();
    
    energyError = coarseSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    bool warnAboutOffRank = false;
//    VarFactoryPtr vf = bf->varFactory();
    //    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    //      cout << "\n\n******************** Dofs for cell " << *cellIDIt << " (fineSoln before subtracting exact) ********************\n";
    //      FieldContainer<double> coefficients = fineSoln->allCoefficientsForCellID(*cellIDIt, warnAboutOffRank);
    //      DofOrderingPtr trialOrder = fineMesh->getElementType(*cellIDIt)->trialOrderPtr;
    //      printLabeledDofCoefficients(vf, trialOrder, coefficients);
    //    }
    
    set<GlobalIndexType> myCellIDs = fineSoln->mesh()->cellIDsInPartition();
    
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = coarseSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (coarseSoln) ********************\n\n";
      DofOrderingPtr trialOrder = coarseMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, formConforming.bf()->varFactory(), trialOrder, coefficients);
    }
    
    SolutionPtr exactSoln = Solution::solution(coarseMesh);
    exactSoln->setIP(ip);
    exactSoln->setRHS(rhs);
    
    // again, a sanity check, now on the exact fine solution:
    exactSoln->projectOntoMesh(exactSolnMap);
    energyError = exactSoln->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);
    
    for (GlobalIndexType cellID : myCellIDs) {
      FieldContainer<double> coefficients = exactSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      out << "\n\n******************** Dofs for cell " << cellID << " (exactSoln) ********************\n\n";
      DofOrderingPtr trialOrder = coarseMesh->getElementType(cellID)->trialOrderPtr;
      printLabeledDofCoefficients(out, formConforming.bf()->varFactory(), trialOrder, coefficients);
    }
    
    coarseSoln->addSolution(exactSoln, -1.0); // should recover zero solution this way
    
    // import global solution data onto each rank:
    coarseSoln->importSolutionForOffRankCells(cellIDs);
    
    for (GlobalIndexType cellID : cellIDs) {
      FieldContainer<double> coefficients = coarseSoln->allCoefficientsForCellID(cellID, warnAboutOffRank);
      FieldContainer<double> expectedCoefficients(coefficients.size()); // zero coefficients
      
      TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO(coefficients, expectedCoefficients, tol, tol);
      //    cout << "\n\n******************** Dofs for cell " << cellID << " (fineSoln after subtracting exact) ********************\n\n";
      //    DofOrderingPtr trialOrder = fineMesh->getElementType(cellID)->trialOrderPtr;
      //    printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, coefficients);
    }
  }

  
  // Commenting out the FineCoarseSwap test because we likely won't need this feature
  // TODO: remove the feature from GMGOperator.
  /*TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_FineCoarseSwap )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    testQuadFineCoarseSwap(simple, useStaticCondensation, out, success);
  }*/
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorLineStandardSolve )
  {
    bool useStaticCondensation = false;
    testProlongationOperatorLine(useStaticCondensation, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorLineCondensedSolve )
  {
    MPIWrapper::CommWorld()->Barrier();
    
    bool useStaticCondensation = true;
    testProlongationOperatorLine(useStaticCondensation, out, success);
  }

  //**** p-Multigrid prolongation tests on the quad *****//
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveNonconforming_p )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveConforming_p )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveNonconforming_p )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveConforming_p )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveNonconforming_p )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveConforming_p )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveNonconforming_p )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveConforming_p )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = false;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  //**** h-Multigrid tests below *****//
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveConforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveConforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveConforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveConforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 1;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }

  /*** Two-level tests ***/
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveTwoLevelNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveTwoLevelHangingNodesNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveTwoLevelConforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleCondensedSolveTwoLevelHangingNodesConforming )
  {
    bool simple = true;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveTwoLevelNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveTwoLevelHangingNodesNonconforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveTwoLevelConforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SimpleStandardSolveTwoLevelHangingNodesConforming )
  {
    bool simple = true;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveTwoLevelNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveTwoLevelHangingNodesNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveTwoLevelConforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowCondensedSolveTwoLevelHangingNodesConforming )
  {
    bool simple = false;
    bool useStaticCondensation = true;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveTwoLevelNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveTwoLevelHangingNodesNonconforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = false;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveTwoLevelConforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = true;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, ProlongationOperatorQuad_SlowStandardSolveTwoLevelHangingNodesConforming )
  {
    bool simple = false;
    bool useStaticCondensation = false;
    bool useConformingTraces = true;
    bool testHMultigrid = true;
    int levels = 2;
    bool uniform = false;
    testProlongationOperatorQuad(simple, useStaticCondensation, useConformingTraces, testHMultigrid,
                                 levels, uniform, out, success);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, SchwarzWeightUniformQuads )
  {
    int polyOrder = 1, delta_k = 1;
    int spaceDim = 2;
    bool useTriangles = false;
    
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    // bilinear form
    BFPtr bf = form.bf();
    
    // set up mesh
    int H1Order = polyOrder + 1;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    int overlap = 0;
    int maxNeighbors = 3;  // cell plus 2 neighbors, in 2x2 mesh
    double weightExpected = 1.0 / (maxNeighbors + 1);
    double weightActual = getSchwarzWeight(mesh, overlap);
    TEST_FLOATING_EQUALITY(weightActual, weightExpected, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, SchwarzWeightQuadsHangingNodes )
  {
    int polyOrder = 1, delta_k = 1;
    int spaceDim = 2;
    bool useTriangles = false;
    
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    // bilinear form
    BFPtr bf = form.bf();
    
    // set up mesh
    int H1Order = polyOrder + 1;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    int overlap = 0;
    int maxNeighbors = 5;  // the fine cell with a vertex on the center of the mesh has overlap region of 5
    double weightExpected = 1.0 / (maxNeighbors + 1);
    double weightActual = getSchwarzWeight(mesh, overlap);
    TEST_FLOATING_EQUALITY(weightActual, weightExpected, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, SchwarzWeightUniformTriangles )
  {
    int polyOrder = 1, delta_k = 1;
    int spaceDim = 2;
    bool useTriangles = true;
    
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    // bilinear form
    BFPtr bf = form.bf();
    
    // set up mesh
    int H1Order = polyOrder + 1;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    int overlap = 0;
    int maxNeighbors = 4;  // cell plus 3 neighbors (two cells have 3 neighbors)
    double weightExpected = 1.0 / (maxNeighbors + 1);
    double weightActual = getSchwarzWeight(mesh, overlap);
    TEST_FLOATING_EQUALITY(weightActual, weightExpected, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST( GMGOperator, SchwarzWeightTrianglesHangingNodes )
  {
    int polyOrder = 1, delta_k = 1;
    int spaceDim = 2;
    bool useTriangles = true;
    
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    // bilinear form
    BFPtr bf = form.bf();
    
    // set up mesh
    int H1Order = polyOrder + 1;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                                elementCounts[0], elementCounts[1], useTriangles);
    
    set<GlobalIndexType> cellsToRefine = {2};
    mesh->hRefine(cellsToRefine);
    
    int overlap = 0;
    int maxNeighbors = 4;  // the fine cell with a vertex on the center of the mesh has overlap region of 5
    double weightExpected = 1.0 / (maxNeighbors + 1);
    double weightActual = getSchwarzWeight(mesh, overlap);
    TEST_FLOATING_EQUALITY(weightActual, weightExpected, 1e-15);
    
    // now, a similar test, but this time we refine the upper left cell (cell 3)
    // In this case, cell 2 has two fine neighbors, and two coarse neighbors, for a total of 5 neighbors,
    // and no fine cell has that many neighbors.

    mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                        elementCounts[0], elementCounts[1], useTriangles);
    
    cellsToRefine = {3};
    mesh->hRefine(cellsToRefine);
    maxNeighbors = 5;  // the fine cell with a vertex on the center of the mesh has overlap region of 5
    weightExpected = 1.0 / (maxNeighbors + 1);
    weightActual = getSchwarzWeight(mesh, overlap);
    TEST_FLOATING_EQUALITY(weightActual, weightExpected, 1e-15);
  }
} // namespace
