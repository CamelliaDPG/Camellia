//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ExtrudedMeshTopologyTests.cpp
//  runTests
//
//  Created by Roberts, Nathan V on 3/22/21.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "ExtrudedMeshTopology.h"
#include "MeshFactory.h"
#include "MeshTestUtility.h"
#include "PoissonFormulation.h"

#include "Epetra_Import.h"

using namespace Camellia;

namespace
{
  //! extracts the vertex coordinates for the corresponding component cell, and finds the matching cell in the component MeshTopology.  Returns its cell ID.
  IndexType lookupComponentCellIndex(const FieldContainer<double> &phaseSpaceVerticesFC,
                                     const int numPhysicalDims, const int numExtrusionDims,
                                     const int componentNumber,
                                     MeshTopologyPtr componentMeshTopology)
  {
    const int numVertices = phaseSpaceVerticesFC.dimension(0);
    const int phaseSpaceDim = phaseSpaceVerticesFC.dimension(1);
    std::vector<std::vector<double>> phaseSpaceVertices(numVertices, std::vector<double>(phaseSpaceDim));
    
    for (int vertexNumber=0; vertexNumber < numVertices; vertexNumber++)
    {
      for (int d=0; d<phaseSpaceDim; d++)
      {
        phaseSpaceVertices[vertexNumber][d] = phaseSpaceVerticesFC(vertexNumber,d);
      }
    }
    
    const int numPhysicalVertices = numVertices >> numExtrusionDims;
    
    CellPtr cellFound;
    if (componentNumber == 0)
    {
      std::vector<std::vector<double> > physicalVertices(numPhysicalVertices,std::vector<double>(numPhysicalDims));
      
      for (int vertexNumber=0; vertexNumber < numPhysicalVertices; vertexNumber++)
      {
        for (int d=0; d<numPhysicalDims; d++)
        {
          physicalVertices[vertexNumber][d] = phaseSpaceVertices[vertexNumber][d];
        }
      }
      cellFound = componentMeshTopology->findCellWithVertices(physicalVertices);
    }
    else
    {
      // each extrusion is 1D
      const int numExtrusionVertices = 2;
      const int numExtrusionDims = 1;
      std::vector<std::vector<double> > extrusionVertices(numExtrusionVertices,std::vector<double>(numExtrusionDims));
      
      const int phaseSpaceDimension = numPhysicalDims + componentNumber - 1;
      const int startPhaseVertex = 0;
      const int endPhaseVertex   = numPhysicalVertices << (componentNumber - 1);
      
      extrusionVertices[0][0] = phaseSpaceVertices[startPhaseVertex][phaseSpaceDimension];
      extrusionVertices[1][0] = phaseSpaceVertices[endPhaseVertex][phaseSpaceDimension];
      
      cellFound = componentMeshTopology->findCellWithVertices(extrusionVertices);
    }
    
    if (cellFound != Teuchos::null)
    {
      return cellFound->cellIndex();
    }
    else
    {
      return -1;
    }
  }

  Teuchos::RCP<ExtrudedMeshTopology> makeTestMeshTopology( int numPhysicalDims, int numExtrusionDims,
                                                          const int physicalMeshWidth=2, const int extrusionMeshWidth=2)
  {
    using std::vector;
    vector<double> physicalDims(numPhysicalDims,1.0);
    vector<int> physicalElementCounts(numPhysicalDims,physicalMeshWidth);
    auto physicalMeshTopo = MeshFactory::rectilinearMeshTopology(physicalDims, physicalElementCounts);
    
    vector<double> extrusionStartCoords(numExtrusionDims, -1.0);
    vector<double> extrusionExtents(numExtrusionDims, 2.0);
    vector<int> extrusionElementCounts(numExtrusionDims,extrusionMeshWidth);
    
    auto extrudedMeshTopology = Teuchos::rcp(new ExtrudedMeshTopology(physicalMeshTopo,extrusionStartCoords,extrusionExtents,extrusionElementCounts));
    return extrudedMeshTopology;
  }

  Teuchos::RCP<MeshTopology> makeTestMeshTopologyNonExtruded(  int numPhysicalDims, int numExtrusionDims,
                                                             const int physicalMeshWidth=2, const int extrusionMeshWidth=2)
  {
    using std::vector;
    vector<double> startCoords(numPhysicalDims, 0.0);
    vector<double> dims(numPhysicalDims,1.0);
    
    vector<int> elementCounts(numPhysicalDims,physicalMeshWidth);
    
    for (int d=0; d<numExtrusionDims; d++)
    {
      startCoords.push_back(-1.);
      dims.push_back(2.0);
      elementCounts.push_back(extrusionMeshWidth);
    }
    
    auto meshTopo = MeshFactory::rectilinearMeshTopology(dims, elementCounts, startCoords);
    
    return meshTopo;
  }
  
  void testComponentMap(const int numPhysicalDims, const int numExtrusionDims, Teuchos::FancyOStream &out, bool &success)
  {
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims);
    
    auto physicalMeshTopo = extrudedMeshTopo->physicalMeshTopology();
    int expectedCellCount = physicalMeshTopo->cellCount();
    
    vector<MeshTopologyPtr> extrusionTopologies(numExtrusionDims);
    for (int d=0; d<numExtrusionDims; d++)
    {
      extrusionTopologies[d] = extrudedMeshTopo->extrusionMeshTopology(d);
      
      expectedCellCount *= extrusionTopologies[d]->cellCount();
    }
    
    int actualCellCount = extrudedMeshTopo->cellCount();
    TEST_EQUALITY(expectedCellCount, actualCellCount);
    
    for (IndexType cellID=0; cellID < actualCellCount; cellID++)
    {
      out << "\n\n*************************************************\n";
      out << "testing phase space cellID " << cellID << endl;
      
      CellPtr cell = extrudedMeshTopo->getCell(cellID);
      const int numVertices = cell->vertices().size();
      
      FieldContainer<double> vertices(numVertices, numPhysicalDims + numExtrusionDims);
      extrudedMeshTopo->verticesForCell(vertices, cellID);
      
      out << "phase space vertices: \n" << vertices;
      
      for (int componentNumber = 0; componentNumber < numExtrusionDims + 1; componentNumber++)
      {
        MeshTopologyPtr componentMeshTopology = (componentNumber == 0) ? physicalMeshTopo : extrusionTopologies[componentNumber-1];
        IndexType expectedCellID = lookupComponentCellIndex(vertices, numPhysicalDims, numExtrusionDims, componentNumber, componentMeshTopology);
        
        IndexType actualCellID;
        if (componentNumber == 0)
        {
          actualCellID = extrudedMeshTopo->physicalCellIDForPhaseSpaceCellID(cellID);
        }
        else
        {
          actualCellID = extrudedMeshTopo->extrusionCellIDForPhaseSpaceCellID(cellID, componentNumber - 1);
        }
        TEST_EQUALITY(expectedCellID, actualCellID);
      }
    }
  }

  void testH1FieldPoisson(const int numPhysicalDims, const int numExtrusionDims, Teuchos::FancyOStream &out, bool &success)
  {
    // Inspired by some weird results when we experimentally try to switch Vlasov
    // field (f) from L^2 to H^1, want to check that solutions are consistent
    // when using extruded versus standard meshes.  Start with a 2D (1D extruded)
    // MeshTopology; use primal Poisson formulation as a simple example to test.
    
    const int physicalMeshWidth = 2;
    const int extrudedMeshWidth = 3;
    
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims, physicalMeshWidth, extrudedMeshWidth);
    auto standardMeshTopo = makeTestMeshTopologyNonExtruded(numPhysicalDims, numExtrusionDims, physicalMeshWidth, extrudedMeshWidth);
    
    const int totalDim = numPhysicalDims + numExtrusionDims;
    
    PoissonFormulation primalForm(totalDim, true, Camellia::PoissonFormulation::PRIMAL);
    
    auto poissonBF = primalForm.bf();
   
    const int delta_k = totalDim;
    std::vector<int> H1Order(totalDim,1);
    auto extrudedMesh = MeshFactory::minRuleMesh(extrudedMeshTopo, poissonBF, H1Order, delta_k);
    auto standardMesh = MeshFactory::minRuleMesh(standardMeshTopo, poissonBF, H1Order, delta_k);
    
    // exact solution: for now, we just use a linear u
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    
    FunctionPtr u_exact = (totalDim == 2) ? -x + y : -x + y + 2. * z;

    // rhs = f * v, where f = \Delta u
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = u_exact->dx()->dx() + u_exact->dy()->dy() + u_exact->dz()->dz();
    rhs->addTerm(f * primalForm.v());
    
    VarPtr v = primalForm.v();
    IPPtr ip = IP::ip();
    ip->addTerm(v->grad());
    ip->addTerm(v);
    
    VarPtr u = primalForm.u();
    BCPtr bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(u, boundary, u_exact);
    
    SolutionPtr solnExtruded = Solution::solution(primalForm.bf(), extrudedMesh, bc, rhs, ip);
    SolutionPtr solnStandard = Solution::solution(primalForm.bf(), standardMesh, bc, rhs, ip);
    
    map<int, FunctionPtr> u_exact_map;
    u_exact_map[u->ID()] = u_exact;
    solnExtruded->projectOntoMesh(u_exact_map);
    solnStandard->projectOntoMesh(u_exact_map);
    
    FunctionPtr u_soln_extruded = Function::solution(u, solnExtruded);
    FunctionPtr u_soln_standard = Function::solution(u, solnStandard);
    FunctionPtr u_err_extruded = u_soln_extruded - u_exact;
    FunctionPtr u_err_standard = u_soln_standard - u_exact;
    
    double tol = 1e-12;
    double u_err_extruded_l2 = u_err_extruded->l2norm(extrudedMesh);
    double u_err_standard_l2 = u_err_standard->l2norm(standardMesh);
    
    if (u_err_extruded_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for extruded ";
      cout << totalDim << "D mesh and exactly recoverable solution, u error after projection is " << u_err_extruded_l2 << endl;
    }
    if (u_err_standard_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for standard ";
      cout << totalDim << "D mesh and exactly recoverable solution, u error after projection is " << u_err_standard_l2 << endl;
    }
    
    solnExtruded->clear();
    solnStandard->clear();
    
    SolverPtr solver = Solver::getSolver(Solver::KLU, false);
    solnExtruded->solve(solver);
    solnStandard->solve(solver);
    
    {
      // check neighbor agreement for standard
      Epetra_MultiVector *lhsVector = solnStandard->getGlobalCoefficients();
      Epetra_SerialComm Comm;
      Epetra_Map partMap = solnStandard->getPartitionMap();

      // Import solution onto current processor
      GlobalIndexTypeToCast numNodesGlobal = standardMesh->numGlobalDofs();
      GlobalIndexTypeToCast numMyNodes = numNodesGlobal;
      Epetra_Map     solnMap(numNodesGlobal, numMyNodes, 0, Comm);
      Epetra_Import  solnImporter(solnMap, partMap);
      Epetra_Vector  solnCoeff(solnMap);
      solnCoeff.Import(*lhsVector, solnImporter, Insert);

      if ( ! MeshTestUtility::neighborBasesAgreeOnSides(standardMesh, solnCoeff))
      {
        cout << "GDAMinimumRuleTests failure: for standard 2D Poisson mesh (after solve), neighboring bases do not agree on sides." << endl;
        success = false;
      }
    }
    {
      // check neighbor agreement for extruded
      Epetra_MultiVector *lhsVector = solnExtruded->getGlobalCoefficients();
      Epetra_SerialComm Comm;
      Epetra_Map partMap = solnExtruded->getPartitionMap();

      // Import solution onto current processor
      GlobalIndexTypeToCast numNodesGlobal = extrudedMesh->numGlobalDofs();
      GlobalIndexTypeToCast numMyNodes = numNodesGlobal;
      Epetra_Map     solnMap(numNodesGlobal, numMyNodes, 0, Comm);
      Epetra_Import  solnImporter(solnMap, partMap);
      Epetra_Vector  solnCoeff(solnMap);
      solnCoeff.Import(*lhsVector, solnImporter, Insert);

      if ( ! MeshTestUtility::neighborBasesAgreeOnSides(extrudedMesh, solnCoeff))
      {
        cout << "GDAMinimumRuleTests failure: for extruded 2D Poisson mesh (after solve), neighboring bases do not agree on sides." << endl;
        success = false;
      }
    }

    u_err_extruded_l2 = u_err_extruded->l2norm(extrudedMesh);
    u_err_standard_l2 = u_err_standard->l2norm(standardMesh);
    
    if (u_err_extruded_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for extruded ";
      cout << totalDim << "D mesh and exactly recoverable solution, u error is " << u_err_extruded_l2 << endl;
    }
    if (u_err_standard_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for standard ";
      cout << totalDim << "D mesh and exactly recoverable solution, u error is " << u_err_standard_l2 << endl;
    }
    
    // check flux
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();
    FunctionPtr sigma_n_exact = u_exact->grad() * (n * parity);
    
    FunctionPtr sigma_n_soln_extruded = Function::solution(primalForm.sigma_n_hat(), solnExtruded, false);
    FunctionPtr sigma_n_soln_standard = Function::solution(primalForm.sigma_n_hat(), solnStandard, false);
    FunctionPtr sigma_n_err_extruded = sigma_n_soln_extruded - sigma_n_exact;
    FunctionPtr sigma_n_err_standard = sigma_n_soln_standard - sigma_n_exact;
    
    double sigma_n_err_extruded_l2 = sigma_n_err_extruded->l2norm(extrudedMesh);
    double sigma_n_err_standard_l2 = sigma_n_err_standard->l2norm(standardMesh);
    
    if (sigma_n_err_extruded_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for extruded ";
      cout << totalDim << "D mesh and exactly recoverable solution, sigma_n error is " << sigma_n_err_extruded_l2 << endl;
    }
    if (sigma_n_err_standard_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for standard ";
      cout << totalDim << "D mesh and exactly recoverable solution, sigma_n error is " << sigma_n_err_standard_l2 << endl;
    }
  
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, CellVertexIndicesPeriodic_1D3V )
  {
    const int numExtrusionDims = 3;
    
    // by the construction below (a 2D mesh, extruded by 1 cell in 2 further dimensions), each vertex should participate in at least two cells, and in at most four.
    std::map<IndexType,int> vertexIncidenceCount;
    
    using std::vector;
    vector<double> physicalDims {1.0};
    vector<int> physicalElementCounts {4};
    vector<double> physicalOrigin {0.0};
    PeriodicBCPtr xLeftToRight     = PeriodicBC::xIdentification(physicalOrigin[0], physicalOrigin[0]+physicalDims[0]);
    auto physicalMeshTopo = MeshFactory::rectilinearMeshTopology(physicalDims, physicalElementCounts, physicalOrigin, {xLeftToRight});
    
    vector<double> extrusionStartCoords(numExtrusionDims, -1.0);
    vector<double> extrusionExtents(numExtrusionDims, 2.0);
    vector<int> extrusionElementCounts{4,1,1};
    
    auto extrudedMeshTopo = Teuchos::rcp(new ExtrudedMeshTopology(physicalMeshTopo,extrusionStartCoords,extrusionExtents,extrusionElementCounts));
    auto locallyKnownCells = extrudedMeshTopo->getLocallyKnownActiveCellIndices();
    
    for (const auto & cellID : locallyKnownCells)
    {
      CellPtr cell = extrudedMeshTopo->getCell(cellID);
      
      auto cellTopo = cell->topology();
      
      const int vertexDim = 0;
      int vertexCount = cellTopo->getSubcellCount(vertexDim);
      for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
      {
        IndexType entityIndex = cell->entityIndex(vertexDim, vertexOrdinal);
        TEST_COMPARE(entityIndex, >=, 0);
        vertexIncidenceCount[entityIndex]++;
      }
    }
    
    for (const auto entry : vertexIncidenceCount)
    {
      out << "Checking incidence count for vertex index " << entry.first << std::endl;
      TEST_COMPARE(entry.second, >=, 2);
      TEST_COMPARE(entry.second, <=, 4);
    }
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, Construction_1D3V )
  {
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 3;
    
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims);
    
    auto physicalMeshTopo = extrudedMeshTopo->physicalMeshTopology();
    int expectedCellCount = physicalMeshTopo->cellCount();
    
    vector<MeshTopologyPtr> extrusionTopologies(numExtrusionDims);
    for (int d=0; d<numExtrusionDims; d++)
    {
      extrusionTopologies[d] = extrudedMeshTopo->extrusionMeshTopology(d);
      
      expectedCellCount *= extrusionTopologies[d]->cellCount();
    }
    
    // check that cell count matches expected:
    int actualCellCount = extrudedMeshTopo->cellCount();
    TEST_EQUALITY(expectedCellCount, actualCellCount);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, ComponentMap_1D1V )
  {
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 1;
    
    testComponentMap(numPhysicalDims, numExtrusionDims, out, success);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, ComponentMap_1D3V )
  {
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 3;
    
    testComponentMap(numPhysicalDims, numExtrusionDims, out, success);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, H1Fields_1D1V )
  {
    // Inspired by some weird results when we experimentally try to switch Vlasov
    // field (f) from L^2 to H^1, want to check that solutions are consistent
    // when using extruded versus standard meshes.  Start with a 2D (1D extruded)
    // MeshTopology; use primal Poisson formulation as a simple example to test.
    
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 1;
    
    testH1FieldPoisson(numPhysicalDims, numExtrusionDims, out, success);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, H1Fields_1D2V )
  {
    // Inspired by some weird results when we experimentally try to switch Vlasov
    // field (f) from L^2 to H^1, want to check that solutions are consistent
    // when using extruded versus standard meshes.  Start with a 2D (1D extruded)
    // MeshTopology; use primal Poisson formulation as a simple example to test.
    
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 2;
    
    testH1FieldPoisson(numPhysicalDims, numExtrusionDims, out, success);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, Unrefine )
  {
    // this test reproduces conditions that led to extra cells being incorrectly added to MeshToologyView::_rootCells (before we fixed the bug).
    
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 1;
    
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims);
    
    const int rootCellCount = extrudedMeshTopo->getRootCellIndicesGlobal().size();
    
    const IndexType cellIDZero = 0;
    
    auto cellZero = extrudedMeshTopo->getCell(cellIDZero);
    
    auto cellTopo = cellZero->topology();
    
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
    
    const IndexType firstChildCellID = extrudedMeshTopo->nextCellID();
    extrudedMeshTopo->refineCell(cellIDZero, refPattern, firstChildCellID);
    
    const IndexType firstGrandchildCellID = extrudedMeshTopo->nextCellID();
    extrudedMeshTopo->refineCell(firstChildCellID, refPattern, firstGrandchildCellID);
    
    extrudedMeshTopo->unrefineCell(firstChildCellID, refPattern);
    
    // root cell count should not have changed:
    const int latestRootCellCount = extrudedMeshTopo->getRootCellIndicesGlobal().size();
    TEST_EQUALITY(rootCellCount, latestRootCellCount);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, VertexOrdering_1D1V )
  {
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 1;
    
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims);
    
    const IndexType cellID = 0;
    CellPtr cell = extrudedMeshTopo->getCell(cellID);
    
    const int numVertices = cell->vertices().size();
    const int numVerticesExpected = 1 << (numPhysicalDims + numExtrusionDims);
    
    TEST_EQUALITY(numVerticesExpected, numVertices);
    
    FieldContainer<double> vertices(numVertices, numPhysicalDims + numExtrusionDims);
    extrudedMeshTopo->verticesForCell(vertices, cellID);
    
    const double tol = 1e-14;
    TEST_FLOATING_EQUALITY( 0.0,vertices(0,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(0,1),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(1,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(1,1),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(2,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(2,1),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(3,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(3,1),tol);
  }

  TEUCHOS_UNIT_TEST( ExtrudedMeshTopology, VertexOrdering_1D2V )
  {
    const int numPhysicalDims = 1;
    const int numExtrusionDims = 2;
    
    auto extrudedMeshTopo = makeTestMeshTopology(numPhysicalDims, numExtrusionDims);
    
    const IndexType cellID = 0;
    CellPtr cell = extrudedMeshTopo->getCell(cellID);
    
    const int numVertices = cell->vertices().size();
    const int numVerticesExpected = 1 << (numPhysicalDims + numExtrusionDims);
    
    TEST_EQUALITY(numVerticesExpected, numVertices);
    
    FieldContainer<double> vertices(numVertices, numPhysicalDims + numExtrusionDims);
    extrudedMeshTopo->verticesForCell(vertices, cellID);
    
    const double tol = 1e-14;
    TEST_FLOATING_EQUALITY( 0.0,vertices(0,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(0,1),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(0,2),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(1,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(1,1),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(1,2),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(2,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(2,1),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(2,2),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(3,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(3,1),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(3,2),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(4,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(4,1),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(4,2),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(5,0),tol);
    TEST_FLOATING_EQUALITY(-1.0,vertices(5,1),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(5,2),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(6,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(6,1),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(6,2),tol);
    TEST_FLOATING_EQUALITY( 0.5,vertices(7,0),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(7,1),tol);
    TEST_FLOATING_EQUALITY( 0.0,vertices(7,2),tol);
  }
} // end anonymous namespace
