//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BCTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/29/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "Boundary.h"
#include "ExpFunction.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "Projector.h"
#include "RHS.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
void testSpaceTimeTraceBCFunction(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
  vector<int> elementCounts(spaceDim,1);   // one-element mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

  double t0 = 0.0, t1 = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

  double epsilon = .1;
  int fieldPolyOrder = 3, delta_k = 1;

  static const double CONST_VALUE = 0.5;
  FunctionPtr u = Function::constant(CONST_VALUE);

  bool useConformingTraces = true;
  SpaceTimeHeatFormulation form(spaceDim, useConformingTraces, epsilon);

  FunctionPtr forcingFunction = SpaceTimeHeatFormulation::forcingFunction(spaceDim, epsilon, u);
  form.initializeSolution(spaceTimeMeshTopo, fieldPolyOrder, delta_k, forcingFunction);

  VarPtr u_hat = form.u_hat();
  BCPtr bc = form.solution()->bc();
  bc->addDirichlet(u_hat, SpatialFilter::allSpace() | SpatialFilter::matchingT(t0) | SpatialFilter::matchingT(t1), u);

  MeshPtr mesh = form.solution()->mesh();

  GlobalIndexType cellID = 0;

  if (spaceTimeMeshTopo->isValidCellIndex(cellID))
  {
    // use our knowledge that we have a one-element mesh: every last dof for u_hat should be present, and have coefficient CONST_VALUE
    DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    CellTopoPtr cellTopo = mesh->getElementType(cellID)->cellTopoPtr;

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    double tol = 1e-13;
    for (int sideOrdinal=0; sideOrdinal < cellTopo->getSideCount(); sideOrdinal++)
    {
      out << "******** SIDE " << sideOrdinal << " ********" << endl;
      BasisPtr basis = trialOrder->getBasis(u_hat->ID(),sideOrdinal);

      int numCells = 1;
      Camellia::FieldContainer<double> dirichletValues(numCells,basis->getCardinality());
      // project bc function onto side basis:
      Teuchos::RCP<BCFunction<double>> bcFunction = BCFunction<double>::bcFunction(bc, u_hat->ID());
      bc->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideOrdinal));
      for (int basisOrdinal=0; basisOrdinal<dirichletValues.dimension(1); basisOrdinal++)
      {
        TEST_FLOATING_EQUALITY(CONST_VALUE, dirichletValues(0,basisOrdinal), tol);
      }
    }
  }
}
  

  // Dirichlet data for the cubature-enrichment tests: a narrow Gaussian ridge in x and one in y (width 0.15), which the
  // default cubature of a coarse, low-order cell under-resolves; and a quadratic, which the trace basis represents.
  FunctionPtr sharpBoundaryData()
  {
    FunctionPtr x = Function::xn(1), y = Function::yn(1);
    FunctionPtr dx = x - Function::constant(0.4), dy = y - Function::constant(0.35);
    const double s2 = 0.15 * 0.15;
    return Exp<double>::exp((-1.0 / s2) * dx * dx) + Exp<double>::exp((-1.0 / s2) * dy * dy);
  }
  
  FunctionPtr quadraticBoundaryData()
  {
    FunctionPtr x = Function::xn(1), y = Function::yn(1);
    return x * x + 2.0 * y + 1.0;
  }
  
  map<GlobalIndexType,double> bcValueMap(MeshPtr mesh, BCPtr bc, DofInterpreter* dofInterpreter, int cubatureEnrichment)
  {
    Camellia::FieldContainer<GlobalIndexType> indices;
    Camellia::FieldContainer<double> values;
    mesh->boundary().bcsToImpose(indices, values, *bc, dofInterpreter, cubatureEnrichment);
    map<GlobalIndexType,double> valueMap;
    for (int i=0; i<indices.size(); i++) valueMap[indices[i]] = values[i];
    return valueMap;
  }
  
  // max |a - b| over the dofs; dof sets must agree
  double maxDifference(const map<GlobalIndexType,double> &a, const map<GlobalIndexType,double> &b,
                       Teuchos::FancyOStream &out, bool &success)
  {
    TEST_EQUALITY(a.size(), b.size());
    double diff = 0;
    for (auto entry : a)
    {
      auto other = b.find(entry.first);
      if (other == b.end())
      {
        out << "dof " << entry.first << " missing\n";
        success = false;
        continue;
      }
      diff = max(diff, abs(entry.second - other->second));
    }
    return diff;
  }
  
  // A one-cell ultraweak Poisson mesh on [0,1]^2 with low order (default cubature degree low), and BCs on u_hat over
  // the whole boundary given by SpatialFilter (legacy) or by tag.
  struct CubatureEnrichmentSetup
  {
    MeshPtr mesh;
    VarPtr u_hat;
    Teuchos::RCP<PoissonFormulation> form;
    int tagID = 17;
  };
  
  CubatureEnrichmentSetup cubatureEnrichmentSetup()
  {
    CubatureEnrichmentSetup setup;
    const int spaceDim = 2, H1Order = 2, delta_k = 1;
    bool conformingTraces = true;
    setup.form = Teuchos::rcp( new PoissonFormulation(spaceDim, conformingTraces, PoissonFormulation::ULTRAWEAK) );
    setup.u_hat = setup.form->u_hat();
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(vector<double>(spaceDim,1.0), vector<int>(spaceDim,1));
    setup.mesh = Teuchos::rcp( new Mesh(meshTopo, setup.form->bf(), H1Order, delta_k) );
    EntitySetPtr allSides = meshTopo->createEntitySet();
    GlobalIndexType cellID = 0;
    if (meshTopo->isValidCellIndex(cellID))
    {
      CellPtr cell = meshTopo->getCell(cellID);
      for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
        allSides->addEntity(spaceDim - 1, cell->entityIndex(spaceDim - 1, sideOrdinal));
    }
    meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, setup.tagID, allSides);
    return setup;
  }
  
  BCPtr boundaryDataBC(const CubatureEnrichmentSetup &setup, FunctionPtr data, bool useTags)
  {
    BCPtr bc = BC::bc();
    if (useTags) bc->addDirichlet(setup.u_hat, setup.tagID, data);
    else         bc->addDirichlet(setup.u_hat, SpatialFilter::allSpace(), data);
    return bc;
  }
  
  // Boundary::bcsToImpose: the cubature enrichment reaches the projection of the data.  Under-resolved data change with
  // the enrichment and converge; data the basis represents exactly do not change.
  void testBCCubatureEnrichment(bool useTags, Teuchos::FancyOStream &out, bool &success)
  {
    auto setup = cubatureEnrichmentSetup();
    SolutionPtr soln = Solution::solution(setup.form->bf(), setup.mesh, BC::bc());
    DofInterpreter* dofInterpreter = soln->getDofInterpreter().get();
    
    BCPtr sharp = boundaryDataBC(setup, sharpBoundaryData(), useTags);
    auto sharp0  = bcValueMap(setup.mesh, sharp, dofInterpreter, 0);
    auto sharp30 = bcValueMap(setup.mesh, sharp, dofInterpreter, 30);
    auto sharp40 = bcValueMap(setup.mesh, sharp, dofInterpreter, 40);
    const double underResolved = maxDifference(sharp0, sharp40, out, success);
    const double resolved      = maxDifference(sharp30, sharp40, out, success);
    out << "sharp data: |default - enriched 40| = " << underResolved << ", |enriched 30 - enriched 40| = " << resolved << endl;
    TEST_COMPARE(underResolved, >, 1e-2); // the default rule misses the ridges: the enrichment matters here (0.108)
    TEST_COMPARE(resolved, <, 1e-8);      // enriched projections converge (exponentially: 2.7e-10 here)
    
    BCPtr quadratic = boundaryDataBC(setup, quadraticBoundaryData(), useTags);
    const double exact = maxDifference(bcValueMap(setup.mesh, quadratic, dofInterpreter, 0),
                                       bcValueMap(setup.mesh, quadratic, dofInterpreter, 20), out, success);
    out << "quadratic data: |default - enriched 20| = " << exact << endl;
    TEST_COMPARE(exact, <, 1e-12);
  }
  
  // u_hat's coefficients on each side of cell 0, as imposed in a solve
  map<int,Camellia::FieldContainer<double>> solvedTraceCoefficients(const CubatureEnrichmentSetup &setup, SolutionPtr soln)
  {
    map<int,Camellia::FieldContainer<double>> coefficients;
    GlobalIndexType cellID = 0;
    DofOrderingPtr trialOrder = setup.mesh->getElementType(cellID)->trialOrderPtr;
    for (int sideOrdinal : trialOrder->getSidesForVarID(setup.u_hat->ID()))
      soln->solnCoeffsForCellID(coefficients[sideOrdinal], cellID, setup.u_hat->ID(), sideOrdinal);
    return coefficients;
  }
  
  // u_hat's coefficients on each side of cell 0 projected directly with the given cubature enrichment
  map<int,Camellia::FieldContainer<double>> projectedTraceCoefficients(const CubatureEnrichmentSetup &setup, FunctionPtr data,
                                                                       int cubatureEnrichment)
  {
    map<int,Camellia::FieldContainer<double>> coefficients;
    GlobalIndexType cellID = 0;
    DofOrderingPtr trialOrder = setup.mesh->getElementType(cellID)->trialOrderPtr;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(setup.mesh, cellID, false, cubatureEnrichment);
    for (int sideOrdinal : trialOrder->getSidesForVarID(setup.u_hat->ID()))
    {
      BasisPtr basis = trialOrder->getBasis(setup.u_hat->ID(), sideOrdinal);
      Camellia::FieldContainer<double> sideCoefficients(1, basis->getCardinality());
      Projector<double>::projectFunctionOntoBasisInterpolating(sideCoefficients, data, basis, basisCache->getSideBasisCache(sideOrdinal));
      sideCoefficients.resize(basis->getCardinality());
      coefficients[sideOrdinal] = sideCoefficients;
    }
    return coefficients;
  }
  
  double maxDifference(const map<int,Camellia::FieldContainer<double>> &a, const map<int,Camellia::FieldContainer<double>> &b)
  {
    double diff = 0;
    for (auto entry : a)
    {
      const auto &other = b.at(entry.first);
      for (int i=0; i<entry.second.size(); i++) diff = max(diff, abs(entry.second[i] - other[i]));
    }
    return diff;
  }
  
  // Solution::imposeBCs (via solve()): the data are projected with the Solution's cubature enrichment, or with the BC's
  // if set (BC::setCubatureEnrichmentDegree).
  void testSolutionBCCubatureEnrichment(Teuchos::FancyOStream &out, bool &success)
  {
    auto setup = cubatureEnrichmentSetup();
    FunctionPtr data = sharpBoundaryData();
    const double tol = 1e-10;
    auto projected0  = projectedTraceCoefficients(setup, data, 0);
    auto projected20 = projectedTraceCoefficients(setup, data, 20);
    TEST_COMPARE(maxDifference(projected0, projected20), >, 1e-3); // the cases below are distinguishable
    
    auto solveWith = [&](int solutionEnrichment, int bcEnrichment) {
      BCPtr bc = boundaryDataBC(setup, data, false);
      if (bcEnrichment >= 0) bc->setCubatureEnrichmentDegree(bcEnrichment);
      SolutionPtr soln = Solution::solution(setup.form->bf(), setup.mesh, bc);
      soln->setRHS(RHS::rhs());
      soln->setIP(setup.form->bf()->graphNorm());
      soln->setCubatureEnrichmentDegree(solutionEnrichment);
      soln->solve();
      return solvedTraceCoefficients(setup, soln);
    };
    // no enrichment anywhere: the default rule (the behavior before BC enrichment existed)
    TEST_COMPARE(maxDifference(solveWith(0, -1), projected0), <, tol);
    // the Solution's enrichment reaches the BC data
    TEST_COMPARE(maxDifference(solveWith(20, -1), projected20), <, tol);
    // the BC's enrichment overrides the Solution's, either way
    TEST_COMPARE(maxDifference(solveWith(0, 20), projected20), <, tol);
    TEST_COMPARE(maxDifference(solveWith(20, 0), projected0), <, tol);
  }

  void testTagCoefficientsMatchLegacy(int spaceDim, bool useFieldBCs, Teuchos::FancyOStream &out, bool &success)
  {
    // test that the coefficients determined for a BC object that uses the new tag-based BCs
    // matches those determined by SpatialFilters
    bool conformingTraces = true;
    PoissonFormulation::PoissonFormulationChoice formChoice = useFieldBCs ? PoissonFormulation::PRIMAL : PoissonFormulation::ULTRAWEAK;
    PoissonFormulation form(spaceDim,conformingTraces,formChoice);
    
    VarPtr var = useFieldBCs ? form.u() : form.u_hat();

    int H1Order = 3, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(vector<double>(spaceDim,1.0), vector<int>(spaceDim,1));
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) );
    
    // add a tag for the Dirichlet BC region (all the sides of the single cell in the mesh)
    int tagID = 34;
    EntitySetPtr allSides = meshTopo->createEntitySet();
    
    // on any ranks that can see cell 0, add its sides to allSides
    GlobalIndexType cellID = 0;
    if (meshTopo->isValidCellIndex(cellID))
    {
      CellPtr cell = meshTopo->getCell(cellID);
      int sideCount = cell->getSideCount();
      int sideDim = spaceDim - 1;
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
        allSides->addEntity(sideDim, sideEntityIndex);
      }
    }
    meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, tagID, allSides);

    FunctionPtr x = Function::xn(1);
    
    FunctionPtr phi_value = x * x + 1;
    BCPtr legacyBC = BC::bc();
    legacyBC->addDirichlet(var, SpatialFilter::allSpace(), phi_value);
    SolutionPtr legacySoln = Solution::solution(form.bf(), mesh, legacyBC);
    
    BCPtr tagBC = BC::bc();
    tagBC->addDirichlet(var, tagID, phi_value);
    SolutionPtr soln = Solution::solution(form.bf(), mesh, tagBC);
    
    Camellia::FieldContainer<GlobalIndexType> bcGlobalIndicesLegacy, bcGlobalIndicesTags;
    Camellia::FieldContainer<double> bcGlobalValuesLegacy, bcGlobalValuesTags;
    
    // we can safely assume that the two Solution objects have equivalent DofInterpreter
    Teuchos::RCP<DofInterpreter> dofInterpreter = legacySoln->getDofInterpreter();
    
    mesh->boundary().bcsToImpose(bcGlobalIndicesLegacy,bcGlobalValuesLegacy,*legacyBC, dofInterpreter.get());
    mesh->boundary().bcsToImpose(bcGlobalIndicesTags,bcGlobalValuesTags,*tagBC, dofInterpreter.get());
    
    map<GlobalIndexType,double> bcValueMapLegacy;
    for (int i=0; i<bcGlobalIndicesLegacy.size(); i++)
    {
      bcValueMapLegacy[bcGlobalIndicesLegacy[i]] = bcGlobalValuesLegacy[i];
    }
    map<GlobalIndexType,double> bcValueMapTags;
    for (int i=0; i<bcGlobalIndicesTags.size(); i++)
    {
      bcValueMapTags[bcGlobalIndicesTags[i]] = bcGlobalValuesTags[i];
    }
    
    TEST_EQUALITY(bcValueMapLegacy.size(), bcValueMapTags.size());
    
    double tol = 1e-14;
    for (int i=0; i<bcGlobalIndicesLegacy.size(); i++)
    {
      GlobalIndexType legacyDofIndex = bcGlobalIndicesLegacy[i];
      double legacyValue = bcValueMapLegacy[legacyDofIndex];
      if (bcValueMapTags.find(legacyDofIndex) == bcValueMapTags.end())
      {
        out << "Dof Index " << legacyDofIndex << " not found in bcValueMapTags.\n";
        success = false;
      }
      else
      {
        double tagValue = bcValueMapTags[legacyDofIndex];
        double diff = abs(legacyValue - tagValue);
        if ((diff > tol) && (diff > tol * min(abs(tagValue),abs(legacyValue))))
        {
          success = false;
          out << "legacy value != tag value (" << legacyValue << " != " << tagValue << ")\n";
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( BC, FieldBCsMinRule_1D)
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 1;
    bool useFieldBCs = true;
    testTagCoefficientsMatchLegacy(spaceDim, useFieldBCs, out, success);
    // ultimately, should check *correctness* of imposition, not just that the two types agree for field BCs
  }
  
  TEUCHOS_UNIT_TEST( BC, FieldBCsMinRule_2D)
  {
    int spaceDim = 2;
    bool useFieldBCs = true;
    testTagCoefficientsMatchLegacy(spaceDim, useFieldBCs, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, FieldBCsMinRule_3D)
  {
    int spaceDim = 3;
    bool useFieldBCs = true;
    testTagCoefficientsMatchLegacy(spaceDim, useFieldBCs, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, SpaceTimeTraceBCCoefficients )
  {
    int spaceDim = 1;
    testSpaceTimeTraceBCFunction(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, TagCoefficientsMatchLegacy_1D )
  {
    int spaceDim = 1;
    bool useFieldBCs = false;
    testTagCoefficientsMatchLegacy(spaceDim, useFieldBCs, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, TagCoefficientsMatchLegacy_2D )
  {
    int spaceDim = 2;
    bool useFieldBCs = false;
    testTagCoefficientsMatchLegacy(spaceDim, useFieldBCs, out, success);
  }
  TEUCHOS_UNIT_TEST( BC, CubatureEnrichmentReachesBoundaryData_Legacy )
  {
    bool useTags = false;
    testBCCubatureEnrichment(useTags, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, CubatureEnrichmentReachesBoundaryData_Tags )
  {
    bool useTags = true;
    testBCCubatureEnrichment(useTags, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BC, CubatureEnrichmentFromSolutionAndBC )
  {
    testSolutionBCCubatureEnrichment(out, success);
  }
} // namespace
