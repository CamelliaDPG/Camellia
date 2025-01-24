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
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
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
} // namespace
