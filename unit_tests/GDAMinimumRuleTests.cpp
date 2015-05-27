//
//  GDAMinimumRuleTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 5/14/15.
//
//

#include "Epetra_Import.h"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "Epetra_MpiDistributor.h"
#endif

#include "Epetra_SerialComm.h"
#include "Epetra_SerialDistributor.h"

#include "CamelliaCellTools.h"
#include "CamelliaTestingHelpers.h"
#include "CubatureFactory.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MeshTestUtility.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TypeDefs.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
/*
 Three essential tests that GDAMinimumRule is doing its job:
 1. Cell and subcell constraints are correctly identified.
 2. Take a coarse (constraining) basis and compute some values on subcell.  Compute
    fine (constrained) basis values along the same physical region, now weighting
    according to getBasisMapOld() (the one we currently use).  Are the values the same?
 3. Actually try solving something.

 Of these, #1 would be good for a few hand-computed instances -- I'm unclear on how
 easily we can check this on a generic mesh topology.  (Sanity checks are certainly
 possible--check that the constraining subcell is in fact an ancestor of the constrained,
 for instance.  This is implemented by testSubcellConstraintIsAncestor, below.)
 #1 is also cheap to run.

 #2 is a bit more expensive, but not crazy.  It should be possible to repurpose some of
 the ideas already present in BasisReconciliationTests for this purpose.  It also seems
 very possible to do this for a generic mesh.

 #3 is what the DPGTests have done to date; the HangingNodePoisson3D_Slow test below,
 moved over from DPGTests, is an example.  This is an "integration" test, and it's
 expensive and when it fails it doesn't reveal much in terms of where the failure came
 from.  However, it does have the advantage of checking that we end up with the right
 answers in the context of a real problem!

 TODO: implement #2.  Implement #3 for a generic mesh (can imitate HangingNodePoisson3D_Slow, perhaps).
 */

// copied from BasisReconciliation.cpp -- likely, this method should be in a utility class somewhere
void filterFCValues(FieldContainer<double> &filteredFC, const FieldContainer<double> &fc,
                    set<unsigned> &ordinals, int basisCardinality)
{
  // we use pointer arithmetic here, which doesn't allow Intrepid's safety checks, for two reasons:
  // 1. It's easier to manage in a way that's independent of the rank of the basis
  // 2. It's faster.
  int numEntriesPerBasisField = fc.size() / basisCardinality;
  int filteredFCIndex = 0;
  // we can do a check of our own, though, that the filteredFC is the right total length:
  TEUCHOS_TEST_FOR_EXCEPTION(filteredFC.size() != numEntriesPerBasisField * ordinals.size(), std::invalid_argument,
                             "filteredFC.size() != numEntriesPerBasisField * ordinals.size()");
  for (auto ordinal : ordinals)
  {
    const double *fcEntry = &fc[ordinal * numEntriesPerBasisField];
    double *filteredEntry = &filteredFC[ filteredFCIndex * numEntriesPerBasisField ];
    for (int i=0; i<numEntriesPerBasisField; i++)
    {
      *filteredEntry = *fcEntry;
      filteredEntry++;
      fcEntry++;
    }

    filteredFCIndex++;
  }
}

void testSubcellConstraintIsAncestor(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
{
  /*
   This test iterates through all the active cells in the mesh, and checks that for each of their
   subcells, the subcell that constrains them is an ancestor of that subcell.

   One other thing we could test, but don't yet, is that the constraining subcells are not themselves
   constrained by some other subcell--the rule is they should be the end of the line.
   */
  GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
  if (gda == NULL)
  {
    out << "Mesh appears not to use GDAMinimumRule.  testSubcellConstraintIsAncestor() requires this.\n";
    success = false;
    return;
  }
  auto activeCellIDs = mesh->getActiveCellIDs();
  MeshTopologyPtr meshTopo = mesh->getTopology();
  int sideDim = meshTopo->getSpaceDim() - 1;
  for (auto cellID : activeCellIDs)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellConstraints cellConstraints = gda->getCellConstraints(cellID);
    CellTopoPtr cellTopo = cell->topology();
    for (int subcdim=0; subcdim<cellTopo->getDimension(); subcdim++)
    {
      int subcellCount = cellTopo->getSubcellCount(subcdim);
      for (int subcord=0; subcord<subcellCount; subcord++)
      {
        IndexType subcellEntityIndex = cell->entityIndex(subcdim, subcord);
        AnnotatedEntity constrainingEntity = cellConstraints.subcellConstraints[subcdim][subcord];
        CellPtr constrainingCell = meshTopo->getCell(constrainingEntity.cellID);
        CellTopoPtr constrainingCellTopo = constrainingCell->topology();
        // When a side is involved in the constraint, then AnnotatedEntity.subcellOrdinal is the subcell ordinal in the side.
        // We therefore map to the subcell in the cell:
        unsigned constrainingCellSubcord = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, sideDim, constrainingEntity.sideOrdinal, constrainingEntity.dimension, constrainingEntity.subcellOrdinal);
        IndexType constrainingEntityIndex = constrainingCell->entityIndex(constrainingEntity.dimension, constrainingCellSubcord);
        bool isAncestor = meshTopo->entityIsGeneralizedAncestor(constrainingEntity.dimension, constrainingEntityIndex,
                          subcdim, subcellEntityIndex);
        if (!isAncestor)
        {
          out << "cell " << cellID << ", " << CamelliaCellTools::entityTypeString(subcdim);
          out << " ordinal " << subcord << " is constrained by cell " << constrainingEntity.cellID;
          out << ", " << CamelliaCellTools::entityTypeString(constrainingEntity.dimension);
          out << " ordinal " << constrainingCellSubcord << ", which is not its ancestor!\n";
          success = false;
        }
      }
    }
  }
}

void testCoarseBasisEqualsWeightedFineBasis(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
{
  /*
   For each active cell:
     1. Construct the BasisCache for that cell.
     2. For each side of the cell:
        a. Compute the BasisMap for that side.
        b. For each subcell of that side:
           i.    Compute cubature points on the subcell.
           ii.   Map the cubature points to the side.
           iii.  Set the reference cell points of the side BasisCache to be the mapped cubature points.
           iv.   Compute transformed basis values on that side.
           v.    Weight those basis values according to the BasisMap.
           vi.   Determine the constraining subcell and side.
           vii.  Map the fine subcell cubature points to the constraining subcell's reference space.
           viii. Map the constraining subcell points to the constraining side's reference space.
           ix.   Create a side BasisCache for the constraining cell/side.
           x.    Set the reference points for the side BasisCache to be those computed.
           xi.   Check that the physical points for the constraining side cache agree with the fine side cache.
           xii.  Compute the constraining basis at those points.
           xiii. Compare the values.

   Notes about BasisMap: this is a typedef:
   typedef vector< SubBasisDofMapperPtr > BasisMap;

   Where SubBasisDofMapper allows computation of a weighted sum of basis values via its mapFineData() method.
   Iterating through the BasisMap will allow such maps for the basis as a whole.

   One does need to attend to the mappedGlobalDofOrdinals() in the SubBasisDofMapper; this tells you which global
   ordinals's data you get from mapFineData().  If a global dof ordinal is mapped by several SubBasisDofMappers,
   one should accumulate the values.

   So to be precise, one will want to construct BasisMaps for both the coarse and the fine domains, and check that
   they agree on the global data once the mapFineData() thing has been done.  (You still call mapFineData for the coarse
   domain; it just happens that this is a 1-1 mapping because the coarse domain constrains itself.)

   */
  VarFactoryPtr vf = mesh->bilinearForm()->varFactory();

  // for now, we just check a single trace var:
  VarPtr traceVar;
  if (vf->traceVars().size() > 0)
  {
    traceVar = vf->traceVars()[0];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(vf->fluxVars().size() == 0, std::invalid_argument,
                               "Both fluxVars and traceVars are empty; this is not supported by this test.");
    traceVar = vf->fluxVars()[0];
  }
  // TODO: test flux var when both traces and fluxes are present (as with Poisson for spaceDim > 1)

  GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
  if (gda == NULL)
  {
    out << "Mesh appears not to use GDAMinimumRule.  testSubcellConstraintIsAncestor() requires this.\n";
    success = false;
    return;
  }

//    gda->printGlobalDofInfo();

  auto activeCellIDs = mesh->getActiveCellIDs();
  MeshTopologyPtr meshTopo = mesh->getTopology();

//    meshTopo->printAllEntities();
  unsigned edgeDim = 1;
//    meshTopo->printConstraintReport(edgeDim);

  int sideDim = meshTopo->getSpaceDim() - 1;
  Camellia::CubatureFactory cubFactory;
  for (auto cellID : activeCellIDs)
  {
    BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(mesh, cellID);
    CellPtr cell = meshTopo->getCell(cellID);
    CellConstraints cellConstraints = gda->getCellConstraints(cellID);
    auto dofOwnershipInfo = gda->getGlobalDofIndices(cellID, cellConstraints);
    CellTopoPtr cellTopo = cell->topology();
    int sideCount = cellTopo->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      BasisCachePtr sideBasisCache = cellBasisCache->getSideBasisCache(sideOrdinal);
      BasisPtr fineBasis = mesh->getElementType(cellID)->trialOrderPtr->getBasis(traceVar->ID(),sideOrdinal);
      auto fineBasisMap = gda->getBasisMapOld(cellID, dofOwnershipInfo, traceVar, sideOrdinal);
      int cubDegree = fineBasis->getDegree();
      CellTopoPtr sideTopo = cellTopo->getSide(sideOrdinal);
      for (int subcdim=0; subcdim<=sideDim; subcdim++)
      {
        int subcellCount = sideTopo->getSubcellCount(subcdim);
        for (int subcord=0; subcord<subcellCount; subcord++)
        {
          CellTopoPtr subcell = sideTopo->getSubcell(subcdim, subcord);
          unsigned subcordInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, sideOrdinal, subcdim, subcord);
          FieldContainer<double> subcellPoints;
          if (subcdim==0)
          {
            subcellPoints.resize(1,1); // vertex; don't need points as such
          }
          else
          {
            CellTopoPtr subcellTopo = sideTopo->getSubcell(subcdim, subcord);
            auto cubature = cubFactory.create(subcellTopo, cubDegree);
            subcellPoints.resize(cubature->getNumPoints(),cubature->getDimension());
            FieldContainer<double> weights(cubature->getNumPoints()); // we ignore these
            cubature->getCubature(subcellPoints, weights);
          }
          int numPoints = subcellPoints.dimension(0);

          // we understand the subcell points to be defined on the subcell as viewed by the *cell* (the volume)
          // therefore, we need to permute the points to match the side's view of the subcell
          FieldContainer<double> sidePermutedSubcellPoints;
          if (subcdim > 0)
          {
            unsigned canonicalToCellSubcellPermutation = cell->subcellPermutation(subcdim, subcordInCell);
            unsigned canonicalToSideSubcellPermutation = cell->sideSubcellPermutation(sideOrdinal, subcdim, subcord);
            unsigned cellToCanonicalSubcellPermutation = CamelliaCellTools::permutationInverse(subcell,
                canonicalToCellSubcellPermutation);
            unsigned cellToSideSubcellPermutation = CamelliaCellTools::permutationComposition(subcell, cellToCanonicalSubcellPermutation, canonicalToSideSubcellPermutation);

            sidePermutedSubcellPoints.resize(numPoints,subcdim);
            CamelliaCellTools::permutedReferenceCellPoints(subcell, cellToSideSubcellPermutation, subcellPoints, sidePermutedSubcellPoints);
          }
          else
          {
            sidePermutedSubcellPoints = subcellPoints;
          }

          // map the subcellPoints to the fine domain
          FieldContainer<double> fineDomainPoints(numPoints,sideDim);
          CamelliaCellTools::mapToReferenceSubcell(fineDomainPoints, sidePermutedSubcellPoints, subcdim, subcord, sideTopo);

          sideBasisCache->setRefCellPoints(fineDomainPoints);
          FieldContainer<double> fineValuesAllPoints = *sideBasisCache->getTransformedValues(fineBasis, OP_VALUE);
          // strip cell dimension:
          fineValuesAllPoints.resize(fineBasis->getCardinality(), numPoints);

          // map the subcellPoints to ancestor's reference space:
          RefinementBranch cellRefinementBranch = cell->refinementBranchForSubcell(subcdim, subcordInCell);
          unsigned canonicalToAncestralSubcellPermutation;
          FieldContainer<double> ancestralSubcellPoints;
          CellTopoPtr ancestralSubcellTopo;
          if (cellRefinementBranch.size() > 0)
          {
            GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(cellRefinementBranch, subcdim, subcordInCell);
            // genRefBranch will end in volume topology; we want it to end in the constraining subcell topology:
            CellTopoPtr ancestralCellTopo = genRefBranch[0].previousTierTopo;
            ancestralSubcellTopo = ancestralCellTopo->getSubcell(genRefBranch[0].rootDimension,
                                   genRefBranch[0].previousTierSubcellOrdinal);
            genRefBranch[0].previousTierTopo = ancestralSubcellTopo;
            genRefBranch[0].previousTierSubcellOrdinal = 0;
            RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, subcellPoints, ancestralSubcellPoints);
            canonicalToAncestralSubcellPermutation = cell->ancestralPermutationForSubcell(subcdim, subcordInCell);
          }
          else
          {
            // no refinement branch:
            ancestralSubcellTopo = subcell;
            ancestralSubcellPoints = subcellPoints;
            canonicalToAncestralSubcellPermutation = cell->subcellPermutation(subcdim, subcordInCell);
          }

          // constraining subcell and side:
          AnnotatedEntity constrainingEntityInfo = cellConstraints.subcellConstraints[subcdim][subcordInCell];
          CellPtr constrainingCell = meshTopo->getCell(constrainingEntityInfo.cellID);
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          unsigned constrainingSideOrdinal = constrainingEntityInfo.sideOrdinal;
          CellTopoPtr constrainingSideTopo = constrainingCellTopo->getSubcell(sideDim, constrainingSideOrdinal);
          unsigned canonicalToConstrainingSideSubcellPermutation = constrainingCell->sideSubcellPermutation(constrainingSideOrdinal, constrainingEntityInfo.dimension, constrainingEntityInfo.subcellOrdinal);
          unsigned ancestralToCanonicalSubcellPermutation = CamelliaCellTools::permutationInverse(ancestralSubcellTopo, canonicalToAncestralSubcellPermutation);
          unsigned ancestralToConstrainingSideSubcellPermutation = CamelliaCellTools::permutationComposition(ancestralSubcellTopo, ancestralToCanonicalSubcellPermutation, canonicalToConstrainingSideSubcellPermutation);
          FieldContainer<double> constrainingSubcellPoints(numPoints,constrainingEntityInfo.dimension);
          CamelliaCellTools::permutedReferenceCellPoints(ancestralSubcellTopo, ancestralToConstrainingSideSubcellPermutation,
              ancestralSubcellPoints, constrainingSubcellPoints);
          FieldContainer<double> constrainingDomainPoints(numPoints,sideDim);
          CamelliaCellTools::mapToReferenceSubcell(constrainingDomainPoints, constrainingSubcellPoints,
              constrainingEntityInfo.dimension, constrainingEntityInfo.subcellOrdinal,
              constrainingSideTopo);
          BasisCachePtr constrainingCellBasisCache = BasisCache::basisCacheForCell(mesh, constrainingEntityInfo.cellID);
          BasisCachePtr constrainingSideBasisCache = constrainingCellBasisCache->getSideBasisCache(constrainingSideOrdinal);
          constrainingSideBasisCache->setRefCellPoints(constrainingDomainPoints);

          unsigned constrainingSubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo,
              sideDim, constrainingSideOrdinal,
              constrainingEntityInfo.dimension,
              constrainingEntityInfo.subcellOrdinal);
          IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(constrainingEntityInfo.dimension,
              constrainingSubcellOrdinalInCell);
          vector< pair<IndexType,unsigned> > activeCellEntries;
          activeCellEntries = meshTopo->getActiveCellIndices(constrainingEntityInfo.dimension, constrainingSubcellEntityIndex);
          TEUCHOS_TEST_FOR_EXCEPTION(activeCellEntries.size() == 0, std::invalid_argument, "constraining subcell must belong to some active cell!");

          // As a sanity check, compare the physical points for coarse and fine:
          FieldContainer<double> finePhysicalPoints = sideBasisCache->getPhysicalCubaturePoints();
          FieldContainer<double> coarsePhysicalPoints = constrainingSideBasisCache->getPhysicalCubaturePoints();
          bool oldSuccess = success;
          success = true;
          TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(finePhysicalPoints, coarsePhysicalPoints, 1e-15);
          if (!success)
          {
            out << "Failure: finePhysicalPoints != coarsePhysicalPoints; therefore skipping weighted basis comparison.\n";
            break;
          }
          success = success && oldSuccess;

          ElementTypePtr constrainingElementType = mesh->getElementType(constrainingEntityInfo.cellID);
          BasisPtr constrainingBasis = constrainingElementType->trialOrderPtr->getBasis(traceVar->ID(), constrainingSideOrdinal);

          FieldContainer<double> constrainingBasisValuesAllPoints = *constrainingSideBasisCache->getTransformedValues(constrainingBasis, OP_VALUE);
          // strip cell dimension:
          constrainingBasisValuesAllPoints.resize(constrainingBasis->getCardinality(), numPoints);

          CellConstraints coarseCellConstraints = gda->getCellConstraints(constrainingEntityInfo.cellID);
          auto coarseGlobalDofInfo = gda->getGlobalDofIndices(constrainingEntityInfo.cellID, cellConstraints);
          auto coarseBasisMap = gda->getBasisMapOld(constrainingEntityInfo.cellID, coarseGlobalDofInfo,
                                traceVar, constrainingSideOrdinal);

          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            FieldContainer<double> fineValues(fineBasis->getCardinality());
            for (int basisOrdinal=0; basisOrdinal<fineBasis->getCardinality(); basisOrdinal++)
            {
              fineValues(basisOrdinal) = fineValuesAllPoints(basisOrdinal,pointOrdinal);
            }
            map<GlobalIndexType,double> fineGlobalValues;
            for (auto subBasisMap : fineBasisMap)
            {
              // filter fine values according to what subBasisMap knows about.
              set<unsigned> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
              FieldContainer<double> fineFilteredValues(basisOrdinalFilter.size());
              filterFCValues(fineFilteredValues, fineValues, basisOrdinalFilter, fineBasis->getCardinality());
              FieldContainer<double> mappedValues = subBasisMap->mapFineData(fineFilteredValues);
              vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
              for (int i=0; i<globalIndices.size(); i++)
              {
                fineGlobalValues[globalIndices[i]] += mappedValues(i);
              }
            }
            FieldContainer<double> constrainingBasisValues(constrainingBasis->getCardinality());
            for (int basisOrdinal=0; basisOrdinal<constrainingBasis->getCardinality(); basisOrdinal++)
            {
              constrainingBasisValues(basisOrdinal) = constrainingBasisValuesAllPoints(basisOrdinal,pointOrdinal);
            }
            map<GlobalIndexType,double> coarseGlobalValues;
            for (auto subBasisMap : coarseBasisMap)
            {
              // filter constrainingBasisValues here according to what subBasisMap knows about.
              set<unsigned> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
              FieldContainer<double> filteredConstrainingValues(basisOrdinalFilter.size());
              filterFCValues(filteredConstrainingValues, constrainingBasisValues, basisOrdinalFilter,
                             constrainingBasis->getCardinality());
              FieldContainer<double> mappedValues = subBasisMap->mapFineData(filteredConstrainingValues);
              vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
              for (int i=0; i<globalIndices.size(); i++)
              {
                coarseGlobalValues[globalIndices[i]] += mappedValues(i);
              }
            }

            map<GlobalIndexType,double> nonzeroCoarseGlobalValues;
            map<GlobalIndexType,double> nonzeroFineGlobalValues;
            double tol = 1e-14;
            for (auto entry : coarseGlobalValues)
            {
              if (abs(entry.second) > tol)
              {
                nonzeroCoarseGlobalValues[entry.first] = entry.second;
              }
            }
            // it can happen that fine basis participates in some global dofs that
            // coarse does not.  Below, we filter not only for nonzeros, but also to
            // eliminate any dofs that coarse basis does not see.
            for (auto entry : fineGlobalValues)
            {
              bool coarseBasisSkips = (coarseGlobalValues.find(entry.first) == coarseGlobalValues.end());
              if ((abs(entry.second) > tol) && !coarseBasisSkips)
              {
                nonzeroFineGlobalValues[entry.first] = entry.second;
              }
            }

            // Compare coarseGlobalValues to fineGlobalValues
            if (nonzeroCoarseGlobalValues.size() != nonzeroFineGlobalValues.size())
            {
              success = false;
              cout << "Failure on fine cell " << cellID << ", side " << sideOrdinal << endl;
              out << "nonzeroCoarseGlobalValues.size() = " << nonzeroCoarseGlobalValues.size();
              out << " != " << nonzeroFineGlobalValues.size() << " = nonzeroFineGlobalValues().size()\n";
              print("nonzeroCoarseGlobalValues", nonzeroCoarseGlobalValues);
              print("nonzeroFineGlobalValues", nonzeroFineGlobalValues);
              cout << "physical point: (";
              for (int d=0; d < meshTopo->getSpaceDim(); d++)
              {
                cout << coarsePhysicalPoints(0,pointOrdinal,d);
                if (d<sideDim) cout << ", ";
              }
              cout << ")\n";
              cout << "fine reference point: (";
              for (int d=0; d < sideDim; d++)
              {
                cout << fineDomainPoints(pointOrdinal,d);
                if (d<sideDim-1) cout << ", ";
              }
              cout << ")\n";
              cout << "coarse reference point: (";
              for (int d=0; d < sideDim; d++)
              {
                cout << constrainingDomainPoints(pointOrdinal,d);
                if (d<sideDim-1) cout << ", ";
              }
              cout << ")\n";

              for (auto subBasisMap : fineBasisMap)
              {
                // filter fine values according to what subBasisMap knows about.
                set<unsigned> basisOrdinalFilter = subBasisMap->basisDofOrdinalFilter();
                FieldContainer<double> fineFilteredValues(basisOrdinalFilter.size());
                filterFCValues(fineFilteredValues, fineValues, basisOrdinalFilter, fineBasis->getCardinality());
                FieldContainer<double> mappedValues = subBasisMap->mapFineData(fineFilteredValues);
                vector<GlobalIndexType> globalIndices = subBasisMap->mappedGlobalDofOrdinals();
                print("basisOrdinalFilter",basisOrdinalFilter);
                print("globalIndices", globalIndices);
                cout << "fineFilteredValues:\n" << fineFilteredValues;
                cout << "mappedValues:\n" << mappedValues;
              }

            }
            else
            {
              for (auto coarseGlobalValue : nonzeroCoarseGlobalValues)
              {
                GlobalIndexType valueIndex = coarseGlobalValue.first;
                double valueCoarse = coarseGlobalValue.second;
                if (nonzeroFineGlobalValues.find(valueIndex) == nonzeroFineGlobalValues.end())
                {
                  out << "failure: nonzeroFineGlobalValues does not have an entry for global dof ordinal " << valueIndex << endl;
                  success = false;
                }
                else
                {
                  double valueFine = fineGlobalValues[valueIndex];
                  TEST_FLOATING_EQUALITY(valueCoarse, valueFine, tol);
                }
              }
            }
          }
        }
      }
    }
  }
}

// ! copied from DPGTests GDAMinimumRuleTests
class GDAMinimumRuleTests_UnitHexahedronBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y, double z)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    bool zMatch = (abs(z) < tol) || (abs(z-1.0) < tol);
    return xMatch || yMatch || zMatch;
  }
};

// ! copied from DPGTests GDAMinimumRuleTests
SolutionPtr poissonExactSolution3D(int horizontalCells, int verticalCells, int depthCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces)
{
  bool usePenaltyBCs = false;

  int spaceDim = 3;
  PoissonFormulation poissonForm(spaceDim, useH1Traces);

  VarPtr tau = poissonForm.tau();
  VarPtr q = poissonForm.q();

  VarPtr phi_hat = poissonForm.phi_hat();
  VarPtr psi_hat = poissonForm.psi_n_hat();

  VarPtr phi = poissonForm.phi();
  VarPtr psi = poissonForm.psi();

  BFPtr bf = poissonForm.bf();

  int testSpaceEnrichment = 3; //
  double width = 1.0, height = 1.0, depth = 1.0;

  vector<double> dimensions;
  dimensions.push_back(width);
  dimensions.push_back(height);
  dimensions.push_back(depth);

  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);
  elementCounts.push_back(depthCells);

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, testSpaceEnrichment);

  // rhs = f * v, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
  rhs->addTerm(f * q);

  IPPtr graphNorm = bf->graphNorm();

  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  SolutionPtr solution;
  if (!usePenaltyBCs)
  {
    bc->addDirichlet(phi_hat, boundary, phi_exact);
    solution = Solution::solution(mesh, bc, rhs, graphNorm);
  }
  else
  {
    solution = Solution::solution(mesh, bc, rhs, graphNorm);
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new GDAMinimumRuleTests_UnitHexahedronBoundary );

    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(phi_hat==phi_exact,entireBoundary);

    solution->setFilter(pc);
  }

  return solution;
}

// ! copied from DPGTests GDAMinimumRuleTests
SolutionPtr poissonExactSolution3DHangingNodes(int irregularity, FunctionPtr phi_exact, int H1Order)
{
  // right now, we support 1-irregular and 2-irregular
  if ((irregularity > 2) || (irregularity < 0))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only 1- and 2-irregularity supported");
  }
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2, depthCellsInitialMesh = 1;

  bool useH1Traces = true; // "true" is the more thorough test

  SolutionPtr soln = poissonExactSolution3D(horizontalCellsInitialMesh, verticalCellsInitialMesh, depthCellsInitialMesh, H1Order, phi_exact, useH1Traces);

  if (irregularity==0) return soln;

  MeshPtr mesh = soln->mesh();

  //  cout << "about to refine to make Poisson 3D hanging node mesh.\n";

  set<GlobalIndexType> cellIDs;
  cellIDs.insert(1);
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());

  if (irregularity==1) return soln;

  // now, repeat the above, but with a 2-irregular mesh.
  vector<CellPtr> children = mesh->getTopology()->getCell(1)->children();

  // childrenForSides outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  vector< vector< pair< unsigned, unsigned > > > childrenForSides = mesh->getTopology()->getCell(1)->refinementPattern()->childrenForSides();
  for (int sideOrdinal=0; sideOrdinal<childrenForSides.size(); sideOrdinal++)
  {
    vector< pair< unsigned, unsigned > > childrenForSide = childrenForSides[sideOrdinal];
    bool didRefine = false;
    for (int i=0; i<childrenForSide.size(); i++)
    {
      unsigned childOrdinal = childrenForSide[i].first;
      CellPtr child = children[childOrdinal];
      unsigned childSideOrdinal = childrenForSide[i].second;
      pair<GlobalIndexType,unsigned> neighborInfo = child->getNeighborInfo(childSideOrdinal);
      GlobalIndexType neighborCellID = neighborInfo.first;
      if (neighborCellID != -1)   // not boundary
      {
        CellPtr neighbor = mesh->getTopology()->getCell(neighborCellID);
        pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(neighborInfo.second);
        bool neighborIsPeer = neighborNeighborInfo.first == child->cellIndex();
        if (!neighborIsPeer)   // then by refining this cell, we induce a 2-irregular mesh
        {
          set<GlobalIndexType> cellIDs;
          cellIDs.insert(child->cellIndex());
          mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
          didRefine = true;
          break;
        }
      }
    }
    if (didRefine) break;
  }

  //if (irregularity==2)
  return soln;
}

MeshPtr poissonUniformMesh(vector<int> elementWidths, int H1Order, bool useConformingTraces)
{
  int spaceDim = elementWidths.size();
  int testSpaceEnrichment = spaceDim; //
  double span = 1.0; // in each spatial dimension

  vector<double> dimensions(spaceDim,span);

  PoissonFormulation poissonForm(spaceDim, useConformingTraces);
  MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
  return mesh;
}

MeshPtr poissonUniformMesh(int spaceDim, int elementWidth, int H1Order, bool useConformingTraces)
{
  vector<int> elementCounts(spaceDim,elementWidth);
  return poissonUniformMesh(elementCounts, H1Order, useConformingTraces);
}

MeshPtr poissonIrregularMesh(int spaceDim, int irregularity, int H1Order)
{
  bool useConformingTraces = true;

  int elementWidth = 2;
  MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);

  int meshIrregularity = 0;
  vector<GlobalIndexType> cellsToRefine = {1};
  CellPtr cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
  unsigned sharedSideOrdinal = -1;
  for (int sideOrdinal=0; sideOrdinal<cellToRefine->getSideCount(); sideOrdinal++)
  {
    if (cellToRefine->getNeighbor(sideOrdinal) != Teuchos::null)
    {
      sharedSideOrdinal = sideOrdinal;
      break;
    }
  }

  while (meshIrregularity < irregularity)
  {
    mesh->hRefine(cellsToRefine);
    meshIrregularity++;

    // setup for the next refinement, if any:
    auto childEntry = cellToRefine->childrenForSide(sharedSideOrdinal)[0];
    GlobalIndexType childWithNeighborCellID = childEntry.first;
    sharedSideOrdinal = childEntry.second;
    cellsToRefine = {childWithNeighborCellID};
    cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
  }
  return mesh;
}

MeshPtr poisson3DUniformMesh()
{
  return poissonUniformMesh(3, 2, 2, true);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson1D)
{
  int spaceDim = 1;
  int elementWidth = 2;
  int H1Order = 2;
  bool useConformingTraces = true;
  MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DUniform)
{
  int spaceDim = 2;
  int elementWidth = 2;
  int H1Order = 2;
  bool useConformingTraces = true;
  MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DHangingNode1Irregular_Slow)
{
  int spaceDim = 2;
  int H1Order = 2;
  int irregularity = 1;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson2DHangingNode2Irregular_Slow)
{
  int spaceDim = 2;
  int H1Order = 2;
  int irregularity = 2;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DUniform_Slow)
{
  int spaceDim = 3;
  int elementWidth = 2;
  int H1Order = 2;
  bool useConformingTraces = true;
  MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DHangingNode1Irregular_Slow)
{
  int irregularity = 1;
  int spaceDim = 3;
  int H1Order = 2;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, BasisMapsAgreePoisson3DHangingNode2Irregular_Slow)
{
  int irregularity = 2;
  int spaceDim = 3;
  int H1Order = 2;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testCoarseBasisEqualsWeightedFineBasis(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DUniform )
{
  MeshPtr mesh = poisson3DUniformMesh();
  testSubcellConstraintIsAncestor(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DHangingNode1Irregular )
{
  int irregularity = 1;
  int spaceDim = 3;
  int H1Order = 2;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testSubcellConstraintIsAncestor(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, CheckConstraintsPoisson3DHangingNode2Irregular )
{
  int irregularity = 2;
  int spaceDim = 3;
  int H1Order = 2;
  MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
  testSubcellConstraintIsAncestor(mesh, out, success);
}

TEUCHOS_UNIT_TEST( GDAMinimumRule, SolvePoisson3DHangingNode_Slow )
{
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  //  FunctionPtr phi_exact = x + y;
  FunctionPtr phi_exact = -x + y + z;
  //  FunctionPtr phi_exact = Function::constant(3.14159);

  int H1Order = 2; // 1 higher than the order of phi_exact, to get an exactly recoverable solution with L^2 fields.
  int spaceDim = 3;
  bool useConformingTraces = true;
  PoissonFormulation poissonForm(spaceDim, useConformingTraces);
  // TODO: try this with 2-irregular.  Once upon a time, there was an exception thrown for that case.
  for (int irregularity = 1; irregularity<=1; irregularity++)
  {
    SolutionPtr soln = poissonExactSolution3DHangingNodes(irregularity,phi_exact,H1Order);

    MeshPtr mesh = soln->mesh();
    VarFactoryPtr vf = soln->mesh()->bilinearForm()->varFactory();

    if (! MeshTestUtility::checkLocalGlobalConsistency(mesh) )
    {
      cout << "FAILURE: " << irregularity << "-irregular Poisson 3D mesh fails local-to-global consistency check.\n";
      success = false;
      //    return success;
    }

    VarPtr phi = poissonForm.phi();
    VarPtr phi_hat = poissonForm.phi_hat();

    map<int, FunctionPtr> phi_exact_map;
    phi_exact_map[phi->ID()] = phi_exact;
    soln->projectOntoMesh(phi_exact_map);

    FunctionPtr phi_soln = Function::solution(phi, soln);
    FunctionPtr phi_err = phi_soln - phi_exact;

    FunctionPtr phi_hat_soln = Function::solution(phi_hat, soln);

    double tol = 1e-12;
    double phi_err_l2 = phi_err->l2norm(mesh);

    soln->clear();
    soln->solve();

    //    cout << irregularity << "-irregular 3D poisson w/hanging node solved.  About to check solution continuities.\n";

    Epetra_MultiVector *lhsVector = soln->getGlobalCoefficients();
    Epetra_SerialComm Comm;
    Epetra_Map partMap = soln->getPartitionMap();

    // Import solution onto current processor
    GlobalIndexTypeToCast numNodesGlobal = mesh->numGlobalDofs();
    GlobalIndexTypeToCast numMyNodes = numNodesGlobal;
    Epetra_Map     solnMap(numNodesGlobal, numMyNodes, 0, Comm);
    Epetra_Import  solnImporter(solnMap, partMap);
    Epetra_Vector  solnCoeff(solnMap);
    solnCoeff.Import(*lhsVector, solnImporter, Insert);

    if ( ! MeshTestUtility::neighborBasesAgreeOnSides(mesh, solnCoeff))
    {
      cout << "GDAMinimumRuleTests failure: for" << irregularity << "-irregular 3D Poisson mesh with hanging nodes (after solving), neighboring bases do not agree on sides." << endl;
      success = false;
    }

    //    cout << "...solution continuities checked.\n";

    phi_err_l2 = phi_err->l2norm(mesh);
    if (phi_err_l2 > tol)
    {
      success = false;
      cout << "GDAMinimumRuleTests failure: for " << irregularity << "-irregular 3D mesh and exactly recoverable solution, phi error is " << phi_err_l2 << endl;

      string outputSuperDir = ".";
      string outputDir = "poisson3DHangingNode";
      HDF5Exporter exporter(mesh, outputDir, outputSuperDir);
      cout << "Writing phi err to " << outputSuperDir << "/" << outputDir << endl;

      exporter.exportFunction(phi_err, "phi_err");
    }
  }
}
} // namespace