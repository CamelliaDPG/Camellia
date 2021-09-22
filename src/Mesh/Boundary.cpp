//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#include "Intrepid_PointTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "BasisFactory.h"
#include "BC.h"
#include "BCFunction.h"
#include "Boundary.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"
#include "Projector.h"
#include "TensorBasis.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace std;

Boundary::Boundary()
{
  _mesh = Teuchos::null;
}

void Boundary::setMesh(MeshPtr mesh)
{
  _mesh = mesh;
  buildLookupTables();
}

void Boundary::buildLookupTables()
{
  _boundaryElements.clear();

  const set<GlobalIndexType>* rankLocalCells = &_mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : *rankLocalCells)
  {
    CellPtr cell = _mesh->getTopology()->getCell(cellID);
    vector<unsigned> boundarySides = cell->boundarySides();
    for (int i=0; i<boundarySides.size(); i++)
    {
      _boundaryElements.insert(make_pair(cellID, boundarySides[i]));
    }
  }
}

template <typename Scalar>
void Boundary::bcsToImpose(Camellia::FieldContainer<GlobalIndexType> &globalIndices,
                           Camellia::FieldContainer<Scalar> &globalValues, TBC<Scalar> &bc,
                           DofInterpreter* dofInterpreter)
{
  set< GlobalIndexType > rankLocalCells = _mesh->cellIDsInPartition();
  vector<pair<GlobalIndexType, double>> bcGlobalIndicesAndValues;

  for (GlobalIndexType cellID : rankLocalCells)
  {
    bcsToImpose(bcGlobalIndicesAndValues, bc, cellID, dofInterpreter);
  }
  singletonBCsToImpose(bcGlobalIndicesAndValues, bc, dofInterpreter);
  
  // ****** New, tag-based BC imposition follows ******
  map< GlobalIndexType, double> bcTagGlobalIndicesAndValues;
  
  map< int, vector<pair<VarPtr, TFunctionPtr<Scalar>>>> tagBCs = bc.getDirichletTagBCs(); // keys are tags
  
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_mesh->getTopology().get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "pure MeshTopologyViews are not yet supported by new tag-based BC imposition");
  
  for (auto tagBC : tagBCs)
  {
    int tagID = tagBC.first;
    
    vector<EntitySetPtr> entitySets = meshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, tagID);
    for (EntitySetPtr entitySet : entitySets)
    {
      // get rank-local cells that match the entity set:
      set<IndexType> matchingCellIDs = entitySet->cellIDsThatMatch(_mesh->getTopology(), rankLocalCells);
      for (IndexType cellID : matchingCellIDs)
      {
        ElementTypePtr elemType = _mesh->getElementType(cellID);
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
        
        for (auto varFunctionPair : tagBC.second)
        {
          VarPtr var = varFunctionPair.first;
          FunctionPtr f = varFunctionPair.second;
          
          vector<int> sideOrdinals = elemType->trialOrderPtr->getSidesForVarID(var->ID());
          
          for (int sideOrdinal : sideOrdinals)
          {
            BasisPtr basis = elemType->trialOrderPtr->getBasis(var->ID(), sideOrdinal);
            bool isVolume = basis->domainTopology()->getDimension() == _mesh->getDimension();
            for (int d=0; d<_mesh->getDimension(); d++)
            {
              vector<unsigned> matchingSubcells;
              if (isVolume)
                matchingSubcells = entitySet->subcellOrdinals(_mesh->getTopology(), cellID, d);
              else
              {
                CellTopoPtr cellTopo = elemType->cellTopoPtr;
                int sideDim = cellTopo->getDimension() - 1;
                vector<unsigned> matchingSubcellsOnSide = entitySet->subcellOrdinalsOnSide(_mesh->getTopology(), cellID, sideOrdinal, d);
                for (unsigned sideSubcellOrdinal : matchingSubcellsOnSide)
                {
                  unsigned cellSubcellOrdinal = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, sideOrdinal, d, sideSubcellOrdinal);
                  matchingSubcells.push_back(cellSubcellOrdinal);
                }
              }
              
              if (matchingSubcells.size() == 0) continue; // nothing to impose
              
              /*
               What follows - projecting the function onto the basis on the whole domain - is more expensive than necessary,
               in the general case: we can do the projection on just the matching subcells, and if we had a way of taking the
               restriction of a basis to a subcell of the domain, then we could avoid computing the whole basis as well.
               
               But for now, this should work, and it's simple to implement.
               */
              BasisCachePtr basisCacheForImposition = isVolume ? basisCache : basisCache->getSideBasisCache(sideOrdinal);
              int numCells = 1;
              Camellia::FieldContainer<double> basisCoefficients(numCells,basis->getCardinality());
              Projector<double>::projectFunctionOntoBasisInterpolating(basisCoefficients, f, basis, basisCacheForImposition);
              basisCoefficients.resize(basis->getCardinality());
              
//              {
//                // DEBUGGING
//                bool hasNonZeros = false;
//                for (int i=0; i<basisCoefficients.size(); i++)
//                {
//                  if (basisCoefficients[i] != 0.0)
//                  {
//                    hasNonZeros = true;
//                  }
//                }
//
//                if (hasNonZeros)
//                {
//                  std::cout << "basisCoefficients:\n" << basisCoefficients;
//                }
//              }
              
              set<GlobalIndexType> matchingGlobalIndices;
              for (unsigned matchingSubcell : matchingSubcells)
              {
                const bool upgradeToContinuousBasis = true;
                const bool tensorTopologyMeansSpaceTime = _mesh->tensorTopologyMeansSpaceTime();
                set<GlobalIndexType> subcellGlobalIndices = dofInterpreter->globalDofIndicesForVarOnSubcell(var->ID(),cellID,d,matchingSubcell,upgradeToContinuousBasis,tensorTopologyMeansSpaceTime);
                matchingGlobalIndices.insert(subcellGlobalIndices.begin(),subcellGlobalIndices.end());
              }
              
              Camellia::FieldContainer<double> globalData;
              Camellia::FieldContainer<GlobalIndexType> globalDofIndices;
//              dofInterpreter->interpretLocalBasisCoefficients(cellID, var->ID(), sideOrdinal, basisCoefficientsToImpose, globalData, globalDofIndices);
              dofInterpreter->interpretLocalBasisCoefficients(cellID, var->ID(), sideOrdinal, basisCoefficients, globalData, globalDofIndices);
//              {
//                // DEBUGGING
//                bool hasNonZeros = false;
//                for (int i=0; i<globalData.size(); i++)
//                {
//                  if (globalData[i] != 0.0)
//                  {
//                    hasNonZeros = true;
//                  }
//                }
//                
//                if (hasNonZeros)
//                {
//                  std::cout << "globalData for side " << sideOrdinal << ":\n" << globalData << std::endl;
//                }
//              }
              
              for (int globalDofOrdinal=0; globalDofOrdinal<globalDofIndices.size(); globalDofOrdinal++)
              {
                GlobalIndexType globalDofIndex = globalDofIndices(globalDofOrdinal);
                if (matchingGlobalIndices.find(globalDofIndex) != matchingGlobalIndices.end())
                {
                  bcTagGlobalIndicesAndValues[globalDofIndex] = globalData(globalDofOrdinal);
//                  std::cout << "setting bcTagGlobalIndicesAndValues[" << globalDofIndex << "] = " << globalData(globalDofOrdinal) << std::endl;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // merge tag-based and legacy BC maps
//  double tol = 1e-15;
  for (auto tagEntry : bcTagGlobalIndicesAndValues)
  {
    bcGlobalIndicesAndValues.push_back(tagEntry);
//    if (bcGlobalIndicesAndValues.find(tagEntry.first) != bcGlobalIndicesAndValues.end())
//    {
//      // then check that they match, within tolerance
//      double diff = abs(bcGlobalIndicesAndValues[tagEntry.first] - tagEntry.second);
//      TEUCHOS_TEST_FOR_EXCEPTION(diff > tol, std::invalid_argument, "Incompatible BC entries encountered");
//    }
//    else
//    {
//      bcGlobalIndicesAndValues[tagEntry.first] = tagEntry.second;
//    }
  }
  
  globalIndices.resize(bcGlobalIndicesAndValues.size());
  globalValues.resize(bcGlobalIndicesAndValues.size());
  globalIndices.initialize(0);
  globalValues.initialize(0.0);
  int entryOrdinal = 0;
  for (auto bcEntry : bcGlobalIndicesAndValues)
  {
    globalIndices[entryOrdinal] = bcEntry.first;
    globalValues[entryOrdinal] = bcEntry.second;
    entryOrdinal++;
  }
}

template <typename Scalar>
void Boundary::bcsToImpose( vector<pair<GlobalIndexType,Scalar>> &globalDofIndicesAndValues, TBC<Scalar> &bc,
                            GlobalIndexType cellID, DofInterpreter* dofInterpreter)
{
  // this is where we actually compute the BCs; the other bcsToImpose variants call this one.
  CellPtr cell = _mesh->getTopology()->getCell(cellID);

  // define a couple of important inner products:
  TIPPtr<Scalar> ipL2 = Teuchos::rcp( new TIP<Scalar> );
  TIPPtr<Scalar> ipH1 = Teuchos::rcp( new TIP<Scalar> );
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr trace = varFactory->traceVar("trace");
  VarPtr flux = varFactory->traceVar("flux");
  ipL2->addTerm(flux);
  ipH1->addTerm(trace);
  ipH1->addTerm(trace->grad());
  ElementTypePtr elemType = _mesh->getElementType(cellID);
  DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();

  vector<unsigned> boundarySides = cell->boundarySides();
  if (boundarySides.size() > 0)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
    for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
    {
      int trialID = *(trialIt);
      if ( bc.bcsImposed(trialID) )
      {
//        // DEBUGGING: keep track of which sides we impose BCs on:
//        set<unsigned> bcImposedSides;
//
        // Determine global dof indices and values, in one pass per side
        for (int i=0; i<boundarySides.size(); i++)
        {
          unsigned sideOrdinal = boundarySides[i];
          // TODO: here, we need to treat the volume basis case.
          /*
           To do this:
           1. (Determine which dofs in the basis have support on the side.)
           2. (Probably should resize dirichletValues to match number of dofs with support on the side.)
           3. (Within coefficientsForBC, and the projection method it calls, when it's a side cache, check whether the basis being projected has a higher dimension.  If so, do the same determination regarding the support of basis on the side as #1.)
           4. DofInterpreter::interpretLocalBasisCoefficients() needs to handle the case that trialID has volume support, and in this case interpret the provided data appropriately.
           */
          
          BasisPtr basis;
          int numDofsSide;
          if (trialOrderingPtr->getSidesForVarID(trialID).size() == 1)
          {
            // volume basis
            basis = trialOrderingPtr->getBasis(trialID);
            // get the dof ordinals for the side (interpreted as a "continuous" basis)
            numDofsSide = basis->dofOrdinalsForSide(sideOrdinal).size();
          }
          else if (! trialOrderingPtr->hasBasisEntry(trialID, sideOrdinal))
          {
            continue;
          }
          else
          {
            basis = trialOrderingPtr->getBasis(trialID,sideOrdinal);
            numDofsSide = basis->getCardinality();
          }
          
          {
            const GlobalIndexType numCells = 1;
            const int cellOrdinal = 0;
            Camellia::FieldContainer<double> dirichletValues(numCells,numDofsSide);
            // project bc function onto side basis:
            BCPtr bcPtr = Teuchos::rcp(&bc, false);
            Teuchos::RCP<BCFunction<double>> bcFunction = BCFunction<double>::bcFunction(bcPtr, trialID);
            bcPtr->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideOrdinal));
            dirichletValues.resize(numDofsSide);
            
            if (bcFunction->imposeOnCell(cellOrdinal))
            {
              Camellia::FieldContainer<double> globalData;
              Camellia::FieldContainer<GlobalIndexType> globalDofIndices;

              dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, dirichletValues, globalData, globalDofIndices);
              for (int globalDofOrdinal=0; globalDofOrdinal<globalDofIndices.size(); globalDofOrdinal++)
              {
                GlobalIndexType globalDofIndex = globalDofIndices(globalDofOrdinal);
                Scalar value = globalData(globalDofOrdinal);
                
//                // sanity check: if this has been previously set, do the two values roughly agree?
//                if (globalDofIndicesAndValues.find(globalDofIndex) != globalDofIndicesAndValues.end())
//                {
//                  double tol = 1e-10;
//                  Scalar prevValue = globalDofIndicesAndValues[globalDofIndex];
//                  double absDiff = abs(prevValue - value);
//                  if (absDiff > tol)
//                  {
//                    double relativeDiff = absDiff / max(abs(prevValue),abs(value));
//                    int rank = _mesh->Comm()->MyPID();
//                    if (relativeDiff > tol)
//                    {
//                      cout << "WARNING: in Boundary::bcsToImpose(), inconsistent values for BC: " << prevValue << " and ";
//                      cout << value << " prescribed for global dof index " << globalDofIndex;
//                      cout << " on rank " << rank << endl;
//                    }
//                  }
//                }
//                globalDofIndicesAndValues[globalDofIndex] = value;
                globalDofIndicesAndValues.push_back({globalDofIndex,value});
              }
            }
          }
        }
      }
    }
  }
}

template <typename Scalar>
void Boundary::singletonBCsToImpose(vector<pair<GlobalIndexType,Scalar>> &dofIndexToValue, TBC<Scalar> &bc,
                                    DofInterpreter* dofInterpreter)
{
  // first, let's check for any singletons (one-point BCs)
  map<IndexType, set < pair<int, unsigned> > > singletonsForCell;
  
  const set<GlobalIndexType>* rankLocalCells = &_mesh->cellIDsInPartition();
  
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  for (int trialID : trialIDs)
  {
    if (bc.singlePointBC(trialID))
    {
      vector<double> spatialVertex = bc.pointForSpatialPointBC(trialID);
      vector<IndexType> matchingVertices = _mesh->getTopology()->getVertexIndicesMatching(spatialVertex);
      
      unsigned vertexDim = 0;
      for (IndexType vertexIndex : matchingVertices)
      {
        set< pair<IndexType, unsigned> > cellsForVertex = _mesh->getTopology()->getCellsContainingEntity(vertexDim, vertexIndex);
        for (pair<IndexType, unsigned> cellForVertex : cellsForVertex)
        {
          // active cell
          IndexType matchingCellID = cellForVertex.first;
          
          if (rankLocalCells->find(matchingCellID) != rankLocalCells->end())   // we own this cell, so we're responsible for imposing the singleton BC
          {
            CellPtr cell = _mesh->getTopology()->getCell(matchingCellID);
            unsigned vertexOrdinal = cell->findSubcellOrdinal(vertexDim, vertexIndex);
            TEUCHOS_TEST_FOR_EXCEPTION(vertexOrdinal == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL, std::invalid_argument, "Internal error: vertexOrdinal not found for cell to which it supposedly belongs");
            singletonsForCell[matchingCellID].insert(make_pair(trialID, vertexOrdinal));
          }
        }
      }
    }
  }
  
  for (auto cellEntry : singletonsForCell)
  {
    GlobalIndexType cellID = cellEntry.first;
    auto singletons = cellEntry.second;
    ElementTypePtr elemType = _mesh->getElementType(cellID);
    
    map<int, vector<unsigned> > vertexOrdinalsForTrialID;
    for (pair<int, unsigned> trialVertexPair : singletons)
    {
      vertexOrdinalsForTrialID[trialVertexPair.first].push_back(trialVertexPair.second);
    }
    
    for (auto trialVertexOrdinals : vertexOrdinalsForTrialID)
    {
      int trialID = trialVertexOrdinals.first;
      vector<unsigned> vertexOrdinalsInCell = trialVertexOrdinals.second;
      
      CellTopoPtr cellTopo = elemType->cellTopoPtr;
      CellTopoPtr spatialCellTopo;
      
      bool spaceTime;
      int vertexOrdinalInSpatialCell;
      if (vertexOrdinalsInCell.size() == 2)
      {
        // we'd better be doing space-time in this case, and the vertices should be the same spatially
        spaceTime = (cellTopo->getTensorialDegree() > 0);
        TEUCHOS_TEST_FOR_EXCEPTION(!spaceTime, std::invalid_argument, "multiple vertices for spatial point only supported for space-time");
        
        spatialCellTopo = cellTopo->getTensorialComponent();
        
        vertexOrdinalInSpatialCell = -1;
        for (unsigned spatialVertexOrdinal = 0; spatialVertexOrdinal < spatialCellTopo->getNodeCount(); spatialVertexOrdinal++)
        {
          vector<unsigned> tensorComponentNodes = {spatialVertexOrdinal,0};
          unsigned spaceTimeVertexOrdinal_t0 = cellTopo->getNodeFromTensorialComponentNodes(tensorComponentNodes);
          if ((spaceTimeVertexOrdinal_t0 == vertexOrdinalsInCell[0]) || (spaceTimeVertexOrdinal_t0 == vertexOrdinalsInCell[1]))
          {
            // then this should be our match.  Confirm this:
            tensorComponentNodes = {spatialVertexOrdinal,1};
            unsigned spaceTimeVertexOrdinal_t1 = cellTopo->getNodeFromTensorialComponentNodes(tensorComponentNodes);
            bool t1VertexMatches = (spaceTimeVertexOrdinal_t1 == vertexOrdinalsInCell[0]) || (spaceTimeVertexOrdinal_t1 == vertexOrdinalsInCell[1]);
            TEUCHOS_TEST_FOR_EXCEPTION(!t1VertexMatches, std::invalid_argument, "Internal error: space-time vertices do not belong to the same spatial vertex");
            vertexOrdinalInSpatialCell = spatialVertexOrdinal;
            break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(vertexOrdinalInSpatialCell == -1, std::invalid_argument, "Internal error: spatial vertex ordinal not found");
      }
      else if (vertexOrdinalsInCell.size() == 1)
      {
        spaceTime = false;
        spatialCellTopo = cellTopo;
        vertexOrdinalInSpatialCell = vertexOrdinalsInCell[0];
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertexOrdinalsInCell must have 1 or 2 vertices");
      }
      
      set<GlobalIndexType> globalIndicesForVariable;
      DofOrderingPtr trialOrderingPtr = elemType->trialOrderPtr;
      
      int numSpatialSides = spatialCellTopo->getSideCount();
      
      vector<unsigned> spatialSidesForVertex;
      vector<unsigned> sideVertexOrdinals; // same index in this container as spatialSidesForVertex: gets the node ordinal of the vertex in that side
      int sideDim = spatialCellTopo->getDimension() - 1;
      if (!_mesh->bilinearForm()->isFluxOrTrace(trialID))
      {
        spatialSidesForVertex.push_back(VOLUME_INTERIOR_SIDE_ORDINAL);
        sideVertexOrdinals.push_back(vertexOrdinalInSpatialCell);
      }
      else
      {
        for (int spatialSideOrdinal=0; spatialSideOrdinal < numSpatialSides; spatialSideOrdinal++)
        {
          CellTopoPtr sideTopo = spatialCellTopo->getSide(spatialSideOrdinal);
          for (unsigned sideVertexOrdinal = 0; sideVertexOrdinal < sideTopo->getNodeCount(); sideVertexOrdinal++)
          {
            unsigned spatialVertexOrdinal = spatialCellTopo->getNodeMap(sideDim, spatialSideOrdinal, sideVertexOrdinal);
            if (spatialVertexOrdinal == vertexOrdinalInSpatialCell)
            {
              spatialSidesForVertex.push_back(spatialSideOrdinal);
              sideVertexOrdinals.push_back(sideVertexOrdinal);
              break; // this is the only match we should find on this side
            }
          }
        }
        if (spatialSidesForVertex.size() == 0)
        {
          cout << "ERROR: no spatial side for vertex found during singleton BC imposition.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no spatial side for vertex found during singleton BC imposition");
        }
      }
      for (int i=0; i<spatialSidesForVertex.size(); i++)
      {
        unsigned spatialSideOrdinal = spatialSidesForVertex[i];
        unsigned vertexOrdinalInSide = sideVertexOrdinals[i];
        unsigned sideForImposition;
        
        BasisPtr spatialBasis, temporalBasis, spaceTimeBasis, basisForImposition;
        if (!spaceTime)
        {
          spatialBasis = trialOrderingPtr->getBasis(trialID,spatialSideOrdinal);
          sideForImposition = spatialSideOrdinal;
        }
        else
        {
          unsigned spaceTimeSideOrdinal;
          if (!_mesh->bilinearForm()->isFluxOrTrace(trialID))
          {
            spaceTimeSideOrdinal = VOLUME_INTERIOR_SIDE_ORDINAL;
          }
          else
          {
            spaceTimeSideOrdinal = cellTopo->getSpatialSideOrdinal(spatialSideOrdinal);
          }
          spaceTimeBasis = trialOrderingPtr->getBasis(trialID,spaceTimeSideOrdinal);
          
          sideForImposition = spaceTimeSideOrdinal;
          
          TensorBasis<>* tensorBasis = dynamic_cast<TensorBasis<>*>(spaceTimeBasis.get());
          
          TEUCHOS_TEST_FOR_EXCEPTION(!tensorBasis, std::invalid_argument, "space-time basis must be a subclass of TensorBasis");
          if (tensorBasis)
          {
            spatialBasis = tensorBasis->getSpatialBasis();
            temporalBasis = tensorBasis->getTemporalBasis();
          }
        }
        // upgrade bases to continuous ones of the same cardinality, if they are discontinuous.
        const bool spaceTime = _mesh->tensorTopologyMeansSpaceTime() && (elemType->cellTopoPtr->sideIsSpatial(sideForImposition) || (sideForImposition == VOLUME_INTERIOR_SIDE_ORDINAL));
        BasisPtr basis = trialOrderingPtr->getBasis(trialID,sideForImposition);
        basisForImposition = BasisFactory::basisFactory()->getContinuousBasis(basis, spaceTime);
        
        const bool basisIsConstant = basis->getCardinality() == 1;
        
        vector<int> spatialDofOrdinalsForVertex = basisIsConstant ? vector<int>{0} : basisForImposition->dofOrdinalsForVertex(vertexOrdinalInSide);
        if (spatialDofOrdinalsForVertex.size() != 1)
        {
          cout << "ERROR: spatialDofOrdinalsForVertex.size() != 1 during singleton BC imposition.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spatialDofOrdinalsForVertex.size() != 1 during singleton BC imposition");
        }
        
        int spatialDofOrdinalForVertex = spatialDofOrdinalsForVertex[0];
        vector<int> basisDofOrdinals;
        if (!spaceTime)
        {
          basisDofOrdinals.push_back(spatialDofOrdinalForVertex);
        }
        else
        {
          int temporalBasisCardinality = temporalBasis->getCardinality();
          TensorBasis<>* tensorBasis = dynamic_cast<TensorBasis<>*>(spaceTimeBasis.get());
          for (int temporalBasisOrdinal=0; temporalBasisOrdinal<temporalBasisCardinality; temporalBasisOrdinal++)
          {
            basisDofOrdinals.push_back(tensorBasis->getDofOrdinalFromComponentDofOrdinals({spatialDofOrdinalForVertex, temporalBasisOrdinal}));
          }
        }
        
        for (int dofOrdinal : basisDofOrdinals)
        {
          Camellia::FieldContainer<double> basisCoefficients(basisForImposition->getCardinality());
          basisCoefficients[dofOrdinal] = 1.0;
          Camellia::FieldContainer<double> globalCoefficients;
          Camellia::FieldContainer<GlobalIndexType> globalDofIndices;
          dofInterpreter->interpretLocalBasisCoefficients(cellID, trialID, sideForImposition, basisCoefficients,
                                                          globalCoefficients, globalDofIndices);
          double tol = 1e-14;
          int nonzeroEntryOrdinal = -1;
          for (int fieldOrdinal=0; fieldOrdinal < globalCoefficients.size(); fieldOrdinal++)
          {
            if (abs(globalCoefficients[fieldOrdinal]) > tol)
            {
              if (nonzeroEntryOrdinal != -1)
              {
                // previous nonzero entry found; this is a problem--it means we have multiple global coefficients that depend on this vertex
                // (could happen if user specified a hanging node)
                cout << "Error: vertex for single-point imposition has multiple global degrees of freedom.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: vertex for single-point imposition has multiple global degrees of freedom.");
              }
              // nonzero entry: store the fact, and impose the constraint
              nonzeroEntryOrdinal = fieldOrdinal;
              
              bool isRankLocal = dofInterpreter->isLocallyOwnedGlobalDofIndex(globalDofIndices[fieldOrdinal]);
              if (isRankLocal)
              {
                dofIndexToValue.push_back({globalDofIndices[fieldOrdinal],
                                           bc.valueForSinglePointBC(trialID) * globalCoefficients[fieldOrdinal]});
              }
              else
              {
                // Rather than being an error, I *think* we can trust that the owner of the global dof index will be able to apply, and we can safely ignore here...
//                cout << "ERROR: global dof index for single-point BC is not locally owned.\n";
//                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: global dof index for single-point BC is not locally owned");
              }
            }
          }
        }
      }
    }
  }
}

namespace Camellia
{
  template void Boundary::bcsToImpose(Camellia::FieldContainer<GlobalIndexType> &globalIndices,
                                      Camellia::FieldContainer<double> &globalValues, TBC<double> &bc,
                                      DofInterpreter* dofInterpreter);
  template void Boundary::bcsToImpose(vector<pair<GlobalIndexType, double>> &globalDofIndicesAndValues,
                                      TBC<double> &bc, GlobalIndexType cellID,
                                      DofInterpreter* dofInterpreter);
  template void Boundary::singletonBCsToImpose(vector<pair<GlobalIndexType,double>> &dofIndexToValue, TBC<double> &bc,
                                               DofInterpreter* dofInterpreter);
}
