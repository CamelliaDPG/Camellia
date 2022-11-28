// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  ExtrudedMeshTopology.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/14/21.
//

#include "ExtrudedMeshTopology.h"

namespace Camellia
{
  RefinementPatternPtr ExtrudedMeshTopology::physicalRefinement(RefinementPatternPtr phaseRefinement)
  {
    auto physicalRefinementEntry = _phaseToPhysicalRefinementMap.find(phaseRefinement->getKey());
    if ( physicalRefinementEntry == _phaseToPhysicalRefinementMap.end())
    {
      const auto numPhysicalDims = _lowerDimensionalMeshTopology->getDimension();
      auto phaseCellTopo = phaseRefinement->parentTopology();
      
      auto refPattern = phaseRefinement;
      auto cellTopo = phaseCellTopo;
      while (cellTopo->getDimension() != numPhysicalDims)
      {
        auto sideOrdinal = cellTopo->getTemporalSideOrdinal(0);
        refPattern = refPattern->sideRefinementPatterns()[sideOrdinal];
        cellTopo = refPattern->parentTopology();
      }
      
      _phaseToPhysicalRefinementMap[phaseRefinement->getKey()] = refPattern;
      return refPattern;
    }
    return physicalRefinementEntry->second;
  }

namespace
{
  //! private method to allow us to use delegate constructor below
  std::vector< std::vector<double> > determineExtrusionVertices(std::vector<double> startCoords,
                                                                std::vector<double> extents,
                                                                std::vector<int> initialSubdivisions)
  {
    const int numExtrusionDims = startCoords.size();
    
    TEUCHOS_TEST_FOR_EXCEPTION(numExtrusionDims != int(extents.size()),             std::invalid_argument, "startCoords, extents, and initialSubdivisions arguments must all match in length");
    TEUCHOS_TEST_FOR_EXCEPTION(numExtrusionDims != int(initialSubdivisions.size()), std::invalid_argument, "startCoords, extents, and initialSubdivisions arguments must all match in length");
    
    using std::vector;
    vector<vector<double>> extrusionVertices(numExtrusionDims);
    for (int d=0; d<numExtrusionDims; d++)
    {
      const double startCoord = startCoords[d];
      const double extent = extents[d];
      extrusionVertices[d].push_back(startCoord);
      const int numSubdivisions = initialSubdivisions[d];
      for (int i=0; i<numSubdivisions; i++)
      {
        const double coord = startCoord + ((i + 1) * extent) / numSubdivisions;
        extrusionVertices[d].push_back(coord);
      }
    }
    return extrusionVertices;
  }
}
  ExtrudedMeshTopology::ExtrudedMeshTopology(MeshTopologyPtr lowerDimensionalMeshTopology,
                                             std::vector<double> startCoords,
                                             std::vector<double> extents,
                                             std::vector<int> initialSubdivisions)
  :
  ExtrudedMeshTopology(lowerDimensionalMeshTopology,
                       determineExtrusionVertices(startCoords, extents, initialSubdivisions)) {}


  ExtrudedMeshTopology::ExtrudedMeshTopology(MeshTopologyPtr lowerDimensionalMeshTopology,
                                             std::vector<std::vector<double>> extrusionVertices)
  :
  _lowerDimensionalMeshTopology(lowerDimensionalMeshTopology),
  MeshTopology(lowerDimensionalMeshTopology->getDimension() + extrusionVertices.size(), lowerDimensionalMeshTopology->getPeriodicBCs())
  {
    const int numPhysicalDimensions = lowerDimensionalMeshTopology->getDimension();
    const int numExtrusionDimensions = extrusionVertices.size();
    
    _componentToPhaseMaps = std::vector<std::map<IndexType,std::set<IndexType>> >(numExtrusionDimensions+1); // first map is for physical MeshTopology, then one per extrusion dimension
    
    _extrusionMeshTopologies = std::vector<MeshTopologyPtr>(numExtrusionDimensions);
    int vertexMultiplier = 1; // doubles with each extrusion dimension
    
    TEUCHOS_TEST_FOR_EXCEPTION(lowerDimensionalMeshTopology->isDistributed(), std::invalid_argument, "distributed lower-dimensional MeshTopology is not (yet) supported");
    const auto & physicalCellIDs = _lowerDimensionalMeshTopology->MeshTopologyView::getLocallyKnownActiveCellIndices();
    
    // Because TensorArgumentIterator has the last entry as fastest-moving, and we want lower dimensions to be fastest-moving,
    // we put tensorComponents in reverse dimensional order (i.e., physical elements come last)
    std::vector< CellDataHost > tensorComponents(numExtrusionDimensions+1);
    {
      const IndexType numElements = physicalCellIDs.size();
      Kokkos::View<IndexType*,Kokkos::HostSpace> cellIndexView("Physical Cells View", numElements);
      for (IndexType componentCellIndex=0; componentCellIndex<numElements; componentCellIndex++)
      {
        cellIndexView(componentCellIndex) = componentCellIndex;
      }
      Kokkos::Array<int,1> extents {numElements};
      Kokkos::Array<Intrepid2::DataVariationType,1> variationTypes {Intrepid2::GENERAL};
      CellDataHost cellIndexData(cellIndexView,extents,variationTypes);
      tensorComponents[numExtrusionDimensions] = cellIndexData;
    }
    
    for (int d=0; d<numExtrusionDimensions; d++)
    {
      const int numElements = extrusionVertices[d].size() - 1;
      _extrusionMeshTopologies[d] = MeshFactory::intervalMeshTopology(extrusionVertices[d]);
      vertexMultiplier *= 2;
      
      Kokkos::View<IndexType*,Kokkos::HostSpace> cellIndexView("1D Cells View", numElements);
      Kokkos::Array<int,1> extents {numElements};
      Kokkos::Array<Intrepid2::DataVariationType,1> variationTypes {Intrepid2::GENERAL};
      
      for (IndexType componentCellIndex=0; componentCellIndex<numElements; componentCellIndex++)
      {
        cellIndexView(componentCellIndex) = componentCellIndex;
      }
      CellDataHost cellIndexData(cellIndexView,extents,variationTypes);
      tensorComponents[numExtrusionDimensions-d-1] = cellIndexData;
    }
    CellTensorDataHost tensorCells(tensorComponents);
    
    Intrepid2::TensorArgumentIterator argumentIterator(tensorCells, 0);
    std::vector<FieldContainer<double>> vertices_1D(numExtrusionDimensions, FieldContainer<double>(2, 1));
      
    do
    {
      IndexType physicalCellID = argumentIterator.argument(numExtrusionDimensions);
      CellPtr physicalCell = _lowerDimensionalMeshTopology->getCell(physicalCellID);
      const int numPhysicalVertices = physicalCell->vertices().size();
      FieldContainer<double> physicalVertices(numPhysicalVertices, numPhysicalDimensions);
      _lowerDimensionalMeshTopology->verticesForCell(physicalVertices, physicalCellID);
      
      const int numVertices = numPhysicalVertices * vertexMultiplier; // obviously, this is not a parsimonious way to store this.  If/when ExtrudedMeshTopology can take over more of the entity tracking responsibility from MeshTopology, we could store in a tensor-product format and reduce expense.
      
      FieldContainer<double> vertices(numVertices, numPhysicalDimensions + numExtrusionDimensions);
      
      for (int d=0; d<numExtrusionDimensions; d++)
      {
        IndexType cellID_d = argumentIterator.argument(numExtrusionDimensions-d-1);
        _extrusionMeshTopologies[d]->verticesForCell(vertices_1D[d], cellID_d);
      }
      
      for (int tensorVertexOrdinal=0; tensorVertexOrdinal<vertexMultiplier; tensorVertexOrdinal++)
      {
        for (int physicalVertexOrdinal=0; physicalVertexOrdinal<numPhysicalVertices; physicalVertexOrdinal++)
        {
          int vertexOrdinal = physicalVertexOrdinal + tensorVertexOrdinal * numPhysicalVertices;
          for (int d=0; d<numPhysicalDimensions; d++)
          {
            vertices(vertexOrdinal,d) = physicalVertices(physicalVertexOrdinal,d);
          }
          int tensorVertexRemainder = tensorVertexOrdinal;
          for (int d=0; d<numExtrusionDimensions; d++)
          {
            int vertexOrdinal_d = tensorVertexRemainder % 2;
            vertices(vertexOrdinal, d+numPhysicalDimensions) = vertices_1D[d](vertexOrdinal_d, 0);
            tensorVertexRemainder >>= 1;
          }
        }
      }
      
      CellTopoPtr cellTopo = CellTopology::cellTopology(physicalCell->topology(), numExtrusionDimensions);
      
      this->MeshTopology::addCell(cellTopo, vertices);
    } while (argumentIterator.increment() != -1);
  }
  
  ExtrudedMeshTopology::~ExtrudedMeshTopology() {}

  IndexType ExtrudedMeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex)
  {
    // cellVertices are ordered so that the physical cell comes first; it gets repeated according to the extrusion
    // TODO: add checks that cellVertices is consistent with this claim.
    const int physicalSpaceDim = _lowerDimensionalMeshTopology->getDimension();
    const int physicalTensorialDegree = physicalSpaceDim - cellTopo->getShardsTopology().getDimension();
    auto physicalTopo = CellTopology::cellTopology(cellTopo->getShardsTopology(), physicalTensorialDegree);
    
    const int numExtrusionDims = _extrusionMeshTopologies.size();
    
    _phaseToComponentMap[cellIndex] = std::vector<IndexType>(1 + numExtrusionDims);
    
    const int numPhysicalVertices = physicalTopo->getVertexCount();
    vector< vector<double> > physicalCellVertices(numPhysicalVertices, vector<double>(physicalSpaceDim));
    vector<double> startCoords(numExtrusionDims), endCoords(numExtrusionDims);
    for (int node=0; node<numPhysicalVertices; node++)
    {
      const auto & vertex = this->getVertex(cellVertices[node]);
      for (int d=0; d<physicalSpaceDim; d++)
      {
        physicalCellVertices[node][d] = vertex[d];
      }
    }
    
    CellPtr physicalCell = _lowerDimensionalMeshTopology->findCellWithVertices(physicalCellVertices);
    if ( physicalCell == Teuchos::null )
    {
      if (parentCellIndex == -1)
      {
        physicalCell = _lowerDimensionalMeshTopology->addCell(physicalTopo, physicalCellVertices);
      }
      else
      {
        IndexType physicalParentIndex = _phaseToComponentMap[parentCellIndex][0];
        
        RefinementPatternPtr phaseRefPattern = getCell(parentCellIndex)->refinementPattern();
        auto physicalRefPattern = physicalRefinement(phaseRefPattern);
        auto nextCellID = _lowerDimensionalMeshTopology->nextCellID();
        _lowerDimensionalMeshTopology->refineCell(physicalParentIndex, physicalRefPattern, nextCellID);
        physicalCell = _lowerDimensionalMeshTopology->findCellWithVertices(physicalCellVertices);
      }
    }
    _componentToPhaseMaps[0][physicalCell->cellIndex()].insert(cellIndex);
    _phaseToComponentMap[cellIndex][0] = physicalCell->cellIndex();
    
    auto lineTopo = CellTopology::line();
    const int numVertices1D = 2;
    const int dimension1D   = 1;
    std::vector< std::vector<double> > cellVertices1D(numVertices1D, std::vector<double>(dimension1D));
    for (int d=0; d<numExtrusionDims; d++)
    {
      const auto & vertex0 = this->getVertex(cellVertices[0]);
      const double & startCoord = vertex0[d+physicalSpaceDim];
      
      const int vertex1_phaseNumber = numPhysicalVertices << d;
      const auto & vertex1 = this->getVertex(cellVertices[vertex1_phaseNumber]);
      const double & endCoord = vertex1[d+physicalSpaceDim];
      
      cellVertices1D[0][0] = startCoord;
      cellVertices1D[1][0] = endCoord;
      
      CellPtr extrusionCell = _extrusionMeshTopologies[d]->findCellWithVertices(cellVertices1D);
      if (extrusionCell == Teuchos::null)
      {
        if (parentCellIndex == -1)
        {
          extrusionCell = _extrusionMeshTopologies[d]->addCell(lineTopo, cellVertices1D);
        }
        else
        {
          IndexType extrusionParentIndex = _phaseToComponentMap[parentCellIndex][d+1];
          auto extrusionRefPattern = RefinementPattern::regularRefinementPatternLine();
          
          auto nextCellID = _extrusionMeshTopologies[d]->nextCellID();
          _extrusionMeshTopologies[d]->refineCell(extrusionParentIndex, extrusionRefPattern, nextCellID);
          extrusionCell = _extrusionMeshTopologies[d]->findCellWithVertices(cellVertices1D);
        }
      }
      
      // TODO: consider anisotropic refinement possibility: should we delete the old component cell ID to phase space cell ID map entry?
      
      _componentToPhaseMaps[d+1][extrusionCell->cellIndex()].insert(cellIndex);
      _phaseToComponentMap[cellIndex][d+1] = extrusionCell->cellIndex();
    }
                
    // call base class
    return this->MeshTopology::addCell(cellIndex, cellTopo, cellVertices, parentCellIndex);
  }
  
  //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
  void ExtrudedMeshTopology::addVertex(const std::vector<double>& vertex)
  {
    // TODO: anything that we want in terms of checking/updating components
    this->MeshTopology::addVertex(vertex);
  }

  //! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs and the like
  Teuchos::RCP<MeshTopology> ExtrudedMeshTopology::deepCopy() const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "deepCopy() is not (yet) supported by ExtrudedMeshTopology");
  }

  bool ExtrudedMeshTopology::cellContainsPoint(GlobalIndexType cellID, const std::vector<double> &point, int cubatureDegree) const
  {
    const auto & phaseToComponentMapEntry = _phaseToComponentMap.find(cellID);
    TEUCHOS_TEST_FOR_EXCEPTION(phaseToComponentMapEntry == _phaseToComponentMap.end(), std::invalid_argument, "cell not found");
    const std::vector<IndexType> & componentCellIDsVector = phaseToComponentMapEntry->second;
    IndexType lowerDimCellID = componentCellIDsVector[0];
    bool componentsContainPoint = _lowerDimensionalMeshTopology->cellContainsPoint(lowerDimCellID, point, cubatureDegree);
    
    int physicalDimension = _lowerDimensionalMeshTopology->getDimension();
    int numExtrusions = _extrusionMeshTopologies.size();
    vector<double> point_1D(1);
    for (int d=0; d<numExtrusions; d++)
    {
      point_1D[0] = point[physicalDimension + d];
      componentsContainPoint &= _extrusionMeshTopologies[d]->cellContainsPoint(componentCellIDsVector[d+1], point_1D, cubatureDegree);
    }
    return componentsContainPoint;
  }
  
  IndexType ExtrudedMeshTopology::extrusionCellIDForPhaseSpaceCellID(const IndexType &phaseSpaceCellID, const int &extrusionDimension) const
  {
    auto entry = _phaseToComponentMap.find(phaseSpaceCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _phaseToComponentMap.end(), std::invalid_argument, "phase space cellID not found");
    return entry->second[extrusionDimension + 1]; // first entry corresponds to the physical mesh topology
  }
  
  //! returns the 1D extrusionMeshTopology (the extrusionDimension is numbered from 0, and is independent of the number of physical/base dimensions).
  MeshTopologyPtr ExtrudedMeshTopology::extrusionMeshTopology(int extrusionDimension) const
  {
    return _extrusionMeshTopologies[extrusionDimension];
  }
  
  //! computes the measure in the specified extrusion dimensions, starting with the first extrusion dimension.  Involves map lookups; consider caching the result if performance is critical.
  void ExtrudedMeshTopology::extrusionVerticesForPhaseSpaceCell(double &v0, double &v1, const IndexType &phaseSpaceCellID, int extrusionDimension) const
  {
    auto entry = _phaseToComponentMap.find(phaseSpaceCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _phaseToComponentMap.end(), std::invalid_argument, "phase space cellID not found");
    const auto &componentCellIDs = entry->second;
    
    const IndexType &extrusionCellID = componentCellIDs[1+extrusionDimension]; // first componentCellID is the physical cellID
    auto extrusionCell = _extrusionMeshTopologies[extrusionDimension]->getCell(extrusionCellID);
    const IndexType &vertexIndex0 = extrusionCell->vertices()[0];
    const IndexType &vertexIndex1 = extrusionCell->vertices()[1];
    v0 = _extrusionMeshTopologies[extrusionDimension]->getVertex(vertexIndex0)[0];
    v1 = _extrusionMeshTopologies[extrusionDimension]->getVertex(vertexIndex1)[0];
  }

  //! computes the measure in the specified extrusion dimensions, starting with the first extrusion dimension.  Involves map lookups; consider caching the result if performance is critical.
  double ExtrudedMeshTopology::extrusionMeasureForPhaseSpaceCell(const IndexType &phaseSpaceCellID, int extrusionDimensionToExclude) const
  {
    double result = 1.0;
    const int numExtrusions = _extrusionMeshTopologies.size();
    for (int extrusionOrdinal=0; extrusionOrdinal < numExtrusions; extrusionOrdinal++)
    {
      if (extrusionDimensionToExclude != extrusionOrdinal)
      {
        double vertex0, vertex1;
        this->extrusionVerticesForPhaseSpaceCell(vertex0, vertex1, phaseSpaceCellID, extrusionOrdinal);
        result *= std::fabs(vertex1 - vertex0);
      }
    }
    return result;
  }
  
  int ExtrudedMeshTopology::numExtrusionDimensions() const
  {
    return _extrusionMeshTopologies.size();
  }
  
  const std::set<IndexType> & ExtrudedMeshTopology::phaseSpaceCellIDsForExtrusionCell(const int &extrusionOrdinal, const IndexType &componentCellID) const
  {
    const int componentNumber = extrusionOrdinal + 1; // physical cell is 0; extrusions start at index 1
    auto entry = _componentToPhaseMaps[componentNumber].find(componentCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _componentToPhaseMaps[componentNumber].end(), std::invalid_argument, "component cellID not found");
    return entry->second;
  }

  const std::set<IndexType> & ExtrudedMeshTopology::phaseSpaceCellIDsForPhysicalCell(const IndexType &componentCellID) const
  {
    const int componentNumber = 0; // physical cell
    auto entry = _componentToPhaseMaps[componentNumber].find(componentCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _componentToPhaseMaps[componentNumber].end(), std::invalid_argument, "component cellID not found");
    return entry->second;
  }
  
  MeshTopologyPtr ExtrudedMeshTopology::physicalMeshTopology() const
  {
    return _lowerDimensionalMeshTopology;
  }
  
  IndexType ExtrudedMeshTopology::physicalCellIDForPhaseSpaceCellID(const IndexType &phaseSpaceCellID) const
  {
    auto entry = _phaseToComponentMap.find(phaseSpaceCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _phaseToComponentMap.end(), std::invalid_argument, "phase space cellID not found");
    return entry->second[0]; // first entry corresponds to the physical mesh topology
  }
  
  //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
  Camellia::FieldContainer<double> ExtrudedMeshTopology::physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension) const
  {
    return this->MeshTopology::physicalCellNodesForCell(cellIndex, includeCellDimension);
  }
  
  //! At present, calls base class's implementation and then didHRefine() for any directly-registered RefinementObservers.  In the future, may do something more sophisticated involving tensor structures.
  void ExtrudedMeshTopology::refineCell(IndexType cellIndex, RefinementPatternPtr refPattern, IndexType firstChildCellIndex)
  {
    this->MeshTopology::refineCell(cellIndex, refPattern, firstChildCellIndex);
    
    const bool didRepartitionAndRebuild = false; // *we* didn't; that's not something that MeshTopology does…
    const set<IndexType> cellIndexSet {cellIndex};
    MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
    for (auto refObserver : _refinementObservers)
    {
      refObserver->didHRefine(thisPtr, cellIndexSet, refPattern, didRepartitionAndRebuild);
    }
  }
  
  void ExtrudedMeshTopology::registerRefinementObserver(Teuchos::RCP<RefinementObserver> observer)
  {
    _refinementObservers.push_back(observer);
  }
  
  //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
  void ExtrudedMeshTopology::unrefineCell(IndexType cellIndex, RefinementPatternPtr refPattern)
  {
    this->MeshTopology::unrefineCell(cellIndex, refPattern);
    
    const bool didRepartitionAndRebuild = false; // *we* didn't; that's not something that MeshTopology does…
    const set<IndexType> cellIndexSet {cellIndex};
    MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
    for (auto refObserver : _refinementObservers)
    {
      refObserver->didHUnrefine(thisPtr, cellIndexSet, didRepartitionAndRebuild);
    }
  }
  
  //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
  void ExtrudedMeshTopology::verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const
  {
    this->MeshTopology::verticesForCell(vertices,cellID);
  }

  PhysicalMeshUpdater::PhysicalMeshUpdater(MeshPtr phaseSpaceMesh, MeshPtr physicalMesh)
  {
    _phaseSpaceMesh = phaseSpaceMesh;
    _physicalMesh = physicalMesh;
    
    _extrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*> (phaseSpaceMesh->getTopology().get());
    TEUCHOS_TEST_FOR_EXCEPTION(_extrudedMeshTopology == NULL, std::invalid_argument, "phaseSpaceMesh must use an ExtrudedMeshTopology as its MeshTopology");
  }
  
  void PhysicalMeshUpdater::didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern,
                                       bool didRepartitionAndRebuild)
  {
    std::set<GlobalIndexType> refinedPhysicalCellIDs; // may include some cells that were previously refined.  GDA::didHRefine() should tolerate this.
    
    std::vector<IndexType> componentCellCounts{_extrudedMeshTopology->physicalMeshTopology()->cellCount()};
    for (int d=0; d<_extrudedMeshTopology->numExtrusionDimensions(); d++)
    {
      componentCellCounts.push_back(_extrudedMeshTopology->extrusionMeshTopology(d)->cellCount());
    }
    
    for (const auto & phaseSpaceCellID : cellIDs)
    {
      IndexType physicalCellID = _extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
      refinedPhysicalCellIDs.insert(physicalCellID);
    }
    
    _physicalMesh->globalDofAssignment()->didHRefine(refinedPhysicalCellIDs);
    
    // get mesh to repartition, if phase space mesh was repartitioned.
    if (didRepartitionAndRebuild)
    {
      _physicalMesh->repartitionAndRebuild();
    }
  }
  
  void PhysicalMeshUpdater::registerForUpdates(MeshPtr phaseSpaceMesh, MeshPtr physicalMesh)
  {
    // make weak pointers
    MeshPtr phaseSpaceMeshWeakPtr = Teuchos::rcp(phaseSpaceMesh.get(), false);
    MeshPtr physicalMeshWeakPtr   = Teuchos::rcp(physicalMesh.get(), false);
    Teuchos::RCP<RefinementObserver> refObserver = Teuchos::rcp(new PhysicalMeshUpdater(phaseSpaceMeshWeakPtr, physicalMeshWeakPtr));
    phaseSpaceMesh->registerObserver(refObserver);
  }
} // namespace Camellia
