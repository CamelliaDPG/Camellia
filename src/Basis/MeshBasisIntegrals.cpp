//
//  MeshBasisIntegrals.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 11/24/22.
//

#include <MeshBasisIntegrals.h>

#include <BasisCache.h>

namespace Camellia
{
  void MeshBasisIntegrals::initialize(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, int quadratureEnrichment, const std::set<GlobalIndexType> &cellIDs)
  {
    _basis = basis;
    
    if (cellIDs.size() == 0) return;
    
    _integralCellIDs = std::vector<GlobalIndexType>(cellIDs.begin(),cellIDs.end());
    
    int cellOrdinal = 0;
    for (auto cellID : cellIDs)
    {
      _cellOrdinalForCell[cellID] = cellOrdinal;
      cellOrdinal++;
    }
    
    // compute the integrals.
    
    auto cellTopo = mesh->getTopology()->getCell(_integralCellIDs[0])->topology();
    int quadratureDegree = basis->getDegree() + quadratureEnrichment;
    auto basisCache = BasisCache::basisCacheForReferenceCell(cellTopo, quadratureDegree);
    
    basisCache->setMesh(mesh);
    
    auto physicalCellNodes = mesh->physicalCellNodes(_integralCellIDs);
    basisCache->setPhysicalCellNodes(physicalCellNodes, _integralCellIDs, false); // false: don't create side cache
    
    // note that the following is not entirely device-optimized; we'd do better to avoid using FieldContainer.
    
    auto transformedWeightedBasisValues = basisCache->getTransformedWeightedValues(basis, OP_VALUE); // (C,F,P)
    const int numCells  = cellIDs.size();
    const int numPoints = transformedWeightedBasisValues->dimension(2);
    
    auto fxnValues = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numPoints},"MeshBasisIntegrals: fxnValues");
    weight->values(fxnValues, ASSIGN, basisCache);
    
    auto transformedWeightedBasisView = transformedWeightedBasisValues->checkoutDeviceView();
    
    // integrate:
    int basisCardinality = basis->getCardinality();
    _integrals = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,basisCardinality},"MeshBasisIntegrals: _integrals");
    auto integrals = _integrals; // local copy for lambda (avoid implicit reference to this)
    Kokkos::fence();
    auto rangePolicy = Camellia::FieldContainer<double>::getMDRangePolicy2(_integrals);
    Kokkos::parallel_for("MeshBasisIntegrals: compute _integrals", rangePolicy,
    KOKKOS_LAMBDA (const int cellOrdinal, const int fieldOrdinal) {
      double integral = 0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double &weightedBasisValue = transformedWeightedBasisView(cellOrdinal,fieldOrdinal,pointOrdinal);
        const double &fxnValue           = fxnValues(cellOrdinal,pointOrdinal);
        integral += weightedBasisValue * fxnValue;
      }
      integrals(cellOrdinal,fieldOrdinal) = integral;
    });
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(fxnValues);
    
    transformedWeightedBasisValues->checkinDeviceView(false);
  }

  MeshBasisIntegrals::MeshBasisIntegrals(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, int quadratureEnrichment, const std::set<GlobalIndexType> &cellIDs)
  {
    initialize(mesh, basis, weight, quadratureEnrichment, cellIDs);
  }

  MeshBasisIntegrals::MeshBasisIntegrals(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, bool includeAllCells, int quadratureEnrichment)
:
  _basis(basis)
  {
    std::set<GlobalIndexType> cellIDs;
    if (includeAllCells)
    {
      auto meshTopo = mesh->getTopology()->baseMeshTopology();
      GlobalIndexType nextCellID = meshTopo->nextCellID();
      for (int cellID=0; cellID<nextCellID; cellID++)
      {
        if (meshTopo->isValidCellIndex(cellID))
        {
          cellIDs.insert(cellID);
        }
      }
    }
    else
    {
      cellIDs = mesh->cellIDsInPartition();
    }
    initialize(mesh, basis, weight, quadratureEnrichment, cellIDs);
  }

  void MeshBasisIntegrals::solutionIntegral(DeviceView<double> solutionIntegrals, DeviceView<double> solutionCoefficients, vector<GlobalIndexType> &cellIDs)
  {
    const int numCells     = solutionCoefficients.extent_int(0);
    const int numSolutions = solutionCoefficients.extent_int(1);
    const int numFields    = solutionCoefficients.extent_int(2);
    
    TEUCHOS_TEST_FOR_EXCEPTION(solutionIntegrals.extent_int(0) != numCells,     std::invalid_argument, "solutionIntegrals and solutionCoefficients must match in their first (cell) dimension.");
    TEUCHOS_TEST_FOR_EXCEPTION(solutionIntegrals.extent_int(1) != numSolutions, std::invalid_argument, "solutionIntegrals and solutionCoefficients must match in their second (solution) dimension.");
    TEUCHOS_TEST_FOR_EXCEPTION(_basis->getCardinality() != numFields, std::invalid_argument, "solutionCoefficients must match basis cardinality in its third (field) dimension.")
    
    TEUCHOS_TEST_FOR_EXCEPTION(int(cellIDs.size()) != numCells, std::invalid_argument, "cellIDs must have as many entries as the first (cell) dimension of solutionIntegrals and solutionCoefficients.");
    
    auto integralsCellOrdinalLookup = SharedScratchStorage<GlobalIndexType>::sharedScratchStorage().checkoutScratchStorage({numCells},"MeshBasisIntegrals: integralsCellOrdinalLookup");
    auto integralsCellOrdinalLookupHost = Kokkos::create_mirror_view(integralsCellOrdinalLookup);
    
    int entryOrdinal = 0;
    for (const auto & cellID : cellIDs)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_cellOrdinalForCell.find(cellID) == _cellOrdinalForCell.end(), std::invalid_argument, "cellID not found");
      auto cellOrdinal = _cellOrdinalForCell[cellID];
      
      integralsCellOrdinalLookup(entryOrdinal) = cellOrdinal;
      
      entryOrdinal++;
    }
    
    Kokkos::deep_copy(integralsCellOrdinalLookup,integralsCellOrdinalLookupHost);
    
    auto integrals = _integrals; // local copy for lambda (avoid implicit reference to this)
    auto rangePolicy = Camellia::FieldContainer<double>::getMDRangePolicy2(solutionIntegrals);
    Kokkos::parallel_for("MeshBasisIntegrals: compute solutionIntegrals", rangePolicy,
    KOKKOS_LAMBDA (const int cellOrdinal, const int solutionOrdinal) {
      double solnIntegral = 0;
      GlobalIndexType integralsCellOrdinal = integralsCellOrdinalLookup(cellOrdinal);
      for (int fieldOrdinal=0; fieldOrdinal<numFields; fieldOrdinal++)
      {
        const double &solnCoefficient = solutionCoefficients(cellOrdinal,solutionOrdinal,fieldOrdinal);
        const double &basisIntegral   = integrals(integralsCellOrdinal,fieldOrdinal);
        solnIntegral += solnCoefficient * basisIntegral;
      }
      solutionIntegrals(cellOrdinal,solutionOrdinal) = solnIntegral;
    });
    
    SharedScratchStorage<GlobalIndexType>::sharedScratchStorage().checkinScratchStorage(integralsCellOrdinalLookup);
  }

  MeshBasisIntegrals::~MeshBasisIntegrals()
  {
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_integrals);
  }
}
