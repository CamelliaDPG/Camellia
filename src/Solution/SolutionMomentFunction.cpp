// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  SolutionMomentFunction.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/15/21.
//

#include "SolutionMomentFunction.h"

namespace Camellia
{
  SolutionMomentFunction::SolutionMomentFunction(SolutionPtr phaseSpaceSoln, VarPtr fieldVar, int momentNumber, MeshPtr physicalMesh, int survivingExtrusionDimension)
  :
  TFunction<double>("SolutionMomentFunction", fieldVar->rank()),
  _survivingExtrusionDimension(survivingExtrusionDimension),
  _basisReconciliation(true) // true: cache results
  {
    TEUCHOS_TEST_FOR_EXCEPTION(momentNumber != 0, std::invalid_argument, "moments beyond the 0 moment not yet (correctly) implemented");
    
    _soln = phaseSpaceSoln;
    _fieldVar = fieldVar;
    _possiblyExtrudedPhysicalMesh = physicalMesh;
    
    // check that physical mesh has the right number of dimensions:
    const int numDimensionsPhysicalMesh = physicalMesh->getDimension();
    const int numDimensionsPhaseSpace   = phaseSpaceSoln->mesh()->getDimension();
    
    int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension == -1) ? 0 : 1;
    _numExtrusionDimensions = numDimensionsPhaseSpace - numDimensionsPhysicalMesh + numSurvivingExtrusionDimensions;

    // we assume p is constant throughout the domain
    // we also assume that all elements have the same topology
    // we also assume that dimensions in which integration is performed are orthogonal extrusions
    
    const IndexType sampleCellID = 0;
    auto elemType = _soln->mesh()->getElementType(sampleCellID);
    auto cellTopo = elemType->cellTopoPtr;
    
    TEUCHOS_TEST_FOR_EXCEPTION(cellTopo->getTensorialDegree() != _numExtrusionDimensions, std::invalid_argument, "CellTopology and _numExtrusionDimensions differ on the number of tensorial extrusions");
    
    auto trialOrdering = elemType->trialOrderPtr;
    _phaseSpaceBasis = trialOrdering->getBasis(fieldVar->ID());
    
    // TODO: when we move to using Intrepid2 TensorBasis, update this accordingly
    
    // peel off the extrusion bases:
    _extrusionBases = std::vector<BasisPtr>(_numExtrusionDimensions);
    auto spatialBasis = _phaseSpaceBasis;
    for (int extrusionOrdinal=_numExtrusionDimensions-1; extrusionOrdinal >= 0; extrusionOrdinal--)
    {
      using TensorBasis = Camellia::TensorBasis<double, Intrepid::FieldContainer<double> >;
      TensorBasis* tensorBasis = dynamic_cast<TensorBasis*>(spatialBasis.get());
      
      TEUCHOS_TEST_FOR_EXCEPTION(tensorBasis == NULL, std::invalid_argument, "fieldVar must be discretized with a TensorBasis");
      
      _extrusionBases[extrusionOrdinal] = tensorBasis->getTemporalBasis();
      _componentBasisCardinalities[extrusionOrdinal+1] = _extrusionBases[extrusionOrdinal]->getCardinality();
      
      spatialBasis = tensorBasis->getSpatialBasis();
    }
    _spatialBasis = spatialBasis;
    _componentBasisCardinalities[0] = _spatialBasis->getCardinality();
    
    _extrudedSpatialBasis = _spatialBasis;
    if (_survivingExtrusionDimension != -1)
    {
      const int componentCardinality = _extrusionBases[_survivingExtrusionDimension]->getCardinality();
      
      _extrudedSpatialBasis = Teuchos::rcp(new TensorBasis<double>(_extrudedSpatialBasis,_extrusionBases[_survivingExtrusionDimension]));
    }
    
    for (int extrusionOrdinal=0; extrusionOrdinal<_numExtrusionDimensions; extrusionOrdinal++)
    {
      const auto &extrusionBasis = _extrusionBases[extrusionOrdinal];
      
      int basisCardinality = extrusionBasis->getCardinality();
      int basisDegree = extrusionBasis->getDegree();
      
      vector<int> basisCardinalityVector { basisCardinality };
      auto extrusionBasisIntegral = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage( basisCardinalityVector, "SolutionMomentFunction extrusionBasisIntegral" );
      
      vector<int> quadratureDegree {basisDegree + momentNumber};
      
      CubatureFactory cubFactory;
      auto cubature = cubFactory.create(CellTopology::line(), quadratureDegree);
      auto refPoints       = cubature->allocateCubaturePoints();
      auto cubatureWeights = cubature->allocateCubatureWeights();
      cubature->getCubature(refPoints, cubatureWeights);
      
      const int numPoints = refPoints.extent(0);
      
      // TODO: when we switch to using Intrepid2 basis, use the refPoints directly (not refPointsFC)
      auto refPointsFC = FieldContainerConversion<double>::fromTensorPoints(refPoints);
      
      FieldContainer<double> valuesFC(basisCardinality, numPoints);
      extrusionBasis->getValues(valuesFC, refPointsFC, Intrepid::OPERATOR_VALUE);
      
      auto values = valuesFC.checkoutDeviceView(true); // true: deep copy from host
      
      Kokkos::parallel_for("integrate", basisCardinality,
      KOKKOS_LAMBDA (const int basisOrdinal) {
        double integralValue = 0.0;
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double &x = refPoints(pointOrdinal,0);
          double xWeight = 1.0;
          for (int i=0; i<momentNumber; i++) // NOTE: this is not enough to define the higher-order moments; higher moments are tensor-valued, etc.
          {
            xWeight *= x;
          }
          const double &quadratureWeight = cubatureWeights(pointOrdinal);
          const double &basisValue = values(basisOrdinal,pointOrdinal);
          integralValue += quadratureWeight * xWeight * basisValue;
        }
        extrusionBasisIntegral(basisOrdinal) = integralValue / 2.0;
      });
      valuesFC.checkinDeviceView(false); // false: don't deep copy to host
      
      _extrusionBasisIntegrals[extrusionOrdinal] = extrusionBasisIntegral;
    }
  }

  const Camellia::DeviceView<double> SolutionMomentFunction::basisCoefficients() const
  {
    return _spatialBasisCoefficients;
  }

  int SolutionMomentFunction::localCellOrdinal(const IndexType &cellID) const
  {
    auto matchingEntry = std::lower_bound(_physicalCellIDs.begin(), _physicalCellIDs.end(), cellID);
    if (matchingEntry == _physicalCellIDs.end())
    {
      // cellID not found
      return -1;
    }
    else
    {
      const int cellOrdinal = matchingEntry - _physicalCellIDs.begin();
      return cellOrdinal;
    }
  }
  
  MeshPtr SolutionMomentFunction::mesh()
  {
    return _possiblyExtrudedPhysicalMesh;
  }

  //! recomputes coefficients on (extruded) spatial mesh.  MPI collective method.
  void SolutionMomentFunction::updateValues()
  {
    MeshPtr phaseSpaceMesh = _soln->mesh();
    const auto & myPhaseSpaceCells = phaseSpaceMesh->cellIDsInPartition();
    const auto & myPhysicalCells = _possiblyExtrudedPhysicalMesh->cellIDsInPartition();
    
    const int numExtrusionDimensions = _extrusionBasisIntegrals.size();
    
    const int numCells = myPhysicalCells.size();
    
    const int numCellsPrevious  = _spatialBasisCoefficients.extent_int(0);
    
    // the "extruded spatial" basis is the spatial basis, times any extrusion basis not integrated away
    int extrudedSpatialBasisCardinality = _extrudedSpatialBasis->getCardinality();
    int compOrdinal = 0;
    // copy containers to avoid implicit references to this in lambdas below:
    auto componentBasisCardinalities    = _componentBasisCardinalities;
    int survivingExtrusionDimension = _survivingExtrusionDimension;
    const int numExtrudedSpatialBasisComponents = compOrdinal;
    
    const int phaseSpaceBasisCardinality = _phaseSpaceBasis->getCardinality();
    const int numPhaseBasisFunctionsPerSpatialBasisFunction = phaseSpaceBasisCardinality / extrudedSpatialBasisCardinality;
    
    // make sure _spatialBasisCoefficients is properly allocated
    if (numCells != numCellsPrevious)
    {
      // reallocate _spatialBasisCoefficients, _physicalCellOrdinalLookup containers
      std::vector<int> spatialBasisCoefficientsDims {numCells, extrudedSpatialBasisCardinality};
      std::vector<int> physicalCellOrdinalLookupDims {numCells};
      
      // check in previous, if any
      if (_spatialBasisCoefficientsHasBeenCheckedOut)
      {
        SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
      }
      if (_physicalCellOrdinalLookupHasBeenCheckedOut)
      {
        SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(_physicalCellOrdinalLookup);
      }
      // check out current
      _spatialBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(spatialBasisCoefficientsDims, "SolutionMomentFunction spatial basis coefficients");
      _spatialBasisCoefficientsHasBeenCheckedOut = true;
      _physicalCellOrdinalLookup = SharedScratchStorage<IndexType>::sharedScratchStorage().checkoutScratchStorage(physicalCellOrdinalLookupDims, "SolutionMomentFunction physical cell ordinal lookup");
      _physicalCellOrdinalLookupHasBeenCheckedOut = true;
    }
    
    // clear values before accumulating
    Kokkos::deep_copy(_spatialBasisCoefficients, 0.0);
    
    auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(_soln->mesh()->getTopology()->baseMeshTopology());
    TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopology == NULL, std::invalid_argument, "phase space MeshTopology must be an ExtrudedMeshTopology");
    
    Camellia::FieldContainer<double> phaseSpaceBasisCoefficientsFC(phaseSpaceBasisCardinality);
    
    Camellia::FieldContainer<double> coarsePhysicalCellContributionsFC(extrudedSpatialBasisCardinality); // just for the current phase space cellID
    
    std::map<IndexType, FieldContainer<double> > activePhysicalCellContributions;
    
    auto coarsePhysicalCellContributions = coarsePhysicalCellContributionsFC.checkoutDeviceView(false); // false: don't deep copy from host
    
    const int physicalCellCount = phaseSpaceMeshTopology->physicalMeshTopology()->cellCount();
    auto possiblyExtrudedPhysicalMeshTopology = _possiblyExtrudedPhysicalMesh->getTopology();
    for (const auto &phaseSpaceCellID : myPhaseSpaceCells)
    {
      const double extrusionMeasure = phaseSpaceMeshTopology->extrusionMeasureForPhaseSpaceCell(phaseSpaceCellID, _survivingExtrusionDimension);
      _soln->getBasisCoefficientsForCell(phaseSpaceBasisCoefficientsFC, _fieldVar->ID(), phaseSpaceCellID, STANDARD_SOLUTION, 0);
      auto phaseSpaceBasisCoefficients = phaseSpaceBasisCoefficientsFC.checkoutDeviceView(true); // true: deep copy from host
      
      int numExtrusionDimensions = _numExtrusionDimensions;
      auto extrusionBasisIntegralsArray = _extrusionBasisIntegrals; // avoiding implicit reference to this in lambda below
      Kokkos::parallel_for("integrate over phase space", extrudedSpatialBasisCardinality,
      KOKKOS_LAMBDA (const int extrudedSpatialBasisOrdinal) {
        Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> phaseFieldOrdinalComponents;
        int remainingExtrudedSpatialBasisOrdinal = extrudedSpatialBasisOrdinal;
        extractNextComponentOrdinal(phaseFieldOrdinalComponents[0], remainingExtrudedSpatialBasisOrdinal, componentBasisCardinalities[0]);
        if (survivingExtrusionDimension != -1)
        {
          extractNextComponentOrdinal(phaseFieldOrdinalComponents[survivingExtrusionDimension+1],
                                      remainingExtrudedSpatialBasisOrdinal,
                                      componentBasisCardinalities[survivingExtrusionDimension+1]);
        }
        double extrusionIntegralValue = 0.0;
        for (int integratedExtrusionFieldOrdinal = 0; integratedExtrusionFieldOrdinal < numPhaseBasisFunctionsPerSpatialBasisFunction; integratedExtrusionFieldOrdinal++)
        {
          int remainingIntegratedExtrusionFieldOrdinal = integratedExtrusionFieldOrdinal;
          for (int d=0; d<numExtrusionDimensions; d++)
          {
            if (_survivingExtrusionDimension != d)
            {
              extractNextComponentOrdinal(phaseFieldOrdinalComponents[d+1], remainingIntegratedExtrusionFieldOrdinal, componentBasisCardinalities[d+1]);
            }
          }
          
          int phaseFieldOrdinal;
          setTensorOrdinal(phaseFieldOrdinal, phaseFieldOrdinalComponents, componentBasisCardinalities, numExtrusionDimensions+1);
          
          const double &phaseFieldCoefficient = phaseSpaceBasisCoefficients(phaseFieldOrdinal);
          double extrusionComponentProduct = phaseFieldCoefficient;
          for (int d=0; d<numExtrusionDimensions; d++)
          {
            if (_survivingExtrusionDimension == d) continue; // this dimension remains un-integrated; skip
            const auto & extrusionBasisIntegrals = extrusionBasisIntegralsArray[d];
            const int extrusionBasisCardinality = extrusionBasisIntegrals.extent_int(0);
            const int &extrusionComponentFieldOrdinal = phaseFieldOrdinalComponents[d+1];
            extrusionComponentProduct *= extrusionBasisIntegrals(extrusionComponentFieldOrdinal);
          }
          extrusionIntegralValue += extrusionComponentProduct;
        }
        extrusionIntegralValue *= extrusionMeasure;
        
        coarsePhysicalCellContributions(extrudedSpatialBasisOrdinal) = extrusionIntegralValue;
      });
              
      phaseSpaceBasisCoefficientsFC.checkinDeviceView(false); // false: don't deep copy back to host
      coarsePhysicalCellContributionsFC.checkinDeviceView(true);
      
      IndexType possiblyExtrudedPhysicalCellID; // extruded, if there is an extrusion dimension in which we do not integrate
      
      {
        const IndexType physicalCellID = phaseSpaceMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
        if (survivingExtrusionDimension == -1)
        {
          possiblyExtrudedPhysicalCellID = physicalCellID;
        }
        else
        {
          IndexType extrusionCellID = phaseSpaceMeshTopology->extrusionCellIDForPhaseSpaceCellID(phaseSpaceCellID, survivingExtrusionDimension);
          possiblyExtrudedPhysicalCellID = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellID, extrusionCellID);
        }
      }
      
      CellPtr physicalCell = possiblyExtrudedPhysicalMeshTopology->getCell(possiblyExtrudedPhysicalCellID);
      
      // physical cell matches the geometry of phase space cell in physical dimension(s), but it may not be active in the physical mesh;
      // it may have been refined.
      const auto physicalCellActiveDescendants = physicalCell->getDescendants(possiblyExtrudedPhysicalMeshTopology, true); // true: leaf nodes only
      
      for (const auto & activePhysicalCellID : physicalCellActiveDescendants)
      {
        auto activePhysicalCellContributionEntry = activePhysicalCellContributions.find(activePhysicalCellID);
        if (activePhysicalCellContributionEntry == activePhysicalCellContributions.end())
        {
          // allocate (will be zero-filled)
          activePhysicalCellContributions[activePhysicalCellID] = FieldContainer<double>(extrudedSpatialBasisCardinality);
        }
        auto &activePhysicalCellContributionFC = activePhysicalCellContributions[activePhysicalCellID];
        CellPtr activePhysicalCell = possiblyExtrudedPhysicalMeshTopology->getCell(activePhysicalCellID);
        
        SubBasisReconciliationWeights trueSpatialBasisWeights; // weights to relate the phase space basis's physical component to the possibly refined physical cell's basis
        SubBasisReconciliationWeights extrusionBasisWeights;   // weights for the surviving extrusion basis, if any
        if (survivingExtrusionDimension == -1)
        {
          // physical mesh really is just a physical mesh
          RefinementBranch refBranch = activePhysicalCell->refinementBranch(possiblyExtrudedPhysicalCellID);
          trueSpatialBasisWeights    = _basisReconciliation.constrainedWeights(_extrudedSpatialBasis, refBranch, _extrudedSpatialBasis, 0);
          
          if (!trueSpatialBasisWeights.isIdentity)
          {
//              cout << "trueSpatialBasisWeights.weights:\n" << trueSpatialBasisWeights.weights;
            
            const auto & fineOrdinals   = trueSpatialBasisWeights.fineOrdinals;
            const auto & coarseOrdinals = trueSpatialBasisWeights.coarseOrdinals;
            auto fineIt = fineOrdinals.begin();
            for (int fineOrdinalOrdinal = 0; fineOrdinalOrdinal < fineOrdinals.size(); fineOrdinalOrdinal++)
            {
              auto coarseIt = coarseOrdinals.begin();
              for (int coarseOrdinalOrdinal = 0; coarseOrdinalOrdinal < coarseOrdinals.size(); coarseOrdinalOrdinal++)
              {
                const double & weight = trueSpatialBasisWeights.weights(fineOrdinalOrdinal,coarseOrdinalOrdinal);
                activePhysicalCellContributionFC(*fineIt) += weight * coarsePhysicalCellContributionsFC(*coarseIt);
                coarseIt++;
              }
              fineIt++;
            }
          }
          else
          {
            for (int basisOrdinal = 0; basisOrdinal < extrudedSpatialBasisCardinality; basisOrdinal++)
            {
              activePhysicalCellContributionFC(basisOrdinal) += coarsePhysicalCellContributionsFC(basisOrdinal);
            }
          }
        }
        else
        {
          // "physical" mesh is a SubExtrudedMeshTopology
          IndexType truePhysicalCellID, extrusionCellID;
          SubExtrudedMeshTopology::componentCellIndices(truePhysicalCellID, extrusionCellID, activePhysicalCellID);
          
          IndexType truePhysicalAncestor, extrusionAncestor;
          SubExtrudedMeshTopology::componentCellIndices(truePhysicalAncestor, extrusionAncestor, possiblyExtrudedPhysicalCellID);
          
          CellPtr trueActivePhysicalCell = phaseSpaceMeshTopology->physicalMeshTopology()->getCell(truePhysicalCellID);
          RefinementBranch physicalRefBranch = trueActivePhysicalCell->refinementBranch(truePhysicalAncestor);
          
          CellPtr activeExtrusionCell = phaseSpaceMeshTopology->extrusionMeshTopology(survivingExtrusionDimension)->getCell(extrusionCellID);
          RefinementBranch extrusionRefBranch = activeExtrusionCell->refinementBranch(extrusionAncestor);
          
          trueSpatialBasisWeights = _basisReconciliation.constrainedWeights(_spatialBasis, physicalRefBranch, _spatialBasis, 0);
          extrusionBasisWeights   = _basisReconciliation.constrainedWeights(_extrusionBases[survivingExtrusionDimension], extrusionRefBranch, _extrusionBases[survivingExtrusionDimension], 0);

          const int spatialCardinality   = _spatialBasis->getCardinality();
          const int extrusionCardinality = _extrusionBases[survivingExtrusionDimension]->getCardinality();
          
          if (!trueSpatialBasisWeights.isIdentity)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(trueSpatialBasisWeights.fineOrdinals.size()   != spatialCardinality, std::logic_error, "weight count should match basis cardinality");
            TEUCHOS_TEST_FOR_EXCEPTION(trueSpatialBasisWeights.coarseOrdinals.size() != spatialCardinality, std::logic_error, "weight count should match basis cardinality");
          }
          if (!extrusionBasisWeights.isIdentity)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(extrusionBasisWeights.fineOrdinals.size()   != extrusionCardinality, std::logic_error, "weight count should match basis cardinality");
            TEUCHOS_TEST_FOR_EXCEPTION(extrusionBasisWeights.coarseOrdinals.size() != extrusionCardinality, std::logic_error, "weight count should match basis cardinality");
          }
          
          for (int spatialFineOrdinal=0; spatialFineOrdinal<spatialCardinality; spatialFineOrdinal++)
          {
            for (int spatialCoarseOrdinal=0; spatialCoarseOrdinal<spatialCardinality; spatialCoarseOrdinal++)
            {
              double spatialWeight;
              if (trueSpatialBasisWeights.isIdentity)
              {
                spatialWeight = (spatialFineOrdinal == spatialCoarseOrdinal) ? 1.0 : 0.0;
              }
              else
              {
                spatialWeight = trueSpatialBasisWeights.weights(spatialFineOrdinal,spatialCoarseOrdinal);
              }
              
              for (int extrusionFineOrdinal=0; extrusionFineOrdinal<extrusionCardinality; extrusionFineOrdinal++)
              {
                int fineTensorOrdinal;
                setTensorOrdinal(fineTensorOrdinal, spatialFineOrdinal, spatialCardinality, extrusionFineOrdinal);
                for (int extrusionCoarseOrdinal=0; extrusionCoarseOrdinal<extrusionCardinality; extrusionCoarseOrdinal++)
                {
                  int coarseTensorOrdinal;
                  setTensorOrdinal(coarseTensorOrdinal, spatialCoarseOrdinal, spatialCardinality, extrusionCoarseOrdinal);
                  
                  double extrusionWeight;
                  if (extrusionBasisWeights.isIdentity)
                  {
                    extrusionWeight = (extrusionFineOrdinal == extrusionCoarseOrdinal) ? 1.0 : 0.0;
                  }
                  else
                  {
                    extrusionWeight = extrusionBasisWeights.weights(extrusionFineOrdinal,extrusionCoarseOrdinal);
                  }
                  activePhysicalCellContributionFC(fineTensorOrdinal) += spatialWeight * extrusionWeight * coarsePhysicalCellContributionsFC(coarseTensorOrdinal);
                }
              }
            }
          }
        }
        activePhysicalCellContributionFC.checkinDeviceView(true); // true: deep copy back to host
      }
      
      phaseSpaceBasisCoefficientsFC.checkinDeviceView(false); // false: don't deep copy back to host
    }
    
    // send the physical cell contributions to their owners, and store in _spatialBasisCoefficients container
    // we use a convenience method in MPIWrapper for this; note that this sends a bunch of extra data that a manual data-packing implementation would not need to.  (The convenience method requires that we have a vector of fixed-length entries; that's where the inefficiency comes from.)
    using DataType = std::tuple<IndexType,int,double>; // (possiblyExtrudedPhysicalCellID,spatialBasisOrdinal,integralContribution)
    std::map<int,std::vector<DataType>> dataToSend;
    
    for (const auto & entry : activePhysicalCellContributions)
    {
      const IndexType & physicalCellID = entry.first;
      const FieldContainer<double> &physicalCellContribution = entry.second;
      
      const PartitionIndexType physicalCellOwner = _possiblyExtrudedPhysicalMesh->partitionForCellID(physicalCellID);
      TEUCHOS_TEST_FOR_EXCEPTION(physicalCellOwner == -1, std::logic_error, "partition not assigned for physicalCellID -- inactive?");
      
      std::vector<DataType> & ownerDataVector = dataToSend[physicalCellOwner];
      
      for (int basisOrdinal=0; basisOrdinal<physicalCellContribution.dimension(0); basisOrdinal++)
      {
        const double & value = physicalCellContribution(basisOrdinal);
        DataType dataEntry = make_tuple(physicalCellID,basisOrdinal,value);
        ownerDataVector.push_back(dataEntry);
      }
    }
    
    std::vector<DataType> dataReceived;
    MPIWrapper::sendDataVectors(_possiblyExtrudedPhysicalMesh->Comm(), dataToSend, dataReceived);
    
    auto spatialBasisCoefficientsHost = Kokkos::create_mirror_view(_spatialBasisCoefficients);
    Kokkos::deep_copy(spatialBasisCoefficientsHost,0.0);
    // fill in momentsHost, and deep_copy to _moments
    // note that the following map logic is not maximally efficient, but it's unlikely to be a bottleneck due to the relatively small size of the data
    using std::map;
    using std::pair;
    map<IndexType, FieldContainer<double> > contributionsToMyPhysicalCells;
    for (const DataType &dataEntry : dataReceived)
    {
      using std::get;
      const IndexType & physicalCellID = get<0>(dataEntry);
      const int &spatialBasisOrdinal   = get<1>(dataEntry);
      const double &value              = get<2>(dataEntry);
      if (contributionsToMyPhysicalCells.find(physicalCellID) == contributionsToMyPhysicalCells.end())
      {
        contributionsToMyPhysicalCells[physicalCellID] = FieldContainer<double>(extrudedSpatialBasisCardinality);
      }
      FieldContainer<double> &physicalCellIntegrals = contributionsToMyPhysicalCells[physicalCellID];
      
      physicalCellIntegrals(spatialBasisOrdinal) += value;
    }
    // confirm that we have the expected number of entries
    if (contributionsToMyPhysicalCells.size() != numCells)
    {
      Camellia::print("locally owned physical cells", myPhysicalCells);
      std::cout << "contributionsToMyPhysicalCells.size() = " << contributionsToMyPhysicalCells.size() << std::endl;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(contributionsToMyPhysicalCells.size() != numCells, std::invalid_argument, "Received cell count differs from number of locally owned physical cells");
    
    _physicalCellIDs = std::vector<IndexType>(myPhysicalCells.begin(),myPhysicalCells.end());
    int cellOrdinal = 0;
    for (const auto &cellEntry : contributionsToMyPhysicalCells)
    {
      const IndexType &cellID = cellEntry.first;
      const FieldContainer<double> & cellIntegrals = cellEntry.second;
      TEUCHOS_TEST_FOR_EXCEPTION(_physicalCellIDs[cellOrdinal] != cellID, std::invalid_argument, "mismatch in physical cell ID");
      for (int spatialBasisOrdinal=0; spatialBasisOrdinal < extrudedSpatialBasisCardinality; spatialBasisOrdinal++)
      {
        spatialBasisCoefficientsHost(cellOrdinal,spatialBasisOrdinal) += cellIntegrals(spatialBasisOrdinal);
      }
      cellOrdinal++;
    }
    Kokkos::deep_copy(_spatialBasisCoefficients,spatialBasisCoefficientsHost);
  }
  
  void SolutionMomentFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SolutionMomentFunction values()");
   
    // this implementation adapted from BasisSumFunction
    
    const int numDofs = _extrudedSpatialBasis->getCardinality();
    const int spaceDim = basisCache->cellTopology()->getDimension();

    bool basisIsVolumeBasis = _extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();

    constFCPtr transformedValues = basisCache->getTransformedValues(_extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
    
    // transformedValues has dimensions (C,F,P,[D,D])
    // therefore, the rank of the sum is transformedValues->rank() - 3
    int rank = transformedValues->rank() - 3;
    TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
    auto dv_transformedValues = transformedValues->checkoutDeviceView();
    
    // local copies to avoid implicit references to this
    auto spatialBasisCoefficients  = _spatialBasisCoefficients;
    auto physicalCellOrdinalLookup = _physicalCellOrdinalLookup;
    
    auto cellOrdinalLookupHost = Kokkos::create_mirror_view(_physicalCellOrdinalLookup);
    int basisCacheCellOrdinal = 0;
    for (const auto &cellID : basisCache->cellIDs())
    {
      auto matchingEntry = std::lower_bound(_physicalCellIDs.begin(), _physicalCellIDs.end(), cellID);
      TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == _physicalCellIDs.end()) || *matchingEntry != cellID,std::invalid_argument,"cellID not found");
      
      int cellOrdinal = matchingEntry - _physicalCellIDs.begin();
      
      cellOrdinalLookupHost(basisCacheCellOrdinal) = cellOrdinal;
      basisCacheCellOrdinal++;
    }
    Kokkos::deep_copy(_physicalCellOrdinalLookup, cellOrdinalLookupHost);
    
    if (modType == ASSIGN)
    {
      Kokkos::deep_copy(values, 0.0); // initialize
      // Note: values.rank() <= 6, because values.rank < transformedValues.rank
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
      Kokkos::parallel_for("SolutionMomentFunction_values", FieldContainer<double>::getMDRangePolicy6(values),
      KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                     const array_index_type i3, const array_index_type i4, const array_index_type i5)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(i0);
        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          values.access(i0,i1,i2,i3,i4,i5,0) += dv_transformedValues.access(i0,dof,i1,i2,i3,i4,i5) * weight;
        }
      });
    }
    else
    {
      auto basisSumValues = FieldContainer<double>::getDeviceView(values);
      Kokkos::deep_copy(basisSumValues, 0.0); // initialize
      // Note: values.rank() <= 6, because values.rank < transformedValues.rank
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
      Kokkos::parallel_for("SolutionMomentFunction_values", FieldContainer<double>::getMDRangePolicy6(basisSumValues),
      KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                     const array_index_type i3, const array_index_type i4, const array_index_type i5)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(i0);
        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          basisSumValues.access(i0,i1,i2,i3,i4,i5,0) += dv_transformedValues.access(i0,dof,i1,i2,i3,i4,i5) * weight;
        }
      });
      Kokkos::fence();
      Kokkos::parallel_for("SolutionMomentFunction_assign_values", FieldContainer<double>::getMDRangePolicy6(values),
      KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                     const array_index_type i3, const array_index_type i4, const array_index_type i5)
      {
        double & entry =         values.access(i0,i1,i2,i3,i4,i5,0);
        double & value = basisSumValues.access(i0,i1,i2,i3,i4,i5,0);
        assignValue(entry, modType, value);
      });
      Allocation<double>::recordDeviceViewDeallocation(basisSumValues);
    }
    transformedValues->checkinDeviceView(false); // false: don't copy back to host (has not changed)
  }
  
  SolutionMomentFunction::~SolutionMomentFunction()
  {
    for (int d=0; d<_numExtrusionDimensions; d++)
    {
      auto extrusionBasisIntegral = _extrusionBasisIntegrals[d];
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(extrusionBasisIntegral);
    }
    if (_momentsHasBeenCheckedOut)
    {
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_moments);
    }
    if (_spatialBasisCoefficientsHasBeenCheckedOut)
    {
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
    }
    if (_physicalCellOrdinalLookupHasBeenCheckedOut)
    {
      SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(_physicalCellOrdinalLookup);
    }
  }

} // namespace Camellia
