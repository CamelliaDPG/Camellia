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

#include "Camellia_TensorIndices.h"
#include "MonomialFunctions.h"
#include "Solution.h"

namespace
{
  // private (anonymous-namespaced) class to define a simple basis integration kernel.  We moved this out of a lambda defined in the SolutionMomentFunction constructor because nvcc complained about not being able to take the address of the enclosing function.
  class BasisIntegral
  {
    const int numPoints_;
    const int momentNumber_;
    
    using DeviceType = ::Camellia::DeviceType;
    
    Intrepid2::TensorPoints<double, DeviceType> refPoints_;
    Intrepid2::TensorData<double,DeviceType> cubatureWeights_;
    Camellia::DeviceView<double> basisValues_;
    Camellia::DeviceView<double> extrusionBasisIntegral_;
  public:
    BasisIntegral(const int &momentNumber,
                  Intrepid2::TensorPoints<double, DeviceType> refPoints,
                  Intrepid2::TensorData<double,DeviceType> cubatureWeights,
                  Camellia::DeviceView<double> basisValues,
                  Camellia::DeviceView<double> extrusionBasisIntegral)
    :
    momentNumber_(momentNumber),
    numPoints_(refPoints.extent_int(0)),
    refPoints_(refPoints),
    cubatureWeights_(cubatureWeights),
    basisValues_(basisValues),
    extrusionBasisIntegral_(extrusionBasisIntegral)
    {
      
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int &basisOrdinal) const
    {
      double integralValue = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints_; pointOrdinal++)
      {
        const double &x = refPoints_(pointOrdinal,0);
        double xWeight = 1.0;
        for (int i=0; i<momentNumber_; i++) // NOTE: this is not enough to define the higher-order moments; higher moments are tensor-valued, etc.
        {
          xWeight *= x;
        }
        const double &quadratureWeight = cubatureWeights_(pointOrdinal);
        const double &basisValue = basisValues_(basisOrdinal,pointOrdinal);
        integralValue += quadratureWeight * xWeight * basisValue;
      }
      extrusionBasisIntegral_(basisOrdinal) = integralValue / 2.0;
    }
  };
}

namespace Camellia
{
  SolutionMomentFunction::SolutionMomentFunction(SolutionMoments &solutionMoments, const int &momentEntryOrdinal)
  :
  _solutionMoments(solutionMoments),
  _momentEntryOrdinal(momentEntryOrdinal)
  {}

  void SolutionMomentFunction::importCellData(std::vector<GlobalIndexType> cellIDs)
  {
    _solutionMoments.importCellData(cellIDs);
  }

  void SolutionMomentFunction::values(Intrepid2::Data<double, Camellia::DeviceType> &values, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SolutionMomentFunction values(Data,BasisCachePtr)");
   
    // this implementation adapted from BasisSumFunction
    
    BasisPtr extrudedSpatialBasis = _solutionMoments.extrudedSpatialBasis();
    
    const int numDofs = extrudedSpatialBasis->getCardinality();
    const int spaceDim = basisCache->cellTopology()->getDimension();

    bool basisIsVolumeBasis = extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
    
    TEUCHOS_TEST_FOR_EXCEPTION(useCubPointsSideRefCell, std::invalid_argument, "Side caches not (yet) supported by SolutionMomentFunction");
    
    Intrepid2::TensorPoints<double,Camellia::DeviceType> points = basisCache->getRefCellPoints();
    Intrepid2::BasisValues<double,Camellia::DeviceType> basisValues = extrudedSpatialBasis->allocateBasisValues(points);
    
    constFCPtr transformedValues = basisCache->getTransformedValues(extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
    
    // transformedValues has dimensions (C,F,P,[D,D])
    auto dv_transformedValues = transformedValues->checkoutDeviceView();
    
    auto allSpatialBasisCoefficients  = _solutionMoments.getSpatialBasisCoefficients();
    
    auto spatialBasisCoefficients  = Kokkos::subview(allSpatialBasisCoefficients, _momentEntryOrdinal, Kokkos::ALL(), Kokkos::ALL());
    auto physicalCellIDs           = _solutionMoments.physicalCellIDs();
    
    int numCellIDs = basisCache->cellIDs().size();
    std::vector<int> physicalCellOrdinalLookupDims {numCellIDs};
    auto physicalCellOrdinalLookup = SharedScratchStorage<IndexType>::sharedScratchStorage().checkoutScratchStorage(physicalCellOrdinalLookupDims, "SolutionMomentFunction physical cell ordinal lookup");
    
    auto cellOrdinalLookupHost = Kokkos::create_mirror_view(physicalCellOrdinalLookup);
    int basisCacheCellOrdinal = 0;
    
    for (const auto &cellID : basisCache->cellIDs())
    {
      auto matchingEntry = std::lower_bound(physicalCellIDs.begin(), physicalCellIDs.end(), cellID);
      if ((matchingEntry == physicalCellIDs.end()) || *matchingEntry != cellID)
      {
        int rank = basisCache->mesh()->Comm()->getRank();
        std::cout << "physical cell " << cellID << " not found in physicalCellIDs container on rank " << rank << std::endl;
      }
      TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == physicalCellIDs.end()) || *matchingEntry != cellID,std::invalid_argument,"cellID not found");
      
      int cellOrdinal = matchingEntry - physicalCellIDs.begin();
      
      cellOrdinalLookupHost(basisCacheCellOrdinal) = cellOrdinal;
      basisCacheCellOrdinal++;
    }
    Kokkos::deep_copy(physicalCellOrdinalLookup, cellOrdinalLookupHost);
    
    using ExecutionSpace = Camellia::DeviceExecutionSpace;

    const int underlyingViewRank = values.getUnderlyingViewRank(); // can be (C,P0,P1) or (C,P)
    if (underlyingViewRank == 2)
    {
      auto view = values.getUnderlyingView<2>();
      Kokkos::deep_copy(view, 0.0); // initialize
      
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{view.extent(0),view.extent(1)});
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
      
      Kokkos::parallel_for("SolutionMomentFunction_values", policy,
      KOKKOS_LAMBDA (const array_index_type &physicalCellOrdinalInBasisCache, const array_index_type &pointOrdinal)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(physicalCellOrdinalInBasisCache);
        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          const double & basisValue = dv_transformedValues(physicalCellOrdinalInBasisCache,dof,pointOrdinal);
          view(physicalCellOrdinalInBasisCache,pointOrdinal) += basisValue * weight;
        }
      });
    }
    else if (underlyingViewRank == 3)
    {
      auto view = values.getUnderlyingView<3>();
      Kokkos::deep_copy(view, 0.0); // initialize
      
      int numPhysicalPoints = view.extent_int(1);
      
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{view.extent(0),view.extent(1),view.extent(2)});
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
      
      TEUCHOS_TEST_FOR_EXCEPTION(basisValues.numTensorDataFamilies() != 1, std::invalid_argument, "basisValues has unexpected number of TensorData families");
      
      Kokkos::parallel_for("SolutionMomentFunction_values", policy,
      KOKKOS_LAMBDA (const array_index_type &physicalCellOrdinalInBasisCache, const array_index_type &physicalPointOrdinal, const array_index_type &extrusionPointOrdinal)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(physicalCellOrdinalInBasisCache);
        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          int pointOrdinal;
          TensorIndices::setTensorOrdinal(pointOrdinal, physicalPointOrdinal, numPhysicalPoints, extrusionPointOrdinal);
          const double & basisValue = dv_transformedValues(physicalCellOrdinalInBasisCache,dof,pointOrdinal);
          view(physicalCellOrdinalInBasisCache,physicalPointOrdinal,extrusionPointOrdinal) += basisValue * weight;
        }
      });
    }
    else if (underlyingViewRank == 4)
    {
      auto view = values.getUnderlyingView<4>();
      Kokkos::deep_copy(view, 0.0); // initialize

      const int numPoint0 = view.extent_int(1);
      const int numPoint1 = view.extent_int(2);
      const int numPoint2 = view.extent_int(3);

      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>>(
        {0,0,0,0},
        {view.extent(0), view.extent(1), view.extent(2), view.extent(3)}
      );
      using array_index_type = Kokkos::MDRangePolicy<Kokkos::Rank<4>>::array_index_type;

      Kokkos::parallel_for("SolutionMomentFunction_values", policy,
      KOKKOS_LAMBDA (const array_index_type &physicalCellOrdinalInBasisCache,
                     const array_index_type &pointOrdinal0,
                     const array_index_type &pointOrdinal1,
                     const array_index_type &pointOrdinal2)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(physicalCellOrdinalInBasisCache);

        int pointOrdinal;
        TensorIndices::setTensorOrdinal(pointOrdinal, pointOrdinal0, numPoint0, pointOrdinal1, numPoint1, pointOrdinal2);

        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          const double & basisValue = dv_transformedValues(physicalCellOrdinalInBasisCache,dof,pointOrdinal);
          view(physicalCellOrdinalInBasisCache,pointOrdinal0,pointOrdinal1,pointOrdinal2) += basisValue * weight;
        }
      });
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported underlyingViewRank");
    }
    transformedValues->checkinDeviceView(false); // false: don't copy back to host (has not changed)
    SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
  }

  void SolutionMomentFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SolutionMomentFunction values()");
   
    BasisPtr extrudedSpatialBasis = _solutionMoments.extrudedSpatialBasis();
    
    const int numDofs = extrudedSpatialBasis->getCardinality();
    const int spaceDim = basisCache->cellTopology()->getDimension();

    bool basisIsVolumeBasis = extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();

    constFCPtr transformedValues = basisCache->getTransformedValues(extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
    int rank = transformedValues->rank() - 3;
    TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
    auto dv_transformedValues = transformedValues->checkoutDeviceView();
    
    auto allSpatialBasisCoefficients  = _solutionMoments.getSpatialBasisCoefficients();
    auto spatialBasisCoefficients  = Kokkos::subview(allSpatialBasisCoefficients, _momentEntryOrdinal, Kokkos::ALL(), Kokkos::ALL());
    auto physicalCellIDs           = _solutionMoments.physicalCellIDs();
    
    int numCellIDs = basisCache->cellIDs().size();
    std::vector<int> physicalCellOrdinalLookupDims {numCellIDs};
    auto physicalCellOrdinalLookup = SharedScratchStorage<IndexType>::sharedScratchStorage().checkoutScratchStorage(physicalCellOrdinalLookupDims, "SolutionMomentFunction physical cell ordinal lookup");
    
    auto cellOrdinalLookupHost = Kokkos::create_mirror_view(physicalCellOrdinalLookup);
    int basisCacheCellOrdinal = 0;
    for (const auto &cellID : basisCache->cellIDs())
    {
      auto matchingEntry = std::lower_bound(physicalCellIDs.begin(), physicalCellIDs.end(), cellID);
      const bool notFound = (matchingEntry == physicalCellIDs.end()) || *matchingEntry != cellID;
      if ( notFound )
      {
        int rank = basisCache->mesh()->Comm()->getRank();
        std::cout << "physical cell " << cellID << " not found in physicalCellIDs container on rank " << rank << std::endl;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(notFound,std::invalid_argument,"cellID not found");
      
      int cellOrdinal = matchingEntry - physicalCellIDs.begin();
      cellOrdinalLookupHost(basisCacheCellOrdinal) = cellOrdinal;
      basisCacheCellOrdinal++;
    }
    Kokkos::deep_copy(physicalCellOrdinalLookup, cellOrdinalLookupHost);
    
    if (modType == ASSIGN)
    {
      Kokkos::deep_copy(values, 0.0);
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
      Kokkos::deep_copy(basisSumValues, 0.0);
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
    transformedValues->checkinDeviceView(false);
    SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
  }

  RelativisticFirstMomentFunction::RelativisticFirstMomentFunction(SolutionMoments* drivingMoments,
                                                                   SolutionPtr phaseSpaceSolution, VarPtr fieldVar,
                                                                   MeshPtr configurationSpaceMesh, double c,
                                                                   int velocityComponent)
  :
  TFunction<double>("RelativisticFirstMomentFunction"),
  _drivingMoments(drivingMoments),
  _phaseSpaceSolution(Teuchos::rcp(phaseSpaceSolution.get(), false)),
  _fieldVar(fieldVar),
  _configurationSpaceMesh(Teuchos::rcp(configurationSpaceMesh.get(), false)),
  _c(c),
  _velocityComponent(velocityComponent)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_drivingMoments == nullptr, std::invalid_argument, "drivingMoments may not be null");
    TEUCHOS_TEST_FOR_EXCEPTION(_phaseSpaceSolution == Teuchos::null, std::invalid_argument, "phase-space solution is null");
    TEUCHOS_TEST_FOR_EXCEPTION(_configurationSpaceMesh == Teuchos::null, std::invalid_argument, "configuration-space mesh is null");

    MeshPtr phaseSpaceMesh = _phaseSpaceSolution->mesh();
    const int configDim = _configurationSpaceMesh->getDimension();
    const int phaseDim = phaseSpaceMesh->getDimension();
    const int velocityDim = phaseDim - configDim;
    TEUCHOS_TEST_FOR_EXCEPTION((_velocityComponent < 0) || (_velocityComponent >= velocityDim), std::invalid_argument,
                               "velocity component out of range");

    const int velocityStartComponent = configDim;
    FunctionPtr u_i = MonomialFunction::xi_n(velocityStartComponent + _velocityComponent, 1);
    FunctionPtr uSquared = Function::zero();
    for (int d=0; d<velocityDim; d++)
    {
      auto u_d = MonomialFunction::xi_n(velocityStartComponent + d, 1);
      uSquared = uSquared + u_d * u_d;
    }
    FunctionPtr gamma = Function::sqrtFunction(1.0 + uSquared / (_c * _c));
    _weightedF = Function::solution(_fieldVar, _phaseSpaceSolution) * (u_i / gamma);

    _weightedSolution = Solution::solution(phaseSpaceMesh);
    _weightedMoments = Teuchos::rcp(new SolutionMoments(0, _weightedSolution, _fieldVar, _configurationSpaceMesh, -1));

    if (velocityDim == 1)      _zeroMomentFunction = _weightedMoments->solutionMomentFunction(0);
    else if (velocityDim == 2) _zeroMomentFunction = _weightedMoments->solutionMomentFunction(0,0);
    else if (velocityDim == 3) _zeroMomentFunction = _weightedMoments->solutionMomentFunction(0,0,0);
    else TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported velocity dimension in RelativisticFirstMomentFunction");

    _drivingMoments->registerUpdateListener(this);
    update();
  }

  void RelativisticFirstMomentFunction::detachFromDrivingMoments()
  {
    _drivingMoments = nullptr;
  }

  RelativisticFirstMomentFunction::~RelativisticFirstMomentFunction()
  {
    if (_drivingMoments != nullptr)
    {
      _drivingMoments->unregisterUpdateListener(this);
    }

    // Explicitly clear the weighted state before global Camellia teardown so that the
    // owned weighted SolutionMoments object is destroyed in time to check its shared
    // scratch storage back in.
    _zeroMomentFunction = Teuchos::null;
    _weightedMoments = Teuchos::null;
    _weightedSolution = Teuchos::null;
    _weightedF = Teuchos::null;
    _drivingMoments = nullptr;
  }

  void RelativisticFirstMomentFunction::update()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_weightedSolution == Teuchos::null, std::invalid_argument, "weighted solution is null");
    TEUCHOS_TEST_FOR_EXCEPTION(_weightedMoments == Teuchos::null, std::invalid_argument, "weighted moments is null");
    TEUCHOS_TEST_FOR_EXCEPTION(_weightedF == Teuchos::null, std::invalid_argument, "weighted function is null");

    std::map<int, FunctionPtr> projectionMap;
    projectionMap[_fieldVar->ID()] = _weightedF;
    _weightedSolution->projectOntoMesh(projectionMap);
    _weightedMoments->update();
    _hasBeenUpdated = true;
  }

  void RelativisticFirstMomentFunction::values(Intrepid2::Data<double, Camellia::DeviceType> &data, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("RelativisticFirstMomentFunction values(Data,BasisCachePtr)");

    const int functionRank = this->rank();
    TEUCHOS_TEST_FOR_EXCEPTION(functionRank != 0, std::invalid_argument,
                               "RelativisticFirstMomentFunction is expected to be scalar-valued");

    const int numCells = basisCache->getPhysicalCubaturePoints().extent_int(0);
    const int numPoints = basisCache->getPhysicalCubaturePoints().extent_int(1);

    std::vector<int> viewDims {numCells, numPoints};
    auto valuesView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(
      viewDims, "RelativisticFirstMomentFunction::values(Data,...) scratch view"
    );

    this->values(valuesView, ASSIGN, basisCache);

    const int numPointComponents = data.rank() - functionRank - 1;
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointBoundsArray;
    for (int d=0; d<numPointComponents; d++)
    {
      pointBoundsArray[d] = data.extent_int(d + 1);
    }
    for (int d=numPointComponents; d<Intrepid2::Parameters::MaxTensorComponents; d++)
    {
      pointBoundsArray[d] = 1;
    }

    auto rangePolicy = Camellia::FieldContainer<double>::getMDRangePolicy2(valuesView);
    Kokkos::parallel_for("RelativisticFirstMomentFunction copy DeviceView to Data", rangePolicy,
    KOKKOS_LAMBDA (const int &cellOrdinal, const int &pointOrdinal)
    {
      Intrepid2::TensorArgumentIterator pi(pointBoundsArray);
      pi.setEnumerationIndex(pointOrdinal);

      const int MAX_ARGS = Intrepid2::Parameters::MaxTensorComponents;
      Kokkos::Array<int,MAX_ARGS> dataArgs;
      for (int d=0; d<MAX_ARGS; d++)
      {
        dataArgs[d] = 0;
      }
      dataArgs[0] = cellOrdinal;
      for (int d=1; d<numPointComponents+1; d++)
      {
        dataArgs[d] = pi.argument(d-1);
      }

      double* data_entry;
      switch (numPointComponents + 1)
      {
        case 1: data_entry = &data.getWritableEntry(dataArgs[0]); break;
        case 2: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1]); break;
        case 3: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2]); break;
        case 4: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3]); break;
        case 5: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4]); break;
        case 6: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4],dataArgs[5]); break;
        case 7: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4],dataArgs[5],dataArgs[6]); break;
        default: INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(true, std::invalid_argument, "unsupported numPointComponents in RelativisticFirstMomentFunction::values(Data,...)");
      }

      *data_entry = valuesView(cellOrdinal,pointOrdinal);
    });

    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(valuesView);
  }

  void RelativisticFirstMomentFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_phaseSpaceSolution == Teuchos::null, std::invalid_argument, "phase-space solution is null");
    TEUCHOS_TEST_FOR_EXCEPTION(_configurationSpaceMesh == Teuchos::null, std::invalid_argument, "configuration-space mesh is null");
    TEUCHOS_TEST_FOR_EXCEPTION(basisCache->mesh().get() != _configurationSpaceMesh.get(), std::invalid_argument,
                               "RelativisticFirstMomentFunction must be evaluated on its configuration-space mesh");
    TEUCHOS_TEST_FOR_EXCEPTION(!_hasBeenUpdated, std::invalid_argument,
                               "RelativisticFirstMomentFunction values() called before the driving SolutionMoments update initialized weighted moments");
    TEUCHOS_TEST_FOR_EXCEPTION(_zeroMomentFunction == Teuchos::null, std::invalid_argument,
                               "zero moment function is null");

    _zeroMomentFunction->values(values, modType, basisCache);
  }

} // namespace Camellia
