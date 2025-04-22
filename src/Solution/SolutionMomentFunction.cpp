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

  void SolutionMomentFunction::values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
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
          const double & basisValue = dv_transformedValues(physicalCellOrdinalInBasisCache,dof,pointOrdinal); //basisValues(dof,pointOrdinal);
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
      
//      auto tensorBasisValues = basisValues.tensorData();
//
//      auto spatialValues   = basisValues.tensorData().getTensorComponent(0);
//      auto extrusionValues = basisValues.tensorData().getTensorComponent(1);
      
      Kokkos::parallel_for("SolutionMomentFunction_values", policy,
      KOKKOS_LAMBDA (const array_index_type &physicalCellOrdinalInBasisCache, const array_index_type &physicalPointOrdinal, const array_index_type &extrusionPointOrdinal)
      {
        const int &cellOrdinal = physicalCellOrdinalLookup(physicalCellOrdinalInBasisCache);
        for (int dof = 0; dof < numDofs; dof++)
        {
          double weight = spatialBasisCoefficients(cellOrdinal,dof);
          int pointOrdinal;
          TensorIndices::setTensorOrdinal(pointOrdinal, physicalPointOrdinal, numPhysicalPoints, extrusionPointOrdinal);
          const double & basisValue = dv_transformedValues(physicalCellOrdinalInBasisCache,dof,pointOrdinal); //basisValues(dof,pointOrdinal);
//          const int fieldIndex = 0;
//          const long long spatialFieldOrdinal   = tensorBasisValues.getTensorComponentIndex(0, fieldIndex, dof);
//          const long long extrusionFieldOrdinal = tensorBasisValues.getTensorComponentIndex(1, fieldIndex, dof);
//          const double & spatialBasisValue   = spatialValues(spatialFieldOrdinal,physicalPointOrdinal);
//          const double & extrusionBasisValue = extrusionValues(extrusionFieldOrdinal,extrusionPointOrdinal);
//          view(physicalCellOrdinalInBasisCache,physicalPointOrdinal,extrusionPointOrdinal) += spatialBasisValue * extrusionBasisValue * weight;
          view(physicalCellOrdinalInBasisCache,physicalPointOrdinal,extrusionPointOrdinal) += basisValue * weight;
        }
      });
    }
    transformedValues->checkinDeviceView(false); // false: don't copy back to host (has not changed)
    SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
  }

  void SolutionMomentFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SolutionMomentFunction values()");
   
    // this implementation adapted from BasisSumFunction
    
    BasisPtr extrudedSpatialBasis = _solutionMoments.extrudedSpatialBasis();
    
    const int numDofs = extrudedSpatialBasis->getCardinality();
    const int spaceDim = basisCache->cellTopology()->getDimension();

    bool basisIsVolumeBasis = extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
    bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();

    constFCPtr transformedValues = basisCache->getTransformedValues(extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
    
    // transformedValues has dimensions (C,F,P,[D,D])
    // therefore, the rank of the sum is transformedValues->rank() - 3
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
    SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
  }

} // namespace Camellia
