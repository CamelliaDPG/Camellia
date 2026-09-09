#ifndef Camellia_FieldContainer_hpp
#define Camellia_FieldContainer_hpp

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  Camellia_FieldContainer.hpp
 *
 *  Created by Nathan Roberts on 10/3/19.
 *
 */

#include "Intrepid2_Basis.hpp"
#include "Shards_Array.hpp"

#include "Kokkos_DualView.hpp"
#include "Kokkos_DynRankView.hpp"

#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "Camellia_KokkosAllocations.h"
#include "SharedScratchStorage.h"
#include "TypeDefs.h"

namespace Camellia
{
  extern const std::string DEFAULT_VIEW_NAME;

  template<class Scalar, int ArrayTypeId=0>
  class FieldContainer
  {
  public:
    using scalar_type       = Scalar;
    using DeviceMemorySpace = Camellia::DeviceMemorySpace;
    using HostMemorySpace   = Camellia::HostMemorySpace;
    using HostView          = Camellia::HostView<Scalar>;
    using HostViewConst     = Camellia::HostViewConst<Scalar>;
    using ViewTraits        = Camellia::ViewTraits<Scalar>;
    using DeviceView        = Camellia::DeviceView<Scalar>;
    using DeviceViewConst   = Camellia::DeviceViewConst<Scalar>;

#ifndef KOKKOS_ENABLE_HIP
    using DualView = Kokkos::DualView<Scalar*******, typename DeviceView::array_layout, Camellia::DeviceType>;
#else
    using DualView = Kokkos::DualView<Scalar*******, Kokkos::LayoutLeft, Camellia::DeviceType>;
#endif

  private:
    Teuchos::Array<int> _dimensions;
    int _dim0 = 0;
    int _dim1 = 0;
    int _dim2 = 0;
    int _dim3 = 0;
    int _dim4 = 0;
    int _size = 0;

    mutable DualView  _dualView;
    mutable DeviceView _deviceView;
    mutable HostView   _hostView;

    mutable bool _deviceViewIsCheckedOut = false;
    mutable bool _hostModified = false;
    mutable bool _deviceModified = false;
    mutable bool _fieldContainerDataAllocationRecorded = false;
    mutable bool _deviceAllocationRecorded = false;
    mutable int  _fieldContainerAllocationNumber = -1;
    mutable Scalar* _externalData = nullptr;

    static int product(const Teuchos::Array<int> &dimensions)
    {
      if (dimensions.size() == 0) return 0;
      int result = 1;
      for (int d=0; d<dimensions.size(); d++) result *= dimensions[d];
      return result;
    }

    void cacheDims()
    {
      _dim0 = (_dimensions.size() > 0) ? _dimensions[0] : 0;
      _dim1 = (_dimensions.size() > 1) ? _dimensions[1] : 0;
      _dim2 = (_dimensions.size() > 2) ? _dimensions[2] : 0;
      _dim3 = (_dimensions.size() > 3) ? _dimensions[3] : 0;
      _dim4 = (_dimensions.size() > 4) ? _dimensions[4] : 0;
      _size = product(_dimensions);
    }

    static size_t extentFor(const Teuchos::Array<int> &dimensions, const int d)
    {
      if (dimensions.size() == 0) return (d == 0) ? 0 : 1;
      return (d < dimensions.size()) ? static_cast<size_t>(dimensions[d]) : static_cast<size_t>(1);
    }

    static size_t extentOrInvalidFor(const Teuchos::Array<int> &dimensions, const int d)
    {
      return (d < dimensions.size()) ? static_cast<size_t>(dimensions[d]) : static_cast<size_t>(KOKKOS_INVALID_INDEX);
    }

    void refreshViews() const
    {
      const int rank = this->rank();
      if (_externalData != nullptr)
      {
        _hostView = HostView(_externalData,
                             extentOrInvalidFor(_dimensions,0), extentOrInvalidFor(_dimensions,1),
                             extentOrInvalidFor(_dimensions,2), extentOrInvalidFor(_dimensions,3),
                             extentOrInvalidFor(_dimensions,4), extentOrInvalidFor(_dimensions,5),
                             extentOrInvalidFor(_dimensions,6));
      }
      else
      {
        _hostView = HostView(_dualView.view_host(), rank);
      }
      _deviceView = DeviceView(_dualView.view_device(), rank);
    }

    void recordAllocation(AllocationRecord &record, long long entryCount) const
    {
      Camellia::recordAllocation<Scalar>(record, entryCount);

      bool recordIsForFieldContainer = (&record == &fcAllocationRecord);
      if (recordIsForFieldContainer)
      {
        _fieldContainerAllocationNumber = record.allocationCount;
        _fieldContainerDataAllocationRecorded = true;
        if (recordOutstandingFCAllocationNumbers)
        {
          outstandingFCAllocationNumbers.insert(_fieldContainerAllocationNumber);
        }
      }
    }

    void recordDeallocation(AllocationRecord &record, long long entryCount) const
    {
      bool recordIsForFieldContainer = (&record == &fcAllocationRecord);
      if (recordIsForFieldContainer && !_fieldContainerDataAllocationRecorded)
      {
        return;
      }

      Camellia::recordDeallocation<Scalar>(record, entryCount);
      if (recordIsForFieldContainer)
      {
        _fieldContainerDataAllocationRecorded = false;
        if (recordOutstandingFCAllocationNumbers)
        {
          if (outstandingFCAllocationNumbers.find(_fieldContainerAllocationNumber) != outstandingFCAllocationNumbers.end())
          {
            outstandingFCAllocationNumbers.erase(_fieldContainerAllocationNumber);
          }
          else
          {
            std::cout << "WARNING: deallocation event for FC allocation number " << _fieldContainerAllocationNumber;
            std::cout << ", but that allocation number is not contained in outstandingFCAllocationNumbers.\n";
          }
        }
      }
    }

    void recordCurrentDeallocation() const
    {
      if (_fieldContainerDataAllocationRecorded)
      {
        recordDeallocation(fcAllocationRecord, this->size());
      }
      if (_deviceAllocationRecorded)
      {
        Camellia::recordDeallocation<Scalar>(deviceViewAllocationRecord, this->size());
        _deviceAllocationRecorded = false;
      }
    }

    Scalar* hostData() const
    {
      return (_externalData != nullptr) ? _externalData : _hostView.data();
    }

    void copyExternalDataToHostView() const
    {
      if (_externalData != nullptr)
      {
        Scalar* hostViewData = _dualView.view_host().data();
        for (int i=0; i<this->size(); i++) hostViewData[i] = _externalData[i];
      }
    }

    void copyHostViewToExternalData() const
    {
      if (_externalData != nullptr)
      {
        const Scalar* hostViewData = _dualView.view_host().data();
        for (int i=0; i<this->size(); i++) _externalData[i] = hostViewData[i];
      }
    }

    void clearModifiedFlags() const
    {
      _hostModified = false;
      _deviceModified = false;
      _dualView.clear_sync_state();
    }

    void markHostModified() const
    {
      if (!_hostModified)
      {
        _dualView.template modify<HostMemorySpace>();
      }
      _hostModified = true;
      _deviceModified = false;
    }

    void markDeviceModified() const
    {
      if (!_deviceModified)
      {
        _dualView.template modify<DeviceMemorySpace>();
      }
      _deviceModified = true;
      _hostModified = false;
    }

    void syncHost() const
    {
      Kokkos::fence();
      Kokkos::deep_copy(_dualView.view_host(), _dualView.view_device());
      copyHostViewToExternalData();
      recordDeepCopy<Scalar>(deviceToHostDeepCopyRecord, this->size());
      clearModifiedFlags();
    }

    void syncDevice() const
    {
      Kokkos::fence();
      copyExternalDataToHostView();
      Kokkos::deep_copy(_dualView.view_device(), _dualView.view_host());
      recordDeepCopy<Scalar>(hostToDeviceDeepCopyRecord, this->size());
      clearModifiedFlags();
    }

    void allocateStorage(const Teuchos::Array<int> &dimensions, const bool initializeToZero)
    {
      recordCurrentDeallocation();

      _externalData = nullptr;
      _dimensions = dimensions;
      cacheDims();

      _dualView = DualView(DEFAULT_VIEW_NAME,
                           extentFor(_dimensions,0), extentFor(_dimensions,1), extentFor(_dimensions,2), extentFor(_dimensions,3),
                           extentFor(_dimensions,4), extentFor(_dimensions,5), extentFor(_dimensions,6));
      refreshViews();

      recordAllocation(fcAllocationRecord, this->size());
      Camellia::recordAllocation<Scalar>(deviceViewAllocationRecord, this->size());
      _deviceAllocationRecorded = true;

      if (initializeToZero)
      {
        Kokkos::deep_copy(_dualView.view_host(), Scalar(0));
      }
      clearModifiedFlags();
    }

  public:
    DeviceView checkoutDeviceView(bool deepCopyFromHost=true)
    {
      if (deepCopyFromHost)
      {
        syncDevice();
      }
      else
      {
        // The caller intends to overwrite the device view.  Newly constructed or
        // resized FieldContainers have host-side initialization marked as modified;
        // if we leave that stale host flag set while the device is written, the
        // subsequent checkinDeviceView(true) can fail to propagate the device
        // result back to host.  Discard existing sync state before handing out an
        // overwrite-only device view.
        clearModifiedFlags();
      }
      _deviceViewIsCheckedOut = true;
      return _deviceView;
    }

    DeviceViewConst checkoutDeviceView() const
    {
      syncDevice();
      _deviceViewIsCheckedOut = true;
      return DeviceViewConst(_deviceView);
    }

    void checkinDeviceView(bool deepCopyToHost) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!_deviceViewIsCheckedOut, std::invalid_argument, "this check-in request is for a device view that is not currently checked out");
      if (deepCopyToHost)
      {
        syncHost();
      }
      _deviceViewIsCheckedOut = false;
    }

    FieldContainer()
    {
      allocateStorage(Teuchos::Array<int>(), true);
    }

    FieldContainer(const FieldContainer& right)
    {
      *this = right;
    }

    FieldContainer(const int dim0)
    {
      Teuchos::Array<int> dimensions(1); dimensions[0] = dim0;
      allocateStorage(dimensions, true);
    }

    FieldContainer(const int dim0, const int dim1)
    {
      Teuchos::Array<int> dimensions(2); dimensions[0] = dim0; dimensions[1] = dim1;
      allocateStorage(dimensions, true);
    }

    FieldContainer(const int dim0, const int dim1, const int dim2)
    {
      Teuchos::Array<int> dimensions(3); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2;
      allocateStorage(dimensions, true);
    }

    FieldContainer(const int dim0, const int dim1, const int dim2, const int dim3)
    {
      Teuchos::Array<int> dimensions(4); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2; dimensions[3] = dim3;
      allocateStorage(dimensions, true);
    }

    FieldContainer(const int dim0, const int dim1, const int dim2, const int dim3, const int dim4)
    {
      Teuchos::Array<int> dimensions(5); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2; dimensions[3] = dim3; dimensions[4] = dim4;
      allocateStorage(dimensions, true);
    }

    FieldContainer(const Teuchos::Array<int>& dimensions)
    {
      allocateStorage(dimensions, true);
    }

    FieldContainer(const Teuchos::Array<int>& dimensions, const bool initializeToZero)
    {
      allocateStorage(dimensions, initializeToZero);
    }

    FieldContainer(const Teuchos::Array<int>& dimensions,
                   const Teuchos::ArrayView<Scalar>& data)
    {
      allocateStorage(dimensions, false);
      setValues(data);
    }

    FieldContainer(const Teuchos::Array<int>& dimensions,
                   const Teuchos::ArrayRCP<Scalar>& data)
    {
      allocateStorage(dimensions, false);
      setValues(data());
    }

    FieldContainer(const Teuchos::Array<int>& dimensions,
                   Scalar* data,
                   const bool deep_copy = false,
                   const bool owns_mem  = false)
    {
      (void)owns_mem;
      allocateStorage(dimensions, false);
      if (!deep_copy)
      {
        _externalData = data;
        refreshViews();
      }
      setValues(data, this->size());
    }

    FieldContainer(const shards::Array<Scalar,shards::NaturalOrder>& data,
                   const bool deep_copy = false,
                   const bool owns_mem  = false)
    {
      (void)owns_mem;
      Teuchos::Array<int> dimensions(data.rank());
      for (int d=0; d<data.rank(); d++) dimensions[d] = data.dimension(d);
      allocateStorage(dimensions, false);
      if (!deep_copy)
      {
        _externalData = data.contiguous_data();
        refreshViews();
      }
      setValues(data.contiguous_data(), this->size());
    }

    ~FieldContainer()
    {
      if (!Kokkos::is_finalized())
      {
        recordCurrentDeallocation();
      }
    }

    int rank() const
    {
      return _dimensions.size();
    }

    int size() const
    {
      return _size;
    }

    template<class Vector>
    void dimensions(Vector& dimensions) const
    {
      dimensions.assign(_dimensions.begin(), _dimensions.end());
    }

    int dimension(const int whichDim) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION((whichDim < 0) || (whichDim >= this->rank()), std::invalid_argument, "dimension order is out of range");
      return _dimensions[whichDim];
    }

    int getEnumeration(const int i0) const { return i0; }
    int getEnumeration(const int i0, const int i1) const { return i0*_dim1 + i1; }
    int getEnumeration(const int i0, const int i1, const int i2) const { return (i0*_dim1 + i1)*_dim2 + i2; }
    int getEnumeration(const int i0, const int i1, const int i2, const int i3) const { return ((i0*_dim1 + i1)*_dim2 + i2)*_dim3 + i3; }
    int getEnumeration(const int i0, const int i1, const int i2, const int i3, const int i4) const { return (((i0*_dim1 + i1)*_dim2 + i2)*_dim3 + i3)*_dim4 + i4; }

    int getEnumeration(const Teuchos::Array<int>& multiIndex) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(multiIndex.size() != _dimensions.size(), std::invalid_argument, "Number of multi-indices does not match rank of container.");
      if (multiIndex.size() == 0) return 0;
      int address = multiIndex[0];
      for (int r=1; r<multiIndex.size(); r++) address = address * _dimensions[r] + multiIndex[r];
      return address;
    }

    void getMultiIndex(int &i0, const int valueEnum) const
    {
      i0 = valueEnum;
    }

    void getMultiIndex(int &i0, int &i1, const int valueEnum) const
    {
      i0 = valueEnum / _dim1;
      i1 = valueEnum - i0 * _dim1;
    }

    void getMultiIndex(int &i0, int &i1, int &i2, const int valueEnum) const
    {
      int tempDim = _dim1 * _dim2;
      int tempEnum = valueEnum;
      i0 = tempEnum / tempDim;
      tempEnum -= i0 * tempDim;
      tempDim /= _dim1;
      i1 = tempEnum / tempDim;
      tempEnum -= i1 * tempDim;
      i2 = tempEnum;
    }

    void getMultiIndex(int &i0, int &i1, int &i2, int &i3, const int valueEnum) const
    {
      int tempDim = _dim1 * _dim2 * _dim3;
      int tempEnum = valueEnum;
      i0 = tempEnum / tempDim;
      tempEnum -= i0 * tempDim;
      tempDim /= _dim1;
      i1 = tempEnum / tempDim;
      tempEnum -= i1 * tempDim;
      tempDim /= _dim2;
      i2 = tempEnum / tempDim;
      tempEnum -= i2 * tempDim;
      i3 = tempEnum;
    }

    void getMultiIndex(int &i0, int &i1, int &i2, int &i3, int &i4, const int valueEnum) const
    {
      int tempDim = _dim1 * _dim2 * _dim3 * _dim4;
      int tempEnum = valueEnum;
      i0 = tempEnum / tempDim;
      tempEnum -= i0 * tempDim;
      tempDim /= _dim1;
      i1 = tempEnum / tempDim;
      tempEnum -= i1 * tempDim;
      tempDim /= _dim2;
      i2 = tempEnum / tempDim;
      tempEnum -= i2 * tempDim;
      tempDim /= _dim3;
      i3 = tempEnum / tempDim;
      tempEnum -= i3 * tempDim;
      i4 = tempEnum;
    }

    template<class Vector>
    void getMultiIndex(Vector &multiIndex, const int valueEnum) const
    {
      const int theRank = this->rank();
      multiIndex.resize(theRank);
      if (theRank == 0) return;
      int tempEnum = valueEnum;
      int tempRange = 1;
      for (int r=1; r<theRank; r++) tempRange *= _dimensions[r];
      multiIndex[0] = tempEnum / tempRange;
      for (int r=1; r<theRank-1; r++)
      {
        tempEnum -= multiIndex[r-1] * tempRange;
        tempRange /= _dimensions[r];
        multiIndex[r] = tempEnum / tempRange;
      }
      if (theRank > 1) multiIndex[theRank-1] = tempEnum - multiIndex[theRank-2] * tempRange;
    }

    void clear()
    {
      allocateStorage(Teuchos::Array<int>(), true);
    }

    void resize(const int dim0)
    {
      Teuchos::Array<int> dimensions(1); dimensions[0] = dim0;
      resize(dimensions);
    }

    void resize(const int dim0, const int dim1)
    {
      Teuchos::Array<int> dimensions(2); dimensions[0] = dim0; dimensions[1] = dim1;
      resize(dimensions);
    }

    void resize(const int dim0, const int dim1, const int dim2)
    {
      Teuchos::Array<int> dimensions(3); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2;
      resize(dimensions);
    }

    void resize(const int dim0, const int dim1, const int dim2, const int dim3)
    {
      Teuchos::Array<int> dimensions(4); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2; dimensions[3] = dim3;
      resize(dimensions);
    }

    void resize(const int dim0, const int dim1, const int dim2, const int dim3, const int dim4)
    {
      Teuchos::Array<int> dimensions(5); dimensions[0] = dim0; dimensions[1] = dim1; dimensions[2] = dim2; dimensions[3] = dim3; dimensions[4] = dim4;
      resize(dimensions);
    }

    void resize(const Teuchos::Array<int>& newDimensions)
    {
      const int oldSize = this->size();
      const int newSize = product(newDimensions);

      if ((_externalData != nullptr) && (newSize == oldSize))
      {
        _dimensions = newDimensions;
        cacheDims();
        refreshViews();
        clearModifiedFlags();
        return;
      }

      std::vector<Scalar> oldValues(oldSize);
      if (oldSize > 0)
      {
        const Scalar* oldData = hostData();
        for (int i=0; i<oldSize; i++) oldValues[i] = oldData[i];
      }

      allocateStorage(newDimensions, true);

      const int copyCount = std::min(oldSize, this->size());
      Scalar* newData = hostData();
      for (int i=0; i<copyCount; i++) newData[i] = oldValues[i];
      clearModifiedFlags();
    }

    void resize(const FieldContainer<Scalar, ArrayTypeId>& anotherContainer)
    {
      Teuchos::Array<int> dimensions;
      anotherContainer.dimensions(dimensions);
      allocateStorage(dimensions, true);
    }

    void resize(const int numPoints,
                const int numFields,
                const Intrepid2::EFunctionSpace spaceType,
                const Intrepid2::EOperator operatorType,
                const int spaceDim)
    {
      const int fieldRank    = Intrepid2::getFieldRank(spaceType);
      const int operatorRank = Intrepid2::getOperatorRank(spaceType,operatorType,spaceDim);
      const int theRank = 1 + 1 + fieldRank + operatorRank;
      Teuchos::Array<int> newDimensions(theRank);
      newDimensions[0] = numPoints;
      newDimensions[1] = numFields;
      switch (operatorType)
      {
        case Intrepid2::OPERATOR_VALUE:
        case Intrepid2::OPERATOR_GRAD:
        case Intrepid2::OPERATOR_D1:
        case Intrepid2::OPERATOR_CURL:
        case Intrepid2::OPERATOR_DIV:
          for (int i=0; i<fieldRank+operatorRank; i++) newDimensions[2+i] = spaceDim;
          break;
        case Intrepid2::OPERATOR_D2:
        case Intrepid2::OPERATOR_D3:
        case Intrepid2::OPERATOR_D4:
        case Intrepid2::OPERATOR_D5:
        case Intrepid2::OPERATOR_D6:
        case Intrepid2::OPERATOR_D7:
        case Intrepid2::OPERATOR_D8:
        case Intrepid2::OPERATOR_D9:
        case Intrepid2::OPERATOR_D10:
          for (int i=0; i<fieldRank; i++) newDimensions[2+i] = spaceDim;
          newDimensions[2+fieldRank] = Intrepid2::getDkCardinality(operatorType,spaceDim);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid operator type");
      }
      resize(newDimensions);
    }

    void initialize(const Scalar value = 0)
    {
      Kokkos::deep_copy(_hostView, value);
      clearModifiedFlags();
    }

    Scalar getValue(const Teuchos::Array<int>& multiIndex) const
    {
      return (*this)[getEnumeration(multiIndex)];
    }

    void setValue(const Scalar dataValue, const Teuchos::Array<int>& multiIndex)
    {
      (*this)[getEnumeration(multiIndex)] = dataValue;
    }

    void setValue(const Scalar dataValue, const int order)
    {
      (*this)[order] = dataValue;
    }

    void setValues(const Teuchos::ArrayView<Scalar>& dataArray)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(dataArray.size() != this->size(), std::invalid_argument, "Size of argument does not match the size of container.");
      Scalar* data = hostData();
      for (int i=0; i<this->size(); i++) data[i] = dataArray[i];
      clearModifiedFlags();
    }

    void setValues(const Scalar* dataPtr, const int numData)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(numData != this->size(), std::invalid_argument, "Number of data does not match the size of container.");
      Scalar* data = hostData();
      for (int i=0; i<numData; i++) data[i] = dataPtr[i];
      clearModifiedFlags();
    }

    Scalar* data()
    {
      markHostModified();
      return hostData();
    }

    const Scalar* data() const
    {
      return hostData();
    }

    Teuchos::ArrayRCP<Scalar> getData()
    {
      return Teuchos::arcp<Scalar>(hostData(), 0, this->size(), false);
    }

    Teuchos::ArrayRCP<const Scalar> getData() const
    {
      return Teuchos::arcp<const Scalar>(hostData(), 0, this->size(), false);
    }

    const Scalar& operator()(const int i0) const { return _hostView(i0); }
    Scalar& operator()(const int i0) { return _hostView(i0); }

    const Scalar& operator()(const int i0, const int i1) const { return _hostView(i0,i1); }
    Scalar& operator()(const int i0, const int i1) { return _hostView(i0,i1); }

    const Scalar& operator()(const int i0, const int i1, const int i2) const { return _hostView(i0,i1,i2); }
    Scalar& operator()(const int i0, const int i1, const int i2) { return _hostView(i0,i1,i2); }

    const Scalar& operator()(const int i0, const int i1, const int i2, const int i3) const { return _hostView(i0,i1,i2,i3); }
    Scalar& operator()(const int i0, const int i1, const int i2, const int i3) { return _hostView(i0,i1,i2,i3); }

    const Scalar& operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const { return _hostView(i0,i1,i2,i3,i4); }
    Scalar& operator()(const int i0, const int i1, const int i2, const int i3, const int i4) { return _hostView(i0,i1,i2,i3,i4); }

    const Scalar& operator[](const int address) const
    {
      return hostData()[address];
    }

    Scalar& operator[](const int address)
    {
      return hostData()[address];
    }

    FieldContainer<Scalar>& operator=(const FieldContainer<Scalar>& right)
    {
      TEUCHOS_TEST_FOR_EXCEPTION((this == &right), std::invalid_argument, "Invalid right-hand side to '='. Self-assignment prohibited.");
      Teuchos::Array<int> dimensions;
      right.dimensions(dimensions);
      allocateStorage(dimensions, false);
      const Scalar* rightData = right.hostData();
      Scalar* data = hostData();
      for (int i=0; i<this->size(); i++) data[i] = rightData[i];
      clearModifiedFlags();
      return *this;
    }

    static DeviceView getDeviceView(typename Camellia::FieldContainer<Scalar>::DeviceView &dv)
    {
      DeviceView deviceView = Kokkos::create_mirror(DeviceMemorySpace(), dv);
      Camellia::recordAllocation<Scalar>(deviceViewAllocationRecord, deviceView.size());
      return deviceView;
    }

    template<int rank>
    static Kokkos::MDRangePolicy< Kokkos::Rank<rank> > getMDRangePolicy(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv)
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<rank> >::array_index_type;
      Kokkos::Array<array_index_type,rank> mdRangeBegin;
      Kokkos::Array<array_index_type,rank> mdRangeEnd;
      for (int i=0; i<rank; i++)
      {
        mdRangeBegin[i] = 0;
        mdRangeEnd[i] = 1;
      }
      int numRanks = std::min((int) dv.rank(), rank);
      for (int i=0; i<numRanks; i++) mdRangeEnd[i] = dv.extent_int(i);
      return Kokkos::MDRangePolicy< Kokkos::Rank<rank> >(mdRangeBegin,mdRangeEnd);
    }

    template<int rank>
    static Kokkos::MDRangePolicy< Kokkos::Rank<rank> > getMDRangePolicy(Kokkos::Array<int,rank> dimSizes)
    {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<rank> >::array_index_type;
      Kokkos::Array<array_index_type,rank> mdRangeBegin;
      Kokkos::Array<array_index_type,rank> mdRangeEnd;
      for (int i=0; i<rank; i++)
      {
        mdRangeBegin[i] = 0;
        mdRangeEnd[i] = dimSizes[i];
      }
      return Kokkos::MDRangePolicy< Kokkos::Rank<rank> >(mdRangeBegin,mdRangeEnd);
    }

    static Kokkos::MDRangePolicy< Kokkos::Rank<2> > getMDRangePolicy2(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv) { return getMDRangePolicy<2>(dv); }
    static Kokkos::MDRangePolicy< Kokkos::Rank<3> > getMDRangePolicy3(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv) { return getMDRangePolicy<3>(dv); }
    static Kokkos::MDRangePolicy< Kokkos::Rank<4> > getMDRangePolicy4(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv) { return getMDRangePolicy<4>(dv); }
    static Kokkos::MDRangePolicy< Kokkos::Rank<5> > getMDRangePolicy5(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv) { return getMDRangePolicy<5>(dv); }
    static Kokkos::MDRangePolicy< Kokkos::Rank<6> > getMDRangePolicy6(const typename Camellia::FieldContainer<Scalar>::DeviceView &dv) { return getMDRangePolicy<6>(dv); }

    Kokkos::MDRangePolicy< Kokkos::Rank<2> > getMDRangePolicy2()
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!(_deviceViewIsCheckedOut), std::invalid_argument, "getMDRangePolicy _deviceView is not checked out.");
      return getMDRangePolicy2(this->_deviceView);
    }

    Kokkos::MDRangePolicy< Kokkos::Rank<6> > getMDRangePolicy6()
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!(_deviceViewIsCheckedOut), std::invalid_argument, "getMDRangePolicy _deviceView is not checked out.");
      return getMDRangePolicy6(this->_deviceView);
    }

    void printHostViewMetadata(std::ostream &out = std::cout)
    {
      out << "  _hostView.rank()    = " << _hostView.rank() << std::endl;
      out << "  _hostView.data()    = " << _hostView.data() << std::endl;
      for (int i=0; i<_hostView.rank(); i++) out << "  _hostView.extent(" << std::setw(1) << i << ") = " << _hostView.extent(i) << std::endl;
      out << "  _hostView.size()    = " << _hostView.size() << std::endl;
      out << "typeid(_hostView  )   = " << typeid(_hostView).name() << std::endl;
    }

    void printDeviceViewMetadata(std::ostream &out = std::cout)
    {
      out << "_deviceView.rank()    = " << _deviceView.rank() << std::endl;
      out << "_deviceView.data()    = " << _deviceView.data() << std::endl;
      for (int i=0; i<_deviceView.rank(); i++) out << "_deviceView.extent(" << std::setw(1) << i << ") = " << _deviceView.extent(i) << std::endl;
      out << "_deviceView.size()    = " << _deviceView.size() << std::endl;
      out << "typeid(_deviceView)   = " << typeid(_deviceView).name() << std::endl;
    }
  };

  template<class Scalar, int ArrayTypeId>
  std::ostream& operator<<(std::ostream& os, const FieldContainer<Scalar, ArrayTypeId>& container)
  {
    Teuchos::oblackholestream oldFormatState;
    oldFormatState.copyfmt(os);
    os.setf(std::ios_base::scientific, std::ios_base::floatfield);
    os.setf(std::ios_base::right);
    int myprec = os.precision();

    int size = container.size();
    int rank = container.rank();
    Teuchos::Array<int> multiIndex(rank);
    Teuchos::Array<int> dimensions;
    container.dimensions(dimensions);

    os << "===============================================================================\n"
       << "\t Container size = " << size << "\n"
       << "\t Container rank = " << rank << "\n";

    if ((rank == 0) && (size == 0))
    {
      os << "====================================================================================\n"
         << "|                        *** This is an empty container ****                       |\n";
    }
    else
    {
      os << "\t Dimensions     = ";
      for (int r=0; r<rank; r++) os << " (" << dimensions[r] << ") ";
      os << "\n";
      os << "====================================================================================\n"
         << "|              Multi-index          Enumeration             Value                  |\n"
         << "====================================================================================\n";
    }

    for (int address=0; address<size; address++)
    {
      container.getMultiIndex(multiIndex,address);
      std::ostringstream mistring;
      for (int r=0; r<rank; r++) mistring << multiIndex[r] << std::dec << " ";
      os.setf(std::ios::right, std::ios::adjustfield);
      os << std::setw(27) << mistring.str();
      os << std::setw(20) << address;
      os << "             ";
      os.setf(std::ios::left, std::ios::adjustfield);
      os << std::setw(myprec+8) << container[address] << "\n";
    }

    os << "====================================================================================\n\n";
    os.copyfmt(oldFormatState);
    return os;
  }

  std::string allocationReport(AllocationRecord &record);

  typedef Teuchos::RCP< Camellia::FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const Camellia::FieldContainer<double> > constFCPtr;
}

#endif // Camellia_FieldContainer_hpp
