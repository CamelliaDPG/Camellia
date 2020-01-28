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

#include "Intrepid_FieldContainer.hpp"

#include "Kokkos_DynRankView.hpp"

#include <set>

namespace Camellia
{
  extern const std::string DEFAULT_VIEW_NAME;
  
  //! AllocationRecord allows us to track allocations.
  //! We store two instances of this:
  //! - one that tracks FieldContainer allocations
  //! - one that tracks "Device View" allocations (which are no-ops when Device memory space is the same as Host)
  struct AllocationRecord
  {
    int allocationCount = 0;
    long long bytesAllocatedTotal = 0;
    long long bytesAllocatedNet = 0;
    long long bytesAllocatedNetHighWaterMark = 0;
  };
  
  extern AllocationRecord fcAllocationRecord;
  extern AllocationRecord deviceViewAllocationRecord;
  extern std::set<long long> outstandingFCAllocationNumbers;
  extern bool recordOutstandingFCAllocationNumbers; // if false, we skip tracking of the outstanding FC numbers, which does have some performance cost (about 8% penalty for the test suite as a whole)
  
  struct DeepCopyRecord
  {
    int deepCopyCount = 0;
    long long bytesDeepCopied = 0;
  };
  
  extern DeepCopyRecord deviceToHostDeepCopyRecord;
  extern DeepCopyRecord hostToDeviceDeepCopyRecord;
  
  template<typename Scalar>
  void recordDeepCopy(DeepCopyRecord &record, long long entryCount)
  {
    record.deepCopyCount++;
    record.bytesDeepCopied += entryCount * sizeof(Scalar);
  }
  
  template<typename Scalar>
  void recordAllocation(AllocationRecord &record, long long entryCount)
  {
    record.allocationCount++;
    long long bytes = sizeof(Scalar) * entryCount;
    record.bytesAllocatedNet   += bytes;
    record.bytesAllocatedTotal += bytes;
    if (record.bytesAllocatedNet > record.bytesAllocatedNetHighWaterMark)
    {
      record.bytesAllocatedNetHighWaterMark = record.bytesAllocatedNet;
    }
  }

  template<typename Scalar>
  void recordDeallocation(AllocationRecord &record, long long entryCount)
  {
    long long bytes = sizeof(Scalar) * entryCount;
    record.bytesAllocatedNet -= bytes;
  }

  template<class Scalar, int ArrayTypeId=0>
  class FieldContainer : public Intrepid::FieldContainer<Scalar, ArrayTypeId>
  {
    
  public:
    using FieldContainerBase = Intrepid::FieldContainer<Scalar, ArrayTypeId>;
 #ifdef KOKKOS_ENABLE_CUDA
    using DeviceMemorySpace = Kokkos::CudaSpace; // we want to avoid UVM; Kokkos default may be Kokkos::CudaUVMSpace
 #else
    using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
 #endif
    using HostView   = Kokkos::DynRankView<Scalar, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using ViewTraits = typename HostView::drvtraits;
    using DeviceView = Kokkos::DynRankView<typename ViewTraits::value_type, typename ViewTraits::array_layout, DeviceMemorySpace>;
    using DeviceViewConst = Kokkos::DynRankView<const typename ViewTraits::value_type, typename ViewTraits::array_layout, DeviceMemorySpace>;
    using HostViewConst   = Kokkos::DynRankView<const Scalar, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    
  private:
    mutable DeviceView      _deviceView;
    mutable HostView        _hostView;
    mutable DeviceViewConst _deviceViewConst;
    mutable HostViewConst   _hostViewConst;
    
    mutable bool _deviceViewIsCheckedOut = false; // it's an error to check out twice without checking back in first; data copies happen at check in
    mutable bool _deviceViewIsValid      = false; // if invalid, call initializeDeviceView() on checkout
    bool _hostViewIsValid        = false; // if invalid, call initializeHostView() before access to _hostView -- right now only used for creating mirror in initializeDeviceView()
    
    mutable bool _hostViewConstIsValid   = false; // if invalid, call initializeHostViewConst() before access to _hostViewConst -- right now only used for creating mirror in initializeDeviceView()
    
    mutable bool _fieldContainerDataAllocationRecorded = false; // we don't count deallocation if we didn't count allocation; this allows us to enforce that.
    mutable int  _fieldContainerAllocationNumber = -1;
    
    template<typename ViewType>
    ViewType allocateView(const std::string &label) const
    {
      const int rank = this->rank();
      Kokkos::Array<int,7> dims = {1,1,1,1,1,1,1}; // maximum rank of DynRankView is 7
      for (int d=0; d<rank; d++)
      {
        dims[d] = this->dimension(d);
      }
      
      switch (rank)
      {
        case 0:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label));
        case 1:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0]);
        case 2:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1]);
        case 3:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2]);
        case 4:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3]);
          break;
        case 5:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4]);
        case 6:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
        case 7:
          return ViewType(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
      }
    }
    
    void initializeHostView()
    {
      const int rank = this->rank();
      Scalar* dataPtr = this->getData().getRawPtr();
      Kokkos::Array<int,7> dims = {1,1,1,1,1,1,1}; // maximum rank of DynRankView is 7
      for (int d=0; d<rank; d++)
      {
        dims[d] = this->dimension(d);
      }
      
      switch (rank)
      {
        case 0:
          _hostView   = HostView(dataPtr);
          break;
        case 1:
          _hostView   = HostView(dataPtr,dims[0]);
          break;
        case 2:
          _hostView   = HostView(dataPtr,dims[0],dims[1]);
          break;
        case 3:
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2]);
          break;
        case 4:
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3]);
          break;
        case 5:
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4]);
          break;
        case 6:
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
          break;
        case 7:
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
      }
      _hostViewIsValid = true;
    }
    
    void initializeHostViewConst() const
    {
      const int rank = this->rank();
      const Scalar* dataPtr = this->getData().getRawPtr();
      Kokkos::Array<int,7> dims = {1,1,1,1,1,1,1}; // maximum rank of DynRankView is 7
      for (int d=0; d<rank; d++)
      {
        dims[d] = this->dimension(d);
      }
      
      switch (rank)
      {
        case 0:
          _hostViewConst  = HostViewConst(dataPtr);
          break;
        case 1:
          _hostViewConst  = HostViewConst(dataPtr,dims[0]);
          break;
        case 2:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1]);
          break;
        case 3:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1],dims[2]);
          break;
        case 4:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1],dims[2],dims[3]);
          break;
        case 5:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4]);
          break;
        case 6:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
          break;
        case 7:
          _hostViewConst  = HostViewConst(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
      }
      _hostViewConstIsValid = true;
    }
    
    void initializeDeviceView()
    {
      if (!_hostViewIsValid)
      {
        initializeHostView();
      }
      
      // if _deviceView has previously been allocated, since we'll implicitly deallocate when we reset _deviceView below, we should
      // subtract off the previous allocation size.  (This comes up during calls to resize() and clear().)
      recordDeallocation(deviceViewAllocationRecord, _deviceView.size());
      _deviceView = Kokkos::create_mirror_view(DeviceMemorySpace(),_hostView);
      recordAllocation(deviceViewAllocationRecord, _deviceView.size());
      
      _deviceViewIsValid = true;
    }
    
    // const version
    void initializeDeviceViewConst() const
    {
      if (!_hostViewConstIsValid)
      {
        initializeHostViewConst();
      }
      
      // if _deviceView has previously been allocated, since we'll implicitly deallocate when we reset _deviceView below, we should
      // subtract off the previous allocation size.  (This comes up during calls to resize() and clear().)
      recordDeallocation(deviceViewAllocationRecord, _deviceView.size());
      const std::string label = "Camellia::FieldContainer device view";
      _deviceView = allocateView<DeviceView>(label);
      recordAllocation(deviceViewAllocationRecord, _deviceView.size());
      
      _deviceViewIsValid = true;
    }
    
    void invalidateViews()
    {
      // called when the FC is resized, e.g.
      _hostViewIsValid = false;
      _hostViewConstIsValid = false;
      
      // record the deallocation
      if (_deviceViewIsValid)
      {
        _deviceViewIsValid = false;
        recordDeallocation(deviceViewAllocationRecord, _deviceView.size());
      }

      _deviceView      = DeviceView();
      _hostView        = HostView();
      _hostViewConst   = HostViewConst();
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
        // then don't actually record the deallocation
        return;
      }
      
      Camellia::recordDeallocation<Scalar>(record, entryCount);
      if (recordIsForFieldContainer)
      {
        _fieldContainerDataAllocationRecorded = false; // don't allow to record deallocation twice without recording an allocation
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
    
  public:
    DeviceView checkoutDeviceView(bool deepCopyFromHost=true)
    {
      if (!_deviceViewIsValid)
      {
        //std::cout << "initializeDeviceView..." << std::endl;
        initializeDeviceView();
      }
      _deviceViewIsCheckedOut = true;
      //printHostViewMetadata();
      //printDeviceViewMetadata();
      
      // ideally, we should guard this so that we don't do unnecessary copies when memory spaces are the same
      // but this is an optimization that we can make at any point
      // the priority right now is to provide the same interface as FieldContainer, so that we have a
      // drop-in replacement, and to allow us to gradually shift to using Kokkos::View instead.
      // the checkout semantics here allow us at least some guard against accidental usage of UVM.
      //std::cout << std::endl << "deep_copy host2device..." << std::endl;
      if (deepCopyFromHost)
      {
        Kokkos::deep_copy(_deviceView,_hostView);
        recordDeepCopy<Scalar>(hostToDeviceDeepCopyRecord, _hostView.size());
      }
      return _deviceView;
    }
    
    DeviceViewConst checkoutDeviceView() const
    {
      if (!_deviceViewIsValid)
      {
        initializeDeviceViewConst();
        Kokkos::deep_copy(_deviceView,_hostViewConst);
        recordDeepCopy<Scalar>(hostToDeviceDeepCopyRecord, _hostViewConst.size());
      }
      _deviceViewIsCheckedOut = true;
      
      return _deviceView;
    }
    
    void checkinDeviceView(bool deepCopyToHost=true)
    {
      // copy back to host
      //std::cout << "deep_copy device2host..." << std::endl;
      if (deepCopyToHost)
      {
        Kokkos::deep_copy(_hostView,_deviceView);
        recordDeepCopy<Scalar>(deviceToHostDeepCopyRecord, _hostView.size());
      }
      _deviceViewIsCheckedOut = false;
    }
    
    void checkinDeviceView() const
    {
      _deviceViewIsCheckedOut = false;
    }
    
    /** \brief Default constructor.
     */
    FieldContainer()
    :
    FieldContainerBase()
    { }
    
    /** \brief Copy constructor.
     */
    FieldContainer(const FieldContainer& right)
    :
    FieldContainerBase(right)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Copy constructor with Camellia::FieldContainer as source.
     */
    FieldContainer(const FieldContainerBase& right)
    :
    FieldContainerBase(right)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a rank-1 FieldContainer with the specified dimension, initialized by 0.
     
     \param dim0    [in]      - dimension for the only index
     */
    FieldContainer(const int dim0)
    :
    FieldContainerBase(dim0)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a rank-2 FieldContainer with the specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     */
    FieldContainer(const int dim0, const int dim1)
    :
    FieldContainerBase(dim0, dim1)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a rank-3 FieldContainer with the specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     */
    FieldContainer(const int dim0, const int dim1, const int dim2)
    :
    FieldContainerBase(dim0, dim1, dim2)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a rank-4 FieldContainer with the specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     \param dim3    [in]      - dimension for the 4th index
     */
    FieldContainer(const int dim0, const int dim1, const int dim2, const int dim3)
    :
    FieldContainerBase(dim0, dim1, dim2, dim3)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a rank-5 FieldContainer with the specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     \param dim3    [in]      - dimension for the 4th index
     \param dim4    [in]      - dimension for the 5th index
     */
    FieldContainer(const int dim0, const int dim1, const int dim2, const int dim3, const int dim4)
    :
    FieldContainerBase(dim0, dim1, dim2, dim3, dim4)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a  FieldContainer of arbitrary rank,, initialized by 0, using dimensions
     specified in an array. The size of the input array implicitly defines the rank of the
     container and its capacity is defined by the specified dimensions.
     
     \param dimensions[in]           - array with container dimensions
     */
    FieldContainer(const Teuchos::Array<int>& dimensions)
    :
    FieldContainerBase(dimensions)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a FieldContainer of arbitrary rank, using dimensions specified in the
     <var><b>dimensions</b></var> array, and fills it by deep-copying data from a
     Teuchos::ArrayView array (which implicitly doubles as Teuchos::ArrayRCP or
     Teuchos::Array). If the input data array is a Teuchos::ArrayRCP, then '()' should
     be appended to it when calling this function. This forces direct conversion to a
     Teuchos::ArrayView, and prevents the call to the shallow-copy constructor that
     takes a Teuchos::ArrayRCP.
     
     \param dimensions[in]           - array with container dimensions
     \param data[in]                 - array with container values
     */
    FieldContainer(const Teuchos::Array<int>&        dimensions,
                   const Teuchos::ArrayView<Scalar>& data)
    :
    FieldContainerBase(dimensions, data)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a FieldContainer of arbitrary rank, using dimensions specified in the
     <var><b>dimensions</b></var> array, and wraps (shallow-copies) the data pointed
     to by the input Teuchos::ArrayRCP array. If a deep copy is desired instead,
     one can force the use of the constructor that takes Teuchos::ArrayView by
     appending () to the input Teuchos::ArrayRCP parameter. This forces direct
     conversion to a Teuchos::ArrayView.
     
     \param dimensions[in]           - array with container dimensions
     \param data[in]                 - array with container values
     */
    FieldContainer(const Teuchos::Array<int>&       dimensions,
                   const Teuchos::ArrayRCP<Scalar>& data)
    :
    FieldContainerBase(dimensions, data)
    {
      invalidateViews();
    }
    
    /** \brief Creates a FieldContainer of arbitrary rank, using dimensions specified in the
     <var><b>dimensions</b></var> array, and either wraps (shallow-copies) Scalar*
     <var><b>data</b></var>, or deep-copies it, based on the value of the parameter
     <var><b>deep_copy</b></var>. Memory management through FieldContainer, via
     its Teuchos::ArrayRCP data member, can be enabled.
     
     \param dimensions[in]           - array with container dimensions
     \param data[in]                 - array with container values
     \param deep_copy[in]            - if true, then deep-copy, otherwise shallow-copy; default: false
     \param owns_mem[in]             - if true, the field container will manage memory; default: false
     */
    FieldContainer(const Teuchos::Array<int>&    dimensions,
                   Scalar*                       data,
                   const bool                    deep_copy = false,
                   const bool                    owns_mem  = false)
    :
    FieldContainerBase(dimensions, data, deep_copy, owns_mem)
    {
      if (deep_copy) recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    /** \brief Creates a FieldContainer either as a wrapper of the shards::Array<Scalar,shards::NaturalOrder>
     array <var><b>data</b></var>, or as its deep copy, based on the value of the parameter
     <var><b>deep_copy</b></var>. Memory management through FieldContainer, via
     its Teuchos::ArrayRCP data member, can be enabled.
     
     \param data[in]                 - array with container values
     \param deep_copy[in]            - if true, then deep-copy, otherwise shallow-copy; default: false
     \param owns_mem[in]             - if true, the field container will manage memory; default: false
     */
    FieldContainer(const shards::Array<Scalar,shards::NaturalOrder>&  data,
                   const bool                                         deep_copy = false,
                   const bool                                         owns_mem  = false)
    :
    FieldContainerBase(data, deep_copy, owns_mem)
    {
      if (deep_copy) recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
    }
    
    ~FieldContainer()
    {
      bool ownsMemory = this->getData().has_ownership();
      
      if (ownsMemory || (this->size() == 0))
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      invalidateViews();
    }
    
    /** \brief Clears FieldContainer to trivial container (one with rank = 0 and size = 0)
     */
    void clear()
    {
      bool ownsMemory = this->getData().has_ownership();
      
      if (ownsMemory || (this->size() == 0))
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::clear();
      invalidateViews();
    }
    
    /** \brief Resizes FieldContainer to a rank-1 container with the specified dimension, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     */
    void resize(const int dim0)
    {
      bool sizeChange = (dim0 != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(dim0);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to a rank-2 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     */
    void resize(const int dim0, const int dim1)
    {
      bool sizeChange = (dim0 * dim1 != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(dim0, dim1);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to a rank-3 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     */
    void resize(const int dim0, const int dim1, const int dim2)
    {
      bool sizeChange = (dim0*dim1*dim2 != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(dim0, dim1, dim2);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to a rank-4 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     \param dim3    [in]      - dimension for the 4th index
     */
    void resize(const int dim0, const int dim1, const int dim2, const int dim3)
    {
      bool sizeChange = (dim0*dim1*dim2*dim3 != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(dim0, dim1, dim2, dim3);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to a rank-5 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     \param dim3    [in]      - dimension for the 4th index
     \param dim4    [in]      - dimension for the 5th index
     */
    void resize(const int dim0,
                const int dim1,
                const int dim2,
                const int dim3,
                const int dim4)
    {
      bool sizeChange = (dim0*dim1*dim2*dim3*dim4 != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(dim0, dim1, dim2, dim3, dim4);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to arbitrary rank container, initialized by 0, with dimensions
     specified in the input array. The size of this array implicitely defined the rank of the FieldContainer.
     
     \param newDimensions[in]          - new upper values for index ranges
     */
    void resize(const Teuchos::Array<int>& newDimensions)
    {
      int newSize = 1;
      for (int i=0; i<newDimensions.size(); i++)
      {
        newSize *= newDimensions[i];
      }
      bool sizeChange = (newSize != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(newDimensions);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to have the same rank and dimensions as another FieldContainer,
     and initializes by 0.
     
     \param anotherContainer[in]          - a FieldContainer
     */
    void resize(const FieldContainer<Scalar, ArrayTypeId>& anotherContainer)
    {
      bool sizeChange = (anotherContainer.size() != this->size());
      if (sizeChange)
      {
        recordDeallocation(fcAllocationRecord,this->size());
      }
      this->FieldContainerBase::resize(anotherContainer);
      invalidateViews();
      if (sizeChange)
      {
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Resizes FieldContainer to a container whose rank depends on the specified field and
     operator types and the space dimension, initialized by 0. The admissible combinations of these
     arguments, the rank of the resulitng container and its dimensions are summarized in the following table:
     \verbatim
     |--------------------|-------------------|-------------------|-------------------|
     |operator/field rank |       rank 0      | rank 1 2D/3D      | rank 2 2D/3D      |
     |--------------------|-------------------|-------------------|-------------------|
     |       VALUE        | (P,F)             | (P,F,D)           | (P,F,D,D)         |
     |--------------------|-------------------|-------------------|-------------------|
     |     GRAD, D1       | (P,F,D)           | (P,F,D,D)         | (P,F,D,D,D)       |
     |--------------------|-------------------|-------------------|-------------------|
     |        CURL        | (P,F,D) (undef3D) | (P,F)/(P,F,D)     | (P,F,D)/(P,F,D,D) |
     |--------------------|-------------------|-------------------|-------------------|
     |        DIV         | (P,F,D) (only 1D) | (P,F)             | (P,F,D)           |
     |--------------------|-------------------|-------------------|-------------------|
     |    D1,D2,..,D10    | (P,F,K)           | (P,F,D,K)         | (P,F,D,D,K)       |
     |--------------------|-------------------|-------------------|-------------------|
     
     |------|----------------------|---------------------------|
     |      |         Index        |         Dimension         |
     |------|----------------------|---------------------------|
     |   P  |         point        |  0 <= P < numPoints       |
     |   F  |         field        |  0 <= F < numFields       |
     |   D  |   field coordinate   |  0 <= D < spaceDim        |
     |   K  |   enumeration of Dk  |  0 <= K < DkCardinality   |
     |------|----------------------|---------------------------|
     \endverbatim
     \remarks
     \li Enumeration of Dk (derivatives of total order k) follows the lexicographical order of
     the partial derivatives; see getDkEnumeration() for details.
     
     
     \param numPoints       [in]        - number of evaluation points
     \param numFields       [in]        - number of fields that will be evaluated
     \param spaceType       [in]        - type of the function space whose basis will be evaluated
     \param operatorType    [in]        - type of the operator that will be applied to the basis
     \param spaceDim        [in]        - dimension of the ambient space
     */
    void resize(const int                       numPoints,
                const int                       numFields,
                const Intrepid::EFunctionSpace  spaceType,
                const Intrepid::EOperator       operatorType,
                const int                       spaceDim)
    {
      int oldSize = this->size();
      this->FieldContainerBase::resize(numPoints,numFields,spaceType,operatorType,spaceDim);
      invalidateViews();
      if (oldSize != this->size())
      {
        recordDeallocation(fcAllocationRecord,oldSize);
        recordAllocation(fcAllocationRecord,this->size());
      }
    }
    
    /** \brief Assignment operator <var>*this = right</var>.
     */
    FieldContainer<Scalar>& operator  = (const FieldContainer<Scalar>& right)
    {
      TEUCHOS_TEST_FOR_EXCEPTION( ( this == &right ),
                                 std::invalid_argument,
                                 ">>> ERROR (FieldContainer): Invalid right-hand side to '='. Self-assignment prohibited.");
      recordDeallocation(fcAllocationRecord,this->size());
      
      this->FieldContainerBase::operator=(right);
      // base assignment does deep copy; record the allocation
      recordAllocation(fcAllocationRecord,this->size());
      return *this;
    }

    static DeviceView getDeviceView(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      DeviceView deviceView = Kokkos::create_mirror(DeviceMemorySpace(), dv);
      Camellia::recordAllocation<Scalar>(deviceViewAllocationRecord, deviceView.size());
      return deviceView;
    }

    static void recordDeviceViewDeallocation(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      Camellia::recordDeallocation<Scalar>(deviceViewAllocationRecord, dv.size());
    }

    static Kokkos::MDRangePolicy< Kokkos::Rank<2> > getMDRangePolicy2(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
      Kokkos::Array<array_index_type,2> mdRangeBegin = {0,0};
      Kokkos::Array<array_index_type,2> mdRangeEnd   = {1,1};
      int numRanks = std::min((int) dv.rank(), 2);
      for (int i=0; i<numRanks; i++)
      {
        mdRangeEnd[i] = dv.extent_int(i);
      }
      return Kokkos::MDRangePolicy< Kokkos::Rank<2> >(mdRangeBegin,mdRangeEnd);
    }

    static Kokkos::MDRangePolicy< Kokkos::Rank<6> > getMDRangePolicy6(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
      Kokkos::Array<array_index_type,6> mdRangeBegin = {0,0,0,0,0,0};
      Kokkos::Array<array_index_type,6> mdRangeEnd   = {1,1,1,1,1,1};
      int numRanks = std::min((int) dv.rank(), 6);
      for (int i=0; i<numRanks; i++)
      {
        mdRangeEnd[i] = dv.extent_int(i);
      }
      return Kokkos::MDRangePolicy< Kokkos::Rank<6> >(mdRangeBegin,mdRangeEnd);
    }

    Kokkos::MDRangePolicy< Kokkos::Rank<2> > getMDRangePolicy2() {
      TEUCHOS_TEST_FOR_EXCEPTION( !(this->_deviceViewIsCheckedOut), std::invalid_argument,
                                  "getMDRangePolicy _deviceView is not checked out." );
      return getMDRangePolicy2(this->_deviceView);
    }

    Kokkos::MDRangePolicy< Kokkos::Rank<6> > getMDRangePolicy6() {
      TEUCHOS_TEST_FOR_EXCEPTION( !(this->_deviceViewIsCheckedOut), std::invalid_argument,
                                  "getMDRangePolicy _deviceView is not checked out." );
      return getMDRangePolicy6(this->_deviceView);
    }

    /** \brief Print metadata info for member variable _hostView
     */
    void printHostViewMetadata(std::ostream &out = std::cout) {
      out << "  _hostView.rank()    = " <<   _hostView.rank() << std::endl;
      out << "  _hostView.data()    = " <<   _hostView.data() << std::endl;
      for (int i = 0; i < _hostView.rank(); i++) {
        out << "  _hostView.extent(" << std::setw(1) << i << ") = " << _hostView.extent(i) << std::endl;
      }
      out << "  _hostView.size()    = " <<   _hostView.size() << std::endl;
      out << "typeid(_hostView  )   = " << typeid(_hostView  ).name() << std::endl;
    }
    
    /** \brief Print metadata info for member variable _deviceView
    */
    void printDeviceViewMetadata(std::ostream &out = std::cout) {
      out << "_deviceView.rank()    = " << _deviceView.rank() << std::endl;
      out << "_deviceView.data()    = " << _deviceView.data() << std::endl;
      for (int i = 0; i < _hostView.rank(); i++) {
        out << "_deviceView.extent(" << std::setw(1) << i << ") = " << _deviceView.extent(i) << std::endl;
      }
      out << "_deviceView.size()    = " << _deviceView.size() << std::endl;
      out << "typeid(_deviceView)   = " << typeid(_deviceView).name() << std::endl;
    }
  };
  
  std::string allocationReport(AllocationRecord &record);
}

#endif // Camellia_FieldContainer_hpp
