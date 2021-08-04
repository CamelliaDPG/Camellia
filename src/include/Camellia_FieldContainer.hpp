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

#include "Camellia_KokkosAllocations.h"
#include "SharedScratchStorage.h"
#include "TypeDefs.h"

namespace Camellia
{
  extern const std::string DEFAULT_VIEW_NAME;
  
  template<class Scalar, int ArrayTypeId=0>
  class FieldContainer : public Intrepid::FieldContainer<Scalar, ArrayTypeId>
  {
    
  public:
    using FieldContainerBase = Intrepid::FieldContainer<Scalar, ArrayTypeId>;
    using DeviceMemorySpace = Camellia::DeviceMemorySpace;
    using HostMemorySpace   = Camellia::HostMemorySpace;
    using HostView          = Camellia::HostView<Scalar>;
    using ViewTraits        = Camellia::ViewTraits<Scalar>;
    using DeviceView        = Camellia::DeviceView<Scalar>;
    using DeviceViewConst   = Camellia::DeviceViewConst<Scalar>;
    
  private:
    mutable DeviceView      _deviceView;
    mutable HostView        _hostView;
    
    mutable bool _deviceViewIsCheckedOut = false; // it's an error to check out twice without checking back in first; data copies happen at check in
    mutable bool _deviceViewIsValid      = false; // if invalid, call initializeDeviceView() on checkout
    mutable bool _hostViewIsValid        = false; // if invalid, call initializeHostView() before access to _hostView -- right now only used for creating mirror in initializeDeviceView()
    
    mutable bool _fieldContainerDataAllocationRecorded = false; // we don't count deallocation if we didn't count allocation; this allows us to enforce that.
    mutable int  _fieldContainerAllocationNumber = -1;
    
    DeviceView allocateDeviceView(const std::string &label) const
    {
      const int rank = this->rank();
      std::vector<int> dims(rank); // maximum rank of DynRankView is 7
      for (int d=0; d<rank; d++)
      {
        dims[d] = this->dimension(d);
      }
      return Allocation<Scalar>::allocateDeviceView(dims,label);
    }
    
    void initializeHostView(Scalar* dataPtr)
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
    
    void initializeHostView()
    {
      Scalar* dataPtr = this->getData().getRawPtr();
      initializeHostView(dataPtr);
    }
    
    void initializeDeviceView()
    {
      if (!_hostViewIsValid)
      {
        initializeHostView();
      }
      
      // if _deviceView has previously been checked out, we should check it back in. (This comes up during calls to resize() and clear().)
      Allocation<Scalar>::checkinMirrorView(_deviceView);
      std::ostringstream mirrorViewStringStream;
      mirrorViewStringStream << "FieldContainer::_deviceView (mirrors _hostView for FC with _fieldContainerAllocationNumber " << _fieldContainerAllocationNumber << ")";
      _deviceView = Allocation<Scalar>::checkoutMirrorView(_hostView, mirrorViewStringStream.str());
      
      _deviceViewIsValid = true;
    }
    
    void initializeDeviceView() const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!_hostViewIsValid, std::invalid_argument, "call to initializeDeviceView() for const FC requires that host view be valid");
      
      // if _deviceView has previously been checked out, we should check it back in. (This comes up during calls to resize() and clear().)
      Allocation<Scalar>::checkinMirrorView(_deviceView);
      _deviceView = Allocation<Scalar>::checkoutMirrorView(_hostView, "FieldContainer::_deviceView (mirrors _hostView)");
      
      _deviceViewIsValid = true;
    }
    
    void invalidateViews()
    {
      // called when the FC is resized, e.g.
      _hostViewIsValid = false;
      
//      {
//        // DEBUGGING:
//        if (_fieldContainerAllocationNumber == 485)
//        {
//          std::cout << "Invalidating views for _fieldContainerAllocationNumber " << _fieldContainerAllocationNumber << std::endl;
//        }
//      }
      
      // record the deallocation
      if (_deviceViewIsValid)
      {
        _deviceViewIsValid = false;
        Allocation<Scalar>::checkinMirrorView(_deviceView);
      }

      _deviceView      = DeviceView();
      _hostView        = HostView();
    }
    
    void recordAllocation(AllocationRecord &record, long long entryCount) const
    {
      Camellia::recordAllocation<Scalar>(record, entryCount);
      
      bool recordIsForFieldContainer = (&record == &fcAllocationRecord);
      if (recordIsForFieldContainer)
      {
        _fieldContainerAllocationNumber = record.allocationCount;
//        {
//          // DEBUGGING:
//          if (_fieldContainerAllocationNumber == 485)
//          {
//            std::cout << "Recording allocation for _fieldContainerAllocationNumber " << _fieldContainerAllocationNumber << std::endl;
//          }
//        }
        
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
        initializeDeviceView();
        Kokkos::deep_copy(_deviceView,_hostView);
        recordDeepCopy<Scalar>(hostToDeviceDeepCopyRecord, _hostView.size());
      }
      _deviceViewIsCheckedOut = true;
      
      return _deviceView;
    }
    
    void checkinDeviceView(bool deepCopyToHost) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!_deviceViewIsCheckedOut, std::invalid_argument, "this check-in request is for a device view that is not currently checked out");
      // copy back to host
      //std::cout << "deep_copy device2host..." << std::endl;
      if (deepCopyToHost)
      {
        Kokkos::deep_copy(_hostView,_deviceView);
        recordDeepCopy<Scalar>(deviceToHostDeepCopyRecord, _hostView.size());
      }
      _deviceViewIsCheckedOut = false;
    }
    
    /** \brief Default constructor.
     */
    FieldContainer()
    :
    FieldContainerBase()
    {
      initializeHostView();
    }
    
    
    /** \brief Copy constructor with Camellia::FieldContainer as source.
     */
    FieldContainer(const FieldContainer& right)
    :
    FieldContainerBase(right)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
      initializeHostView();
    }
    
    /** \brief Copy constructor.
     */
    FieldContainer(const FieldContainerBase& right)
    :
    FieldContainerBase(right)
    {
      recordAllocation(fcAllocationRecord,this->size());
      invalidateViews();
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
    }
    
    ~FieldContainer()
    {
      bool ownsMemory = this->getData().has_ownership();
      
//      if (_fieldContainerAllocationNumber == 485)
//      {
//        std::cout << "Deallocating _fieldContainerAllocationNumber " << _fieldContainerAllocationNumber << std::endl;
//      }
      
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      initializeHostView();
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
      this->invalidateViews();
      this->initializeHostView();
      
      // base assignment above does deep copy; record the allocation
      recordAllocation(fcAllocationRecord,this->size());
      
      return *this;
    }

    static DeviceView getDeviceView(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      DeviceView deviceView = Kokkos::create_mirror(DeviceMemorySpace(), dv);
      Camellia::recordAllocation<Scalar>(deviceViewAllocationRecord, deviceView.size());
      return deviceView;
    }
    
    template<int rank>
    static Kokkos::MDRangePolicy< Kokkos::Rank<rank> > getMDRangePolicy(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<rank> >::array_index_type;
      Kokkos::Array<array_index_type,rank> mdRangeBegin;
      Kokkos::Array<array_index_type,rank> mdRangeEnd;
      for (int i=0; i<rank; i++)
      {
        // set default values
        mdRangeBegin[i] = 0;
        mdRangeEnd[i]   = 1;
      }
      
      int numRanks = std::min((int) dv.rank(), rank);
      for (int i=0; i<numRanks; i++)
      {
        mdRangeEnd[i] = dv.extent_int(i);
      }
      return Kokkos::MDRangePolicy< Kokkos::Rank<rank> >(mdRangeBegin,mdRangeEnd);
    }
    
    static Kokkos::MDRangePolicy< Kokkos::Rank<2> > getMDRangePolicy2(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      return getMDRangePolicy<2>(dv);
    }
    
    static Kokkos::MDRangePolicy< Kokkos::Rank<3> > getMDRangePolicy3(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      return getMDRangePolicy<3>(dv);
    }
    
    static Kokkos::MDRangePolicy< Kokkos::Rank<4> > getMDRangePolicy4(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      return getMDRangePolicy<4>(dv);
    }
    
    static Kokkos::MDRangePolicy< Kokkos::Rank<5> > getMDRangePolicy5(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      return getMDRangePolicy<5>(dv);
    }

    static Kokkos::MDRangePolicy< Kokkos::Rank<6> > getMDRangePolicy6(typename Camellia::FieldContainer<Scalar>::DeviceView &dv) {
      return getMDRangePolicy<6>(dv);
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
  
  // Trilinos typedefs
  typedef Teuchos::RCP< Camellia::FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const Camellia::FieldContainer<double> > constFCPtr;
}

#endif // Camellia_FieldContainer_hpp
