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

namespace Camellia
{
  template<class Scalar, int ArrayTypeId=0>
  class FieldContainer : public Intrepid::FieldContainer<Scalar, ArrayTypeId>
  {
    using FieldContainerBase = Intrepid::FieldContainer<Scalar, ArrayTypeId>;
    using DeviceView = Kokkos::DynRankView<Scalar>;
    using HostView   = Kokkos::DynRankView<Scalar, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    
    DeviceView  _deviceView;
    HostView    _hostView;
    
    bool _deviceViewIsCheckedOut = false; // it's an error to check out twice without checking back in first; data copies happen at check in
    
    const std::string DEFAULT_VIEW_NAME = "Camellia FieldContainer View";
    
    void initializeViews()
    {
      Teuchos::Array<int> dims;
      this->dimensions(dims);
      const int rank = this->rank();
      auto dataPtr = this->getData().getRawPtr();
      
      switch (rank)
      {
        case 0:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME);
          _hostView   = HostView(dataPtr);
          break;
        case 1:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0]);
          _hostView   = HostView(dataPtr,dims[0]);
          break;
        case 2:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1]);
          _hostView   = HostView(dataPtr,dims[0],dims[1]);
          break;
        case 3:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2]);
          break;
        case 4:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2],dims[3]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3]);
          break;
        case 5:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2],dims[3],dims[4]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4]);
          break;
        case 6:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
          break;
        case 7:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
          break;
        case 8:
          _deviceView = DeviceView(DEFAULT_VIEW_NAME,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],dims[7]);
          _hostView   = HostView(dataPtr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],dims[7]);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
      }
    }
  public:
    DeviceView checkoutDeviceView()
    {
      _deviceViewIsCheckedOut = true;
      // ideally, we should guard this so that we don't do unnecessary copies when memory spaces are the same
      // but this is an optimization that we can make at any point
      // the priority right now is to provide the same interface as Intrepid::FieldContainer, so that we have a
      // drop-in replacement, and to allow us to gradually shift to using Kokkos::View instead.
      // the checkout semantics here allow us at least some guard against accidental usage of UVM.
      Kokkos::deep_copy(_deviceView,_hostView);
      return _deviceView;
    }
    
    void checkinDeviceView()
    {
      // copy back to host
      Kokkos::deep_copy(_hostView,_deviceView);
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
      initializeViews();
    }
    
    /** \brief Creates a rank-1 FieldContainer with the specified dimension, initialized by 0.
     
     \param dim0    [in]      - dimension for the only index
     */
    FieldContainer(const int dim0)
    :
    FieldContainerBase(dim0)
    {
      initializeViews();
    }
    
    /** \brief Creates a rank-2 FieldContainer with the specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     */
    FieldContainer(const int dim0, const int dim1)
    :
    FieldContainerBase(dim0, dim1)
    {
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
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
      initializeViews();
    }
    
    /** \brief Clears FieldContainer to trivial container (one with rank = 0 and size = 0)
     */
    void clear()
    {
      this->FieldContainerBase::clear();
      initializeViews();
    }
    
    
    /** \brief Resizes FieldContainer to a rank-1 container with the specified dimension, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     */
    void resize(const int dim0)
    {
      this->FieldContainerBase::resize(dim0);
      initializeViews();
    }
    
    /** \brief Resizes FieldContainer to a rank-2 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     */
    void resize(const int dim0, const int dim1)
    {
      this->FieldContainerBase::resize(dim0, dim1);
      initializeViews();
    }
    
    /** \brief Resizes FieldContainer to a rank-3 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     */
    void resize(const int dim0, const int dim1, const int dim2)
    {
      this->FieldContainerBase::resize(dim0, dim1, dim2);
      initializeViews();
    }
    
    
    /** \brief Resizes FieldContainer to a rank-4 container with specified dimensions, initialized by 0.
     
     \param dim0    [in]      - dimension for the 1st index
     \param dim1    [in]      - dimension for the 2nd index
     \param dim2    [in]      - dimension for the 3rd index
     \param dim3    [in]      - dimension for the 4th index
     */
    void resize(const int dim0, const int dim1, const int dim2, const int dim3)
    {
      this->FieldContainerBase::resize(dim0, dim1, dim2, dim3);
      initializeViews();
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
      this->FieldContainerBase::resize(dim0, dim1, dim2, dim3, dim4);
      initializeViews();
    }
    
    /** \brief Resizes FieldContainer to arbitrary rank container, initialized by 0, with dimensions
     specified in the input array. The size of this array implicitely defined the rank of the FieldContainer.
     
     \param newDimensions[in]          - new upper values for index ranges
     */
    void resize(const Teuchos::Array<int>& newDimensions)
    {
      this->FieldContainerBase::resize(newDimensions);
      initializeViews();
    }
    
    
    /** \brief Resizes FieldContainer to have the same rank and dimensions as another FieldContainer,
     and initializes by 0.
     
     \param anotherContainer[in]          - a FieldContainer
     */
    void resize(const FieldContainer<Scalar, ArrayTypeId>& anotherContainer)
    {
      this->FieldContainerBase::resize(anotherContainer);
      initializeViews();
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
      this->FieldContainerBase::resize(numPoints,numFields,spaceType,operatorType,spaceDim);
      initializeViews();
    }
  };
}

#endif // Camellia_FieldContainer_hpp
