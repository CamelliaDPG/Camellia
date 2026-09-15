#ifndef Camellia_TensorTraceLayout_h
#define Camellia_TensorTraceLayout_h

#include <stdexcept>
#include <utility>

#include "TensorTraceDescriptor.h"

namespace Camellia
{
  enum TensorTraceInterfaceSide
  {
    TENSOR_TRACE_INTERFACE_LOWER = -1,
    TENSOR_TRACE_INTERFACE_UPPER = 1
  };

  class TensorH1PointTraceLayout1D
  {
    TensorTraceDescriptor _descriptor;
    int _interfacePointCount = 0;

  public:
    TensorH1PointTraceLayout1D(const TensorTraceDescriptor &descriptor,
                               int interfacePointCount)
    : _descriptor(descriptor), _interfacePointCount(interfacePointCount)
    {
      if (!_descriptor.isH1PointValueTrace()) throw std::invalid_argument("TensorH1PointTraceLayout1D requires an H1 point-value trace descriptor.");
      if (_descriptor.usesOutwardNormalSign()) throw std::invalid_argument("TensorH1PointTraceLayout1D point-value traces must not carry outward-normal signs.");
      if (_interfacePointCount < 0) throw std::invalid_argument("TensorH1PointTraceLayout1D requires a nonnegative interface-point count.");
    }

    const TensorTraceDescriptor &descriptor() const { return _descriptor; }
    int interfacePointCount() const { return _interfacePointCount; }
    int valueDofOrdinal(int interfacePointOrdinal) const
    {
      if ((interfacePointOrdinal < 0) || (interfacePointOrdinal >= _interfacePointCount)) throw std::invalid_argument("TensorH1PointTraceLayout1D interface point ordinal out of range.");
      return interfacePointOrdinal;
    }

    int endpointValueDofOrdinal(int lowerInterfacePointOrdinal,
                                int upperInterfacePointOrdinal,
                                TensorTraceInterfaceSide side) const
    {
      const int interfacePointOrdinal = (side == TENSOR_TRACE_INTERFACE_LOWER) ? lowerInterfacePointOrdinal : upperInterfacePointOrdinal;
      return valueDofOrdinal(interfacePointOrdinal);
    }

    double referencePointForSide(TensorTraceInterfaceSide side) const
    {
      return side == TENSOR_TRACE_INTERFACE_LOWER ? -1.0 : 1.0;
    }

    double adaptEndpointValue(double lowerEndpointValue,
                              double upperEndpointValue,
                              TensorTraceInterfaceSide side) const
    {
      return side == TENSOR_TRACE_INTERFACE_LOWER ? lowerEndpointValue : upperEndpointValue;
    }

    std::pair<int,int> adjacentCellEndpointDofOrdinals(int sharedInterfacePointOrdinal) const
    {
      const int lowerCellUpperEndpointDofOrdinal = endpointValueDofOrdinal(sharedInterfacePointOrdinal,
                                                                           sharedInterfacePointOrdinal,
                                                                           TENSOR_TRACE_INTERFACE_UPPER);
      const int upperCellLowerEndpointDofOrdinal = endpointValueDofOrdinal(sharedInterfacePointOrdinal,
                                                                           sharedInterfacePointOrdinal,
                                                                           TENSOR_TRACE_INTERFACE_LOWER);
      return std::make_pair(lowerCellUpperEndpointDofOrdinal, upperCellLowerEndpointDofOrdinal);
    }

    bool adjacentCellEndpointsShareDof(int sharedInterfacePointOrdinal) const
    {
      const auto endpointDofs = adjacentCellEndpointDofOrdinals(sharedInterfacePointOrdinal);
      return endpointDofs.first == endpointDofs.second;
    }
  };
}

#endif
