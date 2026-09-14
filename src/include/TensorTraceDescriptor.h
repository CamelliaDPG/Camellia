#ifndef Camellia_TensorTraceDescriptor_h
#define Camellia_TensorTraceDescriptor_h

#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
  enum TensorTraceContinuity
  {
    TENSOR_TRACE_NONCONFORMING_L2,
    TENSOR_TRACE_H1_POINT_VALUE
  };

  enum TensorTraceSupportKind
  {
    TENSOR_TRACE_SUPPORT_COMPONENT_SKELETON,
    TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS
  };

  /**
   * \brief Metadata-only description of a tensor-native skeleton/trace unknown.
   *
   * The descriptor records the trace family and target unknown without implementing a
   * boundary kernel or volume-to-trace constraint.  It is meant to keep Phase 7 trace
   * semantics explicit while the tensor-native adapters are brought up one family at a
   * time.  In particular, the Maxwell/Poisson 1D H1 traces are point-value interface
   * unknowns distinct from the corresponding volume fields.
   */
  class TensorTraceDescriptor
  {
    std::string _fieldName;
    std::string _traceName;
    TensorTraceContinuity _continuity = TENSOR_TRACE_NONCONFORMING_L2;
    TensorTraceSupportKind _supportKind = TENSOR_TRACE_SUPPORT_COMPONENT_SKELETON;
    int _orthogonalComponentOrdinal = -1;
    bool _usesOutwardNormalSign = false;

    void validate() const
    {
      if (_fieldName.empty()) throw std::invalid_argument("TensorTraceDescriptor requires a non-empty field name.");
      if (_traceName.empty()) throw std::invalid_argument("TensorTraceDescriptor requires a non-empty trace name.");
      if (_orthogonalComponentOrdinal < 0) throw std::invalid_argument("TensorTraceDescriptor requires a nonnegative orthogonal component ordinal.");
      if ((_continuity == TENSOR_TRACE_H1_POINT_VALUE) && _usesOutwardNormalSign)
      {
        throw std::invalid_argument("TensorTraceDescriptor H1 point-value traces should not carry an outward-normal sign.");
      }
    }

  public:
    TensorTraceDescriptor() = default;

    TensorTraceDescriptor(const std::string &fieldName,
                          const std::string &traceName,
                          TensorTraceContinuity continuity,
                          TensorTraceSupportKind supportKind,
                          int orthogonalComponentOrdinal,
                          bool usesOutwardNormalSign)
    : _fieldName(fieldName),
      _traceName(traceName),
      _continuity(continuity),
      _supportKind(supportKind),
      _orthogonalComponentOrdinal(orthogonalComponentOrdinal),
      _usesOutwardNormalSign(usesOutwardNormalSign)
    {
      validate();
    }

    const std::string &fieldName() const { return _fieldName; }
    const std::string &traceName() const { return _traceName; }
    TensorTraceContinuity continuity() const { return _continuity; }
    TensorTraceSupportKind supportKind() const { return _supportKind; }
    int orthogonalComponentOrdinal() const { return _orthogonalComponentOrdinal; }
    bool usesOutwardNormalSign() const { return _usesOutwardNormalSign; }
    bool isH1PointValueTrace() const { return _continuity == TENSOR_TRACE_H1_POINT_VALUE; }
    bool isNonconformingL2Trace() const { return _continuity == TENSOR_TRACE_NONCONFORMING_L2; }

    static TensorTraceDescriptor nonconformingL2SkeletonTrace(const std::string &fieldName,
                                                              const std::string &traceName,
                                                              int orthogonalComponentOrdinal,
                                                              bool usesOutwardNormalSign = true)
    {
      return TensorTraceDescriptor(fieldName,
                                   traceName,
                                   TENSOR_TRACE_NONCONFORMING_L2,
                                   TENSOR_TRACE_SUPPORT_COMPONENT_SKELETON,
                                   orthogonalComponentOrdinal,
                                   usesOutwardNormalSign);
    }

    static TensorTraceDescriptor h1PointValueTrace1D(const std::string &fieldName,
                                                     const std::string &traceName,
                                                     int orthogonalComponentOrdinal = 0)
    {
      return TensorTraceDescriptor(fieldName,
                                   traceName,
                                   TENSOR_TRACE_H1_POINT_VALUE,
                                   TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS,
                                   orthogonalComponentOrdinal,
                                   false);
    }

    static std::vector<TensorTraceDescriptor> maxwell1D2VH1PointValueTraces(const std::string &exFieldName,
                                                                            const std::string &exTraceName,
                                                                            const std::string &byFieldName,
                                                                            const std::string &byTraceName,
                                                                            const std::string &phiFieldName,
                                                                            const std::string &phiTraceName,
                                                                            const std::string &aFieldName,
                                                                            const std::string &aTraceName)
    {
      return {
        h1PointValueTrace1D(exFieldName,  exTraceName),
        h1PointValueTrace1D(byFieldName,  byTraceName),
        h1PointValueTrace1D(phiFieldName, phiTraceName),
        h1PointValueTrace1D(aFieldName,   aTraceName)
      };
    }
  };
}

#endif
