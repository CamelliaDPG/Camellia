#ifndef Camellia_TensorCoefficientTraversal_h
#define Camellia_TensorCoefficientTraversal_h

#include <string>
#include <vector>

#include "TensorDofLayout.h"
#include "TensorMeshTopology.h"

namespace Camellia
{
  enum TensorTraversalKind
  {
    TENSOR_VOLUME_TRAVERSAL,
    TENSOR_SKELETON_TRAVERSAL
  };

  struct TensorCoefficientBlock
  {
    IndexType _base = 0;
    IndexType _stride = 1;
    IndexType _count = 0;

    TensorCoefficientBlock() = default;
    TensorCoefficientBlock(IndexType base, IndexType stride, IndexType count)
    : _base(base), _stride(stride), _count(count) {}

    IndexType base() const { return _base; }
    IndexType stride() const { return _stride; }
    IndexType count() const { return _count; }
    bool isContiguous() const { return _stride == 1; }
  };

  class TensorCoefficientTraversal
  {
    TensorTraversalKind _kind;
    TensorSpaceRole _role;
    std::string _varName;
    TensorLocalOrdering _ordering;

    TensorDofLayout::VariableLayout _variableLayout;
    TensorFactorSpaceShape _shape;

    std::vector<IndexType> _componentStrides;
    std::vector<IndexType> _componentFactorBaseCoordinates;

    bool _isTraceTraversal = false;
    int _traceComponentOrdinal = -1;
    IndexType _traceSupportCoordinate = -1;

  public:
    TensorCoefficientTraversal(TensorTraversalKind kind,
                               TensorSpaceRole role,
                               const std::string &varName,
                               TensorLocalOrdering ordering,
                               const TensorDofLayout::VariableLayout &variableLayout,
                               const TensorFactorSpaceShape &shape,
                               const std::vector<IndexType> &componentStrides,
                               const std::vector<IndexType> &componentFactorBaseCoordinates,
                               bool isTraceTraversal = false,
                               int traceComponentOrdinal = -1,
                               IndexType traceSupportCoordinate = -1);

    TensorTraversalKind kind() const;
    TensorSpaceRole role() const;
    const std::string &varName() const;
    TensorLocalOrdering ordering() const;
    const TensorDofLayout::VariableLayout &variableLayout() const;
    const TensorFactorSpaceShape &shape() const;

    bool isTraceTraversal() const;
    int traceComponentOrdinal() const;
    IndexType traceSupportCoordinate() const;

    IndexType localDofCount() const;

    IndexType globalIndexForLocalOrdinal(IndexType localOrdinal,
                                         const TensorDofLayoutPtr &tensorDofLayout) const;

    std::vector<IndexType> globalIndices(const TensorDofLayoutPtr &tensorDofLayout) const;

    bool hasFastestComponentContiguousRuns() const;
    std::vector<TensorCoefficientBlock> fastestComponentBlocks(const TensorDofLayoutPtr &tensorDofLayout) const;
    bool isContiguous() const;
    std::vector<TensorCoefficientBlock> contiguousBlocks(const TensorDofLayoutPtr &tensorDofLayout) const;
  };
}

#endif
