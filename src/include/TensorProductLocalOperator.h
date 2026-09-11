#ifndef Camellia_TensorProductLocalOperator_h
#define Camellia_TensorProductLocalOperator_h

#include <vector>

#include "TensorDofLayout.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
   * \brief Dense row-major one-dimensional factor matrix used by tensor-product local operators.
   */
  struct TensorComponentLocalMatrix
  {
    int _rowCount = 0;
    int _columnCount = 0;
    std::vector<double> _entries;

    TensorComponentLocalMatrix() = default;
    TensorComponentLocalMatrix(int rowCount,
                               int columnCount,
                               const std::vector<double> &entries);

    int rowCount() const { return _rowCount; }
    int columnCount() const { return _columnCount; }
    const std::vector<double> &entries() const { return _entries; }
    double entry(int rowOrdinal,
                 int columnOrdinal) const;
  };

  /**
   * \brief Reusable tensor-product local matrix/action object.
   *
   * The object represents a Kronecker-style product of one-dimensional component
   * matrices without materializing the full local dense matrix.  It is intentionally
   * small and host-callable for the current Phase 2 correctness path; later kernels
   * can consume the same factor data from device views.
   */
  class TensorProductLocalOperator
  {
    TensorFactorSpaceShape _domainShape;
    TensorFactorSpaceShape _rangeShape;
    std::vector<TensorComponentLocalMatrix> _componentMatrices;

    void validate() const;
    static void decodeOrdinal(const TensorFactorSpaceShape &shape,
                              IndexType ordinal,
                              std::vector<int> &componentOrdinals);

  public:
    TensorProductLocalOperator() = default;
    TensorProductLocalOperator(const TensorFactorSpaceShape &domainShape,
                               const TensorFactorSpaceShape &rangeShape,
                               const std::vector<TensorComponentLocalMatrix> &componentMatrices);

    int componentCount() const;
    IndexType domainSize() const;
    IndexType rangeSize() const;
    const TensorFactorSpaceShape &domainShape() const;
    const TensorFactorSpaceShape &rangeShape() const;
    const TensorComponentLocalMatrix &componentMatrix(int componentOrdinal) const;

    double entry(IndexType rangeOrdinal,
                 IndexType domainOrdinal) const;
    void apply(const std::vector<double> &input,
               std::vector<double> &output) const;
  };
}

#endif
