#ifndef Camellia_TensorVarDiscretization_h
#define Camellia_TensorVarDiscretization_h

#include <vector>

#include "BasisFamily.h"
#include "TensorVar.h"

namespace Camellia
{
  class TensorVarDiscretization;
  using TensorVarDiscretizationPtr = Teuchos::RCP<TensorVarDiscretization>;

  /**
   *  \brief Per-variable, per-component discretization metadata for the tensor-native path.
   *
   *  This object stores the discretization choices attached to a TensorVar, rather than
   *  topology state. In the current implementation, it carries an optional basis-family
   *  override together with one polynomial-order delta per tensor component.
   */
  class TensorVarDiscretization
  {
    TensorVarPtr _tensorVar;
    BasisFamilyPtr _basisFamilyOverride;
    std::vector<int> _componentPolyOrderDeltas;

    void validateConstructorInputs() const;
  public:
    TensorVarDiscretization(TensorVarPtr tensorVar,
                            BasisFamilyPtr basisFamilyOverride,
                            const std::vector<int> &componentPolyOrderDeltas);

    TensorVarPtr tensorVar() const;
    BasisFamilyPtr basisFamilyOverride() const;
    bool hasBasisFamilyOverride() const;

    int componentCount() const;
    int componentPolyOrderDelta(int componentOrdinal) const;
    const std::vector<int> &componentPolyOrderDeltas() const;
  };

  inline std::ostream &operator<<(std::ostream &os, const TensorVarDiscretization &discretization)
  {
    os << "TensorVarDiscretization(var=" << discretization.tensorVar()->name() << ",components=" << discretization.componentCount() << ")";
    return os;
  }
}

#endif
