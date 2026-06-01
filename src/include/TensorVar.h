#ifndef Camellia_TensorVar_h
#define Camellia_TensorVar_h

#include <map>
#include <set>
#include <string>
#include <vector>

#include "TensorMeshTopology.h"
#include "TypeDefs.h"

namespace Camellia
{
  class TensorVar;
  using TensorVarPtr = Teuchos::RCP<TensorVar>;

  /**
   *  \brief High-level tensor-native variable category.
   *
   *  At the tensor-native discretization layer, the primary distinction is between
   *  volumetric variables (fields) and skeleton variables (traces). Flux semantics
   *  are expected to be recovered later in the bilinear-form/operator layer through
   *  trace semantics together with side-parity weighting.
   */
  enum TensorVarKind
  {
    TENSOR_FIELD,
    TENSOR_TRACE
  };

  /**
   *  \brief Tensor-native variable descriptor built as an adaptor around per-component Var semantics.
   *
   *  TensorVar is intentionally neutral with respect to trial versus test usage.
   *  Trial/test distinctions are handled by TensorMesh registration rather than by
   *  the variable object itself. The object stores a common tensor-level name and a
   *  per-component collection of legacy Var descriptors whose semantics are adapted
   *  into the tensor-native layer.
   *
   *  For trace variables, the current tensor-native interpretation is additionally
   *  dimension-specific: a trace variable is associated with one distinguished tensor
   *  component ordinal, and its support is understood to live on the tensor-product
   *  skeleton orthogonal to that component.
   */
  class TensorVar
  {
    std::string _name;
    TensorVarKind _kind;
    std::vector<VarPtr> _componentVars;
    int _traceComponentOrdinal = -1;

    void validateConstructorInputs() const;
  public:
    TensorVar(const std::string &name,
              TensorVarKind kind,
              const std::vector<VarPtr> &componentVars,
              int traceComponentOrdinal = -1);

    const std::string &name() const;
    TensorVarKind kind() const;
    bool isField() const;
    bool isTrace() const;
    bool hasFluxLikeComponentVar() const;
    bool hasTraceLikeComponentVar() const;
    bool isFluxLikeTrace() const;
    bool isValueTrace() const;

    bool hasTraceComponentOrdinal() const;
    int traceComponentOrdinal() const;

    int componentCount() const;
    VarPtr componentVar(int componentOrdinal) const;
    const std::vector<VarPtr> &componentVars() const;

    /**
     *  \brief Return a copy of this TensorVar with a different tensor-level name.
     *
     *  This supports the intended pattern in which trial and test variables may have
     *  matching discretization choices while still remaining semantically distinct by name.
     */
    TensorVarPtr copyWithNewName(const std::string &newName) const;
  };

  inline std::ostream &operator<<(std::ostream &os, const TensorVar &tensorVar)
  {
    os << "TensorVar(name=" << tensorVar.name() << ",kind=";
    os << (tensorVar.kind() == TENSOR_FIELD ? "FIELD" : "TRACE");
    if (tensorVar.hasTraceComponentOrdinal())
    {
      os << ",traceComponent=" << tensorVar.traceComponentOrdinal();
    }
    os << ",components=" << tensorVar.componentCount() << ")";
    return os;
  }
}

#endif
