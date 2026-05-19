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

  enum TensorVarKind
  {
    TENSOR_FIELD,
    TENSOR_TRACE
  };

  class TensorVar
  {
    std::string _name;
    TensorVarKind _kind;
    std::vector<VarPtr> _componentVars;

    void validateConstructorInputs() const;
  public:
    TensorVar(const std::string &name, TensorVarKind kind, const std::vector<VarPtr> &componentVars);

    const std::string &name() const;
    TensorVarKind kind() const;
    bool isField() const;
    bool isTrace() const;

    int componentCount() const;
    VarPtr componentVar(int componentOrdinal) const;
    const std::vector<VarPtr> &componentVars() const;

    TensorVarPtr copyWithNewName(const std::string &newName) const;
  };

  inline std::ostream &operator<<(std::ostream &os, const TensorVar &tensorVar)
  {
    os << "TensorVar(name=" << tensorVar.name() << ",kind=";
    os << (tensorVar.kind() == TENSOR_FIELD ? "FIELD" : "TRACE");
    os << ",components=" << tensorVar.componentCount() << ")";
    return os;
  }
}

#endif
