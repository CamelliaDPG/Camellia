#ifndef Camellia_TensorVarFactory_h
#define Camellia_TensorVarFactory_h

#include <map>
#include <string>
#include <vector>

#include "TensorVar.h"

namespace Camellia
{
  class TensorVarFactory;
  using TensorVarFactoryPtr = Teuchos::RCP<TensorVarFactory>;

  /**
   *  \brief Tensor-native registration and lookup object for TensorVar instances.
   *
   *  This class plays a role analogous in spirit to VarFactory, but for tensor-native
   *  variable descriptors. It is intentionally lightweight: registration by name,
   *  lookup by name, and enumeration of the registered TensorVar objects.
   */
  class TensorVarFactory
  {
    std::map<std::string, TensorVarPtr> _tensorVars;
  public:
    TensorVarFactory();

    void addTensorVar(TensorVarPtr tensorVar);
    TensorVarPtr tensorVar(const std::string &name) const;
    bool hasTensorVar(const std::string &name) const;
    std::vector<TensorVarPtr> tensorVars() const;

    static TensorVarFactoryPtr factory()
    {
      return Teuchos::rcp(new TensorVarFactory);
    }
  };
}

#endif
