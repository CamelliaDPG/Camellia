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
