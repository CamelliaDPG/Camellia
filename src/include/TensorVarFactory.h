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
   *
   *  Tensor trace variables are surface-family classified.  A logical flux trace,
   *  such as the Vlasov t_n_hat trace, should be expanded during setup into one
   *  TensorVar per orthogonal tensor skeleton family (for example tn_x, tn_t,
   *  tn_ux, tn_uy).  The directional helper below performs that setup-only
   *  expansion and assigns traceComponentOrdinal values; kernels should then see
   *  ordinary single-family TensorVar instances rather than runtime-filtering one
   *  trace variable over all skeleton directions.
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

    /**
     * \brief Create unregistered directional trace variables for one logical trace family.
     *
     * componentLabels names the tensor components/skeleton families.  The returned
     * trace variables are named baseName + "_" + componentLabels[d], have
     * traceComponentOrdinal d, and copy the supplied component Var descriptors so
     * that all directional variables share the same per-component legacy semantics
     * without sharing mutable Var instances.
     */
    static std::vector<TensorVarPtr> directionalTraceFamily(const std::string &baseName,
                                                            const std::vector<std::string> &componentLabels,
                                                            const std::vector<VarPtr> &componentVars);

    /**
     * \brief Create and register directional trace variables for one logical trace family.
     */
    std::vector<TensorVarPtr> addDirectionalTraceFamily(const std::string &baseName,
                                                        const std::vector<std::string> &componentLabels,
                                                        const std::vector<VarPtr> &componentVars);

    static TensorVarFactoryPtr factory()
    {
      return Teuchos::rcp(new TensorVarFactory);
    }
  };
}

#endif
