#ifndef Camellia_TensorOperatorFactory_h
#define Camellia_TensorOperatorFactory_h

#include "TensorOperator.h"

namespace Camellia
{
  class TensorOperatorFactory;
  using TensorOperatorFactoryPtr = Teuchos::RCP<TensorOperatorFactory>;

  /**
   *  \brief Abstract factory interface for realizing tensor-native operators from descriptor objects.
   */
  class TensorOperatorFactory
  {
  public:
    virtual ~TensorOperatorFactory() {}

    virtual TensorOperatorBackend backend() const = 0;

    virtual bool canBuild(const TensorMeshPtr &tensorMesh,
                          const TensorDofLayoutPtr &tensorDofLayout,
                          const TensorBilinearFormPtr &bilinearForm,
                          TensorSpaceRole domainRole,
                          const std::string &domainVarName,
                          TensorSpaceRole rangeRole,
                          const std::string &rangeVarName) const = 0;

    virtual TensorOperatorPtr build(const TensorMeshPtr &tensorMesh,
                                    const TensorDofLayoutPtr &tensorDofLayout,
                                    const TensorBilinearFormPtr &bilinearForm,
                                    TensorSpaceRole domainRole,
                                    const std::string &domainVarName,
                                    TensorSpaceRole rangeRole,
                                    const std::string &rangeVarName) const = 0;
  };

  class MatrixFreeTensorOperatorFactory;
  using MatrixFreeTensorOperatorFactoryPtr = Teuchos::RCP<MatrixFreeTensorOperatorFactory>;

  /**
   *  \brief First concrete tensor-native operator factory.
   *
   *  This factory is intentionally conservative: it accepts the descriptor subset that the current
   *  issue #298 implementation can realize coherently through explicit local gather/apply/scatter.
   */
  class MatrixFreeTensorOperatorFactory : public TensorOperatorFactory
  {
  public:
    TensorOperatorBackend backend() const override;

    bool canBuild(const TensorMeshPtr &tensorMesh,
                  const TensorDofLayoutPtr &tensorDofLayout,
                  const TensorBilinearFormPtr &bilinearForm,
                  TensorSpaceRole domainRole,
                  const std::string &domainVarName,
                  TensorSpaceRole rangeRole,
                  const std::string &rangeVarName) const override;

    TensorOperatorPtr build(const TensorMeshPtr &tensorMesh,
                            const TensorDofLayoutPtr &tensorDofLayout,
                            const TensorBilinearFormPtr &bilinearForm,
                            TensorSpaceRole domainRole,
                            const std::string &domainVarName,
                            TensorSpaceRole rangeRole,
                            const std::string &rangeVarName) const override;
  };
}

#endif
