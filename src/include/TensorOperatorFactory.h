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

  class MatrixFreeGraphNormSchurTensorOperatorFactory;
  using MatrixFreeGraphNormSchurTensorOperatorFactoryPtr = Teuchos::RCP<MatrixFreeGraphNormSchurTensorOperatorFactory>;

  /**
   *  \brief Matrix-free facade for CRS-free local graph-norm Schur action.
   *
   *  The scalar overload implements a single trial variable and one eliminated volume test
   *  variable.  The vector overloads support concatenated trial variables and can optionally
   *  include volume/boundary trace terms while preserving explicit trial-to-trial Schur metadata.
   */
  class MatrixFreeGraphNormSchurTensorOperatorFactory : public TensorOperatorFactory
  {
    double _l2Weight;

  public:
    explicit MatrixFreeGraphNormSchurTensorOperatorFactory(double l2Weight = 1.0);

    TensorOperatorBackend backend() const override;

    bool canBuild(const TensorMeshPtr &tensorMesh,
                  const TensorDofLayoutPtr &tensorDofLayout,
                  const TensorBilinearFormPtr &bilinearForm,
                  TensorSpaceRole domainRole,
                  const std::string &domainVarName,
                  TensorSpaceRole rangeRole,
                  const std::string &rangeVarName) const override;

    bool canBuild(const TensorMeshPtr &tensorMesh,
                  const TensorDofLayoutPtr &tensorDofLayout,
                  const TensorBilinearFormPtr &bilinearForm,
                  const std::vector<std::string> &trialVarNames,
                  const std::vector<std::string> &testVarNames,
                  bool includeBoundaryTerms = false) const;

    TensorOperatorPtr build(const TensorMeshPtr &tensorMesh,
                            const TensorDofLayoutPtr &tensorDofLayout,
                            const TensorBilinearFormPtr &bilinearForm,
                            TensorSpaceRole domainRole,
                            const std::string &domainVarName,
                            TensorSpaceRole rangeRole,
                            const std::string &rangeVarName) const override;

    TensorOperatorPtr build(const TensorMeshPtr &tensorMesh,
                            const TensorDofLayoutPtr &tensorDofLayout,
                            const TensorBilinearFormPtr &bilinearForm,
                            const std::vector<std::string> &trialVarNames,
                            const std::vector<std::string> &testVarNames,
                            bool includeBoundaryTerms = false) const;

    double l2Weight() const;

    std::string validationMessage(const TensorMeshPtr &tensorMesh,
                                  const TensorDofLayoutPtr &tensorDofLayout,
                                  const TensorBilinearFormPtr &bilinearForm,
                                  TensorSpaceRole domainRole,
                                  const std::string &domainVarName,
                                  TensorSpaceRole rangeRole,
                                  const std::string &rangeVarName) const;

    std::string validationMessage(const TensorMeshPtr &tensorMesh,
                                  const TensorDofLayoutPtr &tensorDofLayout,
                                  const TensorBilinearFormPtr &bilinearForm,
                                  const std::vector<std::string> &trialVarNames,
                                  const std::vector<std::string> &testVarNames,
                                  bool includeBoundaryTerms = false) const;
  };
}

#endif
