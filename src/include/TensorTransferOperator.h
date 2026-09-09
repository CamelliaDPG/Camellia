#ifndef Camellia_TensorTransferOperator_h
#define Camellia_TensorTransferOperator_h

#include <vector>

#include "ComponentTransferOperator1D.h"

namespace Camellia
{
  class TensorTransferOperator;
  using TensorTransferOperatorPtr = Teuchos::RCP<TensorTransferOperator>;

  /**
   *  \brief Tensor-product transfer composition built from ComponentTransferOperator1D factors.
   *
   *  This class is the tensor-product composition layer above the componentwise transfer operators.
   *  In the current first draft, it is:
   *  \begin{itemize}
   *    \item per-variable,
   *    \item role-aware,
   *    \item flat-buffer-first,
   *    \item and based on the same lower-indexed-components-vary-fastest convention used by TensorDofLayout.
   *  \end{itemize}
   *
   *  The implementation composes component transfer factors explicitly rather than introducing a separate
   *  global sparse operator type at this stage.
   */
  class TensorTransferOperator
  {
    TensorMeshPtr _coarseMesh;
    TensorMeshPtr _fineMesh;
    TensorDofLayoutPtr _coarseLayout;
    TensorDofLayoutPtr _fineLayout;
    TensorSpaceRole _role;
    std::string _varName;
    ComponentTransferType _transferType;
    std::vector<ComponentTransferOperator1DPtr> _componentTransfers;

    void validateConstructorInputs() const;
  public:
    TensorTransferOperator(TensorMeshPtr coarseMesh,
                           TensorMeshPtr fineMesh,
                           TensorDofLayoutPtr coarseLayout,
                           TensorDofLayoutPtr fineLayout,
                           TensorSpaceRole role,
                           const std::string &varName,
                           ComponentTransferType transferType);

    TensorMeshPtr coarseMesh() const;
    TensorMeshPtr fineMesh() const;
    TensorSpaceRole role() const;
    const std::string &varName() const;
    ComponentTransferType transferType() const;

    int componentCount() const;
    ComponentTransferOperator1DPtr componentTransfer(int componentOrdinal) const;

    IndexType rowCount() const;
    IndexType colCount() const;

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const;
  };
}

#endif
