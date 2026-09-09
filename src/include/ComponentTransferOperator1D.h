#ifndef Camellia_ComponentTransferOperator1D_h
#define Camellia_ComponentTransferOperator1D_h

#include <vector>

#include "TensorMesh.h"
#include "TensorDofLayout.h"

namespace Camellia
{
  class ComponentTransferOperator1D;
  using ComponentTransferOperator1DPtr = Teuchos::RCP<ComponentTransferOperator1D>;

  enum ComponentTransferType
  {
    COMPONENT_PROLONGATION,
    COMPONENT_RESTRICTION
  };

  /**
   *  \brief Explicit componentwise 1D transfer abstraction for the tensor-native path.
   *
   *  This class provides the first componentwise transfer layer above the current topology and
   *  discretization/runtime metadata. It supports both h- and p-transfer in one component and
   *  is intended to serve as a building block for later tensor-product transfer composition.
   *
   *  The current implementation is:
   *  \begin{itemize}
   *    \item per-variable,
   *    \item role-aware,
   *    \item local-layout-driven,
   *    \item and matrix-explicit in a flat local operator form.
   *  \end{itemize}
   *
   *  The p-transfer path uses the available hierarchical basis prolongation support. The h-transfer path
   *  currently uses the present dyadic parent/child structure together with the current local layout model.
   *  Restriction starts from transpose/averaging-style semantics in this first draft but is not intended to
   *  freeze later design choices.
   */
  class ComponentTransferOperator1D
  {
    TensorMeshPtr _coarseMesh;
    TensorMeshPtr _fineMesh;
    TensorDofLayoutPtr _coarseLayout;
    TensorDofLayoutPtr _fineLayout;
    TensorSpaceRole _role;
    std::string _varName;
    int _componentOrdinal;
    ComponentTransferType _transferType;
    std::vector<std::vector<double>> _operatorMatrix;

    void validateConstructorInputs() const;
    void buildOperatorMatrix();
    void buildPTransferOperatorMatrix();
    void buildHTransferOperatorMatrix();

  public:
    ComponentTransferOperator1D(TensorMeshPtr coarseMesh,
                                TensorMeshPtr fineMesh,
                                TensorDofLayoutPtr coarseLayout,
                                TensorDofLayoutPtr fineLayout,
                                TensorSpaceRole role,
                                const std::string &varName,
                                int componentOrdinal,
                                ComponentTransferType transferType);

    TensorMeshPtr coarseMesh() const;
    TensorMeshPtr fineMesh() const;
    TensorSpaceRole role() const;
    const std::string &varName() const;
    int componentOrdinal() const;
    ComponentTransferType transferType() const;

    IndexType rowCount() const;
    IndexType colCount() const;
    double entry(IndexType row, IndexType col) const;
    const std::vector<std::vector<double>> &operatorMatrix() const;

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const;
  };
}

#endif
