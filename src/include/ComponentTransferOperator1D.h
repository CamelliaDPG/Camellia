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
   *  discretization/runtime metadata. It supports both h- and p-transfer in one spatial/auxiliary
   *  component and is intended to serve as the building block for later tensor-product transfer composition.
   *
   *  The current implementation is per-variable, role-aware, and local-layout-driven. It exposes a flat local
   *  operator matrix representation for the selected variable/component pair, with prolongation and transpose-based
   *  restriction semantics in the initial model.
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
