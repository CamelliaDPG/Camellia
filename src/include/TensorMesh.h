#ifndef Camellia_TensorMesh_h
#define Camellia_TensorMesh_h

#include <map>
#include <string>
#include <vector>

#include "BasisFamily.h"
#include "TensorMeshTopology.h"
#include "TensorVarDiscretization.h"
#include "TensorVarFactory.h"

namespace Camellia
{
  class TensorDofLayout;
  using TensorDofLayoutPtr = Teuchos::RCP<TensorDofLayout>;

  class TensorMesh;
  using TensorMeshPtr = Teuchos::RCP<TensorMesh>;

  /**
   *  \brief Tensor-native discretization/runtime object built on top of TensorMeshTopology.
   *
   *  TensorMesh combines topology, tensor-variable registration, and discretization defaults.
   *  It intentionally does not inherit from legacy Mesh. In the current implementation, it owns
   *  separate trial and test discretization registrations, mesh-wide default basis/p-order data,
   *  and the logic for combining those defaults with per-variable deltas and topology-side p state.
   */
  class TensorMesh
  {
    Teuchos::RCP<TensorMeshTopology> _topology;
    TensorVarFactoryPtr _tensorVarFactory;
    BasisFamilyPtr _defaultBasisFamily;
    int _defaultPolyOrder;
    int _testSpaceEnrichment;

    std::map<std::string, TensorVarDiscretizationPtr> _trialDiscretizations;
    std::map<std::string, TensorVarDiscretizationPtr> _testDiscretizations;

    void validateConstructorInputs() const;
    void validateDiscretization(const TensorVarDiscretizationPtr &discretization) const;
  public:
    TensorMesh(Teuchos::RCP<TensorMeshTopology> topology,
               TensorVarFactoryPtr tensorVarFactory,
               BasisFamilyPtr defaultBasisFamily,
               int defaultPolyOrder,
               int testSpaceEnrichment = 0);

    Teuchos::RCP<TensorMeshTopology> topology() const;
    TensorVarFactoryPtr tensorVarFactory() const;

    BasisFamilyPtr defaultBasisFamily() const;
    int defaultPolyOrder() const;
    int testSpaceEnrichment() const;

    void addTrialDiscretization(TensorVarDiscretizationPtr discretization);
    void addTestDiscretization(TensorVarDiscretizationPtr discretization);

    bool hasTrialVar(const std::string &name) const;
    bool hasTestVar(const std::string &name) const;

    TensorVarDiscretizationPtr trialDiscretization(const std::string &name) const;
    TensorVarDiscretizationPtr testDiscretization(const std::string &name) const;

    std::vector<TensorVarDiscretizationPtr> trialDiscretizations() const;
    std::vector<TensorVarDiscretizationPtr> testDiscretizations() const;

    BasisFamilyPtr effectiveBasisFamily(const TensorVarDiscretizationPtr &discretization) const;
    int effectivePolyOrderDelta(const TensorVarDiscretizationPtr &discretization, int componentOrdinal, bool isTest) const;
    int effectivePolyOrder(const TensorVarDiscretizationPtr &discretization,
                           const ComponentCellID1D &cellID,
                           int componentOrdinal,
                           bool isTest) const;

    /**
     *  \brief Construct the explicit tensor-native local layout object for this TensorMesh.
     *
     *  The returned TensorDofLayout remains local/discretization-focused and does not introduce
     *  global numbering or gather/scatter behavior. Those belong to the next issue.
     */
    TensorDofLayoutPtr createDofLayout() const;
  };
}

#endif
