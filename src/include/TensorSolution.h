#ifndef Camellia_TensorSolution_h
#define Camellia_TensorSolution_h
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TestForException.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include "TensorCoefficientMap.h"
#include "TensorCoefficientTpetraImport.h"
#include "TensorCoefficientValueEvaluator.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
namespace Camellia {
class TensorSolution {
public:
  using map_type = TensorCoefficientTpetraImport::map_type;
  using multivector_type = TensorCoefficientTpetraImport::multivector_type;
private:
  struct VariableStorage {
    TensorSpaceRole role;
    std::string varName;
    TensorCoefficientMap coefficientMap;
    Teuchos::RCP<const map_type> ownedMap;
    Teuchos::RCP<multivector_type> ownedCoefficients;
    VariableStorage(TensorMeshPtr tensorMesh, TensorDofLayoutPtr dofLayout, TensorSpaceRole role, const std::string &varName, TensorLocalOrdering ordering, const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : role(role), varName(varName), coefficientMap(tensorMesh, dofLayout, role, varName, ordering) {
      ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, comm);
      ownedCoefficients = Teuchos::rcp(new multivector_type(ownedMap, 1));
      ownedCoefficients->putScalar(0.0);
    }
  };
  TensorMeshPtr _tensorMesh;
  TensorDofLayoutPtr _dofLayout;
  Teuchos::RCP<const Teuchos::Comm<int>> _comm;
  TensorLocalOrdering _ordering;
  std::map<std::string, Teuchos::RCP<VariableStorage>> _variableStorage;
  static std::string storageKey(TensorSpaceRole role, const std::string &varName) { return std::string(role == TENSOR_TRIAL_SPACE ? "trial:" : "test:") + varName; }
  Teuchos::RCP<VariableStorage> storage(TensorSpaceRole role, const std::string &varName) const {
    auto entry = _variableStorage.find(storageKey(role, varName));
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _variableStorage.end(), std::invalid_argument, "TensorSolution has no registered coefficient variable with the requested name and role.");
    return entry->second;
  }
public:
  /** \brief Tensor-native owner for long-lived solution coefficient storage.
   *
   * TensorSolution owns rank-owned Tpetra coefficient multivectors for tensor variables and exposes TensorCoefficientMap / TensorCoefficientTpetraImport owned-plus-ghost views used by tensor-native coefficient evaluation and moment/RHS handoff.
   *
   * The name is provisional: this class is intended as the seed of a wholesale, general replacement for the legacy Solution class, not as a driver-owned side cache. Solver/time-stepping and visualization responsibilities should be decomposed into focused collaborators as the replacement grows; high-order GLVis-style visualization can layer on this tensor-native coefficient storage contract.
   */
  TensorSolution(TensorMeshPtr tensorMesh, TensorDofLayoutPtr dofLayout, const Teuchos::RCP<const Teuchos::Comm<int>> &comm, TensorLocalOrdering ordering = TENSOR_VARIABLE_ORDERING)
  : _tensorMesh(tensorMesh), _dofLayout(dofLayout), _comm(comm), _ordering(ordering) {
    TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument, "TensorSolution requires a non-null TensorMesh.");
    if (_dofLayout == Teuchos::null) _dofLayout = _tensorMesh->createDofLayout();
    if (_comm == Teuchos::null) _comm = Tpetra::getDefaultComm();
    TEUCHOS_TEST_FOR_EXCEPTION(_comm == Teuchos::null, std::invalid_argument, "TensorSolution requires a non-null communicator.");
  }
  TensorMeshPtr tensorMesh() const { return _tensorMesh; }
  TensorDofLayoutPtr dofLayout() const { return _dofLayout; }
  Teuchos::RCP<const Teuchos::Comm<int>> comm() const { return _comm; }
  TensorLocalOrdering ordering() const { return _ordering; }
  void registerCoefficientVariable(TensorSpaceRole role, const std::string &varName) {
    const std::string key = storageKey(role, varName);
    if (_variableStorage.find(key) != _variableStorage.end()) return;
    _variableStorage[key] = Teuchos::rcp(new VariableStorage(_tensorMesh, _dofLayout, role, varName, _ordering, _comm));
  }
  bool hasCoefficientVariable(TensorSpaceRole role, const std::string &varName) const { return _variableStorage.find(storageKey(role, varName)) != _variableStorage.end(); }
  const TensorCoefficientMap &coefficientMap(TensorSpaceRole role, const std::string &varName) const { return storage(role, varName)->coefficientMap; }
  Teuchos::RCP<const map_type> ownedMap(TensorSpaceRole role, const std::string &varName) const { return storage(role, varName)->ownedMap; }
  Teuchos::RCP<multivector_type> ownedCoefficientVector(TensorSpaceRole role, const std::string &varName) { return storage(role, varName)->ownedCoefficients; }
  Teuchos::RCP<const multivector_type> ownedCoefficientVector(TensorSpaceRole role, const std::string &varName) const { return storage(role, varName)->ownedCoefficients; }
  void putScalar(TensorSpaceRole role, const std::string &varName, double value) { storage(role, varName)->ownedCoefficients->putScalar(value); }
  void setOwnedCoefficients(TensorSpaceRole role, const std::string &varName, const std::vector<double> &localValues) {
    auto data = storage(role, varName);
    TEUCHOS_TEST_FOR_EXCEPTION(localValues.size() != data->ownedMap->getLocalNumElements(), std::invalid_argument, "TensorSolution::setOwnedCoefficients local value count must match the owned map size.");
    auto ownedHost = data->ownedCoefficients->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (size_t localOrdinal=0; localOrdinal<localValues.size(); localOrdinal++) ownedHost(localOrdinal,0) = localValues[localOrdinal];
  }
  void setOwnedCoefficientByGlobalIndex(TensorSpaceRole role, const std::string &varName, IndexType globalIndex, double value) {
    auto data = storage(role, varName);
    const auto localOrdinal = data->ownedMap->getLocalElement(static_cast<typename map_type::global_ordinal_type>(globalIndex));
    TEUCHOS_TEST_FOR_EXCEPTION(localOrdinal == Teuchos::OrdinalTraits<typename map_type::local_ordinal_type>::invalid(), std::invalid_argument, "TensorSolution::setOwnedCoefficientByGlobalIndex requires a coefficient owned by this rank.");
    auto ownedHost = data->ownedCoefficients->getLocalViewHost(Tpetra::Access::ReadWrite);
    ownedHost(localOrdinal,0) = value;
  }
  TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValuesForVolumeCell(TensorSpaceRole role, const std::string &varName, const TensorMeshTopology::TensorCellTuple &cellTuple) const {
    auto data = storage(role, varName);
    return TensorCoefficientTpetraImport::ownedValuesForVolumeCell(data->coefficientMap, *data->ownedCoefficients, cellTuple);
  }
  TensorCoefficientValueEvaluator::HostVector volumeValuesOnTensorCell(TensorSpaceRole role,
                                                                       const std::string &varName,
                                                                       const TensorBasisCacheND &basisCache,
                                                                       const TensorMeshTopology::TensorCellTuple &cellTuple) const {
    auto data = storage(role, varName);
    auto ownedValues = TensorCoefficientTpetraImport::ownedValuesForVolumeCell(data->coefficientMap, *data->ownedCoefficients, cellTuple);
    return TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(data->coefficientMap, basisCache, ownedValues, cellTuple);
  }
  TensorCoefficientValueEvaluator::HostMatrix volumeValuesAsConfigurationVelocityMatrix1D2V(TensorSpaceRole role,
                                                                                            const std::string &varName,
                                                                                            const TensorBasisCacheND &basisCache,
                                                                                            const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                                            const std::vector<int> &configurationComponentOrdinals,
                                                                                            const std::vector<int> &velocityComponentOrdinals) const {
    auto data = storage(role, varName);
    auto ownedValues = TensorCoefficientTpetraImport::ownedValuesForVolumeCell(data->coefficientMap, *data->ownedCoefficients, cellTuple);
    return TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(data->coefficientMap,
                                                                                          basisCache,
                                                                                          ownedValues,
                                                                                          cellTuple,
                                                                                          configurationComponentOrdinals,
                                                                                          velocityComponentOrdinals);
  }
  std::vector<TensorCoefficientValueEvaluator::OwnedCoefficientValues> ownedValuesForVolumeCells(TensorSpaceRole role, const std::string &varName, const std::vector<TensorMeshTopology::TensorCellTuple> &cellTuples) const {
    auto data = storage(role, varName);
    return TensorCoefficientTpetraImport::ownedValuesForVolumeCells(data->coefficientMap, *data->ownedCoefficients, cellTuples);
  }
  TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValuesForSkeletonEntity(TensorSpaceRole role, const std::string &varName, const TensorMeshTopology::TensorSkeletonEntity &entity) const {
    auto data = storage(role, varName);
    return TensorCoefficientTpetraImport::ownedValuesForSkeletonEntity(data->coefficientMap, *data->ownedCoefficients, entity);
  }
};
using TensorSolutionPtr = Teuchos::RCP<TensorSolution>;
}
#endif
