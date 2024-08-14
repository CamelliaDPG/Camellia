// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.

#ifndef DPG_MESH
#define DPG_MESH

/*
 *  Mesh.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "TypeDefs.h"

// Intrepid includes
#include "Camellia_FieldContainer.hpp"

// Epetra includes
#include "EpetraExt_ConfigDefs.h"
#include <Epetra_Map.h>

#include "Epetra_Vector.h"

#include "ElementType.h"
#include "ElementTypeFactory.h"
#include "Element.h"
#include "Boundary.h"
#include "BF.h"
#include "DofOrderingFactory.h"
#include "RefinementPattern.h"
#include "MeshPartitionPolicy.h"
#include "VarFactory.h"

#include "RefinementObserver.h"

#include "Function.h"
#include "ParametricCurve.h"

#include "MeshTopology.h"

#include "DofInterpreter.h"

namespace Camellia
{
class MeshPartitionPolicy;

class Mesh : public RefinementObserver, public DofInterpreter
{
  MeshTopologyViewPtr _meshTopology;

  Teuchos::RCP<GlobalDofAssignment> _gda;

  int _pToAddToTest;
  bool _enforceMBFluxContinuity; // default to false (the historical value)
  bool _usePatchBasis; // use MultiBasis if this is false.
  bool _useConformingTraces; // if true, enforces vertex trace continuity

  TBFPtr<double> _bilinearForm;
  VarFactoryPtr _varFactory;
  // for now, just a uniform mesh, with a rectangular boundary and elements.
  Boundary _boundary;

  // preferred private constructor to use during deepCopy();
  Mesh(MeshTopologyViewPtr meshTopology, Teuchos::RCP<GlobalDofAssignment> gda, VarFactoryPtr varFactory,
       int pToAddToTest, bool useConformingTraces, bool usePatchBasis, bool enforceMBFluxContinuity);

  // deprecated private constructor to use during deepCopy();
  Mesh(MeshTopologyViewPtr meshTopology, Teuchos::RCP<GlobalDofAssignment> gda, TBFPtr<double> bf,
       int pToAddToTest, bool useConformingTraces, bool usePatchBasis, bool enforceMBFluxContinuity);

  void initializePartitionPolicyIfNull(MeshPartitionPolicyPtr &partitionPolicy, Epetra_CommPtr Comm);

  vector< Teuchos::RCP<RefinementObserver> > _registeredObservers; // meshes that should be modified upon refinement (must differ from this only in bilinearForm; must have identical geometry & cellIDs)

  map<IndexType, GlobalIndexType> getGlobalVertexIDs(const Camellia::FieldContainer<double> &vertexCoordinates);

  Camellia::FieldContainer<double> physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr, vector<GlobalIndexType> &cellIDs );

  void setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly);

  void setNeighbor(ElementPtr elemPtr, unsigned elemSide, ElementPtr neighborPtr, unsigned neighborSide);

  GlobalIndexType getVertexIndex(double x, double y, double tol=1e-14);

  void verticesForCells(Camellia::FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs);

  bool _tensorTopologyMeansSpaceTime = false;
  
  static map<int,int> _emptyIntIntMap; // just defined here to implement a default argument to constructor (there's got to be a better way)
public:
  // Preferred Constructor for min rule, n-D, single H1Order
  Mesh(MeshTopologyViewPtr meshTopology, VarFactoryPtr varFactory, int H1Order, int pToAddTest,
       map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
       MeshPartitionPolicyPtr meshPartitionPolicy = Teuchos::null, Epetra_CommPtr Comm = Teuchos::null, bool allowPruning = true);

  // Preferred Constructor for min rule, n-D, vector H1Order for tensor topologies (tensorial degree 0 and 1 supported)
  Mesh(MeshTopologyViewPtr meshTopology, VarFactoryPtr varFactory, vector<int> H1Order, int pToAddTest,
       map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
       MeshPartitionPolicyPtr meshPartitionPolicy = Teuchos::null, Epetra_CommPtr Comm = Teuchos::null, bool allowPruning = true);

  // legacy (max rule 2D) constructor:
  Mesh(const vector<vector<double> > &vertices, vector< vector<IndexType> > &elementVertices,
       TBFPtr<double> bilinearForm, int H1Order, int pToAddTest, bool useConformingTraces = true,
       map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
       vector< PeriodicBCPtr > periodicBCs = vector< PeriodicBCPtr >(), Epetra_CommPtr Comm = Teuchos::null);

  // Constructor for min rule, n-D, single H1Order
  Mesh(MeshTopologyViewPtr meshTopology, TBFPtr<double> bilinearForm, int H1Order, int pToAddTest,
       map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
       MeshPartitionPolicyPtr meshPartitionPolicy = Teuchos::null, Epetra_CommPtr Comm = Teuchos::null);
  
  Mesh(MeshTopologyViewPtr meshTopology, TBFPtr<double> bilinearForm, int H1Order, int pToAddTest,
       map<int,int> trialOrderEnhancements, map<int,int> testOrderEnhancements,
       MeshPartitionPolicyPtr meshPartitionPolicy, Epetra_CommPtr Comm,
       bool allowMeshTopologyPruning);

  // Constructor for min rule, n-D, vector H1Order for tensor topologies (tensorial degree 0 and 1 supported)
  Mesh(MeshTopologyViewPtr meshTopology, TBFPtr<double> bilinearForm, vector<int> H1Order, int pToAddTest,
       map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
       MeshPartitionPolicyPtr meshPartitionPolicy = Teuchos::null, Epetra_CommPtr Comm = Teuchos::null, bool allowPruning = true);

  // ! Constructor for a single-element mesh extracted from an existing mesh
  Mesh(MeshPtr mesh, GlobalIndexType cellID, Epetra_CommPtr Comm);
  
  ~Mesh();
  
#ifdef HAVE_EPETRAEXT_HDF5
  void saveToHDF5(string filename);
#endif

  Epetra_CommPtr& Comm();
  Teuchos_CommPtr& TeuchosComm();
  
  // ! deepCopy makes a deep copy of both MeshTopology and GDA, but not bilinear form
  MeshPtr deepCopy();

  static MeshPtr readMsh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  static MeshPtr readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd);

  // deprecated static constructors (use MeshFactory methods instead):
  static MeshPtr buildQuadMesh(const Camellia::FieldContainer<double> &quadBoundaryPoints,
                               int horizontalElements, int verticalElements,
                               TBFPtr<double> bilinearForm,
                               int H1Order, int pTest, bool triangulate=false, bool useConformingTraces=true,
                               map<int,int> trialOrderEnhancements=_emptyIntIntMap,
                               map<int,int> testOrderEnhancements=_emptyIntIntMap);
  static MeshPtr buildQuadMeshHybrid(const Camellia::FieldContainer<double> &quadBoundaryPoints,
                                     int horizontalElements, int verticalElements,
                                     TBFPtr<double> bilinearForm,
                                     int H1Order, int pTest, bool useConformingTraces=true);
  static void quadMeshCellIDs(Camellia::FieldContainer<int> &cellIDs,
                              int horizontalElements, int verticalElements,
                              bool useTriangles);

  GlobalIndexType activeCellOffset();

  Camellia::FieldContainer<double> cellSideParities( ElementTypePtr elemTypePtr);
  Camellia::FieldContainer<double> cellSideParitiesForCell( GlobalIndexType cellID );
  
  double sideParity(GlobalIndexType cellID, unsigned sideOrdinal);

  TBFPtr<double> bilinearForm();
  void setBilinearForm( TBFPtr<double>);

  //! This method should probably be moved to MeshTopology; its implementation is independent of Mesh.
  //  bool cellContainsPoint(GlobalIndexType cellID, vector<double> &point);
  std::vector<GlobalIndexType> cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints, bool minusOnesForOffRank=true);
  std::vector<ElementPtr> elementsForPoints(const Camellia::FieldContainer<double> &physicalPoints, bool nullElementsIfOffRank=true);

  vector< Teuchos::RCP< ElementType > > elementTypes(PartitionIndexType partitionNumber=-1); // returns *all* elementTypes by default

  Boundary &boundary();

  vector< GlobalIndexType > cellIDsOfType(ElementTypePtr elemType); // for current MPI node.
  vector< GlobalIndexType > cellIDsOfType(int partitionNumber, ElementTypePtr elemTypePtr);
  vector< GlobalIndexType > cellIDsOfTypeGlobal(ElementTypePtr elemTypePtr);

  const set<GlobalIndexType> & cellIDsInPartition(); // rank-local cellIDs
  
  int cellPolyOrder(GlobalIndexType cellID);
  vector<int> cellTensorPolyOrder(GlobalIndexType cellID);

  // ! RefinementObserver method; when invoked, we too should probably repartitionAndRebuild...
  void didRepartition(MeshTopologyPtr meshTopo);
  
  //! This should probably be renamed "enforceRegularityRules" and then made more general to enforce whatever rules are appropriate for the mesh.
  void enforceOneIrregularity(bool repartitionAndMigrate = true);

  vector<double> getCellCentroid(GlobalIndexType cellID);

  // commented out because unused
  //Epetra_Map getCellIDPartitionMap(int rank, Epetra_Comm* Comm);

  ElementPtr getElement(GlobalIndexType cellID);
  ElementTypePtr getElementType(GlobalIndexType cellID);

  std::set<GlobalIndexType> getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal);
  
  const map< pair<GlobalIndexType,IndexType> , GlobalIndexType>& getLocalToGlobalMap();
  //  map< int, pair<int,int> > getGlobalToLocalMap();

  GlobalIndexType globalDofCount();
  GlobalIndexType globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex);
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber);

  GlobalDofAssignmentPtr globalDofAssignment();

  // ! Collective method.  Returns all cell IDs that are active.
  set<GlobalIndexType> getActiveCellIDsGlobal();

  ElementPtr ancestralNeighborForSide(ElementPtr elem, int sideOrdinal, int &elemSideOrdinalInNeighbor);

  vector< ElementPtr > elementsOfType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr);
  vector< ElementPtr > elementsOfTypeGlobal(ElementTypePtr elemTypePtr); // may want to deprecate in favor of cellIDsOfTypeGlobal()

  vector< ElementPtr > elementsInPartition(PartitionIndexType partitionNumber = -1);

  int getDimension(); // spatial dimension of the mesh
  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID);
  
  //!! Returns the global dof indices for the indicated subcell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime);
  
  DofOrderingFactory & getDofOrderingFactory();

  ElementTypeFactory & getElementTypeFactory();

  TFunctionPtr<double> getTransformationFunction(); // will be NULL for meshes without edge curves defined

  // method signature inherited from RefinementObserver:
  void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern);

  using RefinementObserver::hRefine; // avoid compiler warnings about the hRefine() methods below.
  void hRefine(const vector<GlobalIndexType> &cellIDs, bool repartitionAndRebuild = true);
  void hRefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild = true);

  void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild);
  void hRefine(const vector<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern);
  void hRefine(const vector<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild);

  // ! unrefine the specified cells -- at end of unrefinement, these cells should be active, possibly modulo enforcement of 1-irregularity
  void hUnrefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild = true);
  
  // ! For curvilinear geometry, call this after mesh initialization.
  void initializeTransformationFunction();
  
  void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients);
  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Camellia::FieldContainer<double> &basisCoefficients,
                                       Camellia::FieldContainer<double> &globalCoefficients, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices);
  
  using DofInterpreter::interpretLocalData;
  void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localData,
                          Camellia::FieldContainer<double> &globalData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices);

  bool isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const;
  
  bool meshUsesMaximumRule();
  bool meshUsesMinimumRule();

  bool myCellsInclude(GlobalIndexType cellID) const;
  
  GlobalIndexType numActiveElements();
  
  // ! Returns the number of degrees of freedom belonging to field (volume) variables.  Involves MPI collective communication.  Must be called on every rank.
  GlobalIndexType numFieldDofs();
  
  // ! Returns the number of degrees of freedom on the mesh skeleton.  Involves MPI collective communication.  Must be called on every rank.
  GlobalIndexType numFluxDofs();

  // ! Returns the total number of degrees of freedom on the mesh.  Involves MPI collective communication.  Must be called on every rank.
  GlobalIndexType numGlobalDofs();

  GlobalIndexType numElements();

  GlobalIndexType numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr );

  int parityForSide(GlobalIndexType cellID, int sideOrdinal);

  PartitionIndexType partitionForCellID(GlobalIndexType cellID);
  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex );
  PartitionIndexType partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex );

  Camellia::FieldContainer<double> physicalCellNodes( ElementTypePtr elemType);
  Camellia::FieldContainer<double> physicalCellNodesForCell(GlobalIndexType cellID);
  Camellia::FieldContainer<double> physicalCellNodesGlobal( ElementTypePtr elemType );

  // use this method only if you can ensure the provided cellIDs all have the same type
  Camellia::FieldContainer<double> physicalCellNodes(vector<GlobalIndexType> &cellIDs );
  
  void pRefine(const vector<GlobalIndexType> &cellIDsForPRefinements);
  void pRefine(const vector<GlobalIndexType> &cellIDsForPRefinements, int pToAdd);
  void pRefine(const set<GlobalIndexType> &cellIDsForPRefinements);
  void pRefine(const set<GlobalIndexType> &cellIDsForPRefinements, int pToAdd, bool repartitionAndRebuild = true);
  void printLocalToGlobalMap(); // for debugging

  void rebuildLookups();

  void registerObserver(Teuchos::RCP<RefinementObserver> observer);
  
  //! returns the irregularity of the mesh, defined as the maximum depth of refinement hierarchy for an edge belonging to an active cell.  MPI collective method.
  int irregularity();

  //! the "delta p" for DPG test space enrichment.
  int testSpaceEnrichment() const;
  
  template <typename Scalar>
  void registerSolution(TSolutionPtr<Scalar> solution);

  // ! Repartition elements, migrating data as necessary, and rebuild local-to-global degree of freedom maps.
  void repartitionAndRebuild();
  
  int condensedRowSizeUpperBound();
  int rowSizeUpperBound(); // accounts for multiplicity, but isn't a tight bound

  void setEdgeToCurveMap(const map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > &edgeToCurveMap);
  void setEnforceMultiBasisFluxContinuity( bool value );

  vector< ParametricCurvePtr > parametricEdgesForCell(GlobalIndexType cellID, bool neglectCurves=false);

  void setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy );

  void setUsePatchBasis( bool value );
  bool usePatchBasis();
  
  bool tensorTopologyMeansSpaceTime() const;
  void setTensorTopologyMeansSpaceTime(bool value);

  MeshTopologyViewPtr getTopology();

  vector< vector<double> > verticesForCell(GlobalIndexType cellID);
  vector<IndexType> vertexIndicesForCell(GlobalIndexType cellID);
  Camellia::FieldContainer<double> vertexCoordinates(GlobalIndexType vertexIndex);

  void verticesForCell(Camellia::FieldContainer<double>& vertices, GlobalIndexType cellID);
  void verticesForSide(Camellia::FieldContainer<double>& vertices, GlobalIndexType cellID, int sideOrdinal);

  void unregisterObserver(RefinementObserver* observer);
  void unregisterObserver(Teuchos::RCP<RefinementObserver> observer);
  template <typename Scalar>
  void unregisterSolution(TSolutionPtr<Scalar> solution);

  VarFactoryPtr varFactory() const;
  
  void writeMeshPartitionsToFile(const string & fileName);

  double getCellMeasure(GlobalIndexType cellID);
  double getCellXSize(GlobalIndexType cellID);
  double getCellYSize(GlobalIndexType cellID);
  vector<double> getCellOrientation(GlobalIndexType cellID);
};
}

#endif
