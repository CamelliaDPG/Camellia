//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Boltzmann1D1VFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/3/2020.
//
//

#include "Boltzmann1D1VFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

using namespace Camellia;

const string Boltzmann1D1VFormulation::S_F = "f";
const string Boltzmann1D1VFormulation::S_E = "E";

const string Boltzmann1D1VFormulation::S_F_HAT = "\\widehat{f}";
const string Boltzmann1D1VFormulation::S_T_N_HAT = "\\widehat{t}_n";
const string Boltzmann1D1VFormulation::S_E_HAT = "\\widehat{E}";

const string Boltzmann1D1VFormulation::S_W = "w";
const string Boltzmann1D1VFormulation::S_TAU = "\\tau";

// defines a function on the spatial (1D) mesh by integrating the 1D1V function in the velocity dimension
class VelocitySpace1DIntegral : public QuadraturePointFunction
{
  MeshPtr     _spatialMesh;
  SolutionPtr _phaseSpaceSolution;
  FunctionPtr _distributionFunction; // f from the phase space solution
  VarPtr      _f;
  
  const int _numMoments   = 1; // hard-coded thus for now; we'll support more moments in the future.
  
  int _numPoints1D;
  
  std::map<GlobalIndexType, std::vector<GlobalIndexType> > _spatialCellIDToPhaseSpaceCellIDs;
  Teuchos::RCP<PhasePhysicalMeshMap1D1V> phaseSpacePhysicalMeshMap;
  
  //! _basisIntegrals has shape (M,Ps,F,Px), where:
  //! - M is the zero-based moment ordinal (0 is the first moment)
  //! - Ps is the point set number (0 for equal-width physical cell; others follow, details still to be worked out)
  //! - F is the phase space basis ordinal -- we assume uniform cell geometry; when we relax this we'll need something more complicated, but may want to retain the ability to specify where geometry is the same..
  //! - Px is the ordinal of the point in the 1D quadrature rule
  //! Note that the integrals are *relative*; they should be multiplied by the cell height to get the true integral (in 2V, 3V, this generalizes to the velocity space measure).  (The transformation for moments beyond the first is more complex, but it should be possible to construct such a transformation.)
  FieldContainer<double> _basisIntegrals;
  Teuchos::RCP<PhasePhysicalMeshMap1D1V> _phaseSpacePhysicalMeshMap;
  
  std::map< std::pair<int,int>, int> _pointSetNumbers; // (unrefinementLevel,uniformDescendantNumber) --> point set number (2nd index in _basisIntegrals)
  
  std::map< int, int > _physicalCellIDToLocalCellOrdinal;
  FieldContainer<double> _physicalValuesAtQuadraturePoints; // shape: (C,M,Px)
  
  void updateBasisIntegrals()
  {
    // examine sample spatial cell to get spatial quadrature degree (we assume this is uniform)
    const GlobalIndexType sampleSpatialCellID = 0;
    BasisCachePtr sampleSpatialBasisCache = BasisCache::basisCacheForCell(_spatialMesh, sampleSpatialCellID);
    const int spatialQuadratureDegree = sampleSpatialBasisCache->cubatureDegree();
    _numPoints1D = sampleSpatialBasisCache->getRefCellPoints().dimension(0);
    
    auto phaseSpaceMesh = _phaseSpaceSolution->mesh();
    auto & localCellIDs = phaseSpaceMesh->cellIDsInPartition();
    
    const int phaseSpaceDim = 2;
    
    if (localCellIDs.size() > 0)
    {
      // take the first cell; use this to determine basis integrals -- again, this assumes each cell has the same geometry; we'll want to change this eventually
      auto cellID = *localCellIDs.begin();
      
      auto fBasis = phaseSpaceMesh->getElementType(cellID)->trialOrderPtr->getBasis(_f->ID());
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(phaseSpaceMesh, cellID);
      
      CubatureFactory quadratureFactory;
      auto spatialLineQuadrature = quadratureFactory.create(CellTopology::line(), spatialQuadratureDegree);
      FieldContainer<double> spatialQuadraturePoints(spatialLineQuadrature->getNumPoints(),1);
      FieldContainer<double> spatialQuadratureWeights(spatialLineQuadrature->getNumPoints());
      spatialLineQuadrature->getCubature(spatialQuadraturePoints, spatialQuadratureWeights);
      
      const auto levelNumberPairs = _phaseSpacePhysicalMeshMap->referenceSpaceUnrefinementLevelDescendantNumberPairs();
      const int numPointSets = levelNumberPairs.size();
      
      int pointSetNumber = 0; // increments at bottom of levelNumberPair loop
      _pointSetNumbers.clear();
      for (const auto &levelNumberPair : levelNumberPairs)
      {
        _pointSetNumbers[levelNumberPair] = pointSetNumber++;
      }
      
      const int maxMoment = _numMoments;
      int fPolyDegree = fBasis->getDegree();
      const int velocityQuadratureDegree = fPolyDegree + (maxMoment - 1);
      auto velocityLineQuadrature = quadratureFactory.create(CellTopology::line(), velocityQuadratureDegree);
      FieldContainer<double> velocityQuadraturePoints(velocityLineQuadrature->getNumPoints(),1);
      FieldContainer<double> velocityQuadratureWeights(velocityLineQuadrature->getNumPoints());
      velocityLineQuadrature->getCubature(velocityQuadraturePoints, velocityQuadratureWeights);
      
      const int numSpatialPointsPerSet = spatialLineQuadrature->getNumPoints();
      const int numVelocityPoints      = velocityLineQuadrature->getNumPoints();
      const int numSpatialPointsTotal  = numSpatialPointsPerSet * numPointSets;
      const int numPointsPhaseSpace    = numSpatialPointsTotal * numVelocityPoints;
            
      FieldContainer<double> tensorPoints(numPointsPhaseSpace, phaseSpaceDim);
      
      pointSetNumber = 0; // increments at bottom of levelNumberPair loop
      for (const auto &levelNumberPair : levelNumberPairs)
      {
        const int & unrefinementLevel = levelNumberPair.first;
        const int & descendantNumber  = levelNumberPair.second;
        for (int spatialPointOrdinalInDescendant=0; spatialPointOrdinalInDescendant<numSpatialPointsPerSet; spatialPointOrdinalInDescendant++)
        {
          const double & xInDescendant = spatialQuadraturePoints(spatialPointOrdinalInDescendant,0); // lies in [-1,1]
          
          const int numDescendants = 1 << unrefinementLevel;
          const double xLeft = descendantNumber * 2.0 / numDescendants - 1.0; // left vertex of descendant cell in ancestor
          const double x = xLeft + (xInDescendant + 1.0) / numDescendants;
          
          const int spatialPointOrdinal = spatialPointOrdinalInDescendant + pointSetNumber * numSpatialPointsPerSet;
          
          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
          {
            const double & v = velocityQuadraturePoints(velocityPointOrdinal,0);
            const int tensorPointOrdinal = getTensorPointOrdinal(spatialPointOrdinal, velocityPointOrdinal, numSpatialPointsTotal);
            tensorPoints(tensorPointOrdinal, 0) = x;
            tensorPoints(tensorPointOrdinal, 1) = v;
          }
        }
        pointSetNumber++;
      }
      
      basisCache->setRefCellPoints(tensorPoints);
      auto basisValues = basisCache->getTransformedValues(fBasis, OP_VALUE);
      
      int basisCardinality = fBasis->getCardinality();
      _basisIntegrals.resize(_numMoments,numPointSets,basisCardinality,_numPoints1D);
      const int moment = 0;
      for (int pointSetNumber=0; pointSetNumber<numPointSets; pointSetNumber++)
      {
        for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
        {
          for (int spatialPointOrdinalInDescendant=0; spatialPointOrdinalInDescendant<numSpatialPointsPerSet; spatialPointOrdinalInDescendant++)
          {
            double integral = 0.0;
            const int spatialPointOrdinal = pointSetNumber * numSpatialPointsPerSet + spatialPointOrdinalInDescendant;
            for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
            {
              const int tensorPointOrdinal = getTensorPointOrdinal(spatialPointOrdinal, velocityPointOrdinal, numSpatialPointsTotal);
              const double & basisValue = (*basisValues)(0,basisOrdinal,tensorPointOrdinal);
              const double & velocityIntegralWeight = velocityQuadratureWeights(velocityPointOrdinal);
              integral += basisValue * velocityIntegralWeight;
            }
            _basisIntegrals(moment,pointSetNumber,basisOrdinal,spatialPointOrdinalInDescendant) = integral / 2.0; // divide by 2 because the ref line has length 2
          }
        }
      }
    }
    else
    {
      // no cells assigned to this MPI rank
      _basisIntegrals.resize(0,0,0,0);
    }
  }
public:
  VelocitySpace1DIntegral(MeshPtr spatialMesh, SolutionPtr phaseSpaceSolution, VarPtr f,
                                    const std::map<GlobalIndexType, std::vector<GlobalIndexType> > &spatialCellIDToPhaseSpaceCellIDs,
                                    Teuchos::RCP<PhasePhysicalMeshMap1D1V> phaseSpacePhysicalMeshMap)
  :
  QuadraturePointFunction(spatialMesh, phaseSpaceSolution->cubatureEnrichmentDegree(), 0),
  _spatialMesh(spatialMesh),
  _phaseSpaceSolution(phaseSpaceSolution),
  _f(f),
  _spatialCellIDToPhaseSpaceCellIDs(spatialCellIDToPhaseSpaceCellIDs),
  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
  {
    _distributionFunction = Function::solution(f, _phaseSpaceSolution);
    updateBasisIntegrals();
  }
  
  void willUpdateValues() override
  {
    // if the required point sets have changed, update _basisIntegrals now
    const auto requiredPointSets = _phaseSpacePhysicalMeshMap->referenceSpaceUnrefinementLevelDescendantNumberPairs();
    bool pointSetsMatch = true;
    if (requiredPointSets.size() != _pointSetNumbers.size())
    {
      pointSetsMatch = false;
    }
    else
    {
      for (auto & requiredPointSet : requiredPointSets)
      {
        if (_pointSetNumbers.find(requiredPointSet) == _pointSetNumbers.end())
        {
          pointSetsMatch = false;
          break;
        }
      }
    }
    if (!pointSetsMatch)
    {
      updateBasisIntegrals();
    }
    
    // Local integrals for all physical cells for which we have corresponding phase space cells go in integralContributionsFromMyPhaseSpaceCells;
    // we do MPI communication below once we have populated this…
    using std::map;
    using std::pair;
    using std::vector;
    using PhysicalCellPoint = pair<GlobalIndexType,int>; // first: physical cell ID; second: quadrature point ordinal
    using SenderPID = int;
    using RecipientPID = int;
    using MapKey = pair<SenderPID,PhysicalCellPoint>; // first: the sender's MPI rank; second: the physical cell point
    using MapValue = double;
    map<RecipientPID, map<MapKey, MapValue> > integralContributionsFromMyPhaseSpaceCells;
    
    SenderPID myPID = _phaseSpaceSolution->mesh()->Comm()->MyPID();
    const auto &myPhaseSpaceCellIDs = _phaseSpaceSolution->mesh()->cellIDsInPartition();
    const int basisCardinality           = _basisIntegrals.dimension(2);
    const int numSpatialQuadraturePoints = _basisIntegrals.dimension(3);
    FieldContainer<double> basisCoefficients(basisCardinality);
    int stageOrdinal;
    if (_phaseSpaceSolution->getSolutionStage() != Teuchos::null)
    {
      stageOrdinal = -1;
    }
    else
    {
      stageOrdinal = 0;
    }
    
    const int moment = 0;
    for (const auto &phaseSpaceCellID : myPhaseSpaceCellIDs)
    {
      auto cellVertices = _phaseSpaceSolution->mesh()->verticesForCell(phaseSpaceCellID);
      double vMin = cellVertices[0][1];
      double vMax = vMin;
      for (auto & vertex : cellVertices)
      {
        vMin = std::min(vMin, vertex[1]);
        vMax = std::max(vMax, vertex[1]);
      }
      double cellHeight = vMax - vMin;
      _phaseSpaceSolution->getBasisCoefficientsForCell(basisCoefficients, _f->ID(), phaseSpaceCellID, STANDARD_SOLUTION, stageOrdinal);
      
      const std::vector<PhysicalCellRelativeRefinementInfo> &physicalCellInfoVector = _phaseSpacePhysicalMeshMap->physicalCellInfo(phaseSpaceCellID);
      for (const auto & physicalCellInfo : physicalCellInfoVector)
      {
        const auto & physicalCellID    = physicalCellInfo.physicalCellID;
        const auto & recipientPID      = _spatialMesh->partitionForCellID(physicalCellID);
        const auto & unrefinementLevel = physicalCellInfo.unrefinementLevel;
        const auto & descendantNumber  = physicalCellInfo.uniformDescendantNumber;
        const std::pair<int,int> pointSetKey {unrefinementLevel,descendantNumber};
        TEUCHOS_TEST_FOR_EXCEPTION(_pointSetNumbers.find(pointSetKey) == _pointSetNumbers.end(), std::invalid_argument, "point set not found");
        const int pointSetNumber = _pointSetNumbers[pointSetKey];
        
        for (int spatialPointOrdinalInDescendant=0; spatialPointOrdinalInDescendant<numSpatialQuadraturePoints; spatialPointOrdinalInDescendant++)
        {
          const PhysicalCellPoint physicalCellPoint {physicalCellID,spatialPointOrdinalInDescendant};
          const MapKey mapKey {myPID,physicalCellPoint};
          double integral = 0.0;
          for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++)
          {
            const double & basisCoefficient = basisCoefficients(basisOrdinal);
            const double & basisIntegral    = _basisIntegrals(moment,pointSetNumber,basisOrdinal,spatialPointOrdinalInDescendant);
            integral += basisIntegral * basisCoefficient * cellHeight;
          }
          integralContributionsFromMyPhaseSpaceCells[recipientPID][mapKey] += integral;
        }
      }
    }
    
    map<MapKey, MapValue> integralContributionsToMyPhysicalCells;
    MPIWrapper::sendDataMaps<MapKey,MapValue>(_spatialMesh->Comm(), integralContributionsFromMyPhaseSpaceCells, integralContributionsToMyPhysicalCells);
    
    const auto & myPhysicalCells = _spatialMesh->cellIDsInPartition();
    _physicalValuesAtQuadraturePoints.resize(myPhysicalCells.size(),_numMoments,_numPoints1D);
    _physicalValuesAtQuadraturePoints.initialize(0);
    int physicalCellOrdinal = 0;
    for (auto & physicalCellID : myPhysicalCells)
    {
      _physicalCellIDToLocalCellOrdinal[physicalCellID] = physicalCellOrdinal;
      physicalCellOrdinal++;
    }
    for (const auto & entry : integralContributionsToMyPhysicalCells)
    {
      const MapKey & mapKey                       = entry.first;
      const MapValue & integralContribution       = entry.second;
      const PhysicalCellPoint & physicalCellPoint = mapKey.second;
      const GlobalIndexType & physicalCellID      = physicalCellPoint.first;
      const int & pointOrdinal                    = physicalCellPoint.second;
      const auto & physicalCellOrdinalEntryIt     = _physicalCellIDToLocalCellOrdinal.find(physicalCellID);
      TEUCHOS_TEST_FOR_EXCEPTION(physicalCellOrdinalEntryIt == _physicalCellIDToLocalCellOrdinal.end(), std::invalid_argument, "Physical cell ID not found");
      const int & physicalCellOrdinal             = physicalCellOrdinalEntryIt->second;
      _physicalValuesAtQuadraturePoints(physicalCellOrdinal, moment, pointOrdinal) += integralContribution;
    }
  }
  
  void updateValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & physicalCellIDs = basisCache->cellIDs();
    
    const int moment = 0;
    int cellOrdinalInBasisCache = 0;
    const int numPoints = values.dimension(1);
    for (const auto & physicalCellID : physicalCellIDs)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_physicalCellIDToLocalCellOrdinal.find(physicalCellID) == _physicalCellIDToLocalCellOrdinal.end(),
                                 std::invalid_argument, "physical cellID not found");
      const int cellOrdinalInPhysicalValuesContainer = _physicalCellIDToLocalCellOrdinal[physicalCellID];
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        values(cellOrdinalInBasisCache,pointOrdinal) = _physicalValuesAtQuadraturePoints(cellOrdinalInPhysicalValuesContainer, moment, pointOrdinal);
      }
      cellOrdinalInBasisCache++;
    }
  }
};

/*
 Worth noting that VelocitySpaceExtrusion codifies an inefficient use of memory:
 it is simply duplicating values in the velocity dimension.  It would be better to
 avoid the duplication by adding some facilities to Function; I like the idea of
 adding a method
    bool variesInDimension(int dim);
 and then having some mechanism by which the arrays that Functions that are constant
 in some dimension can only specify values for the dimensions in which they vary.
 Doing this more or less requires reworking the way points are stored, though, to
 express tensor-product points.  We have some loose plans to do something like this
 in Intrepid2.
 */
class VelocitySpaceExtrusion : public TFunction<double>
{
  FunctionPtr _g_1D; // arbitrary function in physical space -- can depend on mesh/physical solution
  
  Teuchos::RCP<PhasePhysicalMeshMap1D1V> _phaseSpacePhysicalMeshMap;
public:
  VelocitySpaceExtrusion( FunctionPtr g_1D, Teuchos::RCP<PhasePhysicalMeshMap1D1V> phaseSpacePhysicalMeshMap)
  :
  Function(g_1D->rank()),
  _g_1D(g_1D),
  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported right now");
  }
  
  void updateValues()
  {
    auto & phaseSpaceCellIDs = _phaseSpacePhysicalMeshMap->mesh2D()->cellIDsInPartition();
    
    const set<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    
    const std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDs.begin(), physicalCellIDs.end());
    _g_1D->importCellData(physicalCellIDsVector);
  }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
    const int rank = MPIWrapper::CommWorld()->MyPID();
    
    TEUCHOS_TEST_FOR_EXCEPTION(basisCache == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    TEUCHOS_TEST_FOR_EXCEPTION(_phaseSpacePhysicalMeshMap == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    
    bool isSideCache = (basisCache->getSideIndex() != -1);
    int physicalSpaceSideOrdinal;
    if (!isSideCache)
    {
      physicalSpaceSideOrdinal = -1;
    }
    else // isSideCache == true
    {
      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
      // what follows assumes 1D1V with quad elements; let's throw an exception if we're not on a quad
      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != CellTopology::quad()->getKey(), std::invalid_argument, "Only phase-space quads supported right now");
      if ((phaseSpaceSideOrdinal == 0) || (phaseSpaceSideOrdinal == 2))
      {
        // this side does not map to a physical side; runs parallel to physical space
        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
        if (_g_1D->boundaryValueOnly())
        {
          // if _g_1D is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
          values.initialize(0.0);
          return;
        }
        else
        {
          // we can and should evaluate _g_1D on the spatial element interior
          physicalSpaceSideOrdinal = -1;
        }
      }
      else if (phaseSpaceSideOrdinal == 1)
      {
        physicalSpaceSideOrdinal = 1;
      }
      else
      {
        physicalSpaceSideOrdinal = 0;
      }
    }
    
    using std::map;
    using std::pair;
    using std::set;
    using std::vector;
    
    // determine physical cell IDs we need to compute on
    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
    
    const bool testVsTest = basisCache->testVsTest();
    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();

    int basisCache1DQuadratureEnrichment;
    if (testVsTest)
    {
      // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
    }
    else
    {
      // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
    }
    
    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
    for ( auto & physicalCellID : physicalCellIDs )
    {
      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
    }
    
    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
    for ( auto & physicalCellID : physicalCellIDs )
    {
      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
      auto physicalBasisCache = basisCachePair.first;
      auto offsets            = basisCachePair.second;
      
      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
      
      TEUCHOS_TEST_FOR_EXCEPTION(basisCacheToUse == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
      
      const int numCells  = 1;
      const int numPoints = basisCacheToUse->getRefCellPoints().dimension(0);
      Camellia::FieldContainer<double> values(numCells,numPoints);
      
      _g_1D->values(values, basisCacheToUse);
      physicalValuesMap[physicalCellID] = values;
      physicalPointsMap[physicalCellID] = basisCacheToUse->getPhysicalCubaturePoints();
    }

    if (phaseSpaceCellIDs.size() == 0)
    {
      return;
    }
 
    const int numPhaseSpacePoints = basisCache->getRefCellPoints().dimension(0);
    
    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
    {
      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
      
      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
      
      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
      
      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
      {
        auto physicalCellID             = physicalPointIterator.physicalCellID();
        auto & values1D                 = physicalValuesMap[physicalCellID];
        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
        double & value                  = values1D(0, pointOrdinalInPhysicalCell);
        
        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
        {
          const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
          values(phaseSpaceCellOrdinal,tensorPointOrdinal) = value;
        }
      }
    }
  }
  
  bool boundaryValueOnly() override
  {
    return _g_1D->boundaryValueOnly();
  }
};

Boltzmann1D1VFormulation::Boltzmann1D1VFormulation(Boltzmann1D1VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                                                   const double x0, const double xWidth, const int xElementsCoarse,
                                                   const double v0, const double vWidth, const int vElementsCoarse,
                                                   const int H1Order, const int delta_k, ERKTableauPtr erkTableau,
                                                   const int numUniformRefinements)
:
_formulationChoice(formulationChoice)
{
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At start of Boltzmann1D1VFormulation construction, memory HWM:      " << memoryUsageInGigabytes << " GB.\n";
//  }
  
  const int spaceDim = 1;
//  const int velocityDim = 1;
  _delta_k = delta_k;
  
  SolutionStagePtr solnStage = Teuchos::null;
  _haveTableau = (erkTableau != Teuchos::null);
  if (_haveTableau)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(this->usesImplicitSolve(), std::invalid_argument, "ERK Tableau is only supported for explicit formulations...");
    solnStage = SolutionStage::solutionStage(erkTableau, dt);
    _dt = solnStage->dt();
  }
  else
  {
    _dt   = ParameterFunction::parameterFunction(dt);
  }
  
  bool useConformingTraces = true; // governs E_hat space (though in 1D this is just a point value regardless)
  
  _q    = ParameterFunction::parameterFunction(q);
  _eps0 = ParameterFunction::parameterFunction(eps0);
  _m    = ParameterFunction::parameterFunction(m);
  
  _n0   = ParameterFunction::parameterFunction(0.0); // will be set/updated during solve
  
  const bool divideIntoTriangles = false;
  PeriodicBCPtr xLeftToRight     = PeriodicBC::xIdentification(x0, x0+xWidth);
  std::vector<PeriodicBCPtr> quadMeshPeriodicBCs {xLeftToRight};
  auto quadMeshTopology          = MeshFactory::quadMeshTopology(xWidth,vWidth,xElementsCoarse,vElementsCoarse,divideIntoTriangles,x0,v0,quadMeshPeriodicBCs);
  bool usePeriodicBCs            = true;
  auto intervalMeshTopology      = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElementsCoarse, usePeriodicBCs);
  
  for (int i=0; i<numUniformRefinements; i++)
  {
    const auto & activeQuadCells = quadMeshTopology->getActiveCellIndicesGlobal();
    auto quadRefPattern = RefinementPattern::regularRefinementPatternQuad();
    for (const auto & activeQuadCellIndex : activeQuadCells)
    {
      quadMeshTopology->refineCell(activeQuadCellIndex, quadRefPattern, quadMeshTopology->nextCellID());
    }
    const auto & activeIntervalCells = intervalMeshTopology->getActiveCellIndicesGlobal();
    auto lineRefPattern = RefinementPattern::regularRefinementPatternLine();
    for (const auto & activeIntervalCellIndex : activeIntervalCells)
    {
      intervalMeshTopology->refineCell(activeIntervalCellIndex, lineRefPattern, intervalMeshTopology->nextCellID());
    }
  }
  
  const bool implicit  = usesImplicitSolve();
  const bool l2Trace   = hasL2Trace();
  const bool ultraweak = _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_H1_TRACE
                      || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE;
  
  if (ultraweak)
  {
    Space tauSpace    = (spaceDim > 1)      ? HDIV : HGRAD;
    Space E_hat_space = useConformingTraces ? HGRAD : L2;
    Space ESpace      = (spaceDim > 1)      ? VECTOR_L2 : L2;

    // fields
    VarPtr f; // Boltzmann
    VarPtr E; // Gauss

    // traces
    VarPtr E_hat, t_n_hat, f_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

    // tests
    VarPtr w;   // Boltzmann test function
    VarPtr tau; // Gauss

    VarFactoryPtr gaussVF = VarFactory::varFactory();
    
    E     = gaussVF->fieldVar(S_E, ESpace);
    if (spaceDim == 1)
    {
      auto fluxWeight = Function::normal_1D() * Function::sideParity(); // for "termTraced" argument below
      E_hat = gaussVF->fluxVar(S_E_HAT, E * fluxWeight, E_hat_space);
    }
    else
    {
      E_hat = gaussVF->traceVar(S_E_HAT, E, E_hat_space);
    }
    tau   = gaussVF->testVar(S_TAU, tauSpace);
    
    _gaussBF = BF::bf(gaussVF);
    if (spaceDim == 1)
      _gaussBF->addTerm(-E, tau->dx());
    else
      _gaussBF->addTerm(-E, tau->grad());
    
    _gaussBF->addTerm(E_hat, tau);
    MeshPtr gaussMesh = MeshFactory::minRuleMesh(intervalMeshTopology, _gaussBF, H1Order, delta_k);
    gaussMesh->globalDofAssignment()->setAllowMeshTopologyPruning(false);
    _gaussSoln = Solution::solution(_gaussBF, gaussMesh);
    auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
    
    FunctionPtr dt = FunctionPtr(_dt);
    FunctionPtr E_soln = Function::solution(E, gaussSolnWeakPtr);
    
    FunctionPtr v = Function::yn(1); // we are using the y dimension for velocity space
    FunctionPtr m = FunctionPtr(_m);
    
    VarFactoryPtr boltzmannVF = VarFactory::varFactory();
    
    f = boltzmannVF->fieldVar(S_F);
    w = boltzmannVF->testVar(S_W, HGRAD); // v lies in HGRAD of the whole 1D1V space
    
    if (l2Trace)
    {
      t_n_hat = boltzmannVF->fluxVar(S_T_N_HAT);
    }
    else
    {
      f_hat = boltzmannVF->traceVar(S_F_HAT);
    }
    
    _boltzmannBF = BF::bf(boltzmannVF);
    
    map<int,int> trialOrderEnhancements;
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1;
    }
    
    MeshPtr boltzmannMesh = Teuchos::rcp( new Mesh(quadMeshTopology, _boltzmannBF, H1Order, delta_k, trialOrderEnhancements) );
    
    _phaseSpacePhysicalMeshMap = Teuchos::rcp( new PhasePhysicalMeshMap1D1V( gaussMesh, boltzmannMesh ) );
    auto meshMapWeakPtr        = Teuchos::rcp( _phaseSpacePhysicalMeshMap.get(), false);
    
    _E_extruded = Teuchos::rcp( new VelocitySpaceExtrusion(E_soln, meshMapWeakPtr) );

    _boltzmannBF->addTerm(f / _dt, w);
    
    if (l2Trace)
    {
      _boltzmannBF->addTerm(t_n_hat, w); // for explicit, t_n should be understood as the trace of the solution at prior time…
    }
    else
    {
      FunctionPtr n = Function::normal();
      _boltzmannBF->addTerm((v * n->x() + (q/m) * _E_extruded * n->y()) * f_hat, w); // for explicit, f_hat should be understood as the trace of the solution at prior time…
    }

    if (implicit)
    {
      _boltzmannBF->addTerm(-v * f, w->dx());
      _boltzmannBF->addTerm(-(q/m)*_E_extruded*f, w->dy());
    }
    
    _boltzmannSoln     = Solution::solution(_boltzmannBF,boltzmannMesh);
    if (_haveTableau)
    {
      // then previous solution is in the same Solution object as the "current" solution
      _boltzmannSoln->setSolutionStage(solnStage);
      _boltzmannSolnPrev = _boltzmannSoln;
      // register for notifications
      auto thisPtr = Teuchos::rcp(this,false);
      solnStage->addObserver(thisPtr);
    }
    else
    {
      _boltzmannSolnPrev = Solution::solution(_boltzmannBF,boltzmannMesh);
    }
    
    _f_velocityIntegral = Teuchos::rcp( new VelocitySpace1DIntegral(_gaussSoln->mesh(), _boltzmannSoln, f, _spatialCellIDToPhaseSpaceCellIDs, meshMapWeakPtr) );
    
    // set RHS for the two problems
    auto gaussRHS = RHS::rhs();
    // casts to allow FunctionPtr overloads to be used:
    auto qFunction = FunctionPtr(_q);
    auto mFunction = FunctionPtr(_m);
    auto epsFxn = FunctionPtr(_eps0);
    auto n0Fxn  = FunctionPtr(_n0); // _n0 will be initialized when initial values are specified
    auto f_velocityIntegralFxn = FunctionPtr(_f_velocityIntegral);
    gaussRHS->addTerm(qFunction / epsFxn * (f_velocityIntegralFxn - n0Fxn) * tau);
    _gaussSoln->setRHS(gaussRHS);
    
    auto boltzmannRHS = RHS::rhs();
    auto boltzmannPrevSolnWeakPtr = Teuchos::rcp(_boltzmannSolnPrev.get(), false);
    
    const bool weightFluxesBySideParity = true;
    const SolutionComponent solnComponent = STANDARD_SOLUTION;
    const int stageOrdinal = _haveTableau ? -1 : 0;
    auto f_previous = Function::solution(f, boltzmannPrevSolnWeakPtr, weightFluxesBySideParity,
                                         solnComponent, stageOrdinal, std::string("k"));
    
    boltzmannRHS->addTerm(f_previous / dt * w);
    
    if (!implicit)
    {
      boltzmannRHS->addTerm(v * f_previous * w->dx());
      boltzmannRHS->addTerm((q/m)*_E_extruded*f_previous * w->dy());
    }
    
    _boltzmannSoln->setRHS(boltzmannRHS);
    
    // set the test-space inner products.
    _gaussSoln->setIP(_gaussBF->naiveNorm(spaceDim));
    
    gaussMesh->registerSolution(_gaussSoln);
    boltzmannMesh->registerSolution(_boltzmannSoln);
    boltzmannMesh->registerSolution(_boltzmannSolnPrev);
    
    IPPtr boltzmannTestNorm = IP::ip();
    const bool useSimpleL2Norm = false;
    if (useSimpleL2Norm)
    {
      boltzmannTestNorm->addTerm(w);
      std::cout << "NOTE: using simple L^2 norm for explicit Boltzmann...\n";
    }
    else
    {
      const bool useExperimentalNormForExplicit = false;
      if (!useExperimentalNormForExplicit || implicit)
      {
        // experiments so far suggest that the norm below is the most robust for explicit time-stepping
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v * w->dx() + (q/m)*_E_extruded * w->dy());
      }
      else
      {
        std::cout << "NOTE: using experimental test norm for explicit Boltzmann...\n";
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v * w->dx() + 1e-3 * (q/m)*_E_extruded * w->dy());
      }
    }
    _boltzmannSoln->setIP(boltzmannTestNorm);
    
    // set BCs
    // we are periodic in x; set homogeneous conditions on the v boundaries
    auto boltzmannBC = BC::bc();
    auto traceVar = l2Trace ? t_n_hat : f_hat;
    boltzmannBC->addDirichlet(traceVar, SpatialFilter::matchingY(v0),        Function::zero());
    boltzmannBC->addDirichlet(traceVar, SpatialFilter::matchingY(v0+vWidth), Function::zero());
    _boltzmannSoln->setBC(boltzmannBC);
    
    auto gaussBC = BC::bc();
    gaussBC->addDirichlet(E_hat, SpatialFilter::matchingX(x0), Function::zero());
    _gaussSoln->setBC(gaussBC);
    
    // static condensation does not work properly with GMG due to details of VelocitySpaceExtrusion; it also does not appear to speed things up for the direct solve
//    if (!_haveTableau)
//    {
//      _boltzmannSoln->setUseCondensedSolve(true); // should work without issue given no tableau; we need to test with tableaus before enabling with those
//    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported Boltzmann1D1VFormulationChoice");
  }
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At conclusion of Boltzmann1D1VFormulation construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//  }
}

MeshPtr Boltzmann1D1VFormulation::boltzmannMesh()
{
  return _boltzmannSoln->mesh();
}

SolutionPtr Boltzmann1D1VFormulation::boltzmannPreviousSolution()
{
  return _boltzmannSolnPrev;
}

SolutionPtr Boltzmann1D1VFormulation::boltzmannSolution()
{
  return _boltzmannSoln;
}

FunctionPtr Boltzmann1D1VFormulation::E_extruded()
{
  return _E_extruded;
}

VarPtr Boltzmann1D1VFormulation::f()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr Boltzmann1D1VFormulation::f_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

MeshPtr Boltzmann1D1VFormulation::gaussMesh()
{
  return _gaussSoln->mesh();
}

SolutionPtr Boltzmann1D1VFormulation::gaussSolution()
{
  return _gaussSoln;
}

bool Boltzmann1D1VFormulation::hasL2Trace() const {
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE;
}

VarPtr Boltzmann1D1VFormulation::E()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->trialVar(S_E);
}

VarPtr Boltzmann1D1VFormulation::E_hat()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->trialVar(S_E_HAT);
}

const std::vector< std::vector<int> > & Boltzmann1D1VFormulation::iterationCounts() const
{
  return _iterationCounts;
}

const std::map<GlobalIndexType, std::set<GlobalIndexType> > & Boltzmann1D1VFormulation::phaseSpaceCellIDToSpatialCellIDs()
{
  return _phaseSpaceCellIDToSpatialCellID;
}

Teuchos::RCP<PhasePhysicalMeshMap1D1V> Boltzmann1D1VFormulation::phaseSpacePhysicalMeshMap()
{
  return _phaseSpacePhysicalMeshMap;
}

void Boltzmann1D1VFormulation::setInitialConditions(FunctionPtr f0)
{
  _boltzmannMeshMeasure = Function::constant(1.0)->integrate(_boltzmannSoln->mesh());
  
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[this->f()->ID()] = f0;
  if ( !hasL2Trace() )
  {
    projectionMap[this->f_hat()->ID()] = f0;
  }
  
  const int solutionOrdinal = 0;
  _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  if (!_haveTableau)
  {
    _boltzmannSolnPrev->setSolution(_boltzmannSoln);
  }
  
  _gaussMeshMeasure = Function::constant(1.0)->integrate(_gaussSoln->mesh());

  updateElectricField();
  
  // set initial tn value:
  auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
  
  if (hasL2Trace())
  {
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr q(_q);
    FunctionPtr m(_m);
    FunctionPtr v = Function::yn(1);
    FunctionPtr n = Function::normal();
    FunctionPtr tn_0 = f0 * (v * n->x() + (q/m) * _E_extruded * n->y()) * sgn;
    projectionMap[this->t_n_hat()->ID()] = tn_0;
    _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    if (!_haveTableau)
    {
      _boltzmannSolnPrev->setSolution(_boltzmannSoln);
    }
  }
  _t = 0;
}

void Boltzmann1D1VFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void Boltzmann1D1VFormulation::setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator, bool gateRefinements)
{
  _refinementThreshold = refinementThreshold;
  _maxElements = maxElements;
  _maxRefinementLevel = maxRefinementLevel;
  _lastRefinementMaxElementError = -1.0;
  _gateRefinements = gateRefinements;
  
  if (_refStrategy == Teuchos::null)
  {
    _refStrategy = RefinementStrategy::energyErrorRefinementStrategy(_boltzmannSoln, _refinementThreshold);
  }
  _refStrategy->setRelativeErrorThreshold(_refinementThreshold);
  _refStrategy->setMaxElements(_maxElements);
  _refStrategy->setMaximumRefinementLevel(_maxRefinementLevel);
  _refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = _boltzmannSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  _refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
}

void Boltzmann1D1VFormulation::setStage(const int stageNumber)
{
  updateElectricField();
}

void Boltzmann1D1VFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

void Boltzmann1D1VFormulation::setSubstepsForImplicit(int numSubsteps)
{
  _numSubstepsForImplicit = numSubsteps;
}

void Boltzmann1D1VFormulation::takeTimeStep()
{
  // for implicit solves, we cycle through the f solve and the E solve a few times;
  // for explicit, we do each solve just once per time step.
  const bool implicit = _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
  int numSubsteps = implicit ? _numSubstepsForImplicit : 1;
//  int numSubsteps = implicit ? 1 : 1; // 1 for implicit for faster debugging
  
  const bool useAdaptivity = _refStrategy != Teuchos::null;
  
  if (_haveTableau && useAdaptivity)
  {
    // the logic below only does adaptivity for the non-tableau (implicit, right now) case; probably user mis-specified something
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "adaptivity is only supported for implicit time-stepping");
  }
  
  bool useGMG = true;
  if (useGMG && (_gmgSolver == Teuchos::null))
  {
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "Before GMG construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//    }
    
    int kCoarse = 0;
    const double cgTol = 1e-6;
    const int maxIters = 10000;

    bool useCondensedSolve = _boltzmannSoln->usesCondensedSolve();

    //    auto multigridStrategy = GMGOperator::V_CYCLE;
    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ; // CAMELLIA_ADDITIVE_SCHWARZ defines cells as blocks; Ifpack's defines all cells on the MPI rank as a block, making MPI scaling less predictable.  (Camellia's implementation is likely less well-optimized, though.)
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_boltzmannSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_boltzmannSoln->mesh()};
    }
    
    _gmgSolver = Teuchos::rcp( new GMGSolver(_boltzmannSoln, meshes, maxIters, cgTol, multigridStrategy,
                                             Solver::getDirectSolver(true), useCondensedSolve) );
    
    _gmgSolver->setSmootherType(smootherChoice);
    
    _gmgSolver->setAztecOutput(_aztecOutputLevel);
    
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "After GMG construction, memory HWM:  " << memoryUsageInGigabytes << " GB.\n";
//    }
  }
  
  std::vector<int> substepIterationCounts(numSubsteps);
  
  for (int substepOrdinal=0; substepOrdinal < numSubsteps; substepOrdinal++)
  {
    // solve for f:
    if (_gmgSolver == Teuchos::null)
    {
      _boltzmannSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      _boltzmannSoln->solve(_gmgSolver);
      substepIterationCounts[substepOrdinal] = _gmgSolver->iterationCount();
    }
    
    const bool correctConservation = false;
    
    if (correctConservation)
    {
      // EXPERIMENT: check "conservation" and correct
      auto mesh = _boltzmannSoln->mesh();
      std::map<int, FunctionPtr> testMap;
      testMap[this->w()->ID()] = Function::constant(1.0);
      auto trialFunctional = _boltzmannBF->trialFunctional(testMap);
      std::map<int, FunctionPtr> trialMap;
      auto f_soln = Function::solution(this->f(), _boltzmannSoln, std::string("k+1"));
      trialMap[this->f()->ID()] = f_soln;
      auto lhsFunction = trialFunctional->evaluate(trialMap);
      
      auto rhsLT = _boltzmannSoln->rhs()->linearTerm();
      auto rhsFunctionBoundaryPart = rhsLT->getBoundaryOnlyPart()->evaluate(testMap);
      auto rhsFunctionVolumePart = rhsLT->getNonBoundaryOnlyPart()->evaluate(testMap);
      
      double rhsVolumeIntegral = rhsFunctionVolumePart->integrate(mesh);
      double rhsTraceIntegral  = rhsFunctionBoundaryPart->integrate(mesh);
      double rhsIntegral = rhsVolumeIntegral + rhsTraceIntegral;
      
      double lhsIntegral = lhsFunction->integrate(mesh);
      double residualIntegral = lhsIntegral - rhsIntegral;
      
      double error = _dt->getConstantValue() * residualIntegral;
      std::cout << "Conservation error before correction: " << error << std::endl;
      double correctionFactor = rhsIntegral / lhsIntegral;
      std::cout << "correctionFactor = 1 + " << (correctionFactor-1.0) << std::endl;
      FunctionPtr f_soln_corrected = f_soln * correctionFactor;
      
      std::map<int,FunctionPtr> correctedSolnMap;
      correctedSolnMap[this->f()->ID()] = f_soln_corrected;
      _boltzmannSoln->projectOntoMesh(correctedSolnMap);
      
      lhsIntegral      = lhsFunction->integrate(mesh);
      
      rhsVolumeIntegral = rhsFunctionVolumePart->integrate(mesh);
      rhsTraceIntegral  = rhsFunctionBoundaryPart->integrate(mesh);
      rhsIntegral = rhsVolumeIntegral + rhsTraceIntegral;
      
      error = _dt->getConstantValue() * (lhsIntegral  - rhsIntegral);
      std::cout << "Conservation error after correction: " << error << std::endl;
    }
    if ( !_haveTableau )
    {
      if (substepOrdinal == numSubsteps-1)
      {
        const bool useAdaptivity = _refStrategy != Teuchos::null;
        
        if (useAdaptivity)
        {
          bool shouldRefine;
          if (_gateRefinements)
          {
            // we compute error indicator first; if no refinement-eligible element has error greater than what would have been refined last time, then we don't do a refinement step now.
            
            auto localCellsToExclude = _refStrategy->localCellsIneligibleForRefinement();
            
            ErrorIndicatorPtr errorIndicator = _refStrategy->errorIndicator();
            errorIndicator->measureError();
            const double maxError = errorIndicator->maxError(localCellsToExclude);
            
            shouldRefine = maxError > _refinementThreshold * _lastRefinementMaxElementError;
            
            if (shouldRefine)
            {
              _lastRefinementMaxElementError = maxError;
            }
          }
          else
          {
            shouldRefine = true;
          }
          
          if (shouldRefine)
          {
            // for now, print refinement info to console on rank 0.
            const bool printToConsole = _boltzmannSoln->mesh()->Comm()->MyPID() == 0;
            const bool doUnrefinements = (_maxElements < std::numeric_limits<int>::max());
            _refStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
          }
        }
        
        // last substep
        _boltzmannSolnPrev->setSolution(_boltzmannSoln);
      }
      // if we have a tableau, then electric field will be updated when the solution stage is set.
      updateElectricField();
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  
  _t += _dt->getConstantValue();
}

const std::map<GlobalIndexType, std::vector<GlobalIndexType> > & Boltzmann1D1VFormulation::spatialCellIDToPhaseSpaceCellIDsMap()
{
  return _spatialCellIDToPhaseSpaceCellIDs;
}

void Boltzmann1D1VFormulation::updateElectricField()
{
  // update f integral, n0
  _f_velocityIntegral->updateValues();
  double f_integral   = _f_velocityIntegral->integrate(_gaussSoln->mesh());
  double n0 = f_integral / _gaussMeshMeasure;
  _n0->setValue(n0);
  
  // solve for E, E_hat:
  _gaussSoln->solve();
  
  // post-process to make E have a zero mean:
  FunctionPtr E_soln = Function::solution(E(), _gaussSoln);
  double E_avg = E_soln->integrate(_gaussSoln->mesh()) / _gaussMeshMeasure;
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[E()->ID()] = E_soln - E_avg;
  projectionMap[E_hat()->ID()] = E_soln - E_avg;
  _gaussSoln->projectOntoMesh(projectionMap);
  
  static_cast<VelocitySpaceExtrusion*>(_E_extruded.get())->updateValues();
}

bool Boltzmann1D1VFormulation::usesImplicitSolve() const
{
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
}

// test variables:
VarPtr Boltzmann1D1VFormulation::w()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

VarPtr Boltzmann1D1VFormulation::tau()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->testVar(S_TAU);
}

VarPtr Boltzmann1D1VFormulation::t_n_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

void testVelocitySpace1DIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 1.0;
  
  const int xElements = 4;
  const int vElements = 4;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  
  using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };
  
  set<GlobalIndexType> boltzmannCellsToRefine;
  if (doRefinement)
  {
    boltzmannCellsToRefine = {0,5,8};
  }
  
  for (auto formulationChoice : formulationChoices)
  {
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, Teuchos::null);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y  = Function::yn(1);
    FunctionPtr xy = x * y;
    // integral of xy dy = x * y^2 / 2
    // we integrate from v0 to v0 + vWidth
    const double expectedIntegralY = (0.5 * (v0 + vWidth) * (v0 + vWidth) - 0.5 * v0 * v0);
    FunctionPtr expectedIntegral = x * expectedIntegralY;
    const double expectedIntegralXY = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0) * expectedIntegralY;
    const double actualIntegralXY   = expectedIntegral->integrate(form.gaussMesh());
    
    TEST_FLOATING_EQUALITY(expectedIntegralXY, actualIntegralXY, tol);
    
    if (boltzmannCellsToRefine.size() > 0)
    {
      auto boltzmannMesh = form.boltzmannMesh();
      boltzmannMesh->hRefine(boltzmannCellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    }
    
    auto f = form.f();
    std::map<int, FunctionPtr> projectionMap;
    projectionMap[f->ID()] = xy;
    
    auto boltzmannSoln = form.boltzmannSolution();
    const int solutionOrdinal = 0;
    boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto gaussMesh = form.gaussMesh();
    auto spatialToPhaseSpaceMap = form.spatialCellIDToPhaseSpaceCellIDsMap();
    
    auto velocitySpaceIntegral = Teuchos::rcp(new VelocitySpace1DIntegral(gaussMesh, boltzmannSoln, f, spatialToPhaseSpaceMap, form.phaseSpacePhysicalMeshMap()) );
    FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
    
    velocitySpaceIntegral->QuadraturePointFunction::updateValues();
    
    double magnitudeExpected = expectedIntegral->l2norm(gaussMesh);
    double err_l2 = err->l2norm(gaussMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( VelocitySpace1DEquispacedIntegral_Boltzmann1D1VFormulation, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpace1DEquispacedIntegral_Boltzmann1D1VFormulation, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

class SideRestriction : public TFunction<double>
{
  std::vector<int> _sidesToMatch;
public:
  SideRestriction(std::vector<int> sidesToMatch)
  :
  Function(0),
  _sidesToMatch(sidesToMatch)
  {}
  
  bool boundaryValueOnly() override
  {
    return true;
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int sideOrdinal = basisCache->getSideIndex();
    bool sideMatches = false;
    for (auto sideToMatch : _sidesToMatch)
    {
      if (sideToMatch == sideOrdinal)
      {
        sideMatches = true;
        break;
      }
    }
    const double value = sideMatches ? 1.0 : 0.0;
    values.initialize(value);
  }
};

void testVelocityExtrusion(bool volumeVariable, bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double v0     = 0.0;
  double vWidth = 20.0;
  
  const int xElements = 2;
  const int vElements = 2;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  
  using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };
  
  for (auto formulationChoice : formulationChoices)
  {
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, Teuchos::null);
    
    FunctionPtr x  = Function::xn(1);
    
    FunctionPtr soln_1D;
    
    auto gaussSoln = form.gaussSolution();
    std::map<int, FunctionPtr> projectionMap;
    if (volumeVariable)
    {
      auto E = form.E();
      projectionMap[E->ID()] = x;
      soln_1D = Function::solution(E, gaussSoln);
    }
    else
    {
      auto E     = form.E();
      auto E_hat = form.E_hat();
      projectionMap[E->ID()] = x * (xWidth - x);
      projectionMap[E_hat->ID()] = x * (xWidth - x); // for skeleton values, due to periodic BCs, want something that agrees on left/right side: having zeros there makes this simpler
      soln_1D = Function::solution(E_hat, gaussSoln, false); // false: don't weight by parity (makes uniquely-valued)
    }
    
    TEST_ASSERT(soln_1D->boundaryValueOnly() == !volumeVariable);
    
    auto gaussMesh = form.gaussMesh();
    auto boltzmannMesh = form.boltzmannMesh();
    
    auto meshMap = Teuchos::rcp( new PhasePhysicalMeshMap1D1V( gaussMesh, boltzmannMesh ) );
    auto extrusion = Teuchos::rcp( new VelocitySpaceExtrusion(soln_1D, meshMap) );
    
    if (doRefinement)
    {
      const set<GlobalIndexType> cellIDs_2D {0,1};
      boltzmannMesh->hRefine(cellIDs_2D, RefinementPattern::regularRefinementPatternQuad());
    }
    
    const int solutionOrdinal =0;
    gaussSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    extrusion->updateValues();
    
    FunctionPtr expected;
    if (volumeVariable)
    {
      expected = x;
    }
    else
    {
      // restrict x to the spatial sides of the mesh skeleton
      std::vector<int> spatialSides {1,3}; // left and right sides of the quad
      FunctionPtr sideRestriction = Teuchos::rcp(new SideRestriction(spatialSides) );
      expected = x * (xWidth - x) * sideRestriction;
    }
    
    FunctionPtr err = FunctionPtr(extrusion) - expected;
    
  //  double magnitudeActual = extrusion->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
    
    double magnitudeExpected = expected->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
    double err_l2 = err->l2norm(boltzmannMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, SkeletonExtrusion )
{
  const bool volumeVariable = false;
  const bool doRefinement = false;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, SkeletonExtrusionWithRefinement )
{
  const bool volumeVariable = false;
  const bool doRefinement = true;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, VolumeExtrusion )
{
  const bool volumeVariable = true;
  const bool doRefinement = false;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}
