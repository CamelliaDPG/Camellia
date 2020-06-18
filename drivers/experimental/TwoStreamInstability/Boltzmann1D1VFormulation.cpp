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
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
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

int getTensorPointOrdinal(const int spatialPointOrdinal, const int velocityPointOrdinal, const int numSpatialPoints)
{
  return spatialPointOrdinal + numSpatialPoints * velocityPointOrdinal;
}

// defines a function on the spatial (1D) mesh by integrating the 1D1V function in the velocity dimension
class VelocitySpace1DEquispacedIntegral : public QuadraturePointFunction
{
  MeshPtr     _spatialMesh;
  SolutionPtr _phaseSpaceSolution;
  FunctionPtr _distributionFunction; // f from the phase space solution
  VarPtr      _f;
  
  std::map<GlobalIndexType, std::vector<GlobalIndexType> > _spatialCellIDToPhaseSpaceCellIDs;
  FieldContainer<double> _basisIntegralsAtXQuadraturePoints; // (F,Px), where F is the phase space basis ordinal -- we assume uniform cell geometry; when we relax this we'll need something more complicated, but may want to retain the ability to specify where geometry is the same...
public:
  VelocitySpace1DEquispacedIntegral(MeshPtr spatialMesh, SolutionPtr phaseSpaceSolution, VarPtr f,
                                    const std::map<GlobalIndexType, std::vector<GlobalIndexType> > &spatialCellIDToPhaseSpaceCellIDs)
  :
  QuadraturePointFunction(spatialMesh, phaseSpaceSolution->cubatureEnrichmentDegree(), 0),
  _spatialMesh(spatialMesh),
  _phaseSpaceSolution(phaseSpaceSolution),
  _f(f),
  _spatialCellIDToPhaseSpaceCellIDs(spatialCellIDToPhaseSpaceCellIDs)
  {
    _distributionFunction = Function::solution(f, _phaseSpaceSolution);
    
    // examine sample spatial cell to get spatial quadrature agree (we assume this is uniform)
    const GlobalIndexType sampleSpatialCellID = 0;
    BasisCachePtr sampleSpatialBasisCache = BasisCache::basisCacheForCell(spatialMesh, sampleSpatialCellID);
    const int spatialQuadratureDegree = sampleSpatialBasisCache->cubatureDegree();
    
    auto phaseSpaceMesh = _phaseSpaceSolution->mesh();
    auto & localCellIDs = phaseSpaceMesh->cellIDsInPartition();

    const int phaseSpaceDim = 2;
    
    if (localCellIDs.size() > 0)
    {
      // take the first cell; use this to determine basis integrals -- again, this assumes each cell has the same geometry; we'll want to change this eventually
      auto cellID = *localCellIDs.begin();
      auto cellVertices = phaseSpaceMesh->verticesForCell(cellID);
      double vMin = cellVertices[0][1];
      double vMax = vMin;
      for (auto & vertex : cellVertices)
      {
        vMin = std::min(vMin, vertex[1]);
        vMax = std::max(vMax, vertex[1]);
      }
      double cellHeight = vMax - vMin;
      
      auto fBasis = phaseSpaceMesh->getElementType(cellID)->trialOrderPtr->getBasis(f->ID());
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(phaseSpaceMesh, cellID);
      
      CubatureFactory quadratureFactory;
      auto spatialLineQuadrature = quadratureFactory.create(CellTopology::line(), spatialQuadratureDegree);
      FieldContainer<double> spatialQuadraturePoints(spatialLineQuadrature->getNumPoints(),1);
      FieldContainer<double> spatialQuadratureWeights(spatialLineQuadrature->getNumPoints());
      spatialLineQuadrature->getCubature(spatialQuadraturePoints, spatialQuadratureWeights);
      
      const int maxMoment = 1; // 1st moment: an integral of f (not v-weighted).
      int fPolyDegree = fBasis->getDegree();
      const int velocityQuadratureDegree = fPolyDegree + (maxMoment - 1);
      auto velocityLineQuadrature = quadratureFactory.create(CellTopology::line(), velocityQuadratureDegree);
      FieldContainer<double> velocityQuadraturePoints(velocityLineQuadrature->getNumPoints(),1);
      FieldContainer<double> velocityQuadratureWeights(velocityLineQuadrature->getNumPoints());
      velocityLineQuadrature->getCubature(velocityQuadraturePoints, velocityQuadratureWeights);
      
      const int numSpatialPoints    = spatialLineQuadrature->getNumPoints();
      const int numVelocityPoints   = velocityLineQuadrature->getNumPoints();
      const int numPointsPhaseSpace = numSpatialPoints * numVelocityPoints;
      
      FieldContainer<double> tensorPoints(numPointsPhaseSpace, phaseSpaceDim);
      FieldContainer<double> tensorWeights(numPointsPhaseSpace);
      for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
      {
        double x = spatialQuadraturePoints(spatialPointOrdinal,0);
        double spatialWeight = spatialQuadratureWeights(spatialPointOrdinal);
        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
        {
          double v = velocityQuadraturePoints(velocityPointOrdinal,0);
          double velocityWeight = velocityQuadratureWeights(velocityPointOrdinal);
          
          const int tensorPointOrdinal = getTensorPointOrdinal(spatialPointOrdinal, velocityPointOrdinal, numSpatialPoints);
          tensorPoints(tensorPointOrdinal, 0) = x;
          tensorPoints(tensorPointOrdinal, 1) = v;
          tensorWeights(tensorPointOrdinal) = spatialWeight * velocityWeight;
        }
      }
      
      basisCache->setRefCellPoints(tensorPoints, tensorWeights);
      auto basisValues = basisCache->getTransformedValues(fBasis, OP_VALUE);
      
      int basisCardinality = fBasis->getCardinality();
      _basisIntegralsAtXQuadraturePoints = FieldContainer<double>(basisCardinality,numSpatialPoints);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
        {
          double integral = 0.0;
          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
          {
            const int tensorPointOrdinal =  getTensorPointOrdinal(spatialPointOrdinal, velocityPointOrdinal, numSpatialPoints);
            const double & basisValue = (*basisValues)(0,basisOrdinal,tensorPointOrdinal);
            const double & velocityIntegralWeight = velocityQuadratureWeights(velocityPointOrdinal);
            integral += basisValue * velocityIntegralWeight;
          }
          _basisIntegralsAtXQuadraturePoints(basisOrdinal,spatialPointOrdinal) = integral * cellHeight / 2.0; // divide by 2 because the ref line has length 2
        }
      }
    }
  }
  
  void willUpdateValues() override
  {
    // ensure that all required elements of the phaseSpaceSolution are known locally
    // NOTE: We could reduce communication costs by doing the integrals locally and then doing MPI reduce to
    //       compute the total integrals.  What's here is easier to implement, since Solution already has an import
    //       mechanism...
    std::set<GlobalIndexType> cellsToImport;
    auto phaseSpaceMesh = _phaseSpaceSolution->mesh();
    for (auto spatialCellID : _spatialMesh->cellIDsInPartition())
    {
      auto & phaseSpaceCellIDs = _spatialCellIDToPhaseSpaceCellIDs[spatialCellID];
      cellsToImport.insert(phaseSpaceCellIDs.begin(),phaseSpaceCellIDs.end());
    }
    _phaseSpaceSolution->importSolutionForOffRankCells(cellsToImport); // Solution will check which of these are actually off-rank
  }
  
  void updateValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spatialCellIDs = basisCache->cellIDs();
    
    const int numSpatialPoints = basisCache->getRefCellPoints().dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(numSpatialPoints != _basisIntegralsAtXQuadraturePoints.dimension(1), std::invalid_argument, "quadrature point count does not match expected");
    
    FieldContainer<double> basisCoefficients;
    
    int stageOrdinal;
    if (_phaseSpaceSolution->getSolutionStage() != Teuchos::null)
    {
      stageOrdinal = -1;
//      std::cout << "updateValues() called when solutionStage has stageNumber " << _phaseSpaceSolution->getSolutionStage()->getStage() << std::endl;
    }
    else
    {
      stageOrdinal = 0;
    }
    
    for (int cellOrdinal=0; cellOrdinal<spatialCellIDs.size(); cellOrdinal++)
    {
      auto spatialCellID = spatialCellIDs[cellOrdinal];
      auto & phaseSpaceCellIDs = _spatialCellIDToPhaseSpaceCellIDs[spatialCellID];
      auto elemType = _phaseSpaceSolution->mesh()->getElementType(phaseSpaceCellIDs[0]); // we assume all element types are the same
      const int basisCardinality = elemType->trialOrderPtr->getBasis(_f->ID())->getCardinality();
      basisCoefficients.resize(basisCardinality);
      
      for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
      {
        double integral = 0.0;
        for (int phaseSpaceCellOrdinal=0; phaseSpaceCellOrdinal<phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
        {
          const auto & phaseSpaceCellID = phaseSpaceCellIDs[phaseSpaceCellOrdinal];
          _phaseSpaceSolution->getBasisCoefficientsForCell(basisCoefficients, _f->ID(), phaseSpaceCellID,
                                                           STANDARD_SOLUTION, stageOrdinal);
          for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++)
          {
            const double & basisCoefficient = basisCoefficients(basisOrdinal);
            const double & basisIntegral    = _basisIntegralsAtXQuadraturePoints(basisOrdinal,spatialPointOrdinal);
            integral += basisIntegral * basisCoefficient;
          }
        }
        values(cellOrdinal,spatialPointOrdinal) = integral;
      }
    }
  }
};

/*
// defines a function on the spatial (1D) mesh by integrating the 1D1V function in the velocity dimension
// Quite a few assumptions here:
// - Assumes equispaced 1D mesh.
// - Assumes that the coarsest level geometry agrees between the two meshes (cells coincide in the x dimension)
// - Also assumes that the spatialMesh does not change; this could be relaxed by adding unrefinements to basis integrals
// - Assumes that all phase space refinements are isotropic
// - Assumes constant polynomial order (could be anisotropic, in principle)
class VelocitySpace1DIntegral : public QuadraturePointFunction
{
  MeshPtr     _spatialMesh;
  SolutionPtr _phaseSpaceSolution;
  VarPtr      _f;
  
  // the basis integrals are laid out according to refinement level; coarsest first
  // arguments are (F + refOffset, Px) where F is the phase space field ordinal, Px is the 1D point ordinal,
  //   and refOffset is (2^(refLevel)-1 + childOrdinal) * phaseSpaceBasisCardinality.
  FieldContainer<double> _basisIntegralsAtXQuadraturePoints;
  int _refinementDepth; // the number of refinement levels currently stored in _basisIntegralsAtXQuadraturePoints
  
  // spatial cell IDs that owned phase space cells contribute to (entries are unique):
  FieldContainer<GlobalIndexType> _spatialCellsContributedTo;
  
  // lookup table for owned phase space cell; entries are indices into _spatialCellsContributedTo
  std::map<GlobalIndexType,int> _spatialCellOrdinalForPhaseSpaceCell;
  
  // _integralsAtSpatialPoints has indices (C,P) where C is a spatial cell ordinal (index into _spatialCellsContributedTo)
  FieldContainer<double> _integralsAtSpatialPoints;
public:
  VelocitySpace1DIntegral(MeshPtr spatialMesh, SolutionPtr phaseSpaceSolution, VarPtr f)
  :
  QuadraturePointFunction(spatialMesh, phaseSpaceSolution->cubatureEnrichmentDegree(), 0),
  _spatialMesh(spatialMesh),
  _phaseSpaceSolution(phaseSpaceSolution),
  _f(f)
  {
    auto phaseSpaceMesh = _phaseSpaceSolution->mesh();
    auto & localCellIDs = phaseSpaceMesh->cellIDsInPartition();
    
    if (localCellIDs.size() > 0)
    {
      // take the first cell; use this to determine basis integrals -- again, this assumes each cell has the same geometry; we'll want to change this eventually
      auto cellID = *localCellIDs.begin();
      auto cellVertices = phaseSpaceMesh->verticesForCell(cellID);
      double vMin = cellVertices[0][1];
      double vMax = vMin;
      for (auto & vertex : cellVertices)
      {
        vMin = std::min(vMin, vertex[1]);
        vMax = std::max(vMax, vertex[1]);
      }
      double cellHeight = vMax - vMin;
      bool testVsTest = false;
      bool quadratureEnrichmentDegree = phaseSpaceSolution->cubatureEnrichmentDegree();
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(phaseSpaceMesh, cellID, testVsTest, quadratureEnrichmentDegree);
      
      auto fBasis = phaseSpaceMesh->getElementType(cellID)->trialOrderPtr->getBasis(f->ID());
      auto basisValues = basisCache->getTransformedValues(fBasis, OP_VALUE);
      
      const int quadratureDegree = basisCache->cubatureDegree();
      CubatureFactory quadratureFactory;
      auto lineQuadrature = quadratureFactory.create(CellTopology::line(), quadratureDegree);
      FieldContainer<double> lineQuadraturePoints(lineQuadrature->getNumPoints(),1);
      FieldContainer<double> lineQuadratureWeights(lineQuadrature->getNumPoints());
      lineQuadrature->getCubature(lineQuadraturePoints, lineQuadratureWeights);
      
      const int numSpatialPoints    = lineQuadrature->getNumPoints();
      const int numVelocityPoints   = lineQuadrature->getNumPoints();
      const int numPointsPhaseSpace = numSpatialPoints * numVelocityPoints;
      
      TEUCHOS_TEST_FOR_EXCEPTION(numPointsPhaseSpace != basisCache->getRefCellPoints().dimension(0), std::invalid_argument, "Number of quadrature points does not match expected");
      
      int basisCardinality = fBasis->getCardinality();
      _basisIntegralsAtXQuadraturePoints = FieldContainer<double>(basisCardinality,numSpatialPoints);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
        {
          double integral = 0.0;
          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
          {
            // note that tensorPointOrdinal determination here does make an assumption about the point ordering
            const int tensorPointOrdinal = spatialPointOrdinal + numSpatialPoints * velocityPointOrdinal;
            const double & basisValue = (*basisValues)(0,basisOrdinal,tensorPointOrdinal);
            const double & velocityIntegralWeight = lineQuadratureWeights(velocityPointOrdinal);
            integral += basisValue * velocityIntegralWeight;
          }
          _basisIntegralsAtXQuadraturePoints(basisOrdinal,spatialPointOrdinal) = integral * cellHeight / 2.0; // divide by 2 because the ref line has length 2
        }
      }
    }
  }
  
  void willUpdateValues() override
  {
    // Here, we need to do all MPI communication; updateValues() is called only on the spatial mesh;
    // here, we will compute integral contributions and communicate these to spatial cell owners.
  }
  
  void updateValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spatialCellIDs = basisCache->cellIDs();
    
    const int numSpatialPoints = basisCache->getRefCellPoints().dimension(0);
    
    FieldContainer<double> basisCoefficients;
    
    int stageOrdinal;
    if (_phaseSpaceSolution->getSolutionStage() != Teuchos::null)
    {
      stageOrdinal = -1;
      //      std::cout << "updateValues() called when solutionStage has stageNumber " << _phaseSpaceSolution->getSolutionStage()->getStage() << std::endl;
    }
    else
    {
      stageOrdinal = 0;
    }
    
    
    for (int cellOrdinal=0; cellOrdinal<spatialCellIDs.size(); cellOrdinal++)
    {
      auto spatialCellID = spatialCellIDs[cellOrdinal];
      auto & phaseSpaceCellIDs = _spatialCellIDToPhaseSpaceCellIDs[spatialCellID];
      auto elemType = _phaseSpaceSolution->mesh()->getElementType(phaseSpaceCellIDs[0]); // we assume all element types are the same
      const int basisCardinality = elemType->trialOrderPtr->getBasis(_f->ID())->getCardinality();
      basisCoefficients.resize(basisCardinality);
      
      for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
      {
        double integral = 0.0;
        for (int phaseSpaceCellOrdinal=0; phaseSpaceCellOrdinal<phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
        {
          const auto & phaseSpaceCellID = phaseSpaceCellIDs[phaseSpaceCellOrdinal];
          _phaseSpaceSolution->getBasisCoefficientsForCell(basisCoefficients, _f->ID(), phaseSpaceCellID,
                                                           STANDARD_SOLUTION, stageOrdinal);
          for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++)
          {
            const double & basisCoefficient = basisCoefficients(basisOrdinal);
            const double & basisIntegral    = _basisIntegralsAtXQuadraturePoints(basisOrdinal,spatialPointOrdinal);
            integral += basisIntegral * basisCoefficient;
          }
        }
        values(cellOrdinal,spatialPointOrdinal) = integral;
      }
    }
  }
};
*/

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
  MeshPtr _mesh_1D;
  std::map<GlobalIndexType, GlobalIndexType> _phaseSpaceCellIDToPhysicalCellID;
public:
  VelocitySpaceExtrusion( FunctionPtr g_1D, MeshPtr mesh_1D,
                         const std::map<GlobalIndexType, GlobalIndexType> & phaseSpaceCellIDToPhysicalCellID)
  :
  Function(g_1D->rank()),
  _g_1D(g_1D),
  _mesh_1D(mesh_1D),
  _phaseSpaceCellIDToPhysicalCellID(phaseSpaceCellIDToPhysicalCellID)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported right now");
  }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
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
    
    // determine physical cell IDs we need to compute on
    auto & phaseSpaceCellIDs = basisCache->cellIDs();
    std::vector<GlobalIndexType> physicalCellIDs(phaseSpaceCellIDs.size());
    int physicalCellOrdinal = 0;
    for (auto & phaseSpaceCellID : phaseSpaceCellIDs)
    {
      auto physicalCellIDIt = _phaseSpaceCellIDToPhysicalCellID.find(phaseSpaceCellID);
      TEUCHOS_TEST_FOR_EXCEPTION(physicalCellIDIt == _phaseSpaceCellIDToPhysicalCellID.end(), std::invalid_argument,
                                 "phase space cell ID not found in container");
      GlobalIndexType physicalCellID = physicalCellIDIt->second;
      physicalCellIDs[physicalCellOrdinal++] = physicalCellID;
    }
    // import cell data
    _g_1D->importCellData(physicalCellIDs);
    
    int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();
    bool testVsTest = basisCache->testVsTest();
    
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
    
    // get one physical element type; this should match all the others (since phase space element type is the same for all cells in basisCache)
    ElementTypePtr elemType_1D = _mesh_1D->getElementType(physicalCellIDs[0]);
    BasisCachePtr basisCache_1D = BasisCache::basisCacheForCellType(_mesh_1D, elemType_1D, testVsTest,
                                                                    basisCache1DQuadratureEnrichment);
    
    int quadratureDegree1D = basisCache_1D->cubatureDegree();
    int quadratureDegree2D = basisCache->cubatureDegree();
    TEUCHOS_TEST_FOR_EXCEPTION(quadratureDegree1D != quadratureDegree2D, std::invalid_argument, "Internal error: quadrature degrees do not match");
    
    FieldContainer<double> physicalCellNodes = _mesh_1D->physicalCellNodes(physicalCellIDs);

    bool createSideCache = isSideCache;
    basisCache_1D->setPhysicalCellNodes(physicalCellNodes, physicalCellIDs, createSideCache);
    
    BasisCachePtr basisCacheToUse = (physicalSpaceSideOrdinal == -1) ? basisCache_1D : basisCache_1D->getSideBasisCache(physicalSpaceSideOrdinal);

    const int numPhysicalPoints   = basisCacheToUse->getRefCellPoints().dimension(0);
    const int numPhaseSpacePoints = basisCache->getRefCellPoints().dimension(0);
    const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
    
    TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
    
    // from here on is where we assume _g_1D is scalar-valued
    FieldContainer<double> values_1D(physicalCellIDs.size(), numPhysicalPoints);
    
    _g_1D->values(values_1D, basisCacheToUse);
    
    // copy values_1D into appropriate spots in the values container
    for (int cellOrdinal=0; cellOrdinal<physicalCellIDs.size(); cellOrdinal++)
    {
      // note: cellOrdinal is the same for physical and phase-space cell; there is a one-to-one map
      // (that's the case because we didn't de-duplicate the physicalCellIDs container above; we could optimize by doing so)
      for (int physicalPointOrdinal=0; physicalPointOrdinal<numPhysicalPoints; physicalPointOrdinal++)
      {
        const double value = values_1D(cellOrdinal,physicalPointOrdinal);
        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
        {
          // note that tensorPointOrdinal determination here does make an assumption about the point ordering
          const int tensorPointOrdinal = physicalPointOrdinal + numPhysicalPoints * velocityPointOrdinal;
          values(cellOrdinal,tensorPointOrdinal) = value;
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
                                                   const double x0, const double xWidth, const int xElements,
                                                   const double v0, const double vWidth, const int vElements,
                                                   const int H1Order, const int delta_k, ERKTableauPtr erkTableau)
:
_formulationChoice(formulationChoice)
{
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
  auto quadMeshTopology          = MeshFactory::quadMeshTopology(xWidth,vWidth,xElements,vElements,divideIntoTriangles,x0,v0,quadMeshPeriodicBCs);
  bool usePeriodicBCs            = true;
  auto intervalMeshTopology      = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
  
  const bool implicit  = usesImplicitSolve();
  const bool l2Trace   = hasL2Trace();
  const bool ultraweak = _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_H1_TRACE
                      || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE;
  
  if (ultraweak)
  {
    for (GlobalIndexType spatialCellID=0; spatialCellID<xElements; spatialCellID++)
    {
      std::vector<GlobalIndexType> phaseSpaceCellIDs(vElements);
      for (int i=0; i<vElements; i++)
      {
        phaseSpaceCellIDs[i] = i + spatialCellID * vElements;
        _phaseSpaceCellIDToSpatialCellID[phaseSpaceCellIDs[i]] = spatialCellID;
      }
      _spatialCellIDToPhaseSpaceCellIDs[spatialCellID] = phaseSpaceCellIDs;
    }
    
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
      auto fluxWeight = Function::normal_1D() * Function::sideParity(); // for "termTraced" argument below (which we are not currently using)
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
    _gaussSoln = Solution::solution(_gaussBF, gaussMesh);
    auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
    
    FunctionPtr dt = FunctionPtr(_dt);
    FunctionPtr E_soln = Function::solution(E, gaussSolnWeakPtr);
    _E_extruded = Teuchos::rcp( new VelocitySpaceExtrusion(E_soln, gaussMesh, _phaseSpaceCellIDToSpatialCellID) );
    
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
    
    map<int,int> trialOrderEnhancements;
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1;
    }
    
    MeshPtr boltzmannMesh = Teuchos::rcp( new Mesh(quadMeshTopology, _boltzmannBF, H1Order, delta_k, trialOrderEnhancements) );
    
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
    
    _f_velocityIntegral = Teuchos::rcp( new VelocitySpace1DEquispacedIntegral(_gaussSoln->mesh(), _boltzmannSolnPrev, f, _spatialCellIDToPhaseSpaceCellIDs) );
    
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
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported Boltzmann1D1VFormulationChoice");
  }
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

const std::map<GlobalIndexType, GlobalIndexType> & Boltzmann1D1VFormulation::phaseSpaceCellIDToSpatialCellID()
{
  return _phaseSpaceCellIDToSpatialCellID;
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

void Boltzmann1D1VFormulation::setStage(const int stageNumber)
{
  updateElectricField();
}

void Boltzmann1D1VFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

void Boltzmann1D1VFormulation::takeTimeStep()
{
  // for implicit solves, we cycle through the f solve and the E solve a few times;
  // for explicit, we do each solve just once per time step.
  const bool implicit = _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
  int numSubsteps = implicit ? 3 : 1;
  
  for (int substepOrdinal=0; substepOrdinal < numSubsteps; substepOrdinal++)
  {
    // solve for f:
    _boltzmannSoln->solve();
    
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
        // last substep
        _boltzmannSolnPrev->setSolution(_boltzmannSoln);
      }
      // if we have a tableau, then electric field will be updated when the solution stage is set.
      updateElectricField();
    }
  }
  
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

TEUCHOS_UNIT_TEST( VelocitySpace1DEquispacedIntegral_Boltzmann1D1VFormulation, AnalyticIntegralMatches )
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
    
    auto f = form.f();
    std::map<int, FunctionPtr> projectionMap;
    projectionMap[f->ID()] = xy;
    
    auto boltzmannSoln = form.boltzmannSolution();
    const int solutionOrdinal = 0;
    boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto gaussMesh = form.gaussMesh();
    auto spatialToPhaseSpaceMap = form.spatialCellIDToPhaseSpaceCellIDsMap();
    
    auto velocitySpaceIntegral = Teuchos::rcp(new VelocitySpace1DEquispacedIntegral(gaussMesh, boltzmannSoln, f, spatialToPhaseSpaceMap) );
    FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
    
    velocitySpaceIntegral->QuadraturePointFunction::updateValues();
    
    double magnitudeExpected = expectedIntegral->l2norm(gaussMesh);
    double err_l2 = err->l2norm(gaussMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
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

void testVelocityExtrusion(bool volumeVariable, Teuchos::FancyOStream &out, bool &success)
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
      auto E_hat = form.E_hat();
      projectionMap[E_hat->ID()] = x;
      soln_1D = Function::solution(E_hat, gaussSoln, false); // false: don't weight by parity (makes uniquely-valued)
    }
    
    TEST_ASSERT(soln_1D->boundaryValueOnly() == !volumeVariable);
    
    const int solutionOrdinal =0;
    gaussSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto gaussMesh = form.gaussMesh();
    auto & phaseSpaceToSpatialMap = form.phaseSpaceCellIDToSpatialCellID();
    auto boltzmannMesh = form.boltzmannMesh();
    
    auto extrusion = Teuchos::rcp( new VelocitySpaceExtrusion(soln_1D, gaussMesh, phaseSpaceToSpatialMap) );
    
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
      expected = x * sideRestriction;
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

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, VolumeExtrusion )
{
  testVelocityExtrusion(true, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusion_Boltzmann1D1VFormulation, SkeletonExtrusion )
{
  testVelocityExtrusion(false, out, success);
}
