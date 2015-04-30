//
//  BCTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/29/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "Boundary.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace {
  void testSpaceTimeTraceBCFunction(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    
    double epsilon = .1;
    int fieldPolyOrder = 3, delta_k = 1;
    
    static const double CONST_VALUE = 0.5;
    FunctionPtr u = Function::constant(CONST_VALUE);
    
    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, useConformingTraces, epsilon);
    
    FunctionPtr forcingFunction = SpaceTimeHeatFormulation::forcingFunction(spaceDim, epsilon, u);
    form.initializeSolution(spaceTimeMeshTopo, fieldPolyOrder, delta_k, forcingFunction);
    
    VarPtr u_hat = form.u_hat();
    bool isTrace = true;
    BCPtr bc = form.solution()->bc();
    bc->addDirichlet(u_hat, SpatialFilter::allSpace(), u);
    
    MeshPtr mesh = form.solution()->mesh();
    
    GlobalIndexType cellID = 0;
    
    // use our knowledge that we have a one-element mesh: every last dof for u_hat should be present, and have coefficient CONST_VALUE
    DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    CellTopoPtr cellTopo = mesh->getElementType(cellID)->cellTopoPtr;
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    double tol = 1e-13;
    for (int sideOrdinal=0; sideOrdinal < cellTopo->getSideCount(); sideOrdinal++)
    {
      out << "******** SIDE " << sideOrdinal << " ********" << endl;
      BasisPtr basis = trialOrder->getBasis(u_hat->ID(),sideOrdinal);

      int numCells = 1;
      Intrepid::FieldContainer<double> dirichletValues(numCells,basis->getCardinality());
      // project bc function onto side basis:
      Teuchos::RCP<BCFunction<double>> bcFunction = BCFunction<double>::bcFunction(bc, u_hat->ID(), isTrace);
      bc->coefficientsForBC(dirichletValues, bcFunction, basis, basisCache->getSideBasisCache(sideOrdinal));
      for (int basisOrdinal=0; basisOrdinal<dirichletValues.dimension(1); basisOrdinal++)
      {
        TEST_FLOATING_EQUALITY(CONST_VALUE, dirichletValues(0,basisOrdinal), tol);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BC, SpaceTimeTraceBCCoefficients )
  {
    int spaceDim = 1;
    testSpaceTimeTraceBCFunction(spaceDim, out, success);
  }
} // namespace