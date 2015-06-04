//
//  SpaceTimeHeatDivFormulation
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Boundary.h"
#include "CamelliaDebugUtility.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatDivFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  void projectExactSolution(SpaceTimeHeatDivFormulation &form, SolutionPtr heatSolution, FunctionPtr u)
  {
    double epsilon = form.epsilon();

    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = heatSolution->mesh()->getDimension();
    int spaceDim = spaceTimeDim - 1;

    sigma1 = epsilon * u->dx();
    if (spaceDim > 1) sigma2 = epsilon * u->dy();
    if (spaceDim > 2) sigma3 = epsilon * u->dz();

    LinearTermPtr tc_lt = form.tc()->termTraced();
    LinearTermPtr u_lt = form.uhat()->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u()->ID()] = u;
    exactMap[form.sigma(1)->ID()] = sigma1;
    if (spaceDim > 1) exactMap[form.sigma(2)->ID()] = sigma2;
    if (spaceDim > 2) exactMap[form.sigma(3)->ID()] = sigma3;

    // flux:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr tc = tc_lt->evaluate(exactMap);
    exactMap[form.tc()->ID()] = tc;

    // traces:
    FunctionPtr uhat = u_lt->evaluate(exactMap);
    exactMap[form.uhat()->ID()] = uhat;

    heatSolution->projectOntoMesh(exactMap);
  }

  void setupExactSolution(SpaceTimeHeatDivFormulation &form, FunctionPtr u,
                          MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k)
  {
    double epsilon = form.epsilon();

    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = meshTopo->getSpaceDim();
    int spaceDim = spaceTimeDim - 1;

    FunctionPtr forcingFunction = SpaceTimeHeatDivFormulation::forcingFunction(spaceDim, epsilon, u);

    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);
  }

  void testForcingFunctionForConstantU(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // Forcing function should be zero for constant u
    FunctionPtr f_expected = Function::zero();

    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    FunctionPtr u = Function::constant(0.5);

    bool useConformingTraces = true;
    SpaceTimeHeatDivFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);

    MeshPtr mesh = form.solution()->mesh();

    FunctionPtr f_actual = form.forcingFunction(spaceDim, epsilon, u);

    double l2_diff = (f_expected-f_actual)->l2norm(mesh);
    TEST_COMPARE(l2_diff, <, 1e-14);
  }

  void testSpaceTimeHeatConsistency(int spaceDim, bool useConformingTraces, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = 0.1;
    int fieldPolyOrder = 2, delta_k = 1;

    FunctionPtr u;
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    FunctionPtr t = Function::tn(1);

    if (spaceDim == 1)
    {
      u = x * t;
    }
    else if (spaceDim == 2)
    {
      u = x * t + y;
    }
    else if (spaceDim == 3)
    {
      u = x * t + y - z;
    }

    SpaceTimeHeatDivFormulation form(spaceDim, epsilon, useConformingTraces);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);

    bool DEBUGGING = false; // set to true to print some information to console...
    if (DEBUGGING)
    {
      // DEBUGGING
      if (spaceDim == 2)
      {
        GlobalIndexType cellID = 0;

        GlobalDofAssignmentPtr gda = form.solution()->mesh()->globalDofAssignment();
        GDAMinimumRule* gdaMinRule = dynamic_cast<GDAMinimumRule*>(gda.get());
        gdaMinRule->printConstraintInfo(cellID);

        DofOrderingPtr trialOrder = form.solution()->mesh()->getElementType(cellID)->trialOrderPtr;
        Intrepid::FieldContainer<double> dofCoefficients(trialOrder->totalDofs());
        dofCoefficients[82] = 1.0;
        printLabeledDofCoefficients(form.bf()->varFactory(), trialOrder, dofCoefficients);


        VarPtr uhat = form.uhat();
        int sideOrdinal = 0;
        int basisOrdinal = 1; // the one we seek, corresponding to 82 above
        BasisPtr uhatBasis = trialOrder->getBasis(uhat->ID(),sideOrdinal);

        int sideDim = uhatBasis->domainTopology()->getDimension();
        for (int subcdim=0; subcdim<=sideDim; subcdim++)
        {
          int subcCount = uhatBasis->domainTopology()->getSubcellCount(subcdim);
          for (int subcord=0; subcord<subcCount; subcord++)
          {
            set<int> dofOrdinalsForSubcell = uhatBasis->dofOrdinalsForSubcell(subcdim, subcord);
            if (dofOrdinalsForSubcell.find(basisOrdinal) != dofOrdinalsForSubcell.end())
            {
              cout << "basisOrdinal " << basisOrdinal << " belongs to subcell " << subcord << " of dimension " << subcdim << endl;
            }
          }
        }

      }
    }

    projectExactSolution(form, form.solution(), u);

    form.solution()->clearComputedResiduals();

    double energyError = form.solution()->energyErrorTotal();

//    if (spaceDim != 3)
//    {
//      MeshPtr mesh = form.solution()->mesh();
//      string outputDir = "/tmp";
//      string solnName = (spaceDim == 1) ? "spaceTimeHeatSolution_1D" : "spaceTimeHeatSolution_2D";
//      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
//      HDF5Exporter exporter(mesh, solnName, outputDir);
//      exporter.exportSolution(form.solution());
//    }

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  void testSpaceTimeHeatConsistencyConstantSolution(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    FunctionPtr u = Function::constant(0.5);

    bool useConformingTraces = true;
    SpaceTimeHeatDivFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u);

    form.solution()->clearComputedResiduals();

    double energyError = form.solution()->energyErrorTotal();

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  void testSpaceTimeHeatImposeConstantFluxBCs(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    static const double CONST_VALUE = 0.0;
    FunctionPtr u = Function::constant(CONST_VALUE);
    FunctionPtr sigma = epsilon*u->grad();
    FunctionPtr n_x = Function::normal(); // spatial normal
    FunctionPtr n_xt = Function::normalSpaceTime();

    bool useConformingTraces = true;
    SpaceTimeHeatDivFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);

    VarPtr uhat = form.uhat();
    VarPtr tc = form.tc();
    bool isTrace = true;
    BCPtr bc = form.solution()->bc();
    // bc->addDirichlet(uhat, SpatialFilter::allSpace(), u);
    bc->addDirichlet(tc, SpatialFilter::allSpace(), -sigma*n_x + u*n_xt->t());

    MeshPtr mesh = form.solution()->mesh();

    Boundary boundary = mesh->boundary();
    DofInterpreter* dofInterpreter = form.solution()->getDofInterpreter().get();
    std::map<GlobalIndexType, double> globalDofIndicesAndValues;
    GlobalIndexType cellID = 0;
    set<pair<int, unsigned>> singletons;
    boundary.bcsToImpose<double>(globalDofIndicesAndValues, *bc, cellID, singletons, dofInterpreter, NULL);

    // use our knowledge that we have a one-element mesh: every last dof for uhat should be present, and have coefficient CONST_VALUE
    DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    CellTopoPtr cellTopo = mesh->getElementType(cellID)->cellTopoPtr;

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    double tol = 1e-13;
    for (int sideOrdinal=0; sideOrdinal < cellTopo->getSideCount(); sideOrdinal++)
    {
      out << "******** SIDE " << sideOrdinal << " ********" << endl;
      BasisPtr basis = trialOrder->getBasis(tc->ID(),sideOrdinal);
      Intrepid::FieldContainer<double> fluxValues(basis->getCardinality());
      fluxValues.initialize(CONST_VALUE);
      Intrepid::FieldContainer<double> globalData;
      Intrepid::FieldContainer<GlobalIndexType> globalDofIndices;
      dofInterpreter->interpretLocalBasisCoefficients(cellID, tc->ID(), sideOrdinal, fluxValues, globalData, globalDofIndices);
      // sanity check on the interpreted global values
      for (int basisOrdinal=0; basisOrdinal<globalData.size(); basisOrdinal++)
      {
        TEST_FLOATING_EQUALITY(CONST_VALUE, globalData(basisOrdinal), tol);
      }

      for (int basisOrdinal=0; basisOrdinal<globalData.size(); basisOrdinal++)
      {
        GlobalIndexType globalDofIndex = globalDofIndices(basisOrdinal);
        if (globalDofIndicesAndValues.find(globalDofIndex) != globalDofIndicesAndValues.end())
        {
          double expectedValue = globalData(basisOrdinal);
          double actualValue = globalDofIndicesAndValues[globalDofIndex];
          TEST_FLOATING_EQUALITY(expectedValue, actualValue, tol);
        }
        else
        {
          out << "On side " << sideOrdinal << ", did not find globalDofIndex " << globalDofIndex << endl;
          success = false;
        }
      }
    }
  }

  void testSpaceTimeHeatSolveConstantSolution(int spaceDim, bool useFluxBCsEverywhere, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    FunctionPtr u = Function::constant(0.5);
    FunctionPtr sigma = epsilon*u->grad();
    FunctionPtr n_x = Function::normal(); // spatial normal
    FunctionPtr n_xt = Function::normalSpaceTime();

    bool useConformingTraces = true;
    SpaceTimeHeatDivFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);

    if (!useFluxBCsEverywhere)
    {
      out << "useFluxBCsEverywhere = false not yet supported/implemented in test.\n";
      success = false;
    }
    else
    {
      VarPtr uhat = form.uhat();
      VarPtr tc = form.tc();
      BCPtr bc = form.solution()->bc();
      // bc->addDirichlet(uhat, SpatialFilter::allSpace(), u);
      bc->addDirichlet(tc, SpatialFilter::allSpace(), -sigma*n_x + u*n_xt->t());
    }

    form.solution()->solve();

//    if (spaceDim != 3)
//    {
//      MeshPtr mesh = form.solution()->mesh();
//      string outputDir = "/tmp";
//      string solnName = (spaceDim == 1) ? "spaceTimeHeatSolution_1D" : "spaceTimeHeatSolution_2D";
//      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
//      HDF5Exporter exporter(mesh, solnName, outputDir);
//      exporter.exportSolution(form.solution());
//    }

    double energyError = form.solution()->energyErrorTotal();

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ConsistencyConstantSolution_1D )
  {
    // consistency test for space-time formulation with 1D space
    testSpaceTimeHeatConsistencyConstantSolution(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ConsistencyConstantSolution_2D )
  {
    // consistency test for space-time formulation with 2D space
    testSpaceTimeHeatConsistencyConstantSolution(2, out, success);
  }

  //  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ConsistencyConstantSolution_3D )
  //  {
  //    // consistency test for space-time formulation with 3D space
  //    testSpaceTimeHeatConsistencyConstantSolution(3, out, success);
  //  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Conforming_1D )
  {
    // consistency test for space-time formulation with 1D space
    bool useConformingTraces = true; // conforming and non conforming are the same for 1D
    testSpaceTimeHeatConsistency(1, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Nonconforming_1D )
  {
    // consistency test for space-time formulation with 1D space
    bool useConformingTraces = false; // conforming and non conforming are the same for 1D
    testSpaceTimeHeatConsistency(1, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Conforming_2D )
  {
    // consistency test for space-time formulation with 2D space
    bool useConformingTraces = true;
    testSpaceTimeHeatConsistency(2, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Conforming_3D_Slow )
  {
    // consistency test for space-time formulation with 3D space
    bool useConformingTraces = true;
    testSpaceTimeHeatConsistency(3, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Nonconforming_2D )
  {
    // consistency test for space-time formulation with 2D space
    bool useConformingTraces = false;
    testSpaceTimeHeatConsistency(2, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, Consistency_Nonconforming_3D_Slow )
  {
    // consistency test for space-time formulation with 3D space
    bool useConformingTraces = false;
    testSpaceTimeHeatConsistency(3, useConformingTraces, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ForcingFunctionForConstantU_1D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ForcingFunctionForConstantU_2D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(2, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ForcingFunctionForConstantU_3D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(3, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ImposeConstantFluxBCs_1D )
  {
    // test BC imposition for space-time formulation with 1D space, exact solution with u constant
    testSpaceTimeHeatImposeConstantFluxBCs(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, SolveConstantSolution_1D )
  {
    // test solve for space-time formulation with 1D space, exact solution with u constant
    bool useFluxBCsEverywhere = true;
    testSpaceTimeHeatSolveConstantSolution(1, useFluxBCsEverywhere, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, ImposeConstantFluxBCs_2D )
  {
    // test BC imposition for space-time formulation with 2D space, exact solution with u constant
    testSpaceTimeHeatImposeConstantFluxBCs(2, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatDivFormulation, SolveConstantSolution_2D )
  {
    // test solve for space-time formulation with 1D space, exact solution with u constant
    bool useFluxBCsEverywhere = true;
    testSpaceTimeHeatSolveConstantSolution(2, useFluxBCsEverywhere, out, success);
  }
} // namespace
