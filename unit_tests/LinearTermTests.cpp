//
//  LinearTermTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/14/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Function.h"
#include "MeshFactory.h"
#include "SerialDenseWrapper.h"
#include "SpaceTimeHeatFormulation.h"
#include "TensorBasis.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace {
  void setFauxSpaceTimeHeatFormulation(int spaceDim, double epsilon, bool useConformingTraces, VarPtr &v, BFPtr &bf)
  {
    if ((spaceDim != 1) && (spaceDim != 2)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 1, or 2");
    }

    // declare all possible variables -- will only create the ones we need for spaceDim
    // fields
    VarPtr u;
    VarPtr sigma1, sigma2;

    // traces
    VarPtr u_hat;
    VarPtr sigma_n_hat;

    // tests
    VarPtr tau1, tau2;

    VarFactoryPtr vf = VarFactory::varFactory();

    u = vf->fieldVar("u");

    sigma1 = vf->fieldVar("\\sigma_1");
    if (spaceDim > 1) sigma2 = vf->fieldVar("\\sigma_2");

    Space uHatSpace = useConformingTraces ? HGRAD : L2;

    u_hat = vf->traceVar("\\widehat{u}", 1.0 * u, uHatSpace);

    FunctionPtr n = Function::normal();
    FunctionPtr n_parity = n * Function::sideParity();

    LinearTermPtr sigma_n_lt;
    if (spaceDim == 1)
    {
      sigma_n_lt = sigma1 * n_parity->x();
    }
    else if (spaceDim == 2)
    {
      sigma_n_lt = sigma1 * n_parity->x() + sigma2 * n_parity->y();
    }
    sigma_n_hat = vf->fluxVar("\\", sigma_n_lt); // should be fluxVarSpaceOnly --> we shouldn't actually *solve* with the faux formulation

    v = vf->testVar("v", HGRAD);

    if (spaceDim > 1)
    {
      // tau should be in HDIV, but we split into scalars
      // because of limitations in faux space-time (no setting variables
      // with vector lengths = spaceDim-1).
      tau1 = vf->testVar("\\tau_1", HGRAD);
      tau2 = vf->testVar("\\tau_2", HGRAD);
    }
    else
    {
      tau1 = vf->testVar("\\tau_1", HGRAD); // scalar
    }

    bf = Teuchos::rcp( new BF(vf) );
    // v terms
    VarPtr v_dt;
    FunctionPtr n_t;
    if (spaceDim==1)
    {
      v_dt = v->dy();
      n_t = Function::normal()->y();
    }
    else
    {
      v_dt = v->dz();
      n_t = Function::normal()->z();
    }

    bf->addTerm(-u, v_dt);
    bf->addTerm(u_hat, v * n_t);
    bf->addTerm(sigma1, v->dx());
    if (spaceDim > 1) bf->addTerm(sigma2, v->dy());
    bf->addTerm(-sigma_n_hat, v);

    // tau terms
    if (spaceDim == 2) {
      bf->addTerm((1.0 / epsilon) * sigma1, tau1);
      bf->addTerm((1.0 / epsilon) * sigma2, tau2);
      bf->addTerm(u, tau1->dx() + tau2->dy()); // tau->div()

      bf->addTerm(-u_hat, tau2 * n->x() + tau2 * n->y());
    }
    else if (spaceDim==1)
    {
      bf->addTerm((1.0 / epsilon) * sigma1, tau1);
      bf->addTerm(u, tau1->dx());
      bf->addTerm(-u_hat, tau1 * n->x());
    }
  }

  MeshPtr singleElementSpaceTimeMesh(int spaceDim, int H1Order)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    int delta_k = 1;
    vector<int> H1OrderVector(2);
    H1OrderVector[0] = H1Order;
    H1OrderVector[1] = H1Order;
    MeshPtr mesh = Teuchos::rcp( new Mesh(spaceTimeMeshTopo, form.bf(), H1OrderVector, delta_k) ) ;
    return mesh;
  }

  MeshPtr singleElementFauxSpaceTimeMesh(int spaceDim, int H1Order)
  {
    double t0 = 0.0, t1 = 1.0;
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    dimensions.push_back(t1-t0);
    vector<int> elementCounts(spaceDim+1,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    x0.push_back(t0);
    MeshTopologyPtr fauxSpaceTimeMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double epsilon = 1.0;
    BFPtr bf;
    VarPtr v; // we ignore this here

    bool useConformingTraces = false;
    setFauxSpaceTimeHeatFormulation(spaceDim, epsilon, useConformingTraces, v, bf);

    int delta_k = 1;
    MeshPtr mesh = Teuchos::rcp( new Mesh(fauxSpaceTimeMeshTopo, bf, H1Order, delta_k) ) ;
    return mesh;
  }

  void testSpaceTimeNonzeroTimeDerivative(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // here, we simply test that v->dt() gives something nonzero

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    VarPtr v = form.v();
    FunctionPtr f = Function::xn(1);

    LinearTermPtr lt = 1.0 * v->dt();

    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
    double norm = lt->computeNorm(form.bf()->graphNorm(), mesh); // should be > 0

//    cout << "spaceDim, " << spaceDim << "; norm " << norm << endl;

    TEST_COMPARE(norm, >, 1e-14);
  }

  void getIntegrationByPartsInTimeComponents(LinearTermPtr &lt1, LinearTermPtr &lt2,
                                             int spaceDim, bool fauxSpaceTime)
  {
    double epsilon = 1.0;
    bool useConformingTraces = false;

    if (!fauxSpaceTime)
    {
      SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
      VarPtr v = form.v();
      FunctionPtr f = Function::xn(1);

      FunctionPtr n_xt = Function::normalSpaceTime();

      lt1 = -f * v->dt();
      lt2 = (f * v) * n_xt->t();
    }
    else
    {
      BFPtr bf;
      VarPtr v;

      setFauxSpaceTimeHeatFormulation(spaceDim, epsilon, useConformingTraces, v, bf);

      FunctionPtr f = Function::xn(1);

      VarPtr v_dt;
      FunctionPtr n_t;
      if (spaceDim == 1)
      {
        v_dt = v->dy();
        n_t = Function::normal()->y();
      }
      else if (spaceDim==2)
      {
        v_dt = v->dz();
        n_t = Function::normal()->z();
      }

      lt1 = -f * v_dt;
      lt2 = (f * v) * n_t;
    }
  }

  void testSpaceTimeIntegrationByPartsInTime(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // we consider df/dt where f = x.
    // Integrating by parts 0 = (df/dt, v) = (-f, v->dt()) + < f, v * n->t() >

    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceDim, epsilon);
    VarPtr v = form.v();
    FunctionPtr f = Function::xn(1);

    FunctionPtr n_xt = Function::normalSpaceTime();

    LinearTermPtr lt1 = -f * v->dt();
    LinearTermPtr lt2 = (f * v) * n_xt->t();

    LinearTermPtr lt = lt1 + lt2;

    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
    double norm = lt->computeNorm(form.bf()->graphNorm(), mesh); // should be 0

    double tol = 1e-14;
    if (norm > tol)
    {
      double norm_lt1 = lt1->computeNorm(form.bf()->graphNorm(), mesh);
      double norm_lt2 = lt2->computeNorm(form.bf()->graphNorm(), mesh);
      out << "norm(lt1) = " << norm_lt1 << endl;
      out << "norm(lt2) = " << norm_lt2 << endl;
    }

    TEST_COMPARE(norm, <, tol);
  }

  void testFauxSpaceTimeIntegrationByPartsInTime(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // Faux space-time: use last dimension as time, but without actually using space-time mesh
    if ((spaceDim != 1) && (spaceDim != 2))
    {
      success = false;
      out << "testFauxSpaceTimeIntegrationByPartsInTime() only supports spaceDim of 1 or 2";
    }

    // we consider df/dt where f = x.
    // Integrating by parts 0 = (df/dt, v) = (-f, v->dt()) + < f, v * n->t() >

    double epsilon = 1.0;
    BFPtr bf;
    VarPtr v;

    bool useConformingTraces = false;
    setFauxSpaceTimeHeatFormulation(spaceDim, epsilon, useConformingTraces, v, bf);

    FunctionPtr f = Function::xn(1);

    VarPtr v_dt;
    FunctionPtr n_t;
    if (spaceDim == 1)
    {
      v_dt = v->dy();
      n_t = Function::normal()->y();
    }
    else if (spaceDim==2)
    {
      v_dt = v->dz();
      n_t = Function::normal()->z();
    }

    LinearTermPtr lt1 = -f * v_dt;
    LinearTermPtr lt2 = (f * v) * n_t;

    LinearTermPtr lt = lt1 + lt2;

    int H1Order = 2;
    IPPtr ip = bf->graphNorm();
    MeshPtr mesh = singleElementFauxSpaceTimeMesh(spaceDim, H1Order);
    double norm = lt->computeNorm(ip, mesh); // should be 0

    double tol = 1e-14;
    if (norm > tol)
    {
      double norm_lt1 = lt1->computeNorm(ip, mesh);
      double norm_lt2 = lt2->computeNorm(ip, mesh);
      out << "norm(lt1) = " << norm_lt1 << endl;
      out << "norm(lt2) = " << norm_lt2 << endl;
    }

//    { // DEBUGGING:
//      double norm_lt1 = lt1->computeNorm(ip, mesh);
//      double norm_lt2 = lt2->computeNorm(ip, mesh);
//      cout << "norm(lt1) = " << norm_lt1 << endl;
//      cout << "norm(lt2) = " << norm_lt2 << endl;
//    }

    TEST_COMPARE(norm, <, tol);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, CompareFauxWithTrueSpaceTime_1D )
  {
    /*
     idea here:

     create v_spacetime, v_faux as v in the two contexts above
     set up LinearTerms:
       -f * v_spacetime->dt();
     and
       -f * v_faux_dt;

     as well as
       (f * v_spacetime) * n_xt->t();
     and
       (f * v_faux) * n_t;

     Then, for the Basis that the faux/true space-time mesh uses for v,
     compute LinearTerm::values() for each of these, and compare faux to
     true.

     As of this writing, I expect the first ones to match and the second
     ones to differ, though I don't yet know what's wrong in the second case
     (hence this test).

     */

    int spaceDim = 1; // spaceDim = 1 is the only one for which faux and real can be expected to be precisely the same
    // volume terms:
    LinearTermPtr lt1_faux, lt1_true;
    // boundary terms:
    LinearTermPtr lt2_faux, lt2_true;
    getIntegrationByPartsInTimeComponents(lt1_faux, lt2_faux, spaceDim, true);
    getIntegrationByPartsInTimeComponents(lt1_true, lt2_true, spaceDim, false);

    int H1Order = 2;
    MeshPtr fauxMesh = singleElementFauxSpaceTimeMesh(spaceDim, H1Order);
    MeshPtr trueMesh = singleElementSpaceTimeMesh(spaceDim, H1Order);

    // we can use the space-time ("true") discretization for the faux, but not
    // the other way around, since the space-time basis is required to be a TensorBasis
    GlobalIndexType cellID = 0;
    DofOrderingPtr trueTestOrder = trueMesh->getElementType(cellID)->testOrderPtr;

    // get ID for v (here, we are using the knowledge that lt1, lt2 only involve a single variable)
    int vIDTrue = *lt1_true->varIDs().begin();
    int vIDFaux = *lt1_faux->varIDs().begin();
    BasisPtr vBasisTrue = trueTestOrder->getBasis(vIDTrue);

    // we need to allow the faux basis to compute gradient on the way to OP_DY (aka OP_DT)
    typedef Camellia::TensorBasis<double, Intrepid::FieldContainer<double> > TensorBasis;
    TensorBasis* vTensorBasis = dynamic_cast<TensorBasis*>(vBasisTrue.get());
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(vTensorBasis->getSpatialBasis(),
                                                                          vTensorBasis->getTemporalBasis(),
                                                                          true) );
    BasisPtr vBasisFaux = tensorBasis;
    // set up a DofOrdering for v for both true and faux:
    DofOrderingPtr vOrderFaux = Teuchos::rcp( new DofOrdering(fauxMesh->getElementType(cellID)->cellTopoPtr) );
    DofOrderingPtr vOrderTrue = Teuchos::rcp( new DofOrdering(trueMesh->getElementType(cellID)->cellTopoPtr) );
    vOrderFaux->addEntry(vIDFaux, vBasisFaux, vBasisFaux->rangeRank());
    vOrderTrue->addEntry(vIDTrue, vBasisTrue, vBasisTrue->rangeRank());

    BasisCachePtr fauxBasisCache = BasisCache::basisCacheForCell(fauxMesh, cellID);
    BasisCachePtr trueBasisCache = BasisCache::basisCacheForCell(trueMesh, cellID);
    // make sure the BasisCaches agree on which points and their ordering:
    fauxBasisCache->setRefCellPoints(trueBasisCache->getRefCellPoints());
    int numPoints = fauxBasisCache->getRefCellPoints().dimension(0);

    // compare values:
    Intrepid::FieldContainer<double> lt1FauxValues(1,vBasisFaux->getCardinality(),numPoints);
    Intrepid::FieldContainer<double> lt1TrueValues(1,vBasisTrue->getCardinality(),numPoints);
    lt1_faux->values(lt1FauxValues, vIDFaux, vBasisFaux, fauxBasisCache);
    lt1_true->values(lt1TrueValues, vIDTrue, vBasisTrue, trueBasisCache);

    double tol=1e-14;
    TEST_COMPARE_FLOATING_ARRAYS(lt1FauxValues, lt1TrueValues, tol);

    // compare integrals:
    Intrepid::FieldContainer<double> lt1FauxIntegrals(1,vOrderFaux->totalDofs());
    Intrepid::FieldContainer<double> lt1TrueIntegrals(1,vOrderTrue->totalDofs());
    lt1_faux->integrate(lt1FauxIntegrals, vOrderFaux, fauxBasisCache);
    lt1_true->integrate(lt1TrueIntegrals, vOrderTrue, trueBasisCache);
    SerialDenseWrapper::roundZeros(lt1FauxIntegrals,1e-15);
    SerialDenseWrapper::roundZeros(lt1TrueIntegrals,1e-15);
    TEST_COMPARE_FLOATING_ARRAYS(lt1FauxIntegrals, lt1TrueIntegrals, tol);

    // Here, a couple ugly hard-codings:
    // 1. How the space-time sides map to the faux sides
    // 2. Whether the side orientations are reversed in the two meshes
    map<unsigned,unsigned> trueToFauxSideOrdinal = {{0,0},{1,2},{2,3},{3,1}};
    vector<bool> orientationReversedForTrueSide = {false,true,true,false};

    int sideDim = spaceDim; // 1
    for (unsigned sideOrdinal=0; sideOrdinal<trueBasisCache->cellTopology()->getSideCount(); sideOrdinal++)
    {
      BasisCachePtr trueSideCache = trueBasisCache->getSideBasisCache(sideOrdinal);
      unsigned fauxSideOrdinal = trueToFauxSideOrdinal[sideOrdinal];
      out << "Checking that values on true side " << sideOrdinal << " match faux side " << fauxSideOrdinal << endl;
      BasisCachePtr fauxSideCache = fauxBasisCache->getSideBasisCache(fauxSideOrdinal);
      Intrepid::FieldContainer<double> trueRefPoints = trueSideCache->getRefCellPoints();
      int numPoints = trueRefPoints.dimension(0);

      Intrepid::FieldContainer<double> fauxRefPoints(numPoints,sideDim);
      for (int fauxPointOrdinal=0; fauxPointOrdinal<numPoints; fauxPointOrdinal++)
      {
        int truePointOrdinal = orientationReversedForTrueSide[sideOrdinal] ? numPoints - fauxPointOrdinal - 1 : fauxPointOrdinal;
        // check that, modulo order, the weighted measures agree:
        double trueWeightedMeasure = trueSideCache->getWeightedMeasures()(0,truePointOrdinal);
        double fauxWeightedMeasure = fauxSideCache->getWeightedMeasures()(0,fauxPointOrdinal);
        TEST_FLOATING_EQUALITY(fauxWeightedMeasure, trueWeightedMeasure, tol);

        fauxRefPoints(fauxPointOrdinal,0) = trueRefPoints(truePointOrdinal,0);
      }
      fauxSideCache->setRefCellPoints(fauxRefPoints);

      // compare values:
      Intrepid::FieldContainer<double> lt2FauxValues(1,vBasisFaux->getCardinality(),numPoints);
      Intrepid::FieldContainer<double> lt2TrueValues(1,vBasisTrue->getCardinality(),numPoints);
      lt2_faux->values(lt2FauxValues, vIDFaux, vBasisFaux, fauxSideCache);
      lt2_true->values(lt2TrueValues, vIDTrue, vBasisTrue, trueSideCache);
      TEST_COMPARE_FLOATING_ARRAYS(lt2FauxValues, lt2TrueValues, tol);

      // compare integrals:
      Intrepid::FieldContainer<double> lt2FauxIntegrals(1,vOrderFaux->totalDofs());
      Intrepid::FieldContainer<double> lt2TrueIntegrals(1,vOrderTrue->totalDofs());
      lt2_faux->integrate(lt2FauxIntegrals, vOrderFaux, fauxSideCache);
      lt2_true->integrate(lt2TrueIntegrals, vOrderTrue, trueSideCache);
      SerialDenseWrapper::roundZeros(lt2FauxIntegrals,1e-15);
      SerialDenseWrapper::roundZeros(lt2TrueIntegrals,1e-15);
      TEST_COMPARE_FLOATING_ARRAYS(lt2FauxIntegrals, lt2TrueIntegrals, tol);
    }
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegration_1D)
  {
    int spaceDim = 1;
    // test that LinearTerm::integrate() computes:
    //           LinearTerm::values() dot weightedMeasures
    // volume terms:
    LinearTermPtr lt1, lt2; // volume, boundary term
    bool notFauxSpaceTime = false; // use actual space-time formulation
    getIntegrationByPartsInTimeComponents(lt1, lt2, spaceDim, notFauxSpaceTime);

    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);

    // we can use the space-time ("true") discretization for the faux, but not
    // the other way around, since the space-time basis is required to be a TensorBasis
    GlobalIndexType cellID = 0;
    DofOrderingPtr wholeTestOrder = mesh->getElementType(cellID)->testOrderPtr; // includes tau

    // get ID for v (here, we are using the knowledge that lt1, lt2 only involve a single variable)
    int vID = *lt1->varIDs().begin();
    BasisPtr vBasis = wholeTestOrder->getBasis(vID);

    // set up a DofOrdering for just v:
    DofOrderingPtr vOrder = Teuchos::rcp( new DofOrdering(mesh->getElementType(cellID)->cellTopoPtr) );
    vOrder->addEntry(vID, vBasis, vBasis->rangeRank());

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

    int numPoints = basisCache->getRefCellPoints().dimension(0);
    // compute values:
    Intrepid::FieldContainer<double> lt1Values(1,vBasis->getCardinality(),numPoints);
    lt1->values(lt1Values, vID, vBasis, basisCache);

    double tol=1e-14;

    // compute integrals:
    Intrepid::FieldContainer<double> lt1IntegralsExpected(1,vOrder->totalDofs());
    for (int basisOrdinal=0; basisOrdinal<vBasis->getCardinality(); basisOrdinal++)
    {
      int dofIndex = vOrder->getDofIndex(vID, basisOrdinal);
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        double value = lt1Values(0,basisOrdinal,ptOrdinal);
        double weight = basisCache->getWeightedMeasures()(0,ptOrdinal);
        lt1IntegralsExpected(0,dofIndex) += value * weight;
      }
    }

    Intrepid::FieldContainer<double> lt1IntegralsActual(1,vOrder->totalDofs());
    lt1->integrate(lt1IntegralsActual, vOrder, basisCache);
    SerialDenseWrapper::roundZeros(lt1IntegralsActual,1e-15);
    SerialDenseWrapper::roundZeros(lt1IntegralsExpected,1e-15);
    TEST_COMPARE_FLOATING_ARRAYS(lt1IntegralsExpected, lt1IntegralsActual, tol);

    for (unsigned sideOrdinal=0; sideOrdinal<basisCache->cellTopology()->getSideCount(); sideOrdinal++)
    {
      BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
      out << "Checking space-time LinearTerm integration on side " << sideOrdinal << endl;
      int numPoints = sideCache->getRefCellPoints().dimension(0);

      // compute values:
      Intrepid::FieldContainer<double> lt2Values(1,vBasis->getCardinality(),numPoints);
      lt2->values(lt2Values, vID, vBasis, sideCache);

      // compute integrals:
      Intrepid::FieldContainer<double> lt2IntegralsExpected(1,vOrder->totalDofs());
      for (int basisOrdinal=0; basisOrdinal<vBasis->getCardinality(); basisOrdinal++)
      {
        int dofIndex = vOrder->getDofIndex(vID, basisOrdinal);
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          double value = lt2Values(0,basisOrdinal,ptOrdinal);
          double weight = sideCache->getWeightedMeasures()(0,ptOrdinal);
          lt2IntegralsExpected(0,dofIndex) += value * weight;
        }
      }

      Intrepid::FieldContainer<double> lt2IntegralsActual(1,vOrder->totalDofs());
      lt2->integrate(lt2IntegralsActual, vOrder, sideCache);
      SerialDenseWrapper::roundZeros(lt2IntegralsActual,1e-15);
      SerialDenseWrapper::roundZeros(lt2IntegralsExpected,1e-15);
      TEST_COMPARE_FLOATING_ARRAYS(lt2IntegralsExpected, lt2IntegralsActual, tol);
    }

  }

  TEUCHOS_UNIT_TEST( LinearTerm, FauxSpaceTimeIntegrationByPartsInTime_1D )
  {
    testFauxSpaceTimeIntegrationByPartsInTime(1,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, FauxSpaceTimeIntegrationByPartsInTime_2D )
  {
    testFauxSpaceTimeIntegrationByPartsInTime(2,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_1D )
  {
    testSpaceTimeIntegrationByPartsInTime(1,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_2D )
  {
    testSpaceTimeIntegrationByPartsInTime(2,out,success);
  }

//  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeIntegrationByPartsInTime_3D )
//  {
//    testSpaceTimeIntegrationByPartsInTime(3,out,success);
//  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_1D )
  {
    testSpaceTimeNonzeroTimeDerivative(1,out,success);
  }

  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_2D )
  {
    testSpaceTimeNonzeroTimeDerivative(2,out,success);
  }

//  TEUCHOS_UNIT_TEST( LinearTerm, SpaceTimeNonzeroTimeDerivative_3D )
//  {
//    testSpaceTimeNonzeroTimeDerivative(3,out,success);
//  }
} // namespace
