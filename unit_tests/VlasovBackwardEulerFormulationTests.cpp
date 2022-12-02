//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  VlasovBackwardEulerFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "VlasovBackwardEulerFormulation.h"

#include "BF.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace {
  const double TEST_X0     = 0.0;
  const double TEST_KX     = 0.2; // wavelength
  const double TEST_XWIDTH = 2 * M_PI / TEST_KX;
  const double TEST_V0     = -10.0;
  const double TEST_VWIDTH = 20.0;

  const double TEST_Q      = 1.0;
  const double TEST_M      = 1.0;
  const double TEST_EPS0   = 1e-4;

  VlasovBackwardEulerFormulation getFormulation(const int xElements, const int vElements, const int H1Order, const bool usePeriodicBCs, double q = TEST_Q,
                                                bool homogeneousPotentialBCs = true)
  {
    // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
    double eps0   = TEST_EPS0;
    double m      = TEST_M;
    double x0     = TEST_X0;
    double xWidth = TEST_XWIDTH;
    double v0     = TEST_V0;
    double vWidth = TEST_VWIDTH;
    
    const double dt = 0.1;
    const int delta_k = 2;
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    const bool useSerendipity = true;
    VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
    form.setRunQuietly(true);
    auto poissonMesh = poissonConfig->configurationSpaceMesh();
    
    auto poissonBC = BC::bc();
    if (homogeneousPotentialBCs)
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      EntitySetPtr x1EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({TEST_X0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      IndexType x1Vertex;
      bool x1Found = poissonMeshTopo->getVertexIndex({TEST_X0+TEST_XWIDTH}, x1Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x1Found, std::invalid_argument, "x1 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      x1EntitySet->addEntity(0, x1Vertex); // add entity of dimension 0 with ID x1Vertex to the x1 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=TEST_X0)
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      
      if (!usePeriodicBCs)
      {
        static const int POISSON_TAG_X1 = 1; // right (x=TEST_X0 + TEST_XWIDTH)
        poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, x1EntitySet);
        poissonBC->addDirichlet(VE_hat, POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
      }
    }
    
    poissonConfig->solution()->setBC(poissonBC);
    
    return form;
  }

  class Maxwellian1V : public TFunction<double>
  {
    double _u;
    double _sigma;
    double _n;
  public:
    Maxwellian1V(double u, double sigma, double n)
    :
    _u(u),
    _sigma(sigma),
    _n(n)
    {}
    
    using TFunction<double>::values;
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
      const int numCells  = phaseSpacePoints.dimension(0);
      const int numPoints = phaseSpacePoints.dimension(1);
      
      const double weight = _n / (sqrt(2. * M_PI) * _sigma);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
          const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
          values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
        }
      }
    }
  };

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, InflowConditions )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr v2 = Function::yn(2);
    FunctionPtr v3 = Function::yn(3);
    FunctionPtr f0 = x * v2;
    
    const int xElements = 4;
    const int vElements = 4;
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const bool usePeriodicBCs = false;

    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);
    
    auto vlasovSoln = form.vlasovSolution();
    
    // want to set up RHS/BCs so that f0 is a stationary solution
    auto w = form.w(); // test function for Boltzmann RHS
    vlasovSoln->rhs()->addTerm(w * v3); // this is the result of evaluating the LHS at f0
    
    form.setInitialAndInflowConditions(f0, f0);
        
    const int X_COMP = 0; // zero-based comp for E_fieldVar() call below
    auto poissonSoln = form.configurationSpaceSolution();
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(X_COMP), poissonSoln);
    
    FunctionPtr E0 = Function::zero();
    
    double err_l2 = (E_soln-E0)->l2norm(poissonSoln->mesh());
    
    double tol = 1e-13;
    TEST_COMPARE(err_l2, <=, tol);
        
    auto poissonBC = BC::bc();
    // set 0 potential BCs at both ends of mesh
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      EntitySetPtr x1EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({TEST_X0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      IndexType x1Vertex;
      bool x1Found = poissonMeshTopo->getVertexIndex({TEST_X0+TEST_XWIDTH}, x1Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x1Found, std::invalid_argument, "x1 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      x1EntitySet->addEntity(0, x1Vertex); // add entity of dimension 0 with ID x1Vertex to the x1 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=TEST_X0)
      static const int POISSON_TAG_X1 = 1; // right (x=TEST_X0 + TEST_XWIDTH)
      
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, x1EntitySet);
      
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
    }
    
    poissonSoln->setBC(poissonBC);
//    Teuchos::RCP<HDF5Exporter> vlasovViz = Teuchos::rcp( new HDF5Exporter (vlasovSoln->mesh(), "BoltzmannInflowConditionsTests1D1V_Stationary", "."));
//    vlasovViz->exportSolution(vlasovSoln, form.t());
    
    FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
    out << "Comparing f0 with f_soln after setting initial conditions.\n";
    double soln_magnitude = f0->l2norm(vlasovSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
    
    form.takeTimeStep();
  
//    vlasovViz->exportSolution(vlasovSoln, form.t());
    
    out << "Comparing f0 with f_soln after taking time step.\n";
    soln_magnitude = f0->l2norm(vlasovSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
  }

// Disabling this test for now.  It is based on what appears to be a pretty fragile test in Boltzmann1D1VFormulationTests, where it
// tested an explicit time step (which we don't support in VlasovBackwardEulerFormulation).  Switching the original to implicit causes
// a (different) test failure there…

//  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PatchTestInitialConditions )
//  {
//    const double q      = TEST_Q;
//    const double eps0   = TEST_EPS0;
//    const double m      = TEST_M;
//
//    FunctionPtr x = Function::xn(1);
//    FunctionPtr v = Function::yn(1);
//    const double meshWidth = TEST_XWIDTH;
//    const double vWidth = TEST_VWIDTH;
//    FunctionPtr f0   = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the domain (n0 should be zero)
//    const double n0  = 0.0;
////    FunctionPtr rho0 = q * (x - (TEST_X0 + TEST_XWIDTH) / 2.0 - n0 ); // f0, integrated in v dimension
//
//    const int xElements = 3;
//    const int vElements = 3;
//    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
//    const bool usePeriodicBCs = false;
//
//    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);
//    form.setEnforceQuasiNeutrality(true);
//
//    form.setUseGeometricMultigrid(true);
//    form.setInitialConditions(f0);
//
//    const int X_COMP = 0; // zero-based comp for E_fieldVar() call below
//    auto poissonSoln = form.configurationSpaceSolution();
//    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(X_COMP), poissonSoln);
//
//    FunctionPtr E0 = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x - n0 * x);
//    double E0_avg = E0->integrate(poissonSoln->mesh()) / meshWidth;
//    E0 = E0 - E0_avg;
//
//    double soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
//    double err_l2 = (E_soln-E0)->l2norm(poissonSoln->mesh());
//
//    double tol = 1e-12;
//    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
//
//    auto vlasovSoln = form.vlasovSolution();
//    FunctionPtr n = Function::normal();
//    FunctionPtr sgn = Function::sideParity();
//    FunctionPtr tn_0 = f0 * (v * n->x() + (q/m) * E0 * n->y()) * sgn;
//
//    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
//
//    double actualMagnitude = tn_soln->l2norm(vlasovSoln->mesh());
//    out << "actualMagnitude = " << actualMagnitude << std::endl;
//
//    TEST_ASSERT(tn_0->boundaryValueOnly());
//    TEST_ASSERT(tn_soln->boundaryValueOnly());
//
//    soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
//    err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
//    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
//
//    FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
//    soln_magnitude = f0->l2norm(vlasovSoln->mesh());
//    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
//    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
//
////    PhaseSpaceVisualization phaseSpaceViz(vlasovSoln, form.f(), "phaseSpaceOutput1D1V", ".");
////    phaseSpaceViz.exportTimeStep(form.t());
//
//    out << "Testing solve with patch test.\n";
//
//    // want to set up RHS, bcs so that f0 is a stationary solution
//    // d/dvx of f0 is 0
//    // d/dx of f0 is 1/TEST_VWIDTH
//    auto w = form.w(); // test function for Boltzmann RHS
//    auto vx = Function::yn(1); // 1D1V: y dimension is v_x
//    vlasovSoln->rhs()->addTerm(w * vx / TEST_VWIDTH); // this is the result of evaluating the LHS at f0
//
//    // BCs:
//    BCPtr vlasovBCs = BC::bc();
//    vlasovBCs->addDirichlet(form.t_n_hat(), SpatialFilter::allSpace(), tn_0);
//    form.vlasovSolution()->setBC(vlasovBCs);
//
//    BCPtr poissonBCs = BC::bc();
//    poissonBCs->addDirichlet(form.configurationSpaceFormulation()->En_hat(), SpatialFilter::allSpace(), E0);
//    form.configurationSpaceSolution()->setBC(poissonBCs);
//
//    form.takeTimeStep();
//
////    phaseSpaceViz.exportTimeStep(form.t());
//
//    out << "Comparing tn_0 with tn_soln after taking time step.\n";
//    soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
//    err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
//    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
//
//    out << "Comparing f0 with f_soln after taking time step.\n";
//    soln_magnitude = f0->l2norm(vlasovSoln->mesh());
//    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
//    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
//  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, RefineUniformly )
  {
    const double q      = 1.0;
    const int xElements = 3;
    const int vElements = 4;
    const int H1Order   = 1;
    const bool usePeriodicBCs = false;
    
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);
    
    int expectedActiveElementCountVlasov = xElements * vElements;
    int actualActiveElementCountVlasov   = form.vlasovMesh()->numActiveElements();
    
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    
    int expectedActiveElementCountPoisson = xElements;
    int actualActiveElementCountPoisson   = form.configurationSpaceMesh()->numActiveElements();
    
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
    
    int numUniformRefinements = 1;
    form.refineUniformly(numUniformRefinements);
    
    int vlasovCountMultiplier  = 4;
    int poissonCountMultiplier = 2;
    
    expectedActiveElementCountVlasov  *= vlasovCountMultiplier;
    expectedActiveElementCountPoisson *= poissonCountMultiplier;
    
    actualActiveElementCountVlasov  = form.vlasovMesh()->numActiveElements();
    actualActiveElementCountPoisson = form.configurationSpaceMesh()->numActiveElements();
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
    
    numUniformRefinements = 2;
    form.refineUniformly(numUniformRefinements);
    
    vlasovCountMultiplier  = 4*4;
    poissonCountMultiplier = 2*2;
    
    expectedActiveElementCountVlasov  *= vlasovCountMultiplier;
    expectedActiveElementCountPoisson *= poissonCountMultiplier;
    
    actualActiveElementCountVlasov  = form.vlasovMesh()->numActiveElements();
    actualActiveElementCountPoisson = form.configurationSpaceMesh()->numActiveElements();
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, SideSetInflow )
  {
    const double q = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int spatialInflowTag = form.tagVlasovSpatialInflow();

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, spatialInflowTag);
    
    std::set<IndexType> sideIndices;
    
    const int phaseSpaceDim = vlasovMeshTopo->getDimension();
    const int sideDim       = phaseSpaceDim - 1;
    for (auto sideSet : actualSideSets)
    {
      int maxEntityDimension = sideSet->maxEntityDimension();
      TEST_EQUALITY(maxEntityDimension, sideDim);
      
      auto entityIndices = sideSet->entitiesOfDimension(sideDim);
      sideIndices.insert(entityIndices.begin(), entityIndices.end());
            
      // check that there aren't any non-side entities in the set (that would be weird)
      for (int d=0; d<sideDim; d++)
      {
        TEST_EQUALITY(sideSet->entitiesOfDimension(d).size(), 0);
      }
    }
    
    int expectedSideCount = vElements; // vElements/2 from upper left, vElements/2 from lower right
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int SPACE_COMPONENT_INDEX_ZERO_BASED    = 0;
    const int VELOCITY_COMPONENT_INDEX_ZERO_BASED = 1;
    
    // next, check that the sides that did match each have v>=0, and x==0.
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        const bool leftVertex  = vertex[SPACE_COMPONENT_INDEX_ZERO_BASED] == TEST_X0;
        const bool rightVertex = vertex[SPACE_COMPONENT_INDEX_ZERO_BASED] == TEST_X0 + TEST_XWIDTH;
        const bool leftInflow  = leftVertex  && (vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED] >= 0);
        const bool rightInflow = rightVertex && (vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED] <= 0);
        TEST_ASSERT(leftInflow || rightInflow);
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, TimeStepStabilityUltraweakImplicit )
  {
    // confirm, for an example problem, that || u^{n+1} || ≤ || u^n ||
    // we set everything up as in two-stream instability, and use a coarse (2 x 2) high-order (H1Order = 4) mesh.
    // We run with both our usual test space norm, and the automatic graph norm for the bilinear form.
    // We use dt's that go from 1e0 to 1e-5.
    // run test suite with --details=ALL argument to see numerical details.
    
    // set things up as in two-stream
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double v0     = -10.0;
    double vWidth =  20.0;
    
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    double n = 0.5 * n0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
    FunctionPtr f0           = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    
    const int xElements = 2;
    const int vElements = 2;
    const int H1Order   = 4; // cubic L^2 functions
    
    const int delta_k = 2;
    const double dt = 1.0;
    const bool usePeriodicBCs = true;
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    const bool useSerendipity = true;
    VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
    form.setRunQuietly(true);
    auto poissonMesh = poissonConfig->configurationSpaceMesh();
    // set a zero BC on Poisson at x0
    auto poissonBC = BC::bc();
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({TEST_X0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=TEST_X0)
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
    }
    poissonConfig->solution()->setBC(poissonBC);
    
    form.setEnforceQuasiNeutrality(true);
    form.setInitialAndInflowConditions(f0, f0);
    
    auto vlasovSoln = form.vlasovSolution();
    IPPtr defaultIP = vlasovSoln->ip(); // our default inner product for Boltzmann; at present, this is one inspired by Führer et al.
    IPPtr graphNorm = vlasovSoln->bf()->graphNorm(); // the usual default choice for generic DPG discretization (has unit weight on the extra L^2 terms)
    
    std::vector<IPPtr> ipsToTest {defaultIP, graphNorm};
    
    std::vector<double> dtsToTest {1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    
    for (const auto & ipToTest : ipsToTest)
    {
      out << "************************************************************\n";
      out << "testing IP: " << ipToTest->displayString() << std::endl;
      vlasovSoln->setIP(ipToTest);
      
      for (const auto & dt : dtsToTest)
      {
        out << "testing dt: " << dt << std::endl;
        form.setTimeStep(dt);
        
        form.setInitialConditions(f0);
        
        // we should measure the projected f0 rather than the exact.
        
        FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
        const double f0_l2 = f_soln->l2norm(form.vlasovMesh());
        out << "f0_l2: " << f0_l2 << std::endl;

        form.takeTimeStep();
        
        const double f1_l2 = f_soln->l2norm(form.vlasovMesh());
        
        out << "f1_l2: " << f1_l2 << std::endl;
        out << "f1_l2 / f0_l2: " << f1_l2 / f0_l2 << std::endl;
        out << "1 - f1_l2 / f0_l2 = " << 1 - f1_l2 / f0_l2 << std::endl;
        
        TEST_COMPARE(f1_l2, <=, f0_l2);
      }
    }
  }
} // end anonymous namespace
