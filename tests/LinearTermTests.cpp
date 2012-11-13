#include "LinearTermTests.h"

void LinearTermTests::SetUp()
{
  //  VarPtr v1, v2, v3; // HGRAD members (test variables)
  //  VarPtr q1, q2, q3; // HDIV members (test variables)
  //  VarPtr u1, u2, u3; // L2 members (trial variables)
  //  VarPtr u1_hat, u2_hat; // trace variables
  //  VarPtr u3_hat_n; // flux variable

  sine_x = Teuchos::rcp( new Sine_x );
  cos_y = Teuchos::rcp( new Cosine_y );

  v1 = varFactory.testVar("v_1", HGRAD);
  v2 = varFactory.testVar("v_2", HGRAD);
  v3 = varFactory.testVar("v_3", HGRAD);

  q1 = varFactory.testVar("q_1", HDIV);
  q2 = varFactory.testVar("q_2", HDIV);
  q3 = varFactory.testVar("q_3", HDIV);

  u1 = varFactory.fieldVar("u_1", HGRAD);
  u2 = varFactory.fieldVar("u_2", HGRAD);
  u3 = varFactory.fieldVar("u_3", HGRAD);

  u1_hat = varFactory.traceVar("\\widehat{u}_1");
  u2_hat = varFactory.traceVar("\\widehat{u}_2");

  u3_hat_n = varFactory.fluxVar("\\widehat{u}_3n");

  bf = Teuchos::rcp(new BF(varFactory)); // made-up bf for Mesh + previous solution tests

  bf->addTerm(u1_hat, q1->dot_normal());
  bf->addTerm(u1, q1->x());
  bf->addTerm(u2, q1->y());

  bf->addTerm(u3_hat_n, v1);
  bf->addTerm(u3, v1);

  int polyOrder = 3, testToAdd = 2;
  Teuchos::RCP<shards::CellTopology> quadTopoPtr;
  //  quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));

  // define nodes for mesh
  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  int horizontalElements = 2, verticalElements = 2;

  mesh = Mesh::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bf, polyOrder+1, polyOrder+1+testToAdd);

  ElementTypePtr elemType = mesh->getElement(0)->elementType();
  trialOrder = elemType->trialOrderPtr;
  testOrder = elemType->testOrderPtr;

  basisCache = Teuchos::rcp(new BasisCache(elemType, mesh));

  vector<int> cellIDs;
  cellIDs.push_back(0); 
  cellIDs.push_back(1);
  cellIDs.push_back(2);
  cellIDs.push_back(3);
  bool createSideCacheToo = true;

  basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);
}

TEST_F(LinearTermTests, TestSums)
{
  LinearTermPtr sum = v1 + v2;

  ASSERT_EQ(sum->summands().size(), 2) << "sum has the wrong number of summands";

  LinearSummand first_summand = sum->summands()[0];
  LinearSummand second_summand = sum->summands()[1];

  VarPtr first_var = first_summand.second;
  EOperatorExtended first_op = first_var->op();

  VarPtr second_var = second_summand.second;
  EOperatorExtended second_op = second_var->op();

  ASSERT_EQ(v1->ID(), first_var->ID()) << "first summand isn't v1";
  ASSERT_EQ(OP_VALUE, first_var->op()) << "first op isn't VALUE";
  ASSERT_EQ(v2->ID(), second_var->ID()) << "second summand isn't v1";
  ASSERT_EQ(OP_VALUE, second_var->op()) << "second op isn't VALUE";

  ASSERT_NE(sum->varIDs().end(), sum->varIDs().find(v1->ID())) << "sum->varIDs() doesn't include v1";
  ASSERT_NE(sum->varIDs().end(), sum->varIDs().find(v2->ID())) << "sum->varIDs() doesn't include v2";

  ASSERT_EQ(2, sum->varIDs().size()) << "sum->varIDs() doesn't have the expected size (expected 2; is " << sum->varIDs().size() << ")";
}

TEST_F(LinearTermTests, TestIntegration)
{
  // for now, we just check the consistency: for LinearTerm a = b + c, does a->integrate
  // give the same values as b->integrate + c->integrate ?

  EXPECT_TRUE(checkLTSumConsistency(1 * v1, 1 * v2, testOrder, basisCache))
    << "(v1 + v2)->integrate not consistent with sum of summands integration";
  EXPECT_TRUE(checkLTSumConsistency(sine_x * v1, 1 * v2, testOrder, basisCache))
    << "(sine_x * v1 + v2)->integrate not consistent with sum of summands integration.\n";
  EXPECT_TRUE(checkLTSumConsistency(1 * q1->div(), 1 * q2->x(), testOrder, basisCache))
    << "(q1->div() + q2->x())->integrate not consistent with sum of summands integration.\n";
  EXPECT_TRUE(checkLTSumConsistency(1 * u1, 1 * u2, testOrder, basisCache))
    << "(u1 + u2)->integrate not consistent with sum of summands integration.\n";
  EXPECT_TRUE(checkLTSumConsistency(1 * u1, sine_x * u2, testOrder, basisCache))
    << "(u1 + sine_x * u2)->integrate not consistent with sum of summands integration.\n";

  // now, same thing, but with boundary-value-only functions in the mix:
  // this next is a fairly complex test; may want to add a more granular one above...
  IPPtr ip = Teuchos::rcp(new IP);
  Teuchos::RCP<RHS> rhs = Teuchos::rcp(new RHSEasy);
  Teuchos::RCP<BC> bc = Teuchos::rcp(new BCEasy);
  SolutionPtr solution = Teuchos::rcp( new Solution(mesh,bc,rhs,ip) );
  // project some functions onto solution, so that something interesting is there:
  FunctionPtr u1_proj = sine_x;
  FunctionPtr u2_proj = cos_y;
  FunctionPtr u3_proj = u1_proj * u2_proj;
  map<int, FunctionPtr> solnToProject;
  solnToProject[u1->ID()] = u1_proj;
  solnToProject[u2->ID()] = u2_proj;
  solnToProject[u3->ID()] = u3_proj;
  solnToProject[u1_hat->ID()] = u1_proj;
  solnToProject[u2_hat->ID()] = u2_proj;
  // u3_hat_n isn't too much like a 'real' bilinear form, in that u3 itself is a scalar
  // this is just a test, so I'm not worried about it...
  solnToProject[u3_hat_n->ID()] = u3_proj;

  solution->projectOntoMesh(solnToProject);

  LinearTermPtr bfTestFunctional = bf->testFunctional(solution);

  // bf->addTerm(u1, q1->x());
  // bf->addTerm(u2, q1->y());
  // bf->addTerm(u3, v1);

  //  bf->addTerm(u1_hat, q1->dot_normal());
  //  bf->addTerm(u3_hat_n, v1);

  LinearTermPtr testFunctionalNoBoundaryValues = u1_proj * q1->x() + u2_proj * q1->y() + u3_proj * v1;

  FunctionPtr u1_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u1_hat) );
  FunctionPtr u2_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u2_hat) );
  FunctionPtr u3_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u3_hat_n) );
  LinearTermPtr testFunctionalBoundaryValues = u1_hat_prev * q1->dot_normal() + u3_hat_prev * v1;

  EXPECT_TRUE(checkLTSumConsistency(testFunctionalNoBoundaryValues, testFunctionalBoundaryValues, testOrder, basisCache))
    << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
  EXPECT_TRUE(checkLTSumConsistency(testFunctionalBoundaryValues, bfTestFunctional - testFunctionalBoundaryValues, testOrder, basisCache))
    << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
  EXPECT_TRUE(checkLTSumConsistency(testFunctionalNoBoundaryValues, bfTestFunctional - testFunctionalNoBoundaryValues, testOrder, basisCache))
    << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
}

TEST_F(LinearTermTests, TestBoundaryPlusVolumeTerms)
{
  // notion is integration by parts:
  // (div f, v) = < f * n, v > - (f, grad v)

  // We perform two subtests for each test: first we try with a particular
  // function substituted for the variable.  Second, we integrate over the
  // basis for the mesh (i.e. we test a whole bunch of functions, whose
  // precise definition is a bit complicated).

  // A third test is against the two-term LinearTerm::integrate() method.
  // This doesn't do integration by parts, but rather tests that
  // (u + u->dot_normal(), v) = (u,v) + (u->dot_normal(), v)

  /////////////   FIRST TEST  ////////////////

  // start simply: define f to be (x, 0)
  // (div f, v) = (1, v)
  // < f * n, v > - (f, grad v) = < x n1, v > - ( x, v->dx() )

  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr y2 = Teuchos::rcp( new Yn(2) );
  FunctionPtr x3 = Teuchos::rcp( new Xn(3) );
  FunctionPtr y3 = Teuchos::rcp( new Yn(3) );

  vector< FunctionPtr > f_fxns;
  f_fxns.push_back( Function::vectorize( x,    Function::zero() ) ); // div of this = 1
  f_fxns.push_back( Function::vectorize( x2 / 6.0, x2 * y / 2.0 ) ); // div of this = x / 3 + x^2 / 2

  for ( vector< FunctionPtr >::iterator fIt = f_fxns.begin(); fIt != f_fxns.end(); fIt++) {
    FunctionPtr vector_fxn = *fIt;
    LinearTermPtr lt_v = vector_fxn->div()*v1;

    // part a: substitute v1 = x*y^2

    FunctionPtr v1_value = x*y2;
    map< int, FunctionPtr > var_values;
    var_values[v1->ID()] = v1_value;

    double expectedValue = lt_v->evaluate(var_values, false)->integrate(mesh);

    FunctionPtr n = Function::normal();

    LinearTermPtr ibp = vector_fxn * n * v1 - vector_fxn * v1->grad();

    double boundaryIntegralSum = ibp->evaluate(var_values,true)->integrate(mesh);
    double volumeIntegralSum   = ibp->evaluate(var_values,false)->integrate(mesh);
    double actualValue = boundaryIntegralSum + volumeIntegralSum;

    double tol = 1e-14;
    EXPECT_NEAR(expectedValue, actualValue, tol);

    // part b: integrate the bases over each of the cells:
    int num_dofs = testOrder->totalDofs();
    FieldContainer<double> integrals_expected( mesh->numElements(), num_dofs );
    FieldContainer<double> integrals_actual( mesh->numElements(), num_dofs );

    lt_v->integrate(integrals_expected,testOrder,basisCache);
    ibp->integrate(integrals_actual,testOrder,basisCache);

    double maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual, integrals_expected, tol, maxDiff))
      << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff;

    // just on the odd chance that ordering makes a difference, repeat this test with the opposite order in ibp:
    ibp =  - vector_fxn * v1->grad() + vector_fxn * n * v1;
    ibp->integrate(integrals_actual,testOrder,basisCache, false, false);

    maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual, integrals_expected, tol, maxDiff))
      << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff;

    // part c: two-term integrals
    FieldContainer<double> integrals_expected_two_term( mesh->numElements(), num_dofs, num_dofs);
    FieldContainer<double> integrals_actual_two_term( mesh->numElements(), num_dofs, num_dofs );
    LinearTermPtr ibp1 = vector_fxn * n * v1;
    LinearTermPtr ibp2 = - vector_fxn * v1->grad();
    lt_v->integrate(integrals_expected_two_term, testOrder, ibp1 + ibp2, testOrder, basisCache, false, false);
    lt_v->integrate(integrals_actual_two_term, testOrder, ibp1, testOrder, basisCache, false, false); // don't forceBoundary, don't sumInto
    lt_v->integrate(integrals_actual_two_term, testOrder, ibp2, testOrder, basisCache, false, true);  // DO sumInto

    maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual_two_term, integrals_expected_two_term, tol, maxDiff))
      << "two-term integration is not bilinear; maxDiff: " << maxDiff;

    // now, same thing but with the roles of ibp{1|2} and lt_v reversed:
    (ibp1 + ibp2)->integrate(integrals_expected_two_term, testOrder, lt_v, testOrder, basisCache, false, false);
    ibp1->integrate(integrals_actual_two_term, testOrder, lt_v, testOrder, basisCache, false, false); // don't forceBoundary, don't sumInto
    ibp2->integrate(integrals_actual_two_term, testOrder, lt_v, testOrder, basisCache, false, true);  // DO sumInto

    maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual_two_term, integrals_expected_two_term, tol, maxDiff))
      << "two-term integration is not bilinear; maxDiff: " << maxDiff;

    // now, test that two-term integration commutes in the two terms:
    ibp1->integrate(integrals_expected_two_term, testOrder, lt_v, testOrder, basisCache, false, false);
    lt_v->integrate(integrals_actual_two_term, testOrder, ibp1, testOrder, basisCache, false, false);

    // we expect the integrals to commute up to a transpose, so let's transpose one of the containers:
    transposeFieldContainer(integrals_expected_two_term);
    maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual_two_term, integrals_expected_two_term, tol, maxDiff))
      << "two-term integration does not commute for boundary value (ibp1); maxDiff: " << maxDiff;

    ibp2->integrate(integrals_expected_two_term, testOrder, lt_v, testOrder, basisCache, false, false);
    lt_v->integrate(integrals_actual_two_term, testOrder, ibp2, testOrder, basisCache, false, false);

    // we expect the integrals to commute up to a transpose, so let's transpose one of the containers:
    transposeFieldContainer(integrals_expected_two_term);
    maxDiff = 0;
    EXPECT_TRUE(fcsAgree(integrals_actual_two_term, integrals_expected_two_term, tol, maxDiff))
      << "two-term integration does not commute for volume value (ibp2); maxDiff: " << maxDiff;

    // part d: to suss out where the integration failure happens in the non-commuting case:
    //         1. Substitute v1 = 1 in ibp2; get a function ibp2_at_v1_equals_one back.
    //         2. Substitute v1 = 1 in lt_v; get a function lt_v_at_v1_equals_one back.
    //         3. Integrate ibp2_at_v1_equals_one * lt_v_at_v1_equals_one over the mesh.  Get a double result.
    //         4. Because basis is nodal, the representation for v1 = 1 is just all 1s for coefficients.
    //            Therefore, the sum of the entries in the integrals_*_two_term matrices will should match
    //            the function integral.  Whichever doesn't match is wrong.

    map< int, FunctionPtr > v1_equals_one;
    v1_equals_one[v1->ID()] = Function::constant(1.0);

    FunctionPtr ibp1_at_v1_equals_one = ibp1->evaluate(v1_equals_one,true);  // ibp1 has only a boundary term, so we just ask for this
    FunctionPtr ibp2_at_v1_equals_one = ibp2->evaluate(v1_equals_one,false); // ibp2 has no boundary terms, so we don't ask for these
    FunctionPtr lt_v_at_v1_equals_one = lt_v->evaluate(v1_equals_one,false); // lt_v also has no boundary terms

    FieldContainer<double> integrals_lt_v_first( mesh->numElements(), num_dofs, num_dofs);
    FieldContainer<double> integrals_ibp1_first( mesh->numElements(), num_dofs, num_dofs );
    FieldContainer<double> integrals_ibp2_first( mesh->numElements(), num_dofs, num_dofs );

    double lt_v_first_integral = 0.0, ibp1_first_integral = 0.0, ibp2_first_integral = 0.0;

    double integral = (ibp1_at_v1_equals_one * lt_v_at_v1_equals_one)->integrate(mesh);
    ibp1->integrate(integrals_ibp1_first,  testOrder, lt_v, testOrder, basisCache, false, false);
    lt_v->integrate(integrals_lt_v_first, testOrder, ibp1, testOrder, basisCache, false, false);

    for (int i=0; i<integrals_lt_v_first.size(); i++) {
      lt_v_first_integral += integrals_lt_v_first[i];
      ibp1_first_integral += integrals_ibp1_first[i];
    }

    double diff;
    diff = abs(lt_v_first_integral - integral);
    EXPECT_NEAR(integral, lt_v_first_integral, tol)
      << "Integral with v1=1 substituted does not match two-term integration of (lt_v,ibp1) with lt_v as this. diff = " << diff << "\n"
      << "lt_v_first_integral = " << lt_v_first_integral << endl
      << "    (true) integral = " << integral << endl;

    diff = abs(ibp1_first_integral - integral);
    EXPECT_NEAR(integral, ibp1_first_integral, tol)
      << "Integral with v1=1 substituted does not match two-term integration of (lt_v,ibp1) with ibp1 as this. diff = " << diff << "\n"
      << "ibp1_first_integral = " << ibp1_first_integral << endl
      << "    (true) integral = " << integral << endl;

    // now, do the same but for ibp2
    integral = (ibp2_at_v1_equals_one * lt_v_at_v1_equals_one)->integrate(mesh);
    ibp2->integrate(integrals_ibp2_first,  testOrder, lt_v, testOrder, basisCache, false, false);
    lt_v->integrate(integrals_lt_v_first,  testOrder, ibp2, testOrder, basisCache, false, false);

    // reset the sums:
    lt_v_first_integral = 0.0;
    ibp1_first_integral = 0.0;
    ibp2_first_integral = 0.0;
    for (int i=0; i<integrals_lt_v_first.size(); i++) {
      lt_v_first_integral += integrals_lt_v_first[i];
      ibp2_first_integral += integrals_ibp2_first[i];
    }

    diff = abs(lt_v_first_integral - integral);
    EXPECT_NEAR(integral, lt_v_first_integral, tol)
      << "Integral with v1=1 substituted does not match two-term integration of (lt_v,ibp2) with lt_v as this. diff = " << diff << "\n"
      << "lt_v_first_integral = " << lt_v_first_integral << endl
      << "    (true) integral = " << integral << endl;

    diff = abs(ibp2_first_integral - integral);
    EXPECT_NEAR(integral, ibp2_first_integral, tol)
      << "Integral with v1=1 substituted does not match two-term integration of (lt_v,ibp2) with ibp2 as this. diff = " << diff << "\n"
      << "ibp2_first_integral = " << ibp2_first_integral << endl
      << "    (true) integral = " << integral << endl;
  }
}

TEST_F(LinearTermTests, TestEnergyNorm)
{
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v1); // L^2 on an HGrad var
  ip->addTerm(q1); // L^2 on Hdiv var

  FunctionPtr one = Function::constant(1);
  LinearTermPtr identity = one*v1; 

  double norm = identity->energyNormTotal(mesh,ip); // should be equal to the sqrt of the measure of the domain [-1,1]^2

  double tol = 1e-15;
  EXPECT_NEAR(2.0, norm, tol);
}

TEST_F(LinearTermTests, TestRieszInversionAsProjection)
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  double eps = .01;
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 2; int pToAdd = 2;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
 
  int nCells = 2;
  int horizontalCells = nCells, verticalCells = nCells;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
						  confusionBF, H1Order, H1Order+pToAdd);    

  ElementTypePtr elemType = myMesh->getElement(0)->elementType();
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType, myMesh));
  
  vector<int> cellIDs;
  cellIDs.push_back(0); 
  cellIDs.push_back(1);
  cellIDs.push_back(2);
  cellIDs.push_back(3);
  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(myMesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);


  LinearTermPtr integrand = Teuchos::rcp(new LinearTerm);// residual
  LinearTermPtr integrandIBP = Teuchos::rcp(new LinearTerm);// residual
  LinearTermPtr integrandIBPReordered = Teuchos::rcp(new LinearTerm);// residual

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;  
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  FunctionPtr X = Teuchos::rcp(new Xn(1));
  FunctionPtr Y = Teuchos::rcp(new Yn(1));
  FunctionPtr testFxn1 = X;
  FunctionPtr testFxn2 = Y;
  FunctionPtr fxnToProject = X*Y + Function::constant(1.0);

  integrand->addTerm(fxnToProject*v);

  IPPtr sobolevIP = Teuchos::rcp(new IP);
  sobolevIP->addTerm(v);
  sobolevIP->addTerm(tau);

  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(myMesh, sobolevIP, integrand));
  riesz->computeRieszRep();

  FunctionPtr rieszFxn = Teuchos::rcp(new RepFunction(v,riesz));
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPts = basisCache->getPhysicalCubaturePoints().dimension(1);

  FieldContainer<double> valProject( numCells, numPts);
  FieldContainer<double> valExpected( numCells, numPts);
  rieszFxn->values(valProject,basisCache);
  fxnToProject->values(valExpected,basisCache);
  
  double maxDiff;
  double tol = 1e-13;
  EXPECT_TRUE(fcsAgree(valProject,valExpected,tol,maxDiff))
    << "Failed Riesz Inversion Projection test with maxDiff = " << maxDiff << endl;
}

TEST_F(LinearTermTests, TestMixedTermConsistency)
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  double eps = .01;
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
 
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 1; int pToAdd = 1;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
 
  int nCells = 1;
  int horizontalCells = nCells, verticalCells = nCells;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
						  confusionBF, H1Order, H1Order+pToAdd);    

  ElementTypePtr elemType = myMesh->getElement(0)->elementType();
  //  DofOrderingPtr testOrder = elemType->testOrderPtr;
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType, myMesh, true));
  

  LinearTermPtr integrandIBP = Teuchos::rcp(new LinearTerm);// residual

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;  
  FunctionPtr n = Teuchos::rcp(new UnitNormalFunction );
  FunctionPtr X = Teuchos::rcp(new Xn(1));
  FunctionPtr Y = Teuchos::rcp(new Yn(1));
  FunctionPtr testFxn1 = X;
  FunctionPtr testFxn2 = Y;
  FunctionPtr divTestFxn = testFxn1->dx() + testFxn2->dy();
  FunctionPtr vectorTest = testFxn1*e1 + testFxn2*e2;

  integrandIBP->addTerm(vectorTest*n*v + -vectorTest*v->grad()); // boundary term

  // define dummy IP to initialize riesz rep class, but just integrate RHS
  IPPtr dummyIP = Teuchos::rcp(new IP);
  dummyIP->addTerm(v);
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(myMesh, dummyIP, integrandIBP));  
  map<int,FieldContainer<double> > rieszRHS = riesz->integrateRHS();

  vector< ElementPtr > allElems = myMesh->activeElements(); // CHANGE TO DISTRIBUTED COMPUTATION
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();

    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();
    FieldContainer<double> physicalCellNodes = myMesh->physicalCellNodesForCell(cellID);

    vector<int> cellIDs;
    cellIDs.push_back(cellID); // just do one cell at a time

    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,myMesh, true));
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // create side cache if ip has boundary values 

    FieldContainer<double> rhsIBPValues(1,numTestDofs);
    integrandIBP->integrate(rhsIBPValues, testOrderingPtr, basisCache);
    FieldContainer<double> rieszValues(1,numTestDofs);
    (riesz->getRHS())->integrate(rieszValues, testOrderingPtr, basisCache);
    double maxDiff;
    double tol = 1e-13;
    FieldContainer<double> rhsIBPVals(numTestDofs);
    for (int i = 0;i< numTestDofs; i++){
      rhsIBPVals(i) = rhsIBPValues(0,i);
      //      cout << "riesz rhs values = " << rieszRHS[cellID](i) << ", rhsIBPValues = " << rhsIBPVals(i) << ", riesz returned values = " << rieszValues(0,i) << endl;
    }
    EXPECT_TRUE(fcsAgree(rieszRHS[cellID],rhsIBPVals,tol,maxDiff))
      << "Failed mixed term consistency test with maxDiff = " << maxDiff << " on cellID " << cellID<< endl; 
  }
}

TEST_F(LinearTermTests, TestRieszInversion)
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  double eps = .01;
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 1; int pToAdd = 1;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
 
  int nCells = 1;
  int horizontalCells = nCells, verticalCells = nCells;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> myMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
						  confusionBF, H1Order, H1Order+pToAdd);    

  ElementTypePtr elemType = myMesh->getElement(0)->elementType();
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType, myMesh));
  
  vector<int> cellIDs;
  vector<ElementPtr> elems = myMesh->activeElements();
  vector<ElementPtr>::iterator elemIt;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    cellIDs.push_back(cellID); 
  }
  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(myMesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);

  LinearTermPtr integrand = Teuchos::rcp(new LinearTerm);// residual
  LinearTermPtr integrandIBP = Teuchos::rcp(new LinearTerm);// residual

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;  
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  FunctionPtr X = Teuchos::rcp(new Xn(1));
  FunctionPtr Y = Teuchos::rcp(new Yn(1));
  FunctionPtr testFxn1 = X;
  FunctionPtr testFxn2 = Y;
  FunctionPtr divTestFxn = testFxn1->dx() + testFxn2->dy();
  FunctionPtr vectorTest = testFxn1*e1 + testFxn2*e2;

  integrand->addTerm(divTestFxn*v);
  integrandIBP->addTerm(vectorTest*n*v + -vectorTest*v->grad()); // boundary term

  IPPtr sobolevIP = Teuchos::rcp(new IP);
  sobolevIP->addTerm(v);
  //  sobolevIP->addTerm(tau);

  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(myMesh, sobolevIP, integrand));  
  //  riesz->setPrintOption(true);
  riesz->computeRieszRep();
  Teuchos::RCP<RieszRep> rieszIBP = Teuchos::rcp(new RieszRep(myMesh, sobolevIP, integrandIBP));
  riesz->setFunctional(integrandIBP);
  //  rieszIBP->setPrintOption(true);
  rieszIBP->computeRieszRep();

  FunctionPtr rieszOrigFxn = Teuchos::rcp(new RepFunction(v,riesz));
  FunctionPtr rieszIBPFxn = Teuchos::rcp(new RepFunction(v,rieszIBP));
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPts = basisCache->getPhysicalCubaturePoints().dimension(1);

  FieldContainer<double> valOriginal( numCells, numPts);
  FieldContainer<double> valIBP( numCells, numPts);
  rieszOrigFxn->values(valOriginal,basisCache);
  rieszIBPFxn->values(valIBP,basisCache);

  double maxDiff,maxDiff1,maxDiff2;
  double tol = 1e-15;
  EXPECT_TRUE(fcsAgree(valOriginal,valIBP,tol,maxDiff))
    << "Failed TestRieszInversion with maxDiff = " << maxDiff << endl;
}
