//
// © 2016-2022 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MimeticVlasovSpaceTimeFormulationTests.hpp
//
//  Created by Nate Roberts on 12/19/2022.
//
//  Tests to be included in MimeticVlasovSpaceTimeFormulationTests.cpp, within its anonymous namespace, after the definition of various
//  patch-testing methods, so that we can use much the same tests for both.

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, MimeticVariablesDefined )
{
  const double q    = 0; // 0 charge --> zero electric field
  
  const int xElements = 1;
  const int tElements = 1;
  const int vElements = 1;
  const int H1Order   = 1; // quadratic L^2 functions; allows exact representation of analytic solution trace
  
  const bool usePeriodicBCs    = false;
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
  
  auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, basisFamily, useMimeticDPG);
  
  const std::string & g_a_name = MimeticVlasovSpaceTimeFormulation::S_GA[0];
  const std::string & g_v_name = MimeticVlasovSpaceTimeFormulation::S_GV[0];
  const std::string & g_c_name = MimeticVlasovSpaceTimeFormulation::S_GC;
  
  auto vlasovVF = form->vlasovSolution()->bf()->varFactory();
  
  vector<string> mimeticVarNames {g_a_name, g_c_name, g_v_name};
  
  for (const auto & name : mimeticVarNames)
  {
    auto var = vlasovVF->trialVar( name );
    if (var == Teuchos::null)
    {
      out << "FAILURE: expected variable " << name << " to be defined, but it was not found.\n";
      success = false;
    }
  }
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, NextTimeSlab )
{
  bool useMimeticDPG = true;
  testNextTimeSlab(useMimeticDPG, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PoissonTwoStreamSteadyState )
{
  const double tolRelativeToProjectionError = 2.50; // we allow actual error to be up to 350% of the best approximation error for f, tn
  const bool useMimeticDPG = true;
  testPoissonTwoStreamSteadyState(tolRelativeToProjectionError, useMimeticDPG, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PoissonPatchTest_f_xv2_E_zero )
{
  const double q = 0; // 0 charge --> zero electric field
  
  const int xElements = 4;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 4; // cubic L^2 functions; allows exact representation of analytic solution trace (g_v = x * v^3)
  
  const int spaceDim = 1;
  const bool spaceTimeMesh = true;
  
  FunctionPtr x  = Function::xn(1);
  FunctionPtr v  = getVx(spaceDim, spaceTimeMesh);
  FunctionPtr v2 = v * v;
  FunctionPtr f_exact = x * v2;
  FunctionPtr E_exact = Function::zero();
  FunctionPtr VE_exact = Function::zero();
  
  const bool overrideVlasovBCs = false;
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
  
  const double tol = 1e-12;
  
  patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PoissonPatchTest_f_xv2_E_zero_Serendipity )
{
  const double q    = 0; // 0 charge --> zero electric field
  
  const int xElements = 4;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 6; // cubic L^2 functions; allows exact representation of analytic solution trace (g_v = x * v^3)
  
  const int spaceDim = 1;
  const bool spaceTimeMesh = true;
  
  FunctionPtr x  = Function::xn(1);
  FunctionPtr v  = getVx(spaceDim, spaceTimeMesh);
  FunctionPtr v2 = v * v;
  FunctionPtr f_exact = x * v2;
  FunctionPtr E_exact = Function::zero();
  FunctionPtr VE_exact = Function::zero();
  
  const bool overrideVlasovBCs = false;
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
  
  const double tol = 1e-11;
  
  patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PatchTest_f_zero_E_1 )
{
  const double q = TEST_Q;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr f_exact   = Function::constant(0.0);
  FunctionPtr E_exact = Function::constant(1.0);
  FunctionPtr V_E_exact = -x;
  
  const int xElements = 3;
  const int tElements = 2;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 2; // linear L^2 functions (allows exact VE)
  
  const bool overrideVlasovBCs = false;
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
  
  const double tol = 1e-12;
  
  patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PatchTest_f_zero_E_1_Serendipity )
{
  const double q = TEST_Q;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr f_exact   = Function::constant(0.0);
  FunctionPtr E_exact = Function::constant(1.0);
  FunctionPtr V_E_exact = -x;
  
  const int xElements = 3;
  const int tElements = 2;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 2; // linear L^2 functions (allows exact VE)
  
  const bool overrideVlasovBCs = false;
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
  
  const double tol = 1e-12;
  
  patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PoissonPatchTest_f_constant_E_linear )
{
  const double q      = TEST_Q;
  
  const int spaceDim = 1;
  const bool spaceTimeMesh = true;
  auto vx = getVx(spaceDim, spaceTimeMesh);
  
  FunctionPtr x       = Function::xn(1);
  FunctionPtr f_exact = Function::constant(1.0);
  FunctionPtr E_exact =   (q / TEST_EPS0 * TEST_VWIDTH) * x;
  FunctionPtr V_E     = - (q / TEST_EPS0 * TEST_VWIDTH) / 2.0 * x * x;

  const int xElements = 3;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  const double tol = 1e-10;
  
  const bool outputViz = false;
  const std::string vizPrefix = "PatchTest_f_constant_E_linear";
  const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
  
  patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG,
                         tol, out, success, outputViz, vizPrefix);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PoissonPatchTest_f_constant_E_linear_Serendipity )
{
  const double q      = TEST_Q;
  
  const int spaceDim = 1;
  const bool spaceTimeMesh = true;
  auto vx = getVx(spaceDim, spaceTimeMesh);
  
  FunctionPtr x       = Function::xn(1);
  FunctionPtr f_exact = Function::constant(1.0);
  FunctionPtr E_exact =   (q / TEST_EPS0 * TEST_VWIDTH) * x;
  FunctionPtr V_E     = - (q / TEST_EPS0 * TEST_VWIDTH) / 2.0 * x * x;

  const int xElements = 3;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  const double tol = 1e-9;
  
  const bool outputViz = false;
  const std::string vizPrefix = "PatchTest_f_constant_E_linear";
  const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
  
  patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success, outputViz, vizPrefix);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic )
{
  const double q      = TEST_Q;
  const double eps0   = TEST_EPS0;
  
  FunctionPtr x = Function::xn(1);
  const double vWidth = TEST_VWIDTH;
  FunctionPtr f_exact = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
  
  FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x);
  FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x);

  const int xElements = 3;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  const double tol = 1e-11;
  
  const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
  
  const bool outputViz = true;
  const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
  patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                         out, success, outputViz, testPrefix);
}

TEUCHOS_UNIT_TEST( MimeticVlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_Serendipity )
{
  const double q      = TEST_Q;
  const double eps0   = TEST_EPS0;
  
  FunctionPtr x = Function::xn(1);
  const double vWidth = TEST_VWIDTH;
  FunctionPtr f_exact = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
  
  FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x);
  FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x);

  const int xElements = 3;
  const int tElements = 1;
  const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
  const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  const double tol = 1e-11;
  
  const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
  const bool useMimeticDPG     = true;
  
  const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
  
  const bool outputViz = true;
  const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
  patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                         out, success, outputViz, testPrefix);
}
