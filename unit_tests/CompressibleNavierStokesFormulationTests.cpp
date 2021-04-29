//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CompressibleNavierStokesFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/21/18.
//
//

#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "RHS.h"
#include "SimpleFunction.h"
#include "Solution.h"

using namespace Camellia;

#include <array>

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  static const double TEST_RE = 1e2;
  static const double TEST_PR = 0.713;
  static const double TEST_CV = 1.000;
  static const double TEST_GAMMA = 1.4;
  static const double TEST_CP = TEST_GAMMA * TEST_CV;
  static const double TEST_R = TEST_CP - TEST_CV;
  
  void testForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T,
                      FunctionPtr fc, FunctionPtr fm, FunctionPtr fe,
                      int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 2;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = TEST_RE; // Reynolds number
    
    bool useConformingTraces = true;
    auto form = CompressibleNavierStokesFormulation::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                       meshTopo, polyOrder, delta_k);
    
    // sanity checks that the constructor has set things up the way we assume:
    TEST_FLOATING_EQUALITY(form->Pr(),    TEST_PR,     tol);
    TEST_FLOATING_EQUALITY(form->Cv(),    TEST_CV,     tol);
    TEST_FLOATING_EQUALITY(form->gamma(), TEST_GAMMA,  tol);
    TEST_FLOATING_EQUALITY(form->Cp(),    TEST_CP,     tol);
    TEST_FLOATING_EQUALITY(form->R(),     TEST_R,      tol);
    TEST_FLOATING_EQUALITY(form->mu(),    1.0/TEST_RE, tol);
    
    BFPtr bf = form->bf();
    RHSPtr rhs = form->rhs();
    
    auto soln = form->solution();
    auto solnIncrement = form->solutionIncrement();
    auto mesh = soln->mesh();

    auto exact_fc = form->exactSolution_fc(u, rho, T);
    auto exact_fe = form->exactSolution_fe(u, rho, T);
    auto exact_fm = form->exactSolution_fm(u, rho, T);
    
    double fc_err = (fc - exact_fc)->l2norm(mesh, cubatureEnrichment);
    if (fc_err > tol)
    {
      success = false;
      out << "FAILURE: fc_err " << fc_err << " > tol " << tol << endl;
      out << "fc expected: " << fc->displayString() << endl;
      out << "fc actual:   " << exact_fc->displayString() << endl;
    }
    
    // here, we use the fact that we're operating in 1D:
    double fm_err = (fm - exact_fm[0])->l2norm(mesh);
    if (fm_err > tol)
    {
      success = false;
      out << "FAILURE: fm_err " << fm_err << " > tol " << tol << endl;
      out << "fm expected: " << fm->displayString() << endl;
      out << "fm actual:   " << exact_fm[0]->displayString() << endl;
    }
    
    double fe_err = (fe - exact_fe)->l2norm(mesh, cubatureEnrichment);
    if (fe_err > tol)
    {
      success = false;
      out << "FAILURE: fe_err " << fe_err << " > tol " << tol << endl;
      out << "fe expected: " << fe->displayString() << endl;
      out << "fe actual:   " << exact_fe->displayString() << endl;
    }
  }
  
  void testResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T, int cubatureEnrichment, double tol, bool steady, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 2;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = 1e2; // Reynolds number
    double dt    = 1.0; // only used for unsteady
    
    bool useConformingTraces = true;
    Teuchos::RCP<CompressibleNavierStokesFormulation> form;
    if (steady)
    {
      form = CompressibleNavierStokesFormulation::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                            meshTopo, polyOrder, delta_k);
    }
    else
    {
      form = CompressibleNavierStokesFormulation::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                  meshTopo, polyOrder, delta_k);
      form->setTimeStep(dt);
    }
    
    BFPtr bf = form->bf();
    RHSPtr rhs = form->rhs();
    
    auto soln = form->solution();
    auto solnIncrement = form->solutionIncrement();
    auto solnPrevTime  = form->solutionPreviousTimeStep();
    
    bool includeFluxParity = false; // for fluxes, we will substitute fluxes into the bf object, meaning that we want them to flip sign with the normal.
    auto exactMap = form->exactSolutionMap(u, rho, T, includeFluxParity);
    auto f_c = form->exactSolution_fc(u, rho, T);
    auto f_m = form->exactSolution_fm(u, rho, T);
    auto f_e = form->exactSolution_fe(u, rho, T);
    
    auto vf = bf->varFactory();
    // split the exact solution into traces and fields
    // we want to project the traces onto the increment, and the fields onto the background flow (soln)
    map<int, FunctionPtr> traceMap, fieldMap;
    for (auto entry : exactMap)
    {
      int trialID = entry.first;
      FunctionPtr f = entry.second;
      if (vf->trial(trialID)->isDefinedOnVolume()) // field
      {
        fieldMap[trialID] = f;
      }
      else
      {
        traceMap[trialID] = f;
      }
    }
    int solnOrdinal = 0; // no goal-oriented stuff here...
    soln->projectOntoMesh(fieldMap, solnOrdinal);
    
    // each entry in the following two containers corresponds to a test
    // we have one such test for steady, two for transient
    vector<map<int, FunctionPtr>> previousSolutionFieldMaps;
    vector<array<FunctionPtr,3>> forcingFunctions; // entries are fc, fm, fe
    if (!steady)
    {
      // this is not a full test of time step formulation, but it does check that if we have a steady solution,
      // the time step will not take us away from it.
      //      solnPrevTime->projectOntoMesh(fieldMap, solnOrdinal);
      
      /* New, fuller test:
       Here, actually test for a sort of conservation property - that the time stepper is linear in the conservation
       variables, basically.  This takes advantage of the fact that all conservation variables are linear in rho.
       Later, we might add another round of tests that uses linearity in T.
       
       */
      
      auto prevSolnFieldMapRho = fieldMap;
      auto prevSolnFieldMapT   = fieldMap;
      
      prevSolnFieldMapRho[form->rho()->ID()] = rho - dt;
      prevSolnFieldMapT  [form->T()->ID()]   = T   - dt;
      
      previousSolutionFieldMaps.push_back(prevSolnFieldMapRho);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapT);
      
      array<FunctionPtr,3> rhoForcing = {{f_c + 1, f_m[0] + u, f_e + TEST_CV * T + .5 * u * u}};
      array<FunctionPtr,3> TForcing   = {{f_c,     f_m[0],     f_e + TEST_CV * rho           }};
      forcingFunctions = {{rhoForcing, TForcing}};
    }
    else
    {
      previousSolutionFieldMaps.push_back(fieldMap);
      array<FunctionPtr,3> standardForcingFunctions = {{f_c, f_m[0], f_e}};
      forcingFunctions = {{standardForcingFunctions}};
    }
      
    int numEntries = previousSolutionFieldMaps.size();
    for (int testOrdinal=0; testOrdinal<numEntries; testOrdinal++)
    {
      f_c = forcingFunctions[testOrdinal][0];
      f_m[0] = forcingFunctions[testOrdinal][1];
      f_e = forcingFunctions[testOrdinal][2];
      form->setForcing(f_c, f_m, f_e);
      
      auto prevSolnFieldMap = previousSolutionFieldMaps[testOrdinal];
      
      solnPrevTime->projectOntoMesh(prevSolnFieldMap, solnOrdinal);
      
      auto residual = bf->testFunctional(traceMap) - rhs->linearTerm();
      
      auto testIP = solnIncrement->ip(); // We'll use this inner product to take the norm of the residual components
      
      auto summands = residual->summands();
      auto testVars = vf->testVars();
      for (auto testEntry : testVars)
      {
        int testID = testEntry.first;
        VarPtr testVar = testEntry.second;
        // filter the parts of residual that involve testVar
        LinearTermPtr testResidual = Teuchos::rcp( new LinearTerm );
        for (auto summandEntry : summands)
        {
          FunctionPtr f = summandEntry.first;
          VarPtr v = summandEntry.second;
          if (v->ID() == testID)
          {
            testResidual = testResidual + f * v;
          }
        }
        double residualNorm = testResidual->computeNorm(testIP, soln->mesh(), cubatureEnrichment);
        if (residualNorm > tol)
        {
          success = false;
          out << "FAILURE: residual in " << testVar->name() << " component: " << residualNorm << " exceeds tolerance " << tol << ".\n";
          out << "Residual string: " << testResidual->displayString() << "\n";
          out << "Exact solution under test:\n";
          for (auto traceEntry : traceMap)
          {
            int trialID = traceEntry.first;
            std::string varName = vf->trial(trialID)->name();
            out << "  " << varName << ": " << traceEntry.second->displayString() << std::endl;
          }
          for (auto trialEntry : fieldMap)
          {
            int trialID = trialEntry.first;
            std::string varName = vf->trial(trialID)->name();
            out << "  " << varName << ": " << trialEntry.second->displayString() << std::endl;
          }
          if (!steady)
          {
            out << "Previous solution under test:\n";
            for (auto trialEntry : prevSolnFieldMap)
            {
              int trialID = trialEntry.first;
              std::string varName = vf->trial(trialID)->name();
              out << "  " << varName << ": " << trialEntry.second->displayString() << std::endl;
            }
          }
        }
      }
    }
  }
  
  void testSteadyResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool steady = true;
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, steady, out, success);
  }
  
  void testTransientResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool steady = false;
    testResidual_1D(u, rho, T, cubatureEnrichment, tol, steady, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_AllZero)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero(); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_LinearTemp)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::xn(1);
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero(); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(1.0); // expected forcing for momentum equation
    FunctionPtr f_e = Function::constant(0.5); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_LinearVelocity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    double fe_const = -4./3. * 1. / TEST_RE;
    FunctionPtr f_e = Function::constant(fe_const); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_LinearTempUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    FunctionPtr f_c = Function::zero();            // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(TEST_R);  // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero();            // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);;
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = 2.0 * x;                 // expected forcing for momentum equation
    FunctionPtr f_e = 1.5 * x * x + -4./3. * 1. / TEST_RE; // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Forcing_1D_Steady_QuadraticTemp)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = x * x;
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::constant(-TEST_CP * 2.0 / (TEST_PR * TEST_RE)); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_AllZero)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_AllOne)
  {
    double tol = 1e-13;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_UnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_UnitTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::constant(1.0);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_UnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = 1e-14;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::xn(1);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearTempUnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearVelocityLinearDensityLinearTemp)
  {
    double tol = 1e-13;
    int cubatureEnrichment = 4;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = x;
    FunctionPtr T   = x;
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = 1e-14;
    int cubatureEnrichment = 3;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_QuadraticTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = x * x;
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_AllZero)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_AllOne)
  {
    double tol = 1e-13;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_UnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_UnitTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::constant(1.0);
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_UnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearDensityUnitVelocity)
  {
    double tol = 1e-14;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::xn(1);
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearTempUnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearVelocityLinearDensityLinearTemp)
  {
    double tol = 1e-13;
    int cubatureEnrichment = 4;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = x;
    FunctionPtr T   = x;
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_LinearVelocityUnitDensity)
  {
    double tol = 1e-14;
    int cubatureEnrichment = 3;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Transient_QuadraticTemp)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::zero();
    FunctionPtr T   = x * x;
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  

  // Test below would work for Euler, I think, but compressible NS has viscosity...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesFormulation, Residual_1D_Steady_Shock)
//  {
//    // Impose Rankine-Hugoniot shock conditions (standing shock)
//    // We expect a zero residual
//
//    double tol = 1e-13;
//    int cubatureEnrichment = 2;
//
//    double Ma  = 2.0; // Mach number
//
//    double rho_a = 1.0;     // prescribed density at left
//    double u_a   = Ma;      // Mach number
//    double gamma = TEST_GAMMA;
//
//    double p_a   = rho_a / gamma;
//    double T_a   = p_a / (rho_a * (gamma - 1.) * TEST_CV);
//
//    double p_b   = p_a * (1. + (2. * gamma) / (gamma +  1.) * (Ma * Ma - 1) );
//    double rho_b = rho_a * ( (gamma - 1.) + (gamma + 1.) * p_b / p_a ) / ( (gamma + 1.) + (gamma - 1.) * p_b / p_a );
//    double u_b   = rho_a * u_a / rho_b;
//    double T_b   = p_b / (rho_b * (gamma - 1.) * TEST_CV);
//
//    // use Heaviside functions to define the jumps
//    FunctionPtr H_right = heaviside(0.5); // "traceable" Heaviside is 0 left of center, 1 right of center, and 0.5 at center
//    FunctionPtr H_left  = 1.0 - H_right;
//    FunctionPtr u   = H_left * u_a   + H_right * u_b;
//    FunctionPtr rho = H_left * rho_a + H_right * rho_b;
//    FunctionPtr T   = H_left * T_a   + H_right * T_b;
//
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
} // namespace
