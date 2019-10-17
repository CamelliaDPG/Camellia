//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CompressibleNavierStokesConservationFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/9/18.
//
//

#include "MeshFactory.h"
#include "CompressibleNavierStokesConservationForm.hpp"
#include "HDF5Exporter.h"
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
  
  static const double DEFAULT_RESIDUAL_TOLERANCE = 1e-12; // now that we do a solve, need to be a bit more relaxed
  static const double DEFAULT_NL_SOLVE_TOLERANCE =  1e-6; // for nonlinear stepping
  
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
    auto form = CompressibleNavierStokesConservationForm::steadyFormulation(spaceDim, Re, useConformingTraces,
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
    int polyOrder = 4; // could make some tests cheaper by taking the required poly order as argument...
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.01; // cheat to the right a bit to ensure that we don't get spurious failures due to zero density at LHS of mesh (if using linear density)
    double x_b   = 1.01;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = 1e2; // Reynolds number
    double dt    = 1.0; // only used for unsteady
    
    if (T->isZero())
    {
      // instead of zero, use a small constant...
      T = Function::constant(.01);
    }
    
    bool useConformingTraces = true;
    Teuchos::RCP<CompressibleNavierStokesConservationForm> form;
    if (steady)
    {
      form = CompressibleNavierStokesConservationForm::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                         meshTopo, polyOrder, delta_k);
    }
    else
    {
      form = CompressibleNavierStokesConservationForm::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
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
      /*
       Here, actually test for a sort of conservation property - that the time stepper is linear in the conservation
       variables, basically.
       */
      
      auto prevSolnFieldMapRho = fieldMap;
      auto prevSolnFieldMapm   = fieldMap;
      auto prevSolnFieldMapE   = fieldMap;
      
      FunctionPtr m = fieldMap[form->m(1)->ID()];
      FunctionPtr E = fieldMap[form->E()->ID() ];
      
      prevSolnFieldMapRho[form->rho()->ID()] = rho - dt;
      prevSolnFieldMapm  [form->m(1)->ID() ] = m   - dt;
      prevSolnFieldMapE  [form->E()->ID()  ] = E   - dt;
      
      previousSolutionFieldMaps.push_back(prevSolnFieldMapRho);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapm);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapE);
      
      array<FunctionPtr,3> rhoForcing = {{f_c + 1, f_m[0]    , f_e     }};
      array<FunctionPtr,3> mForcing   = {{f_c,     f_m[0] + 1, f_e     }};
      array<FunctionPtr,3> EForcing   = {{f_c,     f_m[0]    , f_e + 1 }};
      
      forcingFunctions = {{rhoForcing, mForcing, EForcing}};
    }
    else
    {
      previousSolutionFieldMaps.push_back(fieldMap);
      array<FunctionPtr,3> standardForcingFunctions = {{f_c, f_m[0], f_e}};
      forcingFunctions = {{standardForcingFunctions}};
    }
    
    auto printTestInfo = [&] (map<int, FunctionPtr> &prevSolnFieldMap) -> void {
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
    };
    
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
      
      auto testVars = vf->testVars();
      
      for (auto testEntry : testVars)
      {
        VarPtr testVar = testEntry.second;
        // filter the parts of residual that involve testVar
        LinearTermPtr testResidual = residual->getPartMatchingVariable(testVar);
        double residualNorm = testResidual->computeNorm(testIP, soln->mesh(), cubatureEnrichment);
        if (residualNorm > tol)
        {
          success = false;
          out << "FAILURE: residual in " << testVar->name() << " component: " << residualNorm << " exceeds tolerance " << tol << ".\n";
          out << "Residual string: " << testResidual->displayString() << "\n";
          printTestInfo(prevSolnFieldMap);
        }
      }
      
      /*
       Harden this test a bit: add BCs corresponding to the exact solution, and do a solveAndAccumulate().  Since
       the residual is zero before solve, we expect it to remain zero after solve.  This is basically a mild test
       of solveAndAccumulate(), and maybe a bit harder test of the BC imposition methods.
       */
      form->addMassFluxCondition    (SpatialFilter::allSpace(), rho, u, T);
      form->addEnergyFluxCondition  (SpatialFilter::allSpace(), rho, u, T);
      form->addMomentumFluxCondition(SpatialFilter::allSpace(), rho, u, T);
      form->addTemperatureTraceCondition(SpatialFilter::allSpace(), T);
      form->addVelocityTraceCondition(SpatialFilter::allSpace(), u);
      form->solutionIncrement()->setCubatureEnrichmentDegree(cubatureEnrichment);
      double alpha = form->solveAndAccumulate();
      if (alpha < 1.0)
      {
        success = false;
        out << "TEST FAILURE: with exact solution as background flow, line search used a step size of " << alpha << ", below expected 1.0\n";
        printTestInfo(prevSolnFieldMap);
      }
      for (auto testEntry : testVars)
      {
        VarPtr testVar = testEntry.second;
        // filter the parts of residual that involve testVar
        LinearTermPtr testResidual = residual->getPartMatchingVariable(testVar);
        double residualNorm = testResidual->computeNorm(testIP, soln->mesh(), cubatureEnrichment);
        if (residualNorm > tol)
        {
          success = false;
          out << "FAILURE: after solveAndAccumulate(), residual in " << testVar->name() << " component: " << residualNorm << " exceeds tolerance " << tol << ".\n";
          out << "Residual string: " << testResidual->displayString() << "\n";
          printTestInfo(prevSolnFieldMap);
          
          map<int, FunctionPtr> solnIncrementErrorFunctionMap;
          for (auto exactSolnEntry : exactMap)
          {
            int trialID = exactSolnEntry.first;
            FunctionPtr exactSolnFxn = exactSolnEntry.second;
            FunctionPtr expectedSoln;
            VarPtr var = vf->trial(trialID);
            if (var->isDefinedOnVolume()) // field
            {
              expectedSoln = Function::zero();
            }
            else
            {
              // we solve for the traces in each nonlinear step; they stay in solnIncrement
              expectedSoln = exactSolnFxn;
            }
            bool weightFluxesByParity = true; // pretty sure this is the right choice to make uniquely valued...
            FunctionPtr solnIncrementFxn = Function::solution(var, form->solutionIncrement(), weightFluxesByParity);
            FunctionPtr error = solnIncrementFxn - expectedSoln;
            double l2Error = error->l2norm(soln->mesh());
            out << var->name() << " error: " << l2Error << endl;
          }
        }
      }
    }
  }
  
  void testSteadySolve_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr T, FunctionPtr u_guess, FunctionPtr rho_guess, FunctionPtr T_guess, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 4;
    int polyOrder = 3; // need 3rd order to exactly capture E for linear u and linear rho
    
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.01; // cheat to the right a bit to ensure that we don't get spurious failures due to zero density at LHS of mesh (if using linear density)
    double x_b   = 1.01;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double Re    = TEST_RE; // Reynolds number
    
    if (T->isZero())
    {
      // instead of zero, use a small constant...
      T = Function::constant(.01);
    }
    
    bool useConformingTraces = true;
    Teuchos::RCP<CompressibleNavierStokesConservationForm> form;
    form = CompressibleNavierStokesConservationForm::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                       meshTopo, polyOrder, delta_k);
    
    BFPtr bf = form->bf();
    RHSPtr rhs = form->rhs();
    
    auto soln = form->solution();
    auto solnIncrement = form->solutionIncrement();
    
//    HDF5Exporter solutionExporter(soln->mesh(), "testSteadySolve", "/tmp");
//    HDF5Exporter solutionIncrementExporter(solnIncrement->mesh(), "testSteadySolveIncrement", "/tmp");
    
    bool includeFluxParity = false; // for fluxes, we will substitute fluxes into the bf object, meaning that we want them to flip sign with the normal.
    auto exactMap = form->exactSolutionMap(u, rho, T, includeFluxParity);
    auto f_c = form->exactSolution_fc(u, rho, T);
    auto f_m = form->exactSolution_fm(u, rho, T);
    auto f_e = form->exactSolution_fe(u, rho, T);
    
    form->setForcing(f_c, {{f_m}}, f_e);
    
    auto vf = bf->varFactory();
    // split fields out from the initial guess
    // we want to project the traces onto the increment, and the fields onto the background flow (soln)
    auto initialGuessMap = form->exactSolutionMap(u_guess, rho_guess, T_guess, includeFluxParity);
    map<int, FunctionPtr> fieldInitialGuessMap;
    for (auto entry : initialGuessMap)
    {
      int trialID = entry.first;
      FunctionPtr f = entry.second;
      VarPtr trialVar = vf->trial(trialID);
      if (trialVar->isDefinedOnVolume()) // field
      {
        initialGuessMap[trialID] = f;
        out << "set initial guess for " << trialVar->name() << ": " << f->displayString() << endl;
      }
    }
    int solnOrdinal = 0; // no goal-oriented stuff here...
    soln->projectOntoMesh(initialGuessMap, solnOrdinal);
    
    // add basically all possible BCs -- may want to remove some of these at some point...
    form->addMassFluxCondition    (SpatialFilter::allSpace(), rho, u, T);
    form->addEnergyFluxCondition  (SpatialFilter::allSpace(), rho, u, T);
    form->addMomentumFluxCondition(SpatialFilter::allSpace(), rho, u, T);
    form->addTemperatureTraceCondition(SpatialFilter::allSpace(), T);
    form->addVelocityTraceCondition(SpatialFilter::allSpace(), u);
    form->solutionIncrement()->setCubatureEnrichmentDegree(cubatureEnrichment);
    int maxSteps = 30;
//    solutionExporter.exportSolution(soln, double(0));  // output initial guess
    double l2NormTolerance = 1e-10;
    for (int stepNumber = 0; stepNumber<maxSteps; stepNumber++)
    {
      double alpha = form->solveAndAccumulate();
      if (alpha < 1.0)
      {
        out << "in step " << stepNumber << ", alpha = " << alpha << endl;
      }
//      solutionExporter.exportSolution(soln, double(stepNumber+1));  // use stepNumber as the "time" value for export...
//      solutionIncrementExporter.exportSolution(solnIncrement, double(stepNumber));
      double l2Norm = form->L2NormSolutionIncrement();
      out << "Step " << stepNumber << ", L^2 norm of soln increment: " << l2Norm << endl;
      if (l2Norm < l2NormTolerance)
      {
        break;
      }
    }
    
    map<int, FunctionPtr> solnErrorFunctionMap;
    for (auto exactSolnEntry : exactMap)
    {
      int trialID = exactSolnEntry.first;
      FunctionPtr exactSolnFxn = exactSolnEntry.second;
      FunctionPtr expectedSoln;
      VarPtr var = vf->trial(trialID);
      FunctionPtr solnFxn;
      bool weightFluxesByParity = true; // pretty sure this is the right choice to make uniquely valued...
      if (var->isDefinedOnVolume()) // field
      {
        solnFxn = Function::solution(var, form->solution(), weightFluxesByParity);
      }
      else
      {
        // we solve for the traces in each nonlinear step; they stay in solnIncrement
        solnFxn = Function::solution(var, form->solutionIncrement(), weightFluxesByParity);
      }
      FunctionPtr error = solnFxn - exactSolnFxn;
      double l2Error = error->l2norm(soln->mesh());
      if (l2Error > tol)
      {
        success = false;
        out << "Failure in ";
        out << var->name() << "; error: " << l2Error << endl;
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
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_UnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero(); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(1.0); // expected forcing for momentum equation
    FunctionPtr f_e = Function::constant(0.5); // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_LinearVelocity)
//  {
//    double tol = 1e-16;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::xn(1);
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
//    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
//    double fe_const = -4./3. * 1. / TEST_RE;
//    FunctionPtr f_e = Function::constant(fe_const); // expected forcing for energy equation
//    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_LinearTempUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    FunctionPtr f_c = Function::zero();            // expected forcing for continuity equation
    FunctionPtr f_m = Function::constant(TEST_R);  // expected forcing for momentum equation
    FunctionPtr f_e = Function::zero();            // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    FunctionPtr f_c = Function::constant(1.0); // expected forcing for continuity equation
    FunctionPtr f_m = 2.0 * x;                 // expected forcing for momentum equation
    FunctionPtr f_e = 1.5 * x * x + -4./3. * 1. / TEST_RE; // expected forcing for energy equation
    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Solve_1D_Steady_AllUnit)
  {
    double tol = DEFAULT_NL_SOLVE_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);

    FunctionPtr u_guess   = Function::constant(0.50);
    FunctionPtr rho_guess = Function::constant(0.50);
    FunctionPtr T_guess   = Function::constant(0.50);
    
    testSteadySolve_1D(u, rho, T, u_guess, rho_guess, T_guess, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Solve_1D_Steady_LinearVelocityUnitDensityUnitTemp)
  {
    double tol = DEFAULT_NL_SOLVE_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    
    FunctionPtr u_guess   = 0.5 * u;
    FunctionPtr rho_guess = 0.5 * rho;
    FunctionPtr T_guess   = 0.5 * T;
    
    testSteadySolve_1D(u, rho, T, u_guess, rho_guess, T_guess, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Solve_1D_Steady_LinearVelocityLinearDensityUnitTemp)
  {
    double tol = DEFAULT_NL_SOLVE_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::xn(1) + 1.0;
    FunctionPtr T   = Function::constant(1.0);
    
    FunctionPtr u_guess   = 0.5 * u;
    FunctionPtr rho_guess = 0.5 * rho;
    FunctionPtr T_guess   = 0.5 * T;
    
    testSteadySolve_1D(u, rho, T, u_guess, rho_guess, T_guess, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Solve_1D_Steady_UnitVelocityLinearDensityUnitTemp)
  {
    double tol = DEFAULT_NL_SOLVE_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::constant(1.0);
    
    FunctionPtr u_guess   = 0.5 * u;
    FunctionPtr rho_guess = 0.5 * rho;
    FunctionPtr T_guess   = 0.5 * T;
    
    testSteadySolve_1D(u, rho, T, u_guess, rho_guess, T_guess, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Solve_1D_Steady_UnitVelocityUnitDensityLinearTemp)
  {
    double tol = DEFAULT_NL_SOLVE_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::xn(1) + 20;
    
    FunctionPtr u_guess   = 0.5 * u;
    FunctionPtr rho_guess = 0.5 * rho;
    FunctionPtr T_guess   = 0.5 * T;
    
    testSteadySolve_1D(u, rho, T, u_guess, rho_guess, T_guess, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Forcing_1D_Steady_QuadraticTemp)
//  {
//    double tol = 1e-16;
//    int cubatureEnrichment = 0;
//    FunctionPtr x   = Function::xn(1);
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = x * x;
//    FunctionPtr f_c = Function::zero(); // expected forcing for continuity equation
//    FunctionPtr f_m = Function::zero(); // expected forcing for momentum equation
//    FunctionPtr f_e = Function::constant(-TEST_CP * 2.0 / (TEST_PR * TEST_RE)); // expected forcing for energy equation
//    testForcing_1D(u, rho, T, f_c, f_m, f_e, cubatureEnrichment, tol, out, success);
//  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_AllZero)
//  {
//    double tol = 1e-16;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_AllOne)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_UnitDensityUnitTemp)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_UnitTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::constant(1.0);
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_UnitVelocity)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::constant(1.0);
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearDensityUnitTemp)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::constant(1.0);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearDensityLinearVelocityQuadraticEnergy)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr E   = Function::xn(2);
    FunctionPtr T   = (1.0 / TEST_CV) * (1.0 / rho) * E - (1.0 / TEST_CV) * (0.5 * u * u); // comes out to (1/Cv) * x - (1/Cv) * 0.5 * x^2 = (1/Cv) * x * (1 - 0.5 x)
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_UnitDensityUnitVelocityLinearEnergy)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E   = Function::xn(1) + 1.0; // add 1 to bound temp away from zero
    FunctionPtr T   = (1.0 / TEST_CV) * (1.0 / rho) * E - (1.0 / TEST_CV) * (0.5 * u * u);
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 3;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::xn(1);
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearTempUnitDensity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearVelocity)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 3;
//    FunctionPtr u   = Function::xn(1);
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearVelocityLinearDensityLinearTemp)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 4;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = x;
    FunctionPtr T   = x;
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 3;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_LinearVelocityUnitDensityUnitTemp)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 2;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    
    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_QuadraticTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 2;
//    FunctionPtr x   = Function::xn(1);
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = x * x;
//    testSteadyResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_AllZero)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_AllOne)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::constant(1.0);
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_UnitDensity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_UnitTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::constant(1.0);
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_UnitVelocity)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 0;
//    FunctionPtr u   = Function::constant(1.0);
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearDensity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearDensityUnitVelocity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearDensityLinearVelocityQuadraticEnergy)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE * 10;
    int cubatureEnrichment = 5;
    FunctionPtr u   = Function::xn(1);
    FunctionPtr rho = Function::xn(1);
    FunctionPtr E   = Function::xn(2);
    FunctionPtr T   = (1.0 / TEST_CV) * (1.0 / rho) * E - (1.0 / TEST_CV) * (0.5 * u * u); // comes out to (1/Cv) [x - 0.5 * x * x]
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_UnitDensityUnitVelocityLinearEnergy)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 3;
    FunctionPtr u   = Function::constant(1.0);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E   = Function::xn(1) + 1.0; // add 1 to bound temp away from zero
    FunctionPtr T   = (1.0 / TEST_CV) * (1.0 / rho) * E - (1.0 / TEST_CV) * (0.5 * u * u);
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 3;
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::xn(1);
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearTempUnitDensity)
  {
    double tol = 1e-14;
    int cubatureEnrichment = 2;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::zero();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearVelocity)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 3;
//    FunctionPtr u   = Function::xn(1);
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = Function::zero();
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearVelocityLinearDensityLinearTemp)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 4;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = x;
    FunctionPtr T   = x;
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_LinearVelocityUnitDensity)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 3;
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = x;
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = Function::zero();
    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
  }

  TEUCHOS_UNIT_TEST(Euler, TestSpaceSize)
  {
    int spaceDim = 1;
    double x_a = 0.0, x_b = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    bool useConformingTraces = true;
    auto form = CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                       meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numVars = vf->testVars().size();
    TEST_EQUALITY(numVars, 3); // 1D: vc, vm, ve
  }
  
  // Commenting out for now: zero density not allowed...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Transient_QuadraticTemp)
//  {
//    double tol = 1e-15;
//    int cubatureEnrichment = 2;
//    FunctionPtr x   = Function::xn(1);
//    FunctionPtr u   = Function::zero();
//    FunctionPtr rho = Function::zero();
//    FunctionPtr T   = x * x;
//    testTransientResidual_1D(u, rho, T, cubatureEnrichment, tol, out, success);
//  }
  

  // Test below would work for Euler, I think, but compressible NS has viscosity...
//  TEUCHOS_UNIT_TEST(CompressibleNavierStokesConservationForm, Residual_1D_Steady_Shock)
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
