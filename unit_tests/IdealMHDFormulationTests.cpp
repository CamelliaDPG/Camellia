//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  IdealMHDFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/9/18.
//
//

#include "Functions.hpp"
#include "MeshFactory.h"
#include "IdealMHDFormulation.hpp"
#include "HDF5Exporter.h"
#include "RHS.h"
#include "RieszRep.h"
#include "SimpleFunction.h"
#include "Solution.h"

using namespace Camellia;

#include <array>

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  static const double TEST_CV = 1.000;
  static const double TEST_GAMMA = 2.0;
  static const double TEST_CP = TEST_GAMMA * TEST_CV;
  static const double TEST_R = TEST_CP - TEST_CV;
  
  static const double DEFAULT_RESIDUAL_TOLERANCE = 1e-12; // now that we do a solve, need to be a bit more relaxed
//  static const double DEFAULT_NL_SOLVE_TOLERANCE =  1e-6; // for nonlinear stepping
  
  
  // since we have defined things in other tests in terms of temperature: conversion here
  FunctionPtr energy(FunctionPtr rho, FunctionPtr u, FunctionPtr T)
  {
    FunctionPtr u_dot_u = dot(3, u, u);
    FunctionPtr E = TEST_CV * rho * T + 0.5 * rho * u_dot_u;
    return E;
  }
  
  FunctionPtr pressure(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
  {
    FunctionPtr u_dot_u = dot(3, u, u);
    FunctionPtr p = (TEST_R / TEST_CV) * (E - 0.5 * rho * u_dot_u - 0.5 * dot(3,B,B));
    return p;
  }
  
  FunctionPtr pressure_star(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
  {
    FunctionPtr p = pressure(rho, u, E, B);
    FunctionPtr p_star = p + 0.5 * dot(3,B,B);
    return p_star;
  }
  
  FunctionPtr temperature(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
  {
    return 1./TEST_CV * (E / rho - 0.5 * dot(3,u,u) - 0.5 * dot(3,B,B));
  }
  
  void testForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B,
                      FunctionPtr fc, FunctionPtr fm, FunctionPtr fe, FunctionPtr fB,
                      int cubatureEnrichment, double tol, bool spaceTime, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 2;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    Teuchos::RCP<IdealMHDFormulation> form;
    if (spaceTime)
    {
      int temporalDivisions = 2;
      int temporalPolyOrder = 2;
      double t0 = 0.0;
      double t1 = 1.0;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
      form = IdealMHDFormulation::spaceTimeFormulation(spaceDim, meshTopo, polyOrder, temporalPolyOrder, delta_k);
    }
    else
    {
      form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    }
    form->setBx(B->spatialComponent(1));
    
    // sanity checks that the constructor has set things up the way we assume:
    TEST_FLOATING_EQUALITY(form->Cv(),    TEST_CV,     tol);
    TEST_FLOATING_EQUALITY(form->gamma(), TEST_GAMMA,  tol);
    TEST_FLOATING_EQUALITY(form->Cp(),    TEST_CP,     tol);
    TEST_FLOATING_EQUALITY(form->R(),     TEST_R,      tol);
    
    BFPtr bf = form->bf();
    RHSPtr rhs = form->rhs();
    
    auto soln = form->solution();
    auto solnIncrement = form->solutionIncrement();
    auto mesh = soln->mesh();
    
    auto exact_fc = form->exactSolution_fc(rho, u, E, B);
    auto exact_fe = form->exactSolution_fe(rho, u, E, B);
    auto exact_fm = form->exactSolution_fm(rho, u, E, B);
    auto exact_fB = form->exactSolution_fB(rho, u, E, B);
    
    double fc_err = (fc - exact_fc)->l2norm(mesh, cubatureEnrichment);
    if (fc_err > tol)
    {
      success = false;
      out << "FAILURE: fc_err " << fc_err << " > tol " << tol << endl;
      out << "fc expected: " << fc->displayString() << endl;
      out << "fc actual:   " << exact_fc->displayString() << endl;
    }
    
    int trueSpaceDim = 3;
    for (int d=0; d<trueSpaceDim; d++)
    {
      double fm_err = (fm->spatialComponent(d+1) - exact_fm[d])->l2norm(mesh);
      if (fm_err > tol)
      {
        success = false;
        out << "FAILURE: fm" << d+1 << "_err " << fm_err << " > tol " << tol << endl;
        out << "fm" << d+1 << " expected: " << fm->spatialComponent(d+1)->displayString() << endl;
        out << "fm" << d+1 << " actual:   " << exact_fm[d]->displayString() << endl;
      }
    }
    
    double fe_err = (fe - exact_fe)->l2norm(mesh, cubatureEnrichment);
    if (fe_err > tol)
    {
      success = false;
      out << "FAILURE: fe_err " << fe_err << " > tol " << tol << endl;
      out << "fe expected: " << fe->displayString() << endl;
      out << "fe actual:   " << exact_fe->displayString() << endl;
    }
    
    int dStart = 1; // 1D
    for (int d=dStart; d<trueSpaceDim; d++)
    {
      double fB_err = (fB->spatialComponent(d+1) - exact_fB[d])->l2norm(mesh);
      if (fB_err > tol)
      {
        success = false;
        out << "FAILURE: fB" << d+1 << "_err " << fB_err << " > tol " << tol << endl;
        out << "fB" << d+1 << " expected: " << fB->spatialComponent(d+1)->displayString() << endl;
        out << "fB" << d+1 << " actual:   " << exact_fB[d]->displayString() << endl;
      }
    }
  }
  
  void testForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B,
                      int cubatureEnrichment, double tol, bool spaceTime, Teuchos::FancyOStream &out, bool &success)
  {
    FunctionPtr Bx  = B->spatialComponent(1);
    FunctionPtr By  = B->spatialComponent(2);
    FunctionPtr Bz  = B->spatialComponent(3);
    FunctionPtr ux  = u->spatialComponent(1);
    FunctionPtr uy  = u->spatialComponent(2);
    FunctionPtr uz  = u->spatialComponent(3);
    FunctionPtr u_dot_u = dot(3,u,u);
    FunctionPtr B_dot_B = dot(3,B,B);
    FunctionPtr p_star = pressure_star(rho, u, E, B);
    
    FunctionPtr f_c = rho->dt() + (rho * ux)->dx(); // expected forcing for continuity equation: d/dt(rho) div (rho * ux, 0, 0)
    
    FunctionPtr f_m1 = (rho * ux)->dt() + (rho * ux * ux + p_star)->dx();  // d/dx [ rho * u^2 + P* ]
    FunctionPtr f_m2 = (rho * uy)->dt() + (rho * ux * uy - Bx * By)->dx(); // d/dx [ rho * u * v - Bx * By ]
    FunctionPtr f_m3 = (rho * uz)->dt() + (rho * ux * uz - Bx * Bz)->dx(); // d/dx [ rho * u * w - Bx * Bz ]
    FunctionPtr f_m  = Function::vectorize(f_m1, f_m2, f_m3);
    
    FunctionPtr f_e = E->dt() + ((E+p_star)*ux - Bx * dot(3,B,u))->dx();
    FunctionPtr f_B1 = Function::zero(); // 1D: no B1 equation
    FunctionPtr f_B2 = By->dt() + (By * ux - Bx * uy)->dx(); // d/dx [ By * u - Bx * v ] = d/dx [0] = 0
    FunctionPtr f_B3 = Bz->dt() + (Bz * ux - Bx * uz)->dx(); // d/dx [ Bz * u - Bx * w ] = d/dx [0] = 0
    FunctionPtr f_B = Function::vectorize(f_B1, f_B2, f_B3); // expected forcing for magnetism equation
    testForcing_1D(u, rho, E, B, f_c, f_m, f_e, f_B, cubatureEnrichment, tol, spaceTime, out, success);
  }
  
  void testSpaceTimeForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B,
                               int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool spaceTime = true;
    testForcing_1D(u, rho, E, B, cubatureEnrichment, tol, spaceTime, out, success);
  }
  
  void testSteadyForcing_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B,
                            int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool spaceTime = false;
    testForcing_1D(u, rho, E, B, cubatureEnrichment, tol, spaceTime, out, success);
  }
  
  void testResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, int cubatureEnrichment,
                       double tol, bool steady, bool spaceTime, Teuchos::FancyOStream &out, bool &success)
  {
    int meshWidth = 2;
    int polyOrder = 4; // could make some tests cheaper by taking the required poly order as argument...
    int delta_k   = 2;
    int spaceDim = 1;
    
    double x_a   = 0.01; // cheat to the right a bit to ensure that we don't get spurious failures due to zero density at LHS of mesh (if using linear density)
    double x_b   = 1.01;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    double dt    = 0.49; // for reasons having to do with positivity of temperature and density in the transient solves below, we need this to be strictly bounded above by 0.50
    
    Teuchos::RCP<IdealMHDFormulation> form;
    if (spaceTime)
    {
      int temporalDivisions = 2;
      int temporalPolyOrder = 2;
      double t0 = 0.0;
      double t1 = 1.0;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalDivisions);
      form = IdealMHDFormulation::spaceTimeFormulation(spaceDim, meshTopo, polyOrder, temporalPolyOrder, delta_k);
    }
    else
    {
      form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
      form->setTimeStep(dt);
    }
    form->setBx(B->spatialComponent(1));
    
    BFPtr bf = form->bf();
    RHSPtr rhs = form->rhs();
    
    auto soln = form->solution();
    auto solnIncrement = form->solutionIncrement();
    auto solnPrevTime  = form->solutionPreviousTimeStep();
    
    bool includeFluxParity = false; // for fluxes, we will substitute fluxes into the bf object, meaning that we want them to flip sign with the normal.
    auto exactMap = form->exactSolutionMap(rho, u, E, B, includeFluxParity);
    auto f_c = form->exactSolution_fc(rho, u, E, B);
    auto f_m = form->exactSolution_fm(rho, u, E, B);
    auto f_e = form->exactSolution_fe(rho, u, E, B);
    auto f_B = form->exactSolution_fB(rho, u, E, B);
    
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
    vector<array<FunctionPtr,7>> forcingFunctions; // entries are fc, fm1, fm2, fm3, fe, fB2, fB3
    if (!steady)
    {
      /*
       Here, actually test for a sort of conservation property - that the time stepper is linear in the conservation
       variables, basically.
       */
      
      auto prevSolnFieldMapRho = fieldMap;
      auto prevSolnFieldMapm1  = fieldMap;
      auto prevSolnFieldMapm2  = fieldMap;
      auto prevSolnFieldMapm3  = fieldMap;
      auto prevSolnFieldMapE   = fieldMap;
      auto prevSolnFieldMapB2  = fieldMap;
      auto prevSolnFieldMapB3  = fieldMap;
      
      FunctionPtr m = rho * u;
      FunctionPtr E = fieldMap[form->E()->ID() ];
      
      auto m1 = m->spatialComponent(1);
      auto m2 = m->spatialComponent(2);
      auto m3 = m->spatialComponent(3);
      auto B2 = B->spatialComponent(2);
      auto B3 = B->spatialComponent(3);
      
      double EweightDt = 0.49; // in steady solutions, we allow temperatures to go as small as 0.5.  To avoid negative temperatures in the E - dt test, we need to weight dt so that it is at most 0.50
      double rhoWeightDt = 0.49; // for similar reasons, in the rho - dt test, we weight dt so that it is at most 0.50.  (If rho gets small, then in the previous solution, m dot m / rho will be large, and this is subtracted from energy.)
      
      prevSolnFieldMapRho[form->rho()->ID()] = rho - rhoWeightDt * dt;
      prevSolnFieldMapm1 [form->m(1)->ID() ] = m1  - dt;
      prevSolnFieldMapm2 [form->m(2)->ID() ] = m2  - dt;
      prevSolnFieldMapm3 [form->m(3)->ID() ] = m3  - dt;
      prevSolnFieldMapE  [form->E()->ID()  ] = E   - EweightDt * dt;
      prevSolnFieldMapB2 [form->B(2)->ID() ] = B2  - dt;
      prevSolnFieldMapB3 [form->B(3)->ID() ] = B3  - dt;
      
      previousSolutionFieldMaps.push_back(prevSolnFieldMapRho);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapm1);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapm2);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapm3);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapE);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapB2);
      previousSolutionFieldMaps.push_back(prevSolnFieldMapB3);
      
      array<FunctionPtr,7> rhoForcing = {{f_c + rhoWeightDt, f_m[0]    , f_m[1]    , f_m[2]    , f_e            , f_B[1]    , f_B[2]    }};
      array<FunctionPtr,7> m1Forcing  = {{f_c              , f_m[0] + 1, f_m[1]    , f_m[2]    , f_e            , f_B[1]    , f_B[2]    }};
      array<FunctionPtr,7> m2Forcing  = {{f_c              , f_m[0]    , f_m[1] + 1, f_m[2]    , f_e            , f_B[1]    , f_B[2]    }};
      array<FunctionPtr,7> m3Forcing  = {{f_c              , f_m[0]    , f_m[1]    , f_m[2] + 1, f_e            , f_B[1]    , f_B[2]    }};
      array<FunctionPtr,7> EForcing   = {{f_c              , f_m[0]    , f_m[1]    , f_m[2]    , f_e + EweightDt, f_B[1]    , f_B[2]    }};
      array<FunctionPtr,7> B2Forcing  = {{f_c              , f_m[0]    , f_m[1]    , f_m[2]    , f_e            , f_B[1] + 1, f_B[2]    }};
      array<FunctionPtr,7> B3Forcing  = {{f_c              , f_m[0]    , f_m[1]    , f_m[2]    , f_e            , f_B[1]    , f_B[2] + 1}};
      
      forcingFunctions = {{rhoForcing, m1Forcing, m2Forcing, m3Forcing, EForcing, B2Forcing, B3Forcing}};
    }
    else
    {
      previousSolutionFieldMaps.push_back(fieldMap);
      array<FunctionPtr,7> standardForcingFunctions = {{f_c, f_m[0], f_m[1], f_m[2], f_e, f_B[1], f_B[2]}};
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
      f_c    = forcingFunctions[testOrdinal][0];
      f_m[0] = forcingFunctions[testOrdinal][1];
      f_m[1] = forcingFunctions[testOrdinal][2];
      f_m[2] = forcingFunctions[testOrdinal][3];
      f_e    = forcingFunctions[testOrdinal][4];
      f_B[1] = forcingFunctions[testOrdinal][5];
      f_B[2] = forcingFunctions[testOrdinal][6];
      f_B[0] = Function::zero();
      form->setForcing(f_c, f_m, f_e, f_B);
      
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
      
      // sanity check: make sure that the manufactured solution's temperature is positive (otherwise the solve below won't work)
      auto T = temperature(rho, u, E, B);
      auto mesh = form->solutionIncrement()->mesh();
      bool positiveT = T->isPositive(mesh);
      bool positiveRho = rho->isPositive(mesh);
      
      if (!positiveT)
      {
        out << "ERROR in test setup: prescribed solution's temperature is not positive.\n";
        success = false;
      }
      if (!positiveRho)
      {
        out << "ERROR in test setup: prescribed solution's density is not positive.\n";
        success = false;
      }
      bool trySolving = true;
      trySolving = !spaceTime; // it's unclear to me whether the space-time failure to solve indicates a real issue or not.  Turning this off for now.  TODO: work through this analytically (fails for the "AllOne" and "LinearInTime" space-time residual tests, if trySolving is set to true.)
      if (trySolving)
        {
        /*
         Harden this test a bit: add BCs corresponding to the exact solution, and do a solveAndAccumulate().  Since
         the residual is zero before solve, we expect it to remain zero after solve.  This is basically a mild test
         of solveAndAccumulate(), and maybe a bit harder test of the BC imposition methods.
         */
        form->addMassFluxCondition    (SpatialFilter::allSpace(), rho, u, E, B);
        form->addMomentumFluxCondition(SpatialFilter::allSpace(), rho, u, E, B);
        form->addEnergyFluxCondition  (SpatialFilter::allSpace(), rho, u, E, B);
        form->addMagneticFluxCondition(SpatialFilter::allSpace(), rho, u, E, B);
        form->solutionIncrement()->setCubatureEnrichmentDegree(cubatureEnrichment);
        double alpha = form->solveAndAccumulate();
        bool thisSuccess = true;
        if (alpha < 1.0)
        {
          success = false;
          thisSuccess = false;
          out << "TEST FAILURE: with exact solution as background flow (testOrdinal " << testOrdinal << "), line search used a step size of " << alpha << ", below expected 1.0\n";
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
            thisSuccess = false;
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
        if (thisSuccess)
        {
          out << "Solve succeeded for testOrdinal " << testOrdinal << endl;
        }
        else
        {
          out << "Solve failed for testOrdinal " << testOrdinal << endl;
        }
      }
    }
  }
  
  void testSpaceTimeResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool steady = true; // ignored for space-time, but basically what the false case means is perturb the time terms, which won't work out the same way for space-time
    bool spaceTime = true;
    testResidual_1D(u, rho, E, B, cubatureEnrichment, tol, steady, spaceTime, out, success);
  }
  
  void testSteadyResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool steady = true;
    bool spaceTime = false;
    testResidual_1D(u, rho, E, B, cubatureEnrichment, tol, steady, spaceTime, out, success);
  }
  
  void testTransientResidual_1D(FunctionPtr u, FunctionPtr rho, FunctionPtr E, FunctionPtr B, int cubatureEnrichment, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    bool steady = false;
    bool spaceTime = false;
    testResidual_1D(u, rho, E, B, cubatureEnrichment, tol, steady, spaceTime, out, success);
  }
  
  void testTermsMatch(LinearTermPtr ltExpected, LinearTermPtr ltActual, MeshPtr mesh, IPPtr ip, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    LinearTermPtr diff = ltExpected - ltActual;
    
    BFPtr bf = mesh->bilinearForm();
    
    RieszRepPtr rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, diff) );
    
    rieszRep->computeRieszRep();
    
    double err = rieszRep->getNorm();
    
    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, AbstractTemperature)
  {
    double tol = 1e-16;
    
    int meshWidth = 2;
    int polyOrder = 2;
    int delta_k   = 2; // 1 is likely sufficient in 1D
    int spaceDim = 1;
    
    double x_a   = 0.0;
    double x_b   = 1.0;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    // Temperature should be given by (R/Cv) * [E - 0.5 * m * m / rho - 0.5 * B * B]
    FunctionPtr zero = Function::zero();
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr x   = Function::xn(1);
    FunctionPtr rho = 2. * one;
    FunctionPtr u   = Function::vectorize(one, one, one);
    FunctionPtr E   = x;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    FunctionPtr T_expected = temperature(rho, u, E, B); // (1.0 / TEST_CV) / rho * (E - 0.5 * m_dot_m / rho - 0.5 * dot(3,B,B));
    bool includeFluxParity = false; // inconsequential here
    auto exactMap =  form->exactSolutionMap(rho, u, E, B, includeFluxParity);
    
    // in 1D, Bx is not a solution variable, but does enter the abstract temperature; Bx is kept track of
    // as a member variable (a FunctionPtr) in IdealMHDFormulation.  Therefore, we need to specify this to the form:
    form->setBx(B->spatialComponent(1));
    auto abstractT = form->abstractTemperature();
    auto concreteT = abstractT->evaluateAt(exactMap);
    
    out << "abstractT: " << abstractT->displayString() << endl;
    out << "concreteT: " << concreteT->displayString() << endl;
    out << "T_expected: " << T_expected->displayString() << endl;
    
    auto mesh = form->solution()->mesh();
    double err = (concreteT - T_expected)->l2norm(mesh);
    
//    cout << "rho = " << rho->displayString() << endl;
//    cout << "L^2 norm of T_expected: " << T_expected->l2norm(mesh) << endl;
//    cout << "L^2 norm of concreteT:  " << concreteT->l2norm(mesh) << endl;
    
    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_UnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    auto zero = Function::zero();
    FunctionPtr t   = Function::tn(1);
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = Function::constant(1.0) * t;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearDensityUnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr t   = Function::tn(1);
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(t, t, t);
    FunctionPtr rho = x * t;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearTempUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr t   = Function::tn(1);
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = t;
    FunctionPtr T   = x * t;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearVelocityUnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    auto x    = Function::xn(1);
    auto t   = Function::tn(1);
    auto zero = Function::zero();
    FunctionPtr u   = Function::vectorize(x * t, zero, zero);
    FunctionPtr rho = t;
    FunctionPtr T   = zero;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_UnitDensityLinearMagnetism)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto zero = Function::zero();
    auto x    = Function::xn(1);
    auto t   = Function::tn(1);
    FunctionPtr Bx  = Function::constant(0.75);
    
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = t;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(Bx, x*t, x*t);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearDensityUnitVelocityLinearMagnetism)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    auto one  = Function::constant(1.0);
    auto zero = Function::zero();
    auto t    = Function::tn(1);
    FunctionPtr Bx  = Function::constant(0.75);
    
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(t, t, t);
    FunctionPtr rho = x * t;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(Bx, x * t, x * t);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearTempUnitDensityLinearMagnetism)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    auto t    = Function::tn(1);
    FunctionPtr Bx  = Function::constant(0.75);
    
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = t;
    FunctionPtr T   = x * t;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(Bx, x * t, x * t);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_SpaceTime_LinearVelocityUnitDensityLinearMagnetism)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    auto x    = Function::xn(1);
    auto zero = Function::zero();
    auto t    = Function::tn(1);
    FunctionPtr Bx  = Function::constant(0.75);
    FunctionPtr u   = Function::vectorize(x * t, zero, zero);
    FunctionPtr rho = t;
    FunctionPtr T   = zero;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(Bx, x * t, x * t);
    
    testSpaceTimeForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_UnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 0;
    auto zero = Function::zero();
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearDensityUnitVelocity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(one, one, one);
    FunctionPtr rho = x;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }

  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearTempUnitDensity)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(zero, zero, zero);

    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }

  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearVelocityUnitDensity)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    auto x    = Function::xn(1);
    auto zero = Function::zero();
    FunctionPtr u   = Function::vectorize(x, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = zero;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_UnitDensityLinearMagnetism)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto zero = Function::zero();
    auto x    = Function::xn(1);
    FunctionPtr Bx  = Function::constant(0.75);

    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(Bx, x, x);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearDensityUnitVelocityLinearMagnetism)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 0;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr Bx  = Function::constant(0.75);

    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(one, one, one);
    FunctionPtr rho = x;
    FunctionPtr E   = zero;
    FunctionPtr B   = Function::vectorize(Bx, x, x);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearTempUnitDensityLinearMagnetism)
  {
    double tol = 1e-16;
    int cubatureEnrichment = 2;
    auto one = Function::constant(1.0);
    auto zero = Function::zero();
    FunctionPtr Bx  = Function::constant(0.75);

    FunctionPtr x   = Function::xn(1);
    FunctionPtr u   = Function::vectorize(zero, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = x;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(Bx, x, x);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Forcing_1D_Steady_LinearVelocityUnitDensityLinearMagnetism)
  {
    double tol = 1e-15;
    int cubatureEnrichment = 2;
    auto x    = Function::xn(1);
    auto zero = Function::zero();
    FunctionPtr Bx  = Function::constant(0.75);
    FunctionPtr u   = Function::vectorize(x, zero, zero);
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr T   = zero;
    FunctionPtr E   = energy(rho, u, T);
    FunctionPtr B   = Function::vectorize(Bx, x, x);
    
    testSteadyForcing_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_1D)
  {
    double tol = 1e-14;
    int spaceDim = 1;
    double x_a = 0.0, x_b = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 7); // 1D: vc, ve, plus 3 vm's, 2 vBs
    
    int numTrialVars = vf->trialVars().size(); // 1D: rho, E, 3 m's, 2 B's, plus 1 flux per equation: 14 variables
    TEST_EQUALITY(numTrialVars, 14);
    
    auto bf = form->solutionIncrement()->bf();
    // try momentum = 0, E = 1, rho = 1, B = 0 -- project this onto previous solution, and current guess
    // evaluate bf at this solution
    // we expect to have just three test terms survive:
    // one corresponding to the pressure in the x-momentum equation
    // two corresponding to the time derivatives on the rho, E solution increments
    VarPtr vm1 = form->vm(1);
    VarPtr vc  = form->vc();
    VarPtr ve  = form->ve();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E = Function::constant(1.0);
    FunctionPtr m = Function::vectorize(Function::constant(0.0), Function::constant(0.0), Function::constant(0.0));
    FunctionPtr p = (TEST_GAMMA - 1.0) * (E - 0.5 * dot(3,m,m) / rho);
    auto dt = form->getTimeStep();
    auto expectedLT = -p * vm1->dx() + rho / dt * vc + E / dt * ve;
    
    map<int, FunctionPtr> valueMap;
    valueMap[form->rho()->ID()] = rho;
    valueMap[form->E()->ID()] = E;
    valueMap[form->m(1)->ID()] = m->spatialComponent(1);
    valueMap[form->m(2)->ID()] = m->spatialComponent(2);
    valueMap[form->m(3)->ID()] = m->spatialComponent(3);
    
    form->solutionPreviousTimeStep()->projectOntoMesh(valueMap);
    form->solution()->projectOntoMesh(valueMap);
    form->setBx(Function::zero());
    auto lt = bf->testFunctional(valueMap);
    out << "lt: " << lt->displayString() << endl;
    out << "expected lt: " << expectedLT->displayString() << endl;
    
    auto mesh = form->solution()->mesh();
    auto ip = form->solutionIncrement()->ip();
    
//    ip->printInteractions();
//    cout << "IP: " << ip->displayString() << endl;
    testTermsMatch(expectedLT, lt, mesh, ip, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, BFIsWellPosed_1D)
  {
    // a test that we have all non-zero rows in the rectangular stiffness for a 1D
    // time-stepping formulation.  This test motivated by an issue that we have had
    // with the Brio-Wu example.
    
    double tol = 1e-14;
    int spaceDim = 1;
    double x_a = 0.0, x_b = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 7); // 1D: vc, ve, plus 3 vm's, 2 vBs
    
    int numTrialVars = vf->trialVars().size(); // 1D: rho, E, 3 m's, 2 B's, plus 1 flux per equation: 14 variables
    TEST_EQUALITY(numTrialVars, 14);
    
    auto bf = form->solutionIncrement()->bf();
    // try momentum = 0, E = 1, rho = 1, B = 0 -- project this onto previous solution, and current guess
    // evaluate bf at this solution
    // we expect to have just five test terms survive:
    // three corresponding to the pressure in the momentum equation
    // two corresponding to the time derivatives on the rho, E solution increments
    VarPtr vm1 = form->vm(1);
    VarPtr vm2 = form->vm(2);
    VarPtr vm3 = form->vm(3);
    VarPtr vc  = form->vc();
    VarPtr ve  = form->ve();
    FunctionPtr rho = Function::constant(1.0);
    FunctionPtr E = Function::constant(1.0);
    FunctionPtr m = Function::vectorize(Function::constant(0.0), Function::constant(0.0), Function::constant(0.0));
    FunctionPtr p = (TEST_GAMMA - 1.0) * (E - 0.5 * dot(3,m,m) / rho);
    auto dt = form->getTimeStep();
    auto expectedLT = p * vm1 + p * vm2 + p * vm3 + rho / dt * vc + E / dt * ve;
    
    map<int, FunctionPtr> valueMap;
    valueMap[form->rho()->ID()] = rho;
    valueMap[form->E()->ID()] = E;
    valueMap[form->m(1)->ID()] = m->spatialComponent(1);
    valueMap[form->m(2)->ID()] = m->spatialComponent(2);
    valueMap[form->m(3)->ID()] = m->spatialComponent(3);
    
    form->solutionPreviousTimeStep()->projectOntoMesh(valueMap);
    form->solution()->projectOntoMesh(valueMap);
    
    auto mesh = form->solution()->mesh();
    auto ip = form->solutionIncrement()->ip();
    
    auto & myCellIDs = mesh->cellIDsInPartition();
    
    for (auto cellID : myCellIDs)
    {
      auto elemType = mesh->getElementType(cellID);
      auto testDofs = elemType->testOrderPtr->totalDofs();
      auto trialDofs = elemType->trialOrderPtr->totalDofs();
      int numCells = 1;
      Camellia::FieldContainer<double> stiffnessEnriched(numCells,trialDofs,testDofs); // the "rectangular" stiffness matrix
      bool rowMajor = true; // trial, test, not test, trial
      bool warnAboutNonzeros = false;  // false because we'll do our own check, here
      auto basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      bf->stiffnessMatrix(stiffnessEnriched, elemType, basisCache->getCellSideParities(), basisCache, rowMajor, warnAboutNonzeros);
      int cellOrdinal = 0;
      for (int trialOrdinal=0; trialOrdinal<trialDofs; trialOrdinal++)
      {
        bool nonZeroFound = false;
        for (int testOrdinal=0; testOrdinal<testDofs; testOrdinal++)
        {
          if (abs(stiffnessEnriched(cellOrdinal,trialOrdinal,testOrdinal)) > tol)
          {
            nonZeroFound = true;
          }
        }
        if (!nonZeroFound)
        {
          success = false;
          
          VarPtr trialVar;
          int varSideOrdinal = -1;
          {
            // find the var that corresponds to this trialOrdinal -- TODO: factor this out and put it somewhere that will be more generally useful
            auto vf = bf->varFactory();
            auto trialIDs = vf->trialIDs();
            auto trialOrdering = elemType->trialOrderPtr;
            for (auto varID : trialIDs)
            {
              int numSides = trialOrdering->getNumSidesForVarID(varID);
              if (numSides > 1)
              {
                for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
                {
                  auto dofIndices = trialOrdering->getDofIndices(varID,sideOrdinal);
                  for (auto dofIndex : dofIndices)
                  {
                    if (dofIndex == trialOrdinal)
                    {
                      varSideOrdinal = sideOrdinal;
                      trialVar = vf->trial(varID);
                      break;
                    }
                  }
                  if (trialVar != Teuchos::null) break;
                }
              }
              else
              {
                auto dofIndices = trialOrdering->getDofIndices(varID);
                for (auto dofIndex : dofIndices)
                {
                  if (dofIndex == trialOrdinal)
                  {
                    trialVar = vf->trial(varID);
                    break;
                  }
                }
              }
              if (trialVar != Teuchos::null) break;
            }
          }
          
          out << "FAILURE: For cell " << cellID << ", stiffness matrix row for trial dof ordinal " << trialOrdinal;
          out << "(corresponding to var " << trialVar->name();
          if (varSideOrdinal != -1)
          {
            out << ", side " << varSideOrdinal;
          }
          out <<  ") is all zeros.\n";
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_2D)
  {
    int spaceDim = 2;
    double width = 1.0, height = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({width, height}, {meshWidth, meshWidth});
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 9); // 2D: vc, ve, plus 3 vm's, 3 vBs, plus vGauss
    
    int numTrialVars = vf->trialVars().size(); // 2D: rho, E, 3 m's, 3 B's, plus 1 flux per equation: 17 variables
    TEST_EQUALITY(numTrialVars, 17);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Formulation_3D)
  {
    int spaceDim = 3;
    double width = 1.0, height = 1.0, depth = 1.0;
    int meshWidth = 2;
    int polyOrder = 1, delta_k = 1;
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology({width, height, depth}, {meshWidth, meshWidth, meshWidth});
    
    auto form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k);
    
    auto vf = form->solution()->bf()->varFactory();
    int numTestVars = vf->testVars().size();
    TEST_EQUALITY(numTestVars, 9); // 2D: vc, ve, plus 3 vm's, 3 vBs, plus vGauss
    
    int numTrialVars = vf->trialVars().size(); // 2D: rho, E, 3 m's, 3 B's, plus 1 flux per equation: 17 variables
    TEST_EQUALITY(numTrialVars, 17);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Residual_1D_SpaceTime_AllOne)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE * 100; // as we would for higher space dimensions, we relax tolerance for space-time a bit
    int cubatureEnrichment = 0;
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr u   =  1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    FunctionPtr rho = one;
    FunctionPtr E   = one;
    FunctionPtr B   =  1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    testSpaceTimeResidual_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Residual_1D_SpaceTime_LinearInTime)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE * 1000; // as we would for higher space dimensions, we relax tolerance for space-time a bit
    int cubatureEnrichment = 0;
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr t   = Function::tn(1);
    FunctionPtr u   = 1./sqrt(6.) * Function::vectorize(t, t, t);
    FunctionPtr rho = one;
    FunctionPtr E   = one + t;
    FunctionPtr B   = 1./sqrt(6.) * Function::vectorize(t, t, t);
    testSpaceTimeResidual_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Residual_1D_Steady_AllOne)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr u   = 1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    FunctionPtr rho = one;
    FunctionPtr E   = one;
    FunctionPtr B   = 1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    testSteadyResidual_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Residual_1D_Transient_AllOne)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr one = Function::constant(1.0);
    FunctionPtr u   = 1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    FunctionPtr rho = one;
    FunctionPtr E   = one;
    FunctionPtr B   =  1./sqrt(6.) * Function::vectorize(one, one, one); // weight chosen so that T = 0.5
    testTransientResidual_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST(IdealMHDFormulation, Residual_1D_Transient_AllOneZeroMagneticFlux)
  {
    double tol = DEFAULT_RESIDUAL_TOLERANCE;
    int cubatureEnrichment = 0;
    FunctionPtr one  = Function::constant(1.0);
    FunctionPtr zero = Function::zero();
    FunctionPtr u   = 1./sqrt(3.) * Function::vectorize(one, one, one); // define u with unit magnitude
    FunctionPtr rho = one;
    FunctionPtr E   = one;
    FunctionPtr B   = Function::vectorize(zero, zero, zero);
    testTransientResidual_1D(u, rho, E, B, cubatureEnrichment, tol, out, success);
  }
} // namespace
