//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "ExpFunction.h" // defines Ln
#include "Function.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "CompressibleNavierStokesFormulationRefactor.hpp"
#include "CompressibleNavierStokesConservationForm.hpp"
#include "LagrangeConstraints.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

// template logic on the formulation so that we can use either the conservation formulation or the primitive variable formulation
template<class Form>
FunctionPtr energy(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr momentum(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr momentumFlux(Teuchos::RCP<Form> form, bool previousTimeStep); // the thing that we take a divergence of in the strong form (field variables only)

template<class Form>
FunctionPtr pressure(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr temperature(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
FunctionPtr velocity(Teuchos::RCP<Form> form, bool previousTimeStep);

template<class Form>
void addConservationConstraint(Teuchos::RCP<Form> form);

template<>
FunctionPtr pressure<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  // pressure = (gamma - 1) * (E - 1/2 * m dot m / rho)
  double gamma = form->gamma();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr m1  = Function::solution(form->m(1),  soln);
  FunctionPtr E   = Function::solution(form->E(),   soln);
  
  FunctionPtr p = (gamma - 1) * (E - 0.5 * m1 * m1 / rho);
  return p;
}

template<>
FunctionPtr pressure<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  double R = form->R();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr T   = Function::solution(form->T(),   soln);
  FunctionPtr p = R * rho * T;
  return p;
}

template<>
FunctionPtr energy<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr E   = Function::solution(form->E(), soln);
  return E;
}

template<>
FunctionPtr energy<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  double Cv = form->Cv();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr u   = Function::solution(form->u(1),  soln);
  FunctionPtr T   = Function::solution(form->T(),   soln);
  
  FunctionPtr E = rho * (Cv * T + 0.5 * u * u);
  return E;
}

template<>
FunctionPtr momentum<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr m   = Function::solution(form->m(1), soln);
  return m;
}

template<>
FunctionPtr momentum<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr u   = Function::solution(form->u(1),  soln);
  
  FunctionPtr m = rho * u;
  return m;
}

template<>
FunctionPtr momentumFlux<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr rho = Function::solution(form->rho(),  soln);
  FunctionPtr m   = Function::solution(form->m(1),   soln);
  FunctionPtr E   = Function::solution(form->E(),    soln);
  FunctionPtr D   = Function::solution(form->D(1,1), soln);
  
  FunctionPtr p = pressure(form, previousTimeStep);
  FunctionPtr sigma = (D + D - 2./3. * D);
  FunctionPtr mFlux = m * m / rho + p - sigma;
  return mFlux;
}

template<>
FunctionPtr momentumFlux<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr u   = Function::solution(form->u(1),  soln);
  FunctionPtr D   = Function::solution(form->D(1,1), soln);
  
  FunctionPtr p   = pressure(form, previousTimeStep);
  
  FunctionPtr sigma = (D + D - 2./3. * D);
  
  FunctionPtr mFlux = rho * u * u + p - sigma;
  return mFlux;
}

template<>
FunctionPtr velocity<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr m   = Function::solution(form->m(1),  soln);
  return m / rho;
}

template<>
FunctionPtr velocity<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr u   = Function::solution(form->u(1), soln);
  return u;
}

template<>
FunctionPtr temperature<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, bool previousTimeStep)
{
  // Ideal gas:
  //            p = rho * R * T
  // implies
  //            T = p / (rho * R)
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  double R = form->R();
  FunctionPtr rho = Function::solution(form->rho(), soln);
  FunctionPtr p = pressure(form, previousTimeStep);
  
  FunctionPtr T = p / (rho * R);
  return T;
}

template<>
FunctionPtr temperature<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, bool previousTimeStep)
{
  SolutionPtr soln = previousTimeStep ? form->solutionPreviousTimeStep() : form->solution();
  FunctionPtr T = Function::solution(form->T(), soln);
  return T;
}

template<>
void addConservationConstraint(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form)
{
  cout << "WARNING: addConservationConstraint unimplemented for primitive-variable formulation...\n";
  // TODO: implement this
}

template<>
void addConservationConstraint(Teuchos::RCP<CompressibleNavierStokesConservationForm> form)
{
  int rank = MPIWrapper::CommWorld()->MyPID();
  cout << "TRYING CONSERVATION ENFORCEMENT.\n";
  auto soln = form->solution();
  auto solnIncrement = form->solutionIncrement();
  auto prevSoln = form->solutionPreviousTimeStep();
  auto bf  = solnIncrement->bf();
  auto rhs = solnIncrement->rhs();
  auto dt = form->getTimeStep();
  
  Teuchos::RCP<LagrangeConstraints> constraints = Teuchos::rcp(new LagrangeConstraints);
  // vc constraint:
  VarPtr vc = form->vc();
  map<int, FunctionPtr> vcEqualsOne = {{vc->ID(), Function::constant(1.0)}};
  LinearTermPtr vcTrialFunctional = bf->trialFunctional(vcEqualsOne) * dt;
  FunctionPtr   vcRHSFunction     = rhs->linearTerm()->evaluate(vcEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(vcTrialFunctional == vcRHSFunction);
  if (rank == 0) cout << "Added element constraint " << vcTrialFunctional->displayString() << " == " << vcRHSFunction->displayString() << endl;

  const int spaceDim = 1;
  // vm constraint(s):
  for (int d=0; d<spaceDim; d++)
  {
    // test with 1
    VarPtr vm = form->vm(d+1);
    map<int, FunctionPtr> vmEqualsOne = {{vm->ID(), Function::constant(1.0)}};
    LinearTermPtr trialFunctional = bf->trialFunctional(vmEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
    FunctionPtr rhsFxn = rhs->linearTerm()->evaluate(vmEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
    constraints->addConstraint(trialFunctional == rhsFxn);

    if (rank == 0) cout << "Added element constraint " << trialFunctional->displayString() << " == " << rhsFxn->displayString() << endl;
  }
  // ve constraint:
  VarPtr ve = form->ve();
  map<int, FunctionPtr> veEqualsOne = {{ve->ID(), Function::constant(1.0)}};
  LinearTermPtr veTrialFunctional = bf->trialFunctional(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  FunctionPtr   veRHSFunction     = rhs->linearTerm()->evaluate(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(veTrialFunctional == veRHSFunction);
  if (rank == 0) cout << "Added element constraint " << veTrialFunctional->displayString() << " == " << veRHSFunction->displayString() << endl;

  // although enforcement only happens in solnIncrement, the constraints change numbering of dofs, so we need to set constraints in each Solution object
  solnIncrement->setLagrangeConstraints(constraints);
  soln->setLagrangeConstraints(constraints);
  prevSoln->setLagrangeConstraints(constraints);
}

template<class Form>
void setVelocityOrMomentum(Teuchos::RCP<Form> form, std::map<int, FunctionPtr> &initialStateMap, FunctionPtr u, FunctionPtr rho, FunctionPtr T);

template<class Form>
void setTempOrEnergy(Teuchos::RCP<Form> form, std::map<int, FunctionPtr> &initialStateMap, FunctionPtr u, FunctionPtr rho, FunctionPtr T);

template<>
void setVelocityOrMomentum<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                     FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->m(1)->ID()]    = u * rho;
}

template<>
void setVelocityOrMomentum<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                        FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->u(1)->ID()]    = u;
}

template<>
void setTempOrEnergy<CompressibleNavierStokesConservationForm>(Teuchos::RCP<CompressibleNavierStokesConservationForm> form, std::map<int, FunctionPtr> &initialStateMap,
                                                               FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  double Cv = form->Cv();
  FunctionPtr E = rho * (Cv * T + 0.5 * u * u);
  initialStateMap[form->E()->ID()] = E;
}

template<>
void setTempOrEnergy<CompressibleNavierStokesFormulationRefactor>(Teuchos::RCP<CompressibleNavierStokesFormulationRefactor> form, std::map<int, FunctionPtr> &initialStateMap,
                                                                        FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  initialStateMap[form->T()->ID()]     = T;
}

void writeFunctions(MeshPtr mesh, int meshWidth, int polyOrder, double x_a, std::vector<FunctionPtr> &functions, std::vector<string> &functionNames, string filePrefix)
{
  if (meshWidth != mesh->numElements())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "writeFunctions() does not yet support refined meshes");
  }
  if (1 != mesh->getTopology()->getDimension())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "writeFunctions() only supports 1D meshes");
  }
  int numPoints = max(polyOrder*2 + 1, 2); // polyOrder * 2 will, hopefully, give reasonable approximation of high-order curves
  Intrepid::FieldContainer<double> refCellPoints(numPoints,1);
  double dx = 2.0 / (refCellPoints.size() - 1); // 2.0 is the size of the reference element
  for (int i=0; i<refCellPoints.size(); i++)
  {
    refCellPoints(i,0) = -1.0 + dx * i;
  }
//  cout << "numPoints = " << numPoints << endl;
//  cout << "dx = " << dx << endl;
//  cout << "refCellPoints:\n" << refCellPoints;
  
  int numFunctions = functionNames.size();
  vector<Intrepid::FieldContainer<double>> pointData(numFunctions, Intrepid::FieldContainer<double>(meshWidth,numPoints));
  for (int functionOrdinal=0; functionOrdinal<numFunctions; functionOrdinal++)
  {
    pointData[functionOrdinal].initialize(0.0);
  }
  Teuchos::Array<int> cellDim;
  cellDim.push_back(1);  // C
  cellDim.push_back(numPoints); // P
  auto & myCellIDs = mesh->cellIDsInPartition();
  for (auto cellID : myCellIDs)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    basisCache->setRefCellPoints(refCellPoints);
    
    for (int functionOrdinal=0; functionOrdinal<numFunctions; functionOrdinal++)
    {
      auto f = functions[functionOrdinal];
      auto name = functionNames[functionOrdinal];
      
      Intrepid::FieldContainer<double> localData(cellDim, &pointData[functionOrdinal](cellID,0));
      
      f->values(localData, basisCache);
    }
  }
  
  for (auto & data : pointData)
  {
    MPIWrapper::entryWiseSum(*mesh->Comm(), data);
  }
  
  double h = 1.0 / meshWidth;
  for (int functionOrdinal=0; functionOrdinal<functionNames.size(); functionOrdinal++)
  {
    int totalPoints = meshWidth*(numPoints-1); // cells overlap; eliminate duplicates
    Intrepid::FieldContainer<double> xyPoints(totalPoints,2);
    
    ostringstream fileName;
    fileName << filePrefix << "_" << functionNames[functionOrdinal] << ".dat";
    
    ofstream fout(fileName.str());
    fout << setprecision(6);
    
    for (int i=0; i<totalPoints; i++)
    {
      double x = x_a + dx * i / 2.0 * h;
      int cellOrdinal  = i / (numPoints-1);
      int pointOrdinal = i % (numPoints-1);
      if (i == totalPoints - 1)
      {
        // for last cell, we do take its final point...
        pointOrdinal = numPoints - 1;
      }
      
      double y = pointData[functionOrdinal](cellOrdinal,pointOrdinal);
      xyPoints(i,0) = x;
      xyPoints(i,1) = y;
    }
    GnuPlotUtil::writeXYPoints(fileName.str(), xyPoints);
  }
}

enum TestNormChoice
{
  STEADY_GRAPH_NORM,
  TRANSIENT_GRAPH_NORM,
  EXPERIMENTAL_CONSERVATIVE_NORM
};

FunctionPtr ln(FunctionPtr arg)
{
  return Teuchos::rcp(new Ln<double>(arg));
}

template<class Form>
int runSolver(Teuchos::RCP<Form> form, bool conservationVariables, double dt, double finalTime, int meshWidth, double x_a, double x_b,
              int polyOrder, int cubatureEnrichment, bool useCondensedSolve, double nonlinearTolerance, int continuationSteps,
              double continuationTolerance, TestNormChoice normChoice, bool enforceConservationUsingLagrangeMultipliers)
{
  if (enforceConservationUsingLagrangeMultipliers)
  {
    addConservationConstraint(form);
  }
  int rank = Teuchos::GlobalMPISession::getRank();
  const int spaceDim = 1;
  double mu = form->mu();
  bool pureEuler = (mu == 0.0);
  
  SolutionPtr solnIncrement = form->solutionIncrement();
  SolutionPtr soln = form->solution();
    
  double Re = (!pureEuler) ? 1.0 / form->mu() : -1.0 ;
  
  auto ip = solnIncrement->ip(); // this will be the transient graph norm...
  if (normChoice == STEADY_GRAPH_NORM)
  {
    auto steadyBF = form->steadyBF();
    auto steadyIP = steadyBF->graphNorm();
    auto bf = form->bf();
    
    if (conservationVariables)
    {
      // then steadyBF will have no rho terms wherever velocity is zero
      // unclear what the right thing to do is; for now we just add something in
      steadyBF = Teuchos::rcp( new BF(*steadyBF) ); // default copy constructor
      steadyBF->addTerm(form->rho(),form->vc());
    }
    cout << "steadyBF: " << steadyBF->displayString() << endl;
    bf->setBFForOptimalTestSolve(steadyBF);
    cout << "Using steady graph norm:\n";
    steadyIP->printInteractions();
    soln->setIP(steadyIP);
    solnIncrement->setIP(steadyIP);
    form->solutionPreviousTimeStep()->setIP(steadyIP);
  }
  else if (normChoice == EXPERIMENTAL_CONSERVATIVE_NORM)
  {
    auto dt = form->getTimeStep();
    auto steadyIP = IP::ip();
    // let's walk through the members, dropping any summands that involve test terms without gradients
    auto linearTerms = ip->getLinearTerms();
    for (auto lt : linearTerms)
    {
      LinearTermPtr revisedLT = Teuchos::rcp(new LinearTerm);
      auto summands = lt->summands();
      for (auto summand : summands)
      {
        auto weight = summand.first;
        auto var = summand.second;
        if (var->op() == OP_VALUE)
        {
          continue; // skip this summand
        }
        else
        {
          revisedLT = revisedLT + weight * var;
        }
      }
      if (revisedLT->summands().size() > 0)
      {
        steadyIP->addTerm(revisedLT);
      }
    }
    // now, add boundary terms for each test function:
    steadyIP->addBoundaryTerm(form->vc());
    steadyIP->addBoundaryTerm(form->vm(1));
    steadyIP->addBoundaryTerm(form->ve());
    cout << "Using modified steady graph norm:\n";
    steadyIP->printInteractions();
    soln->setIP(steadyIP);
    solnIncrement->setIP(steadyIP);
    form->solutionPreviousTimeStep()->setIP(steadyIP);
  }
  
  int numTimeSteps = finalTime / dt;
  
  if (rank == 0)
  {
    using namespace std;
    cout << "Solving with:\n";
    if (pureEuler)
      cout << "pure Euler\n";
    else
      cout << "Re = " << Re << endl;
    cout << "p  = " << polyOrder << endl;
    cout << "dt = " << dt << endl;
    cout << meshWidth << " elements; " << numTimeSteps << " timesteps.\n";
  }
  
  form->setTimeStep(dt);
  solnIncrement->setUseCondensedSolve(useCondensedSolve);
  solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichment);
//  solnIncrement->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
//  solnIncrement->setWriteRHSToMatrixMarketFile(   true, "/tmp/b.dat");
  
  double gamma = form->gamma();
  double c_v   = form->Cv();
  
  double rho_a = 1.0; // prescribed density at left
  double u_a   = 0.0; // Mach number
  double p_a   = 1.0; // prescribed pressure at left
  double T_a   = p_a / (rho_a * (gamma - 1.) * c_v);
  
  double rho_b = 0.125;
  double p_b   = 0.1;
  double u_b   = 0.0;
  double T_b   = p_b / (rho_b * (gamma - 1.) * c_v);
  
  double R  = form->R();
  double Cv = form->Cv();
  
  if (rank == 0)
  {
    cout << "R =   " << R << endl;
    cout << "Cv =  " << Cv << endl;
    cout << "State on left:\n";
    cout << "rho = " << rho_a << endl;
    cout << "p   = " << p_a   << endl;
    cout << "T   = " << T_a   << endl;
    cout << "u   = " << u_a   << endl;
    cout << "State on right:\n";
    cout << "rho = " << rho_b << endl;
    cout << "p   = " << p_b   << endl;
    cout << "T   = " << T_b   << endl;
    cout << "u   = " << u_b   << endl;
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  map<int, FunctionPtr> initialState;
  
  {
    auto H_right = Function::heaviside((x_a + x_b)/2.0); // Heaviside is 0 left of center, 1 right of center
    auto H_left  = 1.0 - H_right;  // this guy is 1 left of center, 0 right of center
    auto step = [&](double val_a, double val_b)
    {
      return H_left * val_a + H_right * val_b;
    };
    
    FunctionPtr rho = step(rho_a,rho_b);
    FunctionPtr T   = step(T_a, T_b);
    FunctionPtr u   = step(u_a, u_b);
    
    initialState[form->rho()->ID()]   = rho;
    setVelocityOrMomentum(form, initialState, u, rho, T);
    setTempOrEnergy      (form, initialState, u, rho, T);
    if (!pureEuler)
    {
      initialState[form->q(1)->ID()]    = Function::zero();
      initialState[form->D(1,1)->ID()]  = Function::zero();
    }
    
    // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
    // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
    initialState[form->tc()->ID()]    = rho * u * n_x;
    initialState[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
    initialState[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
    
    if (!pureEuler)
    {
      initialState[form->T_hat()->ID()] = T;
      initialState[form->u_hat(1)->ID()] = u;
    }
  }
  
  // define the pressure so we can plot in our solution export
  FunctionPtr p = pressure(form, true); // true: previous time step
  // define u so we can plot in solution export (in case we're in conservation variables)
  FunctionPtr u = velocity(form, true); // true: previous time step
  // define change in entropy so we can plot in our solution export
  // s2 - s1 = c_p ln (T2/T1) - R ln (p2/p1)
  
  FunctionPtr ds;
  {
    FunctionPtr p1 = pressure(form, true);
    FunctionPtr p2 = pressure(form,false);
    FunctionPtr T1 = temperature(form, true);
    FunctionPtr T2 = temperature(form, false);
    double c_p = form->Cp();
    ds = c_p * ln(T2 / T1) - R * ln(p2/p1);
  }
  
  // project the initial state both onto the solution object representing the previous time state,
  // as well as the current state (the latter is the initial guess for the current time step).
  const int solutionOrdinal = 0;
  form->solutionPreviousTimeStep()->projectOntoMesh(initialState, solutionOrdinal);
  form->solution()->projectOntoMesh(initialState, solutionOrdinal);
  
  MeshPtr mesh = form->solutionIncrement()->mesh();
  
  vector<FunctionPtr> functionsToPlot;
  vector<string> functionNames;

  if (conservationVariables)
  {
    functionsToPlot = {p,u,ds};
    functionNames   = {"pressure","velocity","entropy change"};
  }
  else
  {
    functionsToPlot = {p,ds};
    functionNames   = {"pressure","entropy change"};
  }
  
  // history export gets every nonlinear increment as a separate step
  HDF5Exporter solutionHistoryExporter(mesh, "sodShockSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(mesh, "sodShockSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  if (!pureEuler)
    solnName << "sodShockSolutionRe" << Re << "_dt" << dt << "_k" << polyOrder;
  else
    solnName << "sodShockSolutionEuler" << "_dt" << dt << "_k" << polyOrder;
  if (conservationVariables) solnName << "_conservationVariables";
  HDF5Exporter solutionExporter(mesh, solnName.str(), ".");
  
  solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), 0.0);
  solutionHistoryExporter.exportSolution(form->solution(), 0.0);
  solutionExporter.exportSolution(form->solutionPreviousTimeStep(), functionsToPlot, functionNames, 0.0);
  
//  IPPtr naiveNorm = form->solutionIncrement()->bf()->naiveNorm(spaceDim);
//  if (rank == 0)
//  {
//    cout << "*************************************************************************\n";
//    cout << "**********************    USING NAIVE NORM    ***************************\n";
//    cout << "*************************************************************************\n";
//  }
//  form->solutionIncrement()->setIP(naiveNorm);
  
  // Borrowing some from Truman's dissertation code
  SpatialFilterPtr leftX  = SpatialFilter::matchingX(x_a);
  SpatialFilterPtr rightX = SpatialFilter::matchingX(x_b);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  
  if (rank == 0)
  {
    std::cout << "T_a = " << T_a << std::endl;
    std::cout << "T_b = " << T_b << std::endl;
  }
  //  (SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
  form->addMassFluxCondition             ( SpatialFilter::allSpace(), Function::zero());
  form->addEnergyFluxCondition           ( SpatialFilter::allSpace(), Function::zero());
  form->addMomentumFluxCondition(      leftX, Function::constant(rho_a), Function::constant(u_a), Function::constant(T_a));
  form->addMomentumFluxCondition(     rightX, Function::constant(rho_b), Function::constant(u_b), Function::constant(T_b));
  if (!pureEuler)
  {
    form->addVelocityTraceCondition(     leftX, Function::constant(u_a));
    form->addVelocityTraceCondition(    rightX, Function::constant(u_b));
    form->addTemperatureTraceCondition(  leftX, Function::constant(T_a));
    form->addTemperatureTraceCondition( rightX, Function::constant(T_b));
  }
  
  FunctionPtr rho = Function::solution(form->rho(), form->solution());
  FunctionPtr m   = momentum(form, false); // false: not previous time step (current)
  FunctionPtr E   = energy(form, false);
  FunctionPtr tc  = Function::solution(form->tc(),  form->solution(), true);
  FunctionPtr tm  = Function::solution(form->tm(1), form->solution(), true);
  FunctionPtr te  = Function::solution(form->te(),  form->solution(), true);
  
  auto printConservationReport = [&]() -> void
  {
    double totalMass = rho->integrate(mesh);
    double totalMomentum = m->integrate(mesh);
    double totalEnergy = E->integrate(mesh);
    double dsIntegral = ds->integrate(mesh);
    
    if (rank == 0)
    {
      cout << "Total Mass:        " << totalMass << endl;
      cout << "Total Momentum:    " << totalMomentum << endl;
      cout << "Total Energy:      " << totalEnergy << endl;
      cout << "Change in Entropy: " << dsIntegral << endl;
    }
  };
  
  // elementwise local momentum conservation:
  FunctionPtr rho_soln = Function::solution(form->rho(), form->solution());
  FunctionPtr rho_prev = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr rhoTimeStep = (rho_soln - rho_prev) / dt;
  FunctionPtr rhoFlux = Function::solution(form->tc(), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr momentum_soln = momentum(form, false); // false: not previous time step, but current
  FunctionPtr momentum_prev = momentum(form, true);
  FunctionPtr momentumTimeStep = (momentum_soln - momentum_prev) / dt;
  FunctionPtr momentumFlux = Function::solution(form->tm(1), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr energy_soln = energy(form, false); // not previous time step, but current
  FunctionPtr energy_prev = energy(form, true);
  FunctionPtr energyTimeStep = (energy_soln - energy_prev) / dt;
  FunctionPtr energyFlux = Function::solution(form->te(), form->solution(), true); // include sideParity weights

  FunctionPtr pressure_soln = pressure(form, false); // false: current time step (a.k.a background flow)
  
//  vector<FunctionPtr> conservationFluxes = {momentumFlux};
//  vector<FunctionPtr> timeDifferences    = {momentumTimeStep};
  vector<FunctionPtr> conservationFluxes = {rhoFlux,     momentumFlux,     energyFlux};
  vector<FunctionPtr> timeDifferences    = {rhoTimeStep, momentumTimeStep, energyTimeStep};
  
  int numConserved = conservationFluxes.size();
  
  auto printLocalConservationReport = [&]() -> void
  {
    auto & myCellIDs = mesh->cellIDsInPartition();
    int cellOrdinal = 0;
    vector<double> maxConservationFailures(numConserved);
    for (int conservedOrdinal=0; conservedOrdinal<numConserved; conservedOrdinal++)
    {
      double maxConservationFailure = 0.0;
      for (auto cellID : myCellIDs)
      {
//        cout << "timeDifferences function: " << timeDifferences[conservedOrdinal]->displayString() << endl;
        double timeDifferenceIntegral = timeDifferences[conservedOrdinal]->integrate(cellID, mesh);
        double fluxIntegral = conservationFluxes[conservedOrdinal]->integrate(cellID, mesh);
        double cellIntegral = timeDifferenceIntegral + fluxIntegral;
        maxConservationFailure = std::max(abs(cellIntegral), maxConservationFailure);
        cellOrdinal++;
      }
      maxConservationFailures[conservedOrdinal] = maxConservationFailure;
    }
    vector<double> globalMaxConservationFailures(numConserved);
    mesh->Comm()->MaxAll(&maxConservationFailures[0], &globalMaxConservationFailures[0], numConserved);
    if (rank == 0) {
      cout << "Max cellwise (rho,m,E) conservation failures: ";
      for (double failure : globalMaxConservationFailures)
      {
        cout << failure << "\t";
      }
      cout << endl;
    }
//    for (int cellOrdinal=0; cellOrdinal<myCellCount; cellOrdinal++)
//    {
//      cout << "cell ordinal " << cellOrdinal << ", conservation failure: " << cellIntegrals[cellOrdinal] << endl;
//    }
  };
  
  printConservationReport();
  double t = 0;
  double Re_0;
  int timeStepNumber = 0;
  
  // for the moment, time step adjustment only reduces time step size
  auto adjustTimeStep = [&]() -> void
  {
    // Check that dt is reasonable vis-a-vis CFL
    double h = (x_b-x_a) / meshWidth / polyOrder;
    FunctionPtr soundSpeed = Function::sqrtFunction(gamma * pressure_soln / rho_soln);
    FunctionPtr fluidSpeed = Function::sqrtFunction((momentum_soln / rho_soln) * (momentum_soln / rho_soln));
    double maxSoundSpeed = soundSpeed->linfinitynorm(mesh);
    double maxFluidSpeed = fluidSpeed->linfinitynorm(mesh);
    double dtCFL = h / (maxSoundSpeed + maxFluidSpeed);
    if (dt > dtCFL)
    {
      if (rank == 0)
      {
        cout << "Time step " << dt << " exceeds CFL-stable value of " << dtCFL << endl;
      }
      while (dt > dtCFL)
      {
        dt /= 2.0;
      }
      numTimeSteps = (finalTime - t) / dt + timeStepNumber;
      if (rank == 0)
      {
        cout << "Set time step to " << dt;
        cout << "; set numTimeSteps to " << numTimeSteps << endl;
      }
    }
  };
  
  // for now, we hardcode this option -- we are exploring time steps bigger than the CFL
  bool enforceTimeStepSatisfiesInitialCFL = false;
  if (enforceTimeStepSatisfiesInitialCFL)
  {
    adjustTimeStep();
  }
  
  for (timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    double l2NormOfIncrement = 1.0;
    int stepNumber = 0;
    int continuationStepNumber = 0;
    int maxNonlinearSteps = 10;
    double Re_current, Re_multiplier;
    if (!pureEuler && (continuationSteps > 0))
    {
      if (timeStepNumber == 0)
      {
        Re_0 = std::min(1e2,Re); // for continuation in Reynolds number
      }
      else
      {
        Re_0 = std::min(1e4,Re); // for continuation in Reynolds number
      }
      form->setMu(1.0 / Re_0);
      if (rank == 0) cout << "for continuation, set Re to " << Re_0 << endl;
      Re_current = Re_0;
      Re_multiplier = pow(Re / Re_0, 1./continuationSteps);
    }
    else
    {
      // if Euler, no continuation
      continuationStepNumber = continuationSteps;
    }
    double alpha = 1.0;
    while (((continuationStepNumber < continuationSteps) || (l2NormOfIncrement > nonlinearTolerance)) && (stepNumber < maxNonlinearSteps))
    {
      alpha = form->solveAndAccumulate();
      int solveCode = form->getSolveCode();
      if (solveCode != 0)
      {
        if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
        exit(1);
      }
      l2NormOfIncrement = form->L2NormSolutionIncrement();
      if (rank == 0)
      {
        std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
        std::cout << " (alpha = " << alpha << ")" << std::endl;
      }
      
      stepNumber++;
      solutionHistoryExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
      solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
      if ((continuationStepNumber < continuationSteps) && ((l2NormOfIncrement < continuationTolerance) || stepNumber == maxNonlinearSteps))
      {
        Re_current *= Re_multiplier;
        form->setMu(1./Re_current);
        if (rank == 0) cout << "for continuation, set Re to " << Re_current << endl;
        continuationStepNumber++;
        // since we have changed the Re number, reset the step counter, as well as the l2NormOfIncrement
        l2NormOfIncrement = 1.0;
        stepNumber = 0;
      }
    }
    if (l2NormOfIncrement > nonlinearTolerance)
    {
      if (rank == 0)
      {
        cout << "Nonlinear iteration failed.  Exiting...\n";
      }
      return -1;
    }
    t += dt;
    
    adjustTimeStep();
    
    printLocalConservationReport(); // since this depends on the difference between current/previous solution, we need to call before we set prev to current.
    solutionExporter.exportSolution(form->solution(),functionsToPlot,functionNames,t); // similarly, since the entropy compares current and previous, need this to happen before setSolution()
    printConservationReport();
    
    if (rank == 0) std::cout << "========== t = " << t << ", time step number " << timeStepNumber+1 << " ==========\n";
    if (timeStepNumber != numTimeSteps - 1)
    {
      form->solutionPreviousTimeStep()->setSolution(form->solution());
    }
  }
  
  // now that we're at final time, let's output pressure, velocity, density in a format suitable for plotting
  functionsToPlot.push_back(rho);
  functionNames.push_back("density");
  
  // writeFunctions() only supports unrefined meshes right now.  If we want to emit .dat files for our graded meshes, we'll need to fix that.
  if (mesh->numElements() == meshWidth)
  {
    writeFunctions(mesh, meshWidth, polyOrder, x_a, functionsToPlot, functionNames, solnName.str());
  }
  else
  {
    if (rank == 0)
    {
      std::cout << "Skipping emission of .dat files for TeX plotting, because writeFunctions() does not yet support refined meshes.\n";
    }
  }
  
  return 0;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  int meshWidth = 200;
  int polyOrder = 2;
  int delta_k   = 1; // 1 is likely sufficient in 1D
  bool useCondensedSolve = false; // condensed solve UNSUPPORTED for now; before turning this on, make sure the various Solution objects are all set to use this in a compatible way...
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  double nonlinearTolerance    = 1e-2;
  double continuationTolerance = 1e-2;
  int continuationSteps = 4;
  
  double finalTime = 0.20; // standard for Sod is to take it to t = 0.20
  
  double x_a   = -0.5;
  double x_b   = 0.5;

  double Re    = 1e8;   // Reynolds number
  double dt    = 0.001; // time step
  
  bool useConservationFormulation = true;
  bool useEuler = false;
  bool runWriteFunctionTest = false; // option for debugging some output code...
  bool enforceConservationUsingLagrangeMultipliers = false;
  int meshGradingIterations = 0; // number of times to refine at the elements at the center of the discretization
  
  std::map<string, TestNormChoice> normChoices = {
    {"steadyGraph", STEADY_GRAPH_NORM},
    {"transientGraph", TRANSIENT_GRAPH_NORM},
    {"experimental", EXPERIMENTAL_CONSERVATIVE_NORM}
  };
  
  std::string normChoiceString = "transientGraph";
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("meshWidth", &meshWidth);
  cmdp.setOption("Re", &Re);
  cmdp.setOption("dt", &dt);
  cmdp.setOption("deltaP", &delta_k);
  cmdp.setOption("continuationSteps", &continuationSteps);
  cmdp.setOption("nonlinearTol", &nonlinearTolerance);
  cmdp.setOption("continuationTol", &continuationTolerance);
  cmdp.setOption("conservationVariables", "primitiveVariables", &useConservationFormulation);
  cmdp.setOption("euler", "fullNS", &useEuler);
  cmdp.setOption("enforceConservation", "dontEnforceConservation", &enforceConservationUsingLagrangeMultipliers);
  cmdp.setOption("finalTime", &finalTime);
  cmdp.setOption("meshGradingIterations", &meshGradingIterations, "Number of times to refine the elements immediately around center -- adds two elements per iteration");
  
  cmdp.setOption("norm", &normChoiceString);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  TestNormChoice normChoice;
  if (normChoices.find(normChoiceString) != normChoices.end())
  {
    normChoice = normChoices[normChoiceString];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);
  
  if (meshGradingIterations > 0)
  {
    // a bit of a cheat here, in that we use knowledge of how the elements are numbered...
    GlobalIndexType centerCellLeft = meshWidth / 2 - 1;
    GlobalIndexType centerCellRight = meshWidth / 2;
    
    RefinementPatternPtr lineRefinement = RefinementPattern::regularRefinementPatternLine();
    
    for (int meshGradingIteration = 0; meshGradingIteration < meshGradingIterations; meshGradingIteration++)
    {
      GlobalIndexType cellCount = meshTopo->cellCount();
      meshTopo->refineCell(centerCellLeft, lineRefinement, cellCount);
      cellCount += lineRefinement->numChildren();
      
      // update the left center cell number to be the rightmost of the newly introduced cells
      // again using knowledge of how the elements are numbered...
      centerCellLeft = cellCount - 1; //
      
      meshTopo->refineCell(centerCellRight, lineRefinement, cellCount);
      cellCount += lineRefinement->numChildren();
      
      // update the right center cell number to be the leftmost of the newly introduced cells
      // again using knowledge of how the elements are numbered...
      centerCellRight = cellCount - 2;
    }
  }
  
  bool useConformingTraces = true;
  
  if (runWriteFunctionTest)
  {
    auto form = CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                       meshTopo, polyOrder, delta_k);
    vector<FunctionPtr> functions = {Function::xn(1)};
    vector<string> functionNames  = {"linear"};
    writeFunctions(form->solution()->mesh(), meshWidth, polyOrder, -0.5, functions, functionNames, "testWrite");
    cout << "wrote test plot data; exiting...\n";
    return 0;
  }
  
  if (useConservationFormulation)
  {
    if (useEuler)
    {
      auto form = CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                         meshTopo, polyOrder, delta_k);
      
      return runSolver(form, useConservationFormulation, dt, finalTime, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
    else
    {
      auto form = CompressibleNavierStokesConservationForm::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                    meshTopo, polyOrder, delta_k);
      return runSolver(form, useConservationFormulation, dt, finalTime, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
  }
  else
  {
    if (useEuler)
    {
      auto form = CompressibleNavierStokesFormulationRefactor::timeSteppingEulerFormulation(spaceDim, useConformingTraces,
                                                                                            meshTopo, polyOrder, delta_k);
      return runSolver(form, useConservationFormulation, dt, finalTime, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
    else
    {
      auto form = CompressibleNavierStokesFormulationRefactor::timeSteppingFormulation(spaceDim, Re, useConformingTraces,
                                                                                       meshTopo, polyOrder, delta_k);
      return runSolver(form, useConservationFormulation, dt, finalTime, meshWidth, x_a, x_b, polyOrder, cubatureEnrichment, useCondensedSolve,
                       nonlinearTolerance, continuationSteps, continuationTolerance, normChoice, enforceConservationUsingLagrangeMultipliers);
    }
  }
}
