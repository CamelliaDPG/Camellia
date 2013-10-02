#include "InnerProductScratchPad.h"
#include "Mesh.h"
#include "Solution.h"

// class TimeIntegrator;
// typedef Teuchos::RCP<TimeIntegrator> TimeIntegratorPtr;

class InvDtFunction : public Function {
  double __invDt;
  public:
  InvDtFunction(double _dt) : Function(0){
    __invDt = 1./_dt;
  }
  void setDt(double _dt){
    __invDt = 1./_dt;
  }
  double getDt(){
    return 1./__invDt;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(__invDt);
  }
};

class TimeIntegrator
{
  protected:
    BFPtr _bf;
    Teuchos::RCP<RHSEasy> _rhs;
    MeshPtr _mesh;
    SolutionPtr _solution;
    FunctionPtr _invDt;
    double _t;
    double _dt;
    int _timestep;

  public:
    SolutionPtr prevSolution;

    TimeIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition);
    virtual void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    virtual void runToTime(double T, double dt) = 0;
    virtual void calcNextTimeStep(double dt);
    void printMessage();
};

class ImplicitEulerIntegrator : public TimeIntegrator
{
  private:

  public:
    ImplicitEulerIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition);
    void runToTime(double T, double dt);
};

class RungeKuttaIntegrator : public TimeIntegrator
{
  protected:
    // LinearTermPtr _steadyLinearTerm;

  public:
    RungeKuttaIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition);
    virtual void runToTime(double T, double dt);
};

// Also known as the trapezoid rule
class ESDIRK2Integrator : public RungeKuttaIntegrator
{
  private:
    double a21, a22;
    double b1, b2;
    SolutionPtr _k2Solution;
    Teuchos::RCP<RHSEasy> _k2RHS;

  public:

    ESDIRK2Integrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition);
    void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    void calcNextTimeStep(double dt);
};

class ESDIRK4Integrator : public RungeKuttaIntegrator
{
  private:
    // Standard Butcher tables run from 0 to s
    // I am running from 0 to s-1 to match 0 index
    int numStages;
    double a[6][6];
    double b[6];
    // For ESDIRK schemes, stage 1 is prevSolution
    SolutionPtr _stageSolution[6];
    Teuchos::RCP<RHSEasy> _stageRHS[6];
    LinearTermPtr _steadyLinearTerm[5];

  public:

    ESDIRK4Integrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition);
    void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    void calcNextTimeStep(double dt);
};
