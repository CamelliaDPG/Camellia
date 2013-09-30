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
    void calcNextTimeStep(double dt);
    void printMessage();
};

class ImplicitEulerIntegrator : public TimeIntegrator
{
  private:

  public:

    void runToTime(double T, double dt);
};

class TrapezoidRuleIntegrator : public TimeIntegrator
{
  private:
    LinearTermPtr steadyLinearTerm;

  public:

    TrapezoidRuleIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition) :
      TimeIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition) {}
    void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    void runToTime(double T, double dt);
};
