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

class ESDIRKIntegrator : public TimeIntegrator
{
  private:
    // Standard Butcher tables run from 0 to s
    // I am running from 0 to s-1 to match 0 index
    int _numStages;
    vector< vector<double> > a;
    vector<double> b;
    // For ESDIRK schemes, stage 1 is prevSolution
    vector< SolutionPtr > _stageSolution;
    vector< Teuchos::RCP<RHSEasy> > _stageRHS;
    vector< LinearTermPtr > _steadyLinearTerm;

  public:

    ESDIRKIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition, int numStages);
    virtual void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    virtual void runToTime(double T, double dt);
    virtual void calcNextTimeStep(double dt);
};
