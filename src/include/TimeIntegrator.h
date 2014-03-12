#include "InnerProductScratchPad.h"
#include "Mesh.h"
#include "Solution.h"

// TODO: change L2 error to use different variables

class InvDtFunction : public Function
{
  private:
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

class SteadyResidual
{
  protected:
    VarFactory &varFactory;
  public:
    SteadyResidual(VarFactory &varFactory):varFactory(varFactory) {};
    virtual LinearTermPtr createResidual(SolutionPtr solution, bool includeBoundaryTerms) = 0;
};

class TimeIntegrator
{
  protected:
    int _commRank;
    BFPtr _steadyJacobian;
    SteadyResidual &_steadyResidual;
    RHSPtr _rhs;
    BCPtr _bc;
    SolutionPtr _solution;
    SolutionPtr _prevTimeSolution;
    SolutionPtr _prevNLSolution;
    SolutionPtr _zeroSolution;
    FunctionPtr _invDt;
    double _t;
    double _dt;
    int _timestep;
    bool _nonlinear;
    double _nlTolerance;
    double _nlL2Error;
    int _nlIteration;
    int _nlIterationMax;
    vector<VarPtr> testVars;
    vector<VarPtr> trialVars;

  public:
    TimeIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
        BCPtr bc, IPPtr ip, map<int, FunctionPtr> initialCondition, bool nonlinear);
    SolutionPtr solution();
    SolutionPtr solutionUpdate();
    SolutionPtr prevSolution();
    FunctionPtr invDt();
    void setNLTolerance(double tol) { _nlTolerance = tol; }
    double getNLTolerance() { return _nlTolerance; }
    void setNLIterationMax(double nlIterationMax) { _nlIterationMax = nlIterationMax; }
    double getNLIterationMax() { return _nlIterationMax; }
    virtual void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    virtual void runToTime(double T, double dt) = 0;
    virtual void calcNextTimeStep(double dt);
    void printTimeStepMessage();
    void printNLMessage();
};

class ImplicitEulerIntegrator : public TimeIntegrator
{
  private:

  public:
    ImplicitEulerIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
        BCPtr bc, IPPtr ip, map<int, FunctionPtr> initialCondition, bool nonlinear);
    void runToTime(double T, double dt);
};

class ESDIRKIntegrator : public TimeIntegrator
{
  private:
    // Standard Butcher tables run from 1 to s
    // I am running from 0 to s-1 to match 0 index
    int _numStages;
    vector< vector<double> > a;
    vector<double> b;
    vector<double> c;
    // For ESDIRK schemes, first stage is _prevTimeSolution
    vector< SolutionPtr > _stageSolution;
    vector< RHSPtr > _stageRHS;
    vector< LinearTermPtr > _steadyLinearTerm;

  public:

    ESDIRKIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
        BCPtr bc, IPPtr ip, map<int, FunctionPtr> initialCondition, int numStages, bool nonlinear);
    virtual void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    virtual void runToTime(double T, double dt);
    virtual void calcNextTimeStep(double dt);
};
