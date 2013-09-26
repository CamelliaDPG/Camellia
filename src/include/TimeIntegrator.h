#include "InnerProductScratchPad.h"
#include "Mesh.h"
#include "Solution.h"

// class TimeIntegrator;
// typedef Teuchos::RCP<TimeIntegrator> TimeIntegratorPtr;

class InvDtFunction : public Function {
  double _invDt;
  public:
  InvDtFunction(double dt) : Function(0){
    _invDt = 1./dt;
  }
  void setDt(double dt){
    _invDt = 1./dt;
  }
  double getDt(){
    return 1./_invDt;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(_invDt);
  }
};

class ImplicitEulerIntegrator
{
  private:
    BFPtr bf;
    Teuchos::RCP<RHSEasy> rhs;
    MeshPtr mesh;
    SolutionPtr solution;
    FunctionPtr invDt;
    double t;
    double dt;
    int timestep;

  protected:
    void writeSolution();

  public:
    SolutionPtr prevTimeFlow;

    ImplicitEulerIntegrator(BFPtr steadyBF, Teuchos::RCP<RHSEasy> steadyRHS, MeshPtr mesh,
        SolutionPtr solution);
    void addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier);
    void runToTime(double T);
    void calcNextTimeStep(double dt);
};
