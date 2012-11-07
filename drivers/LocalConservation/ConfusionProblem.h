#ifndef CONFUSIONPROBLEM_H
#define CONFUSIONPROBLEM_H

#include "InnerProductScratchPad.h"

#include <Teuchos_Tuple.hpp>

class EpsilonScaling : public hFunction {
  double _epsilon;
  public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class ConfusionProblem
{
  public:
    void init(double _epsilon = 1.0, int _numRefs = 0, int _H1Order = 3, int _pToAdd = 2)
    {
      epsilon = _epsilon;
      numRefs = _numRefs;
      H1Order = _H1Order;
      pToAdd  = _pToAdd;
    }
    void defineVariables();
    void defineBilinearForm(vector<double> beta);
    void defineBilinearForm(FunctionPtr beta);
    virtual void defineInnerProduct(vector<double> beta);
    virtual void defineInnerProduct(FunctionPtr beta);
    virtual void defineRightHandSide();
    virtual void defineBoundaryConditions() = 0;
    virtual void defineMesh() = 0;
    virtual void runProblem(int argc, char *argv[]) = 0;
    virtual void checkConservation(FunctionPtr flux, FunctionPtr source);
    // Functions to swap inner product (defaults to graph norm)
    void setMathIP();
    void setRobustIP(vector<double> beta);
    void setRobustIP(FunctionPtr beta);
    void setRobustZeroMeanIP(vector<double> beta);
    void setRobustZeroMeanIP(FunctionPtr beta);

  protected:
    double epsilon;
    int numRefs;

    int H1Order;
    int pToAdd;

    // define test variables
    VarFactory varFactory; 
    VarPtr tau;
    VarPtr v;
    // define trial variables
    VarPtr uhat;
    VarPtr beta_n_u_minus_sigma_n;
    VarPtr u;
    VarPtr sigma1;
    VarPtr sigma2;

    BFPtr confusionBF;

    Teuchos::RCP<RHSEasy> rhs;

    Teuchos::RCP<BCEasy> bc;

    IPPtr ip;

    Teuchos::RCP<Mesh> mesh;

    Teuchos::RCP<Solution> solution;
};

#endif /* end of include guard: CONFUSIONPROBLEM_H */
