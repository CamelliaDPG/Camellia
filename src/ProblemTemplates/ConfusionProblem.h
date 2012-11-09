#ifndef CONFUSIONPROBLEM_H
#define CONFUSIONPROBLEM_H

#include "InnerProductScratchPad.h"

#include <Teuchos_Tuple.hpp>

class ConfusionProblem
{
  public:
    ConfusionProblem() : epsilon(1e-2), numRefs(0), 
      H1Order(3), pToAdd(2), checkLocalConservation(false), 
      printLocalConservation(false), enforceLocalConservation(false) { }

    void defineVariables();
    void defineBilinearForm(vector<double> beta);
    void defineBilinearForm(FunctionPtr beta);
    virtual void defineInnerProduct(vector<double> beta);
    virtual void defineInnerProduct(FunctionPtr beta);
    virtual void defineRightHandSide();
    virtual void defineBoundaryConditions() = 0;
    virtual void defineMesh() = 0;
    virtual void solveSteady(int argc, char *argv[], string filename="", double energyThreshold=0.2);
    virtual void runProblem(int argc, char *argv[]) = 0;
    virtual Teuchos::Tuple<double, 3> checkConservation(FunctionPtr flux, FunctionPtr source);
    // Functions to swap inner product (defaults to graph norm)
    void setMathIP();
    void setRobustIP(vector<double> beta);
    void setRobustIP(FunctionPtr beta);
    void setRobustZeroMeanIP(vector<double> beta);
    void setRobustZeroMeanIP(FunctionPtr beta);

    // Making these public for easier introspection
  public:
    double epsilon;
    int numRefs;

    int H1Order;
    int pToAdd;

    bool checkLocalConservation;
    bool printLocalConservation;

    bool enforceLocalConservation;

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

    Teuchos::Tuple<double, 3> fluxImbalances;

    BFPtr confusionBF;

    Teuchos::RCP<RHSEasy> rhs;

    Teuchos::RCP<BCEasy> bc;

    IPPtr ip;

    Teuchos::RCP<Mesh> mesh;

    Teuchos::RCP<Solution> solution;
};

#endif /* end of include guard: CONFUSIONPROBLEM_H */
