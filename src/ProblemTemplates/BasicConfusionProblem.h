#ifndef STANDARDCONFUSIONPROBLEM_H
#define STANDARDCONFUSIONPROBLEM_H

#include "ConfusionProblem.h"

class BasicConfusionProblem : public ConfusionProblem
{
  public:
    // vector<double> getBeta()
    // {
    //   return beta;
    // }
    // void setBeta(double beta_x, double beta_y)
    // {
    //   beta.push_back(beta_x);
    //   beta.push_back(beta_y);
    // }

    void defineInnerProduct(vector<double> beta);
    void defineBoundaryConditions();
    void defineMesh();
    void runProblem(int argc, char *argv[]);

    // Making this public for easier introspection
  public:
    vector<double> beta;
};

#endif /* end of include guard: STANDARDCONFUSIONPROBLEM_H */
