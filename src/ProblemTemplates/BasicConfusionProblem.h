#ifndef STANDARDCONFUSIONPROBLEM_H
#define STANDARDCONFUSIONPROBLEM_H

#include "ConfusionProblem.h"

class BasicConfusionProblem : public ConfusionProblem
{
  public:
    void defineInnerProduct(vector<double> beta);
    void defineBoundaryConditions();
    void defineMesh();
    void runProblem(int argc, char *argv[]);

    // Making this public for easier introspection
  public:
    vector<double> beta;
};

#endif /* end of include guard: STANDARDCONFUSIONPROBLEM_H */
