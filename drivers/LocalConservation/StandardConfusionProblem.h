#ifndef STANDARDCONFUSIONPROBLEM_H
#define STANDARDCONFUSIONPROBLEM_H

#include "ConfusionProblem.h"

class StandardConfusionProblem : public ConfusionProblem
{
  public:
    void init()
    {
      ConfusionProblem::init(1e-2, 2);
    }

    void defineInnerProduct(vector<double> beta);
    void defineBoundaryConditions();
    void defineMesh();
    void runProblem(int argc, char *argv[]);

  protected:
    vector<double> beta;
};

#endif /* end of include guard: STANDARDCONFUSIONPROBLEM_H */
