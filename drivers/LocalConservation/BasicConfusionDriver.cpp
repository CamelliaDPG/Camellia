#include "BasicConfusionProblem.h"

int main(int argc, char *argv[]) {
  StandardConfusionProblem confusionProb;
  confusionProb.init();
  confusionProb.runProblem(argc, argv);
};
