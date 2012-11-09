#include "BasicConfusionProblem.h"

int main(int argc, char *argv[]) {
  for (int i=0; i < argc; i++)
    cout << argv[i] << endl;
  BasicConfusionProblem confusionProb;
  confusionProb.init();
  confusionProb.runProblem(argc, argv);
};
