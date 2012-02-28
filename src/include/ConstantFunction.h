#ifndef CONSTANT_FUNCTION
#define CONSTANT_FUNCTION

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class ConstantFunction : public AbstractFunction {
 private:
  double _constantValue;
 public: 
  ConstantFunction(double constantValue){
    _constantValue = constantValue;
  }

  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    functionValues.resize(numCells,numPoints);
    functionValues.initialize(_constantValue);
  }

};

#endif
