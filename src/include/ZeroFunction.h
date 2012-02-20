#ifndef ZERO_FUNCTION
#define ZERO_FUNCTION

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class ZeroFunction : public AbstractFunction {
public:    
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    functionValues.resize(numCells,numPoints);
    functionValues.initialize(0.0);
  }

};

#endif
