#ifndef SIMPLE_FUNCTION_INTERFACE
#define SIMPLE_FUNCTION_INTERFACE

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class SimpleFunction : public AbstractFunction {
public:    
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    functionValues.resize(numCells,numPoints,spaceDim);
    for (int i=0;i<numCells;i++){
      for (int j=0;j<numPoints;j++){
        double x = physicalPoints(i,j,0);
        functionValues(i,j) = x;
      }
    }  
  }

};

#endif
