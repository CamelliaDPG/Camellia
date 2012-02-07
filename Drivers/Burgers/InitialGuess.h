#ifndef BURGERS_INITIAL_GUESS
#define BURGERS_INITIAL_GUESS

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class InitialGuess : public AbstractFunction {
public:    
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    functionValues.resize(numCells,numPoints);
    for (int i=0;i<numCells;i++){
      for (int j=0;j<numPoints;j++){
        double x = physicalPoints(i,j,0);
        double y = physicalPoints(i,j,1);
	functionValues(i,j) = 1.0-2.0*x; // extrapolate the boundary condition
	//	functionValues(i,j) = 1.0; // extrapolate the boundary condition
      }
    }  
  }

};

#endif
