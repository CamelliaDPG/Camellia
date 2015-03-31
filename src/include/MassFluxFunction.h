//
//  MassFluxFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 7/4/13.
//
//

#ifndef Camellia_debug_MassFluxFunction_h
#define Camellia_debug_MassFluxFunction_h

#include "Function.h"

#include "CamelliaCellTools.h"

// The mass flux function is single-valued within elements.
// the input function is integrated over each element's boundary.
// the main idea is to allow easy visualization of where mass conservation
// is violated in a solution.
class MassFluxFunction : public Function {
  FunctionPtr _un;
  bool _absoluteValue;
public:
  MassFluxFunction(FunctionPtr un, bool takeAbsoluteValue = false) : Function(0) {
    _un = un;
    _absoluteValue = takeAbsoluteValue;
  }
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
    if (basisCache->isSideCache()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MassFluxFunction is only valid on element interiors");
    }
    if (basisCache->getSideBasisCache(0).get() == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MassFluxFunction requires side basis caches to be generated");
    }
    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    MeshPtr mesh = basisCache->mesh();
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    Intrepid::FieldContainer<double> cellIntegrals(numCells);
    
//    Intrepid::FieldContainer<double> cellIntegralsOnSide(numCells); // DEBUGGING
    
    int numSides = CamelliaCellTools::getSideCount(basisCache->cellTopology());
    
    bool sumInto = true;
    for (int sideIndex = 0; sideIndex < numSides; sideIndex++) {
      _un->integrate(cellIntegrals, basisCache->getSideBasisCache(sideIndex), sumInto);
//      _un->integrate(cellIntegralsOnSide, basisCache->getSideBasisCache(sideIndex));  // DEBUGGING
//      // DEBUGGING:
//      for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
//        if (cellIDs[cellIndex]==0) {
//          cout << "MassFluxFunction: on cellID 0, integral of side " << sideIndex << ": " << cellIntegralsOnSide[cellIndex] << endl;
//        }
//      }
    }
    
    Intrepid::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
        
    // weight by cellMeasure (so that the integral of this function makes sense), and take absolute value if requested
    for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
//      cout << "MassFluxFunction: cellIntegral for cellID " << cellIDs[cellIndex] << " = " << cellIntegrals[cellIndex] << endl;
      
      if (! _absoluteValue) {
        cellIntegrals(cellIndex) = cellIntegrals(cellIndex) / cellMeasures(cellIndex);
      } else {
        cellIntegrals(cellIndex) = abs(cellIntegrals(cellIndex) / cellMeasures(cellIndex));
      }
    }
    
    for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
      for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
        values(cellIndex,ptIndex) = cellIntegrals(cellIndex);
      }
    }
  }

};


#endif
