//
//  CellCharacteristicFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 7/24/13.
//
//

#ifndef Camellia_debug_CellCharacteristicFunction_h
#define Camellia_debug_CellCharacteristicFunction_h

#include "Function.h"

using namespace std;

namespace Camellia {
  class CellCharacteristicFunction : public Function<double> {
    set<GlobalIndexType> _cellIDs;
  public:
    CellCharacteristicFunction(GlobalIndexType cellID) : Function<double>(0) {
      _cellIDs.insert(cellID);
    }

    CellCharacteristicFunction(set<GlobalIndexType> cellIDs) : Function<double>(0) {
      _cellIDs = cellIDs;
    }
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      CHECK_VALUES_RANK(values);
      vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      values.initialize(0);

      if (numCells != cellIDs.size()) {
        cout << "ERROR: CellCharacteristicFunction requires cellIDs to be defined in BasisCache\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CellCharacteristicFunction requires cellIDs to be defined in BasisCache");
      }
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        GlobalIndexType cellID = cellIDs[cellIndex];
        if (_cellIDs.find(cellID) != _cellIDs.end()) {
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            values(cellIndex,ptIndex) = 1;
          }
        }
      }
    }
  };
}

#endif
