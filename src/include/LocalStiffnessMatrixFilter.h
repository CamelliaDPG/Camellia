#ifndef DPG_LOCAL_STIFFNESS_MATRIX_FILTER
#define DPG_LOCAL_STIFFNESS_MATRIX_FILTER

#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;

class LocalStiffnessMatrixFilter {
public:
  virtual void filter(FieldContainer<double> &localStiffnessMatrix, const FieldContainer<double> &physicalCellNodes) {
    // localStiffnessMatrix has dimensions (numCells, numTrialDofs, numTrialDofs)
    // physicalCellNodes has dimensions (numCells, numVerticesPerCell, numDimensions)
    
    // default implementation doesn't do anything -- override this method to actually filter...
  }
};

#endif