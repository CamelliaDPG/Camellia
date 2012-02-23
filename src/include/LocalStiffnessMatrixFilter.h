#ifndef DPG_LOCAL_STIFFNESS_MATRIX_FILTER
#define DPG_LOCAL_STIFFNESS_MATRIX_FILTER

#include "Mesh.h" // contains bilinear form and boundary too
#include "BC.h" 
#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;
using namespace std;

class LocalStiffnessMatrixFilter {
public:
  LocalStiffnessMatrixFilter() {}
  virtual void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector, const FieldContainer<double> &physicalCellNodes,
                      vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) = 0;
    // localStiffnessMatrix has dimensions (numCells, numTrialDofs, numTrialDofs)
    // physicalCellNodes has dimensions (numCells, numVerticesPerCell, numDimensions)

};

#endif
