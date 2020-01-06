// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef DPG_LOCAL_STIFFNESS_MATRIX_FILTER
#define DPG_LOCAL_STIFFNESS_MATRIX_FILTER

#include "TypeDefs.h"

#include "Mesh.h" // contains bilinear form and boundary too
#include "BC.h"
#include "Camellia_FieldContainer.hpp"

using namespace std;

namespace Camellia
{
class LocalStiffnessMatrixFilter
{
public:
  LocalStiffnessMatrixFilter() {}
  virtual void filter(Camellia::FieldContainer<double> &localStiffnessMatrix, Camellia::FieldContainer<double> &localRHSVector,
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"LocalStiffnessMatrixFilter::filter() unimplemented.");
  }

  //  virtual void filter(Camellia::FieldContainer<double> &localStiffnessMatrix, Camellia::FieldContainer<double> &localRHSVector,
  //                      const Camellia::FieldContainer<double> &physicalCellNodes,
  //                      vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) = 0;
  // localStiffnessMatrix has dimensions (numCells, numTrialDofs, numTrialDofs)
  // physicalCellNodes has dimensions (numCells, numVerticesPerCell, numDimensions)

  virtual ~LocalStiffnessMatrixFilter() {}
};
}

#endif
