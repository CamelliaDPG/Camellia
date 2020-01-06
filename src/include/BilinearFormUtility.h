#ifndef BILINEAR_FORM_UTILITY
#define BILINEAR_FORM_UTILITY

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#include "DofOrdering.h"
#include "BF.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "IP.h"
#include "Mesh.h"
#include "RHS.h"
#include "BasisCache.h"

// Shards includes
#include "Shards_CellTopology.hpp"

using namespace std;

namespace Camellia
{
template <typename Scalar>
class BilinearFormUtility
{
private:
  static bool _warnAboutZeroRowsAndColumns;
public:
  // the "pre-stiffness" (rectangular) matrix methods:
  static void computeStiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, TBFPtr<Scalar> bilinearForm,
                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                     CellTopoPtr cellTopo, Camellia::FieldContainer<double> &physicalCellNodes,
                                     Camellia::FieldContainer<double> &cellSideParities);

  // the real one:
  //  static void computeStiffnessMatrix(Camellia::FieldContainer<double> &stiffness, BFPtr bilinearForm,
  //                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
  //                                     Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);

  static void computeStiffnessMatrixForCell(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<Mesh> mesh, int cellID);

  static void computeStiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Camellia::FieldContainer<Scalar> &innerProductMatrix,
                                     Camellia::FieldContainer<Scalar> &optimalTestWeights);

  // this method is deprecated; use the next one
  static void computeRHS(Camellia::FieldContainer<Scalar> &rhsVector, TBFPtr<Scalar> bilinearForm, RHS &rhs,
                         Camellia::FieldContainer<Scalar> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                         shards::CellTopology &cellTopo, Camellia::FieldContainer<double> &physicalCellNodes);

  //  static void computeRHS(Camellia::FieldContainer<double> &rhsVector, BFPtr bilinearForm, RHS &rhs,
  //                  Camellia::FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
  //                  BasisCachePtr basisCache);

  static void transposeFCMatrices(Camellia::FieldContainer<Scalar> &fcTranspose,
                                  const Camellia::FieldContainer<Scalar> &fc);

  static bool checkForZeroRowsAndColumns(string name, Camellia::FieldContainer<Scalar> &array, bool checkRows = true, bool checkCols = true);

  static void weightCellBasisValues(Camellia::FieldContainer<double> &basisValues,
                                    const Camellia::FieldContainer<double> &weights, int offset);

  static void setWarnAboutZeroRowsAndColumns( bool value );
  static bool warnAboutZeroRowsAndColumns();
};

extern template class BilinearFormUtility<double>;
}
#endif
