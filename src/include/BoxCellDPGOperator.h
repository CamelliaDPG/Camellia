#ifndef Camellia_BoxCellDPGOperator_h
#define Camellia_BoxCellDPGOperator_h

//
//  BoxCellDPGOperator.h
//  Camellia
//
//  Matrix-free, sum-factorized element actions of a DPG bilinear form on axis-aligned box cells: B_K, B_K^T, and the
//  local DPG operator S_K = B_K^T G_K^{-1} B_K with G_K^{-1} from SeparableTensorIP's fast diagonalization.  See
//  notes/separable-gram-fast-diagonalization.md.
//

#include "TypeDefs.h"

#include "BF.h"
#include "RHS.h"
#include "SeparableTensorIP.h"
#include "TensorBasisDecomposition.h"

#include <map>
#include <vector>

namespace Camellia
{
  /**
   \brief Applies B_K, B_K^T and B_K^T G_K^{-1} B_K on box cells without forming any element matrix.

   Supported bilinear forms: every term pairs a single trial variable's value with a single test variable's value or
   first derivative, with coefficients that are products of the two sides' Function coefficients:
     - field (volume) terms  (c(x) u, v) or (c(x) u, d_k v): c may vary arbitrarily on the cell (evaluated at the
       tensor quadrature points);
     - flux terms  <c t, v> on the cell boundary: c constant.
   The test space must be a single variable with a full tensor-product basis; trial bases may be tensor-product or
   serendipity (embedded, with zeros, in the tensor-product layout).  Cells must be axis-aligned boxes.

   B_K is applied by sum factorization on a tensor Gauss-Legendre rule with q_k = n_k + ceil(enrichment / 2) points in
   direction k (n_k the test basis's one-dimensional cardinality), which integrates trial x test x (polynomial
   coefficient of degree <= enrichment + 1) exactly: four one-dimensional contractions per direction and term group,
   O(q^{d+1}) per cell instead of O(n_test n_trial q^d).

   Test-space vectors are in the tensor layout of the test basis (first direction fastest), the layout of
   SeparableGramInverse; testDofIndex() maps it to the element's test dof ordering.  Trial vectors are in the element's
   trial dof ordering.
   */
  class BoxCellDPGOperator
  {
  public:
    BoxCellDPGOperator(BFPtr bf, Teuchos::RCP<SeparableTensorIP> ip, MeshPtr mesh, int quadratureEnrichment = 2);

    //! whether the cell's element type and geometry are supported (false: the reason is in reason, if given)
    bool supports(GlobalIndexType cellID, std::string *reason = nullptr);

    //! the number of trial dofs of the cell, and of test functions (the tensor layout's size)
    int trialDofCount(GlobalIndexType cellID);
    int testSize(GlobalIndexType cellID);
    //! [tensor index] -> test dof index in the element's test ordering
    const std::vector<int> &testDofIndex(GlobalIndexType cellID);

    //! Bu (test, tensor layout) = B_K u (u: trial coefficients in the trial dof ordering)
    void applyB(GlobalIndexType cellID, const double *u, double *Bu);
    //! BTv (trial dof ordering) = B_K^T v (v: test, tensor layout)
    void applyBTranspose(GlobalIndexType cellID, const double *v, double *BTv);
    //! Su = B_K^T G_K^{-1} B_K u (trial dof ordering)
    void applySchur(GlobalIndexType cellID, const double *u, double *Su);
    //! v <- G_K^{-1} v (test, tensor layout)
    void applyInverseGram(GlobalIndexType cellID, double *v);
    //! load (trial dof ordering) = B_K^T G_K^{-1} l_K, l_K the right-hand side tested against the test basis (by the
    //! element's standard quadrature, enriched by the operator's quadratureEnrichment)
    void applyLoad(GlobalIndexType cellID, RHSPtr rhs, double *load);

    //! forget cached per-cell data (coefficient values at the quadrature points): call when coefficients change, e.g.
    //! the fields in a fixed-point iteration
    void clearCellData() { _cellData.clear(); }

  private:
    struct FieldGroup // the volume terms of one field (trial) variable against one test operator
    {
      int trialVarID;
      int testDirection; // -1: value
      std::vector<FunctionPtr> coefficients; // summed
    };
    struct FluxTerm
    {
      int fluxVarID;
      double coefficient;
    };
    struct SideData
    {
      int side;
      int normal;                 // the cell direction normal to the side
      int endpoint;               // 0: the lower endpoint (-1), 1: the upper (+1)
      std::vector<int> cellDirection; // [side coordinate]: the cell direction it runs along
      std::vector<int> sign;          // [side coordinate]: +1 or -1 (side coordinate = sign * cell coordinate)
    };
    struct FluxSideData
    {
      int fluxVarID, side;
      double coefficient;
      TensorBasisDecomposition basis;
      std::vector<std::vector<double>> values; // [side coordinate]: (q of its cell direction) x m, at sign * points
      std::vector<int> dofIndex;               // [basis ordinal]
    };
    struct FieldData
    {
      int trialVarID;
      TensorBasisDecomposition basis;
      std::vector<std::vector<double>> values; // [direction]: q_k x m_k
      std::vector<int> dofIndex;               // [basis ordinal]
    };
    struct TypeData
    {
      bool supported = false;
      std::string reason;
      int dim = 0;
      int testVarID = -1;
      TensorBasisDecomposition testBasis;
      std::vector<int> testDofIndex; // [tensor index]
      std::vector<std::vector<double>> points, weights; // [direction]
      std::vector<std::vector<double>> testValues, testDerivatives; // [direction]: q_k x n_k
      std::vector<std::vector<double>> testEndpoints; // [direction]: 2 x n_k (values at -1, then +1)
      std::vector<FieldData> fields;
      std::vector<FieldGroup> groups;
      std::vector<SideData> sides;
      std::vector<FluxSideData> fluxSides;
      int trialDofCount = 0;
      Camellia::FieldContainer<double> refPoints; // (P, dim), first direction fastest
    };
    struct CellData
    {
      std::vector<double> halfWidths;
      std::vector<double> sideParities;
      std::vector<std::vector<double>> groupCoefficients; // [group]: P values (coefficients times weights, Jacobian, derivative scaling)
    };

    BFPtr _bf;
    Teuchos::RCP<SeparableTensorIP> _ip;
    MeshPtr _mesh;
    int _quadratureEnrichment;
    std::map<const ElementType*, TypeData> _typeData;
    std::map<GlobalIndexType, CellData> _cellData;

    TypeData &typeData(GlobalIndexType cellID);
    void initializeTypeData(TypeData &data, ElementTypePtr elemType);
    CellData &cellData(GlobalIndexType cellID, const TypeData &type);
  };
}

#endif
