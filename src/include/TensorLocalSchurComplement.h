#ifndef Camellia_TensorLocalSchurComplement_h
#define Camellia_TensorLocalSchurComplement_h

#include <string>
#include <vector>

#include "TensorBasisCache.h"
#include "TensorBilinearForm.h"
#include "TensorOperatorData.h"
#include "TensorProductLocalOperator.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
   * \brief Host-side Phase 0 helper for explicit local B/G/K block construction.
   *
   * TensorProductLocalOperator keeps a separable local B action in factorized form.
   * These utilities materialize the corresponding dense local blocks only at the
   * correctness/assembly boundary: local B as a row-major test-by-trial matrix,
   * diagonal or mass-like G entries as a tensor product of one-dimensional diagonal
   * factors, and the trial-space Schur complement K = B^T G^{-1} B for the current
   * local-coefficient convention.
   */
  class TensorLocalSchurComplementBuilder
  {
  public:
    static TensorProductLocalOperator buildVolumeBOperator(const TensorBilinearTerm &term,
                                                           const TensorFactorSpaceShape &trialShape,
                                                           const TensorFactorSpaceShape &testShape,
                                                           const TensorBasisCacheND &trialBasisCache,
                                                           const TensorBasisCacheND &testBasisCache,
                                                           const std::vector<TensorFunctionPtr> &coefficientFactors);

    struct LocalVolumeSchurComplementBlock
    {
      TensorMeshTopology::TensorCellTuple cellTuple;
      TensorFactorSpaceShape trialShape;
      TensorFactorSpaceShape testShape;
      std::vector<double> bMatrix;
      std::vector<double> gDiagonal;
      std::vector<double> gMatrix;
      std::vector<double> kMatrix;
    };

    struct LocalBoundaryBBlock
    {
      TensorMeshTopology::TensorSkeletonEntity entity;
      IndexType entityOrdinal = -1;
      TensorFactorSpaceShape trialShape;
      TensorFactorSpaceShape testShape;
      std::vector<double> bMatrix;
    };

    static std::vector<double> denseBMatrix(const TensorProductLocalOperator &bOperator);

    static std::vector<std::vector<double>> l2TestMassDiagonalFactors(const TensorBasisCacheND &testBasisCache);

    static std::vector<double> l2TestMassDiagonal(const TensorFactorSpaceShape &testShape,
                                                  const TensorBasisCacheND &testBasisCache);

    static std::vector<double> graphTestNormMatrix(const TensorFactorSpaceShape &testShape,
                                                   const TensorBasisCacheND &testBasisCache,
                                                   double l2Weight = 1.0);

    struct LocalGraphNormAction
    {
    private:
      IndexType _testDofCount = 0;
      std::vector<double> _gMatrix;
      std::vector<double> _gFactorization;
      std::vector<IndexType> _pivotRows;

      void factorGraphNormMatrix();

    public:
      LocalGraphNormAction(IndexType testDofCount,
                           const std::vector<double> &gMatrix);

      IndexType testDofCount() const;
      const std::vector<double> &gMatrix() const;

      std::vector<double> applyInverse(const std::vector<double> &rhs) const;
      std::vector<double> applyInverseToColumns(const std::vector<double> &inputMatrix,
                                                IndexType columnCount) const;
      std::vector<double> applySchurComplement(const std::vector<double> &bMatrix,
                                               IndexType trialDofCount,
                                               const std::vector<double> &trialVector) const;
    };

    static LocalGraphNormAction localGraphNormAction(IndexType testDofCount,
                                                     const std::vector<double> &gMatrix);

    static std::vector<double> applyAffineGraphVolumeSchurComplement(const TensorMeshPtr &tensorMesh,
                                                                     const TensorDofLayoutPtr &tensorDofLayout,
                                                                     const TensorBilinearFormPtr &bilinearForm,
                                                                     const std::string &trialVarName,
                                                                     const std::string &testVarName,
                                                                     const std::vector<double> &trialVector,
                                                                     double l2Weight = 1.0);

    static std::vector<double> applyAffineGraphVolumeSchurComplement(const TensorMeshPtr &tensorMesh,
                                                                     const TensorDofLayoutPtr &tensorDofLayout,
                                                                     const TensorBilinearFormPtr &bilinearForm,
                                                                     const std::vector<std::string> &trialVarNames,
                                                                     const std::vector<std::string> &testVarNames,
                                                                     const std::vector<double> &trialVector,
                                                                     double l2Weight = 1.0);

    static std::vector<double> applyAffineGraphVolumeBoundarySchurComplement(const TensorMeshPtr &tensorMesh,
                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                             const std::vector<std::string> &trialVarNames,
                                                                             const std::vector<std::string> &testVarNames,
                                                                             const std::vector<double> &trialVector,
                                                                             double l2Weight = 1.0);

    static std::vector<double> tensorProductDiagonal(const TensorFactorSpaceShape &shape,
                                                     const std::vector<std::vector<double>> &componentDiagonals);

    static std::vector<double> formDiagonalTestNormSchurComplement(const TensorProductLocalOperator &bOperator,
                                                                   const std::vector<double> &gDiagonal);

    static std::vector<double> formDiagonalTestNormSchurComplement(const std::vector<double> &bMatrix,
                                                                   IndexType testDofCount,
                                                                   IndexType trialDofCount,
                                                                   const std::vector<double> &gDiagonal);

    static std::vector<double> formDenseTestNormSchurComplement(const std::vector<double> &bMatrix,
                                                                IndexType testDofCount,
                                                                IndexType trialDofCount,
                                                                const std::vector<double> &gMatrix);

    static std::vector<LocalVolumeSchurComplementBlock> affineL2VolumeCellSchurComplements(const TensorMeshPtr &tensorMesh,
                                                                                           const TensorDofLayoutPtr &tensorDofLayout,
                                                                                           const TensorBilinearFormPtr &bilinearForm,
                                                                                           const std::string &trialVarName,
                                                                                           const std::string &testVarName);

    static std::vector<LocalVolumeSchurComplementBlock> affineGraphVolumeCellSchurComplements(const TensorMeshPtr &tensorMesh,
                                                                                              const TensorDofLayoutPtr &tensorDofLayout,
                                                                                              const TensorBilinearFormPtr &bilinearForm,
                                                                                              const std::string &trialVarName,
                                                                                              const std::string &testVarName,
                                                                                              double l2Weight = 1.0);

    static std::vector<LocalBoundaryBBlock> affineBoundaryBBlocks(const TensorMeshPtr &tensorMesh,
                                                                 const TensorDofLayoutPtr &tensorDofLayout,
                                                                 const TensorBilinearFormPtr &bilinearForm,
                                                                 const std::string &trialVarName,
                                                                 const std::string &testVarName);

    static TMatrixPtr<double> affineL2VolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                  const TensorDofLayoutPtr &tensorDofLayout,
                                                                  const TensorBilinearFormPtr &bilinearForm,
                                                                  const std::string &trialVarName,
                                                                  const std::string &testVarName,
                                                                  Teuchos_CommPtr comm = Teuchos::null);

    static TMatrixPtr<double> affineGraphVolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                     const TensorDofLayoutPtr &tensorDofLayout,
                                                                     const TensorBilinearFormPtr &bilinearForm,
                                                                     const std::string &trialVarName,
                                                                     const std::string &testVarName,
                                                                     double l2Weight = 1.0,
                                                                     Teuchos_CommPtr comm = Teuchos::null);

    static TMatrixPtr<double> affineL2VolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                  const TensorDofLayoutPtr &tensorDofLayout,
                                                                  const TensorBilinearFormPtr &bilinearForm,
                                                                  const std::vector<std::string> &trialVarNames,
                                                                  const std::vector<std::string> &testVarNames,
                                                                  Teuchos_CommPtr comm = Teuchos::null);

    static TMatrixPtr<double> affineGraphVolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                     const TensorDofLayoutPtr &tensorDofLayout,
                                                                     const TensorBilinearFormPtr &bilinearForm,
                                                                     const std::vector<std::string> &trialVarNames,
                                                                     const std::vector<std::string> &testVarNames,
                                                                     double l2Weight = 1.0,
                                                                     Teuchos_CommPtr comm = Teuchos::null);

    static TMatrixPtr<double> affineL2VolumeBoundarySchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                          const TensorDofLayoutPtr &tensorDofLayout,
                                                                          const TensorBilinearFormPtr &bilinearForm,
                                                                          const std::vector<std::string> &trialVarNames,
                                                                          const std::vector<std::string> &testVarNames,
                                                                          Teuchos_CommPtr comm = Teuchos::null);

    static TMatrixPtr<double> affineGraphVolumeBoundarySchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                             const std::vector<std::string> &trialVarNames,
                                                                             const std::vector<std::string> &testVarNames,
                                                                             double l2Weight = 1.0,
                                                                             Teuchos_CommPtr comm = Teuchos::null);
  };
}

#endif
