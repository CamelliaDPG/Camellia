#include "Teuchos_UnitTestHarness.hpp"

#include "BoxForest.h"

#include <random>
#include <set>

using namespace Camellia;

namespace
{
  int interiorPatchCount(const BoxForest::Skeleton &skeleton)
  {
    int count = 0;
    for (const auto &patch : skeleton.patches) if (!patch.isBoundary()) count++;
    return count;
  }

  double legendre(int i, double x)
  {
    double P0 = 1, P1 = x;
    if (i == 0) return P0;
    for (int k=2; k<=i; k++)
    {
      double P2 = ((2*k-1) * x * P1 - (k-1) * P0) / k;
      P0 = P1;
      P1 = P2;
    }
    return P1;
  }

  std::set<std::vector<BoxForest::Coord>> activeBoxes(const BoxForest &forest)
  {
    std::set<std::vector<BoxForest::Coord>> boxes;
    for (int n : forest.leaves())
    {
      std::vector<BoxForest::Coord> box;
      for (int k=0; k<forest.dimension(); k++)
      {
        box.push_back(forest.node(n).box.lo[k]);
        box.push_back(forest.node(n).box.hi(k));
      }
      boxes.insert(box);
    }
    return boxes;
  }

TEUCHOS_UNIT_TEST( BoxForest, ChildrenTileParent )
{
  // all 2^d - 1 direction subsets in d = 1..4: children are disjoint sub-boxes whose volumes sum to the parent's
  for (int d=1; d<=4; d++)
  {
    for (unsigned mask=1; mask < (1u << d); mask++)
    {
      BoxForest forest(std::vector<int>(d, 1));
      forest.refine(0, mask);
      TEST_EQUALITY(int(forest.leaves().size()), 1 << __builtin_popcount(mask));
      TEST_EQUALITY(activeBoxes(forest).size(), forest.leaves().size());
      double volume = 0;
      for (int n : forest.leaves())
      {
        double v = 1;
        for (int k=0; k<d; k++)
        {
          const int expectedLevel = (mask >> k) & 1;
          TEST_EQUALITY(int(forest.node(n).box.level[k]), expectedLevel);
          v *= std::ldexp(1.0, -forest.node(n).box.level[k]);
        }
        volume += v;
      }
      TEST_FLOATING_EQUALITY(volume, 1.0, 1e-15);
    }
  }
}

TEUCHOS_UNIT_TEST( BoxForest, RefinementOrderDoesNotMatter )
{
  BoxForest xThenY({1,1}), yThenX({1,1}), isotropic({1,1});
  int child = xThenY.refine(0, 1);
  xThenY.refine(child, 2);
  xThenY.refine(child+1, 2);
  child = yThenX.refine(0, 2);
  yThenX.refine(child, 1);
  yThenX.refine(child+1, 1);
  isotropic.refine(0, 3);
  TEST_ASSERT(activeBoxes(xThenY) == activeBoxes(yThenX));
  TEST_ASSERT(activeBoxes(xThenY) == activeBoxes(isotropic));
  TEST_EQUALITY(xThenY.skeleton().patches.size(), isotropic.skeleton().patches.size());
  TEST_EQUALITY(yThenX.skeleton().patches.size(), isotropic.skeleton().patches.size());
}

TEUCHOS_UNIT_TEST( BoxForest, Quad_xNextTo_y )
{
  // two quads side by side: left bisected in x, right bisected in y
  BoxForest forest({2,1});
  int left = forest.refine(0, 1), right = forest.refine(1, 2);
  auto skeleton = forest.skeleton();
  TEST_EQUALITY(forest.leaves().size(), 4);
  // interior patches: between the left children, between the right children, and the whole left|right interface
  TEST_EQUALITY(interiorPatchCount(skeleton), 3);
  TEST_EQUALITY(skeleton.patches.size(), 12); // 9 boundary faces
  for (const auto &patch : skeleton.patches) TEST_ASSERT(patch.isBox);

  // the right face of the left cell's right child is constrained by nothing; the right cell's children hang on it
  auto neighbors = forest.faceNeighbors(left+1, 1);
  TEST_EQUALITY(neighbors.size(), 2);
  int patch = skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(left+1), 1)];
  TEST_EQUALITY(patch, skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(right), 0)]);
  TEST_EQUALITY(patch, skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(right+1), 0)]);
  TEST_EQUALITY(skeleton.patches[patch].minusFaces.size(), 1);
  TEST_EQUALITY(skeleton.patches[patch].plusFaces.size(), 2);
}

TEUCHOS_UNIT_TEST( BoxForest, Hex_yNextTo_z_ConstraintIsExact )
{
  // two hexes sharing an x face: left bisected in y, right bisected in z.  The minimum-rule trace on the
  // interface is one polynomial on the whole face, which is no active cell's face.
  BoxForest forest({2,1,1});
  int left = forest.refine(0, 2), right = forest.refine(1, 4);
  auto skeleton = forest.skeleton();
  const int patchOrdinal = skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(left), 1)];
  TEST_EQUALITY(patchOrdinal, skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(left+1), 1)]);
  TEST_EQUALITY(patchOrdinal, skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(right), 0)]);
  TEST_EQUALITY(patchOrdinal, skeleton.patchForFace[forest.faceOrdinal(forest.leafOrdinal(right+1), 0)]);
  const auto &patch = skeleton.patches[patchOrdinal];
  TEST_EQUALITY(patch.minusFaces.size(), 2);
  TEST_EQUALITY(patch.plusFaces.size(), 2);
  TEST_ASSERT(patch.isBox);
  TEST_EQUALITY(interiorPatchCount(skeleton), 3);

  // cell-face Legendre coefficients = (R_y kron R_z) * patch coefficients; compare pointwise
  const int p = 3, n = p + 1;
  std::mt19937 gen(1);
  std::uniform_real_distribution<double> uniform(-1, 1);
  std::vector<double> patchCoefficients(n * n);
  for (auto &c : patchCoefficients) c = uniform(gen);

  double maxError = 0;
  for (const auto *faces : {&patch.minusFaces, &patch.plusFaces})
  {
    for (const auto &face : *faces)
    {
      const auto &box = forest.node(face.cell).box;
      double a[3], b[3]; // the face's extent in the patch's reference coordinates
      for (int k=1; k<3; k++)
      {
        const double width = double(patch.hi[k] - patch.lo[k]);
        a[k] = -1 + 2 * double(box.lo[k] - patch.lo[k]) / width;
        b[k] = -1 + 2 * double(box.hi(k) - patch.lo[k]) / width;
      }
      auto Ry = BoxForest::legendreRestriction(p, a[1], b[1]);
      auto Rz = BoxForest::legendreRestriction(p, a[2], b[2]);
      std::vector<double> faceCoefficients(n * n, 0.0);
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
          for (int k=0; k<n; k++)
            for (int l=0; l<n; l++)
              faceCoefficients[i*n+j] += Ry[i*n+k] * Rz[j*n+l] * patchCoefficients[k*n+l];
      for (int trial=0; trial<20; trial++)
      {
        const double y = uniform(gen), z = uniform(gen); // reference point on the cell face
        const double Y = a[1] + (b[1] - a[1]) * (y + 1) / 2, Z = a[2] + (b[2] - a[2]) * (z + 1) / 2;
        double fineValue = 0, patchValue = 0;
        for (int i=0; i<n; i++)
        {
          for (int j=0; j<n; j++)
          {
            fineValue  += faceCoefficients[i*n+j]  * legendre(i, y) * legendre(j, z);
            patchValue += patchCoefficients[i*n+j] * legendre(i, Y) * legendre(j, Z);
          }
        }
        maxError = std::max(maxError, std::abs(fineValue - patchValue));
      }
    }
  }
  TEST_COMPARE(maxError, <, 1e-13);
}

TEUCHOS_UNIT_TEST( BoxForest, NonBoxPatchIsDetected )
{
  BoxForest forest({2,1,1}); // interface x = 1, tangential directions (y,z)
  int left = forest.refine(0, 4);   // minus side: z halves,
  forest.refine(left+1, 2);         //   upper one bisected in y
  int right = forest.refine(1, 2);  // plus side: y halves,
  forest.refine(right+1, 4);        //   upper one bisected in z
  auto skeleton = forest.skeleton();
  int nonBoxCount = 0;
  for (const auto &patch : skeleton.patches)
  {
    if (!patch.isBox)
    {
      nonBoxCount++;
      TEST_EQUALITY(patch.normal, 0);
      TEST_EQUALITY(patch.minusFaces.size(), 2); // an L: three quarters of the face
      TEST_EQUALITY(patch.plusFaces.size(), 2);
    }
  }
  TEST_EQUALITY(nonBoxCount, 1);
}

TEUCHOS_UNIT_TEST( BoxForest, Tesseract_VelocityOnlyRefinement )
{
  // (x, t, u_x, u_z) mesh, 2 x 1 x 2 x 2 roots; bisect the cells with x in the first root in the velocity directions only
  BoxForest forest({2,1,2,2});
  for (int root : {0, 2, 4, 6}) forest.refine(root, 4 | 8);
  TEST_EQUALITY(forest.leaves().size(), 4 * 4 + 4);
  auto skeleton = forest.skeleton();
  // across x = 1: each unrefined cell's face is one patch, with the four refined neighbours hanging on it
  int hangingPatches = 0;
  for (const auto &patch : skeleton.patches)
  {
    TEST_ASSERT(patch.isBox);
    if ((patch.normal == 0) && !patch.isBoundary())
    {
      hangingPatches++;
      TEST_EQUALITY(patch.minusFaces.size(), 4);
      TEST_EQUALITY(patch.plusFaces.size(), 1);
    }
  }
  TEST_EQUALITY(hangingPatches, 4);
  // volume is preserved
  double volume = 0;
  for (int n : forest.leaves())
  {
    double v = 1;
    for (int k=0; k<4; k++) v *= std::ldexp(1.0, -forest.node(n).box.level[k]);
    volume += v;
  }
  TEST_FLOATING_EQUALITY(volume, 8.0, 1e-15);
}
} // namespace
