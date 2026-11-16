#ifndef Camellia_BoxForest_h
#define Camellia_BoxForest_h

//
//  BoxForest.h
//  Camellia
//
//  PROTOTYPE.  A light-weight mesh topology for hypercube cells of any dimension with anisotropic
//  (any subset of directions) bisection.  See notes/anisotropic-refinement-design.md.
//
//  The idea: a cell is an axis-aligned dyadic box, identified by integer lower coordinates and a
//  refinement level per direction.  Nothing else is stored: no vertices, edges or faces, no
//  entity-to-cell maps, no refinement patterns or recipes.  Neighbors, hanging faces and the
//  supports of the minimum-rule trace spaces all follow from integer interval arithmetic and a
//  descent of the refinement tree, which is kept only as a spatial index.
//
//  The header depends only on the standard library.
//

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Camellia
{
  class BoxForest
  {
  public:
    static constexpr int MAX_DIM   = 6;
    static constexpr int MAX_LEVEL = 24; // a root cell is 2^MAX_LEVEL units wide in every direction
    using Coord = std::uint64_t;

    //! Half-open integer box [lo, hi) with hi[k] = lo[k] + 2^(MAX_LEVEL - level[k]).
    struct Box
    {
      std::array<Coord,MAX_DIM> lo{};
      std::array<std::uint8_t,MAX_DIM> level{};

      Coord width(int k) const { return Coord(1) << (MAX_LEVEL - level[k]); }
      Coord hi(int k)    const { return lo[k] + width(k); }

      //! The child boxes for a bisection in the directions set in mask.  Does not involve a mesh: usable
      //! for building the children of a candidate refinement without committing it.
      std::vector<Box> children(unsigned mask, int dim) const
      {
        std::vector<int> dirs;
        for (int k=0; k<dim; k++) if (mask & (1u << k)) dirs.push_back(k);
        std::vector<Box> result(std::size_t(1) << dirs.size(), *this);
        for (std::size_t j=0; j<result.size(); j++)
        {
          for (std::size_t i=0; i<dirs.size(); i++)
          {
            int k = dirs[i];
            result[j].level[k] = level[k] + 1;
            if (j & (std::size_t(1) << i)) result[j].lo[k] = lo[k] + width(k) / 2;
          }
        }
        return result;
      }
    };

    struct Node
    {
      Box box;
      int parent = -1;
      int firstChild = -1;  // -1: leaf (active cell)
      std::uint8_t mask = 0; // directions bisected to produce the children
    };

    //! A face of an active cell: side = 2 * direction + (0: lower, 1: upper)
    struct Face
    {
      int cell; // node index
      int side;
    };

    /**
     \brief  The support of one minimum-rule trace polynomial.

     On an interface plane the cells on the two sides each tile the plane with their faces.  A
     single-valued function that is a polynomial on every cell face is one polynomial on each connected
     set of faces (faces being connected when they overlap with positive measure).  A patch is such a
     set.  For isotropic 1-irregular meshes a patch is the coarse neighbor's face; with anisotropic
     neighbors it can be a union that is no active cell's face (e.g. one neighbor bisected in y, the
     other in z), which is the case the "related recipes" exist to describe in MeshTopology.
     */
    struct Patch
    {
      int normal = -1;       // direction orthogonal to the patch
      Coord plane = 0;       // coordinate of the patch in that direction
      std::array<Coord,MAX_DIM> lo{}, hi{}; // bounding box (tangential directions)
      std::vector<Face> minusFaces; // upper faces of the cells below the plane
      std::vector<Face> plusFaces;  // lower faces of the cells above the plane
      bool isBox = true;     // the union of the faces is the bounding box
      bool isBoundary() const { return minusFaces.empty() || plusFaces.empty(); }
    };

    struct Skeleton
    {
      std::vector<Patch> patches;
      std::vector<int> patchForFace; // indexed by faceOrdinal(leafOrdinal, side)
    };

  private:
    int _dim;
    std::vector<int> _rootCounts;
    std::vector<Node> _nodes;
    mutable std::vector<int> _leaves;
    mutable std::vector<int> _leafOrdinals;
    mutable bool _leavesValid = false;

    template<class F>
    void visitOverlapping(int nodeIndex, const std::array<Coord,MAX_DIM> &lo, const std::array<Coord,MAX_DIM> &hi, F &f) const
    {
      const Node &node = _nodes[nodeIndex];
      for (int k=0; k<_dim; k++)
      {
        if ((node.box.lo[k] >= hi[k]) || (node.box.hi(k) <= lo[k])) return;
      }
      if (node.firstChild < 0) { f(nodeIndex); return; }
      const int childCount = 1 << __builtin_popcount(node.mask);
      for (int j=0; j<childCount; j++) visitOverlapping(node.firstChild + j, lo, hi, f);
    }

    void rebuildLeaves() const
    {
      _leaves.clear();
      _leafOrdinals.assign(_nodes.size(), -1);
      for (int i=0; i<int(_nodes.size()); i++)
      {
        if (_nodes[i].firstChild < 0)
        {
          _leafOrdinals[i] = int(_leaves.size());
          _leaves.push_back(i);
        }
      }
      _leavesValid = true;
    }
  public:
    //! A Cartesian grid of root cells, rootCounts[k] of them in direction k.
    BoxForest(const std::vector<int> &rootCounts) : _dim(int(rootCounts.size())), _rootCounts(rootCounts)
    {
      if ((_dim < 1) || (_dim > MAX_DIM)) throw std::invalid_argument("BoxForest: unsupported dimension");
      std::size_t rootCount = 1;
      for (int n : rootCounts) rootCount *= n;
      _nodes.resize(rootCount);
      std::vector<int> index(_dim, 0);
      for (std::size_t i=0; i<rootCount; i++) // root i has multi-index with direction 0 varying fastest
      {
        for (int k=0; k<_dim; k++) _nodes[i].box.lo[k] = Coord(index[k]) << MAX_LEVEL;
        for (int k=0; k<_dim; k++)
        {
          if (++index[k] < rootCounts[k]) break;
          index[k] = 0;
        }
      }
    }

    int dimension() const { return _dim; }
    int nodeCount() const { return int(_nodes.size()); }
    const Node &node(int nodeIndex) const { return _nodes[nodeIndex]; }
    bool isLeaf(int nodeIndex) const { return _nodes[nodeIndex].firstChild < 0; }

    //! node indices of the active cells
    const std::vector<int> &leaves() const
    {
      if (!_leavesValid) rebuildLeaves();
      return _leaves;
    }

    int leafOrdinal(int nodeIndex) const
    {
      if (!_leavesValid) rebuildLeaves();
      return _leafOrdinals[nodeIndex];
    }

    int faceOrdinal(int leafOrdinal, int side) const { return leafOrdinal * 2 * _dim + side; }

    //! Bisect an active cell in the directions set in mask; returns the index of the first child.
    int refine(int nodeIndex, unsigned mask)
    {
      if (!isLeaf(nodeIndex)) throw std::invalid_argument("BoxForest::refine: cell is not active");
      if ((mask == 0) || (mask >= (1u << _dim))) throw std::invalid_argument("BoxForest::refine: invalid mask");
      for (int k=0; k<_dim; k++)
      {
        if ((mask & (1u << k)) && (_nodes[nodeIndex].box.level[k] >= MAX_LEVEL)) throw std::invalid_argument("BoxForest::refine: MAX_LEVEL exceeded");
      }
      const int firstChild = int(_nodes.size());
      for (const Box &childBox : _nodes[nodeIndex].box.children(mask, _dim))
      {
        Node child;
        child.box = childBox;
        child.parent = nodeIndex;
        _nodes.push_back(child);
      }
      _nodes[nodeIndex].firstChild = firstChild;
      _nodes[nodeIndex].mask = std::uint8_t(mask);
      _leavesValid = false;
      return firstChild;
    }

    //! Calls f(nodeIndex) for each active cell that overlaps the half-open box [lo, hi).
    template<class F>
    void forEachLeafOverlapping(const std::array<Coord,MAX_DIM> &lo, const std::array<Coord,MAX_DIM> &hi, F &&f) const
    {
      // range of root multi-indices
      std::array<int,MAX_DIM> first{}, last{}, index{};
      for (int k=0; k<_dim; k++)
      {
        if (hi[k] <= lo[k]) return;
        first[k] = int(lo[k] >> MAX_LEVEL);
        last[k]  = std::min(int((hi[k] - 1) >> MAX_LEVEL), _rootCounts[k] - 1);
        if (first[k] > last[k]) return;
        index[k] = first[k];
      }
      while (true)
      {
        int rootIndex = 0;
        for (int k=_dim-1; k>=0; k--) rootIndex = rootIndex * _rootCounts[k] + index[k];
        visitOverlapping(rootIndex, lo, hi, f);
        int k = 0;
        for (; k<_dim; k++)
        {
          if (++index[k] <= last[k]) break;
          index[k] = first[k];
        }
        if (k == _dim) break;
      }
    }

    //! Active cells sharing (part of) the given face of an active cell, with positive (d-1)-measure.
    std::vector<int> faceNeighbors(int nodeIndex, int side) const
    {
      const Box &box = _nodes[nodeIndex].box;
      const int k = side / 2;
      std::array<Coord,MAX_DIM> lo = box.lo, hi{};
      for (int j=0; j<_dim; j++) hi[j] = box.hi(j);
      std::vector<int> neighbors;
      if (side % 2 == 1) { lo[k] = hi[k]; hi[k] = lo[k] + 1; }
      else
      {
        if (lo[k] == 0) return neighbors;
        hi[k] = lo[k]; lo[k] = hi[k] - 1;
      }
      forEachLeafOverlapping(lo, hi, [&](int n){ neighbors.push_back(n); });
      return neighbors;
    }

    /**
     \brief  Enforces 1-irregularity per direction: across every face, the two neighbors' levels differ by at
             most maxLevelDifference in each direction tangential to the face.

     A cell that is too coarse in some tangential direction relative to a face neighbor is bisected in that
     direction only, so that the rule does not turn anisotropic refinements into isotropic ones.  (Levels in the
     direction normal to the face are not constrained: they do not affect how the face is tiled.)  Returns the number
     of bisections performed.
     */
    int balance(int maxLevelDifference = 1)
    {
      int refinementCount = 0;
      bool changed = true;
      while (changed)
      {
        changed = false;
        const std::vector<int> leafNodes = leaves(); // copy: refine() invalidates
        for (int nodeIndex : leafNodes)
        {
          unsigned mask = 0;
          for (int side=0; side<2*_dim; side++)
          {
            for (int neighbor : faceNeighbors(nodeIndex, side))
            {
              for (int j=0; j<_dim; j++)
              {
                if (j == side / 2) continue;
                if (int(_nodes[neighbor].box.level[j]) > int(_nodes[nodeIndex].box.level[j]) + maxLevelDifference) mask |= (1u << j);
              }
            }
          }
          if (mask != 0)
          {
            // each cell in leafNodes is refined at most once per pass, and faceNeighbors() reads the current tree
            refine(nodeIndex, mask);
            refinementCount++;
            changed = true;
          }
        }
      }
      return refinementCount;
    }

    //! The largest difference in level, in a direction tangential to their shared face, between face neighbors.
    int maxTangentialLevelDifference() const
    {
      int maxDifference = 0;
      for (int nodeIndex : leaves())
      {
        for (int side=0; side<2*_dim; side++)
        {
          for (int neighbor : faceNeighbors(nodeIndex, side))
          {
            for (int j=0; j<_dim; j++)
            {
              if (j == side / 2) continue;
              maxDifference = std::max(maxDifference, std::abs(int(_nodes[neighbor].box.level[j]) - int(_nodes[nodeIndex].box.level[j])));
            }
          }
        }
      }
      return maxDifference;
    }

    //! The trace patches of the active mesh (see Patch).  Cost: one tree query per active face.
    Skeleton skeleton() const
    {
      const std::vector<int> &leafNodes = leaves();
      const int sideCount = 2 * _dim;
      const int faceCount = int(leafNodes.size()) * sideCount;

      // union-find over faces
      std::vector<int> rep(faceCount);
      std::iota(rep.begin(), rep.end(), 0);
      auto find = [&](int i)
      {
        while (rep[i] != i) { rep[i] = rep[rep[i]]; i = rep[i]; }
        return i;
      };

      for (int leaf=0; leaf<int(leafNodes.size()); leaf++)
      {
        for (int k=0; k<_dim; k++)
        {
          const int myFace = faceOrdinal(leaf, 2*k+1);
          for (int neighbor : faceNeighbors(leafNodes[leaf], 2*k+1))
          {
            int a = find(myFace), b = find(faceOrdinal(_leafOrdinals[neighbor], 2*k));
            if (a != b) rep[std::max(a,b)] = std::min(a,b);
          }
        }
      }

      Skeleton skeleton;
      skeleton.patchForFace.assign(faceCount, -1);
      std::vector<double> faceMeasure; // per patch: measure of the faces on one side, in root-cell units
      for (int face=0; face<faceCount; face++)
      {
        const int root = find(face);
        const int leaf = face / sideCount, side = face % sideCount, k = side / 2;
        const Box &box = _nodes[leafNodes[leaf]].box;
        if (skeleton.patchForFace[root] < 0) // root is the least face of the patch, so it is seen first
        {
          skeleton.patchForFace[root] = int(skeleton.patches.size());
          Patch patch;
          patch.normal = k;
          patch.plane = (side % 2 == 1) ? box.hi(k) : box.lo[k];
          for (int j=0; j<_dim; j++) { patch.lo[j] = box.lo[j]; patch.hi[j] = box.hi(j); }
          skeleton.patches.push_back(patch);
          faceMeasure.push_back(0.0);
        }
        const int patchOrdinal = skeleton.patchForFace[root];
        skeleton.patchForFace[face] = patchOrdinal;
        Patch &patch = skeleton.patches[patchOrdinal];
        double measure = 1.0;
        for (int j=0; j<_dim; j++)
        {
          if (j == k) continue;
          patch.lo[j] = std::min(patch.lo[j], box.lo[j]);
          patch.hi[j] = std::max(patch.hi[j], box.hi(j));
          measure *= std::ldexp(1.0, -int(box.level[j]));
        }
        const bool minusSide = (side % 2 == 1);
        (minusSide ? patch.minusFaces : patch.plusFaces).push_back({leafNodes[leaf], side});
        if (minusSide) faceMeasure[patchOrdinal] += measure; // one side only; plus-only patches are handled below
      }
      for (std::size_t p=0; p<skeleton.patches.size(); p++)
      {
        Patch &patch = skeleton.patches[p];
        patch.lo[patch.normal] = patch.hi[patch.normal] = patch.plane;
        double measure = faceMeasure[p];
        if (patch.minusFaces.empty())
        {
          measure = 0;
          for (const Face &face : patch.plusFaces)
          {
            double m = 1.0;
            for (int j=0; j<_dim; j++) if (j != patch.normal) m *= std::ldexp(1.0, -int(_nodes[face.cell].box.level[j]));
            measure += m;
          }
        }
        double boxMeasure = 1.0;
        for (int j=0; j<_dim; j++) if (j != patch.normal) boxMeasure *= std::ldexp(double(patch.hi[j] - patch.lo[j]), -MAX_LEVEL);
        patch.isBox = std::abs(measure - boxMeasure) < 1e-12 * boxMeasure;
      }
      return skeleton;
    }

    /**
     \brief  1D restriction of a Legendre expansion to a sub-interval.

     Given u = sum_j c_j P_j on [-1,1], and the sub-interval [a,b] of [-1,1], returns R (row-major,
     (p+1) x (p+1)) such that u restricted to [a,b] is sum_i (R c)_i P_i in the sub-interval's own
     reference coordinate.  The constraint matrix relating a cell face to its patch is the Kronecker
     product of one such matrix per tangential direction (the identity where the extents agree).
     */
    static std::vector<double> legendreRestriction(int p, double a, double b)
    {
      const int n = p + 1;
      // Gauss-Legendre points and weights by Newton iteration
      std::vector<double> x(n), w(n);
      auto legendre = [](int degree, double xi, std::vector<double> &P)
      {
        P.assign(degree + 1, 1.0);
        if (degree >= 1) P[1] = xi;
        for (int i=2; i<=degree; i++) P[i] = ((2*i-1) * xi * P[i-1] - (i-1) * P[i-2]) / i;
      };
      std::vector<double> P;
      for (int q=0; q<n; q++)
      {
        double xi = std::cos(M_PI * (q + 0.75) / (n + 0.5));
        for (int it=0; it<100; it++)
        {
          legendre(n, xi, P);
          const double dP = n * (xi * P[n] - P[n-1]) / (xi * xi - 1.0);
          const double dx = P[n] / dP;
          xi -= dx;
          if (std::abs(dx) < 1e-15) break;
        }
        legendre(n, xi, P);
        const double dP = n * (xi * P[n] - P[n-1]) / (xi * xi - 1.0);
        x[q] = xi;
        w[q] = 2.0 / ((1.0 - xi * xi) * dP * dP);
      }
      std::vector<double> R(n * n, 0.0), Pfine, Pcoarse;
      for (int q=0; q<n; q++)
      {
        legendre(p, x[q], Pfine);
        legendre(p, a + (b - a) * (x[q] + 1.0) / 2.0, Pcoarse);
        for (int i=0; i<n; i++)
          for (int j=0; j<n; j++)
            R[i*n+j] += 0.5 * (2*i+1) * w[q] * Pfine[i] * Pcoarse[j];
      }
      return R;
    }
  };
}

#endif
