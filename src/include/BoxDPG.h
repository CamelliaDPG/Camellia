#ifndef Camellia_BoxDPG_h
#define Camellia_BoxDPG_h

//
//  BoxDPG.h
//  Camellia
//
//  PROTOTYPE.  A DPG discretization on a BoxForest mesh: hypercube cells of any dimension, bisected in any
//  subset of directions, with L2 field variables and L2 flux variables.  See notes/anisotropic-refinement-design.md.
//
//  This class stands where Mesh + MeshTopology + GDAMinimumRule + Solution stand for the existing meshes, for the
//  purpose of establishing that the box-mesh numbering and constraints are right.  Element matrices come from the
//  usual BF / IP / RHS on an ElementType and a BasisCache; no Mesh is involved.  It is serial.
//
//  Degrees of freedom:
//    - each field variable: its volume basis on every active cell;
//    - each flux variable: one tensor-Legendre polynomial per trace patch of the BoxForest skeleton, of the
//      flux's polynomial degree in each tangential direction.  A cell's side coefficients are the L2 projection of
//      the patch polynomial onto the side basis (exact, since the side basis contains the polynomial).  The
//      projection is computed with the side basis as given, so it does not rely on its ordering or scaling.
//

#include <functional>
#include <map>
#include <vector>

#include "BF.h"
#include "BasisCache.h"
#include "BoxForest.h"
#include "CellTopology.h"
#include "ElementType.h"
#include "Function.h"
#include "IP.h"
#include "RHS.h"
#include "TypeDefs.h"

namespace Camellia
{
  class BoxDPG
  {
  public:
    using PointFunction = std::function<double(const double *x)>;
    //! where(direction, upper): true on the boundary faces (orthogonal to direction; upper or lower) where the BC applies
    using BoundarySelector = std::function<bool(int direction, bool upper)>;
    //! value(x, direction, upper): the flux value, with the convention that the outward normal is the positive one
    using BoundaryValue = std::function<double(const double *x, int direction, bool upper)>;

  private:
    struct FluxBC
    {
      int varID;
      BoundarySelector where;
      BoundaryValue value;
    };

    //! Element matrices and the local-to-global map for one active cell
    struct CellData
    {
      Camellia::FieldContainer<double> B; // (1, trial, test)
      Camellia::FieldContainer<double> G; // (1, test, test)
      Camellia::FieldContainer<double> l; // (1, test)
      std::vector<GlobalIndexType> globalIndices;
      std::vector<std::vector<std::pair<int,double>>> rows; // per local trial dof: (position in globalIndices, weight)
    };

    BoxForest _forest;
    std::vector<std::vector<double>> _rootVertices;
    int _dim;
    BFPtr _bf;
    IPPtr _ip;
    RHSPtr _rhs;
    std::vector<FluxBC> _fluxBCs;

    CellTopoPtr _cellTopo;
    ElementTypePtr _elemType;
    BasisCachePtr _basisCache, _ipBasisCache;
    Camellia::FieldContainer<double> _refNodes;
    std::vector<int> _sideForFace; // BoxForest side (2 * direction + upper) -> side ordinal in _cellTopo

    std::vector<VarPtr> _fieldVars, _fluxVars;
    std::vector<int> _fieldOffsets; // offset of each field variable within a cell's field dofs
    int _fieldDofsPerCell = 0;
    std::vector<int> _fluxDegrees;  // polynomial degree of each flux variable
    std::vector<int> _fluxOffsets;  // offset of each flux variable within a patch's dofs
    int _fluxDofsPerPatch = 0;

    BoxForest::Skeleton _skeleton;
    bool _numberingValid = false;
    std::vector<double> _solution;

    void setCell(int leafOrdinal, bool createSideCaches);
    void computeCellData(int leafOrdinal, CellData &cellData);
    void patchBasisValues(const BoxForest::Patch &patch, int degree, const double *x, std::vector<double> &values) const;
    void computeBoundaryValues(std::map<GlobalIndexType,double> &bcValues);
    static void project(const std::vector<double> &phi, const std::vector<double> &psi, const std::vector<double> &weights,
                        int n, int m, std::vector<double> &coefficients);
  public:
    /**
     \param rootVertices  for each direction, the vertex coordinates of the root grid
     \param bf            bilinear form; its trial variables must be fields and fluxes
     \param H1Order       as for Mesh: fields and fluxes have polynomial degree H1Order - 1
     \param delta_k       test space enrichment
     */
    BoxDPG(const std::vector<std::vector<double>> &rootVertices, BFPtr bf, int H1Order, int delta_k);

    //! The mesh.  Refine through this; the numbering is rebuilt at the next solve().
    BoxForest &forest();
    const BoxForest::Skeleton &skeleton();

    void setIP(IPPtr ip);
    void setRHS(RHSPtr rhs);
    void addFluxBC(VarPtr flux, BoundarySelector where, BoundaryValue value);

    int dimension() const { return _dim; }
    CellTopoPtr cellTopology() const { return _cellTopo; }
    ElementTypePtr elementType() const { return _elemType; }

    //! physical coordinate in direction k of an integer BoxForest coordinate
    double coordinate(int direction, BoxForest::Coord c) const;
    //! (1, nodes, dim) physical nodes of an active cell, in the node ordering of cellTopology()
    Camellia::FieldContainer<double> physicalCellNodes(int nodeIndex) const;

    GlobalIndexType globalDofCount();
    GlobalIndexType fieldDofOffset(int leafOrdinal, VarPtr field);
    GlobalIndexType fluxDofOffset(int patchOrdinal, VarPtr flux);

    //! Assembles and solves (sparse direct).
    void solve();
    const std::vector<double> &solution() const { return _solution; }

    //! coefficients for the element's trial ordering on an active cell (constraints applied)
    std::vector<double> cellCoefficients(int leafOrdinal);

    double L2Error(VarPtr field, PointFunction exact);
    //! DPG energy error (norm of the residual's Riesz representation) on each active cell
    std::vector<double> energyErrors();

    //! A Camellia Function defined pointwise in any dimension.
    static FunctionPtr function(PointFunction f);
  };
}

#endif
