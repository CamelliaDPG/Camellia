//
//  MeshBasisIntegrals.h
//  Camellia
//
//  Created by Roberts, Nathan V on 11/23/22.
//

#ifndef MeshBasisIntegrals_h
#define MeshBasisIntegrals_h

#include <Basis.h>
#include <Function.h>
#include <Mesh.h>
#include <TypeDefs.h>

namespace Camellia {
  //! For an arbitrary mesh and specified basis, compute and store the integrals of the (physical) basis members on each element, multiplied by an arbitrary function weight.
  class MeshBasisIntegrals
  {
    std::vector<GlobalIndexType> _integralCellIDs; // cell IDs for the stored integrals
    std::map<GlobalIndexType, int> _cellOrdinalForCell; // reverse lookup for _integralCellIDs
    DeviceView<double> _integrals; // (C,F) container
    
    BasisPtr _basis;
    
    // bottleneck initialization method called by both constructors.
    void initialize(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, int quadratureEnrichment, const std::set<GlobalIndexType> &cellIDs);
  public:
    //! constructor.  Current implementation does *not* support distributed geometry (MeshTopology pruning should be disabled).  Computes the integrals on construction; the present implementation does not support updates to the mesh post-construction.  cellIDs indicates which cells may be present (in some order, possibly with duplicates) when solutionIntegral is called.
    MeshBasisIntegrals(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, int quadratureEnrichment, const std::set<GlobalIndexType> &cellIDs);
    
    //! constructor.  Current implementation does *not* support distributed geometry (MeshTopology pruning should be disabled).  When includeAllCells is false, only computes integrals for locally-owned, active cells.  Otherwise, computes on the mesh as a whole, including on inactive elements.  Computes the integrals on construction; the present implementation does not support updates to the mesh post-construction.
    MeshBasisIntegrals(MeshPtr mesh, BasisPtr basis, FunctionPtr weight, bool includeAllCells, int quadratureEnrichment);
    
    //! Integrate a set of solutions defined as coefficients on the basis passed in at construction against the weight function passed in at construction.
    //! solutionIntegrals - (C,N) output.  C: cell index, N: solution index.
    //! solutionCoefficients - (C,N,F) input.  C: cell index, N: solution index, F: basis ordinal.
    //! cellIDs: (C) input; the cell IDs in the mesh that correspond to the cells in the solutionCoefficients input.
    void solutionIntegral(DeviceView<double> solutionIntegrals, DeviceView<double> solutionCoefficients, vector<GlobalIndexType> &cellIDs);
    
    ~MeshBasisIntegrals();
  };
}

#endif /* MeshBasisIntegrals_h */
