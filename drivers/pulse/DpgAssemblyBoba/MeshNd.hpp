#ifndef MESH_ND_HPP
#define MESH_ND_HPP

#include <Teuchos_RCP.hpp>

#include "BOBA/boba.hpp"

#include "Mesh1d.hpp"

// tensor-product mesh
template <std::size_t numDim>
class MeshNd {
public:
  MeshNd(const boba::Array<Mesh1dPtr, numDim> &mesh) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(mesh[n] != Teuchos::null, "mesh component is null");
    }
    mesh_ = mesh;
  }

  Mesh1dPtr
  getComponent(std::size_t dimId) const {
    boba_assert(dimId < numDim, "dimension index is out of range");
    return mesh_[dimId];
  }

private:
  boba::Array<Mesh1dPtr, numDim> mesh_;
};

template <std::size_t N>
using MeshNdPtr = Teuchos::RCP<MeshNd<N>>;

template <std::size_t N>
MeshNdPtr<N>
newMeshNd(const boba::Array<Mesh1dPtr, N> &mesh) {
  return Teuchos::rcp(new MeshNd(mesh));
}

#endif
