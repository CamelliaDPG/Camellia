//
//  FunctionErrorIndicator.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 7/2/20.
//

#include "FunctionErrorIndicator.h"

using namespace Camellia;

template <typename Scalar>
FunctionErrorIndicator<Scalar>::FunctionErrorIndicator(MeshPtr mesh, TFunctionPtr<Scalar> f, int quadratureEnrichment)
:
ErrorIndicator(mesh),
_mesh(mesh),
_errorFunction(f),
_quadratureEnrichment(quadratureEnrichment)
{}

//! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
template <typename Scalar>
void FunctionErrorIndicator<Scalar>::measureError()
{
  // clear previous measures
  this->_localErrorMeasures.clear();
  
  const auto & myCells = _mesh->cellIDsInPartition();

  // Note that any MPI communication should happen prior to measureError()
  // Note also that the following loop could be made more efficient by performing the integration for like element types at the same time.
  // (The main inefficiency will come from performing basis computations repeatedly; as it is, no two cells share a basis cache.)
  // TODO: make this loop over element types, and do a single Function::integrate() call for each.
  for (const auto & myCellID : myCells)
  {
    // L^2 norm of function on cell:
    _localErrorMeasures[myCellID] = std::sqrt((_errorFunction*_errorFunction)->integrate(myCellID, _mesh, _quadratureEnrichment));
  }
}

// ETI for double
template class Camellia::FunctionErrorIndicator<double>;
