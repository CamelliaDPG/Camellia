//
//  IPFilter.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/10/14.
//
//

#include "IPFilter.h"

using namespace Intrepid;
using namespace Camellia;

IPFilter::IPFilter(IPPtr ip) {
  _ip = ip;
}

void IPFilter::filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector,
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) {
  Teuchos::Array<int> dim(localStiffnessMatrix.rank());
  for (int rank=0; rank<dim.size(); rank++) {
    dim[rank] = localStiffnessMatrix.dimension(rank);
  }
  FieldContainer<double> ipMatrix(dim);
  GlobalIndexType sampleCellID = basisCache->cellIDs()[0];
  ElementTypePtr elemType = mesh->getElementType(sampleCellID);
  DofOrderingPtr dofOrdering = elemType->trialOrderPtr;
  _ip->computeInnerProductMatrix(ipMatrix, dofOrdering, basisCache);
  
  // add to localStiffnessMatrix
  for (int i=0; i<localStiffnessMatrix.size(); i++) {
    localStiffnessMatrix[i] += ipMatrix[i];
  }
}
