//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  IPFilter.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/10/14.
//
//

#include "IPFilter.h"

using namespace Camellia;

IPFilter::IPFilter(IPPtr ip)
{
  _ip = ip;
}

void IPFilter::filter(Intrepid::FieldContainer<double> &localStiffnessMatrix, Intrepid::FieldContainer<double> &localRHSVector,
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc)
{
  Teuchos::Array<int> dim(localStiffnessMatrix.rank());
  for (int rank=0; rank<dim.size(); rank++)
  {
    dim[rank] = localStiffnessMatrix.dimension(rank);
  }
  Intrepid::FieldContainer<double> ipMatrix(dim);
  GlobalIndexType sampleCellID = basisCache->cellIDs()[0];
  ElementTypePtr elemType = mesh->getElementType(sampleCellID);
  DofOrderingPtr dofOrdering = elemType->trialOrderPtr;
  _ip->computeInnerProductMatrix(ipMatrix, dofOrdering, basisCache);

  // add to localStiffnessMatrix
  for (int i=0; i<localStiffnessMatrix.size(); i++)
  {
    localStiffnessMatrix[i] += ipMatrix[i];
  }
}
