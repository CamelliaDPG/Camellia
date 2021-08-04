//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "IPSwitcher.h"

#include "SerialDenseWrapper.h"
#include "Mesh.h"
#include "BasisCache.h"

using namespace Camellia;


IPSwitcher::IPSwitcher(IPPtr ip1, IPPtr ip2, double minH)
{
  _ip1 = ip1;
  _ip2 = ip2;
  _minH = minH;
  
  // would not be hard to add support for BC imposition, but IPSwitcher is not commonly used, so we'll delay that for now...
  TEUCHOS_TEST_FOR_EXCEPTION(ip1->hasBCsImposed(), std::invalid_argument, "IPSwitcher does not (yet) support IPs that have BCs imposed");
  TEUCHOS_TEST_FOR_EXCEPTION(ip2->hasBCsImposed(), std::invalid_argument, "IPSwitcher does not (yet) support IPs that have BCs imposed");
}


// added by Jesse - evaluate inner product at given varFunctions
LinearTermPtr IPSwitcher::evaluate(map< int, TFunctionPtr<double>> &varFunctions, bool boundaryPart)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Evaluation of switched IPs not supported yet");
//  return Teuchos::rcp((LinearTerm *)NULL);
}

void IPSwitcher::computeInnerProductMatrix(Camellia::FieldContainer<double> &innerProduct,
    Teuchos::RCP<DofOrdering> dofOrdering,
    Teuchos::RCP<BasisCache> basisCache)
{

  MeshPtr mesh = basisCache->mesh();
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  int numCells = innerProduct.dimension(0);
  unsigned numDofs = innerProduct.dimension(1);
  innerProduct.initialize(0.0);
  for (int c = 0; c<numCells; c++)
  {
    int cellID = cellIDs[c];
    int cubatureEnrichment = 0; // maybe add some smart way to figure this out?
    BasisCachePtr cellCache = BasisCache::basisCacheForCell(mesh, cellID, true, cubatureEnrichment);

    double h = std::min(mesh->getCellXSize(cellID),mesh->getCellYSize(cellID)); //mesh->getCellMeasure(cellID)); // getCellXSize or getCellYSize
    Camellia::FieldContainer<double> cellIP(1,numDofs,numDofs);
    if (h > _minH)
    {
      _ip1->computeInnerProductMatrix(cellIP,dofOrdering,cellCache);
    }
    else
    {
      _ip2->computeInnerProductMatrix(cellIP,dofOrdering,cellCache);
    }

    for (int i = 0; i<numDofs; i++)
    {
      for (int j = 0; j<numDofs; j++)
      {
        innerProduct(c,i,j) += cellIP(0,i,j);
      }
    }
  }

}

// does some extra work and may not
double IPSwitcher::computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache)
{
  return std::max(_ip1->computeMaxConditionNumber(testSpace,basisCache),_ip2->computeMaxConditionNumber(testSpace,basisCache));
}

// compute IP vector when var==fxn
void IPSwitcher::computeInnerProductVector(Camellia::FieldContainer<double> &ipVector,
                                           VarPtr var, TFunctionPtr<double> fxn,
                                           Teuchos::RCP<DofOrdering> dofOrdering,
                                           Teuchos::RCP<BasisCache> basisCache,
                                           bool sumInto)
{

  MeshPtr mesh = basisCache->mesh();
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  int numCells = cellIDs.size();
  unsigned numDofs = dofOrdering->totalDofs();

  for (int c = 0; c<numCells; c++)
  {
    int cellID = cellIDs[c];
    int cubatureEnrichment = 0; // maybe add some smart way to figure this out?
    BasisCachePtr cellCache = BasisCache::basisCacheForCell(mesh, cellID, true, cubatureEnrichment);

    double h = mesh->getCellMeasure(cellID); // getCellXSize or getCellYSize
    Camellia::FieldContainer<double> cellVec(1,numDofs);
    if (h<_minH)
    {
      _ip1->computeInnerProductVector(cellVec,var,fxn,dofOrdering,cellCache);
    }
    else
    {
      _ip2->computeInnerProductVector(cellVec,var,fxn,dofOrdering,cellCache);
    }

    if (sumInto)
    {
      for (int i = 0; i<numDofs; i++)
      {
        ipVector(c,i) += cellVec(1,i);
      }
    }
    else
    {
      for (int i = 0; i<numDofs; i++)
      {
        ipVector(c,i) = cellVec(1,i);
      }
    }
  }

}

bool IPSwitcher::hasBoundaryTerms()
{
  return (_ip1->hasBoundaryTerms() || _ip2->hasBoundaryTerms());
}

void IPSwitcher::printInteractions()
{
  _ip1->printInteractions();
  _ip2->printInteractions();
}
