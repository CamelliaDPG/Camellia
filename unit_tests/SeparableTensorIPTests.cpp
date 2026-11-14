#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "BasisFamily.h"
#include "BoxGDA.h"
#include "BoxMeshTopology.h"
#include "Function.h"
#include "SeparableTensorIP.h"
#include "VarFactory.h"

#include <cmath>

using namespace Camellia;
using namespace std;

namespace
{
  //! a function constant on each cell (the product of the cell's widths), as the separable norm's weights are
  class CellWidthProduct : public TFunction<double>
  {
  public:
    CellWidthProduct() : TFunction<double>(0) {}
    void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const auto &nodes = basisCache->getPhysicalCellNodes();
      for (int c=0; c<values.dimension(0); c++)
      {
        double product = 1;
        for (int k=0; k<nodes.dimension(2); k++)
        {
          double lo = nodes(c,0,k), hi = nodes(c,0,k);
          for (int n=1; n<nodes.dimension(1); n++) { lo = std::min(lo, nodes(c,n,k)); hi = std::max(hi, nodes(c,n,k)); }
          product *= (hi - lo);
        }
        for (int p=0; p<values.dimension(1); p++) values(c,p) = 1.0 + 10.0 * product;
      }
    }
  };

  //! the largest difference, relative to the largest entry, between the Gram matrices from 1D factors and by quadrature
  double tensorVersusQuadrature(bool serendipity, const vector<int> &increments, Teuchos::FancyOStream &out, long long &tensorCells)
  {
    const int dim = 4, H1Order = 2, delta_k = 2;
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr u = vf->fieldVar("u", L2);
    VarPtr v = vf->testVar("v", HGRAD);
    if (serendipity) vf->setBasisFamily(BasisFamily::serendipityBasisFamily());
    BFPtr bf = Teuchos::rcp(new BF(vf));
    bf->addTerm(u, v);
    auto topology = Teuchos::rcp(new BoxMeshTopology(vector<vector<double>>{{0.0, 0.5, 2.0}, {0.0, 0.1}, {-1.0, 0.0, 3.0}, {-2.0, 2.0}}));
    MeshPtr mesh = BoxGDA::mesh(topology, bf, H1Order, delta_k);
    mesh->hRefine(std::map<GlobalIndexType,unsigned>{{0, 1u | 4u}});
    std::map<GlobalIndexType, std::vector<int>> orderIncrements;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal()) if (cellID % 2 == 0) orderIncrements[cellID] = increments;
    BoxGDA::pRefineInDirections(mesh, orderIncrements);

    // the direction-separable graph norm's form: per-cell weights on each derivative, and on the value
    auto ip = SeparableTensorIP::ip();
    FunctionPtr weight = Teuchos::rcp(new CellWidthProduct);
    ip->addTerm(v->di(2));
    ip->addTerm(weight * v->di(1));
    ip->addTerm(1.8 * v->di(3));
    ip->addTerm(0.65 * v->di(4));
    ip->addTerm(Function::sqrtFunction(weight * 3.0) * v);

    double maxDifference = 0, maxEntry = 0;
    for (GlobalIndexType cellID : mesh->getActiveCellIDsGlobal())
    {
      DofOrderingPtr testOrdering = mesh->getElementType(cellID)->testOrderPtr;
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, true);
      const int n = testOrdering->totalDofs();
      Camellia::FieldContainer<double> tensor(1, n, n), quadrature(1, n, n);
      ip->setUseTensorAssembly(true);
      ip->computeInnerProductMatrix(tensor, testOrdering, basisCache);
      ip->setUseTensorAssembly(false);
      ip->computeInnerProductMatrix(quadrature, testOrdering, basisCache);
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
        {
          maxDifference = std::max(maxDifference, std::abs(tensor(0,i,j) - quadrature(0,i,j)));
          maxEntry = std::max(maxEntry, std::abs(quadrature(0,i,j)));
        }
    }
    tensorCells = ip->tensorCellCount();
    out << (serendipity ? "serendipity" : "hierarchical") << ", increments " << increments[0] << increments[1] << increments[2] << increments[3]
        << ": " << tensorCells << " cells from 1D factors, " << ip->fallbackCellCount() << " by quadrature; max |difference| / max |entry| = "
        << maxDifference / maxEntry << endl;
    return maxDifference / maxEntry;
  }
}

TEUCHOS_UNIT_TEST( SeparableTensorIP, MatchesQuadrature )
{
  // Box cells of different widths, with isotropic and anisotropic orders, hierarchical and serendipity test bases:
  // the Gram matrix from 1D factors agrees with quadrature to roundoff, and every cell takes the 1D path.
  for (bool serendipity : {false, true})
  {
    for (const vector<int> &increments : vector<vector<int>>{{0,0,0,0}, {2,0,1,0}, {1,1,1,1}})
    {
      long long tensorCells = 0;
      const double relativeDifference = tensorVersusQuadrature(serendipity, increments, out, tensorCells);
      TEST_COMPARE(relativeDifference, <, 1e-12);
      TEST_COMPARE(tensorCells, >, 0);
    }
  }
}
