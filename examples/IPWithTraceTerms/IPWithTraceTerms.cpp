//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_GlobalMPISession.hpp"

#include "CellTopology.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "SpatialFilter.h"
#include "PoissonFormulation.h"
#include "TimeSteppingConstants.h"

using namespace Camellia;
using namespace std;

double conditionNumberForGram(CellTopoPtr cellTopo, int polyOrder, Space fs, double scaling, bool useTraceTerms)
{
  // our initial question is: if we use the natural differential operator on the interior and simply the L^2 inner product on the element boundary, do we recover a norm?
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr v;
  LinearTermPtr volumeTerm, traceTerm;
  auto trace = Function::meshSkeletonCharacteristic();
  
  v = vf->testVar("v", fs);
  
  if (useTraceTerms)
  {
    traceTerm = trace * v;
  }
  else
  {
    traceTerm = 1.0 * v;
  }
  
  int spaceDim = cellTopo->getDimension();
  
  switch (fs) {
    case Camellia::HGRAD:
      volumeTerm = 1.0 * v->grad();
      break;
      
    case Camellia::HDIV:
      volumeTerm = 1.0 * v->div();
      break;
      
    case Camellia::HCURL:
      volumeTerm = 1.0 * v->curl(spaceDim);
      break;
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space!")
      break;
  }
  
  IPPtr ip = IP::ip();
  ip->addTerm(volumeTerm);
  ip->addTerm(traceTerm);
  
  DofOrderingFactory dofOrderingFactory(vf);
  vector<int> polyOrderVector(1,polyOrder); // isotropic in p
  auto testOrdering = dofOrderingFactory.testOrdering(polyOrderVector, cellTopo);
  
  bool createSideCacheToo = true;
  BasisCachePtr ipBasisCache = BasisCache::basisCacheForReferenceCell(cellTopo, polyOrder * 2, createSideCacheToo);
  
  auto physicalCellNodes = ipBasisCache->getPhysicalCellNodes();
  int entryCount = physicalCellNodes.size();
  for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
  {
    physicalCellNodes[entryOrdinal] *= scaling;
  }
  vector<GlobalIndexType> cellIDs = {0};
  ipBasisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
  
  int testDofCount = testOrdering->totalDofs();
  Intrepid::FieldContainer<double> ipMatrix(1,testDofCount,testDofCount);
  
  ip->computeInnerProductMatrix(ipMatrix,testOrdering,ipBasisCache);
  
  ip->printInteractions();
  
  double condNumber1NormEst = SerialDenseWrapper::getMatrixConditionNumber(ipMatrix);
  cout << "1-norm condition number estimate: " << condNumber1NormEst << endl;
  
  return condNumber1NormEst;
}


double conditionNumberForTraceTermGram(CellTopoPtr cellTopo, int polyOrder, Space fs, double scaling)
{
  bool useTraceTerms = true;
  return conditionNumberForGram(cellTopo, polyOrder, fs, scaling, useTraceTerms);
}

double conditionNumberForVolumeL2Gram(CellTopoPtr cellTopo, int polyOrder, Space fs, double scaling)
{
  bool useTraceTerms = true;
  return conditionNumberForGram(cellTopo, polyOrder, fs, scaling, useTraceTerms);
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI

  /*
   Quick and dirty driver to output a sample Gram matrix for the Poisson problem
   */
  
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
//  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
//  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
//  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (2 or 3)");
//  cmdp.setOption("outputStiffness", "onlyGram", &outputStiffnessToo, "if true, output the DPG stiffness matrix as well.");
//  cmdp.setOption("meshWidth", &meshWidth, "mesh width");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  double h = 1.0;

  vector<CellTopoPtr> cellTopos_1D = {CellTopology::line()};
  vector<CellTopoPtr> cellTopos_2D = {CellTopology::triangle(), CellTopology::quad()};
  vector<CellTopoPtr> cellTopos_3D = {CellTopology::tetrahedron(), CellTopology::hexahedron()};
  
  vector<Space> spaces_1D = {HGRAD};
  vector<Space> spaces_2D = {HGRAD, HDIV, HCURL};
  vector<Space> spaces_3D = {HGRAD, HDIV, HCURL};
  
  vector<vector<CellTopoPtr>> cellTopos = {cellTopos_1D, cellTopos_2D, cellTopos_3D};
  vector<vector<Space>> spaces = {spaces_1D, spaces_2D, spaces_3D};
  
  for (int d=0; d < 3; d++)
  {
    std::cout << "****************************** " << d+1 << "D Tests ******************************\n";
    auto cellTopos_d = cellTopos[d];
    auto spaces_d = spaces[d];
    
    for (auto cellTopo : cellTopos_d)
    {
      cout << "--- " << cellTopo->getName() << " ---\n";
      for (auto space : spaces_d)
      {
        int polyOrder = 3;
        conditionNumberForTraceTermGram(cellTopo, polyOrder, space, h);
      }
    }
  }
  
  return 0;
}
