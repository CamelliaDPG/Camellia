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
  // Spoiler: the answer is "No" in general, if the fs is not HGRAD.
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr v;
  LinearTermPtr volumeTerm, traceTerm;
  auto trace = Function::meshSkeletonCharacteristic();
  
  v = vf->testVar("v", fs);
  
  // default trace term to the volume L^2 term; override below
  
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
  vector<int> polyOrderVector = {polyOrder}; // isotropic in p
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
  Camellia::FieldContainer<double> ipMatrix(1,testDofCount,testDofCount);
  
  ip->computeInnerProductMatrix(ipMatrix,testOrdering,ipBasisCache);
  
  ip->printInteractions();
  
  ipMatrix.resize(testDofCount,testDofCount);
  
  double condNumber2NormEst = SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix);
  cout << "2-norm condition number estimate: " << condNumber2NormEst << endl;
  
  return condNumber2NormEst;
}


double conditionNumberForTraceTermGram(CellTopoPtr cellTopo, int polyOrder, Space fs, double scaling)
{
  bool useTraceTerms = true;
  return conditionNumberForGram(cellTopo, polyOrder, fs, scaling, useTraceTerms);
}

double conditionNumberForVolumeL2Gram(CellTopoPtr cellTopo, int polyOrder, Space fs, double scaling)
{
  bool useTraceTerms = false;
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
  
  bool useTraceTerm = true; // 'false' allows us to see what the non-trace version does
  double scaling = 1.0; // 1.0 corresponds to reference space.
  int polyOrder = 3;
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  cmdp.setOption("scaling",&scaling,"scaling to use for coordinate space (h value)");
  cmdp.setOption("polyOrder", &polyOrder, "polynomial order");
  cmdp.setOption("useTraceTerm", "useVolumeL2Term", &useTraceTerm, "whether to use the trace term, or the standard volume L^2 term");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  double h = scaling;

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
        if (useTraceTerm)
        {
          conditionNumberForTraceTermGram(cellTopo, polyOrder, space, h);
        }
        else
        {
          conditionNumberForVolumeL2Gram(cellTopo, polyOrder, space, h);
        }
      }
    }
  }
  
  return 0;
}
