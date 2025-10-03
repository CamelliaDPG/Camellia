//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_GlobalMPISession.hpp"

#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "TimeSteppingConstants.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI

  /*
   Quick and dirty driver to output a sample Gram matrix for the Stokes problem
   */
  
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  int spaceDim = 3;
  int polyOrder = 3, delta_k = spaceDim;
  bool outputStiffnessToo = false;
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (2 or 3)");
  cmdp.setOption("outputStiffness", "onlyGram", &outputStiffnessToo, "if true, output the DPG stiffness matrix as well.");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  
  bool useConformingTraces = true;
  double mu = 1.0;
  StokesVGPFormulation form = StokesVGPFormulation::steadyFormulation(spaceDim, mu, useConformingTraces);
  
  // single element, same as in reference space
  vector<double> dims(spaceDim,2.0);
  vector<double> x0(spaceDim,-1.0);
  vector<int> numElements(spaceDim,1);
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
  
  form.initializeSolution(meshTopo, polyOrder, delta_k);
  form.addZeroMeanPressureCondition();

  MeshPtr mesh = form.solution()->mesh();
  
  int cellID = 0;
  DofOrderingPtr testOrdering = mesh->getElementType(cellID)->testOrderPtr;
  
  int testDofCount = testOrdering->totalDofs();
  Camellia::FieldContainer<double> ipMatrix(1,testDofCount,testDofCount);
  
  cout << "For poly order = " << polyOrder << " and delta_k = " << delta_k;
  cout << ", testDofCount is: " << testDofCount << endl;
  
  bool testVsTest = true;
  BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);
  
  IPPtr ip = form.bf()->graphNorm();
  ip->computeInnerProductMatrix(ipMatrix,testOrdering,ipBasisCache);
  
  ip->printInteractions();
  
  Camellia::FieldContainer<double> allDofCoords(testDofCount,spaceDim);
  
  std::set<int> varIDs = testOrdering->getVarIDs();
  for (int varID : varIDs)
  {
    vector<int> dofIndices = testOrdering->getDofIndices(varID);
    int basisCardinality = dofIndices.size();
    Camellia::FieldContainer<double> dofCoords(basisCardinality,spaceDim);
    testOrdering->getDofCoords(dofCoords,varID);
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
    {
      for (int d=0;d<spaceDim;d++)
      {
        allDofCoords(dofIndices[basisOrdinal],d) = dofCoords(basisOrdinal,d);
      }
    }
  }
  
  ipMatrix.resize(testDofCount,testDofCount);
  
//  double condNumber2Norm = SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix,false);
//  if (condNumber2Norm == -1)
//  {
//    cout << "Matrix has zero eigenvalues!\n";
//  }
//  else
//  {
//    cout << "2-norm condition number: " << condNumber2Norm << endl;
//  }
  condNumber2Norm = SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix,true);
  cout << "2-norm condition number (ignoring zero eigenvalues): " << condNumber2Norm << endl;
  
//  double condNumber1NormEst = SerialDenseWrapper::getMatrixConditionNumber(ipMatrix);
//  cout << "1-norm condition number estimate: " << condNumber1NormEst << endl;
  
  ostringstream suffix;
  suffix << "_k" << polyOrder << "_deltak" << delta_k << "_" << spaceDim << "D";
  
  ostringstream matrixFileName;
  matrixFileName << "StokesGram" << suffix.str() << ".dat";
  
  ostringstream coordsFileName;
  coordsFileName << "coords" << suffix.str() << ".dat";
  
  SerialDenseWrapper::writeMatrixToMatlabFile(matrixFileName.str(), ipMatrix);
  SerialDenseWrapper::writeMatrixToMatlabFile(coordsFileName.str(), allDofCoords);
  
  if (outputStiffnessToo)
  {
    auto soln = form.solution();
    ostringstream stiffnessFileName;
    stiffnessFileName << "StokesK" << suffix.str() << ".dat";
    
    form.addWallCondition(SpatialFilter::allSpace());
    
    soln->setWriteMatrixToFile(true, stiffnessFileName.str());
    soln->solve();
  }
  
  return 0;
}
