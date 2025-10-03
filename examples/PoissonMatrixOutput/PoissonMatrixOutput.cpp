//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_GlobalMPISession.hpp"

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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI

  /*
   Quick and dirty driver to output a sample Gram matrix for the Poisson problem
   */
  
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  int spaceDim = 3;
  int polyOrder = 3, delta_k = spaceDim;
  bool outputStiffnessToo = false;
  int meshWidth = 2;
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  cmdp.setOption("polyOrder",&polyOrder,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (2 or 3)");
  cmdp.setOption("outputStiffness", "onlyGram", &outputStiffnessToo, "if true, output the DPG stiffness matrix as well.");
  cmdp.setOption("meshWidth", &meshWidth, "mesh width");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  bool useConformingTraces = true;
  PoissonFormulation form = PoissonFormulation(spaceDim, useConformingTraces);
  
  // single element, same as in reference space
  vector<double> dims(spaceDim,2.0);
  vector<double> x0(spaceDim,-1.0);
  vector<int> numElements(spaceDim,meshWidth);
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dims,numElements,x0);
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dims, numElements, polyOrder+1, delta_k);
  
  RHSPtr rhs = RHS::rhs(); // unit-forcing RHS
  rhs->addTerm(1.0 * form.v());
  IPPtr ip = form.bf()->graphNorm();
  {
    // trying something -- if we replace the L^2 terms with trace terms, is the Gram matrix still non-singular?
    ip = IP::ip();
    VarPtr tau = form.tau();
    VarPtr v = form.v();
    ip->addTerm(tau->div());
    ip->addTerm(v->grad() - tau);

    auto trace = Function::meshSkeletonCharacteristic();
    ip->addTerm(v * trace);
    ip->addTerm(tau * trace);
//    ip->addTerm(tau->dot_normal()); // this choice seems more natural, but it's also not as clear that makes a norm -- and it definitely makes the condition number in a test case go up by an order of magnitude...
  }
  
  BCPtr bc = BC::bc();
  bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), Function::zero());
  
  int cellID = 0;
  DofOrderingPtr testOrdering = mesh->getElementType(cellID)->testOrderPtr;
  
  int testDofCount = testOrdering->totalDofs();
  Camellia::FieldContainer<double> ipMatrix(1,testDofCount,testDofCount);
  
  cout << "For poly order = " << polyOrder << " and delta_k = " << delta_k;
  cout << ", testDofCount is: " << testDofCount << endl;
  
  bool testVsTest = true;
  BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);
  
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
  
  DofOrderingPtr trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
  int trialDofCount = trialOrdering->totalDofs();
  
  Camellia::FieldContainer<double> rectangularBMatrix(1,testDofCount,trialDofCount);
  form.bf()->stiffnessMatrix(rectangularBMatrix, mesh->getElementType(cellID), ipBasisCache->getCellSideParities(), ipBasisCache, false, true);
  
  ipMatrix.resize(testDofCount,testDofCount);
  rectangularBMatrix.resize(testDofCount,trialDofCount);
  
  // rework rectangularBMatrix in a way that corresponds to the elimination of Dirichlet degrees of freedom
  set<int> uHatIndices;
  int sideCount = mesh->getElementType(cellID)->cellTopoPtr->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    auto uHatIndicesVector = trialOrdering->getDofIndices(form.u_hat()->ID(), sideOrdinal);
    uHatIndices.insert(uHatIndicesVector.begin(),uHatIndicesVector.end());
  }
  
  int bcCount = uHatIndices.size();
  
  Camellia::FieldContainer<double> rectangularBMatrixAfterBCs(testDofCount,trialDofCount-bcCount);
  int offset = 0;
  for (int trialOrdinal=0; trialOrdinal < trialDofCount; trialOrdinal++)
  {
    if (uHatIndices.find(trialOrdinal) != uHatIndices.end())
    {
      offset++;
      continue;
    }
    // otherwise, copy the column corresponding to trialOrdinal into the new container:
    for (int testOrdinal=0; testOrdinal< testDofCount; testOrdinal++)
    {
      rectangularBMatrixAfterBCs(testOrdinal,trialOrdinal-offset) = rectangularBMatrix(testOrdinal,trialOrdinal);
    }
  }

//  double condNumber2Norm = SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix,false);
//  if (condNumber2Norm == -1)
//  {
//    cout << "Matrix has zero eigenvalues!\n";
//  }
//  else
//  {
//    cout << "2-norm condition number: " << condNumber2Norm << endl;
//  }
  double condNumber2Norm = SerialDenseWrapper::getMatrixConditionNumber2Norm(ipMatrix,true);
  cout << "2-norm condition number (ignoring zero eigenvalues): " << condNumber2Norm << endl;
  
//  double condNumber1NormEst = SerialDenseWrapper::getMatrixConditionNumber(ipMatrix);
//  cout << "1-norm condition number estimate: " << condNumber1NormEst << endl;
  
  ostringstream suffix;
  suffix << "_k" << polyOrder << "_deltak" << delta_k << "_" << spaceDim << "D";
  
  ostringstream matrixFileName;
  matrixFileName << "PoissonGram" << suffix.str() << ".dat";
  
  ostringstream BFileName;
  BFileName << "PoissonB" << suffix.str() << ".dat";
  
  ostringstream coordsFileName;
  coordsFileName << "coords" << suffix.str() << ".dat";
  
  SerialDenseWrapper::writeMatrixToMatlabFile(matrixFileName.str(), ipMatrix);
  SerialDenseWrapper::writeMatrixToMatlabFile(coordsFileName.str(), allDofCoords);
  SerialDenseWrapper::writeMatrixToMatlabFile(BFileName.str(), rectangularBMatrixAfterBCs);
  
  std::cout << "Wrote Gram matrix to " << matrixFileName.str() << endl;
  std::cout << "Wrote coords file to " << coordsFileName.str() << endl;
  std::cout << "Wrote B matrix to " << BFileName.str() << endl;
  
  if (outputStiffnessToo)
  {
    SolutionPtr solution = Solution::solution(form.bf(), mesh, bc, rhs, ip);
    solution->solve();
    
    ostringstream stiffnessFileName;
    stiffnessFileName << "PoissonK" << suffix.str() << ".dat";
    
    solution->setWriteMatrixToFile(true, stiffnessFileName.str());
    solution->solve();
    
    ostringstream solnName;
    solnName << "PoissonSolution" << suffix.str();
    HDF5Exporter solutionExporter(mesh, solnName.str(), "/tmp");
    solutionExporter.exportSolution(solution, 0.0);
  }
  
  return 0;
}
