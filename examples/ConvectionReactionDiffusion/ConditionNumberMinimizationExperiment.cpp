//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "E_to_T.hpp"
#include "MeshFactory.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;

/*
 
 Trying some explicitly-defined bilinear forms and inner products to
 see what might be possible in terms of reducing condition number of Poisson
 stiffness matrix for ultraweak DPG.
 
 */

double conditionNumberLAPACK(TMatrixPtr<double> stiffnessMatrix, bool diagScaling)
{
  Camellia::FieldContainer<double> A;
  SerialDenseWrapper::extractFCFromTMatrix(stiffnessMatrix, A);
  if (diagScaling)
  {
    for (int i=0; i<A.dimension(0); i++)
    {
      double diag = A(i,i);
      for (int j=0; j<A.dimension(1); j++)
      {
        A(i,j) /= diag;
      }
    }
  }
  
  return SerialDenseWrapper::condest(A);
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, 0);
  
  int spaceDim = 2;
  int meshWidth = 2;
  bool conformingTraces = true;
  int H1Order = 2, delta_k = 3;
  double domainWidth = 1.0e-3;
  bool diagScaling = false;
  double h = domainWidth / meshWidth;
  double weight = h / 4.0; // ratio of area of square with sidelength h to its perimeter
  
  double sigma_weight = 1.0; // h / 4.0; // sigma = sigma_weight * u->grad()

  Space uHatSpace = conformingTraces ? HGRAD : L2;
  
  VarFactoryPtr vf = VarFactory::varFactory();
  
  // fields
  VarPtr u = vf->fieldVar("u");
  VarPtr sigma = vf->fieldVar("sigma", VECTOR_L2);
  
  // traces
  VarPtr u_hat = vf->traceVar("u_hat", uHatSpace);
  VarPtr sigma_n = vf->fluxVar("sigma_n");
  
  // tests
  VarPtr v = vf->testVar("v", HGRAD);
  VarPtr tau = vf->testVar("tau", HDIV);
  
  BFPtr bf = BF::bf(vf);
  
// standard BF:
//  bf->addTerm(sigma, v->grad());
//  bf->addTerm(sigma_n, v);
//  
//  bf->addTerm(sigma, tau);
//  bf->addTerm(u, tau->div());
//  bf->addTerm(-u_hat, tau->dot_normal());
  
  // weighted BF:
  bf->addTerm(sigma, v->grad());
  bf->addTerm(weight * sigma_n, v);
  
  bf->addTerm(sigma, tau);
  bf->addTerm(sigma_weight * u, tau->div());
  bf->addTerm(- sigma_weight * weight * u_hat, tau->dot_normal());
  
  IPPtr ip = IP::ip();
// standard IP:
  ip->addTerm(tau + v->grad());
  ip->addTerm(tau->div());
  ip->addTerm(v);
  ip->addTerm(tau);
  
  // weighted IP:
//  ip->addTerm(tau + v->grad());
//  ip->addTerm(sigma_weight * tau->div());
//  ip->addTerm(max(sigma_weight,1e-3) * v);
//  ip->addTerm(sigma_weight * weight * tau);
  
  BCPtr bc = BC::bc();
  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
  
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(1.0 * sigma_weight * v);
  
  vector<double> dimensions(spaceDim,domainWidth);
  vector<int> elementCounts(spaceDim,meshWidth);
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k);
  
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  
  soln->setUseCondensedSolve(true);
  soln->initializeLHSVector();
  soln->initializeStiffnessAndLoad();
  soln->populateStiffnessAndLoad();
  
  auto eStiffness = soln->getStiffnessMatrix();
  auto tStiffness = E_to_T::matrix(eStiffness);
  
  double condNumber = conditionNumberLAPACK(tStiffness, diagScaling);
  
  cout << "condest (1-norm): " << condNumber << endl;
  
  return 0;
}
