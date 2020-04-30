//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "TrigFunctions.h"

using namespace Camellia;

int main(int argc, char *argv[])
{
  const static double PI = 3.141592;
  FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI) );
  FunctionPtr cos_pi_x = Teuchos::rcp( new Cos_ax(PI) );
  FunctionPtr phi_exact = sin_pi_x * cos_pi_x;
  
  int spaceDim = 2;
  int meshWidth = 2;
  bool conformingTraces = true;
  int H1Order = 2;
  PoissonFormulation form(spaceDim,conformingTraces);
  
  vector<double> dimensions(spaceDim,1.0);
  vector<int> elementCounts(spaceDim,meshWidth);
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dimensions, elementCounts, H1Order);
  
  SolutionPtr soln = Solution::solution(mesh);
  
  VarPtr phi = form.u();
  map<int, FunctionPtr> exactMap;
  exactMap[phi->ID()] = phi_exact;
  
  soln->projectOntoMesh(exactMap);
  
  return 0;
}
