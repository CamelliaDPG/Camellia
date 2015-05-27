#include "ConfusionBilinearForm.h"
#include "VortexManufacturedSolution.h"

typedef Sacado::Fad::SFad<double,2> F2;                          // FAD with # of independent vars fixed at 2 (x and y)
typedef Sacado::Fad::SFad< Sacado::Fad::SFad<double,2>, 2> F2_2; // same thing, but nested so we can take 2 derivatives

VortexManufacturedSolution::VortexManufacturedSolution(double epsilon)
{
  _epsilon = epsilon;

  // set the class variables from ExactSolution:
  _bc = Teuchos::rcp(this,false);  // false: don't let the RCP own the memory
  _rhs = Teuchos::rcp(this,false);
  _bilinearForm = Teuchos::rcp(new ConfusionBilinearForm(epsilon)); // don't specify beta - this is done in ConfusionBilinearForm
  _cbf = Teuchos::rcp(new ConfusionBilinearForm(epsilon)); // don't specify beta - this is done in ConfusionBilinearForm (stores another copy, b/c we need acces in BC, imposeBC, getConstraints)
}

int VortexManufacturedSolution::H1Order()
{
  // -1 for non-polynomial solution...
  return -1;
}

template <typename T> const T VortexManufacturedSolution::u(T &x, T &y)
{
  T xp = x-.5;
  T yp = y-.5;
  T r = sqrt(xp*xp+yp*yp);
  T t;
  if (r > .5)
  {
    t = r-.5;
  }
  else
  {
    t = 0.0;
  }
  return t;
}

double VortexManufacturedSolution::solutionValue(int trialID,
    FieldContainer<double> &physicalPoint)
{

  double x = physicalPoint(0);
  double y = physicalPoint(1);
  F2 sx(2,0,x), sy(2,1,y), su; // s for Sacado
  su = u(sx,sy);
  double value;

  switch(trialID)
  {
  case ConfusionBilinearForm::U:
    value = su.val();
    break;
  case ConfusionBilinearForm::U_HAT:
    value = su.val();
    break;
  case ConfusionBilinearForm::SIGMA_1:
    value = _epsilon * su.dx(0); // SIGMA_1 == eps * d/dx (u)
    break;
  case ConfusionBilinearForm::SIGMA_2:
    value = _epsilon * su.dx(1); // SIGMA_2 == eps * d/dy (u)
    break;
  case ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT:
    TEUCHOS_TEST_FOR_EXCEPTION( trialID == ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT,
                                std::invalid_argument,
                                "for fluxes, you must call solutionValue with unitNormal argument.");
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument,
                                "solutionValues called with unknown trialID.");
  }
  return value;
}

double VortexManufacturedSolution::solutionValue(int trialID,
    FieldContainer<double> &physicalPoint,
    FieldContainer<double> &unitNormal)
{
  if ( trialID != ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT )
  {
    return solutionValue(trialID,physicalPoint);
  }
  // otherwise, get SIGMA_1 and SIGMA_2, and the unit normal
  double sigma1 = solutionValue(ConfusionBilinearForm::SIGMA_1,physicalPoint);
  double sigma2 = solutionValue(ConfusionBilinearForm::SIGMA_2,physicalPoint);
  double u = solutionValue(ConfusionBilinearForm::U,physicalPoint);
  double n1 = unitNormal(0);
  double n2 = unitNormal(1);
  double x = physicalPoint(0);
  double y = physicalPoint(1);
  double sigma_n = sigma1*n1 + sigma2*n2;
  double beta_n = _cbf->getBeta(x,y)[0]*n1+_cbf->getBeta(x,y)[1]*n2;
  return u*beta_n - sigma_n;
}

/********** RHS implementation **********/
bool VortexManufacturedSolution::nonZeroRHS(int testVarID)
{
  return false; //no rhs for vortex problem - driven by BCs only
}

void VortexManufacturedSolution::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values)
{
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  if (testVarID == ConfusionBilinearForm::V)
  {
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = physicalPoints(cellIndex,ptIndex,0);
        double y = physicalPoints(cellIndex,ptIndex,1);
        F2_2 sx(2,0,x), sy(2,1,y), su; // s for Sacado
        sx.val() = F2(2,0,x);
        sy.val() = F2(2,1,y);
        su = u(sx,sy);
        values(cellIndex,ptIndex) = 0.0;
      }
    }
  }
}

/***************** BC Implementation *****************/

// BC
bool VortexManufacturedSolution::bcsImposed(int varID)
{
  return varID == ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT;
  //    return varID == ConfusionBilinearForm::U_HAT;
}

void VortexManufacturedSolution::imposeBC(int varID, FieldContainer<double> &physicalPoints,
    FieldContainer<double> &unitNormals,
    FieldContainer<double> &dirichletValues,
    FieldContainer<bool> &imposeHere)
{
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  double tol = 1e-14;
  TEUCHOS_TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument, "spaceDim != 2" );
  for (int cellIndex=0; cellIndex < numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++)
    {
      double x = physicalPoints(cellIndex, ptIndex, 0);
      double y = physicalPoints(cellIndex, ptIndex, 1);
      double n1 = unitNormals(cellIndex,ptIndex,0);
      double n2 = unitNormals(cellIndex,ptIndex,1);
      double beta_n = _cbf->getBeta(x,y)[0]*n1+_cbf->getBeta(x,y)[1]*n2;

      // inflow = not outflow
      if (beta_n<0)
      {
        double pi = 2.0*acos(0.0);

        FieldContainer<double> physicalPoint(spaceDim);
        physicalPoint(0) = x;
        physicalPoint(1) = y;
        FieldContainer<double> unitNormal(spaceDim);
        unitNormal(0) = n1;
        unitNormal(1) = n2;

        double u0 = solutionValue(ConfusionBilinearForm::U_HAT,physicalPoint);
        //	double u0 = abs((x-.5)*(x-.5) + (y-.5)*(y-.5)-.25);
        //	double u0 = solutionValue(ConfusionBilinearForm::BETA_N_MINUS_SIGMA_HAT,physicalPoint,unitNormal);

        if (bcsImposed(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT))
        {
          dirichletValues(cellIndex,ptIndex) = beta_n*u0;
        }
        else
        {
          dirichletValues(cellIndex,ptIndex) = u0;
        }
        imposeHere(cellIndex,ptIndex) = true;
      }
      else
      {
        imposeHere(cellIndex,ptIndex) = false;
      }

    }
  }
}

void VortexManufacturedSolution::getConstraints(FieldContainer<double> &physicalPoints,
    FieldContainer<double> &unitNormals,
    vector<map<int,FieldContainer<double > > > &constraintCoeffs,
    vector<FieldContainer<double > > &constraintValues)
{

  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  map<int,FieldContainer<double> > outflowConstraint;
  FieldContainer<double> uCoeffs(numCells,numPoints);
  FieldContainer<double> beta_sigmaCoeffs(numCells,numPoints);
  FieldContainer<double> outflowValues(numCells,numPoints);

  // default to no constraints, apply on outflow only
  uCoeffs.initialize(0.0);
  beta_sigmaCoeffs.initialize(0.0);
  outflowValues.initialize(0.0);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
    {
      double x = physicalPoints(cellIndex,pointIndex,0);
      double y = physicalPoints(cellIndex,pointIndex,1);
      vector<double> beta = _cbf->getBeta(x,y);
      double beta_n = beta[0]*unitNormals(cellIndex,pointIndex,0)+beta[1]*unitNormals(cellIndex,pointIndex,1);

      bool isOnABoundary=false;
      if ((abs(x) < 1e-12) || (abs(y) < 1e-12))
      {
        isOnABoundary = true;
      }
      if ((abs(x-1.0) < 1e-12) || (abs(y-1.0) < 1e-12))
      {
        isOnABoundary = true;
      }

      if (beta_n>0 && isOnABoundary)
      {
        // this combo isolates sigma_n
        uCoeffs(cellIndex,pointIndex) = beta_n;
        beta_sigmaCoeffs(cellIndex,pointIndex) = -1.0;
        outflowValues(cellIndex,pointIndex) = 0.0;
      }

    }
  }
  //    outflowConstraint[ConfusionBilinearForm::U_HAT] = beta_sigmaCoeffs;
  outflowConstraint[ConfusionBilinearForm::U_HAT] = uCoeffs;
  outflowConstraint[ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT] = beta_sigmaCoeffs;
  constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
  constraintValues.push_back(outflowValues); // only one constraint on outflow

}
