#include "ConfusionBilinearForm.h"
#include "EricksonManufacturedSolution.h"

typedef Sacado::Fad::SFad<double,2> F2;                          // FAD with # of independent vars fixed at 2 (x and y)
typedef Sacado::Fad::SFad< Sacado::Fad::SFad<double,2>, 2> F2_2; // same thing, but nested so we can take 2 derivatives

EricksonManufacturedSolution::EricksonManufacturedSolution(double epsilon, double beta_x, double beta_y) {
  _epsilon = epsilon;
  _beta_x  = beta_x;
  _beta_y  = beta_y;
  _useWallBC = true; // use wall u=0 bc on outflow    
  
  // set the class variables from ExactSolution:
  _bc = Teuchos::rcp(this,false);  // false: don't let the RCP own the memory
  _rhs = Teuchos::rcp(this,false);
  _bilinearForm = Teuchos::rcp(new ConfusionBilinearForm(epsilon,beta_x,beta_y));
}

int EricksonManufacturedSolution::H1Order() {
  // -1 for non-polynomial solution...
  return -1;
}

template <typename T> const T EricksonManufacturedSolution::u(T &x, T &y) {
  // in Bui, Demkowicz, Ghattas' paper
  double pi = acos(0.0)*2.0;
  
  T t = exp((1.0-x)/_epsilon)*sin(pi*y/(_epsilon*_epsilon)); // Norbert's example
  
  return t;
}

double EricksonManufacturedSolution::solutionValue(int trialID,
						   FieldContainer<double> &physicalPoint) {
  
  double x = physicalPoint(0);
  double y = physicalPoint(1);
  F2 sx(2,0,x), sy(2,1,y), su; // s for Sacado 
  su = u(sx,sy);
  double value;
  
  double pi = acos(0.0)*2.0;

  double C0 = 0.0;// average of u0
  double u = C0;
  double u_x = 0.0;
  double u_y = 0.0;  

  for (int n = 1;n<20;n++){

    double lambda = n*n*pi*pi*_epsilon;
    double d = sqrt(1.0+4.0*_epsilon*lambda);
    double r1 = (1.0+d)/(2.0*_epsilon);
    double r2 = (1.0-d)/(2.0*_epsilon);
    
    double Cn = 0.0;            
    if (n==1){
      Cn = 1.0; // first term only
    }
    /*
    Cn = -1 + cos(n*pi/2)+.5*n*pi*sin(n*pi/2) + sin(n*pi/4)*(n*pi*cos(n*pi/4)-2*sin(3*n*pi/4));
    Cn /= (n*pi);
    Cn /= (n*pi);
    */

    // normal stress outflow
    double Xbottom;
    double Xtop;
    double dXtop;
    if (!_useWallBC){
      Xbottom = r1*exp(-r2) - r2*exp(-r1);
      Xtop = r1*exp(r2*(x-1.0)) - r2*exp(r1*(x-1.0));
      dXtop = r1*r2*(exp(r2*(x-1.0)) - exp(r1*(x-1.0)));
    }else{
      // wall, zero outflow
      Xtop = (exp(r2*(x-1))-exp(r1*(x-1)));
      Xbottom = (exp(-r2)-exp(-r1));
      dXtop = (exp(r2*(x-1))*r2-exp(r1*(x-1))*r1);    
    }

    double X = Xtop/Xbottom;
    double dX = dXtop/Xbottom;
    double Y = Cn*cos(n*pi*y);
    double dY = -Cn*n*pi*sin(n*pi*y);
    
    u += X*Y;
    u_x += _epsilon * dX*Y;
    u_y += _epsilon * X*dY;
  }

  //  cout << "u = " << u << endl;
  switch(trialID) {
  case ConfusionBilinearForm::U:
    value = u;
    break;
  case ConfusionBilinearForm::U_HAT:
    value = u;
    break;
  case ConfusionBilinearForm::SIGMA_1:
    value = u_x; // SIGMA_1 == eps * d/dx (u)
    break;
  case ConfusionBilinearForm::SIGMA_2:
    value = u_y; // SIGMA_2 == eps * d/dy (u)
    break;
  case ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT:
    TEST_FOR_EXCEPTION( trialID == ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT,
			std::invalid_argument,
			"for fluxes, you must call solutionValue with unitNormal argument.");
    break;
  default:
    TEST_FOR_EXCEPTION( true, std::invalid_argument,
			"solutionValues called with unknown trialID.");
  }
  return value;
}

double EricksonManufacturedSolution::solutionValue(int trialID,
						   FieldContainer<double> &physicalPoint,
						   FieldContainer<double> &unitNormal) {
  if ( trialID != ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT ) {
    return solutionValue(trialID,physicalPoint);
  }
  // otherwise, get SIGMA_1 and SIGMA_2, and the unit normal
  double sigma1 = solutionValue(ConfusionBilinearForm::SIGMA_1,physicalPoint);
  double sigma2 = solutionValue(ConfusionBilinearForm::SIGMA_2,physicalPoint);
  double u = solutionValue(ConfusionBilinearForm::U,physicalPoint);
  double n1 = unitNormal(0);
  double n2 = unitNormal(1);
  double sigma_n = sigma1*n1 + sigma2*n2;
  double beta_n = _beta_x*n1 + _beta_y*n2;
  return u*beta_n - sigma_n;
}

/********** RHS implementation **********/
bool EricksonManufacturedSolution::nonZeroRHS(int testVarID) {
  return false; 
}

void EricksonManufacturedSolution::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  if (testVarID == ConfusionBilinearForm::V) {
    // f = - eps * (d^2/dx^2 + d^2/dy^2) ( u ) + beta_x du/dx + beta_y du/dy
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = physicalPoints(cellIndex,ptIndex,0);
        double y = physicalPoints(cellIndex,ptIndex,1);
        F2_2 sx(2,0,x), sy(2,1,y), su; // s for Sacado 
        sx.val() = F2(2,0,x);
        sy.val() = F2(2,1,y);
        su = u(sx,sy);

	/* these are values of convection-diffusion
	   values(cellIndex,ptIndex) = - _epsilon * ( su.dx(0).dx(0) + su.dx(1).dx(1) ) 
	   + _beta_x * su.dx(0).val() + _beta_y * su.dx(1).val();
	*/

	// this RHS corresponds to only convection
	//values(cellIndex,ptIndex) = _beta_x * su.dx(0).val() + _beta_y * su.dx(1).val();
	
	// exact RHS
	double pi = acos(0.0)*2.0;
	double epsSquared = _epsilon*_epsilon;
	//	values(cellIndex,ptIndex) = exp((1.0-x)/_epsilon)*(pi*pi - 2*epsSquared)*sin(pi*y/epsSquared)/(epsSquared*_epsilon);
	values(cellIndex,ptIndex) = 0.0;
	
      }
    }
  }
}

/***************** BC Implementation *****************/
bool EricksonManufacturedSolution::bcsImposed(int varID){
  // returns true if there are any BCs anywhere imposed on varID
  //return (varID == ConfusionBilinearForm::U_HAT);
  return (varID == ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT || (varID == ConfusionBilinearForm::U_HAT));
}

void EricksonManufacturedSolution::imposeBC(int varID, FieldContainer<double> &physicalPoints,
					    FieldContainer<double> &unitNormals,
					    FieldContainer<double> &dirichletValues,
					    FieldContainer<bool> &imposeHere) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  TEST_FOR_EXCEPTION( ( spaceDim != 2  ),
		      std::invalid_argument,
		      "ConfusionBC expects spaceDim==2.");  
  
  TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells ) 
		      || ( dirichletValues.dimension(1) != numPoints ) 
		      || ( dirichletValues.rank() != 2  ),
		      std::invalid_argument,
		      "dirichletValues dimensions should be (numCells,numPoints).");
  TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells ) 
		      || ( imposeHere.dimension(1) != numPoints ) 
		      || ( imposeHere.rank() != 2  ),
		      std::invalid_argument,
		      "imposeHere dimensions should be (numCells,numPoints).");
  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(2);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      FieldContainer<double> physicalPoint(pointDimensions,
                                           &physicalPoints(cellIndex,ptIndex,0));
      FieldContainer<double> unitNormal(pointDimensions,
                                        &unitNormals(cellIndex,ptIndex,0));
      double beta_n = unitNormals(cellIndex,ptIndex,0)*_beta_x + unitNormals(cellIndex,ptIndex,1)*_beta_y;
      double x = physicalPoint(cellIndex,ptIndex,0);
      double y = physicalPoint(cellIndex,ptIndex,1);
      imposeHere(cellIndex,ptIndex) = false;
      if (varID==ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT) {
	dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint, unitNormal);
	//      if (varID==ConfusionBilinearForm::U_HAT) {
	//	dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint);
	if ( abs(x-1.0) > 1e-12) { // if not the outflow (pts on boundary already)
	  imposeHere(cellIndex,ptIndex) = true;
	}
      } else if (varID==ConfusionBilinearForm::U_HAT) {
	// wall boundary 
	if (abs(x-1.0)<1e-12 && _useWallBC){
	  dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint);
	  imposeHere(cellIndex,ptIndex) = true;
	}
      } 
    }
  }
}

void EricksonManufacturedSolution::getConstraints(FieldContainer<double> &physicalPoints, 
						  FieldContainer<double> &unitNormals,
						  vector<map<int,FieldContainer<double > > > &constraintCoeffs,
						  vector<FieldContainer<double > > &constraintValues){
    
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);       
  map<int,FieldContainer<double> > outflowConstraint;
  FieldContainer<double> uCoeffs(numCells,numPoints);
  FieldContainer<double> beta_sigmaCoeffs(numCells,numPoints);
  FieldContainer<double> outflowValues(numCells,numPoints);
  double tol = 1e-12;
  // default to no constraints, apply on outflow only
  uCoeffs.initialize(0.0);
  beta_sigmaCoeffs.initialize(0.0);
  outflowValues.initialize(0.0);

  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(2);    
  for (int cellIndex=0;cellIndex<numCells;cellIndex++){
    for (int pointIndex=0;pointIndex<numPoints;pointIndex++){
      FieldContainer<double> physicalPoint(pointDimensions,
                                           &physicalPoints(cellIndex,pointIndex,0));
      FieldContainer<double> unitNormal(pointDimensions,
                                        &unitNormals(cellIndex,pointIndex,0));

      double x = physicalPoints(cellIndex,pointIndex,0);
      double y = physicalPoints(cellIndex,pointIndex,1);

      double beta_n = _beta_x*unitNormals(cellIndex,pointIndex,0)+_beta_y*unitNormals(cellIndex,pointIndex,1);
      
      if ( abs(x-1.0) < tol ) { // if on outflow boundary
	TEST_FOR_EXCEPTION(beta_n < 0,std::invalid_argument,"Inflow condition on boundary");
	
	// this combo isolates sigma_n
	//uCoeffs(cellIndex,pointIndex) = 1.0;
	uCoeffs(cellIndex,pointIndex) = beta_n;
	beta_sigmaCoeffs(cellIndex,pointIndex) = -1.0;	    
	double beta_n_u_minus_sigma_n = solutionValue(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, physicalPoint, unitNormal);
	double u_hat = solutionValue(ConfusionBilinearForm::U_HAT, physicalPoint, unitNormal);
	outflowValues(cellIndex,pointIndex) = beta_n*u_hat - beta_n_u_minus_sigma_n; // sigma_n
      }	
    }
  }
  outflowConstraint[ConfusionBilinearForm::U_HAT] = uCoeffs;
  outflowConstraint[ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT] = beta_sigmaCoeffs;	        
  if (!_useWallBC){
    constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
    constraintValues.push_back(outflowValues); // only one constraint on outflow
  }
}


// =============== ABSTRACT FUNCTION INTERFACE for trialID U  =================== 

void EricksonManufacturedSolution::getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints){
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  functionValues.resize(numCells,numPoints);

  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(spaceDim);    
  for (int i=0;i<numCells;i++){
    for (int j=0;j<numPoints;j++){
      double x = physicalPoints(i,j,0);
      double y = physicalPoints(i,j,1);
      FieldContainer<double> physicalPoint(pointDimensions);
      physicalPoint(0) = x;
      physicalPoint(1) = y;
      
      functionValues(i,j) = solutionValue(ConfusionBilinearForm::U,physicalPoint);
    }
  }
  
}
