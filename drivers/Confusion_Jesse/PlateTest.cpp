#include "Solver.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

#include "SolutionExporter.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RieszRep.h"
#include "BasisFactory.h" // for test
#include "HessianFilter.h"

#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

#include "Epetra_LinearProblem.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "StandardAssembler.h" // for system assembly
#include "SerialDenseWrapper.h" // for system assembly
#include "TestingUtilities.h" 
#include "MeshPolyOrderFunction.h"

#include "IPSwitcher.h"

double pi = 2.0*acos(0.0);

// ===================== Mesh functions ====================

class MeshInfo {
  MeshPtr _mesh;
public:
  MeshInfo(MeshPtr mesh){
    _mesh = mesh;
  }
  double getMinCellMeasure(){
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      minMeasure = min(minMeasure, _mesh->getCellMeasure((*elemIt)->cellID()));
    }
    return minMeasure;
  }
  vector<int> getMinCellSizeCellIDs(){
    double minMeasure = getMinCellMeasure();
    vector<int> minMeasureCellIDs;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      if (minMeasure <= _mesh->getCellMeasure((*elemIt)->cellID())){
	minMeasureCellIDs.push_back((*elemIt)->cellID());
      }
    }
    return minMeasureCellIDs;
  }
  double getMinCellSideLength(){
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      minMeasure = min(minMeasure, _mesh->getCellXSize((*elemIt)->cellID()));
      minMeasure = min(minMeasure, _mesh->getCellYSize((*elemIt)->cellID()));
    }
    return minMeasure;
  }
};
// =============================================================

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class AnisotropicHScaling : public Function {
  int _spatialCoord;
public:
  AnisotropicHScaling(int spatialCoord){
    _spatialCoord = spatialCoord;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    MeshPtr mesh = basisCache->mesh();
    vector<int> cellIDs = basisCache->cellIDs();

    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      double h = 1.0;
      if (_spatialCoord==0){
	h = mesh->getCellXSize(cellIDs[cellIndex]);
      }else if (_spatialCoord==1){
	h = mesh->getCellYSize(cellIDs[cellIndex]);
      }	
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	values(cellIndex,ptIndex) = sqrt(h);
      }
    }
  }
};
class HSwitch : public hFunction {
  double _minh;
  MeshPtr _mesh;
public:
  HSwitch(double hToSwitchAt,MeshPtr mesh){
    _minh = hToSwitchAt;
    _mesh = mesh;
  }
  double value(double x, double y, double h) {    
    double val = 1.0;
    if (h>_minh){
      //      val = 1.0 - exp(-abs(h-_minh)*10.0);
    }else{
      val = 0.0;
    }

    /*
    // global switch
    MeshInfo meshInfo(_mesh);
    double minSideLength = meshInfo.getMinCellSideLength() ;
    if (minSideLength<_minh){
      val = 0.0;
    }
    */

    return val;
  }
};
class SqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(h);
  }
};
class InvSqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(1.0/h);
  }
};
class InvHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return 1.0/h;
  }
};

class RampTopBoundary : public SpatialFilter{
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol);
    return xMatch;
  }
};

class LeftInflow : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = abs(x)<tol;
    return xMatch;
  }
};

class FreeStreamBoundary : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool topWall = abs(y-1.0)<tol;
    bool bottomWall = (x<=.5) && abs(y)<tol;
    return topWall || bottomWall;
  }
};


class WallBoundary : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool onWall = (x>.5) && (abs(y)<tol);
    return onWall;
  }
};

class WallSmoothBC : public SimpleFunction {
  double _width;
public:
  WallSmoothBC(double width){
    _width = width;
  }
  double value(double x, double y){
    double e = 1.0 + exp(-(x-(.5 + _width)));
    double s= (_width*_width);
    double value = 1.0/(e/s);
    return value;
  }
};

class EnergyErrorFunction : public Function {
  map<int, double> _energyErrorForCell;
public:
  EnergyErrorFunction(map<int, double> energyErrorForCell) : Function(0) {
    _energyErrorForCell = energyErrorForCell;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    for (int i = 0;i<cellIDs.size();i++){
      double energyError = _energyErrorForCell[cellIDs[i]];
      for (int j = 0;j<numPoints;j++){
	values(i,j) = energyError;
      }
    }
  }
};

int main(int argc, char *argv[]) {
 
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
  int nCells = args.Input<int>("--nCells", "num cells",2);  
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  int numPreRefs = args.Input<int>("--numPreRefs","num preemptive adaptive refinements",0);
  int order = args.Input<int>("--order","order of approximation",2);
  double eps = args.Input<double>("--epsilon","diffusion parameter",1e-2);
  double energyThreshold = args.Input<double>("-energyThreshold","energy thresh for adaptivity", .5);
  double rampHeight = args.Input<double>("--rampHeight","ramp height at x = 2", 0.0);
  double ipSwitch = args.Input<double>("--ipSwitch","point at which to switch to graph norm", 0.0); // default to 0 to remain on robust norm
  bool useAnisotropy = args.Input<bool>("--useAnisotropy","aniso flag ", false);

  int H1Order = order+1; 
  int pToAdd = args.Input<int>("--pToAdd","test space enrichment", 2);

  FunctionPtr zero = Function::constant(0.0);
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  vector<double> e1,e2;
  e1.push_back(1.0);e1.push_back(0.0);
  e2.push_back(0.0);e2.push_back(1.0);

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  // first order term with magnitude alpha
  double alpha = 0.0;
  //  confusionBF->addTerm(alpha * u, v);

  ////////////////////   BUILD MESH   ///////////////////////


  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));  
  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);

  /*
   // robust test norm
  FunctionPtr C_h = Teuchos::rcp( new EpsilonScaling(eps) );  
  FunctionPtr invH = Teuchos::rcp(new InvHScaling);
  FunctionPtr invSqrtH = Teuchos::rcp(new InvSqrtHScaling);
  FunctionPtr sqrtH = Teuchos::rcp(new SqrtHScaling);
  FunctionPtr hSwitch = Teuchos::rcp(new HSwitch(ipSwitch,mesh));
  ip->addTerm(hSwitch*sqrt(eps) * v->grad() );
  ip->addTerm(hSwitch*beta * v->grad() );
  ip->addTerm(hSwitch*tau->div() );
  
  // graph norm
  ip->addTerm( (one-hSwitch)*((1.0/eps) * tau + v->grad()));
  ip->addTerm( (one-hSwitch)*(beta * v->grad() - tau->div()));

  // regularizing terms
  ip->addTerm(C_h/sqrt(eps) * tau );    
  ip->addTerm(invSqrtH*v);
  */

   // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr C_h = Teuchos::rcp( new EpsilonScaling(eps) );  
  FunctionPtr invH = Teuchos::rcp(new InvHScaling);
  FunctionPtr invSqrtH = Teuchos::rcp(new InvSqrtHScaling);
  FunctionPtr sqrtH = Teuchos::rcp(new SqrtHScaling);
  FunctionPtr hSwitch = Teuchos::rcp(new HSwitch(ipSwitch,mesh));
  robIP->addTerm(sqrt(eps) * v->grad() );
  robIP->addTerm(beta * v->grad() );
  robIP->addTerm(tau->div() );
  // regularizing terms
  robIP->addTerm(C_h/sqrt(eps) * tau );    
  robIP->addTerm(invSqrtH*v);

  IPPtr graphIP = confusionBF->graphNorm();
  graphIP->addTerm(invSqrtH*v);
  //  graphIP->addTerm(C_h/sqrt(eps) * tau );    
  IPPtr switchIP = Teuchos::rcp(new IPSwitcher(robIP,graphIP,ipSwitch)); // rob IP for h>ipSwitch mesh size, graph norm o/w
  ip = switchIP;
    
  LinearTermPtr vVecLT = Teuchos::rcp(new LinearTerm);
  LinearTermPtr tauVecLT = Teuchos::rcp(new LinearTerm);
  vVecLT->addTerm(sqrt(eps)*v->grad());
  tauVecLT->addTerm(C_h/sqrt(eps)*tau);

  LinearTermPtr restLT = Teuchos::rcp(new LinearTerm);
  restLT->addTerm(alpha*v);
  restLT->addTerm(invSqrtH*v);
  restLT = restLT + beta * v->grad();
  restLT = restLT + tau->div();

  ////////////////////   SPECIFY RHS   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  //  f = one;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  SpatialFilterPtr Inflow = Teuchos::rcp(new LeftInflow);
  SpatialFilterPtr wallBoundary = Teuchos::rcp(new WallBoundary);//MeshUtilities::rampBoundary(rampHeight);
  SpatialFilterPtr freeStream = Teuchos::rcp(new FreeStreamBoundary);

  bc->addDirichlet(uhat, wallBoundary, one);
  //  bc->addDirichlet(uhat, wallBoundary, Teuchos::rcp(new WallSmoothBC(eps)));
  bc->addDirichlet(beta_n_u_minus_sigma_n, Inflow, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, freeStream, zero);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  mesh->registerSolution(backgroundFlow); // to trigger issue with p-refinements
  map<int, Teuchos::RCP<Function> > functionMap; functionMap[u->ID()] = Function::constant(3.14);
  backgroundFlow->projectOntoMesh(functionMap);

  // lower p to p = 1 at SINGULARITY only
  vector<int> ids;
  /*
  for (int i = 0;i<mesh->numActiveElements();i++){
    bool cellIDset = false;
    int cellID = mesh->activeElements()[i]->cellID();
    int elemOrder = mesh->cellPolyOrder(cellID)-1;
    FieldContainer<double> vv(4,2); mesh->verticesForCell(vv, cellID);
    bool vertexOnWall = false; bool vertexAtSingularity = false;
    for (int j = 0;j<4;j++){
      if ((abs(vv(j,0)-.5) + abs(vv(j,1)))<1e-10){
	vertexAtSingularity = true;     
	cellIDset = true;
      }
    }	
    if (!vertexAtSingularity && elemOrder<2 && !cellIDset ){
      ids.push_back(cellID);
      cout << "celliD = " << cellID << endl;
    }
  }
  */
  ids.push_back(1);
  ids.push_back(3);
  mesh->pRefine(ids); // to put order = 1

  return 0;
  
  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution));  
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  rieszResidual->computeRieszRep();
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,rieszResidual));
  FunctionPtr e_tau = Teuchos::rcp(new RepFunction(tau,rieszResidual));
  map<int,FunctionPtr> errRepMap;
  errRepMap[v->ID()] = e_v;
  errRepMap[tau->ID()] = e_tau;
  FunctionPtr errTau = tauVecLT->evaluate(errRepMap,false);
  FunctionPtr errV = vVecLT->evaluate(errRepMap,false);
  FunctionPtr errRest = restLT->evaluate(errRepMap,false);
  FunctionPtr xErr = (errTau->x())*(errTau->x()) + (errV->dx())*(errV->dx());
  FunctionPtr yErr = (errTau->y())*(errTau->y()) + (errV->dy())*(errV->dy());
  FunctionPtr restErr = errRest*errRest;

  RefinementStrategy refinementStrategy( solution, energyThreshold );    

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                     PRE REFINEMENTS 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  

  if (rank==0){
    cout << "Number of pre-refinements = " << numPreRefs << endl;
  }
  for (int i =0;i<=numPreRefs;i++){   
    vector<ElementPtr> elems = mesh->activeElements();
    vector<ElementPtr>::iterator elemIt;
    vector<int> wallCells;    
    for (elemIt=elems.begin();elemIt != elems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      int numSides = mesh->getElement(cellID)->numSides();
      FieldContainer<double> vertices(numSides,2); //for quads

      mesh->verticesForCell(vertices, cellID);
      bool cellIDset = false;	
      for (int j = 0;j<numSides;j++){ 	
	if ((abs(vertices(j,0)-.5)<1e-7) && (abs(vertices(j,1))<1e-7) && !cellIDset){ // if at singularity, i.e. if a vertex is (1,0)
	  wallCells.push_back(cellID);
	  cellIDset = true;
	}
      }
    }
    if (i<numPreRefs){
      refinementStrategy.refineCells(wallCells);
    }
  }

  double minSideLength = meshInfo.getMinCellSideLength() ;
  double minCellMeasure = meshInfo.getMinCellMeasure() ;
  if (rank==0){
    cout << "after prerefs, sqrt min cell measure = " << sqrt(minCellMeasure) << ", min side length = " << minSideLength << endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VTKExporter exporter(solution, mesh, varFactory);

  for (int refIndex=0;refIndex<numRefs;refIndex++){
    if (rank==0){
      cout << "on ref index " << refIndex << endl;
    }    
    rieszResidual->computeRieszRep(); // in preparation to get anisotropy    

    vector<int> cellIDs;
    refinementStrategy.getCellsAboveErrorThreshhold(cellIDs);

    map<int,double> energyError = solution->energyError();  

    map<int,double> xErrMap = xErr->cellIntegrals(cellIDs,mesh,5,true);
    map<int,double> yErrMap = yErr->cellIntegrals(cellIDs,mesh,5,true);
    map<int,double> restErrMap = restErr->cellIntegrals(cellIDs,mesh,5,true);    
    for (vector<ElementPtr>::iterator elemIt = mesh->activeElements().begin();elemIt!=mesh->activeElements().end();elemIt++){
      int cellID = (*elemIt)->cellID();
      double err = xErrMap[cellID]+ yErrMap[cellID] + restErrMap[cellID];
      //      if (rank==0)
	//      cout << "err thru LT = " << sqrt(err) << ", while energy err = " << energyError[cellID] << endl;
    }

    /*
    map<int,double> ratio,xErr,yErr;
    vector<ElementPtr> elems = mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      ratio[cellID] = 0.0;
      xErr[cellID] = 0.0;
      yErr[cellID] = 0.0;
      if (std::find(cellIDs.begin(),cellIDs.end(),cellID)!=cellIDs.end()){ // if this cell is above energy thresh
	ratio[cellID] = yErrMap[cellID]/xErrMap[cellID];
	xErr[cellID] = xErrMap[cellID];
	yErr[cellID] = yErrMap[cellID];
      }
    }   
    FunctionPtr ratioFxn = Teuchos::rcp(new EnergyErrorFunction(ratio));
    FunctionPtr xErrFxn = Teuchos::rcp(new EnergyErrorFunction(xErr));
    FunctionPtr yErrFxn = Teuchos::rcp(new EnergyErrorFunction(yErr));
    exporter.exportFunction(ratioFxn, string("ratio")+oss.str());
    exporter.exportFunction(xErrFxn, string("xErr")+oss.str());
    exporter.exportFunction(yErrFxn, string("yErr")+oss.str());
    */
    if (useAnisotropy){
      refinementStrategy.refine(rank==0,xErrMap,yErrMap); //anisotropic refinements
    }else{
      refinementStrategy.refine(rank==0); // no anisotropy
    }

    // lower p to p = 1 at SINGULARITY only
    vector<int> ids;
    for (int i = 0;i<mesh->numActiveElements();i++){
      int cellID = mesh->activeElements()[i]->cellID();
      int elemOrder = mesh->cellPolyOrder(cellID)-1;
      FieldContainer<double> vv(4,2); mesh->verticesForCell(vv, cellID);
      bool vertexOnWall = false; bool vertexAtSingularity = false;
      for (int j = 0;j<4;j++){
	if ((abs(vv(j,0)-.5) + abs(vv(j,1)))<1e-10)
	  vertexAtSingularity = true;
      }	
      if (!vertexAtSingularity && elemOrder<2){
	ids.push_back(cellID);
      }
    }
    mesh->pRefine(ids); // to put order = 1
    /*
      if (elemOrder>1){
	if (vertexAtSingularity){
	  vector<int> ids;
	  ids.push_back(cellID);
	  mesh->pRefine(ids,1-(elemOrder-1)); // to put order = 1
	  //	  mesh->pRefine(ids); // to put order = 1
	  if (rank==0)
	    cout << "p unrefining elem with elemOrder = " << elemOrder << endl;
	}
      }else{
	if (!vertexAtSingularity){
	  vector<int> ids;
	  ids.push_back(cellID);	    
	  mesh->pRefine(ids,2-elemOrder);
	}	  
      }
      */



    double minSideLength = meshInfo.getMinCellSideLength() ;
    if (rank==0)
      cout << "minSideLength is " << minSideLength << endl;

    solution->condensedSolve();
    std::ostringstream oss;
    oss << refIndex;
    
  }

  // final solve on final mesh
  solution->setWriteMatrixToFile(true,"K.mat");
  solution->condensedSolve();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                          CHECK CONDITIONING 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  bool checkConditioning = true;
  if (checkConditioning){
    double minSideLength = meshInfo.getMinCellSideLength() ;
    StandardAssembler assembler(solution);
    double maxCond = 0.0;
    int maxCellID = 0;
    for (int i = 0;i<mesh->numActiveElements();i++){
      int cellID = mesh->getActiveElement(i)->cellID();
      FieldContainer<double> ipMat = assembler.getIPMatrix(mesh->getElement(cellID));
      double cond = SerialDenseWrapper::getMatrixConditionNumber(ipMat);
      if (cond>maxCond){
	maxCond = cond;
	maxCellID = cellID;
      }
    }
    if (rank==0){
      cout << "cell ID  " << maxCellID << " has minCellLength " << minSideLength << " and condition estimate " << maxCond << endl;
    }
    string ipMatName = string("ipMat.mat");
    ElementPtr maxCondElem = mesh->getElement(maxCellID);
    FieldContainer<double> ipMat = assembler.getIPMatrix(maxCondElem);
    SerialDenseWrapper::writeMatrixToMatlabFile(ipMatName,ipMat);   
  }
  ////////////////////   print to file   ///////////////////////
  
  if (rank==0){
    exporter.exportSolution(string("robustIP"));
    cout << endl;
  }
 
  return 0;
} 


