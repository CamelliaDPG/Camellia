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

class MeshInfo
{
  MeshPtr _mesh;
public:
  MeshInfo(MeshPtr mesh)
  {
    _mesh = mesh;
  }
  double getMinCellMeasure()
  {
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
    {
      minMeasure = min(minMeasure, _mesh->getCellMeasure((*elemIt)->cellID()));
    }
    return minMeasure;
  }
  vector<int> getMinCellSizeCellIDs()
  {
    double minMeasure = getMinCellMeasure();
    vector<int> minMeasureCellIDs;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
    {
      if (minMeasure <= _mesh->getCellMeasure((*elemIt)->cellID()))
      {
        minMeasureCellIDs.push_back((*elemIt)->cellID());
      }
    }
    return minMeasureCellIDs;
  }
  double getMinCellSideLength()
  {
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin(); elemIt!=elems.end(); elemIt++)
    {
      minMeasure = min(minMeasure, _mesh->getCellXSize((*elemIt)->cellID()));
      minMeasure = min(minMeasure, _mesh->getCellYSize((*elemIt)->cellID()));
    }
    return minMeasure;
  }
};
// =============================================================

class EpsilonScaling : public hFunction
{
  double _epsilon;
public:
  EpsilonScaling(double epsilon)
  {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h)
  {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class AnisotropicHScaling : public Function
{
  int _spatialCoord;
public:
  AnisotropicHScaling(int spatialCoord)
  {
    _spatialCoord = spatialCoord;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    MeshPtr mesh = basisCache->mesh();
    vector<int> cellIDs = basisCache->cellIDs();

    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      double h = 1.0;
      if (_spatialCoord==0)
      {
        h = mesh->getCellXSize(cellIDs[cellIndex]);
      }
      else if (_spatialCoord==1)
      {
        h = mesh->getCellYSize(cellIDs[cellIndex]);
      }
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        values(cellIndex,ptIndex) = sqrt(h);
      }
    }
  }
};
class HSwitch : public hFunction
{
  double _minh;
  MeshPtr _mesh;
public:
  HSwitch(double hToSwitchAt,MeshPtr mesh)
  {
    _minh = hToSwitchAt;
    _mesh = mesh;
  }
  double value(double x, double y, double h)
  {
    double val = 1.0;
    if (h>_minh)
    {
      //      val = 1.0 - exp(-abs(h-_minh)*10.0);
    }
    else
    {
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
class SqrtHScaling : public hFunction
{
public:
  double value(double x, double y, double h)
  {
    return sqrt(h);
  }
};
class InvSqrtHScaling : public hFunction
{
public:
  double value(double x, double y, double h)
  {
    return sqrt(1.0/h);
  }
};
class InvHScaling : public hFunction
{
public:
  double value(double x, double y, double h)
  {
    return 1.0/h;
  }
};

class RampTopBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol);
    return xMatch;
  }
};

class LeftInflow : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = abs(x)<tol;
    return xMatch;
  }
};

class FreeStreamBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool topWall = abs(y-1.0)<tol;
    bool bottomWall = (x<=.5) && abs(y)<tol;
    return topWall || bottomWall;
  }
};


class WallBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool onWall = (x>.5) && (abs(y)<tol);
    return onWall;
  }
};

class WallSmoothBC : public SimpleFunction
{
  double _width;
public:
  WallSmoothBC(double width)
  {
    _width = width;
  }
  double value(double x, double y)
  {
    double e = 1.0 + exp(-(x-(.5 + _width)));
    double s= (_width*_width);
    double value = 1.0/(e/s);
    return value;
  }
};

class EnergyErrorFunction : public Function
{
  map<int, double> _energyErrorForCell;
public:
  EnergyErrorFunction(map<int, double> energyErrorForCell) : Function(0)
  {
    _energyErrorForCell = energyErrorForCell;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    for (int i = 0; i<cellIDs.size(); i++)
    {
      double energyError = _energyErrorForCell[cellIDs[i]];
      for (int j = 0; j<numPoints; j++)
      {
        values(i,j) = energyError;
      }
    }
  }
};

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  int nCells = args.Input<int>("--nCells", "num cells",1);
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  int numPreRefs = args.Input<int>("--numPreRefs","num preemptive adaptive refinements",0);
  int order = args.Input<int>("--order","order of approximation",2);
  double eps = args.Input<double>("--epsilon","diffusion parameter",1e-2);
  double energyThreshold = args.Input<double>("-energyThreshold","energy thresh for adaptivity", .5);
  bool useAnisotropy = args.Input<bool>("--useAnisotropy","aniso flag ", false);

  int H1Order = order+1;
  int pToAdd = args.Input<int>("--pToAdd","test space enrichment", 2);

  FunctionPtr zero = Function::constant(0.0);
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  vector<double> e1,e2;
  e1.push_back(1.0);
  e1.push_back(0.0);
  e2.push_back(0.0);
  e2.push_back(1.0);

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

  ////////////////////   BUILD MESH   ///////////////////////

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  robIP->addTerm(v->grad() );
  robIP->addTerm(tau->div() );
  robIP->addTerm(tau );
  robIP->addTerm(v);

  IPPtr xSemi = Teuchos::rcp(new IP);
  IPPtr ySemi = Teuchos::rcp(new IP);
  IPPtr restIP = Teuchos::rcp(new IP);

  xSemi->addTerm(v->dx());
  xSemi->addTerm(tau->x());

  ySemi->addTerm(v->dy());
  ySemi->addTerm(tau->y());

  restIP->addTerm(v);
  restIP->addTerm(tau->div());

  ////////////////////   SPECIFY RHS   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  SpatialFilterPtr Inflow = Teuchos::rcp(new LeftInflow);
  SpatialFilterPtr wallBoundary = Teuchos::rcp(new WallBoundary);
  SpatialFilterPtr freeStream = Teuchos::rcp(new FreeStreamBoundary);

  bc->addDirichlet(uhat, wallBoundary, one);
  bc->addDirichlet(beta_n_u_minus_sigma_n, Inflow, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, freeStream, zero);

  ////////////////////   SOLVE   ///////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution->condensedSolve();

  ////////////////////   CHECK ERROR   ///////////////////////

  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution));
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, robIP, residual));
  rieszResidual->computeRieszRep();

  ElementPtr elem = mesh->activeElements()[0]; // assume one cell
  double xnormsq = rieszResidual->computeAlternativeNormSqOnCell(xSemi, elem);
  double ynormsq = rieszResidual->computeAlternativeNormSqOnCell(ySemi, elem);
  double restnormsq = rieszResidual->computeAlternativeNormSqOnCell(restIP, elem);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  double rieszErr = rieszResidual->getNorm();
  double energyErr = solution->energyErrorTotal();


  if (rank==0)
  {
    cout << "riesz err = " << rieszErr*rieszErr << ", while energy err = " << energyErr*energyErr << endl;
    cout << "xErr = " << xnormsq << ", yErr = " << ynormsq << ", restErr = " << restnormsq << ", sum = " << xnormsq + ynormsq + restnormsq << endl;
  }


  return 0;
}


