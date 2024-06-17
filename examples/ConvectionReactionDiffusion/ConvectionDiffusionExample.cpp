//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BasisFactory.h"
#include "ConvectionDiffusionReactionFormulation.h"
#include "EnergyErrorFunction.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

// Tpetra includes
#include <MatrixMarket_Tpetra.hpp>
// EpetraExt includes
#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"

typedef double Scalar;

using namespace Camellia;
using namespace std;

enum ExactSolutionChoice {
  BroersenStevensonExample4_2,
  LinearPolynomial,
  QuadraticPolynomial,
  CubicPolynomial
};

/*
 Example 4.2 from Broersen and Stevenson:
 
 Manufactured solution:
 
 u(x,y) = g(b_1,x) * g(b_2,y)
 
 where
 
 g(a,z) = [z + (e^(a * z / epsilon) - 1) / (1 - e^(a / epsilon)]
 
 and convective direction beta = (b_1,b_2).
 
 We solve this on domain [0,1]^2.
 
 */

const static int INFLOW_TAG_ID  = 0; // arbitrary, just an identifier
const static int OUTFLOW_TAG_ID = 1;

using namespace Teuchos;
FunctionPtr exp(FunctionPtr power)
{
  return rcp(new Exp<double>(power));
}

template<class WeightType> // WeightType can be FunctionPtr or double
class ExpTerm : public SimpleFunction<double>
{
  WeightType _weight; // beta_1 / epsilon or beta_2 / epsilon
  bool _xTerm; // alternative is y-term; determines which of x or y enters the exponent
  WeightType _denominator;
public:
  ExpTerm(WeightType weight, bool xTerm)
  :
  SimpleFunction<double>("ConvectionDiffusionExample-ExpTerm", VARIES_IN_DIMENSION_0 | VARIES_IN_DIMENSION_1)
  {
    _weight = weight;
    _xTerm = xTerm;
    _denominator = exp(-weight) - 1.0;
  }
  
  double value(double x, double y);
  
  FunctionPtr dx()
  {
    if (_xTerm)
    {
      // derivative will be this function times _weight without the constant
      //  - exp(-_weight)) / _denominator
      FunctionPtr thisFxn = Teuchos::rcp( new ExpTerm(_weight,_xTerm));
      WeightType constPart = - exp(-_weight) / _denominator;
      return _weight * (thisFxn - constPart);
    }
    else
    {
      return Function::zero();
    }
  }
  
  FunctionPtr dy()
  {
    if (!_xTerm)
    {
      // derivative will be this function times _weight without the constant
      //  - exp(-_weight)) / _denominator
      FunctionPtr thisFxn = Teuchos::rcp( new ExpTerm(_weight,_xTerm));
      WeightType constPart = - exp(-_weight) / _denominator;
      return _weight * (thisFxn - constPart);
    }
    else
    {
      return Function::zero();
    }
  }
};

template<> // WeightType can be FunctionPtr or double
double ExpTerm<double>::value(double x, double y)
{
  double arg = _xTerm ? x : y;
  return (exp(_weight * (arg - 1)) - exp(-_weight)) / _denominator;
}

template<> // WeightType can be FunctionPtr or double
double ExpTerm<FunctionPtr>::value(double x, double y)
{
  double arg = _xTerm ? x : y;
  FunctionPtr valueFxn = (exp(_weight * (arg - 1)) - exp(-_weight)) / _denominator;
  return valueFxn->evaluate(x,y);
}

class CheckerboardFunction : public SimpleFunction<double>
{
  double _interval; // beta_1 / epsilon or beta_2 / epsilon
  double _lowerLeftValue;
  double _otherValue;
public:
  CheckerboardFunction(double interval, double lowerLeftValue, double otherValue)
  :
  SimpleFunction<double>("ConvectionDiffusionExample-CheckerboardFunction", VARIES_IN_DIMENSION_0 | VARIES_IN_DIMENSION_1),
  _interval(interval), _lowerLeftValue(lowerLeftValue), _otherValue(otherValue)
  {}
  
  double value(double x, double y)
  {
    // how many intervals are we from 0 in x,y? (rounding down)
    int intervalCount_x = (int) x / _interval;
    int intervalCount_y = (int) y / _interval;
    // if counts are positive, then the parity tells us whether we're at the lower-left value (even in x, even in y: yes, etc.)
    // each negative count has exactly the opposite value: so we add 1 if counts are negative
    if (intervalCount_x < 0) intervalCount_x++;
    if (intervalCount_y < 0) intervalCount_y++;
    int totalCount = intervalCount_x + intervalCount_y;
    bool useLowerLeftValue = (totalCount % 2 == 0);
    if (useLowerLeftValue)
    {
      return _lowerLeftValue;
    }
    else
    {
      return _otherValue;
    }
  }
  
  // zero derivative everywhere *except* at the checkerboard interfaces
  // but I'm unclear on whether we should allow derivatives to be taken, so for
  // now we leave these undefined (with the effect that BroersenStevenson is not allowed
  // as an exact solution)
//  FunctionPtr dx()
//  {
//    return Function::zero();
//  }
//  
//  FunctionPtr dy()
//  {
//    return Function::zero();
//  }
};

template<class EpsilonType>
FunctionPtr exactSolution(ExactSolutionChoice solnChoice, EpsilonType epsilon, double beta_1, double beta_2)
{
  switch (solnChoice)
  {
    case BroersenStevensonExample4_2:
    {
      // example 4.2
      
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      
      EpsilonType x_beta_epsilon = beta_1 / epsilon;
      EpsilonType y_beta_epsilon = beta_2 / epsilon;
      
      FunctionPtr xTerm, yTerm;
      
      if (beta_1 != 0)
      {
        xTerm = x + (FunctionPtr) Teuchos::rcp( new ExpTerm<EpsilonType>(x_beta_epsilon, true));
      }
      else
      {
        xTerm = x;
      }
      if (beta_2 != 0)
      {
        yTerm = y + (FunctionPtr) Teuchos::rcp( new ExpTerm<EpsilonType>(y_beta_epsilon, false));
      }
      else
      {
        yTerm = y;
      }
      
      return xTerm * yTerm;
    }
      break;
    case LinearPolynomial:
    {
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      
      return x + 3 * y;
    }
      break;
    case QuadraticPolynomial:
    {
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      
      return x + x * y + 3 * y * y;
    }
      break;
    case CubicPolynomial:
    {
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      
      return x * x * x + x * y + 3 * y * y;
    }
    break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported case");
  }
}

class CellIndicatorFunction : public TFunction<double>
{
  std::set<GlobalIndexType> _cellIDs;
public:
  CellIndicatorFunction(GlobalIndexType cellID) : TFunction<double>(0)
  {
    _cellIDs.insert(cellID);
  }
  CellIndicatorFunction(std::vector<GlobalIndexType> &cellIDs) : TFunction<double>(0)
  {
    _cellIDs.insert(cellIDs.begin(),cellIDs.end());
  }
  
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    IndexType cellIndex = 0;
    int numPoints = values.dimension(1);
    for (GlobalIndexType cellID : cellIDs)
    {
      double value;
      if (_cellIDs.find(cellID) == _cellIDs.end())
        value = 0;
      else
        value = 1;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        values(cellIndex, ptIndex) = value;
      }
      cellIndex++;
    }
  }
};

template <typename Scalar>
class CustomErrorIndicator : public ErrorIndicator
{
public:
  CustomErrorIndicator(MeshPtr mesh) : ErrorIndicator(mesh) {}
  
  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {
    _localErrorMeasures.clear();
    const set<GlobalIndexType>* myCells = &_mesh->cellIDsInPartition();
    int spaceDim = _mesh->getTopology()->getDimension();
    for (auto cellID : *myCells)
    {
      double h =  pow(_mesh->getCellMeasure(cellID), 1.0 / spaceDim);
      
      CellPtr cell = _mesh->getTopology()->getCell(cellID);
      IndexType topRightVertex = cell->vertices()[2];
      
      double x = _mesh->getTopology()->getVertex(topRightVertex)[0];
      double y = _mesh->getTopology()->getVertex(topRightVertex)[1];
      
      double tol = 1e-10;
      if ((abs(1 - x) < tol) || (abs(1-y) < tol))
      {
        _localErrorMeasures[cellID] = h;
      }
      else
      {
        _localErrorMeasures[cellID] = 0.0;
      }
    }
  }
};

template <typename Scalar>
class MeshMatchingErrorIndicator : public ErrorIndicator
{
  MeshTopologyViewPtr _meshToMatch;
public:
  MeshMatchingErrorIndicator(MeshPtr mesh, MeshTopologyViewPtr meshToMatch) : ErrorIndicator(mesh)
  {
    _meshToMatch = meshToMatch->getGatheredCopy();
  }
  
  void setMeshToMatch(MeshTopologyViewPtr meshToMatch)
  {
    _meshToMatch = meshToMatch->getGatheredCopy();
  }
  
  //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
  virtual void measureError()
  {
    _localErrorMeasures.clear();
    
    // any cell that we own that is not active in meshToMatch should be refined
    
    const set<GlobalIndexType>* myCells = &_mesh->cellIDsInPartition();
    
    vector<vector<double>> myCentroids;
    for (auto cellID : *myCells)
    {
      myCentroids.push_back(_mesh->getTopology()->getCellCentroid(cellID));
    }
    vector<IndexType> equivalentCellIDs = _meshToMatch->cellIDsWithCentroids(myCentroids);
    
    int cellOrdinal = 0;
    for (auto cellID : *myCells)
    {
      CellPtr cellToMatch = _meshToMatch->getCell(equivalentCellIDs[cellOrdinal++]);
      if (cellToMatch->isParent(_meshToMatch))
      {
        _localErrorMeasures[cellID] = 1.0;
      }
      else
      {
        _localErrorMeasures[cellID] = 0.0;
      }
    }
  }
};

double conditionNumberLAPACK(const Epetra_RowMatrix &stiffnessMatrix)
{
  Camellia::FieldContainer<double> A;
  SerialDenseWrapper::extractFCFromEpetra_RowMatrix(stiffnessMatrix, A);
  
  bool use2norm = false;
  
  int rank = stiffnessMatrix.Comm().MyPID();
  
  double condest;
  if (rank == 0)
  {
    if (use2norm)
    {
      Camellia::FieldContainer<double> lambda_real(A.dimension(0)), lambda_imag(A.dimension(0));
      SerialDenseWrapper::eigenvalues(A, lambda_real, lambda_imag);
      double max_eig = 0.0, min_eig = 1e300;
      for (int i=0; i<lambda_real.size(); i++)
      {
        double real_part = lambda_real(i), imag_part = lambda_imag(i);
        double eig_mag = sqrt(real_part * real_part + imag_part * imag_part);
        max_eig = max(eig_mag, max_eig);
        min_eig = min(eig_mag, min_eig);
      }
      //  cout << "A:\n" << A;
      //  cout << "max_eig: " << max_eig << endl;
      //  cout << "min_eig: " << min_eig << endl;
      condest = max_eig / min_eig;
    }
    else // 1-norm
    {
      condest = SerialDenseWrapper::condest(A);
    }
  }
  stiffnessMatrix.Comm().Broadcast(&condest, 1, 0);
  return condest;
}

void writeEigenValues(const Epetra_RowMatrix &stiffnessMatrix, string filename)
{
  Camellia::FieldContainer<double> A;
  SerialDenseWrapper::extractFCFromEpetra_RowMatrix(stiffnessMatrix, A);
  
  int rank = stiffnessMatrix.Comm().MyPID();
  if (rank == 0)
  {
    Camellia::FieldContainer<double> lambda_real(A.dimension(0)), lambda_imag(A.dimension(0));
    SerialDenseWrapper::eigenvalues(A, lambda_real, lambda_imag);
    
    ofstream fout(filename);
    
    for (int i=0; i<lambda_real.size(); i++)
    {
      double real_part = lambda_real(i), imag_part = lambda_imag(i);
      fout << real_part << "\t" << imag_part << endl;
    }
    fout.close();
  }
}

double h_mean(MeshPtr mesh)
{
  FunctionPtr h = Function::h();
  const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
  double my_hMax = 0, my_hMin = 1e10;
  for (GlobalIndexType cellID : *myCells)
  {
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    int numCells = 1;
    int numPoints = basisCache->getRefCellPointsFC().dimension(1);
    Camellia::FieldContainer<double> values(numCells,numPoints);
    h->values(values, basisCache);
    my_hMax = max(values[0],my_hMax);
    my_hMin = min(values[0],my_hMin);
  }
  double hMax, hMin;
  mesh->Comm()->MaxAll(&my_hMax, &hMax, 1);
  mesh->Comm()->MinAll(&my_hMin, &hMin, 1);
  double hMean = sqrt(hMax * hMin);
  int rank = mesh->Comm()->MyPID();
  if (rank == 0) cout << "hMean: " << hMean << endl;
  return hMean;
}

MeshTopologyViewPtr loadMesh(string meshLoadPrefix, int refinementNumber)
{
  ostringstream meshFileName;
  meshFileName << meshLoadPrefix << refinementNumber;
  return MeshTopologyView::readFromFile(MPIWrapper::CommWorld(), meshFileName.str());
}

void saveMesh(MeshPtr mesh, string meshSavePrefix, int refinementNumber)
{
  ostringstream meshFileName;
  meshFileName << meshSavePrefix << refinementNumber;
  return mesh->getTopology()->writeToFile(meshFileName.str());
}

bool meshesMatch(MeshPtr mesh, MeshTopologyViewPtr meshTopoView)
{
  int rank = mesh->Comm()->MyPID();
  int meshElements = mesh->numElements();
  int meshTopoElements = meshTopoView->cellCount();
  if (meshElements != meshTopoElements)
  {
    if (rank == 0) cout << "mesh has " << meshElements << " elements; meshTopoView has " << meshTopoElements << endl;
    return false;
  }
  
  const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : *myCells)
  {
    CellPtr myCell = mesh->getTopology()->getCell(cellID);
    vector<double> myCentroid = mesh->getTopology()->getCellCentroid(cellID);
    GlobalIndexType otherCellID = meshTopoView->cellIDsWithCentroids({myCentroid})[0];
    CellPtr otherCell = meshTopoView->getCell(otherCellID);
    // vertex points should be identical, and in the same order
    vector<vector<double>> myVertices, otherVertices;
    vector<IndexType> myVertexIndices = myCell->vertices();
    vector<IndexType> otherVertexIndices = otherCell->vertices();
    for (IndexType myVertexIndex : myVertexIndices)
    {
      myVertices.push_back(mesh->getTopology()->getVertex(myVertexIndex));
    }
    for (IndexType otherVertexIndex : otherVertexIndices)
    {
      otherVertices.push_back(meshTopoView->getVertex(otherVertexIndex));
    }
    double tol = 1e-14;
    for (int i=0; i<myVertices.size(); i++)
    {
      for (int d=0; d<myVertices[i].size(); d++)
      {
        double diff = abs(myVertices[i][d]-otherVertices[i][d]);
        if (diff > tol)
        {
          cout << "cellID " << cellID << " vertices differ on mesh and meshTopoView.\n";
        }
      }
    }
  }

  
  // TODO: add more thorough checking here
  return true;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  MPIWrapper::CommWorld()->Barrier(); // can set a breakpoint here for debugger attachment
  
  int rank = mpiSession.getRank();
  
  // set defaults to match Broersen and Stevenson's experiment
  int meshWidth = 16;
  int polyOrder = 1, delta_k = 1;
  int spaceDim = 2;
  double epsilon = 1e-3;
  double beta_1 = 2.0, beta_2 = 1.0;
  bool useTriangles = true; // otherwise, quads
  int numRefinements = 0;
  double energyThreshold = 0.2; // for refinements
  string formulationChoice = "Ultraweak";
  bool computeLAPACKConditionNumber = false;
  bool exportMatricesLastRefinement = false;
  bool exportVisualization = false;
  bool reportRefinedCells = false;
  bool reportSolutionTimings = true;
  int quadratureEnrichment = 0;
  int quadratureEnrichmentL2 = 10;
  bool useDirectSolver = false;
  bool useCondensedSolve = false;
  bool useNodalBasis = true;
  bool exportMatrix = false;
  bool refineUniformly = false;
  bool writeEigValues = false;
  int maxIterations = 2000;
  int iterativeOutputLevel = 100;
  double cgTol = 1e-6;
  int smootherApplicationCount = 1;
  double weightForL2TermsGraphNorm = -1;
  bool useGMRESForDPG = false;
  bool usePointSymmetricGSForDPG = false;
  bool useCustomMeshRefinement = false;
  bool useCheckerboardEpsilon = false;
  double checkerboardLargeEpsilon = 1.0;
  double checkerboardAlternationInterval = 0.5;
  double refErrThreshold = 0; // when error is below threshold, stop refinements
  string meshSavePrefix = "";
  string meshLoadPrefix = "";
  double alpha = 0; // no reaction term
  
  // track the following for each refinement:
  vector<double> indicatorMeasures;
  vector<double> L2errors;
  vector<double> bestL2errors;
  vector<double> condNumbers;
  vector<int> activeElementCounts;
  vector<int> dofCounts;
  vector<double> assemblyTimes;
  vector<double> solveTimes;
  string exactSolnChoiceString = "BroersenStevenson";
  
  cmdp.setOption("meshWidth", &meshWidth );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("spaceDim", &spaceDim);
  cmdp.setOption("alpha", &alpha);
  cmdp.setOption("epsilon", &epsilon);
  cmdp.setOption("exportMatrix", "dontExportMatrix", &exportMatrix);
  cmdp.setOption("beta_1", &beta_1);
  cmdp.setOption("beta_2", &beta_2);
  cmdp.setOption("exactSolution",&exactSolnChoiceString,"which exact solution to use; supported values are 'BroersenStevenson', 'Linear','Quadratic','Cubic'");
  
  cmdp.setOption("useCheckerboardEpsilon", "useConstantEpsilon", &useCheckerboardEpsilon, "use discontinuous, checkerboarded epsilon values.  See options checkerboardLargeEpsilon and checkerboardAlternationInterval");
  cmdp.setOption("checkerboardLargeEpsilon", &checkerboardLargeEpsilon, "The second epsilon to use in a checkerboard pattern (the standard epsilon will be used in the lower left corner; the 'large' value will be used in orthogonally adjacent regions).  See options useCheckerboardEpsilon and checkerboardAlternationInterval");
  cmdp.setOption("checkerboardAlternationInterval", &checkerboardAlternationInterval, "The x and y alternation interval for the checkerboard pattern;.  See options useCheckerboardEpsilon and checkerboardLargeEpsilon");
  
//  cmdp.setOption("useHWeightedTraces", "useUnweightedTraces", &useHWeightedTraces);
  cmdp.setOption("useTriangles", "useQuads", &useTriangles);
  cmdp.setOption("useNodalBasis", "useHierarchicalBasis", &useNodalBasis);
  cmdp.setOption("useDirectSolver", "useIterativeSolver", &useDirectSolver);
  cmdp.setOption("numRefinements", &numRefinements);
  cmdp.setOption("quadratureEnrichment", &quadratureEnrichment, "quadrature enrichment for Solution");
  cmdp.setOption("quadratureEnrichmentL2", &quadratureEnrichmentL2);
  cmdp.setOption("energyThreshold", &energyThreshold);
  cmdp.setOption("formulationChoice", &formulationChoice);
  cmdp.setOption("refErrThreshold", &refErrThreshold, "will stop refinements when L^2 error of solution is below refErrThreshold, or numRefinements is reached.");
  cmdp.setOption("refineUniformly", "refineUsingErrorIndicator", &refineUniformly);
  cmdp.setOption("reportRefinedCells", "dontReportRefinedCells", &reportRefinedCells);
  cmdp.setOption("smootherApplicationCount", &smootherApplicationCount, "number of times to apply smoother at each multigrid level");
  cmdp.setOption("computeLAPACKConditionNumber", "dontComputeLAPACKConditionNumber", &computeLAPACKConditionNumber);
  cmdp.setOption("reportSolutionTimings", "dontReportSolutionTimings", &reportSolutionTimings);
  cmdp.setOption("meshLoadPrefix", &meshLoadPrefix, "filename prefix (refinement number will be appended) to use for loading refinements; if non-empty, will use in place of the usual refinement indicator to drive mesh refinements");
  cmdp.setOption("meshSavePrefix", &meshSavePrefix, "filename prefix (refinement number will be appended) to use for saving refinements");
  cmdp.setOption("exportMatricesLastRefinement","dontExportMatricesLastRefinement", &exportMatricesLastRefinement);
  cmdp.setOption("useDirect", "useIterative", &useDirectSolver);
  cmdp.setOption("useGMRESForDPG", "useCGForDPG", &useGMRESForDPG);
  cmdp.setOption("usePointSymmetricGSForDPG", "useSchwarzForDPG", &usePointSymmetricGSForDPG);
  cmdp.setOption("useVisualization","noVisualization",&exportVisualization);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("useCustomMeshRefinement", "useFormulationMeshRefinement", &useCustomMeshRefinement);
  cmdp.setOption("writeEigValues", "dontWriteEigValues", &writeEigValues);
  cmdp.setOption("iterativeOutputLevel", &iterativeOutputLevel, "How many iterations to take before reporting iterative solver progress (0 to suppress output)");
  cmdp.setOption("iterativeTol", &cgTol);
  cmdp.setOption("maxIterations", &maxIterations);
  cmdp.setOption("weightForL2TermsGraphNorm", &weightForL2TermsGraphNorm, "weight to use for the L^2 terms in graph norm (ultraweak only); default value of -1 means use sqrt(epsilon)");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  ExactSolutionChoice exactSolnChoice = BroersenStevensonExample4_2;
  if (exactSolnChoiceString == "BroersenStevenson")
  {
    exactSolnChoice = BroersenStevensonExample4_2;
  }
  else if (exactSolnChoiceString == "Linear")
  {
    exactSolnChoice = LinearPolynomial;
  }
  else if (exactSolnChoiceString == "Quadratic")
  {
    exactSolnChoice = QuadraticPolynomial;
  }
  else if (exactSolnChoiceString == "Cubic")
  {
    exactSolnChoice = CubicPolynomial;
  }
  else
  {
    cout << "Exact solution string '" << exactSolnChoiceString << "' is not supported.\n";
    exit(1);
  }
  
  if (!useNodalBasis)
  {
    BasisFactory::basisFactory()->setUseLegendreForQuadHVol(true);
    BasisFactory::basisFactory()->setUseLobattoForQuadHDiv(true);
    BasisFactory::basisFactory()->setUseLobattoForQuadHGrad(true);
  }
  
  if (weightForL2TermsGraphNorm == -1)
  {
    weightForL2TermsGraphNorm = sqrt(sqrt(epsilon));
  }
  
  ConvectionDiffusionReactionFormulation::FormulationChoice formulation;
  if (formulationChoice == "Ultraweak")
  {
    formulation = ConvectionDiffusionReactionFormulation::ULTRAWEAK;
  }
  else if (formulationChoice == "SUPG")
  {
    formulation = ConvectionDiffusionReactionFormulation::SUPG;
  }
  else if (formulationChoice == "Primal")
  {
    formulation = ConvectionDiffusionReactionFormulation::PRIMAL;
  }
  else if (formulationChoice == "CDPG")
  {
    formulation = ConvectionDiffusionReactionFormulation::CDPG;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported formulation choice!");
  }
  
  bool formulationIsDPG = (formulation == ConvectionDiffusionReactionFormulation::ULTRAWEAK) || (formulation == ConvectionDiffusionReactionFormulation::PRIMAL || (formulation == ConvectionDiffusionReactionFormulation::CDPG));
  
  FunctionPtr epsilonFunction, sqrtEpsilon;
  
  ostringstream thisRunPrefix;
  if (useCheckerboardEpsilon)
  {
    epsilonFunction = rcp( new CheckerboardFunction(checkerboardAlternationInterval,
                                                    epsilon, checkerboardLargeEpsilon) );
    sqrtEpsilon     = rcp( new CheckerboardFunction(checkerboardAlternationInterval,
                                                    sqrt(epsilon), sqrt(checkerboardLargeEpsilon)) );
    thisRunPrefix << formulationChoice << "_checkerboardInterval" << checkerboardAlternationInterval << "_epsLowerLeft" << epsilon << "_epsOther" << checkerboardLargeEpsilon << "_k" << polyOrder << "_";
  }
  else
  {
    epsilonFunction = Function::constant(epsilon);
    sqrtEpsilon = Function::constant(sqrt(epsilon));
    thisRunPrefix << formulationChoice << "_eps" << epsilon << "_k" << polyOrder << "_";
  }

  FunctionPtr u_exact = exactSolution(exactSolnChoice, epsilon, beta_1, beta_2);
//  if (!useCheckerboardEpsilon)
//  {
//    // then prefer using double argument, for performance reasons
//    u_exact = exactSolution(exactSolnChoice, epsilon, beta_1, beta_2);
//  }
//  else
//  {
//    u_exact = exactSolution(exactSolnChoice, epsilonFunction, beta_1, beta_2);
//  }
  
  if (!refineUniformly)
  {
    thisRunPrefix << "AMR_";
  }
  else
  {
    thisRunPrefix << "Uniform_";
  }
  if (alpha != 0.0)
  {
    thisRunPrefix << "alpha" << alpha << "_";
  }
  thisRunPrefix << "meshWidth" << meshWidth << "_" << numRefinements << "refs_";
  
  FunctionPtr beta = Function::constant({beta_1,beta_2});
  ConvectionDiffusionReactionFormulation form(formulation, spaceDim, beta, epsilonFunction, sqrtEpsilon, alpha);
  
  // determine forcing function and RHS
  FunctionPtr f = form.forcingFunction(u_exact);
  RHSPtr rhs = form.rhs(f);
  
  // bilinear form
  BFPtr bf = form.bf();
  
  // inner product
  IPPtr ip, ipForResidual;
  int H1Order = -1;
  if (formulation == ConvectionDiffusionReactionFormulation::ULTRAWEAK)
  {
    ip = bf->graphNorm(weightForL2TermsGraphNorm);
    H1Order = polyOrder + 1;
  }
  if (formulation == ConvectionDiffusionReactionFormulation::PRIMAL)
  {
    H1Order = polyOrder;
    ip = bf->naiveNorm(spaceDim);
  }
  if (formulation == ConvectionDiffusionReactionFormulation::CDPG)
  {
    H1Order = polyOrder;
    ip = bf->graphNorm(weightForL2TermsGraphNorm);
  }
  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    delta_k = 0;
    H1Order = polyOrder;
    ConvectionDiffusionReactionFormulation primalForm(ConvectionDiffusionReactionFormulation::PRIMAL, spaceDim, beta,
                                                      epsilonFunction, sqrtEpsilon, alpha);
    ipForResidual = primalForm.bf()->naiveNorm(spaceDim);
  }
  
  // set up timings printer
  double totalTimeG = 0, totalTimeB = 0, totalTimeT = 0, totalTimeK = 0;
  int totalElementCount = 0;
  std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> optimalTestTimingCallback;
  optimalTestTimingCallback = [&totalTimeG, &totalTimeB, &totalTimeT, &totalTimeK, &totalElementCount]
    (int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType) {
//      cout << "BF timings: on " << numElements << " elements, computed G in " << timeG;
//      cout << " seconds, B in " << timeB << " seconds; solve for T in " << timeT;
//      cout << " seconds; compute K=B^T T in " << timeK << " seconds." << endl;
      totalElementCount += numElements;
      totalTimeG += timeG;
      totalTimeB += timeB;
      totalTimeT += timeT;
      totalTimeK += timeK;
  };
  // same thing, now for RHS computation:
  std::function<void(int numElements, double timeRHS, ElementTypePtr elemType)> rhsTimingCallback;
  double totalTimeRHS = 0;
  int totalRHSElements = 0;
  rhsTimingCallback = [&totalTimeRHS, &totalRHSElements]
  (int numElements, double timeRHS, ElementTypePtr elemType) {
    totalTimeRHS += timeRHS;
    totalRHSElements += numElements;
  };
  // register the timing callbacks
  bf->setOptimalTestTimingCallback(optimalTestTimingCallback);
  bf->setRHSTimingCallback(rhsTimingCallback);
  
  // set up mesh
  vector<double> dimensions = {1.0,1.0};
  vector<int> elementCounts = {meshWidth,meshWidth};

  // we assume that we're loading from ultraweak meshes if we're loading meshes from file
  ConvectionDiffusionReactionFormulation ultraweakForm(ConvectionDiffusionReactionFormulation::ULTRAWEAK, spaceDim, beta,
                                                       epsilonFunction, sqrtEpsilon, alpha);
  BFPtr ultraweakBF = ultraweakForm.bf();
  
  MeshPtr mesh, ultraweakMesh;
  
  int refinementNumber = 0;
  mesh = MeshFactory::quadMeshMinRule(bf, H1Order, delta_k, dimensions[0], dimensions[1],
                                      elementCounts[0], elementCounts[1], useTriangles);
  
  if (meshSavePrefix != "")
  {
    saveMesh(mesh, meshSavePrefix, refinementNumber);
  }
  
//  if (useHWeightedTraces)
//  {
//    // set initial h_mean value
//    form.setTraceWeight(1.0/h_mean(mesh));
//  }
  // define inflow boundary
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
  SpatialFilterPtr inflowFilter = SpatialFilter::matchingX(0.0) | SpatialFilter::matchingY(0.0);
  vector<IndexType> inflowSides = meshTopo->getBoundarySidesThatMatch(inflowFilter);
  EntitySetPtr inflowEntitySet = meshTopo->createEntitySet();
  int sideDim = spaceDim - 1;
  for (IndexType sideEntityIndex : inflowSides)
  {
    inflowEntitySet->addEntity(sideDim, sideEntityIndex);
  }
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INFLOW_TAG_ID, inflowEntitySet);
  
  SpatialFilterPtr outflowFilter = SpatialFilter::matchingX(1.0) | SpatialFilter::matchingY(1.0);
  vector<IndexType> outflowSides = meshTopo->getBoundarySidesThatMatch(outflowFilter);
  EntitySetPtr outflowEntitySet = meshTopo->createEntitySet();
  for (IndexType sideEntityIndex : outflowSides)
  {
    outflowEntitySet->addEntity(sideDim, sideEntityIndex);
  }
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, OUTFLOW_TAG_ID, outflowEntitySet);

  BCPtr bc = BC::bc();
//  FunctionPtr traceWeight = form.traceWeight();
//  bc->addDirichlet(form.u_dirichlet(), INFLOW_TAG_ID, traceWeight * u_exact);
//  bc->addDirichlet(form.u_dirichlet(), OUTFLOW_TAG_ID, traceWeight * u_exact); // u_exact should actually be zero on the whole boundary
  

  bc->addDirichlet(form.u_dirichlet(), INFLOW_TAG_ID, u_exact);
  bc->addDirichlet(form.u_dirichlet(), OUTFLOW_TAG_ID, u_exact); // u_exact should actually be zero on the whole boundary
//  bc->addDirichlet(form.u_dirichlet(), INFLOW_TAG_ID, Function::zero());
//  bc->addDirichlet(form.u_dirichlet(), OUTFLOW_TAG_ID, Function::zero()); // u_exact should actually be zero on the whole boundary
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  solution->setUseCondensedSolve(useCondensedSolve);
  solution->setCubatureEnrichmentDegree(quadratureEnrichment);
  
  // Solution for L^2 projections:
  SolutionPtr bestSolution = Solution::solution(bf, mesh, bc, rhs, ip);
  bestSolution->setCubatureEnrichmentDegree(quadratureEnrichment);
  map<int,FunctionPtr> exactSolutionMap = {{form.u()->ID(), u_exact}};
  bestSolution->projectOntoMesh(exactSolutionMap);
  FunctionPtr u_best = Function::solution(form.u(), bestSolution);
  
  ostringstream matrixFileName;
  matrixFileName << thisRunPrefix.str() << "ref" << refinementNumber << ".dat";
  
  if (exportMatrix)
  {
    solution->setWriteMatrixToFile(true, matrixFileName.str());
  }
  
  auto getGMGSolver = [&solution, &formulationIsDPG, &delta_k, &useCondensedSolve, &maxIterations, &cgTol, &iterativeOutputLevel,
                       &smootherApplicationCount, formulation, useGMRESForDPG, usePointSymmetricGSForDPG] () -> Teuchos::RCP<GMGSolver>
  {
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(solution->mesh(), 1, delta_k);

    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine, maxIterations, cgTol,
                                                                   GMGOperator::MultigridStrategy::V_CYCLE,
                                                                   Solver::getDirectSolver(),
                                                                   useCondensedSolve));

    if (!formulationIsDPG)
    {
      gmgSolver->setSmootherType(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
      gmgSolver->setUseConjugateGradient(false); // won't be SPD
    }
    else
    {
      if (usePointSymmetricGSForDPG)
      {
        gmgSolver->setSmootherType(GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL);
      }
      else
      {
        gmgSolver->setSmootherType(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
      }
      
      if (useGMRESForDPG)
      {
        gmgSolver->setUseConjugateGradient(false);
      }
    }
    
    gmgSolver->setAztecOutput(iterativeOutputLevel);

    gmgSolver->setSmootherApplicationCount(smootherApplicationCount);
    
    return gmgSolver;
  };
  
  SolverPtr solver;
  if (useDirectSolver)
  {
    solver = Solver::getDirectSolver();
  }
  else
  {
    solver = getGMGSolver();
  }
  
  solution->solve(solver);
  assemblyTimes.push_back(solution->totalTimeLocalStiffness());
  solveTimes.push_back(solution->totalTimeSolve());
  
  if (computeLAPACKConditionNumber)
  {
    double condNum = conditionNumberLAPACK(*solution->getStiffnessMatrix());
    condNumbers.push_back(condNum);
    cout << scientific << setprecision(1);
    if (rank == 0) cout << "Condition Number (LAPACK) for refinement " << refinementNumber << ": " << condNum << endl;
  }
  if (writeEigValues)
  {
    ostringstream eigFileName;
    eigFileName << thisRunPrefix.str() << "ref" << refinementNumber << "_eig.txt";
    if (rank == 0) cout << "computing eigValues and writing to " << eigFileName.str() << endl;
    writeEigenValues(*solution->getStiffnessMatrix(), eigFileName.str());
  }
  if (reportSolutionTimings)
  {
    if (rank == 0)
    {
      double totalTime = solution->totalTimeLocalStiffness() + solution->totalTimeSolve();
      cout << scientific << setprecision(1);
      cout << "Assembly time: " << solution->totalTimeLocalStiffness() << " secs.\n";
      cout << "Solve time: " << solution->totalTimeSolve() << " secs.\n";
      cout << "Total time (assembly + solve): " << totalTime << " secs.\n";
    }
  }
  LinearTermPtr residual = form.residual(solution);
  if (ipForResidual == Teuchos::null) ipForResidual = ip;
  RefinementStrategyPtr refStrategy;
  Teuchos::RCP<MeshMatchingErrorIndicator<double>> meshMatchingErrorIndicator;
  if (meshLoadPrefix != "") {
    MeshTopologyViewPtr loadedMesh = loadMesh(meshLoadPrefix, refinementNumber);
    meshMatchingErrorIndicator = Teuchos::rcp( new MeshMatchingErrorIndicator<double>(mesh, loadedMesh) );
    refStrategy = Teuchos::rcp( new TRefinementStrategy<double>((ErrorIndicatorPtr)meshMatchingErrorIndicator, energyThreshold) );
    refStrategy->setEnforceOneIrregularity(false); // just trying something here; shouldn't make a difference, so long as it was enforced when meshes were generated...
  }
  else if (useCustomMeshRefinement)
  {
    auto errorIndicator = Teuchos::rcp( new CustomErrorIndicator<double>(mesh) );
    refStrategy = Teuchos::rcp( new TRefinementStrategy<double>((ErrorIndicatorPtr)errorIndicator, energyThreshold) );
  }
  else if (formulationIsDPG)
  {
    refStrategy = RefinementStrategy::energyErrorRefinementStrategy(mesh, residual, ipForResidual, energyThreshold);
  }
  else
  {
    VarPtr varForHessian = form.u();
    refStrategy = RefinementStrategy::hessianRefinementStrategy(solution, varForHessian, energyThreshold);
  }

  
  RieszRepPtr rieszRep = RieszRep::rieszRep(mesh, ipForResidual, residual);
  FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(rieszRep) );
  rieszRep->computeRieszRep();
  
  vector<FunctionPtr> functionsToExport;
  vector<string> functionsToExportNames;
  
  if (formulation != ConvectionDiffusionReactionFormulation::SUPG)
  {
    functionsToExport = {u_exact, u_best, f, energyErrorFunction};
    functionsToExportNames = {"u_exact", "u_best", "f", "energy_error"};
  }
  else
  {
    // omit energy error for SUPG
    functionsToExport = {u_exact, u_best, f};
    functionsToExportNames = {"u_exact", "u_best", "f"};
  }

  if (formulation == ConvectionDiffusionReactionFormulation::SUPG)
  {
    functionsToExport.push_back(form.SUPGStabilizationWeight());
    functionsToExportNames.push_back("tau");
  }
  FunctionPtr meshIndicator = Function::meshSkeletonCharacteristic();
  vector<FunctionPtr> traceFunctionsToExport = {meshIndicator};
  vector<string> traceFunctionsToExportNames = {"mesh"};
  
  FunctionPtr u_soln = Function::solution(form.u(), solution);
  double errL2, bestErrL2;
  
  // for some reason, ParaView introduces visual artifacts when doing plot over line on traces when
  // the outputted resolution (HDF5Exporter's "num1DPoints" argument) is too low.  (10 is too low.)
  int numLinearPointsPlotting = max(polyOrder,15);

  Teuchos::RCP<HDF5Exporter> functionExporter;
  Teuchos::RCP<HDF5Exporter> solnExporter;
  if (exportVisualization)
  {
    string exporterName = "ConfusionFunctions";
    exporterName = thisRunPrefix.str() + exporterName;
    functionExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));
    functionExporter->exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
    functionExporter->exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
    
    exporterName = "ConfusionSolution";
    exporterName = thisRunPrefix.str() + exporterName;
    solnExporter = Teuchos::rcp(new HDF5Exporter(mesh, exporterName));
    solnExporter->exportSolution(solution, refinementNumber, numLinearPointsPlotting);
  }
  
  errL2 = (u_soln - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);

  while ((refinementNumber < numRefinements) && (errL2 > refErrThreshold))
  {
    GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
    GlobalIndexType numActiveElements = mesh->numActiveElements();

    bestSolution->projectOntoMesh(exactSolutionMap);
    bestErrL2 = (u_best - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
    
    
//    {
//      // DEBUGGING
//      // gather mesh topology so we can output them using GnuPlotUtil
//      MeshTopologyPtr meshTopo = mesh->getTopology()->getGatheredCopy();
//    // now, on rank 0, output:
//    if (rank == 0)
//      {
//        ostringstream meshExportName;
//        meshExportName << "ConvectionDiffusion" << "_mesh" << refinementNumber;
//        int numPointsPerEdge = 2;
//        bool labelCells = true;
//        string meshColor = "black";
//        GnuPlotUtil::writeExactMeshSkeleton(meshExportName.str(), meshTopo.get(), numPointsPerEdge, labelCells, meshColor);
//      }
//    }
    
    MeshTopologyViewPtr loadedMesh;
    if (meshLoadPrefix != "")
    {
      loadedMesh = loadMesh(meshLoadPrefix, refinementNumber + 1);
      meshMatchingErrorIndicator->setMeshToMatch(loadedMesh);
    }
    
    if (!refineUniformly)
    {
      refStrategy->refine(reportRefinedCells && (rank==0));
    }
    else
    {
      refStrategy->hRefineUniformly();
    }
    
    if (meshLoadPrefix != "")
    {
      if (! meshesMatch(mesh, loadedMesh) )
      {
        cout << "Error; mesh does not match loadedMesh after refinement.\n";
        exit(1);
      }
    }
    
//    if (useHWeightedTraces)
//    {
//      // set new h_mean value
//      form.setTraceWeight(1.0 / h_mean(mesh));
//    }
   
    // recreate solver if neccessary, after refinement:
    SolverPtr solver;
    if (useDirectSolver)
    {
      solver = Solver::getDirectSolver();
    }
    else
    {
      solver = getGMGSolver();
    }

    bestL2errors.push_back(bestErrL2);
    L2errors.push_back(errL2);
    activeElementCounts.push_back(numActiveElements);
    dofCounts.push_back(numGlobalDofs);

    double indicatorError = refStrategy->getEnergyError(refinementNumber);
    indicatorMeasures.push_back(indicatorError);
    if (rank == 0) cout << "Refinement " << refinementNumber << " has indicator error " << indicatorError << ",";
    if (rank == 0)
    {
      cout << " L^2 error " << errL2 << " (vs best L^2 error of " << bestErrL2 << ")";
      cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
    }

    refinementNumber++;
    if (meshSavePrefix != "")
    {
      saveMesh(mesh, meshSavePrefix, refinementNumber);
    }
    
    if (exportMatrix)
    {
      ostringstream matrixFileName;
      matrixFileName << thisRunPrefix.str() << "ref" << refinementNumber << ".dat";
      
      solution->setWriteMatrixToFile(true, matrixFileName.str());
    }
    
    solution->solve(solver);
    
    assemblyTimes.push_back(solution->totalTimeLocalStiffness());
    solveTimes.push_back(solution->totalTimeSolve());
    
    errL2 = (u_soln - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
    
    if (reportSolutionTimings)
    {
      if (rank == 0)
      {
        double totalTime = solution->totalTimeLocalStiffness() + solution->totalTimeSolve();
        cout << scientific << setprecision(1);
        cout << "Total time (assembly + solve): " << totalTime << " secs.\n";
      }
    }
    
    if (exportVisualization)
    {
      solnExporter->exportSolution(solution, refinementNumber, numLinearPointsPlotting);
    }
    
    if (computeLAPACKConditionNumber)
    {
      double condNum = conditionNumberLAPACK(*solution->getStiffnessMatrix());
      condNumbers.push_back(condNum);
      if (rank == 0) cout << "Condition Number (LAPACK) for refinement " << refinementNumber << ": " << condNum << endl;
    }
    else if (conditionNumberEstimate)
    {
      int errCode;
      double conditionNumber = solution->conditionNumberEstimate(errCode);
      if (rank==0)
      {
        if (errCode != 0)
        {
          cout << "Condition Number estimation failed with error code " << errCode;
          cout << " (estimate, for what it's worth--very little, probably--was " << conditionNumber << ")\n";
        }
        else
        {
          cout << "Condition Number estimate refinement " << refinementNumber << ": " << conditionNumber << endl;
        }
      }
    }
    if (writeEigValues)
    {
      ostringstream eigFileName;
      eigFileName << thisRunPrefix.str() << "ref" << refinementNumber << "_eig.txt";
      if (rank == 0) cout << "computing eigValues and writing to " << eigFileName.str() << endl;
      writeEigenValues(*solution->getStiffnessMatrix(), eigFileName.str());
    }
    
    if (formulationIsDPG)
    {
      Epetra_Time timer(*MPIWrapper::CommSerial());
      rieszRep->computeRieszRep();
      double rieszRepTime = timer.ElapsedTime();
      if (rank==0)
      {
        cout << "Computed Riesz rep in " << rieszRepTime << " seconds (timed on rank 0).\n";
      }
    }
    if (exportVisualization)
    {
      functionExporter->exportFunction(functionsToExport, functionsToExportNames, refinementNumber, numLinearPointsPlotting);
      functionExporter->exportFunction(traceFunctionsToExport, traceFunctionsToExportNames, refinementNumber, numLinearPointsPlotting);
    }
  }
  
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  GlobalIndexType numActiveElements = mesh->numActiveElements();
  
  bestSolution->projectOntoMesh(exactSolutionMap);
  bestErrL2 = (u_best - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
  errL2 = (u_soln - u_exact)->l2norm(solution->mesh(), quadratureEnrichmentL2);
  
  bestL2errors.push_back(bestErrL2);
  L2errors.push_back(errL2);
  activeElementCounts.push_back(numActiveElements);
  dofCounts.push_back(numGlobalDofs);

  double indicatorError = refStrategy->computeTotalEnergyError();
  indicatorMeasures.push_back(indicatorError);
  if (rank == 0) cout << "Refinement " << refinementNumber << " has indicator error " << indicatorError << ",";

  if (rank == 0)
  {
    cout << " L^2 error " << errL2 << " (vs best L^2 error of " << bestErrL2 << ")";
    cout << " (" << numGlobalDofs << " dofs on " << numActiveElements << " elements)\n";
  }
  
  if (exportMatricesLastRefinement)
  {
    if (rank == 0) cout << "Exporting matrices for final solve.\n";
    
    ostringstream stiffmatrixName;
    stiffmatrixName << thisRunPrefix.str() << "finalStiffness.dat";
    Teuchos::RCP<Epetra_CrsMatrix> stiffness = solution->getStiffnessMatrix();
    EpetraExt::RowMatrixToMatrixMarketFile(stiffmatrixName.str().c_str(),*stiffness, NULL, NULL, false);
    if (rank == 0) cout << "Wrote unpreconditioned matrix to " << stiffmatrixName.str() << endl;
    
    auto gmgSolver = getGMGSolver();
    gmgSolver->gmgOperator()->setFineStiffnessMatrix(stiffness.get());
    Teuchos::RCP<Epetra_CrsMatrix> op = gmgSolver->gmgOperator()->getMatrixRepresentation();
    
    int maxRowSize = 0; // _P->MaxNumEntries();
    
    // compute op * stiffness
    Epetra_CrsMatrix preconditionedStiffness(::Copy, solution->getPartitionMap(), maxRowSize);
    
    int err = EpetraExt::MatrixMatrix::Multiply(*op, false, *stiffness, false, preconditionedStiffness);
    if (err != 0) cout << "Error (code " << err << ") encountered during matrix-matrix multiply.\n";
    
    ostringstream preconditionedStiffmatrixName;
    preconditionedStiffmatrixName << thisRunPrefix.str() << "finalStiffnessPreconditioned.dat";
    EpetraExt::RowMatrixToMatrixMarketFile(preconditionedStiffmatrixName.str().c_str(),preconditionedStiffness, NULL, NULL, false);
    if (rank == 0) cout << "Wrote preconditioned matrix to " << preconditionedStiffmatrixName.str() << endl;
  }
  
  // compute average timings across all MPI ranks.
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  totalTimeG = MPIWrapper::sum(*Comm, totalTimeG);
  totalTimeB = MPIWrapper::sum(*Comm, totalTimeB);
  totalTimeT = MPIWrapper::sum(*Comm, totalTimeT);
  totalTimeK = MPIWrapper::sum(*Comm, totalTimeK);
  totalElementCount = MPIWrapper::sum(*Comm, totalElementCount);

  totalTimeRHS = MPIWrapper::sum(*Comm, totalTimeRHS);
  totalRHSElements = MPIWrapper::sum(*Comm, totalRHSElements);
  
  if ((rank == 0) && (formulation != ConvectionDiffusionReactionFormulation::SUPG))
  {
    cout << "Average time/element for optimal test computations:\n";
    cout << "G:   " <<  totalTimeG / totalElementCount << " sec.\n";
    cout << "B:   " <<  totalTimeB / totalElementCount << " sec.\n";
    cout << "T:   " <<  totalTimeT / totalElementCount << " sec.\n";
    cout << "K:   " <<  totalTimeK / totalElementCount << " sec.\n";
    cout << "RHS: " << totalTimeRHS / totalRHSElements << " sec.\n";
  }
  
  ostringstream errorFileName;
  errorFileName << thisRunPrefix.str() << "convergence_history.dat";

  if (rank==0)
  {
    ofstream fout(errorFileName.str().c_str());
    fout << "Refinement\tElements\tDofs\tBest Error\tActual Error\tAssembly Time\tSolve Time\n";
    for (int i=0; i<indicatorMeasures.size(); i++)
    {
      fout << i << "\t";
      fout << activeElementCounts[i] << "\t";
      fout << dofCounts[i] << "\t";
//      fout << indicatorMeasures[i] << "\t";
      fout << bestL2errors[i] << "\t";
      fout << L2errors[i] << "\t";
      fout << assemblyTimes[i] << "\t";
      fout << solveTimes[i] << "\n";
    }
    fout.close();
    cout << "Wrote convergence history to " << errorFileName.str() << ".\n";
  }
  
  return 0;
}
