//
// © 2016-2024 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "ConfigurationSpaceFormulationPoisson.h"
#include "HDF5Exporter.h"
#include "MonomialFunctions.h"
#include "RHS.h"
#include "RieszRep.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovSpaceTimeFormulation.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace Camellia;

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const double x0, const double xWidth,
                                                              const double t0, const double tWidth,
                                                              const double v0, const double vWidth,
                                                              const int xElements, const int tElements, const int vElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const bool usePeriodicBCs, const EBasisFamily &basisFamily,
                                                              const int numUniformRefinements)
{
  Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME));

  MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);

  std::vector<double> temporalVertices(tElements + 1);
  for (int i=0; i<=tElements; i++)
  {
    const double t = double(i) / double(tElements) * tWidth + t0;
    temporalVertices[i] = t;
  }

  std::vector<double> velocitySpaceVertices_x(vElements + 1);
  for (int i=0; i<=vElements; i++)
  {
    const double v = double(i) / double(vElements) * vWidth + v0;
    velocitySpaceVertices_x[i] = v;
  }

  std::vector<double> velocitySpaceVertices_y;
  std::vector<double> velocitySpaceVertices_z;

  return Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                     temporalVertices,
                                                     velocitySpaceVertices_x,
                                                     velocitySpaceVertices_y,
                                                     velocitySpaceVertices_z,
                                                     H1Order, delta_k,
                                                     numUniformRefinements,
                                                     basisFamily));
}

struct ManufacturedSolution
{
  FunctionPtr f;
  FunctionPtr rhs;
  FunctionPtr E;
  FunctionPtr VE;
};

ManufacturedSolution manufacturedSolution()
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr t = Function::yn(1);
  FunctionPtr v = MonomialFunction::xi_n(2, 1); // 1D1T1V coordinates are x, t, v; xi_n is zero-based.

  FunctionPtr sin_pi_x = Camellia::sin(M_PI * x);
  FunctionPtr cos_pi_x = Camellia::cos(M_PI * x);
  FunctionPtr sin_pi_t = Camellia::sin(M_PI * t);
  FunctionPtr cos_pi_t = Camellia::cos(M_PI * t);
  FunctionPtr sin_pi_v = Camellia::sin(M_PI * v);
  FunctionPtr cos_pi_v = Camellia::cos(M_PI * v);

  ManufacturedSolution ms;
  ms.f = sin_pi_x * sin_pi_t * cos_pi_v;

  // c = f_t + v f_x + f_v, corresponding to unit electric field with q/m = 1.
  ms.rhs = M_PI * sin_pi_x * cos_pi_t * cos_pi_v
         + M_PI * v * cos_pi_x * sin_pi_t * cos_pi_v
         - M_PI * sin_pi_x * sin_pi_t * sin_pi_v;

  ms.E  = Function::constant(1.0);
  ms.VE = -x; // E = -grad(VE) = 1 in 1D.

  return ms;
}

struct TestNormResult
{
  std::string name;
  double fL2Norm;
  double absL2Error;
  double relL2Error;
};

TestNormResult solveWithTestNorm(const std::string &testNormName, const bool useL2TestNorm,
                                 const int xElements, const int tElements, const int vElements,
                                 const int H1Order, const int delta_k,
                                 const bool useHierarchicalBasis, const int numUniformRefinements,
                                 const double iterativeSolveTolerance, const int iterativeSolveMaxIters,
                                 const int iterativeSolveOutputLevel, const bool saveViz,
                                 const std::string &outputDir, const std::string &resultsFilePrefix,
                                 const int rank)
{
  const double q      = 1.0;
  const double eps0   = 1.0;
  const double m      = 1.0;
  const double x0     = 0.0;
  const double xWidth = 1.0;
  const double t0     = 0.0;
  const double tWidth = 1.0;
  const double v0     = -0.5;
  const double vWidth = 1.0;

  const int numTimeSlabs = 1;
  const bool usePeriodicBCs = false;
  EBasisFamily basisFamily = useHierarchicalBasis ? BASIS_FAMILY_HIERARCHICAL : BASIS_FAMILY_NODAL;

  auto form = getFormulation1D1T1V(x0, xWidth, t0, tWidth / numTimeSlabs, v0, vWidth,
                                   xElements, tElements, vElements, q, eps0, m, H1Order, delta_k,
                                   usePeriodicBCs, basisFamily, numUniformRefinements);
  form->setEnforceQuasiNeutrality(false);
  form->setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
  form->setIterativeSolveTolerance(iterativeSolveTolerance);
  form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);

  ManufacturedSolution ms = manufacturedSolution();

  form->vlasovSolution()->rhs()->addTerm(ms.rhs * form->w());
  form->setInitialAndInflowConditions(ms.f, ms.f);

  // Impose potential boundary data whose electric field is the requested unit field.
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(x0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(x0 + xWidth);
  const bool restrictSideSetsToBoundary = true;
  auto xLeftRightTag = form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
  form->addConfigurationSpaceBC(xLeftRightTag, form->VE_configurationSpaceTrace(), ms.VE);

  // Seed the configuration-space solution with E=1 before the Vlasov solve.  We solve only the
  // Vlasov problem below, so this gives the manufactured RHS the matching unit acceleration.
  std::map<int, FunctionPtr> configurationProjection;
  configurationProjection[form->configurationSpaceFormulation()->E_fieldVar(0)->ID()]  = ms.E;
  configurationProjection[form->configurationSpaceFormulation()->VE_fieldVar()->ID()] = ms.VE;
  form->configurationSpaceSolution()->projectOntoMesh(configurationProjection);
  form->configurationSpaceFormulation()->updateConfigurationSpaceFunctions();

  if (useL2TestNorm)
  {
    form->vlasovSolution()->setIP(form->vlasovBF()->l2Norm());
  }
  else
  {
    form->vlasovSolution()->setIP(form->vlasovBF()->graphNorm());
  }

  form->vlasovSolution()->solve();

  auto vlasovSoln = form->vlasovSolution();
  VarPtr w = form->w();
  TRieszRepPtr<double> rieszRep_residual;
  FunctionPtr epsilon;
  if (saveViz || !resultsFilePrefix.empty())
  {
    auto ip = useL2TestNorm ? form->vlasovBF()->l2Norm() : form->vlasovBF()->graphNorm();
    auto bf_at_w = form->vlasovBF()->testFunctional(vlasovSoln);
    auto residual = bf_at_w - vlasovSoln->rhs()->linearTerm();
    rieszRep_residual = RieszRep::rieszRep(vlasovSoln->mesh(), ip, residual);
    rieszRep_residual->computeRieszRep();
    epsilon = TRieszRep<double>::repFunction(w, rieszRep_residual);
  }

  FunctionPtr f_soln = Function::solution(form->f(), vlasovSoln);
  FunctionPtr f_err  = f_soln - ms.f;

  const double fL2Norm    = ms.f->l2norm(form->vlasovMesh());
  const double absL2Error = f_err->l2norm(form->vlasovMesh());

  if (saveViz)
  {
    std::string vizName = "CompareTestNormsDPGBoBa_" + testNormName;
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vizName, outputDir));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr>{ms.f, f_err, epsilon}, std::vector<std::string>{"f exact", "f error", "epsilon"});
  }

  if (!resultsFilePrefix.empty())
  {
    int outputXElements = xElements;
    int outputTElements = tElements;
    int outputVElements = vElements;
    for (int refNumber=0; refNumber<numUniformRefinements; refNumber++)
    {
      outputXElements *= 2;
      outputTElements *= 2;
      outputVElements *= 2;
    }
    const int numOutputCells = outputXElements * outputTElements * outputVElements;

    auto fBasis = form->vlasovMesh()->getElement(0)->elementType()->trialOrderPtr->getBasis(form->f()->ID());
    const int fCardinality = fBasis->getCardinality();
    const int fCardinality1D = fBasis->intrepid2TensorBasisComponents()[0]->getCardinality();

    auto wBasis = form->vlasovMesh()->getElement(0)->elementType()->testOrderPtr->getBasis(w->ID());
    const int wCardinality = wBasis->getCardinality();
    const int wCardinality1D = wBasis->intrepid2TensorBasisComponents()[0]->getCardinality();

    auto globalCellOrdinal = [&](const int &cellX, const int &cellT, const int &cellV) -> GlobalIndexType
    {
      return cellX + cellT * outputXElements + cellV * outputXElements * outputTElements;
    };
    auto localDofOrdinal = [&](const int &basisX, const int &basisT, const int &basisV, const int &basisCardinality1D) -> GlobalIndexType
    {
      return basisX + basisT * basisCardinality1D + basisV * basisCardinality1D * basisCardinality1D;
    };

    Camellia::FieldContainer<double> fCoefficients(numOutputCells, fCardinality);
    Camellia::FieldContainer<double> epsilonCoefficients(numOutputCells, wCardinality);

    const auto &myCells = form->vlasovMesh()->cellIDsInPartition();
    for (auto myCellID : myCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(myCellID >= numOutputCells, std::invalid_argument, "Unexpected cell ID while writing coefficient data");
      const int cellOrdinal = static_cast<int>(myCellID);

      Camellia::FieldContainer<double> fDofs(fCardinality);
      vlasovSoln->getBasisCoefficientsForCell(fDofs, form->f()->ID(), myCellID, STANDARD_SOLUTION, 0);
      for (int dofOrdinal=0; dofOrdinal<fCardinality; dofOrdinal++)
      {
        fCoefficients(cellOrdinal, dofOrdinal) = fDofs(dofOrdinal);
      }

      const auto &epsilonDofs = rieszRep_residual->getCoefficientsForCell(myCellID);
      auto testOrdering = form->vlasovMesh()->getElementType(myCellID)->testOrderPtr;
      for (int basisOrdinal=0; basisOrdinal<wCardinality; basisOrdinal++)
      {
        const int dofOrdinal = testOrdering->getDofIndex(w->ID(), basisOrdinal);
        epsilonCoefficients(cellOrdinal, basisOrdinal) = epsilonDofs(dofOrdinal);
      }
    }
    MPIWrapper::entryWiseSum(form->vlasovMesh()->Comm(), fCoefficients);
    MPIWrapper::entryWiseSum(form->vlasovMesh()->Comm(), epsilonCoefficients);

    auto writeCoefficients = [&](const std::string &filename, const Camellia::FieldContainer<double> &coefficients,
                                 const int basisCardinality1D, const std::string &label)
    {
      if (rank != 0) return;

      std::ofstream fileStream(filename, std::ios::out);
      if (!fileStream)
      {
        std::cerr << "ERROR: Could not open file " << filename << std::endl;
        return;
      }
      fileStream << std::setprecision(15);

      int globalDofIndex = 1; // 1-based for MATLAB-style output
      for (int cv=0; cv<outputVElements; cv++)
      {
        for (int fv=0; fv<basisCardinality1D; fv++)
        {
          for (int ct=0; ct<outputTElements; ct++)
          {
            for (int ft=0; ft<basisCardinality1D; ft++)
            {
              for (int cx=0; cx<outputXElements; cx++)
              {
                for (int fx=0; fx<basisCardinality1D; fx++)
                {
                  const GlobalIndexType cellIndex = globalCellOrdinal(cx, ct, cv);
                  const GlobalIndexType dofIndex = localDofOrdinal(fx, ft, fv, basisCardinality1D);
                  fileStream << globalDofIndex << "\t" << coefficients(cellIndex, dofIndex) << std::endl;
                  globalDofIndex++;
                }
              }
            }
          }
        }
      }
      fileStream.close();
      std::cout << "Wrote " << label << " coefficient data to file: " << filename << std::endl;
      std::cout << "Ordering is layout-left (fx,cx,ft,ct,fv,cv): ";
      std::cout << basisCardinality1D << " x ";
      std::cout << outputXElements << " x ";
      std::cout << basisCardinality1D << " x ";
      std::cout << outputTElements << " x ";
      std::cout << basisCardinality1D << " x ";
      std::cout << outputVElements << std::endl;
    };

    const std::string resultsBasePath = outputDir + "/" + resultsFilePrefix + "_" + testNormName;
    writeCoefficients(resultsBasePath + "_f.dat", fCoefficients, fCardinality1D, "f");
    writeCoefficients(resultsBasePath + "_epsilon.dat", epsilonCoefficients, wCardinality1D, "epsilon");
  }

  TestNormResult result;
  result.name        = testNormName;
  result.fL2Norm     = fL2Norm;
  result.absL2Error  = absL2Error;
  result.relL2Error  = absL2Error / fL2Norm;
  return result;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);

  Teuchos::CommandLineProcessor cmdp(false, true);

  const int rank = mpiSession.getRank();

  int xElements = 4;
  int vElements = 4;
  int tElements = 2;
  int delta_k   = 2;
  int H1Order   = 3;
  int numUniformRefinements = 0;
  bool saveViz = false;
  bool useHierarchicalBasis = true;
  std::string outputDir = ".";
  std::string resultsFilePrefix = "compareTestNorms";
  int iterativeSolveOutputLevel = 100;
  double iterativeSolveTolerance = 1e-10;
  int iterativeSolveMaxIters = 10000;

  cmdp.setOption("xElements", &xElements);
  cmdp.setOption("tElements", &tElements);
  cmdp.setOption("vElements", &vElements);
  cmdp.setOption("numUniformRefinements", &numUniformRefinements);
  cmdp.setOption("p", &H1Order);
  cmdp.setOption("dp", &delta_k);
  cmdp.setOption("viz", "noViz", &saveViz);
  cmdp.setOption("outputDir", &outputDir);
  cmdp.setOption("resultsFilePrefix", &resultsFilePrefix, "prefix for f and epsilon coefficient output files (empty disables coefficient output)");
  cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
  cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
  cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
  cmdp.setOption("useHierarchicalBasis", "useNodalBasis", &useHierarchicalBasis);

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (rank == 0)
  {
    std::cout << "Running CompareTestNormsDPGBoBa manufactured Vlasov problem." << std::endl;
    std::cout << "Manufactured solution: f(x,t,v) = sin(pi x) sin(pi t) cos(pi v)" << std::endl;
    std::cout << "Domain: x in [0,1], t in [0,1], v in [-1/2,1/2]; numTimeSlabs = 1" << std::endl;
    std::cout << "Mesh: " << xElements << " x " << tElements << " x " << vElements
              << ", p=" << H1Order << ", dp=" << delta_k << std::endl;
  }

  std::vector<TestNormResult> results;
  results.push_back(solveWithTestNorm("L2", true, xElements, tElements, vElements, H1Order, delta_k,
                                       useHierarchicalBasis, numUniformRefinements, iterativeSolveTolerance,
                                       iterativeSolveMaxIters, iterativeSolveOutputLevel, saveViz, outputDir,
                                       resultsFilePrefix, rank));
  results.push_back(solveWithTestNorm("Graph", false, xElements, tElements, vElements, H1Order, delta_k,
                                       useHierarchicalBasis, numUniformRefinements, iterativeSolveTolerance,
                                       iterativeSolveMaxIters, iterativeSolveOutputLevel, saveViz, outputDir,
                                       resultsFilePrefix, rank));

  if (rank == 0)
  {
    std::cout << std::endl;
    std::cout << std::left << std::setw(15) << "Test norm"
              << std::setw(25) << "||f_exact||_L2"
              << std::setw(25) << "||f_h-f||_L2"
              << std::setw(25) << "relative L2 error" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    for (const auto &result : results)
    {
      std::cout << std::left << std::setw(15) << result.name
                << std::scientific << std::setprecision(6) << std::setw(25) << result.fL2Norm
                << std::setw(25) << result.absL2Error
                << std::setw(25) << result.relL2Error << std::endl;
    }
  }

  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);

  return 0;
}
