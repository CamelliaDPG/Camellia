//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ErrorIndicatorTests
//  Camellia
//
//  Created by Nate Roberts on 8/10/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "ErrorIndicator.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;

namespace
{
  double cellValue(std::vector<double> &centroid)
  {
    double value = 0;
    for (double coord : centroid)
    {
      value += (coord+0.5)*(coord+0.5); // top-right is displaced by (0.5, 0.5)
    }
    return value;
  }
  
  // ! Initialize a solution with elementwise constant field data in which the constant
  // ! is computed as the sum of the square of the magnitude (Euclidean norm) of the top-right
  // ! vertex coordinate.  Mesh has bottom left coordinate at the origin, and each element
  // ! is a unit square.
  void initializeSolution(SolutionPtr &soln, VarPtr &var, int meshWidth, int meshHeight)
  {
    int spaceDim = 2;
    bool useConformingTraces = true; // inconsequential here
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    var = form.u(); // just picking a field
    int H1Order = 1; // gives constant fields
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, {(double)meshWidth, (double)meshHeight}, {meshWidth,meshHeight}, H1Order);
    soln = Solution::solution(bf, mesh);
    map<int, FunctionPtr> functionMap;
    const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
    int numSolutions = soln->numSolutions();
    int sideOrdinal = -1; // all sides
    for (GlobalIndexType cellID : *myCells)
    {
      vector<double> centroid = mesh->getCellCentroid(cellID);
      double value = cellValue(centroid);
      functionMap[var->ID()] = Function::constant(value);
      for (int solutionOrdinal=0; solutionOrdinal<numSolutions; solutionOrdinal++)
      {
        soln->projectOntoCell(functionMap, cellID, sideOrdinal, solutionOrdinal);
      }
    }
  }

  FunctionPtr highOrderFunction()
  {
    FunctionPtr x = Function::xn(1), y = Function::yn(1);
    FunctionPtr f = x * x + x * y - 2 * y * y;
    return f;
  }
  
  double expectedHighOrderHessian()
  {
    // note hard-coded dependence on the particular choice of high-order function, above
    /*
     H = [ 2.0  1.0 ]
         [ 1.0 -2.0 ]
     
     largest eigenvalue magnitude: 4.16227766016838
     */
    return 4.16227766016838;
  }
  
  // ! Initialize a solution with elementwise quadratic field data.  Mesh has bottom left coordinate at the origin, and each element
  // ! is a unit square.
  void initializeHighOrderSolutionUnitDomain(SolutionPtr &soln, VarPtr &var, int meshWidth, int meshHeight)
  {
    int spaceDim = 2;
    bool useConformingTraces = true; // inconsequential here
    PoissonFormulation form(spaceDim, useConformingTraces);
    BFPtr bf = form.bf();
    var = form.u(); // just picking a field
    int H1Order = 3;
    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, {(double)1.0, (double)1.0}, {meshWidth,meshHeight}, H1Order);
    soln = Solution::solution(bf, mesh);
    map<int, FunctionPtr> functionMap;
    FunctionPtr f = highOrderFunction();
    
    const set<GlobalIndexType>* myCells = &mesh->cellIDsInPartition();
    for (GlobalIndexType cellID : *myCells)
    {
      functionMap[var->ID()] = f;
      const int ALL_SIDES = -1;
      soln->projectOntoCell(functionMap, cellID, ALL_SIDES);
    }
  }
  
  vector<double> expectedGradient(MeshTopologyViewPtr meshTopo, GlobalIndexType myCellID)
  {
    int spaceDim = meshTopo->getDimension();
    Camellia::FieldContainer<double> Y(spaceDim,spaceDim); // matrix we'll invert
    Camellia::FieldContainer<double> valueDiffs(spaceDim);
    vector<double> centroid = meshTopo->getCellCentroid(myCellID);
    double myValue = cellValue(centroid);
    CellPtr cell = meshTopo->getCell(myCellID);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      vector<double> neighborCentroid = meshTopo->getCellCentroid(neighborID);
      double neighborValue = cellValue(neighborCentroid);
      vector<double> y(spaceDim); // distance vector
      double y_mag_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        y[d] = neighborCentroid[d] - centroid[d];
        y_mag_squared += y[d] * y[d];
      }
      for (int d1=0; d1<spaceDim; d1++)
      {
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += y[d1] * y[d2] / y_mag_squared;
        }
        valueDiffs(d1) += y[d1] * (neighborValue - myValue) / y_mag_squared;
      }
    }
    Camellia::FieldContainer<double> gradient(spaceDim);
    
    SerialDenseWrapper::solveSystem(gradient, Y, valueDiffs);
    vector<double> gradientVector(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      gradientVector[d] = gradient(d);
    }
    return gradientVector;
  }
  
  Camellia::FieldContainer<double> expectedHessian(MeshTopologyViewPtr meshTopo, GlobalIndexType myCellID)
  {
    int spaceDim = meshTopo->getDimension();
    Camellia::FieldContainer<double> Y(spaceDim,spaceDim); // matrix we'll invert
    Camellia::FieldContainer<double> gradientDiffsOuterProduct(spaceDim,spaceDim); // RHS
    vector<double> centroid = meshTopo->getCellCentroid(myCellID);
    vector<double> myGradient = expectedGradient(meshTopo, myCellID);
    CellPtr cell = meshTopo->getCell(myCellID);
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      vector<double> neighborCentroid = meshTopo->getCellCentroid(neighborID);
      vector<double> neighborGradient = expectedGradient(meshTopo, neighborID);
      vector<double> y(spaceDim); // distance vector
      double y_mag_squared = 0;
      for (int d=0; d<spaceDim; d++)
      {
        y[d] = neighborCentroid[d] - centroid[d];
        y_mag_squared += y[d] * y[d];
      }
      for (int d1=0; d1<spaceDim; d1++)
      {
        for (int d2=0; d2<spaceDim; d2++)
        {
          Y(d1,d2) += y[d1] * y[d2] / y_mag_squared;
          gradientDiffsOuterProduct(d1,d2) += y[d1] * (neighborGradient[d2] - myGradient[d2]) / y_mag_squared;
        }
      }
    }
    Camellia::FieldContainer<double> hessian(spaceDim, spaceDim);
    SerialDenseWrapper::solveSystem(hessian, Y, gradientDiffsOuterProduct);
    
    // symmetrize:
    for (int d1=0; d1<spaceDim; d1++)
    {
      for (int d2=0; d2<spaceDim; d2++)
      {
        hessian(d1,d2) = 0.5 * (hessian(d1,d2) + hessian(d2,d1));
      }
    }
    
    return hessian;
  }

  TEUCHOS_UNIT_TEST(ErrorIndicator, Gradient)
  {
    int meshWidth = 5, meshHeight = 5;
    
    SolutionPtr soln;
    VarPtr var;
    initializeSolution(soln, var, meshWidth, meshHeight);
    
    ErrorIndicatorPtr gradientIndicator = ErrorIndicator::gradientErrorIndicator(soln, var);
    gradientIndicator->measureError();
    
    MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology()->getGatheredCopy();
    
    double tol = 1e-14;
    const set<GlobalIndexType>* myCells = &soln->mesh()->cellIDsInPartition();
    const map<GlobalIndexType,double>* gradientValues = &gradientIndicator->localErrorMeasures();
    for (GlobalIndexType cellID : *myCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(gradientValues->find(cellID) == gradientValues->end(), std::invalid_argument, "gradient value not populated for rank-local cell");
      double gradientValue = gradientValues->find(cellID)->second;
      vector<double> gradient = expectedGradient(meshTopo, cellID);
      double mag = 0;
      for (double comp : gradient)
      {
        mag += comp * comp;
      }
      mag = sqrt(mag);
      TEST_FLOATING_EQUALITY(mag, gradientValue, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST(ErrorIndicator, Hessian)
  {
    int meshWidth = 5, meshHeight = 5;
    
    SolutionPtr soln;
    VarPtr var;
    initializeSolution(soln, var, meshWidth, meshHeight);
    
    ErrorIndicatorPtr hessianIndicator = ErrorIndicator::hessianErrorIndicator(soln, var);
    hessianIndicator->measureError();
    
    MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology()->getGatheredCopy();
    int spaceDim = meshTopo->getDimension();
    
    double tol = 1e-13;
    const set<GlobalIndexType>* myCells = &soln->mesh()->cellIDsInPartition();
    const map<GlobalIndexType,double>* hessianValues = &hessianIndicator->localErrorMeasures();
    for (GlobalIndexType cellID : *myCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(hessianValues->find(cellID) == hessianValues->end(), std::invalid_argument, "hessian value not populated for rank-local cell");
      double hessianValue = hessianValues->find(cellID)->second;
      
      vector<double> gradient = expectedGradient(meshTopo, cellID);
//      cout << "cell " << cellID << " gradient: (";
//      for (int d=0; d<spaceDim-1; d++)
//      {
//        cout << gradient[d] << ",";
//      }
//      cout << gradient[spaceDim-1] << ")\n";
      
      Camellia::FieldContainer<double> hessian = expectedHessian(meshTopo, cellID);
      Camellia::FieldContainer<double> lambda_real(spaceDim), lambda_imag(spaceDim);
      SerialDenseWrapper::eigenvalues(hessian, lambda_real, lambda_imag);
      double maxLambda = 0;
      for (int d=0; d<spaceDim; d++)
      {
        maxLambda = max(lambda_real(d),maxLambda);
        // while we're at it, let's assert that the imaginary part of each eigenvalue is 0:
        TEST_COMPARE(abs(lambda_imag(d)), <, tol);
      }
      
      TEST_FLOATING_EQUALITY(maxLambda, hessianValue, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST(ErrorIndicator, HessianHighOrder)
  {
    int meshWidth = 5, meshHeight = 5;
    
    SolutionPtr soln;
    VarPtr var;
    initializeHighOrderSolutionUnitDomain(soln, var, meshWidth, meshHeight);
    
    ErrorIndicatorPtr hessianIndicator = ErrorIndicator::hessianErrorIndicator(soln, var);
    hessianIndicator->measureError();
    
    MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology()->getGatheredCopy();
    int spaceDim = meshTopo->getDimension();
    
    double tol = 1e-12;
    const set<GlobalIndexType>* myCells = &soln->mesh()->cellIDsInPartition();
    const map<GlobalIndexType,double>* hessianValues = &hessianIndicator->localErrorMeasures();
    for (GlobalIndexType cellID : *myCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(hessianValues->find(cellID) == hessianValues->end(), std::invalid_argument, "hessian value not populated for rank-local cell");
      double hessianValue = hessianValues->find(cellID)->second;
      
      double h = 1.0 / meshWidth;
      double hPower = pow(h, 2.0 + spaceDim / 2.0);
      
      double expectedValue = expectedHighOrderHessian() * hPower;
      TEST_FLOATING_EQUALITY(expectedValue, hessianValue, tol);
    }
  }

  TEUCHOS_UNIT_TEST(ErrorIndicator, GlobalCellsBelowThreshold)
  {
    // set things up with the hessian error indicator as in HessianHighOrder
    // then check whether globalCellsBelowThreshold() behaves as expected.
    int meshWidth = 5, meshHeight = 5;
    
    SolutionPtr soln;
    VarPtr var;
    initializeSolution(soln, var, meshWidth, meshHeight);
    
    ErrorIndicatorPtr hessianIndicator = ErrorIndicator::hessianErrorIndicator(soln, var);
    hessianIndicator->measureError();
    
    MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology()->getGatheredCopy();
    int spaceDim = meshTopo->getDimension();
    
    double tol = 1e-12;
    const set<GlobalIndexType>* myCells = &soln->mesh()->cellIDsInPartition();
    const map<GlobalIndexType,double> & localHessianValues = hessianIndicator->localErrorMeasures();
    
    using DataType = std::pair<GlobalIndexType, double>;
    std::vector<DataType> localHessianValuesVector(localHessianValues.begin(),localHessianValues.end());
    
    std::vector<DataType> globalHessianValues;
    std::vector<int> offsets;
    MPIWrapper::allGatherVariable(soln->mesh()->Comm(), globalHessianValues, localHessianValuesVector, offsets);
    
    for (const auto & entry : globalHessianValues)
    {
      const double errThreshold = entry.second * (1. + 1e-13); // cheat slightly above this element, so that this element is included
      auto actualCellsBelowThreshold = hessianIndicator->globalCellsBelowThreshold(errThreshold); // note: MPI collective method (important that the outer for loop is over globalHessianValues rather than localHessianValues)
      
      std::set<GlobalIndexType> expectedCellsBelowThreshold;
      for (auto potentialMatchEntry : globalHessianValues)
      {
        if (potentialMatchEntry.second < errThreshold)
        {
          expectedCellsBelowThreshold.insert(potentialMatchEntry.first);
        }
      }
      
      auto numCellsExpected = expectedCellsBelowThreshold.size();
      auto numCellsActual   = actualCellsBelowThreshold.size();
      TEST_EQUALITY(numCellsExpected, numCellsActual);
      
      if (numCellsActual == numCellsExpected)
      {
        auto expectedIt = expectedCellsBelowThreshold.begin();
        auto actualIt   = actualCellsBelowThreshold.begin();
        for (; expectedIt != expectedCellsBelowThreshold.end(); expectedIt++, actualIt++)
        {
          TEST_EQUALITY(*expectedIt, *actualIt);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST(ErrorIndicator, LocalCellsAboveThreshold)
  {
    // set things up with the hessian error indicator as in HessianHighOrder
    // then check whether localCellsAboveErrorThreshold() behaves as expected.
    int meshWidth = 5, meshHeight = 5;
    
    SolutionPtr soln;
    VarPtr var;
    initializeSolution(soln, var, meshWidth, meshHeight);
    
    ErrorIndicatorPtr hessianIndicator = ErrorIndicator::hessianErrorIndicator(soln, var);
    hessianIndicator->measureError();
    
    MeshTopologyViewPtr meshTopo = soln->mesh()->getTopology()->getGatheredCopy();
    int spaceDim = meshTopo->getDimension();
    
    double tol = 1e-12;
    const set<GlobalIndexType>* myCells = &soln->mesh()->cellIDsInPartition();
    const map<GlobalIndexType,double> & hessianValues = hessianIndicator->localErrorMeasures();
    
    std::vector<std::pair<GlobalIndexType, double>> sortedVector(hessianValues.begin(), hessianValues.end());

    // Sort by error
    std::sort(sortedVector.begin(), sortedVector.end(), [](const auto& a, const auto& b) {
        return a.second < b.second; // Compare based on the error value
    });

    for (auto entryIt = sortedVector.begin(); entryIt != sortedVector.end(); entryIt++)
    {
      const double errThreshold = entryIt->second * (1. - 1e-13); // cheat slightly below this element, so that this element is included
      std::vector<GlobalIndexType> cellsAboveThreshold;
      hessianIndicator->localCellsAboveErrorThreshold(errThreshold, cellsAboveThreshold);
      std::set<GlobalIndexType> actualCellsAboveThreshold(cellsAboveThreshold.begin(),cellsAboveThreshold.end());
      
      std::set<GlobalIndexType> expectedCellsAboveThreshold;
      for (auto potentialMatchEntry : hessianValues)
      {
        if (potentialMatchEntry.second > errThreshold)
        {
          expectedCellsAboveThreshold.insert(potentialMatchEntry.first);
        }
      }
      
      auto numCellsExpected = expectedCellsAboveThreshold.size();
      auto numCellsActual   = cellsAboveThreshold.size();
      TEST_EQUALITY(numCellsExpected, numCellsActual);
      
      if (numCellsActual == numCellsExpected)
      {
        auto expectedIt = expectedCellsAboveThreshold.begin();
        auto actualIt   = actualCellsAboveThreshold.begin();
        for (; expectedIt != expectedCellsAboveThreshold.end(); expectedIt++, actualIt++)
        {
          TEST_EQUALITY(*expectedIt, *actualIt);
        }
      }
    }
  }
  
//  // meta-test: just to see that initializeSolution is doing what it should, by looking at it
//  TEUCHOS_UNIT_TEST(ErrorIndicator, VisualizeInitializedSolution)
//  {
//    int meshWidth = 3, meshHeight = 5;
//    SolutionPtr soln;
//    VarPtr var;
//    initializeSolution(soln, var, meshWidth, meshHeight);
//    HDF5Exporter exporter(soln->mesh());
//    exporter.exportSolution("/tmp", "constantSolution", soln);
//  }
} // namespace
