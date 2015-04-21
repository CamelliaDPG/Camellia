#ifndef CAMELLIA_TEST_SUITE
#define CAMELLIA_TEST_SUITE

#include <string>

using namespace std;

#include "Intrepid_FieldContainer.hpp"

#include "MPIWrapper.h"

#include "Function.h"

#include "BasisCache.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;
using namespace Intrepid;

// abstract class for tests
class TestSuite {
public:
  virtual void runTests(int &numTestsRun, int &numTestsPassed) = 0;
  virtual string testSuiteName() = 0;

  static bool fcsAgree(const FieldContainer<double> &fc1, const FieldContainer<double> &fc2, double tol, double &maxDiff) {
    if (fc1.size() != fc2.size()) {
      maxDiff = -1.0; // a signal something's wrong…
      return false;
    }
    maxDiff = 0.0;
    for (int i=0; i<fc1.size(); i++) {
      maxDiff = max(maxDiff, abs(fc1[i] - fc2[i]));
    }
    return (maxDiff <= tol);
  }

  static bool allSuccess(bool mySuccess) {
    int mySuccessInt = mySuccess ? 0 : -1;
    int successSum = MPIWrapper::sum(mySuccessInt);
    return successSum == 0;
  }

  static void reportFunctionValueDifferences(const FieldContainer<double> &physicalPoints, const FieldContainer<double> &values1, const FieldContainer<double> &values2, double tol) {
    int rank = MPIWrapper::rank();
    if (rank != 0) {
      return;
    }
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    int valuesPerPoint = values1.size() / (numCells * numPoints);
    Teuchos::Array<int> indexArray;
    values1.dimensions(indexArray);
    int extraRanks = indexArray.size() - 2;
    for (int i=0; i<extraRanks; i++) {
      indexArray[i+2] = 0;
    }

    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      indexArray[0] = cellIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        indexArray[1] = ptIndex;
        int enumerationIndex = values1.getEnumeration(indexArray);
        const double *value1 = &values1[enumerationIndex];
        const double *value2 = &values2[enumerationIndex];
        bool valuesDiffer = false;
        double diff = 0;
        for (int valueIndex=0; valueIndex<valuesPerPoint; valueIndex++) {
          diff = abs(*value1-*value2);
          if (diff > tol) {
            valuesDiffer = true;
            break;
          }
          value1++;
          value2++;
        }
        if (valuesDiffer) {
          value1 = &values1[enumerationIndex];
          value2 = &values2[enumerationIndex];

          cout << "Function values differ by " << diff << " at point (";
          for (int d=0; d<spaceDim; d++) {
            cout << physicalPoints(cellIndex,ptIndex,d);
            if (d != spaceDim-1) {
              cout << ",";
            } else {
              cout << ")\n";
            }
          }
          cout << "Values:\nf1()\tf2():\n";
          for (int valueIndex=0; valueIndex<valuesPerPoint; valueIndex++) {
            cout << *value1 << "\t" << *value2 << "\n";
            value1++;
            value2++;
          }
        }
      }
    }
  }

  static void reportFunctionValueDifferences(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache, double tol) {
    TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "f1 and f2 must have same rank.");

    const FieldContainer<double>* physPointsPtr = &(basisCache->getPhysicalCubaturePoints());
    Teuchos::Array<int> indexArray;
    int numCells = physPointsPtr->dimension(0);
    int numPoints = physPointsPtr->dimension(1);
    int spaceDim = physPointsPtr->dimension(2);
    indexArray.push_back(numCells);
    indexArray.push_back(numPoints);
    for (int r=0; r<f1->rank(); r++) {
      indexArray.push_back(spaceDim);
    }
    FieldContainer<double> f1values(indexArray);
    FieldContainer<double> f2values(indexArray);
    f1->values(f1values,basisCache);
    f2->values(f2values,basisCache);
    reportFunctionValueDifferences(*physPointsPtr, f1values, f2values, tol);
  }

  static void reportFCDifferences(const FieldContainer<double> &values1, const FieldContainer<double> &values2, double tol) {
    int rank = MPIWrapper::rank();
    if (rank != 0) {
      return;
    }

    for (int i=0; i<values1.size(); i++) {
      double diff = abs(values1[i]-values2[i]);

      if (diff > tol) {
        cout << "values differ by " << diff << " at index " << i << ": ";
        cout << values1[i] << " ≠ " << values2[i] << endl;
      }
    }
  }

  static void serializeOutput(string label, const FieldContainer<double> &fc) {
    int numProcs=Teuchos::GlobalMPISession::getNProc();;
    int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    // serialize output:
    for (int i=0; i<numProcs; i++) {
      cout.flush();
      Comm.Barrier();
      if (i==rank) {
        cout << "on rank " << rank << ", " << label << " is:\n" << fc;
        cout.flush();
      }
    }
    Comm.Barrier();
  }

  virtual ~TestSuite() {}
};

#endif
