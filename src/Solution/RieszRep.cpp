//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "RieszRep.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

#include "SerialDenseWrapper.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"

#include "MPIWrapper.h"

using namespace Camellia;

template <typename Scalar>
TLinearTermPtr<Scalar> TRieszRep<Scalar>::getFunctional()
{
  return _functional;
}

template <typename Scalar>
MeshPtr TRieszRep<Scalar>::mesh()
{
  return _mesh;
}

template <typename Scalar>
map<GlobalIndexType,Camellia::FieldContainer<Scalar> > TRieszRep<Scalar>::integrateFunctional()
{
  // NVR: changed this to only return integrated values for rank-local cells.

  map<GlobalIndexType,Camellia::FieldContainer<Scalar> > cellRHS;
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    int cubEnrich = 0; // set to zero for release
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubEnrich);

    Camellia::FieldContainer<Scalar> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);

    Camellia::FieldContainer<Scalar> rhsVals(numTestDofs);
    for (int i = 0; i<numTestDofs; i++)
    {
      rhsVals(i) = rhsValues(0,i);
    }
    cellRHS[cellID] = rhsVals;
  }
  return cellRHS;
}

template <typename Scalar>
void TRieszRep<Scalar>::computeRieszRep(int cubatureEnrichment)
{
  _rieszRepNormSquared.clear();
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;

    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubatureEnrichment);

    Camellia::FieldContainer<Scalar> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);
    if (_printAll)
    {
      cout << "RieszRep: LinearTerm values for cell " << cellID << ":\n " << rhsValues << endl;
    }

    Camellia::FieldContainer<Scalar> ipMatrix(1,numTestDofs,numTestDofs);
    _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, basisCache);

    bool printOutRiesz = false;
    if (printOutRiesz)
    {
      cout << " ============================ In RIESZ ==========================" << endl;
      cout << "matrix: \n" << ipMatrix;
    }

    // TODO: replace this with something that uses Cholesky (see BF::factoredCholeskySolve())
    rhsValues.resize(numTestDofs,1);
    Camellia::FieldContainer<Scalar> rieszRepDofs = rhsValues; // copy so we can do the dot product below after solving
    ipMatrix.resize(numTestDofs,numTestDofs);
    rhsValues.resize(numTestDofs,1);
    
    // impose any BCs from the test space:
    if (_ip->hasBCsImposed())
    {
      auto testDofOrdinalsToZero = _ip->homogeneousBCDofOrdinalsForCell(_mesh,cellID);
      for (auto testDofOrdinal : testDofOrdinalsToZero)
      {
        rieszRepDofs(testDofOrdinal,0) = 0.0;
      }
    }
    
    int success = SerialDenseWrapper::solveSPDSystemLAPACKCholesky(rieszRepDofs, ipMatrix);// solveSystemUsingQR(rieszRepDofs, ipMatrix, rhsValues);

    if (success != 0)
    {
      cout << "TRieszRep<Scalar>::computeRieszRep: Solve FAILED with error: " << success << endl;
    }

    double normSquared = SerialDenseWrapper::dot(rieszRepDofs, rhsValues);
    _rieszRepNormSquared[cellID] = normSquared;

    if (printOutRiesz)
    {
      cout << "rhs: \n" << rhsValues;
      cout << "dofs: \n" << rieszRepDofs;
      cout << " ================================================================" << endl;
    }

    Camellia::FieldContainer<Scalar> dofs(numTestDofs);
    for (int i = 0; i<numTestDofs; i++)
    {
      dofs(i) = rieszRepDofs(i,0);
    }
    _rieszRepDofs[cellID] = dofs;
  }
  _repsNotComputed = false;
}

template <typename Scalar>
double TRieszRep<Scalar>::getNorm()
{

  if (_repsNotComputed)
  {
//    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  const set<GlobalIndexType>* myCells = &_mesh->cellIDsInPartition();
  
  double normSumLocal = 0.0;
  for (GlobalIndexType cellID : *myCells)
  {
    normSumLocal += _rieszRepNormSquared[cellID];
  }
  double normSumGlobal = 0.0;
  _mesh->Comm()->SumAll(&normSumLocal, &normSumGlobal, 1);
  return sqrt(normSumGlobal);
}

template <typename Scalar>
const map<GlobalIndexType,double> & TRieszRep<Scalar>::getNormsSquared()
{
  return _rieszRepNormSquared;
}

// computes riesz representation over a single element - map is from int (testID) to Camellia::FieldContainer of values (sized cellIndex, numPoints)
template <typename Scalar>
void TRieszRep<Scalar>::computeRepresentationValues(Camellia::FieldContainer<Scalar> &values, int testID, Camellia::EOperator op, BasisCachePtr basisCache)
{

  if (_repsNotComputed)
  {
//    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  int spaceDim = _mesh->getTopology()->getDimension();
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

  // all elems coming in should be of same type
  ElementTypePtr elemTypePtr = _mesh->getElementType(cellIDs[0]);
  DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
  int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, VOLUME_INTERIOR_SIDE_INDEX);
  BasisPtr testBasis = testOrderingPtr->getBasis(testID);

  bool testBasisIsVolumeBasis = (spaceDim == testBasis->domainTopology()->getDimension());
  bool useCubPointsSideRefCell = testBasisIsVolumeBasis && basisCache->isSideCache();

  Teuchos::RCP< const Camellia::FieldContainer<double> > transformedBasisValues = basisCache->getTransformedValues(testBasis,op,useCubPointsSideRefCell);

  int rank = values.rank() - 2; // if values are shaped as (C,P), scalar...
  if (rank > 1)
  {
    cout << "ranks greater than 1 not presently supported...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ranks greater than 1 not presently supported...");
  }

//  Camellia::print("cellIDs",cellIDs);

  values.initialize(0.0);
  for (int cellIndex = 0; cellIndex<numCells; cellIndex++)
  {
    int cellID = cellIDs[cellIndex];
    if (_rieszRepDofs.find(cellID) == _rieszRepDofs.end())
    {
      cout << "cellID " << cellID << " not found in _rieszRepDofs container.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
    }
    for (int j = 0; j<numTestDofsForVarID; j++)
    {
      int dofIndex = testOrderingPtr->getDofIndex(testID, j);
      for (int i = 0; i<numPoints; i++)
      {
        if (rank==0)
        {
          double basisValue = (*transformedBasisValues)(cellIndex,j,i);
          values(cellIndex,i) += basisValue*_rieszRepDofs[cellID](dofIndex);
        }
        else
        {
          for (int d = 0; d<spaceDim; d++)
          {
            double basisValue = (*transformedBasisValues)(cellIndex,j,i,d);
            values(cellIndex,i,d) += basisValue*_rieszRepDofs[cellID](dofIndex);
          }
        }
      }
    }
  }
}

template <typename Scalar>
map<GlobalIndexType,double> TRieszRep<Scalar>::computeAlternativeNormSqOnCells(TIPPtr<Scalar> ip, vector<GlobalIndexType> cellIDs)
{
  map<GlobalIndexType,double> altNorms;
  int numCells = cellIDs.size();
  for (int i = 0; i<numCells; i++)
  {
    altNorms[cellIDs[i]] = computeAlternativeNormSqOnCell(ip, cellIDs[i]);
  }
  return altNorms;
}

template <typename Scalar>
double TRieszRep<Scalar>::computeAlternativeNormSqOnCell(TIPPtr<Scalar> ip, GlobalIndexType cellID)
{
  Teuchos::RCP<DofOrdering> testOrdering= _mesh->getElementType(cellID)->testOrderPtr;
  bool testVsTest = true;
  Teuchos::RCP<BasisCache> basisCache =   BasisCache::basisCacheForCell(_mesh, cellID, testVsTest,1);

  int numDofs = testOrdering->totalDofs();
  Camellia::FieldContainer<Scalar> ipMat(1,numDofs,numDofs);
  ip->computeInnerProductMatrix(ipMat,testOrdering,basisCache);

  if (_rieszRepDofs.find(cellID) == _rieszRepDofs.end())
  {
    cout << "cellID " << cellID << " not found in _riesRepDofs container.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
  }
  
  double sum = 0.0;
  for (int i = 0; i<numDofs; i++)
  {
    for (int j = 0; j<numDofs; j++)
    {
      sum += _rieszRepDofs[cellID](i)*_rieszRepDofs[cellID](j)*ipMat(0,i,j);
    }
  }

  return sum;
}

template <typename Scalar>
TFunctionPtr<Scalar> TRieszRep<Scalar>::repFunction( VarPtr var, TRieszRepPtr<Scalar> rep )
{
  return Teuchos::rcp( new RepFunction<Scalar>(var, rep) );
}

template <typename Scalar>
TRieszRepPtr<Scalar> TRieszRep<Scalar>::rieszRep(MeshPtr mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> rhs)
{
  return Teuchos::rcp( new TRieszRep<Scalar>(mesh,ip,rhs) );
}

template <typename Scalar>
const Camellia::FieldContainer<Scalar> & TRieszRep<Scalar>::getCoefficientsForCell(GlobalIndexType cellID)
{
  if (_rieszRepDofs.find(cellID) == _rieszRepDofs.end())
  {
    // we'll return a container filled with 0s, but we're going to complain about it.
    std::cout << "WARNING: request for getCoefficientsForCell in RieszRep for cell " << cellID;
    std::cout << ", which does not have coefficients (locally) set; initializing coefficients to 0.0.\n";
    int testSize = _mesh->getElementType(cellID)->testOrderPtr->totalDofs();
    Camellia::FieldContainer<Scalar> coefficients(testSize);
    coefficients.initialize(0.0);
    _rieszRepDofs[cellID] = coefficients;
  }
  return _rieszRepDofs[cellID];
}

template <typename Scalar>
void TRieszRep<Scalar>::setCoefficientsForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &coefficients)
{
  int expectedTestSize = _mesh->getElementType(cellID)->testOrderPtr->totalDofs();
  if (coefficients.size() != expectedTestSize)
  {
    std::cout << "Coefficients size " << coefficients.size() << " does not match expected " << expectedTestSize <<std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coefficients.size() != testOrdering.totalDofs()");
  }
  _rieszRepDofs[cellID] = coefficients;
  _repsNotComputed = false; // if coefficients are being set manually, then don't allow them to be overwritten until/unless we get an explicit call to computeRieszRep()
}

namespace Camellia
{
template class TRieszRep<double>;
template class RepFunction<double>;
}
