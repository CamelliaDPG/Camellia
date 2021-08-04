// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef RIESZ_REP
#define RIESZ_REP

/*
 *  RieszRep.h
 *
 *  Created by Jesse Chan on 10/22/12
 *
 */

#include "TypeDefs.h"

// Epetra includes
#include <Epetra_Map.h>

#include "Camellia_FieldContainer.hpp"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "Mesh.h"
#include "ElementType.h"
#include "Element.h"
#include "Function.h"

#include "LinearTerm.h"
#include "BasisCache.h"
#include "IP.h"

namespace Camellia
{
template <typename Scalar>
class TRieszRep
{
private:

  map<GlobalIndexType, Camellia::FieldContainer<Scalar> > _rieszRepDofs; // from cellID to dofs of riesz representation
  map<GlobalIndexType, double > _rieszRepNormSquared; // from cellID to norm squared of riesz inversion

  MeshPtr _mesh;
  TIPPtr<Scalar> _ip;
  TLinearTermPtr<Scalar> _functional;  // the RHS stuff here and below is misnamed -- should just be called functional
  bool _printAll;
  bool _repsNotComputed;
  
public:
  TRieszRep(MeshPtr mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> functional)
  {
    _mesh = mesh;
    _ip = ip;
    _functional = functional;
    _printAll = false;
    _repsNotComputed = true;
  }

  void setPrintOption(bool printAll)
  {
    _printAll = printAll;
  }

  void setFunctional(TLinearTermPtr<Scalar> functional)
  {
    _functional = functional;
  }

  TLinearTermPtr<Scalar> getFunctional();

  MeshPtr mesh();

  // for testing
  map<GlobalIndexType,Camellia::FieldContainer<Scalar> > integrateFunctional();

  void computeRieszRep(int cubatureEnrichment=0);

  double getNorm();

  // ! Returns reference to container for rank-local cells
  const map<GlobalIndexType,double> &getNormsSquared();

  void distributeDofs();

  void computeRepresentationValues(Camellia::FieldContainer<Scalar> &values, int testID, EOperator op, BasisCachePtr basisCache);

  double computeAlternativeNormSqOnCell(TIPPtr<Scalar> ip, GlobalIndexType cellID);
  map<GlobalIndexType,double> computeAlternativeNormSqOnCells(TIPPtr<Scalar> ip, vector<GlobalIndexType> cellIDs);

  // ! if the coefficients are not locally known, will initialize to 0 and return a Camellia::FieldContainer filled with 0s.
  const Camellia::FieldContainer<Scalar>& getCoefficientsForCell(GlobalIndexType cellID);
  
  // ! coefficients should be over all test dofs
  void setCoefficientsForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar>& coefficients);
  
  static TFunctionPtr<Scalar> repFunction( VarPtr var, TRieszRepPtr<Scalar> rep );
  static TRieszRepPtr<Scalar> rieszRep(MeshPtr mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> functional);
};

extern template class TRieszRep<double>;

template <typename Scalar>
class RepFunction : public TFunction<Scalar>
{
private:

  int _testID;
  TRieszRepPtr<Scalar> _rep;
  EOperator _op;
public:
  RepFunction( VarPtr var, TRieszRepPtr<Scalar> rep ) : TFunction<Scalar>( "RepFunction", var->rank() )
  {
    _testID = var->ID();
    _op = var->op();
    _rep = rep;
  }

  // optional specification of operator to apply - default to rank 0
  RepFunction(int testID, TRieszRepPtr<Scalar> rep, EOperator op): TFunction<Scalar>(0)
  {
    _testID = testID;
    _rep = rep;
    _op = op;
  }

  // specification of function rank
  RepFunction(int testID, TRieszRepPtr<Scalar> rep, EOperator op, int fxnRank): TFunction<Scalar>( "RepFunction", fxnRank )
  {
    _testID = testID;
    _rep = rep;
    _op = op;
  }


  TFunctionPtr<Scalar> x()
  {
    return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_X));
  }
  TFunctionPtr<Scalar> y()
  {
    return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_Y));
  }
  TFunctionPtr<Scalar> dx()
  {
    return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DX));
  }
  TFunctionPtr<Scalar> dy()
  {
    return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DY));
  }
  //  TFunctionPtr<Scalar> grad(){
  //    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_GRAD,2)); // default to 2 space dimensions
  //  }
  TFunctionPtr<Scalar> div()
  {
    return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DIV));
  }

  using TFunction<Scalar>::values;
  
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    _rep->computeRepresentationValues(values, _testID, _op, basisCache);
  }

  // for specifying an operator
  void values(Camellia::FieldContainer<Scalar> &values, EOperator op, BasisCachePtr basisCache)
  {
    _rep->computeRepresentationValues(values, _testID, op, basisCache);
  }
  
  size_t getCellDataSize(GlobalIndexType cellID)
  {
    auto numTestDofs = _rep->mesh()->getElementType(cellID)->testOrderPtr->totalDofs();
    return numTestDofs * sizeof(Scalar); // size in bytes
  }
  
  void packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
  {
    size_t requiredLength = getCellDataSize(cellID);
    TEUCHOS_TEST_FOR_EXCEPTION(requiredLength > bufferLength, std::invalid_argument, "Buffer length too small");
    
    auto & cellDofs = _rep->getCoefficientsForCell(cellID);
    size_t objSize = sizeof(Scalar);
    const Scalar* copyFromLocation = &cellDofs[0];
    memcpy(cellData, copyFromLocation, objSize * cellDofs.size());
  }
  
  size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength) // returns bytes consumed
  {
    int numDofs = _rep->mesh()->getElementType(cellID)->testOrderPtr->totalDofs();
    size_t numBytes = numDofs * sizeof(Scalar);
    if (numBytes > bufferLength)
    {
      std::cout << "In RepFunction, bufferLength " << bufferLength << " is not enough to fill required " << numBytes << " bytes.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(numBytes > bufferLength, std::invalid_argument, "buffer is too short");
    }
    Camellia::FieldContainer<Scalar> cellDofs(numDofs);
    Scalar* copyToLocation = &cellDofs[0];
    memcpy(copyToLocation, cellData, numBytes);
    _rep->setCoefficientsForCell(cellID, cellDofs);
    return numBytes;
  }
  
};

extern template class RepFunction<double>;
}



#endif
