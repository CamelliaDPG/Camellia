//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Epetra_Operator_to_Epetra_Matrix.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Vector.h"

#include "E_to_T.hpp"
#include "T_to_E.hpp"

#include "TypeDefs.h"

#include <set>
#include <map>

namespace {

using namespace Camellia;

Epetra_Map mapFromBlockMap(Teuchos::RCP<Epetra_BlockMap> blockMap)
{
  return Epetra_Map(blockMap->NumGlobalElements64(), blockMap->NumMyElements(),
                    blockMap->MyGlobalElements64(), blockMap->IndexBase(),
                    blockMap->Comm());
}

class EOperatorShim : public Epetra_Operator
{
  const TOperator<double> &_tOp;
  bool _useTranspose = false;
  Epetra_Map _rangeMap;
  Epetra_Map _domainMap;
  const Epetra_Comm &_comm;
public:
  EOperatorShim(const TOperator<double> &tOp, const Epetra_Comm &comm)
  :
  _tOp(tOp),
  _comm(comm),
  _rangeMap(mapFromBlockMap(T_to_E::map(tOp.getRangeMap()))),
  _domainMap(mapFromBlockMap(T_to_E::map(tOp.getDomainMap())))
  {}
  
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override
  {
    auto tX = E_to_T::vector(Teuchos::rcp(&X,false));
    auto tY = E_to_T::vector(Teuchos::rcp(&Y,false));
    if (!_useTranspose)
      _tOp.apply(*tX, *tY, Teuchos::NO_TRANS);
    else
      _tOp.apply(*tX, *tY, Teuchos::TRANS);
    
    auto eY = T_to_E::vector(tY);
    Y.Update(1.0, *eY, 0.0);
    return 0;
  }
  
  int SetUseTranspose(bool UseTranspose) override
  {
    if (_tOp.hasTransposeApply() || !UseTranspose)
    {
      _useTranspose = UseTranspose;
      return 0;
    }
    else
    {
      return -1;
    }
  }
  
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override
  {
    return this->Apply(X,Y);
  }

  double NormInf() const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported");
  }
  
  const char * Label() const override
  {
    return "EOperatorShim";
  }

  //! Returns the current UseTranspose setting.
  bool UseTranspose() const override
  {
    return _useTranspose;
  }

  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  bool HasNormInf() const override
  {
    return false;
  }

  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  const Epetra_Comm & Comm() const override
  {
    return _comm;
  }

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const override
  {
    return _domainMap;
  }

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const override
  {
    return _rangeMap;
  }
  
};

} // anonymous namespace

namespace Camellia
{
Teuchos::RCP<Epetra_CrsMatrix> Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(const Epetra_Operator &op, const Epetra_Map &map)
{
  int numEntriesPerRow = 0;
  Teuchos::RCP<Epetra_FECrsMatrix> matrix = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, map, numEntriesPerRow));

  auto numRows = map.NumGlobalElements64();

  Epetra_Vector X(map);
  Epetra_Vector Y(map);

  double tol = 1e-15; // values below this will be considered 0

  for (long long rowIndex=0; rowIndex<numRows; rowIndex++)
  {
    int lid = map.LID(rowIndex);
    if (lid != -1)
    {
      X[lid] = 1.0;
    }
    op.ApplyInverse(X, Y);
    if (lid != -1)
    {
      X[lid] = 0.0;
    }

    std::vector<double> values;
    std::vector<long long> indices;
    for (int i=0; i<map.NumMyElements(); i++)
    {
      if (abs(Y[i]) > tol)
      {
        values.push_back(Y[i]);
        indices.push_back(map.GID64(i));
      }
    }

    matrix->InsertGlobalValues(rowIndex, static_cast<int>(values.size()), values.data(), indices.data());
  }

  matrix->GlobalAssemble();
  return matrix;
}

Teuchos::RCP<Epetra_CrsMatrix> Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(const TOperator<double> &op, const Epetra_Map &map)
{
  EOperatorShim eOp(op, map.Comm());
  return constructInverseMatrix(eOp, map);
}

}
