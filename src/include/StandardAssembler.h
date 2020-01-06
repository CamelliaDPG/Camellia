// NOTE: This is deprecated by the conversion to Tpetra
#ifndef STANDARD_ASSEMBLER
#define STANDARD_ASSEMBLER

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#include "Assembler.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "BF.h" // to compute stiffness

#include "Element.h"

namespace Camellia
{
class StandardAssembler : public Assembler
{
  TSolutionPtr<double> _solution;
public:
  StandardAssembler(TSolutionPtr<double> solution)
  {
    _solution = solution;
  };
  Epetra_Map getPartMap();
  Epetra_FECrsMatrix initializeMatrix();
  Epetra_FEVector initializeVector();

  //  Teuchos::RCP<Epetra_LinearProblem> assembleProblem();
  //  Epetra_FECrsMatrix assembleProblem();
  void assembleProblem(Epetra_FECrsMatrix &globalStiffMatrix, Epetra_FEVector &rhsVector);

  void distributeDofs(Epetra_FEVector dofs);

  Camellia::FieldContainer<double> getRHS(ElementPtr elem);
  Camellia::FieldContainer<double> getOverdeterminedStiffness(ElementPtr elem);
  Camellia::FieldContainer<double> getIPMatrix(ElementPtr elem);
  int numTestDofsForElem(ElementPtr elem);
  int numTrialDofsForElem(ElementPtr elem);
  void applyBCs(Epetra_FECrsMatrix &globalStiffMatrix, Epetra_FEVector &rhsVector);
  Camellia::FieldContainer<double> getSubVector(Epetra_FEVector u, ElementPtr elem);
  void getElemData(ElementPtr elem, Camellia::FieldContainer<double> &K, Camellia::FieldContainer<double> &f);
  TSolutionPtr<double> solution()
  {
    return _solution;
  }
};
}

#endif
