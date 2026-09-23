// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  ApplyInverseTpetraOperatorAdapter.h
//  Camellia
//

#ifndef Camellia_ApplyInverseTpetraOperatorAdapter_h
#define Camellia_ApplyInverseTpetraOperatorAdapter_h

#include "GMGOperator.h"
#include "TypeDefs.h"

#include "Teuchos_Exceptions.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <stdexcept>
#include <string>

namespace Camellia
{
  /** \brief Adapts an object with GMGOperator-style ApplyInverse() semantics to
      the Tpetra::Operator::apply() interface expected by Belos preconditioners.

      Belos right preconditioners are supplied as Tpetra::Operator instances and
      are applied through apply(). GMGOperator exposes the desired preconditioner
      action through ApplyInverse(X,Y). This adapter makes that convention
      explicit: one apply() call computes one ApplyInverse() application and
      honors the Tpetra alpha/beta update contract,

        Y = beta * Y + alpha * ApplyInverse(X).

      The template is intentionally small so MatrixFreeInfrastructure tests can
      exercise the adapter contract with lightweight fake inverse operators while
      production code can use the GMGApplyInversePreconditioner alias below.
   */
  template <typename InverseOperator>
  class ApplyInverseTpetraOperatorAdapter : public TOperator<double>
  {
  public:
    using Scalar = double;
    using MultiVector = TVector<Scalar>;

    explicit ApplyInverseTpetraOperatorAdapter(const Teuchos::RCP<const InverseOperator> &inverseOperator,
                                               const std::string &operatorName = "ApplyInverseTpetraOperatorAdapter")
    : _inverseOperator(inverseOperator), _operatorName(operatorName)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_inverseOperator.is_null(), std::invalid_argument,
                                 _operatorName << " requires a non-null inverse operator");
    }

    void apply(const MultiVector &X, MultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta  = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::invalid_argument,
                                 _operatorName << " does not support transpose apply");
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                                 _operatorName << " requires X and Y to have the same number of vectors");

      MultiVector inverseResult(this->getRangeMap(), X.getNumVectors());
      const int err = _inverseOperator->ApplyInverse(X, inverseResult);
      TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error,
                                 _operatorName << " underlying ApplyInverse() returned error code " << err);

      Y.update(alpha, inverseResult, beta);
    }

    TMapConstPtr getDomainMap() const override
    {
      return _inverseOperator->getDomainMap();
    }

    TMapConstPtr getRangeMap() const override
    {
      return _inverseOperator->getRangeMap();
    }

    bool hasTransposeApply() const override
    {
      return false;
    }

  private:
    Teuchos::RCP<const InverseOperator> _inverseOperator;
    std::string _operatorName;
  };

  using GMGApplyInversePreconditioner = ApplyInverseTpetraOperatorAdapter<GMGOperator>;
}

#endif // Camellia_ApplyInverseTpetraOperatorAdapter_h
