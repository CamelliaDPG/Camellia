#include "Teuchos_UnitTestHarness.hpp"

#include "ApplyInverseTpetraOperatorAdapter.h"
#include "Camellia.h"
#include "TypeDefs.h"

#include "Tpetra_Core.hpp"

using namespace Camellia;

namespace
{
  class FakeApplyInverseOperator
  {
  public:
    FakeApplyInverseOperator(const TMapConstPtr &map, const double scale, const int returnCode = 0)
    : _map(map), _scale(scale), _returnCode(returnCode)
    {}

    int ApplyInverse(const TVector<double> &X, TVector<double> &Y) const
    {
      if (_returnCode != 0)
        return _returnCode;

      Y.update(_scale, X, 0.0);
      return 0;
    }

    TMapConstPtr getDomainMap() const
    {
      return _map;
    }

    TMapConstPtr getRangeMap() const
    {
      return _map;
    }

  private:
    TMapConstPtr _map;
    double _scale;
    int _returnCode;
  };

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseAdapterAppliesInverseAction )
  {
    auto comm = Tpetra::getDefaultComm();
    TMapConstPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 4, 0, comm));

    auto inverseOperator = Teuchos::rcp(new FakeApplyInverseOperator(map, 2.0));
    ApplyInverseTpetraOperatorAdapter<FakeApplyInverseOperator> adapter(inverseOperator, "fake inverse adapter");

    TVector<double> x(map, 2);
    TVector<double> y(map, 2);

    for (size_t vectorOrdinal=0; vectorOrdinal<x.getNumVectors(); vectorOrdinal++)
    {
      auto xData = x.getDataNonConst(vectorOrdinal);
      auto yData = y.getDataNonConst(vectorOrdinal);
      for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
      {
        xData[localOrdinal] = 1.0 + static_cast<double>(localOrdinal) + 10.0 * static_cast<double>(vectorOrdinal);
        yData[localOrdinal] = -1.0;
      }
    }

    adapter.apply(x, y);

    for (size_t vectorOrdinal=0; vectorOrdinal<x.getNumVectors(); vectorOrdinal++)
    {
      auto xData = x.getData(vectorOrdinal);
      auto yData = y.getData(vectorOrdinal);
      for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
      {
        TEST_FLOATING_EQUALITY(2.0 * xData[localOrdinal], yData[localOrdinal], 1e-15);
      }
    }

    TEST_EQUALITY(adapter.getDomainMap().get(), map.get());
    TEST_EQUALITY(adapter.getRangeMap().get(),  map.get());
    TEST_EQUALITY(adapter.hasTransposeApply(), false);
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseAdapterHonorsAlphaBetaUpdate )
  {
    auto comm = Tpetra::getDefaultComm();
    TMapConstPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 3, 0, comm));

    auto inverseOperator = Teuchos::rcp(new FakeApplyInverseOperator(map, 2.0));
    ApplyInverseTpetraOperatorAdapter<FakeApplyInverseOperator> adapter(inverseOperator, "fake inverse adapter");

    TVector<double> x(map, 1);
    TVector<double> y(map, 1);

    auto xData = x.getDataNonConst(0);
    auto yData = y.getDataNonConst(0);
    for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
    {
      xData[localOrdinal] = 1.0 + static_cast<double>(localOrdinal);
      yData[localOrdinal] = 5.0;
    }

    const double alpha = 3.0;
    const double beta  = 4.0;
    adapter.apply(x, y, Teuchos::NO_TRANS, alpha, beta);

    auto xConstData = x.getData(0);
    auto yConstData = y.getData(0);
    for (size_t localOrdinal=0; localOrdinal<x.getLocalLength(); localOrdinal++)
    {
      const double expected = beta * 5.0 + alpha * 2.0 * xConstData[localOrdinal];
      TEST_FLOATING_EQUALITY(expected, yConstData[localOrdinal], 1e-15);
    }
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseAdapterRejectsUnsupportedTranspose )
  {
    auto comm = Tpetra::getDefaultComm();
    TMapConstPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 2, 0, comm));

    auto inverseOperator = Teuchos::rcp(new FakeApplyInverseOperator(map, 2.0));
    ApplyInverseTpetraOperatorAdapter<FakeApplyInverseOperator> adapter(inverseOperator, "fake inverse adapter");

    TVector<double> x(map, 1);
    TVector<double> y(map, 1);

    TEST_THROW(adapter.apply(x, y, Teuchos::TRANS), std::invalid_argument);
  }

  TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseAdapterReportsUnderlyingErrors )
  {
    auto comm = Tpetra::getDefaultComm();
    TMapConstPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 2, 0, comm));

    auto inverseOperator = Teuchos::rcp(new FakeApplyInverseOperator(map, 2.0, 7));
    ApplyInverseTpetraOperatorAdapter<FakeApplyInverseOperator> adapter(inverseOperator, "fake inverse adapter");

    TVector<double> x(map, 1);
    TVector<double> y(map, 1);

    TEST_THROW(adapter.apply(x, y), std::runtime_error);
  }
}
