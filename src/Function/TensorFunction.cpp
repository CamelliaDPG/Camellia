#include "TensorFunction.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  std::vector<std::set<int>> canonicalizedGroups(const std::vector<std::set<int>> &groups)
  {
    std::vector<std::set<int>> nonemptyGroups;
    for (const auto &group : groups)
    {
      if (!group.empty()) nonemptyGroups.push_back(group);
    }

    bool changed = true;
    while (changed)
    {
      changed = false;
      std::vector<std::set<int>> mergedGroups;
      std::vector<bool> consumed(nonemptyGroups.size(), false);
      for (size_t i=0; i<nonemptyGroups.size(); i++)
      {
        if (consumed[i]) continue;
        std::set<int> merged = nonemptyGroups[i];
        consumed[i] = true;
        bool keepMerging = true;
        while (keepMerging)
        {
          keepMerging = false;
          for (size_t j=0; j<nonemptyGroups.size(); j++)
          {
            if (consumed[j]) continue;
            bool intersects = false;
            for (int c : nonemptyGroups[j])
            {
              if (merged.find(c) != merged.end())
              {
                intersects = true;
                break;
              }
            }
            if (intersects)
            {
              merged.insert(nonemptyGroups[j].begin(), nonemptyGroups[j].end());
              consumed[j] = true;
              keepMerging = true;
              changed = true;
            }
          }
        }
        mergedGroups.push_back(merged);
      }
      nonemptyGroups = mergedGroups;
    }
    return nonemptyGroups;
  }

  class ConstantTensorFunction : public TensorFunction
  {
    double _value;
  public:
    ConstantTensorFunction(double value,
                           const std::vector<TensorFunctionDependencySource> &dependencySources = {})
    : TensorFunction(TensorFunctionDependencyScope(),
                     TENSOR_FUNCTION_ATOMIC,
                     TensorFunctionCouplingStructure::atomic({}),
                     dependencySources),
      _value(value) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      Kokkos::deep_copy(valuesOut, _value);
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      return TensorFunction::constant(0.0);
    }

    std::string displayString() const override
    {
      std::ostringstream os;
      os << _value;
      return os.str();
    }
  };

  class ComponentCoordinateTensorFunction : public TensorFunction
  {
    int _componentOrdinal;
  public:
    ComponentCoordinateTensorFunction(int componentOrdinal,
                                      const std::vector<TensorFunctionDependencySource> &dependencySources = {})
    : TensorFunction(TensorFunctionDependencyScope::singleComponent(componentOrdinal),
                     TENSOR_FUNCTION_ATOMIC,
                     TensorFunctionCouplingStructure::atomic({componentOrdinal}),
                     dependencySources),
      _componentOrdinal(componentOrdinal) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_componentOrdinal < 0, std::invalid_argument,
                                 "ComponentCoordinateTensorFunction requires nonnegative component ordinal.");
      TEUCHOS_TEST_FOR_EXCEPTION(_componentOrdinal >= pointsIn.extent_int(1), std::invalid_argument,
                                 "ComponentCoordinateTensorFunction component ordinal out of range for point view.");

      auto coordSubview = Kokkos::subview(pointsIn, Kokkos::ALL, _componentOrdinal);
      Kokkos::deep_copy(valuesOut, coordSubview);
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      return TensorFunction::constant(componentOrdinal == _componentOrdinal ? 1.0 : 0.0);
    }

    std::string displayString() const override
    {
      std::ostringstream os;
      os << "x_" << _componentOrdinal;
      return os.str();
    }
  };

  class DecoratedTensorFunction : public TensorFunction
  {
    TensorFunctionPtr _wrapped;
  public:
    DecoratedTensorFunction(TensorFunctionPtr wrapped,
                            const std::vector<TensorFunctionDependencySource> &dependencySources)
    : TensorFunction(wrapped->dependencyScope(),
                     wrapped->structure(),
                     wrapped->couplingStructure(),
                     dependencySources),
      _wrapped(wrapped) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      _wrapped->values(pointsIn, valuesOut);
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      auto derivative = _wrapped->derivativeFunction(componentOrdinal);
      for (const auto &source : this->dependencySources())
      {
        derivative = TensorFunction::withDependencySource(derivative, source);
      }
      return derivative;
    }

    std::vector<TensorFunctionPtr> childFunctions() const override
    {
      return {_wrapped};
    }

    std::string displayString() const override
    {
      return _wrapped->displayString();
    }
  };

  class SumTensorFunction : public TensorFunction
  {
    TensorFunctionPtr _a, _b;
  public:
    SumTensorFunction(TensorFunctionPtr a, TensorFunctionPtr b)
    : TensorFunction([&]() {
        std::set<int> deps = a->dependencyScope().componentOrdinals();
        deps.insert(b->dependencyScope().componentOrdinals().begin(), b->dependencyScope().componentOrdinals().end());
        return TensorFunctionDependencyScope::components(deps);
      }(),
      TENSOR_FUNCTION_SUM,
      TensorFunctionCouplingStructure::combineForSum(a->couplingStructure(), b->couplingStructure()),
      [&]() {
        std::vector<TensorFunctionDependencySource> deps = a->dependencySources();
        deps.insert(deps.end(), b->dependencySources().begin(), b->dependencySources().end());
        return deps;
      }()),
      _a(a), _b(b) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      const int numPoints = valuesOut.extent_int(0);
      ValueView aValues("SumTensorFunction::aValues", numPoints);
      _a->values(pointsIn, aValues);
      _b->values(pointsIn, valuesOut);
      Kokkos::parallel_for("SumTensorFunction::values", numPoints,
      KOKKOS_LAMBDA (const int pointOrdinal)
      {
        valuesOut(pointOrdinal) += aValues(pointOrdinal);
      });
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      return _a->derivativeFunction(componentOrdinal) + _b->derivativeFunction(componentOrdinal);
    }

    std::vector<TensorFunctionPtr> childFunctions() const override
    {
      return {_a, _b};
    }

    std::string displayString() const override
    {
      return "(" + _a->displayString() + " + " + _b->displayString() + ")";
    }
  };

  class ProductTensorFunction : public TensorFunction
  {
    TensorFunctionPtr _a, _b;
  public:
    ProductTensorFunction(TensorFunctionPtr a, TensorFunctionPtr b)
    : TensorFunction([&]() {
        std::set<int> deps = a->dependencyScope().componentOrdinals();
        deps.insert(b->dependencyScope().componentOrdinals().begin(), b->dependencyScope().componentOrdinals().end());
        return TensorFunctionDependencyScope::components(deps);
      }(),
      TENSOR_FUNCTION_PRODUCT,
      TensorFunctionCouplingStructure::combineForProduct(a->couplingStructure(), b->couplingStructure()),
      [&]() {
        std::vector<TensorFunctionDependencySource> deps = a->dependencySources();
        deps.insert(deps.end(), b->dependencySources().begin(), b->dependencySources().end());
        return deps;
      }()),
      _a(a), _b(b) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      const int numPoints = valuesOut.extent_int(0);
      ValueView bValues("ProductTensorFunction::bValues", numPoints);
      _a->values(pointsIn, valuesOut);
      _b->values(pointsIn, bValues);
      Kokkos::parallel_for("ProductTensorFunction::values", numPoints,
      KOKKOS_LAMBDA (const int pointOrdinal)
      {
        valuesOut(pointOrdinal) *= bValues(pointOrdinal);
      });
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      return _a->derivativeFunction(componentOrdinal) * _b + _a * _b->derivativeFunction(componentOrdinal);
    }

    std::vector<TensorFunctionPtr> childFunctions() const override
    {
      return {_a, _b};
    }

    std::string displayString() const override
    {
      return "(" + _a->displayString() + " * " + _b->displayString() + ")";
    }
  };

  class UnaryTensorFunction : public TensorFunction
  {
    TensorUnaryFunctionType _unaryType;
    TensorFunctionPtr _arg;
  public:
    UnaryTensorFunction(TensorUnaryFunctionType unaryType, TensorFunctionPtr arg)
    : TensorFunction(arg->dependencyScope(),
                     TENSOR_FUNCTION_UNARY_COMPOSITION,
                     arg->couplingStructure(),
                     arg->dependencySources()),
      _unaryType(unaryType), _arg(arg) {}

    void values(PointView pointsIn, ValueView valuesOut) const override
    {
      _arg->values(pointsIn, valuesOut);
      const int numPoints = valuesOut.extent_int(0);
      switch (_unaryType)
      {
        case TENSOR_UNARY_EXP:
          Kokkos::parallel_for("UnaryTensorFunction::exp", numPoints,
          KOKKOS_LAMBDA (const int pointOrdinal)
          {
            valuesOut(pointOrdinal) = std::exp(valuesOut(pointOrdinal));
          });
          break;
        case TENSOR_UNARY_SIN:
          Kokkos::parallel_for("UnaryTensorFunction::sin", numPoints,
          KOKKOS_LAMBDA (const int pointOrdinal)
          {
            valuesOut(pointOrdinal) = std::sin(valuesOut(pointOrdinal));
          });
          break;
        case TENSOR_UNARY_COS:
          Kokkos::parallel_for("UnaryTensorFunction::cos", numPoints,
          KOKKOS_LAMBDA (const int pointOrdinal)
          {
            valuesOut(pointOrdinal) = std::cos(valuesOut(pointOrdinal));
          });
          break;
      }
    }

    TensorFunctionPtr derivativeFunction(int componentOrdinal) const override
    {
      auto darg = _arg->derivativeFunction(componentOrdinal);
      switch (_unaryType)
      {
        case TENSOR_UNARY_EXP:
          return darg * TensorFunction::unary(TENSOR_UNARY_EXP, _arg);
        case TENSOR_UNARY_SIN:
          return darg * TensorFunction::unary(TENSOR_UNARY_COS, _arg);
        case TENSOR_UNARY_COS:
          return TensorFunction::constant(-1.0) * darg * TensorFunction::unary(TENSOR_UNARY_SIN, _arg);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported unary function type.");
    }

    std::vector<TensorFunctionPtr> childFunctions() const override
    {
      return {_arg};
    }

    std::string displayString() const override
    {
      switch (_unaryType)
      {
        case TENSOR_UNARY_EXP: return "exp(" + _arg->displayString() + ")";
        case TENSOR_UNARY_SIN: return "sin(" + _arg->displayString() + ")";
        case TENSOR_UNARY_COS: return "cos(" + _arg->displayString() + ")";
      }
      return "unary(?)";
    }
  };
}

TensorFunctionDependencyScope::TensorFunctionDependencyScope()
{
}

TensorFunctionDependencyScope::TensorFunctionDependencyScope(const std::set<int> &componentOrdinals)
: _componentOrdinals(componentOrdinals)
{
}

const std::set<int> &TensorFunctionDependencyScope::componentOrdinals() const
{
  return _componentOrdinals;
}

bool TensorFunctionDependencyScope::dependsOnComponent(int componentOrdinal) const
{
  return _componentOrdinals.find(componentOrdinal) != _componentOrdinals.end();
}

bool TensorFunctionDependencyScope::isEmpty() const
{
  return _componentOrdinals.empty();
}

TensorFunctionDependencyScope TensorFunctionDependencyScope::singleComponent(int componentOrdinal)
{
  return TensorFunctionDependencyScope({componentOrdinal});
}

TensorFunctionDependencyScope TensorFunctionDependencyScope::components(const std::set<int> &componentOrdinals)
{
  return TensorFunctionDependencyScope(componentOrdinals);
}

TensorFunctionDependencySource::TensorFunctionDependencySource(TensorFunctionDependencySourceType sourceType,
                                                               const std::string &name)
: _sourceType(sourceType), _name(name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_name == "", std::invalid_argument,
                             "TensorFunctionDependencySource requires a nonempty name.");
}

TensorFunctionDependencySourceType TensorFunctionDependencySource::sourceType() const
{
  return _sourceType;
}

const std::string &TensorFunctionDependencySource::name() const
{
  return _name;
}

bool TensorFunctionDependencySource::operator==(const TensorFunctionDependencySource &other) const
{
  return (_sourceType == other._sourceType) && (_name == other._name);
}

TensorFunctionCouplingStructure::TensorFunctionCouplingStructure()
: _wasComposed(false)
{
}

TensorFunctionCouplingStructure::TensorFunctionCouplingStructure(const std::vector<std::set<int>> &coupledComponentGroups,
                                                                 bool wasComposed)
: _coupledComponentGroups(canonicalizedGroups(coupledComponentGroups)), _wasComposed(wasComposed)
{
}

const std::vector<std::set<int>> &TensorFunctionCouplingStructure::coupledComponentGroups() const
{
  return _coupledComponentGroups;
}

bool TensorFunctionCouplingStructure::wasComposed() const
{
  return _wasComposed;
}

bool TensorFunctionCouplingStructure::isFullySeparable() const
{
  if (_coupledComponentGroups.empty()) return true;
  if (!_wasComposed && _coupledComponentGroups.size() <= 1) return true;
  for (const auto &group : _coupledComponentGroups)
  {
    if (group.size() > 1) return false;
  }
  return _coupledComponentGroups.size() > 1;
}

bool TensorFunctionCouplingStructure::isFullyCoupled() const
{
  if (_coupledComponentGroups.empty()) return false;
  if (_coupledComponentGroups.size() == 1)
  {
    return _wasComposed || (_coupledComponentGroups[0].size() > 1);
  }
  return false;
}

bool TensorFunctionCouplingStructure::isPartiallyCoupled() const
{
  return (!_coupledComponentGroups.empty()) && !isFullySeparable() && !isFullyCoupled();
}

TensorFunctionCouplingStructure TensorFunctionCouplingStructure::atomic(const std::set<int> &components)
{
  if (components.empty()) return TensorFunctionCouplingStructure({}, false);
  return TensorFunctionCouplingStructure({components}, false);
}

TensorFunctionCouplingStructure TensorFunctionCouplingStructure::combineForSum(const TensorFunctionCouplingStructure &a,
                                                                               const TensorFunctionCouplingStructure &b)
{
  std::vector<std::set<int>> groups = a.coupledComponentGroups();
  groups.insert(groups.end(), b.coupledComponentGroups().begin(), b.coupledComponentGroups().end());
  return TensorFunctionCouplingStructure(groups, true);
}

TensorFunctionCouplingStructure TensorFunctionCouplingStructure::combineForProduct(const TensorFunctionCouplingStructure &a,
                                                                                   const TensorFunctionCouplingStructure &b)
{
  std::vector<std::set<int>> groups = a.coupledComponentGroups();
  groups.insert(groups.end(), b.coupledComponentGroups().begin(), b.coupledComponentGroups().end());
  return TensorFunctionCouplingStructure(groups, true);
}

TensorFunction::TensorFunction(const TensorFunctionDependencyScope &dependencyScope,
                               TensorFunctionStructure structure,
                               const TensorFunctionCouplingStructure &couplingStructure,
                               const std::vector<TensorFunctionDependencySource> &dependencySources)
: _dependencyScope(dependencyScope), _structure(structure), _couplingStructure(couplingStructure), _dependencySources(dependencySources)
{
}

const TensorFunctionDependencyScope &TensorFunction::dependencyScope() const
{
  return _dependencyScope;
}

TensorFunctionStructure TensorFunction::structure() const
{
  return _structure;
}

const TensorFunctionCouplingStructure &TensorFunction::couplingStructure() const
{
  return _couplingStructure;
}

bool TensorFunction::isAtomic() const
{
  return _structure == TENSOR_FUNCTION_ATOMIC;
}

bool TensorFunction::isSum() const
{
  return _structure == TENSOR_FUNCTION_SUM;
}

bool TensorFunction::isProduct() const
{
  return _structure == TENSOR_FUNCTION_PRODUCT;
}

bool TensorFunction::isUnaryComposition() const
{
  return _structure == TENSOR_FUNCTION_UNARY_COMPOSITION;
}

bool TensorFunction::isSeparableProduct() const
{
  return isProduct() && _couplingStructure.isFullySeparable();
}

bool TensorFunction::isPartiallyCoupled() const
{
  return _couplingStructure.isPartiallyCoupled();
}

bool TensorFunction::isFullyCoupled() const
{
  return _couplingStructure.isFullyCoupled();
}

bool TensorFunction::hasSolutionDependencies() const
{
  return !_dependencySources.empty();
}

const std::vector<TensorFunctionDependencySource> &TensorFunction::dependencySources() const
{
  return _dependencySources;
}

void TensorFunction::values(PointViewHost pointsIn, ValueViewHost valuesOut) const
{
  auto pointsDevice = Kokkos::create_mirror_view_and_copy(DeviceType::execution_space(), pointsIn);
  auto valuesDevice = Kokkos::create_mirror_view(DeviceType::execution_space(), valuesOut);
  values(pointsDevice, valuesDevice);
  Kokkos::deep_copy(valuesOut, valuesDevice);
}

double TensorFunction::value(const std::vector<double> &coordVector) const
{
  PointView points("TensorFunction::value single-point points", 1, coordVector.size());
  auto pointsHost = Kokkos::create_mirror_view(points);
  for (size_t d=0; d<coordVector.size(); d++)
  {
    pointsHost(0,d) = coordVector[d];
  }
  Kokkos::deep_copy(points, pointsHost);

  ValueView valueView("TensorFunction::value single-point values", 1);
  this->values(points, valueView);
  auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), valueView);
  return valuesHost(0);
}

TensorFunctionPtr TensorFunction::constant(double value)
{
  return Teuchos::rcp(new ConstantTensorFunction(value));
}

TensorFunctionPtr TensorFunction::componentCoordinate(int componentOrdinal)
{
  return Teuchos::rcp(new ComponentCoordinateTensorFunction(componentOrdinal));
}

TensorFunctionPtr TensorFunction::centeredMaxwellian1D(int velocityComponentOrdinal,
                                                       double density,
                                                       double mean,
                                                       double temperature)
{
  TEUCHOS_TEST_FOR_EXCEPTION(velocityComponentOrdinal < 0, std::invalid_argument,
                             "TensorFunction::centeredMaxwellian1D requires nonnegative velocity component ordinal.");
  TEUCHOS_TEST_FOR_EXCEPTION(temperature <= 0.0, std::invalid_argument,
                             "TensorFunction::centeredMaxwellian1D requires positive temperature.");

  const double normalization = density / std::sqrt(2.0 * M_PI * temperature);
  auto v = TensorFunction::componentCoordinate(velocityComponentOrdinal);
  auto centered = v + TensorFunction::constant(-mean);
  auto exponent = TensorFunction::constant(-0.5 / temperature) * centered * centered;
  return TensorFunction::constant(normalization) * exp(exponent);
}

TensorFunctionPtr TensorFunction::shiftedMaxwellian1D(int velocityComponentOrdinal,
                                                      double density,
                                                      double shift,
                                                      double temperature)
{
  return centeredMaxwellian1D(velocityComponentOrdinal, density, shift, temperature);
}

TensorFunctionPtr TensorFunction::withDependencySource(TensorFunctionPtr f,
                                                       const TensorFunctionDependencySource &dependencySource)
{
  TEUCHOS_TEST_FOR_EXCEPTION(f == Teuchos::null, std::invalid_argument,
                             "TensorFunction::withDependencySource requires a non-null TensorFunction.");
  std::vector<TensorFunctionDependencySource> deps = f->dependencySources();
  deps.push_back(dependencySource);
  return Teuchos::rcp(new DecoratedTensorFunction(f, deps));
}

TensorFunctionPtr TensorFunction::unary(TensorUnaryFunctionType unaryType, TensorFunctionPtr arg)
{
  TEUCHOS_TEST_FOR_EXCEPTION(arg == Teuchos::null, std::invalid_argument,
                             "TensorFunction::unary requires a non-null argument function.");
  return Teuchos::rcp(new UnaryTensorFunction(unaryType, arg));
}

std::vector<TensorFunctionPtr> TensorFunction::childFunctions() const
{
  return {};
}

TensorFunctionPtr Camellia::operator+(TensorFunctionPtr a, TensorFunctionPtr b)
{
  TEUCHOS_TEST_FOR_EXCEPTION(a == Teuchos::null, std::invalid_argument,
                             "TensorFunction sum requires non-null left operand.");
  TEUCHOS_TEST_FOR_EXCEPTION(b == Teuchos::null, std::invalid_argument,
                             "TensorFunction sum requires non-null right operand.");
  return Teuchos::rcp(new SumTensorFunction(a,b));
}

TensorFunctionPtr Camellia::operator*(TensorFunctionPtr a, TensorFunctionPtr b)
{
  TEUCHOS_TEST_FOR_EXCEPTION(a == Teuchos::null, std::invalid_argument,
                             "TensorFunction product requires non-null left operand.");
  TEUCHOS_TEST_FOR_EXCEPTION(b == Teuchos::null, std::invalid_argument,
                             "TensorFunction product requires non-null right operand.");
  return Teuchos::rcp(new ProductTensorFunction(a,b));
}

TensorFunctionPtr Camellia::exp(TensorFunctionPtr arg)
{
  return TensorFunction::unary(TENSOR_UNARY_EXP, arg);
}

TensorFunctionPtr Camellia::sin(TensorFunctionPtr arg)
{
  return TensorFunction::unary(TENSOR_UNARY_SIN, arg);
}

TensorFunctionPtr Camellia::cos(TensorFunctionPtr arg)
{
  return TensorFunction::unary(TENSOR_UNARY_COS, arg);
}
