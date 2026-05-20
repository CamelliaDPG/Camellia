#include "Teuchos_UnitTestHarness.hpp"

#include "TensorFunction.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorFunction, ConstantFunctionHasEmptyDependencyScope )
{
  auto f = TensorFunction::constant(3.0);
  TEST_ASSERT(f->dependencyScope().isEmpty());
  TEST_ASSERT(f->isAtomic());
  TEST_ASSERT(!f->isSum());
  TEST_ASSERT(!f->isProduct());
  TEST_ASSERT(!f->isUnaryComposition());
  TEST_ASSERT(!f->hasSolutionDependencies());
  TEST_ASSERT(f->couplingStructure().coupledComponentGroups().empty());
}

TEUCHOS_UNIT_TEST( TensorFunction, ComponentCoordinateHasSingleComponentScope )
{
  auto x2 = TensorFunction::componentCoordinate(2);
  TEST_ASSERT(x2->dependencyScope().dependsOnComponent(2));
  TEST_ASSERT(!x2->dependencyScope().dependsOnComponent(1));
  TEST_EQUALITY(x2->couplingStructure().coupledComponentGroups().size(), 1);
  TEST_ASSERT(x2->couplingStructure().coupledComponentGroups()[0].find(2) != x2->couplingStructure().coupledComponentGroups()[0].end());
}

TEUCHOS_UNIT_TEST( TensorFunction, PointEvaluation )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto c2 = TensorFunction::constant(2.0);
  auto f = x0 + c2;

  double value = f->value({1.5, 7.0});
  TEST_FLOATING_EQUALITY(value, 3.5, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, BatchedPointViewEvaluation )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f = x0 * x1;

  TensorFunction::PointView points("points", 3, 2);
  auto pointsHost = Kokkos::create_mirror_view(points);
  pointsHost(0,0) = 1.0; pointsHost(0,1) = 2.0;
  pointsHost(1,0) = 3.0; pointsHost(1,1) = 4.0;
  pointsHost(2,0) = 5.0; pointsHost(2,1) = 6.0;
  Kokkos::deep_copy(points, pointsHost);

  TensorFunction::ValueView values("values", 3);
  f->values(points, values);
  auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);

  TEST_FLOATING_EQUALITY(valuesHost(0),  2.0, 1e-14);
  TEST_FLOATING_EQUALITY(valuesHost(1), 12.0, 1e-14);
  TEST_FLOATING_EQUALITY(valuesHost(2), 30.0, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, SumAndProductExposeChildStructure )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto sum = x0 + x1;
  auto prod = x0 * x1;

  TEST_ASSERT(sum->isSum());
  TEST_ASSERT(prod->isProduct());
  TEST_ASSERT(prod->isSeparableProduct());
  TEST_EQUALITY(sum->childFunctions().size(), 2);
  TEST_EQUALITY(prod->childFunctions().size(), 2);
}

TEUCHOS_UNIT_TEST( TensorFunction, CompositionDependencyScopeIsUnion )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x2 = TensorFunction::componentCoordinate(2);
  auto f = x0 * x2;

  TEST_ASSERT(f->dependencyScope().dependsOnComponent(0));
  TEST_ASSERT(f->dependencyScope().dependsOnComponent(2));
  TEST_ASSERT(!f->dependencyScope().dependsOnComponent(1));
}

TEUCHOS_UNIT_TEST( TensorFunction, DependencySourceMetadataCanBeAttached )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  TensorFunctionDependencySource source(TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "E");
  auto dependent = TensorFunction::withDependencySource(x0, source);

  TEST_ASSERT(dependent->hasSolutionDependencies());
  TEST_EQUALITY(dependent->dependencySources().size(), 1);
  TEST_EQUALITY(dependent->dependencySources()[0], source);
}

TEUCHOS_UNIT_TEST( TensorFunction, CompositionCarriesDependencySources )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f0 = TensorFunction::withDependencySource(x0, TensorFunctionDependencySource(TENSOR_FUNCTION_SOURCE_SAME_TENSOR_MESH, "u"));
  auto f1 = TensorFunction::withDependencySource(x1, TensorFunctionDependencySource(TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH, "E"));

  auto product = f0 * f1;

  TEST_ASSERT(product->hasSolutionDependencies());
  TEST_EQUALITY(product->dependencySources().size(), 2);
}

TEUCHOS_UNIT_TEST( TensorFunction, ProductOfDisjointComponentsIsFullySeparable )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto product = x0 * x1;

  TEST_ASSERT(product->isProduct());
  TEST_ASSERT(product->isSeparableProduct());
  TEST_ASSERT(product->couplingStructure().isFullySeparable());
  TEST_ASSERT(!product->isPartiallyCoupled());
  TEST_ASSERT(!product->isFullyCoupled());
  TEST_EQUALITY(product->couplingStructure().coupledComponentGroups().size(), 2);
}

TEUCHOS_UNIT_TEST( TensorFunction, SumOfOverlappingDependenciesIsFullyCoupled )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x0_again = TensorFunction::componentCoordinate(0);
  auto coupled = x0 + x0_again;

  TEST_ASSERT(coupled->isFullyCoupled() || coupled->couplingStructure().coupledComponentGroups().size() == 1);
}

TEUCHOS_UNIT_TEST( TensorFunction, ProductOfOverlappingDependenciesIsFullyCoupled )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x0_again = TensorFunction::componentCoordinate(0);
  auto product = x0 * x0_again;

  TEST_ASSERT(product->isFullyCoupled() || product->couplingStructure().coupledComponentGroups().size() == 1);
  TEST_ASSERT(!product->isSeparableProduct());
}

TEUCHOS_UNIT_TEST( TensorFunction, UnaryCompositionPreservesDependencyScope )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto f = sin(x0);

  TEST_ASSERT(f->isUnaryComposition());
  TEST_ASSERT(f->dependencyScope().dependsOnComponent(0));
  TEST_ASSERT(!f->dependencyScope().dependsOnComponent(1));
}

TEUCHOS_UNIT_TEST( TensorFunction, SinOfCoordinateCoupledSum )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f = sin(x0 + x1);

  double value = f->value({0.25, 0.5});
  TEST_FLOATING_EQUALITY(value, std::sin(0.75), 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, ExpOfCoordinateCoupledProduct )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f = exp(x0 * x1);

  double value = f->value({2.0, 3.0});
  TEST_FLOATING_EQUALITY(value, std::exp(6.0), 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, DirectionalDerivativeOfCoordinate )
{
  auto x1 = TensorFunction::componentCoordinate(1);
  auto dx0 = x1->derivativeFunction(0);
  auto dx1 = x1->derivativeFunction(1);

  TEST_FLOATING_EQUALITY(dx0->value({2.0, 3.0}), 0.0, 1e-14);
  TEST_FLOATING_EQUALITY(dx1->value({2.0, 3.0}), 1.0, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, DirectionalDerivativeOfSinXPlusY )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f = sin(x0 + x1);
  auto dfdx0 = f->derivativeFunction(0);
  auto dfdx1 = f->derivativeFunction(1);

  double expected = std::cos(0.75);
  TEST_FLOATING_EQUALITY(dfdx0->value({0.25, 0.5}), expected, 1e-14);
  TEST_FLOATING_EQUALITY(dfdx1->value({0.25, 0.5}), expected, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunction, DirectionalDerivativeOfExpXY )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto x1 = TensorFunction::componentCoordinate(1);
  auto f = exp(x0 * x1);
  auto dfdx0 = f->derivativeFunction(0);
  auto dfdx1 = f->derivativeFunction(1);

  double expected_dfdx0 = 3.0 * std::exp(6.0);
  double expected_dfdx1 = 2.0 * std::exp(6.0);
  TEST_FLOATING_EQUALITY(dfdx0->value({2.0, 3.0}), expected_dfdx0, 1e-12);
  TEST_FLOATING_EQUALITY(dfdx1->value({2.0, 3.0}), expected_dfdx1, 1e-12);
}

TEUCHOS_UNIT_TEST( TensorFunction, UnaryCompositionPreservesDependencySources )
{
  auto x0 = TensorFunction::componentCoordinate(0);
  auto f = TensorFunction::withDependencySource(x0, TensorFunctionDependencySource(TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "E"));
  auto g = sin(f);

  TEST_ASSERT(g->hasSolutionDependencies());
  TEST_EQUALITY(g->dependencySources().size(), 1);
  TEST_EQUALITY(g->dependencySources()[0], TensorFunctionDependencySource(TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "E"));
}
}
