#include "TensorCoefficientExpansion.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  bool appendCoefficientAsSingleComponentFactor(const TensorFunctionPtr &coefficient,
                                                int componentCount,
                                                TensorCoefficientFactorTuple &componentFactors)
  {
    if (coefficient == Teuchos::null) return false;
    if (coefficient->hasSolutionDependencies()) return false;

    const auto &dependentComponents = coefficient->dependencyScope().componentOrdinals();
    if (dependentComponents.empty())
    {
      if (componentCount <= 0) return false;
      componentFactors[0] = (componentFactors[0] == Teuchos::null) ? coefficient : componentFactors[0] * coefficient;
      return true;
    }

    if (dependentComponents.size() == 1)
    {
      const int componentOrdinal = *dependentComponents.begin();
      if ((componentOrdinal < 0) || (componentOrdinal >= componentCount)) return false;
      componentFactors[componentOrdinal] = (componentFactors[componentOrdinal] == Teuchos::null) ? coefficient : componentFactors[componentOrdinal] * coefficient;
      return true;
    }

    return false;
  }

  TensorCoefficientFactorTuple productOfFactorTuples(const TensorCoefficientFactorTuple &left,
                                                     const TensorCoefficientFactorTuple &right)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(left.size() != right.size(), std::invalid_argument,
                               "TensorCoefficientExpansion factor tuples must have matching ranks.");
    TensorCoefficientFactorTuple product = left;
    for (size_t componentOrdinal=0; componentOrdinal<right.size(); componentOrdinal++)
    {
      if (right[componentOrdinal] == Teuchos::null) continue;
      product[componentOrdinal] = (product[componentOrdinal] == Teuchos::null) ? right[componentOrdinal]
                                                                             : product[componentOrdinal] * right[componentOrdinal];
    }
    return product;
  }
}

bool Camellia::appendTensorSeparableCoefficientFactorTerms(const TensorFunctionPtr &coefficient,
                                                           int componentCount,
                                                           TensorCoefficientFactorExpansion &coefficientFactorTerms)
{
  if (coefficient == Teuchos::null) return false;
  if (coefficient->hasSolutionDependencies()) return false;

  if (coefficient->isSum())
  {
    for (const auto &child : coefficient->childFunctions())
    {
      if (!appendTensorSeparableCoefficientFactorTerms(child, componentCount, coefficientFactorTerms)) return false;
    }
    return true;
  }

  if (coefficient->isProduct())
  {
    TensorCoefficientFactorExpansion productExpansion(1, TensorCoefficientFactorTuple(componentCount, Teuchos::null));
    for (const auto &child : coefficient->childFunctions())
    {
      TensorCoefficientFactorExpansion childExpansion;
      if (!appendTensorSeparableCoefficientFactorTerms(child, componentCount, childExpansion)) return false;

      TensorCoefficientFactorExpansion nextExpansion;
      nextExpansion.reserve(productExpansion.size() * childExpansion.size());
      for (const auto &leftFactors : productExpansion)
      {
        for (const auto &rightFactors : childExpansion)
        {
          nextExpansion.push_back(productOfFactorTuples(leftFactors, rightFactors));
        }
      }
      productExpansion.swap(nextExpansion);
    }
    coefficientFactorTerms.insert(coefficientFactorTerms.end(), productExpansion.begin(), productExpansion.end());
    return !productExpansion.empty();
  }

  TensorCoefficientFactorTuple componentFactors(componentCount, Teuchos::null);
  if (!appendCoefficientAsSingleComponentFactor(coefficient, componentCount, componentFactors)) return false;
  coefficientFactorTerms.push_back(componentFactors);
  return true;
}
