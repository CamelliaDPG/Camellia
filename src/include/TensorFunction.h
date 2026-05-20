#ifndef Camellia_TensorFunction_h
#define Camellia_TensorFunction_h

#include <set>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "TypeDefs.h"

namespace Camellia
{
  class TensorFunction;
  using TensorFunctionPtr = Teuchos::RCP<TensorFunction>;

  /**
   *  \brief High-level description of tensor-function dependency scope.
   *
   *  The canonical internal form is dependence by component ordinal. Human-facing coordinate labels
   *  may still be useful elsewhere, but this object records the dependence set in the form most natural
   *  to the tensor-native mesh/component stack.
   */
  class TensorFunctionDependencyScope
  {
    std::set<int> _componentOrdinals;
  public:
    TensorFunctionDependencyScope();
    TensorFunctionDependencyScope(const std::set<int> &componentOrdinals);

    const std::set<int> &componentOrdinals() const;
    bool dependsOnComponent(int componentOrdinal) const;
    bool isEmpty() const;

    static TensorFunctionDependencyScope singleComponent(int componentOrdinal);
    static TensorFunctionDependencyScope components(const std::set<int> &componentOrdinals);
  };

  enum TensorFunctionStructure
  {
    TENSOR_FUNCTION_ATOMIC,
    TENSOR_FUNCTION_SUM,
    TENSOR_FUNCTION_PRODUCT,
    TENSOR_FUNCTION_UNARY_COMPOSITION
  };

  enum TensorFunctionDependencySourceType
  {
    TENSOR_FUNCTION_SOURCE_SAME_TENSOR_MESH,
    TENSOR_FUNCTION_SOURCE_RELATED_TENSOR_MESH,
    TENSOR_FUNCTION_SOURCE_CONFIGURATION_SPACE_MESH,
    TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION
  };

  /**
   *  \brief Metadata-only description of a solution dependency for a TensorFunction.
   *
   *  This is intentionally a descriptor rather than a full evaluator bridge. It allows the coefficient layer
   *  to declare dependence on a solution source without forcing the present issue to implement the full
   *  cross-mesh or operator-kernel evaluation path.
   */
  class TensorFunctionDependencySource
  {
    TensorFunctionDependencySourceType _sourceType;
    std::string _name;
  public:
    TensorFunctionDependencySource(TensorFunctionDependencySourceType sourceType,
                                   const std::string &name);

    TensorFunctionDependencySourceType sourceType() const;
    const std::string &name() const;

    bool operator==(const TensorFunctionDependencySource &other) const;
  };

  inline std::ostream &operator<<(std::ostream &os, const TensorFunctionDependencySource &source)
  {
    os << "TensorFunctionDependencySource(type=" << static_cast<int>(source.sourceType())
       << ",name=" << source.name() << ")";
    return os;
  }

  /**
   *  \brief Explicit metadata describing tensor-function coupling structure by component groups.
   *
   *  Each group lists tensor components that are coupled together. Distinct groups indicate separability.
   *  The additional composition provenance flag is used to distinguish atomic singleton groups from structure
   *  obtained by composing multiple child functions with overlapping dependency sets.
   */
  class TensorFunctionCouplingStructure
  {
    std::vector<std::set<int>> _coupledComponentGroups;
    bool _wasComposed;
  public:
    TensorFunctionCouplingStructure();
    TensorFunctionCouplingStructure(const std::vector<std::set<int>> &coupledComponentGroups,
                                    bool wasComposed = false);

    const std::vector<std::set<int>> &coupledComponentGroups() const;
    bool wasComposed() const;
    bool isFullySeparable() const;
    bool isFullyCoupled() const;
    bool isPartiallyCoupled() const;

    static TensorFunctionCouplingStructure atomic(const std::set<int> &components);
    static TensorFunctionCouplingStructure combineForSum(const TensorFunctionCouplingStructure &a,
                                                         const TensorFunctionCouplingStructure &b);
    static TensorFunctionCouplingStructure combineForProduct(const TensorFunctionCouplingStructure &a,
                                                             const TensorFunctionCouplingStructure &b);
  };

  enum TensorUnaryFunctionType
  {
    TENSOR_UNARY_EXP,
    TENSOR_UNARY_SIN,
    TENSOR_UNARY_COS
  };

  /**
   *  \brief Small tensor-native coefficient/function abstraction inspired by the Infusion prototype.
   *
   *  TensorFunction is intentionally not rooted in Camellia's existing Function/BasisCache evaluation model.
   *  The current implementation is point-view oriented, Kokkos-friendly, and suitable for later operator consumption.
   *
   *  In the current first draft, TensorFunction supports:
   *  \begin{itemize}
   *    \item point and batched point-view evaluation,
   *    \item explicit dependency scope by component ordinal,
   *    \item metadata-first solution dependency hooks,
   *    \item explicit separable / partially coupled / fully coupled structure,
   *    \item sum/product composition,
   *    \item unary nonlinear composition through exp, sin, and cos,
   *    \item and directional derivative-function construction.
   *  \end{itemize}
   */
  class TensorFunction
  {
  public:
    using ValueView = Kokkos::View<double*, DeviceType>;
    using PointView = Kokkos::View<double**, DeviceType>; // (P,D)
    using ValueViewHost = typename ValueView::HostMirror;
    using PointViewHost = typename PointView::HostMirror;

  private:
    TensorFunctionDependencyScope _dependencyScope;
    TensorFunctionStructure _structure;
    TensorFunctionCouplingStructure _couplingStructure;
    std::vector<TensorFunctionDependencySource> _dependencySources;

  protected:
    TensorFunction(const TensorFunctionDependencyScope &dependencyScope,
                   TensorFunctionStructure structure,
                   const TensorFunctionCouplingStructure &couplingStructure,
                   const std::vector<TensorFunctionDependencySource> &dependencySources = {});

  public:
    virtual ~TensorFunction() {}

    const TensorFunctionDependencyScope &dependencyScope() const;
    TensorFunctionStructure structure() const;
    const TensorFunctionCouplingStructure &couplingStructure() const;
    bool isAtomic() const;
    bool isSum() const;
    bool isProduct() const;
    bool isUnaryComposition() const;
    bool isSeparableProduct() const;
    bool isPartiallyCoupled() const;
    bool isFullyCoupled() const;

    bool hasSolutionDependencies() const;
    const std::vector<TensorFunctionDependencySource> &dependencySources() const;

    virtual void values(PointView pointsIn, ValueView valuesOut) const = 0;
    virtual void values(PointViewHost pointsIn, ValueViewHost valuesOut) const;

    virtual double value(const std::vector<double> &coordVector) const;
    virtual TensorFunctionPtr derivativeFunction(int componentOrdinal) const = 0;

    virtual std::vector<TensorFunctionPtr> childFunctions() const;
    virtual std::string displayString() const = 0;

    static TensorFunctionPtr constant(double value);
    static TensorFunctionPtr componentCoordinate(int componentOrdinal);
    static TensorFunctionPtr withDependencySource(TensorFunctionPtr f,
                                                  const TensorFunctionDependencySource &dependencySource);
    static TensorFunctionPtr unary(TensorUnaryFunctionType unaryType, TensorFunctionPtr arg);
  };

  TensorFunctionPtr operator+(TensorFunctionPtr a, TensorFunctionPtr b);
  TensorFunctionPtr operator*(TensorFunctionPtr a, TensorFunctionPtr b);
  TensorFunctionPtr exp(TensorFunctionPtr arg);
  TensorFunctionPtr sin(TensorFunctionPtr arg);
  TensorFunctionPtr cos(TensorFunctionPtr arg);

  inline std::ostream &operator<<(std::ostream &os, const TensorFunction &f)
  {
    os << f.displayString();
    return os;
  }
}

#endif
