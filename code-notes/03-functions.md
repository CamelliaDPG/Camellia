# 03 — Function Class Hierarchy (src/Function/)

## 1. Architecture Overview

Composable symbolic-numeric framework for mathematical functions in FE formulations.

- **Templated base**: `TFunction<Scalar>` (only `double` instantiated)
- **Reference-counted**: All via `Teuchos::RCP`; `FunctionPtr = TFunctionPtr<double>`
- **Rank system**: 0=scalar, 1=vector, 2=tensor
- **Operator overloading**: `+`, `-`, `*`, `/` produce SumFunction, ProductFunction, QuotientFunction expression trees
- **Symbolic differentiation**: Virtual `dx()`, `dy()`, `dz()`, `dt()`, `di(int)` with product/chain rules
- **Dual evaluation**: Legacy host-path `values(FieldContainer&, BasisCachePtr)` and device-path `values(DeviceView&, ValueModificationType, BasisCachePtr)` via Kokkos

---

## 2. File Organization

| File | Purpose |
|------|---------|
| Function.cpp (166KB) | Base class TFunction + ~10 private helper classes |
| SumFunction.cpp | Binary addition with derivative propagation |
| ProductFunction.cpp | Binary multiplication with product rule |
| QuotientFunction.cpp | Division with quotient rule |
| SimpleSolutionFunction.cpp | Extract FE solution values for a variable |
| PreviousSolutionFunction.cpp | Older solution extraction via LinearTerm |
| BasisSumFunction.cpp | Weighted sum of basis functions |
| ConstantScalarFunction.cpp | Scalar constants (with isZero optimization) |
| ConstantVectorFunction.cpp | Vector constants |
| TrigFunctions.cpp | sin, cos, tan with derivatives |
| ExpFunction.cpp | exp, ln with derivatives |
| MonomialFunctions.cpp | x^n, y^n, z^n, t^n power functions |
| ParametricCurve.cpp | Parametric curves for mesh geometry |
| ParametricSurface.cpp | Parametric surfaces for 3D geometry |
| MeshTransferFunction.cpp | Transfer values between meshes |
| MeshTransformationFunction.cpp | Curvilinear mesh coordinate transforms |
| BGKFunction.cpp | BGK collision operator for Boltzmann |
| MLCollisionFunction.cpp | ML-based collision operator |
| MaxwellianFunction.cpp | Maxwellian distribution function |
| TensorFunction.cpp | Tensor-valued (rank 2) functions |
| NeighborFunction.cpp | Evaluate on neighboring element |
| ParameterFunction.cpp | Mutable parameter wrapper |
| SpatiallyFilteredFunction.cpp | Spatial region restriction |
| PolarizedFunction.cpp | Interior (non-boundary) restriction |
| SideParityFunction.cpp | Canonical side orientation (+1/-1) |
| AbsFunction.cpp | Absolute value |
| MinMaxFunctions.cpp | Pointwise min/max |
| hFunction.cpp | Element diameter |
| TimeRampFunction.cpp | Smooth time ramp |
| BCFunction.cpp | BC as function |

---

## 3. Key Base Class API

### Core Virtual Methods
- `values(FieldContainer&, BasisCachePtr)` — host-path evaluation
- `values(DeviceView&, ValueModificationType, BasisCachePtr)` — device-path
- `boundaryValueOnly()` — true for fluxes/traces
- `rank()`, `isZero()`, `displayString()`

### Differentiation
- `dx()`, `dy()`, `dz()`, `dt()` — partial derivatives (return null if unimplemented)
- `grad()`, `div()`, `curl()` — vector calculus operators
- `di(int i)` — 1-based indexed derivative

### Static Factories
- `Function::constant(scalar)`, `::zero()`, `::null()`
- `::xn(n)`, `::yn(n)`, `::zn(n)`, `::tn(n)` — monomials
- `::normal()`, `::sideParity()`, `::h()` — mesh geometry
- `::solution(var, soln)` — FE solution extraction
- `::vectorize(f1, f2)`, `::composedFunction(f, g)`

### Integration
- `integrate(BasisCachePtr)` — scalar integral
- `integrate(mesh, tol)` — mesh-wide adaptive quadrature
- `evaluate(mesh, x, y, z)` — point evaluation with MPI coordination

### Operator Overloads
- `f1 * f2` → ProductFunction (with dot-product contraction for matching ranks)
- `f1 + f2` → SumFunction (with constant folding)
- `f1 / f2` → QuotientFunction
- `-f` → -1.0 * f
- Zero checks: `f->isZero()` returns other operand directly

---

## 4. Private Classes in Function.cpp

| Class | Purpose |
|-------|---------|
| ComponentFunction | Extracts component from vector function |
| CellBoundaryRestrictedFunction | Forces boundaryValueOnly=true |
| HeavisideFunction | Step function H(x-shift) |
| MeshBoundaryCharacteristicFunction | 1 on domain boundary |
| MeshSideSetCharacteristicFunction | 1 on tagged side set |
| MeshSkeletonCharacteristicFunction | 1 on mesh skeleton |
| ComposedFunction | f(g(x)) with chain rule |

---

## 5. Code Smells and Issues

1. **God file**: Function.cpp at 166KB contains base class + ~10 private helpers + dead code blocks
2. **Dual values path bootstrapping**: Mutual recursion between host/device paths with fragile re-entrance guard
3. **Inconsistent className tracking**: Many constructors pass "(missing class name)"
4. **dynamic_cast for constant folding**: Should use virtual isConstant()/constantValue()
5. **Null vs Zero semantics**: null()=undefined derivative vs zero()=zero function. Easy to confuse
6. **Global mutable state**: `callsToFunctionsWithoutDevicePathImplementations` map is not thread-safe
7. **Commented-out code**: Old HeavisideFunction variants, old inverse() method, host impl in BasisSumFunction
8. **Only double instantiated**: Template infrastructure unused for complex/multi-precision
9. **ComposedFunction limitations**: Not templated, Jacobian throws "not yet supported", performance concern
10. **ComponentFunction allocation**: Allocates full vector temp every call
