# 03 — Function Class Hierarchy (src/Function/)

> **STALENESS HEADER (annotated 2025)**: The architecture described here is largely accurate. Two areas are stale: (1) any Epetra mentions must be updated to Tpetra; (2) the dual host/device evaluation path via Kokkos is newer than this note and not fully reflected. Curvilinear mapping functions have been significantly extended -- see 20-curvilinear-geometry.md. File sizes in this note are not verified against the manifest.

## 1. Architecture Overview

Composable symbolic-numeric framework for mathematical functions in FE formulations.

- **Templated base**: TFunction<Scalar> (only double instantiated)
- **Reference-counted**: All via Teuchos::RCP; FunctionPtr = TFunctionPtr<double>
- **Rank system**: 0=scalar, 1=vector, 2=tensor
- **Operator overloading**: +, -, *, / produce SumFunction, ProductFunction, QuotientFunction expression trees
- **Symbolic differentiation**: Virtual dx(), dy(), dz(), dt(), di(int) with product/chain rules
- **Dual evaluation**: Legacy host-path values(FieldContainer&, BasisCachePtr) and device-path values(DeviceView&, ValueModificationType, BasisCachePtr) via Kokkos [NOTE: device path is newer; stale if not mentioned in original note]

## 2. Key Source Files

| File | Purpose |
|------|-------|
| Function.cpp (166KB) | Base class TFunction + ~10 private helper classes |
| SumFunction.cpp | Binary addition with derivative propagation |
| ProductFunction.cpp | Binary multiplication with product rule |
| QuotientFunction.cpp | Division with quotient rule |
| SimpleSolutionFunction.cpp | Extract FE solution values for a variable |
| PreviousSolutionFunction.cpp | Older solution extraction via LinearTerm |

## 3. Operator Overloads
- f1 * f2 -> ProductFunction (with dot-product contraction for matching ranks)
- f1 + f2 -> SumFunction (with constant folding for numeric operands)
- Unary: f->dx(), f->dy(), f->grad(), f->div(), f->curl()

## 4. Cross-Cutting Issues

1. **Dual evaluation paths**: The mutual-recursion between host and device evaluation paths is fragile. Default implementations call each other -- stack overflow if neither is overridden. [CONFIRMED STALE RISK: Kokkos path is newer]
2. **Epetra**: No direct Epetra dependency in Function layer itself, but downstream Solution assembly uses Tpetra. Verify any helper that calls TSolution methods.
3. **Curvilinear functions**: MappingFunction and related geometry functions have been updated. Cross-reference CamelliaCellTools.h and 20-curvilinear-geometry.md.
