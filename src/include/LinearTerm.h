// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  TLinearTerm.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_TLinearTerm_h
#define Camellia_TLinearTerm_h

#include "TypeDefs.h"

#include "IP.h"

#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "Intrepid_Utils.hpp"
#include "Intrepid_Basis.hpp"
#include "Var.h"

#include "DofOrdering.h"

namespace Camellia
{
  const static string LINEAR_TERM_INTEGRATION_TIMER_1_STRING = "linear term integration 1";
  const static string LINEAR_TERM_INTEGRATION_TIMER_2_STRING = "linear term integration 2";
  const static string LINEAR_TERM_INTEGRATION_TIMER_3_STRING = "linear term integration 3";
  const static string LINEAR_TERM_INTEGRATION_TIMER_4_STRING = "linear term integration 4";

template <typename Scalar>
class TLinearTerm
{
  // typedef std::pair< TFunctionPtr<Scalar>, VarPtr > TLinearSummand;
  int _rank; // gets set after first var is added
  std::vector< TLinearSummand<Scalar> > _summands;
  std::set<int> _varIDs;
  Camellia::VarType _termType; // shouldn't mix

  // for the Riesz inversion evaluation
  std::map< ElementType*, Camellia::FieldContainer<double> > _rieszRepresentationForElementType;
  std::map< ElementType*, Camellia::FieldContainer<double> > _rieszRHSForElementType;
  std::map< int, double > _energyNormForCellIDGlobal;

  // some private utility methods:
  static void integrate(Camellia::FieldContainer<Scalar> &values,
                        TLinearTermPtr<Scalar> u, DofOrderingPtr uOrdering,
                        TLinearTermPtr<Scalar> v, DofOrderingPtr vOrdering,
                        BasisCachePtr basisCache, bool sumInto=true);
  static void multiplyFluxValuesByParity(Camellia::FieldContainer<Scalar> &fluxValues, BasisCachePtr sideBasisCache);

public: // was protected; changed for debugging (no big deal either way, I think)
  const std::vector< TLinearSummand<Scalar> > & summands() const;
public:
  TLinearTerm();
  TLinearTerm(TFunctionPtr<Scalar> weight, VarPtr var);
  TLinearTerm(Scalar weight, VarPtr var);
  TLinearTerm(std::vector<Scalar> weight, VarPtr var);
  TLinearTerm( VarPtr v );
  // copy constructor:
  TLinearTerm( const TLinearTerm &a );
  void addVar(TFunctionPtr<Scalar> weight, VarPtr var);
  void addVar(Scalar weight, VarPtr var);
  void addVar(std::vector<Scalar> vector_weight, VarPtr var);

  const std::set<int> & varIDs() const;

  Camellia::VarType termType() const;
  //  vector< Camellia::EOperator > varOps(int varID);

  /** \brief  Computes the norm of the TLinearTerm using the Riesz representation corresponding to the inner product ip on the specified mesh.
   *  \param  ip        [in]  - inner product to use for the Riesz representation
   *  \param  mesh      [in]  - mesh over which to measure
   *  \param  cubatureEnrichment - degree to raise the cubature over the default (2 * test poly order)
   */
  double computeNorm(TIPPtr<Scalar> ip, MeshPtr mesh, int cubatureEnrichment = 0);

  void evaluate(Camellia::FieldContainer<Scalar> &values, TSolutionPtr<Scalar> solution, BasisCachePtr basisCache,
                bool applyCubatureWeights = false, SolutionComponent solutionComponent = STANDARD_SOLUTION, int solutionStage = 0);

  TFunctionPtr<Scalar> evaluate(const Teuchos::map< int, TFunctionPtr<Scalar>> &varFunctions);
  TFunctionPtr<Scalar> evaluate(const Teuchos::map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart);

  TLinearTermPtr<Scalar> getBoundaryOnlyPart();
  TLinearTermPtr<Scalar> getNonBoundaryOnlyPart();
  TLinearTermPtr<Scalar> getPart(bool boundaryOnlyPart);
  TLinearTermPtr<Scalar> getPartMatchingVariable( VarPtr var );
  
  bool isPureBoundaryTerm() const;
  bool isPureVolumeTerm() const;

  // integrate into values Camellia::FieldContainers:
  void integrate(Camellia::FieldContainer<Scalar> &values, DofOrderingPtr thisOrdering,
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
  void integrate(Camellia::FieldContainer<Scalar> &values, DofOrderingPtr thisDofOrdering,
                 TLinearTermPtr<Scalar> otherTerm, DofOrderingPtr otherDofOrdering,
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
  void integrate(Camellia::FieldContainer<Scalar> &values, DofOrderingPtr thisDofOrdering,
                 TLinearTermPtr<Scalar> otherTerm, VarPtr otherVarID, TFunctionPtr<Scalar> fxn,
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false);

  // compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
  // values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
  void values(Camellia::FieldContainer<Scalar> &values, int varID, BasisPtr basis, BasisCachePtr basisCache,
              bool applyCubatureWeights = false, bool naturalBoundaryTermsOnly = false);

  // compute the value of linearTerm for varID = fxn
  // values shape: (C,P), (C,P,D), or (C,P,D,D)
  void values(Camellia::FieldContainer<Scalar> &values, int varID, TFunctionPtr<Scalar> fxn, BasisCachePtr basisCache,
              bool applyCubatureWeights, bool naturalBoundaryTermsOnly = false);

  int rank() const;  // 0 for scalar, 1 for vector, etc.

  bool isZero() const; // true if the TLinearTerm is identically zero

  string displayString() const; // TeX by convention

  void addTerm(const TLinearTerm<Scalar> &a, bool overrideTypeCheck=false);
  void addTerm(TLinearTermPtr<Scalar> aPtr, bool overrideTypeCheck=false);
  
  //! when several summands have like var and op, combine them into one summand with the sum of their function weights
  void collectTerms();
  
  // operator overloading niceties:

  TLinearTerm<Scalar>& operator=(const TLinearTerm<Scalar> &rhs);
  TLinearTerm<Scalar>& operator+=(const TLinearTerm<Scalar> &rhs);

  TLinearTerm<Scalar>& operator+=(VarPtr v);
  
  ~TLinearTerm();
};

// operator overloading for syntax sugar:
TLinearTermPtr<double> operator+(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2);
TLinearTermPtr<double> operator+(VarPtr v, TLinearTermPtr<double> a);
TLinearTermPtr<double> operator+(TLinearTermPtr<double> a, VarPtr v);
TLinearTermPtr<double> operator+(VarPtr v1, VarPtr v2);
TLinearTermPtr<double> operator*(TFunctionPtr<double> f, VarPtr v);
TLinearTermPtr<double> operator*(VarPtr v, TFunctionPtr<double> f);
TLinearTermPtr<double> operator*(double weight, VarPtr v);
TLinearTermPtr<double> operator*(VarPtr v, double weight);
TLinearTermPtr<double> operator*(vector<double> weight, VarPtr v);
TLinearTermPtr<double> operator*(VarPtr v, vector<double> weight);
TLinearTermPtr<double> operator*(TFunctionPtr<double> f, TLinearTermPtr<double> a);
TLinearTermPtr<double> operator*(TLinearTermPtr<double> a, TFunctionPtr<double> f);
TLinearTermPtr<double> operator*(TLinearTermPtr<double> a, double weight);
TLinearTermPtr<double> operator*(double weight, TLinearTermPtr<double> a);
TLinearTermPtr<double> operator/(VarPtr v, double weight);
TLinearTermPtr<double> operator/(VarPtr v, TFunctionPtr<double> f);
TLinearTermPtr<double> operator/(TLinearTermPtr<double> a, double weight);
TLinearTermPtr<double> operator/(TLinearTermPtr<double> a, TFunctionPtr<double> f);
TLinearTermPtr<double> operator-(TLinearTermPtr<double> a);
TLinearTermPtr<double> operator-(TLinearTermPtr<double> a, VarPtr v);
TLinearTermPtr<double> operator-(VarPtr v, TLinearTermPtr<double> a);
TLinearTermPtr<double> operator-(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2);
TLinearTermPtr<double> operator-(VarPtr v1, VarPtr v2);
TLinearTermPtr<double> operator-(VarPtr v);



// template <typename Scalar>
// TLinearTermPtr<Scalar> operator+(TLinearTermPtr<Scalar> a1, TLinearTermPtr<Scalar> a2);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator+(VarPtr v, TLinearTermPtr<Scalar> a);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator+(TLinearTermPtr<Scalar> a, VarPtr v);

// TLinearTermPtr<double> operator+(VarPtr v1, VarPtr v2);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(TFunctionPtr<Scalar> f, VarPtr v);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(VarPtr v, TFunctionPtr<Scalar> f);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(Scalar weight, VarPtr v);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(VarPtr v, Scalar weight);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(vector<Scalar> weight, VarPtr v);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(VarPtr v, vector<Scalar> weight);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(TFunctionPtr<Scalar> f, TLinearTermPtr<Scalar> a);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator*(TLinearTermPtr<Scalar> a, TFunctionPtr<Scalar> f);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator/(VarPtr v, Scalar weight);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator/(VarPtr v, TFunctionPtr<Scalar> f);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a, VarPtr v);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator-(VarPtr v, TLinearTermPtr<Scalar> a);

// template <typename Scalar>
// TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a1, TLinearTermPtr<Scalar> a2);

// TLinearTermPtr<double> operator-(VarPtr v1, VarPtr v2);

// TLinearTermPtr<double> operator-(VarPtr v);

extern template class TLinearTerm<double>;
}


#endif
