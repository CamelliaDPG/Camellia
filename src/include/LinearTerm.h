//
//  LinearTerm.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_LinearTerm_h
#define Camellia_LinearTerm_h

#include "DPGInnerProduct.h"

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
#include "Function.h"
#include "Var.h"
#include "BasisCache.h"

class LinearTerm;
class Mesh;
class Solution;
typedef Teuchos::RCP<Solution> SolutionPtr;
typedef Teuchos::RCP<LinearTerm> LinearTermPtr;

typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

class LinearTerm {
  typedef pair< FunctionPtr, VarPtr > LinearSummand;
  int _rank; // gets set after first var is added
  vector< LinearSummand > _summands;
  set<int> _varIDs;
  VarType _termType; // shouldn't mix

  // for the Riesz inversion evaluation
  map< ElementType*, FieldContainer<double> > _rieszRepresentationForElementType;
  map< ElementType*, FieldContainer<double> > _rieszRHSForElementType;
  map< int, double > _energyNormForCellIDGlobal;
public: // was protected; changed for debugging (no big deal either way, I think)
  const vector< LinearSummand > & summands() const;
public:
  LinearTerm();
  LinearTerm(FunctionPtr weight, VarPtr var);
  LinearTerm(double weight, VarPtr var);
  LinearTerm(vector<double> weight, VarPtr var);
  LinearTerm( VarPtr v );
  // copy constructor:
  LinearTerm( const LinearTerm &a );  
  void addVar(FunctionPtr weight, VarPtr var);
  void addVar(double weight, VarPtr var);
  void addVar(vector<double> vector_weight, VarPtr var);
  
  const set<int> & varIDs() const;
  
  VarType termType() const;
  //  vector< IntrepidExtendedTypes::EOperatorExtended > varOps(int varID);
  
  void evaluate(FieldContainer<double> &values, SolutionPtr solution, BasisCachePtr basisCache, 
                bool applyCubatureWeights = false);
  
  // integrate into values:
  void integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering,
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false);
  void integrate(FieldContainer<double> &values, DofOrderingPtr thisDofOrdering, 
                 LinearTermPtr otherTerm, DofOrderingPtr otherDofOrdering, 
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false);
  void integrate(FieldContainer<double> &values, DofOrderingPtr thisDofOrdering, 
                 LinearTermPtr otherTerm, VarPtr otherVarID, FunctionPtr fxn,
                 BasisCachePtr basisCache, bool forceBoundaryTerm = false);
  
  // compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
  // values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
  void values(FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache, 
              bool applyCubatureWeights = false, bool naturalBoundaryTermsOnly = false);

  // compute the value of linearTerm for varID = fxn
  // values shape: (C,P), (C,P,D), or (C,P,D,D)
  void values(FieldContainer<double> &values, int varID, FunctionPtr fxn, BasisCachePtr basisCache, 
              bool applyCubatureWeights, bool naturalBoundaryTermsOnly = false);
  
  int rank() const;  // 0 for scalar, 1 for vector, etc.

  string displayString(); // TeX by convention
  
  // -------------- added by Jesse --------------------

  void computeRieszRep(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip);
  void computeRieszRHS(Teuchos::RCP<Mesh> mesh);
  LinearTermPtr rieszRep(VarPtr v);
  double functionalNorm();
  const map<int,double> & energyNorm(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip);
  double energyNormTotal(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip); // global energy norm

  
  // -------------- end of added by Jesse --------------------

  void addTerm(const LinearTerm &a, bool overrideTypeCheck=false);
  void addTerm(LinearTermPtr aPtr, bool overrideTypeCheck=false);
  // operator overloading niceties:
  
  LinearTerm& operator=(const LinearTerm &rhs);
  LinearTerm& operator+=(const LinearTerm &rhs);
  
  LinearTerm& operator+=(VarPtr v);
};

// operator overloading for syntax sugar:
LinearTermPtr operator+(LinearTermPtr a1, LinearTermPtr a2);

LinearTermPtr operator+(VarPtr v, LinearTermPtr a);

LinearTermPtr operator+(LinearTermPtr a, VarPtr v);

LinearTermPtr operator*(FunctionPtr f, VarPtr v);

LinearTermPtr operator*(double weight, VarPtr v);

LinearTermPtr operator*(VarPtr v, double weight);

LinearTermPtr operator*(vector<double> weight, VarPtr v);

LinearTermPtr operator*(VarPtr v, vector<double> weight);

LinearTermPtr operator*(FunctionPtr f, LinearTermPtr a);

LinearTermPtr operator+(VarPtr v1, VarPtr v2);

LinearTermPtr operator/(VarPtr v, double weight);

LinearTermPtr operator/(VarPtr v, FunctionPtr f);

LinearTermPtr operator-(VarPtr v1, VarPtr v2);

LinearTermPtr operator-(VarPtr v);

LinearTermPtr operator-(LinearTermPtr a);

LinearTermPtr operator-(LinearTermPtr a, VarPtr v);

LinearTermPtr operator-(VarPtr v, LinearTermPtr a);

LinearTermPtr operator-(LinearTermPtr a1, LinearTermPtr a2);

#endif
