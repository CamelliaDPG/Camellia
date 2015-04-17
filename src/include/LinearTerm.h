//
//  LinearTerm.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_LinearTerm_h
#define Camellia_LinearTerm_h

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

class Epetra_CrsMatrix;
namespace Camellia {

  class LinearTerm {
    typedef std::pair< FunctionPtr<double>, VarPtr > LinearSummand;
    int _rank; // gets set after first var is added
    std::vector< LinearSummand > _summands;
    std::set<int> _varIDs;
    Camellia::VarType _termType; // shouldn't mix

    // for the Riesz inversion evaluation
    std::map< ElementType*, Intrepid::FieldContainer<double> > _rieszRepresentationForElementType;
    std::map< ElementType*, Intrepid::FieldContainer<double> > _rieszRHSForElementType;
    std::map< int, double > _energyNormForCellIDGlobal;

    // some private utility methods:
    static void integrate(Epetra_CrsMatrix *valuesCrsMatrix, Intrepid::FieldContainer<double> &valuesFC,
                          LinearTermPtr u, DofOrderingPtr uOrdering,
                          LinearTermPtr v, DofOrderingPtr vOrdering,
                          BasisCachePtr basisCache, bool sumInto=true);
    static void integrate(Intrepid::FieldContainer<double> &values,
                          LinearTermPtr u, DofOrderingPtr uOrdering,
                          LinearTermPtr v, DofOrderingPtr vOrdering,
                          BasisCachePtr basisCache, bool sumInto=true);
    static void multiplyFluxValuesByParity(Intrepid::FieldContainer<double> &fluxValues, BasisCachePtr sideBasisCache);

    // poor man's templating: just provide both versions of the values argument, making the other version null or size 0
    void integrate(Epetra_CrsMatrix *valuesCrsMatrix, Intrepid::FieldContainer<double> &valuesFC, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, DofOrderingPtr otherDofOrdering,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
    void integrate(Epetra_CrsMatrix *valuesCrsMatrix, Intrepid::FieldContainer<double> &valuesFC, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, VarPtr otherVarID, FunctionPtr<double> fxn,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false);

  public: // was protected; changed for debugging (no big deal either way, I think)
    const std::vector< LinearSummand > & summands() const;
  public:
    LinearTerm();
    LinearTerm(FunctionPtr<double> weight, VarPtr var);
    LinearTerm(double weight, VarPtr var);
    LinearTerm(std::vector<double> weight, VarPtr var);
    LinearTerm( VarPtr v );
    // copy constructor:
    LinearTerm( const LinearTerm &a );
    void addVar(FunctionPtr<double> weight, VarPtr var);
    void addVar(double weight, VarPtr var);
    void addVar(std::vector<double> vector_weight, VarPtr var);

    const std::set<int> & varIDs() const;

    Camellia::VarType termType() const;
    //  vector< Camellia::EOperator > varOps(int varID);

    /** \brief  Computes the norm of the LinearTerm using the Riesz representation corresponding to the inner product ip on the specified mesh.
     *  \param  ip        [in]  - inner product to use for the Riesz representation
     *  \param  mesh      [in]  - mesh over which to measure
     */
    double computeNorm(IPPtr ip, MeshPtr mesh);

    void evaluate(Intrepid::FieldContainer<double> &values, SolutionPtr<double> solution, BasisCachePtr basisCache,
                  bool applyCubatureWeights = false);

    FunctionPtr<double> evaluate(const Teuchos::map< int, FunctionPtr<double>> &varFunctions);
    FunctionPtr<double> evaluate(const Teuchos::map< int, FunctionPtr<double>> &varFunctions, bool boundaryPart);

    LinearTermPtr getBoundaryOnlyPart();
    LinearTermPtr getNonBoundaryOnlyPart();
    LinearTermPtr getPart(bool boundaryOnlyPart);
    LinearTermPtr getPartMatchingVariable( VarPtr var );

    // integrate into values FieldContainers:
    void integrate(Intrepid::FieldContainer<double> &values, DofOrderingPtr thisOrdering,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
    void integrate(Intrepid::FieldContainer<double> &values, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, DofOrderingPtr otherDofOrdering,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
    void integrate(Intrepid::FieldContainer<double> &values, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, VarPtr otherVarID, FunctionPtr<double> fxn,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false);

    // CrsMatrix versions (for the two-LT (matrix) variants of integrate)
    void integrate(Epetra_CrsMatrix *values, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, DofOrderingPtr otherDofOrdering,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false, bool sumInto = true);
    void integrate(Epetra_CrsMatrix *values, DofOrderingPtr thisDofOrdering,
                   LinearTermPtr otherTerm, VarPtr otherVarID, FunctionPtr<double> fxn,
                   BasisCachePtr basisCache, bool forceBoundaryTerm = false);

    // compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
    // values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
    void values(Intrepid::FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache,
                bool applyCubatureWeights = false, bool naturalBoundaryTermsOnly = false);

    // compute the value of linearTerm for varID = fxn
    // values shape: (C,P), (C,P,D), or (C,P,D,D)
    void values(Intrepid::FieldContainer<double> &values, int varID, FunctionPtr<double> fxn, BasisCachePtr basisCache,
                bool applyCubatureWeights, bool naturalBoundaryTermsOnly = false);

    int rank() const;  // 0 for scalar, 1 for vector, etc.

    bool isZero() const; // true if the LinearTerm is identically zero

    string displayString() const; // TeX by convention

    /*
     // -------------- added by Jesse --------------------

     void computeRieszRep(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<IP> ip);
     void computeRieszRHS(Teuchos::RCP<Mesh> mesh);
     LinearTermPtr rieszRep(VarPtr v);
     double functionalNorm();
     const map<int,double> & energyNorm(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<IP> ip);
     double energyNormTotal(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<IP> ip); // global energy norm

     // -------------- end of added by Jesse --------------------
     */

    void addTerm(const LinearTerm &a, bool overrideTypeCheck=false);
    void addTerm(LinearTermPtr aPtr, bool overrideTypeCheck=false);
    // operator overloading niceties:

    LinearTerm& operator=(const LinearTerm &rhs);
    LinearTerm& operator+=(const LinearTerm &rhs);

    LinearTerm& operator+=(VarPtr v);

    ~LinearTerm();
  };

  // operator overloading for syntax sugar:
  LinearTermPtr operator+(LinearTermPtr a1, LinearTermPtr a2);

  LinearTermPtr operator+(VarPtr v, LinearTermPtr a);

  LinearTermPtr operator+(LinearTermPtr a, VarPtr v);

  LinearTermPtr operator*(FunctionPtr<double> f, VarPtr v);
  LinearTermPtr operator*(VarPtr v, FunctionPtr<double> f);

  LinearTermPtr operator*(double weight, VarPtr v);

  LinearTermPtr operator*(VarPtr v, double weight);

  LinearTermPtr operator*(vector<double> weight, VarPtr v);

  LinearTermPtr operator*(VarPtr v, vector<double> weight);

  LinearTermPtr operator*(FunctionPtr<double> f, LinearTermPtr a);
  LinearTermPtr operator*(LinearTermPtr a, FunctionPtr<double> f);

  LinearTermPtr operator+(VarPtr v1, VarPtr v2);

  LinearTermPtr operator/(VarPtr v, double weight);

  LinearTermPtr operator/(VarPtr v, FunctionPtr<double> f);

  LinearTermPtr operator-(VarPtr v1, VarPtr v2);

  LinearTermPtr operator-(VarPtr v);

  LinearTermPtr operator-(LinearTermPtr a);

  LinearTermPtr operator-(LinearTermPtr a, VarPtr v);

  LinearTermPtr operator-(VarPtr v, LinearTermPtr a);

  LinearTermPtr operator-(LinearTermPtr a1, LinearTermPtr a2);
}


#endif
