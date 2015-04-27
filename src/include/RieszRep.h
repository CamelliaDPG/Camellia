#ifndef RIESZ_REP
#define RIESZ_REP

/*
 *  RieszRep.h
 *
 *  Created by Jesse Chan on 10/22/12
 *
 */

#include "TypeDefs.h"

// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Intrepid_FieldContainer.hpp"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "Mesh.h"
#include "ElementType.h"
#include "Element.h"
#include "Function.h"

#include "LinearTerm.h"
#include "BasisCache.h"
#include "IP.h"

namespace Camellia {
  template <typename Scalar>
  class TRieszRep {
   private:

    map<GlobalIndexType, Intrepid::FieldContainer<Scalar> > _rieszRepDofs; // from cellID to dofs of riesz representation
    map<GlobalIndexType, Intrepid::FieldContainer<Scalar> > _rieszRepDofsGlobal; // from cellID to dofs of riesz representation
    map<GlobalIndexType, double > _rieszRepNormSquared; // from cellID to norm squared of riesz inversion
    map<GlobalIndexType, double > _rieszRepNormSquaredGlobal; // from cellID to norm squared of riesz inversion

    TMeshPtr<Scalar> _mesh;
    TIPPtr<Scalar> _ip;
    TLinearTermPtr<Scalar> _functional;  // the RHS stuff here and below is misnamed -- should just be called functional
    bool _printAll;
    bool _repsNotComputed;

   public:
    TRieszRep(TMeshPtr<Scalar> mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> functional){
      _mesh = mesh;
      _ip = ip;
      _functional = functional;
      _printAll = false;
      _repsNotComputed = true;
    }

    void setPrintOption(bool printAll){
      _printAll = printAll;
    }

    void setFunctional(TLinearTermPtr<Scalar> functional){
      _functional = functional;
    }

    TLinearTermPtr<Scalar> getFunctional();

    TMeshPtr<Scalar> mesh();

    // for testing
    map<GlobalIndexType,Intrepid::FieldContainer<Scalar> > integrateFunctional();

    void computeRieszRep(int cubatureEnrichment=0);

    double getNorm();

    // ! Returns reference to container for rank-local cells
    const map<GlobalIndexType,double> &getNormsSquared();

    // ! Returns reference to redundantly stored container for *all* active cells
    const map<GlobalIndexType,double> &getNormsSquaredGlobal();

    void distributeDofs();

    void computeRepresentationValues(Intrepid::FieldContainer<Scalar> &values, int testID, EOperator op, BasisCachePtr basisCache);

    double computeAlternativeNormSqOnCell(TIPPtr<Scalar> ip, ElementPtr elem);
    map<GlobalIndexType,double> computeAlternativeNormSqOnCells(TIPPtr<Scalar> ip, vector<GlobalIndexType> cellIDs);

    static TFunctionPtr<Scalar> repFunction( VarPtr var, TRieszRepPtr<Scalar> rep );
    static TRieszRepPtr<Scalar> rieszRep(TMeshPtr<Scalar> mesh, TIPPtr<Scalar> ip, TLinearTermPtr<Scalar> functional);
  };

  extern template class TRieszRep<double>;

  template <typename Scalar>
  class RepFunction : public TFunction<Scalar> {
  private:

    int _testID;
    TRieszRepPtr<Scalar> _rep;
    EOperator _op;
  public:
    RepFunction( VarPtr var, TRieszRepPtr<Scalar> rep ) : TFunction<Scalar>( var->rank() ) {
      _testID = var->ID();
      _op = var->op();
      _rep = rep;
    }

    // optional specification of operator to apply - default to rank 0
   RepFunction(int testID, TRieszRepPtr<Scalar> rep, EOperator op): TFunction<Scalar>(0){
      _testID = testID;
      _rep = rep;
      _op = op;
    }

    // specification of function rank
   RepFunction(int testID, TRieszRepPtr<Scalar> rep, EOperator op, int fxnRank): TFunction<Scalar>(fxnRank){
      _testID = testID;
      _rep = rep;
      _op = op;
    }


    TFunctionPtr<Scalar> x(){
      return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_X));
    }
    TFunctionPtr<Scalar> y(){
      return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_Y));
    }
    TFunctionPtr<Scalar> dx(){
      return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DX));
    }
    TFunctionPtr<Scalar> dy(){
      return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DY));
    }
    //  TFunctionPtr<Scalar> grad(){
    //    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_GRAD,2)); // default to 2 space dimensions
    //  }
    TFunctionPtr<Scalar> div(){
      return Teuchos::rcp(new RepFunction<Scalar>(_testID,_rep,OP_DIV));
    }

    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
      _rep->computeRepresentationValues(values, _testID, _op, basisCache);
    }

    // for specifying an operator
    void values(Intrepid::FieldContainer<Scalar> &values, EOperator op, BasisCachePtr basisCache){
      _rep->computeRepresentationValues(values, _testID, op, basisCache);
    }
  };

  extern template class RepFunction<double>;
}



#endif
