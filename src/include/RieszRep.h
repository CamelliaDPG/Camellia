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
  class RieszRep {
   private:

    map<GlobalIndexType, Intrepid::FieldContainer<double> > _rieszRepDofs; // from cellID to dofs of riesz representation
    map<GlobalIndexType, Intrepid::FieldContainer<double> > _rieszRepDofsGlobal; // from cellID to dofs of riesz representation
    map<GlobalIndexType, double > _rieszRepNormSquared; // from cellID to norm squared of riesz inversion
    map<GlobalIndexType, double > _rieszRepNormSquaredGlobal; // from cellID to norm squared of riesz inversion

    MeshPtr _mesh;
    IPPtr _ip;
    LinearTermPtr _functional;  // the RHS stuff here and below is misnamed -- should just be called functional
    bool _printAll;
    bool _repsNotComputed;

   public:
    RieszRep(MeshPtr mesh, IPPtr ip, LinearTermPtr functional){
      _mesh = mesh;
      _ip = ip;
      _functional = functional;
      _printAll = false;
      _repsNotComputed = true;
    }

    void setPrintOption(bool printAll){
      _printAll = printAll;
    }

    void setFunctional(LinearTermPtr functional){
      _functional = functional;
    }

    LinearTermPtr getFunctional();

    MeshPtr mesh();

    // for testing
    map<GlobalIndexType,Intrepid::FieldContainer<double> > integrateFunctional();

    void computeRieszRep(int cubatureEnrichment=0);

    double getNorm();

    // ! Returns reference to container for rank-local cells
    const map<GlobalIndexType,double> &getNormsSquared();

    // ! Returns reference to redundantly stored container for *all* active cells
    const map<GlobalIndexType,double> &getNormsSquaredGlobal();

    void distributeDofs();

    void computeRepresentationValues(Intrepid::FieldContainer<double> &values, int testID, EOperator op, BasisCachePtr basisCache);

    double computeAlternativeNormSqOnCell(IPPtr ip, ElementPtr elem);
    map<GlobalIndexType,double> computeAlternativeNormSqOnCells(IPPtr ip, vector<GlobalIndexType> cellIDs);

    static FunctionPtr repFunction( VarPtr var, RieszRepPtr rep );
    static RieszRepPtr rieszRep(MeshPtr mesh, IPPtr ip, LinearTermPtr functional);
  };

  class RepFunction;
  typedef Teuchos::RCP<RepFunction> RepFunctionPtr;

  class RepFunction : public Function<double> {
  private:

    int _testID;
    Teuchos::RCP<RieszRep> _rep;
    EOperator _op;
  public:
    RepFunction( VarPtr var, RieszRepPtr rep ) : Function<double>( var->rank() ) {
      _testID = var->ID();
      _op = var->op();
      _rep = rep;
    }

    // optional specification of operator to apply - default to rank 0
   RepFunction(int testID,Teuchos::RCP<RieszRep> rep, EOperator op): Function<double>(0){
      _testID = testID;
      _rep = rep;
      _op = op;
    }

    // specification of function rank
   RepFunction(int testID,Teuchos::RCP<RieszRep> rep, EOperator op, int fxnRank): Function<double>(fxnRank){
      _testID = testID;
      _rep = rep;
      _op = op;
    }


    FunctionPtr x(){
      return Teuchos::rcp(new RepFunction(_testID,_rep,OP_X));
    }
    FunctionPtr y(){
      return Teuchos::rcp(new RepFunction(_testID,_rep,OP_Y));
    }
    FunctionPtr dx(){
      return Teuchos::rcp(new RepFunction(_testID,_rep,OP_DX));
    }
    FunctionPtr dy(){
      return Teuchos::rcp(new RepFunction(_testID,_rep,OP_DY));
    }
    //  FunctionPtr grad(){
    //    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_GRAD,2)); // default to 2 space dimensions
    //  }
    FunctionPtr div(){
      return Teuchos::rcp(new RepFunction(_testID,_rep,OP_DIV));
    }

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
      _rep->computeRepresentationValues(values, _testID, _op, basisCache);
    }

    // for specifying an operator
    void values(Intrepid::FieldContainer<double> &values, EOperator op, BasisCachePtr basisCache){
      _rep->computeRepresentationValues(values, _testID, op, basisCache);
    }
  };
}



#endif
