#ifndef RIESZ_REP
#define RIESZ_REP

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  RieszRep.h
 *
 *  Created by Jesse Chan on 10/22/12
 *
 */

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

class RieszRep;
typedef Teuchos::RCP<RieszRep> RieszRepPtr;

class RieszRep {
 private:

  map<GlobalIndexType, FieldContainer<double> > _rieszRepDofs; // from cellID to dofs of riesz representation
  map<GlobalIndexType, FieldContainer<double> > _rieszRepDofsGlobal; // from cellID to dofs of riesz representation
  map<GlobalIndexType, double > _rieszRepNormSquared; // from cellID to norm squared of riesz inversion
  map<GlobalIndexType, double > _rieszRepNormSquaredGlobal; // from cellID to norm squared of riesz inversion
  
  MeshPtr _mesh;
  IPPtr _ip;
  LinearTermPtr _rhs;  // the RHS stuff here and below is misnamed -- should just be called functional
  bool _printAll;
  bool _repsNotComputed;
 
 public:
  RieszRep(MeshPtr mesh, IPPtr ip, LinearTermPtr rhs){
    _mesh = mesh;
    _ip = ip;
    _rhs = rhs;
    _printAll = false;
    _repsNotComputed = true;
  }

  void setPrintOption(bool printAll){
    _printAll = printAll;
  }

  void setFunctional(LinearTermPtr rhs){
    _rhs = rhs;
  }

  LinearTermPtr getRHS(); // getFunctional()
  
  MeshPtr mesh();

  // for testing
  map<GlobalIndexType,FieldContainer<double> > integrateRHS(); // integrateFunctional()

  void computeRieszRep(int cubatureEnrichment=0);

  double getNorm();
  const map<GlobalIndexType,double> &getNormsSquaredGlobal();

  void distributeDofs();

  void computeRepresentationValues(FieldContainer<double> &values, int testID, Camellia::EOperator op, BasisCachePtr basisCache);

  double computeAlternativeNormSqOnCell(IPPtr ip, ElementPtr elem);
  map<GlobalIndexType,double> computeAlternativeNormSqOnCells(IPPtr ip, vector<GlobalIndexType> cellIDs);
  
  static FunctionPtr repFunction( VarPtr var, RieszRepPtr rep );
  static RieszRepPtr rieszRep(MeshPtr mesh, IPPtr ip, LinearTermPtr rhs);
};

class RepFunction;
typedef Teuchos::RCP<RepFunction> RepFunctionPtr;

class RepFunction : public Function {
private:
  
  int _testID;
  Teuchos::RCP<RieszRep> _rep;
  Camellia::EOperator _op;
public:
  RepFunction( VarPtr var, RieszRepPtr rep ) : Function( var->rank() ) {
    _testID = var->ID();
    _op = var->op();
    _rep = rep;
  }
    
  /*
// WARNING: DOES NOT WORK FOR HIGHER RANK FUNCTIONS
  RepFunction(int testID,Teuchos::RCP<RieszRep> rep): Function(0){
    _testID = testID;
    _rep = rep;   
    _op =  Camellia::OP_VALUE; // default to OPERATOR_VALUE
  }
  */

  // optional specification of operator to apply - default to rank 0
 RepFunction(int testID,Teuchos::RCP<RieszRep> rep, Camellia::EOperator op): Function(0){
    _testID = testID;
    _rep = rep;   
    _op = op;
  }   
 
  // specification of function rank
 RepFunction(int testID,Teuchos::RCP<RieszRep> rep, Camellia::EOperator op, int fxnRank): Function(fxnRank){
    _testID = testID;
    _rep = rep;   
    _op = op;
  }   
  

  FunctionPtr x(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_X));
  }
  FunctionPtr y(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_Y));
  }
  FunctionPtr dx(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_DX));
  }
  FunctionPtr dy(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_DY));
  }  
  //  FunctionPtr grad(){
  //    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_GRAD,2)); // default to 2 space dimensions
  //  }
  FunctionPtr div(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,Camellia::OP_DIV));
  }

  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    _rep->computeRepresentationValues(values, _testID, _op, basisCache);        
  }

  // for specifying an operator
  void values(FieldContainer<double> &values, Camellia::EOperator op, BasisCachePtr basisCache){
    _rep->computeRepresentationValues(values, _testID, op, basisCache);
  }
};


#endif
