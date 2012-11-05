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

#include <Teuchos_GlobalMPISession.hpp>
// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
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

#include "IP.h" // includes LinearTerm.h and DPGInnerProduct.h
//#include "LinearTerm.h"
#include "BasisCache.h"


typedef Teuchos::RCP<Mesh> MeshPtr;
typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<LinearTerm> LtPtr;

typedef Teuchos::RCP< DofOrdering > DofOrderingPtr;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr; 
typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;

class RieszRep {
 private:

  map<int, FieldContainer<double> > _rieszRepDofs; // from cellID to dofs of riesz representation
  map<int, FieldContainer<double> > _rieszRepDofsGlobal; // from cellID to dofs of riesz representation
  map<int, double > _rieszRepNormSquared; // from cellID to norm squared of riesz inversion
  
  MeshPtr _mesh;
  IPPtr _ip;
  LtPtr _rhs;
  bool _printAll;
 
 public:
  RieszRep(MeshPtr mesh, IPPtr ip, LtPtr &rhs){
    _mesh = mesh;
    _ip = ip;
    _rhs = rhs;
    _printAll = false;
  }

  void setPrintOption(bool printAll){
    _printAll = printAll;
  }

  void setFunctional(LtPtr rhs){
    _rhs = rhs;
  }

  LtPtr getRHS();

  // for testing
  map<int,FieldContainer<double> > integrateRHS();

  void computeRieszRep();

  double getNorm();

  void distributeDofs();

  void computeRepresentationValues(FieldContainer<double> &values, int testID, IntrepidExtendedTypes::EOperatorExtended op, BasisCachePtr basisCache);

};

class RepFunction : public Function {
private:
  int _testID;
  Teuchos::RCP<RieszRep> _rep;
  IntrepidExtendedTypes::EOperatorExtended _op;
public:
 RepFunction(int testID,Teuchos::RCP<RieszRep> rep): Function(0){
    _testID = testID;
    _rep = rep;   
    _op =  IntrepidExtendedTypes::OP_VALUE; // default to OPERATOR_VALUE
  }

  // optional specification of operator to apply 
 RepFunction(int testID,Teuchos::RCP<RieszRep> rep, IntrepidExtendedTypes::EOperatorExtended op): Function(0){
    _testID = testID;
    _rep = rep;   
    _op = op;
  }   

  FunctionPtr dx(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,IntrepidExtendedTypes::OP_DX));
  }
  FunctionPtr dy(){
    return Teuchos::rcp(new RepFunction(_testID,_rep,IntrepidExtendedTypes::OP_DY));
  }

  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    _rep->computeRepresentationValues(values, _testID, _op, basisCache);        
  }

  // for specifying an operator
  void values(FieldContainer<double> &values, IntrepidExtendedTypes::EOperatorExtended op, BasisCachePtr basisCache){
    _rep->computeRepresentationValues(values, _testID, op, basisCache);
  }

};


#endif
