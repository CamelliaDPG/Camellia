//
//  IP.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_IP_h
#define Camellia_IP_h

#include "LinearTerm.h"
#include "Var.h"

#include "Function.h"
#include "DofOrdering.h"

class BilinearForm;

class IP {
  std::vector< LinearTermPtr > _linearTerms;
  std::vector< LinearTermPtr > _boundaryTerms;
  std::vector< LinearTermPtr > _zeroMeanTerms;
  
  bool _isLegacySubclass;
protected:
  Teuchos::RCP< BilinearForm > _bilinearForm; // for legacy subclasses (originally subclasses of DPGInnerProduct)
public:
  IP();
  
  // legacy subclass constructor:
  IP(Teuchos::RCP< BilinearForm > bfs);
  
  // legacy DPGInnerProduct::applyInnerProductData() methods:
  virtual void applyInnerProductData(Intrepid::FieldContainer<double> &testValues1,
                                     Intrepid::FieldContainer<double> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     const FieldContainer<double>& physicalPoints);
  virtual void applyInnerProductData(Intrepid::FieldContainer<double> &testValues1,
                                     Intrepid::FieldContainer<double> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     BasisCachePtr basisCache);
  
  // legacy DPGInnerProduct::computeInnerProductMatrix() method
  virtual void computeInnerProductMatrix(Intrepid::FieldContainer<double> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         shards::CellTopology &cellTopo,
                                         Intrepid::FieldContainer<double>& physicalCellNodes);
  
  virtual ~IP() {}
  
  // if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ...  
  void addTerm( LinearTermPtr a);
  void addTerm( VarPtr v );
  void addZeroMeanTerm( LinearTermPtr a);
  void addZeroMeanTerm( VarPtr v);
  
  void addBoundaryTerm( LinearTermPtr a );
  void addBoundaryTerm( VarPtr v );
  
  virtual void computeInnerProductMatrix(Intrepid::FieldContainer<double> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache);
  
  virtual void computeInnerProductVector(Intrepid::FieldContainer<double> &ipVector,
                                         VarPtr var, FunctionPtr fxn,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache);

  double computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache);
  
  // added by Nate
  LinearTermPtr evaluate(std::map< int, FunctionPtr> &varFunctions);
  // added by Jesse
  LinearTermPtr evaluate(std::map< int, FunctionPtr> &varFunctions, bool boundaryPart);
  //  FunctionPtr evaluate(map< int, FunctionPtr> &varFunctions1, map< int, FunctionPtr> &varFunctions2, bool boundaryPart);

  virtual bool hasBoundaryTerms();
  
  virtual void operators(int testID1, int testID2,
                         std::vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                         std::vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2);
  
  virtual void printInteractions();
  
  std::string displayString();

  static Teuchos::RCP<IP> ip();
  
  static pair<Teuchos::RCP<IP>, VarPtr > standardInnerProductForFunctionSpace(IntrepidExtendedTypes::EFunctionSpaceExtended fs, bool useTraceVar, int spaceDim);
};

typedef Teuchos::RCP<IP> IPPtr;

#endif
