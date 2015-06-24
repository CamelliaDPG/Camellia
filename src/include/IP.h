//
//  TIP.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_IP_h
#define Camellia_IP_h

#include "TypeDefs.h"

#include "LinearTerm.h"
#include "Var.h"

#include "Function.h"
#include "DofOrdering.h"

namespace Camellia
{
template <typename Scalar>
class TIP
{
  std::vector< TLinearTermPtr<Scalar> > _linearTerms;
  std::vector< TLinearTermPtr<Scalar> > _boundaryTerms;
  std::vector< TLinearTermPtr<Scalar> > _zeroMeanTerms;

  bool _isLegacySubclass;
protected:
  TBFPtr<Scalar> _bilinearForm; // for legacy subclasses (originally subclasses of DPGInnerProduct)
public:
  TIP();

  // legacy subclass constructor:
  TIP(TBFPtr<Scalar> bfs);

  // legacy DPGInnerProduct::applyInnerProductData() methods:
  virtual void applyInnerProductData(Intrepid::FieldContainer<Scalar> &testValues1,
                                     Intrepid::FieldContainer<Scalar> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     const Intrepid::FieldContainer<double>& physicalPoints);
  virtual void applyInnerProductData(Intrepid::FieldContainer<Scalar> &testValues1,
                                     Intrepid::FieldContainer<Scalar> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     BasisCachePtr basisCache);

  // legacy DPGInnerProduct::computeInnerProductMatrix() method
  virtual void computeInnerProductMatrix(Intrepid::FieldContainer<Scalar> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         shards::CellTopology &cellTopo,
                                         Intrepid::FieldContainer<double>& physicalCellNodes);

  virtual ~TIP() {}

  // if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ...
  void addTerm( TLinearTermPtr<Scalar> a);
  void addTerm( VarPtr v );
  void addZeroMeanTerm( TLinearTermPtr<Scalar> a);
  void addZeroMeanTerm( VarPtr v);

  void addBoundaryTerm( TLinearTermPtr<Scalar> a );
  void addBoundaryTerm( VarPtr v );

  virtual void computeInnerProductMatrix(Intrepid::FieldContainer<Scalar> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache);

  virtual void computeInnerProductVector(Intrepid::FieldContainer<Scalar> &ipVector,
                                         VarPtr var, TFunctionPtr<Scalar> fxn,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache);

  double computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache);

  // added by Nate
  TLinearTermPtr<Scalar> evaluate(const std::map< int, TFunctionPtr<Scalar>> &varFunctions);
  // added by Jesse
  TLinearTermPtr<Scalar> evaluate(const std::map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart);

  virtual bool hasBoundaryTerms();

  int nonZeroEntryCount(DofOrderingPtr testOrdering);
  
  virtual void operators(int testID1, int testID2,
                         std::vector<Camellia::EOperator> &testOp1,
                         std::vector<Camellia::EOperator> &testOp2);

  virtual void printInteractions();

  std::string displayString();

  static TIPPtr<Scalar> ip();

  static pair<TIPPtr<Scalar>, VarPtr > standardInnerProductForFunctionSpace(Camellia::EFunctionSpace fs, bool useTraceVar, int spaceDim);
};

extern template class TIP<double>;
}

#endif
