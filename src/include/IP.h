// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TIP.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
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

  std::vector< std::pair<VarPtr, SpatialFilterPtr> > _homogeneousBCs;
  
  bool _isLegacySubclass;
protected:
  TBFPtr<Scalar> _bilinearForm; // for legacy subclasses (originally subclasses of DPGInnerProduct)
public:
  TIP();

  // legacy subclass constructor:
  TIP(TBFPtr<Scalar> bfs);

  // legacy DPGInnerProduct::applyInnerProductData() methods:
  virtual void applyInnerProductData(Camellia::FieldContainer<Scalar> &testValues1,
                                     Camellia::FieldContainer<Scalar> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     const Camellia::FieldContainer<double>& physicalPoints);
  virtual void applyInnerProductData(Camellia::FieldContainer<Scalar> &testValues1,
                                     Camellia::FieldContainer<Scalar> &testValues2,
                                     int testID1, int testID2, int operatorIndex,
                                     BasisCachePtr basisCache);

  // legacy DPGInnerProduct::computeInnerProductMatrix() method
  virtual void computeInnerProductMatrix(Camellia::FieldContainer<Scalar> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         shards::CellTopology &cellTopo,
                                         Camellia::FieldContainer<double>& physicalCellNodes);

  virtual ~TIP() {}

  // if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ...
  void addTerm( TLinearTermPtr<Scalar> a);
  void addTerm( VarPtr v );
  void addZeroMeanTerm( TLinearTermPtr<Scalar> a);
  void addZeroMeanTerm( VarPtr v);

  void addBoundaryTerm( TLinearTermPtr<Scalar> a );
  void addBoundaryTerm( VarPtr v );

  virtual void computeInnerProductMatrix(Camellia::FieldContainer<Scalar> &innerProduct,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache);

  virtual void computeInnerProductVector(Camellia::FieldContainer<Scalar> &ipVector,
                                         VarPtr var, TFunctionPtr<Scalar> fxn,
                                         Teuchos::RCP<DofOrdering> dofOrdering,
                                         Teuchos::RCP<BasisCache> basisCache,
                                         bool sumInto = false);

  double computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache);

  // added by Nate
  TLinearTermPtr<Scalar> evaluate(const std::map< int, TFunctionPtr<Scalar>> &varFunctions);
  // added by Jesse
  TLinearTermPtr<Scalar> evaluate(const std::map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart);

  std::vector< TLinearTermPtr<Scalar> > getLinearTerms() const;
  std::vector< TLinearTermPtr<Scalar> > getBoundaryTerms() const;
  std::vector< TLinearTermPtr<Scalar> > getZeroMeanTerms() const;
  
  virtual bool hasBoundaryTerms();

  // ! returns true if BCs have been set using imposeHomogeneousBCs()
  bool hasBCsImposed();
  
  // ! specify that homogeneous BCs should be imposed on var on the portion of the mesh boundary that matches the provided filter.
  // ! Note that this only changes the inner product (Gram) matrix by zeroing out appropriate rows and columns, putting 1s in the diagonal.
  // ! Since IP does not compute the RHS of any solves, clients are responsible for calling homogeneousBCDofOrdinalsForCell() and setting
  // ! corresponding RHS entries to 0.
  void imposeHomogeneousBCs(VarPtr var, SpatialFilterPtr meshBoundaryFilter);
  
  // returns the cell-local dof ordinals on which homogeneous BCs are being imposed (necessary for correct handling of RHS in any solves performed by caller)
  std::vector<int> homogeneousBCDofOrdinalsForCell(MeshPtr mesh, GlobalIndexType cellID);
  
  int nonZeroEntryCount(DofOrderingPtr testOrdering);
  
  virtual void operators(int testID1, int testID2,
                         std::vector<Camellia::EOperator> &testOp1,
                         std::vector<Camellia::EOperator> &testOp2);

  virtual void printInteractions();

  std::string displayString() const;

  static TIPPtr<Scalar> ip();

  static pair<TIPPtr<Scalar>, VarPtr > standardInnerProductForFunctionSpace(Camellia::EFunctionSpace fs, bool useTraceVar, int spaceDim);
};

extern template class TIP<double>;
}

#endif
