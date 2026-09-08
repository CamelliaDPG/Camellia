// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef BASIS_FACTORY
#define BASIS_FACTORY

#include "TypeDefs.h"

#include "Camellia_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

#include "VectorizedBasis.h"

#include "CamelliaIntrepidExtendedTypes.h"

#include "CellTopology.h"

namespace Camellia
{
typedef Teuchos::RCP< Camellia::Basis<> > BasisPtr;

class BasisFactory
{
  typedef Camellia::EFunctionSpace FSE;
private:
  map< pair< pair<int,int>, Camellia::EFunctionSpace >, BasisPtr >
  _existingBases; // keys are ((polyOrder,cellTopoKey),fs))
  map< pair< pair<int,int>, Camellia::EFunctionSpace >, BasisPtr >
  _conformingBases; // keys are ((polyOrder,cellTopoKey),fs))
  map< std::tuple<Camellia::Basis<>*, int, Camellia::EFunctionSpace>, BasisPtr >
  _tensorBases; // keys are (one-lower tensorial degree basis, final tensorial polynomial degree, final function space)
  map< std::tuple<Camellia::Basis<>*, int, Camellia::EFunctionSpace>, BasisPtr >
  _conformingTensorBases; // keys are (one-lower tensorial degree basis, final tensorial polynomial degree, final function space)

  map<unsigned, BasisPtr> _nodalBasisForShardsTopology;
  map<CellTopologyKey, BasisPtr> _nodalBasisForTopology;
  
  // the following maps let us remember what arguments were used to create a basis:
  // (this is useful to, say, create a basis again, but now with polyOrder+1)
  map< Camellia::Basis<>*, int > _polyOrders; // allows lookup of poly order used to create basis
  //  static map< Camellia::Basis<>*, int > _ranks; // allows lookup of basis rank
  map< Camellia::Basis<>*, Camellia::EFunctionSpace > _functionSpaces; // allows lookup of function spaces
  map< Camellia::Basis<>*, int > _cellTopoKeys; // allows lookup of cellTopoKeys
  int _defaultTemporalPolyOrder;
  
  bool _useEnrichedTraces; // i.e. p+1, not p (default is true: this is what we need to prove optimal convergence)
  bool _useLobattoForQuadHGRAD;
  bool _useLobattoForQuadHDIV;
  bool _useLobattoForLineHGRAD;
  bool _useLegendreForLineHVOL;
  bool _useLegendreForQuadHVOL;
public:
  BasisFactory();

  BasisPtr getBasis(const std::vector<int> &H1Orders, CellTopoPtr cellTopo, FSE functionSpace);
  BasisPtr getBasis(const std::vector<int> &H1Orders, CellTopoPtr cellTopo, const std::vector<FSE> &functionSpaces);

  BasisPtr getBasis( int polyOrder, unsigned cellTopoKey, FSE fs );
  BasisPtr getBasis( int polyOrder, CellTopoPtr cellTopo, FSE fs );

  BasisPtr getConformingBasis(const std::vector<int> &H1Orders, CellTopoPtr cellTopo, const FSE functionSpace);
  BasisPtr getConformingBasis(const std::vector<int> &H1Orders, CellTopoPtr cellTopo, const std::vector<FSE> &functionSpaces);

  BasisPtr getConformingBasis( int polyOrder, unsigned cellTopoKey, FSE fs );
  BasisPtr getConformingBasis( int polyOrder, CellTopoPtr cellTopo, FSE fs );

  BasisPtr getSpaceTimeConformingBasis( const std::vector<int> &polyOrderSpace, const int polyOrderTime,
                                       CellTopoPtr spaceTimeCellTopo, const std::vector<FSE> &spatialFS, const FSE temporalFS);
  BasisPtr getSpaceTimeConformingBasis( const std::vector<int> &polyOrderSpace, const int polyOrderTime,
                                       CellTopoPtr spaceTimeCellTopo, const FSE spatialFS, const FSE temporalFS);
  
  BasisPtr getSpaceTimeConformingBasis( const int polyOrderSpace, const int polyOrderTime,
                                       CellTopoPtr spaceTimeCellTopo, const FSE spatialFS, const FSE temporalFS);
  
  BasisPtr getNodalBasisForCellTopology(CellTopoPtr cellTopo);
  BasisPtr getNodalBasisForCellTopology(unsigned cellTopoKey);

  BasisPtr addToPolyOrder(BasisPtr basis, int pToAdd);
  BasisPtr setPolyOrder(BasisPtr basis, int polyOrderToSet);

  int basisPolyOrder(BasisPtr basis);
  int getBasisRank(BasisPtr basis, EOperator op=OP_VALUE);
  Camellia::EFunctionSpace getBasisFunctionSpace(BasisPtr basis);

  bool basisKnown(BasisPtr basis);

  void registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, FSE fs );

  void setUseEnrichedTraces( bool value );

  // the following convenience methods belong in Basis or perhaps a wrapper thereof
  set<int> sideFieldIndices( BasisPtr basis, bool includeSideSubcells = true); // includeSideSubcells: e.g. include vertices as part of quad sides

  void setUseLegendreForQuadHVol(bool value);
  void setUseLobattoForQuadHGrad(bool value);
  void setUseLobattoForQuadHDiv(bool value);

  static Teuchos::RCP<BasisFactory> basisFactory(); // shared, global BasisFactory
  static void deleteBasisFactory(); // delete shared, global BasisFactory -- used during Camellia::finalize()
};
}

#endif
