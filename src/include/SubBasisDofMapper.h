//
//  SubBasisDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef Camellia_debug_SubBasisDofMapper_h
#define Camellia_debug_SubBasisDofMapper_h

#include "TypeDefs.h"

#include <set>
#include "Intrepid_FieldContainer.hpp"
#include "Teuchos_RCP.hpp"

using namespace std;

namespace Camellia
{
class SubBasisDofMapper;
typedef Teuchos::RCP<SubBasisDofMapper> SubBasisDofMapperPtr;

struct SubBasisMapInfo
{
  set<unsigned> basisDofOrdinals;
  vector<GlobalIndexType> globalDofOrdinals;
  Intrepid::FieldContainer<double> weights;
};

class SubBasisDofMapper
{
public:
  virtual Intrepid::FieldContainer<double> mapData(bool transposeConstraint, Intrepid::FieldContainer<double> &data, bool applyOnLeftOnly = false) = 0; // constraint matrix is sized "fine x coarse" -- so transposeConstraint should be true when data belongs to coarse discretization, and false when data belongs to fine discretization.  i.e. in a minimum rule, transposeConstraint is true when the map goes from global to local, and false otherwise.

  virtual void mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                          bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData) = 0;

  //  virtual Intrepid::FieldContainer<double> getConstraintMatrix();

  virtual Intrepid::FieldContainer<double> mapCoarseCoefficients(Intrepid::FieldContainer<double> &coarseCoefficients)
  {
    return mapData(false,coarseCoefficients); // 5-18-15: changed "true" to "false"
  }
  virtual Intrepid::FieldContainer<double> mapFineData(Intrepid::FieldContainer<double> &fineData)
  {
    return mapData(true, fineData); // 5-18-15: changed "false" to "true"
  }

  virtual const set<unsigned> &basisDofOrdinalFilter() = 0;
  virtual vector<GlobalIndexType> mappedGlobalDofOrdinals() = 0;

  virtual SubBasisDofMapperPtr negatedDofMapper() = 0; // this dof mapper, but with negated coefficients (useful for fluxes)

  virtual ~SubBasisDofMapper();

  static SubBasisDofMapperPtr subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals);
  static SubBasisDofMapperPtr subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, const Intrepid::FieldContainer<double> &constraintMatrix);
  //  static SubBasisDofMapperPtr subBasisDofMapper(); // determines if the constraint is a permutation--if it is, then
};
}

#endif
