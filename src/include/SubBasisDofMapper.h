// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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
#include "Camellia_FieldContainer.hpp"
#include "Teuchos_RCP.hpp"

using namespace std;

namespace Camellia
{
class SubBasisDofMapper;
typedef Teuchos::RCP<SubBasisDofMapper> SubBasisDofMapperPtr;

struct SubBasisMapInfo
{
  set<int> basisDofOrdinals;
  vector<GlobalIndexType> globalDofOrdinals;
  Camellia::FieldContainer<double> weights;
  bool isIdentity = false;
};

class SubBasisDofMapper
{
public:
  virtual Camellia::FieldContainer<double> mapData(bool transposeConstraint, Camellia::FieldContainer<double> &data, bool applyOnLeftOnly = false) = 0; // constraint matrix is sized "fine x coarse" -- so transposeConstraint should be true when data belongs to coarse discretization, and false when data belongs to fine discretization.  i.e. in a minimum rule, transposeConstraint is true when the map goes from global to local, and false otherwise.
  
  virtual void mapDataIntoGlobalContainer(const Camellia::FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                          bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Camellia::FieldContainer<double> &globalData) = 0;

  virtual void mapDataIntoGlobalContainer(const Camellia::FieldContainer<double> &allLocalData, const vector<int> &basisOrdinalsInLocalData,
                                          const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                          bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Camellia::FieldContainer<double> &globalData) = 0;

  virtual void mapSubBasisDataIntoGlobalContainer(const Camellia::FieldContainer<double> &subBasisData, const std::map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                                  bool fittableDofsOnly, const std::set<GlobalIndexType> &fittableDofIndices, Camellia::FieldContainer<double> &globalData);
  
  //  virtual Camellia::FieldContainer<double> getConstraintMatrix();

  virtual Camellia::FieldContainer<double> mapCoarseCoefficients(Camellia::FieldContainer<double> &coarseCoefficients)
  {
    return mapData(false,coarseCoefficients); // 5-18-15: changed "true" to "false"
  }
  
  virtual Camellia::FieldContainer<double> mapFineData(Camellia::FieldContainer<double> &fineData)
  {
    return mapData(true, fineData); // 5-18-15: changed "false" to "true"
  }

  virtual const set<int> &basisDofOrdinalFilter() = 0;
  virtual const vector<GlobalIndexType> &mappedGlobalDofOrdinals() = 0;

  //! returns true if the sub basis map is a simple permutation, negated
  virtual bool isNegatedPermutation() = 0;
  
  //! returns true if the sub basis map is a simple permutation
  virtual bool isPermutation() = 0;
  
  virtual set<GlobalIndexType> mappedGlobalDofOrdinalsForBasisOrdinals(set<int> &basisDofOrdinals) = 0;

  virtual SubBasisDofMapperPtr negatedDofMapper() = 0; // this dof mapper, but with negated coefficients (useful for fluxes)
  
  virtual SubBasisDofMapperPtr restrictDofOrdinalFilter(const set<int> &newDofOrdinalFilter) = 0; // this dof mapper, restricted to the specified basisDofOrdinals
  virtual SubBasisDofMapperPtr restrictGlobalDofOrdinals(const set<GlobalIndexType> &newGlobalDofOrdinals) = 0; // this dof mapper, restricted to the specified global dof ordinals
  
  virtual ~SubBasisDofMapper();

  static SubBasisDofMapperPtr subBasisDofMapper(const set<int> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals);
  static SubBasisDofMapperPtr subBasisDofMapper(const set<int> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, const Camellia::FieldContainer<double> &constraintMatrix);
  //  static SubBasisDofMapperPtr subBasisDofMapper(); // determines if the constraint is a permutation--if it is, then
};
}

#endif
