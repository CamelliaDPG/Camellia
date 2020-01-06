// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SubBasisPermutationDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/14/14.
//
//

#ifndef Camellia_debug_SubBasisPermutationDofMapper_h
#define Camellia_debug_SubBasisPermutationDofMapper_h

#include "SubBasisDofMapper.h"

namespace Camellia
{
class SubBasisDofPermutationMapper : public SubBasisDofMapper
{
  set<int> _basisDofOrdinalFilter;
  vector<GlobalIndexType> _globalDofOrdinals;
  vector<int> _inversePermutation;
  bool _negate;
public:
  SubBasisDofPermutationMapper(const set<int> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, bool negate=false);
  const set<int> &basisDofOrdinalFilter();
  
  //! returns true if the sub basis map is a simple permutation, negated
  bool isNegatedPermutation();
  
  //! returns true if the sub basis map is a simple permutation -- always true for SubBasisDofPermutationMapper
  bool isPermutation();
  
  Camellia::FieldContainer<double> mapData(bool transposeConstraint, Camellia::FieldContainer<double> &localData, bool applyOnLeftOnly = false);
  
  void mapDataIntoGlobalContainer(const Camellia::FieldContainer<double> &allLocalData, const vector<int> &basisOrdinalsInLocalData,
                                  const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Camellia::FieldContainer<double> &globalData);
  void mapDataIntoGlobalContainer(const Camellia::FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Camellia::FieldContainer<double> &globalData);

  const std::vector<GlobalIndexType> &mappedGlobalDofOrdinals();
  std::set<GlobalIndexType> mappedGlobalDofOrdinalsForBasisOrdinals(std::set<int> &basisDofOrdinals);
  
  Camellia::FieldContainer<double> getConstraintMatrix();

  SubBasisDofMapperPtr negatedDofMapper(); // this mapper, but negated.
  
  SubBasisDofMapperPtr restrictDofOrdinalFilter(const set<int> &newDofOrdinalFilter); // this dof mapper, restricted to the specified basisDofOrdinals
  SubBasisDofMapperPtr restrictGlobalDofOrdinals(const set<GlobalIndexType> &newGlobalDofOrdinals); // this dof mapper, restricted to the specified global dof ordinals
};
}


#endif
