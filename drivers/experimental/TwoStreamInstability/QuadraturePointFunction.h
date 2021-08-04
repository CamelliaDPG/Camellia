#ifndef __QuadraturePointFunction__
#define __QuadraturePointFunction__

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//@HEADER

#include "Function.h"
#include "Mesh.h"
#include "BasisCache.h"

#include "TypeDefs.h"

//! QuadraturePointFunction: define values at quadrature points.
//
/*!

 \author Nathan V. Roberts, Sandia National Laboratories.

 \date Last modified on 3-March-2020.
 */

namespace Camellia
{
class QuadraturePointFunction : public TFunction<double>
{
  MeshPtr _mesh;
  int _quadratureEnrichment;
  
  FieldContainer<double> _valuesForActiveCells; // values for locally owned cells // TODO: make this a map based on element type; this assumes same discretization across all elements
  std::vector<GlobalIndexType> _cellIDs; // corresponds to the ordering in _valueForActiveCells
  
  std::vector< FieldContainer<double> > _sideValuesForActiveCells; // for the boundaryValueOnly=true case.  Again, this should become a map based on element type once we relax the assumption that all elements are of like type
  
  bool _boundaryValueOnly;
public:
  QuadraturePointFunction(MeshPtr mesh, int quadratureEnrichment, int rank=0, bool boundaryValueOnly=false);
  
  using TFunction<double>::values;
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override;

  bool boundaryValueOnly() override;
  
  // this calls the other updateValues() provided by subclasses
  void updateValues();
  
  // subclasses should override this:
  virtual void updateValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) = 0;
  
  //! Called before the virtual updateValues() above; subclasses may override to perform any prerequisite MPI communication.
  virtual void willUpdateValues();

  //! returns a function whose values are (or can be) precomputed.  They are only updated if/when updateValues is called.
  //! If setValuesNow is true, f::values() is called immediately, and its results stored.
  static Teuchos::RCP<QuadraturePointFunction> precomputedFunction(MeshPtr mesh, int quadratureEnrichment, FunctionPtr f, bool setValuesNow);
  
  ~QuadraturePointFunction();
};
}


#endif
