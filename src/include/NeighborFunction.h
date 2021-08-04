// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  NeighborFunction.h
//  Camellia
//
//  Created by Nate Roberts on 11/22/19.
//
//

#ifndef Camellia_NeighborFunction_h
#define Camellia_NeighborFunction_h

#include "Function.h"
#include "Mesh.h"

namespace Camellia
{
  /** \brief NeighborFunction is a boundary-only function that provides the values of a provided function as seen by the immediate neighbor.
   
   This is useful in the context of data that is discontinuous at cell interfaces, such as arises in DG methods.
   
   After any changes to underlying data, and before NeighborFunction::values() is called, importCellData() should be called with
   *all* cellIDs on which the function will be computed (not their neighbors); the implementation performs a neighbor determination
   and ensures that all required data is imported.
   
   */
template <typename Scalar>
class NeighborFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _fxn;
  MeshPtr _mesh; // required for determining neighbors, computing neighbor values
public:
  NeighborFunction(TFunctionPtr<Scalar> fxn, MeshPtr mesh);
  
  using TFunction<Scalar>::values;
  //! BasisCachePtr argument must be a side basis cache; an exception is thrown otherwise
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr sideBasisCache) override;

  // importCellData means: "Be ready to compute this function on these cells."
  // Therefore, NeighborFunction augments cellIDs with their immediate neighbors, and calls _fxn->importCellData() on that set.
  void importCellData(std::vector<GlobalIndexType> cellIDs) override;

  // will propagate this to _fxn
  size_t getCellDataSize(GlobalIndexType cellID) override; // size in bytes
  // will propagate this to _fxn
  void packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength) override;
  // will propagate this to _fxn
  size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength) override; // returns bytes consumed
  
  //!! returns "neighbor(_fxn->displayString())"
  std::string displayString() override;
  bool boundaryValueOnly() override; // returns true
};
}

#endif
