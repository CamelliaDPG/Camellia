// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  Boundary.h
 *
 */

#ifndef DPG_BOUNDARY
#define DPG_BOUNDARY

#include "TypeDefs.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Camellia_FieldContainer.hpp"

#include "Element.h"

#include "DofInterpreter.h"

namespace Camellia
{
class Boundary
{
  std::set<std::pair<GlobalIndexType,unsigned>> _boundaryElements; // first arg is cellID, second arg is sideOrdinal

  MeshPtr _mesh;
public:
  Boundary();
  void setMesh(MeshPtr mesh);

  template <typename Scalar>
  void bcsToImpose(Camellia::FieldContainer<GlobalIndexType> &globalIndices, Camellia::FieldContainer<Scalar> &globalValues, TBC<Scalar> &bc,
                   DofInterpreter* dofInterpreter);

  //! Determine rank-local values to impose for the "point" boundary conditions (e.g., a point condition on a pressure variable)
  /*!
   \param globalDofIndicesAndValues - (Out) keys are the global degree-of-freedom indices, values are their coefficients (weights).
   \param bc - (In) the BC object specifying the boundary conditions
   \param dofInterpreter - (In) the DofInterpreter
   */
  template <typename Scalar>
  void singletonBCsToImpose(std::vector<std::pair<GlobalIndexType,Scalar>> &globalDofIndicesAndValues, TBC<Scalar> &bc,
                            DofInterpreter* dofInterpreter);
  
  //! Determine values to impose on a single cell.
  /*!
   \param globalDofIndicesAndValues - (Out) keys are the global degree-of-freedom indices, values are their coefficients (weights).
   \param bc - (In) the BC object specifying the boundary conditions
   \param cellID - (In) the cell on which boundary conditions are requested
   \param singletons - (In) "point" boundary conditions (e.g., a point condition on a pressure variable); pairs are (trialID, vertexOrdinalInCell).
   \param dofInterpreter - (In) the DofInterpreter
   */
  template <typename Scalar>
  void bcsToImpose(std::vector<std::pair<GlobalIndexType,Scalar>> &globalDofIndicesAndValues, TBC<Scalar> &bc, GlobalIndexType cellID,
                   DofInterpreter* dofInterpreter);
  void buildLookupTables();
};
}

#endif
