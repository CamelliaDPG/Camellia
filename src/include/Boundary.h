// @HEADER
//
// Original version Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
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

#include "Intrepid_FieldContainer.hpp"

#include "Element.h"

#include "DofInterpreter.h"

#include "Epetra_Map.h"

namespace Camellia
{
class Boundary
{
  std::set<std::pair<GlobalIndexType,unsigned>> _boundaryElements; // first arg is cellID, second arg is sideOrdinal

  MeshPtr _mesh;
  bool _imposeSingletonBCsOnThisRank; // this only governs singleton BCs which don't specify a vertex number.  Otherwise, the rule is that a singleton BC is imposed on the rank that owns the active cell of least ID that contains the vertex.
public:
  Boundary();
  void setMesh(MeshPtr mesh);
  template <typename Scalar>
  void bcsToImpose(Intrepid::FieldContainer<GlobalIndexType> &globalIndices, Intrepid::FieldContainer<Scalar> &globalValues,
                   TBC<Scalar> &bc, std::set<GlobalIndexType> &globalIndexFilter,
                   DofInterpreter* dofInterpreter);
  template <typename Scalar>
  void bcsToImpose(Intrepid::FieldContainer<GlobalIndexType> &globalIndices, Intrepid::FieldContainer<Scalar> &globalValues, TBC<Scalar> &bc,
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
  void bcsToImpose(std::map<GlobalIndexType,Scalar> &globalDofIndicesAndValues, TBC<Scalar> &bc, GlobalIndexType cellID,
                   std::set<std::pair<int, unsigned>> &singletons, DofInterpreter* dofInterpreter);
  void buildLookupTables();
};
}

#endif
