//
//  DofInterpreter.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//

/*
 //@HEADER
 // ***********************************************************************
 //
 //                          DofInterpreter
 //
 // ***********************************************************************
 //@HEADER
 */

#ifndef Camellia_debug_DofInterpreter_h
#define Camellia_debug_DofInterpreter_h

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"
#include "Epetra_Vector.h"

#include <set>

using namespace std;

//! DofInterpreter: an abstract class defining methods for converting between element-local and global representations of finite element coefficients and data.

/*!

 In finite elements we have element-local representations of finite element basis functions, as well as global representations of those functions.  The relationship between these representations depends on how the elements are reconciled where the local discretizations differ---two classical choices are the maximum rule and the minimum rule.  Camellia includes GDAMaximumRule2D which implements the maximum rule in two spatial dimensions, as well as GDAMinimumRule, which implements the minimum rule in arbitrary dimensions, including support for space-time elements.  (As of this writing, the support for space-time is limited to nonconforming discretizations---specifically, discretizations that only enforce continuities along element faces.)
 
 DofInterpreter defines methods that allow conversion between the local and global representations---we call this interpretation.  We make a distinction between interpreting data and interpreting coefficients.  Data refers to integral quantities that depend on the corresponding basis functions, while coefficients are weights on those functions, which may represent solutions in terms of those functions.  Camellia employs data interpretation to construct the global stiffness and load vectors from local stiffness and load vectors, and coefficient interpretation to construct element-local solutions from a global solution.  Thus interpretLocalData can be understood as the transpose of interpretGlobalCoefficients.
 
 Additionally, DofInterpreter provides two local coefficient interpretation methods, interpretLocalCoefficients and interpretLocalBasisCoefficients.  These allow the determination of a set of global coefficients which best fits (in the sense of a least-squares fit) the functions represented by the local coefficients.  The usual use of interpretLocalBasisCoefficients is for determining coefficients imposed by boundary conditions.
 
 \author Nathan V. Roberts, ALCF.
 
 \date Last modified on 13-Jul-2015.
 */


namespace Camellia
{
class DofInterpreter
{
protected:
  MeshPtr _mesh;
public:
  DofInterpreter(MeshPtr mesh) : _mesh(mesh) {}
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType rank) = 0;

  virtual void interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localData,
                                  Intrepid::FieldContainer<double> &globalData, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;

  virtual void interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localStiffnessData, const Intrepid::FieldContainer<double> &localLoadData,
                                  Intrepid::FieldContainer<double> &globalStiffnessData, Intrepid::FieldContainer<double> &globalLoadData, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  
  //!! Determines global coefficients corresponding to the provided local coefficients; in the case of minimum-rule meshes, this will be computed using a least-squares fit.  If some of the corresponding global coefficients are off-rank in globalCoefficients, these will be ignored.  (See the version of interpretLocalCoefficients that takes an STL map for an alternative that will include all global coefficients, regardless of data distribution.)
  virtual void interpretLocalCoefficients(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients);
  
  //!! Determines global coefficients corresponding to the provided local coefficients; in the case of minimum-rule meshes, this will be computed using a least-squares fit.
  virtual void interpretLocalCoefficients(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localCoefficients,
                                          std::map<GlobalIndexType,double> &fittedGlobalCoefficients,
                                          const std::set<int> &trialIDsToExclude);

  virtual void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Intrepid::FieldContainer<double> &basisCoefficients,
      Intrepid::FieldContainer<double> &globalCoefficients, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;

  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localDofs, const Epetra_MultiVector &globalDofs) = 0;

  //!! Returns the global dof indices for the cell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  virtual set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) = 0;
  
  //!! Returns the global dof indices for the indicated subcell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  virtual set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal) = 0;

  //!! MPI-communicating method.  Must be called on all ranks.
  virtual std::set<GlobalIndexType> importGlobalIndicesForCells(const std::vector<GlobalIndexType> &cellIDs);

  //!! MPI-communicating method.  Must be called on all ranks.  Keys are cellIDs (the ones requested), values the global dof indices with support on that cell.
  virtual std::map<GlobalIndexType,std::set<GlobalIndexType>> importGlobalIndicesMap(const std::set<GlobalIndexType> &cellIDs);
  
  virtual ~DofInterpreter() {}
};
}

#endif
