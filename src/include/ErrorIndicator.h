// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  ErrorIndicator.h
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
// ***********************************************************************
//
//                  Camellia ErrorIndicator:
//
// Abstract class for error evaluation, used in automatic mesh refinement.
// Subclasses define the actual error measurement technique; preferred
// choice for DPG is the energy error of the residual (EnergyErrorIndicator).
// Other choices include gradient and hessian indicators.
//
// ***********************************************************************
//

#ifndef Camellia_ErrorIndicator_h
#define Camellia_ErrorIndicator_h

#include "Mesh.h"
#include "Solution.h"
#include "TypeDefs.h"

#include <map>

namespace Camellia {
  class ErrorIndicator
  {
  protected:
    MeshPtr _mesh;
    std::map<GlobalIndexType,double> _localErrorMeasures; // cellID -> error measure
    std::map<GlobalIndexType,double> _parentErrorMeasures; // cellID -> error measure
  public:
    ErrorIndicator(MeshPtr mesh);
    
    //! returns the error measures for cells belonging to the local MPI rank
    const std::map<GlobalIndexType,double> &localErrorMeasures() const;
    
    //! adds to the provided container rank-local cells with error above the threshold
    virtual void localCellsAboveErrorThreshold(double threshold, vector<GlobalIndexType> &cellsAboveThreshold);
    
    //! adds to the provided container locally-owned parents cells with error below the threshold
    virtual void localParentCellsBelowErrorThreshold(double threshold, vector<GlobalIndexType> &cellsBelowThreshold);
    
    //! Returns the cellCountTarget cells with maximum error.  May return more, in the case of a tie, or less, if there are not that many active cells in the mesh.  Excludes the cells indicated in excludedCells from consideration.
    std::vector< std::pair<GlobalIndexType,double> > globalMaximumErrorCells(const int cellCountTarget, const std::set<GlobalIndexType> &excludedCells);
    
    std::vector< std::pair<GlobalIndexType,double> > globalMinimumErrorParentCells(const int cellReductionTarget, const std::set<GlobalIndexType> &excludedChildren);
    
    std::set<GlobalIndexType> globalCellsBelowThreshold(const double &threshold, const std::set<GlobalIndexType> localCellsToExclude = std::set<GlobalIndexType>());
    
    std::vector< std::pair<GlobalIndexType,double> > globalParentCellsBelowThreshold(const double &maxError, const std::set<GlobalIndexType> &excludedChildren);
    
    //! determine rank-local error measures.  Subclasses should populate _localErrorMeasures
    virtual void measureError() = 0;
    
    //! returns maximum rank-local error measure
    virtual double maxLocalError(const std::set<GlobalIndexType> localCellsToExclude = std::set<GlobalIndexType>()) const;
    
    //! return maximum error measure (MPI collective method).
    virtual double maxError(const std::set<GlobalIndexType> localCellsToExclude = std::set<GlobalIndexType>()) const;
    
    //! return minimum error measure across all cells (MPI collective method).
    virtual double minError() const;
    
    //! return minimum error measure across all parent cells that have no grandchildren (MPI collective method).
    virtual double minParentError() const;
    
    //! for all parents of leaf cells (in the h-refinement sense) gather the child error sum to the processor that owns the first child (MPI collective method).
    virtual void gatherParentError(const std::set<GlobalIndexType> &excludedChildren);
    
    MeshPtr mesh() const;
    
    //! return total error (default implementation assumes l2 combination of cell errors is appropriate).  MPI-collective method.
    virtual double totalError() const;
    
    template <typename Scalar>
    static ErrorIndicatorPtr energyErrorIndicator(TSolutionPtr<Scalar> solution);
    
    template <typename Scalar>
    static ErrorIndicatorPtr energyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                  int quadratureEnrichment);
    
    //! Use the L^2 norm of the function specified as error indicator.
    template <typename Scalar>
    static ErrorIndicatorPtr functionErrorIndicator(MeshPtr mesh, FunctionPtr f, int quadratureEnrichment);
    
    template <typename Scalar>
    static ErrorIndicatorPtr gradientErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar);
    
    template <typename Scalar>
    static ErrorIndicatorPtr hessianErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar);
    
    //! Measure parent error by doing a projection of the children's data onto the parent and evaluating the difference in the L^2 norm of the specified variable.
    template <typename Scalar>
    static ErrorIndicatorPtr unrefinementProjectionErrorIndicator(TSolutionPtr<Scalar> solution, VarPtr scalarVar);
  };
}

#endif
