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
  public:
    ErrorIndicator(MeshPtr mesh);
    
    //! returns the error measures for cells belonging to the local MPI rank
    const std::map<GlobalIndexType,double> &localErrorMeasures() const;
    
    //! adds to the provided container rank-local cells with error above the threshold
    virtual void localCellsAboveErrorThreshold(double threshold, vector<GlobalIndexType> &cellsAboveThreshold);
    
    //! determine rank-local error measures.  Subclasses should populate _localErrorMeasures
    virtual void measureError() = 0;
    
    //! returns maximum rank-local error measure
    virtual double maxLocalError() const;
    
    //! return maximum error measure (MPI collective method).
    virtual double maxError() const;
    
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
  };
}

#endif
