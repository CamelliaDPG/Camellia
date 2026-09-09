//
//  BGKFunction.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 12/22/21.
//

#ifndef BGKFunction_h
#define BGKFunction_h

#include "Function.h"
#include "SolutionMoments.h"

namespace Camellia
{
  class BGKFunction : public TFunction<double>
  {
    double _tau;
    FunctionPtr _f_soln;
    Teuchos::RCP<SolutionMoments> _f_solnMoments;
    
    FunctionPtr _zeroMoment;
    std::vector<FunctionPtr> _firstMoments; // up to three of these
    FunctionPtr _secondMoment;
    
  public:
    BGKFunction(const double tau, FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments);
    
    //! Import data required to evaluate BGKFunction on the specified phase-space cells.
    //!
    //! The cell IDs passed here are phase-space cell IDs, matching the mesh on which this
    //! BGKFunction is evaluated. BGKFunction internally translates these to the corresponding
    //! configuration-space cell IDs needed by the SolutionMomentFunction objects.
    //!
    //! This is a collective MPI operation.  Ranks with no local phase-space cells should
    //! call this with an empty vector so the delegated moment-data imports do not hang.
    void importCellData(std::vector<GlobalIndexType> phaseSpaceCellIDs) override;
    
    using TFunction<double>::values;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  };
}

#endif /* BGKFunction_h */
