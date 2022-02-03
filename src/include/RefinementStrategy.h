// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  RefinementStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//

#ifndef Camellia_RefinementStrategy_h
#define Camellia_RefinementStrategy_h

#include "Teuchos_RCP.hpp"

#include "ErrorIndicator.h"
#include "IP.h"
#include "LinearTerm.h"
#include "RieszRep.h"
#include "TypeDefs.h"

using namespace std;

namespace Camellia
{
struct RefinementResults
{
  int numElements;
  int numDofs;
  double totalEnergyError;
};

template <typename Scalar>
class TRefinementStrategy
{
public:
  enum RefinementType
  {
    H_REFINEMENT, // distinction among h-refinement types comes in the RefinementPattern
    P_REFINEMENT
  };
protected:

  static RefinementResults setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError);
  ErrorIndicatorPtr _errorIndicator;
  ErrorIndicatorPtr _unrefinementErrorIndicator; // defaults to null, in which case _errorIndicator will be used

  double _relativeErrorThreshold;
  int _maxElementCount = std::numeric_limits<int>::max();
  bool _enforceOneIrregularity;
  bool _reportPerCellErrors;
  vector< RefinementResults > _results;
  double _min_h;
  int _maxRefinementLevel = std::numeric_limits<int>::max(); // as measured from coarse/initial mesh, which is level 0.  Default: no maximum.
  int _max_p;
  bool _preferPRefinements;
  
  MeshPtr mesh();
public:
  TRefinementStrategy( ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
  TRefinementStrategy( TSolutionPtr<Scalar> solution, double relativeEnergyThreshold, double min_h = 0, int max_p = 10,
                      bool preferPRefinements = false);
  TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, double relativeEnergyThreshold,
                      double min_h = 0, int max_p = 10, bool preferPRefinements = false, int cubatureEnrichment = 0);
  
  double computeTotalEnergyError();
  
  ErrorIndicatorPtr errorIndicator() const;
  
  //! unrefinement error indicator is used for determining parent errors to guide unrefinements; if null, the error indicator determined at construction will be used
  void setUnrefinementErrorIndicator(ErrorIndicatorPtr unrefinementErrorIndicator);
  ErrorIndicatorPtr unrefinementErrorIndicator() const;
  
  //! Set the energy threshold (relative to maximum element error) to use for refinements
  void setRelativeErrorThreshold(double value);
  void setEnforceOneIrregularity(bool value);

  //! Set a maximum number of active elements; used in refineAndUnrefine.  Enough unrefinements will be performed to reduce the number of active elements to the specified count.
  void setMaxElements(int maxElementCount);
  
  virtual void refine(bool printToConsole=false);
  virtual void refineAndUnrefine(bool unrefine, bool printToConsole);
  virtual void hRefineUniformly();

  virtual void refineCells(vector<GlobalIndexType> &cellIDs);
  virtual void unrefineCells(const std::set<GlobalIndexType> &cellIDs);
  static void pRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs);
  static void hRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs);
  static void hRefineUniformly(MeshPtr mesh);

  std::set<GlobalIndexType> localCellsIneligibleForRefinement();
  
  void getCellsAboveErrorThreshhold(vector<GlobalIndexType> &cellsToRefine, std::set<GlobalIndexType> localCellsToExclude = std::set<GlobalIndexType>());
  void setMinH(double value);
  void setMaxP(int max_p);
  void setMaximumRefinementLevel(int maxLevel);
  void setReportPerCellErrors(bool value);

  double getEnergyError(int refinementNumber);
  GlobalIndexType getNumElements(int refinementNumber);
  GlobalIndexType getNumDofs(int refinementNumber);
  
  //! return the number of refinements that have been recorded in the internal results container; used for getEnergyError(), getNumElements(), and getNumDofs().
  int getNumRefinements() const;
  
  static RefinementStrategyPtr energyErrorRefinementStrategy(SolutionPtr soln, double relativeEnergyThreshold);
  static RefinementStrategyPtr energyErrorRefinementStrategy(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                             double relativeEnergyThreshold, int cubatureEnrichmentDegree = 0);
  static RefinementStrategyPtr gradientRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold);
  static RefinementStrategyPtr hessianRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold);
  
  virtual ~TRefinementStrategy() {}
};

extern template class TRefinementStrategy<double>;
}


#endif
