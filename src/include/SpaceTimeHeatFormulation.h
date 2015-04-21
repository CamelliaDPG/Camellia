//
//  SpaceTimeHeatFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_SpaceTimeHeatFormulation_h
#define Camellia_SpaceTimeHeatFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"
#include "Solution.h"
#include "RefinementStrategy.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"

namespace Camellia {
  class SpaceTimeHeatFormulation {
    BFPtr _bf;

    int _spaceDim;
    bool _useConformingTraces;
    double _epsilon;

    SolverPtr _solver;

    TSolutionPtr<double> _solution;

    RefinementStrategyPtr _refinementStrategy;

    VarFactory _vf;

    static const string S_U;
    static const string S_SIGMA1, S_SIGMA2, S_SIGMA3;

    static const string S_U_HAT;
    static const string S_SIGMA_N_HAT;

    static const string S_V;
    static const string S_TAU;

    // ! initialize the Solution object(s) using the provided MeshTopology
    void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                            TFunctionPtr<double> forcingFunction, std::string fileToLoadPrefix);
  public:
    SpaceTimeHeatFormulation(int spaceDim, double epsilon, bool useConformingTraces = false);

    // ! the formulation's bilinear form
    BFPtr bf();

    // ! initialize the Solution object(s) using the provided MeshTopology
    void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1,
                            TFunctionPtr<double> forcingFunction = Teuchos::null);

    // ! initialize the Solution object(s) from file
    void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1,
                            TFunctionPtr<double> forcingFunction = Teuchos::null);

    // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
    // ! implementation, assumes that the constructor arguments provided to SpaceTimeHeatFormulation were the same
    // ! on the SpaceTimeHeatFormulation on which save() was invoked as they were for this SpaceTimeHeatFormulation.
    void load(std::string prefixString);

    // ! Returns epsilon.
    double epsilon();

    // ! refine according to energy error in the solution
    void refine();

    // ! returns the RefinementStrategy object being used to drive refinements
    RefinementStrategyPtr getRefinementStrategy();

    // ! Returns an RHSPtr corresponding to the scalar forcing function f and the formulation.
    RHSPtr rhs(TFunctionPtr<double> f);

    // ! Saves the solution(s) and mesh to an HDF5 format.
    void save(std::string prefixString);

    // ! set the RefinementStrategy to use for driving refinements
    void setRefinementStrategy(RefinementStrategyPtr refStrategy);

    // ! Returns the solution (at current time)
    TSolutionPtr<double> solution();

    // ! Solves
    void solve();

    // field variables:
    VarPtr sigma(int i);
    VarPtr u();

    // traces:
    VarPtr sigma_n_hat();
    VarPtr u_hat();

    // test variables:
    VarPtr tau();
    VarPtr v();

    static TFunctionPtr<double> forcingFunction(int spaceDim, double epsilon, TFunctionPtr<double> u);
  };
}


#endif
