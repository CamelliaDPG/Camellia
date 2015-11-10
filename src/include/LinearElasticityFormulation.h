//
//  LinearElasticityFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 11/2/15, starting from some code from Brendan Keith
//
//

#ifndef Camellia_LinearElasticityFormulation_h
#define Camellia_LinearElasticityFormulation_h

#include <Teuchos_ParameterList.hpp>

#include "BF.h"
#include "GMGSolver.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"

namespace Camellia
{
  class LinearElasticityFormulation
  {
    BFPtr _bf;
    
    int _spaceDim;
    double _mu, _lambda;
    
    double _C[3][3][3][3]; // compliance tensor (supports both 2D and 3D)
    double _E[3][3][3][3]; // stiffness tensor (supports both 2D and 3D)
    
    bool _useConformingTraces;
    
    LinearTermPtr _t1, _t2, _t3; // tractions
    
    SolverPtr _solver;
    
    TSolutionPtr<double> _solution;
    
    RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;
    
    std::map<int,int> _trialVariablePolyOrderAdjustments;
    
    VarFactoryPtr _vf;
    
    static const string S_U1, S_U2, S_U3;
    static const string S_W;
    static const string S_SIGMA11, S_SIGMA12, S_SIGMA13, S_SIGMA21, S_SIGMA22, S_SIGMA23, S_SIGMA31, S_SIGMA32, S_SIGMA33;
    
    static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
    static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;
    
    static const string S_V1, S_V2, S_V3;
    static const string S_Q;
    static const string S_TAU1, S_TAU2, S_TAU3;
    
    void CHECK_VALID_COMPONENT(int i); // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
    
    // ! initialize the Solution object(s) using the provided MeshTopology
    void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                            TFunctionPtr<double> forcingFunction, std::string fileToLoadPrefix,
                            int temporalPolyOrder);
    
    void turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver);
  public:
    LinearElasticityFormulation(Teuchos::ParameterList &parameters);
    
    // ! the Stokes VGP formulation bilinear form
    BFPtr bf();
    
    // ! compliance tensor
    double C(int i, int j, int k, int l);
    
    // ! stiffness tensor
    double E(int i, int j, int k, int l);
    
    // ! initialize the Solution object(s) using the provided MeshTopology
    void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1,
                            TFunctionPtr<double> forcingFunction = Teuchos::null,
                            int temporalPolyOrder = 1);

    // ! Returns lambda.
    double lambda();
    
    // ! Returns mu.
    double mu();
    
    // ! refine according to energy error in the solution
    void refine();
    
    // ! h-refine according to energy error in the solution
    void hRefine();
    
    // ! p-refine according to energy error in the solution
    void pRefine();
    
    // ! returns the RefinementStrategy object being used to drive refinements
    RefinementStrategyPtr getRefinementStrategy();
    
    // ! Returns an RHSPtr corresponding to the vector forcing function f and the formulation.
    RHSPtr rhs(TFunctionPtr<double> f);
    
    // ! set the RefinementStrategy to use for driving refinements
    void setRefinementStrategy(RefinementStrategyPtr refStrategy);
    
    // ! Returns the solution
    TSolutionPtr<double> solution();
    
    // ! Solves
    void solve();
    
    // ! Solves iteratively
    void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);
    
    // ! Returns the spatial dimension.
    int spaceDim();
    
    // field variables:
    VarPtr sigma(int i, int j); // sigma_ij is the Reynolds-weighted derivative of u_i in the j dimension
    VarPtr u(int i);
    VarPtr w();
    
    // traces:
    VarPtr tn_hat(int i);
    VarPtr u_hat(int i);
    
    // test variables:
    VarPtr tau(int i);
    VarPtr v(int i);
    
    // ! returns the forcing function for this formulation if u is the exact solution.
    TFunctionPtr<double> forcingFunction(TFunctionPtr<double> u);
    
    // ! returns a map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
    const std::map<int,int> &getTrialVariablePolyOrderAdjustments();
    
    static LinearElasticityFormulation steadyFormulation(int spaceDim, double lambda, double mu, bool useConformingTraces);
  };
}


#endif
