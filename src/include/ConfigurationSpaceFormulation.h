//
//  ConfigurationSpaceFormulation.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/11/22.
//

#ifndef ConfigurationSpaceFormulation_h
#define ConfigurationSpaceFormulation_h

#include "Camellia.h"
#include "Camellia_TimeDiscretizationType.h"
#include "SolutionMoments.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
    \brief Abstract class for specifying configuration-space solve in terms of (moments of) a phase-space distribution.  Concrete subclasses will include a Poisson formulation, as well as a Maxwell formulation.
   */
  class ConfigurationSpaceFormulation
  {
  protected:
    Teuchos::RCP<SolutionMoments> _fMoments;
    SolutionPtr _configurationSoln;
    SolutionPtr _configurationSolnPrev;      // previously-computed solution.  Can be used to check convergence of fixed-point iteration, etc.
    SolutionPtr _configurationSolnPriorTime; // prior-time solution; null for space-time configurations.
    MeshPtr _phaseMesh;
    MeshTopologyPtr _configurationMeshTopo;
  public:
    ConfigurationSpaceFormulation();
    
    //! Create the configuration-space mesh based on the provided space-time MeshTopology.
    virtual void initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k) = 0;
    
    MeshPtr configurationSpaceMesh();
    
    MeshTopologyPtr configurationSpaceMeshTopology();
    
    //! Returns a name distinguishing this configuration-space formulation from others.  "Maxwell", "Poisson", "Gauss" are all possibilities.
    virtual std::string name() const = 0;
    
    //! should be called by subclass during initialization: specify the configuration
    void setConfigurationSolution(SolutionPtr configurationSolution);
    
    //! should be called by subclass during initialization: specify the solution object into which previously-computed solution will be copied prior to solve.
    void setConfigurationSolutionPrevious(SolutionPtr configurationSolutionPrev);
    
    //! should be called by subclass during initialization: specify the solution object which will be used to store the prior-time solution (used in RHS for time-marching).
    void setConfigurationSolutionPriorTime(SolutionPtr configurationSolutionPriorTime);
    
    //! should be called prior to first solve: set the solution moments object, as well as an initial density which is set if quasi-neutrality is being enforced.  Overriding subclasses should call the base class's method.
    virtual void setMoments(Teuchos::RCP<SolutionMoments> fMoments, FunctionPtr n0);
    
    //! When updateMoments is true, will call fMoments->update() prior to solve.
    void solve(const bool updateMoments);
    
    //! Copy the solution to the solution returned by solutionPrevious().  This should be called after a fixed point iteration has completed.
    void copySolutionToPrevious();
    
    //! Copy the solution to the solution returned by solutionPriorTime().  This should be called after a time step solve has been completed.
    void copySolutionToPriorTime();
    
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i(int zeroBasedComp) = 0;
    
    //! returns the specified component of the electric field as a Function, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i(int zeroBasedComp) = 0;

    //! returns the field variable corresponding  to specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual VarPtr B_fieldVar(int zeroBasedComp);
    
    //! returns the field variable corresponding  to specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual VarPtr E_fieldVar(int zeroBasedComp);
    
    //! returns the field variable corresponding to the electric potential, if any.
    virtual VarPtr VE_fieldVar();
    
    //! returns the trace variable corresponding to the specified E component.  Defined in 1D for Poisson and Maxwell; defined in higher dimensions for Maxwell.  See En_hat().
    virtual VarPtr E_hat(int zeroBasedComp);
    
    //! returns the trace variable corresponding to E (in 1D) or E dot n (in higher dimensions).  Defined in higher dimensions for Poisson, but not for Maxwell.
    virtual VarPtr En_hat();
    
    // magnetic field variables
    virtual VarPtr B_hat(int zeroBasedComp);
    
    // potential:
    virtual VarPtr VE_hat();
    
    //! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
    virtual int maxMomentNumber() = 0;
    
    //! The configuration-space solution.
    SolutionPtr solution();
    
    //! The previous configuration-space solution.  Copied from solution when copySolutionToPrevious() is called.
    SolutionPtr solutionPrevious();
    
    //! The prior-time configuration-space solution (generally null for space-time configurations).  Copied from solution when copySolutionToPriorTime() is called.
    SolutionPtr solutionPriorTime();
    
    //! The relative L^2 norm of the difference between (some combination of) the field variables for solution() and solutionPrevious().
    virtual double l2NormOfSolutionUpdate();
    
    //! Perform MPI communication, etc. to ensure that the B_i, E_i function values for a given configuration-space cell are available on each MPI rank that has a corresponding phase-space cell.
    virtual void updateConfigurationSpaceFunctions() = 0;
    
    static Teuchos::RCP<ConfigurationSpaceFormulation> poissonFormulation(const double &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType);
  };
}

#endif /* ConfigurationSpaceFormulation_h */
