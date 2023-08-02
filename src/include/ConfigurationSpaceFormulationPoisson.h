//
//  ConfigurationSpaceFormulationPoisson.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#ifndef ConfigurationSpaceFormulationPoisson_h
#define ConfigurationSpaceFormulationPoisson_h

#include "Camellia.h"
#include "ConfigurationSpaceFormulation.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
    \brief Class for specifying a Poisson configuration-space formulation; intended to be used in context of a Vlasov (phase-space) solve.
   */
  class ConfigurationSpaceFormulationPoisson
:
  public ConfigurationSpaceFormulation
  {
    std::vector<double> _q;
    double _eps0;
    
    TimeDiscretizationType _timeDiscretizationType;
    int _spaceDim;
    
    BFPtr _poissonBF;
  
    FunctionPtr _ESoln, _EDiff;
    
    std::vector<FunctionPtr> _E_extruded;
    std::vector<FunctionPtr> _E_priorTime_extruded;
  public:
    // strings for variables:
    static const string S_EX;     // trial variable - electric field E_x
    static const string S_EY;     // trial variable - electric field E_y
    static const string S_EZ;     // trial variable - electric field E_z
    static const string S_VE;     // trial variable - potential VE
    static const string S_E_HAT;  // trace variable - trace of E dot n in H^-1/2
    static const string S_VE_HAT; // trace variable - trace of VE in H^1/2
    static const string S_V_POISSON;   // scalar test variable
    static const string S_TAU_POISSON; // vector test variable
  public:
    //! one entry in q per species
    ConfigurationSpaceFormulationPoisson(const std::vector<double> &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType);
    ConfigurationSpaceFormulationPoisson(const double &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType);

    void initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k) override;
    
    virtual void setMoments(std::vector< Teuchos::RCP<SolutionMoments> > fMoments, FunctionPtr n0) override;
    
    //! Returns a name distinguishing this configuration-space formulation from others.  "Maxwell", "Poisson", "Gauss" are all possibilities.
    virtual std::string name() const override;
        
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i(int zeroBasedComp) override;
    
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i(int zeroBasedComp) override;
    
    //! returns the specified component of the magnetic field at prior time, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i_priorTime(int zeroBasedComp) override;
    
    //! returns the specified component of the electric field at prior timeas a Function, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i_priorTime(int zeroBasedComp) override;
    
    //! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
    virtual int maxMomentNumber() override;
    
    //! Perform MPI communication, etc. to ensure that the B_i, E_i function values for a given configuration-space cell are available on each MPI rank that has a corresponding phase-space cell.
    virtual void updateConfigurationSpaceFunctions() override;
    
    // electric field variables
    VarPtr E_fieldVar(int zeroBasedComp) override;
    
    //! returns the trace variable corresponding to the specified E component.  Defined in 1D for Poisson and Maxwell; defined in higher dimensions for Maxwell.  See En_hat().
    VarPtr E_hat(int zeroBasedComp) override;
    
    //! returns the trace variable corresponding to E (in 1D) or E dot n (in higher dimensions).  Defined in higher dimensions for Poisson, but not for Maxwell.
    VarPtr En_hat() override;
    
    // potential variables:
    VarPtr VE_fieldVar() override;
    virtual VarPtr VE_hat() override;
    
    // test variables:
    VarPtr v();   // scalar test variable
    VarPtr tau(); // vector test variable
    
    //! The relative L^2 norm of the difference between E for solution() and solutionPrevious().
    double l2NormOfSolutionUpdate() override;
  };
}

#endif /* ConfigurationSpaceFormulationPoisson_h */
