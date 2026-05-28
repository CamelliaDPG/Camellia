//
//  ConfigurationSpaceFormulationMaxwell1D2V.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/7/26.
//

#ifndef ConfigurationSpaceFormulationMaxwell1D2V_h
#define ConfigurationSpaceFormulationMaxwell1D2V_h

#include "Camellia.h"
#include "ConfigurationSpaceFormulation.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
    \brief Class for specifying a Maxwell configuration-space formulation; intended to be used in context of a Vlasov (phase-space) solve.
   */
  class ConfigurationSpaceFormulationMaxwell1D2V
:
  public ConfigurationSpaceFormulation
  {
    std::vector<double> _q;
    double _eps0; // permittivity
    double _mu0;  // permeability
    int _spaceDim;
    
    BFPtr _maxwellBF;
    
    FunctionPtr _BSoln, _BDiff;
    FunctionPtr _ESoln, _EDiff;
    
    FunctionPtr _By_extruded;
    FunctionPtr _Ex_extruded;
    FunctionPtr _By_priorTime_extruded;
    FunctionPtr _Ex_priorTime_extruded;
  public:
    // strings for variables:
    static const string S_EX;      // trial variable - electric field E_x
    static const string S_BY;      // trial variable - magnetic field B_y
    static const string S_PHI;     // trial variable - potential phi
    static const string S_A;       // trial variable - vector potential A
    
    static const string S_EX_HAT;  // trace variable - trace of E_x in L^2 (defined on spatial and temporal interfaces)
    static const string S_BY_HAT;  // trace variable - trace of B_y in L^2 (defined on spatial interfaces)
    static const string S_PHI_HAT; // trace variable - trace of phi in L^2 (defined on spatial interfaces)
    static const string S_A_HAT;   // trace variable - trace of A   in L^2 (defined on spatial interfaces)
    
    static const string S_W_JX;    // scalar test variable for equation loaded with J_x
    static const string S_W_JZ;    // scalar test variable for equation loaded with J_z
    static const string S_W_RHO;   // scalar test variable for equation loaded with rho
    static const string S_TAU_A;   // test variable for equation defining vector potential A
    static const string S_TAU_PHI; // test variable for equation defining electric potential phi
  public:
    //! one entry in q per species
    ConfigurationSpaceFormulationMaxwell1D2V(const std::vector<double> &q, const double eps0, const double mu0);
    
    ConfigurationSpaceFormulationMaxwell1D2V(const double &q, const double eps0, const double mu0);

    void initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k) override;
    
    virtual void setMoments(std::vector< Teuchos::RCP<SolutionMoments> > fMoments, FunctionPtr n0) override;
    
    //! Returns a name distinguishing this configuration-space formulation from others.  "Maxwell", "Maxwell", "Gauss" are all possibilities.
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
    
    //! Allows the owning formulation to provide explicit charge/current source functions.
    virtual void setCurrentSources(FunctionPtr rho, FunctionPtr Jx, FunctionPtr Jy, FunctionPtr Jz) override;
    
    //! Perform MPI communication, etc. to ensure that the B_i, E_i function values for a given configuration-space cell are available on each MPI rank that has a corresponding phase-space cell.
    virtual void updateConfigurationSpaceFunctions() override;
    
    // electric field variables
    VarPtr E_fieldVar(int zeroBasedComp) override;
    VarPtr E_hat(int zeroBasedComp) override;
    
    // magnetic field variables
    VarPtr B_fieldVar(int zeroBasedComp) override;
    VarPtr B_hat(int zeroBasedComp) override;
    
    // electric potential variables:
    VarPtr VE_fieldVar() override;
    VarPtr VE_hat() override;
    
    // vector potential variables:
    VarPtr A_fieldVar(int zeroBasedComp) override;
    VarPtr A_hat(int zeroBasedComp) override;
    
    // test variables
    VarPtr tau_A();
    VarPtr tau_phi();
    VarPtr w_Jx();
    VarPtr w_Jz();
    VarPtr w_rho();
    
    // used in ConfigurationSpaceFormulationMaxwell1D2VTests.cpp:
    void setRHSForTesting(FunctionPtr rho_over_eps0, FunctionPtr Jx_over_eps0, FunctionPtr Jz_over_eps0);
  };
}

#endif /* ConfigurationSpaceFormulationMaxwell1D2V_h */
