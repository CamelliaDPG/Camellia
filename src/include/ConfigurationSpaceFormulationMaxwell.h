//
//  ConfigurationSpaceFormulationMaxwell.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#ifndef ConfigurationSpaceFormulationMaxwell_h
#define ConfigurationSpaceFormulationMaxwell_h

#include "Camellia.h"
#include "ConfigurationSpaceFormulation.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
    \brief Class for specifying a Maxwell configuration-space formulation; intended to be used in context of a Vlasov (phase-space) solve.
   */
  class ConfigurationSpaceFormulationMaxwell
:
  public ConfigurationSpaceFormulation
  {
    double _q;
    double _eps0; // permittivity
    double _mu0;  // permeability
    int _spaceDim;
    
    BFPtr _maxwellBF;
    
    FunctionPtr _BSoln, _BDiff;
    FunctionPtr _ESoln, _EDiff;
    
    std::vector<FunctionPtr> _B_extruded;
    std::vector<FunctionPtr> _E_extruded;
  public:
    // strings for variables:
    static const string S_EX;     // trial variable - electric field E_x
    static const string S_EY;     // trial variable - electric field E_y
    static const string S_EZ;     // trial variable - electric field E_z
    static const string S_BX;     // trial variable - magnetic field B_x
    static const string S_BY;     // trial variable - magnetic field B_y
    static const string S_BZ;     // trial variable - magnetic field B_z
    static const string S_VE;     // trial variable - potential VE
    static const string S_EX_HAT; // trace variable - trace of E_x in H^1/2
    static const string S_EY_HAT; // trace variable - trace of E_y in H^1/2
    static const string S_EZ_HAT; // trace variable - trace of E_z in H^1/2
    static const string S_BX_HAT; // trace variable - trace of E_x in H^1/2
    static const string S_BY_HAT; // trace variable - trace of E_y in H^1/2
    static const string S_BZ_HAT; // trace variable - trace of E_z in H^1/2
    static const string S_VE_HAT; // trace variable - trace of VE in H^1/2
    static const string S_W1;     // scalar test variable
    static const string S_W2;     // scalar test variable
    static const string S_TAU1;   // vector test variable tau_1 -- used in 3D formulation (2D, too?)
    static const string S_TAU2;   // vector test variable tau_2 -- used in 3D formulation (2D, too?)
    static const string S_TAU3;   // vector test variable tau_3 -- used in 3D formulation (2D, too?)
    static const string S_TAU1Y;  // scalar test variable tau_{1y} -- used in 1D formulation
    static const string S_TAU1Z;  // scalar test variable tau_{1z} -- used in 1D formulation
    static const string S_TAU2X;  // scalar test variable tau_{2x} -- used in 1D formulation
    static const string S_TAU2Y;  // scalar test variable tau_{2y} -- used in 1D formulation
    static const string S_TAU2Z;  // scalar test variable tau_{2z} -- used in 1D formulation
    static const string S_TAU3X;  // scalar test variable tau_{2x} -- used in 1D formulation
  public:
    ConfigurationSpaceFormulationMaxwell(const double q, const double eps0, const double mu0);

    void initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k) override;
    
    virtual void setMoments(Teuchos::RCP<SolutionMoments> fMoments, FunctionPtr n0) override;
    
    //! Returns a name distinguishing this configuration-space formulation from others.  "Maxwell", "Maxwell", "Gauss" are all possibilities.
    virtual std::string name() const override;
        
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i(int zeroBasedComp) override;
    
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i(int zeroBasedComp) override;
    
    //! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
    virtual int maxMomentNumber() override;
    
    //! Perform MPI communication, etc. to ensure that the B_i, E_i function values for a given configuration-space cell are available on each MPI rank that has a corresponding phase-space cell.
    virtual void updateConfigurationSpaceFunctions() override;
    
    // electric field variables
    VarPtr E_fieldVar(int zeroBasedComp) override;
    VarPtr E_hat(int zeroBasedComp) override;
    
    // magnetic field variables
    VarPtr B_fieldVar(int zeroBasedComp) override;
    VarPtr B_hat(int zeroBasedComp) override;
    
    // potential variables:
    VarPtr VE_fieldVar() override;
    VarPtr VE_hat() override;
    
    // test variable for Gauss' Law
    VarPtr w1();
    
    // test variable for Gauss' Law for Magnetism
    VarPtr w2(); // null in 1D
    
    // test variables for 1D scalar expansion of Maxwell-Faraday, Ampere equations:
    VarPtr tau1y();
    VarPtr tau1z();
    VarPtr tau2x();
    VarPtr tau2y();
    VarPtr tau2z();
    VarPtr tau3x(); // potential equation
    
    // test variables for spaceDim > 1
    VarPtr tau1(); // H(curl) spatially
    VarPtr tau2(); // H(curl) spatially
    VarPtr tau3(); // H(div) spatially
  };
}

#endif /* ConfigurationSpaceFormulationMaxwell_h */
