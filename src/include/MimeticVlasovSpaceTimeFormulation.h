// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MimeticVlasovSpaceTimeFormulation.h
//  Camellia
//
//  Created by Nate Roberts on December 16, 2022.
//
//

#ifndef Camellia_MimeticVlasovSpaceTimeFormulation_h
#define Camellia_MimeticVlasovSpaceTimeFormulation_h

#include "VlasovSpaceTimeFormulation.h"

namespace Camellia
{

class MimeticVlasovSpaceTimeFormulation : public VlasovSpaceTimeFormulation
{
protected:
  virtual map<int,int> testOrderEnhancements() override;
  virtual map<int,int> trialOrderEnhancements() override;
public:
  static const string S_GV_VX;  // v_x component of field variable g_v, defined as g_v = f v
  static const string S_GV_VY;  // v_y component of field variable g_v, defined as g_v = f v
  static const string S_GV_VZ;  // v_z component of field variable g_v, defined as g_v = f v
  static const string S_GA_X;   // x component of field variable g_a, defined as g_a = a f, where a is the configuration-space convective field
  static const string S_GA_Y;   // y component of field variable g_a, defined as g_a = a f, where a is the configuration-space convective field
  static const string S_GA_Z;   // z component of field variable g_a, defined as g_a = a f, where a is the configuration-space convective field
  static const string S_GC;  // field variable g_c, defined as g_c = C(f), where C is the collision operator.  (Initial implementation does not include g_c)
  static const string S_GT;
  
  static const vector<string> S_GV; // {S_GV_VX, S_GV_VY, S_GV_VZ}
  static const vector<string> S_GA; // { S_GA_X,  S_GA_Y,  S_GA_Z}
  
  static const string S_WV_VX;  // v_x component of test variable w_v
  static const string S_WV_VY;  // v_y component of test variable w_v
  static const string S_WV_VZ;  // v_z component of test variable w_v
  static const string S_WA_X;   //   x component of test variable w_a
  static const string S_WA_Y;   //   y component of test variable w_a
  static const string S_WA_Z;   //   z component of test variable w_a
  static const string S_WC;  // test variable w_c
  static const string S_WT;  // test variable w_t
  
  static const vector<string> S_WV; // {S_WV_VX, S_WV_VY, S_WV_VZ}
  static const vector<string> S_WA; // { S_WA_X,  S_WA_Y,  S_WA_Z}

  MimeticVlasovSpaceTimeFormulation(double q, double m, bool useSerendipityBasis,
                                    Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                    MeshTopologyPtr spatialMeshTopology,
                                    std::vector<double> temporalVertices,
                                    std::vector<double> velocitySpaceVertices_x,
                                    std::vector<double> velocitySpaceVertices_y,
                                    std::vector<double> velocitySpaceVertices_z,
                                    const int H1Order, const int delta_k,
                                    const int numUniformRefinements);
  
  //! called during initialization of Vlasov Solution objects.  Subclasses must define a distribution field variable (f) with name MimeticVlasovSpaceTimeFormulation::S_F, a trace variable (t_n) with name MimeticVlasovSpaceTimeFormulation::S_T_N_HAT.  The former is used for moment computations, conservation diagnostics, and convergence criteria for the fixed-point iteration; the latter is used for boundary condition imposition.
  VarFactoryPtr constructVlasovVarFactory() override;
  
  //! called from constructor, after Vlasov Solution and Mesh objects are constructed from the passed-in BF object.  Subclasses should fill this appropriately by calling addTerm().
  void initializeVlasovBilinearForm(BFPtr emptyVlasovBF) override;
  
  //! called from constructor, after Vlasov BF object is initialized.
  IPPtr constructVlasovInnerProduct() override;
  
  //! Constructs and returns a VlasovSpaceTimeFormulation for the time slab following the present.  Should be called after all refinements and solves have been completed on the present time slab; modifications to the present time slab after the next slab is created may result in an invalid/incorrect application of boundary conditions in the next slab.  Subclasses should override.
  Teuchos::RCP<VlasovSpaceTimeFormulation> nextTimeSlab(double tWidth, int numTemporalElements) override;
  
  // field variables:
  VarPtr g_a(const int zeroBasedSpatialComponent);
  VarPtr g_v(const int zeroBasedVelocityComponent);
  VarPtr g_c();
  VarPtr g_t();
  
  // test variables:
  VarPtr w_a(const int zeroBasedSpatialComponent);
  VarPtr w_v(const int zeroBasedVelocityComponent);
  VarPtr w_c();
  VarPtr w_t();
  
  // local conservation diagnostics
  virtual std::map<int,FunctionPtr> testForMassConservation(int i) override;
  virtual std::map<int,FunctionPtr> testForXMomentumConservation(int i) override;
  virtual std::map<int,FunctionPtr> testForYMomentumConservation(int i) override;
  virtual std::map<int,FunctionPtr> testForZMomentumConservation(int i) override;
  virtual std::map<int,FunctionPtr> testForEnergyConservation(int i) override;
};

}

#endif
