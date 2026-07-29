// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  RelativisticVlasovSpaceTimeFormulation.h
//  Camellia
//
//  Created by Nate Roberts on December 16, 2022.
//
//

#ifndef Camellia_RelativisticVlasovSpaceTimeFormulation_h
#define Camellia_RelativisticVlasovSpaceTimeFormulation_h

#include "VlasovSpaceTimeFormulation.h"

namespace Camellia
{

class RelativisticVlasovSpaceTimeFormulation : public VlasovSpaceTimeFormulation
{
public:
  enum TestSpaceInnerProduct
  {
    GraphTestSpaceInnerProduct,
    L2TestSpaceInnerProduct
  };
protected:
  const double _c; // speed of light.  Must be consistent with the momentum discretization
  const double _testSpaceL2Weight; // weight for "stabilization term" in graph norm.
  const TestSpaceInnerProduct _testSpaceInnerProduct;
  
  // Cache one relativistic first-moment Function per (velocity component, species).
  // The Function objects themselves own the persistent weighted-solution / weighted-moments state,
  // and update that state when the driving Vlasov-owned SolutionMoments object updates.
  mutable std::map<std::pair<int,int>, SolutionPtr> _relativisticWeightedSolutions;
  mutable std::map<std::pair<int,int>, Teuchos::RCP<SolutionMoments>> _relativisticWeightedMoments;
  mutable std::map<std::pair<int,int>, FunctionPtr> _relativisticFirstMomentFunctions;
public:
  RelativisticVlasovSpaceTimeFormulation(const double c, const double q, const double m,
                                         Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                         MeshTopologyPtr spatialMeshTopology,
                                         std::vector<double> temporalVertices,
                                         std::vector<double> momentumSpaceVertices_x,
                                         std::vector<double> momentumSpaceVertices_y,
                                         std::vector<double> momentumSpaceVertices_z,
                                         const int H1Order, const int delta_k,
                                         const int numUniformRefinements,
                                         const TestSpaceInnerProduct = L2TestSpaceInnerProduct,
                                         const double testSpaceL2Weight = 1.0,
                                         const EBasisFamily &basisFamily = BASIS_FAMILY_HIERARCHICAL,
                                         const int maxMomentNumberForDriver = 0,
                                         const bool enforcePositivity = false);
  
  //! called during initialization of Vlasov Solution objects.  Subclasses must define a distribution field variable (f) with name VlasovSpaceTimeFormulation::S_F, a trace variable (t_n) with name VlasovSpaceTimeFormulation::S_T_N_HAT.  The former is used for moment computations, conservation diagnostics, and convergence criteria for the fixed-point iteration; the latter is used for boundary condition imposition.
  VarFactoryPtr constructVlasovVarFactory() override;
  
  //! called from constructor, after Vlasov Solution and Mesh objects are constructed from the passed-in BF object.  Subclasses should fill this appropriately by calling addTerm().
  void initializeVlasovBilinearForm(BFPtr emptyVlasovBF) override;
  
  //! called from constructor, after Vlasov BF object is initialized.
  IPPtr constructVlasovInnerProduct(const bool sumFactorizedAssembly) override;
  
  //! Constructs and returns a VlasovSpaceTimeFormulation for the time slab following the present.  Should be called after all refinements and solves have been completed on the present time slab; modifications to the present time slab after the next slab is created may result in an invalid/incorrect application of boundary conditions in the next slab.  Subclasses should override.
  Teuchos::RCP<VlasovSpaceTimeFormulation> nextTimeSlab(double tWidth, int numTemporalElements) override;
  
  /** \brief  Returns a Function (varying in proper velocity coordinates) representing the Lorentz factor gamma.*/
  FunctionPtr gamma() const;
  
  /** \brief  Returns a Function representing the specified velocity component.  For RelativisticVlasovSpaceTimeFormulation, this is computed from proper velocity coordinates.*/
  FunctionPtr v_i(const int zeroBasedComp) const override;
  
  /**
   \brief Returns a configuration-space Function representing the limited relativistic first-order moment in the indicated velocity component.

   This helper is deliberately scoped to first moments only and is allowed to be computationally more expensive than the existing
   polynomial moment machinery.  The original development target emphasized non-hanging-node meshes, but the final implementation
   proceeds by projecting a relativistically weighted phase-space field and then reusing zeroth-moment integration, so the effective
   supported mesh class is determined by tests rather than by a hard no-hanging-node restriction.
   */
  FunctionPtr relativisticFirstMoment(const int zeroBasedVelocityComponent, const int speciesOrdinal);
  FunctionPtr relativisticFirstMoment(const int zeroBasedVelocityComponent, const int speciesOrdinal,
                                      SolutionMoments* drivingMoments,
                                      SolutionPtr phaseSpaceSolution);
  
  FunctionPtr currentX() const override;
  FunctionPtr currentY() const override;
  FunctionPtr currentZ() const override;
};

}

#endif
