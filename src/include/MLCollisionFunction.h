//
//  MLCollisionFunction.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/14/22.
//

#ifndef MLCollisionFunction_h
#define MLCollisionFunction_h

#include "Function.h"
#include "SolutionMoments.h"

#ifdef HAVE_MLC
#include "MLC_CollisionModel.hpp"
#include "MLC_MomentConversion.hpp"
#endif

namespace Camellia
{
  class MLCollisionFunction : public TFunction<double>
  {
    FunctionPtr _f_soln;
    Teuchos::RCP<SolutionMoments> _f_solnMoments;
    
    FunctionPtr _bgkFunction;

    // MLC arguments:
    std::string _modelDirectory;
    
    // CollisionModel takes numPoints as a construction argument.  Until we make mlc more flexible on this front, we need to reload from disk every time we want a different number of points.
    int _numPointsForLoadedModel = 0;
#ifdef HAVE_MLC
    Teuchos::RCP<mlc::CollisionModel> _loadedModel;
#endif
    // scaling factors
    double _n0, _L0, _u0;
    
  protected:
    //! checks that the model has been loaded, with appropriate storage allocated for the specified number of cells and points.  (If it has not been loaded, loads it.)
    void ensureModelLoaded(const int &numCells, const int &numConfigurationPoints);
  public:
    /** \brief  Constructor.
     *  \param  n0_scaling        [in]  - scaling for the values of the distribution f; multiply f by this to get the physical value, in ____ units
     *  \param  L0_scaling        [in]  - scaling for configuration space; multiply each of x,y,z by this to get the coordinate in ____ units
     *  \param  u0_scaling        [in]  - scaling for velocity space; multiply each of v_x, v_y, v_z by this to get the velocity in ____ units
     *  \param  f_soln                 [in]  - a function representing the distribution in phase space
     *  \param  f_solnMoments [in]  - pointer to a SolutionMoments object with support for moments at least up to the maxMomentNumber() required by the model.
     *  \param  modelDirPath   [in] - path to the directory containing the mlc model.  Within this directory there should be a model subdirectory, as well as a parameters.yaml file, among other things.
     */
    MLCollisionFunction(const double n0_scaling, const double L0_scaling, const double u0_scaling,
                        FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments,
                        std::string &modelDirPath);
    
    /** \brief static method that reads the model from disk and returns the maximum moment number required by the model.  Useful for initializing the SolutionMoments object provided to the MLCollisionFunction constructor.
     */
    static int maxMomentNumber(std::string &modelDirPath);
    
    double getMass();
    
    using TFunction<double>::values;
    void values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    //! Determines and returns a flat array of scalings to go from internal Camellia (non-dimensionalized) moments to the dimensional moments used by MLC.  Moments are indexed as a flat array, with tetrahedral ordering, as in Camellia's MomentIndices and MLC's (standard) moment indexer.
    std::vector<double> computeMomentScaling(const int maxMomentNumber) const;
    
#ifdef HAVE_MLC
    //! get the collisional response values from the MLC model; this is a thin wrapper around the call of the same name in MLC.  Note that all the arrays must be correctly sized according to the size given at the time the MLC model was loaded.
    void generateCollisionalResponse(const mlc::Array &ml_input_conservedMoments,
                                     mlc::Array &ml_output_tau, mlc::Array &ml_output_g0, mlc::Array &ml_output_ux, mlc::Array &ml_output_uy, mlc::Array &ml_output_uz, mlc::Array &ml_output_sigma, mlc::Array &ml_output_G) const;
    
    //! convert the moments computed by Camellia::SolutionMoments to a format suitable for passing to generateCollisionalResponse().
    void convertMomentsForMLCInput(const DeviceView<double> &camelliaMoments, mlc::Array &mlc_input_conservedMoments);
#endif
    
    void contributeBGKResponse(DeviceView<double> responseView, DeviceView<double> f, DeviceView<double> valuesAllMoments, DeviceView<double> tau, DeviceView<double> physPoints_vx, DeviceView<double> physPoints_vy, DeviceView<double> physPoints_vz) const;
    
    //! computes and sums into the response view the ML model's differential collisional response beyond the BGK response.
    void contributeDifferentialResponseFromBGK(DeviceView<double> responseView, DeviceView<double> g0, DeviceView<double> ux, DeviceView<double> uy, DeviceView<double> uz, DeviceView<double> sigma, DeviceView<double> G,
                                               DeviceView<double> physPoints_vx, DeviceView<double> physPoints_vy, DeviceView<double> physPoints_vz) const;
  };
}

#endif /* MLCCollisionFunction_h */
