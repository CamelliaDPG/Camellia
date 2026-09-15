#ifndef Camellia_TensorMaxwellRHSMomentData_h
#define Camellia_TensorMaxwellRHSMomentData_h

#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "BasisCache.h"
#include "Function.h"

namespace Camellia
{
  /**
   * \brief Per-configuration-point scalar fields consumed by the 1D2V Maxwell RHS.
   *
   * The entries use the same algebraic convention as
   * ConfigurationSpaceFormulationMaxwell1D2V::setMoments(): rho is the RHS-ready
   * charge-density coefficient q/eps0*(density-n0), while Jx and Jz are physical
   * current moments before the final Maxwell RHS signs/scalings are applied
   * (-Jx/eps0 for w_Jx and +Jz/eps0 for w_Jz).
   */
  class TensorMaxwellRHSMomentData
  {
    int _numConfigurationCells = 0;
    int _numPointsPerConfigurationCell = 0;
    std::vector<double> _rhoRHSValues;
    std::vector<double> _JxValues;
    std::vector<double> _JzValues;

  public:
    using DeviceVector = Kokkos::View<double*>;

  private:
    DeviceVector _rhoRHSValuesDevice;
    DeviceVector _JxValuesDevice;
    DeviceVector _JzValuesDevice;

    int ordinal(int configurationCellOrdinal, int pointOrdinal) const
    {
      if ((configurationCellOrdinal < 0) || (configurationCellOrdinal >= _numConfigurationCells))
      {
        throw std::invalid_argument("TensorMaxwellRHSMomentData configuration cell ordinal out of range.");
      }
      if ((pointOrdinal < 0) || (pointOrdinal >= _numPointsPerConfigurationCell))
      {
        throw std::invalid_argument("TensorMaxwellRHSMomentData point ordinal out of range.");
      }
      return configurationCellOrdinal * _numPointsPerConfigurationCell + pointOrdinal;
    }

  public:
    TensorMaxwellRHSMomentData() = default;

    TensorMaxwellRHSMomentData(int numConfigurationCells,
                               int numPointsPerConfigurationCell)
    : _numConfigurationCells(numConfigurationCells),
      _numPointsPerConfigurationCell(numPointsPerConfigurationCell),
      _rhoRHSValues(numConfigurationCells * numPointsPerConfigurationCell, 0.0),
      _JxValues(numConfigurationCells * numPointsPerConfigurationCell, 0.0),
      _JzValues(numConfigurationCells * numPointsPerConfigurationCell, 0.0)
    {
      if (_numConfigurationCells < 0) throw std::invalid_argument("TensorMaxwellRHSMomentData requires nonnegative cell count.");
      if (_numPointsPerConfigurationCell < 0) throw std::invalid_argument("TensorMaxwellRHSMomentData requires nonnegative point count.");
    }

    int numConfigurationCells() const { return _numConfigurationCells; }
    int numPointsPerConfigurationCell() const { return _numPointsPerConfigurationCell; }
    int numEntries() const { return static_cast<int>(_rhoRHSValues.size()); }

    double &rhoRHS(int configurationCellOrdinal, int pointOrdinal) { return _rhoRHSValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }
    double &Jx(int configurationCellOrdinal, int pointOrdinal) { return _JxValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }
    double &Jz(int configurationCellOrdinal, int pointOrdinal) { return _JzValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }

    double rhoRHS(int configurationCellOrdinal, int pointOrdinal) const { return _rhoRHSValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }
    double Jx(int configurationCellOrdinal, int pointOrdinal) const { return _JxValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }
    double Jz(int configurationCellOrdinal, int pointOrdinal) const { return _JzValues[ordinal(configurationCellOrdinal, pointOrdinal)]; }

    const std::vector<double> &rhoRHSValues() const { return _rhoRHSValues; }
    const std::vector<double> &JxValues() const { return _JxValues; }
    const std::vector<double> &JzValues() const { return _JzValues; }

    const DeviceVector &rhoRHSValuesDevice() const { return _rhoRHSValuesDevice; }
    const DeviceVector &JxValuesDevice() const { return _JxValuesDevice; }
    const DeviceVector &JzValuesDevice() const { return _JzValuesDevice; }

    bool hasDeviceMirrors() const
    {
      return (static_cast<int>(_rhoRHSValuesDevice.extent(0)) == numEntries()) &&
             (static_cast<int>(_JxValuesDevice.extent(0)) == numEntries()) &&
             (static_cast<int>(_JzValuesDevice.extent(0)) == numEntries());
    }

    void populateDeviceMirrors()
    {
      _rhoRHSValuesDevice = DeviceVector("tensor Maxwell rho RHS device values", _rhoRHSValues.size());
      _JxValuesDevice     = DeviceVector("tensor Maxwell Jx device values",      _JxValues.size());
      _JzValuesDevice     = DeviceVector("tensor Maxwell Jz device values",      _JzValues.size());

      auto rhoHost = Kokkos::create_mirror_view(_rhoRHSValuesDevice);
      auto jxHost  = Kokkos::create_mirror_view(_JxValuesDevice);
      auto jzHost  = Kokkos::create_mirror_view(_JzValuesDevice);
      for (size_t entry=0; entry<_rhoRHSValues.size(); entry++)
      {
        rhoHost(entry) = _rhoRHSValues[entry];
        jxHost(entry)  = _JxValues[entry];
        jzHost(entry)  = _JzValues[entry];
      }
      Kokkos::deep_copy(_rhoRHSValuesDevice, rhoHost);
      Kokkos::deep_copy(_JxValuesDevice,     jxHost);
      Kokkos::deep_copy(_JzValuesDevice,     jzHost);
    }
  };

  class TensorMaxwellRHSDevicePointAssembly
  {
  public:
    using DeviceVector = TensorMaxwellRHSMomentData::DeviceVector;

    static void assemblePointSourceTerms(const TensorMaxwellRHSMomentData &momentData,
                                         double eps0,
                                         DeviceVector rhoTermOut,
                                         DeviceVector jxTermOut,
                                         DeviceVector jzTermOut)
    {
      if (eps0 == 0.0) throw std::invalid_argument("TensorMaxwellRHSDevicePointAssembly requires nonzero eps0.");
      if (!momentData.hasDeviceMirrors()) throw std::invalid_argument("TensorMaxwellRHSDevicePointAssembly requires populated device mirrors.");
      const int entryCount = momentData.numEntries();
      if (static_cast<int>(rhoTermOut.extent(0)) != entryCount) throw std::invalid_argument("TensorMaxwellRHSDevicePointAssembly rho output extent mismatch.");
      if (static_cast<int>(jxTermOut.extent(0)) != entryCount) throw std::invalid_argument("TensorMaxwellRHSDevicePointAssembly Jx output extent mismatch.");
      if (static_cast<int>(jzTermOut.extent(0)) != entryCount) throw std::invalid_argument("TensorMaxwellRHSDevicePointAssembly Jz output extent mismatch.");

      auto rhoValues = momentData.rhoRHSValuesDevice();
      auto jxValues  = momentData.JxValuesDevice();
      auto jzValues  = momentData.JzValuesDevice();
      Kokkos::parallel_for("TensorMaxwellRHSDevicePointAssembly::assemblePointSourceTerms",
                           entryCount,
                           KOKKOS_LAMBDA(const int entry)
                           {
                             rhoTermOut(entry) = rhoValues(entry);
                             jxTermOut(entry)  = -jxValues(entry) / eps0;
                             jzTermOut(entry)  =  jzValues(entry) / eps0;
                           });
      Kokkos::fence();
    }
  };

  class TensorMaxwellRHSPointFunction : public TFunction<double>
  {
  public:
    enum Field
    {
      RHO_RHS,
      JX,
      JZ
    };

  private:
    Teuchos::RCP<const TensorMaxwellRHSMomentData> _momentData;
    Field _field;

    double value(int cellOrdinal, int pointOrdinal) const
    {
      switch (_field)
      {
        case RHO_RHS: return _momentData->rhoRHS(cellOrdinal, pointOrdinal);
        case JX:      return _momentData->Jx(cellOrdinal, pointOrdinal);
        case JZ:      return _momentData->Jz(cellOrdinal, pointOrdinal);
      }
      throw std::invalid_argument("TensorMaxwellRHSPointFunction received an invalid field selector.");
    }

  public:
    TensorMaxwellRHSPointFunction(const TensorMaxwellRHSMomentData &momentData,
                                  Field field,
                                  const std::string &displayName)
    : TFunction<double>(displayName, 0),
      _momentData(Teuchos::rcp(new TensorMaxwellRHSMomentData(momentData))),
      _field(field)
    {}

    using TFunction<double>::values;

    void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      CHECK_VALUES_RANK(values);
      const int numCells = static_cast<int>(basisCache->cellIDs().size());
      const int numPoints = values.dimension(1);
      if (numCells != _momentData->numConfigurationCells())
      {
        throw std::invalid_argument("TensorMaxwellRHSPointFunction basis-cache cell count does not match moment data.");
      }
      if (numPoints != _momentData->numPointsPerConfigurationCell())
      {
        throw std::invalid_argument("TensorMaxwellRHSPointFunction basis-cache point count does not match moment data.");
      }

      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          values(cellOrdinal,pointOrdinal) = value(cellOrdinal, pointOrdinal);
        }
      }
    }
  };

  class TensorMaxwellRHSFunctionAdapter
  {
  public:
    static FunctionPtr rhoRHSFunction(const TensorMaxwellRHSMomentData &momentData)
    {
      return Teuchos::rcp(new TensorMaxwellRHSPointFunction(momentData,
                                                           TensorMaxwellRHSPointFunction::RHO_RHS,
                                                           "tensor Maxwell rho RHS"));
    }

    static FunctionPtr JxFunction(const TensorMaxwellRHSMomentData &momentData)
    {
      return Teuchos::rcp(new TensorMaxwellRHSPointFunction(momentData,
                                                           TensorMaxwellRHSPointFunction::JX,
                                                           "tensor Maxwell Jx"));
    }

    static FunctionPtr JzFunction(const TensorMaxwellRHSMomentData &momentData)
    {
      return Teuchos::rcp(new TensorMaxwellRHSPointFunction(momentData,
                                                           TensorMaxwellRHSPointFunction::JZ,
                                                           "tensor Maxwell Jz"));
    }
  };

  struct TensorPlanar1D2VSpeciesMoments
  {
    std::vector<double> _density;
    std::vector<double> _xProperVelocityMoment;
    std::vector<double> _zProperVelocityMoment;

    TensorPlanar1D2VSpeciesMoments() = default;
    TensorPlanar1D2VSpeciesMoments(const std::vector<double> &density,
                                   const std::vector<double> &xProperVelocityMoment,
                                   const std::vector<double> &zProperVelocityMoment)
    : _density(density),
      _xProperVelocityMoment(xProperVelocityMoment),
      _zProperVelocityMoment(zProperVelocityMoment)
    {}
  };

  class TensorMaxwell1D2VMomentAdapter
  {
  public:
    static TensorMaxwellRHSMomentData buildRHSMomentData(int numConfigurationCells,
                                                         int numPointsPerConfigurationCell,
                                                         const std::vector<double> &speciesCharges,
                                                         double eps0,
                                                         const std::vector<TensorPlanar1D2VSpeciesMoments> &speciesMoments,
                                                         const std::vector<double> &n0Values)
    {
      if (eps0 == 0.0) throw std::invalid_argument("TensorMaxwell1D2VMomentAdapter requires nonzero eps0.");
      const int numEntries = numConfigurationCells * numPointsPerConfigurationCell;
      if (static_cast<int>(speciesCharges.size()) != static_cast<int>(speciesMoments.size()))
      {
        throw std::invalid_argument("TensorMaxwell1D2VMomentAdapter species charge/moment count mismatch.");
      }
      if (static_cast<int>(n0Values.size()) != numEntries)
      {
        throw std::invalid_argument("TensorMaxwell1D2VMomentAdapter n0 entry count mismatch.");
      }

      TensorMaxwellRHSMomentData data(numConfigurationCells, numPointsPerConfigurationCell);
      for (int species=0; species<static_cast<int>(speciesMoments.size()); species++)
      {
        const auto &moments = speciesMoments[species];
        if ((static_cast<int>(moments._density.size()) != numEntries) ||
            (static_cast<int>(moments._xProperVelocityMoment.size()) != numEntries) ||
            (static_cast<int>(moments._zProperVelocityMoment.size()) != numEntries))
        {
          throw std::invalid_argument("TensorMaxwell1D2VMomentAdapter moment entry count mismatch.");
        }

        const double q = speciesCharges[species];
        for (int cell=0; cell<numConfigurationCells; cell++)
        {
          for (int point=0; point<numPointsPerConfigurationCell; point++)
          {
            const int entry = cell * numPointsPerConfigurationCell + point;
            data.rhoRHS(cell, point) += q / eps0 * (moments._density[entry] - n0Values[entry]);
            data.Jx(cell, point)     += q * moments._xProperVelocityMoment[entry];
            data.Jz(cell, point)     += q * moments._zProperVelocityMoment[entry];
          }
        }
      }
      return data;
    }
  };
}

#endif
