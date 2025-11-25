project(papers)

option(BUILD_RelativisticVlasovSpaceTime "Build drivers in RelativisticVlasovSpaceTime directory" ON)
if (BUILD_RelativisticVlasovSpaceTime)
  add_subdirectory(RelativisticVlasovSpaceTime)
endif(BUILD_RelativisticVlasovSpaceTime)

option(BUILD_VlasovPoissonBackwardEuler "Build drivers in VlasovPoissonBackwardEuler directory" ON)

if (BUILD_VlasovPoissonBackwardEuler)
  add_subdirectory(VlasovPoissonBackwardEuler)
endif(BUILD_VlasovPoissonBackwardEuler)

option(BUILD_VlasovDiode "Build drivers in VlasovDiode directory" ON)

if (BUILD_VlasovDiode)
  add_subdirectory(VlasovDiode)
endif(BUILD_VlasovDiode)

option(BUILD_VlasovPoissonSpaceTime "Build drivers in VlasovPoissonSpaceTime directory" ON)
if (BUILD_VlasovPoissonSpaceTime)
  add_subdirectory(VlasovPoissonSpaceTime)
endif(BUILD_VlasovPoissonSpaceTime)

