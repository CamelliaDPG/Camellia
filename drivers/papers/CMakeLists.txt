project(papers)

option(BUILD_VlasovPoissonBackwardEuler "Build drivers in VlasovPoissonBackwardEuler directory" ON)

if (BUILD_VlasovPoissonBackwardEuler)
  add_subdirectory(VlasovPoissonBackwardEuler)
endif(BUILD_VlasovPoissonBackwardEuler)

option(BUILD_VlasovDiode "Build drivers in VlasovDiode directory" ON)

if (BUILD_VlasovDiode)
  add_subdirectory(VlasovDiode)
endif(BUILD_VlasovDiode)
