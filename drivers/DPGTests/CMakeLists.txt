project(DPGTests)

FILE(GLOB TEST_SOURCES "*.cpp")
set( DRIVERS_DIR ".." )
set( POISSON_SOURCES
  "${DRIVERS_DIR}/Poisson/PoissonBilinearForm.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolution.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolutionQuartic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonRHSQuartic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCLinear.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCQuadratic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCCubic.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonBCQuartic.cpp"
)
set( CONFUSION_SOURCES
  "${DRIVERS_DIR}/Confusion/ConfusionManufacturedSolution.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionProblemLegacy.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionBilinearForm.cpp"
)
include_directories("${DRIVERS_DIR}/Poisson") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Confusion") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Stokes") # for certain tests in DPGTests

add_executable(DPGTests ${TEST_SOURCES} ${POISSON_SOURCES} ${CONFUSION_SOURCES})
target_link_libraries(DPGTests 
  Camellia
)
