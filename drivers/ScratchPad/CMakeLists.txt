project(ScratchPad)

set(DRIVERS_DIR "..")
include_directories("${DRIVERS_DIR}/Poisson") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Confusion") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/DPGTests") # for certain tests in DPGTests
set( POISSON_SRC
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
set( CONFUSION_SRC
  "${DRIVERS_DIR}/Confusion/ConfusionManufacturedSolution.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionProblemLegacy.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionBilinearForm.cpp"
)
set( TEST_SRC
  "${DRIVERS_DIR}/DPGTests/SolutionTests.cpp"
)

add_executable(Burgers NewBurgersDriver.cpp)
target_link_libraries(Burgers
  Camellia
)
add_executable(BurgersTest BurgersTest.cpp ${TEST_SRC} ${POISSON_SRC} ${CONFUSION_SRC})
target_link_libraries(BurgersTest
  Camellia
)
