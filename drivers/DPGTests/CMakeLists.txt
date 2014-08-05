project(DPGTests)

FILE(GLOB TEST_SOURCES "*.cpp")
set( DRIVERS_DIR ".." )
set( CONFUSION_SOURCES
  "${DRIVERS_DIR}/Confusion/ConfusionManufacturedSolution.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionProblemLegacy.cpp"
  "${DRIVERS_DIR}/Confusion/ConfusionBilinearForm.cpp"
)
set( POISSON_SOURCES
  "${DRIVERS_DIR}/Poisson/PoissonBilinearForm.cpp"
  "${DRIVERS_DIR}/Poisson/PoissonExactSolution.cpp"
)
include_directories("${DRIVERS_DIR}/Poisson") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Confusion") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Stokes") # for certain tests in DPGTests

add_executable(DPGTests ${TEST_SOURCES} ${CONFUSION_SOURCES} ${POISSON_SOURCES})
target_link_libraries(DPGTests 
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  Camellia
)
