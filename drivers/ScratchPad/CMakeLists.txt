project(ScratchPad)

set(DRIVERS_DIR "..")
include_directories("${DRIVERS_DIR}/Poisson") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/Confusion") # for certain tests in DPGTests
include_directories("${DRIVERS_DIR}/DPGTests") # for certain tests in DPGTests
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
add_executable(BurgersTest BurgersTest.cpp ${TEST_SRC} ${CONFUSION_SRC})
target_link_libraries(BurgersTest
  Camellia
)
