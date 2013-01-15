project(Tests)

################################
# GTest
################################
ADD_SUBDIRECTORY (gtest-1.6.0)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
# Add test cpp file
add_executable( RunTests 
  main.cpp
  ParallelUnitTestPrinter.cpp
  LinearTermTests.cpp 
  BasisCacheTests.cpp 
  ElementTests.cpp 
  FunctionTests.cpp 
  MeshRefinementTests.cpp 
  TestBilinearFormFlux.cpp 
  MeshTests.cpp 
  LocalConservationTests.cpp 
  MPIWrapperTests.cpp 
  #MultiBasisTests.cpp 
)
# Link test executable against gtest & gtest_main

IF( NOT DEFINED CAMELLIA_LIBRARY_NAME ) 
    MESSAGE("Building Camellia-debug project.")
    set(CAMELLIA_LIBRARY_NAME "Camellia")
ENDIF( NOT DEFINED CAMELLIA_LIBRARY_NAME )

IF( DEFINED Trilinos_LIBRARIES ) # Nate's approach: separate libraries for Trilinos and Camellia
    target_link_libraries(RunTests gtest ${CAMELLIA_LIBRARY_NAME} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ELSE()
    target_link_libraries(RunTests gtest ${CAMELLIA_LIBRARY_NAME})
ENDIF( DEFINED Trilinos_LIBRARIES ) 
add_test( RunTests RunTests )
