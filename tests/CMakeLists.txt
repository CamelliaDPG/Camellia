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
  LinearTermTests.cpp 
  BasisCacheTests.cpp 
  ElementTests.cpp 
  FunctionTests.cpp 
  MeshRefinementTests.cpp 
  TestBilinearFormFlux.cpp 
  MeshTests.cpp 
)
# Link test executable against gtest & gtest_main
target_link_libraries(RunTests gtest gtest_main Camellia)
add_test( RunTests RunTests )
