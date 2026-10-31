if(NOT TARGET BOBA::BOBA)
  find_package(BOBA CONFIG REQUIRED)
endif()

add_executable(PlanarMITLBobaDriver planar_mitl_boba_driver.cpp)
target_compile_features(PlanarMITLBobaDriver PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(PlanarMITLBobaDriver PRIVATE -fp-model=precise)
endif()
target_link_libraries(PlanarMITLBobaDriver PRIVATE Camellia BOBA::BOBA)

add_test(NAME PlanarMITLBobaVolume
  COMMAND PlanarMITLBobaDriver --validate --solve --betaMax=0.5)
add_test(NAME PlanarMITLBobaInteriorFlux
  COMMAND PlanarMITLBobaDriver --validate --solve --nx=2 --nux=2 --betaMax=0.5)
add_test(NAME PlanarMITLBobaZeroField
  COMMAND PlanarMITLBobaDriver --validate --Ex=0 --By=0 --nt=2 --nuz=2 --beta0=-0.3)
add_test(NAME PlanarMITLBobaGraded
  COMMAND PlanarMITLBobaDriver --validate --gradedVelocityMesh --alpha=1e6)
set_tests_properties(PlanarMITLBobaVolume PlanarMITLBobaInteriorFlux
  PlanarMITLBobaZeroField PlanarMITLBobaGraded PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OMP_PROC_BIND=false" TIMEOUT 180)
