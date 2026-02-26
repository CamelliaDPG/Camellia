project(DpgAssemblyBobaDriver)

message("Adding executables for DpgAssemblyBobaDriver.")

add_executable(VlasovAssemblyBobaDriver
  assembly_verification.cpp
  boba_types.hpp
  reference_integrals_1d.hpp
  reference_integrals_1d.cpp
  global_assembly.hpp
  global_assembly.cpp
  vlasov_system.hpp
  vlasov_system.cpp
  vlasov_driver.cpp
)
target_link_libraries(VlasovAssemblyBobaDriver
  Camellia
)
target_compile_features(VlasovAssemblyBobaDriver PRIVATE cxx_std_20)

add_executable(VlasovAssemblyBobaTimer
  ../SpaceTimeMixedVlasovBoba/SpaceTimeMixedVlasovSetup.cpp
  boba_types.hpp
  reference_integrals_1d.hpp
  reference_integrals_1d.cpp
  global_assembly.hpp
  global_assembly.cpp
  vlasov_system.hpp
  vlasov_system.cpp
  vlasov_timer.cpp
)
target_link_libraries(VlasovAssemblyBobaTimer
  Camellia
)
target_compile_features(VlasovAssemblyBobaTimer PRIVATE cxx_std_20)

message("Adding Teuchos unit tests for DpgAssemblyBoba.")
add_executable(DPGBobaAssemblyTests
  ../../../unit_tests/Teuchos_StandardParallelUnitTestMain.cpp
  DPGBobaAssemblyTests.cpp
  assembly_verification.cpp
  reference_integrals_1d.cpp
  global_assembly.cpp
)
target_link_libraries(DPGBobaAssemblyTests Camellia)
target_compile_features(DPGBobaAssemblyTests PRIVATE cxx_std_20)


configure_file(readTtVector.m readTtVector.m COPYONLY)
configure_file(readTtMatrix.m readTtMatrix.m COPYONLY)
