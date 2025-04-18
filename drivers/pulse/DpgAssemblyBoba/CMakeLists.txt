project(DpgAssemblyBobaDriver)

message("Adding executables for DpgAssemblyBobaDriver.")

add_executable(VlasovAssemblyBobaDriver
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

configure_file(readTtVector.m readTtVector.m COPYONLY)
configure_file(readTtMatrix.m readTtMatrix.m COPYONLY)
