project(DpgAssemblyBoba7DDriver)

message("Adding executable for DpgAssemblyBoba7DDriver.")

add_executable(Vlasov3X3VAssemblyBobaDriver
  boba_types.hpp
  reference_integrals_1d.hpp
  reference_integrals_1d.cpp
  global_assembly.hpp
  global_assembly.cpp
  schur_solver_nd.hpp
  vlasov_3x3v_driver.cpp
)
set_target_properties(Vlasov3X3VAssemblyBobaDriver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(Vlasov3X3VAssemblyBobaDriver
  Camellia
)
