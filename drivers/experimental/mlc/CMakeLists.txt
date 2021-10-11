project(mlc-drivers)

cmake_policy(SET CMP0074 NEW)

if (DEFINED mlc_ROOT)
  find_package(mlc REQUIRED)

  MESSAGE("Adding executables for mlc-drivers.")
  add_executable( MLCDriver "MLCDriver.cpp" )

  target_link_libraries( MLCDriver 
  PUBLIC
    Camellia
    mlc
  )
else()
  MESSAGE("mlc_ROOT not defined; skipping mlc-drivers.")
endif()
