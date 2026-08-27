project(CompareTestNormsDPGBoBa)

MESSAGE("Adding executable for CompareTestNormsDPGBoBa.")
add_executable(CompareTestNormsDPGBoBa "CompareTestNormsDPGBoBa.cpp")
set_target_properties(CompareTestNormsDPGBoBa PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(CompareTestNormsDPGBoBa
  Camellia
)
