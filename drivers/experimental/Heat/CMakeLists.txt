project(HeatDriver)

MESSAGE("Adding executable for HeatDriver.")
add_executable(HeatDriver "HeatDriver.cpp")
target_link_libraries(HeatDriver 
  Camellia
)

#install (TARGETS HeatDriver DESTINATION bin/examples/)

